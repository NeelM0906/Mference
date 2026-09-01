import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore
import MferenceValidationSupport

/// The QSA indexer and gated-attention-over-the-selected-set kernels, gated
/// against `FlashNextIndexerReference` / `FlashNextAttentionReference` on the
/// real toy install — the same weights, through the same loader, that the CPU
/// reference forward reads.
///
/// Two different bars, deliberately:
///
/// * **Selection is exact.** A selected set is an integer set; a flipped
///   boundary changes which KV a layer may read. Prefill (all positions in one
///   call) and stepped decode (one call per position, carrying the caches) both
///   have to reproduce the reference exactly, on both prompts. Where a boundary
///   is a bit-exact tie the suite says so rather than quietly passing on luck.
/// * **Attention output is a tensor**, so it carries an FP16 tolerance. The
///   measured error is printed on every run so the bound is a report, not a
///   guess.
///
/// The oracle is fed the **FP16-rounded** block input, the same bytes the GPU
/// sees, so what is measured is the kernels' arithmetic rather than the
/// runner's FP32 activations. Agreement with the runner's own FP32 selections is
/// asserted separately: that is the stronger claim and the one the port needs.
///
/// No-ops when `scratch/qwen4exp-toy-ckpt-prodlayout/` is absent.
@Suite struct FlashNextIndexerKernelTests {

    // MARK: - Harness

    private struct Harness {
        let context: MetalContext
        let model: Model
        let config: ArchConfig
        let weights: FlashNextWeights
        let capture: FlashNextReferenceRunner.Capture
        let tokens: [Int]
        let directory: URL
        let indexer: FlashNextIndexer
        let attention: FlashNextAttention
        let matVec: FlashNextMatVec
    }

    private static func make(_ prompt: FlashNextGoldens.Prompt) throws -> Harness? {
        guard FlashNextParity.checkpointIsPresent else { return nil }
        let context = try MetalContext()
        let dir = try FlashNextParity.installToyCheckpoint()
        let config = FlashNextParity.archConfig()
        let model = try Model.load(directoryURL: dir, device: context.device,
                                   expecting: config, streamingMode: .resident)
        let weights = FlashNextWeights(model: model)
        let runner = try FlashNextReferenceRunner(weights: weights)
        let tokens = try FlashNextGoldens.promptTokens(prompt)
        var capture: FlashNextReferenceRunner.Capture? = .init()
        _ = try runner.step(tokens: tokens, capture: &capture)

        let fn = config.flashNext
        let rotaryDim = Int(Double(config.fullHeadDim) * config.partialRotaryFactor)
        let matVec = try FlashNextMatVec(
            context: context, int4: try DequantInt4GEMV(context: context))
        let indexer = try FlashNextIndexer(
            context: context, matVec: matVec,
            geometry: .init(numHeads: fn.indexerNumHeads,
                            numKVHeads: fn.indexerNumKVHeads,
                            headDim: fn.indexerHeadDim,
                            compressRatio: fn.indexerCompressRatio,
                            blockBudget: fn.indexerBlockBudget,
                            rotaryDim: rotaryDim,
                            theta: Float(config.fullRopeTheta),
                            eps: 1e-6))
        let attention = FlashNextAttention(
            context: context, matVec: matVec,
            elementwise: try Elementwise(context: context),
            epilogue: try PrefillQKVEpilogue(context: context),
            attention: try Attention(context: context),
            geometry: .init(hidden: config.hiddenSize,
                            numHeads: config.numHeads,
                            numKVHeads: config.numFullKVHeads,
                            headDim: config.fullHeadDim,
                            rotaryDim: rotaryDim,
                            theta: Float(config.fullRopeTheta),
                            eps: 1e-6,
                            scale: 1 / Float(config.fullHeadDim).squareRoot()))
        return Harness(context: context, model: model, config: config,
                       weights: weights, capture: capture!, tokens: tokens,
                       directory: dir, indexer: indexer, attention: attention,
                       matVec: matVec)
    }

    private static func referenceGeometry(_ h: Harness)
        -> FlashNextIndexerReference.Geometry {
        let g = h.indexer.geometry
        return .init(numHeads: g.numHeads, numKVHeads: g.numKVHeads,
                     headDim: g.headDim, compressRatio: g.compressRatio,
                     blockBudget: g.blockBudget, rotaryDim: g.rotaryDim,
                     theta: g.theta, eps: g.eps)
    }

    private static func fullAttentionLayers(_ c: ArchConfig) -> [Int] {
        (0..<c.numLayers).filter { !c.layerIsLinear($0) }
    }

    /// The FP16 bytes the GPU will see, and the same values widened back so the
    /// oracle is fed an identical input.
    private static func fp16(_ values: [Float]) -> (buffer: [Float16], asFloat: [Float]) {
        let halves = values.map { Float16($0) }
        return (halves, halves.map { Float($0) })
    }

    private static func trunkName(_ layer: Int, _ suffix: String) -> String {
        "model.language_model.layers.\(layer).self_attn.\(suffix)"
    }

    // MARK: - 1. Selection: whole-prompt prefill

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func prefillSelectionIsExactAtEveryPosition(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let h = try Self.make(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let rows = h.tokens.count
        let g = Self.referenceGeometry(h)
        var checkedLayers = 0

        for layer in Self.fullAttentionLayers(h.config) {
            let key = FlashNextGoldens.layerKey(layer) + "."
            let mixed = try #require(h.capture.floats[key + "attn_hc_mixed"])
            let expectedFP32 = try #require(h.capture.integers[key + "indexer_selected"])
            let (halves, rounded) = Self.fp16(mixed)

            let selections = try Self.runIndexer(h, layer: layer, x: halves,
                                                 rows: rows, chunk: rows)
            let w = try h.weights.attention(layer: layer)
            let oracle = FlashNextIndexerReference.run(
                x: rounded, hidden: h.config.hiddenSize, rows: rows,
                startPosition: 0, indexerQK: w.indexerQK,
                qNorm: w.indexerQNorm, kNorm: w.indexerKNorm,
                rawKeys: [], g: g)

            #expect(selections == oracle.selected,
                    "\(prompt.rawValue) L\(layer) kernel vs oracle (same FP16 input)")
            #expect(selections == expectedFP32,
                    "\(prompt.rawValue) L\(layer) kernel vs the reference runner")
            checkedLayers += 1
        }
        #expect(checkedLayers > 0)
    }

    // MARK: - 2. Selection: stepped decode

    /// One call per position, carrying the raw-key and block-key caches. This is
    /// what exercises the incremental pooling: a block is written once, when its
    /// fourth token lands, and every later query reads it back unchanged.
    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func steppedDecodeSelectionMatchesWholePromptPrefill(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let h = try Self.make(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let rows = h.tokens.count
        var checkedLayers = 0

        for layer in Self.fullAttentionLayers(h.config) {
            let key = FlashNextGoldens.layerKey(layer) + "."
            let mixed = try #require(h.capture.floats[key + "attn_hc_mixed"])
            let expected = try #require(h.capture.integers[key + "indexer_selected"])
            let (halves, _) = Self.fp16(mixed)
            let stepped = try Self.runIndexer(h, layer: layer, x: halves,
                                              rows: rows, chunk: 1)
            #expect(stepped == expected,
                    "\(prompt.rawValue) L\(layer) stepped decode selections")
            checkedLayers += 1
        }
        #expect(checkedLayers > 0)
    }

    // MARK: - 3. Attention over the selected set

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func gatedAttentionOverTheSelectedSetMatchesTheReference(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let h = try Self.make(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let rows = h.tokens.count
        let hidden = h.config.hiddenSize
        var worst: Float = 0
        var checkedLayers = 0

        for layer in Self.fullAttentionLayers(h.config) {
            let key = FlashNextGoldens.layerKey(layer) + "."
            let mixed = try #require(h.capture.floats[key + "attn_hc_mixed"])
            let selected = try #require(h.capture.integers[key + "indexer_selected"])
            let expected = try #require(h.capture.floats[key + "block_out"])
            let (halves, _) = Self.fp16(mixed)

            let actual = try Self.runAttention(h, layer: layer, x: halves,
                                               rows: rows, selected: selected)
            let error = RelError.compute(actual: actual, reference: expected)
            worst = max(worst, error)
            #expect(error < Tolerance.fp16ChainedReduction,
                    "\(prompt.rawValue) L\(layer) block_out off by \(error)")
            checkedLayers += 1
            _ = hidden
        }
        print("flashnext gated attention over the selected set "
                + "(\(prompt.rawValue)): worst relative error \(worst)")
        #expect(checkedLayers > 0)
    }

    // MARK: - Drivers

    /// Run the indexer over `rows` positions in chunks of `chunk`, carrying the
    /// GPU caches, and return the per-row selections.
    private static func runIndexer(_ h: Harness, layer: Int,
                                   x halves: [Float16],
                                   rows: Int, chunk: Int) throws -> [[Int]] {
        let hidden = h.config.hiddenSize
        let device = h.context.device
        guard let xBuffer = Fp16Buffer.make(device, halves: halves) else {
            throw CocoaError(.fileReadUnknown)
        }
        let weight = FlashNextWeightMatrix.from(
            try h.model.resident(name: trunkName(layer, "indexer.index_qk_proj.weight")))
        let qNorm = try h.model.normWeight(
            name: trunkName(layer, "indexer.q_layernorm.weight"))
        let kNorm = try h.model.normWeight(
            name: trunkName(layer, "indexer.k_layernorm.weight"))

        let scratch = try h.indexer.makeScratch(device: device, rows: chunk,
                                                maxTokens: rows)
        let cache = try h.indexer.makeLayerCache(device: device, maxTokens: rows)
        var out: [[Int]] = []
        var position = 0
        while position < rows {
            let count = min(chunk, rows - position)
            guard let cb = h.context.queue.makeCommandBuffer() else {
                throw CocoaError(.fileReadUnknown)
            }
            h.indexer.encodeProjection(
                commandBuffer: cb, weight: weight, x: xBuffer,
                xOffset: position * hidden * MemoryLayout<Float16>.stride,
                hidden: hidden, scratch: scratch, rows: count)
            h.indexer.encodePrepare(
                commandBuffer: cb,
                qNorm: qNorm.buffer, qNormOffset: Int(qNorm.offset),
                kNorm: kNorm.buffer, kNormOffset: Int(kNorm.offset),
                scratch: scratch, cache: cache, rows: count,
                startPosition: position)
            h.indexer.encodeScores(commandBuffer: cb, scratch: scratch,
                                   cache: cache, rows: count,
                                   startPosition: position)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            for row in 0..<count where h.indexer.boundaryIsTied(
                scratch: scratch, row: row, startPosition: position) {
                print("flashnext indexer: L\(layer) query \(position + row) has a "
                        + "bit-exact top-k boundary tie — the ordering policy, not "
                        + "the model, decides this row")
            }
            out.append(contentsOf: h.indexer.selections(scratch: scratch,
                                                        rows: count,
                                                        startPosition: position))
            position += count
        }
        return out
    }

    /// Project, cache, gather each row's selected KV and attend, then gate and
    /// project out. One command buffer per row here — the production runner uses
    /// waves of gather slots, and this test's job is the arithmetic.
    private static func runAttention(_ h: Harness, layer: Int,
                                     x halves: [Float16],
                                     rows: Int,
                                     selected: [[Int]]) throws -> [Float] {
        let hidden = h.config.hiddenSize
        let device = h.context.device
        guard let xBuffer = Fp16Buffer.make(device, halves: halves),
              let out = Fp16Buffer.make(device, count: rows * hidden) else {
            throw CocoaError(.fileReadUnknown)
        }
        func matrix(_ suffix: String) throws -> FlashNextWeightMatrix {
            .from(try h.model.resident(name: trunkName(layer, suffix)))
        }
        let qNorm = try h.model.normWeight(name: trunkName(layer, "q_norm.weight"))
        let kNorm = try h.model.normWeight(name: trunkName(layer, "k_norm.weight"))
        let w = FlashNextAttention.Weights(
            q: try matrix("q_proj.weight"),
            k: try matrix("k_proj.weight"),
            v: try matrix("v_proj.weight"),
            o: try matrix("o_proj.weight"),
            qNorm: qNorm.buffer, qNormOffset: Int(qNorm.offset),
            kNorm: kNorm.buffer, kNormOffset: Int(kNorm.offset))

        let maxSelected = selected.map(\.count).max() ?? 1
        let scratch = try h.attention.makeScratch(device: device, rows: rows,
                                                  maxSelected: maxSelected,
                                                  gatherSlots: 1)
        let cache = try h.attention.makeKVCache(device: device, maxTokens: rows)
        // Indexer scratch is only needed for its row-indexed selection buffer.
        let indexScratch = try h.indexer.makeScratch(device: device, rows: rows,
                                                     maxTokens: rows)

        guard let projectCB = h.context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        h.attention.encodeProjectAndCache(
            commandBuffer: projectCB, weights: w, scratch: scratch, cache: cache,
            x: xBuffer, xOffset: 0, rows: rows, startPosition: 0)
        projectCB.commit()
        projectCB.waitUntilCompleted()
        #expect(projectCB.error == nil)

        for row in 0..<rows {
            let count = h.indexer.writeSelection(selected[row], row: row,
                                                 into: indexScratch)
            guard let cb = h.context.queue.makeCommandBuffer() else {
                throw CocoaError(.fileReadUnknown)
            }
            h.indexer.encodeGatherKV(
                commandBuffer: cb,
                kCache: cache.keys, kCacheOffset: 0,
                vCache: cache.values, vCacheOffset: 0,
                scratch: indexScratch, selectionRow: row,
                kOut: scratch.gatheredK, kOutOffset: 0,
                vOut: scratch.gatheredV, vOutOffset: 0,
                kvDim: h.attention.geometry.kvDim, count: count)
            h.attention.encodeAttendRow(commandBuffer: cb, scratch: scratch,
                                        row: row, slot: 0, selectedCount: count)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
        }

        guard let tailCB = h.context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        h.attention.encodeGateAndProject(commandBuffer: tailCB, weights: w,
                                         scratch: scratch, out: out,
                                         outOffset: 0, rows: rows)
        tailCB.commit()
        tailCB.waitUntilCompleted()
        #expect(tailCB.error == nil)
        return Fp16Buffer.read(out, count: rows * hidden)
    }
}
