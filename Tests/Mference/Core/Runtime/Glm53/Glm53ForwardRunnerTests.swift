import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// The FP16 tier of the `glm53flash` parity gates: `Glm53ForwardRunner` on a
/// real install of the toy checkpoint (written through the production planner
/// by `Glm53Parity.installToyCheckpoint`) against the fp32 oracle that passes
/// the goldens (`Glm53ReferenceParityTests`).
///
/// # Gates
///
/// 1. **Kernels on identical inputs.** The latent attention kernel matches its
///    scalar formula on random data, dense and under a selection, to fp32
///    accumulation noise.
/// 2. **Install round trip.** Resident tensors the runner reads come back from
///    the install equal to the checkpoint's stored value — the planner, the
///    writers and the loader agree byte for byte.
/// 3. **Discrete decisions.** Indexer selections and router top-8 agree with
///    the oracle exactly wherever the oracle's own margin is above the floor
///    (`Self.marginFloor`); a mismatch at a below-floor margin is a legitimate
///    near-tie and is reported, not failed, and comparisons stop at the first
///    one.
/// 4. **Tensors.** Up to that first flip, every captured tensor agrees with the
///    oracle at the FP16-activation tier (`Self.atol`/`rtol`), with the oracle
///    anchored to the runner's layer inputs and cache appends so each layer's
///    arithmetic is judged on its own; the worst delta per family is printed
///    so the tier is a measurement, not a guess.
/// 5. **Rollouts.** The greedy rollout is token-exact against the oracle up to
///    the first flip; the count is asserted and printed.
/// 6. **Chunked prefill == sequential decode**, bit for bit, at every chunking
///    (the same per-token path).
@Suite(.serialized) struct Glm53ForwardRunnerTests {

    /// The FP16-activation tier. First measured 2026-09-11 on both prompts with
    /// the oracle anchored to the runner's inputs: worst abs 2.07e-2
    /// (`post_attention_layernorm_out`, long decode), streams 1.58e-2, mHC
    /// coefficients 8.6e-3, logits 3.6e-3 — so the gate sits ~2.4x above the
    /// observed worst. Recorded on the family page.
    static let atol: Float = 5e-2
    static let rtol: Float = 5e-2
    /// Below this oracle-reported boundary margin a discrete decision may
    /// legitimately flip at FP16 (biased router score, pooled indexer score).
    static let marginFloor: Float = 1e-2

    private struct Harness {
        let dir: URL
        let ctx: MetalContext
        let model: Model
        let runner: Glm53ForwardRunner
        let logits: MTLBuffer
        let config: ArchConfig

        func cleanup() { try? FileManager.default.removeItem(at: dir) }

        func logitsRow() -> [Float] {
            Glm53ForwardRunner.readFP16(logits, count: config.vocabSize)
        }
    }

    private static func makeHarness(maxContext: Int = 128) throws -> Harness {
        let dir = try Glm53Parity.installToyCheckpoint()
        let ctx = try MetalContext()
        let config = Glm53Parity.archConfig()
        let model = try Glm53Parity.loadModel(at: dir, device: ctx.device)
        let runner = try Glm53ForwardRunner(
            model: model, context: ctx, maxContext: maxContext,
            runtimeConfiguration: RuntimeConfiguration(prefillEnabled: true, forceLogitsHead: true))
        guard let logits = ctx.device.makeBuffer(
                length: config.vocabSize * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return Harness(dir: dir, ctx: ctx, model: model, runner: runner, logits: logits, config: config)
    }

    private static func argmax(_ row: [Float]) -> Int {
        var best = 0
        for i in 1..<row.count where row[i] > row[best] { best = i }
        return best
    }

    private static func halfBuffer(_ device: MTLDevice, _ v: [Float]) -> MTLBuffer {
        let b = device.makeBuffer(length: max(1, v.count) * 2, options: .storageModeShared)!
        let p = b.contents().bindMemory(to: Float16.self, capacity: v.count)
        for i in 0..<v.count { p[i] = Float16(v[i]) }
        return b
    }

    // MARK: - 1. Kernels on identical inputs

    @Test func latentAttentionKernelMatchesItsFormulaDenseAndSelected() throws {
        let ctx = try MetalContext()
        let kernels = try Glm53Kernels(context: ctx)
        var rng = SystemRandomNumberGenerator()
        func rand(_ n: Int, _ scale: Float) -> [Float] {
            (0..<n).map { _ in Float.random(in: -1...1, using: &rng) * scale }
        }
        let heads = 2, dim = 64, cached = 37
        let scale: Float = 1 / Float(dim).squareRoot()
        let qLat = rand(heads * dim, 1.5)
        let latents = rand(cached * dim, 1)
        let qBuf = Self.halfBuffer(ctx.device, qLat)
        let latBuf = Self.halfBuffer(ctx.device, latents)
        let outBuf = ctx.device.makeBuffer(length: heads * dim * 2, options: .storageModeShared)!
        let q16 = qLat.map { Float(Float16($0)) }, lat16 = latents.map { Float(Float16($0)) }

        func expected(rows: [Int]) -> [Float] {
            var out = [Float](repeating: 0, count: heads * dim)
            for h in 0..<heads {
                var scores = rows.map { t -> Float in
                    var dot: Float = 0
                    for d in 0..<dim { dot += q16[h * dim + d] * lat16[t * dim + d] }
                    return dot * scale
                }
                let mx = scores.max()!
                var sum: Float = 0
                for i in scores.indices { scores[i] = expf(scores[i] - mx); sum += scores[i] }
                for (i, t) in rows.enumerated() {
                    for d in 0..<dim { out[h * dim + d] += scores[i] / sum * lat16[t * dim + d] }
                }
            }
            return out
        }
        func run(selected: [Int]?) throws -> [Float] {
            let selBuf = ctx.device.makeBuffer(length: max(1, selected?.count ?? 0) * 4,
                                               options: .storageModeShared)!
            if let selected {
                let p = selBuf.contents().bindMemory(to: UInt32.self, capacity: selected.count)
                for (i, t) in selected.enumerated() { p[i] = UInt32(t) }
            }
            let cb = try #require(ctx.queue.makeCommandBuffer())
            kernels.encodeLatentAttention(commandBuffer: cb, qLatent: qBuf, latents: latBuf, selected: selBuf,
                                          out: outBuf, heads: heads, latentDim: dim, cachedRows: cached,
                                          selectedCount: selected.map { UInt32($0.count) } ?? Glm53Kernels.attendAll,
                                          scale: scale)
            cb.commit(); cb.waitUntilCompleted()
            return Glm53ForwardRunner.readFP16(outBuf, count: heads * dim)
        }
        let dense = try run(selected: nil)
        var worst: Float = 0
        for (a, b) in zip(dense, expected(rows: Array(0..<cached))) { worst = max(worst, abs(a - b)) }
        #expect(worst < 2e-3, "dense latent attention worst abs delta \(worst)")

        let picks = [0, 1, 6, 7, 20, 21, 34, 35, 36]
        let sparse = try run(selected: picks)
        worst = 0
        for (a, b) in zip(sparse, expected(rows: picks)) { worst = max(worst, abs(a - b)) }
        #expect(worst < 2e-3, "selected latent attention worst abs delta \(worst)")
    }

    // MARK: - 2. Install round trip

    @Test func installReproducesTheCheckpointBytesTheRunnerReads() throws {
        let h = try Self.makeHarness()
        defer { h.cleanup() }
        let ckpt = try Glm53ToyCheckpoint()

        // An INT8 projection, dequantized against the checkpoint's own dequant.
        let wq = try h.model.glm53QAProj(layer: 3)
        let expected = try ckpt.matrix("language_model.model.layers.3.self_attn.q_a_proj")
        let rows = Int(wq.shape.0), cols = Int(wq.shape.1)
        #expect(rows == expected.rows && cols == expected.cols)
        let q = wq.buffer.contents().advanced(by: Int(wq.offset)).assumingMemoryBound(to: UInt8.self)
        let s = wq.buffer.contents().advanced(by: Int(wq.scaleOffset)).assumingMemoryBound(to: UInt16.self)
        let b = wq.buffer.contents().advanced(by: Int(wq.biasOffset)).assumingMemoryBound(to: UInt16.self)
        var worst: Float = 0
        for r in 0..<rows {
            for c in 0..<cols {
                let g = r * (cols / 64) + c / 64
                let v = Float(q[r * cols + c]) * Quantization.bf16ToFloat(s[g]) + Quantization.bf16ToFloat(b[g])
                worst = max(worst, abs(v - expected.values[r * cols + c]))
            }
        }
        #expect(worst == 0, "INT8 q_a_proj dequantizes differently from the checkpoint: \(worst)")

        // An FP32 tensor and the FP32 router bias.
        let aLog = try h.model.glm53KDAALog(layer: 0)
        let aPtr = aLog.buffer.contents().advanced(by: Int(aLog.offset)).assumingMemoryBound(to: Float.self)
        let aExpected = try ckpt.floats("language_model.model.layers.0.self_attn.forget_gate.A_log")
        #expect((0..<aExpected.count).map { aPtr[$0] } == aExpected)
        let bias = try h.model.glm53RouterCorrectionBias(layer: 1)
        let bPtr = bias.buffer.contents().advanced(by: Int(bias.offset)).assumingMemoryBound(to: Float.self)
        let bExpected = try ckpt.floats("language_model.model.layers.1.mlp.gate.e_score_correction_bias")
        #expect((0..<bExpected.count).map { bPtr[$0] } == bExpected)

        // The mHC `fn` the runner reads as fp32, against the BF16 checkpoint value.
        let fn = try h.model.residentAsF32(name: "language_model.model.layers.2.attn_hc.fn")
        let fnPtr = fn.buffer.contents().advanced(by: Int(fn.offset)).assumingMemoryBound(to: Float.self)
        let fnExpected = try ckpt.floats("language_model.model.layers.2.attn_hc.fn")
        #expect((0..<fnExpected.count).map { fnPtr[$0] } == fnExpected)
    }

    // MARK: - 3/4/5. The runner against the oracle

    private struct Drift {
        var maxAbs: Float = 0
        var maxRel: Float = 0
        var mismatched = 0
        var count = 0
        var at = ""
    }

    @Test(arguments: Glm53Goldens.Prompt.allCases)
    func runnerAgreesWithTheOracleUpToTheFirstNearTieFlip(prompt: Glm53Goldens.Prompt) async throws {
        let h = try Self.makeHarness()
        defer { h.cleanup() }
        let oracle = try Glm53ReferenceRunner(checkpoint: try Glm53ToyCheckpoint())
        let tokens = try Glm53Goldens.promptTokens(prompt)
        let steps = try Glm53Goldens.decodeSteps()
        // Each layer judged on the runner's own inputs and cache appends, so
        // the comparison isolates one layer's arithmetic rather than compounding
        // FP16 drift through 45 (here 4) layers of recurrence and cache.
        oracle.anchor = { key in h.runner.capture?.floats[key] }

        var table: [String: Drift] = [:]
        func family(_ key: String) -> String {
            key.hasPrefix("layer") ? String(key.dropFirst("layerNN.".count)) : key
        }

        var firstFlip: String? = nil
        var agreedTokens = 0
        var totalSteps = 0
        var token = Int32(tokens[0])
        var oracleToken = tokens[0]

        for step in 0..<(tokens.count + steps) {
            let isPrompt = step < tokens.count
            if isPrompt { token = Int32(tokens[step]); oracleToken = tokens[step] }
            h.runner.capture = .init()
            oracle.capture = .init()
            try await h.runner.produce(token: token, position: step, into: h.logits)
            let oracleLogits = try oracle.step(token: oracleToken)
            let mine = h.runner.capture!
            let theirs = oracle.capture
            let phase = isPrompt ? "prefill" : String(format: "decode.step%02d", step - tokens.count)

            // Discrete decisions first.
            for key in mine.integers.keys.sorted() where family(key) == "router_indices" {
                guard let a = mine.integers[key], let e = theirs.ints[key] else { continue }
                if Set(a) == Set(e) { continue }
                let margin = theirs.margins[key.replacingOccurrences(of: "router_indices", with: "router")]
                let where0 = "\(phase) \(key) (pos \(step)): runner \(a) oracle \(e), oracle margin \(margin.map { String($0) } ?? "n/a")"
                if let margin, margin < Self.marginFloor {
                    firstFlip = firstFlip ?? "\(where0) — near-tie"
                } else {
                    Issue.record("\(where0) — wider than the \(Self.marginFloor) floor")
                    firstFlip = firstFlip ?? "\(where0) — UNJUSTIFIED"
                }
            }
            for key in mine.selections.keys.sorted() {
                guard let a = mine.selections[key], let e = theirs.selections[key] else { continue }
                if a == e { continue }
                let margin = theirs.margins[key.replacingOccurrences(of: "idx_selected", with: "indexer")]
                let where0 = "\(phase) \(key) (pos \(step)): runner \(a.map { "\($0)" } ?? "dense") oracle \(e.map { "\($0)" } ?? "dense"), oracle margin \(margin.map { String($0) } ?? "n/a")"
                if let margin, margin < Self.marginFloor {
                    firstFlip = firstFlip ?? "\(where0) — near-tie"
                } else {
                    Issue.record("\(where0) — wider than the \(Self.marginFloor) floor")
                    firstFlip = firstFlip ?? "\(where0) — UNJUSTIFIED"
                }
            }
            if firstFlip == nil {
                if step == 0 {
                    print("  [glm53 runner vs oracle, \(prompt)] position 0, per key (maxAbs / maxRel):")
                }
                for key in mine.floats.keys.sorted() {
                    guard var a = mine.floats[key], var e = theirs.floats[key] else { continue }
                    if family(key) == "idx_scores", e.count > a.count {
                        // The oracle scores the incomplete tail pool too; the
                        // runner scores only the complete pools it can select.
                        e = Array(e.prefix(a.count))
                    }
                    guard a.count == e.count else {
                        Issue.record("\(phase) \(key): \(a.count) values vs \(e.count)")
                        continue
                    }
                    if family(key) == "router_weights" {
                        let idxKey = key.replacingOccurrences(of: "router_weights", with: "router_indices")
                        if let mi = mine.integers[idxKey], let oi = theirs.ints[idxKey],
                           mi.count == a.count, oi.count == e.count, Set(mi) == Set(oi) {
                            a = zip(mi, a).sorted { $0.0 < $1.0 }.map { $0.1 }
                            e = zip(oi, e).sorted { $0.0 < $1.0 }.map { $0.1 }
                        }
                    }
                    let d = FlashNextDelta.compare(a, e, atol: Self.atol, rtol: Self.rtol)
                    if step == 0 {
                        print(String(format: "      %-36@ %.3e / %.3e", key as NSString, d.maxAbs, d.maxRel))
                    }
                    var drift = table[family(key)] ?? Drift()
                    drift.count += e.count
                    drift.mismatched += d.mismatched
                    if d.maxAbs > drift.maxAbs { drift.maxAbs = d.maxAbs; drift.at = "\(phase).\(key)" }
                    drift.maxRel = max(drift.maxRel, d.maxRel)
                    table[family(key)] = drift
                }
            }
            let runnerNext = Self.argmax(h.logitsRow())
            let oracleNext = Self.argmax(oracleLogits)
            if !isPrompt || step == tokens.count - 1 {
                totalSteps += 1
                if runnerNext == oracleNext && firstFlip == nil { agreedTokens += 1 }
                if runnerNext != oracleNext && firstFlip == nil {
                    var sorted = oracleLogits.sorted(by: >)
                    let margin = sorted.count > 1 ? sorted[0] - sorted[1] : .infinity
                    sorted.removeAll()
                    let where0 = "\(phase) argmax: runner \(runnerNext) oracle \(oracleNext), oracle top-2 margin \(margin)"
                    if margin < Self.marginFloor {
                        firstFlip = "\(where0) — near-tie"
                    } else {
                        firstFlip = "\(where0) — UNJUSTIFIED"
                        Issue.record(Comment(rawValue: firstFlip!))
                    }
                }
            }
            token = Int32(runnerNext)
            oracleToken = runnerNext
        }

        print("  [glm53 runner vs oracle, \(prompt)] first divergence: \(firstFlip ?? "none")")
        print("  greedy agreement before it: \(agreedTokens)/\(totalSteps) generated tokens")
        for key in table.keys.sorted() {
            let d = table[key]!
            print(String(format: "    %-32@ maxAbs %.3e maxRel %.3e  %d/%d outside %.0e  (%@)",
                         key as NSString, d.maxAbs, d.maxRel, d.mismatched, d.count, Self.atol, d.at as NSString))
        }
        for (fam, d) in table {
            #expect(d.mismatched == 0, "\(fam): \(d.mismatched)/\(d.count) outside the FP16 tier, worst \(d.maxAbs) at \(d.at)")
        }
        #expect(agreedTokens >= 1, "the runner's first generated token must match the oracle")
    }

    // MARK: - 6. Chunked prefill == sequential decode

    @Test func chunkedPrefillEqualsSequentialDecodeBitForBit() async throws {
        let h = try Self.makeHarness()
        defer { h.cleanup() }
        let tokens = try Glm53Goldens.promptTokens(.long).map { Int32($0) }

        var sequential: [[Float]] = []
        h.runner.reset()
        for (p, t) in tokens.enumerated() {
            try await h.runner.produce(token: t, position: p, into: h.logits)
            sequential.append(h.logitsRow())
        }
        var seqDecode: [[Float]] = []
        var next = Int32(Self.argmax(sequential.last!))
        for s in 0..<4 {
            try await h.runner.produce(token: next, position: tokens.count + s, into: h.logits)
            seqDecode.append(h.logitsRow())
            next = Int32(Self.argmax(seqDecode.last!))
        }

        // Odd boundaries that split indexer pools (kpool 2) and the conv tail.
        for chunks in [[13, 1, 17, 17], [5, 43], [48]] {
            h.runner.reset()
            var start = 0
            var lastLogits: [Float] = []
            for (i, n) in chunks.enumerated() {
                let slice = tokens[start..<(start + n)]
                let result = try await h.runner.prefillChunked(
                    tokens: slice, startPosition: start, outputMode: .logits,
                    config: .production(chunkTokens: 32),
                    into: h.logits, onProgress: { _ in })
                #expect(result.newPosition == start + n)
                start += n
                if i == chunks.count - 1 { lastLogits = h.logitsRow() }
            }
            #expect(lastLogits == sequential.last!, "chunking \(chunks): final prompt logits differ")
            var nextC = Int32(Self.argmax(lastLogits))
            for s in 0..<4 {
                try await h.runner.produce(token: nextC, position: tokens.count + s, into: h.logits)
                let row = h.logitsRow()
                #expect(row == seqDecode[s], "chunking \(chunks): decode step \(s) differs")
                nextC = Int32(Self.argmax(row))
            }
        }
    }

    /// The dense A/B knob: while the sparse layer's cache holds at most
    /// `index_topk` tokens the indexer's selection is exhaustive, so the two
    /// arms are the same computation; the first position past that budget is
    /// refused rather than silently attending more than the model does.
    @Test func denseSelectionKnobMatchesTheIndexerBelowTheBudgetAndRefusesAbove() async throws {
        let h = try Self.makeHarness()
        defer { h.cleanup() }
        let tokens = try Glm53Goldens.promptTokens(.short).map { Int32($0) }
        let denseExact = h.config.compressedAttention.indexTopK   // positions 0..<4

        var indexerArm: [[Float]] = []
        h.runner.reset()
        for p in 0..<denseExact {
            try await h.runner.produce(token: tokens[p], position: p, into: h.logits)
            indexerArm.append(h.logitsRow())
        }

        h.runner.denseSelectionForAB = true
        defer { h.runner.denseSelectionForAB = false }
        h.runner.reset()
        for p in 0..<denseExact {
            try await h.runner.produce(token: tokens[p], position: p, into: h.logits)
            #expect(h.logitsRow() == indexerArm[p], "position \(p): the arms differ")
        }
        await #expect(throws: Glm53ForwardRunnerError.self) {
            try await h.runner.produce(token: tokens[denseExact], position: denseExact, into: h.logits)
        }
    }

    @Test func factoryDispatchesTheFamilyToItsRunner() throws {
        let dir = try Glm53Parity.installToyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Glm53Parity.loadModel(at: dir, device: ctx.device)
        let runtime = try ForwardRunnerFactory.make(model: model, context: ctx, maxContext: 64)
        #expect(runtime.producer is Glm53ForwardRunner)
        #expect(runtime.executedPrefillMode == .chunked)
    }
}
