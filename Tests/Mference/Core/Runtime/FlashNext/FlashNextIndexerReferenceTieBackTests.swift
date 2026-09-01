import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// `FlashNextIndexerReference` and `FlashNextAttentionReference` are the oracles
/// the QSA indexer and gated-attention kernels are gated against. Both are
/// transcriptions of `FlashNextReferenceRunner`, and a transcription is a claim:
/// this suite discharges it.
///
/// The runner is replayed over the toy install with capture on. For every
/// full-attention layer its captured block input `attn_hc_mixed` is pushed
/// through the indexer oracle, whose selections must equal the runner's captured
/// `indexer_selected` **exactly** (they are integer sets — there is no
/// tolerance), and then through the attention oracle, whose output must equal the
/// captured `block_out` **bit for bit**.
///
/// The last test pins the property the GPU caches are built on: pooled block keys
/// are immutable, so prefilling `n-1` tokens and then stepping the last one has
/// to produce the same selection for that token as prefilling all `n` at once.
///
/// No-ops when `scratch/qwen4exp-toy-ckpt-prodlayout/` is absent.
@Suite struct FlashNextIndexerReferenceTieBackTests {

    private struct Fixture {
        let config: ArchConfig
        let weights: FlashNextWeights
        let capture: FlashNextReferenceRunner.Capture
        let tokens: [Int]
        let directory: URL
    }

    private static func load(_ prompt: FlashNextGoldens.Prompt) throws -> Fixture? {
        guard FlashNextParity.checkpointIsPresent else { return nil }
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        let dir = try FlashNextParity.installToyCheckpoint()
        let config = FlashNextParity.archConfig()
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: config, streamingMode: .resident)
        let weights = FlashNextWeights(model: model)
        let runner = try FlashNextReferenceRunner(weights: weights)
        let tokens = try FlashNextGoldens.promptTokens(prompt)
        var capture: FlashNextReferenceRunner.Capture? = .init()
        _ = try runner.step(tokens: tokens, capture: &capture)
        return Fixture(config: config, weights: weights,
                       capture: capture!, tokens: tokens, directory: dir)
    }

    private static func indexerGeometry(_ c: ArchConfig)
        -> FlashNextIndexerReference.Geometry {
        let fn = c.flashNext
        return .init(numHeads: fn.indexerNumHeads,
                     numKVHeads: fn.indexerNumKVHeads,
                     headDim: fn.indexerHeadDim,
                     compressRatio: fn.indexerCompressRatio,
                     blockBudget: fn.indexerBlockBudget,
                     rotaryDim: Int(Double(c.fullHeadDim) * c.partialRotaryFactor),
                     theta: Float(c.fullRopeTheta),
                     eps: 1e-6)
    }

    private static func attentionGeometry(_ c: ArchConfig)
        -> FlashNextAttentionReference.Geometry {
        .init(numHeads: c.numHeads,
              numKVHeads: c.numFullKVHeads,
              headDim: c.fullHeadDim,
              rotaryDim: Int(Double(c.fullHeadDim) * c.partialRotaryFactor),
              theta: Float(c.fullRopeTheta),
              eps: 1e-6)
    }

    // MARK: - 1. Selections are exact

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func indexerOracleReproducesTheReferenceRunnersSelections(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let f = try Self.load(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: f.directory) }
        let g = Self.indexerGeometry(f.config)
        var checkedLayers = 0

        for layer in 0..<f.config.numLayers where !f.config.layerIsLinear(layer) {
            let key = FlashNextGoldens.layerKey(layer) + "."
            let mixed = try #require(f.capture.floats[key + "attn_hc_mixed"])
            let expected = try #require(f.capture.integers[key + "indexer_selected"])
            let w = try f.weights.attention(layer: layer)
            let result = FlashNextIndexerReference.run(
                x: mixed, hidden: f.config.hiddenSize,
                rows: f.tokens.count, startPosition: 0,
                indexerQK: w.indexerQK, qNorm: w.indexerQNorm,
                kNorm: w.indexerKNorm, rawKeys: [], g: g)
            #expect(result.selected == expected,
                    "\(prompt.rawValue) L\(layer) indexer selections")
            checkedLayers += 1
        }
        #expect(checkedLayers > 0, "the toy has no full-attention layers")
    }

    // MARK: - 2. Attention output is bit-identical

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func attentionOracleReproducesTheReferenceRunnersBlockOutput(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let f = try Self.load(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: f.directory) }
        let g = Self.attentionGeometry(f.config)
        let scale = 1 / sqrtf(Float(f.config.fullHeadDim))
        var checkedLayers = 0

        for layer in 0..<f.config.numLayers where !f.config.layerIsLinear(layer) {
            let key = FlashNextGoldens.layerKey(layer) + "."
            let mixed = try #require(f.capture.floats[key + "attn_hc_mixed"])
            let selected = try #require(f.capture.integers[key + "indexer_selected"])
            let expected = try #require(f.capture.floats[key + "block_out"])
            let a = try f.weights.attention(layer: layer)
            let result = FlashNextAttentionReference.run(
                x: mixed, hidden: f.config.hiddenSize,
                rows: f.tokens.count, startPosition: 0,
                w: .init(q: a.q, k: a.k, v: a.v, o: a.o,
                         qNorm: a.qNorm, kNorm: a.kNorm),
                selected: selected, keys: [], values: [],
                scale: scale, g: g)
            #expect(result.out == expected,
                    "\(prompt.rawValue) L\(layer) block_out")
            checkedLayers += 1
        }
        #expect(checkedLayers > 0)
    }

    // MARK: - 3. Pooled block keys are immutable across a prefill/decode split

    /// The GPU caches pool a block once, when its last token lands, and never
    /// touch it again. That is only sound if selection is split-invariant, so
    /// this splits the prompt: prefill `n-1` tokens, then step the last one, and
    /// require the final token's selection to equal what a single `n`-token pass
    /// produced. A lag-one policy would fail here, which is exactly why the
    /// design doc rules one out.
    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func selectionIsInvariantUnderAPrefillDecodeSplit(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let f = try Self.load(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: f.directory) }
        let g = Self.indexerGeometry(f.config)
        let n = f.tokens.count
        guard n >= 2 else { return }
        var checkedLayers = 0

        for layer in 0..<f.config.numLayers where !f.config.layerIsLinear(layer) {
            let key = FlashNextGoldens.layerKey(layer) + "."
            let mixed = try #require(f.capture.floats[key + "attn_hc_mixed"])
            let expected = try #require(f.capture.integers[key + "indexer_selected"])
            let w = try f.weights.attention(layer: layer)
            let hidden = f.config.hiddenSize

            let head = FlashNextIndexerReference.run(
                x: Array(mixed[0..<((n - 1) * hidden)]), hidden: hidden,
                rows: n - 1, startPosition: 0,
                indexerQK: w.indexerQK, qNorm: w.indexerQNorm,
                kNorm: w.indexerKNorm, rawKeys: [], g: g)
            #expect(head.selected == Array(expected[0..<(n - 1)]),
                    "\(prompt.rawValue) L\(layer) prefill half")

            let tail = FlashNextIndexerReference.run(
                x: Array(mixed[((n - 1) * hidden)...]), hidden: hidden,
                rows: 1, startPosition: n - 1,
                indexerQK: w.indexerQK, qNorm: w.indexerQNorm,
                kNorm: w.indexerKNorm, rawKeys: head.rawKeys, g: g)
            #expect(tail.selected == [expected[n - 1]],
                    "\(prompt.rawValue) L\(layer) decode step")
            checkedLayers += 1
        }
        #expect(checkedLayers > 0)
    }
}
