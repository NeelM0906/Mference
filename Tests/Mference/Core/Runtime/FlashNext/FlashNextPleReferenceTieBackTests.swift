import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// Two claims, discharged against the CPU reference forward:
///
/// 1. **`FlashNextPleHash` is production code** — unlike the other oracles in
///    this directory it lives in `Sources/`, because the runner needs it at run
///    time. So this suite gates the shipping hash directly against the runner's
///    captured `ple_ngram_row_ids`, in prefill and through cached decode where
///    the two-token id history is what the shift reads.
/// 2. **`FlashNextPleReference` is a faithful transcription** of the runner's
///    `pleBlock`, replayed from captures: the PLE layer's input stream is the
///    previous layer's `stream_out`, and its output is `ple_out`.
///
/// No-ops when `scratch/qwen4exp-toy-ckpt-prodlayout/` is absent.
@Suite struct FlashNextPleReferenceTieBackTests {

    private struct Fixture {
        let config: ArchConfig
        let weights: FlashNextWeights
        let runner: FlashNextReferenceRunner
        let pleLayer: Int
        let directory: URL
    }

    private static func makeFixture() throws -> Fixture? {
        guard FlashNextParity.checkpointIsPresent else { return nil }
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        let dir = try FlashNextParity.installToyCheckpoint()
        let config = FlashNextParity.archConfig()
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: config, streamingMode: .resident)
        let weights = FlashNextWeights(model: model)
        guard let pleLayer = config.flashNext.pleLayerIndices.first else { return nil }
        return Fixture(config: config, weights: weights,
                       runner: try FlashNextReferenceRunner(weights: weights),
                       pleLayer: pleLayer, directory: dir)
    }

    private static func hash(_ p: FlashNextWeights.PLEWeights,
                             eos: Int) -> FlashNextPleHash {
        FlashNextPleHash(multipliers: p.multipliers,
                         headOffsets: p.headOffsets,
                         headVocabSizes: p.headVocabSizes,
                         eosTokenID: eos)
    }

    // MARK: - 1. The shipping hash

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func productionHashMatchesTheReferenceRowIDsInPrefill(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let f = try Self.makeFixture() else { return }
        defer { try? FileManager.default.removeItem(at: f.directory) }
        let ple = try f.weights.ple(layer: f.pleLayer)
        let hash = Self.hash(ple, eos: f.config.flashNext.pleEosTokenID)

        let tokens = try FlashNextGoldens.promptTokens(prompt)
        var capture: FlashNextReferenceRunner.Capture? = .init()
        _ = try f.runner.step(tokens: tokens, capture: &capture)
        let key = FlashNextGoldens.layerKey(f.pleLayer) + ".ple_ngram_row_ids"
        let expected = try #require(capture?.integers[key])

        // Prefill starts from a fresh history: `context_len` copies of EOS, not
        // zeros. Only the trailing `tokens.count` rows are the prompt's.
        let window = hash.initialHistory() + tokens
        let rows = hash.rowIDs(window: window)
        let kept = Array(rows[(rows.count - tokens.count)...])
        #expect(kept == expected, "\(prompt.rawValue) prefill n-gram row ids")
    }

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func productionHashMatchesTheReferenceRowIDsThroughCachedDecode(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let f = try Self.makeFixture() else { return }
        defer { try? FileManager.default.removeItem(at: f.directory) }
        let ple = try f.weights.ple(layer: f.pleLayer)
        let hash = Self.hash(ple, eos: f.config.flashNext.pleEosTokenID)
        let key = FlashNextGoldens.layerKey(f.pleLayer) + ".ple_ngram_row_ids"

        var history = hash.initialHistory()
        var none: FlashNextReferenceRunner.Capture?
        let prompt0 = try FlashNextGoldens.promptTokens(prompt)
        var logits = try f.runner.step(tokens: prompt0, capture: &none)
        history = Array((history + prompt0).suffix(hash.historyLength))
        var next = f.runner.greedyToken(logits)

        for step in 0..<8 {
            var capture: FlashNextReferenceRunner.Capture? = .init()
            logits = try f.runner.step(tokens: [next], capture: &capture)
            let expected = try #require(capture?.integers[key])

            // The decode window is the cached history plus the new token; the
            // shift runs over that window only.
            let window = history + [next]
            let rows = hash.rowIDs(window: window)
            #expect([rows[rows.count - 1]] == expected,
                    "\(prompt.rawValue) decode step \(step) n-gram row ids")

            history = Array(window.suffix(hash.historyLength))
            next = f.runner.greedyToken(logits)
        }
    }

    /// The EOS-aware shift is the part a naive "previous two tokens" would get
    /// wrong: a segment never looks back across its own EOS, and the EOS token
    /// is the LAST token of its own segment, not the first of the next.
    @Test func shiftRightIgnoringEOSKeepsSegmentsClosed() throws {
        let hash = FlashNextPleHash(multipliers: [1, 2, 3],
                                    headOffsets: [0],
                                    headVocabSizes: [7],
                                    eosTokenID: 9)
        //           i:  0  1  2  3  4  5
        let ids = [4, 9, 5, 6, 9, 7]
        // shift 1: position 0 has nothing before it; position 1 reads 4;
        // position 2 is the first of a new segment (the EOS at 1 closed the
        // previous one) so it reads EOS; 3 reads 5; 4 reads 6; 5 opens a new
        // segment after the EOS at 4, so EOS again.
        #expect(hash.shiftRightIgnoringEOS(ids, by: 1) == [9, 4, 9, 5, 6, 9])
        // shift 2: only positions with two same-segment predecessors resolve.
        #expect(hash.shiftRightIgnoringEOS(ids, by: 2) == [9, 9, 9, 9, 5, 9])
        #expect(hash.shiftRightIgnoringEOS(ids, by: 0) == ids)
    }

    // MARK: - 2. The mixing-block oracle

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func oracleReproducesTheReferenceRunnersPleOutput(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let f = try Self.makeFixture() else { return }
        defer { try? FileManager.default.removeItem(at: f.directory) }
        // The PLE layer is not layer 0, so its input stream is the previous
        // layer's captured output — no unobserved state in between.
        guard f.pleLayer > 0 else { return }

        let tokens = try FlashNextGoldens.promptTokens(prompt)
        var capture: FlashNextReferenceRunner.Capture? = .init()
        _ = try f.runner.step(tokens: tokens, capture: &capture)
        let recorded = try #require(capture)

        let reference = try Self.makeReference(f)
        let key = FlashNextGoldens.layerKey(f.pleLayer) + "."
        let out = reference.oracle.mix(
            hyper: try #require(recorded.floats[
                FlashNextGoldens.layerKey(f.pleLayer - 1) + ".stream_out"]),
            embeds: try #require(recorded.floats[key + "ple_ngram_embeds"]),
            w: reference.weights, rows: tokens.count)
        #expect(out == (try #require(recorded.floats[key + "ple_out"])),
                "\(prompt.rawValue) PLE output")
    }

    /// The same, carried through cached decode — which is where the nine-row
    /// conv state actually does something.
    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func oracleReproducesThePleOutputThroughCachedDecode(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let f = try Self.makeFixture() else { return }
        defer { try? FileManager.default.removeItem(at: f.directory) }
        guard f.pleLayer > 0 else { return }

        let reference = try Self.makeReference(f)
        let key = FlashNextGoldens.layerKey(f.pleLayer) + "."
        let previous = FlashNextGoldens.layerKey(f.pleLayer - 1) + ".stream_out"

        // Prefill through both the runner and the oracle so their conv states
        // start the decode in the same place.
        var capture: FlashNextReferenceRunner.Capture? = .init()
        let prompt0 = try FlashNextGoldens.promptTokens(prompt)
        var logits = try f.runner.step(tokens: prompt0, capture: &capture)
        _ = reference.oracle.mix(
            hyper: try #require(capture?.floats[previous]),
            embeds: try #require(capture?.floats[key + "ple_ngram_embeds"]),
            w: reference.weights, rows: prompt0.count)
        var next = f.runner.greedyToken(logits)

        for step in 0..<8 {
            var stepCapture: FlashNextReferenceRunner.Capture? = .init()
            logits = try f.runner.step(tokens: [next], capture: &stepCapture)
            let out = reference.oracle.mix(
                hyper: try #require(stepCapture?.floats[previous]),
                embeds: try #require(stepCapture?.floats[key + "ple_ngram_embeds"]),
                w: reference.weights, rows: 1)
            #expect(out == (try #require(stepCapture?.floats[key + "ple_out"])),
                    "\(prompt.rawValue) PLE output at decode step \(step)")
            next = f.runner.greedyToken(logits)
        }
    }

    private static func makeReference(_ f: Fixture) throws
        -> (oracle: FlashNextPleReference, weights: FlashNextPleReference.Weights) {
        let p = try f.weights.ple(layer: f.pleLayer)
        let fn = f.config.flashNext
        let geometry = FlashNextHyperConnectionReference.Geometry(
            hidden: f.config.hiddenSize, hcCount: fn.hcCount,
            lowRank: fn.hcLowRank, eps: 1e-6)
        return (FlashNextPleReference(geometry: geometry,
                                      convKernel: fn.pleConvKernelSize,
                                      dilation: p.multipliers.count),
                .init(keyProj: p.keyProj, valueProj: p.valueProj, conv: p.conv,
                      normKey: p.normKey, normQuery: p.normQuery,
                      normConv: p.normConv))
    }
}
