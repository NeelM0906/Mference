import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// `FlashNextForwardRunner` against `FlashNextReferenceRunner` on the BF16 toy
/// install — the production layer loop versus the float32 CPU forward that
/// passes all six golden parity gates.
///
/// # What these gates found, and why they are shaped this way
///
/// The runner reproduces the reference exactly for a while and then diverges at
/// a near-tie, on the long prompt earlier than on the short one. That is
/// measured, attributed, and pinned here rather than tuned away. The chain, from
/// `FlashNextForwardRunnerAttributionTests` and the layer-0 trace this suite
/// prints:
///
/// 1. Block inputs (`attn_hc_mixed`, `mlp_hc_mixed`) carry the FP16 storage
///    floor and nothing more — 4.7e-4 max-abs on a magnitude of 1.1, i.e. 4e-4
///    relative, at layer 0 where the stream reaching them is still bit-exact.
/// 2. Fed that same input, the GPU MoE block reproduces the CPU oracle to
///    **7.8e-7** (`FlashNextForwardRunnerAttributionTests`). The kernels are not
///    the source.
/// 3. The **toy** is what amplifies. Its block outputs are ~1e-3 against block
///    inputs of ~1.1 — three orders of magnitude of attenuation — while the
///    block's Jacobian stays O(1). A 4.7e-4 input difference therefore comes out
///    as a ~4e-4 output difference on an output whose own magnitude is ~8.8e-4,
///    and lands in a residual stream of magnitude ~4e-2 as about 1% per layer.
/// 4. By the last layer the stream is ~2% off, which is enough to reorder a
///    top-2 router boundary or a top-2 indexer block boundary wherever the two
///    candidates sit within that of each other.
///
/// A trained model does not attenuate its blocks by 1000x, so the amplification
/// in step 3 is a property of the toy's random initialization rather than of the
/// architecture. It is still real here, so these gates assert what actually
/// holds — exact agreement up to the first near-tie flip, and where that flip
/// is — which makes them regression gates rather than aspirations.
///
/// **The family gate stays down on the strength of this.** The port's rule is
/// token-exact or report; the long prompt is 7/8.
///
/// The install is opened in `pread` streaming mode with 16 slots, so the routed
/// experts go through the real LFU slot cache rather than a resident mapping.
///
/// No-ops when `scratch/qwen4exp-toy-ckpt-prodlayout/` is absent.
@Suite struct FlashNextForwardRunnerParityTests {

    private static let slotCount = 16
    private static let decodeSteps = 8

    /// The measured position of the first discrete-decision divergence, per
    /// prompt. A regression that flips *earlier* than this fails; one that flips
    /// later, or not at all, does not.
    private static func firstDivergenceFloor(
        _ prompt: FlashNextGoldens.Prompt) -> Int {
        switch prompt {
        case .short: return 17
        case .long: return 11
        }
    }

    /// Measured greedy-rollout agreement over `decodeSteps` generated tokens.
    private static func expectedAgreement(
        _ prompt: FlashNextGoldens.Prompt) -> Int {
        switch prompt {
        case .short: return 8
        case .long: return 7
        }
    }

    private struct Harness {
        let context: MetalContext
        let config: ArchConfig
        let oracle: FlashNextReferenceRunner
        let runner: FlashNextForwardRunner
        let logits: MTLBuffer
        let tokens: [Int]
        let directory: URL
    }

    private static func make(_ prompt: FlashNextGoldens.Prompt,
                             maxContext: Int) throws -> Harness? {
        guard FlashNextParity.checkpointIsPresent else { return nil }
        let context = try MetalContext()
        let dir = try FlashNextParity.installToyCheckpoint()
        let config = FlashNextParity.archConfig()
        let streamed = try Model.load(directoryURL: dir, device: context.device,
                                      expecting: config,
                                      streamingMode: .pread(slotCount: slotCount))
        let resident = try Model.load(directoryURL: dir, device: context.device,
                                      expecting: config, streamingMode: .resident)
        let oracle = try FlashNextReferenceRunner(
            weights: FlashNextWeights(model: resident))
        let runner = try FlashNextForwardRunner(model: streamed, context: context,
                                                maxContext: maxContext)
        guard let logits = context.device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else { return nil }
        return Harness(context: context, config: config, oracle: oracle,
                       runner: runner, logits: logits,
                       tokens: try FlashNextGoldens.promptTokens(prompt),
                       directory: dir)
    }

    /// One position's comparison of every discrete decision the layer loop makes.
    private struct DecisionDiff {
        var routerOrder: [String] = []
        var routerSet: [String] = []
        var indexer: [String] = []
        var ple: [String] = []
        var isClean: Bool { routerOrder.isEmpty && indexer.isEmpty && ple.isEmpty }
    }

    private static func compareDecisions(
        _ actual: FlashNextForwardRunner.Capture,
        _ expected: FlashNextReferenceRunner.Capture,
        config c: ArchConfig) -> DecisionDiff {
        var diff = DecisionDiff()
        let pleLayer = c.flashNext.pleLayerIndices.first ?? -1
        for layer in 0..<c.numLayers {
            let key = FlashNextGoldens.layerKey(layer) + "."
            if let e = expected.integers[key + "router_indices"],
               let a = actual.integers[key + "router_indices"] {
                if a != e { diff.routerOrder.append("L\(layer) \(a) vs \(e)") }
                if Set(a.flatMap { $0 }) != Set(e.flatMap { $0 }) {
                    diff.routerSet.append("L\(layer) \(a) vs \(e)")
                }
            }
            if layer == pleLayer,
               let e = expected.integers[key + "ple_ngram_row_ids"],
               let a = actual.integers[key + "ple_ngram_row_ids"], a != e {
                diff.ple.append("L\(layer)")
            }
            if !c.layerIsLinear(layer),
               let e = expected.integers[key + "indexer_selected"],
               let a = actual.integers[key + "indexer_selected"], a != e {
                diff.indexer.append("L\(layer) \(a) vs \(e)")
            }
        }
        return diff
    }

    // MARK: - Gate 1: discrete decisions

    /// Router expert ids, indexer selections and PLE n-gram row ids, at every
    /// prefill position and through eight cached decode steps, on both prompts.
    ///
    /// Both sides consume the SAME token stream — the prompt, then the oracle's
    /// greedy continuations — so this isolates the layer loop from the rollout
    /// divergence gate 3 measures.
    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func discreteDecisionsAgreeUntilTheFirstNearTieFlip(
        prompt: FlashNextGoldens.Prompt) async throws {
        let steps = Self.decodeSteps
        guard let h = try Self.make(prompt, maxContext: 64) else { return }
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let c = h.config

        var stream = h.tokens
        var index = 0
        var firstDivergence: Int?
        var routerOrderFlips = 0
        var routerSetFlips = 0
        var indexerFlips = 0
        var pleFlips = 0

        while index < h.tokens.count + steps {
            let token = stream[index]
            var oracleCapture: FlashNextReferenceRunner.Capture? = .init()
            let oracleLogits = try h.oracle.step(tokens: [token],
                                                 capture: &oracleCapture)
            h.runner.capture = .init()
            try await h.runner.produce(token: Int32(token), position: index,
                                       into: h.logits)
            let diff = Self.compareDecisions(try #require(h.runner.capture),
                                             try #require(oracleCapture),
                                             config: c)
            routerOrderFlips += diff.routerOrder.count
            routerSetFlips += diff.routerSet.count
            indexerFlips += diff.indexer.count
            pleFlips += diff.ple.count

            // The PLE row ids are a pure 64-bit integer hash of token ids. Both
            // sides see the same ids, and there is no float anywhere in that
            // path, so they must agree at EVERY position.
            #expect(diff.ple.isEmpty,
                    Comment(rawValue: "\(prompt.rawValue) t\(index) PLE row ids "
                                + "diverged at \(diff.ple)"))

            if !diff.isClean, firstDivergence == nil {
                firstDivergence = index
                print("flashnext decisions (\(prompt.rawValue)): first divergence "
                        + "at position \(index) — router \(diff.routerOrder), "
                        + "indexer \(diff.indexer)")
            }
            if firstDivergence == nil {
                #expect(diff.isClean,
                        Comment(rawValue: "\(prompt.rawValue) t\(index) decisions"))
            }

            index += 1
            if index >= stream.count {
                stream.append(h.oracle.greedyToken(oracleLogits))
            }
        }
        h.runner.capture = nil

        let total = h.tokens.count + steps
        print("flashnext decisions (\(prompt.rawValue)) over \(total) positions x "
                + "\(c.numLayers) layers: \(routerOrderFlips) router rank flips "
                + "(\(routerSetFlips) of which changed the expert SET), "
                + "\(indexerFlips) indexer selection flips, \(pleFlips) PLE flips; "
                + "first divergence at "
                + "\(firstDivergence.map(String.init) ?? "never")")

        // The regression gate: exact agreement must reach at least as far as it
        // does today.
        let floor = Self.firstDivergenceFloor(prompt)
        #expect((firstDivergence ?? Int.max) >= floor,
                Comment(rawValue: "\(prompt.rawValue) first decision divergence "
                            + "moved earlier: "
                            + "\(String(describing: firstDivergence)) < \(floor)"))
    }

    // MARK: - Gate 2: full-forward tensor drift

    /// Measured over the positions **before** the first discrete divergence.
    /// After a flip the two runs are on different trajectories and their tensors
    /// legitimately differ; measuring across that boundary would report the
    /// consequences of a decision rather than the arithmetic.
    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func fullForwardTensorDriftIsTheFP16FloorAmplifiedByTheToy(
        prompt: FlashNextGoldens.Prompt) async throws {
        guard let h = try Self.make(prompt, maxContext: 64) else { return }
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let c = h.config

        struct Drift {
            var maxAbs: Float = 0
            var referenceMagnitude: Float = 0
            var worstAt = ""
        }
        var drift: [String: Drift] = [:]
        var layerZero: [String: Drift] = [:]

        func measure(_ point: String, _ at: String, _ isLayerZero: Bool,
                     _ actual: [Float], _ expected: [Float]) {
            guard actual.count == expected.count, !expected.isEmpty else {
                Issue.record(Comment(rawValue:
                    "capture \(point) length \(actual.count) vs \(expected.count)"))
                return
            }
            var maxDiff: Float = 0
            var norm: Float = 0
            for i in 0..<actual.count {
                maxDiff = max(maxDiff, abs(actual[i] - expected[i]))
                norm = max(norm, abs(expected[i]))
            }
            func fold(_ table: inout [String: Drift]) {
                var entry = table[point] ?? Drift()
                if maxDiff > entry.maxAbs {
                    entry = Drift(maxAbs: maxDiff, referenceMagnitude: norm,
                                  worstAt: at)
                }
                table[point] = entry
            }
            fold(&drift)
            if isLayerZero { fold(&layerZero) }
        }

        var measuredPositions = 0
        for (index, token) in h.tokens.enumerated() {
            var oracleCapture: FlashNextReferenceRunner.Capture? = .init()
            _ = try h.oracle.step(tokens: [token], capture: &oracleCapture)
            h.runner.capture = .init()
            try await h.runner.produce(token: Int32(token), position: index,
                                       into: h.logits)
            let recorded = try #require(oracleCapture)
            let actual = try #require(h.runner.capture)
            guard Self.compareDecisions(actual, recorded, config: c).isClean else {
                break
            }
            measuredPositions += 1

            measure("embed_out", "t\(index)", false,
                    try #require(actual.floats["embed_out"]),
                    try #require(recorded.floats["embed_out"]))
            for layer in 0..<c.numLayers {
                let key = FlashNextGoldens.layerKey(layer) + "."
                for point in ["attn_hc_stream_in", "attn_hc_mixed", "block_out",
                              "mlp_hc_stream_in", "mlp_hc_mixed", "moe_out",
                              "stream_out"] {
                    guard let e = recorded.floats[key + point],
                          let a = actual.floats[key + point] else { continue }
                    measure(point, "t\(index) L\(layer)", layer == 0, a, e)
                }
            }
            measure("last_hidden_state", "t\(index)", false,
                    try #require(actual.floats["last_hidden_state"]),
                    try #require(recorded.floats["last_hidden_state"]))
            measure("logits", "t\(index)", false,
                    try #require(actual.floats["logits"]),
                    try #require(recorded.floats["logits"]))
        }
        h.runner.capture = nil
        #expect(measuredPositions > 0, "no decision-aligned positions to measure")

        func report(_ title: String, _ table: [String: Drift]) {
            print("flashnext \(title) (\(prompt.rawValue)), \(measuredPositions) "
                    + "decision-aligned positions:")
            for point in table.keys.sorted() {
                let d = table[point]!
                let padded = point.padding(toLength: 20, withPad: " ", startingAt: 0)
                print("  \(padded) max-abs \(String(format: "%.3e", d.maxAbs))"
                        + "  |ref| \(String(format: "%.3e", d.referenceMagnitude))"
                        + "  at \(d.worstAt)")
            }
        }
        // Layer 0 is the honest measurement of the kernels: the stream reaching
        // it is bit-exact, so nothing has accumulated.
        report("layer-0 drift", layerZero)
        report("full-forward drift", drift)

        // Layer 0's block inputs are the FP16 storage floor and nothing more.
        // This is the tolerance to state: ~4e-4 relative on a magnitude of ~1.1.
        for point in ["attn_hc_mixed", "mlp_hc_mixed"] {
            guard let d = layerZero[point] else { continue }
            let relative = d.maxAbs / max(d.referenceMagnitude, 1e-6)
            #expect(relative < 2e-3,
                    Comment(rawValue: "\(prompt.rawValue) layer-0 \(point) "
                                + "relative drift \(relative) is above the FP16 "
                                + "activation floor"))
        }
        // Everything downstream inherits the toy's ~1000x block attenuation; the
        // bound below is what that amplification actually costs, measured.
        for (point, d) in drift {
            #expect(d.maxAbs < 5e-2,
                    Comment(rawValue: "\(prompt.rawValue) \(point) max-abs "
                                + "\(d.maxAbs) at \(d.worstAt) (reference "
                                + "magnitude \(d.referenceMagnitude))"))
        }
    }

    // MARK: - Gate 3: greedy rollout

    /// Each side rolls out on its own tokens: the end-to-end claim, and the one
    /// that decides whether the family gate can come up.
    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func greedyRolloutAgreementIsMeasuredAndAttributed(
        prompt: FlashNextGoldens.Prompt) async throws {
        let steps = Self.decodeSteps
        guard let h = try Self.make(prompt, maxContext: 64) else { return }
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let vocab = h.config.vocabSize

        func argmax(_ logits: [Float]) -> Int {
            var best = 0
            for i in 1..<logits.count where logits[i] > logits[best] { best = i }
            return best
        }

        var oracleTokens: [Int] = []
        var oracleFinal: [[Float]] = []
        for i in 0..<(h.tokens.count + steps - 1) {
            let token = i < h.tokens.count ? h.tokens[i]
                                           : oracleTokens[i - h.tokens.count]
            var capture: FlashNextReferenceRunner.Capture?
            let out = try h.oracle.step(tokens: [token], capture: &capture)
            if i >= h.tokens.count - 1 {
                let row = Array(out.suffix(vocab))
                oracleFinal.append(row)
                oracleTokens.append(argmax(row))
            }
        }

        var runnerTokens: [Int] = []
        var runnerFinal: [[Float]] = []
        for i in 0..<(h.tokens.count + steps - 1) {
            let token = i < h.tokens.count ? h.tokens[i]
                                           : runnerTokens[i - h.tokens.count]
            try await h.runner.produce(token: Int32(token), position: i,
                                       into: h.logits)
            if i >= h.tokens.count - 1 {
                let row = FlashNextForwardRunner.readFP16(h.logits, count: vocab)
                runnerFinal.append(row)
                runnerTokens.append(argmax(row))
            }
        }

        let agreed = zip(oracleTokens, runnerTokens).prefix { $0 == $1 }.count
        print("flashnext greedy rollout (\(prompt.rawValue)): reference "
                + "\(oracleTokens)")
        print("flashnext greedy rollout (\(prompt.rawValue)): runner    "
                + "\(runnerTokens)")
        print("flashnext greedy rollout (\(prompt.rawValue)): "
                + "\(agreed)/\(oracleTokens.count) tokens agree")

        if agreed < oracleTokens.count {
            // METH-01: attribute the flip, do not tune it away. If the
            // reference's own top-2 margin at that step is smaller than the
            // measured logit difference between the two sides, the argmax was
            // never determined at FP16 precision to begin with.
            let o = oracleFinal[agreed].sorted(by: >)
            let r = runnerFinal[agreed].sorted(by: >)
            let noise = zip(oracleFinal[agreed], runnerFinal[agreed])
                .map { abs($0 - $1) }.max() ?? 0
            let margin = o[0] - o[1]
            print("flashnext greedy rollout (\(prompt.rawValue)): diverged at "
                    + "generated token \(agreed) — reference top-2 margin "
                    + "\(margin), runner top-2 margin \(r[0] - r[1]), "
                    + "max-abs logit drift \(noise)")
            #expect(margin < noise,
                    Comment(rawValue: "\(prompt.rawValue) diverged with a top-2 "
                                + "margin of \(margin), WIDER than the \(noise) "
                                + "logit drift — that is not an FP16 tie, it is a "
                                + "wrong answer"))
        }
        // The regression gate: agreement must not get worse.
        #expect(agreed >= Self.expectedAgreement(prompt),
                Comment(rawValue: "\(prompt.rawValue) greedy rollout agreement "
                            + "fell to \(agreed)/\(oracleTokens.count)"))
    }
}
