import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// The `qwen38flashnext` stop/go parity milestone: a float32 CPU forward, built
/// on the real install/loader path, proved module by module against the
/// committed reference goldens.
///
/// # Which golden set, and why
///
/// These gates run against `Tests/Mference/Fixtures/qwen4exp-bf16/`, not the
/// original `qwen4exp/` set. The original goldens were captured from the
/// **float32 weights `Qwen4ExpForCausalLM(cfg)` initialized**, while the
/// checkpoint the harness emits — the only artifact a port can consume — is a
/// lossy **bfloat16** copy of them. Measured, by running the reference twice:
///
/// | | SHORT | LONG |
/// |---|---|---|
/// | logits max-abs, bf16 weights vs fp32 goldens | 1.16e-3 | 3.09e-2 |
///
/// against a recommended gate of `atol = rtol = 1e-4`. It is not only a
/// tolerance problem: under bf16 weights the LONG prompt's router top-k indices
/// flip at layer 2 query 18, `layer03.indexer_selected` changes at query 28, and
/// the LONG cached-decode greedy rollout diverges from token 7 onward
/// (`...28, 48, 36, 14...` becomes `...28, 2, 56, 43...`). No implementation
/// that loads the checkpoint can pass the fp32 integer gates, however correct.
///
/// `Scripts/parity/qwen4exp_make_goldens.py --weight-dtype bf16` re-captures the
/// same 65 tensors and 4 integer files after rounding every parameter through
/// bfloat16, so the goldens describe the weights the checkpoint carries. The
/// fp32 set is untouched and still reproduces byte-identically; it remains the
/// record of the reference's own arithmetic.
///
/// # Skipping
///
/// Every test no-ops when `scratch/qwen4exp-toy-ckpt-prodlayout/` is absent. It
/// is regenerable but not committed:
///
/// ```
/// ./scratch/qwen4exp-parity-venv/bin/python \
///     Scripts/parity/qwen4exp_make_goldens.py --emit-checkpoint
/// ```
@Suite struct FlashNextReferenceParityTests {

    // MARK: - Fixture

    private static func makeRunner(int4RoundTrip: Bool = false) throws
        -> (runner: FlashNextReferenceRunner, cleanup: URL)? {
        guard FlashNextParity.checkpointIsPresent else { return nil }
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        let dir = try FlashNextParity.installToyCheckpoint()
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: FlashNextParity.archConfig(),
                                   streamingMode: .resident)
        let weights = FlashNextWeights(model: model, int4RoundTrip: int4RoundTrip)
        return (try FlashNextReferenceRunner(weights: weights), dir)
    }

    /// One full prefill, capturing everything.
    private static func prefill(_ prompt: FlashNextGoldens.Prompt,
                                int4RoundTrip: Bool = false) throws
        -> (capture: FlashNextReferenceRunner.Capture,
            logits: [Float], cleanup: URL)? {
        guard let made = try makeRunner(int4RoundTrip: int4RoundTrip) else { return nil }
        let tokens = try FlashNextGoldens.promptTokens(prompt)
        var capture: FlashNextReferenceRunner.Capture? = .init()
        let logits = try made.runner.step(tokens: tokens, capture: &capture)
        return (capture!, logits, made.cleanup)
    }

    /// Prefill, then `steps` cached single-token greedy decode calls.
    private static func decode(_ prompt: FlashNextGoldens.Prompt, steps: Int) throws
        -> (tokens: [Int], captures: [FlashNextReferenceRunner.Capture],
            cleanup: URL)? {
        guard let made = try makeRunner() else { return nil }
        let runner = made.runner
        var none: FlashNextReferenceRunner.Capture?
        var logits = try runner.step(
            tokens: try FlashNextGoldens.promptTokens(prompt), capture: &none)
        var generated = [runner.greedyToken(logits)]
        var captures: [FlashNextReferenceRunner.Capture] = []
        for _ in 1..<steps {
            var capture: FlashNextReferenceRunner.Capture? = .init()
            logits = try runner.step(tokens: [generated.last!], capture: &capture)
            captures.append(capture!)
            generated.append(runner.greedyToken(logits))
        }
        return (generated, captures, made.cleanup)
    }

    // MARK: - Gate 1: PLE n-gram ids and row indices

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func gate1_pleNgramRowIDsMatchExactly(prompt: FlashNextGoldens.Prompt) throws {
        guard let run = try Self.prefill(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: run.cleanup) }
        let goldens = try FlashNextGoldens(prompt: prompt, phase: .prefill)
        let key = "layer01.ple_ngram_row_ids"
        let expected = try goldens.intLists(key)
        let actual = try #require(run.capture.integers[key])
        #expect(actual == expected, "\(prompt.rawValue): PLE row ids diverge")

        // The gathered rows must also reconstruct the reference's embedding.
        let (atol, rtol) = try FlashNextGoldens.tolerances()
        let embeds = try #require(run.capture.floats["layer01.ple_ngram_embeds"])
        let delta = FlashNextDelta.compare(embeds,
                                           try goldens.tensor("layer01.ple_ngram_embeds").values,
                                           atol: atol, rtol: rtol)
        #expect(delta.passes, "\(prompt.rawValue) ple_ngram_embeds \(delta.description)")
    }

    // MARK: - Gate 2: indexer selected sets

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func gate2_indexerSelectedSetsMatchExactly(prompt: FlashNextGoldens.Prompt) throws {
        guard let run = try Self.prefill(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: run.cleanup) }
        let goldens = try FlashNextGoldens(prompt: prompt, phase: .prefill)
        for layer in 0..<FlashNextParity.archConfig().numLayers
        where !FlashNextParity.archConfig().layerIsLinear(layer) {
            let prefix = FlashNextGoldens.layerKey(layer)
            for suffix in ["indexer_selected", "indexer_visible"] {
                let key = "\(prefix).\(suffix)"
                let expected = try goldens.intLists(key)
                let actual = try #require(run.capture.integers[key])
                let detail = firstDifference(actual, expected, run.capture)
                #expect(actual == expected,
                        Comment(rawValue: "\(prompt.rawValue) \(key): \(detail)"))
            }
        }
    }

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func gate2_indexerSelectedSetsMatchThroughCachedDecode(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let run = try Self.decode(prompt, steps: 16) else { return }
        defer { try? FileManager.default.removeItem(at: run.cleanup) }
        let goldens = try FlashNextGoldens(prompt: prompt, phase: .decode)
        let config = FlashNextParity.archConfig()
        for layer in 0..<config.numLayers where !config.layerIsLinear(layer) {
            let key = "\(FlashNextGoldens.layerKey(layer)).indexer_selected"
            for (step, capture) in run.captures.enumerated() {
                let expected = try goldens.intLists(key, step: step)
                let actual = try #require(capture.integers[key])
                #expect(actual == expected,
                        "\(prompt.rawValue) \(key) decode step \(step)")
            }
        }
    }

    // MARK: - Gate 3: router top-k

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func gate3_routerTopKIndicesExactAndWeightsInTolerance(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let run = try Self.prefill(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: run.cleanup) }
        let goldens = try FlashNextGoldens(prompt: prompt, phase: .prefill)
        let (atol, rtol) = try FlashNextGoldens.tolerances()
        for layer in 0..<FlashNextParity.archConfig().numLayers {
            let prefix = FlashNextGoldens.layerKey(layer)
            let indices = try #require(run.capture.integers["\(prefix).router_indices"])
            #expect(indices == (try goldens.intLists("\(prefix).router_indices")),
                    "\(prompt.rawValue) \(prefix) router indices")
            let weights = try #require(run.capture.floats["\(prefix).router_weights"])
            let delta = FlashNextDelta.compare(
                weights, try goldens.tensor("\(prefix).router_weights").values,
                atol: atol, rtol: rtol)
            #expect(delta.passes,
                    "\(prompt.rawValue) \(prefix) router weights \(delta.description)")
        }
    }

    // MARK: - Gate 4: per-layer tensors

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func gate4_perLayerTensorsMatchWithinTolerance(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let run = try Self.prefill(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: run.cleanup) }
        let goldens = try FlashNextGoldens(prompt: prompt, phase: .prefill)
        let (atol, rtol) = try FlashNextGoldens.tolerances()
        let config = FlashNextParity.archConfig()

        var firstFailure: String?
        func check(_ key: String) throws {
            guard let expected = goldens.tensors[key] else { return }
            let actual = try #require(run.capture.floats[key], "runner never captured \(key)")
            let delta = FlashNextDelta.compare(actual, expected.values,
                                               atol: atol, rtol: rtol)
            if !delta.passes, firstFailure == nil {
                firstFailure = "\(key): \(delta.description)"
            }
            #expect(delta.passes, "\(prompt.rawValue) \(key) \(delta.description)")
        }

        try check("embed_out")
        for layer in 0..<config.numLayers {
            let p = FlashNextGoldens.layerKey(layer) + "."
            for suffix in ["ple_out", "attn_hc_stream_in", "attn_hc_mixed",
                           "attn_hc_inject", "block_out", "mlp_hc_stream_in",
                           "mlp_hc_mixed", "mlp_hc_inject", "moe_out", "stream_out"] {
                try check(p + suffix)
            }
        }
        try check("last_hidden_state")
        if let firstFailure {
            Issue.record(Comment(rawValue:
                "\(prompt.rawValue): first diverging tensor \(firstFailure)"))
        }
    }

    // MARK: - Gate 5: logits and greedy rollouts

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func gate5_prefillLogitsAndArgmaxMatch(prompt: FlashNextGoldens.Prompt) throws {
        guard let run = try Self.prefill(prompt) else { return }
        defer { try? FileManager.default.removeItem(at: run.cleanup) }
        let goldens = try FlashNextGoldens(prompt: prompt, phase: .prefill)
        let (atol, rtol) = try FlashNextGoldens.tolerances()
        let expected = try goldens.tensor("logits")
        let delta = FlashNextDelta.compare(run.logits, expected.values,
                                           atol: atol, rtol: rtol)
        #expect(delta.passes, "\(prompt.rawValue) logits \(delta.description)")

        let vocab = FlashNextParity.archConfig().vocabSize
        let rows = expected.shape[0]
        var argmax: [Int] = []
        for t in 0..<rows {
            var best = 0
            for i in 1..<vocab
            where run.logits[t * vocab + i] > run.logits[t * vocab + best] { best = i }
            argmax.append(best)
        }
        #expect(argmax == (try goldens.ints("argmax_all_positions")),
                "\(prompt.rawValue) prefill argmax")
    }

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func gate5_cachedDecodeGreedyRolloutMatchesExactly(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let run = try Self.decode(prompt, steps: 16) else { return }
        defer { try? FileManager.default.removeItem(at: run.cleanup) }
        let goldens = try FlashNextGoldens(prompt: prompt, phase: .decode)
        #expect(run.tokens == (try goldens.ints("generated_token_ids")),
                "\(prompt.rawValue) cached greedy rollout")
        #expect(run.tokens == (try FlashNextGoldens.greedyRollout(prompt)),
                "\(prompt.rawValue) rollout vs manifest")

        // Per-step tensors: decode row i is step i + 1.
        let (atol, rtol) = try FlashNextGoldens.tolerances()
        for key in ["last_hidden_state", "step_logits"] {
            guard let expected = goldens.tensors[key] else { continue }
            let source = key == "step_logits" ? "logits" : key
            var actual: [Float] = []
            for capture in run.captures {
                actual.append(contentsOf: try #require(capture.floats[source]))
            }
            let delta = FlashNextDelta.compare(actual, expected.values,
                                               atol: atol, rtol: rtol)
            #expect(delta.passes, "\(prompt.rawValue) decode \(key) \(delta.description)")
        }
    }

    // MARK: - Gate 6: cache equivalence

    /// The cached decode must equal this runner's own re-prefill of the same
    /// context, step for step — the reference proves the same property about
    /// itself (`uncached_rollout_token_ids`), and the port must too.
    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func gate6_cachedDecodeEqualsUncachedRePrefill(
        prompt: FlashNextGoldens.Prompt) throws {
        guard let made = try Self.makeRunner() else { return }
        defer { try? FileManager.default.removeItem(at: made.cleanup) }
        let runner = made.runner
        let (atol, rtol) = try FlashNextGoldens.tolerances()
        let base = try FlashNextGoldens.promptTokens(prompt)
        let steps = 8

        var none: FlashNextReferenceRunner.Capture?
        var logits = try runner.step(tokens: base, capture: &none)
        var context = base
        var cachedTokens: [Int] = []
        var cachedLogits: [[Float]] = []
        cachedTokens.append(runner.greedyToken(logits))
        for _ in 1..<steps {
            logits = try runner.step(tokens: [cachedTokens.last!], capture: &none)
            cachedLogits.append(logits)
            cachedTokens.append(runner.greedyToken(logits))
        }

        // Re-prefill the whole context from a cold cache at every step.
        var uncachedTokens: [Int] = []
        var uncachedLogits: [[Float]] = []
        runner.reset()
        logits = try runner.step(tokens: context, capture: &none)
        uncachedTokens.append(runner.greedyToken(logits))
        for _ in 1..<steps {
            context.append(uncachedTokens.last!)
            runner.reset()
            logits = try runner.step(tokens: context, capture: &none)
            uncachedLogits.append(Array(logits.suffix(FlashNextParity
                .archConfig().vocabSize)))
            uncachedTokens.append(runner.greedyToken(logits))
        }

        #expect(cachedTokens == uncachedTokens,
                "\(prompt.rawValue) cached \(cachedTokens) != re-prefill \(uncachedTokens)")
        for (i, (a, b)) in zip(cachedLogits, uncachedLogits).enumerated() {
            let delta = FlashNextDelta.compare(a, b, atol: atol, rtol: rtol)
            #expect(delta.passes,
                    "\(prompt.rawValue) step \(i + 1) cache equivalence \(delta.description)")
        }
    }

    // MARK: - The dequant caveat, quantified

    /// What an INT4 affine group-64 install would cost, measured rather than
    /// asserted. `FlashNextWeights(int4RoundTrip: true)` pushes every tensor the
    /// production planner would quantize through `Int4AffineEncoder` and back
    /// through `Quantization.dequantizeInt4Affine`, which
    /// `Int4AffineEncoderParityTests` locks bit-for-bit to each other.
    ///
    /// This is deliberately not a pass/fail gate: it is the number that says
    /// why the parity install is BF16 passthrough.
    @Test func int4RoundTripQuantifiesTheDequantCaveat() throws {
        guard let run = try Self.prefill(.short, int4RoundTrip: true) else { return }
        defer { try? FileManager.default.removeItem(at: run.cleanup) }
        let goldens = try FlashNextGoldens(prompt: .short, phase: .prefill)
        let delta = FlashNextDelta.compare(run.logits,
                                           try goldens.tensor("logits").values,
                                           atol: 1e-4, rtol: 1e-4)
        // Reported, not gated: the point is that it lands far outside 1e-4,
        // which is why the parity install is BF16 passthrough. The only
        // assertion is that the round-trip actually ran.
        print("INT4 g64 round-trip vs bf16 goldens, SHORT logits: \(delta.description)")
        #expect(delta.maxAbs > 1e-4,
                Comment(rawValue: "INT4 round-trip is suspiciously exact "
                            + "(\(delta.description)) — did it run?"))
    }

    /// Prints the worst observed delta per capture family, so the margin
    /// against the 1e-4 gate is visible rather than inferred. Whoever ports a
    /// module to Metal is diffing against these numbers, not against the gate.
    @Test func reportsObservedParityMargins() throws {
        for prompt in FlashNextGoldens.Prompt.allCases {
            guard let run = try Self.prefill(prompt) else { return }
            defer { try? FileManager.default.removeItem(at: run.cleanup) }
            let goldens = try FlashNextGoldens(prompt: prompt, phase: .prefill)
            var worst: (key: String, delta: FlashNextDelta)?
            for (key, expected) in goldens.tensors {
                guard let actual = run.capture.floats[key] else { continue }
                let delta = FlashNextDelta.compare(actual, expected.values,
                                                   atol: 1e-4, rtol: 1e-4)
                if worst == nil || delta.maxAbs > worst!.delta.maxAbs {
                    worst = (key, delta)
                }
            }
            if let worst {
                print("prefill \(prompt.rawValue): worst tensor \(worst.key) "
                        + worst.delta.description)
            }
            let ties = run.capture.indexerBoundaryTies
            print("prefill \(prompt.rawValue): \(ties.count) indexer boundary tie(s)"
                    + (ties.isEmpty ? "" : " at "
                        + ties.map { "L\($0.layer)/q\($0.query)@\($0.score)" }
                            .joined(separator: ", ")))
        }
        for prompt in FlashNextGoldens.Prompt.allCases {
            guard let run = try Self.decode(prompt, steps: 16) else { return }
            defer { try? FileManager.default.removeItem(at: run.cleanup) }
            let goldens = try FlashNextGoldens(prompt: prompt, phase: .decode)
            for (key, source) in [("step_logits", "logits"),
                                  ("last_hidden_state", "last_hidden_state")] {
                guard let expected = goldens.tensors[key] else { continue }
                var actual: [Float] = []
                for capture in run.captures {
                    actual.append(contentsOf: capture.floats[source] ?? [])
                }
                let delta = FlashNextDelta.compare(actual, expected.values,
                                                   atol: 1e-4, rtol: 1e-4)
                print("decode \(prompt.rawValue): \(key) \(delta.description)")
            }
        }
    }

    // MARK: - Helpers

    private func firstDifference(_ actual: [[Int]], _ expected: [[Int]],
                                 _ capture: FlashNextReferenceRunner.Capture) -> String {
        for (i, (a, b)) in zip(actual, expected).enumerated() where a != b {
            let tie = capture.indexerBoundaryTies.first { $0.query == i }
            let note = tie.map { " (boundary tie at score \($0.score))" } ?? ""
            return "first difference at query \(i)\(note): \(a) vs \(b)"
        }
        return "lengths \(actual.count) vs \(expected.count)"
    }
}
