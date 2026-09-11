import Foundation
import Testing
@testable import Mference

/// The fp32 tier of the `glm53flash` parity gates: the Swift CPU oracle
/// (`Glm53ReferenceRunner`) against the goldens PipeNetwork's runtime produced
/// from the same toy checkpoint (`Scripts/parity/README.md`, "glm53flash").
///
/// Every integer decision is exact — pooled-indexer selections (including the
/// dense bypass and the tail rule), router top-8, greedy argmax — and every
/// continuous capture is compared elementwise at `atol = rtol = 1e-4`
/// (`Self.tolerance`); both sides are fp32, so the residual is accumulation
/// order. The worst delta per capture family is printed so a drift is visible
/// as such rather than as a silent tolerance bump.
@Suite(.serialized) struct Glm53ReferenceParityTests {

    static let tolerance: Float = 1e-4

    @Test func toyConfigMatchesTheGoldensManifest() throws {
        #expect(try Glm53Fixtures.toyConfigMismatches() == [])
    }

    @Test func toyCheckpointCarriesThePipeNetworkStoragePolicy() throws {
        let ckpt = try Glm53ToyCheckpoint()
        #expect(try ckpt.entry("language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight").dtype == "U32")
        #expect(try ckpt.entry("language_model.model.layers.0.self_attn.q_proj.weight").dtype == "U32")
        #expect(try ckpt.entry("language_model.model.layers.1.mlp.gate.weight").dtype == "BF16")
        #expect(try ckpt.entry("language_model.model.layers.0.attn_hc.fn").dtype == "BF16")
        #expect(try ckpt.entry("language_model.model.layers.0.attn_hc.base").dtype == "F32")
        #expect(try ckpt.entry("language_model.model.layers.0.self_attn.forget_gate.A_log").dtype == "F32")
        #expect(try ckpt.entry("language_model.model.layers.0.self_attn.conv1d.weight").dtype == "BF16")
        #expect(try ckpt.entry("language_model.model.layers.3.self_attn.embed_q.weight").shape == [2, 64, 16])
        // INT8 experts would be a different width: the routed triplets are x8 packed.
        #expect(try ckpt.entry("language_model.model.layers.1.mlp.switch_mlp.down_proj.weight").shape == [16, 128, 8])
    }

    private struct Worst {
        var maxAbs: Float = 0
        var maxRel: Float = 0
        var mismatched = 0
        var count = 0
        var at = ""
    }

    /// Elementwise `|a - b| <= atol + rtol * |b|`, accumulating the worst.
    private static func compare(_ actual: [Float], _ expected: [Float], key: String,
                                into worst: inout [String: Worst]) {
        let family = key.split(separator: ".").last.map(String.init) ?? key
        var w = worst[family] ?? Worst()
        guard actual.count == expected.count else {
            Issue.record(Comment(rawValue: "\(key): count \(actual.count) != \(expected.count)"))
            return
        }
        for i in 0..<actual.count {
            let d = abs(actual[i] - expected[i])
            let rel = d / max(abs(expected[i]), 1e-30)
            if d > w.maxAbs { w.maxAbs = d; w.at = "\(key)[\(i)]" }
            if d > tolerance + tolerance * abs(expected[i]) {
                w.mismatched += 1
                w.maxRel = max(w.maxRel, rel)
            }
            w.count += 1
        }
        worst[family] = w
    }

    private static func checkPosition(_ oracle: Glm53ReferenceRunner, goldens: Glm53Goldens,
                                      prefix: String, worst: inout [String: Worst],
                                      integerFailures: inout [String]) {
        let cap = oracle.capture
        for (key, tensor) in goldens.tensors where key.hasPrefix(prefix) {
            let local = String(key.dropFirst(prefix.count))
            if local == "logits" { continue }   // compared by the caller
            if local.hasSuffix("idx_pool_keys") {
                // Only the complete (visible) pools carry a defined key.
                guard let mine = cap.floats[local],
                      let visible = cap.poolVisible[local.replacingOccurrences(of: "idx_pool_keys", with: "idx_pool_visible")] else {
                    integerFailures.append("\(key): oracle captured nothing"); continue
                }
                let width = tensor.shape[1]
                for (j, v) in visible.enumerated() where v {
                    compare(Array(mine[(j * width)..<((j + 1) * width)]), tensor.row(j),
                            key: key, into: &worst)
                }
                continue
            }
            if local.hasSuffix("idx_scores") {
                guard let mine = cap.floats[local],
                      let visible = cap.poolVisible[local.replacingOccurrences(of: "idx_scores", with: "idx_pool_visible")] else {
                    integerFailures.append("\(key): oracle captured nothing"); continue
                }
                let picked = visible.indices.filter { visible[$0] }
                compare(picked.map { mine[$0] }, picked.map { tensor.values[$0] }, key: key, into: &worst)
                continue
            }
            guard let mine = cap.floats[local] else {
                integerFailures.append("\(key): oracle captured nothing")
                continue
            }
            if local.hasSuffix("router_weights") {
                // The reference lists the chosen experts in selection-kernel
                // order, the oracle in descending biased score; pair the
                // weights by expert. (The index sets are compared below.)
                let idxLocal = local.replacingOccurrences(of: "router_weights", with: "router_indices")
                if let theirs = goldens.integers[prefix + idxLocal] as? [Int], let ours = cap.ints[idxLocal],
                   theirs.count == tensor.values.count, ours.count == mine.count, Set(theirs) == Set(ours) {
                    compare(zip(ours, mine).sorted { $0.0 < $1.0 }.map { $0.1 },
                            zip(theirs, tensor.values).sorted { $0.0 < $1.0 }.map { $0.1 },
                            key: key, into: &worst)
                    continue
                }
            }
            compare(mine, tensor.values, key: key, into: &worst)
        }
        for (key, value) in goldens.integers where key.hasPrefix(prefix) {
            let local = String(key.dropFirst(prefix.count))
            if local.hasSuffix("router_indices") {
                let expected = Set(value as? [Int] ?? [])
                if Set(cap.ints[local] ?? []) != expected {
                    integerFailures.append("\(key): \(cap.ints[local] ?? []) != \(expected.sorted())")
                }
            } else if local.hasSuffix("idx_selected") {
                let expected = value as? [Int]      // nil for "dense"
                guard let mine = cap.selections[local] else {
                    integerFailures.append("\(key): oracle captured nothing"); continue
                }
                if mine != expected {
                    integerFailures.append("\(key): \(String(describing: mine)) != \(String(describing: expected))")
                }
            } else if local.hasSuffix("idx_visible") {
                if cap.ints[local] != [(value as? NSNumber)?.intValue ?? -1] {
                    integerFailures.append("\(key): \(cap.ints[local] ?? []) != \(value)")
                }
            } else if local.hasSuffix("idx_pool_visible") {
                if cap.poolVisible[local] != (value as? [Bool]) {
                    integerFailures.append("\(key): \(String(describing: cap.poolVisible[local])) != \(value)")
                }
            }
        }
    }

    private struct RunResult {
        var worst: [String: Worst] = [:]
        var integerFailures: [String] = []
        var argmaxMismatches: [String] = []
        var rolloutMatches = 0
        var rolloutSteps = 0
    }

    /// Drives the oracle through the prompt and the goldens' rollout,
    /// comparing every capture. With `anchored` the reference's own layer
    /// inputs and cache appends are forced, so each layer is judged on the
    /// reference's inputs.
    private static func run(prompt: Glm53Goldens.Prompt, anchored: Bool) throws -> RunResult {
        let oracle = try Glm53ReferenceRunner(checkpoint: try Glm53ToyCheckpoint())
        let tokens = try Glm53Goldens.promptTokens(prompt)
        let promptGoldens = try Glm53Goldens(prompt: prompt, phase: .prompt)
        let decodeGoldens = try Glm53Goldens(prompt: prompt, phase: .decode)
        var worst: [String: Worst] = [:]
        var integerFailures: [String] = []
        var argmaxMismatches: [String] = []
        var rolloutMatches = 0

        func stepAndCheck(goldens: Glm53Goldens, prefix: String, token: Int,
                          expectedLogits: [Float], expectedArgmax: Int?) throws {
            if anchored {
                oracle.anchor = { local in goldens.tensors[prefix + local]?.values }
            }
            let logits = try oracle.step(token: token)
            compare(logits, expectedLogits, key: prefix + "logits", into: &worst)
            let mine = logits.indices.max(by: { logits[$0] < logits[$1] })!
            let expected = expectedArgmax
                ?? expectedLogits.indices.max(by: { expectedLogits[$0] < expectedLogits[$1] })!
            if mine != expected { argmaxMismatches.append(prefix) }
            checkPosition(oracle, goldens: goldens, prefix: prefix,
                          worst: &worst, integerFailures: &integerFailures)
        }

        let seqLogits = try promptGoldens.tensor("seq.logits")
        for (p, token) in tokens.enumerated() {
            try stepAndCheck(goldens: promptGoldens, prefix: Glm53Goldens.phasePrefix(position: p),
                             token: token, expectedLogits: seqLogits.row(p), expectedArgmax: nil)
        }
        for L in 0..<oracle.config.numLayers {
            let key = "seq.final." + Glm53Goldens.layerKey(L) + "."
            if promptGoldens.has(key + "kda_state") {
                compare(oracle.kdaState(layer: L), try promptGoldens.tensor(key + "kda_state").values,
                        key: key + "kda_state", into: &worst)
                compare(oracle.kdaConvTail(layer: L), try promptGoldens.tensor(key + "kda_conv_state").values,
                        key: key + "kda_conv_state", into: &worst)
            }
            if promptGoldens.has(key + "dsa_latent_cache") {
                compare(oracle.latentCache(layer: L), try promptGoldens.tensor(key + "dsa_latent_cache").values,
                        key: key + "dsa_latent_cache", into: &worst)
                compare(oracle.indexerPackedCache(layer: L), try promptGoldens.tensor(key + "idx_packed_cache").values,
                        key: key + "idx_packed_cache", into: &worst)
            }
        }
        let rollout = try decodeGoldens.ints("greedy_rollout")
        let next = try decodeGoldens.ints("greedy_next")
        let stepLogits = try decodeGoldens.tensor("decode.step_logits")
        for (s, token) in rollout.enumerated() {
            let before = argmaxMismatches.count
            try stepAndCheck(goldens: decodeGoldens, prefix: Glm53Goldens.phasePrefix(step: s),
                             token: token, expectedLogits: stepLogits.row(s), expectedArgmax: next[s])
            if argmaxMismatches.count == before { rolloutMatches += 1 }
        }
        for L in 0..<oracle.config.numLayers {
            let key = "decode.final." + Glm53Goldens.layerKey(L) + "."
            if decodeGoldens.has(key + "kda_state") {
                compare(oracle.kdaState(layer: L), try decodeGoldens.tensor(key + "kda_state").values,
                        key: key + "kda_state", into: &worst)
            }
            if decodeGoldens.has(key + "dsa_latent_cache") {
                compare(oracle.latentCache(layer: L), try decodeGoldens.tensor(key + "dsa_latent_cache").values,
                        key: key + "dsa_latent_cache", into: &worst)
            }
        }
        return RunResult(worst: worst, integerFailures: integerFailures,
                         argmaxMismatches: argmaxMismatches, rolloutMatches: rolloutMatches,
                         rolloutSteps: rollout.count)
    }

    private static func report(_ label: String, _ r: RunResult) -> (mismatched: Int, worstLogits: Float) {
        var text = "[glm53 oracle vs goldens: \(label)] tol \(tolerance)\n"
        var total = 0
        for family in r.worst.keys.sorted() {
            let w = r.worst[family]!
            total += w.mismatched
            text += String(format: "  %-34@ worst abs %.3e at %@; %d/%d outside tolerance\n",
                           family as NSString, w.maxAbs, w.at as NSString, w.mismatched, w.count)
        }
        text += "  rollout argmax \(r.rolloutMatches)/\(r.rolloutSteps); "
        text += "argmax mismatches \(r.argmaxMismatches.count); integer failures \(r.integerFailures.count)\n"
        FileHandle.standardError.write(Data(text.utf8))
        return (total, r.worst["logits"]?.maxAbs ?? .infinity)
    }

    /// Every layer, judged on the reference's own inputs: continuous captures
    /// within `tolerance`, every integer decision exact, every argmax exact.
    @Test(arguments: Glm53Goldens.Prompt.allCases)
    func anchoredOracleReproducesEveryLayer(prompt: Glm53Goldens.Prompt) throws {
        let r = try Self.run(prompt: prompt, anchored: true)
        let (mismatched, _) = Self.report("\(prompt.rawValue), anchored", r)
        #expect(mismatched == 0, "values outside tolerance; see the report above")
        #expect(r.integerFailures.isEmpty, Comment(rawValue: r.integerFailures.prefix(10).joined(separator: "\n")))
        #expect(r.argmaxMismatches.isEmpty, Comment(rawValue: r.argmaxMismatches.joined(separator: ", ")))
        #expect(r.worst["logits"] != nil && r.worst["kda_state"] != nil && r.worst["idx_scores"] != nil)
    }

    /// Free-running: the oracle carries its own fp32 drift through 48 + 16
    /// tokens. The discrete decisions and the greedy rollout must still be
    /// exact; the logits must stay inside the band the reference's own
    /// batched-vs-per-token disagreement defines (`integers_prompt_*.json`,
    /// `sequential_vs_single_max_abs`), scaled x4 for the two extra
    /// accumulation orders in play. The worst deltas are printed as the
    /// measured drift of this toy, not asserted below the band.
    @Test(arguments: Glm53Goldens.Prompt.allCases)
    func freeRunningOracleKeepsEveryDecision(prompt: Glm53Goldens.Prompt) throws {
        let r = try Self.run(prompt: prompt, anchored: false)
        let (_, worstLogits) = Self.report("\(prompt.rawValue), free-running", r)
        let goldens = try Glm53Goldens(prompt: prompt, phase: .prompt)
        let referenceGap = Float((goldens.integers["sequential_vs_single_max_abs"] as? NSNumber)?.doubleValue ?? 0)
        #expect(r.integerFailures.isEmpty, Comment(rawValue: r.integerFailures.prefix(10).joined(separator: "\n")))
        #expect(r.argmaxMismatches.isEmpty, Comment(rawValue: r.argmaxMismatches.joined(separator: ", ")))
        #expect(r.rolloutMatches == r.rolloutSteps)
        #expect(worstLogits <= max(4 * referenceGap, 1e-3),
                "free-running logits drift \(worstLogits) exceeds 4x the reference's own gap \(referenceGap)")
    }
}
