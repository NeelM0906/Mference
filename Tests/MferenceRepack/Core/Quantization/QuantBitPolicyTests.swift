import Foundation
import Testing
@testable import MferenceRepackCore

/// The per-tensor bit policy the original-repo planner consults.
///
/// The policy must reproduce the mixture a trusted community conversion chose
/// — for Qwen 3.6 that is mlx-community's, which overrides every layer's
/// `mlp.gate` and `mlp.shared_expert_gate` to INT8 group-64 and leaves the rest
/// at INT4 — and the override set must be exactly those gating tensors and
/// nothing adjacent to them. The family table below pins which families share
/// that policy. (The Flash-Next tensor-name cases were removed 2026-11-18 with
/// that family's tests.)
@Suite struct QuantBitPolicyTests {

    /// Qwen 3.6's real resident inventory, per layer, abbreviated to the
    /// tensors that matter for width. Names use the vendor repo's
    /// `model.language_model.` prefix, which is what the original-repo planner
    /// sees (the mlx conversion's own names are `language_model.model.`).
    private static func residentNames(layers: Int) -> [String] {
        let prefix = "model.language_model."
        var names = [prefix + "embed_tokens.weight", "lm_head.weight",
                     prefix + "norm.weight"]
        for layer in 0..<layers {
            let base = "\(prefix)layers.\(layer)."
            names += [
                base + "self_attn.q_proj.weight",
                base + "self_attn.k_proj.weight",
                base + "self_attn.v_proj.weight",
                base + "self_attn.o_proj.weight",
                base + "linear_attn.in_proj_qkv.weight",
                base + "linear_attn.out_proj.weight",
                base + "mlp.gate.weight",                    // the router
                base + "mlp.shared_expert_gate.weight",      // the shared gate
                base + "mlp.shared_expert.gate_proj.weight",
                base + "mlp.shared_expert.up_proj.weight",
                base + "mlp.shared_expert.down_proj.weight",
                base + "input_layernorm.weight",
            ]
        }
        return names
    }

    // MARK: - The Qwen 3.6 mixture

    /// The count is derived from the layer count, not written down: 40 layers
    /// give the control's 80 overrides, and a checkpoint with a different depth
    /// gets the right number without the policy being edited.
    @Test("the MoE-router policy overrides exactly two gating tensors per layer",
          arguments: [1, 4, 40, 61])
    func overridesTwoTensorsPerLayer(layers: Int) {
        let policy = QuantBitPolicy.moeRouterInt8
        let names = Self.residentNames(layers: layers)
        let overridden = names.filter { policy.overrides($0) }
        #expect(overridden.count == 2 * layers)
        #expect(overridden.filter { $0.hasSuffix(".mlp.gate.weight") }.count == layers)
        #expect(overridden.filter { $0.hasSuffix(".mlp.shared_expert_gate.weight") }
            .count == layers)
        for name in overridden {
            #expect(policy.bits(forTensorNamed: name) == 8)
        }
        for name in names where !policy.overrides(name) {
            #expect(policy.bits(forTensorNamed: name) == 4, "\(name)")
        }
    }

    /// Qwen 3.6 has 40 layers, so the policy must land on the control's own
    /// `bitWidthOverridesHonored` value. That number is read off
    /// mlx-community's conversion, not chosen here.
    @Test("40 layers reproduce the control conversion's 80 overrides")
    func matchesTheControlOverrideCount() {
        let names = Self.residentNames(layers: 40)
        #expect(names.filter { QuantBitPolicy.moeRouterInt8.overrides($0) }.count == 80)
    }

    /// The shared expert's *projections* are not the shared expert's *gate*.
    /// A prefix or `contains` match would sweep them up; the suffix rule must
    /// not.
    @Test("the shared expert's projections stay at the base width")
    func sharedExpertProjectionsAreNotGates() {
        let policy = QuantBitPolicy.moeRouterInt8
        for suffix in ["mlp.shared_expert.gate_proj.weight",
                       "mlp.shared_expert.up_proj.weight",
                       "mlp.shared_expert.down_proj.weight",
                       "mlp.experts.gate_up_proj",
                       "mlp.experts.down_proj",
                       "mlp.switch_mlp.gate_proj.weight"] {
            let name = "model.language_model.layers.7." + suffix
            #expect(policy.bits(forTensorNamed: name) == 4, "\(name)")
        }
    }

    // MARK: - Family table

    @Test("the MoE original-repo families share one policy; everyone else is uniform INT4")
    func bothOriginalRepoFamiliesShareOnePolicy() {
        #expect(QuantBitPolicy.originalRepo(family: .qwen36) == .moeRouterInt8)
        #expect(QuantBitPolicy.originalRepo(family: .qwen38flashnext) == .moeRouterInt8)
        // MiniCPM5 also has an original-repo entry, and its answer is uniform
        // INT4 on purpose rather than by default: the vendor's own MLX
        // conversion carries no per-tensor overrides, and a dense llama has no
        // router to keep wider.
        #expect(QuantBitPolicy.originalRepo(family: .minicpm5) == .uniformInt4)
        // Every family without an original-repo installer entry still has no
        // examined conversion, so none of them may carry a table.
        for family in [RepackModelFamily.gemma4, .qwen38, .deepseekV4Flash,
                       .inklingSmall, .maple] {
            #expect(QuantBitPolicy.originalRepo(family: family) == .uniformInt4,
                    "\(family.rawValue)")
        }
    }

    // MARK: - Mechanism

    @Test("the longest matching suffix wins regardless of table order")
    func longestSuffixWins() {
        let policy = QuantBitPolicy(defaultBits: 4, rules: [
            QuantBitPolicy.Rule(suffix: ".gate.weight", bits: 8),
            QuantBitPolicy.Rule(suffix: ".mlp.gate.weight", bits: 4),
        ])
        #expect(policy.bits(forTensorNamed: "l.0.mlp.gate.weight") == 4)
        #expect(policy.bits(forTensorNamed: "l.0.ffn.gate.weight") == 8)
        // Construction order must not matter.
        let reversed = QuantBitPolicy(defaultBits: 4, rules: policy.rules.reversed())
        #expect(reversed.rules == policy.rules)
    }

    @Test("a width with no streaming quantizer is rejected at plan time")
    func unsupportedWidthsThrow() {
        for bits in [2, 3, 6, 16] {
            let policy = QuantBitPolicy(defaultBits: 4, rules: [
                QuantBitPolicy.Rule(suffix: ".mlp.gate.weight", bits: bits),
            ])
            #expect(throws: RepackError.self) {
                _ = try policy.validated(for: .qwen36)
            }
        }
        #expect(throws: Never.self) {
            _ = try QuantBitPolicy.moeRouterInt8.validated(for: .qwen36)
            _ = try QuantBitPolicy.moeRouterInt8.validated(for: .qwen38flashnext)
            _ = try QuantBitPolicy.uniformInt4.validated(for: .qwen38flashnext)
        }
    }

    /// The widths the policy may name and the widths the range planner can
    /// actually stream have to stay in step; this is the assertion that fails
    /// if someone adds one without the other.
    @Test("every supported width has a streaming transform")
    func supportedWidthsHaveTransforms() throws {
        for bits in QuantBitPolicy.supportedBits.sorted() {
            let transform: RangeCopyTransform = bits == 4
                ? .quantizeInt4G64(component: .weights)
                : .quantizeInt8G64(component: .weights)
            // One group of BF16 source produces `64 * bits / 8` bytes.
            let produced = try transform.destinationByteCount(for: 128)
            #expect(produced == UInt64(64 * bits / 8), "\(bits)-bit")
        }
    }
}
