import Foundation
import Testing
@testable import MferenceRepackCore

/// The per-tensor bit policy the original-repo planner consults.
///
/// Two things are being protected here. The first is that the policy actually
/// reproduces the mixture a trusted community conversion chose — for Qwen 3.6
/// that is mlx-community's, which overrides every layer's `mlp.gate` and
/// `mlp.shared_expert_gate` to INT8 group-64 and leaves the rest at INT4. The
/// second is that the *same two suffixes* resolve against Flash-Next's real
/// tensor names, which are the vendor's own and are never rewritten by the
/// planner — including the `mtp.` draft layer's, which share no prefix with the
/// text stack at all and would be quietly missed by a rule keyed on anything
/// but a suffix.
///
/// Flash-Next used to be the negative case here: it shipped uniform INT4 and
/// this suite asserted that it stayed that way. The 2026-09-10 measurement on
/// its own install reversed that (see `QuantBitPolicy.moeRouterInt8`), so the
/// assertions are inverted rather than deleted — the property is still that the
/// override set is exactly the gating tensors and nothing adjacent to them.
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

    // MARK: - Flash-Next's real tensor names

    /// The exact names `docs/families/qwen38flashnext.tensors.json` records for
    /// the MoE block, at the exact prefix the planner emits. Flash-Next's
    /// `residentName(for:family:)` is the identity, so these are simultaneously
    /// the vendor's source names and the installed index's names — which is why
    /// getting them wrong here would not be caught anywhere downstream.
    private static func flashNextMoENames(layers: Int) -> [String] {
        var names: [String] = []
        for layer in 0..<layers {
            let base = "model.language_model.layers.\(layer)."
            names += [
                base + "mlp.gate.weight",
                base + "mlp.shared_expert_gate.weight",
                base + "mlp.shared_expert.gate_proj.weight",
                base + "mlp.shared_expert.up_proj.weight",
                base + "mlp.shared_expert.down_proj.weight",
                base + "self_attn.q_proj.weight",
                base + "linear_attn.in_proj_qkv.weight",
                base + "attn_hyper_connection.input_mix_weight_down.weight",
                base + "mlp_hyper_connection.input_mix_weight_up.weight",
                base + "ple.key_proj.weight",
            ]
        }
        // The MTP draft layer's own MoE block. Its names sit under an `mtp.`
        // prefix with no `model.language_model.` in sight; only the suffix is
        // shared with the text stack.
        names += ["mtp.layers.0.mlp.gate.weight",
                  "mtp.layers.0.mlp.shared_expert_gate.weight",
                  "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                  "mtp.fc_embedding.weight",
                  "mtp.fc_hidden.weight"]
        return names
    }

    @Test("Flash-Next keeps both gating tensors at INT8, on 48 layers + MTP")
    func flashNextKeepsBothGatingTensorsAtInt8() {
        let policy = QuantBitPolicy.originalRepo(family: .qwen38flashnext)
        #expect(policy == .moeRouterInt8)
        let names = Self.flashNextMoENames(layers: 48)
        let overridden = names.filter { policy.overrides($0) }
        // 48 text layers + the MTP draft layer, two gating tensors each. This
        // is the number the install's `bitWidthOverridesHonored` must report.
        #expect(overridden.count == 98)
        for name in overridden { #expect(policy.bits(forTensorNamed: name) == 8) }
        for name in names where !policy.overrides(name) {
            #expect(policy.bits(forTensorNamed: name) == 4, "\(name)")
        }
    }

    /// The draft layer routes over the same 512 experts as the layers it drafts
    /// for. A rule that reached the text stack but not `mtp.` would give the two
    /// routers different fidelity and quietly cost verification acceptances —
    /// exactly the kind of miss no install-time check would report.
    @Test("the MTP draft layer's router gets the same width")
    func theMTPDraftLayerGetsTheSameTreatment() {
        let policy = QuantBitPolicy.originalRepo(family: .qwen38flashnext)
        #expect(policy.bits(forTensorNamed: "mtp.layers.0.mlp.gate.weight") == 8)
        #expect(policy.bits(
            forTensorNamed: "mtp.layers.0.mlp.shared_expert_gate.weight") == 8)
        #expect(policy.bits(
            forTensorNamed: "mtp.layers.0.mlp.shared_expert.gate_proj.weight") == 4)
    }

    /// A rule matching a fused expert tensor is a configuration error the
    /// planner throws on — the expert pools are planned at the base width — so
    /// the policy must not come near them, under either prefix.
    @Test("the fused expert tensors keep the base width")
    func flashNextFusedExpertTensorsKeepTheBaseWidth() {
        let policy = QuantBitPolicy.originalRepo(family: .qwen38flashnext)
        for name in ["model.language_model.layers.0.mlp.experts.gate_up_proj",
                     "model.language_model.layers.0.mlp.experts.down_proj",
                     "mtp.layers.0.mlp.experts.gate_up_proj",
                     "mtp.layers.0.mlp.experts.down_proj"] {
            #expect(!policy.overrides(name), "\(name)")
        }
    }

    @Test("both original-repo families share one policy; nobody else has one")
    func bothOriginalRepoFamiliesShareOnePolicy() {
        #expect(QuantBitPolicy.originalRepo(family: .qwen36) == .moeRouterInt8)
        #expect(QuantBitPolicy.originalRepo(family: .qwen38flashnext) == .moeRouterInt8)
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
