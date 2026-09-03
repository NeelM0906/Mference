import Foundation
import Testing
@testable import MferenceRepackCore

/// The names an original-repo install writes into its resident index.
///
/// `Model.resident(name:)` is an exact dictionary lookup — a miss is
/// `tensorNotFound`, with no aliasing anywhere — and each family's runner asks
/// for one fixed spelling. Qwen 3.6's runner uses `Model.trunkPrefix`
/// `language_model.model.` and the head `language_model.lm_head.weight`, which
/// the pre-quantized path inherits for free because that is what mlx-lm's
/// conversion writes. The vendor's own repo does not agree: it ships the trunk
/// as `model.language_model.` and a top-level `lm_head.weight`.
///
/// This was invisible for as long as Flash-Next was the only original-repo
/// family, because Flash-Next has no runner to disagree with. It is not
/// invisible now, and it fails *after* the manifest gate rather than at it, so
/// it is worth pinning here rather than rediscovering it at the end of a
/// 72 GB install.
///
/// The literal expectations below are the runner's spellings, transcribed from
/// `Model.trunkPrefix` and `Model.lmHeadName`; the real end-to-end proof is
/// that `qwen36original` loads.
@Suite struct OriginalRepoResidentNamingTests {

    /// The vendor repo's spelling for the tensors Qwen 3.6 actually ships.
    private static let vendorNames = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.7.mlp.gate.weight",
        "model.language_model.layers.7.mlp.shared_expert_gate.weight",
        "model.language_model.layers.39.mlp.shared_expert.down_proj.weight",
        "model.language_model.layers.39.input_layernorm.weight",
        "lm_head.weight",
    ]

    @Test("Qwen 3.6 is written under the spelling its runner looks up")
    func qwen36IsRenamedToTheRunnersSpelling() {
        let renamed = Self.vendorNames.map {
            FlashNextPlanner.residentName(for: $0, family: .qwen36)
        }
        #expect(renamed == [
            "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight",
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.7.mlp.gate.weight",
            "language_model.model.layers.7.mlp.shared_expert_gate.weight",
            "language_model.model.layers.39.mlp.shared_expert.down_proj.weight",
            "language_model.model.layers.39.input_layernorm.weight",
            "language_model.lm_head.weight",
        ])
    }

    /// A renaming, not a remapping: two tensors must never collide, or one
    /// would silently overwrite the other's index entry.
    @Test("the renaming is injective")
    func renamingIsInjective() {
        var names = Self.vendorNames
        for layer in 0..<40 {
            names.append("model.language_model.layers.\(layer).mlp.gate.weight")
            names.append("model.language_model.layers.\(layer).self_attn.o_proj.weight")
        }
        let renamed = names.map {
            FlashNextPlanner.residentName(for: $0, family: .qwen36)
        }
        #expect(Set(renamed).count == Set(names).count)
    }

    /// The suffix the bit policy keys on survives renaming, so the width a
    /// tensor gets does not depend on which spelling it is looked up under.
    @Test("renaming does not disturb the bit policy")
    func renamingPreservesPolicyDecisions() {
        let policy = QuantBitPolicy.moeRouterInt8
        for name in Self.vendorNames {
            let renamed = FlashNextPlanner.residentName(for: name, family: .qwen36)
            #expect(policy.bits(forTensorNamed: name)
                == policy.bits(forTensorNamed: renamed), "\(name)")
        }
    }

    /// Flash-Next ships. Renaming it would change every byte of its install,
    /// and it has no runner whose spelling would justify that.
    @Test("Flash-Next names are untouched")
    func flashNextIsIdentity() {
        for name in Self.vendorNames + [
            "model.language_model.layers.1.ple.key_proj.weight",
            "model.language_model.hyper_connection_mixer.weight",
        ] {
            #expect(FlashNextPlanner.residentName(for: name,
                                                  family: .qwen38flashnext) == name,
                    "\(name)")
        }
    }

    /// A name the mapping does not recognise passes through rather than being
    /// mangled, so an unexpected tensor fails loudly at load instead of landing
    /// under a plausible-looking wrong name.
    @Test("unrecognised names pass through unchanged")
    func unknownNamesPassThrough() {
        for name in ["mtp.fc_embedding.weight",
                     "model.visual.patch_embed.proj.weight",
                     "something.else.entirely"] {
            #expect(FlashNextPlanner.residentName(for: name, family: .qwen36) == name,
                    "\(name)")
        }
    }
}
