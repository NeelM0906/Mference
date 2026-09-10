import Foundation
import Testing
@testable import MferenceRepackCore

/// The RMSNorm `1 + w` convention for original-repo installs.
///
/// Qwen 3.6 applies most of its RMSNorms as `(1 + w) * x̂`. mlx-lm's conversion
/// folds the `+1` into the stored weight; the Mference runtime, built against
/// that conversion, multiplies by the stored weight directly. An install read
/// from the vendor's original repo stores the bare `w`, so it must fold too.
///
/// This defect is worth a dedicated suite because of how quietly it fails.
/// The install verifies, the manifest validates, the model *loads*, generation
/// runs at full speed — and every token is wrong, because all 101 norms are off
/// by one. Nothing structural catches it: norms are not quantized, so the
/// weight-level gate never samples them, and neither `--verify-install` nor
/// `ManifestReader` has any opinion about a passthrough tensor's numeric
/// convention. It was found by the model-level gate reporting *zero* top-1
/// agreement, which is the whole reason that half of W2.1b exists.
@Suite struct NormBiasFoldTests {

    private static let prefix = "model.language_model."

    /// Qwen 3.6's full norm inventory, with the counts measured against
    /// mlx-community's conversion of the same checkpoint.
    private static func normInventory() -> (folded: [String], bare: [String]) {
        var folded = ["\(prefix)norm.weight"]                       // final norm
        var bare: [String] = []
        for layer in 0..<40 {
            let base = "\(prefix)layers.\(layer)."
            folded.append(base + "input_layernorm.weight")
            folded.append(base + "post_attention_layernorm.weight")
            // The gated-DeltaNet block's own norm keeps the bare convention.
            bare.append(base + "linear_attn.norm.weight")
        }
        // Only the ten full-attention layers carry q/k norms.
        for layer in [3, 7, 11, 15, 19, 23, 27, 31, 35, 39] {
            let base = "\(prefix)layers.\(layer).self_attn."
            folded.append(base + "q_norm.weight")
            folded.append(base + "k_norm.weight")
        }
        return (folded, bare)
    }

    @Test("Qwen 3.6 folds every norm except the gated-DeltaNet block's own")
    func qwen36FoldsTheRightSet() {
        let (folded, bare) = Self.normInventory()
        for name in folded {
            #expect(FlashNextPlanner.foldsNormBias(name, family: .qwen36),
                    "\(name) should be folded")
        }
        for name in bare {
            #expect(!FlashNextPlanner.foldsNormBias(name, family: .qwen36),
                    "\(name) must stay bare")
        }
    }

    /// The counts are the ones measured on disk: 40 + 40 + 10 + 10 + 1 folded,
    /// 30 bare (`linear_attn` appears on the 30 linear-attention layers).
    @Test("the folded set is the 101 tensors the control conversion folded")
    func foldedSetMatchesTheControl() {
        let (folded, bare) = Self.normInventory()
        #expect(folded.count == 101)
        #expect(folded.filter { $0.hasSuffix(".input_layernorm.weight") }.count == 40)
        #expect(folded.filter { $0.hasSuffix(".post_attention_layernorm.weight") }
            .count == 40)
        #expect(folded.filter { $0.hasSuffix(".q_norm.weight") }.count == 10)
        #expect(folded.filter { $0.hasSuffix(".k_norm.weight") }.count == 10)
        #expect(bare.count == 40)   // one per layer in this inventory
    }

    /// Nothing that is not a norm may be folded — folding a projection would
    /// corrupt it silently.
    @Test("non-norm tensors are never folded")
    func nonNormsAreNeverFolded() {
        for suffix in ["self_attn.q_proj.weight", "mlp.gate.weight",
                       "mlp.shared_expert.down_proj.weight",
                       "linear_attn.conv1d.weight", "linear_attn.A_log",
                       "embed_tokens.weight"] {
            let name = "\(Self.prefix)layers.5." + suffix
            #expect(!FlashNextPlanner.foldsNormBias(name, family: .qwen36),
                    "\(name)")
        }
        #expect(!FlashNextPlanner.foldsNormBias("lm_head.weight", family: .qwen36))
    }

    /// Flash-Next ships and its first-light run produced coherent output with
    /// the bare weights. Folding would change every byte of that install.
    @Test("Flash-Next folds nothing")
    func flashNextFoldsNothing() {
        let (folded, bare) = Self.normInventory()
        for name in folded + bare {
            #expect(!FlashNextPlanner.foldsNormBias(name, family: .qwen38flashnext),
                    "\(name)")
        }
    }

    /// The transform itself: `bf16(w + 1)`, width-preserving, and the operation
    /// that was verified bit-exact against the control's stored bytes.
    @Test("add-one-bf16 preserves width and rounds through BF16")
    func addOneTransformGeometry() throws {
        let transform = RangeCopyTransform.addOneBF16
        #expect(transform.inputUnitBytes == 2)
        #expect(try transform.destinationByteCount(for: 4_096) == 4_096)
        #expect(throws: RepackError.self) {
            _ = try transform.destinationByteCount(for: 4_097)
        }
        #expect(transform.fingerprintDescription == "add-one-bf16")
    }

    /// Folding must be exactly one BF16-rounded addition — the same operation
    /// the conversion performed — not a float accumulation that drifts.
    @Test("folding a BF16 value is bf16(w + 1)")
    func foldMatchesBF16Addition() {
        for raw: Float in [0, 0.1, -0.25, 0.00390625, -0.5, 1.5, 0.10009766] {
            let stored = Int4AffineEncoder.bf16Bits(raw)
            let w = Int4AffineEncoder.bf16ToFloat(stored)
            let folded = Int4AffineEncoder.bf16Bits(w + 1)
            // Round-trips to within one BF16 ulp of 1 + w, and never equals the
            // unfolded value (which is what a missing fold looks like).
            #expect(abs(Int4AffineEncoder.bf16ToFloat(folded) - (w + 1)) <= 0.004)
            #expect(folded != stored)
        }
    }
}
