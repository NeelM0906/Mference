import Foundation
import Metal

/// Inkling-Small resident-tensor accessors. The family's attention carries no
/// RoPE (`wr_du` + `rel_logits_proj` encode position), K/V pass through
/// depthwise short convolutions, layers 0-1 are dense FFN, and the two shared
/// experts ship stacked in one `[2, ...]` tensor per projection.
extension Model {

    // MARK: - Attention projections

    public func inklingWqDu(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn.wq_du.weight")
    }
    public func inklingWkDv(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn.wk_dv.weight")
    }
    public func inklingWvDv(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn.wv_dv.weight")
    }
    public func inklingWoUd(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn.wo_ud.weight")
    }
    /// Relative-state projection, width `numHeads * dRel`.
    public func inklingWrDu(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn.wr_du.weight")
    }
    /// Bias-vs-distance profile bank `[dRel, extent]` — BF16, unquantized.
    /// Extent is `slidingWindow` on local layers and `rel_extent` on global.
    public func inklingRelProj(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn.rel_logits_proj.proj")
    }
    public func inklingQNorm(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn.q_norm.weight")
    }
    public func inklingKNorm(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn.k_norm.weight")
    }

    // MARK: - Short convolutions (BF16 taps, checkpoint layout [C, K, 1])

    public func inklingKSconv(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn.k_sconv.weight")
    }
    public func inklingVSconv(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn.v_sconv.weight")
    }
    public func inklingAttnSconv(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).attn_sconv.weight")
    }
    public func inklingMlpSconv(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).mlp_sconv.weight")
    }

    // MARK: - Router sidecars

    /// `e_score_correction_bias` — FP32 [numExperts]; selection only.
    public func inklingGateBias(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).mlp.gate.bias")
    }
    /// Learned per-layer router output scalar — FP32 [1].
    public func inklingGateGlobalScale(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).mlp.gate.global_scale")
    }

    // MARK: - Dense FFN (layers 0..<numDenseLayers)

    public func inklingDenseGate(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).mlp.gate_proj.weight")
    }
    public func inklingDenseUp(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).mlp.up_proj.weight")
    }
    public func inklingDenseDown(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).mlp.down_proj.weight")
    }
    /// Learned output gain of the dense MLP — FP32 [1].
    public func inklingDenseGlobalScale(layer L: Int) throws -> TensorView {
        try resident(name: "model.llm.layers.\(L).mlp.global_scale")
    }

    // MARK: - Stacked shared experts

    /// Sub-view of one shared expert from the stacked `[numShared, ...]`
    /// tensor: weights, scales, and biases are each contiguous per expert, so
    /// the slice at index `expert` is a fixed stride into all three regions.
    public func inklingSharedExpert(_ proj: String, layer L: Int,
                                    expert e: Int, of numShared: Int) throws -> TensorView {
        let stacked = try resident(
            name: "model.llm.layers.\(L).mlp.shared_experts.\(proj).weight")
        let n = UInt64(numShared)
        precondition(stacked.length % n == 0 && stacked.scaleLength % n == 0
                        && stacked.biasLength % n == 0,
                     "stacked shared-expert tensor not divisible by \(numShared)")
        let w = stacked.length / n
        let s = stacked.scaleLength / n
        let b = stacked.biasLength / n
        let idx = UInt64(e)
        return TensorView(buffer: stacked.buffer,
                          offset: stacked.offset + idx * w, length: w,
                          scaleOffset: stacked.scaleOffset + idx * s, scaleLength: s,
                          biasOffset: stacked.biasOffset + idx * b, biasLength: b,
                          shape: (stacked.shape.1, stacked.shape.2, 1, 1),
                          dtype: stacked.dtype)
    }

    /// Reads a resident scalar CPU-side, decoding by the view's dtype. The
    /// checkpoint mixes precisions here: `mlp.gate.global_scale` and
    /// `mlp.gate.bias` are FP32, but the dense layers' `mlp.global_scale` is
    /// BF16 — reading that as raw FP32 yields garbage (2 BF16 bytes + 2
    /// neighbor bytes), which corrupted layers 0-1 and, through them, every
    /// downstream activation.
    public func inklingScalar(_ view: TensorView) -> Float {
        let base = view.buffer.contents().advanced(by: Int(view.offset))
        switch view.dtype {
        case 1:  // BF16
            return Quantization.bf16ToFloat(base.assumingMemoryBound(to: UInt16.self)[0])
        case 2:  // FP16
            return Float(base.assumingMemoryBound(to: Float16.self)[0])
        case 3:  // FP32
            return base.assumingMemoryBound(to: Float.self)[0]
        default:
            preconditionFailure("inklingScalar: unsupported dtype \(view.dtype)")
        }
    }
}
