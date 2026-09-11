import Foundation
import Metal

/// Resident accessors for the `glm53Flash` family.
///
/// The install keeps PipeNetwork's mlx-vlm names verbatim:
/// `language_model.model.layers.N.…`, `language_model.model.embed_tokens`,
/// `language_model.model.norm`, `language_model.lm_head`. The generic
/// accessors (`embedding`, `lmHead`, `finalNorm`, `router`, `inputNorm`,
/// `postAttnNorm`, `sharedExpert*`) resolve through `trunkPrefix`; what the
/// family adds on top of them lives here:
///
///   * **KDA layers** (mask 7): `self_attn.{q,k,v}_proj` INT8, a BF16
///     depthwise `conv1d.weight` `[3 * qkvDim, kernel, 1]`, the forget gate's
///     `f_a_proj` / `f_b_proj` INT8 low-rank pair with FP32 `A_log`
///     `[numVHeads]` and `dt_bias` `[numVHeads * valueHeadDim]`, the output
///     gate's `g_a_proj` / `g_b_proj` INT8 pair, `b_proj` INT8 `[numVHeads,
///     hidden]`, the BF16 per-head `o_norm.weight` and `o_proj` INT8.
///   * **Sparse layers** (mask 8): `q_a_proj`, `q_a_layernorm`, `q_b_proj`,
///     `kv_a_proj_with_mqa`, `kv_a_layernorm`, the per-head INT8
///     `embed_q` `[heads, kvLoraRank, qkNopeHeadDim]` and `unembed_out`
///     `[heads, vHeadDim, kvLoraRank]`, `o_proj`; and the indexer's `wq_b`,
///     `wk`, `weights_proj` (INT8), `k_norm.{weight,bias}` (BF16) and the
///     BF16 pooling `index_kpool_compress_gate` `[indexHeadDim, hidden]` /
///     `index_kpool_compress_ape` `[indexKPool, indexHeadDim]`.
///   * **mHC** sites `attn_hc.{fn,base,scale}` / `ffn_hc.{fn,base,scale}`:
///     `fn` is BF16 `[(2 + mult) * mult, mult * hidden]` in this conversion,
///     `base` and `scale` FP32.
///   * **MoE**: the router gate is BF16 with an FP32
///     `e_score_correction_bias`; the three leading dense layers carry
///     `mlp.{gate,up,down}_proj` INT8 of width `denseIntermediateSize`.
extension Model {

    private func glm53Layer(_ L: Int) -> String { "language_model.model.layers.\(L)." }

    // MARK: - Kimi Delta Attention

    public func glm53KDAQProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.q_proj.weight")
    }
    public func glm53KDAKProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.k_proj.weight")
    }
    public func glm53KDAVProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.v_proj.weight")
    }
    /// Depthwise causal conv over `[q ; k ; v]`, BF16 `[3 * qkvDim, kernel, 1]`.
    public func glm53KDAConv1d(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.conv1d.weight")
    }
    public func glm53KDAForgetAProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.forget_gate.f_a_proj.weight")
    }
    public func glm53KDAForgetBProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.forget_gate.f_b_proj.weight")
    }
    /// Per-head decay base, FP32 `[numVHeads]`.
    public func glm53KDAALog(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.forget_gate.A_log")
    }
    /// Per-channel decay bias, FP32 `[numVHeads * valueHeadDim]`.
    public func glm53KDADtBias(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.forget_gate.dt_bias")
    }
    public func glm53KDAGateAProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.g_a_proj.weight")
    }
    public func glm53KDAGateBProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.g_b_proj.weight")
    }
    /// Write-strength projection, INT8 `[numVHeads, hidden]`; `beta = sigmoid`.
    public func glm53KDABetaProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.b_proj.weight")
    }
    /// Sigmoid-gated per-head RMSNorm gain, BF16 `[valueHeadDim]`.
    public func glm53KDAOutNorm(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.o_norm.weight")
    }
    /// Output projection of either attention kind, INT8 `[hidden, heads * dim]`.
    public func glm53OProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.o_proj.weight")
    }

    // MARK: - NoPE latent sparse attention

    public func glm53QAProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.q_a_proj.weight")
    }
    public func glm53QANorm(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.q_a_layernorm.weight")
    }
    public func glm53QBProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.q_b_proj.weight")
    }
    public func glm53KVAProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.kv_a_proj_with_mqa.weight")
    }
    public func glm53KVANorm(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.kv_a_layernorm.weight")
    }
    /// Per-head query fold into the latent, INT8 `[heads, kvLoraRank, qkNopeHeadDim]`.
    public func glm53EmbedQ(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.embed_q.weight")
    }
    /// Per-head latent unfold, INT8 `[heads, vHeadDim, kvLoraRank]`.
    public func glm53UnembedOut(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.unembed_out.weight")
    }

    // MARK: - Pooled lightning indexer

    public func glm53IndexerQBProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.indexer.wq_b.weight")
    }
    public func glm53IndexerKProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.indexer.wk.weight")
    }
    public func glm53IndexerKNormWeight(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.indexer.k_norm.weight")
    }
    public func glm53IndexerKNormBias(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.indexer.k_norm.bias")
    }
    public func glm53IndexerWeightsProj(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.indexer.weights_proj.weight")
    }
    /// Pooling gate, BF16 `[indexHeadDim, hidden]`.
    public func glm53IndexerPoolGate(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.indexer.index_kpool_compress_gate")
    }
    /// Pooling position bias, BF16 `[indexKPool, indexHeadDim]`.
    public func glm53IndexerPoolAPE(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "self_attn.indexer.index_kpool_compress_ape")
    }

    // MARK: - Hyper-connections

    /// Attention-site mix: `fn` BF16 `[(2 + mult) * mult, mult * hidden]`,
    /// `base` FP32 `[(2 + mult) * mult]`, `scale` FP32 `[3]`.
    public func glm53AttnHCFn(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "attn_hc.fn")
    }
    public func glm53AttnHCBase(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "attn_hc.base")
    }
    public func glm53AttnHCScale(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "attn_hc.scale")
    }
    public func glm53FFNHCFn(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "ffn_hc.fn")
    }
    public func glm53FFNHCBase(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "ffn_hc.base")
    }
    public func glm53FFNHCScale(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "ffn_hc.scale")
    }

    // MARK: - MoE and dense FFN

    /// Router selection bias, FP32 `[numExperts]`. Selection only: the
    /// routing weights are the raw sigmoid scores of the selected experts,
    /// renormalized and scaled.
    public func glm53RouterCorrectionBias(layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "mlp.gate.e_score_correction_bias")
    }
    /// The dense FFN of the leading `numDenseLayers` layers; `proj` is one
    /// of `gate_proj`, `up_proj`, `down_proj`.
    public func glm53DenseFFN(_ proj: String, layer L: Int) throws -> TensorView {
        try resident(name: glm53Layer(L) + "mlp.\(proj).weight")
    }
}
