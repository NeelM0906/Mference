import Foundation

/// The four message gates of an Inkling 2 core layer, in the order the
/// checkpoint's message queries use (`loop_msg_query` columns 0 to 3).
public enum Inkling2MessageGate: Int, Sendable, CaseIterable {
    case query = 0, key, value, ffn

    /// The GGUF infix: `blk.N.loop_<infix>_ctrl_{down,gain,forget}.weight`.
    var ggufInfix: String {
        switch self {
        case .query: "q"
        case .key: "k"
        case .value: "v"
        case .ffn: "ffn"
        }
    }
}

/// Resident tensors of an `inkling2Loop` install, under the names of the
/// Unsloth GGUF it was copied from (llama.cpp `inkling2_loop`, tag b12635).
/// Nothing is renamed or converted at install, so each accessor returns the
/// GGUF's own bytes and its index dtype says the format: 3 = F32, 1 = BF16,
/// `GGMLBlockFormat.q8_0.residentDtype` for Q8_0 blocks.
///
/// Shapes below are row-major `[rows, columns]` (the GGUF's `ne` reversed).
/// Every layer-indexed accessor takes the **stored** layer 0 to 23; the three
/// passes over the core share these tensors (see `Inkling2Config.layerVisits`).
extension Model {
    static let inkling2EmbeddingName = "token_embd.weight"
    static let inkling2OutputHeadName = "output.weight"

    /// `blk.<L>.<suffix>`.
    static func inkling2LayerName(_ L: Int, _ suffix: String) -> String { "blk.\(L).\(suffix)" }

    private func inkling2Layer(_ L: Int, _ suffix: String) throws -> TensorView {
        try resident(name: Self.inkling2LayerName(L, suffix))
    }

    // MARK: - Embedding, head and output mix

    /// `token_embd.weight`, Q8_0 `[201024, 2304]` (`embedding` returns the same).
    public var inkling2Embedding: TensorView { get throws { try resident(name: Self.inkling2EmbeddingName) } }
    /// `token_embd_norm.weight`, F32 `[2304]`: the weighted RMSNorm right after the lookup.
    public var inkling2EmbeddingNorm: TensorView { get throws { try resident(name: "token_embd_norm.weight") } }
    /// `output.weight`, Q8_0 `[201024, 2304]`; rows past `unpaddedVocabSize` are padding.
    public var inkling2OutputHead: TensorView { get throws { try resident(name: Self.inkling2OutputHeadName) } }
    /// `output_norm.weight`, F32 `[2304]`: the final RMSNorm before the head.
    public var inkling2OutputNorm: TensorView { get throws { try resident(name: "output_norm.weight") } }
    /// `output_res_query.weight`, F32 `[2304]`: the block-residual query that
    /// mixes the block sums into the final hidden state.
    public var inkling2OutputResQuery: TensorView { get throws { try resident(name: "output_res_query.weight") } }

    // MARK: - Attention

    /// `attn_norm.weight`, F32 `[2304]`; sliding layers only.
    public func inkling2AttnNorm(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_norm.weight") }
    /// `attn_{q,k,v}_inp_norm.weight`, F32 `[2304]`; global layers only (separate q, k, v inputs).
    public func inkling2AttnQInputNorm(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_q_inp_norm.weight") }
    public func inkling2AttnKInputNorm(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_k_inp_norm.weight") }
    public func inkling2AttnVInputNorm(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_v_inp_norm.weight") }
    /// `attn_q.weight`, Q8_0 `[3072, 2304]` (24 heads x 128).
    public func inkling2AttnQ(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_q.weight") }
    /// `attn_k.weight` / `attn_v.weight`, Q8_0 `[1024, 2304]` (8 KV heads x 128).
    public func inkling2AttnK(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_k.weight") }
    public func inkling2AttnV(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_v.weight") }
    /// `attn_output.weight`, Q8_0 `[2304, 3072]`.
    public func inkling2AttnOutput(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_output.weight") }
    /// `attn_gate.weight`, Q8_0 `[3072, 2304]`: `sigmoid(gate(x))` scales the
    /// attention output before `attn_output`.
    public func inkling2AttnGate(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_gate.weight") }
    /// `attn_q_clip.weight`, F32 `[24, 128]`: per-head query scale before the L2 clip.
    public func inkling2AttnQClip(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_q_clip.weight") }
    /// `attn_sink_proj.weight`, `[24, 2304]` (BF16 in the shipped file): one sink logit per head per token.
    public func inkling2AttnSinkProj(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_sink_proj.weight") }
    /// `attn_rel.weight`, Q8_0 `[384, 2304]`: the relative states, 24 heads x 16.
    public func inkling2AttnRel(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_rel.weight") }
    /// `attn_rel_bias.weight`, F32 `[1024, 16]`: bias profile per backward distance.
    public func inkling2AttnRelBias(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_rel_bias.weight") }
    /// `attn_rel_clip.weight`, F32 `[24, 16]`: per-head relative-state scale before its L2 clip.
    public func inkling2AttnRelClip(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_rel_clip.weight") }

    // MARK: - Short convolutions

    /// `attn_k_conv.weight` / `attn_v_conv.weight`, F32 `[1024, 4]`: the K and V
    /// causal taps (the static taps on global layers).
    public func inkling2AttnKConv(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_k_conv.weight") }
    public func inkling2AttnVConv(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_v_conv.weight") }
    /// `attn_{k,v}_conv_proj.weight`, `[32, 2304]` (BF16 in the shipped file):
    /// the per-token taps of the dynamic K and V convolutions, 8 heads x 4; global layers only.
    public func inkling2AttnKConvProj(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_k_conv_proj.weight") }
    public func inkling2AttnVConvProj(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_v_conv_proj.weight") }
    /// `attn_post_conv.weight` / `ffn_post_conv.weight`, F32 `[2304, 4]`: the
    /// convolutions after attention and after the MoE.
    public func inkling2AttnPostConv(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_post_conv.weight") }
    public func inkling2FFNPostConv(layer L: Int) throws -> TensorView { try inkling2Layer(L, "ffn_post_conv.weight") }

    // MARK: - Block attention residuals

    /// `attn_res_query.weight`, F32 `[1, 2304]` on sliding layers and
    /// `[3, 2304]` (q, k, v) on global layers; absent on layer 0.
    public func inkling2AttnResQuery(layer L: Int) throws -> TensorView { try inkling2Layer(L, "attn_res_query.weight") }
    /// `ffn_res_query.weight`, F32 `[2304]`: the MoE input's block-residual query.
    public func inkling2FFNResQuery(layer L: Int) throws -> TensorView { try inkling2Layer(L, "ffn_res_query.weight") }

    // MARK: - MoE

    /// `ffn_norm.weight`, F32 `[2304]`.
    public func inkling2FFNNorm(layer L: Int) throws -> TensorView { try inkling2Layer(L, "ffn_norm.weight") }
    /// `ffn_gate_inp.weight`, F32 `[128, 2304]`: router logits, sigmoid scores.
    public func inkling2Router(layer L: Int) throws -> TensorView { try inkling2Layer(L, "ffn_gate_inp.weight") }
    /// `exp_probs_b.bias`, F32 `[128]`: added to the scores for selection only.
    public func inkling2RouterSelectionBias(layer L: Int) throws -> TensorView { try inkling2Layer(L, "exp_probs_b.bias") }
    /// `ffn_gate_up_exps.scale` / `ffn_down_exps.scale`, F32 `[128]`: the
    /// per-expert factor an NVFP4 projection's output is multiplied by
    /// (`amax / outer_scale_denominator` of the vendor file). NVFP4 layers only.
    public func inkling2ExpertGateUpScale(layer L: Int) throws -> TensorView { try inkling2Layer(L, "ffn_gate_up_exps.scale") }
    public func inkling2ExpertDownScale(layer L: Int) throws -> TensorView { try inkling2Layer(L, "ffn_down_exps.scale") }

    // MARK: - Loop (core layers 6 to 17)

    /// `loop_msg_query.weight`, F32 `[4, 2304]`: the message queries, rows in
    /// `Inkling2MessageGate` order.
    public func inkling2LoopMessageQuery(layer L: Int) throws -> TensorView { try inkling2Layer(L, "loop_msg_query.weight") }
    /// `loop_<g>_ctrl_down.weight`, F32 `[64, 4608]`: `z = down([norm(x); norm(m)])`.
    public func inkling2LoopGateDown(layer L: Int, _ gate: Inkling2MessageGate) throws -> TensorView {
        try inkling2Layer(L, "loop_\(gate.ggufInfix)_ctrl_down.weight")
    }
    /// `loop_<g>_ctrl_gain.weight`, F32 `[2304, 64]`: the message gate `tanh(gain(z))`.
    public func inkling2LoopGateGain(layer L: Int, _ gate: Inkling2MessageGate) throws -> TensorView {
        try inkling2Layer(L, "loop_\(gate.ggufInfix)_ctrl_gain.weight")
    }
    /// `loop_<g>_ctrl_forget.weight`, F32 `[2304, 64]`: the forget gate `tanh(forget(z))`.
    public func inkling2LoopGateForget(layer L: Int, _ gate: Inkling2MessageGate) throws -> TensorView {
        try inkling2Layer(L, "loop_\(gate.ggufInfix)_ctrl_forget.weight")
    }
}
