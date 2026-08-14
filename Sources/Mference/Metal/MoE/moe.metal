#include <metal_stdlib>
using namespace metal;

constant constexpr uint kMoEGroupSize = 64;
constant constexpr uint kMaxStreamedExperts = 8;
constant constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kGeluCubicCoeff = 0.044715f;

constant uint FC_ROUTER_NUM_EXPERTS [[function_constant(40)]];
constant uint FC_ROUTER_D [[function_constant(41)]];
constant uint FC_ROUTER_TOP_K [[function_constant(42)]];
constant bool FC_ROUTER_USE_FC [[function_constant(43)]];

constant uint FC_MOE_D [[function_constant(0)]];
constant uint FC_MOE_F [[function_constant(1)]];
constant uint FC_MOE_TOP_K [[function_constant(2)]];
constant bool FC_MOE_USE_FC [[function_constant(3)]];
// Hidden activation for expert FFNs: unset/false = Gemma's gelu_pytorch_tanh,
// true = silu (Qwen 3.6 SwiGLU). Applies to routed decode, routed prefill,
// and the fused INT8 shared-expert kernel.
constant bool FC_MOE_ACT_SILU [[function_constant(4)]];
// Pre-activation clamp for routed experts (DeepSeek V4 swiglu_limit): gate is
// clamped to <= limit, up to [-limit, limit], before activation. Unset or
// <= 0 means no clamp.
constant float FC_MOE_SWIGLU_LIMIT [[function_constant(5)]];

static inline float2 moe_swiglu_clamp(float gate, float up) {
    if (is_function_constant_defined(FC_MOE_SWIGLU_LIMIT) &&
        FC_MOE_SWIGLU_LIMIT > 0.0f) {
        gate = min(gate, FC_MOE_SWIGLU_LIMIT);
        up = clamp(up, -FC_MOE_SWIGLU_LIMIT, FC_MOE_SWIGLU_LIMIT);
    }
    return float2(gate, up);
}

static inline uint router_fc_num_experts(constant uint& num_experts) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_NUM_EXPERTS))
        ? FC_ROUTER_NUM_EXPERTS
        : num_experts;
}

static inline uint router_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_D))
        ? FC_ROUTER_D
        : D;
}

static inline uint moe_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_D)) ? FC_MOE_D : D;
}

static inline uint moe_fc_f(constant uint& F) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_F)) ? FC_MOE_F : F;
}

static inline uint moe_fc_top_k(constant uint& top_k) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_TOP_K)) ? FC_MOE_TOP_K : top_k;
}

static inline float gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kGeluSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    // Clamping avoids Metal tanh producing NaN at large magnitudes while being
    // equivalent to the saturated result at FP32 precision.
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

static inline float moe_hidden_activation(float x) {
    if (is_function_constant_defined(FC_MOE_ACT_SILU) && FC_MOE_ACT_SILU) {
        return x / (1.0f + exp(-x));
    }
    return gelu_pytorch_tanh(x);
}

struct ExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

struct RoutedBlobs {
    device const uint8_t* blob[kMaxStreamedExperts];
};

static inline void router_gemv_gemma4_body(
    device const uint8_t* W,
    device const bfloat* scales,
    device const bfloat* biases,
    device const half* hidden,
    device const bfloat* effective_scale,
    device float* out_logits,
    constant uint& num_experts,
    constant uint& D,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint DD = router_fc_d(D);
    const uint e = tg_idx * rows_per_tg + sg_idx;
    if (e >= NE) return;

    const uint n_groups = DD / kMoEGroupSize;
    device const uint8_t* W_row = W + uint(e) * DD;
    device const bfloat* s_row = scales + uint(e) * n_groups;
    device const bfloat* b_row = biases + uint(e) * n_groups;

    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint idx = g * kMoEGroupSize + lane * 2u;
        const float q0 = float(uint(W_row[idx]));
        const float q1 = float(uint(W_row[idx + 1u]));
        const float x0 = float(hidden[idx]) * float(effective_scale[idx]);
        const float x1 = float(hidden[idx + 1u]) * float(effective_scale[idx + 1u]);
        acc = fma(s, q0 * x0 + q1 * x1, acc);
        acc = fma(b, x0 + x1, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

kernel void router_gemv_gemma4_r4(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* hidden [[buffer(3)]],
    device const bfloat* effective_scale [[buffer(4)]],
    device float* out_logits [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    router_gemv_gemma4_body(W, scales, biases, hidden, effective_scale,
                            out_logits, num_experts, D, 4, tg_idx, sg_idx, lane);
}

// DeepSeek V4's router gate rides unquantized BF16 `[num_experts, D]`
// (the mlx-community conversion leaves `ffn.gate.weight` out of the quant
// table). One SIMD per expert row; four rows per threadgroup.
kernel void router_gemv_bf16_r4(
    device const bfloat* W [[buffer(0)]],
    device const half* hidden [[buffer(1)]],
    device const bfloat* effective_scale [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint DD = router_fc_d(D);
    const uint e = tg_idx * 4u + sg_idx;
    if (e >= NE) return;
    device const bfloat* W_row = W + uint(e) * DD;
    float acc = 0.0f;
    for (uint i = lane; i < DD; i += 32u) {
        acc = fma(float(W_row[i]),
                  float(hidden[i]) * float(effective_scale[i]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

// Serial single-thread selection. Superseded in production by the one-SIMD
// `_par` kernels below; kept because the parity tests compare the parallel
// results against it bit for bit.
kernel void router_topk_select_k8(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    const uint NE = router_fc_num_experts(num_experts);
    uint top_idx[8];
    float top_score[8];
    for (uint i = 0; i < 8; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
    }

    for (uint e = 0; e < NE; ++e) {
        const float s = logits[e];
        if (s <= top_score[7]) continue;
        uint pos = 8u;
        for (uint i = 0; i < 8; ++i) {
            if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= 8u) continue;
        for (uint i = 7; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = s;
    }

    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[8];
    for (uint i = 0; i < 8; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < 8; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

// --- One-SIMD parallel top-k selection ------------------------------------
//
// The serial selects above walk all `num_experts` logits on a single thread of
// a whole dispatch, on the per-layer critical path. These replacements keep the
// same single 32-thread dispatch but spread the scan across the SIMD, and are
// bit-identical to the serial result rather than merely equivalent.
//
// Ownership is blocked: lane L owns experts [L*per, L*per + per) with
// per = ceil(NE/32) <= 8 for NE <= 256, and out-of-range slots carry a -INFINITY
// sentinel so short or non-multiple-of-32 expert counts need no special case.
// Each of the K steps takes a lane-local strict-`>` ascending max (lowest index
// wins inside a lane, matching the serial ascending scan) and then a
// shuffle-down argmax whose ties prefer the LOWER index. Together those
// reproduce the serial "first strictly-greater score wins, equal score keeps the
// earlier expert" order exactly. A per-lane taken bitmask retires winners.
// The normalization tail stays serial on lane 0 with the same operations in the
// same order, so the emitted weights match bit for bit.
constant constexpr uint kRouterMaxPerLane = 8;   // num_experts <= 256
constant constexpr uint kRouterNoWinner = 0xFFFFFFFFu;

static inline uint router_select_next_one_simd(
    thread const float* ch,
    thread uint& taken,
    uint per,
    uint base
) {
    float bv = -INFINITY;
    uint bi = kRouterNoWinner;
    for (uint j = 0; j < per; ++j) {
        if (!(taken & (1u << j)) && ch[j] > bv) {
            bv = ch[j];
            bi = base + j;
        }
    }
    for (uint off = 16u; off > 0u; off >>= 1) {
        const float ov = simd_shuffle_down(bv, off);
        const uint oi = simd_shuffle_down(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    bi = simd_broadcast(bi, 0);
    if (bi != kRouterNoWinner && bi >= base && bi < base + per) {
        taken |= 1u << (bi - base);
    }
    return bi;
}

kernel void router_topk_select_k8_par(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint NE = router_fc_num_experts(num_experts);
    // Out-of-contract dispatches are a visible no-op, never an OOB write.
    if (sg_idx != 0u || NE > 32u * kRouterMaxPerLane) return;

    const uint per = (NE + 31u) / 32u;
    const uint base = lane * per;
    float ch[kRouterMaxPerLane];
    for (uint j = 0; j < per; ++j) {
        const uint e = base + j;
        ch[j] = (e < NE) ? logits[e] : -INFINITY;
    }

    uint taken = 0u;
    uint top_idx[8];
    for (uint k = 0; k < 8; ++k) {
        top_idx[k] = router_select_next_one_simd(ch, taken, per, base);
    }
    if (lane != 0u) return;

    // Fewer experts than K leaves trailing steps without a winner. The serial
    // kernel leaves those slots at index 0 with score -INFINITY; mirror that.
    float top_score[8];
    for (uint i = 0; i < 8; ++i) {
        if (top_idx[i] == kRouterNoWinner) {
            top_idx[i] = 0u;
            top_score[i] = -INFINITY;
        } else {
            top_score[i] = logits[top_idx[i]];
        }
    }

    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[8];
    for (uint i = 0; i < 8; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < 8; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

// Each SIMD computes one affine INT4 row. Four adjacent groups are loaded as
// aligned 32-bit chunks; remaining groups use one byte per lane.
static inline float moe_int4_gemv_row_simd_dev_vec(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const bfloat* s_row = S + uint(row) * n_groups;
    device const bfloat* b_row = B + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 = w4 & 0xFFu;
        const uint b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

// Gate and up rows share activation loads. Two 16-bit loads assemble each
// 4-byte weight chunk because packed sub-tensor offsets need only be 2-byte aligned.
static inline float2 moe_int4_gate_up_rows_simd_dev_vec_u16load(
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        const uint gb0 = gw4 & 0xFFu;
        const uint gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float g_dot = 0.0f;
        g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot);
        g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot);
        g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot);
        g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot);

        const uint ub0 = uw4 & 0xFFu;
        const uint ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot);
        u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot);
        u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot);
        u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint8_t gbv = gW_row[g * (kMoEGroupSize / 2) + lane];
        const uint8_t ubv = uW_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

static inline void moe_phase1_gate_up_act_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

static inline void moe_phase1_gate_up_act_subset_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    device const uint* active_slots,
    uint active_count,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * F) return;
    const uint active_idx = rowg / F;
    const uint slot = active_slots[active_idx];
    if (slot >= top_k) return;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_u16load_body(
        routed, routed_offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), rows_per_tg, tg_idx, sg_idx, lane);
}

// ---------------------------------------------------------------------------
// GPU-resident slot map (S3): resolve the router's top-k experts to slab
// byte offsets without the CPU, and run the whole routed FFN inside cb1 when
// every expert is already cached. The three guarded kernels are no-ops when
// `all_hit[0] == 0`; the CPU fallback then does the work exactly as before.
// The math bodies are shared with the production kernels, so an all-hit
// layer's output is byte-identical to the argument-buffer path.
// ---------------------------------------------------------------------------

kernel void router_slot_lookup_k8(
    device const uint* indices [[buffer(0)]],
    device const short* slot_of [[buffer(1)]],
    device uint* out_slot_offsets [[buffer(2)]],
    device uint* out_all_hit [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    constant uint& slot_stride [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    uint all_hit = 1u;
    for (uint k = 0; k < top_k; ++k) {
        const uint e = min(indices[k], num_experts - 1u);
        const short slot = slot_of[e];
        if (slot < 0) {
            all_hit = 0u;
            out_slot_offsets[k] = 0u;
        } else {
            out_slot_offsets[k] = uint(slot) * slot_stride;
        }
    }
    out_all_hit[0] = all_hit;
}

kernel void moe_phase1_gate_up_act_slotmap(
    device const uint8_t* slab [[buffer(0)]],
    device const uint* slot_offsets [[buffer(1)]],
    device const uint* all_hit [[buffer(2)]],
    constant ExpertOffsets& routed_offsets [[buffer(3)]],
    device const half* x [[buffer(4)]],
    device half* acts [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (all_hit[0] == 0u) return;
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    const uint K = moe_fc_top_k(top_k);
    const uint rowg = tg_idx * 8u + sg_idx;
    if (rowg >= K * FF) return;
    const uint slot = rowg / FF;
    const uint f = rowg % FF;

    device const uint8_t* base = slab + slot_offsets[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, DD, lane);
    if (lane == 0) acts[slot * FF + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_phase2_down_reduce_k8_slotmap(
    device const uint8_t* slab [[buffer(0)]],
    device const uint* slot_offsets [[buffer(1)]],
    device const uint* all_hit [[buffer(2)]],
    constant ExpertOffsets& routed_offsets [[buffer(3)]],
    device const half* acts [[buffer(4)]],
    device const half* routing_w [[buffer(5)]],
    device const half* residual [[buffer(6)]],
    device half* y [[buffer(7)]],
    constant uint& D [[buffer(8)]],
    constant uint& F [[buffer(9)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (all_hit[0] == 0u) return;
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = slab + slot_offsets[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int4_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void residual_add_fp16_guarded(
    device half*       hidden [[buffer(0)]],
    device const half* delta  [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    device const uint* all_hit [[buffer(3)]],
    uint               tid   [[thread_position_in_grid]]
) {
    if (all_hit[0] == 0u) return;
    if (tid >= count) return;
    hidden[tid] = half(float(hidden[tid]) + float(delta[tid]));
}

kernel void moe_phase1_gate_up_act_subset_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_subset_u16load_body(
        routed, routed_offsets, x, acts, active_slots, active_count,
        moe_fc_d(D), moe_fc_f(F), moe_fc_top_k(top_k), rows_per_tg,
        tg_idx, sg_idx, lane);
}

// --- DeepSeek V4: sqrtsoftplus routing (top-6) and INT2 experts -------------
//
// The V4 router scores logits with sqrt(softplus(x)). Selection adds a per-
// expert correction bias (selection only); weights are the raw scores of the
// selected experts, renormalized over the 6 and scaled by route_scale.

static inline float sqrtsoftplus(float x) {
    // softplus in a numerically safe form: max(x, 0) + log1p(exp(-|x|)).
    const float sp = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    return sqrt(sp);
}

// Serial single-thread selection, superseded in production by
// `router_topk_select_sqrtsoftplus_k6_par`; kept as the parity reference.
kernel void router_topk_select_sqrtsoftplus_k6(
    device const float* logits [[buffer(0)]],
    device const float* correction_bias [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant float& route_scale [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    const uint NE = router_fc_num_experts(num_experts);
    uint top_idx[6];
    float top_sel[6];
    for (uint i = 0; i < 6; ++i) {
        top_idx[i] = 0u;
        top_sel[i] = -INFINITY;
    }

    for (uint e = 0; e < NE; ++e) {
        const float s = sqrtsoftplus(logits[e]) + correction_bias[e];
        if (s <= top_sel[5]) continue;
        uint pos = 6u;
        for (uint i = 0; i < 6; ++i) {
            if (s > top_sel[i] || (s == top_sel[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= 6u) continue;
        for (uint i = 5; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_sel[i] = top_sel[i - 1];
        }
        top_idx[pos] = e;
        top_sel[pos] = s;
    }

    float scores[6];
    float sum = 0.0f;
    for (uint i = 0; i < 6; ++i) {
        scores[i] = sqrtsoftplus(logits[top_idx[i]]);
        sum += scores[i];
    }
    const float inv = route_scale / (sum + 1e-20f);
    for (uint i = 0; i < 6; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = half(scores[i] * inv);
    }
}

// One-SIMD replacement for `router_topk_select_sqrtsoftplus_k6`. Selection uses
// the same blocked-ownership argmax reduction as `router_topk_select_k8_par`;
// the score is sqrtsoftplus(logit) + correction bias, and the weight tail is
// the serial code verbatim on lane 0, so the emitted indices and FP16 weights
// are bit-identical.
kernel void router_topk_select_sqrtsoftplus_k6_par(
    device const float* logits [[buffer(0)]],
    device const float* correction_bias [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant float& route_scale [[buffer(5)]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint NE = router_fc_num_experts(num_experts);
    if (sg_idx != 0u || NE > 32u * kRouterMaxPerLane) return;

    const uint per = (NE + 31u) / 32u;
    const uint base = lane * per;
    float ch[kRouterMaxPerLane];
    for (uint j = 0; j < per; ++j) {
        const uint e = base + j;
        ch[j] = (e < NE) ? (sqrtsoftplus(logits[e]) + correction_bias[e])
                         : -INFINITY;
    }

    uint taken = 0u;
    uint top_idx[6];
    for (uint k = 0; k < 6; ++k) {
        top_idx[k] = router_select_next_one_simd(ch, taken, per, base);
    }
    if (lane != 0u) return;

    // The serial kernel leaves unfilled slots at index 0; mirror that so the
    // sqrtsoftplus(logits[0]) contribution below matches.
    for (uint i = 0; i < 6; ++i) {
        if (top_idx[i] == kRouterNoWinner) top_idx[i] = 0u;
    }
    float scores[6];
    float sum = 0.0f;
    for (uint i = 0; i < 6; ++i) {
        scores[i] = sqrtsoftplus(logits[top_idx[i]]);
        sum += scores[i];
    }
    const float inv = route_scale / (sum + 1e-20f);
    for (uint i = 0; i < 6; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = half(scores[i] * inv);
    }
}

// Hash-routed layers: expert indices are a frozen tid2eid[token] lookup done
// on the CPU; only the weights come from the learned gate. Scores the 6
// given experts, renormalizes, scales.
//
// Serial reference; production uses `router_hash_weights_k6_par`.
kernel void router_hash_weights_k6(
    device const float* logits [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device half* out_weights [[buffer(2)]],
    constant float& route_scale [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    float scores[6];
    float sum = 0.0f;
    for (uint i = 0; i < 6; ++i) {
        scores[i] = sqrtsoftplus(logits[indices[i]]);
        sum += scores[i];
    }
    const float inv = route_scale / (sum + 1e-20f);
    for (uint i = 0; i < 6; ++i) {
        out_weights[i] = half(scores[i] * inv);
    }
}

// One-SIMD variant of `router_hash_weights_k6`: lanes 0..5 evaluate the six
// sqrtsoftplus scores concurrently, then lane 0 runs the serial accumulation
// and normalization in the original order. `simd_shuffle` moves exact bits, so
// the result matches the serial kernel exactly.
kernel void router_hash_weights_k6_par(
    device const float* logits [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device half* out_weights [[buffer(2)]],
    constant float& route_scale [[buffer(3)]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (sg_idx != 0u) return;
    const float own = (lane < 6u) ? sqrtsoftplus(logits[indices[lane]]) : 0.0f;
    float scores[6];
    for (uint i = 0; i < 6; ++i) scores[i] = simd_shuffle(own, i);
    if (lane != 0u) return;

    float sum = 0.0f;
    for (uint i = 0; i < 6; ++i) sum += scores[i];
    const float inv = route_scale / (sum + 1e-20f);
    for (uint i = 0; i < 6; ++i) {
        out_weights[i] = half(scores[i] * inv);
    }
}

// Each SIMD computes one affine INT2 row: 16 elements per u32, group-64
// scales/biases (16 bytes of packed weights per group). Blocks of 4 groups
// load 2 bytes per lane (8 elements); packed sub-tensor offsets are 2-byte
// aligned, so ushort loads are safe.
static inline float moe_int2_gemv_row_simd_dev_vec(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 4;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const bfloat* s_row = S + uint(row) * n_groups;
    device const bfloat* b_row = B + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 64u + lane * 2u;
        const uint w2 = uint(*((device const ushort*)(W_row + byte_base)));
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 4u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(w2 & 0x3u), e0, dot);
        dot = fma(float((w2 >> 2) & 0x3u), e1, dot);
        dot = fma(float((w2 >> 4) & 0x3u), e2, dot);
        dot = fma(float((w2 >> 6) & 0x3u), e3, dot);
        dot = fma(float((w2 >> 8) & 0x3u), e4, dot);
        dot = fma(float((w2 >> 10) & 0x3u), e5, dot);
        dot = fma(float((w2 >> 12) & 0x3u), e6, dot);
        dot = fma(float((w2 >> 14) & 0x3u), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        // Two elements per lane: byte k holds elements 4k..4k+3, element j at
        // bit 2*(j%4).
        const uint8_t byte = W_row[g * (kMoEGroupSize / 4) + (lane >> 1)];
        const uint shift = (lane & 1u) * 4u;
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint((byte >> shift) & 0x3u)), x0, 0.0f);
        dot = fma(float(uint((byte >> (shift + 2u)) & 0x3u)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

// `gate_group_size` may be 32 or 64 (DeepSeek V4 ships most layers'
// gate_proj at group 32); the up projection always uses kMoEGroupSize.
// The packed 2-bit weight layout is group-size independent — only the
// scale/bias row stride and per-element group index change.
static inline float2 moe_int2_gate_up_rows_simd_dev_vec(
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    device const half* x,
    uint row,
    uint N,
    uint gate_group_size,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint gate_n_groups = N / gate_group_size;
    const uint row_bytes = N / 4;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * gate_n_groups;
    device const bfloat* gB_row = gateB + uint(row) * gate_n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 64u + lane * 2u;
        const uint gw2 = uint(*((device const ushort*)(gW_row + byte_base)));
        const uint uw2 = uint(*((device const ushort*)(uW_row + byte_base)));
        const uint g = blk * 4u + (lane >> 3);
        // Each lane's 8 contiguous elements sit inside one group for any
        // group size >= 8 with this block layout.
        const uint gg = (byte_base * 4u) / gate_group_size;
        const float gs = float(gS_row[gg]);
        const float gb = float(gB_row[gg]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint elem = byte_base * 4u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        float g_dot = 0.0f;
        g_dot = fma(float(gw2 & 0x3u), e0, g_dot);
        g_dot = fma(float((gw2 >> 2) & 0x3u), e1, g_dot);
        g_dot = fma(float((gw2 >> 4) & 0x3u), e2, g_dot);
        g_dot = fma(float((gw2 >> 6) & 0x3u), e3, g_dot);
        g_dot = fma(float((gw2 >> 8) & 0x3u), e4, g_dot);
        g_dot = fma(float((gw2 >> 10) & 0x3u), e5, g_dot);
        g_dot = fma(float((gw2 >> 12) & 0x3u), e6, g_dot);
        g_dot = fma(float((gw2 >> 14) & 0x3u), e7, g_dot);

        float u_dot = 0.0f;
        u_dot = fma(float(uw2 & 0x3u), e0, u_dot);
        u_dot = fma(float((uw2 >> 2) & 0x3u), e1, u_dot);
        u_dot = fma(float((uw2 >> 4) & 0x3u), e2, u_dot);
        u_dot = fma(float((uw2 >> 6) & 0x3u), e3, u_dot);
        u_dot = fma(float((uw2 >> 8) & 0x3u), e4, u_dot);
        u_dot = fma(float((uw2 >> 10) & 0x3u), e5, u_dot);
        u_dot = fma(float((uw2 >> 12) & 0x3u), e6, u_dot);
        u_dot = fma(float((uw2 >> 14) & 0x3u), e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        // `g` walks 64-element chunks (== up groups); the gate group is
        // derived per lane so a 32-wide gate grouping stays correct.
        const uint gg = (g * kMoEGroupSize + lane * 2u) / gate_group_size;
        const float gs = float(gS_row[gg]);
        const float gb = float(gB_row[gg]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint8_t gbv = gW_row[g * (kMoEGroupSize / 4) + (lane >> 1)];
        const uint8_t ubv = uW_row[g * (kMoEGroupSize / 4) + (lane >> 1)];
        const uint shift = (lane & 1u) * 4u;
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint((gbv >> shift) & 0x3u)), x0, 0.0f);
        g_dot = fma(float(uint((gbv >> (shift + 2u)) & 0x3u)), x1, g_dot);
        float u_dot = fma(float(uint((ubv >> shift) & 0x3u)), x0, 0.0f);
        u_dot = fma(float(uint((ubv >> (shift + 2u)) & 0x3u)), x1, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

static inline void moe_phase1_int2_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    uint D,
    uint F,
    uint top_k,
    uint gate_group_size,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    const float2 gu = moe_int2_gate_up_rows_simd_dev_vec(
        base + re.gate_W_off,
        (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off),
        base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off),
        x, f, D, gate_group_size, lane);
    if (lane == 0) {
        const float2 cgu = moe_swiglu_clamp(gu.x, gu.y);
        acts[slot * F + f] = half(moe_hidden_activation(cgu.x) * cgu.y);
    }
}

kernel void moe_phase1_gate_up_act_int2(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    constant uint& gate_group_size [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    moe_phase1_int2_body(
        routed, routed_offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), gate_group_size, rows_per_tg, tg_idx, sg_idx, lane);
}

kernel void moe_phase1_gate_up_act_subset_int2(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]],
    constant uint& gate_group_size [[buffer(9)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    const uint F_ = moe_fc_f(F);
    if (rowg >= active_count * F_) return;
    const uint active_idx = rowg / F_;
    const uint slot = active_slots[active_idx];
    if (slot >= moe_fc_top_k(top_k)) return;
    const uint f = rowg % F_;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    const float2 gu = moe_int2_gate_up_rows_simd_dev_vec(
        base + re.gate_W_off,
        (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off),
        base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off),
        x, f, moe_fc_d(D), gate_group_size, lane);
    if (lane == 0) {
        const float2 cgu = moe_swiglu_clamp(gu.x, gu.y);
        acts[slot * F_ + f] = half(moe_hidden_activation(cgu.x) * cgu.y);
    }
}

kernel void moe_phase2_down_reduce_int2_k6(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[6];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;
    if (sg_idx >= 6u) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int2_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2];
        acc += partial[3]; acc += partial[4]; acc += partial[5];
        y[d] = half(acc);
    }
}

kernel void moe_phase2_down_reduce_k6(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[6];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int4_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2];
        acc += partial[3]; acc += partial[4]; acc += partial[5];
        y[d] = half(acc);
    }
}

kernel void moe_phase2_down_reduce_k8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int4_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}

// ============================================================================
// DeepSeek-V4 chunked-prefill routed MoE — pair-major (additive; Wave-2B).
//
// Decode runs one token at a time: six experts are streamed in, phase 1 fills
// `acts[slot]`, and `moe_phase2_down_reduce_int2_k6` fuses the down projection
// with the weighted reduce. Prefill cannot afford that: every token re-reads
// its six 8 MB expert blobs, so a 629-token prompt reads ~1.3 TB.
//
// The prefill variants below are *pair-major*: the chunk's (token, rank)
// routes are sorted by expert and processed one expert tile at a time, so an
// expert blob is read once per chunk instead of once per token. The
// arithmetic is unchanged — the same `moe_int2_*` helpers, the same clamp,
// the same activation, the same `routing_w * value` product, the same
// residual-first FP32 accumulation in rank order — so results are
// bit-identical to the decode path. The only structural difference is that the
// down projection writes an FP32 per-route partial (decode keeps it in
// threadgroup memory), which carries strictly more precision than the
// threadgroup float the fused kernel reduces.
// ============================================================================

struct DSV4PrefillRoute {
    uint token;       // row of `x` / of the output
    uint rank;        // routing slot 0..top_k-1 (selects the routing weight)
    uint local_slot;  // index into the tile's RoutedBlobs
    uint reserved;
};

// acts[(token * top_k + rank) * F + f], one simdgroup per (route, f) row.
kernel void dsv4_prefill_moe_phase1_pairs_int2(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],            // [tokens, D]
    device half* acts [[buffer(3)]],               // [tokens * top_k, F]
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    constant uint& gate_group_size [[buffer(7)]],
    device const DSV4PrefillRoute* routes [[buffer(8)]],
    constant uint& route_start [[buffer(9)]],
    constant uint& route_count [[buffer(10)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    const uint KK = moe_fc_top_k(top_k);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= route_count * FF) return;
    const uint route_index = rowg / FF;
    const uint f = rowg % FF;
    const DSV4PrefillRoute r = routes[route_start + route_index];
    if (r.local_slot >= kMaxStreamedExperts) return;

    device const uint8_t* base = routed.blob[r.local_slot];
    const ExpertOffsets re = routed_offsets;
    const float2 gu = moe_int2_gate_up_rows_simd_dev_vec(
        base + re.gate_W_off,
        (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off),
        base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off),
        x + uint(r.token) * DD, f, DD, gate_group_size, lane);
    if (lane == 0) {
        const float2 cgu = moe_swiglu_clamp(gu.x, gu.y);
        acts[(uint(r.token) * KK + uint(r.rank)) * FF + f] =
            half(moe_hidden_activation(cgu.x) * cgu.y);
    }
}

// partials[(token * top_k + rank) * D + d] = routing_w[...] * down_row(acts).
// FP32 output so the reduce below matches the fused decode kernel exactly.
kernel void dsv4_prefill_moe_down_pairs_int2(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],         // [tokens * top_k, F]
    device const half* routing_w [[buffer(3)]],    // [tokens * top_k]
    device float* partials [[buffer(4)]],          // [tokens * top_k, D]
    constant uint& D [[buffer(5)]],
    constant uint& F [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],
    device const DSV4PrefillRoute* routes [[buffer(8)]],
    constant uint& route_start [[buffer(9)]],
    constant uint& route_count [[buffer(10)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    const uint KK = moe_fc_top_k(top_k);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= route_count * DD) return;
    const uint route_index = rowg / DD;
    const uint d = rowg % DD;
    const DSV4PrefillRoute r = routes[route_start + route_index];
    if (r.local_slot >= kMaxStreamedExperts) return;
    const uint pair = uint(r.token) * KK + uint(r.rank);

    device const uint8_t* base = routed.blob[r.local_slot];
    const ExpertOffsets re = routed_offsets;
    const float value = moe_int2_gemv_row_simd_dev_vec(
        base + re.down_W_off,
        (device const bfloat*)(base + re.down_s_off),
        (device const bfloat*)(base + re.down_b_off),
        acts + pair * FF, d, FF, lane);
    if (lane == 0) {
        partials[pair * DD + d] = float(routing_w[pair]) * value;
    }
}

// y[token] = residual[token] + sum_{rank} partials[token, rank] — the same
// residual-first, rank-ordered FP32 accumulation as
// `moe_phase2_down_reduce_int2_k6`.
kernel void dsv4_prefill_moe_reduce_pairs_k6(
    device const float* partials [[buffer(0)]],    // [tokens * 6, D]
    device const half* residual [[buffer(1)]],     // [tokens, D]
    device half* y [[buffer(2)]],                  // [tokens, D]
    constant uint& D [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint DD = moe_fc_d(D);
    if (gid.x >= DD) return;
    const uint row = gid.y * DD + gid.x;
    const uint base = gid.y * 6u * DD + gid.x;
    float acc = float(residual[row]);
    acc += partials[base + 0u * DD];
    acc += partials[base + 1u * DD];
    acc += partials[base + 2u * DD];
    acc += partials[base + 3u * DD];
    acc += partials[base + 4u * DD];
    acc += partials[base + 5u * DD];
    y[row] = half(acc);
}
