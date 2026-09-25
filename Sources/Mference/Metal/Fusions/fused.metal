#include <metal_stdlib>
using namespace metal;

#ifndef MFERENCE_GEMMA_SOURCE_NORM
#define MFERENCE_GEMMA_SOURCE_NORM
// Preserve MLX's four-contiguous-value partials and SIMD merge order while
// retaining our 256-thread dispatch. Explicit stores and precise division
// preserve its arithmetic inside the shared fast-math library.
// The pointer template handles resident/device and fused/threadgroup inputs.
template <typename InputPointer>
static inline float gemma_source_norm_inv(
    InputPointer x, uint D, float eps, uint lane, uint sg, uint sgs,
    threadgroup float* partial
) {
    const uint groups = min(32u, (D + 127u) / 128u);
    for (uint group = sg; group < groups; group += sgs) {
        volatile float acc = 0.0f;
        for (uint base = (group * 32u + lane) * 4u; base < D; base += 4096u) {
            for (uint j = 0; j < 4u; ++j) {
                const float v = base + j < D ? float(x[base + j]) : 0.0f;
                acc = acc + v * v;
            }
        }
        const float sum = simd_sum(float(acc));
        if (lane == 0u) partial[group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const float sum = simd_sum(lane < groups ? partial[lane] : 0.0f);
        if (lane == 0u) {
            volatile float mean = precise::divide(sum, float(D));
            volatile float regularized = mean + eps;
            partial[0] = precise::rsqrt(regularized);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}
#endif


#ifndef MFERENCE_GEMMA_SOURCE_FP16
#define MFERENCE_GEMMA_SOURCE_FP16
// Preserve the source activation boundary before learned scaling. The original
// Gemma checkpoint keeps its existing fused arithmetic unless explicitly set.
constant bool FC_GEMMA_SOURCE_FP16 [[function_constant(110)]];
constant bool kGemmaSourceFP16 = is_function_constant_defined(FC_GEMMA_SOURCE_FP16)
    ? FC_GEMMA_SOURCE_FP16 : false;
static inline half gemma_weighted_norm(float x, float inv, float weight) {
    return kGemmaSourceFP16 ? half(x * inv) * half(weight) : half(x * inv * weight);
}
static inline half gemma_scaled_embedding(float value, float scale) {
    return kGemmaSourceFP16 ? half(value) * half(scale) : half(value * scale);
}

// MLX's quantized GEMV forms its affine-bias input sum in FP16 quads.
static inline float gemma_source_quad_sum(half4 x) {
    const half a = x.x + x.y;
    const half b = a + x.z;
    return float(half(b + x.w));
}
static inline half gemma_source_geglu(float gate_value, float up_value) {
    // Gate/up projections are FP16 tensors in the source, even when fused.
    volatile half gate = half(gate_value);
    volatile half up = half(up_value);
    // Preserve source half stores across fast-math contraction, particularly
    // 1 + tanh(x): its rounded negative tail is exactly zero in the source.
    volatile half cube = half(float(gate) * float(gate) * float(gate));
    volatile half term = half(0.044715f) * cube;
    volatile half sum = gate + term;
    volatile half inner = half(0.7978845608028654f) * sum;
    // FP16 tanh has already rounded to +/-1 at these bounds. Keep that
    // exact source result while avoiding fast-tanh's positive exp overflow.
    volatile half curve = half(tanh(clamp(float(inner), -20.0f, 20.0f)));
    volatile half shifted = half(1.0h + curve);
    volatile half scaled = half(0.5h * gate);
    volatile half activation = scaled * shifted;
    return half(activation * up);
}
static inline float gemma_source_bias_correction(
    device const half* x, device const bfloat* biases, uint width, uint group_size
) {
    float correction = 0.0f;
    for (uint k = 0; k < width; k += 4u) {
        const half4 quad(x[k], x[k + 1u], x[k + 2u], x[k + 3u]);
        const float exact = float(quad.x) + float(quad.y) + float(quad.z) + float(quad.w);
        correction = fma(float(biases[k / group_size]),
                        gemma_source_quad_sum(quad) - exact, correction);
    }
    return correction;
}
#endif

// ============================================================================
// fused.metal — current decode fusions:
//   fused_qkv_epilogue:     per-head Q/K/V RMSNorm + NeoX RoPE after projection.
//   fused_post_attn_setup:  post-attention residual + pre-FFN/router norms.
//   fused_layer_tail:       real Gemma 4 decoder-layer tail.
// ============================================================================

constant constexpr uint kFusedThreads        = 256;
constant constexpr uint kFusedMaxSimdGroups  = 32;  // QAT uses up to 32 logical SIMD partials
constant constexpr uint kFusedGroupSize      = 64;
// Threadgroup-memory cap for the normalized-hidden staging slot. Sized for
// D=2816 (production hidden dim) with headroom; consumes 8 KB per threadgroup.
constant constexpr uint kFusedMaxD           = 4096;
constant constexpr uint kFusedMaxHeadDim     = 512;
constant uint FC_FUSED_D [[function_constant(80)]];
constant uint FC_FUSED_N [[function_constant(81)]];
constant uint FC_FUSED_HEAD_DIM [[function_constant(82)]];
constant uint FC_FUSED_NUM_Q_HEADS [[function_constant(83)]];
constant uint FC_FUSED_NUM_KV_HEADS [[function_constant(84)]];
constant uint FC_FUSED_ROTARY [[function_constant(85)]];
constant bool FC_FUSED_USE_FC [[function_constant(86)]];

static inline bool fused_use_fc() {
    return is_function_constant_defined(FC_FUSED_USE_FC) && FC_FUSED_USE_FC;
}

static inline uint fused_fc_d(constant uint& D) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_D)) ? FC_FUSED_D : D;
}

static inline uint fused_fc_n(constant uint& N) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_N)) ? FC_FUSED_N : N;
}

static inline uint fused_fc_head_dim(constant uint& head_dim) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_HEAD_DIM)) ? FC_FUSED_HEAD_DIM : head_dim;
}

static inline uint fused_fc_num_q_heads(constant uint& num_q_heads) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_NUM_Q_HEADS)) ? FC_FUSED_NUM_Q_HEADS : num_q_heads;
}

static inline uint fused_fc_num_kv_heads(constant uint& num_kv_heads) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_NUM_KV_HEADS)) ? FC_FUSED_NUM_KV_HEADS : num_kv_heads;
}

static inline uint fused_fc_rotary(constant uint& rotary) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_ROTARY)) ? FC_FUSED_ROTARY : rotary;
}

inline void fused_rope_neox_pair(thread float& x0,
                                 thread float& x1,
                                 uint pair_index,
                                 uint head_dim,
                                 float position,
                                 float theta_base,
                                 bool proportional)
{
    const float exponent = -float(2u * pair_index) / float(head_dim);
    // Source proportional RoPE constructs positive powers in FP32, then
    // reciprocates them. Reversing the exponent is not rounding-equivalent.
    const float freq = kGemmaSourceFP16 && proportional
        ? precise::divide(1.0f, precise::pow(theta_base, -exponent))
        : pow(theta_base, exponent);
    const float angle    = position * freq;
    const float c = kGemmaSourceFP16 ? fast::cos(angle) : cos(angle);
    const float s = kGemmaSourceFP16 ? fast::sin(angle) : sin(angle);
    const float r0 = x0 * c - x1 * s;
    const float r1 = x0 * s + x1 * c;
    x0 = r0;
    x1 = r1;
}

// ============================================================================
// fused_qkv_epilogue — real Gemma 4 Q/K/V per-head post-processing.
//
// Replaces the cb1 chain after Q/K/V projection:
//     rmsnorm_bf16w_perhead(Q, q_norm) -> rmsnorm_bf16w_perhead(K, k_norm)
//     -> rmsnorm_no_scale_perhead(V) -> rope_neox(Q) -> rope_neox(K)
//
// One threadgroup owns one logical head. Q and K use BF16 weights shared across
// heads; V uses no-scale RMSNorm. For Q/K, the normalized per-head values are
// rounded to half in threadgroup memory before RoPE, preserving the standalone
// kernel boundary where RMSNorm writes FP16 and RoPE reads FP16.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kFusedThreads)]]
void fused_qkv_epilogue(
    device       half*   q              [[buffer(0)]],  // [num_q_heads, head_dim]
    device       half*   k              [[buffer(1)]],  // [num_kv_heads, head_dim]
    device       half*   v              [[buffer(2)]],  // [num_kv_heads, head_dim]
    device const bfloat* q_weight       [[buffer(3)]],  // [head_dim]
    device const bfloat* k_weight       [[buffer(4)]],  // [head_dim]
    constant     uint&   head_dim       [[buffer(5)]],
    constant     uint&   num_q_heads    [[buffer(6)]],
    constant     uint&   num_kv_heads   [[buffer(7)]],
    constant     uint&   position       [[buffer(8)]],
    constant     float&  theta_base     [[buffer(9)]],
    constant     uint&   rotated_pairs  [[buffer(10)]],
    constant     float&  rms_eps        [[buffer(11)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]],
    uint  head_group       [[threadgroup_position_in_grid]]
) {
    threadgroup half  head_tg[kFusedMaxHeadDim];
    threadgroup float partial[kFusedMaxSimdGroups];

    const uint HD = fused_fc_head_dim(head_dim);
    const uint NQ = fused_fc_num_q_heads(num_q_heads);
    const uint NKV = fused_fc_num_kv_heads(num_kv_heads);
    const uint RP = fused_fc_rotary(rotated_pairs);

    const bool is_q = head_group < NQ;
    const bool is_k = !is_q && head_group < (NQ + NKV);
    const bool is_v = !is_q && !is_k && head_group < (NQ + 2u * NKV);
    if (!is_q && !is_k && !is_v) return;

    const uint local_head = is_q ? head_group : (head_group - NQ) % NKV;
    device half* dst = is_q ? (q + local_head * HD)
                    : (is_k ? (k + local_head * HD)
                            : (v + local_head * HD));
    device const half* src = dst;
    device const bfloat* w = is_q ? q_weight : k_weight;

    if (kGemmaSourceFP16) {
        gemma_source_norm_inv(src, HD, rms_eps, simd_lane_id, simd_group_id, simdgroups, partial);
    } else {
        float acc = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            float xv = float(src[i]);
            acc = fma(xv, xv, acc);
        }
        acc = simd_sum(acc);
        if (simd_lane_id == 0) {
            partial[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float sum = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
            sum = simd_sum(sum);
            if (simd_lane_id == 0) {
                partial[0] = rsqrt(sum / float(HD) + rms_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = partial[0];
    for (uint i = lid; i < HD; i += lsize) {
        head_tg[i] = is_v ? half(float(src[i]) * inv)
            : gemma_weighted_norm(float(src[i]), inv, float(w[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_v) {
        for (uint i = lid; i < HD; i += lsize) {
            dst[i] = head_tg[i];
        }
        return;
    }

    const uint half_dim = HD / 2u;
    for (uint pair = lid; pair < half_dim; pair += lsize) {
        float x0 = float(head_tg[pair]);
        float x1 = float(head_tg[half_dim + pair]);
        if (pair < RP) {
            fused_rope_neox_pair(x0, x1, pair, HD, float(position), theta_base, RP * 2u < HD);
        }
        dst[pair] = half(x0);
        dst[half_dim + pair] = half(x1);
    }
}

// ============================================================================
// fused_layer_tail — real Gemma 4 decoder-layer tail.
//
// Replaces the cb2 chain:
//     rmsnorm_bf16w(h2, post_ffn_2) -> residual_add(h1, ...)
//     -> rmsnorm_bf16w(..., post_ffn) -> residual_add(hidden, ...)
//     -> layer-scalar multiply
//
// The two RMSNorm reductions copy rmsnorm.metal's two-stage reduction order.
// Elementwise adds and scalar multiply preserve the same FP16 stage boundaries
// as the standalone kernels: h1+h2_norm is rounded to half before the second
// reduction, and hidden+post_norm is rounded to half before multiplying by the
// half-cast layer scalar.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kFusedThreads)]]
void fused_post_attn_setup(
    device       half*   hidden         [[buffer(0)]],  // [D] FP16 in-place
    device const half*   attn           [[buffer(1)]],  // [D] FP16
    device       half*   dense_x        [[buffer(2)]],  // [D] FP16
    device       half*   routed_x       [[buffer(3)]],  // [D] FP16
    device       half*   router_x       [[buffer(4)]],  // [D] FP16
    device const bfloat* w_post_attn    [[buffer(5)]],  // [D] BF16
    device const bfloat* w_pre_ffn      [[buffer(6)]],  // [D] BF16
    device const bfloat* w_pre_ffn2     [[buffer(7)]],  // [D] BF16
    constant     uint&   D              [[buffer(8)]],
    constant     float&  rms_eps        [[buffer(9)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup half  attn_norm_tg[kFusedMaxD];
    threadgroup half  hidden_tg[kFusedMaxD];
    threadgroup float partial[kFusedMaxSimdGroups];
    const uint DD = fused_fc_d(D);

    if (kGemmaSourceFP16) {
        gemma_source_norm_inv(attn, DD, rms_eps, simd_lane_id, simd_group_id, simdgroups, partial);
    } else {
        float acc = 0.0f;
        for (uint i = lid; i < DD; i += lsize) {
            float v = float(attn[i]);
            acc = fma(v, v, acc);
        }
        acc = simd_sum(acc);
        if (simd_lane_id == 0) {
            partial[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float sum = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
            sum = simd_sum(sum);
            if (simd_lane_id == 0) {
                partial[0] = rsqrt(sum / float(DD) + rms_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float attn_inv = partial[0];
    for (uint i = lid; i < DD; i += lsize) {
        attn_norm_tg[i] = gemma_weighted_norm(float(attn[i]), attn_inv, float(w_post_attn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = lid; i < DD; i += lsize) {
        half h = half(float(hidden[i]) + float(attn_norm_tg[i]));
        hidden_tg[i] = h;
        hidden[i] = h;
        float hf = float(h);
        acc = fma(hf, hf, acc);
    }
    if (kGemmaSourceFP16) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemma_source_norm_inv(hidden_tg, DD, rms_eps, simd_lane_id, simd_group_id, simdgroups, partial);
    } else {
        acc = simd_sum(acc);
        if (simd_lane_id == 0) {
            partial[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float sum = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
            sum = simd_sum(sum);
            if (simd_lane_id == 0) {
                partial[0] = rsqrt(sum / float(DD) + rms_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float hidden_inv = partial[0];
    for (uint i = lid; i < DD; i += lsize) {
        const float h = float(hidden_tg[i]) * hidden_inv;
        dense_x[i] = gemma_weighted_norm(float(hidden_tg[i]), hidden_inv, float(w_pre_ffn[i]));
        routed_x[i] = gemma_weighted_norm(float(hidden_tg[i]), hidden_inv, float(w_pre_ffn2[i]));
        router_x[i] = half(h);
    }
}

[[kernel, max_total_threads_per_threadgroup(kFusedThreads)]]
void fused_layer_tail(
    device const half*   h2             [[buffer(0)]],  // [D] FP16
    device const half*   h1             [[buffer(1)]],  // [D] FP16
    device       half*   hidden         [[buffer(2)]],  // [D] FP16 in-place
    device const bfloat* w_postffn2     [[buffer(3)]],  // [D] BF16
    device const bfloat* w_postffn      [[buffer(4)]],  // [D] BF16
    constant     uint&   D              [[buffer(5)]],
    constant     float&  rms_eps        [[buffer(6)]],
    constant     float&  layer_scalar   [[buffer(7)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup half  tmp_tg[kFusedMaxD];
    threadgroup half  h12_tg[kFusedMaxD];
    threadgroup float partial[kFusedMaxSimdGroups];
    const uint DD = fused_fc_d(D);

    if (kGemmaSourceFP16) {
        gemma_source_norm_inv(h2, DD, rms_eps, simd_lane_id, simd_group_id, simdgroups, partial);
    } else {
        float acc = 0.0f;
        for (uint i = lid; i < DD; i += lsize) {
            float v = float(h2[i]);
            acc = fma(v, v, acc);
        }
        acc = simd_sum(acc);
        if (simd_lane_id == 0) {
            partial[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
            v = simd_sum(v);
            if (simd_lane_id == 0) {
                float mean_sq = v / float(DD);
                partial[0] = rsqrt(mean_sq + rms_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_h2 = partial[0];
    for (uint i = lid; i < DD; i += lsize) {
        tmp_tg[i] = gemma_weighted_norm(float(h2[i]), inv_h2, float(w_postffn2[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint D4 = DD / 4u;
    threadgroup half4* h12_4 = reinterpret_cast<threadgroup half4*>(h12_tg);
    threadgroup half4* tmp_4 = reinterpret_cast<threadgroup half4*>(tmp_tg);
    device const half4* h1_4 = reinterpret_cast<device const half4*>(h1);
    for (uint i = lid; i < D4; i += lsize) {
        h12_4[i] = h1_4[i] + tmp_4[i];
    }
    const uint tailStart = D4 * 4u;
    for (uint i = tailStart + lid; i < DD; i += lsize) {
        h12_tg[i] = h1[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (kGemmaSourceFP16) {
        gemma_source_norm_inv(h12_tg, DD, rms_eps, simd_lane_id, simd_group_id, simdgroups, partial);
    } else {
        float acc = 0.0f;
        for (uint i = lid; i < DD; i += lsize) {
            float v = float(h12_tg[i]);
            acc = fma(v, v, acc);
        }
        acc = simd_sum(acc);
        if (simd_lane_id == 0) {
            partial[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
            v = simd_sum(v);
            if (simd_lane_id == 0) {
                float mean_sq = v / float(DD);
                partial[0] = rsqrt(mean_sq + rms_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_h12 = partial[0];
    for (uint i = lid; i < DD; i += lsize) {
        tmp_tg[i] = gemma_weighted_norm(float(h12_tg[i]), inv_h12, float(w_postffn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half4* hidden4 = reinterpret_cast<device half4*>(hidden);
    for (uint i = lid; i < D4; i += lsize) {
        hidden4[i] = hidden4[i] + tmp_4[i];
    }
    for (uint i = tailStart + lid; i < DD; i += lsize) {
        hidden[i] = hidden[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    const half hScale = half(layer_scalar);
    for (uint i = lid; i < D4; i += lsize) {
        hidden4[i] = hidden4[i] * hScale;
    }
    for (uint i = tailStart + lid; i < DD; i += lsize) {
        hidden[i] = hidden[i] * hScale;
    }
}
