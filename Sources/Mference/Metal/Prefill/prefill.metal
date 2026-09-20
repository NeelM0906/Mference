#include <metal_stdlib>
using namespace metal;

#ifndef MFERENCE_GEMMA_SOURCE_ROUTER_DOT
#define MFERENCE_GEMMA_SOURCE_ROUTER_DOT
// MLX 0.32.2 gemv.h reduction, adapted to one physical SIMD per expert.
// Copyright © 2023-2024 Apple Inc. MIT license; see LICENSE-MLX.
// The pinned 128x2816 router uses eight logical SIMD groups, four contiguous
// values per lane, then an ordered merge. Compile its entry points safely.
static inline float gemma_source_router_dot(
    device const bfloat* row, device const half* input,
    device const half* scale, uint D, uint lane
) {
    float total = 0.0f;
    for (uint group = 0; group < 8u; ++group) {
        float sum = 0.0f;
        for (uint base = group * 128u + lane * 4u; base < D; base += 1024u) {
            for (uint j = 0; j < 4u; ++j) {
                const half x = input[base + j] * scale[base + j];
                sum += float(half(row[base + j])) * float(x);
            }
        }
        for (ushort delta = 16; delta > 0; delta >>= 1) sum += simd_shuffle_down(sum, delta);
        total = group == 0 ? sum : total + sum;
    }
    return total;
}
#endif

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

#ifndef MFERENCE_GEMMA_ROUTING_PRECISION
#define MFERENCE_GEMMA_ROUTING_PRECISION
static inline float gemma_source_weighted_expert(float value, half weight) {
    // A fused projection must retain both source tensor-store boundaries.
    volatile half projected = half(value);
    volatile half product = projected * weight;
    return float(product);
}

static inline void gemma_source_softmax8(
    thread const float* descending_scores, thread half* probabilities
) {
    // Pinned MLX's argpartition is an ascending sort. Its default FP16
    // softmax uses two lanes with four ascending values each, half partial
    // sums and a half reciprocal. Keep our descending route-slot order while
    // reproducing that source normalization order and precision.
    volatile half exps[8];
    for (uint i = 0; i < 8u; ++i) {
        volatile half shifted = half(descending_scores[i]) - half(descending_scores[0]);
        exps[i] = fast::exp(shifted);
    }
    volatile half lower = 0.0h;
    volatile half upper = 0.0h;
    for (uint i = 0; i < 4u; ++i) lower = lower + exps[7u - i];
    for (uint i = 0; i < 4u; ++i) upper = upper + exps[3u - i];
    volatile half total = lower + upper;
    volatile half inverse = 1.0h / total;
    for (uint i = 0; i < 8u; ++i) probabilities[i] = exps[i] * inverse;
}
#endif

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#endif

constant constexpr uint kPrefillGroupSize = 64;
#ifndef MFERENCE_AFFINE_GROUP_SIZE
#define MFERENCE_AFFINE_GROUP_SIZE
constant uint FC_AFFINE_GROUP_SIZE [[function_constant(108)]];
constant uint kAffineGroupSize = is_function_constant_defined(FC_AFFINE_GROUP_SIZE)
    ? FC_AFFINE_GROUP_SIZE : 64u;
#endif
constant constexpr uint kPrefillRmsMaxSimdGroups = 32;
#ifndef MFERENCE_ROUTER_BF16
#define MFERENCE_ROUTER_BF16
constant bool FC_ROUTER_BF16 [[function_constant(109)]];
constant bool kRouterBF16 = is_function_constant_defined(FC_ROUTER_BF16) && FC_ROUTER_BF16;
#endif
constant constexpr uint kPrefillPostMaxD = 4096;
// Flash-Next routes over 512 experts. The bound only sizes the threadgroup
// score staging array (2 KiB at 512) and clamps `num_experts`; for the shipped
// families (<= 256) the selection reads the same values in the same order, so
// widening it is a capacity change, not a numerical one.
constant constexpr uint kPrefillRouterMaxExperts = 512;
constant constexpr uint kPrefillRouterMaxTopK = 64;
constant constexpr uint kPrefillAttentionMaxSimdGroups = 16;
constant constexpr uint kPrefillMaxTileExperts = 16;
constant constexpr float kPrefillGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kPrefillGeluCubicCoeff = 0.044715f;
constant uint FC_PREFILL_KV_RING_CAP [[function_constant(76)]];
// Unset/false = gelu_pytorch_tanh (Gemma), true = silu (Qwen 3.6 SwiGLU).
constant bool FC_PREFILL_ACT_SILU [[function_constant(77)]];

static inline float prefill_gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kPrefillGeluSqrt2OverPi * (x + kPrefillGeluCubicCoeff * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

static inline float prefill_hidden_activation(float x) {
    if (is_function_constant_defined(FC_PREFILL_ACT_SILU) &&
        FC_PREFILL_ACT_SILU) {
        return x / (1.0f + exp(-x));
    }
    return prefill_gelu_pytorch_tanh(x);
}

// Qwen shared-expert scalar gate for an entire prompt block. One SIMD owns a
// token row and reproduces the scalar INT8 GEMV's accumulation order. The
// accumulated gate is rounded through FP16 before sigmoid, matching the old
// GEMV-output buffer boundary exactly, then the same SIMD gates the row in
// place. Eight rows share a 256-thread threadgroup.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void prefill_qwen_shared_scalar_gate(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        rows   [[buffer(5)]],
    constant uint&        D      [[buffer(6)]],
    constant uint&        xStride [[buffer(7)]],
    constant uint&        yStride [[buffer(8)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * 8u + sg_idx;
    if (row >= rows) return;

    device const half* xRow = x + row * xStride;
    device half* yRow = y + row * yStride;
    const uint groups = D / kPrefillGroupSize;
    float acc = 0.0f;
    for (uint group = 0; group < groups; ++group) {
        const uint i0 = group * kPrefillGroupSize + lane * 2u;
        const uint i1 = i0 + 1u;
        const float x0 = float(xRow[i0]);
        const float x1 = float(xRow[i1]);
        const float s = float(scales[group]);
        const float b = float(biases[group]);
        acc = fma(s, float(uint(W[i0])) * x0 + float(uint(W[i1])) * x1, acc);
        acc = fma(b, x0 + x1, acc);
    }
    acc = simd_sum(acc);
    const float roundedGate = float(half(simd_broadcast(acc, 0u)));
    const float multiplier = 1.0f / (1.0f + exp(-roundedGate));
    for (uint col = lane; col < D; col += 32u) {
        yRow[col] = half(float(yRow[col]) * multiplier);
    }
}

kernel void prefill_embed_lookup_int4_block(
    device const uint8_t* table     [[buffer(0)]],
    device const bfloat*  scales    [[buffer(1)]],
    device const bfloat*  biases    [[buffer(2)]],
    device const uint*    tokens    [[buffer(3)]],
    device half*          out       [[buffer(4)]],
    constant uint&        T         [[buffer(5)]],
    constant uint&        D         [[buffer(6)]],
    constant float&       out_scale [[buffer(7)]],
    uint2                 gid       [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    const uint token = tokens[t];
    const uint groups_per_row = D / kAffineGroupSize;
    device const uint8_t* row_q = table  + token * (D / 2u);
    device const bfloat*  row_s = scales + token * groups_per_row;
    device const bfloat*  row_b = biases + token * groups_per_row;

    const uint8_t byte = row_q[d >> 1];
    const uint q = (d & 1u) == 0u ? uint(byte & 0x0Fu) : uint(byte >> 4);
    const float s = float(row_s[d / kAffineGroupSize]);
    const float b = float(row_b[d / kAffineGroupSize]);
    out[t * D + d] = gemma_scaled_embedding(float(q) * s + b, out_scale);
}

static inline float prefill_rms_block_inv(
    device const half* x,
    uint D,
    float eps,
    uint lid,
    uint lsize,
    uint simd_lane_id,
    uint simd_group_id,
    uint simdgroups,
    threadgroup float* partial
) {
    if (kGemmaSourceFP16) return gemma_source_norm_inv(x, D, eps, simd_lane_id,
        simd_group_id, simdgroups, partial);
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(x[i]);
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
            partial[0] = rsqrt(v / float(D) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_bf16w_block(
    device const half*   x       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device half*         out     [[buffer(2)]],
    constant uint&       T       [[buffer(3)]],
    constant uint&       D       [[buffer(4)]],
    constant float&      eps     [[buffer(5)]],
    uint                 row     [[threadgroup_position_in_grid]],
    uint                 lid     [[thread_position_in_threadgroup]],
    uint                 lsize   [[threads_per_threadgroup]],
    uint                 lane    [[thread_index_in_simdgroup]],
    uint                 sg      [[simdgroup_index_in_threadgroup]],
    uint                 sgs     [[simdgroups_per_threadgroup]]
) {
    if (row >= T) return;
    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xr = x + row * D;
    device half* yr = out + row * D;
    const float inv = prefill_rms_block_inv(xr, D, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < D; i += lsize) {
        yr[i] = gemma_weighted_norm(float(xr[i]), inv, float(weight[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_bf16w_perhead_block(
    device const half*   x                   [[buffer(0)]],
    device const bfloat* weight              [[buffer(1)]],
    device half*         out                 [[buffer(2)]],
    constant uint&       T                   [[buffer(3)]],
    constant uint&       head_dim            [[buffer(4)]],
    constant uint&       num_heads           [[buffer(5)]],
    constant uint&       token_stride_elems  [[buffer(6)]],
    constant float&      eps                 [[buffer(7)]],
    uint3                tg                  [[threadgroup_position_in_grid]],
    uint3                lid3                [[thread_position_in_threadgroup]],
    uint3                lsize3              [[threads_per_threadgroup]],
    uint                 lane                [[thread_index_in_simdgroup]],
    uint                 sg                  [[simdgroup_index_in_threadgroup]],
    uint                 sgs                 [[simdgroups_per_threadgroup]]
) {
    const uint h = tg.x;
    const uint t = tg.y;
    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    if (t >= T || h >= num_heads) return;

    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xh = x + t * token_stride_elems + h * head_dim;
    device half* yh = out + t * token_stride_elems + h * head_dim;
    const float inv = prefill_rms_block_inv(xh, head_dim, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < head_dim; i += lsize) {
        yh[i] = gemma_weighted_norm(float(xh[i]), inv, float(weight[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_no_scale_perhead_block(
    device const half*   x                   [[buffer(0)]],
    device half*         out                 [[buffer(1)]],
    constant uint&       T                   [[buffer(2)]],
    constant uint&       head_dim            [[buffer(3)]],
    constant uint&       num_heads           [[buffer(4)]],
    constant uint&       token_stride_elems  [[buffer(5)]],
    constant float&      eps                 [[buffer(6)]],
    uint3                tg                  [[threadgroup_position_in_grid]],
    uint3                lid3                [[thread_position_in_threadgroup]],
    uint3                lsize3              [[threads_per_threadgroup]],
    uint                 lane                [[thread_index_in_simdgroup]],
    uint                 sg                  [[simdgroup_index_in_threadgroup]],
    uint                 sgs                 [[simdgroups_per_threadgroup]]
) {
    const uint h = tg.x;
    const uint t = tg.y;
    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    if (t >= T || h >= num_heads) return;

    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xh = x + t * token_stride_elems + h * head_dim;
    device half* yh = out + t * token_stride_elems + h * head_dim;
    const float inv = prefill_rms_block_inv(xh, head_dim, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < head_dim; i += lsize) {
        yh[i] = half(float(xh[i]) * inv);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_post_attn_setup_block(
    device       half*   hidden                [[buffer(0)]],
    device const half*   attn                  [[buffer(1)]],
    device       half*   dense_x               [[buffer(2)]],
    device       half*   routed_x              [[buffer(3)]],
    device       half*   router_x              [[buffer(4)]],
    device const bfloat* w_post_attn           [[buffer(5)]],
    device const bfloat* w_pre_ffn             [[buffer(6)]],
    device const bfloat* w_pre_ffn2            [[buffer(7)]],
    constant uint&       T                     [[buffer(8)]],
    constant uint&       D                     [[buffer(9)]],
    constant uint&       hidden_stride_elems   [[buffer(10)]],
    constant uint&       attn_stride_elems     [[buffer(11)]],
    constant uint&       dense_stride_elems    [[buffer(12)]],
    constant uint&       routed_stride_elems   [[buffer(13)]],
    constant uint&       router_stride_elems   [[buffer(14)]],
    constant float&      rms_eps               [[buffer(15)]],
    uint                 row                   [[threadgroup_position_in_grid]],
    uint                 lid                   [[thread_position_in_threadgroup]],
    uint                 lsize                 [[threads_per_threadgroup]],
    uint                 lane                  [[thread_index_in_simdgroup]],
    uint                 sg                    [[simdgroup_index_in_threadgroup]],
    uint                 sgs                   [[simdgroups_per_threadgroup]]
) {
    if (row >= T || D > kPrefillPostMaxD) return;

    threadgroup half attn_norm_tg[kPrefillPostMaxD];
    threadgroup half hidden_tg[kPrefillPostMaxD];
    threadgroup float partial[kPrefillRmsMaxSimdGroups];

    device half* hidden_row = hidden + row * hidden_stride_elems;
    device const half* attn_row = attn + row * attn_stride_elems;
    device half* dense_row = dense_x + row * dense_stride_elems;
    device half* routed_row = routed_x + row * routed_stride_elems;
    device half* router_row = router_x + row * router_stride_elems;

    const float attn_inv = prefill_rms_block_inv(attn_row, D, rms_eps,
                                                 lid, lsize, lane, sg, sgs,
                                                 partial);
    for (uint i = lid; i < D; i += lsize) {
        attn_norm_tg[i] = gemma_weighted_norm(float(attn_row[i]), attn_inv, float(w_post_attn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        half h = half(float(hidden_row[i]) + float(attn_norm_tg[i]));
        hidden_tg[i] = h;
        hidden_row[i] = h;
        float hf = float(h);
        acc = fma(hf, hf, acc);
    }
    if (kGemmaSourceFP16) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemma_source_norm_inv(hidden_tg, D, rms_eps, lane, sg, sgs, partial);
    } else {
        acc = simd_sum(acc);
        if (lane == 0) {
            partial[sg] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            float sum = (lane < sgs) ? partial[lane] : 0.0f;
            sum = simd_sum(sum);
            if (lane == 0) {
                partial[0] = rsqrt(sum / float(D) + rms_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float hidden_inv = partial[0];
    for (uint i = lid; i < D; i += lsize) {
        const float h = float(hidden_tg[i]) * hidden_inv;
        dense_row[i] = gemma_weighted_norm(float(hidden_tg[i]), hidden_inv, float(w_pre_ffn[i]));
        routed_row[i] = gemma_weighted_norm(float(hidden_tg[i]), hidden_inv, float(w_pre_ffn2[i]));
        router_row[i] = half(h);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_layer_tail_block(
    device const half*   h2                    [[buffer(0)]],
    device const half*   h1                    [[buffer(1)]],
    device       half*   hidden                [[buffer(2)]],
    device const bfloat* w_postffn2            [[buffer(3)]],
    device const bfloat* w_postffn             [[buffer(4)]],
    constant uint&       T                     [[buffer(5)]],
    constant uint&       D                     [[buffer(6)]],
    constant uint&       h2_stride_elems       [[buffer(7)]],
    constant uint&       h1_stride_elems       [[buffer(8)]],
    constant uint&       hidden_stride_elems   [[buffer(9)]],
    constant float&      rms_eps               [[buffer(10)]],
    constant float&      layer_scalar          [[buffer(11)]],
    uint                 row                   [[threadgroup_position_in_grid]],
    uint                 lid                   [[thread_position_in_threadgroup]],
    uint                 lsize                 [[threads_per_threadgroup]],
    uint                 lane                  [[thread_index_in_simdgroup]],
    uint                 sg                    [[simdgroup_index_in_threadgroup]],
    uint                 sgs                   [[simdgroups_per_threadgroup]]
) {
    if (row >= T || D > kPrefillPostMaxD) return;

    threadgroup half tmp_tg[kPrefillPostMaxD];
    threadgroup half h12_tg[kPrefillPostMaxD];
    threadgroup float partial[kPrefillRmsMaxSimdGroups];

    device const half* h2_row = h2 + row * h2_stride_elems;
    device const half* h1_row = h1 + row * h1_stride_elems;
    device half* hidden_row = hidden + row * hidden_stride_elems;

    const float inv_h2 = prefill_rms_block_inv(h2_row, D, rms_eps,
                                               lid, lsize, lane, sg, sgs,
                                               partial);
    for (uint i = lid; i < D; i += lsize) {
        tmp_tg[i] = gemma_weighted_norm(float(h2_row[i]), inv_h2, float(w_postffn2[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < D; i += lsize) {
        h12_tg[i] = h1_row[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (kGemmaSourceFP16) {
        gemma_source_norm_inv(h12_tg, D, rms_eps, lane, sg, sgs, partial);
    } else {
        float acc = 0.0f;
        for (uint i = lid; i < D; i += lsize) {
            float v = float(h12_tg[i]);
            acc = fma(v, v, acc);
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            partial[sg] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            float sum = (lane < sgs) ? partial[lane] : 0.0f;
            sum = simd_sum(sum);
            if (lane == 0) {
                partial[0] = rsqrt(sum / float(D) + rms_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_h12 = partial[0];
    for (uint i = lid; i < D; i += lsize) {
        tmp_tg[i] = gemma_weighted_norm(float(h12_tg[i]), inv_h12, float(w_postffn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < D; i += lsize) {
        hidden_row[i] = hidden_row[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    const half h_scale = half(layer_scalar);
    for (uint i = lid; i < D; i += lsize) {
        hidden_row[i] = hidden_row[i] * h_scale;
    }
}

struct PrefillTokenExpertPairMSL {
    uint token;
    uint expert;
    uint rank;
    uint weight_bits_and_reserved;
};

struct PrefillDeviceGroupMSL { uint expert; uint pair_start; uint pair_count; };

kernel void prefill_route_count_by_expert(
    device const uint* ids [[buffer(0)]], device uint* counts [[buffer(1)]],
    constant uint& pairs [[buffer(2)]], constant uint& experts [[buffer(3)]],
    uint e [[thread_position_in_grid]]) {
    if (e >= experts) return;
    uint count = 0;
    for (uint p = 0; p < pairs; ++p) count += min(ids[p], experts - 1u) == e;
    counts[e] = count;
}

kernel void prefill_route_group_prefix(
    device const uint* counts [[buffer(0)]],
    device PrefillDeviceGroupMSL* groups [[buffer(1)]],
    device uint* dispatch [[buffer(2)]], constant uint& experts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]]) {
    uint offset = 0, maximum = 0;
    for (uint e = 0; e < experts; ++e) {
        groups[e] = {e, offset, counts[e]};
        offset += counts[e];
        maximum = max(maximum, counts[e]);
    }
    // Two MTLDispatchThreadgroupsIndirectArguments (gate/up, then down).
    dispatch[0] = (F + 31u) / 32u;
    dispatch[1] = (maximum + 63u) / 64u;
    dispatch[2] = experts;
    dispatch[3] = (D + 31u) / 32u;
    dispatch[4] = dispatch[1];
    dispatch[5] = experts;
}

kernel void prefill_route_scatter_by_expert(
    device const uint* ids [[buffer(0)]], device const half* weights [[buffer(1)]],
    device const PrefillDeviceGroupMSL* groups [[buffer(2)]],
    device PrefillTokenExpertPairMSL* sorted [[buffer(3)]],
    constant uint& pairs [[buffer(4)]], constant uint& experts [[buffer(5)]],
    constant uint& top_k [[buffer(6)]], uint e [[thread_position_in_grid]]) {
    if (e >= experts) return;
    uint destination = groups[e].pair_start;
    for (uint p = 0; p < pairs; ++p) {
        if (min(ids[p], experts - 1u) == e) {
            sorted[destination++] = {p / top_k, e, p % top_k,
                                     uint(as_type<ushort>(weights[p]))};
        }
    }
}

struct PrefillStreamedRoutedBlobsMSL {
    device const uint8_t* blob[kPrefillMaxTileExperts];
};

struct PrefillGroupedRoutedMoEStreamedParamsMSL {
    uint pair_start;
    uint pair_count;
    uint D;
    uint F;
    uint top_k;
    uint hidden_stride_elements;
    uint live_expert_count;
    uint local_expert_0;
    uint local_expert_1;
    uint local_expert_2;
    uint local_expert_3;
    uint local_expert_4;
    uint local_expert_5;
    uint local_expert_6;
    uint local_expert_7;
    uint local_expert_8;
    uint local_expert_9;
    uint local_expert_10;
    uint local_expert_11;
    uint local_expert_12;
    uint local_expert_13;
    uint local_expert_14;
    uint local_expert_15;
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

static inline uint prefill_streamed_local_expert_id(
    constant PrefillGroupedRoutedMoEStreamedParamsMSL& p,
    uint slot
) {
    switch (slot) {
        case 0: return p.local_expert_0;
        case 1: return p.local_expert_1;
        case 2: return p.local_expert_2;
        case 3: return p.local_expert_3;
        case 4: return p.local_expert_4;
        case 5: return p.local_expert_5;
        case 6: return p.local_expert_6;
        case 7: return p.local_expert_7;
        case 8: return p.local_expert_8;
        case 9: return p.local_expert_9;
        case 10: return p.local_expert_10;
        case 11: return p.local_expert_11;
        case 12: return p.local_expert_12;
        case 13: return p.local_expert_13;
        case 14: return p.local_expert_14;
        default: return p.local_expert_15;
    }
}

static inline float prefill_moe_int4_gemv_row_dev(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kAffineGroupSize;
    const uint row_bytes = N / 2u;
    device const uint8_t* W_row = W + row * row_bytes;
    device const bfloat* s_row = S + row * groups;
    device const bfloat* b_row = B + row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        device const uint8_t* Wg = W_row + g * (kAffineGroupSize / 2u);
        device const half* xg = x + g * kAffineGroupSize;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint k = 0; k < kAffineGroupSize / 2u; ++k) {
            const uint8_t packed = Wg[k];
            const float x0 = float(xg[2u * k]);
            const float x1 = float(xg[2u * k + 1u]);
            dot_qx = fma(float(uint(packed & 0x0Fu)), x0, dot_qx);
            dot_qx = fma(float(uint(packed >> 4)), x1, dot_qx);
            sum_x += x0 + x1;
        }
        if (kGemmaSourceFP16) {
            sum_x = 0.0f;
            for (uint k = 0; k < kAffineGroupSize; k += 4u) {
                sum_x += gemma_source_quad_sum(half4(xg[k], xg[k + 1u], xg[k + 2u], xg[k + 3u]));
            }
        }
        acc = fma(scale, dot_qx, acc);
        acc = fma(bias, sum_x, acc);
    }
    return acc;
}

static inline float prefill_moe_int4_gemv_row_tg(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    threadgroup const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kAffineGroupSize;
    const uint row_bytes = N / 2u;
    device const uint8_t* W_row = W + row * row_bytes;
    device const bfloat* s_row = S + row * groups;
    device const bfloat* b_row = B + row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        device const uint8_t* Wg = W_row + g * (kAffineGroupSize / 2u);
        threadgroup const half* xg = x + g * kAffineGroupSize;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint k = 0; k < kAffineGroupSize / 2u; ++k) {
            const uint8_t packed = Wg[k];
            const float x0 = float(xg[2u * k]);
            const float x1 = float(xg[2u * k + 1u]);
            dot_qx = fma(float(uint(packed & 0x0Fu)), x0, dot_qx);
            dot_qx = fma(float(uint(packed >> 4)), x1, dot_qx);
            sum_x += x0 + x1;
        }
        if (kGemmaSourceFP16) {
            sum_x = 0.0f;
            for (uint k = 0; k < kAffineGroupSize; k += 4u) {
                sum_x += gemma_source_quad_sum(half4(xg[k], xg[k + 1u], xg[k + 2u], xg[k + 3u]));
            }
        }
        acc = fma(scale, dot_qx, acc);
        acc = fma(bias, sum_x, acc);
    }
    return acc;
}

kernel void prefill_router_gemma4_block(
    device const uint8_t* W                [[buffer(0)]],
    device const bfloat*  scales           [[buffer(1)]],
    device const bfloat*  biases           [[buffer(2)]],
    device const half*    hidden           [[buffer(3)]],
    device const bfloat*  effective_scale  [[buffer(4)]],
    device const bfloat*  per_expert_scale [[buffer(5)]],
    device uint*          out_indices      [[buffer(6)]],
    device half*          out_weights      [[buffer(7)]],
    constant uint&        T                [[buffer(8)]],
    constant uint&        num_experts      [[buffer(9)]],
    constant uint&        D                [[buffer(10)]],
    constant uint&        top_k            [[buffer(11)]],
    constant uint&        hidden_stride    [[buffer(12)]],
    uint                  row              [[threadgroup_position_in_grid]],
    uint                  tid              [[thread_position_in_threadgroup]],
    uint                  tg_size          [[threads_per_threadgroup]]
) {
    if (row >= T) return;
    threadgroup float scores[kPrefillRouterMaxExperts];
    const uint NE = min(num_experts, kPrefillRouterMaxExperts);
    const uint KK = min(top_k, kPrefillRouterMaxTopK);
    device const half* row_hidden = hidden + row * hidden_stride;

    if (kRouterBF16 && kGemmaSourceFP16) {
        const uint lane = tid % 32u, group = tid / 32u;
        for (uint e = group; e < NE; e += tg_size / 32u) {
            const float score = gemma_source_router_dot(
                reinterpret_cast<device const bfloat*>(W) + e * D, row_hidden,
                reinterpret_cast<device const half*>(effective_scale), D, lane);
            if (lane == 0) scores[e] = float(half(score));
        }
    } else {
        for (uint e = tid; e < NE; e += tg_size) {
            float acc = 0.0f;
            if (kRouterBF16) {
                device const bfloat* W_row = reinterpret_cast<device const bfloat*>(W) + e * D;
                for (uint col = 0; col < D; ++col) {
                    const float x = kGemmaSourceFP16
                        ? float(half(row_hidden[col] * reinterpret_cast<device const half*>(effective_scale)[col]))
                        : float(row_hidden[col]) * float(effective_scale[col]);
                    const float w = kGemmaSourceFP16 ? float(half(W_row[col])) : float(W_row[col]);
                    acc = fma(w, x, acc);
                }
            } else {
                const uint n_groups = D / kPrefillGroupSize;
                device const uint8_t* W_row = W + e * D;
                device const bfloat* s_row = scales + e * n_groups;
                device const bfloat* b_row = biases + e * n_groups;
                for (uint g = 0; g < n_groups; ++g) {
                    float s = float(s_row[g]);
                    float b = float(b_row[g]);
                    device const uint8_t* Wg = W_row + g * kPrefillGroupSize;
                    device const half* xg = row_hidden + g * kPrefillGroupSize;
                    device const bfloat* eg = effective_scale + g * kPrefillGroupSize;
                    float dot_qx = 0.0f;
                    float sum_x = 0.0f;
                    for (uint k = 0; k < kPrefillGroupSize; ++k) {
                        float q = float(uint(Wg[k]));
                        float xv = float(xg[k]) * float(eg[k]);
                        dot_qx = fma(q, xv, dot_qx);
                        sum_x += xv;
                    }
                    acc = fma(s, dot_qx, acc);
                    acc = fma(b, sum_x, acc);
                }
            }
            scores[e] = kGemmaSourceFP16 ? float(half(acc)) : acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint top_idx[kPrefillRouterMaxTopK];
        float top_score[kPrefillRouterMaxTopK];
        for (uint i = 0; i < kPrefillRouterMaxTopK; ++i) {
            top_idx[i] = 0u;
            top_score[i] = -INFINITY;
        }

        for (uint e = 0; e < NE; ++e) {
            float s = scores[e];
            if (KK > 0 && (s < top_score[KK - 1] ||
                (!kGemmaSourceFP16 && s == top_score[KK - 1]))) continue;
            uint pos = KK;
            for (uint i = 0; i < KK; ++i) {
                if (s > top_score[i] || (s == top_score[i] && (kGemmaSourceFP16 ? e > top_idx[i] : e < top_idx[i]))) {
                    pos = i;
                    break;
                }
            }
            if (pos >= KK) continue;
            for (uint i = KK - 1; i > pos; --i) {
                top_idx[i] = top_idx[i - 1];
                top_score[i] = top_score[i - 1];
            }
            top_idx[pos] = e;
            top_score[pos] = s;
        }

        half source_probabilities[8];
        if (kGemmaSourceFP16 && KK == 8u) {
            gemma_source_softmax8(top_score, source_probabilities);
        }
        float max_s = top_score[0];
        float sum_exp = 0.0f;
        float exps[kPrefillRouterMaxTopK];
        for (uint i = 0; i < KK; ++i) {
            float e = fast::exp(top_score[i] - max_s);
            exps[i] = e;
            sum_exp += e;
        }
        for (uint i = 0; i < KK; ++i) {
            const uint expert_idx = top_idx[i];
            const float w = exps[i] / sum_exp;
            const float gain = float(per_expert_scale[expert_idx]);
            out_indices[row * top_k + i] = expert_idx;
            out_weights[row * top_k + i] = (kGemmaSourceFP16 && KK == 8u)
                ? source_probabilities[i] * half(gain)
                : (kGemmaSourceFP16 ? half(w) * half(gain) : half(w * gain));
        }
    }
}

kernel void prefill_moe_reduce_token_major(
    device const half* route_partials [[buffer(0)]],
    device const half* route_weights  [[buffer(1)]],
    device half*       h2             [[buffer(2)]],
    constant uint&     T              [[buffer(3)]],
    constant uint&     top_k          [[buffer(4)]],
    constant uint&     D              [[buffer(5)]],
    uint2              gid            [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    if (kGemmaSourceFP16) {
        volatile half acc = 0.0h;
        for (uint rank = top_k; rank > 0u; --rank) {
            const uint r = rank - 1u;
            const uint partial_index = (t * top_k + r) * D + d;
            const half product = half(gemma_source_weighted_expert(
                float(route_partials[partial_index]), route_weights[t * top_k + r]));
            acc = acc + product;
        }
        h2[t * D + d] = acc;
        return;
    }

    float acc = 0.0f;
    for (uint r = 0; r < top_k; ++r) {
        const uint partial_index = (t * top_k + r) * D + d;
        if (kGemmaSourceFP16) {
            acc += gemma_source_weighted_expert(float(route_partials[partial_index]), route_weights[t * top_k + r]);
        } else {
            acc = fma(float(route_weights[t * top_k + r]),
                      float(route_partials[partial_index]), acc);
        }
    }
    h2[t * D + d] = half(acc);
}

// ----------------------------------------------------------------------------
// Inkling batched expert application.
//
// The v1 Inkling prefill applied one expert to one token per dispatch: three
// GEMVs plus an accumulate, repeated for every (token, expert) pair. These two
// kernels invert the loop: one dispatch pair per EXPERT covers every chunk
// token routed to it, so an expert's weight rows are read once per expert
// instead of once per pair and the grid's token dimension supplies the reuse
// through the cache.
//
// Measured effect, 2 785-token prompt on an M3 Ultra: prefill 228.9 s -> 205.7 s
// (1.11x). Do not read more into it than that. `MFERENCE_PREFILL_BREAKDOWN=1`
// shows why the ceiling is low — the routed-expert loop is only ~15 % of
// prefill (8 % streaming, 6.6 % this compute); the other ~85 % is the
// per-token attention / norm / short-conv replay elsewhere in
// `prefillInklingChunk`, which is where the remaining time actually is.
//
// The accumulate needs no atomics: within a single expert a token appears at
// most once, so no two threads of one dispatch touch the same `acc` element.
// `acc` is FP32 because a raw Inkling expert row reaches ~6e4 before its
// routing weight is applied and clips an FP16 store (docs/INKLING_SMALL.md,
// "FFN output range").
//
// The same pair applies a SHARED expert too: pass every chunk token with the
// per-token shared gamma as the weight.
// ----------------------------------------------------------------------------
struct InklingPrefillExpertParamsMSL {
    uint d;                 // hidden size
    uint f;                 // expert intermediate width
    uint pair_start;        // first pair index for this expert
    uint pair_count;        // tokens routed to this expert
    uint hidden_stride;     // elements between token rows of `hidden`
    uint gate_W_off, gate_s_off, gate_b_off;
    uint up_W_off,   up_s_off,   up_b_off;
    uint down_W_off, down_s_off, down_b_off;
};

kernel void inkling_prefill_expert_act(
    device const half*                          hidden      [[buffer(0)]],
    device const uint*                          pair_tokens [[buffer(1)]],
    device const uint8_t*                       expert      [[buffer(2)]],
    device half*                                act         [[buffer(3)]],
    constant InklingPrefillExpertParamsMSL&     p           [[buffer(4)]],
    uint2                                       gid         [[thread_position_in_grid]]
) {
    const uint f = gid.x;
    const uint t = gid.y;
    if (f >= p.f || t >= p.pair_count) return;

    device const half* x = hidden
        + uint(pair_tokens[p.pair_start + t]) * p.hidden_stride;
    const float gate = prefill_moe_int4_gemv_row_dev(
        expert + p.gate_W_off,
        reinterpret_cast<device const bfloat*>(expert + p.gate_s_off),
        reinterpret_cast<device const bfloat*>(expert + p.gate_b_off),
        x, f, p.d);
    const float up = prefill_moe_int4_gemv_row_dev(
        expert + p.up_W_off,
        reinterpret_cast<device const bfloat*>(expert + p.up_s_off),
        reinterpret_cast<device const bfloat*>(expert + p.up_b_off),
        x, f, p.d);
    // Round gate and up to FP16 *before* combining. The per-token path this
    // replaces stored both GEMV results to FP16 scratch and `silu_mul_fp16`
    // read them back, so keeping the same two roundings keeps prefill's
    // activations consistent with the decode path rather than slightly more
    // accurate than it.
    const float g = float(half(gate));
    const float u = float(half(up));
    act[t * p.f + f] = half(prefill_hidden_activation(g) * u);
}

kernel void inkling_prefill_expert_down_accum(
    device const half*                          act          [[buffer(0)]],
    device const uint*                          pair_tokens  [[buffer(1)]],
    device const float*                         pair_weights [[buffer(2)]],
    device const uint8_t*                       expert       [[buffer(3)]],
    device float*                               acc          [[buffer(4)]],
    constant InklingPrefillExpertParamsMSL&     p            [[buffer(5)]],
    uint2                                       gid          [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (d >= p.d || t >= p.pair_count) return;

    const float value = prefill_moe_int4_gemv_row_dev(
        expert + p.down_W_off,
        reinterpret_cast<device const bfloat*>(expert + p.down_s_off),
        reinterpret_cast<device const bfloat*>(expert + p.down_b_off),
        act + t * p.f, d, p.f);
    const uint token = pair_tokens[p.pair_start + t];
    const uint index = token * p.d + d;
    acc[index] = fma(pair_weights[p.pair_start + t], value, acc[index]);
}

#ifndef MFERENCE_GEMMA_SOURCE_PROJECTION
#define MFERENCE_GEMMA_SOURCE_PROJECTION
// QAT's source GEMV rounds the input sum in half quads, accumulates four
// products at a time, and adds each complete affine sub-result. Callers of
// this path compile this module with safe math: fast-math compilation does
// not preserve the source's FP16 rounding boundaries.
static inline float gemma_source_projection_row(
    device const uint8_t* weights,
    device const bfloat* scales,
    device const bfloat* biases,
    device const half* x,
    uint width, uint group_size, bool fast_shape, uint lane
) {
    device const ushort* packed_weights = (device const ushort*)weights;
    const uint values = fast_shape ? 16u : 8u;
    float result = 0.0f;
    for (uint base = lane * values; base < width; base += 32u * values) {
        float sum = 0.0f, dot = 0.0f;
        for (uint i = 0; i < values; i += 4u) {
            const uint k = base + i;
            const ushort packed = packed_weights[k / 4u];
            sum += x[k] + x[k + 1u] + x[k + 2u] + x[k + 3u];
            dot += float(x[k]) * (packed & 15u)
                + (float(x[k + 1u]) / 16.0f) * (packed & 240u)
                + (float(x[k + 2u]) / 256.0f) * (packed & 3840u)
                + (float(x[k + 3u]) / 4096.0f) * (packed & 61440u);
        }
        const uint group = base / group_size;
        result += float(half(scales[group])) * dot + sum * float(half(biases[group]));
    }
    return simd_sum(result);
}
#endif

static inline float prefill_grouped_moe_projection_row(
    device const uint8_t* W, device const bfloat* S, device const bfloat* B,
    device const half* x, uint row, uint width, uint lane
) {
    if (kGemmaSourceFP16) {
        const uint groups = width / kAffineGroupSize;
        return gemma_source_projection_row(W + row * (width / 2u),
            S + row * groups, B + row * groups, x,
            width, kAffineGroupSize, width % 512u == 0u, lane);
    }
    return prefill_moe_int4_gemv_row_dev(W, S, B, x, row, width);
}

kernel void prefill_grouped_routed_moe_batched_phase1(
    device const half*                                   hidden               [[buffer(0)]],
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         gate_up_act_scratch  [[buffer(7)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    uint2                                                gid                  [[thread_position_in_grid]]
) {
    const uint f = kGemmaSourceFP16 ? gid.x / 32u : gid.x;
    const uint lane = gid.x % 32u;
    const uint pair_local = gid.y;
    if (f >= p.F || pair_local >= p.pair_count) return;

    const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + pair_local];
    uint local_slot = kPrefillMaxTileExperts;
    for (uint slot = 0; slot < p.live_expert_count; ++slot) {
        if (prefill_streamed_local_expert_id(p, slot) == pair.expert) {
            local_slot = slot;
            break;
        }
    }
    if (local_slot >= p.live_expert_count) return;

    device const uint8_t* expert = routed.blob[local_slot];
    device const half* x = hidden + pair.token * p.hidden_stride_elements;
    device const uint8_t* gate_W = expert + p.gate_W_off;
    device const bfloat* gate_s = reinterpret_cast<device const bfloat*>(expert + p.gate_s_off);
    device const bfloat* gate_b = reinterpret_cast<device const bfloat*>(expert + p.gate_b_off);
    device const uint8_t* up_W = expert + p.up_W_off;
    device const bfloat* up_s = reinterpret_cast<device const bfloat*>(expert + p.up_s_off);
    device const bfloat* up_b = reinterpret_cast<device const bfloat*>(expert + p.up_b_off);

    const float gate = prefill_grouped_moe_projection_row(gate_W, gate_s, gate_b, x, f, p.D, lane);
    const float up = prefill_grouped_moe_projection_row(up_W, up_s, up_b, x, f, p.D, lane);
    if (kGemmaSourceFP16 && lane != 0u) return;
    const uint row_elements = p.pair_count * p.F;
    const uint index = pair_local * p.F + f;
    gate_up_act_scratch[index] = half(gate);
    gate_up_act_scratch[row_elements + index] = half(up);
    gate_up_act_scratch[2u * row_elements + index] =
        (kGemmaSourceFP16 ? gemma_source_geglu(half(gate), half(up))
            : half(prefill_hidden_activation(gate) * up));
}

kernel void prefill_grouped_routed_moe_batched_down(
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         route_partials       [[buffer(5)]],
    device const half*                                   gate_up_act_scratch  [[buffer(7)]],
    device half*                                         down_scratch         [[buffer(8)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    uint2                                                gid                  [[thread_position_in_grid]]
) {
    const uint d = kGemmaSourceFP16 ? gid.x / 32u : gid.x;
    const uint lane = gid.x % 32u;
    const uint pair_local = gid.y;
    if (d >= p.D || pair_local >= p.pair_count) return;

    const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + pair_local];
    uint local_slot = kPrefillMaxTileExperts;
    for (uint slot = 0; slot < p.live_expert_count; ++slot) {
        if (prefill_streamed_local_expert_id(p, slot) == pair.expert) {
            local_slot = slot;
            break;
        }
    }
    if (local_slot >= p.live_expert_count) return;

    device const uint8_t* expert = routed.blob[local_slot];
    device const uint8_t* down_W = expert + p.down_W_off;
    device const bfloat* down_s = reinterpret_cast<device const bfloat*>(expert + p.down_s_off);
    device const bfloat* down_b = reinterpret_cast<device const bfloat*>(expert + p.down_b_off);
    device const half* act = gate_up_act_scratch + 2u * p.pair_count * p.F + pair_local * p.F;
    const half value = half(prefill_grouped_moe_projection_row(down_W, down_s, down_b, act, d, p.F, lane));
    if (kGemmaSourceFP16 && lane != 0u) return;
    down_scratch[pair_local * p.D + d] = value;
    route_partials[(pair.token * p.top_k + pair.rank) * p.D + d] = value;
}

// Same arithmetic as streamed batched experts, addressed directly in the
// immutable resident slab. All sorted pairs share one dispatch per phase.
kernel void prefill_grouped_routed_moe_resident_phase1(
    device const half* hidden [[buffer(0)]],
    device const PrefillTokenExpertPairMSL* pairs [[buffer(1)]],
    device half* act [[buffer(2)]], device half* partials [[buffer(3)]],
    device const uint8_t* slab [[buffer(4)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL& p [[buffer(5)]],
    constant uint& stride [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
    const uint f = kGemmaSourceFP16 ? gid.x / 32u : gid.x;
    const uint lane = gid.x % 32u;
    if (f >= p.F || gid.y >= p.pair_count) return;
    const PrefillTokenExpertPairMSL pair = pairs[gid.y];
    device const uint8_t* expert = slab + ulong(pair.expert) * ulong(stride);
    device const half* x = hidden + pair.token * p.hidden_stride_elements;
    const float gate = prefill_grouped_moe_projection_row(expert + p.gate_W_off,
        reinterpret_cast<device const bfloat*>(expert + p.gate_s_off),
        reinterpret_cast<device const bfloat*>(expert + p.gate_b_off), x, f, p.D, lane);
    const float up = prefill_grouped_moe_projection_row(expert + p.up_W_off,
        reinterpret_cast<device const bfloat*>(expert + p.up_s_off),
        reinterpret_cast<device const bfloat*>(expert + p.up_b_off), x, f, p.D, lane);
    if (kGemmaSourceFP16 && lane != 0u) return;
    act[gid.y * p.F + f] = (kGemmaSourceFP16 ? gemma_source_geglu(half(gate), half(up))
            : half(prefill_hidden_activation(gate) * up));
}

kernel void prefill_grouped_routed_moe_resident_down(
    device const half* hidden [[buffer(0)]],
    device const PrefillTokenExpertPairMSL* pairs [[buffer(1)]],
    device const half* act [[buffer(2)]], device half* partials [[buffer(3)]],
    device const uint8_t* slab [[buffer(4)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL& p [[buffer(5)]],
    constant uint& stride [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
    const uint d = kGemmaSourceFP16 ? gid.x / 32u : gid.x;
    const uint lane = gid.x % 32u;
    if (d >= p.D || gid.y >= p.pair_count) return;
    const PrefillTokenExpertPairMSL pair = pairs[gid.y];
    device const uint8_t* expert = slab + ulong(pair.expert) * ulong(stride);
    const half value = half(prefill_grouped_moe_projection_row(expert + p.down_W_off,
        reinterpret_cast<device const bfloat*>(expert + p.down_s_off),
        reinterpret_cast<device const bfloat*>(expert + p.down_b_off),
        act + gid.y * p.F, d, p.F, lane));
    if (kGemmaSourceFP16 && lane != 0u) return;
    partials[(pair.token * p.top_k + pair.rank) * p.D + d] = value;
}

kernel void prefill_dequant_int4_qmm_f16_block(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    X      [[buffer(3)]],
    device half*          Y      [[buffer(4)]],
    constant uint&        T      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        K      [[buffer(7)]],
    uint2                 tid    [[thread_position_in_threadgroup]],
    uint2                 tgid   [[threadgroup_position_in_grid]],
    uint                  simd   [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    if (kGemmaSourceFP16) {
        const uint n = tgid.x * 8u + simd;
        if (n >= N) return;
        const uint groups = K / kAffineGroupSize;
        for (uint row = 0; row < 8u; ++row) {
            const uint t = tgid.y * 8u + row;
            if (t >= T) break;
            const float result = gemma_source_projection_row(
                W + n * (K / 2u), scales + n * groups, biases + n * groups,
                X + t * K, K, kAffineGroupSize, N % 8u == 0u && K % 512u == 0u, lane);
            if (lane == 0u) Y[t * N + n] = half(result);
        }
        return;
    }
    const uint n = tgid.x * 8u + tid.x;
    const uint t = tgid.y * 8u + tid.y;
    if (t >= T || n >= N) return;

    const uint groups = K / kAffineGroupSize;
    const uint row_bytes = K / 2u;
    device const uint8_t* w_row = W + n * row_bytes;
    device const bfloat* s_row = scales + n * groups;
    device const bfloat* b_row = biases + n * groups;
    device const half* x_row = X + t * K;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        const uint group_base = g * kAffineGroupSize;
        for (uint kk = 0; kk < kAffineGroupSize; ++kk) {
            const uint k = group_base + kk;
            const uint8_t packed = w_row[k >> 1];
            const uint q = (k & 1u) == 0u ? uint(packed & 0x0Fu) : uint(packed >> 4);
            const float w = fma(float(q), scale, bias);
            acc = fma(w, float(x_row[k]), acc);
        }
    }
    Y[t * N + n] = half(acc);
}

static inline void prefill_rope_apply_neox_pair(
    device half* head_ptr,
    uint i,
    uint half_dim,
    uint freq_divisor,
    float position,
    float theta_base,
    bool proportional = false
) {
    const float exponent = -float(2u * i) / float(freq_divisor);
    const float freq = kGemmaSourceFP16 && proportional
        ? precise::divide(1.0f, precise::pow(theta_base, -exponent))
        : pow(theta_base, exponent);
    const float angle = position * freq;
    const float c = kGemmaSourceFP16 ? fast::cos(angle) : cos(angle);
    const float s = kGemmaSourceFP16 ? fast::sin(angle) : sin(angle);

    const uint i0 = i;
    const uint i1 = half_dim + i;
    const float x0 = float(head_ptr[i0]);
    const float x1 = float(head_ptr[i1]);
    head_ptr[i0] = half(x0 * c - x1 * s);
    head_ptr[i1] = half(x0 * s + x1 * c);
}

kernel void prefill_rope_default_neox_block(
    device half*   data                [[buffer(0)]],
    constant uint& start_position      [[buffer(1)]],
    constant uint& head_dim            [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& token_stride_elems  [[buffer(4)]],
    constant float& theta_base         [[buffer(5)]],
    uint3          gid                 [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    const uint half_dim = head_dim / 2u;
    if (i >= half_dim) return;
    if (h >= num_heads) return;

    device half* head_ptr = data + t * token_stride_elems + h * head_dim;
    prefill_rope_apply_neox_pair(head_ptr, i, half_dim, head_dim,
                                 float(start_position + t), theta_base);
}

kernel void prefill_rope_proportional_neox_block(
    device half*   data                [[buffer(0)]],
    constant uint& start_position      [[buffer(1)]],
    constant uint& head_dim            [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& token_stride_elems  [[buffer(4)]],
    constant float& theta_base         [[buffer(5)]],
    constant uint& rotated_pairs       [[buffer(6)]],
    uint3          gid                 [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    if (i >= rotated_pairs) return;
    if (h >= num_heads) return;

    const uint half_dim = head_dim / 2u;
    device half* head_ptr = data + t * token_stride_elems + h * head_dim;
    prefill_rope_apply_neox_pair(head_ptr, i, half_dim, head_dim,
                                 float(start_position + t), theta_base, true);
}

// Qwen-style partial RoPE: rotation confined to the first `rotary_dim`
// elements per head, pairing (i, rotary_dim/2 + i), frequency divisor =
// rotary_dim; the remaining elements pass through untouched.
kernel void prefill_rope_neox_subdim_block(
    device half*   data                [[buffer(0)]],
    constant uint& start_position      [[buffer(1)]],
    constant uint& head_dim            [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& token_stride_elems  [[buffer(4)]],
    constant float& theta_base         [[buffer(5)]],
    constant uint& rotary_dim          [[buffer(6)]],
    uint3          gid                 [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    const uint half_rotary = rotary_dim / 2u;
    if (i >= half_rotary) return;
    if (h >= num_heads) return;

    device half* head_ptr = data + t * token_stride_elems + h * head_dim;
    prefill_rope_apply_neox_pair(head_ptr, i, half_rotary, rotary_dim,
                                 float(start_position + t), theta_base);
}

struct PrefillAttentionParams {
    uint startPosition;
    uint queryCount;
    uint headDim;
    uint numQHeads;
    uint numKVHeads;
    uint kvValidCount;
    uint slidingWindow;
    uint kvTokenStrideElements;
    uint qTokenStrideElements;
    uint oTokenStrideElements;
    float scale;
};

static inline uint prefill_kv_slot(uint logical) {
    return (is_function_constant_defined(FC_PREFILL_KV_RING_CAP) &&
            FC_PREFILL_KV_RING_CAP != 0u)
        ? (logical % FC_PREFILL_KV_RING_CAP)
        : logical;
}

static inline float prefill_attention_tg_sum(
    float value,
    uint lane,
    uint simd_group,
    uint simdgroups,
    threadgroup float* partial
) {
    float s = simd_sum(value);
    if (lane == 0u) {
        partial[simd_group] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u) {
        float v = lane < simdgroups ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) {
            partial[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

static inline float prefill_attention_tg_sum_single_bank(
    float value,
    uint lane,
    uint simd_group,
    uint simdgroups,
    threadgroup float* partial
) {
    const float result = prefill_attention_tg_sum(
        value, lane, simd_group, simdgroups, partial);
    // A single scratch bank needs an explicit reader-to-next-writer edge.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

[[kernel, max_total_threads_per_threadgroup(512)]]
kernel void attention_prefill_causal_tiled(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]
) {
    const uint t = tg.x;
    const uint qh = tg.y;
    if (t >= p.queryCount || qh >= p.numQHeads) return;

    threadgroup float partial[2u * kPrefillAttentionMaxSimdGroups];

    const uint d = tid.x;
    const bool owns = d < p.headDim;
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh / q_per_kv;
    const uint abs_q = p.startPosition + t;
    uint first = 0u;
    if (p.slidingWindow != 0u && abs_q + 1u > p.slidingWindow) {
        first = abs_q + 1u - p.slidingWindow;
    }
    const uint last_exclusive = min(p.kvValidCount, abs_q + 1u);

    device const half* q_row = Q + t * p.qTokenStrideElements + qh * p.headDim;
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (uint key = first; key < last_exclusive; ++key) {
        const uint phys_key = prefill_kv_slot(key);
        device const half* k_row = K + phys_key * p.kvTokenStrideElements + kvh * p.headDim;
        const float qv = owns ? float(q_row[d]) : 0.0f;
        const float kv = owns ? float(k_row[d]) : 0.0f;
        const uint bank = key & 1u;
        const float score = prefill_attention_tg_sum(
            qv * kv,
            lane,
            simd_group,
            simdgroups,
            partial + bank * kPrefillAttentionMaxSimdGroups) * p.scale;

        const float new_max = max(row_max, score);
        const float old_scale = row_sum > 0.0f ? fast::exp(row_max - new_max) : 0.0f;
        const float new_scale = fast::exp(score - new_max);
        if (owns) {
            device const half* v_row = V + phys_key * p.kvTokenStrideElements + kvh * p.headDim;
            acc = fma(new_scale, float(v_row[d]), acc * old_scale);
        }
        row_sum = row_sum * old_scale + new_scale;
        row_max = new_max;
    }

    if (owns) {
        device half* out_row = O + t * p.oTokenStrideElements + qh * p.headDim;
        out_row[d] = row_sum > 0.0f ? half(acc / row_sum) : half(0.0f);
    }
}

#if defined(__HAVE_TENSOR__)

constant constexpr int kPrefillTensorOpsOutputs = 8;
constant constexpr int kPrefillTensorOpsKeys = 64;
constant constexpr int kPrefillTensorOpsHeadDim = 512;

static inline void attention_prefill_full_tensorops_2d_validity_v2_impl(
    device const half* Q,
    device half* K,
    device half* V,
    device half* O,
    constant PrefillAttentionParams& p,
    uint3 tg,
    uint lid,
    uint threads,
    threadgroup half* query_tile,
    threadgroup float* score_tile,
    threadgroup float* weight_tile,
    threadgroup float* row_max,
    threadgroup float* row_sum,
    threadgroup float* row_old_scale
) {
    constexpr auto qk_desc = matmul2d_descriptor(
        kPrefillTensorOpsOutputs,
        kPrefillTensorOpsKeys,
        kPrefillTensorOpsHeadDim,
        false, true, false);
    constexpr auto pv_desc = matmul2d_descriptor(
        kPrefillTensorOpsOutputs,
        kPrefillTensorOpsHeadDim,
        kPrefillTensorOpsKeys,
        false, false, false);
    matmul2d<qk_desc, execution_simdgroups<4>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<4>> pv_op;

    using device_half_tensor =
        tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor =
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_float_tensor =
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

    const uint query_start = tg.x;
    const uint qh_start = tg.y * uint(kPrefillTensorOpsOutputs);
    const uint valid_query_rows =
        min(1u, p.queryCount - min(query_start, p.queryCount));
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh_start / q_per_kv;

    for (uint linear = lid;
         linear < uint(kPrefillTensorOpsOutputs * kPrefillTensorOpsHeadDim);
         linear += threads) {
        const uint output_row =
            linear / uint(kPrefillTensorOpsHeadDim);
        const uint d = linear % uint(kPrefillTensorOpsHeadDim);
        if (valid_query_rows != 0u) {
            query_tile[linear] = Q[
                query_start * p.qTokenStrideElements
                + (qh_start + output_row) * p.headDim
                + d];
        } else {
            query_tile[linear] = half(0.0f);
        }
    }
    if (lid < uint(kPrefillTensorOpsOutputs)) {
        row_max[lid] = -INFINITY;
        row_sum[lid] = 0.0f;
        row_old_scale[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup_half_tensor query_tensor(
        query_tile,
        dextents<int32_t, 2>(
            kPrefillTensorOpsHeadDim,
            kPrefillTensorOpsOutputs),
        array<int32_t, 2>({1, kPrefillTensorOpsHeadDim}));
    threadgroup_float_tensor weight_tensor(
        weight_tile,
        dextents<int32_t, 2>(
            kPrefillTensorOpsKeys,
            kPrefillTensorOpsOutputs),
        array<int32_t, 2>({1, kPrefillTensorOpsKeys}));
    device_half_tensor key_tensor(
        K + kvh * p.headDim,
        dextents<int32_t, 2>(
            int32_t(p.headDim),
            int32_t(p.kvValidCount)),
        array<int32_t, 2>({1, int32_t(p.kvTokenStrideElements)}));
    device_half_tensor value_tensor(
        V + kvh * p.headDim,
        dextents<int32_t, 2>(
            int32_t(p.headDim),
            int32_t(p.kvValidCount)),
        array<int32_t, 2>({1, int32_t(p.kvTokenStrideElements)}));

    auto query_slice = query_tensor.slice(0, 0);
    auto first_value_slice = value_tensor.slice(0, 0);
    auto output_accumulator =
        pv_op.get_destination_cooperative_tensor<
            decltype(weight_tensor), decltype(first_value_slice), float>();
    #pragma clang loop unroll(full)
    for (int element = 0;
         element < output_accumulator.get_capacity();
         ++element) {
        if (output_accumulator.is_valid_element(element)) {
            output_accumulator[element] = 0.0f;
        }
    }

    // This full-attention kernel starts at key zero and ignores slidingWindow.
    // The Swift selector must dispatch it only when every prior key is visible.
    const uint last =
        min(p.kvValidCount, p.startPosition + query_start + valid_query_rows);
    for (uint key_start = 0u;
         key_start < last;
         key_start += uint(kPrefillTensorOpsKeys)) {
        auto key_slice = key_tensor.slice(0, int32_t(key_start));
        auto score_product =
            qk_op.get_destination_cooperative_tensor<
                decltype(query_slice), decltype(key_slice), float>();
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < score_product.get_capacity();
             ++element) {
            if (score_product.is_valid_element(element)) {
                score_product[element] = 0.0f;
            }
        }
        qk_op.run(query_slice, key_slice, score_product);

        #pragma clang loop unroll(full)
        for (int element = 0;
             element < score_product.get_capacity();
             ++element) {
            if (!score_product.is_valid_element(element)) continue;
            const auto position =
                score_product.get_multidimensional_index(element);
            const uint key_column = uint(position[0]);
            const uint output_row = uint(position[1]);
            score_tile[
                output_row * uint(kPrefillTensorOpsKeys) + key_column] =
                score_product[element] * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < uint(kPrefillTensorOpsOutputs)) {
            const uint output_row = lid;
            const uint causal_last = valid_query_rows != 0u
                ? min(p.kvValidCount, p.startPosition + query_start + 1u)
                : 0u;
            const uint visible =
                causal_last > key_start
                    ? min(uint(kPrefillTensorOpsKeys), causal_last - key_start)
                    : 0u;

            float tile_max = -INFINITY;
            for (uint key = 0u; key < visible; ++key) {
                tile_max = max(
                    tile_max,
                    score_tile[
                        output_row * uint(kPrefillTensorOpsKeys) + key]);
            }
            const float next_max = max(row_max[output_row], tile_max);
            const float old_scale = row_sum[output_row] > 0.0f
                ? fast::exp(row_max[output_row] - next_max)
                : 0.0f;
            float tile_sum = 0.0f;
            for (uint key = 0u;
                 key < uint(kPrefillTensorOpsKeys);
                 ++key) {
                const float weight = key < visible
                    ? fast::exp(
                        score_tile[
                            output_row * uint(kPrefillTensorOpsKeys) + key]
                        - next_max)
                    : 0.0f;
                weight_tile[
                    output_row * uint(kPrefillTensorOpsKeys) + key] = weight;
                tile_sum += weight;
            }
            row_old_scale[output_row] = old_scale;
            row_sum[output_row] =
                row_sum[output_row] * old_scale + tile_sum;
            row_max[output_row] = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto value_slice = value_tensor.slice(0, int32_t(key_start));
        auto output_product =
            pv_op.get_destination_cooperative_tensor<
                decltype(weight_tensor), decltype(value_slice), float>();
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < output_product.get_capacity();
             ++element) {
            if (output_product.is_valid_element(element)) {
                output_product[element] = 0.0f;
            }
        }
        pv_op.run(weight_tensor, value_slice, output_product);
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < output_accumulator.get_capacity();
             ++element) {
            if (!output_accumulator.is_valid_element(element)
                || !output_product.is_valid_element(element)) {
                continue;
            }
            const auto position =
                output_accumulator.get_multidimensional_index(element);
            const uint output_row = uint(position[1]);
            output_accumulator[element] =
                fma(
                    1.0f,
                    output_product[element],
                    output_accumulator[element] * row_old_scale[output_row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (int element = 0;
         element < output_accumulator.get_capacity();
         ++element) {
        if (!output_accumulator.is_valid_element(element)) continue;
        const auto position =
            output_accumulator.get_multidimensional_index(element);
        const uint d = uint(position[0]);
        const uint output_row = uint(position[1]);
        if (valid_query_rows != 0u) {
            const float denominator = row_sum[output_row];
            O[
                query_start * p.oTokenStrideElements
                + (qh_start + output_row) * p.headDim
                + d] = denominator > 0.0f
                    ? half(output_accumulator[element] / denominator)
                    : half(0.0f);
        }
    }
}

kernel void attention_prefill_full_tensorops_2d_validity_v2(
    device const half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]
) {
    threadgroup half query_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsHeadDim];
    threadgroup float score_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsKeys];
    threadgroup float weight_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsKeys];
    threadgroup float row_max[kPrefillTensorOpsOutputs];
    threadgroup float row_sum[kPrefillTensorOpsOutputs];
    threadgroup float row_old_scale[kPrefillTensorOpsOutputs];
    attention_prefill_full_tensorops_2d_validity_v2_impl(
        Q, K, V, O, p, tg, lid, threads3.x,
        query_tile, score_tile, weight_tile,
        row_max, row_sum, row_old_scale);
}

#endif
