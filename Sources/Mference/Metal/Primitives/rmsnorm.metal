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
// rmsnorm — RMS-norm over hidden dim.
//
//   inv     = rsqrt(mean(x[i]^2) + eps)
//   y[i]    = x[i] * inv * weight[i]
//
// FP32 accumulator (numerical stability: D=2816 with FP16 inputs can overflow
// FP16 sum-of-squares once activations grow past ~1.2 in magnitude).
// FP16 storage in and out. Learned weights are BF16 where present.
//
// Dispatch: one threadgroup per row, 256 threads per group. Two-stage block
// reduce — SIMD-group simd_sum, then a single SIMD-group merges the partials.
// ============================================================================

// Original profiles use eight physical SIMD partials. QAT uses up to 32
// logical four-value partials within the same dispatch. Slot 0 broadcasts inv.
constant constexpr uint kRmsMaxSimdGroups = 32;
constant uint FC_RMS_D [[function_constant(30)]];
constant bool FC_RMS_USE_FC [[function_constant(31)]];

static inline uint rms_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_RMS_USE_FC) &&
            FC_RMS_USE_FC &&
            is_function_constant_defined(FC_RMS_D)) ? FC_RMS_D : D;
}

// Common block reduction. Returns `inv = rsqrt(mean(x^2) + eps)` broadcast to
// every thread via threadgroup memory slot 0.
static inline float rms_block_inv(
    device const half* x,
    uint  D,
    float eps,
    uint  lid,
    uint  lsize,
    uint  simd_lane_id,
    uint  simd_group_id,
    uint  simdgroups,
    threadgroup float* partial
) {
    if (kGemmaSourceFP16) return gemma_source_norm_inv(
        x, D, eps, simd_lane_id, simd_group_id, simdgroups, partial);
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
            float mean_sq = v / float(D);
            partial[0] = rsqrt(mean_sq + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

// Gemma 4 RMS norms ship as BF16 weight vectors (all 30 layers' input /
// post-attn / pre-FFN / post-FFN norms, plus q/k_norm). Math is identical to
// the no-scale form below, with a learned weight applied after normalization.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w(
    device const half*   x          [[buffer(0)]],   // [D] FP16
    device const bfloat* weight     [[buffer(1)]],   // [D] BF16
    device       half*   out        [[buffer(2)]],   // [D] FP16
    constant     uint&   D          [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint DD = rms_fc_d(D);
    const float inv = rms_block_inv(x, DD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups,
                                    partial);

    for (uint i = lid; i < DD; i += lsize) {
        float xv = float(x[i]);
        float wv = float(weight[i]);
        out[i] = gemma_weighted_norm(xv, inv, wv);
    }
}

// Gemma 4 applies q_norm/k_norm
// (BF16 weight, shared across heads) and v_norm (no-scale) to each attention
// head independently. These kernels process all heads in one dispatch, with
// one threadgroup per head, avoiding a chain of tiny serialized encoders.
// Math is identical to the single-row kernels applied per head.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_perhead(
    device const half*   x          [[buffer(0)]],   // [numHeads * headDim] FP16
    device const bfloat* weight     [[buffer(1)]],   // [headDim] BF16, shared per head
    device       half*   out        [[buffer(2)]],   // [numHeads * headDim] FP16
    constant     uint&   headDim    [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint HD = rms_fc_d(headDim);
    device const half* xh = x   + head * HD;
    device       half* oh = out + head * HD;
    const float inv = rms_block_inv(xh, HD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(xh[i]);
        float wv = float(weight[i]);
        oh[i] = gemma_weighted_norm(xv, inv, wv);
    }
}

// Qwen4-Exp (`qwen38flashnext`) group RMSNorm: `group_size = hidden`.
//
// The residual stream is `hc_count` copies of a `hidden`-wide vector laid end to
// end. Each copy is normalized over its own `hidden` channels — like the
// per-head kernel above — but the learned weight spans the WHOLE bundle, one
// scale per (stream, channel), so it cannot be shared the way `q_norm` is. That
// single indexing difference is the whole kernel.
//
// Dispatch: one threadgroup per (row, group); grid width = rows * groups. The
// weight is reused across rows, so the group index is recovered modulo `groups`.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_grouped(
    device const half*   x          [[buffer(0)]],   // [rows * groups * G] FP16
    device const bfloat* weight     [[buffer(1)]],   // [groups * G] BF16
    device       half*   out        [[buffer(2)]],   // [rows * groups * G] FP16
    constant     uint&   groupSize  [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    constant     uint&   groups     [[buffer(5)]],
    uint  slot             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint G = rms_fc_d(groupSize);
    device const half*   xg = x      + slot * G;
    device       half*   og = out    + slot * G;
    device const bfloat* wg = weight + (slot % groups) * G;
    const float inv = rms_block_inv(xg, G, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < G; i += lsize) {
        og[i] = gemma_weighted_norm(float(xg[i]), inv, float(wg[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_no_scale_perhead(
    device const half*  x          [[buffer(0)]],   // [numHeads * headDim] FP16
    device       half*  out        [[buffer(1)]],   // [numHeads * headDim] FP16
    constant     uint&  headDim    [[buffer(2)]],
    constant     float& eps        [[buffer(3)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint HD = rms_fc_d(headDim);
    device const half* xh = x   + head * HD;
    device       half* oh = out + head * HD;
    const float inv = rms_block_inv(xh, HD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < HD; i += lsize) {
        oh[i] = half(float(xh[i]) * inv);
    }
}

// Gemma 4 v_norm and the MoE router's internal norm are no-scale RMSNorm:
// y[i] = x[i] * rsqrt(mean(x^2) + eps). There is no resident weight tensor.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_no_scale(
    device const half*  x          [[buffer(0)]],   // [D] FP16
    device       half*  out        [[buffer(1)]],   // [D] FP16
    constant     uint&  D          [[buffer(2)]],
    constant     float& eps        [[buffer(3)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint DD = rms_fc_d(D);
    const float inv = rms_block_inv(x, DD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups,
                                    partial);

    for (uint i = lid; i < DD; i += lsize) {
        float xv = float(x[i]);
        out[i] = half(xv * inv);
    }
}
