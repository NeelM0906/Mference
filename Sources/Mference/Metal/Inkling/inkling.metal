#include <metal_stdlib>
using namespace metal;

// ============================================================================
// FP32-row helpers kept from the retired Inkling-Small kernels. The DFlash2
// drafter (`DFlash2Kernels`) binds them by these names; the rest of the
// family's kernels were removed with the family.
// ============================================================================

kernel void inkling_f16_to_f32(
    device const half*  src   [[buffer(0)]],
    device       float* dst   [[buffer(1)]],
    constant     uint&  count [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    dst[i] = float(src[i]);
}

// Batched FP32-residual RMSNorm. Grid X is the token row; each threadgroup
// runs a 256-thread reduction and writes one contiguous FP16 row for the
// following matrix projections.
[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling_rms_f32in_prefill(
    device const float*  x      [[buffer(0)]], // [T, D]
    device const bfloat* weight [[buffer(1)]],
    device       half*   out    [[buffer(2)]], // [T, D]
    constant     uint&   d      [[buffer(3)]],
    constant     uint&   rows   [[buffer(4)]],
    constant     float&  eps    [[buffer(5)]],
    uint token [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups    [[simdgroups_per_threadgroup]]
) {
    threadgroup float scratch[8];
    threadgroup float bcast;
    if (token >= rows) return;
    device const float* x_row = x + token * d;
    device half* out_row = out + token * d;
    float acc = 0.0f;
    for (uint i = lid; i < d; i += lsize) {
        const float value = x_row[i];
        acc = fma(value, value, acc);
    }
    float sum = simd_sum(acc);
    if (simd_lane_id == 0) { scratch[simd_group_id] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float value = simd_lane_id < simdgroups ? scratch[simd_lane_id] : 0.0f;
        value = simd_sum(value);
        if (simd_lane_id == 0) { bcast = value; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(bcast / float(d) + eps);
    for (uint i = lid; i < d; i += lsize) {
        out_row[i] = half(x_row[i] * inv * float(weight[i]));
    }
}
