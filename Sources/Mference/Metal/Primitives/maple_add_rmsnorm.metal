#include <metal_stdlib>
using namespace metal;

constant constexpr uint kMapleHiddenSize = 2048;
constant constexpr uint kMapleThreads = 256;
constant constexpr uint kMapleValuesPerThread = kMapleHiddenSize / kMapleThreads;
constant constexpr uint kMapleSIMDGroups = kMapleThreads / 32;

[[kernel, max_total_threads_per_threadgroup(256)]]
void maple_add_rmsnorm_bf16_d2048(
    device       bfloat* hidden [[buffer(0)]],
    device const bfloat* delta  [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device       bfloat* normed [[buffer(3)]],
    constant     float& eps     [[buffer(4)]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint simdLane               [[thread_index_in_simdgroup]],
    uint simdGroup              [[simdgroup_index_in_threadgroup]])
{
    float roundedHidden[kMapleValuesPerThread];
    float sumSquares = 0.0f;

    for (uint i = 0; i < kMapleValuesPerThread; ++i) {
        const uint j = tid * kMapleValuesPerThread + i;
        const bfloat rounded = bfloat(float(hidden[j]) + float(delta[j]));
        hidden[j] = rounded;
        roundedHidden[i] = float(rounded);
        sumSquares += roundedHidden[i] * roundedHidden[i];
    }

    sumSquares = simd_sum(sumSquares);
    threadgroup float partial[kMapleSIMDGroups];
    if (simdLane == 0) partial[simdGroup] = sumSquares;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    for (uint i = 0; i < kMapleSIMDGroups; ++i) total += partial[i];
    const float scale = metal::rsqrt(total / float(kMapleHiddenSize) + eps);

    for (uint i = 0; i < kMapleValuesPerThread; ++i) {
        const uint j = tid * kMapleValuesPerThread + i;
        normed[j] = bfloat(roundedHidden[i] * scale * float(weight[j]));
    }
}
