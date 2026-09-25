// Adapted from Apple MLX 0.32.2 sdpa_vector.h, gemv.h and softmax.h.
// Copyright © 2023-2024 Apple Inc. MIT license; see LICENSE-MLX.
#include <metal_stdlib>
using namespace metal;

// QAT's pinned FP16 execution profile. Compile this module with safe math:
// MLX 0.32.2 uses contiguous dot-product lanes and explicit FP16 boundaries
// for its unfused, 512-wide attention. The existing model profiles retain
// attention.metal. These kernels consume Mference's time-major KV layout.

// Matches GemmaQATPrefillAttention.Parameters. Query tiles share bounded
// scratch; each row uses its own causal end and source reduction geometry.
struct GemmaQATPrefillParams {
    uint startPosition, queryCount, headDim, numQHeads, numKVHeads;
    uint kvValidCount, slidingWindow, kvTokenStrideElements;
    uint qTokenStrideElements, oTokenStrideElements;
    float scale;
    uint queryOffset, scratchStride;
};

kernel void gemma_qat_prefill_attention_swa(
    device const half* query [[buffer(0)]], device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]], device half* output [[buffer(3)]],
    constant GemmaQATPrefillParams& p [[buffer(4)]],
    constant uint& ringCapacity [[buffer(5)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint head = grid.x, row = grid.y;
    const uint end = p.startPosition + row + 1u;
    const uint start = end > p.slidingWindow ? end - p.slidingWindow : 0u;
    query += row * p.qTokenStrideElements;
    output += row * p.oTokenStrideElements;
    threadgroup float maxima[32], denominators[32], transpose[1024];
    float q[8], accumulator[8];
    const uint kvHead = head / 2u;
    for (uint j = 0; j < 8u; ++j) {
        q[j] = float(query[head * 256u + lane * 8u + j]);
        accumulator[j] = 0.0f;
    }
    float maximum = -FLT_MAX, denominator = 0.0f;
    for (uint position = start + group; position < end; position += 32u) {
        const uint slot = ringCapacity == 0u ? position : position % ringCapacity;
        const uint base = slot * p.kvTokenStrideElements + kvHead * 256u + lane * 8u;
        float score = 0.0f;
        for (uint j = 0; j < 8u; ++j) score += q[j] * float(keys[base + j]);
        score = simd_sum(score);
        const float updated = max(maximum, score);
        const float factor = fast::exp(maximum - updated);
        const float probability = fast::exp(score - updated);
        denominator = denominator * factor + probability;
        maximum = updated;
        for (uint j = 0; j < 8u; ++j) {
            accumulator[j] = accumulator[j] * factor + probability * float(values[base + j]);
        }
    }
    if (lane == 0) { maxima[group] = maximum; denominators[group] = denominator; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = maxima[lane];
    const float updated = simd_max(maximum);
    const float factor = fast::exp(maximum - updated);
    denominator = simd_sum(denominators[lane] * factor);
    for (uint j = 0; j < 8u; ++j) {
        transpose[lane * 32u + group] = accumulator[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float sum = simd_sum(transpose[group * 32u + lane] * factor);
        if (lane == 0) output[head * 256u + group * 8u + j] = half(sum / denominator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gemma_qat_prefill_attention_scores(
    device const half* query [[buffer(0)]], device const half* keys [[buffer(1)]],
    device half* scores [[buffer(2)]], constant GemmaQATPrefillParams& p [[buffer(3)]],
    uint3 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    const uint head = grid.x, position = grid.y;
    const uint row = p.queryOffset + grid.z;
    const uint length = p.startPosition + row + 1u;
    if (position >= length) return;
    query += row * p.qTokenStrideElements;
    scores += grid.z * 16u * p.scratchStride;
    // The source GEMV uses eight column SIMD groups for short output vectors,
    // and one SIMD group per output row once 512 < 16 * length.
    const uint groups = length <= 32u ? 8u : 1u;
    float sum = 0.0f;
    if (group < groups) {
        for (uint base = tid * 4u; base < 512u; base += groups * 128u) {
            for (uint j = 0; j < 4u; ++j) {
                const uint d = base + j;
                sum += float(query[head * 512u + d]) * float(keys[position * p.kvTokenStrideElements + (head / 8u) * 512u + d]);
            }
        }
    }
    for (ushort delta = 16; delta > 0; delta >>= 1) sum += simd_shuffle_down(sum, delta);
    if (lane == 0) partial[group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        sum = partial[0];
        for (uint g = 1; g < groups; ++g) sum += partial[g];
        scores[head * p.scratchStride + position] = half(sum);
    }
}

kernel void gemma_qat_prefill_attention_probabilities(
    device const half* scores [[buffer(0)]], device half* probabilities [[buffer(1)]],
    constant GemmaQATPrefillParams& p [[buffer(2)]], uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint2 threadgroupSize [[threads_per_threadgroup]],
    uint group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint head = grid.x, row = p.queryOffset + grid.y;
    const uint length = p.startPosition + row + 1u;
    const uint threads = threadgroupSize.x;
    threadgroup float maxima[32], denominators[32];
    scores += (grid.y * 16u + head) * p.scratchStride;
    probabilities += (grid.y * 16u + head) * p.scratchStride;
    if (group == 0) { maxima[lane] = -INFINITY; denominators[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = -FLT_MAX, denominator = 0.0f;
    float value[4];
    if (length <= 4096u) {
        for (uint j = 0; j < 4u; ++j) {
            const uint position = tid * 4u + j;
            value[j] = position < length ? float(scores[position]) : -INFINITY;
            maximum = max(maximum, value[j]);
        }
        maximum = simd_max(maximum);
        if (lane == 0) maxima[group] = maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group == 0) {
            maximum = simd_max(maxima[lane]);
            if (lane == 0) maxima[0] = maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = maxima[0];
        for (uint j = 0; j < 4u; ++j) {
            value[j] = fast::exp(value[j] - maximum);
            denominator += value[j];
        }
        denominator = simd_sum(denominator);
        if (lane == 0) denominators[group] = denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group == 0) {
            denominator = simd_sum(denominators[lane]);
            if (lane == 0) denominators[0] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inverse = 1.0f / denominators[0];
        for (uint j = 0; j < 4u; ++j) {
            const uint position = tid * 4u + j;
            if (position < length) probabilities[position] = half(value[j] * inverse);
        }
    } else {
        for (uint base = tid * 4u; base < length; base += threads * 4u) {
            const float previous = maximum;
            for (uint j = 0; j < 4u; ++j) {
                value[j] = base + j < length ? float(scores[base + j]) : -INFINITY;
                maximum = max(maximum, value[j]);
            }
            denominator *= fast::exp(previous - maximum);
            for (uint j = 0; j < 4u; ++j) denominator += fast::exp(value[j] - maximum);
        }
        float previous = maximum;
        maximum = simd_max(maximum);
        denominator *= fast::exp(previous - maximum);
        denominator = simd_sum(denominator);
        previous = maximum;
        if (lane == 0) maxima[group] = maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = simd_max(maxima[lane]);
        denominator *= fast::exp(previous - maximum);
        if (lane == 0) denominators[group] = denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inverse = 1.0f / simd_sum(denominators[lane]);
        for (uint base = tid * 4u; base < length; base += threads * 4u) {
            for (uint j = 0; j < 4u; ++j) {
                if (base + j < length) probabilities[base + j] = half(fast::exp(float(scores[base + j]) - maximum) * inverse);
            }
        }
    }
}

kernel void gemma_qat_prefill_attention_values(
    device const half* probabilities [[buffer(0)]], device const half* values [[buffer(1)]],
    device half* output [[buffer(2)]], constant GemmaQATPrefillParams& p [[buffer(3)]],
    uint3 grid [[threadgroup_position_in_grid]],
    uint group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint head = grid.y;
    const uint queryRow = p.queryOffset + grid.z;
    const uint length = p.startPosition + queryRow + 1u;
    probabilities += grid.z * 16u * p.scratchStride;
    output += queryRow * p.oTokenStrideElements;
    const uint row = lane / 4u, column = grid.x * 64u + group * 16u + (lane % 4u) * 4u;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint base = row * 4u; base < length; base += 32u) {
        for (uint j = 0; j < 4u && base + j < length; ++j) {
            const float probability = float(probabilities[head * p.scratchStride + base + j]);
            for (uint d = 0; d < 4u; ++d) {
                sum[d] += probability * float(values[(base + j) * p.kvTokenStrideElements + (head / 8u) * 512u + column + d]);
            }
        }
    }
    for (uint d = 0; d < 4u; ++d) {
        for (ushort delta = 16; delta >= 4; delta >>= 1) sum[d] += simd_shuffle_down(sum[d], delta);
        if (row == 0) output[head * 512u + column + d] = half(sum[d]);
    }
}
