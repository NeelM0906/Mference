// Vector-SDPA kernels adapted from Apple MLX v0.32.0 `sdpa_vector.h`.
// Copyright © 2024 Apple Inc. Licensed under the MIT License.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant uint maple_qk_rope_dim [[function_constant(0)]];
constant int maple_attention_two_pass_blocks [[function_constant(1)]];

constant constexpr uint kMapleHeadDim = 128;
constant constexpr uint kMapleQHeads = 16;
constant constexpr uint kMapleKVHeads = 4;

[[kernel, max_total_threads_per_threadgroup(32)]]
void maple_qk_norm_rope_decode(
    device       bfloat* q [[buffer(0)]],
    device       bfloat* k [[buffer(1)]],
    device const bfloat* q_weight [[buffer(2)]],
    device const bfloat* k_weight [[buffer(3)]],
    device const float* inv_freq [[buffer(4)]],
    constant uint& position [[buffer(5)]],
    constant float& rms_eps [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint head [[threadgroup_position_in_grid]]) {
    const bool is_q = head < kMapleQHeads;
    const uint local_head = is_q ? head : head - kMapleQHeads;
    if (!is_q && local_head >= kMapleKVHeads) return;

    device bfloat* xh = is_q
        ? q + local_head * kMapleHeadDim
        : k + local_head * kMapleHeadDim;
    device const bfloat* wh = is_q ? q_weight : k_weight;
    constexpr uint values_per_lane = kMapleHeadDim / 32;

    float source[values_per_lane];
    float sum_squares = 0.0f;
    for (uint i = 0; i < values_per_lane; ++i) {
        const uint j = lane * values_per_lane + i;
        const float value = float(xh[j]);
        source[i] = value;
        sum_squares += value * value;
    }
    sum_squares = simd_sum(sum_squares);
    const float scale = metal::rsqrt(sum_squares / float(kMapleHeadDim) + rms_eps);

    float normalized[values_per_lane];
    for (uint i = 0; i < values_per_lane; ++i) {
        const uint j = lane * values_per_lane + i;
        normalized[i] = source[i] * scale * float(wh[j]);
    }

    const uint rope_half = maple_qk_rope_dim > 0 ? maple_qk_rope_dim / 2 : 1;
    for (uint i = 0; i < values_per_lane; ++i) {
        const uint j = lane * values_per_lane + i;
        float value = normalized[i];
        if (maple_qk_rope_dim > 0 && j < maple_qk_rope_dim) {
            const uint partner_lane = lane < 8 ? lane + 8 : lane - 8;
            const float partner = simd_shuffle(normalized[i], partner_lane);
            const uint pair = j < rope_half ? j : j - rope_half;
            const float angle = float(position) * inv_freq[pair];
            const float cosine = metal::cos(angle);
            const float sine = metal::sin(angle);
            value = j < rope_half
                ? value * cosine - partner * sine
                : value * cosine + partner * sine;
        }
        xh[j] = bfloat(value);
    }
}

[[kernel, max_total_threads_per_threadgroup(1024)]]
void maple_sdpa_vector_bf16_d128(
    device const bfloat* queries [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant int& sequence_length [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int D = 128;
    constexpr int V = 128;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = V / BD;
    constexpr int gqa_factor = 4;
    constexpr int k_head_stride = D;
    constexpr int k_seq_stride = 4 * D;
    constexpr int v_head_stride = D;
    constexpr int v_seq_stride = 4 * D;
    const int inner_k_stride = BN * k_seq_stride;
    const int inner_v_stride = BN * v_seq_stride;

    typedef float U;
    thread U q[qk_per_thread];
    thread U k[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    const int q_head = tid.x;
    const int kv_head = q_head / gqa_factor;
    queries += q_head * D + simd_lid * qk_per_thread;
    keys += kv_head * k_head_stride + simd_gid * k_seq_stride
        + simd_lid * qk_per_thread;
    values += kv_head * v_head_stride + simd_gid * v_seq_stride
        + simd_lid * v_per_thread;
    out += q_head * V + simd_gid * v_per_thread;

    for (int i = 0; i < qk_per_thread; ++i) q[i] = U(scale) * queries[i];
    for (int i = 0; i < v_per_thread; ++i) o[i] = 0;

    U max_score = -metal::numeric_limits<U>::max();
    U sum_exp_score = 0;
    for (int position = simd_gid; position < sequence_length; position += BN) {
        for (int i = 0; i < qk_per_thread; ++i) k[i] = keys[i];
        U score = 0;
        for (int i = 0; i < qk_per_thread; ++i) score += q[i] * k[i];
        score = simd_sum(score);

        const U new_max = max(max_score, score);
        const U factor = fast::exp(max_score - new_max);
        const U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int i = 0; i < v_per_thread; ++i) {
            o[i] = o[i] * factor + exp_score * values[i];
        }
        keys += inner_k_stride;
        values += inner_v_stride;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    const U new_max = simd_max(max_score);
    const U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
    for (int i = 0; i < v_per_thread; ++i) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : o[i] / sum_exp_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; ++i) out[i] = bfloat(o[i]);
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void maple_sdpa_vector_2pass_1_bf16_d128(
    device const bfloat* queries [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device bfloat* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant int& sequence_length [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int D = 128;
    constexpr int V = 128;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = V / BD;
    constexpr int k_head_stride = D;
    constexpr int k_seq_stride = 4 * D;
    constexpr int v_head_stride = D;
    constexpr int v_seq_stride = 4 * D;

    typedef float U;
    thread U q[qk_per_thread];
    thread U o[v_per_thread] = {0};

    const int kv_head = tid.x;
    const int block = tid.z;
    const int q_head = tptg.y * kv_head + tidtg.y;
    const int q_batch_head = q_head;
    queries += q_batch_head * D + simd_lid * qk_per_thread;
    keys += kv_head * k_head_stride + block * k_seq_stride + simd_lid * qk_per_thread;
    values += kv_head * v_head_stride + block * v_seq_stride + simd_lid * v_per_thread;
    partials += q_batch_head * maple_attention_two_pass_blocks * V
        + block * V + simd_lid * v_per_thread;
    sums += q_batch_head * maple_attention_two_pass_blocks + block;
    maxs += q_batch_head * maple_attention_two_pass_blocks + block;

    for (int i = 0; i < qk_per_thread; ++i) q[i] = U(scale) * queries[i];
    U max_score = -metal::numeric_limits<U>::max();
    U sum_exp_score = 0;
    for (int position = block; position < sequence_length;
         position += maple_attention_two_pass_blocks) {
        U score = 0;
        for (int i = 0; i < qk_per_thread; ++i) score += q[i] * keys[i];
        score = simd_sum(score);
        const U new_max = max(max_score, score);
        const U factor = fast::exp(max_score - new_max);
        const U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int i = 0; i < v_per_thread; ++i) {
            o[i] = o[i] * factor + exp_score * values[i];
        }
        keys += maple_attention_two_pass_blocks * k_seq_stride;
        values += maple_attention_two_pass_blocks * v_seq_stride;
    }
    if (simd_lid == 0) {
        sums[0] = sum_exp_score;
        maxs[0] = max_score;
    }
    for (int i = 0; i < v_per_thread; ++i) partials[i] = bfloat(o[i]);
}

[[kernel, max_total_threads_per_threadgroup(1024)]]
void maple_sdpa_vector_2pass_2_bf16_d128(
    device const bfloat* partials [[buffer(0)]],
    device const float* sums [[buffer(1)]],
    device const float* maxs [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant int& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int D = 128;
    constexpr int elem_per_thread = D / BD;
    typedef float U;

    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];
    const int head = tid.x;
    const int q_offset = head;
    partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
    sums += q_offset * blocks;
    maxs += q_offset * blocks;
    out += q_offset * D + simd_gid * elem_per_thread;

    U sum_exp_score = 0.0;
    U max_score = -metal::numeric_limits<U>::max();
    for (int b = 0; b < blocks / BN; ++b) {
        max_score = max(max_score, maxs[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);
    for (int b = 0; b < blocks / BN; ++b) {
        const U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * sums[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);
    for (int b = 0; b < blocks / BN; ++b) {
        const U factor = fast::exp(maxs[simd_gid] - max_score);
        for (int i = 0; i < elem_per_thread; ++i) o[i] += factor * U(partials[i]);
        maxs += BN;
        sums += BN;
        partials += BN * D;
    }
    for (int i = 0; i < elem_per_thread; ++i) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : o[i] / sum_exp_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        for (int i = 0; i < elem_per_thread; ++i) out[i] = bfloat(o[i]);
    }
}
