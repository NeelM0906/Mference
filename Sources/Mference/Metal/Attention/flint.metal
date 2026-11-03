#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant constexpr int kFlintQHeads = 32;
constant constexpr int kFlintKVHeads = 4;
constant constexpr int kFlintGQAFactor = 8;
constant constexpr int kFlintHeadDim = 128;
constant constexpr int kFlintBN = 32;
constant constexpr int kFlintBD = 32;
constant constexpr int kFlintTopK = 6;

// One threadgroup per query head. Each SIMD group scans a strided subset of
// logical positions, then the threadgroup merges the online-softmax states.
// The learned sink starts with denominator weight one and contributes no
// value.
[[kernel, max_total_threads_per_threadgroup(1024)]]
void flint_attention_sink_fp16_d128(
    device const half* queries [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device half* out [[buffer(3)]],
    device const bfloat* sinks [[buffer(4)]],
    constant uint& first [[buffer(5)]],
    constant uint& last [[buffer(6)]],
    constant uint& ring_capacity [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int qk_per_thread = kFlintHeadDim / kFlintBD;
    constexpr int v_per_thread = kFlintHeadDim / kFlintBD;

    thread float q[qk_per_thread];
    thread float k[qk_per_thread];
    thread float o[v_per_thread];
    threadgroup float outputs[kFlintBN * kFlintBD];
    threadgroup float max_scores[kFlintBN];
    threadgroup float sum_exp_scores[kFlintBN];

    const int q_head = tid.x;
    const int kv_head = q_head / kFlintGQAFactor;
    device const half* q_row = queries + q_head * kFlintHeadDim
        + simd_lid * qk_per_thread;

    for (int i = 0; i < qk_per_thread; ++i) {
        q[i] = float(q_row[i]) * scale;
        o[i] = 0.0f;
    }

    float max_score = -metal::numeric_limits<float>::max();
    float sum_exp_score = 0.0f;

    for (uint p = first + simd_gid; p < last; p += kFlintBN) {
        const uint physical = ring_capacity > 0 ? p % ring_capacity : p;
        device const half* k_row = keys
            + (physical * kFlintKVHeads + kv_head) * kFlintHeadDim
            + simd_lid * qk_per_thread;
        device const half* v_row = values
            + (physical * kFlintKVHeads + kv_head) * kFlintHeadDim
            + simd_lid * v_per_thread;

        for (int i = 0; i < qk_per_thread; ++i) {
            k[i] = float(k_row[i]);
        }
        float score = 0.0f;
        for (int i = 0; i < qk_per_thread; ++i) {
            score += q[i] * k[i];
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int i = 0; i < v_per_thread; ++i) {
            o[i] = o[i] * factor + exp_score * float(v_row[i]);
        }
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float group_max = max_scores[simd_lid];
    const float global_max = simd_max(group_max);
    const float sink = float(sinks[q_head]);
    const float final_max = max(global_max, sink);
    const float merge_factor = fast::exp(group_max - final_max);
    const float denom = simd_sum(sum_exp_scores[simd_lid] * merge_factor)
        + fast::exp(sink - final_max);

    for (int i = 0; i < v_per_thread; ++i) {
        outputs[simd_lid * kFlintBD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * kFlintBD + simd_lid]
                        * merge_factor) / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        device half* out_row = out + q_head * kFlintHeadDim
            + simd_gid * v_per_thread;
        for (int i = 0; i < v_per_thread; ++i) {
            out_row[i] = half(o[i]);
        }
    }
}

// BF16 router matrix times FP16 activation. One SIMD group per expert row.
[[kernel, max_total_threads_per_threadgroup(128)]]
void flint_router_gemv_bf16(
    device const bfloat* weights [[buffer(0)]],
    device const half* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& experts [[buffer(3)]],
    constant uint& dimension [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint expert = tg * 4u + sg;
    if (expert >= experts) return;
    device const bfloat* row = weights + expert * dimension;
    float acc = 0.0f;
    for (uint i = lane; i < dimension; i += 32u) {
        acc = fma(float(row[i]), float(hidden[i]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) logits[expert] = acc;
}

// Select top-6 by raw logit (sigmoid is monotone), apply sigmoid to those
// selected scores, then renormalize their sum to one.
[[kernel, max_total_threads_per_threadgroup(32)]]
void flint_router_sigmoid_top6_norm(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device half* weights [[buffer(2)]],
    constant uint& experts [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]]) {
    if (tid != 0) return;
    float best[kFlintTopK];
    uint best_index[kFlintTopK];
    for (int i = 0; i < kFlintTopK; ++i) {
        best[i] = -metal::numeric_limits<float>::max();
        best_index[i] = 0xFFFFFFFFu;
    }
    for (uint e = 0; e < experts; ++e) {
        const float score = logits[e];
        int pos = kFlintTopK;
        for (int i = 0; i < kFlintTopK; ++i) {
            if (score > best[i] || (score == best[i] && e < best_index[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= kFlintTopK) continue;
        for (int i = kFlintTopK - 1; i > pos; --i) {
            best[i] = best[i - 1];
            best_index[i] = best_index[i - 1];
        }
        best[pos] = score;
        best_index[pos] = e;
    }

    float activated[kFlintTopK];
    float denom = 0.0f;
    for (int i = 0; i < kFlintTopK; ++i) {
        const float x = best[i];
        activated[i] = x >= 0.0f
            ? 1.0f / (1.0f + fast::exp(-x))
            : fast::exp(x) / (1.0f + fast::exp(x));
        denom += activated[i];
    }
    for (int i = 0; i < kFlintTopK; ++i) {
        indices[i] = best_index[i];
        weights[i] = half(activated[i] / denom);
    }
}

// Batched prefill router: one threadgroup covers four experts for one token.
// `hidden` is [tokens, hiddenSize] FP16 and `logits` is
// [tokens, experts] FP32.
[[kernel, max_total_threads_per_threadgroup(128)]]
void flint_router_gemv_bf16_batched(
    device const bfloat* weights [[buffer(0)]],
    device const half* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& experts [[buffer(3)]],
    constant uint& dimension [[buffer(4)]],
    constant uint& hidden_stride [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint expert = tg.x * 4u + sg;
    const uint token = tg.y;
    if (expert >= experts) return;
    device const bfloat* row = weights + expert * dimension;
    device const half* x = hidden + token * hidden_stride;
    float acc = 0.0f;
    for (uint i = lane; i < dimension; i += 32u) {
        acc = fma(float(row[i]), float(x[i]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        logits[token * experts + expert] = acc;
    }
}

// One threadgroup per token; lane zero selects and normalizes the top-k.
[[kernel, max_total_threads_per_threadgroup(32)]]
void flint_router_sigmoid_top6_norm_batched(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device half* weights [[buffer(2)]],
    constant uint& experts [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    if (tid != 0 || top_k == 0u || top_k > 6u) return;
    device const float* token_logits = logits + tg * experts;
    device uint* token_indices = indices + tg * top_k;
    device half* token_weights = weights + tg * top_k;

    float best[6];
    uint best_index[6];
    for (uint i = 0; i < top_k; ++i) {
        best[i] = -metal::numeric_limits<float>::max();
        best_index[i] = 0xFFFFFFFFu;
    }
    for (uint e = 0; e < experts; ++e) {
        const float score = token_logits[e];
        uint pos = top_k;
        for (uint i = 0; i < top_k; ++i) {
            if (score > best[i] || (score == best[i] && e < best_index[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= top_k) continue;
        for (uint i = top_k - 1; i > pos; --i) {
            best[i] = best[i - 1];
            best_index[i] = best_index[i - 1];
        }
        best[pos] = score;
        best_index[pos] = e;
    }

    float activated[6];
    float denom = 0.0f;
    for (uint i = 0; i < top_k; ++i) {
        const float x = best[i];
        activated[i] = x >= 0.0f
            ? 1.0f / (1.0f + fast::exp(-x))
            : fast::exp(x) / (1.0f + fast::exp(x));
        denom += activated[i];
    }
    for (uint i = 0; i < top_k; ++i) {
        token_indices[i] = best_index[i];
        token_weights[i] = half(activated[i] / denom);
    }
}
