#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Inkling-Small family kernels.
//
// Semantics transcribed from the reference `inkling_mlx` package shipped in
// the checkpoint repo (see docs/INKLING_SMALL.md "Forward-pass contract"):
//
//  - inkling_sconv_step: depthwise causal 1-D conv, kernel K, groups == C,
//    no bias, no activation, PLUS the module-internal residual add. State is
//    the last K-1 inputs, fp32, updated in place. Reference computes in fp32
//    regardless of model dtype.
//  - inkling_qk_norm: per-head RMS norm (weighted) on Q and K. V is NOT
//    normalized in this family, which is why the Gemma fused epilogue (which
//    RMS-norms V) is not reused.
//  - inkling_attention_decode: single-token GQA attention with the learned
//    relative-position bias computed inline, optional sliding window over a
//    ring-buffered cache, and the long-context log-scaling factor `tau`
//    multiplying both q·k and the bias (causal mask unaffected).
//  - inkling_router_select: sigmoid router. Correction bias affects SELECTION
//    only; weights are softmax(logsigmoid(raw logits)) over the 6 selected
//    experts AND the 2 shared-expert sink logits, scaled by
//    route_scale * global_scale. Whole router runs in fp32 — bf16 rounding
//    flips near-tied top-k picks (reference warning).
//  - inkling_gamma_combine: y = g0*a + g1*b for the two shared experts.
//  - inkling_scale_f16: in-place scalar multiply (muP logit scaling).
// ============================================================================

constant constexpr uint kInklingAttnThreads = 128;
constant constexpr uint kInklingAttnMaxSimdGroups = 4;
constant constexpr uint kInklingMaxHeadDim = 128;
constant constexpr uint kInklingMaxDRel = 16;

// ----------------------------------------------------------------------------
// Depthwise short convolution, one decode step.
//   x_in   : [C] half — this step's input column.
//   state  : [C, K-1] float — previous K-1 inputs per channel (oldest first).
//   w      : [C, K] bfloat — per-channel taps (checkpoint layout [C, K, 1]).
//   out    : [C] half — conv(x) + x, written at out_offset.
// out[c] = x[c] + sum_{j<K-1} w[c,j]*state[c,j] + w[c,K-1]*x[c]; then the
// state row shifts left and x enters at the end.
// ----------------------------------------------------------------------------
kernel void inkling_sconv_step(
    device const half*   x_in    [[buffer(0)]],
    device       float*  state   [[buffer(1)]],
    device const bfloat* w       [[buffer(2)]],
    device       half*   out     [[buffer(3)]],
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   K       [[buffer(5)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= C) return;
    const uint km1 = K - 1u;
    const float xv = float(x_in[c]);
    device float* row = state + c * km1;
    device const bfloat* taps = w + c * K;
    float acc = xv * float(taps[km1]);
    for (uint j = 0; j < km1; ++j) {
        acc = fma(row[j], float(taps[j]), acc);
    }
    out[c] = half(acc + xv);
    // Shift the state row and append this input.
    for (uint j = 0; j + 1u < km1; ++j) { row[j] = row[j + 1u]; }
    row[km1 - 1u] = xv;
}

// Chunked prefill counterpart, specialized to Inkling's fixed K=4 contract.
// One thread owns a channel and walks the token axis in causal order, keeping
// the three history values in registers. This replaces T separate dispatches
// without changing the recurrence or the FP16 store boundary.
kernel void inkling_sconv_prefill_f16out(
    device const half*   x      [[buffer(0)]],  // [T, C]
    device       float*  state  [[buffer(1)]],  // [C, 3]
    device const bfloat* w      [[buffer(2)]],  // [C, 4]
    device       half*   out    [[buffer(3)]],  // [T, C]
    constant     uint&   C      [[buffer(4)]],
    constant     uint&   T      [[buffer(5)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= C) return;
    device float* hist = state + c * 3u;
    device const bfloat* taps = w + c * 4u;
    float x0 = hist[0];
    float x1 = hist[1];
    float x2 = hist[2];
    const float w0 = float(taps[0]);
    const float w1 = float(taps[1]);
    const float w2 = float(taps[2]);
    const float w3 = float(taps[3]);
    for (uint t = 0; t < T; ++t) {
        const uint index = t * C + c;
        const float xv = float(x[index]);
        float acc = xv * w3;
        acc = fma(x0, w0, acc);
        acc = fma(x1, w1, acc);
        acc = fma(x2, w2, acc);
        out[index] = half(acc + xv);
        x0 = x1;
        x1 = x2;
        x2 = xv;
    }
    hist[0] = x0;
    hist[1] = x1;
    hist[2] = x2;
}

// The attention/MLP short-convolution sites feed the FP32 residual stream.
// Fuse their FP32 output and residual add while walking the causal token axis;
// `scale` is 1 for attention and the FFN un-prescale for the MLP tail.
kernel void inkling_sconv_prefill_residual_f32(
    device const half*   x       [[buffer(0)]],  // [T, C]
    device       float*  state   [[buffer(1)]],  // [C, 3]
    device const bfloat* w       [[buffer(2)]],  // [C, 4]
    device       float*  hidden  [[buffer(3)]],  // [T, C]
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   T       [[buffer(5)]],
    constant     float&  scale   [[buffer(6)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= C) return;
    device float* hist = state + c * 3u;
    device const bfloat* taps = w + c * 4u;
    float x0 = hist[0];
    float x1 = hist[1];
    float x2 = hist[2];
    const float w0 = float(taps[0]);
    const float w1 = float(taps[1]);
    const float w2 = float(taps[2]);
    const float w3 = float(taps[3]);
    for (uint t = 0; t < T; ++t) {
        const uint index = t * C + c;
        const float xv = float(x[index]);
        float acc = xv * w3;
        acc = fma(x0, w0, acc);
        acc = fma(x1, w1, acc);
        acc = fma(x2, w2, acc);
        hidden[index] = fma(acc + xv, scale, hidden[index]);
        x0 = x1;
        x1 = x2;
        x2 = xv;
    }
    hist[0] = x0;
    hist[1] = x1;
    hist[2] = x2;
}

// ----------------------------------------------------------------------------
// Per-head weighted RMS norm over Q [NQ, HD] and K [NKV, HD], fp32 math.
// Threadgroup per head; heads [0, NQ) are Q, [NQ, NQ+NKV) are K.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(kInklingAttnThreads)]]
void inkling_qk_norm(
    device       half*   q        [[buffer(0)]],
    device       half*   k        [[buffer(1)]],
    device const bfloat* q_weight [[buffer(2)]],   // [HD]
    device const bfloat* k_weight [[buffer(3)]],   // [HD]
    constant     uint&   head_dim [[buffer(4)]],
    constant     uint&   nq       [[buffer(5)]],
    constant     uint&   nkv      [[buffer(6)]],
    constant     float&  eps      [[buffer(7)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups    [[simdgroups_per_threadgroup]]
) {
    threadgroup float scratch[kInklingAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = head_dim;
    const bool is_q = tg < nq;
    if (!is_q && tg >= nq + nkv) return;
    device half* row = is_q ? (q + tg * HD) : (k + (tg - nq) * HD);
    device const bfloat* w = is_q ? q_weight : k_weight;

    float acc = 0.0f;
    for (uint i = lid; i < HD; i += lsize) {
        const float v = float(row[i]);
        acc = fma(v, v, acc);
    }
    float s = simd_sum(acc);
    if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(bcast / float(HD) + eps);
    for (uint i = lid; i < HD; i += lsize) {
        row[i] = half(float(row[i]) * inv * float(w[i]));
    }
}

// Batched Q/K norm: grid = [NQ + NKV, T]. The arithmetic inside each
// threadgroup is deliberately identical to `inkling_qk_norm` so batching
// changes scheduling, not the model's reduction order.
[[kernel, max_total_threads_per_threadgroup(kInklingAttnThreads)]]
void inkling_qk_norm_prefill(
    device       half*   q        [[buffer(0)]],
    device       half*   k        [[buffer(1)]],
    device const bfloat* q_weight [[buffer(2)]],
    device const bfloat* k_weight [[buffer(3)]],
    constant     uint&   head_dim [[buffer(4)]],
    constant     uint&   nq       [[buffer(5)]],
    constant     uint&   nkv      [[buffer(6)]],
    constant     uint&   rows     [[buffer(7)]],
    constant     float&  eps      [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 lid_xy  [[thread_position_in_threadgroup]],
    uint2 lsize_xy [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups    [[simdgroups_per_threadgroup]]
) {
    threadgroup float scratch[kInklingAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint lid = lid_xy.x;
    const uint lsize = lsize_xy.x;
    const uint head = tg.x;
    const uint token = tg.y;
    if (token >= rows || head >= nq + nkv) return;
    const bool is_q = head < nq;
    const uint HD = head_dim;
    device half* row = is_q
        ? (q + (token * nq + head) * HD)
        : (k + (token * nkv + head - nq) * HD);
    device const bfloat* gain = is_q ? q_weight : k_weight;

    float acc = 0.0f;
    for (uint i = lid; i < HD; i += lsize) {
        const float v = float(row[i]);
        acc = fma(v, v, acc);
    }
    float s = simd_sum(acc);
    if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float value = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        value = simd_sum(value);
        if (simd_lane_id == 0) { bcast = value; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(bcast / float(HD) + eps);
    for (uint i = lid; i < HD; i += lsize) {
        row[i] = half(float(row[i]) * inv * float(gain[i]));
    }
}

// ----------------------------------------------------------------------------
// Single-token GQA attention with inline relative-position bias.
//
//   Q    : [NQ, HD] half (post per-head RMS norm)
//   K/V  : [slots, NKV, HD] half — ring-buffered when ring_cap > 0.
//   rel  : [NQ, D_REL] half — this token's relative states (wr_du output).
//   proj : [D_REL, rel_extent] bfloat — bias-vs-distance profile bank.
//
// For kv position p: dist = q_pos - p (>= 0 by construction of [kv_start,
// seq_len)); bias = dot(rel[h], proj[:, dist]) if dist < rel_extent else 0.
// logit = tau * (scale * q·k + bias). tau folds the global-layer log-scaling
// of both q and the bias; the caller passes 1.0 for sliding layers and for
// positions below the floor.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(kInklingAttnThreads)]]
void inkling_attention_decode(
    device const half*   Q          [[buffer(0)]],
    device const half*   K          [[buffer(1)]],
    device const half*   V          [[buffer(2)]],
    device const half*   rel        [[buffer(3)]],
    device const bfloat* proj       [[buffer(4)]],
    device       half*   out        [[buffer(5)]],  // [NQ, HD]
    constant     uint&   head_dim   [[buffer(6)]],
    constant     uint&   nq         [[buffer(7)]],
    constant     uint&   nkv        [[buffer(8)]],
    constant     uint&   seq_len    [[buffer(9)]],
    constant     uint&   kv_start   [[buffer(10)]],
    constant     uint&   rel_extent [[buffer(11)]],
    constant     uint&   d_rel      [[buffer(12)]],
    constant     uint&   ring_cap   [[buffer(13)]],
    constant     float&  scale      [[buffer(14)]],
    constant     float&  tau        [[buffer(15)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups    [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kInklingMaxHeadDim];
    threadgroup float rel_smem[kInklingMaxDRel];
    threadgroup float scratch[kInklingAttnMaxSimdGroups];
    threadgroup float bcast;

    const uint HD = head_dim;
    const uint q_head = tg;
    if (q_head >= nq) return;
    const uint kv_head = q_head / (nq / nkv);
    const uint q_pos = seq_len - 1u;

    device const half* Q_row = Q + q_head * HD;
    for (uint i = lid; i < HD; i += lsize) { q_smem[i] = float(Q_row[i]); }
    device const half* rel_row = rel + q_head * d_rel;
    if (lid < d_rel) { rel_smem[lid] = float(rel_row[lid]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float o_local = 0.0f;  // one output element per thread (HD <= lsize)
    const uint my_i = lid; // element index this thread owns
    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = kv_start; p < seq_len; ++p) {
        const uint phys = (ring_cap != 0u) ? (p % ring_cap) : p;
        device const half* K_row = K + (phys * nkv + kv_head) * HD;
        device const half* V_row = V + (phys * nkv + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i], float(K_row[i]), partial);
        }
        float s = simd_sum(partial);
        if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
            float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
            t = simd_sum(t);
            if (simd_lane_id == 0) { bcast = t; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qk = bcast;

        const uint dist = q_pos - p;
        float bias = 0.0f;
        if (dist < rel_extent) {
            for (uint i = 0; i < d_rel; ++i) {
                bias = fma(rel_smem[i], float(proj[i * rel_extent + dist]), bias);
            }
        }
        const float logit = tau * fma(qk, scale, bias);

        const float m_new = max(m_run, logit);
        const float alpha = fast::exp(m_run - m_new);
        const float p_exp = fast::exp(logit - m_new);
        d_run = d_run * alpha + p_exp;
        if (my_i < HD) {
            o_local = o_local * alpha + p_exp * float(V_row[my_i]);
        }
        m_run = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device half* out_row = out + q_head * HD;
    if (my_i < HD) {
        out_row[my_i] = half((d_run > 0.0f) ? (o_local / d_run) : 0.0f);
    }
}

// Chunked causal GQA attention with Inkling's learned relative-position bias.
// Grid = [NQ, T], so all token/head threadgroups from a layer are visible to
// the GPU scheduler at once instead of being serialized behind T dispatches.
// K/V for the complete chunk must be present before this kernel is encoded.
[[kernel, max_total_threads_per_threadgroup(kInklingAttnThreads)]]
void inkling_attention_prefill(
    device const half*   Q              [[buffer(0)]],  // [T, NQ, HD]
    device const half*   K              [[buffer(1)]],  // cache
    device const half*   V              [[buffer(2)]],  // cache
    device const half*   rel            [[buffer(3)]],  // [T, NQ, D_REL]
    device const bfloat* proj           [[buffer(4)]],
    device       half*   out            [[buffer(5)]],  // [T, NQ, HD]
    constant     uint&   head_dim       [[buffer(6)]],
    constant     uint&   nq             [[buffer(7)]],
    constant     uint&   nkv            [[buffer(8)]],
    constant     uint&   start_position [[buffer(9)]],
    constant     uint&   query_count    [[buffer(10)]],
    constant     uint&   sliding_window [[buffer(11)]], // 0 = global
    constant     uint&   rel_extent     [[buffer(12)]],
    constant     uint&   d_rel          [[buffer(13)]],
    constant     uint&   ring_cap       [[buffer(14)]],
    constant     uint&   log_floor      [[buffer(15)]],
    constant     float&  scale          [[buffer(16)]],
    constant     float&  log_alpha      [[buffer(17)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 lid_xy  [[thread_position_in_threadgroup]],
    uint2 lsize_xy [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups    [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kInklingMaxHeadDim];
    threadgroup float rel_smem[kInklingMaxDRel];
    threadgroup float scratch[kInklingAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint lid = lid_xy.x;
    const uint lsize = lsize_xy.x;

    const uint q_head = tg.x;
    const uint token = tg.y;
    if (q_head >= nq || token >= query_count) return;
    const uint HD = head_dim;
    const uint kv_head = q_head / (nq / nkv);
    const uint q_pos = start_position + token;
    const uint seq_len = q_pos + 1u;
    const uint kv_start = sliding_window == 0u || seq_len <= sliding_window
        ? 0u : seq_len - sliding_window;
    float tau = 1.0f;
    if (sliding_window == 0u && log_floor > 0u && seq_len > log_floor) {
        tau += log_alpha * log(float(seq_len) / float(log_floor));
    }

    device const half* Q_row = Q + (token * nq + q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) { q_smem[i] = float(Q_row[i]); }
    device const half* rel_row = rel + (token * nq + q_head) * d_rel;
    if (lid < d_rel) { rel_smem[lid] = float(rel_row[lid]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float o_local = 0.0f;
    const uint my_i = lid;
    float m_run = -INFINITY;
    float d_run = 0.0f;
    for (uint p = kv_start; p < seq_len; ++p) {
        const uint phys = ring_cap != 0u ? p % ring_cap : p;
        device const half* K_row = K + (phys * nkv + kv_head) * HD;
        device const half* V_row = V + (phys * nkv + kv_head) * HD;
        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i], float(K_row[i]), partial);
        }
        float sum = simd_sum(partial);
        if (simd_lane_id == 0) { scratch[simd_group_id] = sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
            float value = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
            value = simd_sum(value);
            if (simd_lane_id == 0) { bcast = value; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint dist = q_pos - p;
        float bias = 0.0f;
        if (dist < rel_extent) {
            for (uint i = 0; i < d_rel; ++i) {
                bias = fma(rel_smem[i], float(proj[i * rel_extent + dist]), bias);
            }
        }
        const float logit = tau * fma(bcast, scale, bias);
        const float m_new = max(m_run, logit);
        const float alpha = fast::exp(m_run - m_new);
        const float p_exp = fast::exp(logit - m_new);
        d_run = d_run * alpha + p_exp;
        if (my_i < HD) {
            o_local = o_local * alpha + p_exp * float(V_row[my_i]);
        }
        m_run = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device half* out_row = out + (token * nq + q_head) * HD;
    if (my_i < HD) {
        out_row[my_i] = half(d_run > 0.0f ? o_local / d_run : 0.0f);
    }
}

// Batched BF16 router GEMV. This is the 2-D form of router_gemv_bf16_r4:
// four expert rows per threadgroup on X and one prompt row on Y.
[[kernel, max_total_threads_per_threadgroup(128)]]
void inkling_router_gemv_prefill(
    device const bfloat* W               [[buffer(0)]],
    device const half*   hidden          [[buffer(1)]],
    device const bfloat* effective_scale [[buffer(2)]],
    device       float*  out_logits      [[buffer(3)]],
    constant     uint&   num_outputs     [[buffer(4)]],
    constant     uint&   D               [[buffer(5)]],
    constant     uint&   rows            [[buffer(6)]],
    uint2 tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint token = tg_idx.y;
    const uint e = tg_idx.x * 4u + sg_idx;
    if (token >= rows || e >= num_outputs) return;
    device const bfloat* W_row = W + e * D;
    device const half* x_row = hidden + token * D;
    float acc = 0.0f;
    for (uint i = lane; i < D; i += 32u) {
        acc = fma(float(W_row[i]),
                  float(x_row[i]) * float(effective_scale[i]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { out_logits[token * num_outputs + e] = acc; }
}

// ----------------------------------------------------------------------------
// Sigmoid router selection over 258 logits (256 routed + 2 shared sinks).
// Single threadgroup; the scan is 256 elements so one thread does selection.
//
//   logits       : [n_routed + n_shared] float, from router_gemv_bf16_r4.
//   bias         : [n_routed] float — e_score_correction_bias; selection only.
//   global_scale : [1] float — per-layer learned scalar.
//   out_indices  : [top_k] uint
//   out_weights  : [top_k] half — routed expert weights (phase-2 layout).
//   out_gammas   : [n_shared] float — shared-expert weights.
// ----------------------------------------------------------------------------
kernel void inkling_router_select(
    device const float* logits       [[buffer(0)]],
    device const float* bias         [[buffer(1)]],
    device const float* global_scale [[buffer(2)]],
    device       uint*  out_indices  [[buffer(3)]],
    device       half*  out_weights  [[buffer(4)]],
    device       float* out_gammas   [[buffer(5)]],
    constant     uint&  n_routed     [[buffer(6)]],
    constant     uint&  n_shared     [[buffer(7)]],
    constant     uint&  top_k        [[buffer(8)]],
    constant     float& route_scale  [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    // Selection score: sigmoid(logit) + bias. Insertion into a small
    // descending top-k array, matching argpartition's set semantics (intra-set
    // order is irrelevant downstream).
    uint  top_idx[8];
    float top_score[8];
    const uint KK = min(top_k, 8u);
    for (uint i = 0; i < KK; ++i) { top_idx[i] = 0u; top_score[i] = -INFINITY; }
    for (uint e = 0; e < n_routed; ++e) {
        const float s = 1.0f / (1.0f + fast::exp(-logits[e])) + bias[e];
        if (s <= top_score[KK - 1u]) continue;
        uint pos = KK;
        while (pos > 0u && top_score[pos - 1u] < s) { --pos; }
        for (uint j = KK - 1u; j > pos; --j) {
            top_score[j] = top_score[j - 1u];
            top_idx[j]   = top_idx[j - 1u];
        }
        top_score[pos] = s;
        top_idx[pos]   = e;
    }

    // Weights: softmax over logsigmoid of the RAW logits of the selected
    // experts plus the shared sinks, then route_scale * global_scale.
    // logsigmoid(x) = -log(1 + e^{-x}) = -log1p(exp(-x)); use the stable form
    // min(x, 0) - log1p(exp(-|x|)).
    float lp[10];
    const uint total = KK + n_shared;
    for (uint i = 0; i < total; ++i) {
        const float x = (i < KK) ? logits[top_idx[i]] : logits[n_routed + (i - KK)];
        lp[i] = min(x, 0.0f) - log(1.0f + fast::exp(-fabs(x)));
    }
    float mx = -INFINITY;
    for (uint i = 0; i < total; ++i) { mx = max(mx, lp[i]); }
    float denom = 0.0f;
    for (uint i = 0; i < total; ++i) { denom += fast::exp(lp[i] - mx); }
    const float s_all = route_scale * global_scale[0] / denom;
    for (uint i = 0; i < KK; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = half(fast::exp(lp[i] - mx) * s_all);
    }
    for (uint s = 0; s < n_shared; ++s) {
        out_gammas[s] = fast::exp(lp[KK + s] - mx) * s_all;
    }
}

// Batched selector for logits produced by `inkling_router_gemv_prefill`.
// One threadgroup owns one token row. The lane-0 serial scan intentionally
// mirrors `inkling_router_select` exactly; the expensive GEMV is parallelized
// across all prompt rows while selection keeps its established tie behavior.
kernel void inkling_router_select_prefill(
    device const float* logits       [[buffer(0)]], // [T, routed + shared]
    device const float* bias         [[buffer(1)]],
    device const float* global_scale [[buffer(2)]],
    device       uint*  out_indices  [[buffer(3)]], // [T, 8]
    device       half*  out_weights  [[buffer(4)]], // [T, 8]
    device       float* out_gammas   [[buffer(5)]], // [T, shared]
    constant     uint&  n_routed     [[buffer(6)]],
    constant     uint&  n_shared     [[buffer(7)]],
    constant     uint&  top_k        [[buffer(8)]],
    constant     float& route_scale  [[buffer(9)]],
    constant     uint&  rows         [[buffer(10)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0 || token >= rows) return;
    const uint num_outputs = n_routed + n_shared;
    device const float* row_logits = logits + token * num_outputs;
    device uint* row_indices = out_indices + token * 8u;
    device half* row_weights = out_weights + token * 8u;
    device float* row_gammas = out_gammas + token * n_shared;

    uint top_idx[8];
    float top_score[8];
    const uint KK = min(top_k, 8u);
    for (uint i = 0; i < KK; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
    }
    for (uint e = 0; e < n_routed; ++e) {
        const float s = 1.0f / (1.0f + fast::exp(-row_logits[e])) + bias[e];
        if (s <= top_score[KK - 1u]) continue;
        uint pos = KK;
        while (pos > 0u && top_score[pos - 1u] < s) { --pos; }
        for (uint j = KK - 1u; j > pos; --j) {
            top_score[j] = top_score[j - 1u];
            top_idx[j] = top_idx[j - 1u];
        }
        top_score[pos] = s;
        top_idx[pos] = e;
    }

    float lp[10];
    const uint total = KK + n_shared;
    for (uint i = 0; i < total; ++i) {
        const float value = i < KK
            ? row_logits[top_idx[i]] : row_logits[n_routed + i - KK];
        lp[i] = min(value, 0.0f) - log(1.0f + fast::exp(-fabs(value)));
    }
    float mx = -INFINITY;
    for (uint i = 0; i < total; ++i) { mx = max(mx, lp[i]); }
    float denom = 0.0f;
    for (uint i = 0; i < total; ++i) { denom += fast::exp(lp[i] - mx); }
    const float all_scale = route_scale * global_scale[0] / denom;
    for (uint i = 0; i < KK; ++i) {
        row_indices[i] = top_idx[i];
        row_weights[i] = half(fast::exp(lp[i] - mx) * all_scale);
    }
    for (uint s = 0; s < n_shared; ++s) {
        row_gammas[s] = fast::exp(lp[KK + s] - mx) * all_scale;
    }
}

// ----------------------------------------------------------------------------
// y = gammas[0]*a + gammas[1]*b  (shared-expert combine, fp32 accumulate).
// ----------------------------------------------------------------------------
kernel void inkling_gamma_combine(
    device const half*  a      [[buffer(0)]],
    device const half*  b      [[buffer(1)]],
    device const float* gammas [[buffer(2)]],
    device       half*  y      [[buffer(3)]],
    constant     uint&  count  [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    y[i] = half(gammas[0] * float(a[i]) + gammas[1] * float(b[i]));
}

// FP32-input gamma combine. The two shared-expert down projections write FP32
// rows (their raw magnitudes leave FP16 range — see
// `inkling_scale_accum_f32_from_f32`); the gammas carry the 1/32 FFN prescale,
// so the combined output is back inside FP16 range with room to spare.
kernel void inkling_gamma_combine_f32in(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device const float* gammas [[buffer(2)]],
    device       half*  y      [[buffer(3)]],
    constant     uint&  count  [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    y[i] = half(gammas[0] * a[i] + gammas[1] * b[i]);
}

// ----------------------------------------------------------------------------
// FP32 residual-stream helpers. Inkling's residual grows past 55 000 by layer
// 22 (BF16-trained, muP-style outlier channels) and overflows an FP16 stream
// at layer 23; the hidden state therefore lives in FP32 for this family.
// Every consumer is an RMS norm, so only these three ops touch the stream.
// ----------------------------------------------------------------------------
kernel void inkling_f16_to_f32(
    device const half*  src   [[buffer(0)]],
    device       float* dst   [[buffer(1)]],
    constant     uint&  count [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    dst[i] = float(src[i]);
}

// Same taps/state semantics as inkling_sconv_step, but FP32 output: the
// sublayer-output convolutions (attn/mlp sites) feed deltas that exceed the
// FP16 range deep in the stack (~3.5x per-channel gain on outlier channels).
kernel void inkling_sconv_step_f32out(
    device const half*   x_in    [[buffer(0)]],
    device       float*  state   [[buffer(1)]],
    device const bfloat* w       [[buffer(2)]],
    device       float*  out     [[buffer(3)]],
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   K       [[buffer(5)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= C) return;
    const uint km1 = K - 1u;
    const float xv = float(x_in[c]);
    device float* row = state + c * km1;
    device const bfloat* taps = w + c * K;
    float acc = xv * float(taps[km1]);
    for (uint j = 0; j < km1; ++j) {
        acc = fma(row[j], float(taps[j]), acc);
    }
    out[c] = acc + xv;
    for (uint j = 0; j + 1u < km1; ++j) { row[j] = row[j + 1u]; }
    row[km1 - 1u] = xv;
}

// `scale` un-folds the FFN output pre-scale (see the runner: router weights
// and shared gammas are divided by kInklingFFNPrescale so the FP16 MoE
// intermediates stay inside half range on BF16-magnitude channels; the
// sconv-with-residual is linear, so scaling back here is exact).
kernel void inkling_residual_add_f32d(
    device       float* hidden [[buffer(0)]],
    device const float* delta  [[buffer(1)]],
    constant     uint&  count  [[buffer(2)]],
    constant     float& scale  [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    hidden[i] = fma(delta[i], scale, hidden[i]);
}

kernel void inkling_residual_add_f32(
    device       float* hidden [[buffer(0)]],
    device const half*  delta  [[buffer(1)]],
    constant     uint&  count  [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    hidden[i] += float(delta[i]);
}

// RMS norm over an FP32 input row with BF16 gains, FP16 output (feeds the
// INT4/INT8 GEMVs). One threadgroup, fp32 accumulation.
[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling_rms_f32in(
    device const float*  x      [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device       half*   out    [[buffer(2)]],
    constant     uint&   d      [[buffer(3)]],
    constant     float&  eps    [[buffer(4)]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups    [[simdgroups_per_threadgroup]]
) {
    threadgroup float scratch[8];
    threadgroup float bcast;
    float acc = 0.0f;
    for (uint i = lid; i < d; i += lsize) {
        const float v = x[i];
        acc = fma(v, v, acc);
    }
    float ssum = simd_sum(acc);
    if (simd_lane_id == 0) { scratch[simd_group_id] = ssum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(bcast / float(d) + eps);
    for (uint i = lid; i < d; i += lsize) {
        out[i] = half(x[i] * inv * float(weight[i]));
    }
}

// Batched FP32-residual RMSNorm. Grid X is the token row; each threadgroup
// uses the same 256-thread reduction as `inkling_rms_f32in` and writes one
// contiguous FP16 row for the following matrix projections.
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

// ----------------------------------------------------------------------------
// Logits epilogue: i < valid → x[i] *= c (muP 1/16); i >= valid → -INF so the
// padding rows of the 201 024-wide unembed can never be sampled (the real
// vocabulary is 200 058).
// ----------------------------------------------------------------------------
// `bad` counts non-finite real-vocabulary logits. A NaN or infinity here is
// always an engine fault upstream, and it used to be *invisible*: the CPU
// argmax seeds its running best at (index 0, -infinity), every `v > best`
// comparison against NaN is false, so an all-NaN row silently returned token
// id 0 — which decodes to "!". Counting them lets the caller fail loudly
// instead of emitting a plausible-looking exclamation mark.
kernel void inkling_head_epilogue(
    device       half*         x     [[buffer(0)]],
    constant     float&        c     [[buffer(1)]],
    constant     uint&         valid [[buffer(2)]],
    constant     uint&         total [[buffer(3)]],
    device       atomic_uint*  bad   [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= total) return;
    if (i < valid) {
        const float v = float(x[i]) * c;
        if (!isfinite(v)) {
            atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
        }
        x[i] = half(v);
    } else {
        x[i] = half(-INFINITY);
    }
}

// ----------------------------------------------------------------------------
// acc(f32) += w * x(f16) — expert-major prefill accumulation: each streamed
// expert's GLU output is folded into the per-token FFN accumulator with its
// routing weight.
// ----------------------------------------------------------------------------
kernel void inkling_scale_accum_f32(
    device       float* acc   [[buffer(0)]],
    device const half*  x     [[buffer(1)]],
    constant     float& w     [[buffer(2)]],
    constant     uint&  count [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    acc[i] = fma(float(x[i]), w, acc[i]);
}

// As above, but the expert output is already FP32. The GLU down projection
// writes FP32 for this family because a single Inkling channel (layer 41,
// channel 3895 on the released checkpoint) runs at 1e4-6e4 *before* its
// routing weight or shared gamma is applied, so an FP16 store of the raw row
// clips to infinity on the tokens that push it over 65 504.
kernel void inkling_scale_accum_f32_from_f32(
    device       float* acc   [[buffer(0)]],
    device const float* x     [[buffer(1)]],
    constant     float& w     [[buffer(2)]],
    constant     uint&  count [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    acc[i] = fma(x[i], w, acc[i]);
}

// f32 -> f16 narrowing copy (per-token staging of chunk vectors into the
// fixed GEMV scratch).
kernel void inkling_f32_to_f16(
    device const float* src   [[buffer(0)]],
    device       half*  dst   [[buffer(1)]],
    constant     uint&  count [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    dst[i] = half(src[i]);
}

// ----------------------------------------------------------------------------
// In-place x *= c over half elements (muP logit scaling, dense global_scale).
// ----------------------------------------------------------------------------
kernel void inkling_scale_f16(
    device       half*  x     [[buffer(0)]],
    constant     float& c     [[buffer(1)]],
    constant     uint&  count [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    x[i] = half(float(x[i]) * c);
}
