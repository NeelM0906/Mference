#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention — split-KV tiled softmax attention for single-token decode.
//
// Decode path only: M_q = 1 (one query token), arbitrary seq_len history.
// The MPP prefill path handles M_q > 1 separately.
//
// Layout (caller-side contract):
//   Q   : [num_q_heads,  head_dim]                      FP16, contiguous.
//   K   : [seq_len, num_kv_heads, head_dim]             FP16, contiguous.
//   V   : [seq_len, num_kv_heads, head_dim]             FP16, same shape as K.
//         Full attention reuses the raw K projection for V, but its separate
//         normalization and RoPE paths make these buffers distinct here.
//   out : [num_q_heads,  head_dim]                      FP16.
//
// GQA: q_head -> kv_head = q_head / (num_q_heads / num_kv_heads).
//      Multiple Q heads share one KV head; the dispatch indexes Q heads.
//
// Online softmax recurrence (FP32 accumulators) — Milakov & Gimelshein 2018,
// also FlashAttention:
//   m_new   = max(m, s)
//   alpha   = exp(m - m_new)                 // rescale factor for past state
//   d       = d * alpha + exp(s - m_new)
//   o[i]    = o[i] * alpha + exp(s - m_new) * V[p, i]
//   m       = m_new
// Final normalization: out[i] = o[i] / d.
//
// ============================================================================

constant constexpr uint kAttnThreads      = 256;
// kAttnMaxSimdGroups must cover kAttnThreads / 32 = 8.
constant constexpr uint kAttnMaxSimdGroups = 8;
constant constexpr uint kAttnMaxQPerKV     = 2;
constant constexpr uint kAttnMaxFullQPerKV = 8;
constant constexpr uint kAttnFullQPerThreadgroup = 2;
// Largest head_dim we run with (full-attention layers). SWA uses 256 — the
// kernel still allocates the 512-slot scratch but only touches the live half.
constant constexpr uint kAttnMaxHeadDim   = 512;
constant uint FC_ATTN_HEAD_DIM [[function_constant(60)]];
constant uint FC_ATTN_NUM_Q_HEADS [[function_constant(61)]];
constant uint FC_ATTN_NUM_KV_HEADS [[function_constant(62)]];
constant bool FC_ATTN_USE_FC [[function_constant(63)]];
constant float FC_ATTN_SCALE [[function_constant(64)]];
constant uint FC_ATTN_NUM_CHUNKS [[function_constant(65)]];
constant uint FC_ATTN_RING_CAP [[function_constant(69)]];

static inline uint attn_fc_head_dim(constant uint& head_dim) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_HEAD_DIM))
        ? FC_ATTN_HEAD_DIM
        : head_dim;
}
static inline uint attn_fc_num_q_heads(constant uint& num_q_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_Q_HEADS))
        ? FC_ATTN_NUM_Q_HEADS
        : num_q_heads;
}

static inline uint attn_fc_num_kv_heads(constant uint& num_kv_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_KV_HEADS))
        ? FC_ATTN_NUM_KV_HEADS
        : num_kv_heads;
}

static inline float attn_fc_scale(float scale) {
    return is_function_constant_defined(FC_ATTN_SCALE) ? FC_ATTN_SCALE : scale;
}

static inline uint attn_fc_num_chunks(constant uint& num_chunks) {
    return is_function_constant_defined(FC_ATTN_NUM_CHUNKS) ? FC_ATTN_NUM_CHUNKS : num_chunks;
}

static inline uint attn_ring_slot(uint p) {
    return (is_function_constant_defined(FC_ATTN_RING_CAP) &&
            FC_ATTN_RING_CAP != 0u)
        ? (p % FC_ATTN_RING_CAP)
        : p;
}

static inline float attn_softmax_exp(float x) {
    return fast::exp(x);
}

// Block reduce: per-SIMD-group simd_sum, write partial to scratch, lane 0 of
// SIMD-group 0 finishes the merge with a second simd_sum and broadcasts.
// `scratch` must hold at least `simdgroups` floats; `bcast` is one float used
// to publish the final reduced value to all threads.
inline float block_reduce_sum(float v,
                              uint simd_lane_id,
                              uint simd_group_id,
                              uint simdgroups,
                              threadgroup float* scratch,
                              threadgroup float* bcast) {
    float s = simd_sum(v);
    if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { *bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *bcast;
}


// ============================================================================
// Split-KV (Flash-Decoding) decode attention — the default path.
//

// Pass 1 (attention_decode_partial): grid = num_q_heads * num_chunks. Each TG
//   runs the same online-softmax recurrence over its chunk [p_start, p_end) and
//   writes the UN-normalized partial state (m_chunk, d_chunk, o_chunk[head_dim])
//   to scratch — no division yet.
// Pass 2 (attention_decode_combine): grid = num_q_heads. Each TG merges its
//   head's num_chunks partials with the standard online-softmax rescale
//   (m_glob = max_c m_c; D = Σ d_c·e^{m_c−m_glob}; O = Σ o_c·e^{m_c−m_glob}) and
//   writes out[i] = O[i] / D in FP16.
//
// At num_chunks == 1 the chunk spans the whole [kv_start, seq_len) range and
// the partial is the exact single-pass accumulation; the combine's only chunk
// has m_glob == m_chunk so e^0 == 1 and out == o/d — byte-identical to the
// single-pass kernels above. num_chunks > 1 changes the FP rounding of the
// partial sums only (same position summation order), not the algorithm.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    // p_start can land past the end when num_chunks > range length (the tail
    // chunks are empty); the loop simply does not execute and the partial is
    // (-inf, 0, 0), which the combine weights to zero via e^{-inf}.
    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i], float(K_row[i]), partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            o_local[slot] = o_local[slot] * alpha + p_exp * float(V_row[i]);
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// ============================================================================
// Paged decode attention — split-KV partial over a page-table selection.
//
// The KV cache lives in per-layer page pools (KVPageStore): fixed 64-token
// pages at arbitrary pool slots. `page_table[i]` is the pool slot of the
// i-th *selected* page, in ascending logical-position order; `sel_tokens`
// counts the selected logical tokens (the last listed page may be a partial
// tail). Softmax runs over exactly the selected subset — the sparse
// (Quest-style) decode path. With an identity table and a full selection the
// accumulation order matches `attention_decode_partial` bit for bit.
// Combine pass is shared (`attention_decode_combine`).
// ============================================================================

constant constexpr uint kAttnPageTokens = 64;

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_paged_partial(
    device const half*   Q             [[buffer(0)]],
    device const half*   K_pool        [[buffer(1)]],
    device const half*   V_pool        [[buffer(2)]],
    device       float*  m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float*  d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float*  o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&   head_dim      [[buffer(6)]],
    constant     uint&   num_q_heads   [[buffer(7)]],
    constant     uint&   num_kv_heads  [[buffer(8)]],
    constant     uint&   sel_tokens    [[buffer(9)]],
    device const uint*   page_table    [[buffer(10)]],
    constant     uint&   chunk_len     [[buffer(11)]],
    constant     uint&   num_chunks    [[buffer(12)]],
    constant     float&  scale         [[buffer(13)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint l_start = chunk * chunk_len;
    uint l_end = l_start + chunk_len;
    if (l_end > sel_tokens) { l_end = sel_tokens; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint l = l_start; l < l_end; ++l) {
        const uint slot = page_table[l / kAttnPageTokens];
        const uint phys_p = slot * kAttnPageTokens + (l % kAttnPageTokens);
        device const half* K_row = K_pool + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V_pool + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i], float(K_row[i]), partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot_i = 0;
        for (uint i = lid; i < HD; i += lsize) {
            o_local[slot_i] = o_local[slot_i] * alpha + p_exp * float(V_row[i]);
            slot_i += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot_i = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot_i];
        slot_i += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_gqa_swa_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxQPerKV * kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxQPerKV * kAttnMaxSimdGroups];
    threadgroup float bcast[kAttnMaxQPerKV];
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_per_kv = NQ / NKV;
    if (q_per_kv > kAttnMaxQPerKV) { return; }

    const uint kv_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint q_base = kv_head * q_per_kv;
    for (uint qg = 0; qg < q_per_kv; ++qg) {
        device const half* Q_row = Q + (q_base + qg) * HD;
        threadgroup float* Q_s = q_smem + qg * kAttnMaxHeadDim;
        for (uint i = lid; i < HD; i += lsize) {
            Q_s[i] = float(Q_row[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint groups_per_q = max(1u, simdgroups / q_per_kv);
    const uint active_q = min(q_per_kv - 1u, simd_group_id / groups_per_q);
    const uint local_group = simd_group_id - active_q * groups_per_q;
    const uint threads_per_q = groups_per_q * 32u;
    const uint local_lid = local_group * 32u + simd_lane_id;

    constexpr uint kGQAPerThread =
        (kAttnMaxHeadDim + (kAttnThreads / kAttnMaxQPerKV) - 1) /
        (kAttnThreads / kAttnMaxQPerKV);
    float o_local[kGQAPerThread];
    for (uint k = 0; k < kGQAPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            const float k_val = float(K_row[i]);
            partial = fma(q_smem[active_q * kAttnMaxHeadDim + i], k_val, partial);
        }

        float s = simd_sum(partial);
        if (simd_lane_id == 0) {
            reduce_scratch[active_q * kAttnMaxSimdGroups + local_group] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_group == 0) {
            float t = (simd_lane_id < groups_per_q)
                ? reduce_scratch[active_q * kAttnMaxSimdGroups + simd_lane_id]
                : 0.0f;
            t = simd_sum(t);
            if (simd_lane_id == 0) { bcast[active_q] = t; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = bcast[active_q] * attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        for (uint slot = 0; slot < kGQAPerThread; ++slot) { o_local[slot] *= alpha; }
        m_run = m_new;

        uint slot = 0;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            o_local[slot] += p_exp * float(V_row[i]);
            slot += 1;
        }
    }

    const uint q_head = q_base + active_q;
    const uint base = uint(q_head) * NC + chunk;
    if (local_lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = local_lid; i < HD; i += threads_per_q) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// ============================================================================
// Blocked (streamed) prefill attention — beyond-RAM contexts.
//
// A prefill chunk's queries attend the whole past, which no longer fits the
// pool. The past streams through staging windows; each window dispatch folds
// its positions into per-(query, q-head) running online-softmax state
// (m, d, o[head_dim], FP32 in device memory), and a finalize pass writes
// out = o/d in FP16. The chunk's own pages run as a final window with the
// causal predicate p <= q_pos.
//
// Geometry: one simdgroup per (query token, q head) — 32 lanes split the
// head_dim, positions run sequentially with simd-level reduction only (no
// threadgroup barriers). K/V rows resolve through a page table like the
// paged decode kernel; staging passes an identity table.
// ============================================================================

constant constexpr uint kFlashSimdgroupsPerTG = 8;

[[kernel]]
void attention_prefill_flash_init(
    device float* m_state [[buffer(0)]],   // [query_count * num_q_heads]
    device float* d_state [[buffer(1)]],
    device float* o_state [[buffer(2)]],   // [query_count * num_q_heads * head_dim]
    constant uint& rows   [[buffer(3)]],   // query_count * num_q_heads
    constant uint& head_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < rows) {
        m_state[gid] = -INFINITY;
        d_state[gid] = 0.0f;
    }
    const uint total = rows * head_dim;
    for (uint i = gid; i < total; i += rows) {
        // rows threads stride the o_state clear; grid is sized to `rows`.
        o_state[i] = 0.0f;
    }
}

[[kernel, max_total_threads_per_threadgroup(kFlashSimdgroupsPerTG * 32)]]
void attention_prefill_flash_update(
    device const half*  Q             [[buffer(0)]],   // [query_count, q_stride] FP16
    device const half*  K_pool        [[buffer(1)]],
    device const half*  V_pool        [[buffer(2)]],
    device const uint*  page_table    [[buffer(3)]],   // window pages -> pool/staging slots
    device       float* m_state       [[buffer(4)]],
    device       float* d_state       [[buffer(5)]],
    device       float* o_state       [[buffer(6)]],
    constant     uint&  query_count   [[buffer(7)]],
    constant     uint&  q_start       [[buffer(8)]],   // global position of query row 0
    constant     uint&  head_dim      [[buffer(9)]],
    constant     uint&  num_q_heads   [[buffer(10)]],
    constant     uint&  num_kv_heads  [[buffer(11)]],
    constant     uint&  win_start     [[buffer(12)]],  // global position of window token 0 (page-aligned)
    constant     uint&  win_tokens    [[buffer(13)]],
    constant     uint&  q_stride      [[buffer(14)]],  // elements per query row
    constant     float& scale         [[buffer(15)]],
    constant     uint&  causal        [[buffer(16)]],  // 1: apply p <= q_pos within the window
    uint tg_id           [[threadgroup_position_in_grid]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]]
) {
    const uint row = tg_id * kFlashSimdgroupsPerTG + simd_group_id;
    const uint total_rows = query_count * num_q_heads;
    if (row >= total_rows) { return; }
    const uint t = row / num_q_heads;
    const uint q_head = row % num_q_heads;
    const uint kv_head = q_head / (num_q_heads / num_kv_heads);
    const uint q_pos = q_start + t;

    // Effective window span for this query under the causal predicate.
    uint span = win_tokens;
    if (causal != 0u) {
        if (q_pos + 1 <= win_start) { return; }
        span = min(span, q_pos + 1 - win_start);
    }
    if (span == 0u) { return; }

    device const half* Q_row = Q + t * q_stride + q_head * head_dim;

    // Lane-strided registers: head_dim <= 512 -> at most 16 elems per lane.
    constexpr uint kMaxPerLane = kAttnMaxHeadDim / 32;
    float q_reg[kMaxPerLane];
    const uint per_lane = (head_dim + 31) / 32;
    for (uint k = 0; k < per_lane; ++k) {
        const uint i = simd_lane_id + k * 32;
        q_reg[k] = i < head_dim ? float(Q_row[i]) : 0.0f;
    }

    float m_run = m_state[row];
    float d_run = d_state[row];
    device float* o_row = o_state + row * head_dim;
    float o_reg[kMaxPerLane];
    for (uint k = 0; k < per_lane; ++k) {
        const uint i = simd_lane_id + k * 32;
        o_reg[k] = i < head_dim ? o_row[i] : 0.0f;
    }

    const uint elems = num_kv_heads * head_dim;
    for (uint w = 0; w < span; ++w) {
        const uint slot = page_table[w / kAttnPageTokens];
        const uint phys = slot * kAttnPageTokens + (w % kAttnPageTokens);
        device const half* K_row = K_pool + phys * elems + kv_head * head_dim;
        device const half* V_row = V_pool + phys * elems + kv_head * head_dim;

        float partial = 0.0f;
        for (uint k = 0; k < per_lane; ++k) {
            const uint i = simd_lane_id + k * 32;
            if (i < head_dim) { partial = fma(q_reg[k], float(K_row[i]), partial); }
        }
        const float s = simd_sum(partial) * scale;

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        for (uint k = 0; k < per_lane; ++k) {
            const uint i = simd_lane_id + k * 32;
            if (i < head_dim) {
                o_reg[k] = o_reg[k] * alpha + p_exp * float(V_row[i]);
            }
        }
        m_run = m_new;
    }

    if (simd_lane_id == 0) { m_state[row] = m_run; d_state[row] = d_run; }
    for (uint k = 0; k < per_lane; ++k) {
        const uint i = simd_lane_id + k * 32;
        if (i < head_dim) { o_row[i] = o_reg[k]; }
    }
}

[[kernel]]
void attention_prefill_flash_finalize(
    device const float* m_state   [[buffer(0)]],
    device const float* d_state   [[buffer(1)]],
    device const float* o_state   [[buffer(2)]],
    device       half*  out       [[buffer(3)]],   // [query_count, o_stride]
    constant     uint&  query_count [[buffer(4)]],
    constant     uint&  head_dim  [[buffer(5)]],
    constant     uint&  num_q_heads [[buffer(6)]],
    constant     uint&  o_stride  [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = query_count * num_q_heads * head_dim;
    if (gid >= total) { return; }
    const uint row = gid / head_dim;
    const uint i = gid % head_dim;
    const uint t = row / num_q_heads;
    const uint q_head = row % num_q_heads;
    const float d = d_state[row];
    const float v = d > 0.0f ? o_state[row * head_dim + i] / d : 0.0f;
    out[t * o_stride + q_head * head_dim + i] = half(v);
}

// ============================================================================
// KV page maintenance kernels (paged long-context path).
// ============================================================================

// Element-wise min/max over a sealed page's K rows — the Quest (arXiv
// 2406.10774) page summary used to estimate a page's attention criticality
// without reading it. One threadgroup; runs once per page seal.
// `metadata` points at the page's metadata slot: min[NKV*HD] then max[NKV*HD].
[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void kv_page_minmax(
    device const half*  K_pool        [[buffer(0)]],
    device       half*  metadata      [[buffer(1)]],
    constant     uint&  slot          [[buffer(2)]],
    constant     uint&  valid_tokens  [[buffer(3)]],
    constant     uint&  num_kv_heads  [[buffer(4)]],
    constant     uint&  head_dim      [[buffer(5)]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    const uint elems = num_kv_heads * head_dim;
    const uint base = slot * kAttnPageTokens * elems;
    for (uint e = lid; e < elems; e += lsize) {
        float mn = INFINITY;
        float mx = -INFINITY;
        for (uint t = 0; t < valid_tokens; ++t) {
            const float v = float(K_pool[base + t * elems + e]);
            mn = min(mn, v);
            mx = max(mx, v);
        }
        metadata[e] = half(mn);
        metadata[elems + e] = half(mx);
    }
}

// Quest page criticality: for each page, per q-head upper bound of q·k over
// the page — sum_d max(q_d·min_d, q_d·max_d) against the head's kv-head
// min/max summary — reduced with max over q heads. One threadgroup per page;
// `metadata` points at the layer's metadata base.
[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_page_scores(
    device const half*  Q             [[buffer(0)]],   // [num_q_heads, head_dim]
    device const half*  metadata      [[buffer(1)]],
    device       float* scores        [[buffer(2)]],   // [num_pages]
    constant     uint&  num_pages     [[buffer(3)]],
    constant     uint&  head_dim      [[buffer(4)]],
    constant     uint&  num_q_heads   [[buffer(5)]],
    constant     uint&  num_kv_heads  [[buffer(6)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint page = tg_id;
    if (page >= num_pages) { return; }

    const uint elems = num_kv_heads * head_dim;
    device const half* mn = metadata + page * 2 * elems;
    device const half* mx = mn + elems;

    const uint q_per_kv = num_q_heads / num_kv_heads;
    float best = -INFINITY;
    for (uint qh = 0; qh < num_q_heads; ++qh) {
        const uint kvh = qh / q_per_kv;
        device const half* Q_row = Q + qh * head_dim;
        device const half* lo = mn + kvh * head_dim;
        device const half* hi = mx + kvh * head_dim;
        float partial = 0.0f;
        for (uint i = lid; i < head_dim; i += lsize) {
            const float q = float(Q_row[i]);
            partial += max(q * float(lo[i]), q * float(hi[i]));
        }
        const float s = block_reduce_sum(partial,
                                         simd_lane_id, simd_group_id, simdgroups,
                                         reduce_scratch, &bcast);
        best = max(best, s);
    }
    if (lid == 0) { scores[page] = best; }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_combine(
    device const float* m_in         [[buffer(0)]],    // [num_q_heads * num_chunks]
    device const float* d_in         [[buffer(1)]],
    device const float* o_in         [[buffer(2)]],    // [num_q_heads * num_chunks * head_dim]
    device       half*  out          [[buffer(3)]],    // [num_q_heads * head_dim]
    constant     uint&  head_dim     [[buffer(4)]],
    constant     uint&  num_chunks   [[buffer(5)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]]
) {
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint q_head = tg_id;
    device const float* m_row  = m_in + uint(q_head) * NC;
    device const float* d_row  = d_in + uint(q_head) * NC;
    device const float* o_base = o_in + uint(q_head) * NC * HD;

    // num_chunks is small (<= a few dozen); each thread recomputes the global
    // max and denominator rather than pay a threadgroup reduction + barriers.
    float m_glob = -INFINITY;
    for (uint c = 0; c < NC; ++c) { m_glob = max(m_glob, m_row[c]); }
    float D = 0.0f;
    for (uint c = 0; c < NC; ++c) { D += d_row[c] * attn_softmax_exp(m_row[c] - m_glob); }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    device half* out_row = out + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) {
            acc += o_base[c * HD + i] * attn_softmax_exp(m_row[c] - m_glob);
        }
        out_row[i] = half(acc * inv_d);
    }
}
