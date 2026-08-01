#include <metal_stdlib>
using namespace metal;

// DeepSeek-V4 decode kernels: shared-KV MQA attention with per-head sinks
// over [sliding-window ring ‖ compressed entries], the CSA/HCA softmax-gated
// window compressor, the lightning-indexer scorer, interleaved trailing
// partial RoPE, Manifold-Constrained Hyper-Connection stream mixing, and the
// clamped-SwiGLU elementwise for the shared expert.
//
// Function-constant namespace: 100–109.

constant uint FC_DSV4_HEAD_DIM [[function_constant(100)]];     // 512
constant uint FC_DSV4_NUM_HEADS [[function_constant(101)]];    // 64
constant uint FC_DSV4_ROPE_DIM [[function_constant(102)]];     // 64
constant uint FC_DSV4_HC_MULT [[function_constant(103)]];      // 4
constant uint FC_DSV4_HIDDEN [[function_constant(104)]];       // 4096
constant bool FC_DSV4_USE_FC [[function_constant(105)]];

static inline uint dsv4_head_dim(constant uint& v) {
    return (is_function_constant_defined(FC_DSV4_USE_FC) && FC_DSV4_USE_FC &&
            is_function_constant_defined(FC_DSV4_HEAD_DIM)) ? FC_DSV4_HEAD_DIM : v;
}
static inline uint dsv4_rope_dim(constant uint& v) {
    return (is_function_constant_defined(FC_DSV4_USE_FC) && FC_DSV4_USE_FC &&
            is_function_constant_defined(FC_DSV4_ROPE_DIM)) ? FC_DSV4_ROPE_DIM : v;
}

// Interleaved partial RoPE on the trailing `rope_dim` channels of each of
// `num_heads` contiguous heads: pair (2i, 2i+1) rotates by angle
// pos / theta^(2i / rope_dim). `direction` is +1 for the forward rotation and
// -1 for the conjugate (applied to the attention output at the query
// position, undoing the rotation V carried because K == V).
//
// Grid: num_heads threadgroups, >= rope_dim/2 threads each.
kernel void dsv4_rope_interleaved_trailing(
    device half* x [[buffer(0)]],
    constant uint& head_dim_in [[buffer(1)]],
    constant uint& rope_dim_in [[buffer(2)]],
    constant uint& position [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant float& direction [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint head_dim = dsv4_head_dim(head_dim_in);
    const uint rope_dim = dsv4_rope_dim(rope_dim_in);
    const uint half_dim = rope_dim / 2u;
    if (tid >= half_dim) return;
    const uint base = head * head_dim + (head_dim - rope_dim);
    const float freq = float(position)
        * pow(theta, -2.0f * float(tid) / float(rope_dim));
    const float c = cos(freq);
    const float s = sin(freq) * direction;
    const float x0 = float(x[base + 2u * tid]);
    const float x1 = float(x[base + 2u * tid + 1u]);
    x[base + 2u * tid] = half(x0 * c - x1 * s);
    x[base + 2u * tid + 1u] = half(x1 * c + x0 * s);
}

// Softmax attention for one token: 64 query heads over one shared K=V head.
// KV rows come from two regions: the sliding-window ring (`window_count`
// valid rows of `ring_capacity`, slot = absolute_position % ring_capacity)
// and the compressed-entry cache. `selected` restricts the compressed region
// to `selected_count` entries (lightning-indexer output); `selected_count ==
// 0xFFFFFFFF` means "all compressed entries". Every head folds its learnable
// sink logit into the softmax denominator (the sink contributes no value).
//
// Grid: num_heads threadgroups x 256 threads. Threadgroup memory holds the
// logits (window + compressed <= kDSV4MaxEntries).
constant constexpr uint kDSV4MaxEntries = 2176;   // 128 window + 2048 compressed
constant constexpr uint kDSV4Threads = 256;

kernel void dsv4_attention_decode(
    device const half* q [[buffer(0)]],                 // [H, head_dim], roped
    device const half* window_kv [[buffer(1)]],         // ring [capacity, head_dim]
    device const half* compressed_kv [[buffer(2)]],     // [T, head_dim]
    device const uint* selected [[buffer(3)]],          // indexer picks
    device const float* sinks [[buffer(4)]],            // [H]
    device half* out [[buffer(5)]],                     // [H, head_dim]
    constant uint& head_dim_in [[buffer(6)]],
    constant uint& window_count [[buffer(7)]],
    constant uint& window_start_pos [[buffer(8)]],      // abs pos of oldest row
    constant uint& ring_capacity [[buffer(9)]],
    constant uint& compressed_count [[buffer(10)]],
    constant uint& selected_count [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float logits[kDSV4MaxEntries];
    threadgroup float red[kDSV4Threads / 32];
    const uint head_dim = dsv4_head_dim(head_dim_in);
    const bool use_all = selected_count == 0xFFFFFFFFu;
    const uint comp_used = use_all
        ? compressed_count
        : min(selected_count, compressed_count);
    const uint total = window_count + comp_used;

    device const half* q_head = q + head * head_dim;

    // Pass 1: logits.
    for (uint e = tid; e < total; e += kDSV4Threads) {
        device const half* row;
        if (e < window_count) {
            const uint slot = (window_start_pos + e) % ring_capacity;
            row = window_kv + slot * head_dim;
        } else {
            const uint ci = e - window_count;
            const uint entry = use_all ? ci : selected[ci];
            row = compressed_kv + min(entry, compressed_count - 1u) * head_dim;
        }
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += 4u) {
            const half4 qa = *((device const half4*)(q_head + d));
            const half4 ka = *((device const half4*)(row + d));
            dot += float(qa.x) * float(ka.x) + float(qa.y) * float(ka.y)
                 + float(qa.z) * float(ka.z) + float(qa.w) * float(ka.w);
        }
        logits[e] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Max over logits and the sink.
    float local_max = sinks[head];
    for (uint e = tid; e < total; e += kDSV4Threads) {
        local_max = max(local_max, logits[e]);
    }
    local_max = simd_max(local_max);
    if ((tid & 31u) == 0u) red[tid >> 5] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = red[0];
    for (uint i = 1; i < kDSV4Threads / 32; ++i) m = max(m, red[i]);
    // `red` is reused for the sum reduction below; every thread must finish
    // reading the max partials first.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exponentials and denominator (sink included, no value contribution).
    float local_sum = 0.0f;
    for (uint e = tid; e < total; e += kDSV4Threads) {
        const float ex = fast::exp(logits[e] - m);
        logits[e] = ex;
        local_sum += ex;
    }
    local_sum = simd_sum(local_sum);
    if ((tid & 31u) == 0u) red[tid >> 5] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denom = fast::exp(sinks[head] - m);
    for (uint i = 0; i < kDSV4Threads / 32; ++i) denom += red[i];
    const float inv_denom = 1.0f / denom;

    // Pass 2: weighted value sum, threads stride the head dim.
    for (uint d = tid; d < head_dim; d += kDSV4Threads) {
        float acc = 0.0f;
        for (uint e = 0; e < total; ++e) {
            device const half* row;
            if (e < window_count) {
                const uint slot = (window_start_pos + e) % ring_capacity;
                row = window_kv + slot * head_dim;
            } else {
                const uint ci = e - window_count;
                const uint entry = use_all ? ci : selected[ci];
                row = compressed_kv + min(entry, compressed_count - 1u) * head_dim;
            }
            acc = fma(logits[e], float(row[d]), acc);
        }
        out[head * head_dim + d] = half(acc * inv_denom);
    }
}

// One compressor window emission (paper eqs. 20-23): per output channel c,
// softmax over the window slots of (gate + position_bias), convex-combine the
// kv rows, RMS-normalize with `norm_weight`. CSA (`dual == 1`) pools
// 2*rate slots — the prior window's Ca slices (channels [0, dim)) then the
// current window's Cb slices (channels [dim, 2*dim)); a missing prior window
// (`has_prior == 0`) contributes -inf gates (softmax weight 0). HCA
// (`dual == 0`) pools `rate` slots of width `dim` directly. RoPE at the
// window's first source position is applied by a separate
// `dsv4_rope_interleaved_trailing` dispatch on the emitted entry.
//
// After emission the caller's pending buffers are drained; for CSA the
// current window's Ca slices become the next window's prior — this kernel
// copies them to `next_prior_kv` / `next_prior_gate`.
//
// Grid: 1 threadgroup x >= dim threads.
kernel void dsv4_compress_emit(
    device const half* pending_kv [[buffer(0)]],     // [rate, row_width]
    device const half* pending_gate [[buffer(1)]],   // [rate, row_width]
    device const half* prior_kv [[buffer(2)]],       // [rate, dim] (CSA)
    device const half* prior_gate [[buffer(3)]],     // [rate, dim] (CSA)
    device const float* pos_bias [[buffer(4)]],      // [rate, row_width], FP32
    device const bfloat* norm_weight [[buffer(5)]],  // [dim], BF16
    device half* out_entry [[buffer(6)]],            // [dim]
    device half* next_prior_kv [[buffer(7)]],        // [rate, dim] (CSA)
    device half* next_prior_gate [[buffer(8)]],      // [rate, dim] (CSA)
    constant uint& rate [[buffer(9)]],
    constant uint& dim [[buffer(10)]],
    constant uint& dual [[buffer(11)]],
    constant uint& has_prior [[buffer(12)]],
    constant float& eps [[buffer(13)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float sumsq_red[16];   // up to 512 threads = 16 simdgroups
    const uint row_width = dual == 1u ? 2u * dim : dim;

    float pooled = 0.0f;
    if (tid < dim) {
        const uint slots = dual == 1u ? 2u * rate : rate;
        float max_g = -INFINITY;
        for (uint j = 0; j < slots; ++j) {
            float g;
            if (dual == 1u && j < rate) {
                g = (has_prior == 1u)
                    ? float(prior_gate[j * dim + tid])
                    : -INFINITY;
            } else {
                const uint pj = dual == 1u ? j - rate : j;
                const uint ch = dual == 1u ? dim + tid : tid;
                g = float(pending_gate[pj * row_width + ch])
                    + pos_bias[pj * row_width + ch];
            }
            max_g = max(max_g, g);
        }
        float denom = 0.0f;
        for (uint j = 0; j < slots; ++j) {
            float g;
            float v;
            if (dual == 1u && j < rate) {
                if (has_prior != 1u) continue;
                g = float(prior_gate[j * dim + tid]);
                v = float(prior_kv[j * dim + tid]);
            } else {
                const uint pj = dual == 1u ? j - rate : j;
                const uint ch = dual == 1u ? dim + tid : tid;
                g = float(pending_gate[pj * row_width + ch])
                    + pos_bias[pj * row_width + ch];
                v = float(pending_kv[pj * row_width + ch]);
            }
            const float w = fast::exp(g - max_g);
            denom += w;
            pooled = fma(w, v, pooled);
        }
        pooled /= denom;
    }

    // RMS over the pooled dim.
    float local = (tid < dim) ? pooled * pooled : 0.0f;
    local = simd_sum(local);
    if ((tid & 31u) == 0u) sumsq_red[min(tid >> 5, 15u)] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumsq = 0.0f;
    const uint n_sg = (tg_size + 31u) / 32u;
    for (uint i = 0; i < min(n_sg, 16u); ++i) sumsq += sumsq_red[i];
    const float inv = rsqrt(sumsq / float(dim) + eps);

    if (tid < dim) {
        out_entry[tid] = half(pooled * inv * float(norm_weight[tid]));
        // Persist the current window's Ca slices (+ position bias, matching
        // the reference which biases before splitting the series).
        if (dual == 1u) {
            for (uint j = 0; j < rate; ++j) {
                next_prior_kv[j * dim + tid] = pending_kv[j * row_width + tid];
                next_prior_gate[j * dim + tid] =
                    half(float(pending_gate[j * row_width + tid])
                         + pos_bias[j * row_width + tid]);
            }
        }
    }
}

// Lightning-indexer scores at decode: score[t] = sum_h w[h]*w_scale *
// relu(q_h . k_t * head_scale), over the compressed index keys. `w` is the
// raw weights_proj output (FP16 from the GEMV); `w_scale` carries the
// heads^-0.5 factor. Grid: one threadgroup per compressed entry, 128 threads.
kernel void dsv4_indexer_score(
    device const half* q [[buffer(0)]],           // [H_idx, idx_dim], roped
    device const half* keys [[buffer(1)]],        // [T, idx_dim]
    device const half* w [[buffer(2)]],           // [H_idx]
    device float* scores [[buffer(3)]],           // [T]
    constant uint& num_heads [[buffer(4)]],
    constant uint& idx_dim [[buffer(5)]],
    constant float& head_scale [[buffer(6)]],
    constant float& w_scale [[buffer(7)]],
    uint entry [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float head_red[4];
    device const half* key = keys + entry * idx_dim;
    // 4 simdgroups sweep the heads round-robin.
    float acc = 0.0f;
    for (uint h = sg; h < num_heads; h += 4u) {
        device const half* q_head = q + h * idx_dim;
        float dot = 0.0f;
        for (uint d = lane * 4u; d < idx_dim; d += 128u) {
            const half4 qa = *((device const half4*)(q_head + d));
            const half4 ka = *((device const half4*)(key + d));
            dot += float(qa.x) * float(ka.x) + float(qa.y) * float(ka.y)
                 + float(qa.z) * float(ka.z) + float(qa.w) * float(ka.w);
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            acc = fma(float(w[h]) * w_scale, max(dot * head_scale, 0.0f), acc);
        }
    }
    if (lane == 0) head_red[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        scores[entry] = head_red[0] + head_red[1] + head_red[2] + head_red[3];
    }
}

// mHC mixing weights for one sublayer site (paper §2.2 eq. 8). Computes
// mix = fn @ unweighted_rmsnorm(flatten(streams)), splits into pre / post /
// comb, applies the sigmoid / sigmoid*2 / softmax+Sinkhorn maps, and writes
// pre[H], post[H], comb[H*H] (fp32). fn rows are (2 + H) * H, H = hc_mult.
//
// Grid: 1 threadgroup x 256 threads (8 simdgroups; row r handled by
// simdgroup r % 8 in a round-robin).
kernel void dsv4_hc_weights(
    device const half* streams [[buffer(0)]],      // [H, hidden]
    device const float* fn [[buffer(1)]],          // [(2+H)*H, H*hidden]
    device const float* base_b [[buffer(2)]],      // [(2+H)*H]
    device const float* scale3 [[buffer(3)]],      // [3]
    device float* out_pre [[buffer(4)]],           // [H]
    device float* out_post [[buffer(5)]],          // [H]
    device float* out_comb [[buffer(6)]],          // [H, H] row-major
    constant uint& hc_mult [[buffer(7)]],
    constant uint& hidden [[buffer(8)]],
    constant uint& sinkhorn_iters [[buffer(9)]],
    constant float& hc_eps [[buffer(10)]],
    constant float& rms_eps [[buffer(11)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float red[8];
    threadgroup float mix[24];          // (2 + 4) * 4 for the production shape
    threadgroup float inv_norm_tg;
    const uint H = hc_mult;
    const uint flat = H * hidden;
    const uint rows = (2u + H) * H;

    // RMS of the flattened streams.
    float sumsq = 0.0f;
    for (uint i = tid; i < flat; i += 256u) {
        const float v = float(streams[i]);
        sumsq = fma(v, v, sumsq);
    }
    sumsq = simd_sum(sumsq);
    if (lane == 0) red[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) total += red[i];
        inv_norm_tg = rsqrt(total / float(flat) + rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_norm = inv_norm_tg;

    // fn rows, round-robin across simdgroups.
    for (uint r = sg; r < rows; r += 8u) {
        device const float* fn_row = fn + r * flat;
        float dot = 0.0f;
        for (uint i = lane; i < flat; i += 32u) {
            dot = fma(fn_row[i], float(streams[i]) * inv_norm, dot);
        }
        dot = simd_sum(dot);
        if (lane == 0) mix[r] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        const float pre_scale = scale3[0];
        const float post_scale = scale3[1];
        const float comb_scale = scale3[2];
        for (uint i = 0; i < H; ++i) {
            out_pre[i] = 1.0f / (1.0f + fast::exp(-(mix[i] * pre_scale + base_b[i])))
                + hc_eps;
            out_post[i] = 2.0f / (1.0f + fast::exp(-(mix[H + i] * post_scale
                                                     + base_b[H + i])));
        }
        // Row softmax + eps, then Sinkhorn: column, then (iters-1) x
        // (row, column) alternations — matching the reference order.
        float comb[16];
        for (uint r = 0; r < H; ++r) {
            float mx = -INFINITY;
            for (uint c2 = 0; c2 < H; ++c2) {
                const uint i = 2u * H + r * H + c2;
                comb[r * H + c2] = mix[i] * comb_scale + base_b[i];
                mx = max(mx, comb[r * H + c2]);
            }
            float sum = 0.0f;
            for (uint c2 = 0; c2 < H; ++c2) {
                comb[r * H + c2] = fast::exp(comb[r * H + c2] - mx);
                sum += comb[r * H + c2];
            }
            for (uint c2 = 0; c2 < H; ++c2) {
                comb[r * H + c2] = comb[r * H + c2] / sum + hc_eps;
            }
        }
        for (uint c2 = 0; c2 < H; ++c2) {
            float col = 0.0f;
            for (uint r = 0; r < H; ++r) col += comb[r * H + c2];
            for (uint r = 0; r < H; ++r) comb[r * H + c2] /= (col + hc_eps);
        }
        for (uint it = 1; it < sinkhorn_iters; ++it) {
            for (uint r = 0; r < H; ++r) {
                float row_sum = 0.0f;
                for (uint c2 = 0; c2 < H; ++c2) row_sum += comb[r * H + c2];
                for (uint c2 = 0; c2 < H; ++c2) comb[r * H + c2] /= (row_sum + hc_eps);
            }
            for (uint c2 = 0; c2 < H; ++c2) {
                float col = 0.0f;
                for (uint r = 0; r < H; ++r) col += comb[r * H + c2];
                for (uint r = 0; r < H; ++r) comb[r * H + c2] /= (col + hc_eps);
            }
        }
        for (uint i = 0; i < H * H; ++i) out_comb[i] = comb[i];
    }
}

// Collapse the H streams into one sequence row: x[d] = sum_j pre[j] *
// streams[j][d]. Grid: >= hidden threads.
kernel void dsv4_hc_collapse(
    device const half* streams [[buffer(0)]],
    device const float* pre [[buffer(1)]],
    device half* x [[buffer(2)]],
    constant uint& hc_mult [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= hidden) return;
    float acc = 0.0f;
    for (uint j = 0; j < hc_mult; ++j) {
        acc = fma(pre[j], float(streams[j * hidden + d]), acc);
    }
    x[d] = half(acc);
}

// Place the sublayer output back into the streams and mix the residual:
// out_streams[k][d] = post[k] * sub[d] + sum_j comb[j][k] * streams[j][d]
// (comb consumed transposed, matching the reference). Reads `streams`,
// writes `out_streams` — the caller ping-pongs the two buffers.
// Grid: hc_mult x hidden threads (2D).
kernel void dsv4_hc_place_mix(
    device const half* streams [[buffer(0)]],
    device const half* sub [[buffer(1)]],
    device const float* post [[buffer(2)]],
    device const float* comb [[buffer(3)]],
    device half* out_streams [[buffer(4)]],
    constant uint& hc_mult [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint k = gid.y;
    const uint d = gid.x;
    if (k >= hc_mult || d >= hidden) return;
    float acc = post[k] * float(sub[d]);
    for (uint j = 0; j < hc_mult; ++j) {
        acc = fma(comb[j * hc_mult + k], float(streams[j * hidden + d]), acc);
    }
    out_streams[k * hidden + d] = half(acc);
}

// Final stream collapse (DeepseekV4HyperHead): pre = sigmoid(hc_fn @
// unweighted_rmsnorm(flat) * scale + base) + eps, then the pre-weighted sum.
// Grid: 1 threadgroup x 256 threads, then a second pass over hidden.
kernel void dsv4_hyper_head(
    device const half* streams [[buffer(0)]],
    device const float* hc_fn [[buffer(1)]],       // [H, H*hidden]
    device const float* hc_base [[buffer(2)]],     // [H]
    device const float* hc_scale [[buffer(3)]],    // [1]
    device half* x [[buffer(4)]],                  // [hidden]
    constant uint& hc_mult [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    constant float& hc_eps [[buffer(7)]],
    constant float& rms_eps [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float red[8];
    threadgroup float pre[8];
    threadgroup float inv_norm_tg;
    const uint H = hc_mult;
    const uint flat = H * hidden;

    float sumsq = 0.0f;
    for (uint i = tid; i < flat; i += 256u) {
        const float v = float(streams[i]);
        sumsq = fma(v, v, sumsq);
    }
    sumsq = simd_sum(sumsq);
    if (lane == 0) red[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) total += red[i];
        inv_norm_tg = rsqrt(total / float(flat) + rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_norm = inv_norm_tg;

    for (uint r = sg; r < H; r += 8u) {
        device const float* fn_row = hc_fn + r * flat;
        float dot = 0.0f;
        for (uint i = lane; i < flat; i += 32u) {
            dot = fma(fn_row[i], float(streams[i]) * inv_norm, dot);
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            pre[r] = 1.0f / (1.0f + fast::exp(-(dot * hc_scale[0] + hc_base[r])))
                + hc_eps;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < hidden; d += 256u) {
        float acc = 0.0f;
        for (uint j = 0; j < H; ++j) {
            acc = fma(pre[j], float(streams[j * hidden + d]), acc);
        }
        x[d] = half(acc);
    }
}

// Clamped SwiGLU for the DeepSeek shared expert: y = silu(min(g, limit)) *
// clamp(u, -limit, limit). Grid: >= n threads.
kernel void dsv4_swiglu_clamp_mul(
    device const half* g [[buffer(0)]],
    device const half* u [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& limit [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    const float gv = min(float(g[i]), limit);
    const float uv = clamp(float(u[i]), -limit, limit);
    const float act = gv / (1.0f + fast::exp(-gv));
    y[i] = half(act * uv);
}

// Broadcast one hidden row into the H residual streams (embedding entry).
kernel void dsv4_broadcast_streams(
    device const half* x [[buffer(0)]],
    device half* streams [[buffer(1)]],
    constant uint& hc_mult [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.y >= hc_mult || gid.x >= hidden) return;
    streams[gid.y * hidden + gid.x] = x[gid.x];
}
