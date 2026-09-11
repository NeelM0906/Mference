#include <metal_stdlib>
using namespace metal;

// ============================================================================
// glm53.metal — GLM-5.3-Flash (`glm53Flash`) kernel additions. The family's
// mHC mixes, swiglu clamp, indexer scoring and the depthwise conv reuse the
// `dsv4_*` and `gdn_*` kernels of the shared library; what is new here is
// the Kimi Delta Attention recurrence with its per-channel decay and
// sigmoid-gated output norm, the NoPE latent attention over a selected row
// set, the pooled indexer keys, a LayerNorm with bias for the indexer keys,
// a per-head batched INT8 GEMV for the absorbed-MLA folds, and an INT8
// embedding gather. All arithmetic is fp32; storage is FP16 (activations,
// caches), BF16 (norm gains, companions), FP32 (KDA decay params, state).
// ============================================================================

// INT8 affine group-64 embedding row gather: out[d] = q * scale + bias.
kernel void glm53_embed_lookup_int8(
    device const uint8_t* table [[buffer(0)]],    // [V, D] bytes
    device const bfloat* scales [[buffer(1)]],    // [V, D/64]
    device const bfloat* biases [[buffer(2)]],    // [V, D/64]
    device half* out [[buffer(3)]],               // [D]
    constant uint& token_id [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint groups = D / 64u;
    const uint q = uint(table[uint(token_id) * D + gid]);
    const float s = float(scales[uint(token_id) * groups + gid / 64u]);
    const float b = float(biases[uint(token_id) * groups + gid / 64u]);
    out[gid] = half(float(q) * s + b);
}

// ----------------------------------------------------------------------------
// Kimi Delta Attention, one token. One threadgroup per head (256 threads,
// 8 simdgroups). Reference (glm53_flash_mlx.language / mlx_vlm.gated_delta):
//   q, k, v  = head slices of the post-SiLU conv output
//   q = l2norm(q) * D^-0.5 ; k = l2norm(k)      (l2norm: x * rsqrt(sum x^2 + 1e-6))
//   decay[d] = exp(lower_bound * sigmoid(exp(A_log[h]) * (a[h,d] + dt_bias[h,d])))
//   beta     = sigmoid(b[h])
//   S[dv, :] *= decay ; kv = S k ; delta = (v - kv) * beta ; S += k (x) delta ; y = S q
//   out = rmsnorm_D(y) * o_norm * sigmoid(gate)
// State layout [H][Dv][Dk], dk fastest — the reference's own `[Hv, Dv, Dk]`.
// D must be a multiple of 32 and at most 128.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53_kda_decode(
    device const half*   conv_out [[buffer(0)]],   // [3*H*D]: q | k | v
    device const half*   a        [[buffer(1)]],   // [H*D]  f_b(f_a(x))
    device const half*   b        [[buffer(2)]],   // [H]    b_proj(x)
    device const half*   gate     [[buffer(3)]],   // [H*D]  g_b(g_a(x))
    device const float*  A_log    [[buffer(4)]],   // [H]
    device const float*  dt_bias  [[buffer(5)]],   // [H*D]
    device const bfloat* o_norm   [[buffer(6)]],   // [D]
    device float*        state    [[buffer(7)]],   // [H][D][D]
    device half*         out      [[buffer(8)]],   // [H*D]
    device half*         y_out    [[buffer(9)]],   // [H*D] the recurrence read-out (capture)
    constant uint&       H        [[buffer(10)]],
    constant uint&       D        [[buffer(11)]],
    constant float&      lower_bound [[buffer(12)]],
    constant float&      eps      [[buffer(13)]],
    uint h    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float qs[128];
    threadgroup float ks[128];
    threadgroup float vs[128];
    threadgroup float decay[128];
    threadgroup float ys[128];
    threadgroup float red_q[8];
    threadgroup float red_k[8];
    threadgroup float scalars[2];   // inv_q (with the D^-0.5 folded in), inv_k
    if (h >= H) return;
    const uint qkv = H * D;
    const uint base = h * D;

    if (tid < D) {
        qs[tid] = float(conv_out[base + tid]);
        ks[tid] = float(conv_out[qkv + base + tid]);
        vs[tid] = float(conv_out[2u * qkv + base + tid]);
        const float g = exp(A_log[h]) * (float(a[base + tid]) + dt_bias[base + tid]);
        decay[tid] = exp(lower_bound / (1.0f + exp(-g)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qq = tid < D ? qs[tid] * qs[tid] : 0.0f;
    float kk = tid < D ? ks[tid] * ks[tid] : 0.0f;
    qq = simd_sum(qq);
    kk = simd_sum(kk);
    if (lane == 0) { red_q[sg] = qq; red_k[sg] = kk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sq = 0.0f, sk = 0.0f;
        for (uint i = 0; i < 8u; ++i) { sq += red_q[i]; sk += red_k[i]; }
        scalars[0] = rsqrt(sq + 1e-6f) * rsqrt(float(D));
        scalars[1] = rsqrt(sk + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < D) {
        qs[tid] *= scalars[0];
        ks[tid] *= scalars[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float beta = 1.0f / (1.0f + exp(-float(b[h])));
    const uint per_lane = D / 32u;
    for (uint dv = sg; dv < D; dv += 8u) {
        device float* srow = state + (uint(h) * D + dv) * D;
        float s[4];
        float kvm = 0.0f;
        for (uint i = 0; i < per_lane; ++i) {
            const uint idx = lane * per_lane + i;
            s[i] = srow[idx] * decay[idx];
            kvm = fma(s[i], ks[idx], kvm);
        }
        kvm = simd_sum(kvm);
        const float delta = (vs[dv] - kvm) * beta;
        float yv = 0.0f;
        for (uint i = 0; i < per_lane; ++i) {
            const uint idx = lane * per_lane + i;
            s[i] = fma(ks[idx], delta, s[i]);
            yv = fma(s[i], qs[idx], yv);
            srow[idx] = s[i];
        }
        yv = simd_sum(yv);
        if (lane == 0) ys[dv] = yv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sigmoid-gated RMSNorm over the head's D outputs.
    float yy = tid < D ? ys[tid] * ys[tid] : 0.0f;
    yy = simd_sum(yy);
    if (lane == 0) red_q[sg] = yy;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < 8u; ++i) sum += red_q[i];
        scalars[0] = rsqrt(sum / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < D) {
        const float gv = 1.0f / (1.0f + exp(-float(gate[base + tid])));
        out[base + tid] = half(ys[tid] * scalars[0] * float(o_norm[tid]) * gv);
        y_out[base + tid] = half(ys[tid]);
    }
}

// ----------------------------------------------------------------------------
// LayerNorm with gain and bias over one row of `d` values (the indexer key
// path: `nn.LayerNorm(index_head_dim, eps=1e-6)` with bias). One threadgroup,
// 256 threads; d up to 4096.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53_layernorm_bias(
    device const half*   x      [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* bias   [[buffer(2)]],
    device half*         out    [[buffer(3)]],
    constant uint&       d      [[buffer(4)]],
    constant float&      eps    [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float red[8];
    threadgroup float stats[2];
    float sum = 0.0f;
    for (uint i = tid; i < d; i += 256u) sum += float(x[i]);
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8u; ++i) total += red[i];
        stats[0] = total / float(d);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = stats[0];
    float var = 0.0f;
    for (uint i = tid; i < d; i += 256u) {
        const float c = float(x[i]) - mean;
        var = fma(c, c, var);
    }
    var = simd_sum(var);
    if (lane == 0) red[sg] = var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8u; ++i) total += red[i];
        stats[1] = rsqrt(total / float(d) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = stats[1];
    for (uint i = tid; i < d; i += 256u) {
        out[i] = half((float(x[i]) - mean) * inv * float(weight[i]) + float(bias[i]));
    }
}

// ----------------------------------------------------------------------------
// One pooled indexer key: pool `j` covers tokens j*kp .. j*kp+kp-1 (complete
// pools only). Per channel d: w_c = softmax_c(gate[t_c][d] + ape[c][d]),
// pooled[j][d] = sum_c w_c * key[t_c][d]. Grid: >= dim threads.
// ----------------------------------------------------------------------------
kernel void glm53_pool_keys(
    device const half*   keys   [[buffer(0)]],   // [T][dim]
    device const half*   gates  [[buffer(1)]],   // [T][dim]
    device const bfloat* ape    [[buffer(2)]],   // [kp][dim]
    device half*         pooled [[buffer(3)]],   // [P][dim]
    constant uint&       pool   [[buffer(4)]],   // j
    constant uint&       kp     [[buffer(5)]],
    constant uint&       dim    [[buffer(6)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= dim) return;
    const uint first = pool * kp;
    float logits[8];
    float mx = -INFINITY;
    for (uint c = 0; c < kp; ++c) {
        logits[c] = float(gates[(first + c) * dim + d]) + float(ape[c * dim + d]);
        mx = max(mx, logits[c]);
    }
    float sum = 0.0f;
    for (uint c = 0; c < kp; ++c) { logits[c] = exp(logits[c] - mx); sum += logits[c]; }
    float acc = 0.0f;
    for (uint c = 0; c < kp; ++c) {
        acc = fma(logits[c] / sum, float(keys[(first + c) * dim + d]), acc);
    }
    pooled[pool * dim + d] = half(acc);
}

// ----------------------------------------------------------------------------
// NoPE latent attention, one token: per head h, softmax(q_lat[h] . latent[t]
// * scale) over the selected rows (or every cached row when selected_count is
// 0xFFFFFFFF), with the same latent row as the value. One threadgroup per
// head, 8 simdgroups stride the rows with an online softmax each, merged at
// the end. kv_dim: a multiple of 32, at most 512.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53_latent_attention(
    device const half* q_lat    [[buffer(0)]],   // [H][kv]
    device const half* latents  [[buffer(1)]],   // [T][kv]
    device const uint* selected [[buffer(2)]],   // [selected_count]
    device half*       out      [[buffer(3)]],   // [H][kv]
    constant uint&     kv_dim   [[buffer(4)]],
    constant uint&     total    [[buffer(5)]],   // cached rows T
    constant uint&     selected_count [[buffer(6)]],
    constant float&    scale    [[buffer(7)]],
    uint h    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float acc_tg[8 * 512];
    threadgroup float m_tg[8];
    threadgroup float d_tg[8];
    const uint per = kv_dim / 32u;
    float q[16];
    for (uint i = 0; i < per; ++i) q[i] = float(q_lat[h * kv_dim + lane * per + i]);

    const bool dense = selected_count == 0xFFFFFFFFu;
    const uint n = dense ? total : selected_count;
    float m = -FLT_MAX / 2.0f;
    float denom = 0.0f;
    float acc[16];
    for (uint i = 0; i < per; ++i) acc[i] = 0.0f;
    for (uint r = sg; r < n; r += 8u) {
        const uint row = dense ? r : selected[r];
        device const half* k = latents + row * kv_dim + lane * per;
        float kv[16];
        float dot = 0.0f;
        for (uint i = 0; i < per; ++i) { kv[i] = float(k[i]); dot = fma(q[i], kv[i], dot); }
        dot = simd_sum(dot) * scale;
        const float new_m = max(m, dot);
        const float rescale = exp(m - new_m);
        const float w = exp(dot - new_m);
        denom = fma(denom, rescale, w);
        for (uint i = 0; i < per; ++i) acc[i] = fma(acc[i], rescale, kv[i] * w);
        m = new_m;
    }
    for (uint i = 0; i < per; ++i) acc_tg[sg * kv_dim + lane * per + i] = acc[i];
    if (lane == 0) { m_tg[sg] = m; d_tg[sg] = denom; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -FLT_MAX / 2.0f;
    for (uint s = 0; s < 8u; ++s) M = max(M, m_tg[s]);
    float Dn = 0.0f;
    for (uint s = 0; s < 8u; ++s) Dn = fma(d_tg[s], exp(m_tg[s] - M), Dn);
    const float inv = Dn > 0.0f ? 1.0f / Dn : 0.0f;
    for (uint d = tid; d < kv_dim; d += 256u) {
        float o = 0.0f;
        for (uint s = 0; s < 8u; ++s) o = fma(acc_tg[s * kv_dim + d], exp(m_tg[s] - M), o);
        out[h * kv_dim + d] = half(o * inv);
    }
}

// ----------------------------------------------------------------------------
// Per-head batched INT8 affine group-64 GEMV: y[h][m] = W[h] x[h], with
// W stored `[H, M, N]` bytes and companions `[H, M, N/64]`. The absorbed-MLA
// folds (`embed_q`: N = qk head dim, M = latent; `unembed_out`: the reverse).
// Same row math as `dequant_int8_gemv_simd`; grid (ceil(M/8), H) threadgroups
// of 256 threads, one simdgroup per row.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53_headed_int8_gemv(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],   // [H][N]
    device half*          y      [[buffer(4)]],   // [H][M]
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint2                 tg     [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint h = tg.y;
    const uint row = tg.x * 8u + sg_idx;
    if (row >= M) return;
    const uint n_groups = N / 64u;
    device const uint8_t* W_row = W + (uint(h) * M + row) * N;
    device const bfloat*  s_row = scales + (uint(h) * M + row) * n_groups;
    device const bfloat*  b_row = biases + (uint(h) * M + row) * n_groups;
    device const half*    xh = x + uint(h) * N;
    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint i0 = g * 64u + lane * 2u;
        const float q0 = float(uint(W_row[i0]));
        const float q1 = float(uint(W_row[i0 + 1u]));
        const float x0 = float(xh[i0]);
        const float x1 = float(xh[i0 + 1u]);
        acc = fma(s, q0 * x0 + q1 * x1, acc);
        acc = fma(b, x0 + x1, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[uint(h) * M + row] = half(acc);
}


// ---------------------------------------------------------------------------
// Router selection: sigmoid scores, selection on score + correction bias
// (stable descending, the lower index on ties, as MLX's argsort), weights the
// unbiased scores of the chosen experts renormalized to one and scaled by
// routed_scaling_factor. Serial on one thread: 288 experts x 8 slots is a few
// thousand flops, well below one kernel launch on the layer's critical path,
// and the serial scan pins the tie rule exactly. Both expert-streaming modes
// route through this kernel, so their outputs are byte-identical.
// ---------------------------------------------------------------------------
kernel void glm53_router_select_k8(
    device const float* logits [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant float& route_scale [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    constexpr uint K = 8;
    uint top_idx[K];
    float top_key[K];
    float top_score[K];
    for (uint i = 0; i < K; ++i) { top_idx[i] = 0u; top_key[i] = -INFINITY; top_score[i] = 0.0f; }
    for (uint e = 0; e < num_experts; ++e) {
        const float s = 1.0f / (1.0f + precise::exp(-logits[e]));
        const float key = s + bias[e];
        if (key <= top_key[K - 1]) continue;
        uint pos = K;
        for (uint i = 0; i < K; ++i) {
            if (key > top_key[i]) { pos = i; break; }
        }
        if (pos >= K) continue;
        for (uint i = K - 1; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_key[i] = top_key[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_key[pos] = key;
        top_score[pos] = s;
    }
    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) sum += top_score[i];
    for (uint i = 0; i < K; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = half(top_score[i] / sum * route_scale);
    }
}

// ---------------------------------------------------------------------------
// mHC mixing weights, split so the fn matvec runs wide. `dsv4_hc_weights`
// does the whole site in one threadgroup: 24 rows x 16,384 fp32 walked with a
// dependent load-fma chain per lane, which measured 0.58 ms per site on the
// M3 Ultra (2.9 GB/s) — two sites x 45 layers was 52 ms of an 86 ms token.
//
// `glm53_hc_dots`: threadgroup r < rows computes the raw dot of fn row r with
// the flattened streams (float4 / half4 loads, 256 threads); threadgroup
// `rows` computes the streams' sum of squares. `glm53_hc_finalize` (one
// thread) applies the RMS normalization (linear, so it factors out of the
// dots), the sigmoid / 2*sigmoid / softmax + Sinkhorn maps — the same
// arithmetic as `dsv4_hc_weights`' tail, in the same order.
// ---------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53_hc_dots(
    device const half*  streams [[buffer(0)]],   // [H * hidden]
    device const float* fn      [[buffer(1)]],   // [rows, H * hidden]
    device float*       partials [[buffer(2)]],  // [rows + 1]: dots, then sumsq
    constant uint&      flat    [[buffer(3)]],   // H * hidden, a multiple of 4
    constant uint&      rows    [[buffer(4)]],
    uint r    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float red[8];
    float acc = 0.0f;
    if (r < rows) {
        device const float4* f4 = (device const float4*)(fn + uint(r) * flat);
        device const half4*  s4 = (device const half4*)streams;
        for (uint i = tid; i < flat / 4u; i += 256u) {
            const float4 f = f4[i];
            const float4 s = float4(s4[i]);
            acc = fma(f.x, s.x, acc);
            acc = fma(f.y, s.y, acc);
            acc = fma(f.z, s.z, acc);
            acc = fma(f.w, s.w, acc);
        }
    } else {
        device const half4* s4 = (device const half4*)streams;
        for (uint i = tid; i < flat / 4u; i += 256u) {
            const float4 s = float4(s4[i]);
            acc = fma(s.x, s.x, acc);
            acc = fma(s.y, s.y, acc);
            acc = fma(s.z, s.z, acc);
            acc = fma(s.w, s.w, acc);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) red[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8u; ++i) total += red[i];
        partials[r] = total;
    }
}

kernel void glm53_hc_finalize(
    device const float* partials [[buffer(0)]],  // [rows + 1]
    device const float* base_b   [[buffer(1)]],  // [rows]
    device const float* scale3   [[buffer(2)]],  // [3]
    device float*       out_pre  [[buffer(3)]],  // [H]
    device float*       out_post [[buffer(4)]],  // [H]
    device float*       out_comb [[buffer(5)]],  // [H, H]
    constant uint&      hc_mult  [[buffer(6)]],
    constant uint&      flat     [[buffer(7)]],
    constant uint&      sinkhorn_iters [[buffer(8)]],
    constant float&     hc_eps   [[buffer(9)]],
    constant float&     rms_eps  [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    const uint H = hc_mult;
    const uint rows = (2u + H) * H;
    const float inv_norm = rsqrt(partials[rows] / float(flat) + rms_eps);
    float mix[24];
    for (uint i = 0; i < rows; ++i) mix[i] = partials[i] * inv_norm;

    const float pre_scale = scale3[0];
    const float post_scale = scale3[1];
    const float comb_scale = scale3[2];
    for (uint i = 0; i < H; ++i) {
        out_pre[i] = 1.0f / (1.0f + fast::exp(-(mix[i] * pre_scale + base_b[i]))) + hc_eps;
        out_post[i] = 2.0f / (1.0f + fast::exp(-(mix[H + i] * post_scale + base_b[H + i])));
    }
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
        for (uint c2 = 0; c2 < H; ++c2) comb[r * H + c2] = comb[r * H + c2] / sum + hc_eps;
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

// ---------------------------------------------------------------------------
// Router select, one simdgroup: lane L owns experts L, L+32, ... (at most 16
// per lane, so up to 512 experts). Each of the K steps takes the lane-local
// best (highest biased key, lowest index among equals), then the simdgroup's
// highest key and, among lanes holding it, the lowest index — the serial
// kernel's order exactly, so selection and weights are bit-identical to
// `glm53_router_select_k8`.
// ---------------------------------------------------------------------------
kernel void glm53_router_select_k8_par(
    device const float* logits [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant float& route_scale [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint K = 8;
    constexpr uint PER = 16;
    float key[PER];
    float score[PER];
    for (uint j = 0; j < PER; ++j) {
        const uint e = lane + 32u * j;
        if (e < num_experts) {
            score[j] = 1.0f / (1.0f + precise::exp(-logits[e]));
            key[j] = score[j] + bias[e];
        } else {
            score[j] = 0.0f;
            key[j] = -INFINITY;
        }
    }
    float chosen_score[K];
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint best_j = PER;
        for (uint j = 0; j < PER; ++j) {
            if (key[j] > best) { best = key[j]; best_j = j; }
        }
        const float top = simd_max(best);
        const uint mine = (best_j < PER && best == top) ? lane + 32u * best_j : 0xFFFFFFFFu;
        const uint winner = simd_min(mine);
        const float s = 1.0f / (1.0f + precise::exp(-logits[winner]));
        chosen_score[k] = s;
        if (lane == 0) out_indices[k] = winner;
        if (winner == mine) key[best_j] = -INFINITY;
    }
    if (lane == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < K; ++k) sum += chosen_score[k];
        for (uint k = 0; k < K; ++k) out_weights[k] = half(chosen_score[k] / sum * route_scale);
    }
}
