#include <metal_stdlib>
using namespace metal;

// ============================================================================
// GLM-5.3-Flash batched (chunked) prefill kernels.
//
// The per-token decode path streams every weight once per prompt token, so
// prefill runs at decode speed. These kernels process a chunk of T tokens per
// dispatch: the projections read each weight row once and apply it to T
// activations, the mHC / conv / indexer bookkeeping runs over the chunk, and
// the KDA recurrence walks the chunk inside one threadgroup per head with the
// state held in registers. Same arithmetic as the decode kernels wherever an
// element is computed (the KDA update, the mHC maps, the latent softmax); the
// GEMMs accumulate in a different order than the GEMVs, so the prefill tier is
// FP16-close to sequential decode rather than bit-identical.
// ============================================================================

constant uint kG53PGroup = 64;

// ---------------------------------------------------------------------------
// INT8 affine group-64 GEMM: Y[t][m] = sum_n W[m][n] X[t][n] for T tokens.
// Threadgroup: 8 simdgroups (one weight row each) x a tile of 32 tokens; grid
// (ceil(M/8), ceil(T/32)). N is walked in chunks of 256 (4 groups): the chunk's
// activation tile [32 tokens][256] is staged once in threadgroup memory by all
// 256 threads, then each lane applies its 2 weight bytes per group (as
// `dequant_int8_gemv_simd`) to the 32 tokens from that tile, so global traffic
// is the weight row once plus the activations once per 8 rows.
// ---------------------------------------------------------------------------
constant uint kG53PTokenTile = 32;
constant uint kG53PChunk = 256;

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53p_int8_gemm(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    X      [[buffer(3)]],
    device half*          Y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        T      [[buffer(7)]],
    constant uint&        x_stride [[buffer(8)]],
    constant uint&        y_stride [[buffer(9)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup half xt[kG53PTokenTile * kG53PChunk];   // 16 KB
    const uint tid = tid2.x;
    const uint m = tg.x * 8u + sg;
    const uint row_valid = m < M ? 1u : 0u;
    const uint t0 = tg.y * kG53PTokenTile;
    const uint tn = min(kG53PTokenTile, T - t0);
    const uint n_groups = N / kG53PGroup;
    device const uint8_t* W_row = W + uint(min(m, M - 1u)) * N;
    device const bfloat*  s_row = scales + uint(min(m, M - 1u)) * n_groups;
    device const bfloat*  b_row = biases + uint(min(m, M - 1u)) * n_groups;
    float acc[kG53PTokenTile];
    for (uint t = 0; t < kG53PTokenTile; ++t) acc[t] = 0.0f;

    for (uint n0 = 0; n0 < N; n0 += kG53PChunk) {
        const uint nc = min(kG53PChunk, N - n0);
        // Stage the tile: element e of the flattened [tn][nc] tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Stage [32 tokens][nc] as half2 pairs: thread `tid` covers pair
        // indices tid, tid + 256, ... Rows past `tn` are zeroed so the token
        // loop below can run to the compile-time tile bound.
        {
            threadgroup half2* xt2 = (threadgroup half2*)xt;
            const uint pairs_per_row = nc / 2u;
            for (uint e = tid; e < kG53PTokenTile * pairs_per_row; e += 256u) {
                const uint t = e / pairs_per_row;
                const uint i = (e - t * pairs_per_row) * 2u;
                half2 v = half2(0.0h, 0.0h);
                if (t < tn) v = *((device const half2*)(X + (t0 + t) * x_stride + n0 + i));
                xt2[t * (kG53PChunk / 2u) + i / 2u] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row_valid == 0u) continue;
        const uint groups_here = nc / kG53PGroup;
        for (uint gi = 0; gi < groups_here; ++gi) {
            const uint g = n0 / kG53PGroup + gi;
            const float s = float(s_row[g]);
            const float b = float(b_row[g]);
            const uint i0 = g * kG53PGroup + lane * 2u;
            const float q0 = float(uint(W_row[i0]));
            const float q1 = float(uint(W_row[i0 + 1u]));
            threadgroup const half2* col = (threadgroup const half2*)(xt + gi * kG53PGroup + lane * 2u);
#pragma unroll
            for (uint t = 0; t < kG53PTokenTile; ++t) {
                const float2 xv = float2(col[t * (kG53PChunk / 2u)]);
                acc[t] = fma(s, q0 * xv.x + q1 * xv.y, acc[t]);
                acc[t] = fma(b, xv.x + xv.y, acc[t]);
            }
        }
    }
    if (row_valid == 0u) return;
#pragma unroll
    for (uint t = 0; t < kG53PTokenTile; ++t) {
        const float v = simd_sum(acc[t]);
        if (lane == 0 && t < tn) Y[(t0 + t) * y_stride + m] = half(v);
    }
}

// BF16 GEMM, same shape rules (the router gate, the indexer pooling gate).
// `f32_out` selects fp32 output (router logits).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53p_bf16_gemm(
    device const bfloat* W      [[buffer(0)]],
    device const half*   X      [[buffer(1)]],
    device half*         Y16    [[buffer(2)]],
    device float*        Y32    [[buffer(3)]],
    constant uint&       M      [[buffer(4)]],
    constant uint&       N      [[buffer(5)]],
    constant uint&       T      [[buffer(6)]],
    constant uint&       x_stride [[buffer(7)]],
    constant uint&       y_stride [[buffer(8)]],
    constant uint&       f32_out [[buffer(9)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m = tg.x * 8u + sg;
    if (m >= M) return;
    const uint t0 = tg.y * 16u;
    const uint tn = min(16u, T - t0);
    device const bfloat* W_row = W + uint(m) * N;
    float acc[16];
#pragma unroll
    for (uint t = 0; t < 16u; ++t) acc[t] = 0.0f;
    for (uint i = lane * 2u; i < N; i += 64u) {
        const float w0 = float(W_row[i]);
        const float w1 = float(W_row[i + 1u]);
#pragma unroll
        for (uint t = 0; t < 16u; ++t) {
            device const half* x = X + (t0 + min(t, tn - 1u)) * x_stride;
            acc[t] = fma(w0, float(x[i]), acc[t]);
            acc[t] = fma(w1, float(x[i + 1u]), acc[t]);
        }
    }
#pragma unroll
    for (uint t = 0; t < 16u; ++t) {
        const float v = simd_sum(acc[t]);
        if (lane == 0 && t < tn) {
            if (f32_out != 0u) Y32[(t0 + t) * y_stride + m] = v;
            else Y16[(t0 + t) * y_stride + m] = half(v);
        }
    }
}

// Per-head INT8 GEMV for T tokens: y[t][h][m] = W[h] x[t][h]. Grid
// (ceil(M/8), H, T). Weights are re-read per token (the folds are small).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53p_headed_int8_gemv_batched(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],   // [T][H][N]
    device half*          y      [[buffer(4)]],   // [T][H][M]
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        H      [[buffer(7)]],
    uint3                 tg     [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint h = tg.y;
    const uint t = tg.z;
    const uint row = tg.x * 8u + sg_idx;
    if (row >= M) return;
    const uint n_groups = N / kG53PGroup;
    device const uint8_t* W_row = W + (uint(h) * M + row) * N;
    device const bfloat*  s_row = scales + (uint(h) * M + row) * n_groups;
    device const bfloat*  b_row = biases + (uint(h) * M + row) * n_groups;
    device const half*    xh = x + (uint(t) * H + h) * N;
    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint i0 = g * kG53PGroup + lane * 2u;
        const float q0 = float(uint(W_row[i0]));
        const float q1 = float(uint(W_row[i0 + 1u]));
        const float x0 = float(xh[i0]);
        const float x1 = float(xh[i0 + 1u]);
        acc = fma(s, q0 * x0 + q1 * x1, acc);
        acc = fma(b, x0 + x1, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(uint(t) * H + h) * M + row] = half(acc);
}

// ---------------------------------------------------------------------------
// Embedding rows for T tokens: out[t] = dequant(table[token[t]]). Grid (D, T).
// ---------------------------------------------------------------------------
kernel void glm53p_embed_lookup_int8_batched(
    device const uint8_t* table  [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const uint*    tokens [[buffer(3)]],
    device half*          out    [[buffer(4)]],   // [T][D]
    constant uint&        D      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (d >= D) return;
    const uint tok = tokens[t];
    const uint groups = D / kG53PGroup;
    const uint g = tok * groups + d / kG53PGroup;
    out[t * D + d] = half(fma(float(uint(table[tok * D + d])), float(scales[g]), float(biases[g])));
}

// streams[t][k][d] = x[t][d]. Grid (D, H * T).
kernel void glm53p_broadcast_streams_batched(
    device const half* x [[buffer(0)]],          // [T][D]
    device half* streams [[buffer(1)]],          // [T][H][D]
    constant uint& hc_mult [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    if (d >= hidden) return;
    const uint t = gid.y / hc_mult;
    streams[gid.y * hidden + d] = x[t * hidden + d];
}

// ---------------------------------------------------------------------------
// mHC over the chunk: dots (grid (rows + 1, T)), finalize (T threads),
// collapse (grid (hidden, T)), place-mix (grid (hidden, H * T)). The maps are
// `glm53_hc_finalize`'s, per token.
// ---------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53p_hc_dots_batched(
    device const half*  streams  [[buffer(0)]],   // [T][flat]
    device const float* fn       [[buffer(1)]],   // [rows][flat]
    device float*       partials [[buffer(2)]],   // [T][rows + 1]
    constant uint&      flat     [[buffer(3)]],
    constant uint&      rows     [[buffer(4)]],
    uint2 tgp  [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]
) {
    threadgroup float red[8];
    const uint tid = tid2.x;
    const uint r = tgp.x;
    const uint t = tgp.y;
    device const half4* s4 = (device const half4*)(streams + uint(t) * flat);
    float acc = 0.0f;
    if (r < rows) {
        device const float4* f4 = (device const float4*)(fn + uint(r) * flat);
        for (uint i = tid; i < flat / 4u; i += 256u) {
            const float4 f = f4[i];
            const float4 s = float4(s4[i]);
            acc = fma(f.x, s.x, acc); acc = fma(f.y, s.y, acc);
            acc = fma(f.z, s.z, acc); acc = fma(f.w, s.w, acc);
        }
    } else {
        for (uint i = tid; i < flat / 4u; i += 256u) {
            const float4 s = float4(s4[i]);
            acc = fma(s.x, s.x, acc); acc = fma(s.y, s.y, acc);
            acc = fma(s.z, s.z, acc); acc = fma(s.w, s.w, acc);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) red[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8u; ++i) total += red[i];
        partials[uint(t) * (rows + 1u) + r] = total;
    }
}

kernel void glm53p_hc_finalize_batched(
    device const float* partials [[buffer(0)]],  // [T][rows + 1]
    device const float* base_b   [[buffer(1)]],
    device const float* scale3   [[buffer(2)]],
    device float*       out_pre  [[buffer(3)]],  // [T][H]
    device float*       out_post [[buffer(4)]],  // [T][H]
    device float*       out_comb [[buffer(5)]],  // [T][H*H]
    constant uint&      hc_mult  [[buffer(6)]],
    constant uint&      flat     [[buffer(7)]],
    constant uint&      sinkhorn_iters [[buffer(8)]],
    constant float&     hc_eps   [[buffer(9)]],
    constant float&     rms_eps  [[buffer(10)]],
    constant uint&      T        [[buffer(11)]],
    uint t [[thread_position_in_grid]]
) {
    if (t >= T) return;
    const uint H = hc_mult;
    const uint rows = (2u + H) * H;
    device const float* p = partials + t * (rows + 1u);
    const float inv_norm = rsqrt(p[rows] / float(flat) + rms_eps);
    float mix[24];
    for (uint i = 0; i < rows; ++i) mix[i] = p[i] * inv_norm;
    const float pre_scale = scale3[0], post_scale = scale3[1], comb_scale = scale3[2];
    for (uint i = 0; i < H; ++i) {
        out_pre[t * H + i] = 1.0f / (1.0f + fast::exp(-(mix[i] * pre_scale + base_b[i]))) + hc_eps;
        out_post[t * H + i] = 2.0f / (1.0f + fast::exp(-(mix[H + i] * post_scale + base_b[H + i])));
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
        for (uint c2 = 0; c2 < H; ++c2) { comb[r * H + c2] = fast::exp(comb[r * H + c2] - mx); sum += comb[r * H + c2]; }
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
    for (uint i = 0; i < H * H; ++i) out_comb[t * H * H + i] = comb[i];
}

kernel void glm53p_hc_collapse_batched(
    device const half*  streams [[buffer(0)]],   // [T][H][hidden]
    device const float* pre     [[buffer(1)]],   // [T][H]
    device half*        x       [[buffer(2)]],   // [T][hidden]
    constant uint&      hc_mult [[buffer(3)]],
    constant uint&      hidden  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint d = gid.x, t = gid.y;
    if (d >= hidden) return;
    float acc = 0.0f;
    for (uint j = 0; j < hc_mult; ++j) {
        acc = fma(pre[t * hc_mult + j], float(streams[(t * hc_mult + j) * hidden + d]), acc);
    }
    x[t * hidden + d] = half(acc);
}

kernel void glm53p_hc_place_mix_batched(
    device const half*  streams     [[buffer(0)]],   // [T][H][hidden]
    device const half*  sub         [[buffer(1)]],   // [T][hidden]
    device const float* post        [[buffer(2)]],   // [T][H]
    device const float* comb        [[buffer(3)]],   // [T][H*H]
    device half*        out_streams [[buffer(4)]],   // [T][H][hidden]
    constant uint&      hc_mult     [[buffer(5)]],
    constant uint&      hidden      [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    if (d >= hidden) return;
    const uint H = hc_mult;
    const uint t = gid.y / H;
    const uint k = gid.y % H;
    float acc = post[t * H + k] * float(sub[t * hidden + d]);
    for (uint j = 0; j < H; ++j) {
        acc = fma(comb[t * H * H + j * H + k], float(streams[(t * H + j) * hidden + d]), acc);
    }
    out_streams[(t * H + k) * hidden + d] = half(acc);
}

// ---------------------------------------------------------------------------
// KDA over a chunk: one threadgroup per head walks the T tokens with the
// head's [D][D] state in registers (simdgroup sg owns rows dv = sg, sg + 8,
// ...; lane owns columns lane*per .. lane*per + per). Per token the same
// arithmetic as `glm53_kda_decode`, then the sigmoid-gated RMSNorm.
// ---------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53p_kda_chunk(
    device const half*   conv_out [[buffer(0)]],   // [T][3*H*D]
    device const half*   a        [[buffer(1)]],   // [T][H*D]
    device const half*   b        [[buffer(2)]],   // [T][H]
    device const half*   gate     [[buffer(3)]],   // [T][H*D]
    device const float*  A_log    [[buffer(4)]],   // [H]
    device const float*  dt_bias  [[buffer(5)]],   // [H*D]
    device const bfloat* o_norm   [[buffer(6)]],   // [D]
    device float*        state    [[buffer(7)]],   // [H][D][D]
    device half*         out      [[buffer(8)]],   // [T][H*D]
    constant uint&       H        [[buffer(9)]],
    constant uint&       D        [[buffer(10)]],
    constant uint&       T        [[buffer(11)]],
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
    threadgroup float scalars[2];
    if (h >= H) return;
    const uint qkv = H * D;
    const uint base = h * D;
    const uint per_lane = D / 32u;       // <= 4
    const uint rows_per_sg = D / 8u;     // <= 16

    // State into registers.
    float s[16][4];
    for (uint r = 0; r < rows_per_sg; ++r) {
        const uint dv = sg + r * 8u;
        device const float* srow = state + (uint(h) * D + dv) * D;
        for (uint i = 0; i < per_lane; ++i) s[r][i] = srow[lane * per_lane + i];
    }

    for (uint t = 0; t < T; ++t) {
        device const half* cv = conv_out + uint(t) * 3u * qkv;
        if (tid < D) {
            qs[tid] = float(cv[base + tid]);
            ks[tid] = float(cv[qkv + base + tid]);
            vs[tid] = float(cv[2u * qkv + base + tid]);
            const float g = exp(A_log[h]) * (float(a[uint(t) * qkv + base + tid]) + dt_bias[base + tid]);
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
        if (tid < D) { qs[tid] *= scalars[0]; ks[tid] *= scalars[1]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float beta = 1.0f / (1.0f + exp(-float(b[uint(t) * H + h])));
        for (uint r = 0; r < rows_per_sg; ++r) {
            const uint dv = sg + r * 8u;
            float kvm = 0.0f;
            for (uint i = 0; i < per_lane; ++i) {
                const uint idx = lane * per_lane + i;
                s[r][i] = s[r][i] * decay[idx];
                kvm = fma(s[r][i], ks[idx], kvm);
            }
            kvm = simd_sum(kvm);
            const float delta = (vs[dv] - kvm) * beta;
            float yv = 0.0f;
            for (uint i = 0; i < per_lane; ++i) {
                const uint idx = lane * per_lane + i;
                s[r][i] = fma(ks[idx], delta, s[r][i]);
                yv = fma(s[r][i], qs[idx], yv);
            }
            yv = simd_sum(yv);
            if (lane == 0) ys[dv] = yv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
            const float gv = 1.0f / (1.0f + exp(-float(gate[uint(t) * qkv + base + tid])));
            out[uint(t) * qkv + base + tid] = half(ys[tid] * scalars[0] * float(o_norm[tid]) * gv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < rows_per_sg; ++r) {
        const uint dv = sg + r * 8u;
        device float* srow = state + (uint(h) * D + dv) * D;
        for (uint i = 0; i < per_lane; ++i) srow[lane * per_lane + i] = s[r][i];
    }
}

// ---------------------------------------------------------------------------
// Dense causal latent attention for T queries: query t attends latent rows
// 0 ..< base + t + 1 (valid while base + T <= index_topk, where the model's
// own selection is exhaustive). Grid (H, T); same online softmax as
// `glm53_latent_attention`.
// ---------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53p_latent_attention_causal(
    device const half* q_lat    [[buffer(0)]],   // [T][H][kv]
    device const half* latents  [[buffer(1)]],   // [rows][kv]
    device half*       out      [[buffer(2)]],   // [T][H][kv]
    constant uint&     kv_dim   [[buffer(3)]],
    constant uint&     base     [[buffer(4)]],   // rows cached before the chunk
    constant uint&     H        [[buffer(5)]],
    constant float&    scale    [[buffer(6)]],
    uint2 tgp  [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]
) {
    const uint tid = tid2.x;
    threadgroup float acc_tg[8 * 512];
    threadgroup float m_tg[8];
    threadgroup float d_tg[8];
    const uint h = tgp.x;
    const uint t = tgp.y;
    const uint per = kv_dim / 32u;
    const uint n = base + t + 1u;
    float q[16];
    for (uint i = 0; i < per; ++i) q[i] = float(q_lat[(uint(t) * H + h) * kv_dim + lane * per + i]);
    float m = -FLT_MAX / 2.0f;
    float denom = 0.0f;
    float acc[16];
    for (uint i = 0; i < per; ++i) acc[i] = 0.0f;
    for (uint r = sg; r < n; r += 8u) {
        device const half* k = latents + uint(r) * kv_dim + lane * per;
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
        out[(uint(t) * H + h) * kv_dim + d] = half(o * inv);
    }
}

// LayerNorm with gain and bias for T rows (grid T threadgroups x 256).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void glm53p_layernorm_bias_batched(
    device const half*   x      [[buffer(0)]],   // [T][d]
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* bias   [[buffer(2)]],
    device half*         out    [[buffer(3)]],   // rows at out_row0 + t
    constant uint&       d      [[buffer(4)]],
    constant float&      eps    [[buffer(5)]],
    uint t    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float red[8];
    threadgroup float stats[2];
    device const half* row = x + uint(t) * d;
    float s = 0.0f;
    for (uint i = tid; i < d; i += 256u) s += float(row[i]);
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0.0f; for (uint i = 0; i < 8u; ++i) tot += red[i]; stats[0] = tot / float(d); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = stats[0];
    float v = 0.0f;
    for (uint i = tid; i < d; i += 256u) { const float c = float(row[i]) - mean; v = fma(c, c, v); }
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0.0f; for (uint i = 0; i < 8u; ++i) tot += red[i]; stats[1] = rsqrt(tot / float(d) + eps); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = stats[1];
    for (uint i = tid; i < d; i += 256u) {
        out[uint(t) * d + i] = half((float(row[i]) - mean) * inv * float(weight[i]) + float(bias[i]));
    }
}

// Pooled indexer keys for `count` consecutive complete pools starting at
// `first_pool`. Grid (dim, count). Same math as `glm53_pool_keys`.
kernel void glm53p_pool_keys_batched(
    device const half*   keys   [[buffer(0)]],
    device const half*   gates  [[buffer(1)]],
    device const bfloat* ape    [[buffer(2)]],
    device half*         pooled [[buffer(3)]],
    constant uint&       first_pool [[buffer(4)]],
    constant uint&       kp     [[buffer(5)]],
    constant uint&       dim    [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    if (d >= dim) return;
    const uint pool = first_pool + gid.y;
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
    for (uint c = 0; c < kp; ++c) acc = fma(logits[c] / sum, float(keys[(first + c) * dim + d]), acc);
    pooled[pool * dim + d] = half(acc);
}

// Router select for T tokens: one simdgroup per token (grid T x 32), the
// `glm53_router_select_k8_par` rule per token.
kernel void glm53p_router_select_k8_batched(
    device const float* logits [[buffer(0)]],        // [T][E]
    device const float* bias [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],          // [T][8]
    device half* out_weights [[buffer(3)]],          // [T][8]
    constant uint& num_experts [[buffer(4)]],
    constant float& route_scale [[buffer(5)]],
    uint t [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint K = 8;
    constexpr uint PER = 16;
    device const float* row = logits + uint(t) * num_experts;
    float key[PER];
    for (uint j = 0; j < PER; ++j) {
        const uint e = lane + 32u * j;
        key[j] = e < num_experts ? (1.0f / (1.0f + precise::exp(-row[e])) + bias[e]) : -INFINITY;
    }
    float chosen_score[K];
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint best_j = PER;
        for (uint j = 0; j < PER; ++j) { if (key[j] > best) { best = key[j]; best_j = j; } }
        const float top = simd_max(best);
        const uint mine = (best_j < PER && best == top) ? lane + 32u * best_j : 0xFFFFFFFFu;
        const uint winner = simd_min(mine);
        chosen_score[k] = 1.0f / (1.0f + precise::exp(-row[winner]));
        if (lane == 0) out_indices[uint(t) * K + k] = winner;
        if (winner == mine) key[best_j] = -INFINITY;
    }
    if (lane == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < K; ++k) sum += chosen_score[k];
        for (uint k = 0; k < K; ++k) out_weights[uint(t) * K + k] = half(chosen_score[k] / sum * route_scale);
    }
}

// ---------------------------------------------------------------------------
// INT8 affine group-64 GEMM on the simdgroup matrix units. Threadgroup: 4
// simdgroups x 32 rows (8 per simdgroup) x a tile of 32 tokens; grid
// (ceil(M/32), ceil(T/32)). N is walked in chunks of 256 staged as an
// activation tile [32 tokens][256] in threadgroup memory; per 8-wide k step a
// simdgroup dequantizes its 8 x 8 weight tile to half (2 elements per lane,
// `fma(q, s, b)` rounded once), loads it as A, loads four transposed 8 x 8
// activation tiles as B and accumulates C[8 rows][8 tokens] x 4 in fp32.
// The dequantized weight rounds to fp16 here (the GEMV keeps it fp32); the
// step is ~16x below the INT8 quantization step itself.
// ---------------------------------------------------------------------------
#include <metal_simdgroup_matrix>

// fp32 operands: the dequantized weight `fma(q, s, b)` stays fp32 as in the
// GEMV, the activations convert exactly, products and sums are fp32 — the
// GEMV's arithmetic up to accumulation order. Chunks of 128 columns (two
// groups) keep the fp32 activation tile inside threadgroup memory.
constant uint kG53PMMAChunk = 128;
constant uint kG53PXTStride = kG53PMMAChunk + 4;   // padded fp32 row stride (elements)

[[kernel, max_total_threads_per_threadgroup(128)]]
kernel void glm53p_int8_gemm_mma(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    X      [[buffer(3)]],
    device half*          Y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        T      [[buffer(7)]],
    constant uint&        x_stride [[buffer(8)]],
    constant uint&        y_stride [[buffer(9)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup float xt[kG53PTokenTile * kG53PXTStride];   // [32 t][132], 16.9 KB
    threadgroup float at[4][8 * 64];                         // per simdgroup A group, 2 KB each
    threadgroup float ct[4][8 * kG53PTokenTile];
    const uint tid = tid2.x;
    const uint m0 = tg.x * 32u + sg * 8u;
    const uint t0 = tg.y * kG53PTokenTile;
    const uint tn = min(kG53PTokenTile, T - t0);
    const uint n_groups = N / kG53PGroup;
    const uint a_row = lane / 4u;
    const uint a_col = (lane % 4u) * 16u;
    const uint m = min(m0 + a_row, M - 1u);
    device const uint8_t* W_row = W + uint(m) * N;
    device const bfloat*  s_row = scales + uint(m) * n_groups;
    device const bfloat*  b_row = biases + uint(m) * n_groups;
    threadgroup float* a_mine = at[sg] + (a_col / 8u) * 64u + a_row * 8u;

    simdgroup_float8x8 C[4];
    for (uint i = 0; i < 4u; ++i) C[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint n0 = 0; n0 < N; n0 += kG53PMMAChunk) {
        const uint nc = min(kG53PMMAChunk, N - n0);
        const uint segs = nc / 8u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < kG53PTokenTile * segs; e += 128u) {
            const uint row = e / segs;
            const uint seg = e - row * segs;
            // Rows past T are in-bounds scratch whose results are never stored.
            const half4 v0 = *((device const half4*)(X + (t0 + row) * x_stride + n0 + seg * 8u));
            const half4 v1 = *((device const half4*)(X + (t0 + row) * x_stride + n0 + seg * 8u + 4u));
            threadgroup float* dst = xt + row * kG53PXTStride + seg * 8u;
            *((threadgroup float4*)dst) = float4(v0);
            *((threadgroup float4*)(dst + 4)) = float4(v1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gi = 0; gi < nc / kG53PGroup; ++gi) {
            const uint g = n0 / kG53PGroup + gi;
            const float s = float(s_row[g]);
            const float b = float(b_row[g]);
            const uint4 w16 = *((device const uint4*)(W_row + g * kG53PGroup + a_col));
            const uint words[4] = { w16.x, w16.y, w16.z, w16.w };
#pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                const uint w = words[j];
                threadgroup float* dst = a_mine + (j / 2u) * 64u + (j % 2u) * 4u;
                dst[0] = fma(float(w & 0xFFu), s, b);
                dst[1] = fma(float((w >> 8) & 0xFFu), s, b);
                dst[2] = fma(float((w >> 16) & 0xFFu), s, b);
                dst[3] = fma(float(w >> 24), s, b);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 A[8];
#pragma unroll
            for (uint kk = 0; kk < 8u; ++kk) simdgroup_load(A[kk], at[sg] + kk * 64u, 8);
            threadgroup const float* xg = xt + gi * kG53PGroup;
#pragma unroll
            for (uint tt = 0; tt < 4u; ++tt) {
#pragma unroll
                for (uint kk = 0; kk < 8u; ++kk) {
                    simdgroup_float8x8 B;
                    simdgroup_load(B, xg + tt * 8u * kG53PXTStride + kk * 8u, kG53PXTStride, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(C[tt], A[kk], B, C[tt]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint tt = 0; tt < 4u; ++tt) {
        simdgroup_store(C[tt], ct[sg] + tt * 8u, kG53PTokenTile);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint out_row = m0 + a_row;
    if (out_row < M) {
        for (uint j = 0; j < 8u; ++j) {
            const uint t = (lane % 4u) * 8u + j;
            if (t < tn) Y[(t0 + t) * y_stride + out_row] = half(ct[sg][a_row * kG53PTokenTile + t]);
        }
    }
}
