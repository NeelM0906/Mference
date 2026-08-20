// DFlash2 block-diffusion drafter kernels (Qwen 3.8 target).
//
// The drafter is 5 BF16 transformer layers that predict a whole block of
// masked future tokens in one forward pass, conditioned on target hidden
// states injected into every layer's K/V (z-lab dflash `model_mlx.py`
// semantics). Everything here is fp32-accumulated; the drafter only has to
// be *close* to the reference — draft quality affects acceptance length,
// never output bytes (the target verify pass owns correctness).

#include <metal_stdlib>
using namespace metal;

// BF16 weights [M, N] row-major x FP16 rows [T, N] -> FP16 [T, M], weights
// read once for all T rows (the multi-x contract). One SIMD per output row,
// four rows per threadgroup, 128 threads. T <= 8; callers chunk larger row
// counts.
kernel void dflash2_bf16_gemv_multix(
    device const bfloat* W [[buffer(0)]],
    device const half* X [[buffer(1)]],
    device half* Y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint m = tg * 4u + sg;
    if (m >= M) return;
    device const bfloat* row = W + ulong(m) * N;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = lane; i < N; i += 32u) {
        const float w = float(row[i]);
        for (uint t = 0; t < T; ++t) {
            acc[t] = fma(w, float(X[ulong(t) * N + i]), acc[t]);
        }
    }
    for (uint t = 0; t < T; ++t) {
        const float s = simd_sum(acc[t]);
        if (lane == 0) Y[ulong(t) * M + m] = half(s);
    }
}

// FP32-output twin of the multi-x GEMV: the o/down projections feed the
// drafter's FP32 residual stream (their sums exceed the FP16 range on
// outlier channels, the same failure mode the Inkling residual fix
// documents).
kernel void dflash2_bf16_gemv_multix_f32out(
    device const bfloat* W [[buffer(0)]],
    device const half* X [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint m = tg * 4u + sg;
    if (m >= M) return;
    device const bfloat* row = W + ulong(m) * N;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = lane; i < N; i += 32u) {
        const float w = float(row[i]);
        for (uint t = 0; t < T; ++t) {
            acc[t] = fma(w, float(X[ulong(t) * N + i]), acc[t]);
        }
    }
    for (uint t = 0; t < T; ++t) {
        const float s = simd_sum(acc[t]);
        if (lane == 0) Y[ulong(t) * M + m] = s;
    }
}

// hidden[i] += delta[i], both FP32 — the drafter residual stream.
kernel void dflash2_residual_add_f32f32(
    device float* hidden [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) return;
    hidden[i] += delta[i];
}

// Block attention: T block queries over [ctx keys | block keys].
// Context part carries the sliding-window mask in *kept-row index space*
// (query index = ctxLen + t), the block part is fully bidirectional
// (`is_causal: false` in the drafter config). Q/K arrive per-head normed
// and roped. One SIMD per (query, q head); GQA via head ratio. Online
// softmax, fp32 accumulators, sequential keys — the blocked-streamed
// prefill shape from attention.metal.
kernel void dflash2_block_attention(
    device const half* Q [[buffer(0)]],       // [T, HQ*128]
    device const half* ctxK [[buffer(1)]],    // [ctxLen, HKV*128]
    device const half* ctxV [[buffer(2)]],    // [ctxLen, HKV*128]
    device const half* blkK [[buffer(3)]],    // [T, HKV*128]
    device const half* blkV [[buffer(4)]],    // [T, HKV*128]
    device half* out [[buffer(5)]],           // [T, HQ*128]
    constant uint& T [[buffer(6)]],
    constant uint& ctxLen [[buffer(7)]],
    constant uint& window [[buffer(8)]],
    constant uint& numQHeads [[buffer(9)]],
    constant uint& numKVHeads [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint HD = 128u;
    const uint h = tgid.x;
    const uint t = tgid.y;
    if (h >= numQHeads || t >= T) return;
    const uint kvh = h / (numQHeads / numKVHeads);

    device const half* q = Q + (ulong(t) * numQHeads + h) * HD;
    float4 qv;
    const uint e0 = lane * 4u;
    qv = float4(float(q[e0]), float(q[e0 + 1u]), float(q[e0 + 2u]), float(q[e0 + 3u]));

    float m = -INFINITY;
    float d = 0.0f;
    float4 o = float4(0.0f);
    const uint total = ctxLen + T;
    for (uint j = 0; j < total; ++j) {
        device const half* k;
        device const half* v;
        if (j < ctxLen) {
            // Sliding-window visibility in kept-row index space.
            if ((ctxLen + t) - j >= window) continue;
            k = ctxK + (ulong(j) * numKVHeads + kvh) * HD;
            v = ctxV + (ulong(j) * numKVHeads + kvh) * HD;
        } else {
            k = blkK + (ulong(j - ctxLen) * numKVHeads + kvh) * HD;
            v = blkV + (ulong(j - ctxLen) * numKVHeads + kvh) * HD;
        }
        float partial = qv.x * float(k[e0]) + qv.y * float(k[e0 + 1u])
                      + qv.z * float(k[e0 + 2u]) + qv.w * float(k[e0 + 3u]);
        const float score = simd_sum(partial) * scale;
        const float mNew = max(m, score);
        const float corr = fast::exp(m - mNew);
        const float w = fast::exp(score - mNew);
        d = d * corr + w;
        o = o * corr + w * float4(float(v[e0]), float(v[e0 + 1u]),
                                  float(v[e0 + 2u]), float(v[e0 + 3u]));
        m = mNew;
    }
    device half* y = out + (ulong(t) * numQHeads + h) * HD;
    const float inv = d > 0.0f ? (1.0f / d) : 0.0f;
    y[e0]      = half(o.x * inv);
    y[e0 + 1u] = half(o.y * inv);
    y[e0 + 2u] = half(o.z * inv);
    y[e0 + 3u] = half(o.w * inv);
}

// Grouped dynamic causal convolution over the block dimension:
//   out[i,c] = sum_t (base[t,c] + dyn[i, plane*K*G + t*G + c/groupSize])
//              * x[i-t, c],   x[i-t] = 0 for i-t < 0.
// `base` points at one [K, H] plane of the checkpoint's [2, K, H] tensor;
// `plane` selects the prepare (0) or finish (1) slice of the projected
// dynamic kernels. One thread per (channel, block row).
kernel void dflash2_dynconv(
    device const half* x [[buffer(0)]],       // [T, H]
    device const half* dyn [[buffer(1)]],     // [T, 2*K*G]
    device const bfloat* base [[buffer(2)]],  // [K, H]
    device half* out [[buffer(3)]],           // [T, H]
    constant uint& H [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& groupSize [[buffer(6)]],
    constant uint& plane [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint i = gid.y;
    if (c >= H) return;
    const uint G = H / groupSize;
    const uint g = c / groupSize;
    device const half* dynRow = dyn + ulong(i) * (2u * K * G) + plane * K * G;
    float acc = 0.0f;
    for (uint t = 0; t < K && t <= i; ++t) {
        const float kernelValue = float(base[t * H + c]) + float(dynRow[t * G + g]);
        acc = fma(kernelValue, float(x[ulong(i - t) * H + c]), acc);
    }
    out[ulong(i) * H + c] = half(acc);
}

// FP32-in/FP32-out twin of the dynamic conv, applied to the o/down
// projection outputs on the FP32 residual path (the "finish" site).
kernel void dflash2_dynconv_f32io(
    device const float* x [[buffer(0)]],      // [T, H]
    device const half* dyn [[buffer(1)]],     // [T, 2*K*G]
    device const bfloat* base [[buffer(2)]],  // [K, H]
    device float* out [[buffer(3)]],          // [T, H]
    constant uint& H [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& groupSize [[buffer(6)]],
    constant uint& plane [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint i = gid.y;
    if (c >= H) return;
    const uint G = H / groupSize;
    const uint g = c / groupSize;
    device const half* dynRow = dyn + ulong(i) * (2u * K * G) + plane * K * G;
    float acc = 0.0f;
    for (uint t = 0; t < K && t <= i; ++t) {
        const float kernelValue = float(base[t * H + c]) + float(dynRow[t * G + g]);
        acc = fma(kernelValue, x[ulong(i - t) * H + c], acc);
    }
    out[ulong(i) * H + c] = acc;
}

// Per-row top-16 of an FP16 logit row. 128 threads keep strided local
// top-16 lists, then one thread merges them. Candidate-set semantics: the
// selector scores all 16, so intra-set order is irrelevant.
kernel void dflash2_topk16(
    device const half* logits [[buffer(0)]],  // [T, V]
    device uint* outIdx [[buffer(1)]],        // [T, 16]
    device float* outVal [[buffer(2)]],       // [T, 16]
    constant uint& V [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    constexpr uint KTOP = 16u;
    float bestVal[KTOP];
    uint bestIdx[KTOP];
    for (uint i = 0; i < KTOP; ++i) { bestVal[i] = -INFINITY; bestIdx[i] = 0u; }
    device const half* src = logits + ulong(row) * V;
    for (uint i = tid; i < V; i += threads) {
        const float v = float(src[i]);
        if (v <= bestVal[KTOP - 1u]) continue;
        uint pos = KTOP - 1u;
        while (pos > 0u && bestVal[pos - 1u] < v) {
            bestVal[pos] = bestVal[pos - 1u];
            bestIdx[pos] = bestIdx[pos - 1u];
            --pos;
        }
        bestVal[pos] = v;
        bestIdx[pos] = i;
    }
    threadgroup float shVal[128 * KTOP];
    threadgroup uint shIdx[128 * KTOP];
    for (uint i = 0; i < KTOP; ++i) {
        shVal[tid * KTOP + i] = bestVal[i];
        shIdx[tid * KTOP + i] = bestIdx[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;
    float outV[KTOP];
    uint outI[KTOP];
    for (uint i = 0; i < KTOP; ++i) { outV[i] = -INFINITY; outI[i] = 0u; }
    const uint entries = threads * KTOP;
    for (uint i = 0; i < entries; ++i) {
        const float v = shVal[i];
        if (v <= outV[KTOP - 1u]) continue;
        uint pos = KTOP - 1u;
        while (pos > 0u && outV[pos - 1u] < v) {
            outV[pos] = outV[pos - 1u];
            outI[pos] = outI[pos - 1u];
            --pos;
        }
        outV[pos] = v;
        outI[pos] = shIdx[i];
    }
    for (uint i = 0; i < KTOP; ++i) {
        outIdx[row * KTOP + i] = outI[i];
        outVal[row * KTOP + i] = outV[i];
    }
}

// Strided tap gather: copy T rows of [T, D] into a [., tapCount*D]
// staging matrix at column tapIndex*D, row offset dstRow. Used to build
// the concatenated target-hidden features without a CPU round trip.
kernel void dflash2_tap_gather(
    device const half* src [[buffer(0)]],   // [T, D]
    device half* dst [[buffer(1)]],         // [., tapCount*D]
    constant uint& D [[buffer(2)]],
    constant uint& tapCount [[buffer(3)]],
    constant uint& tapIndex [[buffer(4)]],
    constant uint& dstRow [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= D) return;
    dst[ulong(dstRow + t) * (tapCount * D) + tapIndex * D + c] =
        src[ulong(t) * D + c];
}
