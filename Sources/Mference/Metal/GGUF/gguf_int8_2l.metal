// INT8-2L resident GEMV (WS4 U1 §3b, U3). Compiled on demand (module
// "gguf_int8_2l"). A tensor of `rows × N` weights has three dense regions in
// block order: `values` (int8, natural order), `d` (fp16 bits, one per BD
// weights: 256 for Q6_K, 32 for Q8_0) and, for Q6_K, `m` (int8, one per 16).
// A weight is d · m · value (m ≡ 1 without `m`), exact in fp32.
//
// One simdgroup per output row; a lane takes 8 consecutive weights, which
// always share one d and one m (BD ≥ 32, BM = 16), so it scales once.

#include <metal_stdlib>
using namespace metal;

struct Int8TwoLevelParams {
    uint rows;
    uint n;          // weights per row; a multiple of BD and of 8
    uint bd_shift;   // log2(BD)
    uint has_m;      // 1 for Q6_K
};

static inline float int8_2l_row_dot(
    device const char* values, device const ushort* dbits, device const char* m,
    constant Int8TwoLevelParams& p, device const half* x, uint row, uint lane)
{
    const uint n = p.n;
    device const char* row_v = values + ulong(row) * n;
    const ulong row_w = ulong(row) * n;
    float acc = 0.0f;
    for (uint c = lane; c < n / 8; c += 32) {
        const uint w = 8 * c;
        const char4 va = *((device const char4*)(row_v + w));
        const char4 vb = *((device const char4*)(row_v + w + 4));
        const half4 xa = *((device const half4*)(x + w));
        const half4 xb = *((device const half4*)(x + w + 4));
        float s = 0.0f;
        s = fma(float(va.x), float(xa.x), s); s = fma(float(va.y), float(xa.y), s);
        s = fma(float(va.z), float(xa.z), s); s = fma(float(va.w), float(xa.w), s);
        s = fma(float(vb.x), float(xb.x), s); s = fma(float(vb.y), float(xb.y), s);
        s = fma(float(vb.z), float(xb.z), s); s = fma(float(vb.w), float(xb.w), s);
        const ulong gw = row_w + w;
        float scale = float(as_type<half>(dbits[gw >> p.bd_shift]));
        if (p.has_m != 0) { scale *= float(m[gw >> 4]); }
        acc = fma(scale, s, acc);
    }
    return simd_sum(acc);
}

kernel void gguf_int8_2l_gemv_f16out(
    device const char* values [[buffer(0)]],
    device const ushort* dbits [[buffer(1)]],
    device const char* m [[buffer(2)]],
    constant Int8TwoLevelParams& p [[buffer(3)]],
    device const half* x [[buffer(4)]],
    device half* y [[buffer(5)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tg_idx * 8 + sg_idx;
    if (row >= p.rows) { return; }
    const float v = int8_2l_row_dot(values, dbits, m, p, x, row, lane);
    if (lane == 0) { y[row] = half(v); }
}

kernel void gguf_int8_2l_gemv_f32out(
    device const char* values [[buffer(0)]],
    device const ushort* dbits [[buffer(1)]],
    device const char* m [[buffer(2)]],
    constant Int8TwoLevelParams& p [[buffer(3)]],
    device const half* x [[buffer(4)]],
    device float* y [[buffer(5)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tg_idx * 8 + sg_idx;
    if (row >= p.rows) { return; }
    const float v = int8_2l_row_dot(values, dbits, m, p, x, row, lane);
    if (lane == 0) { y[row] = v; }
}

// Batched form for prefill: Y[t][row] = W[row] · X[t] for `tokens` rows of X
// (row stride x_stride halves) and Y (row stride y_stride). One simdgroup per
// (output row, group of 8 tokens): each 8-weight chunk and its scale are
// loaded once and feed eight FP32 accumulators. Per token the arithmetic is
// the GEMV's (same chunking, same per-chunk FMA order, same simd_sum), so a
// batched row equals the single-token GEMV of that token bit for bit.
struct Int8TwoLevelBatchParams {
    uint tokens;
    uint x_stride;
    uint y_stride;
    uint pad;
};

kernel void gguf_int8_2l_gemm_t8_f16out(
    device const char* values [[buffer(0)]],
    device const ushort* dbits [[buffer(1)]],
    device const char* m [[buffer(2)]],
    constant Int8TwoLevelParams& p [[buffer(3)]],
    device const half* x [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant Int8TwoLevelBatchParams& bp [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tg.x * 8 + sg_idx;
    const uint t0 = tg.y * 8;
    if (row >= p.rows || t0 >= bp.tokens) { return; }
    const uint nt = min(8u, bp.tokens - t0);
    const uint n = p.n;
    device const char* row_v = values + ulong(row) * n;
    const ulong row_w = ulong(row) * n;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint c = lane; c < n / 8; c += 32) {
        const uint w = 8 * c;
        const char4 va = *((device const char4*)(row_v + w));
        const char4 vb = *((device const char4*)(row_v + w + 4));
        const ulong gw = row_w + w;
        float scale = float(as_type<half>(dbits[gw >> p.bd_shift]));
        if (p.has_m != 0) { scale *= float(m[gw >> 4]); }
        for (uint t = 0; t < nt; ++t) {
            device const half* xt = x + ulong(t0 + t) * bp.x_stride;
            const half4 xa = *((device const half4*)(xt + w));
            const half4 xb = *((device const half4*)(xt + w + 4));
            float s = 0.0f;
            s = fma(float(va.x), float(xa.x), s); s = fma(float(va.y), float(xa.y), s);
            s = fma(float(va.z), float(xa.z), s); s = fma(float(va.w), float(xa.w), s);
            s = fma(float(vb.x), float(xb.x), s); s = fma(float(vb.y), float(xb.y), s);
            s = fma(float(vb.z), float(xb.z), s); s = fma(float(vb.w), float(xb.w), s);
            acc[t] = fma(scale, s, acc[t]);
        }
    }
    for (uint t = 0; t < nt; ++t) {
        const float v = simd_sum(acc[t]);
        if (lane == 0) { y[ulong(t0 + t) * bp.y_stride + row] = half(v); }
    }
}
