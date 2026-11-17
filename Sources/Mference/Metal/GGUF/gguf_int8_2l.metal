// INT8-2L resident GEMV (WS4 U1 §3b, U3). Compiled on demand (module
// "gguf_int8_2l"). A tensor of `rows × N` weights has three dense regions in
// block order: `values` (int8, natural order), `d` (fp16 bits, one per BD
// weights: 256 for Q6_K, 32 for Q8_0) and, for Q6_K, `m` (int8, one per 16).
// A weight is d · m · value (m ≡ 1 without `m`), exact in fp32.
//
// One simdgroup per output row; a lane takes 8 consecutive weights, which
// always share one d and one m (BD ≥ 32, BM = 16), so it scales once.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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

template <typename T>
static inline void int8_2l_gemm_t8(
    device const char* values, device const ushort* dbits, device const char* m,
    constant Int8TwoLevelParams& p, device const half* x, device T* y,
    constant Int8TwoLevelBatchParams& bp, uint2 tg, uint sg_idx, uint lane)
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
        if (lane == 0) { y[ulong(t0 + t) * bp.y_stride + row] = T(v); }
    }
}

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
    int8_2l_gemm_t8<half>(values, dbits, m, p, x, y, bp, tg, sg_idx, lane);
}

// FP32-store twin for the consumers that keep FP32 pre-activations (the
// hyper-connection low-rank and mix gates): each token's row equals
// `gguf_int8_2l_gemv_f32out` on that token alone, bit for bit.
kernel void gguf_int8_2l_gemm_t8_f32out(
    device const char* values [[buffer(0)]],
    device const ushort* dbits [[buffer(1)]],
    device const char* m [[buffer(2)]],
    constant Int8TwoLevelParams& p [[buffer(3)]],
    device const half* x [[buffer(4)]],
    device float* y [[buffer(5)]],
    constant Int8TwoLevelBatchParams& bp [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    int8_2l_gemm_t8<float>(values, dbits, m, p, x, y, bp, tg, sg_idx, lane);
}

// Simdgroup-matrix prefill GEMM (WS1 (b), class T): Y[t][row] = W[row] · X[t]
// for a tile of 64 tokens x 128 output rows per threadgroup, four simdgroups
// as 2 x 2 (32 tokens x 64 rows each, 4 x 8 float 8x8 accumulators). K steps
// by 32: X is staged as half (exact), W is dequantized to (d · m) · v into a
// float threadgroup tile (every weight exact), then 8x8 multiply-accumulates
// in float. The K reduction order differs from the GEMV's, so this is not
// bit-identical to it. Requires n % 32 == 0.
#define I8MMA_UNROLL _Pragma("clang loop unroll(full)")
constant constexpr uint kI8Tokens = 64;
constant constexpr uint kI8Rows = 128;
constant constexpr uint kI8K = 32;
constant constexpr uint kI8Ldk = kI8K + 8;

static inline ushort2 i8mma_frag_coord(ushort lane) {
    const ushort qid = lane / 4;
    return ushort2((qid & 4) + ((lane / 2) % 4), (qid & 2) * 2 + (lane % 2) * 2);
}

static inline void int8_2l_gemm_mma(
    device const char* values, device const ushort* dbits, device const char* m,
    constant Int8TwoLevelParams& p, device const half* x, device half* y,
    constant Int8TwoLevelBatchParams& bp,
    threadgroup half* x_tile, threadgroup float* w_tile,
    uint2 tg, ushort tid, ushort sg, ushort lane)
{
    const uint t_base = tg.y * kI8Tokens;
    const uint r_base = tg.x * kI8Rows;
    const uint n = p.n;

    // X loader: token tid / 2, k half (tid % 2) * 16.
    const uint x_tok = tid / 2u, x_k = (tid % 2u) * (kI8K / 2u);
    const bool x_valid = t_base + x_tok < bp.tokens;
    device const half* x_src = x + ulong(x_valid ? t_base + x_tok : 0u) * bp.x_stride + x_k;
    // W loader: row tid, all 32 k of the step.
    const uint w_row = r_base + tid;
    const bool w_valid = w_row < p.rows;
    device const char* w_src = values + ulong(w_valid ? w_row : 0u) * n;
    const ulong w_base = ulong(w_valid ? w_row : 0u) * n;

    simdgroup_float8x8 acc[4][8];
    I8MMA_UNROLL for (uint i = 0; i < 4; ++i) I8MMA_UNROLL for (uint j = 0; j < 8; ++j) {
        acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint sg_m = (sg / 2u) * 32u, sg_n = (sg % 2u) * 64u;
    const ushort2 fc = i8mma_frag_coord(lane);

    for (uint k0 = 0; k0 < n; k0 += kI8K) {
        threadgroup half4* xd = reinterpret_cast<threadgroup half4*>(x_tile + x_tok * kI8Ldk + x_k);
        if (x_valid) {
            device const half4* xs = reinterpret_cast<device const half4*>(x_src + k0);
            I8MMA_UNROLL for (uint i = 0; i < kI8K / 8u; ++i) xd[i] = xs[i];
        } else {
            I8MMA_UNROLL for (uint i = 0; i < kI8K / 8u; ++i) xd[i] = half4(0.0h);
        }
        threadgroup float* wd = w_tile + tid * kI8Ldk;
        if (w_valid) {
            const ulong gw = w_base + k0;
            const float d = float(as_type<half>(dbits[gw >> p.bd_shift]));
            const float s0 = p.has_m != 0 ? d * float(m[gw >> 4]) : d;
            const float s1 = p.has_m != 0 ? d * float(m[(gw + 16) >> 4]) : d;
            device const char4* vs = reinterpret_cast<device const char4*>(w_src + k0);
            I8MMA_UNROLL for (uint i = 0; i < kI8K / 4u; ++i) {
                const char4 v = vs[i];
                const float s = i < 4u ? s0 : s1;
                wd[4 * i + 0] = s * float(v.x); wd[4 * i + 1] = s * float(v.y);
                wd[4 * i + 2] = s * float(v.z); wd[4 * i + 3] = s * float(v.w);
            }
        } else {
            I8MMA_UNROLL for (uint i = 0; i < kI8K; ++i) wd[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        I8MMA_UNROLL for (uint kk = 0; kk < kI8K; kk += 8) {
            simdgroup_float8x8 a[4], b[8];
            I8MMA_UNROLL for (uint i = 0; i < 4; ++i) {
                threadgroup const half* src = x_tile + (sg_m + i * 8u + fc.x) * kI8Ldk + kk + fc.y;
                a[i].thread_elements()[0] = float(src[0]);
                a[i].thread_elements()[1] = float(src[1]);
            }
            I8MMA_UNROLL for (uint j = 0; j < 8; ++j) {
                const uint at = (sg_n + j * 8u + fc.y) * kI8Ldk + kk + fc.x;
                b[j].thread_elements()[0] = float(w_tile[at]);
                b[j].thread_elements()[1] = float(w_tile[at + kI8Ldk]);
            }
            I8MMA_UNROLL for (uint i = 0; i < 4; ++i) I8MMA_UNROLL for (uint j = 0; j < 8; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage the 64 x 128 half result through the weight tile's memory, then
    // store with token and row bounds.
    constexpr uint kStage = kI8Rows + 8;
    threadgroup half* stage = reinterpret_cast<threadgroup half*>(w_tile);
    I8MMA_UNROLL for (uint i = 0; i < 4; ++i) I8MMA_UNROLL for (uint j = 0; j < 8; ++j) {
        const uint r = sg_m + i * 8u + fc.x, c = sg_n + j * 8u + fc.y;
        stage[r * kStage + c] = half(acc[i][j].thread_elements()[0]);
        stage[r * kStage + c + 1] = half(acc[i][j].thread_elements()[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint out_tok = tid / 2u, out_col = (tid % 2u) * (kI8Rows / 2u);
    if (t_base + out_tok < bp.tokens) {
        device half* dst = y + ulong(t_base + out_tok) * bp.y_stride + r_base + out_col;
        threadgroup const half* src = stage + out_tok * kStage + out_col;
        const uint live = p.rows > r_base + out_col ? min(kI8Rows / 2u, p.rows - r_base - out_col) : 0u;
        for (uint i = 0; i < live; ++i) dst[i] = src[i];
    }
}

kernel void gguf_int8_2l_gemm_mma_wf32_f16out(
    device const char* values [[buffer(0)]],
    device const ushort* dbits [[buffer(1)]],
    device const char* m [[buffer(2)]],
    constant Int8TwoLevelParams& p [[buffer(3)]],
    device const half* x [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant Int8TwoLevelBatchParams& bp [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    // 128 x 40 floats (20,480 B) also holds the 64 x 136 half stage.
    threadgroup float w_tile[kI8Rows * kI8Ldk];
    threadgroup half x_tile[kI8Tokens * kI8Ldk];
    int8_2l_gemm_mma(values, dbits, m, p, x, y, bp, x_tile, w_tile, tg, tid, sg, lane);
}

#undef I8MMA_UNROLL

// Q6_K row gather for `token_embd` (WS4 U1 R2): the table stays native Q6_K
// (block_q6_K: ql[128], qh[64], int8 scales[16], fp16 d; 210 B per 256), and
// each selected row is decoded as ggml's dequantize_row_q6_K does,
// (d · scale) · q with q = 6-bit code − 32, exact in fp32, then rounded once
// to half (the contract of flashnext_embed_row_bf16). out[t][e] for
// ids[t], t < tokens; one thread per (element, token).
kernel void gguf_q6k_embed_rows(
    device const uchar* table [[buffer(0)]],   // [vocab][hidden/256 blocks]
    device const uint* ids [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint e = gid.x, t = gid.y;
    if (e >= hidden || t >= tokens) { return; }
    device const uchar* b = table + (ulong(ids[t]) * (hidden / 256) + e / 256) * 210;
    const uint i = e % 256, h = i / 128, r = i % 128, k = r / 32, l = r % 32;
    const uint ql = b[h * 64 + l + (k & 1) * 32];
    const uint low4 = (k < 2) ? (ql & 0xF) : (ql >> 4);
    const uint high2 = (b[128 + h * 32 + l] >> (2 * k)) & 0x3;
    const int q = int(low4 | high2 << 4) - 32;
    const float d = float(as_type<half>(ushort(b[208] | (ushort(b[209]) << 8))));
    const float sc = float(as_type<char>(b[192 + i / 16]));
    out[ulong(t) * hidden + e] = half(d * sc * float(q));
}
