#include <metal_stdlib>
using namespace metal;

#ifndef MFERENCE_GEMMA_SOURCE_FP16
#define MFERENCE_GEMMA_SOURCE_FP16
// Preserve the source activation boundary before learned scaling. The original
// Gemma checkpoint keeps its existing fused arithmetic unless explicitly set.
constant bool FC_GEMMA_SOURCE_FP16 [[function_constant(110)]];
constant bool kGemmaSourceFP16 = is_function_constant_defined(FC_GEMMA_SOURCE_FP16)
    ? FC_GEMMA_SOURCE_FP16 : false;
static inline half gemma_weighted_norm(float x, float inv, float weight) {
    return kGemmaSourceFP16 ? half(x * inv) * half(weight) : half(x * inv * weight);
}
static inline half gemma_scaled_embedding(float value, float scale) {
    return kGemmaSourceFP16 ? half(value) * half(scale) : half(value * scale);
}

// MLX's quantized GEMV forms its affine-bias input sum in FP16 quads.
static inline float gemma_source_quad_sum(half4 x) {
    const half a = x.x + x.y;
    const half b = a + x.z;
    return float(half(b + x.w));
}
static inline half gemma_source_geglu(float gate_value, float up_value) {
    // Gate/up projections are FP16 tensors in the source, even when fused.
    volatile half gate = half(gate_value);
    volatile half up = half(up_value);
    // Preserve source half stores across fast-math contraction, particularly
    // 1 + tanh(x): its rounded negative tail is exactly zero in the source.
    volatile half cube = half(float(gate) * float(gate) * float(gate));
    volatile half term = half(0.044715f) * cube;
    volatile half sum = gate + term;
    volatile half inner = half(0.7978845608028654f) * sum;
    // FP16 tanh has already rounded to +/-1 at these bounds. Keep that
    // exact source result while avoiding fast-tanh's positive exp overflow.
    volatile half curve = half(tanh(clamp(float(inner), -20.0f, 20.0f)));
    volatile half shifted = half(1.0h + curve);
    volatile half scaled = half(0.5h * gate);
    volatile half activation = scaled * shifted;
    return half(activation * up);
}
static inline float gemma_source_bias_correction(
    device const half* x, device const bfloat* biases, uint width, uint group_size
) {
    float correction = 0.0f;
    for (uint k = 0; k < width; k += 4u) {
        const half4 quad(x[k], x[k + 1u], x[k + 2u], x[k + 3u]);
        const float exact = float(quad.x) + float(quad.y) + float(quad.z) + float(quad.w);
        correction = fma(float(biases[k / group_size]),
                        gemma_source_quad_sum(quad) - exact, correction);
    }
    return correction;
}
#endif

// ============================================================================
// dequant_int4 — MLX `affine` 4-bit dequant.
//
// Layout (per row of length N):
//   W       : N/2 bytes. Low nibble of byte k = component 2k (unsigned 0..15),
//             high nibble = component 2k+1.
//   scales/biases: N/G BF16, with G=32 for Gemma QAT and G=64 by default.
//   value   : w[i] = float(nibble[i]) * scale[i/G] + bias[i/G].
//
// Affine factoring for GEMV (sum over one storage group):
//   sum_k (q_k * s + b) * x_k = s * sum_k(q_k * x_k) + b * sum_k x_k
// so scale and bias each cost one mul + one FMA per group instead of per
// element; the per-element inner loop keeps the scalar path's FMA count.
// ============================================================================

#ifndef MFERENCE_AFFINE_GROUP_SIZE
#define MFERENCE_AFFINE_GROUP_SIZE
constant uint FC_AFFINE_GROUP_SIZE [[function_constant(108)]];
constant uint kAffineGroupSize = is_function_constant_defined(FC_AFFINE_GROUP_SIZE)
    ? FC_AFFINE_GROUP_SIZE : 64u;
#endif
constant uint kGroupSize = kAffineGroupSize;
constant uint FC_INT4_M [[function_constant(20)]];
constant uint FC_INT4_N [[function_constant(21)]];
constant bool FC_INT4_USE_FC [[function_constant(22)]];
constant uint FC_INT4_QKV_MQ [[function_constant(23)]];
constant uint FC_INT4_QKV_MKV [[function_constant(24)]];
constant uint FC_INT4_QKV_N [[function_constant(25)]];
constant bool FC_INT4_QKV_USE_FC [[function_constant(26)]];
constant bool FC_SHARED_INT4_ACT_SILU [[function_constant(27)]];

static inline uint int4_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_M)) ? FC_INT4_M : M;
}

static inline uint int4_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_N)) ? FC_INT4_N : N;
}

static inline uint int4_qkv_fc_mq(constant uint& Mq) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MQ)) ? FC_INT4_QKV_MQ : Mq;
}

static inline uint int4_qkv_fc_mkv(constant uint& Mkv) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MKV)) ? FC_INT4_QKV_MKV : Mkv;
}

static inline uint int4_qkv_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_N)) ? FC_INT4_QKV_N : N;
}

inline uint nib_lo(uint8_t b) { return uint(b & 0x0F); }
inline uint nib_hi(uint8_t b) { return uint(b >> 4); }

#ifndef MFERENCE_GEMMA_SOURCE_PROJECTION
#define MFERENCE_GEMMA_SOURCE_PROJECTION
// QAT's source GEMV rounds the input sum in half quads, accumulates four
// products at a time, and adds each complete affine sub-result. Callers of
// this path compile this module with safe math: fast-math compilation does
// not preserve the source's FP16 rounding boundaries.
static inline float gemma_source_projection_row(
    device const uint8_t* weights,
    device const bfloat* scales,
    device const bfloat* biases,
    device const half* x,
    uint width, uint group_size, bool fast_shape, uint lane
) {
    device const ushort* packed_weights = (device const ushort*)weights;
    const uint values = fast_shape ? 16u : 8u;
    float result = 0.0f;
    for (uint base = lane * values; base < width; base += 32u * values) {
        float sum = 0.0f, dot = 0.0f;
        for (uint i = 0; i < values; i += 4u) {
            const uint k = base + i;
            const ushort packed = packed_weights[k / 4u];
            sum += x[k] + x[k + 1u] + x[k + 2u] + x[k + 3u];
            dot += float(x[k]) * (packed & 15u)
                + (float(x[k + 1u]) / 16.0f) * (packed & 240u)
                + (float(x[k + 2u]) / 256.0f) * (packed & 3840u)
                + (float(x[k + 3u]) / 4096.0f) * (packed & 61440u);
        }
        const uint group = base / group_size;
        result += float(half(scales[group])) * dot + sum * float(half(biases[group]));
    }
    return simd_sum(result);
}
#endif


kernel void embed_lookup_int4(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/2] nibbles
    device const bfloat*  scales    [[buffer(1)]],   // [V, D/G] BF16
    device const bfloat*  biases    [[buffer(2)]],   // [V, D/G] BF16
    device half*          out       [[buffer(3)]],   // [D] FP16
    constant uint&        token_id  [[buffer(4)]],
    constant uint&        D         [[buffer(5)]],
    constant float&       out_scale [[buffer(6)]],   // pass 1.0 to disable
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint groups_per_row = D / kGroupSize;
    device const uint8_t* row_q = table  + uint(token_id) * (D / 2u);
    device const bfloat*  row_s = scales + uint(token_id) * groups_per_row;
    device const bfloat*  row_b = biases + uint(token_id) * groups_per_row;
    uint8_t byte = row_q[gid >> 1];
    uint    q    = (gid & 1u) ? uint(byte >> 4) : uint(byte & 0xFu);
    float   s    = float(row_s[gid / kGroupSize]);
    float   b    = float(row_b[gid / kGroupSize]);
    out[gid] = gemma_scaled_embedding(float(q) * s + b, out_scale);
}

// y[m] = sum_{n} W[m, n] * x[n]. One-SIMD-per-row variant: 32 threads
// cooperate on a single output row, each handling 2 elements per group of 64
// (one byte → two nibbles). simd_sum reduces across the group; lane 0 writes.
//
// Requires N % G == 0. A group-32 tail activates only its first 16 lanes.
// Each threadgroup handles eight consecutive rows, one SIMD per row. The
// larger work unit gives the scheduler enough independent rows while sharing
// the L1-cached input-vector reads.
//
// `OutT` is the store type only — the dot product always accumulates in FP32.
// `half` is the default everywhere; `float` exists for rows whose magnitude
// can leave FP16 range before a downstream scale brings them back, which is
// the case for Inkling's shared-expert down projection (see
// docs/INKLING_SMALL.md, "FFN output range").
template <typename OutT>
static inline float dequant_int4_gemv_simd_body_t(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device OutT*          y,
    uint                  M,
    uint                  N,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane,
    bool                  store_output
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return 0.0f;
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    if (kGemmaSourceFP16) {
        const float result = gemma_source_projection_row(
            W_row, s_row, b_row, x, N, kGroupSize, M % 8u == 0u && N % 512u == 0u, lane);
        if (store_output && lane == 0u) y[row] = OutT(result);
        return result;
    }

    float acc = 0.0f;
    // The vectorized row path reads
    // weights a uint (4 bytes = 8 nibbles) and x as half4 in 128-byte blocks,
    // with a scalar byte-per-lane remainder. Within a block the 32 lanes
    // split 8-per-group (G=64) or 4-per-group (G=32), each handling eight
    // elements from one pair, so the affine factoring s·Σqx + b·Σx is preserved
    // (simd_sum aggregates; s/b are constant within a group). Aligned: row
    // stride N/2 and weightsOffset are multiples of 4; x is
    // half4-aligned (lane*8 elements). N=2816/4096/8192 has exact 256-value
    // blocks; the remainder also covers widths such as 704 and 2112.
    // A SIMD block always covers 256 values. Its storage has four group-64
    // pairs or eight group-32 pairs; each lane's eight values stay in one pair.
    const uint full_blocks = kGemmaSourceFP16 ? (N + 255u) / 256u : N / 256u;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        if (kGemmaSourceFP16 && byte_base * 2u >= N) continue;
        // Read the 4-byte weight chunk as two ushorts. The resident weight
        // tensors are 2-byte aligned but NOT 4-byte aligned (BF16 scale/bias
        // regions leave a 2-aligned weightsOffset), so a `uint*` load would be
        // misaligned (undefined → garbage); a `ushort*` load is safe (row stride
        // N/2, weightsOffset, and byte_base are all even) and halves the loads
        // vs byte-by-byte.
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = (byte_base * 2u) / kGroupSize;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = kGemmaSourceFP16
            ? gemma_source_quad_sum(xa) + gemma_source_quad_sum(xb)
            : e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint tile = full_blocks * 4u; tile * 64u < N; ++tile) {
        const uint elem = tile * 64u + lane * 2u;
        if (elem >= N) continue;
        const uint g = elem / kGroupSize;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[elem / 2u];
        const float x0 = float(x[elem]);
        const float x1 = float(x[elem + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (store_output && lane == 0) {
        y[row] = OutT(acc);
    }
    return acc;
}

static inline void dequant_int4_gemv_simd_body(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device half*          y,
    uint                  M,
    uint                  N,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    dequant_int4_gemv_simd_body_t<half>(W, scales, biases, x, y, M, N,
                                        rows_per_tg, tg_idx, sg_idx, lane, true);
}

// Token-parallel projection with the decode kernel's exact affine factoring
// and reduction order. One dispatch covers every prompt row; there is no
// host-side token loop or full-model replay.
kernel void prefill_dequant_int4_gemv_simd_block(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* X [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (tg.y >= T) return;
    dequant_int4_gemv_simd_body_t<half>(W, scales, biases,
        X + tg.y * K, Y + tg.y * N, N, K, 8u, tg.x, sg, lane, true);
}

static inline float shared_int4_activation(float x) {
    if (is_function_constant_defined(FC_SHARED_INT4_ACT_SILU)
        && FC_SHARED_INT4_ACT_SILU) {
        return x / (1.0f + exp(-x));
    }
    const float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

/// Shared INT4 gate/up/activation in one encoder. The two GEMV bodies are the
/// production body above. The FP32 results round through local half values
/// before activation, preserving the exact three-dispatch numerical boundary
/// without writing and rereading the gate/up scratch vectors.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void shared_int4_gate_up_act_simd(
    device const uint8_t* gateW      [[buffer(0)]],
    device const bfloat* gateScales  [[buffer(1)]],
    device const bfloat* gateBiases  [[buffer(2)]],
    device const uint8_t* upW        [[buffer(3)]],
    device const bfloat* upScales    [[buffer(4)]],
    device const bfloat* upBiases    [[buffer(5)]],
    device const half* x             [[buffer(6)]],
    device half* act                 [[buffer(7)]],
    constant uint& M                 [[buffer(8)]],
    constant uint& N                 [[buffer(9)]],
    uint tg_idx                      [[threadgroup_position_in_grid]],
    uint sg_idx                      [[simdgroup_index_in_threadgroup]],
    uint lane                        [[thread_index_in_simdgroup]]) {
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    const float gateValue = dequant_int4_gemv_simd_body_t<half>(
        gateW, gateScales, gateBiases, x, act,
        MM, NN, 8u, tg_idx, sg_idx, lane, false);
    const float upValue = dequant_int4_gemv_simd_body_t<half>(
        upW, upScales, upBiases, x, act,
        MM, NN, 8u, tg_idx, sg_idx, lane, false);
    const uint row = tg_idx * 8u + sg_idx;
    if (lane == 0u && row < MM) {
        const half roundedGate = half(gateValue);
        const half roundedUp = half(upValue);
        act[row] = kGemmaSourceFP16 ? gemma_source_geglu(roundedGate, roundedUp)
            : half(shared_int4_activation(float(roundedGate)) * float(roundedUp));
    }
}

kernel void dequant_int4_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, MM, NN,
                                rows_per_tg, tg_idx, sg_idx, lane);
}

// Same GEMV, FP32 output rows. Bit-identical arithmetic to
// `dequant_int4_gemv_simd` — only the final store widens — so a caller can
// swap it in wherever an output row can exceed FP16's 65 504 before the
// scale that brings it back into range is applied downstream.
kernel void dequant_int4_gemv_simd_f32out(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device float*         y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    dequant_int4_gemv_simd_body_t<float>(W, scales, biases, x, y, MM, NN,
                                         rows_per_tg, tg_idx, sg_idx, lane, true);
}


kernel void dequant_int4_qkv_gemv_simd(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row;
        M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - QQ;
        M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - QQ - KK;
        M = KK;
    }
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, M, NN,
                                1u, local_row, 0u, lane);
}

// ============================================================================
// Multi-token GEMV for the MTP speculative-verify pass.
//
// y[t][m] = sum_n W[m, n] * x[t][n] for a small batch of T token rows
// (T <= 8), reading each packed weight row ONCE and applying it to every
// token. Per-token arithmetic — operand order, affine factoring, FP32
// accumulation, the final simd_sum — is an exact replica of
// `dequant_int4_gemv_simd_body_t`, so each output row is bit-identical to
// running the single-token decode GEMV once per token. That bit-identity is
// what lets the speculative verify chunk emit the same greedy tokens as
// plain decode while paying the weight-read cost of one decode step.
// ============================================================================

constant constexpr uint kMultiXMaxT = 8;
// Token count specialized per pipeline so the per-token loops fully unroll
// and the accumulators stay in registers (a runtime T spills the accumulator
// array and multiplies the per-token marginal cost).
constant uint FC_MULTIX_T [[function_constant(45)]];
constant bool FC_MULTIX_USE_FC [[function_constant(46)]];

static inline uint multix_fc_t(uint T) {
    return (is_function_constant_defined(FC_MULTIX_USE_FC) &&
            FC_MULTIX_USE_FC &&
            is_function_constant_defined(FC_MULTIX_T)) ? FC_MULTIX_T : T;
}

template <typename OutT>
static inline void dequant_int4_gemv_simd_multix_body(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,      // [T, N] row-major
    device OutT*          y,      // [T, M] row-major
    uint                  M,
    uint                  N,
    uint                  T_param,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    const uint T = multix_fc_t(T_param);
    constexpr uint rows_per_tg = 8;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float accs[kMultiXMaxT];
    for (uint t = 0; t < kMultiXMaxT; ++t) { accs[t] = 0.0f; }

    const uint full_blocks = kGemmaSourceFP16 ? (N + 255u) / 256u : N / 256u;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        if (kGemmaSourceFP16 && byte_base * 2u >= N) continue;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = (byte_base * 2u) / kGroupSize;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        for (uint t = 0; t < T; ++t) {
            device const half* xt = x + t * N;
            const half4 xa = *((device const half4*)(xt + elem));
            const half4 xb = *((device const half4*)(xt + elem + 4u));
            const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
            const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
            float dot = 0.0f;
            dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
            dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
            dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
            dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
            const float sum = kGemmaSourceFP16
            ? gemma_source_quad_sum(xa) + gemma_source_quad_sum(xb)
            : e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            accs[t] = fma(s, dot, accs[t]);
            accs[t] = fma(b, sum, accs[t]);
        }
    }
    for (uint tile = full_blocks * 4u; tile * 64u < N; ++tile) {
        const uint elem = tile * 64u + lane * 2u;
        if (elem >= N) continue;
        const uint g = elem / kGroupSize;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[elem / 2u];
        for (uint t = 0; t < T; ++t) {
            device const half* xt = x + t * N;
            const float x0 = float(xt[elem]);
            const float x1 = float(xt[elem + 1u]);
            float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
            dot = fma(float(uint(byte >> 4)), x1, dot);
            const float sum = x0 + x1;
            accs[t] = fma(s, dot, accs[t]);
            accs[t] = fma(b, sum, accs[t]);
        }
    }
    for (uint t = 0; t < T; ++t) {
        const float acc = simd_sum(accs[t]);
        if (lane == 0) {
            y[t * M + row] = OutT(acc);
        }
    }
}

// Prompt-sized grid of small MultiX tiles: at most two host dispatches cover
// the full prompt and its ragged tail, with no per-token host loop.
kernel void prefill_dequant_int4_multix_block(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* X [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint tile = multix_fc_t(1u);
    const uint first = tg.y * tile;
    if (first >= T) return;
    dequant_int4_gemv_simd_multix_body<half>(W, scales, biases,
        X + first * K, Y + first * N, N, K, tile, tg.x, sg, lane);
}

kernel void dequant_int4_gemv_simd_multix(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        T      [[buffer(7)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    dequant_int4_gemv_simd_multix_body<half>(W, scales, biases, x, y,
                                             M, N, T, tg_idx, sg_idx, lane);
}

kernel void dequant_int4_gemv_simd_multix_f32out(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device float*         y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        T      [[buffer(7)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    dequant_int4_gemv_simd_multix_body<float>(W, scales, biases, x, y,
                                              M, N, T, tg_idx, sg_idx, lane);
}
