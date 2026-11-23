#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Inkling 2 Mini routed experts (128 experts, top-8, D 2304, F 1152), as the
// Unsloth UD-Q4_K_XL install stores them: ggml NVFP4 blocks (layers 0-19) or
// Q8_0 blocks (layers 20-23), byte for byte. An expert blob is its merged
// gate-up matrix (2F rows of D weights: gate rows 0..F-1, up rows F..2F-1)
// then its down matrix (D rows of F weights). A layer's blobs sit at a fixed
// stride in one buffer. Two per-expert scale arrays (gate-up, down) are the
// NVFP4 `.scale` tensors as stored (F32 [128] each; ones for Q8_0).
//
// Pieces, each one or two dispatches, with no CPU readback between them:
//   router        F32 logits, sigmoid, biased top-8, normalized weights x 8
//   route plan    count, prefix, scatter: pairs grouped by expert and a list
//                 of 64-row work blocks, plus indirect dispatch arguments
//   gate-up       grouped GEMM, weights decoded a tile at a time into half
//                 threadgroup tiles, float accumulators, the per-expert scale
//                 and the SITU activation in the epilogue
//   down          grouped GEMM; epilogue (acc * scale) * route weight into
//                 FP32 partials per (token, rank)
//   reduce        partials summed in rank order
//   decode        one token: gate-up GEMV + SITU, then down GEMV summing the
//                 eight experts in rank order
//
// The weight encoding is a function constant, so one source builds both.
// NVFP4 values (E2M1 x UE4M3) are exact in half; Q8_0 values are rounded
// to half as ggml's Metal mul_mm does.

#define I2_UNROLL _Pragma("clang loop unroll(full)")

constant uint FC_I2_ENCODING [[function_constant(200)]];
constant uint kI2Encoding = is_function_constant_defined(FC_I2_ENCODING) ? FC_I2_ENCODING : 0u;
constant constexpr uint kI2NVFP4 = 0;

constant constexpr uint kI2Rows = 64;          // routed rows per threadgroup
constant constexpr uint kI2K = 32;             // K step
constant constexpr uint kI2Ld = kI2K + 8;      // padded tile row, halves
constant constexpr uint kI2GateUpCols = 64;    // gate and up outputs each
constant constexpr uint kI2DownCols = 128;     // down outputs
constant constexpr uint kI2Threads = 128;      // four simdgroups, 2 x 2

struct I2Pair { uint token; uint rank; float weight; uint reserved; };
struct I2Work { uint expert; uint pair_start; uint row_count; uint reserved; };

// Layout-identical to `Inkling2MoEParams`.
struct I2Params {
    uint D;
    uint F;
    uint top_k;
    uint hidden_stride;     // halves between hidden rows
    uint expert_stride;     // bytes between expert blobs
    uint gate_up_offset;    // bytes from a blob to gate-up row 0
    uint down_offset;       // bytes from a blob to down row 0
    uint experts;
};

static inline uint i2_row_bytes(uint k) {
    return kI2Encoding == kI2NVFP4 ? (k / 64u) * 36u : (k / 32u) * 34u;
}

// ggml_ue4m3_to_fp32 times two: the sign bit is ignored, 0x00 and 0x7F are 0.
static inline float i2_ue4m3(uint x) {
    if (x == 0u || x == 0x7Fu) return 0.0f;
    const uint e = (x >> 3) & 15u, m = x & 7u;
    return e == 0u ? float(m) * 0.001953125f : as_type<float>(((e + 120u) << 23) | (m << 20));
}

static inline half i2_e2m1(uint code) {
    const uint mag = code & 7u;
    const half v = mag < 2u ? half(mag) * 0.5h
                            : half(1u << ((mag >> 1) - 1u)) * (1.0h + 0.5h * half(mag & 1u));
    return (code & 8u) != 0u ? -v : v;
}

// lut[b] = (E2M1(low nibble of b), E2M1(high nibble of b)).
static inline void i2_fill_lut(threadgroup half2* lut, uint tid, uint threads) {
    for (uint b = tid; b < 256u; b += threads) lut[b] = half2(i2_e2m1(b & 15u), i2_e2m1(b >> 4));
}

// (row, col) of the two elements a lane holds in an 8x8 simdgroup matrix:
// (row, col) and (row, col + 1).
static inline ushort2 i2_frag_coord(ushort lane) {
    const ushort qid = lane / 4;
    return ushort2((qid & 4) + ((lane / 2) % 4), (qid & 2) * 2 + (lane % 2) * 2);
}

// Weight rows [row0, row0 + rows) x K [k0, k0 + 32) of a matrix with `k`
// columns into threadgroup half, row-major with stride kI2Ld. Each thread
// owns whole rows.
static inline void i2_dequant_tile(device const uchar* mat, uint k, uint row0, uint rows, uint k0,
                                   threadgroup const half2* lut, threadgroup half* dst,
                                   uint tid, uint threads) {
    const uint row_bytes = i2_row_bytes(k);
    for (uint r = tid; r < rows; r += threads) {
        threadgroup half* d = dst + r * kI2Ld;
        device const uchar* row = mat + ulong(row0 + r) * row_bytes;
        if (kI2Encoding == kI2NVFP4) {
            device const uchar* block = row + (k0 / 64u) * 36u;
            const uint scales = *reinterpret_cast<device const uint*>(block);
            const uint sub0 = (k0 % 64u) / 16u;
            device const uint* q = reinterpret_cast<device const uint*>(block + 4u + sub0 * 8u);
            I2_UNROLL for (uint s = 0; s < 2; ++s) {
                const half2 scale = half2(half(i2_ue4m3((scales >> (8u * (sub0 + s))) & 0xFFu)));
                I2_UNROLL for (uint w = 0; w < 2; ++w) {
                    const uint u = q[s * 2u + w];
                    const half2 v0 = lut[u & 0xFFu] * scale, v1 = lut[(u >> 8) & 0xFFu] * scale;
                    const half2 v2 = lut[(u >> 16) & 0xFFu] * scale, v3 = lut[u >> 24] * scale;
                    threadgroup half4* lo = reinterpret_cast<threadgroup half4*>(d + s * 16u + w * 4u);
                    threadgroup half4* hi = reinterpret_cast<threadgroup half4*>(d + s * 16u + 8u + w * 4u);
                    *lo = half4(v0.x, v1.x, v2.x, v3.x);
                    *hi = half4(v0.y, v1.y, v2.y, v3.y);
                }
            }
        } else {
            device const uchar* block = row + (k0 / 32u) * 34u;
            const float scale = float(*reinterpret_cast<device const half*>(block));
            device const ushort* q = reinterpret_cast<device const ushort*>(block + 2u);
            I2_UNROLL for (uint i = 0; i < 16; ++i) {
                const ushort pair = q[i];
                d[2u * i] = half(float(as_type<char>(uchar(pair & 0xFFu))) * scale);
                d[2u * i + 1u] = half(float(as_type<char>(uchar(pair >> 8))) * scale);
            }
        }
    }
}

// Rows [0, 64) x K [k0, k0 + 32); rows past the block's end are zero. Each
// thread copies half a row (16 halves).
static inline void i2_load_rows(device const half* src_row, bool valid, uint k0, threadgroup half* dst_row) {
    threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(dst_row);
    if (valid) {
        device const half4* src = reinterpret_cast<device const half4*>(src_row + k0);
        I2_UNROLL for (uint i = 0; i < kI2K / 8u; ++i) dst[i] = src[i];
    } else {
        I2_UNROLL for (uint i = 0; i < kI2K / 8u; ++i) dst[i] = half4(0.0h);
    }
}

// SITU with clamped arguments: Metal's fast tanh and exp return NaN far
// outside their useful range; at the clamps tanh is 1 and the sigmoid 0 or 1
// to FP32 precision.
static inline float i2_situ(float g, float u) {
    const float capped = tanh(clamp(g, -64.0f, 64.0f) * 0.25f) * 4.0f;
    const float sigmoid = 1.0f / (1.0f + exp(-clamp(g, -80.0f, 80.0f)));
    const float linear = tanh(clamp(u, -800.0f, 800.0f) * (1.0f / 25.0f)) * 25.0f;
    return capped * sigmoid * linear;
}

// MARK: - Prefill grouped GEMMs

kernel void inkling2_moe_gate_up(
    device const half* hidden            [[buffer(0)]],
    device const I2Pair* pairs           [[buffer(1)]],
    device const I2Work* work            [[buffer(2)]],
    device const uint* work_count        [[buffer(3)]],
    device half* activation              [[buffer(4)]],
    device const uchar* pool             [[buffer(5)]],
    device const float* gate_up_scales   [[buffer(6)]],
    constant I2Params& p                 [[buffer(7)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    ushort tid  [[thread_index_in_threadgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (tgid.y >= work_count[0]) return;
    const I2Work w = work[tgid.y];
    device const uchar* gate = pool + ulong(w.expert) * p.expert_stride + p.gate_up_offset;
    device const uchar* up = gate + ulong(p.F) * i2_row_bytes(p.D);
    const uint n_base = tgid.x * kI2GateUpCols;

    threadgroup half smem[3 * kI2Rows * kI2Ld];
    threadgroup half2 lut[256];
    threadgroup half* rows_tile = smem;
    threadgroup half* gate_tile = smem + kI2Rows * kI2Ld;
    threadgroup half* up_tile = gate_tile + kI2GateUpCols * kI2Ld;
    if (kI2Encoding == kI2NVFP4) {
        i2_fill_lut(lut, tid, kI2Threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint a_row = tid / 2u, a_k = (tid % 2u) * (kI2K / 2u);
    const bool a_valid = a_row < w.row_count;
    device const half* a_src = hidden;
    if (a_valid) a_src = hidden + ulong(pairs[w.pair_start + a_row].token) * p.hidden_stride + a_k;

    simdgroup_float8x8 acc_gate[4][4], acc_up[4][4];
    I2_UNROLL for (uint i = 0; i < 4; ++i) I2_UNROLL for (uint j = 0; j < 4; ++j) {
        acc_gate[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc_up[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint sg_m = (sg / 2u) * 32u, sg_n = (sg % 2u) * 32u;

    for (uint k0 = 0; k0 < p.D; k0 += kI2K) {
        i2_load_rows(a_src, a_valid, k0, rows_tile + a_row * kI2Ld + a_k);
        // Simdgroups 0-1 decode the gate rows and 2-3 the up rows.
        if (tid < kI2GateUpCols) {
            i2_dequant_tile(gate, p.D, n_base, kI2GateUpCols, k0, lut, gate_tile, tid, kI2GateUpCols);
        } else {
            i2_dequant_tile(up, p.D, n_base, kI2GateUpCols, k0, lut, up_tile, tid - kI2GateUpCols, kI2GateUpCols);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        I2_UNROLL for (uint kk = 0; kk < kI2K; kk += 8) {
            simdgroup_half8x8 a[4], g[4], u[4];
            I2_UNROLL for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a[i], rows_tile + (sg_m + i * 8u) * kI2Ld + kk, kI2Ld);
            }
            I2_UNROLL for (uint j = 0; j < 4; ++j) {
                simdgroup_load(g[j], gate_tile + (sg_n + j * 8u) * kI2Ld + kk, kI2Ld, ulong2(0, 0), true);
                simdgroup_load(u[j], up_tile + (sg_n + j * 8u) * kI2Ld + kk, kI2Ld, ulong2(0, 0), true);
            }
            I2_UNROLL for (uint i = 0; i < 4; ++i) I2_UNROLL for (uint j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(acc_gate[i][j], a[i], g[j], acc_gate[i][j]);
                simdgroup_multiply_accumulate(acc_up[i][j], a[i], u[j], acc_up[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Per-expert scale in FP32, then SITU, staged so partial blocks write
    // only their own rows.
    const float scale = gate_up_scales[w.expert];
    constexpr uint kStage = kI2GateUpCols + 8;
    threadgroup half* stage = smem;
    const ushort2 fc = i2_frag_coord(lane);
    I2_UNROLL for (uint i = 0; i < 4; ++i) I2_UNROLL for (uint j = 0; j < 4; ++j) {
        const uint r = sg_m + i * 8u + fc.x, c = sg_n + j * 8u + fc.y;
        I2_UNROLL for (uint e = 0; e < 2; ++e) {
            const float gv = acc_gate[i][j].thread_elements()[e] * scale;
            const float uv = acc_up[i][j].thread_elements()[e] * scale;
            stage[r * kStage + c + e] = half(i2_situ(gv, uv));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint out_row = tid / 2u, out_col = (tid % 2u) * (kI2GateUpCols / 2u);
    if (out_row < w.row_count) {
        device half4* dst = reinterpret_cast<device half4*>(
            activation + ulong(w.pair_start + out_row) * p.F + n_base + out_col);
        threadgroup const half4* src = reinterpret_cast<threadgroup const half4*>(stage + out_row * kStage + out_col);
        I2_UNROLL for (uint i = 0; i < kI2GateUpCols / 8u; ++i) dst[i] = src[i];
    }
}

kernel void inkling2_moe_down(
    device const half* activation        [[buffer(0)]],
    device const I2Pair* pairs           [[buffer(1)]],
    device const I2Work* work            [[buffer(2)]],
    device const uint* work_count        [[buffer(3)]],
    device float* partials               [[buffer(4)]],
    device const uchar* pool             [[buffer(5)]],
    device const float* down_scales      [[buffer(6)]],
    constant I2Params& p                 [[buffer(7)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    ushort tid  [[thread_index_in_threadgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (tgid.y >= work_count[0]) return;
    const I2Work w = work[tgid.y];
    device const uchar* down = pool + ulong(w.expert) * p.expert_stride + p.down_offset;
    const uint n_base = tgid.x * kI2DownCols;

    threadgroup half smem[(kI2Rows + kI2DownCols) * kI2Ld];
    threadgroup half2 lut[256];
    threadgroup half* rows_tile = smem;
    threadgroup half* weight_tile = smem + kI2Rows * kI2Ld;
    if (kI2Encoding == kI2NVFP4) {
        i2_fill_lut(lut, tid, kI2Threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint a_row = tid / 2u, a_k = (tid % 2u) * (kI2K / 2u);
    const bool a_valid = a_row < w.row_count;
    device const half* a_src = activation + ulong(w.pair_start + (a_valid ? a_row : 0u)) * p.F + a_k;

    simdgroup_float8x8 acc[4][8];
    I2_UNROLL for (uint i = 0; i < 4; ++i) I2_UNROLL for (uint j = 0; j < 8; ++j) {
        acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint sg_m = (sg / 2u) * 32u, sg_n = (sg % 2u) * 64u;

    for (uint k0 = 0; k0 < p.F; k0 += kI2K) {
        i2_load_rows(a_src, a_valid, k0, rows_tile + a_row * kI2Ld + a_k);
        i2_dequant_tile(down, p.F, n_base, kI2DownCols, k0, lut, weight_tile, tid, kI2Threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        I2_UNROLL for (uint kk = 0; kk < kI2K; kk += 8) {
            simdgroup_half8x8 a[4], b[8];
            I2_UNROLL for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a[i], rows_tile + (sg_m + i * 8u) * kI2Ld + kk, kI2Ld);
            }
            I2_UNROLL for (uint j = 0; j < 8; ++j) {
                simdgroup_load(b[j], weight_tile + (sg_n + j * 8u) * kI2Ld + kk, kI2Ld, ulong2(0, 0), true);
            }
            I2_UNROLL for (uint i = 0; i < 4; ++i) I2_UNROLL for (uint j = 0; j < 8; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // (acc * expert scale) * route weight, FP32, into the pair's partial row.
    const float scale = down_scales[w.expert];
    const ushort2 fc = i2_frag_coord(lane);
    I2_UNROLL for (uint i = 0; i < 4; ++i) {
        const uint r = sg_m + i * 8u + fc.x;
        if (r >= w.row_count) continue;
        const I2Pair pair = pairs[w.pair_start + r];
        device float* dst = partials + ulong(pair.token * p.top_k + pair.rank) * p.D + n_base + sg_n + fc.y;
        I2_UNROLL for (uint j = 0; j < 8; ++j) {
            const float2 v = float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
            *reinterpret_cast<device float2*>(dst + j * 8u) = (v * scale) * pair.weight;
        }
    }
}

// out[t][i] = sum over ranks, in rank order, of partials[t * top_k + r][i].
kernel void inkling2_moe_reduce(
    device const float4* partials        [[buffer(0)]],
    device float4* out                   [[buffer(1)]],
    constant uint& D                     [[buffer(2)]],
    constant uint& top_k                 [[buffer(3)]],
    constant uint& out_stride            [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint quads = D / 4u;
    if (gid.x >= quads) return;
    device const float4* src = partials + ulong(gid.y) * top_k * quads + gid.x;
    float4 sum = src[0];
    for (uint r = 1; r < top_k; ++r) sum += src[ulong(r) * quads];
    out[ulong(gid.y) * (out_stride / 4u) + gid.x] = sum;
}

// MARK: - Router

constant constexpr uint kI2RouterTokens = 32;
constant constexpr uint kI2RouterK = 32;
constant constexpr uint kI2RouterExperts = 128;
constant constexpr uint kI2RouterLd = kI2RouterK + 4;
constant constexpr uint kI2RouterLogitLd = kI2RouterExperts + 4;

// logits = W x in FP32 (W [128][D] F32, x [T][x_stride] F32), then per token
// sigmoid, top-8 of (sigmoid + bias) with ties to the lower expert, weights
// = sigmoid / (sum + 1e-20) * route_scale (`expert_weights_norm_eps`), FP32
// throughout as upstream runs this router. Writes ids [T][top_k] and
// weights [T][top_k]. 32 tokens per threadgroup of 128 threads.
kernel void inkling2_moe_router(
    device const float* x                [[buffer(0)]],
    device const float* weight           [[buffer(1)]],
    device const float* bias             [[buffer(2)]],
    device uint* ids                     [[buffer(3)]],
    device float* route_weights          [[buffer(4)]],
    constant uint& tokens                [[buffer(5)]],
    constant uint& D                     [[buffer(6)]],
    constant uint& x_stride              [[buffer(7)]],
    constant uint& top_k                 [[buffer(8)]],
    constant float& route_scale          [[buffer(9)]],
    uint tgid   [[threadgroup_position_in_grid]],
    ushort tid  [[thread_index_in_threadgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint kTiles = (kI2RouterTokens + kI2RouterExperts) * kI2RouterLd;
    constexpr uint kLogits = kI2RouterTokens * kI2RouterLogitLd;
    threadgroup float smem[kTiles > kLogits ? kTiles : kLogits];
    threadgroup float* x_tile = smem;
    threadgroup float* w_tile = smem + kI2RouterTokens * kI2RouterLd;
    const uint t_base = tgid * kI2RouterTokens;

    simdgroup_float8x8 acc[4][4];
    I2_UNROLL for (uint i = 0; i < 4; ++i) I2_UNROLL for (uint j = 0; j < 4; ++j) {
        acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint e_base = sg * 32u;
    for (uint k0 = 0; k0 < D; k0 += kI2RouterK) {
        // x: 32 tokens x 32 K, 8 floats per thread.
        {
            const uint row = tid / 4u, col = (tid % 4u) * 8u;
            const uint t = t_base + row;
            threadgroup float* dst = x_tile + row * kI2RouterLd + col;
            if (t < tokens) {
                device const float4* src = reinterpret_cast<device const float4*>(x + ulong(t) * x_stride + k0 + col);
                const float4 a = src[0], b = src[1];
                dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
                dst[4] = b.x; dst[5] = b.y; dst[6] = b.z; dst[7] = b.w;
            } else {
                I2_UNROLL for (uint i = 0; i < 8; ++i) dst[i] = 0.0f;
            }
        }
        // W: 128 experts x 32 K, one expert row per thread.
        {
            device const float4* src = reinterpret_cast<device const float4*>(weight + ulong(tid) * D + k0);
            threadgroup float* dst = w_tile + tid * kI2RouterLd;
            I2_UNROLL for (uint i = 0; i < 8; ++i) {
                const float4 v = src[i];
                dst[4 * i] = v.x; dst[4 * i + 1] = v.y; dst[4 * i + 2] = v.z; dst[4 * i + 3] = v.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        I2_UNROLL for (uint kk = 0; kk < kI2RouterK; kk += 8) {
            simdgroup_float8x8 a[4], b[4];
            I2_UNROLL for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], x_tile + (i * 8u) * kI2RouterLd + kk, kI2RouterLd);
            I2_UNROLL for (uint j = 0; j < 4; ++j) {
                simdgroup_load(b[j], w_tile + (e_base + j * 8u) * kI2RouterLd + kk, kI2RouterLd, ulong2(0, 0), true);
            }
            I2_UNROLL for (uint i = 0; i < 4; ++i) I2_UNROLL for (uint j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* logits = smem;
    I2_UNROLL for (uint i = 0; i < 4; ++i) I2_UNROLL for (uint j = 0; j < 4; ++j) {
        simdgroup_store(acc[i][j], logits + (i * 8u) * kI2RouterLogitLd + e_base + j * 8u, kI2RouterLogitLd);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup selects for 8 tokens; lane l holds experts l + 32 j.
    for (uint q = 0; q < 8; ++q) {
        const uint row = sg * 8u + q, t = t_base + row;
        if (t >= tokens) break;
        float prob[4], sel[4];
        I2_UNROLL for (uint j = 0; j < 4; ++j) {
            const uint e = lane + 32u * j;
            const float l = clamp(logits[row * kI2RouterLogitLd + e], -80.0f, 80.0f);
            prob[j] = 1.0f / (1.0f + exp(-l));
            sel[j] = prob[j] + bias[e];
        }
        // Lane r keeps rank r; the sum runs in rank order.
        float sum = 0.0f, my_prob = 0.0f;
        uint my_id = 0u;
        for (uint r = 0; r < top_k; ++r) {
            float best = sel[0];
            uint best_e = lane;
            I2_UNROLL for (uint j = 1; j < 4; ++j) {
                if (sel[j] > best) { best = sel[j]; best_e = lane + 32u * j; }
            }
            const float top = simd_max(best);
            const uint chosen = simd_min(best == top ? best_e : 0xFFFFFFFFu);
            float chosen_prob = 0.0f;
            I2_UNROLL for (uint j = 0; j < 4; ++j) {
                if (lane + 32u * j == chosen) { chosen_prob = prob[j]; sel[j] = -FLT_MAX; }
            }
            chosen_prob = simd_sum(chosen_prob);
            sum += chosen_prob;
            if (lane == r) { my_prob = chosen_prob; my_id = chosen; }
        }
        if (lane < top_k) {
            const ulong at = ulong(t) * top_k + lane;
            ids[at] = my_id;
            route_weights[at] = (my_prob / (sum + 1e-20f)) * route_scale;
        }
    }
}

// MARK: - Route planning (no readback)

kernel void inkling2_route_clear(device uint* counts [[buffer(0)]], constant uint& experts [[buffer(1)]],
                                 uint e [[thread_position_in_grid]]) {
    if (e < experts) counts[e] = 0u;
}

kernel void inkling2_route_count(device const uint* ids [[buffer(0)]], device atomic_uint* counts [[buffer(1)]],
                                 constant uint& pairs [[buffer(2)]], constant uint& experts [[buffer(3)]],
                                 uint p [[thread_position_in_grid]]) {
    if (p >= pairs) return;
    atomic_fetch_add_explicit(&counts[min(ids[p], experts - 1u)], 1u, memory_order_relaxed);
}

struct I2PlanParams {
    uint experts;
    uint rows_per_block;
    uint gate_up_columns;    // threadgroups across gate-up outputs
    uint down_columns;       // threadgroups across down outputs
    uint work_capacity;
};

// One threadgroup with one thread per expert (experts <= 1024). Writes each
// expert's first pair (also its scatter cursor), its 64-row work blocks, the
// work count, and two MTLDispatchThreadgroupsIndirectArguments (gate-up,
// then down).
kernel void inkling2_route_plan(
    device const uint* counts            [[buffer(0)]],
    device uint* cursors                 [[buffer(1)]],
    device I2Work* work                  [[buffer(2)]],
    device uint* work_count              [[buffer(3)]],
    device uint* dispatch                [[buffer(4)]],
    constant I2PlanParams& pp            [[buffer(5)]],
    uint e [[thread_position_in_threadgroup]]) {
    threadgroup uint pair_start[1024];
    threadgroup uint block_start[1025];
    if (e == 0) {
        uint pairs = 0, blocks = 0;
        for (uint x = 0; x < pp.experts; ++x) {
            pair_start[x] = pairs;
            block_start[x] = blocks;
            pairs += counts[x];
            blocks += (counts[x] + pp.rows_per_block - 1u) / pp.rows_per_block;
        }
        block_start[pp.experts] = min(blocks, pp.work_capacity);
        work_count[0] = block_start[pp.experts];
        dispatch[0] = pp.gate_up_columns; dispatch[1] = block_start[pp.experts]; dispatch[2] = 1u;
        dispatch[3] = pp.down_columns;    dispatch[4] = block_start[pp.experts]; dispatch[5] = 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e >= pp.experts) return;
    const uint count = counts[e];
    cursors[e] = pair_start[e];
    for (uint b = 0; b * pp.rows_per_block < count; ++b) {
        const uint slot = block_start[e] + b;
        if (slot >= pp.work_capacity) break;
        work[slot] = I2Work{e, pair_start[e] + b * pp.rows_per_block,
                            min(pp.rows_per_block, count - b * pp.rows_per_block), 0u};
    }
}

// Places each (token, rank) pair in its expert's group. The order inside a
// group is not fixed, and does not need to be: every routed row is computed
// on its own, and partials are indexed by (token, rank).
kernel void inkling2_route_scatter(
    device const uint* ids               [[buffer(0)]],
    device const float* route_weights    [[buffer(1)]],
    device atomic_uint* cursors          [[buffer(2)]],
    device I2Pair* sorted                [[buffer(3)]],
    constant uint& pairs                 [[buffer(4)]],
    constant uint& top_k                 [[buffer(5)]],
    constant uint& experts               [[buffer(6)]],
    uint p [[thread_position_in_grid]]) {
    if (p >= pairs) return;
    const uint e = min(ids[p], experts - 1u);
    const uint at = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
    sorted[at] = I2Pair{p / top_k, p % top_k, route_weights[p], 0u};
}

// MARK: - Decode (one token)

constant constexpr uint kI2DecodeRows = 4;   // output rows per simdgroup

// Doubled E2M1 magnitudes (0, 1, 2, 3, 4, 6, 8, 12) as nibbles of one word,
// so a code decodes in registers.
constant constexpr uint kI2E2M1Doubled = 0xC8643210u;

static inline float i2_e2m1_doubled(uint code) {
    const float magnitude = float((kI2E2M1Doubled >> ((code & 7u) << 2)) & 15u);
    return as_type<float>(as_type<uint>(magnitude) | ((code & 8u) << 28));
}

// The 8 weights of unit `u` (of k / 8) of a row against x, FP32. NVFP4: block
// u / 8, sub-block (u % 8) / 2; bytes (u % 2) * 4 ..< + 4 of the sub-block
// hold elements j (low nibbles) and j + 8 (high) for j in (u % 2) * 4 ..< + 4.
static inline float i2_unit_dot(device const uchar* row, uint u, device const half* x) {
    if (kI2Encoding == kI2NVFP4) {
        const uint b = u / 8u, s = (u % 8u) / 2u, h = u % 2u;
        device const uchar* block = row + b * 36u;
        const uint codes = *reinterpret_cast<device const uint*>(block + 4u + s * 8u + h * 4u);
        const uint k = b * 64u + s * 16u + h * 4u;
        const float4 xl = float4(*reinterpret_cast<device const half4*>(x + k));
        const float4 xh = float4(*reinterpret_cast<device const half4*>(x + k + 8u));
        const float4 wl = float4(i2_e2m1_doubled(codes), i2_e2m1_doubled(codes >> 8),
                                 i2_e2m1_doubled(codes >> 16), i2_e2m1_doubled(codes >> 24));
        const float4 wh = float4(i2_e2m1_doubled(codes >> 4), i2_e2m1_doubled(codes >> 12),
                                 i2_e2m1_doubled(codes >> 20), i2_e2m1_doubled(codes >> 28));
        return (dot(wl, xl) + dot(wh, xh)) * (i2_ue4m3(block[s]) * 0.5f);
    } else {
        const uint b = u / 4u, part = (u % 4u) * 8u;
        device const uchar* block = row + b * 34u;
        const float scale = float(*reinterpret_cast<device const half*>(block));
        device const ushort* q = reinterpret_cast<device const ushort*>(block + 2u + part);
        const uint k = b * 32u + part;
        const float4 x0 = float4(*reinterpret_cast<device const half4*>(x + k));
        const float4 x1 = float4(*reinterpret_cast<device const half4*>(x + k + 4u));
        const ushort q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
        const float4 w0 = float4(float(as_type<char>(uchar(q0 & 0xFFu))), float(as_type<char>(uchar(q0 >> 8))),
                                 float(as_type<char>(uchar(q1 & 0xFFu))), float(as_type<char>(uchar(q1 >> 8))));
        const float4 w1 = float4(float(as_type<char>(uchar(q2 & 0xFFu))), float(as_type<char>(uchar(q2 >> 8))),
                                 float(as_type<char>(uchar(q3 & 0xFFu))), float(as_type<char>(uchar(q3 >> 8))));
        return (dot(w0, x0) + dot(w1, x1)) * scale;
    }
}

// Gate-up and SITU for one token's experts: grid (F / 16, top_k), four
// simdgroups of four (gate, up) row pairs each. activation [top_k][F] half.
kernel void inkling2_moe_decode_gate_up(
    device const half* x                 [[buffer(0)]],
    device const uint* ids               [[buffer(1)]],
    device half* activation              [[buffer(2)]],
    device const uchar* pool             [[buffer(3)]],
    device const float* gate_up_scales   [[buffer(4)]],
    constant I2Params& p                 [[buffer(5)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint expert = min(ids[tgid.y], p.experts - 1u);
    const uint row_bytes = i2_row_bytes(p.D);
    device const uchar* gate = pool + ulong(expert) * p.expert_stride + p.gate_up_offset;
    device const uchar* up = gate + ulong(p.F) * row_bytes;
    const uint row0 = (tgid.x * 4u + sg) * kI2DecodeRows;
    if (row0 >= p.F) return;
    const uint units = p.D / 8u;
    float g[kI2DecodeRows], u[kI2DecodeRows];
    I2_UNROLL for (uint q = 0; q < kI2DecodeRows; ++q) { g[q] = 0.0f; u[q] = 0.0f; }
    for (uint unit = lane; unit < units; unit += 32u) {
        I2_UNROLL for (uint q = 0; q < kI2DecodeRows; ++q) {
            g[q] += i2_unit_dot(gate + ulong(row0 + q) * row_bytes, unit, x);
            u[q] += i2_unit_dot(up + ulong(row0 + q) * row_bytes, unit, x);
        }
    }
    const float scale = gate_up_scales[expert];
    I2_UNROLL for (uint q = 0; q < kI2DecodeRows; ++q) {
        const float gs = simd_sum(g[q]) * scale, us = simd_sum(u[q]) * scale;
        if (lane == 0) activation[ulong(tgid.y) * p.F + row0 + q] = half(i2_situ(gs, us));
    }
}

// Down for one token, the experts summed in rank order: grid D / 16, four
// simdgroups of four output rows. out [D] float.
kernel void inkling2_moe_decode_down(
    device const half* activation        [[buffer(0)]],
    device const uint* ids               [[buffer(1)]],
    device const float* route_weights    [[buffer(2)]],
    device float* out                    [[buffer(3)]],
    device const uchar* pool             [[buffer(4)]],
    device const float* down_scales      [[buffer(5)]],
    constant I2Params& p                 [[buffer(6)]],
    uint tgid   [[threadgroup_position_in_grid]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint row_bytes = i2_row_bytes(p.F);
    const uint row0 = (tgid * 4u + sg) * kI2DecodeRows;
    if (row0 >= p.D) return;
    const uint units = p.F / 8u;
    float total[kI2DecodeRows];
    I2_UNROLL for (uint q = 0; q < kI2DecodeRows; ++q) total[q] = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        const uint expert = min(ids[r], p.experts - 1u);
        device const uchar* down = pool + ulong(expert) * p.expert_stride + p.down_offset;
        device const half* act = activation + ulong(r) * p.F;
        float acc[kI2DecodeRows];
        I2_UNROLL for (uint q = 0; q < kI2DecodeRows; ++q) acc[q] = 0.0f;
        for (uint unit = lane; unit < units; unit += 32u) {
            I2_UNROLL for (uint q = 0; q < kI2DecodeRows; ++q) {
                acc[q] += i2_unit_dot(down + ulong(row0 + q) * row_bytes, unit, act);
            }
        }
        const float scale = down_scales[expert], weight = route_weights[r];
        I2_UNROLL for (uint q = 0; q < kI2DecodeRows; ++q) total[q] += (simd_sum(acc[q]) * scale) * weight;
    }
    if (lane == 0) {
        I2_UNROLL for (uint q = 0; q < kI2DecodeRows; ++q) out[row0 + q] = total[q];
    }
}

#undef I2_UNROLL
