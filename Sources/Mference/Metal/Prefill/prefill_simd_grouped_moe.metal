#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Grouped routed-expert GEMM for GPUs without the TensorOps prefill path
// (Apple7 class), built from 8x8 simdgroup matrices only, so it compiles on
// every supported macOS. Same ABI as `mpp_grouped_routed_moe_*` in
// tensorops.metal: buffers 0-6, a 16-expert argument buffer, sorted pairs and
// per-expert groups; the dispatch's z is the group, y a 64-row block.
//
// Phase 1 computes gate and up together for 64 routed rows x 64 outputs, so
// each hidden-row load feeds 128 output columns; down computes 64 rows x 128
// outputs. Both step K by 32, dequantizing weights into padded threadgroup
// half tiles through `psgm_dequant_tile`, the format seam: family 0 is INT4
// affine, family 1 the GGUF LUT4 execution format (`GGUFLUT4`), selected by
// function constant 113. Accumulation is float; the SwiGLU product and the
// route partials are stored as half, as the MPP kernel does.

#ifndef MFERENCE_AFFINE_GROUP_SIZE
#define MFERENCE_AFFINE_GROUP_SIZE
constant uint FC_AFFINE_GROUP_SIZE [[function_constant(108)]];
constant uint kAffineGroupSize = is_function_constant_defined(FC_AFFINE_GROUP_SIZE)
    ? FC_AFFINE_GROUP_SIZE : 64u;
#endif

#define PSGM_UNROLL _Pragma("clang loop unroll(full)")

// Dequant family: 0 = INT4 affine (default), 1 = LUT4.
constant uint FC_PSGM_DEQUANT [[function_constant(113)]];
constant uint kPSGMDequant = is_function_constant_defined(FC_PSGM_DEQUANT) ? FC_PSGM_DEQUANT : 0u;
constant bool kPSGMLUT4 = kPSGMDequant == 1u;

constant constexpr uint kPSGMMaxExperts = 16;
constant constexpr uint kPSGMRows = 64;            // routed rows per threadgroup
constant constexpr uint kPSGMK = 32;               // K step; k0 % 32 == 0
constant constexpr uint kPSGMLdk = kPSGMK + 8;     // padded tile row, halves
constant constexpr uint kPSGMPhase1Cols = 64;      // gate and up outputs each
constant constexpr uint kPSGMDownCols = 128;       // down outputs
constant constexpr uint kPSGMThreads = 128;        // four simdgroups, 2 x 2

struct PSGMPair { uint token; uint expert; uint rank; uint weight_bits_and_reserved; };
struct PSGMGroup { uint expert; uint pair_start; uint pair_count; };
struct PSGMExperts { device const uint8_t* blob[kPSGMMaxExperts]; };

// Layout-identical to `PrefillGroupedRoutedMoEStreamedParams` plus the
// resident stride word, as `MPPGroupedParams`.
struct PSGMParams {
    uint pair_start;
    uint pair_count;
    uint D;
    uint F;
    uint top_k;
    uint hidden_stride;
    uint live_expert_count;
    uint local_expert[kPSGMMaxExperts];
    uint gate_W_off; uint gate_s_off; uint gate_b_off;
    uint up_W_off; uint up_s_off; uint up_b_off;
    uint down_W_off; uint down_s_off; uint down_b_off;
    uint resident_expert_stride;
};

// One projection inside an expert blob. Bound per dispatch, never baked into
// a pipeline: GGUF families vary formats and strides by layer. INT4 affine
// reads (W, scale, bias) at (w_off, s_off, b_off); LUT4 reads (indices, D, M)
// there, with its block geometry and 2^K below.
struct PSGMProjFormat {
    uint w_off; uint s_off; uint b_off; uint k_extent;
    uint bd_shift; uint bm_shift; uint has_m; float scale_k;
};

// Layout-identical to `GGUFLUT4MoE.ProjParams` / `LUT4ProjParams`: one per
// projection (gate, up, down), bound at buffer 7 for the LUT4 family.
struct PSGMLUT4Proj {
    uint bd_shift; uint bm_shift; uint has_m; float scale_k; float table[16];
};

static inline PSGMProjFormat psgm_format(uint w, uint s, uint b, uint k_extent,
                                         constant PSGMLUT4Proj* lut4, uint proj) {
    PSGMProjFormat f = {w, s, b, k_extent, 0u, 0u, 0u, 1.0f};
    if (kPSGMLUT4) {
        f.bd_shift = lut4[proj].bd_shift;
        f.bm_shift = lut4[proj].bm_shift;
        f.has_m = lut4[proj].has_m;
        f.scale_k = lut4[proj].scale_k;
    }
    return f;
}

// (row, col) of the two elements a lane holds in an 8x8 simdgroup matrix:
// (row, col) and (row, col + 1).
static inline ushort2 psgm_frag_coord(ushort lane) {
    const ushort qid = lane / 4;
    return ushort2((qid & 4) + ((lane / 2) % 4), (qid & 2) * 2 + (lane % 2) * 2);
}

static inline uint psgm_expert_slot(constant PSGMParams& p, uint expert) {
    for (uint slot = 0; slot < p.live_expert_count; ++slot) {
        if (p.local_expert[slot] == expert) return slot;
    }
    return kPSGMMaxExperts;
}

// The format seam: dequantize weight rows [row0, row0 + rows) x K
// [k0, k0 + 32) into threadgroup half, row-major with stride `ldk`. Each
// thread owns whole row segments (row r for r = tid, tid + threads, ...).
// Family 0 is INT4 affine: sequential nibbles, bf16 scale and bias per
// `kAffineGroupSize` weights. `table` is reserved for a 16-entry LUT family.
static inline void psgm_dequant_tile(device const uint8_t* blob, PSGMProjFormat fmt,
                                     threadgroup const float* table,
                                     uint row0, uint rows, uint k0,
                                     threadgroup half* dst, uint ldk,
                                     uint tid, uint threads) {
    if (kPSGMLUT4) {
        // LUT4: weight = ((D * M) * T[idx]) * 2^K, exact in fp32, rounded to
        // half once. Within each 32 weights, index byte r holds weight r (low
        // nibble) and weight r + 16 (high nibble); k0 % 32 == 0, so one step
        // is one such group: one D (BD >= 32) and one M per 16 or 32 weights.
        for (uint r = tid; r < rows; r += threads) {
            const uint n = row0 + r;
            const ulong first = ulong(n) * fmt.k_extent + k0;
            device const uint* q = reinterpret_cast<device const uint*>(
                blob + fmt.w_off + n * (fmt.k_extent / 2u) + k0 / 2u);
            const float d = float(reinterpret_cast<device const half*>(blob + fmt.s_off)[first >> fmt.bd_shift]);
            float m_lo = 1.0f, m_hi = 1.0f;
            if (fmt.has_m != 0u) {
                device const char* m = reinterpret_cast<device const char*>(blob + fmt.b_off);
                m_lo = float(m[first >> fmt.bm_shift]);
                m_hi = float(m[(first + 16u) >> fmt.bm_shift]);
            }
            const float dm_lo = d * m_lo, dm_hi = d * m_hi;
            threadgroup half* out = dst + r * ldk;
            PSGM_UNROLL for (uint c = 0; c < 4u; ++c) {
                const uint packed = q[c];
                PSGM_UNROLL for (uint j = 0; j < 4u; ++j) {
                    const uint byte = (packed >> (8u * j)) & 0xFFu;
                    out[4u * c + j] = half((dm_lo * table[byte & 0xFu]) * fmt.scale_k);
                    out[4u * c + j + 16u] = half((dm_hi * table[byte >> 4]) * fmt.scale_k);
                }
            }
        }
        return;
    }
    const uint groups_per_row = fmt.k_extent / kAffineGroupSize;
    device const bfloat* scales = reinterpret_cast<device const bfloat*>(blob + fmt.s_off);
    device const bfloat* biases = reinterpret_cast<device const bfloat*>(blob + fmt.b_off);
    for (uint r = tid; r < rows; r += threads) {
        const uint n = row0 + r;
        device const uint* q = reinterpret_cast<device const uint*>(
            blob + fmt.w_off + n * (fmt.k_extent / 2u) + k0 / 2u);
        threadgroup half* d = dst + r * ldk;
        PSGM_UNROLL for (uint c = 0; c < kPSGMK / 8u; ++c) {
            const uint group = n * groups_per_row + (k0 + c * 8u) / kAffineGroupSize;
            const float scale = float(scales[group]);
            const float bias = float(biases[group]);
            const uint packed = q[c];
            PSGM_UNROLL for (uint j = 0; j < 8; ++j) {
                d[c * 8u + j] = half(fma(float((packed >> (4u * j)) & 0xFu), scale, bias));
            }
        }
    }
}

// Rows [0, 64) x K [k0, k0 + 32) of the threadgroup's routed rows; rows past
// the group's end are zero. Each thread copies half a row (16 halves).
static inline void psgm_load_rows(device const half* src_row, bool valid, uint k0,
                                  threadgroup half* dst_row) {
    threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(dst_row);
    if (valid) {
        device const half4* src = reinterpret_cast<device const half4*>(src_row + k0);
        PSGM_UNROLL for (uint i = 0; i < kPSGMK / 8u; ++i) dst[i] = src[i];
    } else {
        PSGM_UNROLL for (uint i = 0; i < kPSGMK / 8u; ++i) dst[i] = half4(0.0h);
    }
}

kernel void prefill_simd_grouped_moe_phase1(
    device const half* hidden                [[buffer(0)]],
    device const PSGMPair* pairs             [[buffer(1)]],
    device const PSGMGroup* groups           [[buffer(2)]],
    device half* activation                  [[buffer(3)]],
    device half* route_partials              [[buffer(4)]],
    device const PSGMExperts& experts        [[buffer(5)]],
    constant PSGMParams& p                   [[buffer(6)]],
    constant PSGMLUT4Proj* lut4              [[buffer(7)]],   // LUT4 only; INT4 binds zeros
    uint3 tgid  [[threadgroup_position_in_grid]],
    ushort tid  [[thread_index_in_threadgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (tgid.z >= p.pair_count) return;
    const PSGMGroup group = groups[p.pair_start + tgid.z];
    const uint row_base = tgid.y * kPSGMRows;
    if (row_base >= group.pair_count) return;
    const uint slot = psgm_expert_slot(p, group.expert);
    if (slot >= p.live_expert_count) return;
    device const uint8_t* expert = experts.blob[slot];
    const uint n_base = tgid.x * kPSGMPhase1Cols;
    const uint tile_pair_start = groups[p.pair_start].pair_start;

    threadgroup half smem[kPSGMRows * kPSGMLdk + 2 * kPSGMPhase1Cols * kPSGMLdk];
    threadgroup half* rows_tile = smem;
    threadgroup half* gate_tile = smem + kPSGMRows * kPSGMLdk;
    threadgroup half* up_tile = gate_tile + kPSGMPhase1Cols * kPSGMLdk;

    const uint a_row = tid / 2u, a_k = (tid % 2u) * (kPSGMK / 2u);
    const bool a_valid = row_base + a_row < group.pair_count;
    device const half* a_src = hidden;
    if (a_valid) {
        a_src = hidden + pairs[group.pair_start + row_base + a_row].token * p.hidden_stride + a_k;
    }
    const PSGMProjFormat gate_fmt = psgm_format(p.gate_W_off, p.gate_s_off, p.gate_b_off, p.D, lut4, 0u);
    const PSGMProjFormat up_fmt = psgm_format(p.up_W_off, p.up_s_off, p.up_b_off, p.D, lut4, 1u);
    // LUT4 tables, gate's then up's, staged once.
    threadgroup float tables[32];
    if (kPSGMLUT4) {
        if (tid < 32u) tables[tid] = lut4[tid / 16u].table[tid % 16u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_float8x8 acc_gate[4][4], acc_up[4][4];
    PSGM_UNROLL for (uint i = 0; i < 4; ++i) PSGM_UNROLL for (uint j = 0; j < 4; ++j) {
        acc_gate[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc_up[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint sg_m = (sg / 2u) * 32u, sg_n = (sg % 2u) * 32u;
    const ushort2 fc = psgm_frag_coord(lane);

    for (uint k0 = 0; k0 < p.D; k0 += kPSGMK) {
        psgm_load_rows(a_src, a_valid, k0, rows_tile + a_row * kPSGMLdk + a_k);
        // Simdgroups 0-1 dequantize gate and 2-3 up, so each half can carry
        // its own format (and table) without divergence inside a simdgroup.
        if (tid < kPSGMPhase1Cols) {
            psgm_dequant_tile(expert, gate_fmt, tables, n_base, kPSGMPhase1Cols, k0,
                              gate_tile, kPSGMLdk, tid, kPSGMPhase1Cols);
        } else {
            psgm_dequant_tile(expert, up_fmt, tables + 16, n_base, kPSGMPhase1Cols, k0,
                              up_tile, kPSGMLdk, tid - kPSGMPhase1Cols, kPSGMPhase1Cols);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        PSGM_UNROLL for (uint kk = 0; kk < kPSGMK; kk += 8) {
            simdgroup_float8x8 a[4], g[4], u[4];
            PSGM_UNROLL for (uint i = 0; i < 4; ++i) {
                threadgroup const half* src = rows_tile + (sg_m + i * 8u + fc.x) * kPSGMLdk + kk + fc.y;
                a[i].thread_elements()[0] = float(src[0]);
                a[i].thread_elements()[1] = float(src[1]);
            }
            PSGM_UNROLL for (uint j = 0; j < 4; ++j) {
                // B[k][n] = W[n][k]: the lane's (k, n) and (k, n + 1).
                const uint at = (sg_n + j * 8u + fc.y) * kPSGMLdk + kk + fc.x;
                g[j].thread_elements()[0] = float(gate_tile[at]);
                g[j].thread_elements()[1] = float(gate_tile[at + kPSGMLdk]);
                u[j].thread_elements()[0] = float(up_tile[at]);
                u[j].thread_elements()[1] = float(up_tile[at + kPSGMLdk]);
            }
            PSGM_UNROLL for (uint i = 0; i < 4; ++i) PSGM_UNROLL for (uint j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(acc_gate[i][j], a[i], g[j], acc_gate[i][j]);
                simdgroup_multiply_accumulate(acc_up[i][j], a[i], u[j], acc_up[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // SwiGLU in float, staged through threadgroup memory so partial row
    // blocks write only their own rows.
    constexpr uint kStage = kPSGMPhase1Cols + 8;
    threadgroup half* stage = smem;
    PSGM_UNROLL for (uint i = 0; i < 4; ++i) PSGM_UNROLL for (uint j = 0; j < 4; ++j) {
        const uint r = sg_m + i * 8u + fc.x, c = sg_n + j * 8u + fc.y;
        PSGM_UNROLL for (uint e = 0; e < 2; ++e) {
            const float gv = acc_gate[i][j].thread_elements()[e];
            const float uv = acc_up[i][j].thread_elements()[e];
            stage[r * kStage + c + e] = half((gv / (1.0f + exp(-gv))) * uv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint out_row = tid / 2u, out_col = (tid % 2u) * (kPSGMPhase1Cols / 2u);
    if (row_base + out_row < group.pair_count) {
        const uint local_pair = group.pair_start + row_base + out_row - tile_pair_start;
        device half* dst = activation + local_pair * p.F + n_base + out_col;
        threadgroup const half* src = stage + out_row * kStage + out_col;
        for (uint i = 0; i < kPSGMPhase1Cols / 2u; ++i) dst[i] = src[i];
    }
}

kernel void prefill_simd_grouped_moe_down(
    device const half* hidden                [[buffer(0)]],
    device const PSGMPair* pairs             [[buffer(1)]],
    device const PSGMGroup* groups           [[buffer(2)]],
    device const half* activation            [[buffer(3)]],
    device half* route_partials              [[buffer(4)]],
    device const PSGMExperts& experts        [[buffer(5)]],
    constant PSGMParams& p                   [[buffer(6)]],
    constant PSGMLUT4Proj* lut4              [[buffer(7)]],   // LUT4 only; INT4 binds zeros
    uint3 tgid  [[threadgroup_position_in_grid]],
    ushort tid  [[thread_index_in_threadgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (tgid.z >= p.pair_count) return;
    const PSGMGroup group = groups[p.pair_start + tgid.z];
    const uint row_base = tgid.y * kPSGMRows;
    if (row_base >= group.pair_count) return;
    const uint slot = psgm_expert_slot(p, group.expert);
    if (slot >= p.live_expert_count) return;
    device const uint8_t* expert = experts.blob[slot];
    const uint n_base = tgid.x * kPSGMDownCols;
    const uint tile_pair_start = groups[p.pair_start].pair_start;

    constexpr uint kStage = kPSGMDownCols + 8;
    constexpr uint kTiles = (kPSGMRows + kPSGMDownCols) * kPSGMLdk;
    constexpr uint kSmem = kPSGMRows * kStage > kTiles ? kPSGMRows * kStage : kTiles;
    threadgroup half smem[kSmem];
    threadgroup half* rows_tile = smem;
    threadgroup half* weight_tile = smem + kPSGMRows * kPSGMLdk;

    const uint a_row = tid / 2u, a_k = (tid % 2u) * (kPSGMK / 2u);
    const bool a_valid = row_base + a_row < group.pair_count;
    device const half* a_src = activation;
    if (a_valid) {
        a_src = activation + (group.pair_start + row_base + a_row - tile_pair_start) * p.F + a_k;
    }
    const PSGMProjFormat down_fmt = psgm_format(p.down_W_off, p.down_s_off, p.down_b_off, p.F, lut4, 2u);
    threadgroup float table[16];
    if (kPSGMLUT4) {
        if (tid < 16u) table[tid] = lut4[2].table[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_float8x8 acc[4][8];
    PSGM_UNROLL for (uint i = 0; i < 4; ++i) PSGM_UNROLL for (uint j = 0; j < 8; ++j) {
        acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint sg_m = (sg / 2u) * 32u, sg_n = (sg % 2u) * 64u;
    const ushort2 fc = psgm_frag_coord(lane);

    for (uint k0 = 0; k0 < p.F; k0 += kPSGMK) {
        psgm_load_rows(a_src, a_valid, k0, rows_tile + a_row * kPSGMLdk + a_k);
        psgm_dequant_tile(expert, down_fmt, table, n_base, kPSGMDownCols, k0,
                          weight_tile, kPSGMLdk, tid, kPSGMThreads);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        PSGM_UNROLL for (uint kk = 0; kk < kPSGMK; kk += 8) {
            simdgroup_float8x8 a[4], b[8];
            PSGM_UNROLL for (uint i = 0; i < 4; ++i) {
                threadgroup const half* src = rows_tile + (sg_m + i * 8u + fc.x) * kPSGMLdk + kk + fc.y;
                a[i].thread_elements()[0] = float(src[0]);
                a[i].thread_elements()[1] = float(src[1]);
            }
            PSGM_UNROLL for (uint j = 0; j < 8; ++j) {
                const uint at = (sg_n + j * 8u + fc.y) * kPSGMLdk + kk + fc.x;
                b[j].thread_elements()[0] = float(weight_tile[at]);
                b[j].thread_elements()[1] = float(weight_tile[at + kPSGMLdk]);
            }
            PSGM_UNROLL for (uint i = 0; i < 4; ++i) PSGM_UNROLL for (uint j = 0; j < 8; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup half* stage = smem;
    PSGM_UNROLL for (uint i = 0; i < 4; ++i) PSGM_UNROLL for (uint j = 0; j < 8; ++j) {
        const uint r = sg_m + i * 8u + fc.x, c = sg_n + j * 8u + fc.y;
        stage[r * kStage + c] = half(acc[i][j].thread_elements()[0]);
        stage[r * kStage + c + 1] = half(acc[i][j].thread_elements()[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint out_row = tid / 2u, out_col = (tid % 2u) * (kPSGMDownCols / 2u);
    if (row_base + out_row < group.pair_count) {
        const PSGMPair pair = pairs[group.pair_start + row_base + out_row];
        device half* dst = route_partials + (pair.token * p.top_k + pair.rank) * p.D + n_base + out_col;
        threadgroup const half* src = stage + out_row * kStage + out_col;
        for (uint i = 0; i < kPSGMDownCols / 2u; ++i) dst[i] = src[i];
    }
}

#undef PSGM_UNROLL
