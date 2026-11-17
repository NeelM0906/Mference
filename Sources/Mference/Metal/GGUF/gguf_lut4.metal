// LUT4 expansion of native GGUF blocks (WS4 U1 §3a). Compiled on demand
// (MetalContext.moduleLibrary, module "gguf_lut4"); not part of the shared
// runtime library.
//
// A job expands `n_blocks` consecutive source blocks of one projection from
// `src` into the planar LUT4 sections of a slot in `dst`:
//   indices: 4 bits per weight, row-major; within each 32 weights byte r holds
//            weight r (low nibble) and weight r + 16 (high nibble);
//   D:       the block's fp16 scale bits, one per block;
//   M:       int8 multipliers, BLOCK_WEIGHTS / BM per block (types with M).
// The integer work here must match GGUFLUT4Reference.decompose bit for bit.

#include <metal_stdlib>
using namespace metal;

#ifndef MFERENCE_GGUF_LUT4_COMMON
#define MFERENCE_GGUF_LUT4_COMMON

// ggml type ids (ggml.h at llama.cpp b1b770d9).
constant uint kLUT4IQ3XXS = 18;
constant uint kLUT4IQ4NL = 20;
constant uint kLUT4IQ2S = 22;
constant uint kLUT4IQ4XS = 23;
constant uint kLUT4Q20 = 42;

constant uint lut4_source_type [[function_constant(0)]];

struct LUT4ExpandJob {
    ulong src_offset;        // first source block, bytes into src
    ulong dst_index_offset;  // first index byte of the block range, bytes into dst (16-aligned)
    ulong dst_d_offset;      // first D (fp16) of the block range, bytes into dst
    ulong dst_m_offset;      // first M (int8) of the block range, bytes into dst
    uint n_blocks;
    uint pad;
};

static inline uint lut4_block_bytes(uint t) {
    return t == kLUT4IQ3XXS ? 98u : t == kLUT4IQ2S ? 82u : t == kLUT4IQ4XS ? 136u : 18u;
}

static inline uint lut4_block_groups(uint t) {
    return (t == kLUT4IQ4NL) ? 1u : (t == kLUT4Q20) ? 2u : 8u;
}

static inline uint lut4_u32(device const uchar* p) {
    return uint(p[0]) | uint(p[1]) << 8 | uint(p[2]) << 16 | uint(p[3]) << 24;
}

// Pack 32 indices (natural order) into 16 bytes, IQ4_NL nibble order.
static inline uint4 lut4_pack32(thread const uchar* idx) {
    uint w[4];
    for (uint q = 0; q < 4; ++q) {
        uint v = 0;
        for (uint b = 0; b < 4; ++b) {
            uint r = 4 * q + b;
            v |= (uint(idx[r]) | uint(idx[r + 16]) << 4) << (8 * b);
        }
        w[q] = v;
    }
    return uint4(w[0], w[1], w[2], w[3]);
}

#endif

kernel void gguf_lut4_expand(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant LUT4ExpandJob* jobs [[buffer(2)]],
    constant ushort* index_grid [[buffer(3)]],   // iq3xxs (256) or iq2s (1024) magnitude-index grid
    constant uchar* ksigns [[buffer(4)]],        // ksigns_iq2xs (128)
    uint2 gid [[thread_position_in_grid]])
{
    const LUT4ExpandJob job = jobs[gid.y];
    const uint block = gid.x;
    if (block >= job.n_blocks) { return; }
    const uint t = lut4_source_type;
    device const uchar* b = src + job.src_offset + ulong(block) * lut4_block_bytes(t);
    const uint groups = lut4_block_groups(t);
    device uint4* out = (device uint4*)(dst + job.dst_index_offset + ulong(block) * groups * 16);

    // D: the block's fp16 scale bits, unchanged.
    device uchar* dptr = dst + job.dst_d_offset + ulong(block) * 2;
    dptr[0] = b[0];
    dptr[1] = b[1];

    uchar idx[32];
    if (t == kLUT4IQ4NL) {
        // The 16 qs bytes are already LUT4 index bytes.
        uint w[4];
        for (uint q = 0; q < 4; ++q) { w[q] = lut4_u32(b + 2 + 4 * q); }
        out[0] = uint4(w[0], w[1], w[2], w[3]);
    } else if (t == kLUT4IQ4XS) {
        const uint scales_h = uint(b[2]) | uint(b[3]) << 8;
        device char* mptr = (device char*)(dst + job.dst_m_offset + ulong(block) * 8);
        for (uint g = 0; g < 8; ++g) {
            const uint low = (uint(b[4 + g / 2]) >> (4 * (g % 2))) & 0xF;
            const uint high = (scales_h >> (2 * g)) & 0x3;
            mptr[g] = char(int(low | high << 4) - 32);
            uint w[4];
            for (uint q = 0; q < 4; ++q) { w[q] = lut4_u32(b + 8 + 16 * g + 4 * q); }
            out[g] = uint4(w[0], w[1], w[2], w[3]);
        }
    } else if (t == kLUT4Q20) {
        for (uint g = 0; g < 2; ++g) {
            for (uint r = 0; r < 32; ++r) {
                const uint e = 32 * g + r;
                idx[r] = (b[2 + e / 4] >> (2 * (e % 4))) & 0x3;
            }
            out[g] = lut4_pack32(idx);
        }
    } else if (t == kLUT4IQ3XXS) {
        device char* mptr = (device char*)(dst + job.dst_m_offset + ulong(block) * 8);
        for (uint g = 0; g < 8; ++g) {
            const uint word = lut4_u32(b + 2 + 64 + 4 * g);
            mptr[g] = char(2 * int(word >> 28) + 1);
            for (uint l = 0; l < 4; ++l) {
                const uint signs = ksigns[(word >> (7 * l)) & 127];
                for (uint h = 0; h < 2; ++h) {
                    const uint packed = index_grid[b[2 + 8 * g + 2 * l + h]];
                    for (uint j = 0; j < 4; ++j) {
                        const uint mag = (packed >> (3 * j)) & 0x7;
                        const uint neg = (signs >> (4 * h + j)) & 1;
                        idx[8 * l + 4 * h + j] = uchar(mag | neg << 3);
                    }
                }
            }
            out[g] = lut4_pack32(idx);
        }
    } else if (t == kLUT4IQ2S) {
        device char* mptr = (device char*)(dst + job.dst_m_offset + ulong(block) * 16);
        for (uint s = 0; s < 16; ++s) {
            const uint byte = b[74 + s / 2];
            mptr[s] = char(2 * int((s % 2 == 0) ? (byte & 0xF) : (byte >> 4)) + 1);
        }
        for (uint g = 0; g < 8; ++g) {
            for (uint l = 0; l < 4; ++l) {
                const uint grid_index = uint(b[2 + 4 * g + l]) | ((uint(b[66 + g]) >> (2 * l)) & 0x3) << 8;
                const uint packed = index_grid[grid_index];
                const uint signs = b[2 + 32 + 4 * g + l];
                for (uint j = 0; j < 8; ++j) {
                    const uint mag = (packed >> (2 * j)) & 0x3;
                    const uint neg = (signs >> j) & 1;
                    idx[8 * l + j] = uchar(mag | neg << 3);
                }
            }
            out[g] = lut4_pack32(idx);
        }
    }
}

// ============================================================================
// LUT4 routed-expert decode (one token): phase 1 (gate + up, SiLU · up) and
// phase 2 (down, routing-weighted, residual-first FP32 reduce in rank order).
// The structure mirrors moe_phase1_gate_up_act_u16load and
// moe_phase2_down_reduce_k10: one simdgroup per output row, x / acts in half,
// FP32 accumulation, half outputs.
//
// A LUT4 weight is D · M · 2^K · T[idx] (GGUFLUT4). A lane takes 8 index bytes
// at a time: weights w0..w0+7 (low nibbles) and w0+16..w0+23 (high nibbles)
// of one 32-weight group, which share one D, so it sums T[idx]·x per (D, M)
// pair and scales once.
// ============================================================================

constant uint kLUT4RoutedSlots = 10;

struct LUT4RoutedBlobs {
    device const uchar* blob[kLUT4RoutedSlots];
};

/// Byte offsets of each projection's sections inside an expert slot.
struct LUT4ExpertOffsets {
    uint gate_idx; uint gate_d; uint gate_m;
    uint up_idx;   uint up_d;   uint up_m;
    uint down_idx; uint down_d; uint down_m;
};

/// One projection type's LUT4 parameters.
struct LUT4ProjParams {
    uint bd_shift;   // log2(BD)
    uint bm_shift;   // log2(BM); unused when has_m == 0
    uint has_m;
    float scale_k;   // 2^K
    float table[16]; // T as float
};

static inline float lut4_row_dot(
    device const uchar* idx, device const ushort* dbits, device const char* m,
    constant LUT4ProjParams& p, threadgroup const float* T,
    device const half* x, uint row, uint N, uint lane)
{
    device const uchar* row_idx = idx + row * (N / 2);
    const uint row_w = row * N;
    float acc = 0.0f;
    for (uint hb = lane; hb < N / 16; hb += 32) {
        const uint w0 = 32 * (hb >> 1) + 8 * (hb & 1);
        const uint2 q = *((device const uint2*)(row_idx + 8 * hb));
        const half4 xa = *((device const half4*)(x + w0));
        const half4 xb = *((device const half4*)(x + w0 + 4));
        const half4 ya = *((device const half4*)(x + w0 + 16));
        const half4 yb = *((device const half4*)(x + w0 + 20));
        float lo = 0.0f, hi = 0.0f;
        lo = fma(T[(q.x >>  0) & 0xF], float(xa.x), lo); hi = fma(T[(q.x >>  4) & 0xF], float(ya.x), hi);
        lo = fma(T[(q.x >>  8) & 0xF], float(xa.y), lo); hi = fma(T[(q.x >> 12) & 0xF], float(ya.y), hi);
        lo = fma(T[(q.x >> 16) & 0xF], float(xa.z), lo); hi = fma(T[(q.x >> 20) & 0xF], float(ya.z), hi);
        lo = fma(T[(q.x >> 24) & 0xF], float(xa.w), lo); hi = fma(T[(q.x >> 28) & 0xF], float(ya.w), hi);
        lo = fma(T[(q.y >>  0) & 0xF], float(xb.x), lo); hi = fma(T[(q.y >>  4) & 0xF], float(yb.x), hi);
        lo = fma(T[(q.y >>  8) & 0xF], float(xb.y), lo); hi = fma(T[(q.y >> 12) & 0xF], float(yb.y), hi);
        lo = fma(T[(q.y >> 16) & 0xF], float(xb.z), lo); hi = fma(T[(q.y >> 20) & 0xF], float(yb.z), hi);
        lo = fma(T[(q.y >> 24) & 0xF], float(xb.w), lo); hi = fma(T[(q.y >> 28) & 0xF], float(yb.w), hi);
        const float d = float(as_type<half>(dbits[(row_w + w0) >> p.bd_shift])) * p.scale_k;
        if (p.has_m != 0) {
            const float m0 = float(m[(row_w + w0) >> p.bm_shift]);
            const float m1 = float(m[(row_w + w0 + 16) >> p.bm_shift]);
            acc = fma(d, fma(m0, lo, m1 * hi), acc);
        } else {
            acc = fma(d, lo + hi, acc);
        }
    }
    return simd_sum(acc);
}

kernel void gguf_lut4_moe_phase1_gate_up_act(
    device const LUT4RoutedBlobs& routed [[buffer(0)]],
    constant LUT4ExpertOffsets& off [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    constant LUT4ProjParams& gu [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float T[16];
    if (tid < 16) { T[tid] = gu.table[tid]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rowg = tg_idx * 8 + sg_idx;
    if (rowg >= top_k * F) { return; }
    const uint slot = rowg / F, f = rowg % F;
    device const uchar* base = routed.blob[slot];
    const float g = lut4_row_dot(base + off.gate_idx, (device const ushort*)(base + off.gate_d),
                                 (device const char*)(base + off.gate_m), gu, T, x, f, D, lane);
    const float u = lut4_row_dot(base + off.up_idx, (device const ushort*)(base + off.up_d),
                                 (device const char*)(base + off.up_m), gu, T, x, f, D, lane);
    if (lane == 0) { acts[slot * F + f] = half((g / (1.0f + exp(-g))) * u); }
}

kernel void gguf_lut4_moe_phase2_down_reduce(
    device const LUT4RoutedBlobs& routed [[buffer(0)]],
    constant LUT4ExpertOffsets& off [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    constant LUT4ProjParams& dp [[buffer(9)]],
    uint d [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float T[16];
    threadgroup float partial[kLUT4RoutedSlots];
    if (tid < 16) { T[tid] = dp.table[tid]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= D) { return; }
    device const uchar* base = routed.blob[sg_idx];
    const float value = lut4_row_dot(base + off.down_idx, (device const ushort*)(base + off.down_d),
                                     (device const char*)(base + off.down_m), dp, T,
                                     acts + sg_idx * F, d, F, lane);
    if (lane == 0) { partial[sg_idx] = float(routing_w[sg_idx]) * value; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float acc = float(residual[d]);
        for (uint r = 0; r < top_k; ++r) { acc += partial[r]; }
        y[d] = half(acc);
    }
}

// ============================================================================
// LUT4 routed-expert prefill, streamed tile (row form). Same interface as
// prefill_grouped_routed_moe_batched_phase1 / _down in prefill.metal, so the
// runner's tile binding, pair sorting and token-major reduce are reused:
//   - pairs are PrefillTokenExpertPairMSL {token, expert, rank, weight bits};
//   - routed is the 16-slot tile blob table;
//   - params is PrefillGroupedRoutedMoEStreamedParamsMSL with the nine
//     per-projection offsets read as LUT4 (index, D, M) for gate, up, down;
//   - phase 1 writes gate, up and silu(gate)·up into gate_up_act_scratch
//     (three planes of pair_count × F half); down writes the unweighted
//     down value to down_scratch and route_partials[(token·top_k + rank)·D + d].
// One simdgroup per (output row, pair), using lut4_row_dot (decode-tested).
// A grouped simdgroup-matrix form is a later optimization (WS1 P3b's tiling).
// ============================================================================

constant uint kLUT4PrefillMaxTileExperts = 16;

struct LUT4PrefillPair { uint token; uint expert; uint rank; uint weight_bits; };

struct LUT4PrefillBlobs { device const uchar* blob[kLUT4PrefillMaxTileExperts]; };

struct LUT4PrefillParams {
    uint pair_start, pair_count, D, F, top_k, hidden_stride_elements, live_expert_count;
    uint local_expert[kLUT4PrefillMaxTileExperts];
    uint gate_idx, gate_d, gate_m, up_idx, up_d, up_m, down_idx, down_d, down_m;
};

static inline uint lut4_prefill_local_slot(constant LUT4PrefillParams& p, uint expert) {
    for (uint s = 0; s < p.live_expert_count; ++s) {
        if (p.local_expert[s] == expert) { return s; }
    }
    return kLUT4PrefillMaxTileExperts;
}

// Grid: x = F * 32 (one simdgroup per row f), y = pair_count.
// Threadgroup: (256, 1, 1).
kernel void gguf_lut4_prefill_phase1(
    device const half* hidden [[buffer(0)]],
    device const LUT4PrefillPair* sorted_pairs [[buffer(1)]],
    device half* gate_up_act_scratch [[buffer(7)]],
    device const LUT4PrefillBlobs& routed [[buffer(9)]],
    constant LUT4PrefillParams& p [[buffer(10)]],
    constant LUT4ProjParams& gu [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float T[16];
    if (tid < 16) { T[tid] = gu.table[tid]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint f = gid.x / 32, lane = gid.x % 32, pair_local = gid.y;
    if (f >= p.F || pair_local >= p.pair_count) { return; }
    const LUT4PrefillPair pair = sorted_pairs[p.pair_start + pair_local];
    const uint slot = lut4_prefill_local_slot(p, pair.expert);
    if (slot >= p.live_expert_count) { return; }
    device const uchar* base = routed.blob[slot];
    device const half* x = hidden + pair.token * p.hidden_stride_elements;
    const float g = lut4_row_dot(base + p.gate_idx, (device const ushort*)(base + p.gate_d),
                                 (device const char*)(base + p.gate_m), gu, T, x, f, p.D, lane);
    const float u = lut4_row_dot(base + p.up_idx, (device const ushort*)(base + p.up_d),
                                 (device const char*)(base + p.up_m), gu, T, x, f, p.D, lane);
    if (lane != 0) { return; }
    const uint plane = p.pair_count * p.F, index = pair_local * p.F + f;
    gate_up_act_scratch[index] = half(g);
    gate_up_act_scratch[plane + index] = half(u);
    gate_up_act_scratch[2 * plane + index] = half((g / (1.0f + exp(-g))) * u);
}

// Grid: x = D * 32 (one simdgroup per output row d), y = pair_count.
kernel void gguf_lut4_prefill_down(
    device const LUT4PrefillPair* sorted_pairs [[buffer(1)]],
    device half* route_partials [[buffer(5)]],
    device const half* gate_up_act_scratch [[buffer(7)]],
    device half* down_scratch [[buffer(8)]],
    device const LUT4PrefillBlobs& routed [[buffer(9)]],
    constant LUT4PrefillParams& p [[buffer(10)]],
    constant LUT4ProjParams& dp [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float T[16];
    if (tid < 16) { T[tid] = dp.table[tid]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d = gid.x / 32, lane = gid.x % 32, pair_local = gid.y;
    if (d >= p.D || pair_local >= p.pair_count) { return; }
    const LUT4PrefillPair pair = sorted_pairs[p.pair_start + pair_local];
    const uint slot = lut4_prefill_local_slot(p, pair.expert);
    if (slot >= p.live_expert_count) { return; }
    device const uchar* base = routed.blob[slot];
    device const half* act = gate_up_act_scratch + 2 * p.pair_count * p.F + pair_local * p.F;
    const half value = half(lut4_row_dot(base + p.down_idx, (device const ushort*)(base + p.down_d),
                                         (device const char*)(base + p.down_m), dp, T, act, d, p.F, lane));
    if (lane != 0) { return; }
    down_scratch[pair_local * p.D + d] = value;
    route_partials[(pair.token * p.top_k + pair.rank) * p.D + d] = value;
}
