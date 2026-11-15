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
