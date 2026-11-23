#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Inkling 2 Mini attention side: stacked Q8_0 projections, the QKV epilogue
// (clips, short convolutions, value norm, KV cache store), relative-bias
// attention with a per-query sink for prefill and decode, and the FP32 short
// convolution of the attention and MoE outputs.
//
// Compiled on demand as the module `inkling2_attention`
// (`MetalContext.moduleLibrary`), not in the shared runtime library. The CPU
// references are in `Sources/MferenceValidation/Support/Reference/Inkling2`.
//
// 8x8 simdgroup fragments: lane l holds row fm = ((l/4)&4) + ((l/2)%4) and
// columns fn, fn+1 with fn = ((l/4)&2)*2 + (l%2)*2 (the layout llama.cpp's
// mul_mv_mma and MLX's steel kernels rely on). The lanes that share a row
// differ in lane bits 0 and 3, so a row reduction is two xor shuffles.
// ============================================================================

// Loops over arrays of simdgroup fragments must unroll fully, or the arrays
// are indexed at run time and spill.
#define INK2_UNROLL _Pragma("clang loop unroll(full)")

static inline short ink2_frag_row(ushort lane) { return ((lane / 4) & 4) + ((lane / 2) % 4); }
static inline short ink2_frag_col(ushort lane) { return ((lane / 4) & 2) * 2 + (lane % 2) * 2; }

static inline float ink2_row_max(float v) {
    v = max(v, simd_shuffle_xor(v, 1));
    return max(v, simd_shuffle_xor(v, 8));
}
static inline float ink2_row_sum(float v) {
    v += simd_shuffle_xor(v, 1);
    return v + simd_shuffle_xor(v, 8);
}

// ----------------------------------------------------------------------------
// Stacked Q8_0 projection
//
// out[t, outColumn + r] = sum_k input[t, k] * W[r, k] for every segment, where
// W is a GGUF Q8_0 weight (rows of K/32 blocks of {half d; int8 q[32]}, 34
// bytes, value d*q), a BF16 weight (rows of K bfloats; the file stores the
// sink and dynamic-tap projections that way) or an F32 weight (the loop
// gates' down projections). Segments let every projection that shares a visit's
// inputs run as one dispatch: the query-side group (q, relative states, sink,
// output gate), keys (+ dynamic taps) and values, each with its own input.
// All segments read one weight buffer and one input buffer at their own
// offsets and write disjoint column ranges of one FP32 output.
//
// Tile: 64 weight rows x 32 tokens per threadgroup, K step 32 = one Q8_0 block
// per row; 4 simdgroups each own 32 rows x 16 tokens. Weights are decoded to
// half once per tile into threadgroup memory in 8x8 blocks, activations are
// half, accumulation is FP32. Grid: x = ceil(tokens / 32), y = total 64-row
// tiles over all segments.
// ----------------------------------------------------------------------------

#define INK2_PROJ_MAX_SEGMENTS 8

struct Inkling2ProjSegment {
    ulong weightOffset;   // bytes into `weights` of the segment's row 0
    uint  inputOffset;    // elements into `input` of token 0
    uint  inputStride;    // elements between token rows of the input
    uint  rows;           // weight rows (output columns) of this segment
    uint  outColumn;      // first output column
    uint  tileStart;      // first tile (or row pair) of the segment along the grid
    uint  format;         // 0 = GGUF Q8_0 blocks, 1 = BF16 rows, 2 = F32 rows
};

struct Inkling2ProjParams {
    uint tokens;
    uint inputs;          // K, a multiple of 32
    uint outStride;       // floats between output rows
    uint segmentCount;
    Inkling2ProjSegment segments[INK2_PROJ_MAX_SEGMENTS];
};

[[kernel, max_total_threads_per_threadgroup(128)]]
void inkling2_q8_stacked_mm(
    constant Inkling2ProjParams& p [[buffer(0)]],
    device const uchar* weights    [[buffer(1)]],
    device const half*  input      [[buffer(2)]],
    device float*       output     [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float shmem[2048];                 // 8 KB: A (4 KB) + B (2 KB); edge store (8 KB)
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = sa + 64 * 32;

    const uint tile = tgpig.y;
    uint si = 0;
    for (uint i = 1; i < p.segmentCount; ++i) {
        if (tile >= p.segments[i].tileStart) { si = i; }
    }
    const Inkling2ProjSegment seg = p.segments[si];
    const uint r0 = (tile - seg.tileStart) * 64u;
    const uint t0 = tgpig.x * 32u;
    if (r0 >= seg.rows || t0 >= p.tokens) { return; }
    const short nr0 = short(min(64u, seg.rows - r0));
    const short nr1 = short(min(32u, p.tokens - t0));

    // A: thread -> (row tiitg/2, 16 values of the block's half tiitg%2).
    const short ar  = short(tiitg / 2);
    const short il0 = short(tiitg % 2);
    const short lr0 = min(ar, short(nr0 - 1));
    // B: thread -> (token tiitg/4, 8 values at 8*(tiitg%4)).
    const short br  = short(tiitg / 4);
    const short lr1 = min(br, short(nr1 - 1));
    const short iy  = short(8 * (tiitg % 4));

    const uint blocks = p.inputs / 32u;
    const uint format = seg.format;
    const uint rowBytes = format == 2u ? p.inputs * 4u : format == 1u ? p.inputs * 2u : blocks * 34u;
    device const uchar* arow = weights + seg.weightOffset + ulong(r0 + uint(lr0)) * rowBytes;
    device const half* y = input + seg.inputOffset + ulong(t0 + uint(lr1)) * seg.inputStride + uint(iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }

    const short asy = ar / 8, alx = ar % 8;
    const short bsx = short(tiitg % 4), bsy = br / 8, bly = br % 8;

    for (uint kb = 0; kb < blocks; ++kb) {
        half w[16];
        if (format == 2u) {
            device const float4* src = (device const float4*)(arow + (kb * 32u + 16u * uint(il0)) * 4u);
            for (short i = 0; i < 4; ++i) {
                const half4 v = half4(src[i]);
                w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w;
            }
        } else if (format == 1u) {
            device const bfloat4* src = (device const bfloat4*)(arow + (kb * 32u + 16u * uint(il0)) * 2u);
            for (short i = 0; i < 4; ++i) {
                const half4 v = half4(float4(src[i]));
                w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w;
            }
        } else {
            device const uchar* blk = arow + kb * 34u;
            const half d = *((device const half*)blk);
            device const ushort* qs = (device const ushort*)(blk + 2u + 16u * uint(il0));
            for (short i = 0; i < 8; ++i) {
                const char2 c = as_type<char2>(qs[i]);
                w[2 * i]     = d * half(c.x);
                w[2 * i + 1] = d * half(c.y);
            }
        }
        const half4 y0 = *((device const half4*)y);
        const half4 y1 = *((device const half4*)(y + 4));

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short ib = 8 * (2 * il0 + i / 8) + asy;     // [k block][row block]
            sa[64 * ib + 8 * (i % 8) + alx] = w[i];          // block stored [k][row]
        }
        {
            const short ib = 4 * bsx + bsy;                   // [k block][token block]
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * bly);
            dst[0] = y0;
            dst[1] = y1;
        }
        y += 32;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        INK2_UNROLL
        for (short ik = 0; ik < 4; ++ik) {
            INK2_UNROLL
            for (short i = 0; i < 4; ++i) { simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false); }
            INK2_UNROLL
            for (short i = 0; i < 2; ++i) { simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false); }
            INK2_UNROLL
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    const ulong column = ulong(seg.outColumn) + r0 + 32u * (sgitg & 1);
    if (nr0 == 64 && nr1 == 32) {
        device float* C = output + ulong(t0 + 16u * (sgitg >> 1)) * p.outStride + column;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + ulong(8 * (i / 4)) * p.outStride, p.outStride, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* tmp = shmem;                       // [32 tokens][64 rows]
        threadgroup float* base = tmp + (16 * (sgitg >> 1)) * 64 + 32 * (sgitg & 1);
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], base + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tiitg; e < 32u * 64u; e += 128u) {
            const uint tok = e / 64u, row = e % 64u;
            if (tok < uint(nr1) && row < uint(nr0)) {
                output[ulong(t0 + tok) * p.outStride + seg.outColumn + r0 + row] = tmp[e];
            }
        }
    }
}

// Same product with 64 x 64 tiles: each simdgroup owns 32 rows x 32 tokens
// (16 accumulators), halving the fragment loads per multiply. Grid: x =
// ceil(tokens / 64), y = 64-row tiles as for the 32-token kernel.
[[kernel, max_total_threads_per_threadgroup(128)]]
void inkling2_q8_stacked_mm64(
    constant Inkling2ProjParams& p [[buffer(0)]],
    device const uchar* weights    [[buffer(1)]],
    device const half*  input      [[buffer(2)]],
    device float*       output     [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float shmem[4096];                 // 16 KB: A (4 KB) + B (4 KB); edge store (16 KB)
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = sa + 64 * 32;

    const uint tile = tgpig.y;
    uint si = 0;
    for (uint i = 1; i < p.segmentCount; ++i) {
        if (tile >= p.segments[i].tileStart) { si = i; }
    }
    const Inkling2ProjSegment seg = p.segments[si];
    const uint r0 = (tile - seg.tileStart) * 64u;
    const uint t0 = tgpig.x * 64u;
    if (r0 >= seg.rows || t0 >= p.tokens) { return; }
    const short nr0 = short(min(64u, seg.rows - r0));
    const short nr1 = short(min(64u, p.tokens - t0));

    // A: thread -> (row tiitg/2, 16 values of the block's half tiitg%2).
    const short ar  = short(tiitg / 2);
    const short il0 = short(tiitg % 2);
    const short lr0 = min(ar, short(nr0 - 1));
    // B: thread -> (token tiitg/2, 16 values at 16*(tiitg%2)).
    const short br  = short(tiitg / 2);
    const short lr1 = min(br, short(nr1 - 1));
    const short iy  = short(16 * (tiitg % 2));

    const uint blocks = p.inputs / 32u;
    const uint format = seg.format;
    const uint rowBytes = format == 2u ? p.inputs * 4u : format == 1u ? p.inputs * 2u : blocks * 34u;
    device const uchar* arow = weights + seg.weightOffset + ulong(r0 + uint(lr0)) * rowBytes;
    device const half* y = input + seg.inputOffset + ulong(t0 + uint(lr1)) * seg.inputStride + uint(iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    INK2_UNROLL
    for (short i = 0; i < 16; ++i) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }

    const short asy = ar / 8, alx = ar % 8;
    const short bsx = short(2 * (tiitg % 2)), bsy = br / 8, bly = br % 8;

    for (uint kb = 0; kb < blocks; ++kb) {
        half w[16];
        if (format == 2u) {
            device const float4* src = (device const float4*)(arow + (kb * 32u + 16u * uint(il0)) * 4u);
            for (short i = 0; i < 4; ++i) {
                const half4 v = half4(src[i]);
                w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w;
            }
        } else if (format == 1u) {
            device const bfloat4* src = (device const bfloat4*)(arow + (kb * 32u + 16u * uint(il0)) * 2u);
            for (short i = 0; i < 4; ++i) {
                const half4 v = half4(float4(src[i]));
                w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w;
            }
        } else {
            device const uchar* blk = arow + kb * 34u;
            const half d = *((device const half*)blk);
            device const ushort* qs = (device const ushort*)(blk + 2u + 16u * uint(il0));
            for (short i = 0; i < 8; ++i) {
                const char2 c = as_type<char2>(qs[i]);
                w[2 * i]     = d * half(c.x);
                w[2 * i + 1] = d * half(c.y);
            }
        }
        const half4 y0 = *((device const half4*)y);
        const half4 y1 = *((device const half4*)(y + 4));
        const half4 y2 = *((device const half4*)(y + 8));
        const half4 y3 = *((device const half4*)(y + 12));

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short ib = 8 * (2 * il0 + i / 8) + asy;     // [k block][row block]
            sa[64 * ib + 8 * (i % 8) + alx] = w[i];          // block stored [k][row]
        }
        {
            const short ib0 = 8 * bsx + bsy, ib1 = 8 * (bsx + 1) + bsy;   // [k block][token block]
            threadgroup half4* d0 = (threadgroup half4*)(sb + 64 * ib0 + 8 * bly);
            threadgroup half4* d1 = (threadgroup half4*)(sb + 64 * ib1 + 8 * bly);
            d0[0] = y0; d0[1] = y1;
            d1[0] = y2; d1[1] = y3;
        }
        y += 32;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 4 * 64 * (sgitg / 2);
        INK2_UNROLL
        for (short ik = 0; ik < 4; ++ik) {
            INK2_UNROLL
            for (short i = 0; i < 4; ++i) { simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false); }
            INK2_UNROLL
            for (short i = 0; i < 4; ++i) { simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false); }
            INK2_UNROLL
            for (short i = 0; i < 16; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 8 * 64;
        }
    }

    const ulong column = ulong(seg.outColumn) + r0 + 32u * (sgitg & 1);
    if (nr0 == 64 && nr1 == 64) {
        device float* C = output + ulong(t0 + 32u * (sgitg >> 1)) * p.outStride + column;
        INK2_UNROLL
        for (short i = 0; i < 16; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + ulong(8 * (i / 4)) * p.outStride, p.outStride, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* tmp = shmem;                       // [64 tokens][64 rows]
        threadgroup float* base = tmp + (32 * (sgitg >> 1)) * 64 + 32 * (sgitg & 1);
        INK2_UNROLL
        for (short i = 0; i < 16; ++i) {
            simdgroup_store(mc[i], base + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tiitg; e < 64u * 64u; e += 128u) {
            const uint tok = e / 64u, row = e % 64u;
            if (tok < uint(nr1) && row < uint(nr0)) {
                output[ulong(t0 + tok) * p.outStride + seg.outColumn + r0 + row] = tmp[e];
            }
        }
    }
}

// FP32 product with 64 x 64 tiles: F32 weight rows, FP32 inputs, FP32
// fragments. For the loop gates' small products (the vendor keeps them in
// FP32). Same segment table as the stacked product (segments must be F32);
// inputOffset and inputStride count floats.
[[kernel, max_total_threads_per_threadgroup(128)]]
void inkling2_f32_mm64(
    constant Inkling2ProjParams& p [[buffer(0)]],
    device const uchar* weights    [[buffer(1)]],
    device const float* input      [[buffer(2)]],
    device float*       output     [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float shmem[4096];                 // 16 KB: A (8 KB) + B (8 KB); edge store (16 KB)
    threadgroup float* sa = shmem;
    threadgroup float* sb = sa + 64 * 32;

    const uint tile = tgpig.y;
    uint si = 0;
    for (uint i = 1; i < p.segmentCount; ++i) {
        if (tile >= p.segments[i].tileStart) { si = i; }
    }
    const Inkling2ProjSegment seg = p.segments[si];
    const uint r0 = (tile - seg.tileStart) * 64u;
    const uint t0 = tgpig.x * 64u;
    if (r0 >= seg.rows || t0 >= p.tokens) { return; }
    const short nr0 = short(min(64u, seg.rows - r0));
    const short nr1 = short(min(64u, p.tokens - t0));

    // A: thread -> (row tiitg/2, 16 values of the block's half tiitg%2).
    const short ar  = short(tiitg / 2);
    const short il0 = short(tiitg % 2);
    const short lr0 = min(ar, short(nr0 - 1));
    // B: thread -> (token tiitg/2, 16 values at 16*(tiitg%2)).
    const short br  = short(tiitg / 2);
    const short lr1 = min(br, short(nr1 - 1));
    const short iy  = short(16 * (tiitg % 2));

    const uint blocks = p.inputs / 32u;
    const uint rowBytes = p.inputs * 4u;    // F32 rows only
    device const uchar* arow = weights + seg.weightOffset + ulong(r0 + uint(lr0)) * rowBytes;
    device const float* y = input + seg.inputOffset + ulong(t0 + uint(lr1)) * seg.inputStride + uint(iy);

    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[4];
    simdgroup_float8x8 mc[16];
    INK2_UNROLL
    for (short i = 0; i < 16; ++i) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }

    const short asy = ar / 8, alx = ar % 8;
    const short bsx = short(2 * (tiitg % 2)), bsy = br / 8, bly = br % 8;

    for (uint kb = 0; kb < blocks; ++kb) {
        float w[16];
        {
            device const float4* src = (device const float4*)(arow + (kb * 32u + 16u * uint(il0)) * 4u);
            INK2_UNROLL
            for (short i = 0; i < 4; ++i) {
                const float4 v = src[i];
                w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w;
            }
        }
        const float4 y0 = *((device const float4*)y);
        const float4 y1 = *((device const float4*)(y + 4));
        const float4 y2 = *((device const float4*)(y + 8));
        const float4 y3 = *((device const float4*)(y + 12));

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short ib = 8 * (2 * il0 + i / 8) + asy;     // [k block][row block]
            sa[64 * ib + 8 * (i % 8) + alx] = w[i];          // block stored [k][row]
        }
        {
            const short ib0 = 8 * bsx + bsy, ib1 = 8 * (bsx + 1) + bsy;   // [k block][token block]
            threadgroup float4* d0 = (threadgroup float4*)(sb + 64 * ib0 + 8 * bly);
            threadgroup float4* d1 = (threadgroup float4*)(sb + 64 * ib1 + 8 * bly);
            d0[0] = y0; d0[1] = y1;
            d1[0] = y2; d1[1] = y3;
        }
        y += 32;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const float* lsmb = sb + 4 * 64 * (sgitg / 2);
        INK2_UNROLL
        for (short ik = 0; ik < 4; ++ik) {
            INK2_UNROLL
            for (short i = 0; i < 4; ++i) { simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false); }
            INK2_UNROLL
            for (short i = 0; i < 4; ++i) { simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false); }
            INK2_UNROLL
            for (short i = 0; i < 16; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 8 * 64;
        }
    }

    const ulong column = ulong(seg.outColumn) + r0 + 32u * (sgitg & 1);
    if (nr0 == 64 && nr1 == 64) {
        device float* C = output + ulong(t0 + 32u * (sgitg >> 1)) * p.outStride + column;
        INK2_UNROLL
        for (short i = 0; i < 16; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + ulong(8 * (i / 4)) * p.outStride, p.outStride, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* tmp = shmem;                       // [64 tokens][64 rows]
        threadgroup float* base = tmp + (32 * (sgitg >> 1)) * 64 + 32 * (sgitg & 1);
        INK2_UNROLL
        for (short i = 0; i < 16; ++i) {
            simdgroup_store(mc[i], base + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tiitg; e < 64u * 64u; e += 128u) {
            const uint tok = e / 64u, row = e % 64u;
            if (tok < uint(nr1) && row < uint(nr0)) {
                output[ulong(t0 + tok) * p.outStride + seg.outColumn + r0 + row] = tmp[e];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Small-batch Q8_0 projection (decode and tiny chunks): each simdgroup owns
// two weight rows over the whole K; lane l reads eight quants of block
// l/4 + 8i at 8*(l%4), so 8 blocks (272 bytes per row) are in flight per
// step, and every activation load serves both rows. NT (1, 2, 4 or 8) bounds
// the token rows sharing each weight read. Same segment table as the stacked
// product with tileStart counting row pairs. Grid: x = ceil(pairs / 4)
// threadgroups of 4 simdgroups.
// ----------------------------------------------------------------------------

// Weight decode of one lane's eight values of block b, per format.
template <uint FMT>
static inline float ink2_mv_weights(device const uchar* row, uint b, uint part, thread float4& wa, thread float4& wb) {
    if (FMT == 2u) {
        device const float4* src = (device const float4*)(row + (b * 32u + 8u * part) * 4u);
        wa = src[0];
        wb = src[1];
        return 1.0f;
    } else if (FMT == 1u) {
        device const bfloat4* src = (device const bfloat4*)(row + (b * 32u + 8u * part) * 2u);
        wa = float4(src[0]);
        wb = float4(src[1]);
        return 1.0f;
    } else {
        device const uchar* blk = row + b * 34u;
        device const ushort* qs = (device const ushort*)(blk + 2u + 8u * part);
        const char2 c0 = as_type<char2>(qs[0]), c1 = as_type<char2>(qs[1]);
        const char2 c2 = as_type<char2>(qs[2]), c3 = as_type<char2>(qs[3]);
        wa = float4(c0.x, c0.y, c1.x, c1.y);
        wb = float4(c2.x, c2.y, c3.x, c3.y);
        return float(*((device const half*)blk));
    }
}

template <short NT, uint FMT>
static inline void ink2_mv_rows(constant Inkling2ProjParams& p, Inkling2ProjSegment seg, uint r, bool second,
                                device const uchar* weights, device const half* input, device float* output,
                                ushort lane) {
    const uint blocks = p.inputs / 32u;
    const uint rowBytes = FMT == 2u ? p.inputs * 4u : FMT == 1u ? p.inputs * 2u : blocks * 34u;
    device const uchar* row0 = weights + seg.weightOffset + ulong(r) * rowBytes;
    device const uchar* row1 = second ? row0 + rowBytes : row0;
    const uint tokens = min(p.tokens, uint(NT));
    float acc0[NT], acc1[NT];
    INK2_UNROLL
    for (short t = 0; t < NT; ++t) { acc0[t] = 0.0f; acc1[t] = 0.0f; }
    const uint part = lane % 4u;
    device const half* xbase = input + seg.inputOffset + 8u * part;
    for (uint b = lane / 4u; b < blocks; b += 8u) {
        // Activations once per block, shared by both rows (as the first form of this kernel).
        float4 xa[NT], xb[NT];
        INK2_UNROLL
        for (short t = 0; t < NT; ++t) {
            if (uint(t) < tokens) {
                device const half4* x = (device const half4*)(xbase + ulong(t) * seg.inputStride + b * 32u);
                xa[t] = float4(x[0]);
                xb[t] = float4(x[1]);
            } else {
                xa[t] = float4(0.0f);
                xb[t] = float4(0.0f);
            }
        }
        float4 wa, wb;
        const float d0 = ink2_mv_weights<FMT>(row0, b, part, wa, wb);
        INK2_UNROLL
        for (short t = 0; t < NT; ++t) { acc0[t] += d0 * (dot(wa, xa[t]) + dot(wb, xb[t])); }
        const float d1 = ink2_mv_weights<FMT>(row1, b, part, wa, wb);
        INK2_UNROLL
        for (short t = 0; t < NT; ++t) { acc1[t] += d1 * (dot(wa, xa[t]) + dot(wb, xb[t])); }
    }
    INK2_UNROLL
    for (short t = 0; t < NT; ++t) {
        const float s0 = simd_sum(acc0[t]);
        const float s1 = simd_sum(acc1[t]);
        if (lane == 0 && uint(t) < tokens) {
            device float* o = output + ulong(t) * p.outStride + seg.outColumn + r;
            o[0] = s0;
            if (second) { o[1] = s1; }
        }
    }
}

template <short NT>
[[kernel]] void inkling2_q8_stacked_mv(
    constant Inkling2ProjParams& p [[buffer(0)]],
    device const uchar* weights    [[buffer(1)]],
    device const half*  input      [[buffer(2)]],
    device float*       output     [[buffer(3)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]])
{
    const uint pair = tgpig * 4u + sgitg;
    uint si = 0;
    for (uint i = 1; i < p.segmentCount; ++i) {
        if (pair >= p.segments[i].tileStart) { si = i; }
    }
    const Inkling2ProjSegment seg = p.segments[si];
    const uint r = (pair - seg.tileStart) * 2u;
    if (r >= seg.rows) { return; }
    const bool second = r + 1u < seg.rows;
    // The format is uniform per simdgroup; each case runs its own specialised loop.
    switch (seg.format) {
    case 2u: ink2_mv_rows<NT, 2u>(p, seg, r, second, weights, input, output, lane); break;
    case 1u: ink2_mv_rows<NT, 1u>(p, seg, r, second, weights, input, output, lane); break;
    default: ink2_mv_rows<NT, 0u>(p, seg, r, second, weights, input, output, lane); break;
    }
}

template [[host_name("inkling2_q8_stacked_mv_t1")]] [[kernel]] decltype(inkling2_q8_stacked_mv<1>) inkling2_q8_stacked_mv<1>;
template [[host_name("inkling2_q8_stacked_mv_t2")]] [[kernel]] decltype(inkling2_q8_stacked_mv<2>) inkling2_q8_stacked_mv<2>;
template [[host_name("inkling2_q8_stacked_mv_t4")]] [[kernel]] decltype(inkling2_q8_stacked_mv<4>) inkling2_q8_stacked_mv<4>;
template [[host_name("inkling2_q8_stacked_mv_t8")]] [[kernel]] decltype(inkling2_q8_stacked_mv<8>) inkling2_q8_stacked_mv<8>;

// ----------------------------------------------------------------------------
// Relative-bias attention with a sink (prefill, any chunk)
//
// logit(h,i,j) = tau_i * (scale * q_i . k_j + r_i . P[pos_i - pos_j])   (bias 0 past relExtent)
// p = softmax over visible keys plus the sink logit s(h,i) (no value);
// out = (sum_j p_j v_j) * sigmoid(gate)   (gate optional)
//
// The sink initialises the online softmax (m = s, l = 1), so it needs no key.
// The bias is a 16-wide dot product per (query, key) inside the kernel
// against the rows of P staged per key tile: no per-query table, no mask
// buffer. A sliding window reads the ring at slot = position % capacity (the
// logical window is at most two physical spans; each key row is addressed on
// its own); global layers read linear storage (capacity 0) and apply tau.
//
// Threadgroup: BQT query rows of HG query heads, one simdgroup per 8 rows of a
// head (HG = 1: 32 rows, 4 simdgroups, grid x = ceil(rows / 32), y = query
// heads; HG = 3: 16 rows of the three heads of one KV head, 6 simdgroups,
// grid x = ceil(rows / 16), y = KV heads); keys in tiles of BK. Q, S, P and
// O live in simdgroup fragments; K, V and the P rows of the tile are staged
// in threadgroup memory.
// ----------------------------------------------------------------------------

struct Inkling2AttnParams {
    uint  queryCount;     // rows in this chunk
    uint  startPosition;  // position of chunk row 0
    uint  qStride;        // halves between query rows
    uint  relStride;      // floats between relative-state rows
    uint  sinkStride;     // floats between sink-logit rows
    uint  gateStride;     // floats between gate-logit rows; 0 = no gate
    uint  outStride;      // halves between output rows
    uint  kvStride;       // halves between KV cache slots
    uint  ringCapacity;   // 0 = linear storage (slot = position)
    uint  window;         // 0 = global attention
    uint  relExtent;      // distances 0 ..< relExtent carry a bias
    uint  logFloor;       // 0 = no log scaling
    float logAlpha;
    float scale;          // 1 / sqrt(head dim)
    uint  numQHeads;
    uint  numKVHeads;
};

constant constexpr short INK2_D = 128;
constant constexpr short INK2_R = 16;

static inline float ink2_tau(constant Inkling2AttnParams& p, uint position) {
    if (p.logFloor == 0u) { return 1.0f; }
    const float ratio = float(position + 1u) / float(p.logFloor);
    return ratio > 1.0f ? 1.0f + p.logAlpha * log(ratio) : 1.0f;
}

static inline float ink2_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// BK keys per tile; BQT query tokens and HG query heads per threadgroup
// (HG = 1: one head, 32 tokens, 4 simdgroups; HG = 3: the three heads of one
// KV head share each K/V tile, 16 tokens, 6 simdgroups). Simdgroup s owns 8
// tokens of head s / (BQT / 8).
template <short BK, short BQT, short HG>
[[kernel]] void inkling2_attention_prefill(
    constant Inkling2AttnParams& p [[buffer(0)]],
    device const half*  Q     [[buffer(1)]],
    device const half*  Kc    [[buffer(2)]],
    device const half*  Vc    [[buffer(3)]],
    device const float* rel   [[buffer(4)]],
    device const float* table [[buffer(5)]],
    device const float* sink  [[buffer(6)]],
    device const float* gate  [[buffer(7)]],
    device half*        out   [[buffer(8)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]])
{
    constexpr short BQ = BQT;
    constexpr short SUB = BQT / 8;           // simdgroups per head
    constexpr uint NT = 32u * uint(HG) * uint(SUB);
    constexpr short D = INK2_D;
    constexpr short DP = INK2_D + 8;     // padded row: 8x8 loads hit distinct banks
    constexpr short R = INK2_R;
    constexpr short NB = BK / 8;
    // K tile then V tile; one head's Q is staged here first (BQ * DP <= 2 * BK * DP).
    static_assert(BQ <= 2 * BK, "Q staging must fit the K/V tiles");
    threadgroup half kv[2 * BK * DP];
    threadgroup float pTable[(BQ + BK) * R];
    threadgroup half* kTile = kv;
    threadgroup half* vTile = kv + BK * DP;

    const uint myHead = uint(sgitg) / uint(SUB);
    const uint sub = uint(sgitg) % uint(SUB);
    const uint h = HG == 1 ? tgpig.y : tgpig.y * uint(HG) + myHead;
    const uint kvh = HG == 1 ? h / (p.numQHeads / p.numKVHeads) : tgpig.y;
    const uint q0 = tgpig.x * uint(BQ);
    if (q0 >= p.queryCount) { return; }
    const uint rowsHere = min(uint(BQ), p.queryCount - q0);
    const uint pos0 = p.startPosition + q0;

    simdgroup_half8x8 qf[D / 8];
    INK2_UNROLL
    for (short j = 0; j < HG; ++j) {
        const uint stagedHead = HG == 1 ? h : tgpig.y * uint(HG) + uint(j);
        for (uint e = tiitg; e < uint(BQ * D / 4); e += NT) {
            const uint row = e / uint(D / 4), c4 = e % uint(D / 4);
            half4 v = half4(0.0h);
            if (row < rowsHere) {
                v = ((device const half4*)(Q + ulong(q0 + row) * p.qStride + stagedHead * uint(D)))[c4];
            }
            ((threadgroup half4*)(kv + row * uint(DP)))[c4] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (myHead == uint(j)) {
            INK2_UNROLL
            for (short ds = 0; ds < D / 8; ++ds) {
                simdgroup_load(qf[ds], kv + (8 * sub) * DP + ds * 8, DP, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const short fm = ink2_frag_row(lane);
    const short fn = ink2_frag_col(lane);
    const uint myRow = q0 + 8u * sub + uint(fm);
    const bool rowValid = myRow < p.queryCount;
    const uint myPos = p.startPosition + myRow;
    const float tau = ink2_tau(p, myPos);
    const float qScale = p.scale * tau;
    float4 relReg[R / 4];
    float m = 0.0f;
    INK2_UNROLL
    for (short i = 0; i < R / 4; ++i) { relReg[i] = float4(0.0f); }
    if (rowValid) {
        device const float4* rrow = (device const float4*)(rel + ulong(myRow) * p.relStride + h * uint(R));
        INK2_UNROLL
        for (short i = 0; i < R / 4; ++i) { relReg[i] = rrow[i] * tau; }
        m = sink[ulong(myRow) * p.sinkStride + h];
    }
    float l = 1.0f;
    simdgroup_float8x8 of[D / 8];
    INK2_UNROLL
    for (short db = 0; db < D / 8; ++db) { of[db] = make_filled_simdgroup_matrix<float, 8>(0.0f); }

    const uint keyEnd = pos0 + rowsHere;
    const uint keyBegin = (p.window > 0u && pos0 + 1u > p.window) ? pos0 + 1u - p.window : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0 = keyBegin; k0 < keyEnd; k0 += uint(BK)) {
        for (uint e = tiitg; e < uint(BK * D / 4); e += NT) {
            const uint kr = e / uint(D / 4), c4 = e % uint(D / 4);
            const uint key = k0 + kr;
            half4 kk = half4(0.0h), vv = half4(0.0h);
            if (key < keyEnd) {
                const uint slot = p.ringCapacity != 0u ? key % p.ringCapacity : key;
                const ulong base = ulong(slot) * p.kvStride + kvh * uint(D);
                kk = ((device const half4*)(Kc + base))[c4];
                vv = ((device const half4*)(Vc + base))[c4];
            }
            ((threadgroup half4*)(kTile + kr * uint(DP)))[c4] = kk;
            ((threadgroup half4*)(vTile + kr * uint(DP)))[c4] = vv;
        }
        // Distances in this tile: pos0 + i - (k0 + j), i < BQ, j < BK.
        const int dBase = int(pos0) - int(k0) - (BK - 1);
        for (uint e = tiitg; e < uint((BQ + BK - 1) * R / 4); e += NT) {
            const uint dr = e / uint(R / 4), c4 = e % uint(R / 4);
            const int dist = dBase + int(dr);
            float4 v = float4(0.0f);
            if (dist >= 0 && uint(dist) < p.relExtent) {
                v = ((device const float4*)(table + uint(dist) * uint(R)))[c4];
            }
            ((threadgroup float4*)pTable)[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sf[NB];
        INK2_UNROLL
        for (short kb = 0; kb < NB; ++kb) {
            sf[kb] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            INK2_UNROLL
            for (short ds = 0; ds < D / 8; ++ds) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, kTile + kb * 8 * DP + ds * 8, DP, 0, true);
                simdgroup_multiply_accumulate(sf[kb], qf[ds], kt, sf[kb]);
            }
        }

        float tileMax = -INFINITY;
        INK2_UNROLL
        for (short kb = 0; kb < NB; ++kb) {
            INK2_UNROLL
            for (short c = 0; c < 2; ++c) {
                const uint key = k0 + uint(kb * 8 + fn + c);
                const bool visible = rowValid && key < keyEnd && key <= myPos
                    && (p.window == 0u || myPos - key < p.window);
                float s = -INFINITY;
                if (visible) {
                    s = sf[kb].thread_elements()[c] * qScale;
                    const uint dist = myPos - key;
                    if (dist < p.relExtent) {
                        threadgroup const float4* pr =
                            (threadgroup const float4*)(pTable + (int(dist) - dBase) * R);
                        float b = 0.0f;
                        INK2_UNROLL
                        for (short i = 0; i < R / 4; ++i) { b += dot(relReg[i], pr[i]); }
                        s += b;
                    }
                }
                sf[kb].thread_elements()[c] = s;
                tileMax = max(tileMax, s);
            }
        }
        tileMax = ink2_row_max(tileMax);
        const float mNew = max(m, tileMax);
        const float factor = exp(m - mNew);
        float tileSum = 0.0f;
        simdgroup_half8x8 pf[NB];
        INK2_UNROLL
        for (short kb = 0; kb < NB; ++kb) {
            const float e0 = exp(sf[kb].thread_elements()[0] - mNew);
            const float e1 = exp(sf[kb].thread_elements()[1] - mNew);
            tileSum += e0 + e1;
            pf[kb].thread_elements()[0] = half(e0);
            pf[kb].thread_elements()[1] = half(e1);
        }
        tileSum = ink2_row_sum(tileSum);
        l = l * factor + tileSum;
        m = mNew;
        if (factor != 1.0f) {
            INK2_UNROLL
            for (short db = 0; db < D / 8; ++db) {
                of[db].thread_elements()[0] *= factor;
                of[db].thread_elements()[1] *= factor;
            }
        }
        INK2_UNROLL
        for (short kb = 0; kb < NB; ++kb) {
            INK2_UNROLL
            for (short db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, vTile + kb * 8 * DP + db * 8, DP, 0, false);
                simdgroup_multiply_accumulate(of[db], pf[kb], vf, of[db]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (rowValid) {
        const float inv = 1.0f / l;
        device half* orow = out + ulong(myRow) * p.outStride + h * uint(D);
        device const float* grow = gate + ulong(myRow) * p.gateStride + h * uint(D);
        INK2_UNROLL
        for (short db = 0; db < D / 8; ++db) {
            const uint col = uint(db * 8 + fn);
            float a = of[db].thread_elements()[0] * inv;
            float b = of[db].thread_elements()[1] * inv;
            if (p.gateStride != 0u) {
                a *= ink2_sigmoid(grow[col]);
                b *= ink2_sigmoid(grow[col + 1]);
            }
            *((device half2*)(orow + col)) = half2(half(a), half(b));
        }
    }
}

template [[host_name("inkling2_attention_prefill_bk16")]] [[kernel]]
decltype(inkling2_attention_prefill<16, 32, 1>) inkling2_attention_prefill<16, 32, 1>;
template [[host_name("inkling2_attention_prefill_bk32")]] [[kernel]]
decltype(inkling2_attention_prefill<32, 32, 1>) inkling2_attention_prefill<32, 32, 1>;
template [[host_name("inkling2_attention_prefill_gqa16")]] [[kernel]]
decltype(inkling2_attention_prefill<16, 16, 3>) inkling2_attention_prefill<16, 16, 3>;

// ----------------------------------------------------------------------------
// QKV epilogue: everything between the stacked projection and attention.
//
// Reads the FP32 projection output `y` (one row per token, columns as given)
// and, per token:
//   q   = clip_t(qScale * y_q)                  per query head -> FP16 queries
//   r   = clip_8(relScale * y_r)                per query head -> FP32 relative states
//   k   = clip_t(sconv(y_k))                    per KV head    -> FP16 key cache slot
//   v   = rms_noweight(sconv(y_v))              per KV head    -> FP16 value cache slot
// sconv is the kernel-4 causal convolution over the token axis with the
// carried tail (the last 3 pre-convolution inputs, FP32 `[3][1024]`, row 2
// the newest): static `(sum_j w[c,j] x[t-3+j] + x_t) / sqrt(5)`, or for the
// global layers static plus per-token taps `a[t,h,lag]` on `x[t-lag]`,
// divided by sqrt(9). clip_t(v) = v * min(1, t / (|v| + eps)).
//
// The tails are read, not written: run `inkling2_conv_tail_update` on the
// same `y` columns afterwards. With one row, `advanceTails` updates them in
// place instead (each thread owns its channels), saving that dispatch. Grid: (tokens, 16) threadgroups of 128
// threads; simdgroup (y*4 + s) handles vector 0..23 q heads, 24..47 relative
// heads, 48..55 key heads, 56..63 value heads.
// ----------------------------------------------------------------------------

struct Inkling2QKVParams {
    uint  tokens;
    uint  startPosition;
    uint  yStride;        // floats between rows of `y`
    uint  qColumn;        // 24 x 128 query projections
    uint  relColumn;      // 24 x 16 relative-state projections
    uint  kColumn;        // 8 x 128 key projections (pre-convolution)
    uint  vColumn;        // 8 x 128 value projections (pre-convolution)
    uint  kTapColumn;     // 8 x 4 dynamic key taps (global layers)
    uint  vTapColumn;     // 8 x 4 dynamic value taps (global layers)
    uint  dynamicConv;    // 1 = global layer convolution
    uint  qOutStride;     // halves between query rows
    uint  relOutStride;   // floats between relative-state rows
    uint  kvStride;       // halves between cache slots
    uint  ringCapacity;   // 0 = linear storage
    uint  advanceTails;   // 1 = also advance the key/value tails in place (one row only)
    float qkClip;
    float relClip;
    float clipEps;
    float rmsEps;
};

static inline float ink2_conv_context(device const float* y, uint yStride, uint column,
                                      device const float* tail, int row, uint c) {
    return row >= 0 ? y[ulong(row) * yStride + column + c] : tail[uint(3 + row) * 1024u + c];
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void inkling2_qkv_epilogue(
    constant Inkling2QKVParams& p  [[buffer(0)]],
    device const float* y          [[buffer(1)]],
    device const float* qScale     [[buffer(2)]],
    device const float* relScale   [[buffer(3)]],
    device const float* kConv      [[buffer(4)]],
    device const float* vConv      [[buffer(5)]],
    device float*       kTail      [[buffer(6)]],
    device float*       vTail      [[buffer(7)]],
    device half*        qOut       [[buffer(8)]],
    device float*       relOut     [[buffer(9)]],
    device half*        kCache     [[buffer(10)]],
    device half*        vCache     [[buffer(11)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]])
{
    const uint t = tgpig.x;
    if (t >= p.tokens) { return; }
    const uint vec = tgpig.y * 4u + sgitg;
    device const float* yrow = y + ulong(t) * p.yStride;

    if (vec < 24u) {
        const uint h = vec;
        const float4 x = ((device const float4*)(yrow + p.qColumn + h * 128u))[lane]
                       * ((device const float4*)(qScale + h * 128u))[lane];
        const float norm = sqrt(simd_sum(dot(x, x)));
        const float factor = min(1.0f, p.qkClip / (norm + p.clipEps));
        ((device half4*)(qOut + ulong(t) * p.qOutStride + h * 128u))[lane] = half4(x * factor);
        return;
    }
    if (vec < 48u) {
        const uint h = vec - 24u;
        const float x = lane < 16u ? yrow[p.relColumn + h * 16u + lane] * relScale[h * 16u + lane] : 0.0f;
        const float norm = sqrt(simd_sum(x * x));
        const float factor = min(1.0f, p.relClip / (norm + p.clipEps));
        if (lane < 16u) { relOut[ulong(t) * p.relOutStride + h * 16u + lane] = x * factor; }
        return;
    }
    const bool isKey = vec < 56u;
    const uint h = isKey ? vec - 48u : vec - 56u;
    const uint column = isKey ? p.kColumn : p.vColumn;
    device const float* conv = isKey ? kConv : vConv;
    device float* tail = isKey ? kTail : vTail;
    float taps[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (p.dynamicConv != 0u) {
        const uint tapColumn = isKey ? p.kTapColumn : p.vTapColumn;
        for (short lag = 0; lag < 4; ++lag) { taps[lag] = yrow[tapColumn + h * 4u + uint(lag)]; }
    }
    const float divisor = p.dynamicConv != 0u ? 3.0f : sqrt(5.0f);
    float4 outv;
    for (short i = 0; i < 4; ++i) {
        const uint c = h * 128u + 4u * lane + uint(i);
        const float4 w = ((device const float4*)conv)[c];        // taps oldest first
        const float x3 = ink2_conv_context(y, p.yStride, column, tail, int(t) - 3, c);
        const float x2 = ink2_conv_context(y, p.yStride, column, tail, int(t) - 2, c);
        const float x1 = ink2_conv_context(y, p.yStride, column, tail, int(t) - 1, c);
        const float x0 = yrow[column + c];
        float acc = w.x * x3 + w.y * x2 + w.z * x1 + w.w * x0;
        acc += taps[0] * x0 + taps[1] * x1 + taps[2] * x2 + taps[3] * x3;
        outv[i] = (acc + x0) / divisor;
        if (p.advanceTails != 0u) {
            // One row: this thread alone reads and writes channel c of the tail.
            tail[c] = x2;
            tail[1024u + c] = x1;
            tail[2048u + c] = x0;
        }
    }
    const uint slot = p.ringCapacity != 0u ? (p.startPosition + t) % p.ringCapacity : p.startPosition + t;
    device half4* dst = (device half4*)((isKey ? kCache : vCache) + ulong(slot) * p.kvStride + h * 128u) + lane;
    if (isKey) {
        const float norm = sqrt(simd_sum(dot(outv, outv)));
        *dst = half4(outv * min(1.0f, p.qkClip / (norm + p.clipEps)));
    } else {
        const float meanSquare = simd_sum(dot(outv, outv)) / 128.0f;
        *dst = half4(outv * rsqrt(meanSquare + p.rmsEps));
    }
}

// ----------------------------------------------------------------------------
// Convolution tails. After a chunk of `tokens` rows, a tail holds the last 3
// pre-convolution inputs of `history ++ x` (FP32 `[3][channels]`, row 2 the
// newest). Up to two runs per dispatch (keys and values read the same
// projection output). Each thread owns one channel and reads before it
// writes, so updating in place is safe for any chunk length.
// ----------------------------------------------------------------------------

struct Inkling2TailRun {
    uint xColumn;     // first column of the run in `x`
    uint tailOffset;  // floats into `tails` of the run's [3][channels] block
    uint channels;
    uint reserved;
};

struct Inkling2TailParams {
    uint tokens;
    uint xStride;
    uint runCount;
    uint reserved;
    Inkling2TailRun runs[2];
};

kernel void inkling2_conv_tail_update(
    constant Inkling2TailParams& p [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device float*       tails      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint c = gid;
    uint r = 0;
    if (c >= p.runs[0].channels) {
        if (p.runCount < 2u) { return; }
        c -= p.runs[0].channels;
        r = 1;
        if (c >= p.runs[1].channels) { return; }
    }
    const Inkling2TailRun run = p.runs[r];
    device float* tail = tails + run.tailOffset;
    float v[3];
    for (uint n = 0; n < 3u; ++n) {
        const uint m = p.tokens + n;     // row in history ++ x
        v[n] = m < 3u ? tail[m * run.channels + c] : x[ulong(m - 3u) * p.xStride + run.xColumn + c];
    }
    for (uint n = 0; n < 3u; ++n) { tail[n * run.channels + c] = v[n]; }
}

// ----------------------------------------------------------------------------
// Static short convolution of the attention and MoE outputs, FP32:
//   out[t,c] = (sum_j w[c,j] x[t-3+j] + x[t,c]) / sqrt(5)  (+ residual[t,c])
// with the carried tail as rows -3..-1. `out` may alias `residual`, never `x`.
// With one row, `advanceTail` also advances the tail in place.
// Grid: (channels / 4, tokens), one thread per four channels.
// ----------------------------------------------------------------------------

struct Inkling2ConvParams {
    uint tokens;
    uint channels;
    uint xStride;
    uint outStride;
    uint residualStride;  // 0 = no residual
    uint advanceTail;     // 1 = also advance the tail in place (one row only)
    uint reserved1;
    uint reserved2;
};

kernel void inkling2_sconv_f32(
    constant Inkling2ConvParams& p [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device const float* weight     [[buffer(2)]],
    device float*       tail       [[buffer(3)]],
    device const float* residual   [[buffer(4)]],
    device float*       out        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c4 = gid.x, t = gid.y;
    if (c4 * 4u >= p.channels || t >= p.tokens) { return; }
    float4 ctx[4];
    for (short j = 0; j < 4; ++j) {
        const int row = int(t) - 3 + j;
        ctx[j] = row >= 0 ? ((device const float4*)(x + ulong(row) * p.xStride))[c4]
                          : ((device const float4*)(tail + uint(3 + row) * p.channels))[c4];
    }
    float4 result;
    for (short i = 0; i < 4; ++i) {
        const float4 w = ((device const float4*)weight)[c4 * 4u + uint(i)];
        result[i] = (w.x * ctx[0][i] + w.y * ctx[1][i] + w.z * ctx[2][i] + w.w * ctx[3][i] + ctx[3][i])
                  * rsqrt(5.0f);
    }
    if (p.residualStride != 0u) {
        result += ((device const float4*)(residual + ulong(t) * p.residualStride))[c4];
    }
    ((device float4*)(out + ulong(t) * p.outStride))[c4] = result;
    if (p.advanceTail != 0u) {
        // One row: rows -2, -1 and x become the tail; this thread owns its four channels.
        ((device float4*)(tail))[c4] = ctx[1];
        ((device float4*)(tail + p.channels))[c4] = ctx[2];
        ((device float4*)(tail + 2u * p.channels))[c4] = ctx[3];
    }
}

// ----------------------------------------------------------------------------
// Weighted RMS norm from the FP32 stream to the FP16 projection input:
//   out[t,:] = half(x[t,:] * rsqrt(mean(x^2) + eps) * weight)
// FP32 weights (the GGUF norm tensors). One threadgroup of 256 per row.
// ----------------------------------------------------------------------------

struct Inkling2NormParams {
    uint  width;
    uint  xStride;
    uint  outStride;
    float eps;
};

// One row's norm by one threadgroup of 256; shared by both norm kernels, so
// they write the same bytes.
static inline void ink2_rms_row(uint width, float eps, device const float* xr, device const float* weight,
                                device half* orow, ushort tid, ushort lane, ushort sgitg,
                                threadgroup float* partial) {
    float acc = 0.0f;
    for (uint i = tid; i < width; i += 256u) { acc = fma(xr[i], xr[i], acc); }
    acc = simd_sum(acc);
    if (lane == 0) { partial[sgitg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (short s = 0; s < 8; ++s) { total += partial[s]; }
    const float inv = rsqrt(total / float(width) + eps);
    for (uint i = tid; i < width; i += 256u) { orow[i] = half(xr[i] * inv * weight[i]); }
}

// `ink2_rms_row` that also writes the normed row in FP32; its FP16 output is the same bytes.
static inline void ink2_rms_row_dual(uint width, float eps, device const float* xr, device const float* weight,
                                     device half* orow, device float* frow, ushort tid, ushort lane, ushort sgitg,
                                     threadgroup float* partial) {
    float acc = 0.0f;
    for (uint i = tid; i < width; i += 256u) { acc = fma(xr[i], xr[i], acc); }
    acc = simd_sum(acc);
    if (lane == 0) { partial[sgitg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (short s = 0; s < 8; ++s) { total += partial[s]; }
    const float inv = rsqrt(total / float(width) + eps);
    for (uint i = tid; i < width; i += 256u) {
        const float y = xr[i] * inv * weight[i];
        frow[i] = y;
        orow[i] = half(y);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling2_rms_f32_to_f16(
    constant Inkling2NormParams& p [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device const float* weight     [[buffer(2)]],
    device half*        out        [[buffer(3)]],
    uint   row   [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[8];
    ink2_rms_row(p.width, p.eps, x + ulong(row) * p.xStride, weight, out + ulong(row) * p.outStride,
                 tid, lane, sgitg, partial);
}

// Up to three inputs (an attention block's q, k and v inputs), each with its own
// weight and output, normed in one dispatch: grid (tokens, inputs). The same
// bytes as one `inkling2_rms_f32_to_f16` dispatch per input; at one token the
// threadgroups run side by side, so the norm's latency is paid once.
[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling2_rms_f32_to_f16_set(
    constant Inkling2NormParams& p    [[buffer(0)]],
    device const float* x0            [[buffer(1)]],
    device const float* x1            [[buffer(2)]],
    device const float* x2            [[buffer(3)]],
    device const float* w0            [[buffer(4)]],
    device const float* w1            [[buffer(5)]],
    device const float* w2            [[buffer(6)]],
    device half*        o0            [[buffer(7)]],
    device half*        o1            [[buffer(8)]],
    device half*        o2            [[buffer(9)]],
    uint2  tg    [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[8];
    const uint token = tg.x, k = tg.y;
    device const float* x = k == 0u ? x0 : (k == 1u ? x1 : x2);
    device const float* w = k == 0u ? w0 : (k == 1u ? w1 : w2);
    device half*        o = k == 0u ? o0 : (k == 1u ? o1 : o2);
    ink2_rms_row(p.width, p.eps, x + ulong(token) * p.xStride, w, o + ulong(token) * p.outStride,
                 tid, lane, sgitg, partial);
}

// ----------------------------------------------------------------------------
// Decode attention (one query row), split over key blocks.
//
// Pass 1, grid (blocks, KV heads): a threadgroup of 8 simdgroups takes the
// three query heads of one KV head over one block of keys; each simdgroup walks
// every eighth key with the lanes on head dims (4 each) and folds the 16-wide
// bias into the same simd_sum as q.k. Partial (max, sum, out[128]) per (query
// head, block). Pass 2, grid (query heads): merges the blocks with the sink as
// the starting state (block maxima and sums by simd reductions) and applies the
// sigmoid gate. 8 simdgroups and 128-key blocks, with that merge, measured 16%
// (window, 1,024 keys) and 20% (global, 4,096 keys) faster than 4 and 64.
// ----------------------------------------------------------------------------

struct Inkling2DecodeParams {
    uint  position;       // the query row's position
    uint  keyBegin;       // first visible key
    uint  keyCount;       // keys keyBegin ..< keyBegin + keyCount
    uint  blockKeys;      // keys per pass-1 block
    uint  blocks;
    uint  kvStride;       // halves between cache slots
    uint  ringCapacity;   // 0 = linear storage
    uint  relExtent;
    float tau;            // log scaling of this position (1 for window layers)
    float scale;
    uint  numQHeads;
    uint  numKVHeads;
    uint  hasGate;
    uint  reserved0;
    uint  reserved1;
    uint  reserved2;
};

constant constexpr uint INK2_PARTIAL = 2u + 128u;   // max, sum, 128 outputs

[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling2_attention_decode_partial(
    constant Inkling2DecodeParams& p [[buffer(0)]],
    device const half*  Q        [[buffer(1)]],
    device const half*  Kc       [[buffer(2)]],
    device const half*  Vc       [[buffer(3)]],
    device const float* rel      [[buffer(4)]],
    device const float* table    [[buffer(5)]],
    device float*       partials [[buffer(6)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]])
{
    threadgroup float stage[8][3][INK2_PARTIAL];
    const uint block = tgpig.x, g = tgpig.y;
    const uint group = p.numQHeads / p.numKVHeads;     // 3
    const float qs = p.scale * p.tau;
    float4 q4[3];
    float relv[3];
    for (uint j = 0; j < 3u; ++j) {
        const uint h = g * group + j;
        q4[j] = float4(((device const half4*)(Q + h * 128u))[lane]) * qs;
        relv[j] = lane < 16u ? rel[h * 16u + lane] * p.tau : 0.0f;
    }
    float m[3] = {-INFINITY, -INFINITY, -INFINITY};
    float l[3] = {0.0f, 0.0f, 0.0f};
    float4 o[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};

    const uint first = block * p.blockKeys;
    const uint last = min(first + p.blockKeys, p.keyCount);
    for (uint n = first + sgitg; n < last; n += 8u) {
        const uint key = p.keyBegin + n;
        const uint slot = p.ringCapacity != 0u ? key % p.ringCapacity : key;
        const ulong base = ulong(slot) * p.kvStride + g * 128u;
        const float4 k4 = float4(((device const half4*)(Kc + base))[lane]);
        const float4 v4 = float4(((device const half4*)(Vc + base))[lane]);
        const uint dist = p.position - key;
        const float pd = (lane < 16u && dist < p.relExtent) ? table[dist * 16u + lane] : 0.0f;
        const float4 parts = float4(dot(q4[0], k4) + relv[0] * pd,
                                    dot(q4[1], k4) + relv[1] * pd,
                                    dot(q4[2], k4) + relv[2] * pd, 0.0f);
        const float4 logits = simd_sum(parts);
        for (uint j = 0; j < 3u; ++j) {
            const float mNew = max(m[j], logits[j]);
            const float f = exp(m[j] - mNew);
            const float e = exp(logits[j] - mNew);
            l[j] = l[j] * f + e;
            o[j] = o[j] * f + e * v4;
            m[j] = mNew;
        }
    }
    for (uint j = 0; j < 3u; ++j) {
        if (lane == 0) { stage[sgitg][j][0] = m[j]; stage[sgitg][j][1] = l[j]; }
        for (short i = 0; i < 4; ++i) { stage[sgitg][j][2u + 4u * lane + uint(i)] = o[j][i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The first 128 threads: thread = output dim; loop the three heads.
    if (tiitg >= 128u) { return; }
    const uint d = tiitg;
    for (uint j = 0; j < 3u; ++j) {
        float M = -INFINITY;
        for (short s = 0; s < 8; ++s) { M = max(M, stage[s][j][0]); }
        float L = 0.0f, O = 0.0f;
        for (short s = 0; s < 8; ++s) {
            const float ms = stage[s][j][0];
            const float w = ms == -INFINITY ? 0.0f : exp(ms - M);
            L += stage[s][j][1] * w;
            O += stage[s][j][2u + d] * w;
        }
        device float* dst = partials + (ulong(g * group + j) * p.blocks + block) * INK2_PARTIAL;
        if (d == 0) { dst[0] = M; dst[1] = L; }
        dst[2u + d] = O;
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void inkling2_attention_decode_merge(
    constant Inkling2DecodeParams& p [[buffer(0)]],
    device const float* partials [[buffer(1)]],
    device const float* sink     [[buffer(2)]],
    device const float* gate     [[buffer(3)]],
    device half*        out      [[buffer(4)]],
    uint   h     [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float shared[2];
    device const float* base = partials + ulong(h) * p.blocks * INK2_PARTIAL;
    const float s = sink[h];
    // The block maxima and sums: one simdgroup, a lane per block, simd reductions.
    if (sgitg == 0) {
        float M = s;
        for (uint b = lane; b < p.blocks; b += 32u) { M = max(M, base[b * INK2_PARTIAL]); }
        M = simd_max(M);
        float L = 0.0f;
        for (uint b = lane; b < p.blocks; b += 32u) {
            const float mb = base[b * INK2_PARTIAL];
            L += mb == -INFINITY ? 0.0f : base[b * INK2_PARTIAL + 1u] * exp(mb - M);
        }
        L = simd_sum(L) + exp(s - M);
        if (lane == 0) { shared[0] = M; shared[1] = L; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float M = shared[0], L = shared[1];
    float O = 0.0f;
    for (uint b = 0; b < p.blocks; ++b) {
        const float mb = base[b * INK2_PARTIAL];
        O += mb == -INFINITY ? 0.0f : base[b * INK2_PARTIAL + 2u + tid] * exp(mb - M);
    }
    float result = O / L;
    if (p.hasGate != 0u) { result *= ink2_sigmoid(gate[h * 128u + tid]); }
    out[h * 128u + tid] = half(result);
}

// ============================================================================
// Loop core: block attention residuals, cross-pass message retrieval and the
// message gate.
// ============================================================================

// ----------------------------------------------------------------------------
// Depth mix: the block attention residuals and the cross-pass message
// retrieval. Per token, each state s_n (FP32 [rows][stride], up to 6) is
// normalised with a weightless RMS norm; each learned query q_j (FP32
// [queries][2304], up to 4) attends over them with scale 1/sqrt(2304);
// out_j = sum_n softmax_n(q_j . s^_n / sqrt(D)) s^_n. One threadgroup of 256
// per token; outputs FP32 [rows][outStride] with query j at column j * width.
// ----------------------------------------------------------------------------

struct Inkling2MixParams {
    uint  rows;
    uint  width;          // 2304
    uint  stateCount;     // 1 ... 6 (equals the function constant)
    uint  queryCount;     // 1 ... 4 (equals the function constant)
    uint  stateStride[6]; // floats between rows of each state
    uint  outStride;      // floats between output rows
    float eps;
};

// State and query counts are function constants, so every loop over them
// unrolls and the per-thread sums stay in registers.
constant uint FC_MIX_STATES  [[function_constant(200)]];
constant uint FC_MIX_QUERIES [[function_constant(201)]];
constant uint kMixStates  = is_function_constant_defined(FC_MIX_STATES) ? FC_MIX_STATES : 1u;
constant uint kMixQueries = is_function_constant_defined(FC_MIX_QUERIES) ? FC_MIX_QUERIES : 1u;

// The depth mix of one token row, shared by the mix kernel and the mix + norm kernel.
static inline void ink2_depth_mix_row(constant Inkling2MixParams& p,
                                      device const float* s0, device const float* s1, device const float* s2,
                                      device const float* s3, device const float* s4, device const float* s5,
                                      device const float* queries, device float* out, uint row,
                                      ushort tid, ushort lane, ushort sgitg,
                                      threadgroup float (*partial)[30], threadgroup float (*weights)[6],
                                      threadgroup float* inv)
{
    constexpr uint NS = 6, NQ = 4, NA = NS + NQ * NS;   // NA = 30: the threadgroup arrays' widths
    device const float* rowp[NS] = {
        s0 + ulong(row) * p.stateStride[0], s1 + ulong(row) * p.stateStride[1], s2 + ulong(row) * p.stateStride[2],
        s3 + ulong(row) * p.stateStride[3], s4 + ulong(row) * p.stateStride[4], s5 + ulong(row) * p.stateStride[5]};
    float acc[NA];
    INK2_UNROLL
    for (uint i = 0; i < NA; ++i) { acc[i] = 0.0f; }
    for (uint i = tid; i < p.width; i += 256u) {
        float sv[NS];
        INK2_UNROLL
        for (uint n = 0; n < NS; ++n) {
            sv[n] = 0.0f;
            if (n < kMixStates) { sv[n] = rowp[n][i]; acc[n] = fma(sv[n], sv[n], acc[n]); }
        }
        INK2_UNROLL
        for (uint j = 0; j < NQ; ++j) {
            if (j < kMixQueries) {
                const float q = queries[j * p.width + i];
                INK2_UNROLL
                for (uint n = 0; n < NS; ++n) {
                    if (n < kMixStates) { acc[NS + j * NS + n] = fma(q, sv[n], acc[NS + j * NS + n]); }
                }
            }
        }
    }
    INK2_UNROLL
    for (uint i = 0; i < NA; ++i) {
        const bool used = i < NS ? i < kMixStates : ((i - NS) / NS < kMixQueries && (i - NS) % NS < kMixStates);
        if (used) {
            const float total = simd_sum(acc[i]);
            if (lane == 0) { partial[sgitg][i] = total; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sums[NA];
        INK2_UNROLL
        for (uint i = 0; i < NA; ++i) {
            const bool used = i < NS ? i < kMixStates : ((i - NS) / NS < kMixQueries && (i - NS) % NS < kMixStates);
            float t = 0.0f;
            if (used) { for (short s = 0; s < 8; ++s) { t += partial[s][i]; } }
            sums[i] = t;
        }
        const float scale = rsqrt(float(p.width));
        INK2_UNROLL
        for (uint n = 0; n < NS; ++n) { if (n < kMixStates) { inv[n] = rsqrt(sums[n] / float(p.width) + p.eps); } }
        INK2_UNROLL
        for (uint j = 0; j < NQ; ++j) {
            if (j >= kMixQueries) { continue; }
            float sc[NS];
            float m = -INFINITY;
            INK2_UNROLL
            for (uint n = 0; n < NS; ++n) {
                sc[n] = 0.0f;
                if (n < kMixStates) { sc[n] = sums[NS + j * NS + n] * inv[n] * scale; m = max(m, sc[n]); }
            }
            float total = 0.0f;
            INK2_UNROLL
            for (uint n = 0; n < NS; ++n) { if (n < kMixStates) { sc[n] = exp(sc[n] - m); total += sc[n]; } }
            INK2_UNROLL
            for (uint n = 0; n < NS; ++n) { if (n < kMixStates) { weights[j][n] = sc[n] / total; } }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[NQ][NS];
    INK2_UNROLL
    for (uint j = 0; j < NQ; ++j) {
        INK2_UNROLL
        for (uint n = 0; n < NS; ++n) { w[j][n] = (j < kMixQueries && n < kMixStates) ? weights[j][n] * inv[n] : 0.0f; }
    }
    device float* orow = out + ulong(row) * p.outStride;
    for (uint i = tid; i < p.width; i += 256u) {
        float sv[NS];
        INK2_UNROLL
        for (uint n = 0; n < NS; ++n) { sv[n] = n < kMixStates ? rowp[n][i] : 0.0f; }
        INK2_UNROLL
        for (uint j = 0; j < NQ; ++j) {
            if (j < kMixQueries) {
                float o = 0.0f;
                INK2_UNROLL
                for (uint n = 0; n < NS; ++n) { if (n < kMixStates) { o = fma(w[j][n], sv[n], o); } }
                orow[j * p.width + i] = o;
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling2_depth_mix(
    constant Inkling2MixParams& p [[buffer(0)]],
    device const float* s0 [[buffer(1)]],
    device const float* s1 [[buffer(2)]],
    device const float* s2 [[buffer(3)]],
    device const float* s3 [[buffer(4)]],
    device const float* s4 [[buffer(5)]],
    device const float* s5 [[buffer(6)]],
    device const float* queries [[buffer(7)]],
    device float*       out     [[buffer(8)]],
    uint   row   [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float mixPartial[8][30];
    threadgroup float mixWeights[4][6];
    threadgroup float mixInv[6];
    ink2_depth_mix_row(p, s0, s1, s2, s3, s4, s5, queries, out, row, tid, lane, sgitg, mixPartial, mixWeights, mixInv);
}

// ----------------------------------------------------------------------------
// The depth mix fused with the weighted RMS norm of each of its outputs, for
// the visits where nothing sits between them (first-pass attention inputs, and
// the MoE input outside the loop passes): the mix as `inkling2_depth_mix` into
// `out` (FP32 scratch), then each query's row normed as `inkling2_rms_f32_to_f16`
// does (each thread reads back the elements it wrote, in that kernel's order),
// so `normed` is byte-identical to the two kernels. Up to three queries, each
// with its own norm weight and FP16 output; `normed32` (optional) also takes
// the normed rows in FP32 (the MoE router's input). One threadgroup per token.
// ----------------------------------------------------------------------------

struct Inkling2MixNormParams {
    uint  normedStride;   // halves between token rows of each FP16 output
    uint  normed32Stride; // floats between token rows of each FP32 output (0 = none)
    float normEps;
    uint  reserved;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling2_depth_mix_norm(
    constant Inkling2MixParams& p        [[buffer(0)]],
    device const float* s0               [[buffer(1)]],
    device const float* s1               [[buffer(2)]],
    device const float* s2               [[buffer(3)]],
    device const float* s3               [[buffer(4)]],
    device const float* s4               [[buffer(5)]],
    device const float* s5               [[buffer(6)]],
    device const float* queries          [[buffer(7)]],
    device float*       out              [[buffer(8)]],
    constant Inkling2MixNormParams& n    [[buffer(9)]],
    device const float* w0               [[buffer(10)]],
    device const float* w1               [[buffer(11)]],
    device const float* w2               [[buffer(12)]],
    device half*        h0               [[buffer(13)]],
    device half*        h1               [[buffer(14)]],
    device half*        h2               [[buffer(15)]],
    device float*       f0               [[buffer(16)]],
    device float*       f1               [[buffer(17)]],
    device float*       f2               [[buffer(18)]],
    uint   row   [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float mixPartial[8][30];
    threadgroup float mixWeights[4][6];
    threadgroup float mixInv[6];
    ink2_depth_mix_row(p, s0, s1, s2, s3, s4, s5, queries, out, row, tid, lane, sgitg, mixPartial, mixWeights, mixInv);
    threadgroup float partial[8];
    for (uint j = 0; j < kMixQueries && j < 3u; ++j) {
        device const float* xr = out + ulong(row) * p.outStride + j * p.width;
        device const float* w = j == 0u ? w0 : (j == 1u ? w1 : w2);
        device half* hrow = (j == 0u ? h0 : (j == 1u ? h1 : h2)) + ulong(row) * n.normedStride;
        if (n.normed32Stride == 0u) {
            ink2_rms_row(p.width, n.normEps, xr, w, hrow, tid, lane, sgitg, partial);
        } else {
            device float* frow = (j == 0u ? f0 : (j == 1u ? f1 : f2)) + ulong(row) * n.normed32Stride;
            ink2_rms_row_dual(p.width, n.normEps, xr, w, hrow, frow, tid, lane, sgitg, partial);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // `partial` is reused by the next query
    }
}

// ----------------------------------------------------------------------------
// Message gate, around two FP32 products (inkling2_f32_mm64, F32 weights):
//   gate inputs:  h[t] = [rms(x_c); rms(m_c)] in FP32, per controller c (width 2 * 2304)
//   z = W_down h                 (F32 down [64][4608] per controller)
//   f, g = W_forget z, W_gain z  (F32 [2304][64])
//   out = (1 - tanh f) x + tanh g m
// ----------------------------------------------------------------------------

struct Inkling2GateInputParams {
    uint  rows;
    uint  width;           // 2304
    uint  xStride;
    uint  mStride;
    uint  outStride;       // floats between rows of h (>= 2 * width)
    float eps;
};

// h[t] = [rms(x); rms(m)] in FP32, the input of the controller's down product.
[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling2_gate_inputs(
    constant Inkling2GateInputParams& p [[buffer(0)]],
    device const float* x  [[buffer(1)]],
    device const float* m  [[buffer(2)]],
    device float*       h  [[buffer(3)]],
    uint   row   [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[8][2];
    device const float* xr = x + ulong(row) * p.xStride;
    device const float* mr = m + ulong(row) * p.mStride;
    float ax = 0.0f, am = 0.0f;
    for (uint i = tid; i < p.width; i += 256u) { ax = fma(xr[i], xr[i], ax); am = fma(mr[i], mr[i], am); }
    ax = simd_sum(ax);
    am = simd_sum(am);
    if (lane == 0) { partial[sgitg][0] = ax; partial[sgitg][1] = am; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tx = 0.0f, tm = 0.0f;
    for (short s = 0; s < 8; ++s) { tx += partial[s][0]; tm += partial[s][1]; }
    const float ix = rsqrt(tx / float(p.width) + p.eps), im = rsqrt(tm / float(p.width) + p.eps);
    device float* hr = h + ulong(row) * p.outStride;
    for (uint i = tid; i < p.width; i += 256u) {
        hr[i] = xr[i] * ix;
        hr[p.width + i] = mr[i] * im;
    }
}

struct Inkling2GateCombineParams {
    uint rows;
    uint width;
    uint xStride;
    uint mStride;
    uint fgStride;     // floats between rows of the forget/gain product
    uint forgetColumn;
    uint gainColumn;
    uint outStride;
};

// out = (1 - tanh f) x + tanh g m (tanh clamped: fast-math tanh misbehaves
// for large arguments on some GPUs). `out` may alias `x`.
kernel void inkling2_gate_combine(
    constant Inkling2GateCombineParams& p [[buffer(0)]],
    device const float* x  [[buffer(1)]],
    device const float* m  [[buffer(2)]],
    device const float* fg [[buffer(3)]],
    device float*       out [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y;
    if (i >= p.width || t >= p.rows) { return; }
    const float xv = x[ulong(t) * p.xStride + i], mv = m[ulong(t) * p.mStride + i];
    const float f = tanh(clamp(fg[ulong(t) * p.fgStride + p.forgetColumn + i], -20.0f, 20.0f));
    const float g = tanh(clamp(fg[ulong(t) * p.fgStride + p.gainColumn + i], -20.0f, 20.0f));
    out[ulong(t) * p.outStride + i] = xv - f * xv + g * mv;
}

// ----------------------------------------------------------------------------
// Message gate for one row (decode): up to four controllers in two dispatches.
// All tensors live in one weight buffer and one activation buffer at the
// offsets of `Inkling2GateDecodeParams`.
//   down:    grid (controllers, 64 / 8); a threadgroup reduces rms(x), rms(m)
//            of its controller, then each of its 8 simdgroups forms one z row:
//            z = inv_x * (down[:, :2304] . x) + inv_m * (down[:, 2304:] . m)
//   combine: grid (ceil(width / 256), controllers); a thread forms f = F z and
//            g = G z for its element in FP32 and writes (1 - tanh f) x + tanh g m.
// ----------------------------------------------------------------------------

struct Inkling2GateController {
    ulong down;       // bytes into `weights`: F32 [rank][2 * width]
    ulong forget;     // bytes: F32 [width][rank]
    ulong gain;       // bytes: F32 [width][rank]
    uint  x;          // floats into `acts`
    uint  m;          // floats into `acts`
    uint  out;        // floats into `acts` (may equal x)
    uint  reserved;
};

struct Inkling2GateDecodeParams {
    uint  width;
    uint  rank;
    uint  controllers;
    float eps;
    Inkling2GateController c[4];
};

// Down product: each z row's dot product is split over 4 simdgroups (x first
// half, x second half, m first half, m second half), two rows per threadgroup of
// 8; every threadgroup also reduces the RMS of x and m. Grid: (controllers,
// rank / 2), 256 threads. (One simdgroup per row measured 46 us per three
// controllers with the old combine; this form and the combine below, 28 us.)
[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling2_gate_down_decode(
    constant Inkling2GateDecodeParams& p [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* acts    [[buffer(2)]],
    device float*       z       [[buffer(3)]],     // [controllers][rank]
    uint2  tg    [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[8][2];
    threadgroup float dots[8];
    const Inkling2GateController c = p.c[tg.x];
    device const float* x = acts + c.x;
    device const float* m = acts + c.m;
    const uint row = tg.y * 2u + sgitg / 4u, quarter = sgitg % 4u;
    const uint halfWidth = p.width / 2u;
    float d = 0.0f;
    if (row < p.rank) {
        device const float* w = (device const float*)(weights + c.down) + ulong(row) * 2u * p.width
                                + (quarter / 2u) * p.width + (quarter % 2u) * halfWidth;
        device const float* v = (quarter < 2u ? x : m) + (quarter % 2u) * halfWidth;
        for (uint i = lane * 4u; i < halfWidth; i += 128u) {
            d += dot(*((device const float4*)(w + i)), *((device const float4*)(v + i)));
        }
    }
    d = simd_sum(d);
    float ax = 0.0f, am = 0.0f;
    for (uint i = tid; i < p.width; i += 256u) { ax = fma(x[i], x[i], ax); am = fma(m[i], m[i], am); }
    ax = simd_sum(ax);
    am = simd_sum(am);
    if (lane == 0) { partial[sgitg][0] = ax; partial[sgitg][1] = am; dots[sgitg] = d; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= 2u) { return; }
    float tx = 0.0f, tm = 0.0f;
    for (short s = 0; s < 8; ++s) { tx += partial[s][0]; tm += partial[s][1]; }
    const float ix = rsqrt(tx / float(p.width) + p.eps), im = rsqrt(tm / float(p.width) + p.eps);
    const uint r = tg.y * 2u + tid;
    if (r < p.rank) {
        const uint s0 = tid * 4u;
        z[tg.x * p.rank + r] = ix * (dots[s0] + dots[s0 + 1u]) + im * (dots[s0 + 2u] + dots[s0 + 3u]);
    }
}

// Forget, gain and the mix: 16 lanes per channel read its forget and gain rows
// as float4 (rank / 4 of them), two channels per simdgroup, 16 per threadgroup.
// Grid: (width / 16, controllers), 256 threads.
[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling2_gate_combine_decode(
    constant Inkling2GateDecodeParams& p [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device float*       acts    [[buffer(2)]],
    device const float* z       [[buffer(3)]],
    uint2  tg    [[threadgroup_position_in_grid]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const Inkling2GateController c = p.c[tg.y];
    const uint part = lane % 16u;
    const uint i = tg.x * 16u + sgitg * 2u + lane / 16u;          // channel; width % 16 == 0
    device const float4* zv = (device const float4*)(z + tg.y * p.rank);
    device const float4* fr = (device const float4*)((device const float*)(weights + c.forget) + ulong(i) * p.rank);
    device const float4* gr = (device const float4*)((device const float*)(weights + c.gain) + ulong(i) * p.rank);
    float f = 0.0f, g = 0.0f;
    for (uint q = part; q < p.rank / 4u; q += 16u) {
        const float4 zq = zv[q];
        f += dot(fr[q], zq);
        g += dot(gr[q], zq);
    }
    for (ushort off = 8; off > 0; off >>= 1) { f += simd_shuffle_xor(f, off); g += simd_shuffle_xor(g, off); }
    if (part == 0u) {
        const float xv = acts[c.x + i], mv = acts[c.m + i];
        const float tf = tanh(clamp(f, -20.0f, 20.0f)), tgv = tanh(clamp(g, -20.0f, 20.0f));
        acts[c.out + i] = xv - tf * xv + tgv * mv;
    }
}

// ============================================================================
// Metal 4 tensor-ops variants (matmul2d), for GPUs that build them. The Swift
// side selects them only where `supportsApple10TensorOps` holds; everywhere
// else the simdgroup kernels above run, and on GPUs without tensor support
// this block drops out of the library.
// ============================================================================
#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// Stacked Q8_0 projection: 64 tokens x 32 weight rows per threadgroup, K step
// 64 (two Q8_0 blocks per row) decoded to half in threadgroup memory. Same
// segment table as `inkling2_q8_stacked_mm`, with tileStart counting 32-row
// tiles. Grid: x = total 32-row tiles, y = ceil(tokens / 64).
[[kernel, max_total_threads_per_threadgroup(128)]]
void inkling2_q8_stacked_mm_tensorops(
    constant Inkling2ProjParams& p [[buffer(0)]],
    device const uchar* weights    [[buffer(1)]],
    device half*        input      [[buffer(2)]],
    device float*       output     [[buffer(3)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint3 lid3    [[thread_position_in_threadgroup]],
    uint3 threads3[[threads_per_threadgroup]])
{
    constexpr int TM = 64, TN = 32, TK = 64;
    const uint tile = tgid.x;
    uint si = 0;
    for (uint i = 1; i < p.segmentCount; ++i) {
        if (tile >= p.segments[i].tileStart) { si = i; }
    }
    const Inkling2ProjSegment seg = p.segments[si];
    const uint r0 = (tile - seg.tileStart) * uint(TN);
    const uint t0 = tgid.y * uint(TM);
    if (r0 >= seg.rows || t0 >= p.tokens) { return; }

    constexpr auto descriptor = matmul2d_descriptor(TM, TN, TK, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[TN * TK];
    threadgroup_half_tensor tileB(weightTile, dextents<int32_t, 2>(TK, TN), array<int32_t, 2>({1, TK}));
    device_half_tensor firstA(input + seg.inputOffset, dextents<int32_t, 2>(TK, int32_t(p.tokens)),
                              array<int32_t, 2>({1, int32_t(seg.inputStride)}));
    auto firstTileA = firstA.slice(0, int32_t(t0));
    auto acc = op.get_destination_cooperative_tensor<decltype(firstTileA), decltype(tileB), float>();
    for (int e = 0; e < acc.get_capacity(); ++e) { acc[e] = 0.0f; }

    const uint format = seg.format;
    const uint rowBytes = format == 2u ? p.inputs * 4u : format == 1u ? p.inputs * 2u : (p.inputs / 32u) * 34u;
    const uint lid = lid3.x, threads = threads3.x;
    for (uint k0 = 0; k0 < p.inputs; k0 += uint(TK)) {
        for (uint linear = lid; linear < uint(TN * TK); linear += threads) {
            const uint n = linear / uint(TK), k = linear % uint(TK);
            half w = 0.0h;
            if (r0 + n < seg.rows) {
                device const uchar* row = weights + seg.weightOffset + ulong(r0 + n) * rowBytes;
                if (format == 2u) {
                    w = half(((device const float*)row)[k0 + k]);
                } else if (format == 1u) {
                    w = half(float(((device const bfloat*)row)[k0 + k]));
                } else {
                    device const uchar* blk = row + ((k0 + k) / 32u) * 34u;
                    w = *((device const half*)blk) * half(as_type<char>(blk[2u + (k0 + k) % 32u]));
                }
            }
            weightTile[linear] = w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device_half_tensor A(input + seg.inputOffset + k0, dextents<int32_t, 2>(TK, int32_t(p.tokens)),
                             array<int32_t, 2>({1, int32_t(seg.inputStride)}));
        auto tileA = A.slice(0, int32_t(t0));
        auto part = op.get_destination_cooperative_tensor<decltype(tileA), decltype(tileB), float>();
        for (int e = 0; e < part.get_capacity(); ++e) { part[e] = 0.0f; }
        op.run(tileA, tileB, part);
        for (int e = 0; e < acc.get_capacity(); ++e) { acc[e] += part[e]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int e = 0; e < acc.get_capacity(); ++e) {
        if (!acc.is_valid_element(e)) { continue; }
        const auto pos = acc.get_multidimensional_index(e);
        const uint n = r0 + uint(pos[0]), m = t0 + uint(pos[1]);
        if (n < seg.rows && m < p.tokens) {
            output[ulong(m) * p.outStride + seg.outColumn + n] = acc[e];
        }
    }
}

// Relative-bias attention with a sink, tensor-ops form: eight query rows of
// one head per threadgroup, 64-key tiles for both products. Generalised from
// the retired family's `inkling_attention_prefill_tensorops` (left as it is in
// tensorops.metal): any GQA head count, the sink as the initial softmax state,
// FP32 relative states and the distance-major FP32 bias table, the optional
// output gate. A ring tile never crosses the ring end. Same parameters and
// buffers as `inkling2_attention_prefill`. Grid: x = ceil(rows / 8), y = heads.
constant constexpr int INK2_TQ = 8;
constant constexpr int INK2_TK = 64;

[[kernel, max_total_threads_per_threadgroup(128)]]
void inkling2_attention_prefill_tensorops(
    constant Inkling2AttnParams& p [[buffer(0)]],
    device const half*  Q     [[buffer(1)]],
    device const half*  Kc    [[buffer(2)]],
    device const half*  Vc    [[buffer(3)]],
    device const float* rel   [[buffer(4)]],
    device const float* table [[buffer(5)]],
    device const float* sink  [[buffer(6)]],
    device const float* gate  [[buffer(7)]],
    device half*        out   [[buffer(8)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint3 lid3    [[thread_position_in_threadgroup]],
    uint3 threads3[[threads_per_threadgroup]])
{
    const uint lid = lid3.x, threads = threads3.x;
    constexpr auto qkDescriptor = matmul2d_descriptor(INK2_TQ, INK2_TK, INK2_D, false, true, false);
    constexpr auto pvDescriptor = matmul2d_descriptor(INK2_TQ, INK2_D, INK2_TK, false, false, false);
    matmul2d<qkDescriptor, execution_simdgroups<4>> qkOp;
    matmul2d<pvDescriptor, execution_simdgroups<4>> pvOp;
    using deviceHalfTensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroupHalfTensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroupFloatTensor = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half  queryTile[INK2_TQ * INK2_D];
    threadgroup float relTile[INK2_TQ * INK2_R];
    threadgroup float scoreTile[INK2_TQ * INK2_TK];
    threadgroup float weightTile[INK2_TQ * INK2_TK];
    threadgroup float rowMax[INK2_TQ];
    threadgroup float rowSum[INK2_TQ];
    threadgroup float rowOldScale[INK2_TQ];

    const uint q0 = tgid.x * uint(INK2_TQ);
    const uint h = tgid.y;
    if (q0 >= p.queryCount || h >= p.numQHeads) { return; }
    const uint validRows = min(uint(INK2_TQ), p.queryCount - q0);
    const uint kvh = h / (p.numQHeads / p.numKVHeads);
    const uint pos0 = p.startPosition + q0;

    for (uint linear = lid; linear < uint(INK2_TQ * INK2_D); linear += threads) {
        const uint row = linear / uint(INK2_D), d = linear % uint(INK2_D);
        queryTile[linear] = row < validRows ? Q[ulong(q0 + row) * p.qStride + h * uint(INK2_D) + d] : half(0.0h);
    }
    for (uint linear = lid; linear < uint(INK2_TQ * INK2_R); linear += threads) {
        const uint row = linear / uint(INK2_R), d = linear % uint(INK2_R);
        relTile[linear] = row < validRows ? rel[ulong(q0 + row) * p.relStride + h * uint(INK2_R) + d] : 0.0f;
    }
    if (lid < uint(INK2_TQ)) {
        rowMax[lid] = lid < validRows ? sink[ulong(q0 + lid) * p.sinkStride + h] : 0.0f;
        rowSum[lid] = 1.0f;
        rowOldScale[lid] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroupHalfTensor queryTensor(queryTile, dextents<int32_t, 2>(INK2_D, INK2_TQ), array<int32_t, 2>({1, INK2_D}));
    threadgroupFloatTensor weightTensor(weightTile, dextents<int32_t, 2>(INK2_TK, INK2_TQ), array<int32_t, 2>({1, INK2_TK}));
    const uint keyEnd = pos0 + validRows;
    const uint cacheCount = p.ringCapacity != 0u ? p.ringCapacity : keyEnd;
    deviceHalfTensor keyTensor((device half*)(Kc + kvh * uint(INK2_D)), dextents<int32_t, 2>(INK2_D, int32_t(cacheCount)),
                               array<int32_t, 2>({1, int32_t(p.kvStride)}));
    deviceHalfTensor valueTensor((device half*)(Vc + kvh * uint(INK2_D)), dextents<int32_t, 2>(INK2_D, int32_t(cacheCount)),
                                 array<int32_t, 2>({1, int32_t(p.kvStride)}));
    auto querySlice = queryTensor.slice(0, 0);
    auto firstValueSlice = valueTensor.slice(0, 0);
    auto outAcc = pvOp.get_destination_cooperative_tensor<decltype(weightTensor), decltype(firstValueSlice), float>();
    for (int e = 0; e < outAcc.get_capacity(); ++e) { if (outAcc.is_valid_element(e)) { outAcc[e] = 0.0f; } }

    const uint firstKey = (p.window > 0u && pos0 + 1u > p.window) ? pos0 + 1u - p.window : 0u;
    for (uint keyStart = firstKey; keyStart < keyEnd;) {
        const uint physicalStart = p.ringCapacity != 0u ? keyStart % p.ringCapacity : keyStart;
        uint tileCount = min(uint(INK2_TK), keyEnd - keyStart);
        if (p.ringCapacity != 0u) { tileCount = min(tileCount, p.ringCapacity - physicalStart); }
        auto keySlice = keyTensor.slice(0, int32_t(physicalStart));
        auto scores = qkOp.get_destination_cooperative_tensor<decltype(querySlice), decltype(keySlice), float>();
        for (int e = 0; e < scores.get_capacity(); ++e) { if (scores.is_valid_element(e)) { scores[e] = 0.0f; } }
        qkOp.run(querySlice, keySlice, scores);
        for (int e = 0; e < scores.get_capacity(); ++e) {
            if (!scores.is_valid_element(e)) { continue; }
            const auto pos = scores.get_multidimensional_index(e);
            scoreTile[uint(pos[1]) * uint(INK2_TK) + uint(pos[0])] = scores[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < uint(INK2_TQ)) {
            const uint row = lid;
            const bool validRow = row < validRows;
            const uint position = pos0 + row;
            const float tau = ink2_tau(p, position);
            float tileMax = -INFINITY;
            for (uint c = 0; c < uint(INK2_TK); ++c) {
                const uint key = keyStart + c;
                const bool visible = validRow && c < tileCount && key <= position
                    && (p.window == 0u || position - key < p.window);
                float logit = -INFINITY;
                if (visible) {
                    const uint dist = position - key;
                    float bias = 0.0f;
                    if (dist < p.relExtent) {
                        for (uint d = 0; d < uint(INK2_R); ++d) {
                            bias = fma(relTile[row * uint(INK2_R) + d], table[dist * uint(INK2_R) + d], bias);
                        }
                    }
                    logit = tau * fma(scoreTile[row * uint(INK2_TK) + c], p.scale, bias);
                }
                scoreTile[row * uint(INK2_TK) + c] = logit;
                tileMax = max(tileMax, logit);
            }
            const float nextMax = max(rowMax[row], tileMax);
            const float oldScale = exp(rowMax[row] - nextMax);
            float tileSum = 0.0f;
            for (uint c = 0; c < uint(INK2_TK); ++c) {
                const float w = exp(scoreTile[row * uint(INK2_TK) + c] - nextMax);
                weightTile[row * uint(INK2_TK) + c] = w;
                tileSum += w;
            }
            rowOldScale[row] = oldScale;
            rowSum[row] = rowSum[row] * oldScale + tileSum;
            rowMax[row] = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto valueSlice = valueTensor.slice(0, int32_t(physicalStart));
        auto product = pvOp.get_destination_cooperative_tensor<decltype(weightTensor), decltype(valueSlice), float>();
        for (int e = 0; e < product.get_capacity(); ++e) { if (product.is_valid_element(e)) { product[e] = 0.0f; } }
        pvOp.run(weightTensor, valueSlice, product);
        for (int e = 0; e < outAcc.get_capacity(); ++e) {
            if (!outAcc.is_valid_element(e) || !product.is_valid_element(e)) { continue; }
            const auto pos = outAcc.get_multidimensional_index(e);
            outAcc[e] = fma(outAcc[e], rowOldScale[uint(pos[1])], product[e]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        keyStart += tileCount;
    }

    for (int e = 0; e < outAcc.get_capacity(); ++e) {
        if (!outAcc.is_valid_element(e)) { continue; }
        const auto pos = outAcc.get_multidimensional_index(e);
        const uint d = uint(pos[0]), row = uint(pos[1]);
        if (row < validRows) {
            float value = outAcc[e] / rowSum[row];
            if (p.gateStride != 0u) { value *= ink2_sigmoid(gate[ulong(q0 + row) * p.gateStride + h * uint(INK2_D) + d]); }
            out[ulong(q0 + row) * p.outStride + h * uint(INK2_D) + d] = half(value);
        }
    }
}
#endif

// ============================================================================
// MoE router at one token (decode). The prefill router (`inkling2_moe_router`)
// tiles 32 tokens on one threadgroup, so at one token a single threadgroup
// streams the 1.2 MB router matrix (210 us). Here: the logits `W x` by one
// simdgroup per expert row, then that kernel's selection for one token (the
// same code: sigmoid of the clamped logit, top-k of sigmoid + bias with ties to
// the lower expert, weights sigmoid / (sum + 1e-20) * routeScale).
// ============================================================================

struct Inkling2RouterDecodeParams {
    uint  width;          // the hidden size (a multiple of 4)
    uint  experts;        // up to 128
    uint  topK;           // up to 32
    float routeScale;
};

// Grid: ceil(experts / 2) threadgroups of 256: two expert rows per threadgroup,
// each row's dot product split over four simdgroups (a quarter of the width each).
[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling2_router_logits_decode(
    constant Inkling2RouterDecodeParams& p [[buffer(0)]],
    device const float* x       [[buffer(1)]],    // [width]
    device const float* weight  [[buffer(2)]],    // [experts][width]
    device float*       logits  [[buffer(3)]],    // [experts]
    uint   tg    [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]])
{
    threadgroup float partial[8];
    const uint e = tg * 2u + sgitg / 4u, q = sgitg % 4u;
    const uint quarter = p.width / 16u;                 // float4s in a quarter row
    float acc = 0.0f;
    if (e < p.experts) {
        device const float4* w = (device const float4*)(weight + ulong(e) * p.width) + q * quarter;
        device const float4* xv = (device const float4*)x + q * quarter;
        for (uint i = lane; i < quarter; i += 32u) { acc += dot(w[i], xv[i]); }
    }
    acc = simd_sum(acc);
    if (lane == 0) { partial[sgitg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2u) {
        const uint row = tg * 2u + tid;
        if (row < p.experts) {
            logits[row] = (partial[tid * 4u] + partial[tid * 4u + 1u]) + (partial[tid * 4u + 2u] + partial[tid * 4u + 3u]);
        }
    }
}

// Grid: one threadgroup of 32 (one simdgroup); lane l holds experts l + 32 j.
[[kernel, max_total_threads_per_threadgroup(32)]]
void inkling2_router_select_decode(
    constant Inkling2RouterDecodeParams& p [[buffer(0)]],
    device const float* logits       [[buffer(1)]],
    device const float* bias         [[buffer(2)]],
    device uint*        ids          [[buffer(3)]],
    device float*       routeWeights [[buffer(4)]],
    ushort lane [[thread_index_in_simdgroup]])
{
    float prob[4], sel[4];
    for (uint j = 0; j < 4; ++j) {
        const uint e = lane + 32u * j;
        const float l = e < p.experts ? clamp(logits[e], -80.0f, 80.0f) : 0.0f;
        prob[j] = e < p.experts ? 1.0f / (1.0f + exp(-l)) : 0.0f;
        sel[j] = e < p.experts ? prob[j] + bias[e] : -FLT_MAX;
    }
    // Lane r keeps rank r; the sum runs in rank order.
    float sum = 0.0f, myProb = 0.0f;
    uint myId = 0u;
    for (uint r = 0; r < p.topK; ++r) {
        float best = sel[0];
        uint bestE = lane;
        for (uint j = 1; j < 4; ++j) {
            if (sel[j] > best) { best = sel[j]; bestE = lane + 32u * j; }
        }
        const float top = simd_max(best);
        const uint chosen = simd_min(best == top ? bestE : 0xFFFFFFFFu);
        float chosenProb = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            if (lane + 32u * j == chosen) { chosenProb = prob[j]; sel[j] = -FLT_MAX; }
        }
        chosenProb = simd_sum(chosenProb);
        sum += chosenProb;
        if (lane == r) { myProb = chosenProb; myId = chosen; }
    }
    if (lane < p.topK) {
        ids[lane] = myId;
        routeWeights[lane] = (myProb / (sum + 1e-20f)) * p.routeScale;
    }
}
