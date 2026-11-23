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
// bytes, value d*q) or a BF16 weight (rows of K bfloats; the file stores the
// sink and dynamic-tap projections that way). Segments let every projection that shares a visit's
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
    uint  format;         // 0 = GGUF Q8_0 blocks, 1 = BF16 rows
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
    const bool bf16 = seg.format != 0u;
    const uint rowBytes = bf16 ? p.inputs * 2u : blocks * 34u;
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
        if (bf16) {
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
        for (short ik = 0; ik < 4; ++ik) {
            for (short i = 0; i < 4; ++i) { simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false); }
            for (short i = 0; i < 2; ++i) { simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false); }
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

// ----------------------------------------------------------------------------
// Small-batch Q8_0 projection (decode and tiny chunks): each simdgroup owns
// two weight rows over the whole K; lane l reads eight quants of block
// l/4 + 8i at 8*(l%4), so 8 blocks (272 bytes per row) are in flight per
// step, and every activation load serves both rows. NT (1, 2, 4 or 8) bounds
// the token rows sharing each weight read. Same segment table as the stacked
// product with tileStart counting row pairs. Grid: x = ceil(pairs / 4)
// threadgroups of 4 simdgroups.
// ----------------------------------------------------------------------------

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
    const uint blocks = p.inputs / 32u;
    const bool bf16 = seg.format != 0u;
    const uint rowBytes = bf16 ? p.inputs * 2u : blocks * 34u;
    device const uchar* row0 = weights + seg.weightOffset + ulong(r) * rowBytes;
    device const uchar* row1 = second ? row0 + rowBytes : row0;
    const uint tokens = min(p.tokens, uint(NT));

    float acc0[NT], acc1[NT];
    for (short t = 0; t < NT; ++t) { acc0[t] = 0.0f; acc1[t] = 0.0f; }
    const uint part = lane % 4u;
    device const half* xbase = input + seg.inputOffset + 8u * part;
    for (uint b = lane / 4u; b < blocks; b += 8u) {
        float4 xa[NT], xb[NT];
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
        for (short which = 0; which < 2; ++which) {
            device const uchar* rowp = which == 0 ? row0 : row1;
            float d;
            float4 wa, wb;
            if (bf16) {
                device const bfloat4* src = (device const bfloat4*)(rowp + (b * 32u + 8u * part) * 2u);
                d = 1.0f;
                wa = float4(src[0]);
                wb = float4(src[1]);
            } else {
                device const uchar* blk = rowp + b * 34u;
                d = float(*((device const half*)blk));
                device const ushort* qs = (device const ushort*)(blk + 2u + 8u * part);
                const char2 c0 = as_type<char2>(qs[0]), c1 = as_type<char2>(qs[1]);
                const char2 c2 = as_type<char2>(qs[2]), c3 = as_type<char2>(qs[3]);
                wa = float4(c0.x, c0.y, c1.x, c1.y);
                wb = float4(c2.x, c2.y, c3.x, c3.y);
            }
            for (short t = 0; t < NT; ++t) {
                const float v = d * (dot(wa, xa[t]) + dot(wb, xb[t]));
                if (which == 0) { acc0[t] += v; } else { acc1[t] += v; }
            }
        }
    }
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
// Threadgroup: 32 query rows of one query head, 4 simdgroups of 8 rows; keys
// in tiles of BK. Q, S, P and O live in simdgroup fragments; K, V and the P
// rows of the tile are staged in threadgroup memory. Grid: x = ceil(rows/32),
// y = query heads.
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

template <short BK>
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
    constexpr short BQ = 32;
    constexpr short D = INK2_D;
    constexpr short DP = INK2_D + 8;     // padded row: 8x8 loads hit distinct banks
    constexpr short R = INK2_R;
    constexpr short NB = BK / 8;
    // K tile then V tile; Q is staged here first (BQ * DP <= 2 * BK * DP for BK >= 16).
    threadgroup half kv[2 * BK * DP];
    threadgroup float pTable[(BQ + BK) * R];
    threadgroup half* kTile = kv;
    threadgroup half* vTile = kv + BK * DP;

    const uint h = tgpig.y;
    const uint kvh = h / (p.numQHeads / p.numKVHeads);
    const uint q0 = tgpig.x * uint(BQ);
    if (q0 >= p.queryCount) { return; }
    const uint rowsHere = min(uint(BQ), p.queryCount - q0);
    const uint pos0 = p.startPosition + q0;

    for (uint e = tiitg; e < uint(BQ * D / 4); e += 128u) {
        const uint row = e / uint(D / 4), c4 = e % uint(D / 4);
        half4 v = half4(0.0h);
        if (row < rowsHere) {
            v = ((device const half4*)(Q + ulong(q0 + row) * p.qStride + h * uint(D)))[c4];
        }
        ((threadgroup half4*)(kv + row * uint(DP)))[c4] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qf[D / 8];
    INK2_UNROLL
    for (short ds = 0; ds < D / 8; ++ds) {
        simdgroup_load(qf[ds], kv + (8 * sgitg) * DP + ds * 8, DP, 0, false);
    }

    const short fm = ink2_frag_row(lane);
    const short fn = ink2_frag_col(lane);
    const uint myRow = q0 + 8u * sgitg + uint(fm);
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
        for (uint e = tiitg; e < uint(BK * D / 4); e += 128u) {
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
        for (uint e = tiitg; e < uint((BQ + BK - 1) * R / 4); e += 128u) {
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
decltype(inkling2_attention_prefill<16>) inkling2_attention_prefill<16>;
template [[host_name("inkling2_attention_prefill_bk32")]] [[kernel]]
decltype(inkling2_attention_prefill<32>) inkling2_attention_prefill<32>;

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
// same `y` columns afterwards. Grid: (tokens, 16) threadgroups of 128
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
    device const float* kTail      [[buffer(6)]],
    device const float* vTail      [[buffer(7)]],
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
    device const float* tail = isKey ? kTail : vTail;
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
// Grid: (channels / 4, tokens), one thread per four channels.
// ----------------------------------------------------------------------------

struct Inkling2ConvParams {
    uint tokens;
    uint channels;
    uint xStride;
    uint outStride;
    uint residualStride;  // 0 = no residual
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

kernel void inkling2_sconv_f32(
    constant Inkling2ConvParams& p [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device const float* weight     [[buffer(2)]],
    device const float* tail       [[buffer(3)]],
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
    device const float* xr = x + ulong(row) * p.xStride;
    float acc = 0.0f;
    for (uint i = tid; i < p.width; i += 256u) { acc = fma(xr[i], xr[i], acc); }
    acc = simd_sum(acc);
    if (lane == 0) { partial[sgitg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (short s = 0; s < 8; ++s) { total += partial[s]; }
    const float inv = rsqrt(total / float(p.width) + p.eps);
    device half* orow = out + ulong(row) * p.outStride;
    for (uint i = tid; i < p.width; i += 256u) { orow[i] = half(xr[i] * inv * weight[i]); }
}

// ----------------------------------------------------------------------------
// Decode attention (one query row), split over key blocks.
//
// Pass 1, grid (blocks, KV heads): a threadgroup takes the three query heads
// of one KV head over one block of keys; each simdgroup walks every fourth
// key with the lanes on head dims (4 each) and folds the 16-wide bias into the
// same simd_sum as q.k. Partial (max, sum, out[128]) per (query head, block).
// Pass 2, grid (query heads): merges the blocks with the sink as the starting
// state and applies the sigmoid gate.
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

[[kernel, max_total_threads_per_threadgroup(128)]]
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
    threadgroup float stage[4][3][INK2_PARTIAL];
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
    for (uint n = first + sgitg; n < last; n += 4u) {
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
    // 128 threads: thread = output dim; loop the three heads.
    const uint d = tiitg;
    for (uint j = 0; j < 3u; ++j) {
        float M = -INFINITY;
        for (short s = 0; s < 4; ++s) { M = max(M, stage[s][j][0]); }
        float L = 0.0f, O = 0.0f;
        for (short s = 0; s < 4; ++s) {
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
    uint   h   [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    device const float* base = partials + ulong(h) * p.blocks * INK2_PARTIAL;
    const float s = sink[h];
    float M = s;
    for (uint b = 0; b < p.blocks; ++b) { M = max(M, base[b * INK2_PARTIAL]); }
    float L = exp(s - M), O = 0.0f;
    for (uint b = 0; b < p.blocks; ++b) {
        const float mb = base[b * INK2_PARTIAL];
        const float w = mb == -INFINITY ? 0.0f : exp(mb - M);
        L += base[b * INK2_PARTIAL + 1u] * w;
        O += base[b * INK2_PARTIAL + 2u + tid] * w;
    }
    float result = O / L;
    if (p.hasGate != 0u) { result *= ink2_sigmoid(gate[h * 128u + tid]); }
    out[h * 128u + tid] = half(result);
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

    const bool bf16 = seg.format != 0u;
    const uint rowBytes = bf16 ? p.inputs * 2u : (p.inputs / 32u) * 34u;
    const uint lid = lid3.x, threads = threads3.x;
    for (uint k0 = 0; k0 < p.inputs; k0 += uint(TK)) {
        for (uint linear = lid; linear < uint(TN * TK); linear += threads) {
            const uint n = linear / uint(TK), k = linear % uint(TK);
            half w = 0.0h;
            if (r0 + n < seg.rows) {
                device const uchar* row = weights + seg.weightOffset + ulong(r0 + n) * rowBytes;
                if (bf16) {
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
