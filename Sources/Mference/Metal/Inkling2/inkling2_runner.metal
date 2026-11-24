// Inkling 2 Mini runner kernels: the parts of a layer visit between the
// attention block (inkling2_attention*.metal) and the routed experts
// (inkling2_moe*.metal), plus the embedding and the output head.
//
// - inkling2_depth_mix: block attention residuals and cross-pass messages.
//   Every state is RMS-normalized without a weight, each learned query
//   scores the normalized states with 1/sqrt(d), the output is the
//   softmax-weighted sum of the normalized states.
// - inkling2_message_gate: z = down [rms(x); rms(m)],
//   out = (1 - tanh(forget z)) x + tanh(gain z) m, 16 tokens per
//   threadgroup so each weight read serves the whole tile.
// - inkling2_embed_q8_0: Q8_0 embedding rows, then the weighted RMSNorm.
// - inkling2_rms_norm: weighted RMSNorm, FP32 in and out; the _dual form
//   also writes an FP16 copy (router and experts read one each).
// - inkling2_head_q8_0: Q8_0 output head over a normalized row, logit
//   softcap, FP16 logits.
//
// FP32 everywhere except the FP16 logits. One threadgroup of 256 threads per
// token row for the per-token kernels.

#include <metal_stdlib>
using namespace metal;

constant uint I2R_THREADS = 256;
constant uint I2R_SIMDS = I2R_THREADS / 32;
constant uint I2R_MAX_STATES = 8;
constant uint I2R_MAX_QUERIES = 4;
constant uint I2R_MAX_RANK = 64;

// Sum over the threadgroup; every thread gets the total.
static inline float i2r_sum(float v, threadgroup float *scratch, uint tid, uint simd, uint lane) {
    v = simd_sum(v);
    if (lane == 0) { scratch[simd] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < I2R_SIMDS; ++s) { total += scratch[s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// ---------------------------------------------------------------- depth mix

struct I2RMixParams {
    uint d;
    uint tokens;
    uint states;              // 1 ... I2R_MAX_STATES
    uint queries;             // 1 ... I2R_MAX_QUERIES
    uint stateStride;         // floats between token rows of a state
    uint outStride;           // floats between token rows of an output
    uint outQueryStride;      // floats between the outputs of consecutive queries
    float eps;
    uint stateOffsets[I2R_MAX_STATES];  // float offset of each state's row 0 in `arena`
};

kernel void inkling2_depth_mix(device const float *arena    [[buffer(0)]],
                               device const float *queries  [[buffer(1)]],   // [queries][d]
                               device float *out            [[buffer(2)]],
                               constant I2RMixParams &p     [[buffer(3)]],
                               uint t    [[threadgroup_position_in_grid]],
                               uint tid  [[thread_index_in_threadgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[I2R_SIMDS];
    threadgroup float invRms[I2R_MAX_STATES];
    threadgroup float weight[I2R_MAX_QUERIES][I2R_MAX_STATES];
    if (t >= p.tokens) { return; }
    const float scale = rsqrt(float(p.d));

    float scores[I2R_MAX_QUERIES][I2R_MAX_STATES];
    for (uint s = 0; s < p.states; ++s) {
        device const float *row = arena + p.stateOffsets[s] + t * p.stateStride;
        float ss = 0.0f;
        float dots[I2R_MAX_QUERIES] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = tid; i < p.d; i += I2R_THREADS) {
            const float v = row[i];
            ss += v * v;
            for (uint q = 0; q < p.queries; ++q) { dots[q] += v * queries[q * p.d + i]; }
        }
        const float total = i2r_sum(ss, scratch, tid, simd, lane);
        const float inv = rsqrt(total / float(p.d) + p.eps);
        if (tid == 0) { invRms[s] = inv; }
        for (uint q = 0; q < p.queries; ++q) {
            scores[q][s] = i2r_sum(dots[q], scratch, tid, simd, lane) * inv * scale;
        }
    }
    if (tid == 0) {
        for (uint q = 0; q < p.queries; ++q) {
            float m = scores[q][0];
            for (uint s = 1; s < p.states; ++s) { m = max(m, scores[q][s]); }
            float z = 0.0f;
            for (uint s = 0; s < p.states; ++s) { z += precise::exp(scores[q][s] - m); }
            for (uint s = 0; s < p.states; ++s) { weight[q][s] = precise::exp(scores[q][s] - m) / z; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.d; i += I2R_THREADS) {
        float acc[I2R_MAX_QUERIES] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint s = 0; s < p.states; ++s) {
            const float v = arena[p.stateOffsets[s] + t * p.stateStride + i] * invRms[s];
            for (uint q = 0; q < p.queries; ++q) { acc[q] += weight[q][s] * v; }
        }
        for (uint q = 0; q < p.queries; ++q) {
            out[q * p.outQueryStride + t * p.outStride + i] = acc[q];
        }
    }
}

// ------------------------------------------------------------- message gate

struct I2RGateParams {
    uint d;               // multiple of I2R_GATE_CHUNK / 2
    uint tokens;
    uint rank;            // <= I2R_MAX_RANK
    uint xStride;
    uint mStride;
    uint outStride;
    float eps;
};

constant uint I2R_GATE_TILE = 16;     // tokens per threadgroup, sharing each weight read
constant uint I2R_GATE_CHUNK = 64;    // input columns staged per step

// One threadgroup per I2R_GATE_TILE tokens. Thread `tid` owns rank
// `tid % 64` for tokens `tid / 64 + 4k`, k < 4, while z is accumulated.
kernel void inkling2_message_gate(device const float *x       [[buffer(0)]],
                                  device const float *m       [[buffer(1)]],
                                  device const float *down    [[buffer(2)]],   // [rank][2d]
                                  device const float *gain    [[buffer(3)]],   // [d][rank]
                                  device const float *forget  [[buffer(4)]],   // [d][rank]
                                  device float *out           [[buffer(5)]],
                                  constant I2RGateParams &p   [[buffer(6)]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint tid   [[thread_index_in_threadgroup]],
                                  uint simd  [[simdgroup_index_in_threadgroup]],
                                  uint lane  [[thread_index_in_simdgroup]]) {
    threadgroup float invX[I2R_GATE_TILE];
    threadgroup float invM[I2R_GATE_TILE];
    threadgroup float z[I2R_GATE_TILE][I2R_MAX_RANK];
    threadgroup float gIn[I2R_GATE_TILE][I2R_GATE_CHUNK];
    threadgroup float wDown[I2R_MAX_RANK][I2R_GATE_CHUNK];

    const uint t0 = group * I2R_GATE_TILE;
    if (t0 >= p.tokens) { return; }
    const uint nT = min(I2R_GATE_TILE, p.tokens - t0);
    const uint twoD = 2 * p.d;

    // Weightless norms of x and m, one simdgroup per token.
    for (uint k = simd; k < nT; k += I2R_SIMDS) {
        device const float *xr = x + (t0 + k) * p.xStride;
        device const float *mr = m + (t0 + k) * p.mStride;
        float sx = 0.0f, sm = 0.0f;
        for (uint i = lane; i < p.d; i += 32) {
            sx += xr[i] * xr[i];
            sm += mr[i] * mr[i];
        }
        sx = simd_sum(sx);
        sm = simd_sum(sm);
        if (lane == 0) {
            invX[k] = rsqrt(sx / float(p.d) + p.eps);
            invM[k] = rsqrt(sm / float(p.d) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // z = down [rms(x); rms(m)] for the tile, a chunk of columns at a time.
    const uint r = tid % I2R_MAX_RANK;
    const uint kBase = tid / I2R_MAX_RANK;           // 0 ..< 4
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint c0 = 0; c0 < twoD; c0 += I2R_GATE_CHUNK) {
        for (uint e = tid; e < I2R_GATE_TILE * I2R_GATE_CHUNK; e += I2R_THREADS) {
            const uint k = e / I2R_GATE_CHUNK, j = e % I2R_GATE_CHUNK, c = c0 + j;
            float v = 0.0f;
            if (k < nT) {
                v = c < p.d ? x[(t0 + k) * p.xStride + c] * invX[k]
                            : m[(t0 + k) * p.mStride + (c - p.d)] * invM[k];
            }
            gIn[k][j] = v;
        }
        for (uint e = tid; e < I2R_MAX_RANK * I2R_GATE_CHUNK; e += I2R_THREADS) {
            const uint rr = e / I2R_GATE_CHUNK, j = e % I2R_GATE_CHUNK;
            wDown[rr][j] = rr < p.rank ? down[rr * twoD + c0 + j] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < I2R_GATE_CHUNK; ++j) {
            const float w = wDown[r][j];
            for (uint k = 0; k < 4; ++k) { acc[k] += w * gIn[kBase + 4 * k][j]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint k = 0; k < 4; ++k) { z[kBase + 4 * k][r] = acc[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // out = (1 - tanh(forget z)) x + tanh(gain z) m, one column per thread.
    for (uint i = tid; i < p.d; i += I2R_THREADS) {
        device const float *g = gain + i * p.rank;
        device const float *f = forget + i * p.rank;
        float gs[I2R_GATE_TILE], fs[I2R_GATE_TILE];
        for (uint k = 0; k < I2R_GATE_TILE; ++k) { gs[k] = 0.0f; fs[k] = 0.0f; }
        for (uint rr = 0; rr < p.rank; ++rr) {
            const float gr = g[rr], fr = f[rr];
            for (uint k = 0; k < I2R_GATE_TILE; ++k) {
                gs[k] += gr * z[k][rr];
                fs[k] += fr * z[k][rr];
            }
        }
        for (uint k = 0; k < nT; ++k) {
            const float xv = x[(t0 + k) * p.xStride + i];
            const float mv = m[(t0 + k) * p.mStride + i];
            out[(t0 + k) * p.outStride + i] = (1.0f - precise::tanh(fs[k])) * xv + precise::tanh(gs[k]) * mv;
        }
    }
}

// ----------------------------------------------------- Q8_0 rows and norms

// ggml block_q8_0: FP16 scale then 32 int8, 34 bytes. Row starts are even.
static inline float i2r_q8_0(device const uchar *row, uint i) {
    device const uchar *block = row + (i / 32) * 34;
    const half scale = as_type<half>(ushort(ushort(block[0]) | (ushort(block[1]) << 8)));
    return float(scale) * float(as_type<char>(block[2 + (i % 32)]));
}

struct I2RNormParams {
    uint d;
    uint tokens;
    uint inStride;     // embed: unused; norm: floats between input rows
    uint outStride;
    uint rowBytes;     // embed: bytes per Q8_0 row
    float eps;
};

kernel void inkling2_embed_q8_0(device const uint *ids        [[buffer(0)]],
                                device const uchar *table     [[buffer(1)]],
                                device const float *weight    [[buffer(2)]],
                                device float *out             [[buffer(3)]],
                                constant I2RNormParams &p     [[buffer(4)]],
                                uint t    [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[I2R_SIMDS];
    if (t >= p.tokens) { return; }
    device const uchar *row = table + ulong(ids[t]) * p.rowBytes;
    device float *o = out + t * p.outStride;
    float ss = 0.0f;
    for (uint i = tid; i < p.d; i += I2R_THREADS) {
        const float v = i2r_q8_0(row, i);
        o[i] = v;
        ss += v * v;
    }
    const float inv = rsqrt(i2r_sum(ss, scratch, tid, simd, lane) / float(p.d) + p.eps);
    for (uint i = tid; i < p.d; i += I2R_THREADS) {
        o[i] = o[i] * inv * weight[i];
    }
}

kernel void inkling2_rms_norm(device const float *x        [[buffer(0)]],
                              device const float *weight   [[buffer(1)]],
                              device float *out            [[buffer(2)]],
                              constant I2RNormParams &p    [[buffer(3)]],
                              uint t    [[threadgroup_position_in_grid]],
                              uint tid  [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[I2R_SIMDS];
    if (t >= p.tokens) { return; }
    device const float *xr = x + t * p.inStride;
    float ss = 0.0f;
    for (uint i = tid; i < p.d; i += I2R_THREADS) { ss += xr[i] * xr[i]; }
    const float inv = rsqrt(i2r_sum(ss, scratch, tid, simd, lane) / float(p.d) + p.eps);
    for (uint i = tid; i < p.d; i += I2R_THREADS) {
        out[t * p.outStride + i] = xr[i] * inv * weight[i];
    }
}

// The MoE input in both forms K2's kernels read: FP32 for the router, FP16
// for the grouped experts. `halfStride` is in halves.
kernel void inkling2_rms_norm_dual(device const float *x        [[buffer(0)]],
                                   device const float *weight   [[buffer(1)]],
                                   device float *out            [[buffer(2)]],
                                   device half *outHalf         [[buffer(3)]],
                                   constant I2RNormParams &p    [[buffer(4)]],
                                   constant uint &halfStride    [[buffer(5)]],
                                   uint t    [[threadgroup_position_in_grid]],
                                   uint tid  [[thread_index_in_threadgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[I2R_SIMDS];
    if (t >= p.tokens) { return; }
    device const float *xr = x + t * p.inStride;
    float ss = 0.0f;
    for (uint i = tid; i < p.d; i += I2R_THREADS) { ss += xr[i] * xr[i]; }
    const float inv = rsqrt(i2r_sum(ss, scratch, tid, simd, lane) / float(p.d) + p.eps);
    for (uint i = tid; i < p.d; i += I2R_THREADS) {
        const float v = xr[i] * inv * weight[i];
        out[t * p.outStride + i] = v;
        outHalf[t * halfStride + i] = half(v);
    }
}

// ------------------------------------------------------------- output head

struct I2RHeadParams {
    uint d;
    uint rows;         // logits written (the unpadded vocabulary)
    uint rowBytes;
    float softcap;     // 0 disables
};

constant uint I2R_HEAD_ROWS_PER_SIMD = 4;

// One normalized FP32 row x -> FP16 logits. Each simdgroup takes
// I2R_HEAD_ROWS_PER_SIMD consecutive rows; lanes split each row's blocks
// (lane k: blocks k, k + 32, ... in order, each summed in order, then
// simd_sum). The simdgroup's rows run interleaved, so one x load serves them
// all and their weight loads are in flight together; x is read from device
// memory (it stays in cache) and a block's codes in eight 4-byte loads.
kernel void inkling2_head_q8_0(device const float *x        [[buffer(0)]],
                               device const uchar *table    [[buffer(1)]],
                               device half *logits          [[buffer(2)]],
                               constant I2RHeadParams &p    [[buffer(3)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint simd  [[simdgroup_index_in_threadgroup]],
                               uint lane  [[thread_index_in_simdgroup]]) {
    const uint blocks = p.d / 32;
    const uint firstRow = (group * I2R_SIMDS + simd) * I2R_HEAD_ROWS_PER_SIMD;
    if (firstRow >= p.rows) { return; }
    device const uchar *rows[I2R_HEAD_ROWS_PER_SIMD];
    float acc[I2R_HEAD_ROWS_PER_SIMD];
    for (uint k = 0; k < I2R_HEAD_ROWS_PER_SIMD; ++k) {
        // The simdgroup's rows at or past `rows` re-read row rows - 1 (no
        // read past the rows asked for) and are not written.
        rows[k] = table + ulong(min(firstRow + k, p.rows - 1)) * p.rowBytes;
        acc[k] = 0.0f;
    }
    for (uint b = lane; b < blocks; b += 32) {
        device const float4 *xb = reinterpret_cast<device const float4 *>(x + b * 32);
        float4 xv[8];
        for (uint j = 0; j < 8; ++j) { xv[j] = xb[j]; }
        for (uint k = 0; k < I2R_HEAD_ROWS_PER_SIMD; ++k) {
            device const uchar *block = rows[k] + b * 34;
            const float scale = float(*reinterpret_cast<device const half *>(block));
            device const packed_char4 *q = reinterpret_cast<device const packed_char4 *>(block + 2);
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                const char4 c = char4(q[j]);
                s += float(c.x) * xv[j].x;
                s += float(c.y) * xv[j].y;
                s += float(c.z) * xv[j].z;
                s += float(c.w) * xv[j].w;
            }
            acc[k] += scale * s;
        }
    }
    for (uint k = 0; k < I2R_HEAD_ROWS_PER_SIMD; ++k) {
        const float total = simd_sum(acc[k]);
        const uint r = firstRow + k;
        if (lane == 0 && r < p.rows) {
            float v = total;
            if (p.softcap > 0.0f) { v = p.softcap * precise::tanh(v / p.softcap); }
            logits[r] = half(v);
        }
    }
}
