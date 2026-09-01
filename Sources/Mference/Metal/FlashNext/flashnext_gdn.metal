#include <metal_stdlib>
using namespace metal;

// ============================================================================
// flashnext_gdn — a dimension-generic gated DeltaNet decode step.
//
// # Why this exists next to the shipped `gdn.metal`
//
// Flash-Next's GDN block IS `Qwen3_5GatedDeltaNet` at Qwen 3.8's exact geometry
// (Hk 16, Hv 48, Dk 128, Dv 128, conv 4), so the production path is the shipped
// kernels — including the fused Hv=48 decode — with the gated norm's activation
// switched to sigmoid via `FC_GDN_GATE_SIGMOID`.
//
// Those kernels are 32-lane tiled: `GDN.init` requires `key_head_dim % 32 == 0`
// and `value_head_dim % 4 == 0`. The parity toy's GDN is Hk 2 / Hv 4 / Dk 8 /
// Dv 8, which does not qualify and cannot be made to without regenerating every
// committed golden. Without a path that runs at those dims, the toy could gate
// the hyper-connections, the PLE, the indexer, attention and the MoE — and then
// four of its six layers would have no block at all.
//
// So this file is the fallback: same recurrence, same order, no tiling
// assumptions. It is a correctness path, not a performance one — the runner
// takes the shipped kernels whenever the geometry qualifies, which is every real
// install. `FlashNextGDNTests` gates it against the CPU reference forward's own
// `gatedDeltaNet` at toy dims, and gates the shipped **sigmoid** variant at the
// production Hv=48 geometry separately, so both halves of the branch are covered.
//
// Semantics, transcribed from `FlashNextReferenceRunner.gatedDeltaNet` (which
// implements `torch_recurrent_gated_delta_rule`):
//
//   conv    = silu(causal_depthwise_conv(qkv))          // dilation 1, kernel K
//   q, k    = l2norm(conv heads) with q scaled by 1/sqrt(Dk)
//   beta    = sigmoid(b[h])
//   decay   = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))
//   S      *= decay;  mem = S^T k;  delta = (v - mem) * beta
//   S      += outer(k, delta);      y = S^T q
//   out     = rmsnorm(y) * SIGMOID(z)                   // not silu
//
// Two details that are easy to get wrong and are load-bearing:
//   * `l2norm` divides by `sqrt(sum + 1e-6)` — the SUM, not the mean, with the
//     epsilon inside the root.
//   * the gated norm's RMS uses the MEAN plus eps, and its weight is
//     ones-initialized, so it must NOT carry the `(1 + w)` bake.
//
// State layout here is `[head][dk][dv]`, the reference's own indexing. That is a
// different order from the shipped kernels' `[Hv, Dv, Dk]`, which is fine
// because the two paths are never mixed on one install — but it is why a runner
// must not switch paths mid-sequence.
// ============================================================================

constant constexpr uint kFlashNextGDNMaxKeyHeadDim = 128;
constant constexpr uint kFlashNextGDNMaxSimdGroups = 32;

static inline float flashnext_gdn_silu(float x) { return x / (1.0f + exp(-x)); }
static inline float flashnext_gdn_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float flashnext_gdn_softplus(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

// ---------------------------------------------------------------------------
// Causal depthwise conv over `[cached tail | current row]`, then SiLU.
//
// `padded` is `[(K - 1) + rows, qkvDim]`: the caller blits the carried tail into
// its head and the new rows after it, so the taps index backwards without a
// boundary case — the same layout `flashnext_ple_conv` uses. Dilation is 1 here
// (the PLE conv is the dilated one).
// ---------------------------------------------------------------------------
kernel void flashnext_gdn_conv(
    device const half*   padded      [[buffer(0)]],  // [(K-1+rows), qkvDim] FP16
    device       half*   out         [[buffer(1)]],  // [rows, qkvDim] FP16
    device const bfloat* weight      [[buffer(2)]],  // [qkvDim, K] BF16
    constant     uint&   qkvDim      [[buffer(3)]],
    constant     uint&   kernelWidth [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]            // x = channel, y = row
) {
    if (gid.x >= qkvDim) return;
    float acc = 0.0f;
    for (uint j = 0; j < kernelWidth; ++j) {
        acc = fma(float(weight[gid.x * kernelWidth + j]),
                  float(padded[ulong(gid.y + j) * ulong(qkvDim) + gid.x]), acc);
    }
    out[ulong(gid.y) * ulong(qkvDim) + gid.x] = half(flashnext_gdn_silu(acc));
}

// ---------------------------------------------------------------------------
// One decode step of the gated delta rule plus the sigmoid gated norm.
//
// One threadgroup per value head; one thread per value channel. `q` and `k` are
// reduced cooperatively into threadgroup memory because every `dv` thread needs
// the whole head.
// ---------------------------------------------------------------------------
kernel void flashnext_gdn_recurrence(
    device const half*   conv     [[buffer(0)]],   // [qkvDim] FP16, post-conv
    device const half*   zProj    [[buffer(1)]],   // [Hv * Dv] FP16
    device const half*   aProj    [[buffer(2)]],   // [Hv] FP16
    device const half*   bProj    [[buffer(3)]],   // [Hv] FP16
    device const bfloat* aLog     [[buffer(4)]],   // [Hv] BF16
    device const bfloat* dtBias   [[buffer(5)]],   // [Hv] BF16
    device const bfloat* normW    [[buffer(6)]],   // [Dv] BF16, ones-centered
    device       float*  state    [[buffer(7)]],   // [Hv, Dk, Dv] FP32
    device       half*   out      [[buffer(8)]],   // [Hv * Dv] FP16
    constant     uint&   numKHeads [[buffer(9)]],
    constant     uint&   numVHeads [[buffer(10)]],
    constant     uint&   keyHeadDim [[buffer(11)]],
    constant     uint&   valueHeadDim [[buffer(12)]],
    constant     float&  eps      [[buffer(13)]],
    uint  h    [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  tsize [[threads_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  simdgroups [[simdgroups_per_threadgroup]]
) {
    threadgroup float qHead[kFlashNextGDNMaxKeyHeadDim];
    threadgroup float kHead[kFlashNextGDNMaxKeyHeadDim];
    threadgroup float partial[kFlashNextGDNMaxSimdGroups];
    threadgroup float shared[2];

    const uint Hk = numKHeads;
    const uint Hv = numVHeads;
    const uint Dk = keyHeadDim;
    const uint Dv = valueHeadDim;
    if (h >= Hv || Dk > kFlashNextGDNMaxKeyHeadDim) return;
    const uint keyDim = Hk * Dk;
    const uint hk = h / (Hv / Hk);

    // Load the head's q and k, and reduce their squared sums.
    float qsum = 0.0f;
    float ksum = 0.0f;
    for (uint d = tid; d < Dk; d += tsize) {
        const float qv = float(conv[hk * Dk + d]);
        const float kv = float(conv[keyDim + hk * Dk + d]);
        qHead[d] = qv;
        kHead[d] = kv;
        qsum = fma(qv, qv, qsum);
        ksum = fma(kv, kv, ksum);
    }
    qsum = simd_sum(qsum);
    ksum = simd_sum(ksum);
    if (lane == 0) partial[sg] = qsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < simdgroups) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) shared[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[sg] = ksum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < simdgroups) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) shared[1] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // l2norm divides by sqrt(SUM + 1e-6); the query additionally by sqrt(Dk).
    const float qScale = (1.0f / sqrt(shared[0] + 1e-6f)) / sqrt(float(Dk));
    const float kScale = 1.0f / sqrt(shared[1] + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < Dk; d += tsize) {
        qHead[d] *= qScale;
        kHead[d] *= kScale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float beta = flashnext_gdn_sigmoid(float(bProj[h]));
    const float decay = exp(-exp(float(aLog[h]))
                            * flashnext_gdn_softplus(float(aProj[h])
                                                     + float(dtBias[h])));
    device float* S = state + ulong(h) * ulong(Dk) * ulong(Dv);
    const uint vBase = 2u * keyDim + h * Dv;

    // Pass 1: decay the state and read the memory term.
    float memory = 0.0f;
    float y = 0.0f;
    if (tid < Dv) {
        for (uint dk = 0; dk < Dk; ++dk) {
            const uint i = dk * Dv + tid;
            S[i] *= decay;
            memory = fma(S[i], kHead[dk], memory);
        }
        const float delta = (float(conv[vBase + tid]) - memory) * beta;
        // Pass 2: rank-one update, and read the output in the same sweep — the
        // reference reads the UPDATED state, so the two cannot be reordered.
        for (uint dk = 0; dk < Dk; ++dk) {
            const uint i = dk * Dv + tid;
            S[i] = fma(kHead[dk], delta, S[i]);
            y = fma(S[i], qHead[dk], y);
        }
    }

    // Gated norm: RMS over the head's Dv channels (MEAN plus eps), a
    // ones-centered weight, and a SIGMOID gate on z.
    float sumsq = (tid < Dv) ? y * y : 0.0f;
    sumsq = simd_sum(sumsq);
    if (lane == 0) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < simdgroups) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) shared[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= Dv) return;
    const float inv = 1.0f / sqrt(shared[0] / float(Dv) + eps);
    const uint o = h * Dv + tid;
    out[o] = half(y * inv * float(normW[tid])
                  * flashnext_gdn_sigmoid(float(zProj[o])));
}
