#include <metal_stdlib>
using namespace metal;

// ============================================================================
// flashnext — kernels for the `qwen38flashnext` (upstream `qwen4_exp`) axes
// that no shipped family has: low-rank hyper-connections over a 4x2560 residual
// stream, the PLE n-gram layer, and the QSA indexer.
//
// Semantics come from
// docs/superpowers/specs/2026-09-01-qwen38flashnext-runtime-design.md and are
// gated kernel by kernel against `FlashNextReferenceRunner`, the float32 CPU
// forward that passes all six golden parity gates. Where this file names a
// reference expression, it is transcribing that runner.
//
// Convention, matching the rest of the runtime: activations are FP16 in memory,
// every reduction and gate accumulates in FP32. Values that are cheap to keep
// wide and expensive to round — the pre-sigmoid mix gate, the low-rank vector
// before its SiLU, the four injection scalars — stay FP32 end to end.
//
// ---------------------------------------------------------------------------
// Decode-state contract (read from the pinned reference, not reconstructed)
//
// `transformers` `cache_utils.py` `update_conv_state(x, layer_idx, state_idx,
// conv_kernel_size = L)` — the exact commit the goldens were generated with,
// in `scratch/qwen4exp-parity-venv/`:
//
//   * `L` is the RETAINED STATE LENGTH, not the kernel width.
//   * Prefill (no previous state for that `state_idx`): returns `x` itself,
//     LEFT-PADDED WITH ZEROS to length `L` when `x` is shorter; stores the last
//     `L` columns and marks `has_previous_state`.
//   * Decode: returns `cat([stored_state(L), x], dim=-1)` — the full stored
//     context ahead of the new columns; stores the last `L` of that.
//   * `update_recurrent_state` is a plain overwrite.
//
// The zero-padding is why the PLE id-history call site pads with `eos_token_id`
// BEFORE calling: the cache would otherwise pad token ids with zeros.
//
// State lengths this family needs, in these terms: GDN conv `L = 4`, PLE conv
// `L = 9` ((kernel - 1) * dilation = 3 * 3), PLE id history `L = 2`
// (ngram_size - 1), pre-filled with `eos_token_id` rather than zero.
// `FlashNextReferenceRunner` already models all three — its gate 6
// (cached decode == its own re-prefill) is what proves it — so this is the
// contract the Metal state layouts must reproduce, not a new degree of freedom.
// ============================================================================

constant constexpr float kFlashNextGemvRowsPerThreadgroup = 8;

// ---------------------------------------------------------------------------
// BF16 GEMV
//
// The parity install (and any BF16-passthrough install) carries its projections
// unquantized, so the Flash-Next paths need a dense BF16 mat-vec next to the
// INT4 one `DequantInt4GEMV` provides. One SIMD group per output row, eight
// rows per threadgroup — the same shape as `dequant_int4_gemv_simd`.
// ---------------------------------------------------------------------------

static inline float flashnext_gemv_bf16_row(
    device const bfloat* W,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    device const bfloat* W_row = W + ulong(row) * ulong(N);
    float acc = 0.0f;
    for (uint i = lane; i < N; i += 32u) {
        acc = fma(float(W_row[i]), float(x[i]), acc);
    }
    return simd_sum(acc);
}

kernel void flashnext_gemv_bf16(
    device const bfloat* W   [[buffer(0)]],   // [M, N] BF16, row-major
    device const half*   x   [[buffer(1)]],   // [N] FP16
    device       half*   y   [[buffer(2)]],   // [M] FP16
    constant     uint&   M   [[buffer(3)]],
    constant     uint&   N   [[buffer(4)]],
    uint tg_idx  [[threadgroup_position_in_grid]],
    uint sg_idx  [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * uint(kFlashNextGemvRowsPerThreadgroup) + sg_idx;
    if (row >= M) return;
    const float acc = flashnext_gemv_bf16_row(W, x, row, N, lane);
    if (lane == 0) y[row] = half(acc);
}

// FP32-output variant. Only the store differs, so for rows inside FP16 range
// the two agree exactly before the cast.
kernel void flashnext_gemv_bf16_f32out(
    device const bfloat* W   [[buffer(0)]],
    device const half*   x   [[buffer(1)]],
    device       float*  y   [[buffer(2)]],
    constant     uint&   M   [[buffer(3)]],
    constant     uint&   N   [[buffer(4)]],
    uint tg_idx  [[threadgroup_position_in_grid]],
    uint sg_idx  [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * uint(kFlashNextGemvRowsPerThreadgroup) + sg_idx;
    if (row >= M) return;
    const float acc = flashnext_gemv_bf16_row(W, x, row, N, lane);
    if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// Hyper-connections (`Qwen4ExpTextGatedResidual`)
//
// Per sub-block, from the group-normed stream `h_n` (10240 wide):
//
//   m     = silu( W_down . h_n / hc_count )                 // 320
//   g     = sigmoid( W_up . m )                             // 10240
//   mixed = mean over the 4 streams of (g * h_n)            // 2560
//   inj   = 2 * sigmoid( W_inject . h_n / hc_count )        // 4
//   after the block: hyper += flatten(block_out * inj)
//
// Note all three `/hc_count` divisions, and that the mix and inject read the
// NORMED stream while the residual add lands on the RAW one.
// ---------------------------------------------------------------------------

static inline float flashnext_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// `m = silu(raw / divisor)`. The low-rank vector is FP32 out of the GEMV and
// FP16 going into the next one, so the SiLU is the only place it rounds.
kernel void flashnext_hc_lowrank_activation(
    device const float* raw     [[buffer(0)]],   // [count] FP32
    device       half*  out     [[buffer(1)]],   // [count] FP16
    constant     uint&  count   [[buffer(2)]],
    constant     float& divisor [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    const float v = raw[gid] / divisor;
    out[gid] = half(v / (1.0f + exp(-v)));
}

// `mixed[d] = (sum over streams j of sigmoid(up[j*H + d]) * normed[j*H + d]) / hc`.
// `up` stays FP32: it is the pre-sigmoid gate, where FP16 rounding would be
// amplified by the sigmoid slope on every one of the 10240 channels.
kernel void flashnext_hc_mix(
    device const float* up      [[buffer(0)]],   // [rows * hc * H] FP32
    device const half*  normed  [[buffer(1)]],   // [rows * hc * H] FP16
    device       half*  mixed   [[buffer(2)]],   // [rows * H] FP16
    constant     uint&  hidden  [[buffer(3)]],
    constant     uint&  hcCount [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]        // x = channel, y = row
) {
    const uint d = gid.x;
    if (d >= hidden) return;
    const uint bundle = hidden * hcCount;
    device const float* u = up     + ulong(gid.y) * ulong(bundle);
    device const half*  n = normed + ulong(gid.y) * ulong(bundle);
    float acc = 0.0f;
    for (uint j = 0; j < hcCount; ++j) {
        const uint i = j * hidden + d;
        acc = fma(flashnext_sigmoid(u[i]), float(n[i]), acc);
    }
    mixed[ulong(gid.y) * ulong(hidden) + d] = half(acc / float(hcCount));
}

// `inj[j] = 2 * sigmoid(raw[j] / divisor)`. Four values per token; kept FP32
// because they multiply the whole 2560-wide block output on the way back into
// the residual stream.
kernel void flashnext_hc_inject_gate(
    device const float* raw     [[buffer(0)]],   // [rows * hcCount] FP32
    device       float* gate    [[buffer(1)]],   // [rows * hcCount] FP32
    constant     uint&  count   [[buffer(2)]],
    constant     float& divisor [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    gate[gid] = 2.0f * flashnext_sigmoid(raw[gid] / divisor);
}

// `hyper[j*H + d] += gate[j] * block[d]`, in place on the RAW stream.
kernel void flashnext_hc_inject_accumulate(
    device       half*  hyper   [[buffer(0)]],   // [rows * hc * H] FP16, in/out
    device const half*  block   [[buffer(1)]],   // [rows * H] FP16
    device const float* gate    [[buffer(2)]],   // [rows * hcCount] FP32
    constant     uint&  hidden  [[buffer(3)]],
    constant     uint&  hcCount [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]        // x = bundle channel, y = row
) {
    const uint bundle = hidden * hcCount;
    if (gid.x >= bundle) return;
    const uint j = gid.x / hidden;
    const uint d = gid.x - j * hidden;
    device half* h = hyper + ulong(gid.y) * ulong(bundle) + gid.x;
    const float delta = gate[ulong(gid.y) * ulong(hcCount) + j]
        * float(block[ulong(gid.y) * ulong(hidden) + d]);
    *h = half(float(*h) + delta);
}

// ---------------------------------------------------------------------------
// PLE (`Qwen4ExpTextPLELayer`)
//
// The n-gram hash and the row gather are CPU work (16 int64 hashes and 16 row
// reads per token, through `PleRowPool`'s LFU cache). What lands here is the
// mixing, per token, from the gathered embedding `e` and the raw hyper stream:
//
//   k      = group_rmsnorm(W_key . e)            viewed [hc, H]
//   v      = W_value . e                                    // H
//   qn     = group_rmsnorm(hyper)                viewed [hc, H]
//   gate_s = (k_s . qn_s) / sqrt(H)                          // per stream
//   gate_s = sigmoid( sign(gate_s) * sqrt(max(|gate_s|, 1e-6)) )
//   gv     = flatten(gate_s * v)                             // hc*H
//   out    = gv + silu(depthwise_conv1d(group_rmsnorm(gv)))
//   hyper += out
//
// The projections and the three group norms reuse the kernels above; the three
// kernels here are the parts with no analogue elsewhere in the runtime.
// ---------------------------------------------------------------------------

constant constexpr uint kFlashNextPleMaxSimdGroups = 8;

// `gate_s = sigmoid(signed_sqrt((k_s . qn_s) / sqrt(H)))`, one threadgroup per
// (row, stream). The 1e-6 floor is applied to the MAGNITUDE before the square
// root, so |g'| >= 1e-3 unless the dot product is exactly zero — the sign is
// taken first and re-applied, which is not the same as sqrt of a clamped value.
kernel void flashnext_ple_stream_gate(
    device const half*  keyNormed   [[buffer(0)]],   // [rows * hc * H] FP16
    device const half*  queryNormed [[buffer(1)]],   // [rows * hc * H] FP16
    device       float* gate        [[buffer(2)]],   // [rows * hc] FP32
    constant     uint&  hidden      [[buffer(3)]],
    constant     uint&  hcCount     [[buffer(4)]],
    uint  slot             [[threadgroup_position_in_grid]],   // row * hc + stream
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kFlashNextPleMaxSimdGroups];
    const uint H = hidden;
    // The bundle is laid out [row][stream][H], so `row * hc * H + stream * H`
    // is just `slot * H` — the threadgroup index already IS the stream index.
    const ulong base = ulong(slot) * ulong(H);
    float acc = 0.0f;
    for (uint d = lid; d < H; d += lsize) {
        acc = fma(float(keyNormed[base + d]), float(queryNormed[base + d]), acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) partial[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id != 0) return;
    float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
    v = simd_sum(v);
    if (simd_lane_id != 0) return;
    float g = v / sqrt(float(H));
    const float sign = g > 0.0f ? 1.0f : (g < 0.0f ? -1.0f : 0.0f);
    g = sqrt(max(abs(g), 1e-6f)) * sign;
    gate[slot] = flashnext_sigmoid(g);
}

// `gv[j*H + d] = gate[j] * v[d]` — the 2560-wide value broadcast into all four
// streams under its own per-stream gate.
kernel void flashnext_ple_apply_gate(
    device const half*  value   [[buffer(0)]],   // [rows * H] FP16
    device const float* gate    [[buffer(1)]],   // [rows * hc] FP32
    device       half*  gated   [[buffer(2)]],   // [rows * hc * H] FP16
    constant     uint&  hidden  [[buffer(3)]],
    constant     uint&  hcCount [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]        // x = bundle channel, y = row
) {
    const uint bundle = hidden * hcCount;
    if (gid.x >= bundle) return;
    const uint j = gid.x / hidden;
    const uint d = gid.x - j * hidden;
    gated[ulong(gid.y) * ulong(bundle) + gid.x] =
        half(gate[ulong(gid.y) * ulong(hcCount) + j]
                 * float(value[ulong(gid.y) * ulong(hidden) + d]));
}

// Depthwise causal conv over the normed gated value, kernel 4, dilation
// `ngram_size` (3), accumulated into `out` — which the caller has already
// seeded with the un-normed gated value, because the reference's output is
// `gv + silu(conv(norm(gv)))`.
//
// `padded` is `[stateLength | rows]`: the caller writes the normed rows at
// offset `stateLength` so no copy is needed, and the taps step back by the
// dilation from `stateLength + row`. `stateLength` is `(kernel - 1) * dilation`
// = 9, matching the decode-state contract in this file's header.
kernel void flashnext_ple_conv(
    device const half*   padded      [[buffer(0)]],  // [(L + rows) * bundle] FP16
    device       half*   out         [[buffer(1)]],  // [rows * bundle] FP16, seeded
    device const bfloat* weight      [[buffer(2)]],  // [bundle, K] BF16
    constant     uint&   bundle      [[buffer(3)]],
    constant     uint&   kernelWidth [[buffer(4)]],
    constant     uint&   dilation    [[buffer(5)]],
    constant     uint&   stateLength [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]            // x = channel, y = row
) {
    if (gid.x >= bundle) return;
    float acc = 0.0f;
    for (uint j = 0; j < kernelWidth; ++j) {
        const uint row = gid.y + stateLength - (kernelWidth - 1u - j) * dilation;
        acc = fma(float(weight[gid.x * kernelWidth + j]),
                  float(padded[ulong(row) * ulong(bundle) + gid.x]), acc);
    }
    const ulong i = ulong(gid.y) * ulong(bundle) + gid.x;
    out[i] = half(float(out[i]) + acc / (1.0f + exp(-acc)));
}

// The embedding tile: `hidden_states = embed(input_ids).repeat(1, 1, hc_count)`
// is a TILE, so stream j is an exact copy of the 2560-wide embedding row, not
// an interleave. One dispatch instead of `hc_count` blits.
kernel void flashnext_hc_tile_embedding(
    device const half* embedding [[buffer(0)]],   // [rows * H] FP16
    device       half* hyper     [[buffer(1)]],   // [rows * hc * H] FP16
    constant     uint& hidden    [[buffer(2)]],
    constant     uint& hcCount   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]         // x = bundle channel, y = row
) {
    const uint bundle = hidden * hcCount;
    if (gid.x >= bundle) return;
    const uint d = gid.x % hidden;
    hyper[ulong(gid.y) * ulong(bundle) + gid.x] =
        embedding[ulong(gid.y) * ulong(hidden) + d];
}

// ---------------------------------------------------------------------------
// BF16 embedding row gather.
//
// The production install quantizes `embed_tokens.weight` to INT4 affine g64 and
// the shipped `embed_lookup_int4` serves it. The parity install carries it as
// dense BF16, so the runner needs the other half of the same dtype split it
// already makes for every projection. There is no output scale: this family does
// NOT scale the embedding by sqrt(hidden) — `hidden_states = embed(ids)` and then
// the hyper-connection tile, nothing else.
// ---------------------------------------------------------------------------
kernel void flashnext_embed_row_bf16(
    device const bfloat* table  [[buffer(0)]],   // [vocab, hidden] BF16
    device       half*   out    [[buffer(1)]],   // [hidden] FP16
    constant     uint&   row    [[buffer(2)]],
    constant     uint&   hidden [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= hidden) return;
    out[gid] = half(float(table[ulong(row) * ulong(hidden) + gid]));
}
