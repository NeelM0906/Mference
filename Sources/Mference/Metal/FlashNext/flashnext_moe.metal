#include <metal_stdlib>
using namespace metal;

// ============================================================================
// flashnext_moe — routed-expert compute for a BF16-passthrough install.
//
// The production `qwen38flashnext` install quantizes every routed expert to INT4
// affine group-64, and that path is the shipped `moe_phase1_gate_up_act_u16load`
// + `moe_phase2_down_reduce_k10` pair in `moe.metal`. The parity install carries
// every tensor at its source BF16 (the toy's `moe_intermediate_size` is 32, which
// group-64 cannot quantize at all), so the same runner has to drive a dense BF16
// expert as well — exactly the dual-dtype split `FlashNextWeightMatrix` already
// makes for the resident projections.
//
// The two paths share the argument-buffer struct (10 device pointers, matching
// `RoutedBlobs` in `moe.metal`) and `ExpertOffsets`, so a runner swaps pipelines
// without re-encoding its expert bindings.
//
// Arithmetic convention, matching the INT4 kernels: activations FP16 in memory,
// every dot product accumulated in FP32 via `simd_sum`, the weighted reduce
// seeded with the residual and summed in rank order so the accumulation order is
// the same one the shipped reduce uses.
// ============================================================================

constant constexpr uint kFlashNextRoutedBlobSlots = 10;

struct FlashNextExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

struct FlashNextRoutedBlobs {
    device const uint8_t* blob[kFlashNextRoutedBlobSlots];
};

static inline float flashnext_moe_silu(float x) {
    return x / (1.0f + exp(-x));
}

// One SIMD group per output row. `W` is [rows, N] dense BF16.
static inline float flashnext_moe_bf16_row(
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

static inline void flashnext_moe_phase1_bf16_body(
    device const FlashNextRoutedBlobs& routed,
    constant FlashNextExpertOffsets& offsets,
    device const half* x,
    device half* acts,
    uint slot,
    uint f,
    uint D,
    uint F,
    uint lane
) {
    device const uint8_t* base = routed.blob[slot];
    device const bfloat* gW = (device const bfloat*)(base + offsets.gate_W_off);
    device const bfloat* uW = (device const bfloat*)(base + offsets.up_W_off);
    const float g = flashnext_moe_bf16_row(gW, x, f, D, lane);
    const float u = flashnext_moe_bf16_row(uW, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(flashnext_moe_silu(g) * u);
}

// acts[slot * F + f] = silu(gate_f . x) * (up_f . x), for every routed slot.
kernel void flashnext_moe_phase1_gate_up_bf16(
    device const FlashNextRoutedBlobs& routed [[buffer(0)]],
    constant FlashNextExpertOffsets& offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    flashnext_moe_phase1_bf16_body(routed, offsets, x, acts,
                                   rowg / F, rowg % F, D, F, lane);
}

// The hit/miss split form: only the slots named in `active_slots` are computed,
// so a layer whose expert blobs arrive in two waves can start on the cached ones
// without waiting. Identical arithmetic, so the union of two subset dispatches is
// bit-identical to one full dispatch.
kernel void flashnext_moe_phase1_gate_up_bf16_subset(
    device const FlashNextRoutedBlobs& routed [[buffer(0)]],
    constant FlashNextExpertOffsets& offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * F) return;
    const uint slot = active_slots[rowg / F];
    if (slot >= top_k) return;
    flashnext_moe_phase1_bf16_body(routed, offsets, x, acts,
                                   slot, rowg % F, D, F, lane);
}

// y[d] = residual[d] + sum over ranks of routing_w[rank] * (down_d . acts[rank]).
//
// General in `top_k` (<= 10) rather than one kernel per width: the reduce runs
// `top_k` simdgroups and sums `partial[0..top_k)` in rank order, which is the
// same FP32 sequence the unrolled `moe_phase2_down_reduce_k{6,8,10}` kernels
// produce at their own width. Flash-Next's production width is 10 and the parity
// install's is 2, and one kernel covers both.
kernel void flashnext_moe_phase2_down_reduce_bf16(
    device const FlashNextRoutedBlobs& routed [[buffer(0)]],
    constant FlashNextExpertOffsets& offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[kFlashNextRoutedBlobSlots];
    if (d >= D) return;
    if (sg_idx >= top_k) return;

    device const uint8_t* base = routed.blob[sg_idx];
    device const bfloat* dW = (device const bfloat*)(base + offsets.down_W_off);
    const float value = flashnext_moe_bf16_row(dW, acts + sg_idx * F, d, F, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        for (uint j = 0; j < top_k; ++j) acc += partial[j];
        y[d] = half(acc);
    }
}

// ---------------------------------------------------------------------------
// The shared expert's scalar gate.
//
// `Qwen3NextSparseMoeBlock` multiplies the shared FFN's whole output by
// `sigmoid(shared_expert_gate . x)` — one `[1, hidden]` row, so one scalar per
// token. The pre-sigmoid value arrives in FP32 rather than FP16 for the same
// reason the hyper-connection gates do: rounding a pre-activation is amplified
// by the sigmoid slope, and a single float costs nothing to keep wide.
// ---------------------------------------------------------------------------
kernel void flashnext_moe_shared_gate_scale(
    device       half*  out    [[buffer(0)]],   // [count] FP16, in place
    device const float* scalar [[buffer(1)]],   // [1] FP32, pre-sigmoid
    constant     uint&  count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    const float gate = 1.0f / (1.0f + exp(-scalar[0]));
    out[gid] = half(float(out[gid]) * gate);
}
