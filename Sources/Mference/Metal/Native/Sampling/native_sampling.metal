// Native engine sampling kernels (module `native_sampling`,
// docs/NATIVE_ENGINE.md). Compiled on demand through
// MetalContext.modulePipeline; never part of the shared runtime library.
//
// Greedy choice: a two-pass argmax over one F32 logits row. The order is
// the CPU sampler's first-maximum scan: a higher value wins, equal values go
// to the lower index, NaN never wins; a NaN at index 0 chooses 0.

#include <metal_stdlib>
using namespace metal;

struct NativeArgBest {
    float value;
    uint index;
};

// a is better than b: a is not NaN, and b is NaN, or a is greater, or equal
// with a lower index.
static inline bool native_argbest_better(float av, uint ai, float bv, uint bi) {
    if (isnan(av)) { return false; }
    if (isnan(bv)) { return true; }
    return av > bv || (av == bv && ai < bi);
}

// Reduce (v, i) over one threadgroup; thread 0 of the group gets the result.
static inline NativeArgBest native_argbest_reduce(float v, uint i, uint lane, uint simd, uint simds,
                                                  threadgroup NativeArgBest *shared) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        float ov = simd_shuffle_down(v, offset);
        uint oi = simd_shuffle_down(i, offset);
        if (native_argbest_better(ov, oi, v, i)) { v = ov; i = oi; }
    }
    if (lane == 0) { shared[simd] = NativeArgBest{v, i}; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NativeArgBest best = NativeArgBest{NAN, 0xFFFFFFFFu};
    if (simd == 0) {
        if (lane < simds) { best = shared[lane]; }
        v = best.value;
        i = best.index;
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float ov = simd_shuffle_down(v, offset);
            uint oi = simd_shuffle_down(i, offset);
            if (native_argbest_better(ov, oi, v, i)) { v = ov; i = oi; }
        }
        best = NativeArgBest{v, i};
    }
    return best;
}

// Pass 1: each threadgroup reduces a strided share of the row into
// partials[group].
kernel void native_argmax_partial(device const float *logits [[buffer(0)]],
                                  device NativeArgBest *partials [[buffer(1)]],
                                  constant uint &count [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]],
                                  uint grid [[threads_per_grid]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup NativeArgBest shared[32];
    float v = NAN;
    uint i = 0xFFFFFFFFu;
    for (uint j = gid; j < count; j += grid) {
        float x = logits[j];
        if (native_argbest_better(x, j, v, i)) { v = x; i = j; }
    }
    NativeArgBest best = native_argbest_reduce(v, i, lane, simd, simds, shared);
    if (tid == 0) {
        partials[group] = best;
    }
}

// Pass 2: one threadgroup reduces the partials and writes the token id.
kernel void native_argmax_final(device const float *logits [[buffer(0)]],
                                device const NativeArgBest *partials [[buffer(1)]],
                                constant uint &groups [[buffer(2)]],
                                device uint *output [[buffer(3)]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint threads [[threads_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]],
                                uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup NativeArgBest shared[32];
    float v = NAN;
    uint i = 0xFFFFFFFFu;
    for (uint j = tid; j < groups; j += threads) {
        NativeArgBest p = partials[j];
        if (native_argbest_better(p.value, p.index, v, i)) { v = p.value; i = p.index; }
    }
    NativeArgBest best = native_argbest_reduce(v, i, lane, simd, simds, shared);
    if (tid == 0) {
        output[0] = isnan(logits[0]) ? 0u : best.index;
    }
}
