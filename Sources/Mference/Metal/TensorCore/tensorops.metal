#include <metal_stdlib>
using namespace metal;

#ifndef MFERENCE_GEMMA_SOURCE_FP16
#define MFERENCE_GEMMA_SOURCE_FP16
// Preserve the source activation boundary before learned scaling. The original
// Gemma checkpoint keeps its existing fused arithmetic unless explicitly set.
constant bool FC_GEMMA_SOURCE_FP16 [[function_constant(110)]];
constant bool kGemmaSourceFP16 = is_function_constant_defined(FC_GEMMA_SOURCE_FP16)
    ? FC_GEMMA_SOURCE_FP16 : false;
static inline half gemma_weighted_norm(float x, float inv, float weight) {
    return kGemmaSourceFP16 ? half(x * inv) * half(weight) : half(x * inv * weight);
}
static inline half gemma_scaled_embedding(float value, float scale) {
    return kGemmaSourceFP16 ? half(value) * half(scale) : half(value * scale);
}

// MLX's quantized GEMV forms its affine-bias input sum in FP16 quads.
static inline float gemma_source_quad_sum(half4 x) {
    const half a = x.x + x.y;
    const half b = a + x.z;
    return float(half(b + x.w));
}
static inline half gemma_source_geglu(float gate_value, float up_value) {
    // Gate/up projections are FP16 tensors in the source, even when fused.
    volatile half gate = half(gate_value);
    volatile half up = half(up_value);
    // Preserve source half stores across fast-math contraction, particularly
    // 1 + tanh(x): its rounded negative tail is exactly zero in the source.
    volatile half cube = half(float(gate) * float(gate) * float(gate));
    volatile half term = half(0.044715f) * cube;
    volatile half sum = gate + term;
    volatile half inner = half(0.7978845608028654f) * sum;
    // FP16 tanh has already rounded to +/-1 at these bounds. Keep that
    // exact source result while avoiding fast-tanh's positive exp overflow.
    volatile half curve = half(tanh(clamp(float(inner), -20.0f, 20.0f)));
    volatile half shifted = half(1.0h + curve);
    volatile half scaled = half(0.5h * gate);
    volatile half activation = scaled * shifted;
    return half(activation * up);
}
static inline float gemma_source_bias_correction(
    device const half* x, device const bfloat* biases, uint width, uint group_size
) {
    float correction = 0.0f;
    for (uint k = 0; k < width; k += 4u) {
        const half4 quad(x[k], x[k + 1u], x[k + 2u], x[k + 3u]);
        const float exact = float(quad.x) + float(quad.y) + float(quad.z) + float(quad.w);
        correction = fma(float(biases[k / group_size]),
                        gemma_source_quad_sum(quad) - exact, correction);
    }
    return correction;
}
#endif

#ifndef MFERENCE_GEMMA_SOURCE_PROJECTION
#define MFERENCE_GEMMA_SOURCE_PROJECTION
// QAT's source GEMV rounds the input sum in half quads, accumulates four
// products at a time, and adds each complete affine sub-result. Callers of
// this path compile this module with safe math: fast-math compilation does
// not preserve the source's FP16 rounding boundaries.
static inline float gemma_source_projection_row(
    device const uint8_t* weights,
    device const bfloat* scales,
    device const bfloat* biases,
    device const half* x,
    uint width, uint group_size, bool fast_shape, uint lane
) {
    device const ushort* packed_weights = (device const ushort*)weights;
    const uint values = fast_shape ? 16u : 8u;
    float result = 0.0f;
    for (uint base = lane * values; base < width; base += 32u * values) {
        float sum = 0.0f, dot = 0.0f;
        for (uint i = 0; i < values; i += 4u) {
            const uint k = base + i;
            const ushort packed = packed_weights[k / 4u];
            sum += x[k] + x[k + 1u] + x[k + 2u] + x[k + 3u];
            dot += float(x[k]) * (packed & 15u)
                + (float(x[k + 1u]) / 16.0f) * (packed & 240u)
                + (float(x[k + 2u]) / 256.0f) * (packed & 3840u)
                + (float(x[k + 3u]) / 4096.0f) * (packed & 61440u);
        }
        const uint group = base / group_size;
        result += float(half(scales[group])) * dot + sum * float(half(biases[group]));
    }
    return simd_sum(result);
}
#endif

// QAT requires the source GEMV's affine accumulation even when query rows
// are batched. The normal MPP tensor product uses a different reduction tree.
// Keep the query/output tile dispatch, and use one SIMD per output row for
// this checkpoint; all other profiles retain the tensor operation below.
template <typename OutT>
static inline void gemma_qat_affine_projection_tile(
    device const uint8_t* weights, device const bfloat* scales,
    device const bfloat* biases, device const half* x, device OutT* output,
    uint M, uint N, uint K, uint groupSize, uint3 grid, uint tid, uint threads
) {
    const uint lane = tid % 32u, simd = tid / 32u;
    const uint groups = K / groupSize;
    for (uint localM = 0; localM < 64u; ++localM) {
        const uint m = grid.y * 64u + localM;
        if (m >= M) break;
        for (uint localN = simd; localN < 32u; localN += threads / 32u) {
            const uint n = grid.x * 32u + localN;
            if (n >= N) continue;
            const float result = gemma_source_projection_row(
                weights + n * (K / 2u), scales + n * groups, biases + n * groups,
                x + m * K, K, groupSize, N % 8u == 0u && K % 512u == 0u, lane);
            if (lane == 0u) output[m * N + n] = OutT(result);
        }
    }
}

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

#ifndef MFERENCE_AFFINE_GROUP_SIZE
#define MFERENCE_AFFINE_GROUP_SIZE
constant uint FC_AFFINE_GROUP_SIZE [[function_constant(108)]];
constant uint kAffineGroupSize = is_function_constant_defined(FC_AFFINE_GROUP_SIZE)
    ? FC_AFFINE_GROUP_SIZE : 64u;
#endif
constant constexpr int kMPPAffineTileM = 64;
constant constexpr int kMPPAffineTileN = 32;
constant constexpr int kMPPAffineTileK = 64;

kernel void mpp_prefill_affine_threadgroup_f16(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device half* activations            [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    if (kGemmaSourceFP16) {
        gemma_qat_affine_projection_tile(packedWeights, scales, biases,
            activations, output, M, N, K, kAffineGroupSize, tgid, lid3.x, threads3.x);
        return;
    }
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineTileM, kMPPAffineTileN, kMPPAffineTileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[kMPPAffineTileN * kMPPAffineTileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, kMPPAffineTileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(
        0,
        int32_t(tgid.y) * kMPPAffineTileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K / 2u;
    const uint groupsPerRow = K / kAffineGroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < K / uint(kMPPAffineTileK); ++group) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineTileN * kMPPAffineTileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineTileK);
            const uint localK = linear % uint(kMPPAffineTileK);
            const uint globalN = tgid.x * uint(kMPPAffineTileN) + localN;
            if (globalN < N) {
                const uint globalK = group * uint(kMPPAffineTileK) + localK;
                const uint8_t packed = packedWeights[globalN * rowBytes + (globalK >> 1)];
                const uint q = (globalK & 1u) == 0u
                    ? uint(packed & 0x0fu)
                    : uint(packed >> 4);
                const uint scaleIndex = globalN * groupsPerRow + globalK / kAffineGroupSize;
                const float scale = float(scales[scaleIndex]);
                const float bias = float(biases[scaleIndex]);
                weightTile[linear] = half(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor groupA(
            activations + group * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(
            0,
            int32_t(tgid.y) * kMPPAffineTileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineTileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineTileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            float value = accumulator[element];
            output[globalM * N + globalN] = half(value);
        }
    }
}

// Same cooperative INT4 affine product as the FP16-output path, retaining the
// FP32 accumulator for continuous gates that immediately feed SiLU/sigmoid.
kernel void mpp_prefill_affine_threadgroup_f32(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device half* activations            [[buffer(3)]],
    device float* output                [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    if (kGemmaSourceFP16) {
        gemma_qat_affine_projection_tile(packedWeights, scales, biases,
            activations, output, M, N, K, kAffineGroupSize, tgid, lid3.x, threads3.x);
        return;
    }
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineTileM, kMPPAffineTileN, kMPPAffineTileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    using device_half_tensor = tensor<device half,
        dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half,
        dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[kMPPAffineTileN * kMPPAffineTileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, kMPPAffineTileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(0, int32_t(tgid.y) * kMPPAffineTileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K / 2u;
    const uint groupsPerRow = K / kAffineGroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < K / uint(kMPPAffineTileK); ++group) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineTileN * kMPPAffineTileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineTileK);
            const uint localK = linear % uint(kMPPAffineTileK);
            const uint globalN = tgid.x * uint(kMPPAffineTileN) + localN;
            if (globalN < N) {
                const uint globalK = group * uint(kMPPAffineTileK) + localK;
                const uint8_t packed = packedWeights[
                    globalN * rowBytes + (globalK >> 1)];
                const uint q = (globalK & 1u) == 0u
                    ? uint(packed & 0x0fu) : uint(packed >> 4);
                const uint scaleIndex = globalN * groupsPerRow + globalK / kAffineGroupSize;
                weightTile[linear] = half(fma(
                    float(q),
                    float(scales[scaleIndex]),
                    float(biases[scaleIndex])));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device_half_tensor groupA(
            activations + group * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(0, int32_t(tgid.y) * kMPPAffineTileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineTileN)
            + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineTileM)
            + uint(position[1]);
        if (globalM < M && globalN < N) {
            float value = accumulator[element];
            output[globalM * N + globalN] = value;
        }
    }
}

constant constexpr int kMPPGroupedM = 64;
constant constexpr int kMPPGroupedN = 32;
constant constexpr int kMPPGroupedK = 64;
constant constexpr uint kMPPGroupedMaxExperts = 16;

struct MPPGroupedPair {
    uint token;
    uint expert;
    uint rank;
    uint weight_bits_and_reserved;
};

struct MPPGroupedGroup {
    uint expert;
    uint pair_start;
    uint pair_count;
};

struct MPPGroupedExperts {
    device const uint8_t* blob[kMPPGroupedMaxExperts];
};

struct MPPGroupedParams {
    uint pair_start;
    uint pair_count;
    uint D;
    uint F;
    uint top_k;
    uint hidden_stride;
    uint live_expert_count;
    uint local_expert_0;
    uint local_expert_1;
    uint local_expert_2;
    uint local_expert_3;
    uint local_expert_4;
    uint local_expert_5;
    uint local_expert_6;
    uint local_expert_7;
    uint local_expert_8;
    uint local_expert_9;
    uint local_expert_10;
    uint local_expert_11;
    uint local_expert_12;
    uint local_expert_13;
    uint local_expert_14;
    uint local_expert_15;
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
    uint resident_expert_stride;
};

static inline uint mpp_grouped_local_expert(
    constant MPPGroupedParams& p, uint slot
) {
    switch (slot) {
        case 0: return p.local_expert_0;
        case 1: return p.local_expert_1;
        case 2: return p.local_expert_2;
        case 3: return p.local_expert_3;
        case 4: return p.local_expert_4;
        case 5: return p.local_expert_5;
        case 6: return p.local_expert_6;
        case 7: return p.local_expert_7;
        case 8: return p.local_expert_8;
        case 9: return p.local_expert_9;
        case 10: return p.local_expert_10;
        case 11: return p.local_expert_11;
        case 12: return p.local_expert_12;
        case 13: return p.local_expert_13;
        case 14: return p.local_expert_14;
        default: return p.local_expert_15;
    }
}

static inline float mpp_grouped_silu(float x) {
    return x / (1.0f + exp(-x));
}

// Unset/false = SiLU (Flash-Next), true = gelu_pytorch_tanh with the same
// clamp as prefill.metal's Gemma activation.
constant bool FC_MPP_GROUPED_GELU [[function_constant(112)]];
static inline float mpp_grouped_activation(float x) {
    if (is_function_constant_defined(FC_MPP_GROUPED_GELU) && FC_MPP_GROUPED_GELU) {
        const float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3);
        inner = clamp(inner, -20.0f, 20.0f);
        return 0.5f * x * (1.0f + tanh(inner));
    }
    return mpp_grouped_silu(x);
}

kernel void mpp_grouped_routed_moe_phase1(
    device const half* hidden                 [[buffer(0)]],
    device const MPPGroupedPair* pairs        [[buffer(1)]],
    device const MPPGroupedGroup* groups      [[buffer(2)]],
    device half* activation                   [[buffer(3)]],
    device half* route_partials               [[buffer(4)]],
    device const MPPGroupedExperts& experts   [[buffer(5)]],
    constant MPPGroupedParams& p              [[buffer(6)]],
    device const uint8_t* resident_slab       [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPGroupedM, kMPPGroupedN, kMPPGroupedK, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    using tg_half_tensor = tensor<threadgroup half,
        dextents<int32_t, 2>, tensor_inline>;

    if (tgid.z >= p.pair_count) return;
    const MPPGroupedGroup group = groups[p.pair_start + tgid.z];
    const uint row_base = tgid.y * uint(kMPPGroupedM);
    if (row_base >= group.pair_count) return;
    uint local_slot = kMPPGroupedMaxExperts;
    if (p.live_expert_count != 0u) {
        for (uint slot = 0; slot < p.live_expert_count; ++slot) {
            if (mpp_grouped_local_expert(p, slot) == group.expert) {
                local_slot = slot;
                break;
            }
        }
        if (local_slot >= p.live_expert_count) return;
    }

    const uint lid = lid3.x;
    const uint threads = threads3.x;
    const uint n_base = tgid.x * uint(kMPPGroupedN);
    const bool resident = p.live_expert_count == 0u;
    const uint tile_pair_start = resident ? 0u : groups[p.pair_start].pair_start;
    device const uint8_t* expert = resident
        ? resident_slab + ulong(group.expert) * ulong(p.resident_expert_stride)
        : experts.blob[local_slot];
    device const uint8_t* gate_w = expert + p.gate_W_off;
    device const bfloat* gate_s =
        reinterpret_cast<device const bfloat*>(expert + p.gate_s_off);
    device const bfloat* gate_b =
        reinterpret_cast<device const bfloat*>(expert + p.gate_b_off);
    device const uint8_t* up_w = expert + p.up_W_off;
    device const bfloat* up_s =
        reinterpret_cast<device const bfloat*>(expert + p.up_s_off);
    device const bfloat* up_b =
        reinterpret_cast<device const bfloat*>(expert + p.up_b_off);

    threadgroup half a_tile[kMPPGroupedM * kMPPGroupedK];
    threadgroup half gate_tile[kMPPGroupedN * kMPPGroupedK];
    threadgroup half up_tile[kMPPGroupedN * kMPPGroupedK];
    tg_half_tensor tile_a(a_tile,
        dextents<int32_t, 2>(kMPPGroupedK, kMPPGroupedM),
        array<int32_t, 2>({1, kMPPGroupedK}));
    tg_half_tensor tile_gate(gate_tile,
        dextents<int32_t, 2>(kMPPGroupedK, kMPPGroupedN),
        array<int32_t, 2>({1, kMPPGroupedK}));
    tg_half_tensor tile_up(up_tile,
        dextents<int32_t, 2>(kMPPGroupedK, kMPPGroupedN),
        array<int32_t, 2>({1, kMPPGroupedK}));
    auto gate_acc = operation.get_destination_cooperative_tensor<
        decltype(tile_a), decltype(tile_gate), float>();
    auto up_acc = operation.get_destination_cooperative_tensor<
        decltype(tile_a), decltype(tile_up), float>();
    auto product = operation.get_destination_cooperative_tensor<
        decltype(tile_a), decltype(tile_gate), float>();
    for (int i = 0; i < gate_acc.get_capacity(); ++i) {
        gate_acc[i] = 0.0f;
        up_acc[i] = 0.0f;
    }

    const uint row_bytes = p.D / 2u;
    // Column tiles stay 64 wide; scales and biases follow the checkpoint's
    // storage group (the same index when that is 64).
    const uint groups_per_row = p.D / uint(kMPPGroupedK);
    const uint scale_groups = p.D / kAffineGroupSize;
    for (uint kg = 0; kg < groups_per_row; ++kg) {
        for (uint linear = lid; linear < uint(kMPPGroupedM * kMPPGroupedK);
             linear += threads) {
            const uint m = linear / uint(kMPPGroupedK);
            const uint k = linear % uint(kMPPGroupedK);
            const uint pair_row = row_base + m;
            if (pair_row < group.pair_count) {
                const MPPGroupedPair pair = pairs[group.pair_start + pair_row];
                a_tile[linear] = hidden[pair.token * p.hidden_stride
                    + kg * uint(kMPPGroupedK) + k];
            } else {
                a_tile[linear] = half(0.0f);
            }
        }
        for (uint linear = lid; linear < uint(kMPPGroupedN * kMPPGroupedK);
             linear += threads) {
            const uint n = linear / uint(kMPPGroupedK);
            const uint k = linear % uint(kMPPGroupedK);
            const uint global_n = n_base + n;
            if (global_n < p.F) {
                const uint global_k = kg * uint(kMPPGroupedK) + k;
                const uint scale_index = global_n * scale_groups + global_k / kAffineGroupSize;
                const uint8_t gpacked = gate_w[global_n * row_bytes
                    + (global_k >> 1)];
                const uint8_t upacked = up_w[global_n * row_bytes
                    + (global_k >> 1)];
                const uint gq = (global_k & 1u) == 0u
                    ? uint(gpacked & 0x0fu) : uint(gpacked >> 4);
                const uint uq = (global_k & 1u) == 0u
                    ? uint(upacked & 0x0fu) : uint(upacked >> 4);
                gate_tile[linear] = half(fma(float(gq),
                    float(gate_s[scale_index]), float(gate_b[scale_index])));
                up_tile[linear] = half(fma(float(uq),
                    float(up_s[scale_index]), float(up_b[scale_index])));
            } else {
                gate_tile[linear] = half(0.0f);
                up_tile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < product.get_capacity(); ++i) product[i] = 0.0f;
        operation.run(tile_a, tile_gate, product);
        for (int i = 0; i < gate_acc.get_capacity(); ++i) gate_acc[i] += product[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < product.get_capacity(); ++i) product[i] = 0.0f;
        operation.run(tile_a, tile_up, product);
        for (int i = 0; i < up_acc.get_capacity(); ++i) up_acc[i] += product[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < gate_acc.get_capacity(); ++i) {
        if (!gate_acc.is_valid_element(i)) continue;
        const auto position = gate_acc.get_multidimensional_index(i);
        const uint n = n_base + uint(position[0]);
        const uint m = row_base + uint(position[1]);
        if (n < p.F && m < group.pair_count) {
            const uint local_pair = group.pair_start + m - tile_pair_start;
            activation[local_pair * p.F + n] =
                half(mpp_grouped_activation(gate_acc[i]) * up_acc[i]);
        }
    }
}

kernel void mpp_grouped_routed_moe_down(
    device const half* hidden                 [[buffer(0)]],
    device const MPPGroupedPair* pairs        [[buffer(1)]],
    device const MPPGroupedGroup* groups      [[buffer(2)]],
    device const half* activation             [[buffer(3)]],
    device half* route_partials               [[buffer(4)]],
    device const MPPGroupedExperts& experts   [[buffer(5)]],
    constant MPPGroupedParams& p              [[buffer(6)]],
    device const uint8_t* resident_slab       [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPGroupedM, kMPPGroupedN, kMPPGroupedK, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    using tg_half_tensor = tensor<threadgroup half,
        dextents<int32_t, 2>, tensor_inline>;

    if (tgid.z >= p.pair_count) return;
    const MPPGroupedGroup group = groups[p.pair_start + tgid.z];
    const uint row_base = tgid.y * uint(kMPPGroupedM);
    if (row_base >= group.pair_count) return;
    uint local_slot = kMPPGroupedMaxExperts;
    if (p.live_expert_count != 0u) {
        for (uint slot = 0; slot < p.live_expert_count; ++slot) {
            if (mpp_grouped_local_expert(p, slot) == group.expert) {
                local_slot = slot;
                break;
            }
        }
        if (local_slot >= p.live_expert_count) return;
    }

    const uint lid = lid3.x;
    const uint threads = threads3.x;
    const uint n_base = tgid.x * uint(kMPPGroupedN);
    const bool resident = p.live_expert_count == 0u;
    const uint tile_pair_start = resident ? 0u : groups[p.pair_start].pair_start;
    device const uint8_t* expert = resident
        ? resident_slab + ulong(group.expert) * ulong(p.resident_expert_stride)
        : experts.blob[local_slot];
    device const uint8_t* down_w = expert + p.down_W_off;
    device const bfloat* down_s =
        reinterpret_cast<device const bfloat*>(expert + p.down_s_off);
    device const bfloat* down_b =
        reinterpret_cast<device const bfloat*>(expert + p.down_b_off);

    threadgroup half a_tile[kMPPGroupedM * kMPPGroupedK];
    threadgroup half weight_tile[kMPPGroupedN * kMPPGroupedK];
    tg_half_tensor tile_a(a_tile,
        dextents<int32_t, 2>(kMPPGroupedK, kMPPGroupedM),
        array<int32_t, 2>({1, kMPPGroupedK}));
    tg_half_tensor tile_b(weight_tile,
        dextents<int32_t, 2>(kMPPGroupedK, kMPPGroupedN),
        array<int32_t, 2>({1, kMPPGroupedK}));
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(tile_a), decltype(tile_b), float>();
    auto product = operation.get_destination_cooperative_tensor<
        decltype(tile_a), decltype(tile_b), float>();
    for (int i = 0; i < accumulator.get_capacity(); ++i) accumulator[i] = 0.0f;

    const uint row_bytes = p.F / 2u;
    const uint groups_per_row = p.F / uint(kMPPGroupedK);
    const uint scale_groups = p.F / kAffineGroupSize;
    for (uint kg = 0; kg < groups_per_row; ++kg) {
        for (uint linear = lid; linear < uint(kMPPGroupedM * kMPPGroupedK);
             linear += threads) {
            const uint m = linear / uint(kMPPGroupedK);
            const uint k = linear % uint(kMPPGroupedK);
            const uint pair_row = row_base + m;
            if (pair_row < group.pair_count) {
                const uint local_pair = group.pair_start + pair_row - tile_pair_start;
                a_tile[linear] = activation[local_pair * p.F
                    + kg * uint(kMPPGroupedK) + k];
            } else {
                a_tile[linear] = half(0.0f);
            }
        }
        for (uint linear = lid; linear < uint(kMPPGroupedN * kMPPGroupedK);
             linear += threads) {
            const uint n = linear / uint(kMPPGroupedK);
            const uint k = linear % uint(kMPPGroupedK);
            const uint global_n = n_base + n;
            if (global_n < p.D) {
                const uint global_k = kg * uint(kMPPGroupedK) + k;
                const uint8_t packed = down_w[global_n * row_bytes
                    + (global_k >> 1)];
                const uint q = (global_k & 1u) == 0u
                    ? uint(packed & 0x0fu) : uint(packed >> 4);
                const uint scale_index = global_n * scale_groups + global_k / kAffineGroupSize;
                weight_tile[linear] = half(fma(float(q),
                    float(down_s[scale_index]), float(down_b[scale_index])));
            } else {
                weight_tile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < product.get_capacity(); ++i) product[i] = 0.0f;
        operation.run(tile_a, tile_b, product);
        for (int i = 0; i < accumulator.get_capacity(); ++i) accumulator[i] += product[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < accumulator.get_capacity(); ++i) {
        if (!accumulator.is_valid_element(i)) continue;
        const auto position = accumulator.get_multidimensional_index(i);
        const uint n = n_base + uint(position[0]);
        const uint m = row_base + uint(position[1]);
        if (n < p.D && m < group.pair_count) {
            const MPPGroupedPair pair = pairs[group.pair_start + m];
            route_partials[(pair.token * p.top_k + pair.rank) * p.D + n] =
                half(accumulator[i]);
        }
    }
}

// Inkling's production attention shape is GQA 32/8 with a 128-wide head and
// a learned 16-wide relative-position projection. Eight consecutive query
// tokens share each QK/PV tile. This is deliberately separate from the generic
// 512-wide prefill attention kernel: it keeps Inkling's relative bias and
// per-query log scaling inside the online softmax while using TensorOps for the
// two matrix products that dominate long-context attention.
constant constexpr int kInklingAttentionQueries = 8;
constant constexpr int kInklingAttentionKeys = 64;
constant constexpr int kInklingAttentionHeadDim = 128;
constant constexpr int kInklingAttentionDRel = 16;

kernel void inkling_attention_prefill_tensorops(
    device const half*   Q              [[buffer(0)]],
    device const half*   K              [[buffer(1)]],
    device const half*   V              [[buffer(2)]],
    device const half*   rel            [[buffer(3)]],
    device const bfloat* proj           [[buffer(4)]],
    device       half*   output         [[buffer(5)]],
    constant     uint&   headDim        [[buffer(6)]],
    constant     uint&   numQHeads      [[buffer(7)]],
    constant     uint&   numKVHeads     [[buffer(8)]],
    constant     uint&   startPosition  [[buffer(9)]],
    constant     uint&   queryCount     [[buffer(10)]],
    constant     uint&   slidingWindow  [[buffer(11)]],
    constant     uint&   relExtent      [[buffer(12)]],
    constant     uint&   dRel           [[buffer(13)]],
    constant     uint&   ringCapacity   [[buffer(14)]],
    constant     uint&   logFloor       [[buffer(15)]],
    constant     float&  scale          [[buffer(16)]],
    constant     float&  logAlpha       [[buffer(17)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]) {
    // GPUCompiler 32023 requires stage-in attribute declarations to be all
    // scalar or all vector of one width; keep the all-uint3 convention.
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    constexpr auto qkDescriptor = matmul2d_descriptor(
        kInklingAttentionQueries,
        kInklingAttentionKeys,
        kInklingAttentionHeadDim,
        false, true, false);
    constexpr auto pvDescriptor = matmul2d_descriptor(
        kInklingAttentionQueries,
        kInklingAttentionHeadDim,
        kInklingAttentionKeys,
        false, false, false);
    matmul2d<qkDescriptor, execution_simdgroups<4>> qkOperation;
    matmul2d<pvDescriptor, execution_simdgroups<4>> pvOperation;

    using deviceHalfTensor =
        tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroupHalfTensor =
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroupFloatTensor =
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

    // The Swift dispatcher only selects this kernel for the fixed Inkling
    // shape. Keep the checks here as a hard safety net because the cooperative
    // tensor descriptors are compile-time.
    if (headDim != uint(kInklingAttentionHeadDim)
        || numQHeads != 32u
        || numKVHeads != 8u
        || dRel != uint(kInklingAttentionDRel)) {
        return;
    }

    threadgroup half queryTile[
        kInklingAttentionQueries * kInklingAttentionHeadDim];
    threadgroup half relativeTile[
        kInklingAttentionQueries * kInklingAttentionDRel];
    threadgroup float scoreTile[
        kInklingAttentionQueries * kInklingAttentionKeys];
    threadgroup float weightTile[
        kInklingAttentionQueries * kInklingAttentionKeys];
    threadgroup float rowMax[kInklingAttentionQueries];
    threadgroup float rowSum[kInklingAttentionQueries];
    threadgroup float rowOldScale[kInklingAttentionQueries];

    const uint queryStart = tgid.x * uint(kInklingAttentionQueries);
    const uint queryHead = tgid.y;
    if (queryStart >= queryCount || queryHead >= numQHeads) return;
    const uint validRows = min(
        uint(kInklingAttentionQueries), queryCount - queryStart);
    const uint kvHead = queryHead / (numQHeads / numKVHeads);
    const uint kvStride = numKVHeads * headDim;
    const uint qStride = numQHeads * headDim;

    for (uint linear = lid;
         linear < uint(kInklingAttentionQueries * kInklingAttentionHeadDim);
         linear += threads) {
        const uint row = linear / uint(kInklingAttentionHeadDim);
        const uint d = linear % uint(kInklingAttentionHeadDim);
        queryTile[linear] = row < validRows
            ? Q[(queryStart + row) * qStride + queryHead * headDim + d]
            : half(0.0f);
    }
    for (uint linear = lid;
         linear < uint(kInklingAttentionQueries * kInklingAttentionDRel);
         linear += threads) {
        const uint row = linear / uint(kInklingAttentionDRel);
        const uint d = linear % uint(kInklingAttentionDRel);
        relativeTile[linear] = row < validRows
            ? rel[((queryStart + row) * numQHeads + queryHead) * dRel + d]
            : half(0.0f);
    }
    if (lid < uint(kInklingAttentionQueries)) {
        rowMax[lid] = -INFINITY;
        rowSum[lid] = 0.0f;
        rowOldScale[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroupHalfTensor queryTensor(
        queryTile,
        dextents<int32_t, 2>(
            kInklingAttentionHeadDim, kInklingAttentionQueries),
        array<int32_t, 2>({1, kInklingAttentionHeadDim}));
    threadgroupFloatTensor weightTensor(
        weightTile,
        dextents<int32_t, 2>(
            kInklingAttentionKeys, kInklingAttentionQueries),
        array<int32_t, 2>({1, kInklingAttentionKeys}));
    const uint cacheCount = ringCapacity != 0u
        ? ringCapacity : startPosition + queryCount;
    // The tensor handle type is non-const; the kernel only reads through
    // these views, so shedding the const qualifier is safe.
    deviceHalfTensor keyTensor(
        (device half *)(K + kvHead * headDim),
        dextents<int32_t, 2>(
            int32_t(headDim), int32_t(cacheCount)),
        array<int32_t, 2>({1, int32_t(kvStride)}));
    deviceHalfTensor valueTensor(
        (device half *)(V + kvHead * headDim),
        dextents<int32_t, 2>(
            int32_t(headDim), int32_t(cacheCount)),
        array<int32_t, 2>({1, int32_t(kvStride)}));

    auto querySlice = queryTensor.slice(0, 0);
    auto firstValueSlice = valueTensor.slice(0, 0);
    auto outputAccumulator =
        pvOperation.get_destination_cooperative_tensor<
            decltype(weightTensor), decltype(firstValueSlice), float>();
    for (int element = 0;
         element < outputAccumulator.get_capacity(); ++element) {
        if (outputAccumulator.is_valid_element(element)) {
            outputAccumulator[element] = 0.0f;
        }
    }

    const uint firstQueryPosition = startPosition + queryStart;
    const uint firstKey = slidingWindow != 0u
        && firstQueryPosition + 1u > slidingWindow
        ? firstQueryPosition + 1u - slidingWindow
        : 0u;
    const uint lastKey = startPosition + queryStart + validRows;
    // A logical sliding-window range occupies at most two physical spans in
    // the KV ring. Never let a cooperative tile cross the ring boundary:
    // invalid columns are masked to zero, then the next iteration resumes at
    // physical slot zero without staging or copying K/V.
    for (uint keyStart = firstKey; keyStart < lastKey;) {
        const uint physicalStart = ringCapacity != 0u
            ? keyStart % ringCapacity : keyStart;
        uint tileCount = min(
            uint(kInklingAttentionKeys), lastKey - keyStart);
        if (ringCapacity != 0u) {
            tileCount = min(tileCount, ringCapacity - physicalStart);
        }
        auto keySlice = keyTensor.slice(0, int32_t(physicalStart));
        auto scoreProduct =
            qkOperation.get_destination_cooperative_tensor<
                decltype(querySlice), decltype(keySlice), float>();
        for (int element = 0;
             element < scoreProduct.get_capacity(); ++element) {
            if (scoreProduct.is_valid_element(element)) {
                scoreProduct[element] = 0.0f;
            }
        }
        qkOperation.run(querySlice, keySlice, scoreProduct);
        for (int element = 0;
             element < scoreProduct.get_capacity(); ++element) {
            if (!scoreProduct.is_valid_element(element)) continue;
            const auto position =
                scoreProduct.get_multidimensional_index(element);
            const uint keyColumn = uint(position[0]);
            const uint row = uint(position[1]);
            scoreTile[row * uint(kInklingAttentionKeys) + keyColumn] =
                scoreProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < uint(kInklingAttentionQueries)) {
            const uint row = lid;
            const bool validRow = row < validRows;
            const uint queryPosition = startPosition + queryStart + row;
            const uint rowFirst = validRow && slidingWindow != 0u
                && queryPosition + 1u > slidingWindow
                ? queryPosition + 1u - slidingWindow
                : 0u;
            const uint rowLast = validRow ? queryPosition + 1u : 0u;
            float tau = 1.0f;
            if (validRow && slidingWindow == 0u && logFloor > 0u
                && rowLast > logFloor) {
                tau += logAlpha * log(float(rowLast) / float(logFloor));
            }

            float tileMax = -INFINITY;
            for (uint keyColumn = 0u;
                 keyColumn < uint(kInklingAttentionKeys); ++keyColumn) {
                if (keyColumn >= tileCount) continue;
                const uint key = keyStart + keyColumn;
                if (key < rowFirst || key >= rowLast) continue;
                const uint distance = queryPosition - key;
                float bias = 0.0f;
                if (distance < relExtent) {
                    for (uint d = 0u; d < uint(kInklingAttentionDRel); ++d) {
                        bias = fma(
                            float(relativeTile[
                                row * uint(kInklingAttentionDRel) + d]),
                            float(proj[d * relExtent + distance]),
                            bias);
                    }
                }
                const uint index =
                    row * uint(kInklingAttentionKeys) + keyColumn;
                const float logit = tau * fma(scoreTile[index], scale, bias);
                scoreTile[index] = logit;
                tileMax = max(tileMax, logit);
            }
            const float nextMax = max(rowMax[row], tileMax);
            const float oldScale = rowSum[row] > 0.0f
                ? fast::exp(rowMax[row] - nextMax)
                : 0.0f;
            float tileSum = 0.0f;
            for (uint keyColumn = 0u;
                 keyColumn < uint(kInklingAttentionKeys); ++keyColumn) {
                const uint key = keyStart + keyColumn;
                const bool visible = validRow
                    && keyColumn < tileCount
                    && key >= rowFirst && key < rowLast;
                const float weight = visible
                    ? fast::exp(scoreTile[
                        row * uint(kInklingAttentionKeys) + keyColumn] - nextMax)
                    : 0.0f;
                weightTile[
                    row * uint(kInklingAttentionKeys) + keyColumn] = weight;
                tileSum += weight;
            }
            rowOldScale[row] = oldScale;
            rowSum[row] = rowSum[row] * oldScale + tileSum;
            rowMax[row] = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto valueSlice = valueTensor.slice(0, int32_t(physicalStart));
        auto outputProduct =
            pvOperation.get_destination_cooperative_tensor<
                decltype(weightTensor), decltype(valueSlice), float>();
        for (int element = 0;
             element < outputProduct.get_capacity(); ++element) {
            if (outputProduct.is_valid_element(element)) {
                outputProduct[element] = 0.0f;
            }
        }
        pvOperation.run(weightTensor, valueSlice, outputProduct);
        for (int element = 0;
             element < outputAccumulator.get_capacity(); ++element) {
            if (!outputAccumulator.is_valid_element(element)
                || !outputProduct.is_valid_element(element)) continue;
            const auto position =
                outputAccumulator.get_multidimensional_index(element);
            const uint row = uint(position[1]);
            outputAccumulator[element] = fma(
                1.0f,
                outputProduct[element],
                outputAccumulator[element] * rowOldScale[row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        keyStart += tileCount;
    }

    for (int element = 0;
         element < outputAccumulator.get_capacity(); ++element) {
        if (!outputAccumulator.is_valid_element(element)) continue;
        const auto position =
            outputAccumulator.get_multidimensional_index(element);
        const uint d = uint(position[0]);
        const uint row = uint(position[1]);
        if (row < validRows) {
            const float denominator = rowSum[row];
            output[((queryStart + row) * numQHeads + queryHead) * headDim + d] =
                denominator > 0.0f
                ? half(outputAccumulator[element] / denominator)
                : half(0.0f);
        }
    }
}

#endif
