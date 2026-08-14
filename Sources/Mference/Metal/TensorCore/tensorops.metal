#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

constant constexpr uint kW4A8GroupSize = 64;
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
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < groupsPerRow; ++group) {
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
                const float scale = float(scales[globalN * groupsPerRow + group]);
                const float bias = float(biases[globalN * groupsPerRow + group]);
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
            output[globalM * N + globalN] = half(accumulator[element]);
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
