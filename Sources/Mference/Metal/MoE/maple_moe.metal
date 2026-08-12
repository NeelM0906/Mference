#include <metal_stdlib>
using namespace metal;

constant constexpr uint kMapleD = 2048u;
constant constexpr uint kMapleF = 512u;
constant constexpr uint kMapleExperts = 256u;
constant constexpr uint kMapleTopK = 8u;

struct ExpertOffsets {
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

struct RoutedBlobs {
    device const uchar* blob[kMapleTopK];
};

kernel void maple_router_bf16_gemv(
    device const bfloat* weights [[buffer(0)]],
    device const bfloat* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (tid >= 256u) return;
    const uint row0 = group * 32u + simd * 4u;
    if (row0 >= kMapleExperts) return;

    float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint column = lane * 4u;
    for (uint block = 0u; block < kMapleD / 128u; ++block) {
        float values[4];
        for (uint n = 0u; n < 4u; ++n) values[n] = float(hidden[column + n]);
        for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
            const uint row = row0 + rowOffset;
            if (row >= kMapleExperts) continue;
            device const bfloat* w = weights + ulong(row) * kMapleD + column;
            for (uint n = 0u; n < 4u; ++n) accumulators[rowOffset] += float(w[n]) * values[n];
        }
        column += 128u;
    }
    for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
        for (ushort offset = 16u; offset >= 1u; offset >>= 1u) {
            accumulators[rowOffset] += simd_shuffle_down(accumulators[rowOffset], offset);
        }
    }
    if (lane == 0u) {
        for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
            if (row0 + rowOffset < kMapleExperts) logits[row0 + rowOffset] = accumulators[rowOffset];
        }
    }
}

kernel void maple_router_top8_full_softmax(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* weights [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float scores[kMapleExperts];
    if (tid != 0u) return;

    float maximum = -1.0e30f;
    for (uint expert = 0u; expert < kMapleExperts; ++expert) {
        const float logit = logits[expert];
        if (!isfinite(logit)) {
            for (uint rank = 0u; rank < kMapleTopK; ++rank) {
                indices[rank] = kMapleExperts;
                weights[rank] = 0.0f;
            }
            return;
        }
        if (logit > maximum) maximum = logit;
    }
    for (uint expert = 0u; expert < kMapleExperts; ++expert) {
        scores[expert] = metal::exp(logits[expert] - maximum);
    }

    float sum = 0.0f;
    for (uint group = 0u; group < 8u; ++group) {
        float partial[32];
        for (uint lane = 0u; lane < 32u; ++lane) partial[lane] = scores[group * 32u + lane];
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            for (uint lane = 0u; lane < offset; ++lane) partial[lane] += partial[lane + offset];
        }
        sum += partial[0];
    }
    const float reciprocal = 1.0f / (sum + 1.0e-20f);
    for (uint expert = 0u; expert < kMapleExperts; ++expert) scores[expert] *= reciprocal;

    uint topIndices[kMapleTopK];
    float topValues[kMapleTopK];
    for (uint rank = 0u; rank < kMapleTopK; ++rank) {
        float bestValue = -1.0e30f;
        uint bestIndex = 0u;
        for (uint expert = 0u; expert < kMapleExperts; ++expert) {
            if (scores[expert] > bestValue) {
                bestValue = scores[expert];
                bestIndex = expert;
            }
        }
        topValues[rank] = bestValue;
        topIndices[rank] = bestIndex;
        scores[bestIndex] = -1.0e30f;
    }
    float selectedSum = 0.0f;
    for (uint rank = 0u; rank < kMapleTopK; ++rank) selectedSum += topValues[rank];
    for (uint rank = 0u; rank < kMapleTopK; ++rank) {
        indices[rank] = topIndices[rank];
        weights[rank] = topValues[rank] / (selectedSum + 1.0e-20f);
    }
}

static inline float maple_load16(
    device const bfloat* x, uint element, thread float* values) {
    float sum = 0.0f;
    for (uint i = 0u; i < 16u; i += 4u) {
        const bfloat x0 = x[element + i];
        const bfloat x1 = x[element + i + 1u];
        const bfloat x2 = x[element + i + 2u];
        const bfloat x3 = x[element + i + 3u];
        sum += float(x0 + x1 + x2 + x3);
        values[i] = float(x0);
        values[i + 1u] = float(x1) / 4.0f;
        values[i + 2u] = float(x2) / 16.0f;
        values[i + 3u] = float(x3) / 64.0f;
    }
    return sum;
}

static inline float maple_qdot16(
    device const uchar* packed, thread const float* values,
    float scale, float bias, float sum) {
    float dot = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint code = uint(packed[i]);
        const uint index = i * 4u;
        dot += values[index] * float(code & 0x03u)
             + values[index + 1u] * float(code & 0x0cu)
             + values[index + 2u] * float(code & 0x30u)
             + values[index + 3u] * float(code & 0xc0u);
    }
    return scale * dot + sum * bias;
}

static inline float maple_swiglu(float gateProjection, float upProjection) {
    const bfloat limit = bfloat(7.0f);
    const bfloat gate = bfloat(min(float(bfloat(gateProjection)), float(limit)));
    const bfloat up = bfloat(clamp(float(bfloat(upProjection)), -float(limit), float(limit)));
    const bfloat absolute = gate < 0 ? bfloat(-float(gate)) : gate;
    const bfloat exponent = bfloat(metal::exp(float(absolute)));
    const bfloat denominator = bfloat(1.0f + float(exponent));
    const bfloat reciprocal = bfloat(1.0f / float(denominator));
    const bfloat sigmoid = gate < 0 ? reciprocal : bfloat(1.0f - float(reciprocal));
    const bfloat silu = bfloat(float(gate) * float(sigmoid));
    return float(bfloat(float(silu) * float(up)));
}

static inline float maple_row(
    device const uchar* weights, device const bfloat* scales,
    device const bfloat* biases, device const bfloat* x,
    uint row, uint columns, uint lane) {
    const uint rowBytes = columns / 4u;
    const uint runtimeGroups = columns / 64u;
    device const uchar* rowWeights = weights + ulong(row) * rowBytes;
    device const bfloat* rowScales = scales + ulong(row) * runtimeGroups;
    device const bfloat* rowBiases = biases + ulong(row) * runtimeGroups;
    float result = 0.0f;
    for (uint block = 0u; block < columns; block += 512u) {
        float values[16];
        const uint element = block + lane * 16u;
        const float sum = maple_load16(x, element, values);
        const uint sourceGroup = block / 128u + lane / 8u;
        const uint runtimeGroup = sourceGroup * 2u;
        const uint packedByte = block / 4u + lane * 4u;
        result += maple_qdot16(rowWeights + packedByte, values,
                               float(rowScales[runtimeGroup]),
                               float(rowBiases[runtimeGroup]), sum);
    }
    return simd_sum(result);
}

static inline void maple_phase1_rows(
    device const uchar* base, constant ExpertOffsets& offsets,
    device const bfloat* x, device bfloat* acts, uint row0, uint lane) {
    for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
        const uint row = row0 + rowOffset;
        const float gate = maple_row(base + offsets.gate_W_off,
                                     (device const bfloat*)(base + offsets.gate_s_off),
                                     (device const bfloat*)(base + offsets.gate_b_off),
                                     x, row, kMapleD, lane);
        const float up = maple_row(base + offsets.up_W_off,
                                   (device const bfloat*)(base + offsets.up_s_off),
                                   (device const bfloat*)(base + offsets.up_b_off),
                                   x, row, kMapleD, lane);
        if (lane == 0u) acts[row] = bfloat(maple_swiglu(gate, up));
    }
}

kernel void maple_moe_phase1(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& offsets [[buffer(1)]],
    device const bfloat* x [[buffer(2)]],
    device bfloat* acts [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (simd >= 2u) return;
    const uint rowGroup = group * 8u + simd * 4u;
    if (rowGroup >= kMapleTopK * kMapleF) return;
    const uint slot = rowGroup / kMapleF;
    const uint row = rowGroup % kMapleF;
    maple_phase1_rows(routed.blob[slot], offsets, x, acts + slot * kMapleF, row, lane);
}

kernel void maple_moe_phase1_subset(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& offsets [[buffer(1)]],
    device const bfloat* x [[buffer(2)]],
    device bfloat* acts [[buffer(3)]],
    device const uint* activeSlots [[buffer(4)]],
    constant uint& activeCount [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (simd >= 2u) return;
    const uint activeRow = group * 8u + simd * 4u;
    if (activeRow >= activeCount * kMapleF) return;
    const uint slot = activeSlots[activeRow / kMapleF];
    if (slot >= kMapleTopK) return;
    maple_phase1_rows(routed.blob[slot], offsets, x,
                      acts + slot * kMapleF, activeRow % kMapleF, lane);
}

kernel void maple_moe_phase2(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& offsets [[buffer(1)]],
    device const bfloat* acts [[buffer(2)]],
    device const float* routingWeights [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[kMapleTopK];
    if (row >= kMapleD || simd >= kMapleTopK) return;
    device const uchar* base = routed.blob[simd];
    const float down = maple_row(base + offsets.down_W_off,
                                 (device const bfloat*)(base + offsets.down_s_off),
                                 (device const bfloat*)(base + offsets.down_b_off),
                                 acts + simd * kMapleF, row, kMapleF, lane);
    if (lane == 0u) partials[simd] = routingWeights[simd] * float(bfloat(down));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u && lane == 0u) {
        float sum = partials[0];
        for (uint rank = 1u; rank < kMapleTopK; ++rank) sum += partials[rank];
        output[row] = bfloat(sum);
    }
}
