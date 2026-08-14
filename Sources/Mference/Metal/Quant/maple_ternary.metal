#include <metal_stdlib>
using namespace metal;

constant constexpr uint kMapleD = 2048u;
constant constexpr uint kMapleRowsPerThreadgroup = 8u;
constant constexpr uint kMapleRowsPerSIMD = 4u;
constant constexpr uint kMapleValuesPerLane = 16u;
constant constexpr uint kMapleBlock = 512u;
constant constexpr uint kMapleGroup = 64u;
constant constexpr uint kMapleSourceGroup = 128u;

static inline uint mapleInt4Code(device const uchar* row, uint element) {
    const uchar packed = row[element >> 1u];
    return (element & 1u) == 0u ? uint(packed & 0x0fu) : uint(packed >> 4u);
}

[[kernel, max_total_threads_per_threadgroup(64)]]
void maple_ternary_qmv_d2048(
    device const uchar* weights [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row0 = tg * kMapleRowsPerThreadgroup + simdGroup * kMapleRowsPerSIMD;
    if (row0 >= rows) return;

    constexpr uint rowBytes = kMapleD / 2u;
    constexpr uint groupsPerRow = kMapleD / kMapleGroup;
    float result[kMapleRowsPerSIMD] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint block = 0; block < kMapleD / kMapleBlock; ++block) {
        const uint element0 = block * kMapleBlock + lane * kMapleValuesPerLane;
        float xv[kMapleValuesPerLane];
        float xsum = 0.0f;
        for (uint group4 = 0; group4 < 4; ++group4) {
            const uint j = group4 * 4;
            const bfloat bx0 = x[element0 + j];
            const bfloat bx1 = x[element0 + j + 1u];
            const bfloat bx2 = x[element0 + j + 2u];
            const bfloat bx3 = x[element0 + j + 3u];
            xv[j] = float(bx0);
            xv[j + 1u] = float(bx1) / 4.0f;
            xv[j + 2u] = float(bx2) / 16.0f;
            xv[j + 3u] = float(bx3) / 64.0f;
            xsum += float(bx0 + bx1 + bx2 + bx3);
        }
        const uint group = block * (kMapleBlock / kMapleGroup)
            + (lane / (kMapleSourceGroup / kMapleValuesPerLane)) * 2u;
        for (uint r = 0; r < kMapleRowsPerSIMD; ++r) {
            const uint row = row0 + r;
            if (row >= rows) continue;
            device const uchar* wrow = weights + ulong(row) * rowBytes;
            float qdot = 0.0f;
            for (uint group4 = 0; group4 < 4; ++group4) {
                const uint j = group4 * 4;
                const uint c0 = mapleInt4Code(wrow, element0 + j);
                const uint c1 = mapleInt4Code(wrow, element0 + j + 1u);
                const uint c2 = mapleInt4Code(wrow, element0 + j + 2u);
                const uint c3 = mapleInt4Code(wrow, element0 + j + 3u);
                const uint packed2 = c0 | (c1 << 2u) | (c2 << 4u) | (c3 << 6u);
                qdot += xv[j] * float(packed2 & 0x03u)
                    + xv[j + 1u] * float(packed2 & 0x0cu)
                    + xv[j + 2u] * float(packed2 & 0x30u)
                    + xv[j + 3u] * float(packed2 & 0xc0u);
            }
            const ulong parameter = ulong(row) * groupsPerRow + group;
            result[r] += float(scales[parameter]) * qdot + xsum * float(biases[parameter]);
        }
    }

    for (uint r = 0; r < kMapleRowsPerSIMD; ++r) {
        result[r] = simd_sum(result[r]);
        const uint row = row0 + r;
        if (lane == 0 && row < rows) y[row] = bfloat(result[r]);
    }
}

[[kernel, max_total_threads_per_threadgroup(64)]]
void maple_int4_qmv_d2048(
    device const uchar* weights [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row0 = tg * kMapleRowsPerThreadgroup + simdGroup * kMapleRowsPerSIMD;
    if (row0 >= rows) return;

    constexpr uint rowBytes = kMapleD / 2u;
    constexpr uint groupsPerRow = kMapleD / kMapleGroup;
    float result[kMapleRowsPerSIMD] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint block = 0; block < kMapleD / kMapleBlock; ++block) {
        const uint element0 = block * kMapleBlock + lane * kMapleValuesPerLane;
        float scaled[kMapleValuesPerLane];
        float xsum = 0.0f;
        for (uint group4 = 0; group4 < 4; ++group4) {
            const uint j = group4 * 4;
            const bfloat bx0 = x[element0 + j];
            const bfloat bx1 = x[element0 + j + 1u];
            const bfloat bx2 = x[element0 + j + 2u];
            const bfloat bx3 = x[element0 + j + 3u];
            xsum += float(bx0 + bx1 + bx2 + bx3);
            scaled[j] = float(bx0);
            scaled[j + 1u] = float(bx1) / 16.0f;
            scaled[j + 2u] = float(bx2) / 256.0f;
            scaled[j + 3u] = float(bx3) / 4096.0f;
        }
        const uint group = block * (kMapleBlock / kMapleGroup)
            + lane / (kMapleGroup / kMapleValuesPerLane);
        for (uint r = 0; r < kMapleRowsPerSIMD; ++r) {
            const uint row = row0 + r;
            if (row >= rows) continue;
            device const uchar* wrow = weights + ulong(row) * rowBytes;
            float qdot = 0.0f;
            for (uint group4 = 0; group4 < 4; ++group4) {
                const uint j = group4 * 4;
                device const ushort* packed = (device const ushort*)(wrow + (element0 + j) / 2u);
                const uint word = uint(*packed);
                qdot += scaled[j] * float(word & 0x000fu)
                    + scaled[j + 1u] * float(word & 0x00f0u)
                    + scaled[j + 2u] * float(word & 0x0f00u)
                    + scaled[j + 3u] * float(word & 0xf000u);
            }
            const ulong parameter = ulong(row) * groupsPerRow + group;
            result[r] += float(scales[parameter]) * qdot + xsum * float(biases[parameter]);
        }
    }

    for (uint r = 0; r < kMapleRowsPerSIMD; ++r) {
        result[r] = simd_sum(result[r]);
        const uint row = row0 + r;
        if (lane == 0 && row < rows) y[row] = half(float(bfloat(result[r])));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void maple_embed_lookup_int4_bf16(
    device const uchar* table [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& tokenID [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= kMapleD) return;
    constexpr uint rowBytes = kMapleD / 2u;
    constexpr uint groupsPerRow = kMapleD / kMapleGroup;
    device const uchar* row = table + ulong(tokenID) * rowBytes;
    const uint code = mapleInt4Code(row, gid);
    const ulong parameter = ulong(tokenID) * groupsPerRow + gid / kMapleGroup;
    out[gid] = scales[parameter] * code + biases[parameter];
}
