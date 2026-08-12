#include <metal_stdlib>
using namespace metal;

constant constexpr uint kMapleFlashD = 2048u;
constant constexpr uint kMapleFlashRowsPerThreadgroup = 8u;
constant constexpr uint kMapleFlashRowsPerSIMD = 4u;
constant constexpr uint kMapleFlashValuesPerLane = 16u;
constant constexpr uint kMapleFlashBlock = 512u;
constant constexpr uint kMapleFlashGroup = 64u;

[[kernel, max_total_threads_per_threadgroup(256)]]
void maple_flash_head_fill_negative_infinity(
    device half* logits [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) logits[gid] = -INFINITY;
}

/// Evaluates exact rows from the original Maple INT4 LM head. `tokenIDs`
/// maps each compact output row to its original vocabulary row, so FlashHead
/// does not retain a second, cluster-ordered copy of the full head.
[[kernel, max_total_threads_per_threadgroup(64)]]
void maple_flash_head_gather_int4_qmv_d2048(
    device const uchar* weights [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device const uint* tokenIDs [[buffer(4)]],
    device half* logits [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row0 = tg * kMapleFlashRowsPerThreadgroup
        + simdGroup * kMapleFlashRowsPerSIMD;
    if (row0 >= rows) return;

    constexpr uint rowBytes = kMapleFlashD / 2u;
    constexpr uint groupsPerRow = kMapleFlashD / kMapleFlashGroup;
    float result[kMapleFlashRowsPerSIMD] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint block = 0; block < kMapleFlashD / kMapleFlashBlock; ++block) {
        const uint element0 = block * kMapleFlashBlock + lane * kMapleFlashValuesPerLane;
        float scaled[kMapleFlashValuesPerLane];
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
        const uint group = block * (kMapleFlashBlock / kMapleFlashGroup)
            + lane / (kMapleFlashGroup / kMapleFlashValuesPerLane);
        for (uint r = 0; r < kMapleFlashRowsPerSIMD; ++r) {
            const uint row = row0 + r;
            if (row >= rows) continue;
            const uint token = tokenIDs[row];
            device const uchar* wrow = weights + ulong(token) * rowBytes;
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
            const ulong parameter = ulong(token) * groupsPerRow + group;
            result[r] += float(scales[parameter]) * qdot + xsum * float(biases[parameter]);
        }
    }

    for (uint r = 0; r < kMapleFlashRowsPerSIMD; ++r) {
        result[r] = simd_sum(result[r]);
        const uint row = row0 + r;
        if (lane == 0 && row < rows) {
            logits[tokenIDs[row]] = half(float(bfloat(result[r])));
        }
    }
}
