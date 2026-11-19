// native_reference.metal: the slow reference kernels of the native engine
// (docs/NATIVE_ENGINE.md, "The six layers"). They are the Phase 0 oracle and
// frozen after Phase 0.
//
// Clarity over speed: one thread computes one output value (or one state row),
// accumulates in F32 in a fixed sequential order, and dequantizes one ggml
// block at a time with a direct port of ggml's `dequantize_row_<type>` at the
// pin (`Sources/LlamaCpp/ggml/src/ggml-quants.c`). There is no tiling, no
// simdgroup work and no threadgroup memory. The Swift CPU twins in
// `Native/Reference/ReferenceCPU*.swift` follow the same order, so the unit
// tests can compare closely.
//
// Every weight read goes through `ref_packed_offset`, the Metal twin of
// `Native.TileLayout.packedOffset`, so the reference reads every layout.

#include <metal_stdlib>
using namespace metal;

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// ---------------------------------------------------------------------------
// ggml lookup tables, copied from the vendored ggml-common.h
// (`ReferenceFormatPinTests` pins the Swift copies to the header and
// `ReferenceKernelPinTests` pins these to the Swift copies).
// ---------------------------------------------------------------------------

constant uchar ref_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128,
};

constant uchar ref_ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

constant char ref_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

constant uint ref_iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

constant uint ref_iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

// ---------------------------------------------------------------------------
// Block loads and dequantization (ports of ggml dequantize_row_<type>)
// ---------------------------------------------------------------------------

#define REF_MAX_BLOCK_BYTES 256
#define REF_MAX_BLOCK_ELEMS 256
#define REF_MAX_HEAD_DIM 256

enum : uint {
    REF_F32 = 0, REF_F16 = 1, REF_Q8_0 = 8, REF_Q4_K = 12, REF_Q5_K = 13, REF_Q6_K = 14,
    REF_IQ3_XXS = 18, REF_IQ3_S = 21, REF_IQ4_XS = 23, REF_BF16 = 30,
};

static inline float ref_load_f16(thread const uchar * b) {
    return float(as_type<half>(ushort(ushort(b[0]) | (ushort(b[1]) << 8))));
}

static inline float ref_load_f32(thread const uchar * b) {
    return as_type<float>(uint(b[0]) | (uint(b[1]) << 8) | (uint(b[2]) << 16) | (uint(b[3]) << 24));
}

static inline float ref_load_bf16(thread const uchar * b) {
    return as_type<float>((uint(b[0]) | (uint(b[1]) << 8)) << 16);
}

static inline uint ref_load_u32(thread const uchar * b) {
    return uint(b[0]) | (uint(b[1]) << 8) | (uint(b[2]) << 16) | (uint(b[3]) << 24);
}

static inline uchar ref_grid_byte(uint word, int j) {
    return uchar((word >> (8 * j)) & 0xff);
}

static void ref_scale_min_k4(int j, thread const uchar * q, thread uchar & d, thread uchar & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// block_q8_0: d (f16), qs[32] (int8)
static void ref_dequant_q8_0(thread const uchar * b, thread float * y) {
    const float d = ref_load_f16(b);
    for (int j = 0; j < 32; ++j) {
        y[j] = float(as_type<char>(b[2 + j])) * d;
    }
}

// block_q4_K: d, dmin (f16), scales[12], qs[128]
static void ref_dequant_q4_K(thread const uchar * b, thread float * y) {
    const float d = ref_load_f16(b);
    const float min = ref_load_f16(b + 2);
    thread const uchar * scales = b + 4;
    thread const uchar * q = b + 16;
    int is = 0;
    int o = 0;
    uchar sc, m;
    for (int j = 0; j < 256; j += 64) {
        ref_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * float(sc); const float m1 = min * float(m);
        ref_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * float(sc); const float m2 = min * float(m);
        for (int l = 0; l < 32; ++l) y[o++] = d1 * float(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) y[o++] = d2 * float(q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}

// block_q5_K: d, dmin (f16), scales[12], qh[32], qs[128]
static void ref_dequant_q5_K(thread const uchar * b, thread float * y) {
    const float d = ref_load_f16(b);
    const float min = ref_load_f16(b + 2);
    thread const uchar * scales = b + 4;
    thread const uchar * qh = b + 16;
    thread const uchar * ql = b + 48;
    int is = 0;
    int o = 0;
    uchar sc, m;
    uchar u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        ref_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * float(sc); const float m1 = min * float(m);
        ref_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * float(sc); const float m2 = min * float(m);
        for (int l = 0; l < 32; ++l) y[o++] = d1 * float((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) y[o++] = d2 * float((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32; is += 2;
        u1 <<= 2; u2 <<= 2;
    }
}

// block_q6_K: ql[128], qh[64], scales[16] (int8), d (f16)
static void ref_dequant_q6_K(thread const uchar * b, thread float * y) {
    const float d = ref_load_f16(b + 208);
    thread const uchar * ql = b;
    thread const uchar * qh = b + 128;
    thread const uchar * sc = b + 192;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = int(as_type<char>(uchar((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)))) - 32;
            const int q2 = int(as_type<char>(uchar((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)))) - 32;
            const int q3 = int(as_type<char>(uchar((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)))) - 32;
            const int q4 = int(as_type<char>(uchar((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)))) - 32;
            y[l +  0] = d * float(as_type<char>(sc[is + 0])) * float(q1);
            y[l + 32] = d * float(as_type<char>(sc[is + 2])) * float(q2);
            y[l + 64] = d * float(as_type<char>(sc[is + 4])) * float(q3);
            y[l + 96] = d * float(as_type<char>(sc[is + 6])) * float(q4);
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

// block_iq4_xs: d (f16), scales_h (u16), scales_l[4], qs[128]
static void ref_dequant_iq4_xs(thread const uchar * b, thread float * y) {
    const float d = ref_load_f16(b);
    const uint scales_h = uint(b[2]) | (uint(b[3]) << 8);
    thread const uchar * scales_l = b + 4;
    thread const uchar * qs = b + 8;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = int(((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4));
        const float dl = d * float(ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j +  0] = dl * float(ref_kvalues_iq4nl[qs[j] & 0xf]);
            y[j + 16] = dl * float(ref_kvalues_iq4nl[qs[j] >> 4]);
        }
        y += 32; qs += 16;
    }
}

// block_iq3_xxs: d (f16), qs[96] (64 grid indices, then 32 bytes of scales and signs)
static void ref_dequant_iq3_xxs(thread const uchar * b, thread float * y) {
    const float d = ref_load_f16(b);
    thread const uchar * qs = b + 2;
    thread const uchar * scales_and_signs = b + 2 + 64;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint aux32 = ref_load_u32(scales_and_signs + 4 * ib32);
        const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uchar signs = ref_ksigns_iq2xs[(aux32 >> 7 * l) & 127];
            const uint grid1 = ref_iq3xxs_grid[qs[2 * l + 0]];
            const uint grid2 = ref_iq3xxs_grid[qs[2 * l + 1]];
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db * float(ref_grid_byte(grid1, j)) * ((signs & ref_kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                y[j + 4] = db * float(ref_grid_byte(grid2, j)) * ((signs & ref_kmask_iq2xs[j + 4]) ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
    }
}

// block_iq3_s: d (f16), qs[64], qh[8], signs[32], scales[4]
static void ref_dequant_iq3_s(thread const uchar * b, thread float * y) {
    const float d = ref_load_f16(b);
    thread const uchar * qs = b + 2;
    thread const uchar * qh = b + 66;
    thread const uchar * signs = b + 74;
    thread const uchar * scales = b + 106;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const float db1 = d * float(1 + 2 * (scales[ib32 / 2] & 0xf));
        const float db2 = d * float(1 + 2 * (scales[ib32 / 2] >> 4));
        for (int l = 0; l < 4; ++l) {
            const uint grid1 = ref_iq3s_grid[int(qs[2 * l + 0]) | ((int(qh[0]) << (8 - 2 * l)) & 256)];
            const uint grid2 = ref_iq3s_grid[int(qs[2 * l + 1]) | ((int(qh[0]) << (7 - 2 * l)) & 256)];
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db1 * float(ref_grid_byte(grid1, j)) * ((signs[l] & ref_kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                y[j + 4] = db1 * float(ref_grid_byte(grid2, j)) * ((signs[l] & ref_kmask_iq2xs[j + 4]) ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8; signs += 4;
        for (int l = 0; l < 4; ++l) {
            const uint grid1 = ref_iq3s_grid[int(qs[2 * l + 0]) | ((int(qh[1]) << (8 - 2 * l)) & 256)];
            const uint grid2 = ref_iq3s_grid[int(qs[2 * l + 1]) | ((int(qh[1]) << (7 - 2 * l)) & 256)];
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db2 * float(ref_grid_byte(grid1, j)) * ((signs[l] & ref_kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                y[j + 4] = db2 * float(ref_grid_byte(grid2, j)) * ((signs[l] & ref_kmask_iq2xs[j + 4]) ? -1.f : 1.f);
            }
            y += 8;
        }
        qh += 2; qs += 8; signs += 4;
    }
}

// Dequantize one block of `type` (`blockElems` values) into `y`.
static void ref_dequant_block(uint type, thread const uchar * b, thread float * y) {
    switch (type) {
        case REF_F32: y[0] = ref_load_f32(b); break;
        case REF_F16: y[0] = ref_load_f16(b); break;
        case REF_BF16: y[0] = ref_load_bf16(b); break;
        case REF_Q8_0: ref_dequant_q8_0(b, y); break;
        case REF_Q4_K: ref_dequant_q4_K(b, y); break;
        case REF_Q5_K: ref_dequant_q5_K(b, y); break;
        case REF_Q6_K: ref_dequant_q6_K(b, y); break;
        case REF_IQ4_XS: ref_dequant_iq4_xs(b, y); break;
        case REF_IQ3_XXS: ref_dequant_iq3_xxs(b, y); break;
        case REF_IQ3_S: ref_dequant_iq3_s(b, y); break;
        default: y[0] = NAN; break;
    }
}

// ---------------------------------------------------------------------------
// Tile layout (Metal twin of Native.TileLayout.packedOffset)
// ---------------------------------------------------------------------------

struct RefLayout {
    uint type;
    uint blockElems;
    uint blockBytes;
    uint blocksPerRow;
    uint rowsPerTile;
    uint blocksPerTile;
    uint planes;
    uint fieldCount;
    uint fieldBytes[8];
};

static ulong ref_packed_offset(constant RefLayout & L, uint row, uint block, uint byteInBlock) {
    const ulong r = L.rowsPerTile, k = L.blocksPerTile, bb = L.blockBytes;
    const ulong chunksPerRow = L.blocksPerRow / k;
    const ulong tileIndex = (ulong(row) / r) * chunksPerRow + (ulong(block) / k);
    const ulong tileBytes = r * k * bb;
    const ulong blockInTile = (ulong(row) % r) * k + (ulong(block) % k);
    const ulong tileBase = tileIndex * tileBytes;
    if (L.planes == 0) {
        return tileBase + blockInTile * bb + byteInBlock;
    }
    ulong fieldStart = 0;
    ulong planeBase = 0;
    for (uint f = 0; f < L.fieldCount; ++f) {
        const ulong fb = L.fieldBytes[f];
        if (byteInBlock < fieldStart + fb) {
            return tileBase + planeBase + blockInTile * fb + (byteInBlock - fieldStart);
        }
        fieldStart += fb;
        planeBase += r * k * fb;
    }
    return 0;
}

// The GGUF-order bytes of one block, byte by byte through ref_packed_offset
// (the Metal twin of ReferenceDequant.gatherBlock).
static void ref_gather_block(device const uchar * W, constant RefLayout & L, uint row, uint block,
                             thread uchar * out) {
    for (uint i = 0; i < L.blockBytes; ++i) {
        out[i] = W[ref_packed_offset(L, row, block, i)];
    }
}

// ---------------------------------------------------------------------------
// QuantGEMM: y[r][o] (+)= sum_i W[o][i] * x[r][i]
// One thread per (o, r); one kernel serves GEMV, verify and prefill.
// ---------------------------------------------------------------------------

struct RefGemmParams {
    uint outRows;
    uint rows;
    uint inputStride;   // bytes between input rows
    uint outputStride;  // bytes between output rows
    uint accumulate;
};

kernel void ref_quant_gemm(device const uchar * W [[buffer(0)]],
                           device const uchar * X [[buffer(1)]],
                           device uchar * Y [[buffer(2)]],
                           constant RefGemmParams & p [[buffer(3)]],
                           constant RefLayout & L [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
    const uint o = gid.x;
    const uint r = gid.y;
    if (o >= p.outRows || r >= p.rows) return;
    device const float * x = (device const float *)(X + ulong(r) * p.inputStride);
    uchar blk[REF_MAX_BLOCK_BYTES];
    float w[REF_MAX_BLOCK_ELEMS];
    float acc = 0.0f;
    for (uint b = 0; b < L.blocksPerRow; ++b) {
        ref_gather_block(W, L, o, b, blk);
        ref_dequant_block(L.type, blk, w);
        const uint base = b * L.blockElems;
        for (uint e = 0; e < L.blockElems; ++e) {
            acc += w[e] * x[base + e];
        }
    }
    device float * y = (device float *)(Y + ulong(r) * p.outputStride) + o;
    if (p.accumulate != 0) {
        *y = *y + acc;
    } else {
        *y = acc;
    }
}

struct RefGetRowsParams {
    uint tokens;
    uint vocab;
    uint outputStride;
};

// output[t] = dequant(W row tokens[t]); one thread per (block, t).
kernel void ref_get_rows(device const uchar * W [[buffer(0)]],
                         device const uint * T [[buffer(1)]],
                         device uchar * Y [[buffer(2)]],
                         constant RefGetRowsParams & p [[buffer(3)]],
                         constant RefLayout & L [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint b = gid.x;
    const uint t = gid.y;
    if (b >= L.blocksPerRow || t >= p.tokens) return;
    device float * y = (device float *)(Y + ulong(t) * p.outputStride) + b * L.blockElems;
    const uint token = T[t];
    if (token >= p.vocab) {
        for (uint e = 0; e < L.blockElems; ++e) y[e] = NAN;
        return;
    }
    uchar blk[REF_MAX_BLOCK_BYTES];
    float w[REF_MAX_BLOCK_ELEMS];
    ref_gather_block(W, L, token, b, blk);
    ref_dequant_block(L.type, blk, w);
    for (uint e = 0; e < L.blockElems; ++e) y[e] = w[e];
}

// ---------------------------------------------------------------------------
// Glue: RMSNorm, add, SwiGLU, copy
// ---------------------------------------------------------------------------

struct RefRowParams {
    uint rows;
    uint cols;
    uint aStride;
    uint bStride;
    uint outStride;
    float eps;
    uint weightType;
};

static inline float ref_weight_at(device const uchar * W, uint type, uint i) {
    if (type == REF_F16) {
        thread uchar b[2] = { W[2 * i], W[2 * i + 1] };
        return ref_load_f16(b);
    }
    return ((device const float *)W)[i];
}

// y = x / sqrt(mean(x^2) + eps) * w (ggml rms_norm, then mul); one thread per row.
kernel void ref_rms_norm(device const uchar * X [[buffer(0)]],
                         device const uchar * W [[buffer(1)]],
                         device uchar * Y [[buffer(2)]],
                         constant RefRowParams & p [[buffer(3)]],
                         uint r [[thread_position_in_grid]]) {
    if (r >= p.rows) return;
    device const float * x = (device const float *)(X + ulong(r) * p.aStride);
    device float * y = (device float *)(Y + ulong(r) * p.outStride);
    float sum = 0.0f;
    for (uint i = 0; i < p.cols; ++i) sum += x[i] * x[i];
    const float mean = sum / float(p.cols);
    const float scale = 1.0f / sqrt(mean + p.eps);
    for (uint i = 0; i < p.cols; ++i) y[i] = (x[i] * scale) * ref_weight_at(W, p.weightType, i);
}

kernel void ref_add(device const uchar * A [[buffer(0)]],
                    device const uchar * B [[buffer(1)]],
                    device uchar * Y [[buffer(2)]],
                    constant RefRowParams & p [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.cols || gid.y >= p.rows) return;
    const float a = ((device const float *)(A + ulong(gid.y) * p.aStride))[gid.x];
    const float b = ((device const float *)(B + ulong(gid.y) * p.bStride))[gid.x];
    ((device float *)(Y + ulong(gid.y) * p.outStride))[gid.x] = a + b;
}

static inline float ref_silu(float x) { return x / (1.0f + exp(-x)); }
static inline float ref_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// y = silu(gate) * up (gate in A, up in B).
kernel void ref_swiglu(device const uchar * A [[buffer(0)]],
                       device const uchar * B [[buffer(1)]],
                       device uchar * Y [[buffer(2)]],
                       constant RefRowParams & p [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.cols || gid.y >= p.rows) return;
    const float g = ((device const float *)(A + ulong(gid.y) * p.aStride))[gid.x];
    const float u = ((device const float *)(B + ulong(gid.y) * p.bStride))[gid.x];
    ((device float *)(Y + ulong(gid.y) * p.outStride))[gid.x] = ref_silu(g) * u;
}

// Byte copy of `cols` bytes per row (the host passes cols * element size).
kernel void ref_copy_bytes(device const uchar * A [[buffer(0)]],
                           device uchar * Y [[buffer(2)]],
                           constant RefRowParams & p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.cols || gid.y >= p.rows) return;
    Y[ulong(gid.y) * p.outStride + gid.x] = A[ulong(gid.y) * p.aStride + gid.x];
}

// ---------------------------------------------------------------------------
// Full attention over paged KV (Native.KVPageGeometry)
// ---------------------------------------------------------------------------

struct RefAttnParams {
    uint rows;
    uint startPosition;
    uint headCount;
    uint kvHeadCount;
    uint headDim;
    uint ropeDims;
    float eps;
    float freqBase;
    float scale;
    uint qGateStride;
    uint kStride;
    uint vStride;
    uint outStride;
    uint tokensPerPage;
    uint attnLayer;
    uint attentionLayers;
    uint int8;
    uint pageBytes;
};

static ulong ref_kv_row(constant RefAttnParams & p, uint kv, uint head, uint token) {
    return ((ulong(p.attnLayer) * 2 + kv) * p.kvHeadCount + head) * p.tokensPerPage + token;
}

static ulong ref_kv_value_offset(constant RefAttnParams & p, uint page, uint kv, uint head, uint token, uint dim) {
    const ulong vb = p.int8 != 0 ? 1 : 2;
    return ulong(page) * p.pageBytes + (ref_kv_row(p, kv, head, token) * p.headDim + dim) * vb;
}

static ulong ref_kv_scale_offset(constant RefAttnParams & p, uint page, uint kv, uint head, uint token) {
    const ulong rowsPerPage = ulong(p.attentionLayers) * 2 * p.kvHeadCount * p.tokensPerPage;
    return ulong(page) * p.pageBytes + rowsPerPage * p.headDim + ref_kv_row(p, kv, head, token) * 2;
}

// Per-head RMSNorm with weight w (ggml rms_norm then mul).
static void ref_head_rms_norm(thread float * v, uint n, device const float * w, float eps) {
    float sum = 0.0f;
    for (uint i = 0; i < n; ++i) sum += v[i] * v[i];
    const float mean = sum / float(n);
    const float scale = 1.0f / sqrt(mean + eps);
    for (uint i = 0; i < n; ++i) v[i] = (v[i] * scale) * w[i];
}

// NEOX RoPE on the first n dims at position pos (ggml-metal theta form).
static void ref_rope_neox(thread float * v, uint n, float pos, float base) {
    const float inv_ndims = -1.0f / float(n);
    for (uint i = 0; i < n / 2; ++i) {
        const float theta = pos * pow(base, inv_ndims * float(2 * i));
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = v[i];
        const float x1 = v[i + n / 2];
        v[i] = x0 * c - x1 * s;
        v[i + n / 2] = x0 * s + x1 * c;
    }
}

static float ref_kv_load(device const uchar * arena, constant RefAttnParams & p, uint page, uint kv,
                         uint head, uint token, uint dim) {
    const ulong off = ref_kv_value_offset(p, page, kv, head, token, dim);
    if (p.int8 != 0) {
        const ulong so = ref_kv_scale_offset(p, page, kv, head, token);
        const float s = float(as_type<half>(ushort(ushort(arena[so]) | (ushort(arena[so + 1]) << 8))));
        return s * float(as_type<char>(arena[off]));
    }
    return float(as_type<half>(ushort(ushort(arena[off]) | (ushort(arena[off + 1]) << 8))));
}

static void ref_kv_store_row(device uchar * arena, constant RefAttnParams & p, uint page, uint kv,
                             uint head, uint token, thread const float * v) {
    if (p.int8 != 0) {
        float amax = 0.0f;
        for (uint d = 0; d < p.headDim; ++d) amax = max(amax, fabs(v[d]));
        const half s = half(amax / 127.0f);
        const float fs = float(s);
        for (uint d = 0; d < p.headDim; ++d) {
            const float q = fs == 0.0f ? 0.0f : clamp(rint(v[d] / fs), -127.0f, 127.0f);
            arena[ref_kv_value_offset(p, page, kv, head, token, d)] = as_type<uchar>(char(q));
        }
        const ushort sb = as_type<ushort>(s);
        const ulong so = ref_kv_scale_offset(p, page, kv, head, token);
        arena[so] = uchar(sb & 0xff);
        arena[so + 1] = uchar(sb >> 8);
        return;
    }
    for (uint d = 0; d < p.headDim; ++d) {
        const ushort hb = as_type<ushort>(half(v[d]));
        const ulong off = ref_kv_value_offset(p, page, kv, head, token, d);
        arena[off] = uchar(hb & 0xff);
        arena[off + 1] = uchar(hb >> 8);
    }
}

// Append K (after k-norm and RoPE) and V of every row to the pages.
// gid.x = kvHead * 2 + (0 = K, 1 = V), gid.y = row.
kernel void ref_attn_append_kv(device const uchar * K [[buffer(0)]],
                               device const uchar * V [[buffer(1)]],
                               device const float * kNorm [[buffer(2)]],
                               device uchar * arena [[buffer(3)]],
                               device const uint * pageTable [[buffer(4)]],
                               constant RefAttnParams & p [[buffer(5)]],
                               uint2 gid [[thread_position_in_grid]]) {
    const uint head = gid.x / 2;
    const uint kind = gid.x % 2;
    const uint r = gid.y;
    if (head >= p.kvHeadCount || r >= p.rows) return;
    const uint pos = p.startPosition + r;
    const uint page = pageTable[pos / p.tokensPerPage];
    const uint token = pos % p.tokensPerPage;
    float v[REF_MAX_HEAD_DIM];
    device const float * src = kind == 0
        ? (device const float *)(K + ulong(r) * p.kStride) + head * p.headDim
        : (device const float *)(V + ulong(r) * p.vStride) + head * p.headDim;
    for (uint d = 0; d < p.headDim; ++d) v[d] = src[d];
    if (kind == 0) {
        ref_head_rms_norm(v, p.headDim, kNorm, p.eps);
        ref_rope_neox(v, p.ropeDims, float(pos), p.freqBase);
    }
    ref_kv_store_row(arena, p, page, kind, head, token, v);
}

// Causal GQA attention of each (row, head) over positions 0...pos, gated.
// gid.x = head, gid.y = row. Two passes (max, then exp-sum and weighted V).
kernel void ref_attn_core(device const uchar * QG [[buffer(0)]],
                          device const float * qNorm [[buffer(1)]],
                          device const uchar * arena [[buffer(2)]],
                          device const uint * pageTable [[buffer(3)]],
                          device uchar * OUT [[buffer(4)]],
                          constant RefAttnParams & p [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    const uint h = gid.x;
    const uint r = gid.y;
    if (h >= p.headCount || r >= p.rows) return;
    const uint D = p.headDim;
    const uint pos = p.startPosition + r;
    device const float * qg = (device const float *)(QG + ulong(r) * p.qGateStride) + h * 2 * D;
    float q[REF_MAX_HEAD_DIM];
    for (uint d = 0; d < D; ++d) q[d] = qg[d];
    ref_head_rms_norm(q, D, qNorm, p.eps);
    ref_rope_neox(q, p.ropeDims, float(pos), p.freqBase);
    const uint kvh = h / (p.headCount / p.kvHeadCount);

    float m = -INFINITY;
    for (uint t = 0; t <= pos; ++t) {
        const uint page = pageTable[t / p.tokensPerPage];
        const uint tok = t % p.tokensPerPage;
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) s += q[d] * ref_kv_load(arena, p, page, 0, kvh, tok, d);
        m = max(m, s * p.scale);
    }
    float acc[REF_MAX_HEAD_DIM];
    for (uint d = 0; d < D; ++d) acc[d] = 0.0f;
    float sum = 0.0f;
    for (uint t = 0; t <= pos; ++t) {
        const uint page = pageTable[t / p.tokensPerPage];
        const uint tok = t % p.tokensPerPage;
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) s += q[d] * ref_kv_load(arena, p, page, 0, kvh, tok, d);
        const float e = exp(s * p.scale - m);
        sum += e;
        for (uint d = 0; d < D; ++d) acc[d] += e * ref_kv_load(arena, p, page, 1, kvh, tok, d);
    }
    device const float * gate = qg + D;
    device float * out = (device float *)(OUT + ulong(r) * p.outStride) + h * D;
    for (uint d = 0; d < D; ++d) out[d] = (acc[d] / sum) * ref_sigmoid(gate[d]);
}

// ---------------------------------------------------------------------------
// Gated DeltaNet over Native.RecurrentLayout state
// ---------------------------------------------------------------------------

struct RefGDNParams {
    uint rows;
    uint convKernel;
    uint convChannels;
    uint valueHeads;
    uint groups;
    uint dim;
    float eps;
    uint qkvStride;
    uint alphaStride;
    uint betaStride;
    uint outStride;
    uint zStride;
    uint ssmOffset;
    uint writeArena;
    uint arenaStride;
};

// Conv input of channel c at step time `tt` (negative = from the conv state).
static float ref_gdn_conv_input(device const uchar * QKV, device const uchar * stateIn,
                                constant RefGDNParams & p, uint c, int tt) {
    if (tt < 0) {
        const int row = int(p.convKernel) - 1 + tt;
        return ((device const float *)stateIn)[ulong(row) * p.convChannels + c];
    }
    return ((device const float *)(QKV + ulong(tt) * p.qkvStride))[c];
}

// silu(conv1d) of channel c at row r.
static float ref_gdn_conv(device const uchar * QKV, device const uchar * stateIn,
                          device const float * convW, constant RefGDNParams & p, uint c, uint r) {
    float s = 0.0f;
    for (uint k = 0; k < p.convKernel; ++k) {
        const int tt = int(r) - int(p.convKernel - 1) + int(k);
        s += ref_gdn_conv_input(QKV, stateIn, p, c, tt) * convW[ulong(c) * p.convKernel + k];
    }
    return ref_silu(s);
}

// ggml build_gdn_l2_norm: rms_norm(x, eps / n) * (1 / sqrt(n)).
static void ref_gdn_l2_norm(thread float * v, uint n, float eps) {
    float sum = 0.0f;
    for (uint i = 0; i < n; ++i) sum += v[i] * v[i];
    const float mean = sum / float(n);
    const float scale = 1.0f / sqrt(mean + eps / float(n));
    const float post = 1.0f / sqrt(float(n));
    for (uint i = 0; i < n; ++i) v[i] = (v[i] * scale) * post;
}

// Write the conv state after `consumed` rows (the last convKernel - 1 inputs)
// for every channel this thread owns.
static void ref_gdn_write_conv_state(device const uchar * QKV, device const uchar * stateIn,
                                     device uchar * dst, constant RefGDNParams & p,
                                     uint thread_index, uint thread_count, uint consumed) {
    device float * conv = (device float *)dst;
    for (uint c = thread_index; c < p.convChannels; c += thread_count) {
        for (uint k = 0; k + 1 < p.convKernel; ++k) {
            const int tt = int(consumed) - int(p.convKernel - 1) + int(k);
            conv[ulong(k) * p.convChannels + c] = ref_gdn_conv_input(QKV, stateIn, p, c, tt);
        }
    }
}

// The delta-rule scan. One thread per (value head h, value dim j) owns state
// row S[h][j][:] and recomputes the conv, SiLU and L2 norm of its q/k head
// (head h % groups) for every row. Output is o before the gated norm.
kernel void ref_gdn_scan(device const uchar * QKV [[buffer(0)]],
                         device const uchar * ALPHA [[buffer(1)]],
                         device const uchar * BETA [[buffer(2)]],
                         device const float * convW [[buffer(3)]],
                         device const float * dtBias [[buffer(4)]],
                         device const float * A [[buffer(5)]],
                         device const uchar * stateIn [[buffer(6)]],
                         device uchar * stateOut [[buffer(7)]],
                         device uchar * arena [[buffer(8)]],
                         device uchar * OUT [[buffer(9)]],
                         constant RefGDNParams & p [[buffer(10)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint h = gid.y;
    const uint d = p.dim;
    if (j >= d || h >= p.valueHeads) return;
    const uint g = h % p.groups;
    const uint qBase = g * d;
    const uint kBase = p.groups * d + g * d;
    const uint vChannel = 2 * p.groups * d + h * d + j;
    const float scale = 1.0f / sqrt(float(d));

    float S[REF_MAX_HEAD_DIM];
    device const float * sIn = (device const float *)(stateIn + p.ssmOffset) + (ulong(h) * d + j) * d;
    for (uint i = 0; i < d; ++i) S[i] = sIn[i];

    float q[REF_MAX_HEAD_DIM];
    float k[REF_MAX_HEAD_DIM];
    const uint threadIndex = h * d + j;
    const uint threadCount = p.valueHeads * d;
    for (uint r = 0; r < p.rows; ++r) {
        for (uint i = 0; i < d; ++i) {
            q[i] = ref_gdn_conv(QKV, stateIn, convW, p, qBase + i, r);
            k[i] = ref_gdn_conv(QKV, stateIn, convW, p, kBase + i, r);
        }
        ref_gdn_l2_norm(q, d, p.eps);
        ref_gdn_l2_norm(k, d, p.eps);
        const float v = ref_gdn_conv(QKV, stateIn, convW, p, vChannel, r);

        const float alpha = ((device const float *)(ALPHA + ulong(r) * p.alphaStride))[h];
        const float betaRaw = ((device const float *)(BETA + ulong(r) * p.betaStride))[h];
        const float biased = alpha + dtBias[h];
        const float softplus = biased > 20.0f ? biased : log(1.0f + exp(biased));
        const float gate = softplus * A[h];
        const float beta = ref_sigmoid(betaRaw);
        const float decay = exp(gate);

        for (uint i = 0; i < d; ++i) S[i] = S[i] * decay;
        float kv = 0.0f;
        for (uint i = 0; i < d; ++i) kv += S[i] * k[i];
        const float delta = (v - kv) * beta;
        for (uint i = 0; i < d; ++i) S[i] = S[i] + k[i] * delta;
        float o = 0.0f;
        for (uint i = 0; i < d; ++i) o += S[i] * q[i];
        ((device float *)(OUT + ulong(r) * p.outStride))[h * d + j] = o * scale;

        if (p.writeArena != 0) {
            device uchar * slot = arena + ulong(r) * p.arenaStride;
            device float * sSlot = (device float *)(slot + p.ssmOffset) + (ulong(h) * d + j) * d;
            for (uint i = 0; i < d; ++i) sSlot[i] = S[i];
            ref_gdn_write_conv_state(QKV, stateIn, slot, p, threadIndex, threadCount, r + 1);
        }
    }
    device float * sOut = (device float *)(stateOut + p.ssmOffset) + (ulong(h) * d + j) * d;
    for (uint i = 0; i < d; ++i) sOut[i] = S[i];
    ref_gdn_write_conv_state(QKV, stateIn, stateOut, p, threadIndex, threadCount, p.rows);
}

// out[r][h] = RMSNorm(o[r][h], norm) * silu(z[r][h]), in place; one thread per (h, r).
kernel void ref_gdn_gated_norm(device uchar * OUT [[buffer(0)]],
                               device const uchar * Z [[buffer(1)]],
                               device const float * normW [[buffer(2)]],
                               constant RefGDNParams & p [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    const uint h = gid.x;
    const uint r = gid.y;
    if (h >= p.valueHeads || r >= p.rows) return;
    const uint d = p.dim;
    device float * o = (device float *)(OUT + ulong(r) * p.outStride) + h * d;
    device const float * z = (device const float *)(Z + ulong(r) * p.zStride) + h * d;
    float sum = 0.0f;
    for (uint i = 0; i < d; ++i) sum += o[i] * o[i];
    const float mean = sum / float(d);
    const float scale = 1.0f / sqrt(mean + p.eps);
    for (uint i = 0; i < d; ++i) o[i] = ((o[i] * scale) * normW[i]) * ref_silu(z[i]);
}

// ---------------------------------------------------------------------------
// Sampling: greedy argmax (ties to the lowest id), one thread.
// ---------------------------------------------------------------------------

kernel void ref_argmax(device const float * logits [[buffer(0)]],
                       device uint * out [[buffer(1)]],
                       constant uint & count [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint best = 0;
    float bestValue = logits[0];
    for (uint i = 1; i < count; ++i) {
        if (logits[i] > bestValue) {
            bestValue = logits[i];
            best = i;
        }
    }
    out[0] = best;
}
