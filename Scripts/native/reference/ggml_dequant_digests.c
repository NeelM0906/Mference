// Prints FNV-1a 64 digests of ggml's dequantize_row_<type> output over
// deterministic pseudo-random blocks. ReferenceDequantTests regenerates the
// same blocks in Swift and checks Native.ReferenceDequant against these
// digests bit for bit. Build and run (no model, no download):
//
//   Scripts/native/reference/run_ggml_dequant_digests.sh
//
// Compiled with -ffp-contract=off so `a * b - c` is not fused, matching Swift.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "ggml-quants.h"
#include "ggml-impl.h"

static uint64_t state;
static uint64_t next64(void) {  // splitmix64
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// A finite F16 scale with a random sign and a moderate exponent.
static uint16_t scale16(void) {
    uint64_t r = next64();
    return (uint16_t)(((r & 1) << 15) | (0x1800 + ((r >> 1) % 0x1800)));
}

static void put16(uint8_t * p, uint16_t v) { p[0] = (uint8_t)(v & 0xff); p[1] = (uint8_t)(v >> 8); }

static uint64_t fnv(const float * v, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        memcpy(&bits, &v[i], 4);
        for (int b = 0; b < 4; ++b) { h ^= (bits >> (8 * b)) & 0xff; h *= 0x100000001b3ull; }
    }
    return h;
}

#define NB 64
static uint8_t blocks[NB * 256];
static float out[NB * 256];

// Fill NB blocks of `bytes` bytes; d (and dmin) fields at `d0`, `d1` (-1: none).
static void fill(uint64_t seed, int bytes, int d0, int d1) {
    state = seed;
    for (int i = 0; i < NB * bytes; ++i) blocks[i] = (uint8_t)(next64() & 0xff);
    for (int b = 0; b < NB; ++b) {
        put16(blocks + b * bytes + d0, scale16());
        if (d1 >= 0) put16(blocks + b * bytes + d1, scale16());
    }
}

// `n` finite F16 (or BF16) bit patterns from scale16().
static void fill16(uint64_t seed, int n) {
    state = seed;
    for (int i = 0; i < n; ++i) put16(blocks + 2 * i, scale16());
}

int main(void) {
    // Unquantized types: the conversions ggml's to_float uses for them.
    fill16(8, 1024);
    for (int i = 0; i < 1024; ++i) { ggml_fp16_t h; memcpy(&h, blocks + 2 * i, 2); out[i] = GGML_FP16_TO_FP32(h); }
    printf("f16 %016llx\n", (unsigned long long) fnv(out, 1024));
    fill16(9, 1024);
    for (int i = 0; i < 1024; ++i) { ggml_bf16_t h; memcpy(&h, blocks + 2 * i, 2); out[i] = GGML_BF16_TO_FP32(h); }
    printf("bf16 %016llx\n", (unsigned long long) fnv(out, 1024));
    state = 10;
    for (int i = 0; i < 1024; ++i) {
        uint32_t bits = (uint32_t)(next64() & 0x807FFFFFull) | 0x3F000000u;
        memcpy(blocks + 4 * i, &bits, 4);
        memcpy(&out[i], blocks + 4 * i, 4);
    }
    printf("f32 %016llx\n", (unsigned long long) fnv(out, 1024));

    fill(1, 34, 0, -1);   dequantize_row_q8_0((const block_q8_0 *)blocks, out, NB * 32);
    printf("q8_0 %016llx\n", (unsigned long long) fnv(out, NB * 32));
    fill(2, 144, 0, 2);   dequantize_row_q4_K((const block_q4_K *)blocks, out, NB * 256);
    printf("q4_K %016llx\n", (unsigned long long) fnv(out, NB * 256));
    fill(3, 176, 0, 2);   dequantize_row_q5_K((const block_q5_K *)blocks, out, NB * 256);
    printf("q5_K %016llx\n", (unsigned long long) fnv(out, NB * 256));
    fill(4, 210, 208, -1); dequantize_row_q6_K((const block_q6_K *)blocks, out, NB * 256);
    printf("q6_K %016llx\n", (unsigned long long) fnv(out, NB * 256));
    fill(5, 136, 0, -1);  dequantize_row_iq4_xs((const block_iq4_xs *)blocks, out, NB * 256);
    printf("iq4_xs %016llx\n", (unsigned long long) fnv(out, NB * 256));
    fill(6, 98, 0, -1);   dequantize_row_iq3_xxs((const block_iq3_xxs *)blocks, out, NB * 256);
    printf("iq3_xxs %016llx\n", (unsigned long long) fnv(out, NB * 256));
    fill(7, 110, 0, -1);  dequantize_row_iq3_s((const block_iq3_s *)blocks, out, NB * 256);
    printf("iq3_s %016llx\n", (unsigned long long) fnv(out, NB * 256));
    return 0;
}
