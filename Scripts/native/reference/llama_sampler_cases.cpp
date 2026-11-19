// Runs llama.cpp's own sampler chain (src/llama-sampler.cpp at the pin), built
// the way MferenceEngine's LlamaSampler builds it, over deterministic random
// logits, histories and seeds, and prints the chosen tokens as a Swift array
// literal. ReferenceSamplerTests regenerates the same inputs and checks that
// Native.ReferenceSampler chooses the same tokens. No model, no download:
//
//   Scripts/native/reference/run_llama_sampler_cases.sh
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "llama.h"

static uint64_t state;
static uint64_t next64() {  // splitmix64, as RefTest.SplitMix64
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float unit() { return (float)((double)(next64() >> 11) / 9007199254740992.0 * 2.0 - 1.0); }

int main() {
    const float temps[] = {0.0f, 0.3f, 0.7f, 1.0f, 1.5f};
    const int topKs[] = {0, 1, 5, 40, 300};
    const float topPs[] = {1.0f, 0.5f, 0.9f, 0.95f};
    const float minPs[] = {0.0f, 0.05f, 0.2f};
    const float repeats[] = {1.0f, 1.1f, 1.3f};
    const float freqs[] = {0.0f, 0.2f};
    const float presents[] = {0.0f, 0.5f};
    const int lastNs[] = {0, 16, 64};
    const int steps = 4, historyLength = 24;
    printf("    // case: (inputSeed, vocab, temperature, topK, topP, minP, repeat, frequency, presence, lastN, bias, drawSeed, tokens)\n");
    printf("    static let llamaCppCases: [Case] = [\n");
    for (int c = 0; c < 240; ++c) {
        state = 0xC0FFEE00ull + c;
        uint64_t pick = next64();
        float temp = temps[pick % 5]; pick /= 5;
        int topK = topKs[pick % 5]; pick /= 5;
        float topP = topPs[pick % 4]; pick /= 4;
        float minP = minPs[pick % 3]; pick /= 3;
        float rep = repeats[pick % 3]; pick /= 3;
        float freq = freqs[pick % 2]; pick /= 2;
        float pres = presents[pick % 2]; pick /= 2;
        int lastN = lastNs[pick % 3]; pick /= 3;
        int bias = (int)(pick % 3);
        int vocab = (c % 20 == 19) ? 248320 : 2000;
        uint32_t drawSeed = (uint32_t)(next64() & 0x7fffffffu);
        uint64_t inputSeed = 0xABCD0000ull + c;

        state = inputSeed;
        std::vector<float> logits(vocab);
        for (auto & l : logits) l = unit() * 6.0f;
        std::vector<int32_t> history(historyLength);
        for (auto & t : history) t = (int32_t)(next64() % (uint64_t)vocab);

        llama_sampler_chain_params params = llama_sampler_chain_default_params();
        params.no_perf = true;
        llama_sampler * chain = llama_sampler_chain_init(params);
        if (bias != 0) {
            llama_logit_bias b[2] = {{history[0], bias == 1 ? -INFINITY : 2.0f}, {7, bias == 1 ? 1.5f : -3.0f}};
            llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(vocab, 2, b));
        }
        bool penalised = lastN != 0 && (rep != 1.0f || freq != 0.0f || pres != 0.0f);
        if (penalised) llama_sampler_chain_add(chain, llama_sampler_init_penalties(vocab, lastN, rep, freq, pres));
        if (temp <= 0) {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        } else {
            if (topK > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(topK < vocab ? topK : vocab));
            if (topP < 1) llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1));
            if (minP > 0) llama_sampler_chain_add(chain, llama_sampler_init_min_p(minP, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(drawSeed));
        }
        for (int32_t t : history) llama_sampler_accept(chain, t);
        printf("        (0x%llx, %d, %g, %d, %g, %g, %g, %g, %g, %d, %d, %u, [", (unsigned long long) inputSeed, vocab,
               temp, topK, topP, minP, rep, freq, pres, lastN, bias, drawSeed);
        std::vector<llama_token_data> cur(vocab);
        for (int s = 0; s < steps; ++s) {
            for (int i = 0; i < vocab; ++i) cur[i] = {i, logits[i], 0.0f};
            llama_token_data_array arr = {cur.data(), cur.size(), -1, false};
            llama_sampler_apply(chain, &arr);
            int32_t tok = arr.data[arr.selected].id;
            llama_sampler_accept(chain, tok);
            printf("%s%d", s ? ", " : "", tok);
        }
        printf("]),\n");
        llama_sampler_free(chain);
    }
    printf("    ]\n");
    return 0;
}
