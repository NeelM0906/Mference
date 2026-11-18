// Multi-token prediction (MTP) speculative decoding through llama.cpp's own
// `common_speculative` (`draft-mtp`), as a small C API for Swift. The loop is
// upstream's `examples/speculative-simple` at the pinned commit, so a request
// decodes as that tool decodes it; see mference_mtp.cpp. Not part of upstream
// llama.cpp.
#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mference_mtp mference_mtp;

struct mference_mtp_params {
    int32_t  n_draft_max;      // tokens drafted per step (upstream default 3)
    int32_t  n_batch;          // logical batch for the prompt (the target context's)
    int32_t  n_ubatch;         // physical batch
    int32_t  n_threads;
    int32_t  n_threads_batch;
};

// The target's sampler for one generation, as common_params_sampling
// (temp <= 0 is greedy).
struct mference_mtp_sampling {
    float    temp;
    int32_t  top_k;            // <= 0 disables
    float    top_p;            // 1 disables
    float    min_p;            // 0 disables
    int32_t  penalty_last_n;   // 0 disables, -1 is the context
    float    penalty_repeat;   // 1 disables
    float    penalty_freq;
    float    penalty_present;
    uint32_t seed;
};

struct mference_mtp_params mference_mtp_default_params(void);
struct mference_mtp_sampling mference_mtp_default_sampling(void);

// Creates the MTP draft context against `ctx_tgt`, whose model was loaded with
// `load_mtp`. NULL if the model has no MTP head or the context fails.
mference_mtp * mference_mtp_init(struct llama_model * model,
                                 struct llama_context * ctx_tgt,
                                 struct mference_mtp_params params);

void mference_mtp_free(mference_mtp * mtp);

// The MTP draft context (for an abort callback); owned by `mtp`.
struct llama_context * mference_mtp_draft_context(mference_mtp * mtp);

// Starts a generation from an empty memory: clears both contexts, evaluates
// prompt[0 .. n_prompt - 1) on the target and the MTP head in n_batch chunks,
// and keeps the last prompt token for the first step. Returns 0, or the
// failing llama_decode status (2: aborted).
int32_t mference_mtp_begin(mference_mtp * mtp, const llama_token * prompt, int32_t n_prompt,
                           struct mference_mtp_sampling sampling);

// One draft and verify step. Writes the tokens accepted by the target's
// sampler (at least one) to `out`, whose capacity must be at least
// n_draft_max + 1. `n_remaining` (> 0) is the generation budget left, which
// bounds the draft. `n_drafted` receives the number of tokens drafted.
// Returns the number of tokens written, or a negative llama_decode status
// (-2: aborted) or -1 on failure.
int32_t mference_mtp_step(mference_mtp * mtp, int32_t n_remaining,
                          llama_token * out, int32_t capacity, int32_t * n_drafted);

#ifdef __cplusplus
}
#endif
