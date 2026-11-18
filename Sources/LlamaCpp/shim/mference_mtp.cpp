// MTP speculative decoding for the engine, through llama.cpp's own
// common_speculative (`draft-mtp`). The draft context is created as
// common_speculative_init_result creates it, and each step follows
// examples/speculative-simple/speculative-simple.cpp at the pinned commit:
// draft from the MTP head, evaluate [id_last, draft...] on the target, accept
// with the target's sampler (common_sampler_sample_and_accept_n), and restore a
// state checkpoint on partial acceptance when the target memory cannot drop a
// partial sequence (hybrid recurrent models such as qwen35). Not part of
// upstream llama.cpp.

#include "mference_mtp.h"

#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <memory>
#include <vector>

struct mference_mtp {
    llama_model * model = nullptr;
    llama_context * ctx_tgt = nullptr;
    llama_context_ptr ctx_dft;

    common_params_speculative params_spec;
    common_speculative_ptr spec;
    common_sampler_ptr smpl;

    int32_t n_batch = 2048;
    bool use_ckpt_tgt = false;
    bool use_ckpt_dft = false;

    // Generation state, as the locals of speculative-simple's main().
    llama_seq_id seq_id = 0;
    llama_token id_last = LLAMA_TOKEN_NULL;
    llama_tokens prompt_tgt;
    llama_tokens draft;
    common_prompt_checkpoint ckpt;
    int n_past = 0;
};

extern "C" struct mference_mtp_params mference_mtp_default_params(void) {
    common_params_speculative_draft draft;
    struct mference_mtp_params params;
    params.n_draft_max = draft.n_max;
    params.n_batch = 2048;
    params.n_ubatch = 512;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    return params;
}

extern "C" struct mference_mtp_sampling mference_mtp_default_sampling(void) {
    common_params_sampling sampling;
    struct mference_mtp_sampling params;
    params.temp = sampling.temp;
    params.top_k = sampling.top_k;
    params.top_p = sampling.top_p;
    params.min_p = sampling.min_p;
    params.penalty_last_n = sampling.penalty_last_n;
    params.penalty_repeat = sampling.penalty_repeat;
    params.penalty_freq = sampling.penalty_freq;
    params.penalty_present = sampling.penalty_present;
    params.seed = sampling.seed;
    return params;
}

extern "C" mference_mtp * mference_mtp_init(struct llama_model * model,
                                            struct llama_context * ctx_tgt,
                                            struct mference_mtp_params params) {
    if (model == nullptr || ctx_tgt == nullptr || llama_model_n_layer_nextn(model) <= 0) {
        return nullptr;
    }
    auto mtp = std::make_unique<mference_mtp>();
    mtp->model = model;
    mtp->ctx_tgt = ctx_tgt;
    mtp->n_batch = params.n_batch;

    // The MTP context, as common_speculative_init_result makes it: the target's
    // context parameters, context type MTP, the target's n_ctx, no recurrent
    // snapshots, and the target as the source of the hidden states.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = llama_n_ctx(ctx_tgt);
    cparams.n_batch = (uint32_t) params.n_batch;
    cparams.n_ubatch = (uint32_t) params.n_ubatch;
    cparams.n_seq_max = 1;
    cparams.n_threads = params.n_threads;
    cparams.n_threads_batch = params.n_threads_batch;
    cparams.no_perf = true;
    cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cparams.n_rs_seq = 0;
    cparams.ctx_other = ctx_tgt;
    mtp->ctx_dft.reset(llama_init_from_model(model, cparams));
    if (!mtp->ctx_dft) {
        return nullptr;
    }

    mtp->params_spec.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    mtp->params_spec.draft.n_max = std::max(1, params.n_draft_max);
    mtp->params_spec.draft.ctx_tgt = ctx_tgt;
    mtp->params_spec.draft.ctx_dft = mtp->ctx_dft.get();

    mtp->use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    mtp->use_ckpt_dft = common_context_can_seq_rm(mtp->ctx_dft.get()) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    return mtp.release();
}

extern "C" void mference_mtp_free(mference_mtp * mtp) {
    delete mtp;
}

extern "C" struct llama_context * mference_mtp_draft_context(mference_mtp * mtp) {
    return mtp ? mtp->ctx_dft.get() : nullptr;
}

extern "C" int32_t mference_mtp_begin(mference_mtp * mtp, const llama_token * prompt, int32_t n_prompt,
                                      struct mference_mtp_sampling sampling) {
    if (mtp == nullptr || prompt == nullptr || n_prompt < 1) {
        return -1;
    }
    common_params_sampling s;
    s.temp = sampling.temp;
    s.top_k = sampling.top_k;
    s.top_p = sampling.top_p;
    s.min_p = sampling.min_p;
    s.penalty_last_n = sampling.penalty_last_n;
    s.penalty_repeat = sampling.penalty_repeat;
    s.penalty_freq = sampling.penalty_freq;
    s.penalty_present = sampling.penalty_present;
    s.seed = sampling.seed;
    s.no_perf = true;
    llama_memory_clear(llama_get_memory(mtp->ctx_tgt), true);
    llama_memory_clear(llama_get_memory(mtp->ctx_dft.get()), true);

    mtp->spec.reset(common_speculative_init(mtp->params_spec, 1));
    mtp->smpl.reset(common_sampler_init(mtp->model, s));
    if (!mtp->spec || !mtp->smpl) {
        return -1;
    }
    // The prompt goes through the target's sampler, as in llama-server.
    for (int32_t i = 0; i < n_prompt; ++i) {
        common_sampler_accept(mtp->smpl.get(), prompt[i], false);
    }

    // Eval the prompt (all but its last token) on the target and feed it to the
    // speculative implementation, in n_batch chunks.
    for (int32_t start = 0; start < n_prompt - 1; start += mtp->n_batch) {
        const int32_t end = std::min(n_prompt - 1, start + mtp->n_batch);
        common_batch batch_prompt(mtp->ctx_tgt);
        for (int32_t i = start; i < end; ++i) {
            batch_prompt.add(prompt[i], i, mtp->seq_id, false);
        }
        const int32_t status = llama_process(mtp->ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get());
        if (status != 0) {
            return status;
        }
        if (!common_speculative_process(mtp->spec.get(), batch_prompt)) {
            return -1;
        }
    }

    // note: keep the last token separate!
    mtp->id_last = prompt[n_prompt - 1];
    mtp->prompt_tgt.assign(prompt, prompt + n_prompt - 1);
    mtp->prompt_tgt.reserve(llama_n_ctx(mtp->ctx_tgt));
    mtp->n_past = n_prompt - 1;
    mtp->draft.clear();
    mtp->ckpt = common_prompt_checkpoint();

    common_speculative_begin(mtp->spec.get(), mtp->seq_id, mtp->prompt_tgt);
    return 0;
}

extern "C" int32_t mference_mtp_step(mference_mtp * mtp, int32_t n_remaining,
                                     llama_token * out, int32_t capacity, int32_t * n_drafted) {
    if (mtp == nullptr || !mtp->spec || out == nullptr || n_remaining < 1
        || capacity < mtp->params_spec.draft.n_max + 1) {
        return -1;
    }
    llama_context * ctx_tgt = mtp->ctx_tgt;
    llama_context * ctx_dft = mtp->ctx_dft.get();
    const llama_seq_id seq_id = mtp->seq_id;
    common_batch batch_tgt(ctx_tgt);

    if (n_drafted) {
        *n_drafted = 0;
    }
    // Loops only to re-evaluate a partially accepted draft after restoring the
    // checkpoint; every other path returns accepted tokens.
    while (true) {
        if (mtp->draft.empty()) {
            mtp->ckpt.update_pos(
                    mtp->prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (mtp->use_ckpt_dft) {
                mtp->ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // the max draft that fits the remaining context and generation budget
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - mtp->n_past - 2;
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
            n_draft_max = std::max(n_draft_max, 0);

            common_speculative_get_draft_params(mtp->spec.get(), seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .pos0       = */ mtp->n_past,
                /* .id_last    = */ mtp->id_last,
                /* .prompt     = */ &mtp->prompt_tgt,
                /* .result     = */ &mtp->draft,
            };
            common_speculative_draft(mtp->spec.get());
            if (n_drafted) {
                *n_drafted = (int32_t) mtp->draft.size();
            }

            // checkpoint the target before evaluating the draft
            if (!mtp->draft.empty() && mtp->use_ckpt_tgt) {
                mtp->ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // reset the draft context to the checkpoint before verification
            if (mtp->use_ckpt_dft) {
                if (!mtp->ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                    return -1;
                }
            }
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, mtp->ckpt.pos_max + 1, -1);
        } else if (mtp->use_ckpt_tgt && mtp->ckpt.empty()) {
            return -1;
        }
        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        batch_tgt.clear();
        batch_tgt.add(mtp->id_last, mtp->n_past++, seq_id, true);
        for (size_t i = 0; i < mtp->draft.size(); ++i) {
            batch_tgt.add(mtp->draft[i], mtp->n_past + i, seq_id, true);
        }
        const int32_t status = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
        if (status != 0) {
            return status == 2 ? -2 : (status < 0 ? status : -status);
        }
        if (!common_speculative_process(mtp->spec.get(), batch_tgt)) {
            return -1;
        }

        common_sampler_ptr smpl_save;
        if (mtp->use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(mtp->smpl.get()));
        }
        const size_t n_draft = mtp->draft.size();

        auto ids = common_sampler_sample_and_accept_n(mtp->smpl.get(), ctx_tgt, mtp->draft);
        if (ids.empty()) {
            return -1;
        }

        // partial acceptance without partial seq_rm: restore the checkpoint and
        // make the accepted tokens the draft to re-evaluate
        if (mtp->use_ckpt_tgt && ids.size() - 1 < n_draft) {
            mtp->draft = std::move(ids);
            if (!mtp->ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                return -1;
            }
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, mtp->ckpt.pos_max + 1, -1);
            if (!mtp->ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                return -1;
            }
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, mtp->ckpt.pos_max + 1, -1);
            mtp->prompt_tgt.resize(mtp->ckpt.n_tokens);
            mtp->smpl = std::move(smpl_save);
            mtp->n_past = (int) mtp->prompt_tgt.size();
            continue;
        }

        common_speculative_accept(mtp->spec.get(), seq_id, (uint16_t) (ids.size() - 1));
        mtp->n_past += ids.size() - 1;

        const int32_t n_out = (int32_t) std::min<size_t>(ids.size(), (size_t) capacity);
        for (size_t i = 0; i < ids.size(); ++i) {
            mtp->prompt_tgt.push_back(mtp->id_last);
            mtp->id_last = ids[i];
            if (i < (size_t) capacity) {
                out[i] = ids[i];
            }
        }
        mtp->draft.clear();

        // clear the memory of any extra tokens
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, mtp->n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, mtp->n_past, -1);
        return n_out;
    }
}
