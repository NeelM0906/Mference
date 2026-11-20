import Foundation
import LlamaCpp
import Mference

/// A llama.cpp sampler chain built from a `GenerationConfig`, in the order
/// llama.cpp's own `common_sampler` uses (penalties, top-k, top-p, min-p,
/// temperature, then the final pick), so a request samples as it would
/// through `llama-server` with the same settings. Temperature 0 picks the
/// argmax (`llama_sampler_init_greedy`), as `common_sampler` does.
///
/// Samplers llama.cpp includes by default but that are neutral at their
/// defaults (DRY, top-n-sigma, typical-p, XTC) are left out; they change
/// nothing when disabled.
final class LlamaSampler: @unchecked Sendable {
    private var chain: UnsafeMutablePointer<llama_sampler>?

    /// The seed used when the request gives none, as llama.cpp's
    /// `LLAMA_DEFAULT_SEED` does: a fresh random one per request.
    static func resolvedSeed(_ seed: UInt64?) -> UInt32 {
        seed.map { UInt32(truncatingIfNeeded: $0) } ?? UInt32.random(in: 0..<UInt32.max)
    }

    /// `contextSize` stands in for a `repeatLastN` of -1 ("all history"),
    /// as llama.cpp's own argument parsing maps `--repeat-last-n -1`.
    /// `suppressedTokens` are masked out first, as `common_sampler` does with
    /// the GGUF's suppress list.
    init(config: GenerationConfig, vocabularySize: Int32, contextSize: Int32,
         suppressedTokens: [Int32] = []) {
        var params = llama_sampler_chain_default_params()
        params.no_perf = true
        let chain = llama_sampler_chain_init(params)!
        if !suppressedTokens.isEmpty {
            let biases = suppressedTokens.map { llama_logit_bias(token: $0, bias: -Float.infinity) }
            llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(
                vocabularySize, Int32(biases.count), biases))
        }
        let penalised = config.repeatLastN != 0
            && (config.repetitionPenalty != 1 || config.presencePenalty != 0 || config.frequencyPenalty != 0)
        if penalised {
            // No more history than the context can exist, so a larger window
            // penalises the same tokens; clamping also keeps the Int32
            // conversion from trapping and llama.cpp from sizing its history
            // buffer by the request.
            let window = config.repeatLastN < 0
                ? contextSize : Int32(min(config.repeatLastN, Int(contextSize)))
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                vocabularySize, window, config.repetitionPenalty,
                config.frequencyPenalty, config.presencePenalty))
        }
        if config.temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            if let topK = config.topK, topK > 0 {
                // Top-k at or above the vocabulary keeps every token.
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(
                    Int32(min(topK, Int(vocabularySize)))))
            }
            if let topP = config.topP, topP < 1 {
                llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1))
            }
            if config.minP > 0 {
                llama_sampler_chain_add(chain, llama_sampler_init_min_p(config.minP, 1))
            }
            llama_sampler_chain_add(chain, llama_sampler_init_temp(config.temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(Self.resolvedSeed(config.seed)))
        }
        self.chain = chain
    }

    /// The same settings for llama.cpp's `common_sampler`, which the MTP loop
    /// verifies drafts with (same order and neutral thresholds as above).
    static func mtpSampling(_ config: GenerationConfig, contextSize: Int32) -> mference_mtp_sampling {
        var sampling = mference_mtp_default_sampling()
        sampling.temp = config.temperature
        sampling.top_k = config.topK.map { Int32(min($0, Int(Int32.max))) } ?? 0
        sampling.top_p = config.topP ?? 1
        sampling.min_p = config.minP
        let penalised = config.repeatLastN != 0
            && (config.repetitionPenalty != 1 || config.presencePenalty != 0 || config.frequencyPenalty != 0)
        sampling.penalty_last_n = penalised
            ? (config.repeatLastN < 0 ? contextSize : Int32(min(config.repeatLastN, Int(contextSize)))) : 0
        sampling.penalty_repeat = config.repetitionPenalty
        sampling.penalty_freq = config.frequencyPenalty
        sampling.penalty_present = config.presencePenalty
        sampling.seed = resolvedSeed(config.seed)
        return sampling
    }

    /// Samples from the logits of output `index` of the last decode, and
    /// records the token for the penalty window.
    func sample(context: OpaquePointer, index: Int32 = -1) -> Int32 {
        llama_sampler_sample(chain, context, index)
    }

    /// Feeds prompt tokens to the penalty window without sampling, as
    /// `common_sampler` does for the prompt.
    func accept(_ token: Int32) {
        llama_sampler_accept(chain, token)
    }

    deinit {
        if let chain { llama_sampler_free(chain) }
    }
}
