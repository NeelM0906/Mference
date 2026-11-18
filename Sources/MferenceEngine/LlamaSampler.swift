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
    init(config: GenerationConfig, vocabularySize: Int32, contextSize: Int32) {
        var params = llama_sampler_chain_default_params()
        params.no_perf = true
        let chain = llama_sampler_chain_init(params)!
        let penalised = config.repeatLastN != 0
            && (config.repetitionPenalty != 1 || config.presencePenalty != 0 || config.frequencyPenalty != 0)
        if penalised {
            let window = config.repeatLastN < 0 ? contextSize : Int32(config.repeatLastN)
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                vocabularySize, window, config.repetitionPenalty,
                config.frequencyPenalty, config.presencePenalty))
        }
        if config.temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            if let topK = config.topK, topK > 0 {
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(Int32(topK)))
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
