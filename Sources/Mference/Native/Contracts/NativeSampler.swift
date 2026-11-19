import Metal

extension Native {
    /// Sampling settings, the `GenerationConfig` subset a sampler uses.
    /// Semantics are llama.cpp `common_sampler`'s at the pin
    /// (`docs/LLAMA_SAMPLING.md`).
    public struct SamplingParams: Sendable, Equatable {
        public var temperature: Float
        /// nil or <= 0 disables top-k.
        public var topK: Int?
        /// nil or >= 1 disables top-p.
        public var topP: Float?
        public var minP: Float
        public var repetitionPenalty: Float
        public var presencePenalty: Float
        public var frequencyPenalty: Float
        /// 0 disables penalties, -1 uses the whole history.
        public var repeatLastN: Int
        public var seed: UInt64?
        /// Additive logit biases applied first (llama.cpp `logit_bias`).
        public var logitBias: [Int32: Float]

        public init(temperature: Float, topK: Int?, topP: Float?, minP: Float,
                    repetitionPenalty: Float, presencePenalty: Float, frequencyPenalty: Float,
                    repeatLastN: Int, seed: UInt64?, logitBias: [Int32: Float] = [:]) {
            self.temperature = temperature
            self.topK = topK
            self.topP = topP
            self.minP = minP
            self.repetitionPenalty = repetitionPenalty
            self.presencePenalty = presencePenalty
            self.frequencyPenalty = frequencyPenalty
            self.repeatLastN = repeatLastN
            self.seed = seed
            self.logitBias = logitBias
        }

        public init(_ config: GenerationConfig) {
            self.init(temperature: config.temperature, topK: config.topK, topP: config.topP,
                      minP: config.minP, repetitionPenalty: config.repetitionPenalty,
                      presencePenalty: config.presencePenalty, frequencyPenalty: config.frequencyPenalty,
                      repeatLastN: config.repeatLastN, seed: config.seed)
        }

        /// llama.cpp ends the chain with greedy instead of a seeded draw only
        /// when temperature is exactly 0 (see `docs/LLAMA_SAMPLING.md`).
        public var isGreedy: Bool { temperature == 0 }
    }

    /// The sampler chain exactly as llama.cpp `common_sampler` builds it at
    /// the pin (`common/sampling.cpp`): logit bias, penalties, then the
    /// truncation samplers in its order (top-k, top-p, min-p for the settings
    /// `SamplingParams` carries; the others at their neutral defaults),
    /// temperature, and finally greedy (temperature 0: the first maximum in
    /// the chain's candidate order) or the seeded draw.
    ///
    /// Token choices must equal llama.cpp's on the same logits, history and
    /// seed for every setting. The existing `MferenceEngine/LlamaSampler.swift`
    /// shows the chain llama.cpp builds.
    public protocol Sampler: AnyObject {
        /// Start a generation: set params, clear history, seed the RNG.
        func reset(params: SamplingParams)
        /// Add `token` to the penalty history (prompt and generated tokens),
        /// as `common_sampler_accept`.
        func accept(_ token: Int32)
        /// Choose a token from one row of F32 logits (CPU).
        func sample(logits: UnsafeBufferPointer<Float>) -> Int32
        /// Whether `encodeChoice` is exact for the current params (greedy with
        /// neutral penalties at least). When false the scheduler reads logits
        /// back and calls `sample`.
        var canChooseOnGPU: Bool { get }
        /// Encode the choice for one logits row into `output` (one u32), so the
        /// next step can read it without a CPU round trip. Must equal
        /// `sample` on the same logits.
        func encodeChoice(_ encoder: MTLComputeCommandEncoder, logits: TensorRef,
                          output: TokenSlot) throws
    }
}
