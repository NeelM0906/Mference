import Foundation
import Metal

/// Generation knobs threaded from the caller through the `Generator` into the
/// sampler. Pure value type; one per `generate(...)` call.
///
/// Canonical home is here (the sampler is the primary consumer); `Generator`
/// reuses the same type rather than redeclaring it.
public struct GenerationConfig: Sendable {
    public var maxNewTokens: Int
    public var temperature: Float
    public var topK: Int?                  // nil = no truncation
    public var topP: Float?                // nil = no nucleus truncation
    public var repetitionPenalty: Float
    public var presencePenalty: Float
    public var frequencyPenalty: Float
    public var repeatLastN: Int            // 0 disables penalties; -1 uses all history
    public var minP: Float
    public var seed: UInt64?               // nil = nondeterministic
    public var stopStrings: [String]
    public var extraStopTokens: Set<Int32>

    /// The sampling defaults are declared once, in `init` below (llama.cpp's
    /// built-in preset; docs/LLAMA_SAMPLING.md). The CLI and server fall back
    /// to these only for a parameter the caller omitted, so an explicit flag
    /// or request field always wins.
    public static let defaults = GenerationConfig()

    public init(maxNewTokens: Int = 256,
                temperature: Float = 0.8,
                topK: Int? = 40,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                presencePenalty: Float = 0.0,
                frequencyPenalty: Float = 0.0,
                repeatLastN: Int = 64,
                minP: Float = 0.05,
                seed: UInt64? = nil,
                stopStrings: [String] = [],
                extraStopTokens: Set<Int32> = []) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.repeatLastN = repeatLastN
        self.minP = minP
        self.seed = seed
        self.stopStrings = stopStrings
        self.extraStopTokens = extraStopTokens
    }

    public func validate() throws {
        guard presencePenalty.isFinite, (-2...2).contains(presencePenalty) else {
            throw GeneratorError.invalidGenerationConfig(
                "presencePenalty must be finite and between -2 and 2")
        }
        guard frequencyPenalty.isFinite, (-2...2).contains(frequencyPenalty) else {
            throw GeneratorError.invalidGenerationConfig("frequencyPenalty must be finite and between -2 and 2")
        }
        guard repetitionPenalty.isFinite, repetitionPenalty > 0,
              (1 / repetitionPenalty).isFinite else {
            throw GeneratorError.invalidGenerationConfig("repetitionPenalty must be finite and greater than zero")
        }
        guard repeatLastN >= -1 else {
            throw GeneratorError.invalidGenerationConfig("repeatLastN must be -1 or nonnegative")
        }
        guard minP.isFinite, (0...1).contains(minP) else {
            throw GeneratorError.invalidGenerationConfig("minP must be finite and between 0 and 1")
        }
        guard maxNewTokens > 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "maxNewTokens must be greater than zero")
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "temperature must be finite and nonnegative")
        }
        if let topK, !(1...256).contains(topK) {
            throw GeneratorError.invalidGenerationConfig(
                "topK must be between 1 and 256")
        }
        if let topP, (!topP.isFinite || topP <= 0 || topP > 1) {
            throw GeneratorError.invalidGenerationConfig(
                "topP must be greater than zero and at most one")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw GeneratorError.invalidGenerationConfig(
                "topP below one requires topK; full-vocabulary nucleus sampling is not implemented")
        }
    }

}

/// Which path a `sample(...)` call took.
enum SamplePath: Sendable, Equatable {
    case greedyGPU
    case gpuSampled
    case hostPenalty // host prepares history counts; GPU applies penalties
}

/// Turns `GenerationConfig` + a logits buffer into one token id, staying
/// GPU-resident wherever the kernels allow.
///
/// The built `sample` kernel already does temperature / top-k / top-p / min-p / seeded
/// draw / greedy argmax on GPU reading softmaxed probs, so this type's job is:
/// (1) run the softcap+softmax front-end (`logit_softcap_softmax`), (2) apply
/// repetition, frequency and presence penalties in post-softcap logit space,
/// using counts from the configured history window,
/// and (3) derive a per-position seed so a fixed `seed` is reproducible across
/// token positions.
///
/// The chosen id lands in a 1-element UInt32 buffer. The generation loop reads
/// that value after the command buffer completes.
///
/// Truncation follows llama.cpp's default order: Top-K, Top-P normalized over
/// that set, Min-P relative to its peak, then temperature and the random draw.
final class Sampler {
    private let softcap: LogitSoftcapSoftmax
    private let sampleKernel: Sample
    private let topK64Kernel: SampleTopK64
    let vocab: Int
    private let logitSoftcap: Float
    private let historyCounts: MTLBuffer
    private var countedTokens: [Int32] = []
    var diagnosticBufferBytes: UInt64 {
        UInt64(historyCounts.length + topK64Kernel.scratchBytes)
    }

    init(context: MetalContext, vocab: Int = 262_144,
                logitSoftcap: Float = 30.0) throws {
        self.softcap = try LogitSoftcapSoftmax(context: context)
        self.sampleKernel = try Sample(context: context)
        self.topK64Kernel = try SampleTopK64(context: context, vocab: vocab)
        self.vocab = vocab
        self.logitSoftcap = logitSoftcap
        guard let counts = context.device.makeBuffer(length: vocab * MemoryLayout<UInt32>.stride,
                                                     options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        memset(counts.contents(), 0, counts.length)
        self.historyCounts = counts
    }

    /// Encode the sampler onto `commandBuffer`. `logits` is FP16 [vocab],
    /// post-lm_head and pre-softcap. It is read without modification, including
    /// when applying history penalties. `probs` is a preallocated
    /// FP16 [vocab] scratch. `outToken` holds one UInt32. `position` indexes the
    /// per-position seed advance. Returns the path taken.
    @discardableResult
    func sample(commandBuffer: MTLCommandBuffer,
                       logits: MTLBuffer,
                       probs: MTLBuffer,
                       history: [Int32],
                       config: GenerationConfig,
                       position: Int,
                       outToken: MTLBuffer) -> SamplePath {
        let v = UInt32(vocab)

        let appliedPenalty = config.repeatLastN != 0 && !history.isEmpty
            && (config.repetitionPenalty != 1 || config.presencePenalty != 0 || config.frequencyPenalty != 0)
        if appliedPenalty { updateHistoryCounts(history, lastN: config.repeatLastN) }

        softcap.encode(commandBuffer: commandBuffer,
                       logits: logits, probs: probs, v: v, softcap: logitSoftcap,
                       historyCounts: appliedPenalty ? historyCounts : nil,
                       repetitionPenalty: config.repetitionPenalty,
                       frequencyPenalty: config.frequencyPenalty,
                       presencePenalty: config.presencePenalty)

        let isGreedy = config.temperature == 0
        let seed = Self.seedFor(config: config, position: position)
        if config.temperature > 0,
           let topK = config.topK, topK <= 64 {
            topK64Kernel.encode(commandBuffer: commandBuffer,
                                probs: probs,
                                outToken: outToken,
                                temperature: config.temperature,
                                topP: config.topP ?? 1.0,
                                minP: config.minP,
                                topK: UInt32(topK),
                                seed: seed)
        } else {
            sampleKernel.encode(commandBuffer: commandBuffer,
                                probs: probs, outToken: outToken, v: v,
                                temperature: isGreedy ? 0.0 : config.temperature,
                                topK: UInt32(config.topK ?? 0),
                                topP: config.topP ?? 1.0,
                                minP: config.minP,
                                seed: seed,
                                position: UInt32(position))
        }

        if appliedPenalty { return .hostPenalty }
        return isGreedy ? .greedyGPU : .gpuSampled
    }

    // MARK: - History counts (host); penalty math runs after softcap on GPU.

    private func updateHistoryCounts(_ history: [Int32], lastN: Int) {
        let counts = historyCounts.contents().bindMemory(to: UInt32.self, capacity: vocab)
        for id in countedTokens { counts[Int(id)] = 0 }
        countedTokens.removeAll(keepingCapacity: true)
        let window = lastN < 0 ? history[...] : history.suffix(lastN)
        for id in window where id >= 0 && Int(id) < vocab {
            if counts[Int(id)] == 0 { countedTokens.append(id) }
            counts[Int(id)] += 1
        }
    }

    // MARK: - Seed

    /// Deterministic per-position seed when `config.seed != nil` so a fixed seed
    /// reproduces across token positions; clock-derived (non-zero) otherwise.
    /// xorshift64 in the kernel has a fixed point at 0, so we never emit 0.
    static func seedFor(config: GenerationConfig, position: Int) -> UInt64 {
        if let s = config.seed {
            let mixed = Self.splitmix64(s &+ UInt64(bitPattern: Int64(position)))
            return mixed == 0 ? 0x9E3779B97F4A7C15 : mixed
        }
        var t = timespec()
        clock_gettime(CLOCK_MONOTONIC, &t)
        let raw = UInt64(bitPattern: Int64(t.tv_nsec)) &* 0x9E3779B97F4A7C15
            &+ UInt64(bitPattern: Int64(t.tv_sec))
        return raw == 0 ? 0x9E3779B97F4A7C15 : raw
    }

    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
