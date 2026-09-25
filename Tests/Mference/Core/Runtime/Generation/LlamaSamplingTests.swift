import Foundation
import Metal
import Testing
@testable import Mference

/// Reference contract: ggml-org/llama.cpp b23701f77d47dad9de834d59ebfcbe25c9e8b46f,
/// common/common.h and src/llama-sampler.cpp. These tests check distributions
/// and eligible tokens, not RNG identity across two different engines.
@Suite struct LlamaSamplingTests {
    private struct Rig {
        let context: MetalContext
        let sampler: Sampler
        let logits: MTLBuffer
        let probs: MTLBuffer
        let output: MTLBuffer
        let vocab: Int
        init(vocab: Int = 128, cap: Float = 0) throws {
            context = try MetalContext()
            sampler = try Sampler(context: context, vocab: vocab, logitSoftcap: cap)
            self.vocab = vocab
            logits = try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
            probs = try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
            output = try #require(context.device.makeBuffer(length: 4, options: .storageModeShared))
        }
        func draw(_ values: [Float], _ config: GenerationConfig, history: [Int32] = []) throws -> UInt32 {
            try config.validate()
            let p = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
            for i in 0..<vocab { p[i] = Float16(values[i]) }
            let command = try #require(context.queue.makeCommandBuffer())
            _ = sampler.sample(commandBuffer: command, logits: logits, probs: probs,
                history: history, config: config, position: 0, outToken: output)
            command.commit(); command.waitUntilCompleted()
            #expect(command.status == .completed)
            return output.contents().load(as: UInt32.self)
        }
    }

    @Test func defaultsMatchLlamaFallbackPreset() throws {
        let c = GenerationConfig()
        try c.validate()
        #expect(c.temperature == 0.8 && c.topK == 40 && c.topP == 0.95 && c.minP == 0.05)
        #expect(c.repetitionPenalty == 1 && c.presencePenalty == 0 && c.frequencyPenalty == 0)
        #expect(c.repeatLastN == 64)
    }

    @Test func frequencyCountsOccurrencesWhilePresenceActsOnce() throws {
        let rig = try Rig()
        var values = [Float](repeating: -10, count: 128)
        values[0] = 4; values[1] = 3.75
        #expect(try rig.draw(values, .init(temperature: 0, presencePenalty: 0.2), history: [0, 0]) == 0)
        #expect(try rig.draw(values, .init(temperature: 0, frequencyPenalty: 0.2), history: [0, 0]) == 1)
        #expect(try rig.draw(values, .init(temperature: 0, frequencyPenalty: -0.2), history: [1, 1]) == 1)
        #expect(try rig.draw(values, .init(temperature: 0, frequencyPenalty: 1, repeatLastN: 0), history: [0]) == 0)
        let old: [Int32] = [0, 0] + Array(repeating: 3, count: 64)
        #expect(try rig.draw(values, .init(temperature: 0, frequencyPenalty: 0.2), history: old) == 0)
        #expect(try rig.draw(values, .init(temperature: 0, frequencyPenalty: 0.2, repeatLastN: -1), history: old) == 1)
    }

    @Test(arguments: [Float(0), 30], [Float(0.8), 1.25])
    func penaltyProbabilitiesMatchIndependentPostSoftcapFormula(cap: Float, repetition: Float) throws {
        let rig = try Rig(cap: cap)
        var values = [Float](repeating: -10, count: 128)
        values[0] = cap == 0 ? 4 : 400
        values[1] = cap == 0 ? 4 : 300
        values[2] = -4; values[3] = 0
        let history: [Int32] = [0, 0, 1, 2, 2, 2, 3]
        let config = GenerationConfig(temperature: 0, repetitionPenalty: repetition,
            presencePenalty: -1.5, frequencyPenalty: 0.5)
        _ = try rig.draw(values, config, history: history)
        let counts = [2, 1, 3, 1]
        let reference = values.enumerated().map { i, raw -> Double in
            let z = Double(Float(Float16(raw)))
            var final = cap == 0 ? z : Double(cap) * tanh(z / Double(cap))
            if i < counts.count {
                final = final <= 0 ? final * Double(repetition) : final / Double(repetition)
                final -= Double(counts[i]) * 0.5 - 1.5
            }
            return final
        }
        let peak = reference.max()!
        let weights = reference.map { exp($0 - peak) }
        let sum = weights.reduce(0, +)
        let actual = rig.probs.contents().bindMemory(to: Float16.self, capacity: 128)
        let input = rig.logits.contents().bindMemory(to: Float16.self, capacity: 128)
        for i in 0..<128 {
            let expected = Float16(weights[i] / sum)
            #expect(abs(Float(actual[i]) - Float(expected)) <= Float(expected.ulp) * 2)
            #expect(input[i] == Float16(values[i]), "penalties must not rewrite raw model logits")
        }
    }

    @Test(arguments: [0, 4, 40, 64, 128])
    func minPFiltersBeforeTemperatureAndSupportsNoTopK(topK: Int) throws {
        let rig = try Rig()
        var values = [Float](repeating: -10, count: 128)
        values[3] = 2; values[7] = 1.5; values[9] = 1
        var unfiltered = Set<UInt32>()
        for seed in UInt64(1)...32 {
            let k = topK == 0 ? nil : topK
            let filtered = GenerationConfig(temperature: 2, topK: k, topP: 1, minP: 0.8, seed: seed)
            #expect(try rig.draw(values, filtered) == 3)
            unfiltered.insert(try rig.draw(values, .init(temperature: 2, topK: k, topP: 1, minP: 0, seed: seed)))
        }
        #expect(unfiltered.count > 1, "zero must disable Min-P")
    }

    @Test func frequencyPenaltyRequiresRealLogitsUnlessWindowDisabled() throws {
        #expect(!GenerationConfig(temperature: 0, frequencyPenalty: 1).isPureGreedy)
        #expect(GenerationConfig(temperature: 0, frequencyPenalty: 1, repeatLastN: 0).isPureGreedy)
        for value: Float in [-2.01, 2.01, .infinity, .nan] {
            #expect(throws: GeneratorError.self) { try GenerationConfig(frequencyPenalty: value).validate() }
        }
        for value: Float in [0, -1, .infinity, .nan] {
            #expect(throws: GeneratorError.self) { try GenerationConfig(repetitionPenalty: value).validate() }
        }
        #expect(throws: GeneratorError.self) { try GenerationConfig(repeatLastN: -2).validate() }
    }
}

extension RawCompletionLoopTests {
    @Test(arguments: [false, true], [false, true])
    func frequencyUsesPromptAndGeneratedCountsOnResetAndResume(resume: Bool, chunked: Bool) async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load()
        let ids = try ["a", "b", "c"].map { try #require(tokenizer.encode($0, addBOS: false).first) }
        let prompt = [ids[0], ids[0]]
        let cached = resume ? 1 : 0
        let producer = HistorySamplingProducer(vocabSize: tokenizer.vocabSize, ranked: ids, cached: cached)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize, logitSoftcap: 0)
        var generated: [Int32] = []
        let result = try await runRawCompletion(producer: producer, tokenizer: tokenizer, promptIds: prompt,
            config: .init(maxNewTokens: 2, temperature: 0, frequencyPenalty: 0.2),
            context: context, scratch: scratch, prefillConfig: chunked ? .defaultChunked : .off,
            start: resume ? .resume(cachedPromptTokens: cached) : .reset) { event in
                if case .token(_, let id, _) = event { generated.append(id) }
            }
        #expect(generated == [ids[1], ids[0]])
        #expect(result.cachedPromptTokens == cached)
        #expect(result.computedPrefillTokens == prompt.count - cached)
    }
}
