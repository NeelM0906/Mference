import Metal
import Testing

@testable import Mference

extension RawCompletionLoopTests {
    /// Constant ranked logits make history effects observable as token IDs.
    /// The same producer covers reset/resume, scalar/chunked prefill, and the
    /// fused-head guard without loading any checkpoint weights.
    final class HistorySamplingProducer: ContinuableLogitProducer, ChunkedPrefillRunner,
        FusedHeadLogitProducer, @unchecked Sendable
    {
        let vocabSize: Int
        let ranked: [Int32]
        let usesFusedGreedyHead: Bool
        var lastGreedyToken: UInt32 { UInt32(ranked[0]) }
        private(set) var continuationPosition: Int
        private(set) var resets = 0
        private(set) var produces = 0
        private(set) var outputModes: [PrefillOutputMode] = []

        init(vocabSize: Int, ranked: [Int32], cached: Int = 0, fused: Bool = false) {
            self.vocabSize = vocabSize
            self.ranked = ranked
            self.continuationPosition = cached
            self.usesFusedGreedyHead = fused
        }

        func reset() {
            resets += 1
            continuationPosition = 0
        }

        func prepareForContinuation(expectedPosition: Int) throws {
            #expect(continuationPosition == expectedPosition)
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            #expect(position == continuationPosition)
            produces += 1
            continuationPosition += 1
            writeLogits(logits)
        }

        func prefillChunked(tokens: ArraySlice<Int32>, startPosition: Int,
                            outputMode: PrefillOutputMode, config: PrefillRuntimeConfig,
                            into logits: MTLBuffer, onProgress: (Int) -> Void) async throws -> PrefillResult {
            #expect(startPosition == continuationPosition)
            outputModes.append(outputMode)
            continuationPosition += tokens.count
            writeLogits(logits)
            onProgress(tokens.count)
            return PrefillResult(newPosition: continuationPosition,
                                 seed: usesFusedGreedyHead ? .greedyToken(lastGreedyToken) : .logitsWritten)
        }

        private func writeLogits(_ logits: MTLBuffer) {
            let pointer = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for i in 0..<vocabSize { pointer[i] = -10 }
            for (rank, token) in ranked.enumerated() {
                pointer[Int(token)] = Float16(4 - Float(rank) * 0.25)
            }
        }
    }

    @Test(arguments: [false, true], [false, true])
    func presenceUsesCachedNewPromptAndGeneratedHistory(resume: Bool, chunked: Bool) async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load()
        let ids = try ["a", "b", "c", "d"].map {
            try #require(tokenizer.encode($0, addBOS: false).first)
        }
        try #require(Set(ids).count == 4)
        let prompt = [ids[0], ids[0], ids[1]]
        let cached = resume ? 2 : 0
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize,
                                               logitSoftcap: 0)
        for presence: Float in [0, 1.5] {
            let producer = HistorySamplingProducer(vocabSize: tokenizer.vocabSize, ranked: ids,
                                                    cached: cached)
            var generated: [Int32] = []
            let result = try await runRawCompletion(
                producer: producer, tokenizer: tokenizer, promptIds: prompt,
                config: GenerationConfig(maxNewTokens: 2, temperature: 0, presencePenalty: presence),
                context: context, scratch: scratch,
                prefillConfig: chunked ? .defaultChunked : .off,
                start: resume ? .resume(cachedPromptTokens: cached) : .reset
            ) { event in
                if case .token(_, let id, _) = event { generated.append(id) }
            }
            // Both prompt tokens must be penalized before the first draw,
            // and the first generated token must be penalized before the next.
            let expected = presence == 0 ? [ids[0], ids[0]] : [ids[2], ids[3]]
            #expect(generated == expected)
            #expect(result.cachedPromptTokens == cached)
            #expect(result.computedPrefillTokens == prompt.count - cached)
            #expect(producer.resets == (resume ? 0 : 1))
            #expect(producer.outputModes == (chunked ? [.logits] : []))
        }
    }

    @Test(arguments: [false, true])
    func presenceExcludesFusedGreedyBeforeProducerMutation(chunked: Bool) async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load()
        let token = try #require(tokenizer.encode("a", addBOS: false).first)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        for presence: Float in [-1.5, 1.5] {
            let producer = HistorySamplingProducer(vocabSize: tokenizer.vocabSize,
                                                    ranked: [token], fused: true)
            await #expect(throws: PrefillError.unsupportedPrefillSeed(
                "the fused-head producer cannot serve this sampling configuration; use a logits head")) {
                _ = try await runRawCompletion(
                    producer: producer, tokenizer: tokenizer, promptIds: [token],
                    config: GenerationConfig(maxNewTokens: 1, temperature: 0, presencePenalty: presence),
                    context: context, scratch: scratch,
                    prefillConfig: chunked ? .defaultChunked : .off) { _ in }
            }
            #expect(producer.resets == 0)
            #expect(producer.produces == 0)
            #expect(producer.outputModes.isEmpty)
        }
        for config in [GenerationConfig(maxNewTokens: 1, temperature: 0),
                       GenerationConfig(maxNewTokens: 1, temperature: 0, presencePenalty: 0)] {
            let producer = HistorySamplingProducer(vocabSize: tokenizer.vocabSize,
                                                    ranked: [token], fused: true)
            var generated: [Int32] = []
            _ = try await runRawCompletion(
                producer: producer, tokenizer: tokenizer, promptIds: [token], config: config,
                context: context, scratch: scratch,
                prefillConfig: chunked ? .defaultChunked : .off
            ) { event in
                if case .token(_, let id, _) = event { generated.append(id) }
            }
            #expect(generated == [token])
            #expect(producer.resets == 1)
            #expect(producer.outputModes == (chunked ? [.greedyIfAvailable] : []))
        }
    }

    @Test func presenceRejectsGreedyChunkedSeedOnLogitsPath() async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load()
        let token = try #require(tokenizer.encode("a", addBOS: false).first)
        let producer = ChunkedTestProducer(vocabSize: tokenizer.vocabSize, firstToken: token,
                                           seed: .greedyToken(UInt32(token)))
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        await #expect(throws: PrefillError.self) {
            _ = try await runRawCompletion(
                producer: producer, tokenizer: tokenizer, promptIds: [token],
                config: GenerationConfig(maxNewTokens: 1, temperature: 0, presencePenalty: 1.5),
                context: context, scratch: scratch, prefillConfig: .defaultChunked) { _ in }
        }
        #expect(producer.lastOutputMode == .logits)
        #expect(producer.produceCalls == 0)
    }
}
