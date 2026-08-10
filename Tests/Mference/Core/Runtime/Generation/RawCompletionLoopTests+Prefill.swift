import Testing

@testable import Mference

extension RawCompletionLoopTests {
    @Test func prefillProgressCoversEveryPromptToken() async throws {
        let tokenizer = try await MFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let promptIDs = tokenizer.encode("one two three", addBOS: true)
        let (collected, result) = try await runLoop(
            seq: [tokenA],
            end: tokenizer.eosID,
            prompt: "one two three",
            config: GenerationConfig(maxNewTokens: 4, temperature: 0))

        #expect(result.prefillTokens == promptIDs.count)
        #expect(collected.prefills.count == promptIDs.count)
        #expect(collected.prefills.last?.0 == promptIDs.count)
        #expect(collected.prefills.allSatisfy { $0.1 == promptIDs.count })
    }

    @Test func disabledChunkedPrefillUsesScalarReplay() async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load()
        let promptIDs = tokenizer.encode("one two three", addBOS: true)
        let producer = CountingProducer(
            vocabSize: tokenizer.vocabSize,
            step: automaton([], end: tokenizer.eosID))
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        _ = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 4, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off) { _ in }

        #expect(producer.resetCalls == 1)
        #expect(producer.produceCalls == promptIDs.count)
    }

    @Test func headlessSequentialPrefillSkipsOnlyNonfinalTokensAfterReset() async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load()
        let prompt = tokenizer.encode("one two three", addBOS: true)
        let producer = HeadlessContinuationProducer(vocabSize: tokenizer.vocabSize,
                                                    terminalToken: tokenizer.eosID)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        var progress: [(Int, Int)] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: prompt,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off
        ) { event in
            if case .prefill(let done, let total) = event {
                progress.append((done, total))
            }
        }

        #expect(producer.resetCalls == 1)
        #expect(producer.headlessPositions == Array(0..<(prompt.count - 1)))
        #expect(producer.exactPrefillPositions == [prompt.count - 1])
        #expect(producer.producePositions.isEmpty)
        #expect(producer.continuationPosition == prompt.count)
        #expect(progress.map(\.0) == Array(1...prompt.count))
        #expect(progress.allSatisfy { $0.1 == prompt.count })
        #expect(result.kvPosition == prompt.count)
    }

    @Test func headlessSequentialPrefillResumesWithMultipleUncachedTokens() async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load()
        let prompt = tokenizer.encode("one two three four", addBOS: true)
        let cached = prompt.count - 2
        let producer = HeadlessContinuationProducer(vocabSize: tokenizer.vocabSize,
                                                    terminalToken: tokenizer.eosID,
                                                    position: cached)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        _ = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: prompt,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off,
            start: .resume(cachedPromptTokens: cached)
        ) { _ in }

        #expect(producer.resetCalls == 0)
        #expect(producer.prepareCalls == [cached])
        #expect(producer.headlessPositions == [cached])
        #expect(producer.exactPrefillPositions == [prompt.count - 1])
        #expect(producer.producePositions.isEmpty)
        #expect(producer.continuationPosition == prompt.count)
    }

    @Test func headlessSequentialPrefillResumesOneUncachedTokenWithLogits() async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load()
        let prompt = tokenizer.encode("one two three four", addBOS: true)
        let cached = prompt.count - 1
        let producer = HeadlessContinuationProducer(vocabSize: tokenizer.vocabSize,
                                                    terminalToken: tokenizer.eosID,
                                                    position: cached)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        _ = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: prompt,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off,
            start: .resume(cachedPromptTokens: cached)
        ) { _ in }

        #expect(producer.prepareCalls == [cached])
        #expect(producer.headlessPositions.isEmpty)
        #expect(producer.exactPrefillPositions == [prompt.count - 1])
        #expect(producer.producePositions.isEmpty)
        #expect(producer.continuationPosition == prompt.count)
    }
}
