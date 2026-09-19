import Metal
import Testing

@testable import Mference

extension RawCompletionLoopTests {
    @Test(arguments: [false, true])
    func checkpointSplitsOnlyPrefillAndPreservesAccounting(chunked: Bool) async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load(from: GemmaThinkingTests.fixtureFolder(), family: .gemma4)
        let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7]
        let producer = ContinuationProducer(vocabSize: tokenizer.vocabSize,
            terminalToken: tokenizer.eosID, position: 2)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        producer.reportsEachBatch = chunked
        var checkpoints: [Int] = []
        var progress: [Int] = []
        let result = try await runRawCompletion(producer: producer, tokenizer: tokenizer,
            promptIds: prompt, config: .init(maxNewTokens: 1, temperature: 0),
            context: context, scratch: scratch, prefillConfig: chunked ? .defaultChunked : .off,
            start: .resume(cachedPromptTokens: 2), prefillCheckpoint: (4, {
                checkpoints.append(producer.continuationPosition)
            })) { event in
                if case .prefill(let done, _) = event { progress.append(done) }
            }
        #expect(checkpoints == [4])
        #expect(producer.resetCalls == 0)
        #expect(result.kvBackedTokenIDs == prompt)
        #expect(result.cachedPromptTokens == 2)
        #expect(result.computedPrefillTokens == 5)
        #expect(result.uncommittedBoundaryTokenIDs == [tokenizer.eosID])
        #expect(progress == progress.sorted() && progress.last == prompt.count)
        #expect(result.prefillExecution?.computedTokens == 5)
        if chunked {
            #expect(producer.prefillRanges == [2..<4, 4..<7])
            #expect(result.prefillExecution?.batchedChunkSizes == [2, 3])
        }
    }

    final class ContinuationProducer: ChunkedPrefillRunner, ContinuableLogitProducer,
        @unchecked Sendable
    {
        let vocabSize: Int
        private let terminalToken: Int32
        private(set) var continuationPosition: Int
        private(set) var resetCalls = 0
        private(set) var prepareCalls: [Int] = []
        private(set) var prefillRanges: [Range<Int>] = []
        var reportedExecution: PrefillExecutionReport?
        var reportsEachBatch = false

        init(vocabSize: Int, terminalToken: Int32, position: Int) {
            self.vocabSize = vocabSize
            self.terminalToken = terminalToken
            self.continuationPosition = position
        }

        func reset() {
            resetCalls += 1
            continuationPosition = 0
        }

        func prepareForContinuation(expectedPosition: Int) throws {
            guard continuationPosition == expectedPosition else {
                throw PrefillError.prefillCursorMismatch("test cursor mismatch")
            }
            prepareCalls.append(expectedPosition)
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            guard continuationPosition == position else {
                throw PrefillError.prefillCursorMismatch("test scalar cursor mismatch")
            }
            continuationPosition += 1
            writeTerminal(to: logits)
        }

        func prefillChunked(tokens: ArraySlice<Int32>,
                            startPosition: Int,
                            outputMode: PrefillOutputMode,
                            config: PrefillRuntimeConfig,
                            into logits: MTLBuffer,
                            onProgress: (Int) -> Void) async throws -> PrefillResult {
            guard continuationPosition == startPosition else {
                throw PrefillError.prefillCursorMismatch("test prefill cursor mismatch")
            }
            prefillRanges.append(startPosition..<(startPosition + tokens.count))
            continuationPosition += tokens.count
            onProgress(tokens.count)
            writeTerminal(to: logits)
            var execution = reportedExecution
            if reportsEachBatch {
                var actual = PrefillExecutionReport()
                actual.recordBatch(tokens.count)
                execution = actual
            }
            return PrefillResult(newPosition: continuationPosition,
                                 seed: .logitsWritten, execution: execution)
        }

        private func writeTerminal(to logits: MTLBuffer) {
            let pointer = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for index in 0..<vocabSize { pointer[index] = -30 }
            pointer[Int(terminalToken)] = 30
        }
    }

    @Test(arguments: [false, true])
    func resumedChunkedPrefillUsesNonzeroStart(reportsExecution: Bool) async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load()
        let prompt = tokenizer.encode("one two three four", addBOS: true)
        let cached = prompt.count - 1
        let producer = ContinuationProducer(
            vocabSize: tokenizer.vocabSize,
            terminalToken: tokenizer.eosID,
            position: cached)
        if reportsExecution {
            var report = PrefillExecutionReport()
            report.recordBatch(1)
            producer.reportedExecution = report
        }
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        var progress: [(Int, Int)] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: prompt,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .defaultChunked,
            start: .resume(cachedPromptTokens: cached)
        ) { event in
            if case .prefill(let done, let total) = event {
                progress.append((done, total))
            }
        }

        #expect(producer.resetCalls == 0)
        #expect(producer.prepareCalls == [cached])
        #expect(producer.prefillRanges == [cached..<prompt.count])
        #expect(progress.last?.0 == prompt.count)
        #expect(progress.last?.1 == prompt.count)
        #expect(result.prefillTokens == prompt.count)
        #expect(result.cachedPromptTokens == cached)
        #expect(result.computedPrefillTokens == 1)
        // Preserve either actual work or the legacy producer's unknown state.
        #expect(result.prefillExecution == producer.reportedExecution)
        if reportsExecution {
            #expect(result.prefillExecution?.computedTokens == 1)
            #expect(result.prefillExecution?.batchedChunkSizes == [1])
        }
        #expect(result.kvPosition == prompt.count)
        #expect(result.kvBackedTokenIDs == prompt)
        #expect(result.uncommittedBoundaryTokenIDs == [tokenizer.eosID])
    }

    @Test func resumeRejectsInvalidCachedCountsBeforeMutatingProducer() async throws {
        let context = try MetalContext()
        let tokenizer = try await MFTokenizer.load()
        let prompt = tokenizer.encode("one two", addBOS: true)
        let producer = ContinuationProducer(
            vocabSize: tokenizer.vocabSize,
            terminalToken: tokenizer.eosID,
            position: 0)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        for count in [0, prompt.count, prompt.count + 1] {
            await #expect(throws: GeneratorError.self) {
                _ = try await runRawCompletion(
                    producer: producer,
                    tokenizer: tokenizer,
                    promptIds: prompt,
                    config: GenerationConfig(maxNewTokens: 1, temperature: 0),
                    context: context,
                    scratch: scratch,
                    prefillConfig: .off,
                    start: .resume(cachedPromptTokens: count)
                ) { _ in }
            }
        }

        #expect(producer.resetCalls == 0)
        #expect(producer.prepareCalls.isEmpty)
        #expect(producer.prefillRanges.isEmpty)
    }
}
