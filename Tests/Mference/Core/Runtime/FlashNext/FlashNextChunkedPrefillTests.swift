import Foundation
import Metal
import Testing
@testable import Mference

/// Production-path gates for Flash-Next chunked prefill. The synthetic install
/// uses INT4 resident projections and INT4 streamed experts, so this exercises
/// the same batched projection, QSA, GDN and expert-major MoE paths as the pinned
/// checkpoint without requiring its 175 GB install.
@Suite struct FlashNextChunkedPrefillTests {
    private static func makeRunner(
        maxContext: Int = 96,
        streamingMode: ExpertStreamingMode = .pread(slotCount: 16)
    ) throws
        -> (URL, MetalContext, Model, FlashNextForwardRunner) {
        let directory = try FlashNextToySynthetic.write()
        let context = try MetalContext()
        let config = ArchConfig.qwen38FlashNextToy()
        let model = try Model.load(directoryURL: directory,
                                   device: context.device,
                                   expecting: config,
                                   streamingMode: streamingMode)
        let runner = try FlashNextForwardRunner(model: model, context: context,
                                                maxContext: maxContext)
        return (directory, context, model, runner)
    }

    private static func logits(_ context: MetalContext, vocab: Int) throws
        -> MTLBuffer {
        guard let value = context.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return value
    }

    private static func bits(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
        let values = Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: UInt16.self, capacity: count),
            count: count))
        #expect(values.allSatisfy { Float16(bitPattern: $0).isFinite })
        #expect(values.contains { Float16(bitPattern: $0) != 0 })
        return values
    }

    private static func prompt(_ count: Int, vocab: Int) -> [Int32] {
        (0..<count).map { Int32(4 + ($0 * 37 + 11) % (vocab - 4)) }
    }

    private static func argmax(_ bits: [UInt16]) -> Int32 {
        Int32(bits.indices.max { Float16(bitPattern: bits[$0]) < Float16(bitPattern: bits[$1]) }!)
    }

    /// Different batched reductions need not round identically to GEMV. The
    /// original unaligned fixture compared NaN bit patterns; that was not a
    /// valid zero-tolerance gate. Require finite/nonzero rows, exact decisions,
    /// and at most two FP16 relative-precision units at the row's scale. This
    /// is tighter than the existing installed-model 5% bound, not an assertion
    /// that chunked prefill is an exact speculative verifier.
    private static func expectNumericalMatch(_ actual: [UInt16], _ expected: [UInt16]) {
        #expect(actual.count == expected.count)
        let a = actual.map { Float(Float16(bitPattern: $0)) }
        let e = expected.map { Float(Float16(bitPattern: $0)) }
        #expect(a.allSatisfy { $0.isFinite } && e.allSatisfy { $0.isFinite })
        let error = zip(a, e).map { abs($0 - $1) }.max() ?? .infinity
        let scale = e.map { abs($0) }.max() ?? 0
        #expect(scale > 0)
        #expect(error <= scale / 512)
        #expect(argmax(actual) == argmax(expected))
    }

    @Test func factoryDoesNotClaimExecutionBeforePrefill() throws {
        let (directory, context, model, _) = try Self.makeRunner(maxContext: 64)
        defer { try? FileManager.default.removeItem(at: directory) }
        let requested = RuntimeConfiguration(prefillEnabled: true,
                                             forceLogitsHead: true)
        let runtime = try ForwardRunnerFactory.make(
            model: model, context: context, maxContext: 64,
            runtimeConfiguration: requested)
        #expect(runtime.producer is FlashNextForwardRunner)
        #expect(runtime.producer is any ChunkedPrefillRunner)
        #expect(runtime.prefillConfig == requested.prefillConfig)
        #expect(runtime.executedPrefillMode == .unreported)
        #expect(runtime.kvStorageMode == .fp16)
    }

    private static func expectChunkedMatchesSequential(
        promptLength: Int, streamingMode: ExpertStreamingMode
    ) async throws {
        let (directory, context, _, runner) = try Self.makeRunner(
            streamingMode: streamingMode)
        defer { try? FileManager.default.removeItem(at: directory) }
        let vocab = ArchConfig.qwen38FlashNextToy().vocabSize
        let logits = try Self.logits(context, vocab: vocab)
        let prompt = Self.prompt(promptLength, vocab: vocab)
        let continuation: Int32 = 17

        for (position, token) in prompt.enumerated() {
            try await runner.produce(token: token, position: position,
                                     into: logits)
        }
        let sequentialPrompt = Self.bits(logits, count: vocab)
        try await runner.produce(token: continuation, position: prompt.count,
                                 into: logits)
        let sequentialNext = Self.bits(logits, count: vocab)
        var greedyTokens: [Int32] = []
        var greedyRows: [[UInt16]] = []
        for step in 0..<8 {
            let token = Self.argmax(Self.bits(logits, count: vocab))
            greedyTokens.append(token)
            try await runner.produce(token: token, position: prompt.count + 1 + step, into: logits)
            greedyRows.append(Self.bits(logits, count: vocab))
        }

        runner.reset()
        var progress: [Int] = []
        let result = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { progress.append($0) })
        #expect(result.newPosition == prompt.count)
        #expect(result.seed == .logitsWritten)
        #expect(result.execution?.batchedTokens == prompt.count)
        #expect(result.execution?.replayedTokens == 0)
        #expect(result.execution?.batchedChunkSizes == (prompt.count <= 32
            ? [prompt.count] : [32, prompt.count - 32]))
        #expect(progress.last == prompt.count)
        #expect(runner.continuationPosition == prompt.count)
        if case .resident = streamingMode {
            #expect(runner.deviceGroupedPrefillLayers > 0)
        } else {
            #expect(runner.deviceGroupedPrefillLayers == 0)
        }
        let chunkedPrompt = Self.bits(logits, count: vocab)
        #expect(runner.tensorOpsPrefillEncodings == 0, "toy geometry uses the portable path")
        try await runner.produce(token: continuation, position: prompt.count,
                                 into: logits)
        let chunkedNext = Self.bits(logits, count: vocab)

        Self.expectNumericalMatch(chunkedPrompt, sequentialPrompt)
        Self.expectNumericalMatch(chunkedNext, sequentialNext)
        for (step, token) in greedyTokens.enumerated() {
            #expect(Self.argmax(Self.bits(logits, count: vocab)) == token)
            try await runner.produce(token: token, position: prompt.count + 1 + step, into: logits)
            Self.expectNumericalMatch(Self.bits(logits, count: vocab), greedyRows[step])
        }
    }

    @Test("chunked prefill matches sequential state",
          arguments: [5, 40])
    func chunkedMatchesSequentialState(promptLength: Int) async throws {
        try await Self.expectChunkedMatchesSequential(
            promptLength: promptLength, streamingMode: .pread(slotCount: 16))
    }

    @Test func residentChunkedPrefillMatchesSequentialState() async throws {
        try await Self.expectChunkedMatchesSequential(
            promptLength: 40, streamingMode: .resident)
    }

    @Test(arguments: [false, true])
    func warmAppendsPreserveObservedBatchingAndDecodeState(resident: Bool) async throws {
        let (directory, context, _, runner) = try Self.makeRunner(
            streamingMode: resident ? .resident : .pread(slotCount: 16))
        defer { try? FileManager.default.removeItem(at: directory) }
        let vocab = ArchConfig.qwen38FlashNextToy().vocabSize
        let out = try Self.logits(context, vocab: vocab)
        let tokens = Self.prompt(49, vocab: vocab)
        let continuation: [Int32] = [5, 17, 29, 41, 53, 65]
        for (position, token) in tokens.enumerated() {
            try await runner.produce(token: token, position: position, into: out)
        }
        var expected = [Self.bits(out, count: vocab)]
        for (i, token) in continuation.enumerated() {
            try await runner.produce(token: token, position: tokens.count + i, into: out)
            expected.append(Self.bits(out, count: vocab))
        }
        runner.reset()
        var start = 0
        for length in [3, 29, 17] {
            let result = try await runner.prefillChunked(tokens: tokens[start..<(start + length)],
                startPosition: start, outputMode: .logits, config: .production(chunkTokens: 32),
                into: out, onProgress: { _ in })
            #expect(result.execution?.batchedTokens == length)
            #expect(result.execution?.replayedTokens == 0)
            #expect(result.execution?.batchedChunkSizes == [length])
            start += length
            try runner.prepareForContinuation(expectedPosition: start)
        }
        Self.expectNumericalMatch(Self.bits(out, count: vocab), expected[0])
        for (i, token) in continuation.enumerated() {
            try await runner.produce(token: token, position: start + i, into: out)
            Self.expectNumericalMatch(Self.bits(out, count: vocab), expected[i + 1])
        }
    }

    @Test(arguments: [false, true])
    func interruptedWarmAppendRequiresReset(resident: Bool) async throws {
        let (directory, context, _, runner) = try Self.makeRunner(
            streamingMode: resident ? .resident : .pread(slotCount: 16))
        defer { try? FileManager.default.removeItem(at: directory) }
        let vocab = ArchConfig.qwen38FlashNextToy().vocabSize
        let out = try Self.logits(context, vocab: vocab)
        let tokens = Self.prompt(40, vocab: vocab)
        _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32), into: out,
            onProgress: { _ in })
        let expected = Self.bits(out, count: vocab)
        runner.reset()
        _ = try await runner.prefillChunked(tokens: tokens[..<3], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32), into: out,
            onProgress: { _ in })
        runner.prefillDidCompleteLayer = { layer in
            if layer == 1 { throw CancellationError() }
        }
        do {
            _ = try await runner.prefillChunked(tokens: tokens[3...], startPosition: 3,
                outputMode: .logits, config: .production(chunkTokens: 32), into: out,
                onProgress: { _ in })
            Issue.record("expected cancellation after GPU state advanced")
        } catch is CancellationError { }
        #expect(throws: PrefillError.self) {
            try runner.prepareForContinuation(expectedPosition: 3)
        }
        do {
            try await runner.produce(token: tokens[3], position: 3, into: out)
            Issue.record("dirty state must reject decode")
        } catch let error as PrefillError {
            guard case .chunkedRunnerDirty = error else { throw error }
        }
        runner.prefillDidCompleteLayer = nil
        runner.reset()
        _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32), into: out,
            onProgress: { _ in })
        #expect(Self.bits(out, count: vocab) == expected)
        try runner.prepareForContinuation(expectedPosition: tokens.count)
    }

    @Test func memorySnapshotReadsExistingAllocationsWithoutOpeningExperts() throws {
        let directory = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: context.device,
            expecting: .qwen38FlashNextToy(), streamingMode: .pread(slotCount: 16))
        let before = model.diagnosticMemoryBytes
        let again = model.diagnosticMemoryBytes
        #expect(before == again)
        #expect(before["expertSlotBuffers"]! == 0)
        #expect(before["mappedExpertRegions"]! == 0)
        let alias = try model.resident(name: "lm_head.weight")
        model.streamersQueue.sync { model.convertedBox.views["test-core-alias"] = alias }
        #expect(model.diagnosticMemoryBytes["convertedWeightBuffers"]! == 0)
        let converted = try model.residentAsF32(
            name: "model.language_model.hyper_connection_mixer.hc_norm.weight")
        model.streamersQueue.sync { model.convertedBox.views["test-converted-alias"] = converted }
        #expect(model.diagnosticMemoryBytes["convertedWeightBuffers"]! == UInt64(converted.buffer.length))
        let runner = try FlashNextForwardRunner(model: model, context: context, maxContext: 64)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize)
        let snapshot = RuntimeMemorySnapshot.capture(model: model, producer: runner, scratch: scratch)
        #expect(try #require(snapshot.bytes["targetKVStateBuffers"]!) > 0)
        #expect(try #require(snapshot.bytes["mappedCoreWeightBuffers"]!) > 0)
        #expect(snapshot.bytes["completionScratchBuffers"]! == scratch.diagnosticBufferBytes)
        #expect(try #require(snapshot.bytes["processPhysicalFootprint"]!) > 0)
        #expect(try #require(snapshot.bytes["processRSS"]!) > 0)
        #expect(snapshot.bytes["runnerScratchBuffers"]! == nil)
        #expect(snapshot.bytes["filesystemCache"]! == nil)
    }
}
