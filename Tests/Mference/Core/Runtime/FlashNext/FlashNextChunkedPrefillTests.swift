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
        Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: UInt16.self, capacity: count),
            count: count))
    }

    private static func prompt(_ count: Int, vocab: Int) -> [Int32] {
        (0..<count).map { Int32(4 + ($0 * 37 + 11) % (vocab - 4)) }
    }

    @Test func factoryReportsChunkedProductionPrefill() throws {
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
        #expect(runtime.executedPrefillMode == .chunked)
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

        runner.reset()
        var progress: [Int] = []
        let result = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { progress.append($0) })
        #expect(result == PrefillResult(newPosition: prompt.count,
                                       seed: .logitsWritten))
        #expect(progress.last == prompt.count)
        #expect(runner.continuationPosition == prompt.count)
        let chunkedPrompt = Self.bits(logits, count: vocab)
        try await runner.produce(token: continuation, position: prompt.count,
                                 into: logits)
        let chunkedNext = Self.bits(logits, count: vocab)

        #expect(chunkedPrompt == sequentialPrompt)
        #expect(chunkedNext == sequentialNext)
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
}
