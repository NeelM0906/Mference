import Foundation
import Metal
import Testing
@testable import Mference

/// Current installed checkpoint only; never downloads or copies weights.
@Suite(.serialized) struct InklingPrefillRecoveryTests {
    @Test func cancelledWarmAppendResetsKVAndConvolutions() async throws {
        guard let path = ProcessInfo.processInfo.environment["MFERENCE_INKLING_GTURBO"] else { return }
        let ctx = try MetalContext()
        let url = URL(fileURLWithPath: path)
        let config = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: url, device: ctx.device,
            expecting: config, streamingMode: .pread(slotCount: 16))
        let tokenizer = try await MFTokenizer.load(forModelDirectory: url)
        let runner = try RealForwardRunner(model: model, context: ctx, maxContext: 128,
            runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 16,
                prefillChunkTokens: 32, forceLogitsHead: true))
        let logits = try #require(ctx.device.makeBuffer(length: model.config.vocabSize * 2,
                                                        options: .storageModeShared))
        let source = tokenizer.encode(String(repeating:
            "A small town records river levels every morning and compares weekly averages.\n", count: 12),
            addBOS: true)
        try #require(source.count >= 65)
        let tokens = Array(source.prefix(65))
        func bits() -> [UInt16] {
            Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: UInt16.self),
                                      count: model.config.vocabSize))
        }
        func prefix() async throws {
            let result = try await runner.prefillChunked(tokens: tokens.prefix(33), startPosition: 0,
                outputMode: .logits, config: .production(chunkTokens: 32),
                into: logits, onProgress: { _ in })
            #expect(result.execution?.batchedTokens == 33)
            #expect(result.execution?.replayedTokens == 0)
        }
        try await prefix()
        let reference = bits()
        try await runner.produce(token: tokens[33], position: 33, into: logits)
        let referenceNext = bits()
        runner.reset()
        try await prefix()
        runner.prefillWillEncodeLayer = { layer in
            // Dense layers and the first routed layer have completed, including
            // their KV, attention/output short convolutions and expert pipeline.
            if layer == 3 { withUnsafeCurrentTask { $0?.cancel() } }
        }
        struct SerialBuffer: @unchecked Sendable { let value: MTLBuffer }
        let output = SerialBuffer(value: logits)
        let interrupted = Task { @Sendable in
            _ = try await runner.prefillChunked(tokens: tokens[33..<65], startPosition: 33,
                outputMode: .logits, config: .production(chunkTokens: 32),
                into: output.value, onProgress: { _ in })
        }
        do {
            try await interrupted.value
            Issue.record("expected task cancellation after routed layer writes")
        } catch is CancellationError {}
        #expect(throws: PrefillError.self) { try runner.prepareForContinuation(expectedPosition: 33) }
        do {
            try await runner.produce(token: tokens[33], position: 33, into: logits)
            Issue.record("dirty state must reject decode")
        } catch let error as PrefillError {
            guard case .chunkedRunnerDirty = error else { throw error }
        }
        runner.prefillWillEncodeLayer = nil
        runner.reset()
        try await prefix()
        #expect(bits() == reference, "reset reproduces the full prefix logits row")
        try runner.prepareForContinuation(expectedPosition: 33)
        try await runner.produce(token: tokens[33], position: 33, into: logits)
        #expect(bits() == referenceNext, "decode handoff reproduces the full logits row")
        FileHandle.standardError.write(Data(
            "[inkling recovery] strict verification; slots=16; warm=33; cancelled append=32; reset and next full-logit rows exact\n".utf8))
    }
}
