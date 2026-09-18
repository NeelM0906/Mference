import Foundation
import Metal
import Testing
@testable import Mference

/// Execution-path contract, not a model-quality fixture. Numerical parity is
/// covered by the family suites. These assertions consume actual dispatch
/// reports after cold/warm multi-token calls, rather than factory preferences.
@Suite(.serialized) struct ProductionPrefillContractTests {
    @Test(arguments: ["gemma", "qwen"])
    func realRunnerBatchesWarmAppends(family: String) async throws {
        let streamed = try await Self.warmAppendRow(family: family, resident: false)
        let resident = try await Self.warmAppendRow(family: family, resident: true)
        #expect(streamed == resident, "bounded expert tiles preserve the resident result")
    }

    @Test(arguments: ["gemma", "qwen"], [false, true])
    func interruptedWarmAppendRequiresReset(family: String, resident: Bool) async throws {
        let gemma = family == "gemma"
        let config: ArchConfig = gemma ? .gemma4Toy(topKExperts: 8) : .qwen36Toy()
        let directory = try (gemma
            ? ModelLoaderTests.writeToySynthetic(config: config, finiteNorms: true)
            : QwenToySynthetic.write())
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: config, streamingMode: resident ? .resident : .pread(slotCount: 8))
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 128,
            runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 8,
                prefillChunkTokens: 64, forceLogitsHead: true))
        let logits = try #require(context.device.makeBuffer(length: config.vocabSize * 2,
                                                            options: .storageModeShared))
        let tokens = (0..<65).map { Int32(4 + $0 % 16) }
        func row() -> [UInt16] {
            Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: UInt16.self),
                                      count: config.vocabSize))
        }
        _ = try await runner.prefillChunked(tokens: tokens.prefix(33), startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 64),
            into: logits, onProgress: { _ in })
        let reference = row()
        runner.prefillWillEncodeLayer = { layer in
            if layer == 1 { withUnsafeCurrentTask { $0?.cancel() } }
        }
        // No parent access occurs until value is awaited. Metal's buffer
        // protocol itself has no Sendable annotation on the supported SDKs.
        struct SerialBuffer: @unchecked Sendable { let value: MTLBuffer }
        let output = SerialBuffer(value: logits)
        let interrupted = Task { @Sendable in
            _ = try await runner.prefillChunked(tokens: tokens[33..<65], startPosition: 33,
                outputMode: .logits, config: .production(chunkTokens: 64),
                into: output.value, onProgress: { _ in })
        }
        do {
            try await interrupted.value
            Issue.record("expected cancellation after the first layer advanced")
        } catch is CancellationError {}
        #expect(throws: (any Error).self) {
            try runner.prepareForContinuation(expectedPosition: 33)
        }
        do {
            try await runner.produce(token: 7, position: 33, into: logits)
            Issue.record("dirty state must reject decode")
        } catch is PrefillError {}
        runner.prefillWillEncodeLayer = nil
        runner.reset()
        _ = try await runner.prefillChunked(tokens: tokens.prefix(33), startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 64),
            into: logits, onProgress: { _ in })
        #expect(row() == reference)
        try runner.prepareForContinuation(expectedPosition: 33)
        try await runner.produce(token: 7, position: 33, into: logits)
        #expect(row().allSatisfy { Float16(bitPattern: $0).isFinite })
    }

    private static func warmAppendRow(family: String, resident: Bool) async throws -> [UInt16] {
        let gemma = family == "gemma"
        // The loader's historical top-2 toy does not satisfy the production
        // MoE kernel contract. Keep its defaults, but run this fixture at top-8.
        let config: ArchConfig = gemma ? .gemma4Toy(topKExperts: 8) : .qwen36Toy()
        let directory = try (gemma ? ModelLoaderTests.writeToySynthetic(config: config, finiteNorms: true) : QwenToySynthetic.write())
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: config, streamingMode: resident ? .resident : .pread(slotCount: 8))
        let runtime = RuntimeConfiguration(expertCacheSlots: 8, prefillChunkTokens: 64, forceLogitsHead: true)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 416,
                                          runtimeConfiguration: runtime)
        let logits = try #require(context.device.makeBuffer(length: config.vocabSize * 2, options: .storageModeShared))
        let tokenRange: Int = config.vocabSize - 4
        var tokens: [Int32] = []
        for index in 0..<385 {
            let mixed: Int = index * 37 + 11
            tokens.append(Int32(4 + mixed % tokenRange))
        }
        func row() -> [UInt16] {
            Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: UInt16.self),
                                      count: config.vocabSize))
        }
        var first: [UInt16] = []
        for pass in 0..<2 {
            runner.reset()
            var start = 0
            // Ragged cold and warm appends, multi-chunk tail, and Gemma's
            // sliding-window/ring boundary (256/320) inside the final append.
            for length in [31, 34, 320] {
                var progress: [Int] = []
                let result = try await runner.prefillChunked(tokens: tokens[start..<(start + length)],
                    startPosition: start, outputMode: .logits, config: runtime.prefillConfig,
                    into: logits, onProgress: { progress.append($0) })
                let report: PrefillExecutionReport = try #require(result.execution)
                #expect(report.executedMode == PrefillExecutedMode.chunked)
                #expect(report.batchedTokens == length)
                #expect(report.replayedTokens == 0)
                #expect(report.replayReasons.isEmpty)
                #expect(report.batchedChunkSizes == (length <= 64 ? [length] : [64, 64, 64, 64, 64]))
                #expect(progress.last == length)
                #expect(progress == progress.sorted())
                start += length
                #expect(result.newPosition == start)
                try runner.prepareForContinuation(expectedPosition: start)
            }
            try await runner.produce(token: 7, position: start, into: logits)
            #expect(runner.continuationPosition == start + 1)
            #expect(row().allSatisfy { Float16(bitPattern: $0).isFinite })
            if pass == 0 { first = row() }
            else { #expect(row() == first, "reset reproduces warm-append state") }
        }
        return first
    }
}
