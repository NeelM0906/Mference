import Foundation
import Metal
import Testing
@testable import Mference

@Suite(.serialized) struct FlashNextCheckpointTests {
    private struct Harness {
        let directory: URL
        let context: MetalContext
        let model: Model
        let runner: FlashNextForwardRunner
        let output: MTLBuffer

        func bits() -> [UInt16] {
            Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt16.self),
                                      count: model.config.vocabSize))
        }
    }

    private func make() throws -> Harness {
        let directory = try FlashNextToySynthetic.write()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: .qwen38FlashNextToy(), streamingMode: .pread(slotCount: 16))
        let runner = try FlashNextForwardRunner(model: model, context: context, maxContext: 96)
        let output = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                            options: .storageModeShared))
        return Harness(directory: directory, context: context, model: model, runner: runner, output: output)
    }

    @Test(arguments: [3, 35])
    func rejectedDraftRestoresRecurrentAndPLEState(prefixCount: Int) async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let tokens: [Int32] = (0..<prefixCount).map { Int32(4 + ($0 * 17) % 53) }
        _ = try await h.runner.prefillChunked(tokens: tokens[...], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32), into: h.output, onProgress: { _ in })
        let checkpoint = try h.runner.captureDecodeCheckpoint()
        var expected: [[UInt16]] = []
        for (step, token) in [Int32(7), 11, 19, 23].enumerated() {
            try await h.runner.produce(token: token, position: prefixCount + step, into: h.output)
            expected.append(h.bits())
        }
        try h.runner.restoreDecodeCheckpoint(checkpoint)
        let retry = try h.runner.captureDecodeCheckpoint()
        // Wrong draft includes EOS, which resets PLE's token history. Crossing
        // a pooled-key boundary also leaves future indexer rows to be hidden.
        let wrong = [Int32(h.model.config.flashNext.pleEosTokenID), 43, 47, 53]
        for (step, token) in wrong.enumerated() {
            try await h.runner.produce(token: token, position: prefixCount + step, into: h.output)
        }
        try h.runner.restoreDecodeCheckpoint(retry)
        #expect(h.runner.continuationPosition == prefixCount)
        for (step, token) in [Int32(7), 11, 19, 23].enumerated() {
            try await h.runner.produce(token: token, position: prefixCount + step, into: h.output)
            #expect(h.bits() == expected[step], "every logit after rollback")
        }
        #expect(throws: FlashNextForwardRunnerError.self) { try h.runner.restoreDecodeCheckpoint(retry) }
    }

    @Test func resetAndBranchChangesInvalidateOldCheckpoints() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        try await h.runner.produce(token: 7, position: 0, into: h.output)
        let parent = try h.runner.captureDecodeCheckpoint()
        try await h.runner.produce(token: 11, position: 1, into: h.output)
        let child = try h.runner.captureDecodeCheckpoint()
        try h.runner.restoreDecodeCheckpoint(parent)
        try await h.runner.produce(token: 19, position: 1, into: h.output)
        #expect(throws: FlashNextForwardRunnerError.self) { try h.runner.restoreDecodeCheckpoint(child) }
        let fresh = try h.runner.captureDecodeCheckpoint()
        h.runner.reset()
        #expect(throws: FlashNextForwardRunnerError.self) { try h.runner.restoreDecodeCheckpoint(fresh) }
    }

    @Test func checkpointsCannotCrossRunnerOwnership() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        // Share immutable weights, not sequence state. No second model install.
        let other = try FlashNextForwardRunner(model: h.model, context: h.context, maxContext: 96)
        try await h.runner.produce(token: 7, position: 0, into: h.output)
        let checkpoint = try h.runner.captureDecodeCheckpoint()
        try await other.produce(token: 11, position: 0, into: h.output)
        #expect(throws: FlashNextForwardRunnerError.self) { try other.restoreDecodeCheckpoint(checkpoint) }
        #expect(other.continuationPosition == 1)
        try other.prepareForContinuation(expectedPosition: 1)
    }

    @Test func interruptedAppendCanRestoreCommittedCheckpointAfterScratchResize() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let tokens: [Int32] = [7, 11, 19]
        _ = try await h.runner.prefillChunked(tokens: tokens[...], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 8), into: h.output, onProgress: { _ in })
        let checkpoint = try h.runner.captureDecodeCheckpoint()
        try await h.runner.produce(token: 23, position: 3, into: h.output)
        let expected = h.bits()
        try h.runner.restoreDecodeCheckpoint(checkpoint)
        let clean = try h.runner.captureDecodeCheckpoint()
        h.runner.prefillDidCompleteLayer = { if $0 == 2 { throw CancellationError() } }
        do {
            _ = try await h.runner.prefillChunked(tokens: [Int32(43), 47][...], startPosition: 3,
                outputMode: .logits, config: .production(chunkTokens: 32), into: h.output, onProgress: { _ in })
            Issue.record("expected interrupted append")
        } catch is CancellationError { }
        #expect(throws: PrefillError.self) { _ = try h.runner.captureDecodeCheckpoint() }
        h.runner.prefillDidCompleteLayer = nil
        try h.runner.restoreDecodeCheckpoint(clean)
        try h.runner.prepareForContinuation(expectedPosition: 3)
        try await h.runner.produce(token: 23, position: 3, into: h.output)
        #expect(h.bits() == expected)
    }

    @Test func interruptedDecodeRejectsReuseButCanRestoreAnEarlierCheckpoint() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        try await h.runner.produce(token: 7, position: 0, into: h.output)
        let checkpoint = try h.runner.captureDecodeCheckpoint()
        try await h.runner.produce(token: 11, position: 1, into: h.output)
        let expected = h.bits()
        try h.runner.restoreDecodeCheckpoint(checkpoint)
        let clean = try h.runner.captureDecodeCheckpoint()
        h.runner.decodeWillEncodeLayer = { if $0 == 2 { throw CancellationError() } }
        do {
            try await h.runner.produce(token: 43, position: 1, into: h.output)
            Issue.record("expected interruption after GDN and PLE advanced")
        } catch is CancellationError { }
        #expect(throws: PrefillError.self) { _ = try h.runner.captureDecodeCheckpoint() }
        #expect(throws: PrefillError.self) { try h.runner.prepareForContinuation(expectedPosition: 1) }
        h.runner.decodeWillEncodeLayer = nil
        do {
            try await h.runner.produce(token: 11, position: 1, into: h.output)
            Issue.record("dirty decode state must reject reuse")
        } catch is PrefillError { }
        try h.runner.restoreDecodeCheckpoint(clean)
        try await h.runner.produce(token: 11, position: 1, into: h.output)
        #expect(h.bits() == expected)
    }
}
