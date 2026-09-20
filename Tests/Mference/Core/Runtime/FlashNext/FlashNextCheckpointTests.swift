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
            let values = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt16.self),
                                                   count: model.config.vocabSize))
            #expect(values.allSatisfy { Float16(bitPattern: $0).isFinite })
            #expect(values.contains { Float16(bitPattern: $0) != 0 })
            return values
        }

        func hiddenBits(_ snapshot: FlashNextForwardRunner.TargetHiddenBundle) throws -> [UInt16] {
            let readback = try #require(context.device.makeBuffer(length: snapshot.buffer.length,
                                                                 options: .storageModeShared))
            let cb = try #require(context.queue.makeCommandBuffer())
            let blit = try #require(cb.makeBlitCommandEncoder())
            blit.copy(from: snapshot.buffer, sourceOffset: 0, to: readback,
                      destinationOffset: 0, size: readback.length)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            let values = Array(UnsafeBufferPointer(start: readback.contents().assumingMemoryBound(to: UInt16.self),
                                                   count: readback.length / 2))
            #expect(values.allSatisfy { Float16(bitPattern: $0).isFinite })
            #expect(values.contains { Float16(bitPattern: $0) != 0 })
            return values
        }
    }

    @Test func targetBundleIsUnmixedOwnedAndUnavailableBeforeCommit() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.runner.captureTargetHiddenBundle() }
        h.runner.capture = .init()
        try await h.runner.produce(token: 7, position: 0, into: h.output)
        let snapshot = try h.runner.captureTargetHiddenBundle()
        let bits = try h.hiddenBits(snapshot)
        #expect(snapshot.processedTokenCount == 1)
        #expect(bits.count == h.model.config.residualStreamWidth)
        let reference = try #require(h.runner.capture?.floats["layer03.stream_out"])
        #expect(reference.allSatisfy { $0.isFinite })
        #expect(bits.map { Float(Float16(bitPattern: $0)) } == reference)
        h.runner.capture = nil
        try await h.runner.produce(token: 11, position: 1, into: h.output)
        #expect(try h.hiddenBits(snapshot) == bits, "caller owns a copy, not mutable decode scratch")
        h.runner.reset()
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.runner.captureTargetHiddenBundle() }
    }

    @Test(arguments: [3, 35])
    func targetBundleSurvivesPrefillWarmAppendAndRollback(prefixCount: Int) async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let tokens = (0..<prefixCount).map { Int32(4 + ($0 * 17) % 53) }
        for (position, token) in tokens.enumerated() {
            try await h.runner.produce(token: token, position: position, into: h.output)
        }
        let sequential = try h.hiddenBits(h.runner.captureTargetHiddenBundle())
        h.runner.reset()
        _ = try await h.runner.prefillChunked(tokens: tokens[...], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32), into: h.output, onProgress: { _ in })
        let snapshot = try h.runner.captureTargetHiddenBundle()
        #expect(snapshot.processedTokenCount == prefixCount)
        let prefilling = try h.hiddenBits(snapshot)
        // Batched arithmetic may round differently; use the same two-FP16-
        // precision-unit bound as the finite prefill gate. Ownership and
        // checkpoint restoration below remain bit-exact, without tolerances.
        let expected = sequential.map { Float(Float16(bitPattern: $0)) }
        let actual = prefilling.map { Float(Float16(bitPattern: $0)) }
        let delta = zip(actual, expected).map { abs($0 - $1) }.max() ?? .infinity
        let scale = expected.map { abs($0) }.max() ?? 0
        #expect(scale > 0 && delta <= scale / 512, "last ragged prefill row, not row zero or mixed state")
        let checkpoint = try h.runner.captureDecodeCheckpoint()
        _ = try await h.runner.prefillChunked(tokens: [Int32(7), 11, 19][...], startPosition: prefixCount,
            outputMode: .logits, config: .production(chunkTokens: 64), into: h.output, onProgress: { _ in })
        #expect(try h.runner.captureTargetHiddenBundle().processedTokenCount == prefixCount + 3)
        #expect(try h.hiddenBits(snapshot) == prefilling, "scratch resize must not invalidate owned copies")
        try h.runner.restoreDecodeCheckpoint(checkpoint)
        #expect(try h.hiddenBits(h.runner.captureTargetHiddenBundle()) == prefilling)
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

    @Test func allTargetRowsAreOwnedOrderedAndDoNotReplayPrefill() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let tokens: [Int32] = (0..<44).map { Int32(4 + ($0 * 17) % 53) }
        var batches: [FlashNextForwardRunner.TargetHiddenRows] = []
        h.runner.consumeTargetHiddenRows = { batches.append($0) }
        let first = try await h.runner.prefillChunked(tokens: tokens.prefix(40), startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32), into: h.output, onProgress: { _ in })
        #expect(first.execution?.batchedTokens == 40 && first.execution?.replayedTokens == 0)
        let warm = try await h.runner.prefillChunked(tokens: tokens[40..<43], startPosition: 40,
            outputMode: .logits, config: .production(chunkTokens: 64), into: h.output, onProgress: { _ in })
        #expect(warm.execution?.batchedTokens == 3 && warm.execution?.replayedTokens == 0)
        try await h.runner.produce(token: tokens[43], position: 43, into: h.output)
        #expect(batches.map(\.startPosition) == [0, 32, 40, 43])
        #expect(batches.map { $0.tokens.count } == [32, 8, 3, 1])
        #expect(batches.flatMap(\.tokens) == tokens)
        let width = h.model.config.residualStreamWidth
        var captured: [[UInt16]] = []
        for batch in batches {
            #expect(batch.buffer.length == batch.tokens.count * width * 2)
            let bits = try h.hiddenBits(.init(processedTokenCount: batch.startPosition + batch.tokens.count,
                                               buffer: batch.buffer))
            for row in batch.tokens.indices {
                captured.append(Array(bits[(row * width)..<((row + 1) * width)]))
            }
        }
        #expect(captured.last == (try h.hiddenBits(h.runner.captureTargetHiddenBundle())))
        h.runner.consumeTargetHiddenRows = nil
        h.runner.reset()
        // Independently capture each sequential target row. Across ALL raw
        // intermediate HC bundles the fixture accumulates up to 0.208% drift
        // (row 21), slightly beyond the existing last-row/head two-unit bound.
        // Use an explicit four-FP16-unit semantic band for this new all-row
        // check; existing final-row/head limits and exact ownership/rollback
        // checks remain unchanged. This cannot qualify an exact verifier.
        for (position, token) in tokens.enumerated() {
            try await h.runner.produce(token: token, position: position, into: h.output)
            let reference = try h.hiddenBits(h.runner.captureTargetHiddenBundle())
            let expected = reference.map { Float(Float16(bitPattern: $0)) }
            let actual = captured[position].map { Float(Float16(bitPattern: $0)) }
            let scale = expected.map { abs($0) }.max()!
            let error = zip(actual, expected).map { abs($0 - $1) }.max()!
            print("[target hidden rows] row=\(position) maxAbs=\(error) scale=\(scale)")
            #expect(scale > 0 && error <= scale / 256, "target hidden row \(position)")
        }
        var preserved: [UInt16] = []
        for batch in batches {
            preserved += try h.hiddenBits(.init(processedTokenCount: 0, buffer: batch.buffer))
        }
        #expect(preserved == captured.flatMap { $0 }, "owned rows survive scratch resize, reset and replay")
    }

    @Test(arguments: [false, true])
    func targetRowConsumerFailureRequiresRollbackOrReset(prefill: Bool) async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        try await h.runner.produce(token: 7, position: 0, into: h.output)
        let checkpoint = try h.runner.captureDecodeCheckpoint()
        try await h.runner.produce(token: 11, position: 1, into: h.output)
        let expected = h.bits()
        try h.runner.restoreDecodeCheckpoint(checkpoint)
        let retry = try h.runner.captureDecodeCheckpoint()
        var calls = 0
        h.runner.consumeTargetHiddenRows = { rows in
            #expect(rows.startPosition == 1 && rows.tokens == [11])
            calls += 1
            throw CancellationError()
        }
        do {
            if prefill {
                _ = try await h.runner.prefillChunked(tokens: [Int32(11)][...], startPosition: 1,
                    outputMode: .logits, config: .production(chunkTokens: 32), into: h.output, onProgress: { _ in })
            } else {
                try await h.runner.produce(token: 11, position: 1, into: h.output)
            }
            Issue.record("expected target-row consumer failure")
        } catch is CancellationError { }
        #expect(calls == 1 && h.runner.continuationPosition == 1)
        #expect(throws: PrefillError.self) { try h.runner.prepareForContinuation(expectedPosition: 1) }
        #expect(throws: PrefillError.self) { _ = try h.runner.captureTargetHiddenBundle() }
        h.runner.consumeTargetHiddenRows = nil
        try h.runner.restoreDecodeCheckpoint(retry)
        try await h.runner.produce(token: 11, position: 1, into: h.output)
        #expect(h.bits() == expected)
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
            outputMode: .logits, config: .production(chunkTokens: 32), into: h.output, onProgress: { _ in })
        let checkpoint = try h.runner.captureDecodeCheckpoint()
        try await h.runner.produce(token: 23, position: 3, into: h.output)
        let expected = h.bits()
        try h.runner.restoreDecodeCheckpoint(checkpoint)
        let clean = try h.runner.captureDecodeCheckpoint()
        h.runner.prefillDidCompleteLayer = { if $0 == 2 { throw CancellationError() } }
        do {
            _ = try await h.runner.prefillChunked(tokens: [Int32(43), 47][...], startPosition: 3,
                outputMode: .logits, config: .production(chunkTokens: 64), into: h.output, onProgress: { _ in })
            Issue.record("expected interrupted append")
        } catch is CancellationError { }
        #expect(throws: PrefillError.self) { _ = try h.runner.captureDecodeCheckpoint() }
        #expect(throws: PrefillError.self) { _ = try h.runner.captureTargetHiddenBundle() }
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
        #expect(throws: PrefillError.self) { _ = try h.runner.captureTargetHiddenBundle() }
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
