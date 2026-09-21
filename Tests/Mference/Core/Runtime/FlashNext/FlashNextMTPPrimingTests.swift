import Foundation
import Metal
import Testing
@testable import Mference

@Suite(.serialized) struct FlashNextMTPPrimingTests {
    private struct Harness {
        let directory: URL
        let context: MetalContext
        let model: Model
        let primer: FlashNextMTPPrimer
        let output: MTLBuffer

        func rows(_ tokens: [Int32], at position: Int) throws -> FlashNextForwardRunner.TargetHiddenRows {
            let width = model.config.residualStreamWidth
            var values: [Float16] = []
            for row in tokens.indices {
                for column in 0..<width {
                    let integer = (column * 7 + (row + position) * 3) % 23 - 11
                    values.append(Float16(Float(integer) / 8))
                }
            }
            let buffer = try #require(context.device.makeBuffer(bytes: values, length: values.count * 2, options: .storageModeShared))
            return .init(startPosition: position, tokens: tokens, buffer: buffer)
        }

        func bits(_ buffer: MTLBuffer) throws -> [UInt16] {
            let copy = try #require(context.device.makeBuffer(length: buffer.length, options: .storageModeShared))
            let cb = try #require(context.queue.makeCommandBuffer())
            let blit = try #require(cb.makeBlitCommandEncoder())
            blit.copy(from: buffer, sourceOffset: 0, to: copy, destinationOffset: 0, size: copy.length)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.error == nil)
            let result = Array(UnsafeBufferPointer(start: copy.contents().assumingMemoryBound(to: UInt16.self), count: copy.length / 2))
            #expect(result.allSatisfy { Float16(bitPattern: $0).isFinite })
            #expect(result.contains { Float16(bitPattern: $0) != 0 })
            return result
        }
    }

    private func make(resident: Bool = false, maxContext: Int = 48) throws -> Harness {
        let directory = try FlashNextToySynthetic.write(includeMTP: true)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: .qwen38FlashNextToy(), streamingMode: .pread(slotCount: 16))
        let primer = try FlashNextMTPPrimer(model: model, context: context, maxContext: maxContext,
            policy: resident ? .resident : .bounded(slots: 6))
        let output = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        return Harness(directory: directory, context: context, model: model, primer: primer, output: output)
    }

    @Test(arguments: [false, true])
    func shiftedPairsAreIndependentOfChunkPartition(resident: Bool) throws {
        let h = try make(resident: resident)
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let tokens = (0..<40).map { Int32(4 + ($0 * 17) % 53) }
        let reference = try FlashNextMTPDraftRunner(model: h.model, context: h.context,
            maxContext: 48, policy: resident ? .resident : .bounded(slots: 6))
        var expected: [UInt16] = []
        for row in tokens.indices {
            let input = try h.rows([tokens[row]], at: row)
            let result = try reference.append(token: row + 1 < tokens.count ? tokens[row + 1] : 11,
                targetHidden: input.buffer, at: row, into: h.output)
            if row == tokens.count - 1 { expected = try h.bits(result.hidden) + h.bits(h.output) }
        }
        for chunks in [[40], [32, 8], [1, 31, 1, 7], Array(repeating: 1, count: 40)] {
            h.primer.reset()
            var position = 0
            for count in chunks {
                try h.primer.consume(h.rows(Array(tokens[position..<(position + count)]), at: position))
                position += count
                #expect(h.primer.targetPosition == position)
                #expect(h.primer.draftPosition == position - 1, "tail waits for its actual next token")
            }
            let result = try h.primer.finish(nextToken: 11, into: h.output)
            #expect(h.primer.draftPosition == 40)
            #expect(try h.bits(result.hidden) + h.bits(h.output) == expected)
            #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.primer.finish(nextToken: 11, into: h.output) }
        }
    }

    @Test func tailOwnershipWarmAppendAndRollback() throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let initial = try h.rows([7, 11, 19], at: 0)
        try h.primer.consume(initial)
        let checkpoint = try h.primer.checkpoint()
        // A tail owns a row, not the caller's batch (nor a mutable row scratch).
        memset(initial.buffer.contents(), 0, initial.buffer.length)
        let first = try h.primer.finish(nextToken: 23, into: h.output)
        let expected = try h.bits(first.hidden) + h.bits(h.output)
        let finished = try h.primer.checkpoint()
        #expect(throws: FlashNextForwardRunnerError.self) { try h.primer.consume(h.rows([24], at: 3)) }
        #expect(h.primer.targetPosition == 3 && h.primer.draftPosition == 3)
        try h.primer.consume(h.rows([23, 29, 31], at: 3))
        let warm = try h.primer.finish(nextToken: 37, into: h.output)
        let warmExpected = try h.bits(warm.hidden) + h.bits(h.output)
        try h.primer.restore(finished)
        #expect(throws: FlashNextForwardRunnerError.self) { try h.primer.restore(checkpoint) }
        try h.primer.consume(h.rows([23, 29, 31], at: 3))
        #expect(try h.bits(h.primer.finish(nextToken: 37, into: h.output).hidden) + h.bits(h.output) == warmExpected)
        h.primer.reset()
        try h.primer.consume(h.rows([7, 11, 19], at: 0))
        let pending = try h.primer.checkpoint()
        _ = try h.primer.finish(nextToken: 53, into: h.output)
        try h.primer.consume(h.rows([53, 59], at: 3))
        try h.primer.restore(pending)
        #expect(try h.bits(h.primer.finish(nextToken: 23, into: h.output).hidden) + h.bits(h.output) == expected)
    }

    @Test func partialPrimingFailureRequiresPairedRecovery() throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        try h.primer.consume(h.rows([7, 11], at: 0))
        let before = try h.primer.checkpoint()
        try h.primer.consume(h.rows([19, 23, 29], at: 2))
        let expected = try h.bits(h.primer.finish(nextToken: 31, into: h.output).hidden) + h.bits(h.output)
        try h.primer.restore(before)
        let recovery = try h.primer.checkpoint()
        h.primer.didPrimeRow = { if $0 == 2 { throw CancellationError() } }
        #expect(throws: CancellationError.self) { try h.primer.consume(h.rows([19, 23, 29], at: 2)) }
        #expect(h.primer.targetPosition == 2 && h.primer.draftPosition == 3)
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.primer.checkpoint() }
        #expect(throws: FlashNextForwardRunnerError.self) { try h.primer.consume(h.rows([19], at: 2)) }
        h.primer.didPrimeRow = nil
        try h.primer.restore(recovery)
        try h.primer.consume(h.rows([19, 23, 29], at: 2))
        #expect(try h.bits(h.primer.finish(nextToken: 31, into: h.output).hidden) + h.bits(h.output) == expected)
    }

    @Test func badPositionsTokensBuffersAndCheckpointOwnersReject() throws {
        let h = try make(maxContext: 2)
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let other = try FlashNextMTPPrimer(model: h.model, context: h.context, maxContext: 2, policy: .bounded(slots: 6))
        let checkpoint = try h.primer.checkpoint()
        #expect(throws: FlashNextForwardRunnerError.self) { try other.restore(checkpoint) }
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.primer.finish(nextToken: 7, into: h.output) }
        #expect(throws: FlashNextForwardRunnerError.self) { try h.primer.consume(h.rows([7], at: 1)) }
        #expect(throws: FlashNextForwardRunnerError.self) { try h.primer.consume(h.rows([-1], at: 0)) }
        #expect(throws: FlashNextForwardRunnerError.self) { try h.primer.consume(h.rows([Int32(h.model.config.vocabSize)], at: 0)) }
        let row = try h.rows([7], at: 0)
        #expect(throws: FlashNextForwardRunnerError.self) { try h.primer.consume(.init(startPosition: 0, tokens: [7, 11], buffer: row.buffer)) }
        try h.primer.consume(h.rows([7, 11], at: 0))
        _ = try h.primer.finish(nextToken: 19, into: h.output)
        #expect(throws: FlashNextForwardRunnerError.self) { try h.primer.consume(h.rows([19], at: 2)) }
        h.primer.reset()
        #expect(throws: FlashNextForwardRunnerError.self) { try h.primer.restore(checkpoint) }
    }

    @Test func actualTargetRowsPrimeAcrossColdWarmAndDecodeBoundaries() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        try await verifyTargetPriming(h)
    }

    @Test func installedTargetRowsPrimeNativeDraft() async throws {
        guard let path = ProcessInfo.processInfo.environment["MFERENCE_FLASHNEXT_GTURBO"] else { return }
        let context = try MetalContext()
        let directory = URL(fileURLWithPath: path)
        let model = try Model.load(directoryURL: directory, device: context.device, streamingMode: .pread(slotCount: 16))
        let primer = try FlashNextMTPPrimer(model: model, context: context, maxContext: 48, policy: .bounded(slots: 16))
        let output = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        try await verifyTargetPriming(.init(directory: directory, context: context, model: model, primer: primer, output: output))
        print("[installed MTP priming] 43 actual target HC rows; cold/warm/decode shifted pairs match manual native feed exactly; zero target replay")
    }

    private func verifyTargetPriming(_ h: Harness) async throws {
        let target = try FlashNextForwardRunner(model: h.model, context: h.context, maxContext: 48)
        var captured: [FlashNextForwardRunner.TargetHiddenRows] = []
        target.consumeTargetHiddenRows = { rows in
            captured.append(rows)
            try h.primer.consume(rows)
        }
        let tokens = (0..<43).map { Int32(4 + ($0 * 17) % 53) }
        let result = try await target.prefillChunked(tokens: tokens.prefix(35), startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32), into: h.output, onProgress: { _ in })
        #expect(result.execution?.batchedTokens == 35 && result.execution?.replayedTokens == 0)
        _ = try h.primer.finish(nextToken: tokens[35], into: h.output)
        _ = try await target.prefillChunked(tokens: tokens[35..<42], startPosition: 35,
            outputMode: .logits, config: .production(chunkTokens: 32), into: h.output, onProgress: { _ in })
        _ = try h.primer.finish(nextToken: tokens[42], into: h.output)
        try await target.produce(token: tokens[42], position: 42, into: h.output)
        let final = try h.primer.finish(nextToken: 61, into: h.output)
        let expected = try h.bits(final.hidden) + h.bits(h.output)
        #expect(captured.map(\.startPosition) == [0, 32, 35, 42])
        // Independent single-row feeding of exactly the captured target rows,
        // not a second target prefill (which has a different rounding path).
        let manual = try FlashNextMTPDraftRunner(model: h.model, context: h.context, maxContext: 48, policy: .bounded(slots: 16))
        for batch in captured {
            let words = try h.bits(batch.buffer)
            let width = h.model.config.residualStreamWidth
            for row in batch.tokens.indices {
                let position = batch.startPosition + row
                let values = Array(words[(row * width)..<((row + 1) * width)])
                let buffer = try #require(h.context.device.makeBuffer(bytes: values, length: values.count * 2, options: .storageModeShared))
                let output = try manual.append(token: position + 1 < tokens.count ? tokens[position + 1] : 61,
                    targetHidden: buffer, at: position, into: h.output)
                if position == tokens.count - 1 { #expect(try h.bits(output.hidden) + h.bits(h.output) == expected) }
            }
        }
        #expect(target.continuationPosition == 43 && h.primer.draftPosition == 43)
    }

    @Test func interruptedTargetAndPrimerRecoverTogether() async throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let target = try FlashNextForwardRunner(model: h.model, context: h.context, maxContext: 48)
        target.consumeTargetHiddenRows = { try h.primer.consume($0) }
        func append(_ tokens: [Int32], at position: Int) async throws {
            _ = try await target.prefillChunked(tokens: tokens[...], startPosition: position,
                outputMode: .logits, config: .production(chunkTokens: 32), into: h.output, onProgress: { _ in })
        }
        try await append([7, 11], at: 0)
        let targetBase = try target.captureDecodeCheckpoint()
        let draftBase = try h.primer.checkpoint()
        try await append([19, 23, 29], at: 2)
        let targetExpected = try h.bits(h.output)
        let draftExpected = try h.bits(h.primer.finish(nextToken: 31, into: h.output).hidden) + h.bits(h.output)
        try target.restoreDecodeCheckpoint(targetBase)
        try h.primer.restore(draftBase)
        let targetRecovery = try target.captureDecodeCheckpoint()
        let draftRecovery = try h.primer.checkpoint()
        h.primer.didPrimeRow = { if $0 == 2 { throw CancellationError() } }
        do {
            try await append([19, 23, 29], at: 2)
            Issue.record("expected interrupted priming")
        } catch is CancellationError { }
        #expect(throws: PrefillError.self) { try target.prepareForContinuation(expectedPosition: 2) }
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.primer.checkpoint() }
        h.primer.didPrimeRow = nil
        try target.restoreDecodeCheckpoint(targetRecovery)
        try h.primer.restore(draftRecovery)
        try await append([19, 23, 29], at: 2)
        #expect(try h.bits(h.output) == targetExpected)
        #expect(try h.bits(h.primer.finish(nextToken: 31, into: h.output).hidden) + h.bits(h.output) == draftExpected)
    }
}
