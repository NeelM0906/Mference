import Foundation
import Metal
import Testing
@testable import Mference

/// Mechanical execution/state gates only. These do not establish target-token
/// alignment, upstream native-draft parity, acceptance or a speed improvement.
@Suite(.serialized) struct FlashNextMTPDraftRunnerTests {
    private struct Harness {
        let directory: URL
        let context: MetalContext
        let model: Model
        let runner: FlashNextMTPDraftRunner
        let embedding: MTLBuffer
        let hidden: MTLBuffer
        let logits: MTLBuffer

        func append(_ row: Int, variant: Int = 0) throws -> FlashNextMTPDraftRunner.Output {
            let e = embedding.contents().assumingMemoryBound(to: Float16.self)
            let h = hidden.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<model.config.hiddenSize { e[i] = Float16(Float((i * 3 + row + variant) % 17 - 8) / 16) }
            for i in 0..<model.config.residualStreamWidth { h[i] = Float16(Float((i * 7 + row * 3 + variant) % 23 - 11) / 8) }
            logits.contents().assumingMemoryBound(to: Float16.self).update(repeating: .nan, count: model.config.vocabSize)
            return try runner.append(embedding: embedding, targetHidden: hidden, at: row, into: logits)
        }

        func bits(_ buffer: MTLBuffer) throws -> [UInt16] {
            let readback = try #require(context.device.makeBuffer(length: buffer.length, options: .storageModeShared))
            let cb = try #require(context.queue.makeCommandBuffer())
            let blit = try #require(cb.makeBlitCommandEncoder())
            blit.copy(from: buffer, sourceOffset: 0, to: readback, destinationOffset: 0, size: buffer.length)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.error == nil)
            let values = Array(UnsafeBufferPointer(start: readback.contents().assumingMemoryBound(to: UInt16.self), count: buffer.length / 2))
            #expect(values.allSatisfy { Float16(bitPattern: $0).isFinite })
            #expect(values.contains { Float16(bitPattern: $0) != 0 })
            return values
        }
    }

    private func make(resident: Bool = false, maxContext: Int = 48) throws -> Harness {
        let directory = try FlashNextToySynthetic.write(includeMTP: true)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: .qwen38FlashNextToy(), streamingMode: .pread(slotCount: 16))
        let runner = try FlashNextMTPDraftRunner(model: model, context: context, maxContext: maxContext,
            policy: resident ? .resident : .bounded(slots: 6))
        func buffer(_ count: Int) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(length: count * 2, options: .storageModeShared))
        }
        return Harness(directory: directory, context: context, model: model, runner: runner,
            embedding: try buffer(model.config.hiddenSize), hidden: try buffer(model.config.residualStreamWidth),
            logits: try buffer(model.config.vocabSize))
    }

    @Test func boundedAndResidentMatchAcrossSparseSelection() throws {
        func run(resident: Bool) throws -> [[UInt16]] {
            let h = try make(resident: resident)
            defer { try? FileManager.default.removeItem(at: h.directory) }
            var result: [[UInt16]] = []
            for row in 0..<40 {
                let out = try h.append(row)
                #expect(out.processedRows == row + 1)
                #expect(out.hidden.length == h.model.config.residualStreamWidth * 2)
                result.append(try h.bits(out.hidden))
                result.append(try h.bits(h.logits))
            }
            return result
        }
        let bounded = try run(resident: false)
        let resident = try run(resident: true)
        #expect(bounded == resident)
    }

    @Test(arguments: [false, true])
    func rejectedBranchRestoresCacheAndOwnedOutputs(resident: Bool) throws {
        let h = try make(resident: resident)
        defer { try? FileManager.default.removeItem(at: h.directory) }
        for row in 0..<31 { _ = try h.append(row) }
        let checkpoint = try h.runner.checkpoint()
        var expected: [[UInt16]] = []
        let owned = try h.append(31)
        let saved = try h.bits(owned.hidden)
        expected.append(saved)
        expected.append(try h.bits(h.logits))
        for row in 32..<37 {
            expected.append(try h.bits(h.append(row).hidden))
            expected.append(try h.bits(h.logits))
        }
        try h.runner.restore(checkpoint)
        let retry = try h.runner.checkpoint()
        for row in 31..<39 { _ = try h.append(row, variant: 9) }
        try h.runner.restore(retry)
        #expect(throws: FlashNextForwardRunnerError.self) { try h.runner.restore(retry) }
        var actual: [[UInt16]] = []
        for row in 31..<37 {
            actual.append(try h.bits(h.append(row).hidden))
            actual.append(try h.bits(h.logits))
        }
        #expect(actual == expected)
        #expect(try h.bits(owned.hidden) == saved, "owned HC row cannot alias mutable draft scratch")
        let stale = try h.runner.checkpoint()
        h.runner.reset()
        #expect(throws: FlashNextForwardRunnerError.self) { try h.runner.restore(stale) }
        _ = try h.append(0)
        #expect(try h.bits(owned.hidden) == saved)
    }

    @Test func partialGPUFailureRequiresRestoreOrReset() throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        _ = try h.append(0)
        let parent = try h.runner.checkpoint()
        let expected = try h.bits(h.append(1).hidden)
        let expectedLogits = try h.bits(h.logits)
        try h.runner.restore(parent)
        let checkpoint = try h.runner.checkpoint()
        h.runner.didPrepareAttention = { throw CancellationError() }
        #expect(throws: CancellationError.self) { _ = try h.append(1, variant: 9) }
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.runner.checkpoint() }
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.append(1) }
        h.runner.didPrepareAttention = nil
        try h.runner.restore(checkpoint)
        #expect(try h.bits(h.append(1).hidden) == expected)
        #expect(try h.bits(h.logits) == expectedLogits)
    }

    @Test func missingPrefixContextExhaustionAndForeignCheckpointsReject() throws {
        let h = try make(maxContext: 2)
        defer { try? FileManager.default.removeItem(at: h.directory) }
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.append(1) }
        #expect(h.runner.position == 0)
        _ = try h.append(0)
        let checkpoint = try h.runner.checkpoint()
        let other = try FlashNextMTPDraftRunner(model: h.model, context: h.context, maxContext: 2, policy: .bounded(slots: 6))
        #expect(throws: FlashNextForwardRunnerError.self) { try other.restore(checkpoint) }
        _ = try h.append(1)
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.append(2) }
        #expect(h.runner.position == 2)
        _ = try h.runner.checkpoint()
        #expect(throws: FlashNextForwardRunnerError.self) {
            _ = try FlashNextMTPDraftRunner(model: h.model, context: h.context, maxContext: 2, policy: .bounded(slots: 5))
        }
    }
}
