import Foundation
import Metal
import Testing
@testable import Mference

@Suite(.serialized) struct FlashNextMTPGreedyVerifierTests {
    private func bits(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt16.self), count: count))
    }

    private func sample(_ scratch: RawCompletionScratch, context: MetalContext, position: Int) throws -> Int32 {
        let cb = try #require(context.queue.makeCommandBuffer())
        scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
            history: [], config: .init(temperature: 0), position: position, outToken: scratch.outToken)
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.error == nil)
        return Int32(bitPattern: scratch.outToken.contents().assumingMemoryBound(to: UInt32.self).pointee)
    }

    @Test(arguments: [false, true])
    func acceptedRejectedAndInterruptedRoundsMatchPlainTarget(resident: Bool) async throws {
        let directory = try FlashNextToySynthetic.write(includeMTP: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: .qwen38FlashNextToy(), streamingMode: .pread(slotCount: 16))
        try await verify(model: model, context: context, resident: resident)
    }

    @Test func installedAcceptedRejectedAndInterruptedRoundsMatchPlainTarget() async throws {
        guard let path = ProcessInfo.processInfo.environment["MFERENCE_FLASHNEXT_GTURBO"] else { return }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device,
            streamingMode: .pread(slotCount: 16))
        try await verify(model: model, context: context, resident: false)
        print("[installed MTP verifier] accepted/rejected/bonus/budget/stop/context/recovery match plain target full logits exactly; sequential reference only")
    }

    private func verify(model: Model, context: MetalContext, resident: Bool) async throws {
        let verifier = try FlashNextMTPGreedyVerifier(model: model, context: context, maxContext: 48,
            policy: resident ? .resident : .bounded(slots: 16))
        let plain = try FlashNextForwardRunner(model: model, context: context, maxContext: 48)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
            logitSoftcap: Float(model.config.finalLogitSoftcap))
        let logits = scratch.logits
        // Cross the toy's compressed-indexer boundary and a prefill chunk.
        let prefix = (0..<35).map { Int32(4 + ($0 * 17) % 53) }
        _ = try await plain.prefillChunked(tokens: prefix[...], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32), into: logits, onProgress: { _ in })
        var expected: [Int32] = []
        var heads: [[UInt16]] = []
        for _ in 0..<8 {
            let token = try sample(scratch, context: context, position: plain.continuationPosition)
            expected.append(token)
            try await plain.produce(token: token, position: plain.continuationPosition, into: logits)
            heads.append(bits(logits, count: model.config.vocabSize))
        }

        // Reject at each proposal position; a fully matching round adds bonus.
        for matches in 0...3 {
            try await verifier.prefill(prefix)
            var proposals = Array(expected.prefix(3))
            if matches < 3 { proposals[matches] = (proposals[matches] + 1) % Int32(model.config.vocabSize) }
            let result = try await verifier.verify(proposals, budget: 8)
            #expect(result.accepted == matches)
            #expect(result.tokens == Array(expected.prefix(matches + 1)))
            #expect(!result.reachedStop)
            #expect(bits(verifier.logits, count: model.config.vocabSize) == heads[matches])
            #expect(verifier.target.continuationPosition == prefix.count + matches + 1)
            #expect(verifier.primer.targetPosition == verifier.target.continuationPosition)
            #expect(verifier.primer.draftPosition == verifier.target.continuationPosition - 1)
            let next = try await verifier.verify([], budget: 1)
            #expect(next.tokens == [expected[matches + 1]])
            #expect(bits(verifier.logits, count: model.config.vocabSize) == heads[matches + 1])
        }

        try await verifier.prefill(prefix)
        let initialHead = bits(verifier.logits, count: model.config.vocabSize)
        verifier.didVerifyToken = { if $0 == 2 { throw CancellationError() } }
        do {
            _ = try await verifier.verify(Array(expected.prefix(3)), budget: 8)
            Issue.record("expected interrupted verification")
        } catch is CancellationError { }
        #expect(verifier.target.continuationPosition == prefix.count)
        #expect(verifier.primer.targetPosition == prefix.count)
        #expect(bits(verifier.logits, count: model.config.vocabSize) == initialHead)
        verifier.didVerifyToken = nil
        let recovered = try await verifier.verify(Array(expected.prefix(3)), budget: 8)
        #expect(recovered.tokens == Array(expected.prefix(4)))
        #expect(bits(verifier.logits, count: model.config.vocabSize) == heads[3])

        // Throw inside the target-row consumer, before target commit, leaving
        // both owners partially advanced/dirty rather than only a clean round.
        try await verifier.prefill(prefix)
        verifier.primer.didPrimeRow = { if $0 == prefix.count { throw CancellationError() } }
        do {
            _ = try await verifier.verify(Array(expected.prefix(3)), budget: 8)
            Issue.record("expected partial consumer interruption")
        } catch is CancellationError { }
        #expect(verifier.target.continuationPosition == prefix.count)
        #expect(verifier.primer.targetPosition == prefix.count)
        #expect(bits(verifier.logits, count: model.config.vocabSize) == initialHead)
        verifier.primer.didPrimeRow = nil
        let retried = try await verifier.verify(Array(expected.prefix(3)), budget: 8)
        #expect(retried.tokens == Array(expected.prefix(4)))
        #expect(bits(verifier.logits, count: model.config.vocabSize) == heads[3])

        try await verifier.prefill(prefix)
        let stopped = try await verifier.verify(Array(expected.prefix(3)), budget: 8, stopTokens: [expected[0]])
        #expect(stopped.tokens == [expected[0]] && stopped.reachedStop)
        #expect(bits(verifier.logits, count: model.config.vocabSize) == heads[0])
        try await verifier.prefill(prefix)
        let capped = try await verifier.verify(Array(expected.prefix(3)), budget: 2)
        #expect(capped.tokens == Array(expected.prefix(2)) && capped.accepted == 2)
        #expect(bits(verifier.logits, count: model.config.vocabSize) == heads[1])
        let invalidHead = bits(verifier.logits, count: model.config.vocabSize)
        do {
            _ = try await verifier.verify([-1], budget: 1)
            Issue.record("expected invalid proposal rejection")
        } catch is FlashNextForwardRunnerError { }
        #expect(bits(verifier.logits, count: model.config.vocabSize) == invalidHead)

        let limited = try FlashNextMTPGreedyVerifier(model: model, context: context, maxContext: 37,
            policy: .bounded(slots: 16))
        try await limited.prefill(prefix)
        let exhausted = try await limited.verify(Array(expected.prefix(3)), budget: 8)
        #expect(exhausted.tokens == Array(expected.prefix(2)))
        #expect(limited.target.continuationPosition == 37)
        do {
            _ = try await limited.verify([], budget: 1)
            Issue.record("expected exhausted context rejection")
        } catch is FlashNextForwardRunnerError { }
        limited.reset()
        do {
            _ = try await limited.verify([], budget: 1)
            Issue.record("expected unprimed state rejection")
        } catch is FlashNextForwardRunnerError { }
    }
}
