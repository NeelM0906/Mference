import Foundation
import Metal
import Testing
@testable import Mference

/// Qwen 3.8 dense runtime integration against the qwen38 toy fixture:
/// factory dispatch, deterministic decode replay across runner instances,
/// KV + GDN state reset correctness, and the sequential-replay prefill v1
/// matching pure decode.
@Suite struct Qwen38ForwardRunnerTests {
    private static let vocab = 1024

    private func makeRunner(maxContext: Int = 64) throws -> (URL, MetalContext, Qwen38ForwardRunner) {
        let dir = try Qwen38ToySynthetic.write()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen38Toy())
        let runner = try Qwen38ForwardRunner(model: model,
                                             context: ctx,
                                             maxContext: maxContext)
        return (dir, ctx, runner)
    }

    private func makeLogits(_ ctx: MetalContext) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: Self.vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    private func bits(_ logits: MTLBuffer) -> [UInt16] {
        Array(UnsafeBufferPointer(
            start: logits.contents().bindMemory(to: UInt16.self, capacity: Self.vocab),
            count: Self.vocab))
    }

    private static func prompt(_ count: Int) -> [Int32] {
        (0..<count).map { Int32(($0 * 37 + 11) % vocab) }
    }

    /// Factory dispatch: a qwen38 model selects the dense runner with FP16
    /// KV storage, and the qwen36 selection is unchanged (mirrors
    /// `qwenFactory_preservesChunkedFusedRunnerSelection`).
    @Test func qwen38Factory_selectsDenseRunnerAndPreservesQwen36() throws {
        let dir = try Qwen38ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: .qwen38Toy())
        let requested = RuntimeConfiguration(prefillEnabled: true, forceLogitsHead: false)
        let runtime = try ForwardRunnerFactory.make(model: model, context: ctx,
                                                    maxContext: 64,
                                                    runtimeConfiguration: requested)
        #expect(runtime.producer is Qwen38ForwardRunner)
        #expect(runtime.producer is any ChunkedPrefillRunner)
        #expect(runtime.producer is any HeadlessSequentialPrefillRunner)
        #expect(runtime.producer is any ExactPrefillLogitProducer)
        #expect(runtime.prefillConfig == requested.prefillConfig)
        #expect(runtime.executedPrefillMode == .chunked)
        #expect(runtime.kvStorageMode == .fp16)
        #expect((runtime.producer as? any FusedHeadLogitProducer)?.usesFusedGreedyHead == true)

        let qwen36Dir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: qwen36Dir) }
        let qwen36Model = try Model.load(directoryURL: qwen36Dir, device: ctx.device,
                                         expecting: .qwen36Toy())
        let qwen36Runtime = try ForwardRunnerFactory.make(model: qwen36Model, context: ctx,
                                                          maxContext: 64,
                                                          runtimeConfiguration: requested)
        #expect(qwen36Runtime.producer is RealForwardRunner)
        #expect(qwen36Runtime.kvStorageMode == .fp16)
    }

    /// Deterministic decode replay: two fresh runner instances fed the same
    /// tokens must produce identical greedy tokens at every step.
    @Test func decodeReplay_isDeterministicAcrossInstances() async throws {
        let (dirA, ctxA, runnerA) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dirA) }
        let (dirB, ctxB, runnerB) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dirB) }
        let logitsA = try makeLogits(ctxA)
        let logitsB = try makeLogits(ctxB)

        let tokens: [Int32] = [1, 17, 300, 5]
        var greedyA: [UInt32] = []
        var greedyB: [UInt32] = []
        for (position, token) in tokens.enumerated() {
            try await runnerA.produce(token: token, position: position, into: logitsA)
            try await runnerB.produce(token: token, position: position, into: logitsB)
            greedyA.append(runnerA.lastGreedyToken)
            greedyB.append(runnerB.lastGreedyToken)
        }
        #expect(greedyA == greedyB)
        #expect(greedyA.allSatisfy { $0 < UInt32(Self.vocab) })
        #expect(runnerA.continuationPosition == tokens.count)
    }

    /// Sequential-vs-restart reset correctness: the same input from the empty
    /// state must reproduce the same argmax after reset() — this fails if
    /// reset() leaves stale KV rows, GDN recurrent state, or conv tail.
    @Test func reset_restoresEmptyContextState() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)

        try await runner.produce(token: 1, position: 0, into: logits)
        let first = runner.lastGreedyToken
        try await runner.produce(token: Int32(first), position: 1, into: logits)
        #expect(runner.continuationPosition == 2)

        runner.reset()
        #expect(runner.continuationPosition == 0)
        try await runner.produce(token: 1, position: 0, into: logits)
        #expect(runner.lastGreedyToken == first)
    }

    /// Prefill v1 is a sequential decode replay, so prefilling a prompt then
    /// decoding must match pure decode from a fresh state exactly (shared
    /// KV + GDN recurrent state + conv tail).
    @Test func prefillChunked_matchesPureDecode() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)

        // Pure decode reference over a mixed linear/full-layer state history.
        let prompt: [Int32] = [11, 42, 7]
        for (position, token) in prompt.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
        }
        try await runner.produce(token: 9, position: prompt.count, into: logits)
        let reference = runner.lastGreedyToken

        runner.reset()
        var progress: [Int] = []
        let result = try await runner.prefillChunked(
            tokens: prompt[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { progress.append($0) })
        #expect(result.newPosition == prompt.count)
        if case .greedyToken(let seed) = result.seed {
            #expect(seed < UInt32(Self.vocab))
        } else {
            Issue.record("expected a greedy seed token from the fused head")
        }
        #expect(progress.last == prompt.count)
        #expect(runner.continuationPosition == prompt.count)

        try await runner.produce(token: 9, position: prompt.count, into: logits)
        #expect(runner.lastGreedyToken == reference)
    }

    /// The chunked-prefill correctness gate (mirrors the Maple
    /// `mapleChunkedPrefill_matchesSequentialLogitsAndContinuation` pattern):
    /// the final prompt-token logits and every continuation-decode logits row
    /// after a chunked prefill must be identical to sequential decode from a
    /// fresh state. Shapes cover a prompt shorter than one chunk (5/32), a
    /// prompt that is not a chunk multiple (40/32), a multi-chunk prompt
    /// (70/32), and exactly one full chunk (64/64).
    @Test("chunked prefill equals sequential decode",
          arguments: [(5, 32), (40, 32), (70, 32), (64, 64)])
    func chunkedPrefill_matchesSequentialDecodeLogits(promptLength: Int,
                                                      chunkTokens: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(maxContext: 96)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)
        let prompt = Self.prompt(promptLength)
        let continuation: [Int32] = [5, 91, 200]

        // Sequential-decode reference: every prompt and continuation token
        // through the decode path with the exact logits head.
        var reference: [[UInt16]] = []
        for (position, token) in prompt.enumerated() {
            try await runner.produceExactPrefill(token: token, position: position,
                                                 into: logits)
        }
        reference.append(bits(logits))
        for (index, token) in continuation.enumerated() {
            try await runner.produceExactPrefill(token: token,
                                                 position: prompt.count + index,
                                                 into: logits)
            reference.append(bits(logits))
        }

        runner.reset()
        var progress: [Int] = []
        let result = try await runner.prefillChunked(
            tokens: prompt[...],
            startPosition: 0,
            outputMode: .logits,
            config: .production(chunkTokens: chunkTokens),
            into: logits,
            onProgress: { progress.append($0) })
        #expect(result == PrefillResult(newPosition: prompt.count, seed: .logitsWritten))
        #expect(progress.last == prompt.count)
        #expect(runner.continuationPosition == prompt.count)
        var chunked: [[UInt16]] = [bits(logits)]
        for (index, token) in continuation.enumerated() {
            try await runner.produceExactPrefill(token: token,
                                                 position: prompt.count + index,
                                                 into: logits)
            chunked.append(bits(logits))
        }

        #expect(reference.count == chunked.count)
        for (step, (want, got)) in zip(reference, chunked).enumerated() {
            let mismatches = zip(want, got).filter { $0 != $1 }.count
            #expect(mismatches == 0,
                    "prompt \(promptLength) chunk \(chunkTokens) step \(step): \(mismatches) mismatched logits")
        }
    }

    /// Fused-greedy continuation across a chunk boundary: a multi-chunk
    /// prefill must seed the same greedy token as pure decode and continue
    /// decoding identically (KV rows, GDN state, and conv tails all carry
    /// across chunks).
    @Test func chunkedPrefill_multiChunkGreedyContinuation_matchesPureDecode() async throws {
        let (dir, ctx, runner) = try makeRunner(maxContext: 96)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)
        let prompt = Self.prompt(40)

        for (position, token) in prompt.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
        }
        let referenceSeed = runner.lastGreedyToken
        try await runner.produce(token: 9, position: prompt.count, into: logits)
        let referenceNext = runner.lastGreedyToken

        runner.reset()
        let result = try await runner.prefillChunked(
            tokens: prompt[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        #expect(result.newPosition == prompt.count)
        if case .greedyToken(let seed) = result.seed {
            #expect(seed == referenceSeed)
        } else {
            Issue.record("expected a greedy seed token from the fused head")
        }
        try await runner.produce(token: 9, position: prompt.count, into: logits)
        #expect(runner.lastGreedyToken == referenceNext)
    }

    /// Headless and exact-prefill paths: produceWithoutLogits advances state
    /// without touching the logits buffer; produceExactPrefill then writes a
    /// full logits row identical to the plain logits-head decode.
    @Test func headlessAndExactPrefillPaths() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)
        let sentinel: UInt16 = 0x7BFF
        let values = logits.contents().bindMemory(to: UInt16.self, capacity: Self.vocab)
        for index in 0..<Self.vocab { values[index] = sentinel }

        let headless: any HeadlessSequentialPrefillRunner = runner
        try await headless.produceWithoutLogits(token: 3, position: 0)
        #expect(bits(logits).allSatisfy { $0 == sentinel })
        #expect(runner.continuationPosition == 1)

        let exact: any ExactPrefillLogitProducer = runner
        try await exact.produceExactPrefill(token: 4, position: 1, into: logits)
        let exactBits = bits(logits)
        #expect(exactBits != Array(repeating: sentinel, count: Self.vocab))
        #expect(runner.continuationPosition == 2)

        // The exact head must be reproducible from a replayed state.
        runner.reset()
        try await headless.produceWithoutLogits(token: 3, position: 0)
        try await exact.produceExactPrefill(token: 4, position: 1, into: logits)
        #expect(bits(logits) == exactBits)
    }

    /// Cursor discipline mirrors the other runners: produce at the wrong
    /// position and stale continuation cursors are rejected.
    @Test func cursorMismatch_isRejected() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)

        try await runner.produce(token: 1, position: 0, into: logits)
        await #expect(throws: Qwen38ForwardRunnerError.self) {
            try await runner.produce(token: 1, position: 0, into: logits)
        }
        try runner.prepareForContinuation(expectedPosition: 1)
        #expect(throws: PrefillError.self) {
            try runner.prepareForContinuation(expectedPosition: 2)
        }
    }
}
