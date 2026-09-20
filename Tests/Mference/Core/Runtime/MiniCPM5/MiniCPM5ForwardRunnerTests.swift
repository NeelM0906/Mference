import Foundation
import Metal
import Testing
@testable import Mference

/// MiniCPM5 runtime integration against the toy install: factory dispatch,
/// deterministic replay across instances, reset correctness, chunked prefill
/// equal to sequential decode (bit for bit on the per-row path, FP16-tier
/// and argmax-identical on the batched QMM path), and cursor discipline.
/// Mirrors `Qwen38ForwardRunnerTests` minus the DeltaNet state.
@Suite struct MiniCPM5ForwardRunnerTests {
    private static let vocab = 128

    private func makeRunner(maxContext: Int = 128,
                            runtimeConfiguration: RuntimeConfiguration = .production) throws -> (URL, MetalContext, MiniCPM5ForwardRunner) {
        let dir = try MiniCPM5Parity.installToyCheckpoint()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: .miniCPM5Toy())
        let runner = try MiniCPM5ForwardRunner(model: model, context: ctx,
                                               maxContext: maxContext,
                                               runtimeConfiguration: runtimeConfiguration)
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
        (0..<count).map { Int32(4 + ($0 * 37 + 11) % (vocab - 4)) }
    }

    @Test("cancelled warm append requires reset across KV backends",
          arguments: ["dense", "paged", "spilled"])
    func interruptedWarmAppendRequiresReset(backend: String) async throws {
        let runtime = RuntimeConfiguration(kvPagedPolicy: backend == "dense" ? .off : .on,
            kvTopKPages: 0, kvSinkPages: 1, kvRecentPages: 2,
            kvPoolPagesPerLayer: backend == "spilled" ? 5 : nil)
        let (dir, ctx, runner) = try makeRunner(maxContext: 512, runtimeConfiguration: runtime)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)
        let tokens = Self.prompt(433)
        _ = try await runner.prefillChunked(tokens: tokens.prefix(400), startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 64),
            into: logits, onProgress: { _ in })
        let reference = bits(logits)
        try await runner.produceExactPrefill(token: 7, position: 400, into: logits)
        let referenceNext = bits(logits)
        runner.reset()
        _ = try await runner.prefillChunked(tokens: tokens.prefix(400), startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 64),
            into: logits, onProgress: { _ in })
        runner.prefillDidCompleteLayer = { layer in
            if layer == 0 { withUnsafeCurrentTask { $0?.cancel() } }
        }
        // Exclusively used by this task until its value is awaited below.
        struct SerialBuffer: @unchecked Sendable { let value: MTLBuffer }
        let output = SerialBuffer(value: logits)
        let interrupted = Task { @Sendable in
            _ = try await runner.prefillChunked(tokens: tokens[400..<433], startPosition: 400,
                outputMode: .logits, config: .production(chunkTokens: 64),
                into: output.value, onProgress: { _ in })
        }
        do {
            try await interrupted.value
            Issue.record("expected failure after a real layer wrote KV")
        } catch is CancellationError {}
        #expect(throws: PrefillError.self) {
            try runner.prepareForContinuation(expectedPosition: 400)
        }
        do {
            try await runner.produce(token: 7, position: 400, into: logits)
            Issue.record("dirty state must reject decode")
        } catch let error as PrefillError {
            guard case .chunkedRunnerDirty = error else { throw error }
        }
        runner.prefillDidCompleteLayer = nil
        runner.reset()
        _ = try await runner.prefillChunked(tokens: tokens.prefix(400), startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 64),
            into: logits, onProgress: { _ in })
        #expect(bits(logits) == reference)
        try runner.prepareForContinuation(expectedPosition: 400)
        try await runner.produceExactPrefill(token: 7, position: 400, into: logits)
        #expect(bits(logits) == referenceNext)
    }

    @Test func factory_selectsTheMiniCPM5RunnerWithChunkedPrefillAndFP16KV() throws {
        let dir = try MiniCPM5Parity.installToyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: .miniCPM5Toy())
        let requested = RuntimeConfiguration(prefillEnabled: true, forceLogitsHead: false)
        let runtime = try ForwardRunnerFactory.make(model: model, context: ctx,
                                                    maxContext: 64,
                                                    runtimeConfiguration: requested)
        #expect(runtime.producer is MiniCPM5ForwardRunner)
        #expect(runtime.producer is any ChunkedPrefillRunner)
        #expect(runtime.producer is any HeadlessSequentialPrefillRunner)
        #expect(runtime.producer is any ExactPrefillLogitProducer)
        #expect(runtime.prefillConfig == requested.prefillConfig)
        #expect(runtime.executedPrefillMode == .unreported)
        #expect(runtime.kvStorageMode == .fp16)
        #expect((runtime.producer as? any FusedHeadLogitProducer)?.usesFusedGreedyHead == true)
    }

    /// The runner refuses a config that still carries q/k norms: that family
    /// belongs to the fused-epilogue runners.
    @Test func runner_refusesAQKNormConfig() throws {
        let dir = try MiniCPM5Parity.installToyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: .miniCPM5Toy())
        #expect(model.config.qkNorm == false)
        let qwen38Dir = try Qwen38ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: qwen38Dir) }
        let qwen38 = try Model.load(directoryURL: qwen38Dir, device: ctx.device,
                                    expecting: .qwen38Toy())
        #expect(throws: MiniCPM5ForwardRunnerError.self) {
            _ = try MiniCPM5ForwardRunner(model: qwen38, context: ctx, maxContext: 64)
        }
    }

    @Test func decodeReplay_isDeterministicAcrossInstances() async throws {
        let (dirA, ctxA, runnerA) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dirA) }
        let (dirB, ctxB, runnerB) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dirB) }
        let logitsA = try makeLogits(ctxA)
        let logitsB = try makeLogits(ctxB)

        let tokens: [Int32] = [5, 17, 100, 9]
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

    @Test func reset_restoresEmptyContextState() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)

        try await runner.produce(token: 5, position: 0, into: logits)
        let first = runner.lastGreedyToken
        try await runner.produce(token: Int32(first), position: 1, into: logits)
        #expect(runner.continuationPosition == 2)

        runner.reset()
        #expect(runner.continuationPosition == 0)
        try await runner.produce(token: 5, position: 0, into: logits)
        #expect(runner.lastGreedyToken == first)
    }

    /// The chunked-prefill correctness gate: the final prompt-token logits and
    /// every continuation-decode logits row after a chunked prefill against
    /// sequential decode from a fresh state. Shapes cover a prompt shorter
    /// than one chunk (per-row path: bit-exact), a non-multiple, multi-chunk,
    /// and exactly one full chunk (batched INT4 QMM path: FP16 tier, argmax
    /// identical — the toy's INT4 MLP takes the batched path the real install
    /// takes, which the INT8 Qwen 3.8 toy never exercised).
    @Test("chunked prefill equals sequential decode",
          arguments: [(5, 32), (40, 32), (70, 32), (64, 64)])
    func chunkedPrefill_matchesSequentialDecodeLogits(promptLength: Int,
                                                      chunkTokens: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(maxContext: 96)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)
        let prompt = Self.prompt(promptLength)
        let continuation: [Int32] = [5, 91, 100]

        var reference: [[UInt16]] = []
        for (position, token) in prompt.enumerated() {
            try await runner.produceExactPrefill(token: token, position: position, into: logits)
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
        #expect(result.newPosition == prompt.count)
        #expect(result.seed == .logitsWritten)
        #expect(result.execution?.batchedTokens == prompt.count)
        #expect(result.execution?.replayedTokens == 0)
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
            let wantF = want.map { Float(Float16(bitPattern: $0)) }
            let gotF = got.map { Float(Float16(bitPattern: $0)) }
            let maxAbs = zip(wantF, gotF).map { abs($0 - $1) }.max() ?? 0
            let argmaxWant = wantF.indices.max { wantF[$0] < wantF[$1] }
            let argmaxGot = gotF.indices.max { gotF[$0] < gotF[$1] }
            if promptLength < 32 {
                // Sub-tile chunks replay the decode GEMV row by row: exact.
                #expect(mismatches == 0,
                        "prompt \(promptLength) chunk \(chunkTokens) step \(step): \(mismatches) mismatched logits")
            } else {
                // Tile-tall chunks take the batched INT4 QMM for the projections
                // and the MLP, whose summation order differs from the decode
                // GEMV: equal within the FP16 tier and argmax-identical, not
                // bit-identical. Measured and recorded on the family page.
                #expect(maxAbs <= 1e-2,
                        "prompt \(promptLength) chunk \(chunkTokens) step \(step): maxAbs \(maxAbs) (\(mismatches) differing bits)")
                #expect(argmaxWant == argmaxGot,
                        "prompt \(promptLength) chunk \(chunkTokens) step \(step): argmax differs")
            }
            print("[minicpm5 chunked-vs-sequential] prompt \(promptLength) chunk \(chunkTokens) step \(step): maxAbs \(maxAbs) bits \(mismatches)")
        }
    }

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

    @Test func headlessAndExactPrefillPaths() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)
        let sentinel: UInt16 = 0x7BFF
        let values = logits.contents().bindMemory(to: UInt16.self, capacity: Self.vocab)
        for index in 0..<Self.vocab { values[index] = sentinel }

        let headless: any HeadlessSequentialPrefillRunner = runner
        try await headless.produceWithoutLogits(token: 6, position: 0)
        #expect(bits(logits).allSatisfy { $0 == sentinel })
        #expect(runner.continuationPosition == 1)

        let exact: any ExactPrefillLogitProducer = runner
        try await exact.produceExactPrefill(token: 7, position: 1, into: logits)
        let exactBits = bits(logits)
        #expect(exactBits != Array(repeating: sentinel, count: Self.vocab))
        #expect(runner.continuationPosition == 2)

        runner.reset()
        try await headless.produceWithoutLogits(token: 6, position: 0)
        try await exact.produceExactPrefill(token: 7, position: 1, into: logits)
        #expect(bits(logits) == exactBits)
    }

    @Test func cursorMismatch_isRejected() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)

        try await runner.produce(token: 5, position: 0, into: logits)
        await #expect(throws: MiniCPM5ForwardRunnerError.self) {
            try await runner.produce(token: 5, position: 0, into: logits)
        }
        try runner.prepareForContinuation(expectedPosition: 1)
        #expect(throws: PrefillError.self) {
            try runner.prepareForContinuation(expectedPosition: 2)
        }
    }
}
