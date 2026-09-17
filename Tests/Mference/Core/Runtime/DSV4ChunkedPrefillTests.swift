import Foundation
import Metal
import Testing
@testable import Mference

/// Pins the DeepSeek-V4 chunked-prefill contract: `prefillChunked(T tokens)`
/// must be indistinguishable from `T` sequential `produce(...)` calls, both in
/// the logits it leaves behind and in the attention state it hands to the
/// decoder afterwards.
///
/// The fixture (`DSV4ToySynthetic`) covers every DSV4 layer flavour in four
/// layers — window-only, CSA (compressor + lightning-indexer key emission),
/// HCA — with hash-routed and learned-router MoE layers, INT2 experts, a
/// 16-slot sliding-window ring that wraps several times over these prompts,
/// and enough live experts per chunk to span more than one routed tile.
@Suite(.serialized)
struct DSV4ChunkedPrefillTests {

    private struct Harness {
        let dir: URL
        let ctx: MetalContext
        let runner: RealForwardRunner
        let logits: MTLBuffer
        let vocab: Int
    }

    private static func makeHarness(maxContext: Int = 96,
                                    chunkTokens: Int = 32,
                                    streamingMode: ExpertStreamingMode = .pread(slotCount: 16)) throws -> Harness {
        let dir = try DSV4ToySynthetic.write()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .deepseekV4Toy(),
                                   streamingMode: streamingMode)
        let runtime = RuntimeConfiguration(expertCacheSlots: 16,
                                           prefillChunkTokens: chunkTokens,
                                           forceLogitsHead: true)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: maxContext,
                                           runtimeConfiguration: runtime)
        let vocab = model.config.vocabSize
        guard let buf = ctx.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return Harness(dir: dir, ctx: ctx, runner: runner, logits: buf, vocab: vocab)
    }

    /// Deterministic prompt inside the toy vocabulary.
    private static func prompt(_ count: Int) -> [Int32] {
        (0..<count).map { Int32(($0 &* 37 &+ 11) % 251) }
    }

    private static func snapshot(_ harness: Harness) -> [UInt16] {
        let ptr = harness.logits.contents().bindMemory(to: UInt16.self,
                                                       capacity: harness.vocab)
        return (0..<harness.vocab).map { ptr[$0] }
    }

    /// Runs `tokens` through the decode path one at a time, then `extra` more
    /// continuation steps, returning the raw logits bits after each of the
    /// last `1 + extra` positions.
    private static func decodeReference(_ harness: Harness,
                                        tokens: [Int32],
                                        continuation: [Int32]) async throws -> [[UInt16]] {
        harness.runner.reset()
        var out: [[UInt16]] = []
        for (i, token) in tokens.enumerated() {
            try await harness.runner.produce(token: token, position: i,
                                             into: harness.logits)
        }
        out.append(snapshot(harness))
        for (i, token) in continuation.enumerated() {
            try await harness.runner.produce(token: token,
                                             position: tokens.count + i,
                                             into: harness.logits)
            out.append(snapshot(harness))
        }
        return out
    }

    private static func chunkedRun(_ harness: Harness,
                                   tokens: [Int32],
                                   continuation: [Int32],
                                   chunkTokens: Int,
                                   expectedBatches: [Int]? = nil) async throws -> [[UInt16]] {
        harness.runner.reset()
        var out: [[UInt16]] = []
        let result = try await harness.runner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .logits,
            config: .production(chunkTokens: chunkTokens),
            into: harness.logits,
            onProgress: { _ in })
        #expect(result.newPosition == tokens.count)
        let execution = try #require(result.execution)
        #expect(execution.computedTokens == tokens.count)
        #expect(execution.replayedTokens == 0)
        #expect(execution.batchedTokens == tokens.count)
        if let expectedBatches {
            #expect(execution.batchedChunkSizes == expectedBatches)
        }
        #expect(execution.executedMode == .chunked)
        #expect(execution.replayReasons.isEmpty)
        #expect(harness.runner.continuationPosition == tokens.count)
        out.append(snapshot(harness))
        for (i, token) in continuation.enumerated() {
            try await harness.runner.produce(token: token,
                                             position: tokens.count + i,
                                             into: harness.logits)
            out.append(snapshot(harness))
        }
        return out
    }

    private static func expectIdentical(_ want: [[UInt16]], _ got: [[UInt16]],
                                        label: String) {
        #expect(want.count == got.count, "\(label): step count")
        for (step, (w, g)) in zip(want, got).enumerated() {
            let mismatches = zip(w, g).filter { $0 != $1 }.count
            #expect(mismatches == 0,
                    "\(label): step \(step) has \(mismatches) mismatched logits")
            if mismatches > 0, let first = zip(w, g).enumerated()
                .first(where: { $0.element.0 != $0.element.1 }) {
                let want = String(first.element.0, radix: 16)
                let got = String(first.element.1, radix: 16)
                Issue.record("\(label): first mismatch at vocab \(first.offset): want 0x\(want) got 0x\(got)")
            }
        }
    }

    /// The core contract. Every case is a *batched* chunk: the CSA lightning
    /// selection cutover for the toy config is absolute position 51
    /// (the 13th compressed entry at rate 4).
    ///
    /// - 24 tokens / chunk 32: one full chunk.
    /// - 45 tokens / chunk 64: one ragged chunk.
    /// - 45 tokens / chunk 32: a full chunk plus a 13-token ragged tail, so
    ///   the compressor's pending window, the prior-Ca carry, and the window
    ///   ring all have to survive a chunk boundary mid-window.
    @Test("chunked prefill equals sequential decode",
          arguments: [(24, 32), (45, 64), (45, 32), (32, 32), (51, 32), (52, 32), (85, 32)])
    func chunkedPrefillMatchesSequentialDecode(promptLength: Int,
                                              chunkTokens: Int) async throws {
        let harness = try Self.makeHarness(chunkTokens: chunkTokens)
        defer { try? FileManager.default.removeItem(at: harness.dir) }
        let tokens = Self.prompt(promptLength)
        let continuation: [Int32] = [5, 91, 200]
        let reference = try await Self.decodeReference(harness, tokens: tokens,
                                                       continuation: continuation)
        let chunked = try await Self.chunkedRun(harness, tokens: tokens,
                                                continuation: continuation,
                                                chunkTokens: chunkTokens)
        Self.expectIdentical(reference, chunked,
                             label: "prompt \(promptLength) chunk \(chunkTokens)")
    }

    /// Both spans stay batched even when the second crosses the cutover.
    @Test("batched sparse spans equal sequential decode")
    func sparseSpansMatchSequentialDecode() async throws {
        let harness = try Self.makeHarness(chunkTokens: 32)
        defer { try? FileManager.default.removeItem(at: harness.dir) }
        let tokens = Self.prompt(60)
        #expect(DSV4ChunkedPrefill.supports(config: .deepseekV4Toy(),
                                            startPosition: 0, tokenCount: 32,
                                            expertCacheSlots: 16))
        #expect(DSV4ChunkedPrefill.supports(config: .deepseekV4Toy(),
                                             startPosition: 32, tokenCount: 28,
                                             expertCacheSlots: 16))
        let continuation: [Int32] = [17, 42]
        let reference = try await Self.decodeReference(harness, tokens: tokens,
                                                       continuation: continuation)
        let chunked = try await Self.chunkedRun(harness, tokens: tokens,
                                                continuation: continuation,
                                                chunkTokens: 32,
                                                expectedBatches: [32, 28])
        Self.expectIdentical(reference, chunked, label: "sparse spans")
    }

    /// A single span crossing the cutover stays one batch, including queries
    /// that share compressed entries and those that emit a new entry.
    @Test("span crossing the lightning cutover batches every token")
    func cutoverCrossingSpanBatchesEveryToken() async throws {
        let harness = try Self.makeHarness(chunkTokens: 64)
        defer { try? FileManager.default.removeItem(at: harness.dir) }
        let tokens = Self.prompt(60)
        #expect(DSV4ChunkedPrefill.batchedTokenPrefix(config: .deepseekV4Toy(),
                                                      startPosition: 0,
                                                      tokenCount: 60,
                                                      expertCacheSlots: 16) == 60)
        #expect(DSV4ChunkedPrefill.supports(config: .deepseekV4Toy(),
                                             startPosition: 0, tokenCount: 60,
                                             expertCacheSlots: 16))
        let continuation: [Int32] = [17, 42]
        let reference = try await Self.decodeReference(harness, tokens: tokens,
                                                       continuation: continuation)
        let chunked = try await Self.chunkedRun(harness, tokens: tokens,
                                                continuation: continuation,
                                                chunkTokens: 64,
                                                expectedBatches: [60])
        Self.expectIdentical(reference, chunked, label: "cutover batch")
    }

    @Test func warmAppendsAcrossSparseCutoverMatchDecode() async throws {
        let h = try Self.makeHarness()
        defer { try? FileManager.default.removeItem(at: h.dir) }
        let tokens = Self.prompt(79)
        let reference = try await Self.decodeReference(h, tokens: tokens, continuation: [5, 17, 42, 91])
        h.runner.reset()
        var start = 0
        for length in [47, 4, 1, 27] {
            let result = try await h.runner.prefillChunked(tokens: tokens[start..<(start + length)],
                startPosition: start, outputMode: .logits, config: .production(chunkTokens: 32),
                into: h.logits, onProgress: { _ in })
            #expect(result.execution?.batchedTokens == length)
            #expect(result.execution?.replayedTokens == 0)
            start += length
            try h.runner.prepareForContinuation(expectedPosition: start)
        }
        var actual = [Self.snapshot(h)]
        for (i, token) in [Int32(5), 17, 42, 91].enumerated() {
            try await h.runner.produce(token: token, position: start + i, into: h.logits)
            actual.append(Self.snapshot(h))
        }
        Self.expectIdentical(reference, actual, label: "warm sparse append")
    }

    @Test(arguments: ["8", "16", "resident"])
    func sparsePrefillProfilesKeepExactState(profile: String) async throws {
        let mode: ExpertStreamingMode = profile == "resident" ? .resident : .pread(slotCount: Int(profile)!)
        let h = try Self.makeHarness(streamingMode: mode)
        defer { try? FileManager.default.removeItem(at: h.dir) }
        let tokens = Self.prompt(71)
        let reference = try await Self.decodeReference(h, tokens: tokens, continuation: [7, 19, 31, 43, 55, 67])
        let actual = try await Self.chunkedRun(h, tokens: tokens,
            continuation: [7, 19, 31, 43, 55, 67], chunkTokens: 32, expectedBatches: [32, 32, 7])
        Self.expectIdentical(reference, actual, label: "expert profile \(profile)")
    }

    @Test func partialChunkCancellationRequiresResetEvenOnWarmContinuation() async throws {
        let h = try Self.makeHarness()
        defer { try? FileManager.default.removeItem(at: h.dir) }
        let tokens = Self.prompt(60)
        let reference = try await Self.chunkedRun(h, tokens: tokens, continuation: [3], chunkTokens: 32)
        h.runner.reset()
        _ = try await h.runner.prefillChunked(tokens: tokens.prefix(16), startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32), into: h.logits, onProgress: { _ in })
        h.runner.dsv4PrefillDidCompleteLayer = { if $0 == 2 { throw CancellationError() } }
        do {
            _ = try await h.runner.prefillChunked(tokens: tokens.dropFirst(16), startPosition: 16,
                outputMode: .logits, config: .production(chunkTokens: 32), into: h.logits, onProgress: { _ in })
            Issue.record("expected mid-chunk cancellation")
        } catch is CancellationError { }
        #expect(throws: PrefillError.self) { try h.runner.prepareForContinuation(expectedPosition: 16) }
        do {
            try await h.runner.produce(token: tokens[16], position: 16, into: h.logits)
            Issue.record("dirty state must not decode")
        } catch let error as PrefillError {
            guard case .chunkedRunnerDirty = error else { throw error }
        }
        h.runner.dsv4PrefillDidCompleteLayer = nil
        let recovered = try await Self.chunkedRun(h, tokens: tokens, continuation: [3], chunkTokens: 32)
        Self.expectIdentical(reference, recovered, label: "reset after cancellation")
    }

    /// Two runs of the same chunked prefill must agree, so the scratch reuse
    /// across chunks and layers carries no state between calls.
    @Test("chunked prefill is reproducible across runs")
    func chunkedPrefillIsReproducible() async throws {
        let harness = try Self.makeHarness(chunkTokens: 32)
        defer { try? FileManager.default.removeItem(at: harness.dir) }
        let tokens = Self.prompt(40)
        let first = try await Self.chunkedRun(harness, tokens: tokens,
                                              continuation: [3], chunkTokens: 32)
        let second = try await Self.chunkedRun(harness, tokens: tokens,
                                               continuation: [3], chunkTokens: 32)
        Self.expectIdentical(first, second, label: "repeat")
    }
}
