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
                                    chunkTokens: Int = 32) throws -> Harness {
        let dir = try DSV4ToySynthetic.write()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .deepseekV4Toy(),
                                   streamingMode: .pread(slotCount: 16))
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
                                   chunkTokens: Int) async throws -> [[UInt16]] {
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
    /// selection cutover for the toy config is absolute position 48
    /// (`indexTopK 12 * csaCompressRate 4`).
    ///
    /// - 24 tokens / chunk 32: one full chunk.
    /// - 45 tokens / chunk 64: one ragged chunk.
    /// - 45 tokens / chunk 32: a full chunk plus a 13-token ragged tail, so
    ///   the compressor's pending window, the prior-Ca carry, and the window
    ///   ring all have to survive a chunk boundary mid-window.
    @Test("chunked prefill equals sequential decode",
          arguments: [(24, 32), (45, 64), (45, 32), (32, 32)])
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

    /// A prompt long enough that the second span crosses the lightning-indexer
    /// cutover, so one call mixes a batched chunk and a token-by-token
    /// fallback chunk. The result must still match pure decode.
    @Test("mixed batched and fallback spans equal sequential decode")
    func mixedSpansMatchSequentialDecode() async throws {
        let harness = try Self.makeHarness(chunkTokens: 32)
        defer { try? FileManager.default.removeItem(at: harness.dir) }
        let tokens = Self.prompt(60)
        #expect(DSV4ChunkedPrefill.supports(config: .deepseekV4Toy(),
                                            startPosition: 0, tokenCount: 32,
                                            expertCacheSlots: 16))
        #expect(!DSV4ChunkedPrefill.supports(config: .deepseekV4Toy(),
                                             startPosition: 32, tokenCount: 28,
                                             expertCacheSlots: 16))
        let continuation: [Int32] = [17, 42]
        let reference = try await Self.decodeReference(harness, tokens: tokens,
                                                       continuation: continuation)
        let chunked = try await Self.chunkedRun(harness, tokens: tokens,
                                                continuation: continuation,
                                                chunkTokens: 32)
        Self.expectIdentical(reference, chunked, label: "mixed spans")
    }

    /// A single span that crosses the lightning-selection cutover (the
    /// `--prefill-chunk auto` shape for long prompts) must batch its eligible
    /// prefix and replay only the remainder token-by-token, splitting at the
    /// cutover even mid compressor window — position 51 with the toy's
    /// `indexTopK 12 × rate 4`.
    @Test("span crossing the lightning cutover batches its eligible prefix")
    func cutoverCrossingSpanBatchesPrefix() async throws {
        let harness = try Self.makeHarness(chunkTokens: 64)
        defer { try? FileManager.default.removeItem(at: harness.dir) }
        let tokens = Self.prompt(60)
        #expect(DSV4ChunkedPrefill.batchedTokenPrefix(config: .deepseekV4Toy(),
                                                      startPosition: 0,
                                                      tokenCount: 60,
                                                      expertCacheSlots: 16) == 51)
        #expect(!DSV4ChunkedPrefill.supports(config: .deepseekV4Toy(),
                                             startPosition: 0, tokenCount: 60,
                                             expertCacheSlots: 16))
        let continuation: [Int32] = [17, 42]
        let reference = try await Self.decodeReference(harness, tokens: tokens,
                                                       continuation: continuation)
        let chunked = try await Self.chunkedRun(harness, tokens: tokens,
                                                continuation: continuation,
                                                chunkTokens: 64)
        Self.expectIdentical(reference, chunked, label: "cutover split")
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
