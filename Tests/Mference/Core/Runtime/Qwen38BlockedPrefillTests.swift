import Foundation
import Metal
import Testing
@testable import Mference

/// The SSD tier live: a pool smaller than the context forces sealed pages to
/// spill and the blocked (streamed) prefill path to run. Blocked prefill is
/// exact — same math as the resident path, different summation order — so
/// the prefill head must agree with the dense runner. Growing-chat flows
/// (prefill → decode → prefill continuation) must be deterministic under
/// eviction, fetch, and pinning.
@Suite struct Qwen38BlockedPrefillTests {
    private static let vocab = 1024

    private func makeRunner(maxContext: Int,
                            paged: Bool,
                            poolPages: Int? = nil,
                            topK: Int = 0,
                            sink: Int = 1,
                            recent: Int = 2) throws -> (URL, MetalContext, Qwen38ForwardRunner) {
        let dir = try Qwen38ToySynthetic.write()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen38Toy())
        let config = RuntimeConfiguration(prefillEnabled: true,
                                          kvPagedPolicy: paged ? .on : .off,
                                          kvTopKPages: topK,
                                          kvSinkPages: sink,
                                          kvRecentPages: recent,
                                          kvPoolPagesPerLayer: poolPages)
        let runner = try Qwen38ForwardRunner(model: model,
                                             context: ctx,
                                             maxContext: maxContext,
                                             runtimeConfiguration: config)
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

    private static func prompt(_ count: Int, seed: Int = 0) -> [Int32] {
        (0..<count).map { Int32((($0 + seed) * 37 + 11) % vocab) }
    }

    private func prefill(_ runner: Qwen38ForwardRunner,
                         tokens: [Int32], start: Int,
                         logits: MTLBuffer,
                         chunk: Int = 64) async throws {
        _ = try await runner.prefillChunked(tokens: tokens[...],
                                            startPosition: start,
                                            outputMode: .greedyIfAvailable,
                                            config: .production(chunkTokens: chunk),
                                            into: logits,
                                            onProgress: { _ in })
    }

    /// Pool of 5 pages, 400-token prompt (7 pages): later chunks run the
    /// blocked streamed path over spilled pages. The prefill head must agree
    /// with the dense runner's.
    @Test func blockedPrefill_headMatchesDense() async throws {
        let tokens = Self.prompt(400)
        let (dirD, ctxD, dense) = try makeRunner(maxContext: 512, paged: false)
        defer { try? FileManager.default.removeItem(at: dirD) }
        let (dirP, ctxP, paged) = try makeRunner(maxContext: 512, paged: true, poolPages: 5)
        defer { try? FileManager.default.removeItem(at: dirP) }
        let logitsD = try makeLogits(ctxD)
        let logitsP = try makeLogits(ctxP)

        try await prefill(dense, tokens: tokens, start: 0, logits: logitsD)
        try await prefill(paged, tokens: tokens, start: 0, logits: logitsP)
        #expect(dense.lastGreedyToken == paged.lastGreedyToken)
        #expect(paged.continuationPosition == 400)
    }

    /// Same, with a chunk size that does not divide the page size, so chunk
    /// boundaries land mid-page and the tail window carries an unsealed
    /// prefix.
    @Test func blockedPrefill_unalignedChunks_headMatchesDense() async throws {
        let tokens = Self.prompt(410, seed: 3)
        let (dirD, ctxD, dense) = try makeRunner(maxContext: 512, paged: false)
        defer { try? FileManager.default.removeItem(at: dirD) }
        let (dirP, ctxP, paged) = try makeRunner(maxContext: 512, paged: true, poolPages: 5)
        defer { try? FileManager.default.removeItem(at: dirP) }
        let logitsD = try makeLogits(ctxD)
        let logitsP = try makeLogits(ctxP)

        try await prefill(dense, tokens: tokens, start: 0, logits: logitsD, chunk: 32)
        try await prefill(paged, tokens: tokens, start: 0, logits: logitsP, chunk: 32)
        #expect(dense.lastGreedyToken == paged.lastGreedyToken)
    }

    /// Growing chat under a tight pool: prefill, sparse decode, prefill
    /// continuation (blocked, starting mid-page), more decode. Deterministic:
    /// a reset + replay reproduces the identical stream, including LRU
    /// evictions, spill fetches, and selection pinning.
    @Test func growingChat_tightPool_replaysDeterministically() async throws {
        let (dir, ctx, runner) = try makeRunner(maxContext: 512, paged: true,
                                                poolPages: 6, topK: 1)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)
        let turn1 = Self.prompt(200)
        let turn2 = Self.prompt(150, seed: 7)

        func run() async throws -> [UInt32] {
            var stream: [UInt32] = []
            try await prefill(runner, tokens: turn1, start: 0, logits: logits)
            stream.append(runner.lastGreedyToken)
            var position = 200
            var token: Int32 = 9
            for _ in 0..<10 {
                try await runner.produce(token: token, position: position, into: logits)
                stream.append(runner.lastGreedyToken)
                token = Int32(runner.lastGreedyToken % UInt32(Self.vocab))
                position += 1
            }
            try runner.prepareForContinuation(expectedPosition: position)
            try await prefill(runner, tokens: turn2, start: position, logits: logits)
            stream.append(runner.lastGreedyToken)
            position += turn2.count
            for _ in 0..<5 {
                try await runner.produce(token: token, position: position, into: logits)
                stream.append(runner.lastGreedyToken)
                token = Int32(runner.lastGreedyToken % UInt32(Self.vocab))
                position += 1
            }
            #expect(runner.continuationPosition == position)
            return stream
        }

        let first = try await run()
        runner.reset()
        let second = try await run()
        #expect(first == second)
        #expect(first.allSatisfy { $0 < UInt32(Self.vocab) })
    }
}
