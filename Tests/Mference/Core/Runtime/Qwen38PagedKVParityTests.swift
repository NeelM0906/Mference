import Foundation
import Metal
import Testing
@testable import Mference

/// E2E parity for the paged long-context mode against the qwen38 toy: with a
/// selection budget that covers every page, paged decode must reproduce the
/// dense runner's exact greedy stream (the paged kernel is bit-identical
/// under full selection), through pure decode, reset, and prefill + decode.
/// A tight-budget sparse run must still decode without faulting.
@Suite struct Qwen38PagedKVParityTests {
    private static let vocab = 1024

    private func makeRunner(maxContext: Int,
                            paged: Bool,
                            topK: Int = 1024,
                            sink: Int = 2,
                            recent: Int = 4) throws -> (URL, MetalContext, Qwen38ForwardRunner) {
        let dir = try Qwen38ToySynthetic.write()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen38Toy())
        let config = RuntimeConfiguration(prefillEnabled: true,
                                          kvPagedPolicy: paged ? .on : .off,
                                          kvTopKPages: topK,
                                          kvSinkPages: sink,
                                          kvRecentPages: recent)
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

    private static func prompt(_ count: Int) -> [Int32] {
        (0..<count).map { Int32(($0 * 37 + 11) % vocab) }
    }

    /// Pure decode across several page boundaries: greedy streams identical.
    @Test func pagedFullSelection_decodeMatchesDense() async throws {
        let steps = 200        // crosses three 64-token page boundaries
        let (dirD, ctxD, dense) = try makeRunner(maxContext: 256, paged: false)
        defer { try? FileManager.default.removeItem(at: dirD) }
        let (dirP, ctxP, paged) = try makeRunner(maxContext: 256, paged: true)
        defer { try? FileManager.default.removeItem(at: dirP) }
        let logitsD = try makeLogits(ctxD)
        let logitsP = try makeLogits(ctxP)

        var token: Int32 = 7
        var denseStream: [UInt32] = []
        var pagedStream: [UInt32] = []
        for position in 0..<steps {
            try await dense.produce(token: token, position: position, into: logitsD)
            try await paged.produce(token: token, position: position, into: logitsP)
            denseStream.append(dense.lastGreedyToken)
            pagedStream.append(paged.lastGreedyToken)
            token = Int32(dense.lastGreedyToken % UInt32(Self.vocab))
        }
        #expect(denseStream == pagedStream)
    }

    /// Chunked prefill + decode continuation: identical to the dense runner.
    @Test func pagedFullSelection_prefillPlusDecodeMatchesDense() async throws {
        let promptTokens = Self.prompt(150)
        let decodeSteps = 40
        let (dirD, ctxD, dense) = try makeRunner(maxContext: 256, paged: false)
        defer { try? FileManager.default.removeItem(at: dirD) }
        let (dirP, ctxP, paged) = try makeRunner(maxContext: 256, paged: true)
        defer { try? FileManager.default.removeItem(at: dirP) }
        let logitsD = try makeLogits(ctxD)
        let logitsP = try makeLogits(ctxP)
        let config = PrefillRuntimeConfig.production(chunkTokens: 64)

        _ = try await dense.prefillChunked(tokens: promptTokens[...],
                                           startPosition: 0,
                                           outputMode: .greedyIfAvailable,
                                           config: config,
                                           into: logitsD,
                                           onProgress: { _ in })
        _ = try await paged.prefillChunked(tokens: promptTokens[...],
                                           startPosition: 0,
                                           outputMode: .greedyIfAvailable,
                                           config: config,
                                           into: logitsP,
                                           onProgress: { _ in })
        #expect(dense.lastGreedyToken == paged.lastGreedyToken)

        var token = Int32(dense.lastGreedyToken % UInt32(Self.vocab))
        var denseStream: [UInt32] = []
        var pagedStream: [UInt32] = []
        for step in 0..<decodeSteps {
            let position = promptTokens.count + step
            try await dense.produce(token: token, position: position, into: logitsD)
            try await paged.produce(token: token, position: position, into: logitsP)
            denseStream.append(dense.lastGreedyToken)
            pagedStream.append(paged.lastGreedyToken)
            token = Int32(dense.lastGreedyToken % UInt32(Self.vocab))
        }
        #expect(denseStream == pagedStream)
    }

    /// Reset drops paged state completely: a rerun reproduces the first run.
    @Test func pagedReset_replaysIdentically() async throws {
        let (dir, ctx, runner) = try makeRunner(maxContext: 256, paged: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)

        func run() async throws -> [UInt32] {
            var out: [UInt32] = []
            var token: Int32 = 3
            for position in 0..<130 {
                try await runner.produce(token: token, position: position, into: logits)
                out.append(runner.lastGreedyToken)
                token = Int32(runner.lastGreedyToken % UInt32(Self.vocab))
            }
            return out
        }
        let first = try await run()
        runner.reset()
        let second = try await run()
        #expect(first == second)
    }

    /// A genuinely sparse budget (fewer pages than exist) must stay stable
    /// and produce valid tokens — the quality gate for real sparsity runs on
    /// the real model, not the toy.
    @Test func sparseBudget_decodesWithoutFaulting() async throws {
        let (dir, ctx, runner) = try makeRunner(maxContext: 512, paged: true,
                                                topK: 1, sink: 1, recent: 2)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)

        var token: Int32 = 11
        for position in 0..<400 {     // 6+ pages; selection covers at most 4
            try await runner.produce(token: token, position: position, into: logits)
            #expect(runner.lastGreedyToken < UInt32(Self.vocab))
            token = Int32(runner.lastGreedyToken % UInt32(Self.vocab))
        }
        #expect(runner.continuationPosition == 400)
    }
}
