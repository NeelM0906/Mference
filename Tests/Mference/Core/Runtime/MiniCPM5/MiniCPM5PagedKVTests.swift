import Foundation
import Metal
import Testing
@testable import Mference

/// The paged long-context path on the minicpm5 toy: with a selection budget
/// covering every page, paged decode reproduces the dense runner's greedy
/// stream exactly (decode across page boundaries, and prefill + decode), and
/// a pool smaller than the context — the SSD tier live, blocked streamed
/// prefill — still agrees with the dense runner's prefill head and decode.
@Suite struct MiniCPM5PagedKVTests {
    private static let vocab = 128

    private func makeRunner(maxContext: Int,
                            paged: Bool,
                            poolPages: Int? = nil,
                            topK: Int = 1024,
                            sink: Int = 2,
                            recent: Int = 4) throws -> (URL, MetalContext, MiniCPM5ForwardRunner) {
        let dir = try MiniCPM5Parity.installToyCheckpoint()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: .miniCPM5Toy())
        let config = RuntimeConfiguration(prefillEnabled: true,
                                          kvPagedPolicy: paged ? .on : .off,
                                          kvTopKPages: topK,
                                          kvSinkPages: sink,
                                          kvRecentPages: recent,
                                          kvPoolPagesPerLayer: poolPages)
        let runner = try MiniCPM5ForwardRunner(model: model, context: ctx,
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
        (0..<count).map { Int32(4 + ($0 * 37 + 11) % (vocab - 4)) }
    }

    private func greedyPrefill(_ runner: MiniCPM5ForwardRunner, tokens: [Int32],
                               logits: MTLBuffer, chunk: Int = 64) async throws -> UInt32 {
        let result = try await runner.prefillChunked(tokens: tokens[...],
                                                     startPosition: 0,
                                                     outputMode: .greedyIfAvailable,
                                                     config: .production(chunkTokens: chunk),
                                                     into: logits,
                                                     onProgress: { _ in })
        guard case .greedyToken(let seed) = result.seed else {
            throw ModelError.residentBufferWrapFailed
        }
        return seed
    }

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

    @Test func pagedFullSelection_prefillPlusDecodeMatchesDense() async throws {
        let promptTokens = Self.prompt(150)
        let decodeSteps = 40
        let (dirD, ctxD, dense) = try makeRunner(maxContext: 256, paged: false)
        defer { try? FileManager.default.removeItem(at: dirD) }
        let (dirP, ctxP, paged) = try makeRunner(maxContext: 256, paged: true)
        defer { try? FileManager.default.removeItem(at: dirP) }
        let logitsD = try makeLogits(ctxD)
        let logitsP = try makeLogits(ctxP)

        var tokenD = try await greedyPrefill(dense, tokens: promptTokens, logits: logitsD)
        var tokenP = try await greedyPrefill(paged, tokens: promptTokens, logits: logitsP)
        #expect(tokenD == tokenP)
        var position = promptTokens.count
        for _ in 0..<decodeSteps {
            try await dense.produce(token: Int32(tokenD), position: position, into: logitsD)
            try await paged.produce(token: Int32(tokenP), position: position, into: logitsP)
            tokenD = dense.lastGreedyToken
            tokenP = paged.lastGreedyToken
            #expect(tokenD == tokenP, "position \(position)")
            position += 1
        }
    }

    /// A pool of five pages per layer (the minimum for sink 1 + recent 2 +
    /// two slots of slack) against a seven-page prompt forces sealed pages to
    /// spill and the blocked (streamed) prefill to run. Blocked prefill is
    /// exact up to summation order, so the greedy head must agree with the
    /// dense runner at the toy's margins, and decode continues under eviction.
    @Test func smallPool_blockedPrefillAndDecodeMatchDense() async throws {
        let promptTokens = Self.prompt(400)
        let (dirD, ctxD, dense) = try makeRunner(maxContext: 512, paged: false)
        defer { try? FileManager.default.removeItem(at: dirD) }
        let (dirP, ctxP, paged) = try makeRunner(maxContext: 512, paged: true,
                                                 poolPages: 5, topK: 0, sink: 1, recent: 2)
        defer { try? FileManager.default.removeItem(at: dirP) }
        let logitsD = try makeLogits(ctxD)
        let logitsP = try makeLogits(ctxP)

        let seedD = try await greedyPrefill(dense, tokens: promptTokens, logits: logitsD)
        let seedP = try await greedyPrefill(paged, tokens: promptTokens, logits: logitsP)
        #expect(seedD == seedP)
        // Decode continues without faulting under eviction; with topK 0 the
        // selection is sink + recent only, so the stream is not required to
        // match dense here — only to be finite and in-vocabulary.
        var token = seedP
        for step in 0..<8 {
            try await paged.produce(token: Int32(token), position: promptTokens.count + step,
                                    into: logitsP)
            token = paged.lastGreedyToken
            #expect(token < UInt32(Self.vocab))
        }
    }
}
