import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// MTP speculative decoding composed with the paged KV mode: with a
/// covering selection budget, spec rounds run per-position paged attention
/// over the same tables as plain paged decode, so the greedy stream must be
/// byte-identical to plain paged decode — through page-boundary crossings,
/// rollbacks (cursor rewind un-seals pages), and reset. A tight pool run
/// exercises rounds whose selections fetch from the spill file.
@Suite(.serialized) struct Qwen38PagedMTPTests {
    private static let vocab = 1024

    private static func makeAttachedDirectory() throws -> URL {
        let dir = try Qwen38ToySynthetic.write()
        let shard = try Qwen38ToySynthetic.writeMTPShard()
        defer { try? FileManager.default.removeItem(at: shard) }
        _ = try MTPAttachTool.run(gturboDirectory: dir.path, shardPath: shard.path)
        return dir
    }

    private func makeRunner(_ dir: URL,
                            maxContext: Int,
                            paged: Bool,
                            poolPages: Int? = nil,
                            topK: Int = 1024,
                            sink: Int = 2,
                            recent: Int = 4) throws -> (MetalContext, Qwen38ForwardRunner) {
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
        return (ctx, runner)
    }

    private func makeLogits(_ ctx: MetalContext) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: Self.vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    /// Greedy self-drive for `steps` tokens from a fixed seed.
    private func drive(_ runner: Qwen38ForwardRunner,
                       _ logits: MTLBuffer,
                       steps: Int,
                       seed: Int32 = 7) async throws -> [UInt32] {
        var out: [UInt32] = []
        var token = seed
        for position in 0..<steps {
            try await runner.produce(token: token, position: position, into: logits)
            out.append(runner.lastGreedyToken)
            token = Int32(runner.lastGreedyToken % UInt32(Self.vocab))
        }
        return out
    }

    /// Spec rounds in paged mode must reproduce plain paged decode exactly —
    /// the MTP byte-identity contract, now over paged attention. 200 steps
    /// cross three page boundaries, so rounds straddle seals and rewinds.
    @Test func pagedSpecDecode_byteIdenticalToPagedPlain() async throws {
        let dir = try Self.makeAttachedDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let (ctxSpec, spec) = try makeRunner(dir, maxContext: 256, paged: true)
        let logitsSpec = try makeLogits(ctxSpec)
        let specStream = try await drive(spec, logitsSpec, steps: 200)

        // Disable MTP per-instance — setenv would race runners constructed
        // concurrently by other suites.
        let (ctxPlain, plain) = try makeRunner(dir, maxContext: 256, paged: true)
        plain.mtp = nil
        let logitsPlain = try makeLogits(ctxPlain)
        let plainStream = try await drive(plain, logitsPlain, steps: 200)

        #expect(specStream == plainStream)
    }

    /// Rewinds across a page boundary: rejected drafts whose span sealed a
    /// page must un-seal it cleanly. Reset + replay must reproduce the run.
    @Test func pagedSpecDecode_resetReplaysIdentically() async throws {
        let dir = try Self.makeAttachedDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (ctx, runner) = try makeRunner(dir, maxContext: 256, paged: true)
        let logits = try makeLogits(ctx)

        let first = try await drive(runner, logits, steps: 150)
        runner.reset()
        let second = try await drive(runner, logits, steps: 150)
        #expect(first == second)
    }

    /// A selection budget the context outgrows mid-run: spec rounds run
    /// while the selection is exhaustive, then the gate hands decode off to
    /// plain paged tokens — the stream must stay byte-identical to MTP-off
    /// paged decode across the crossover into sparse selection.
    @Test func pagedSpecDecode_sparseBudget_byteIdenticalToPagedPlain() async throws {
        let dir = try Self.makeAttachedDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Coverage ends at 5 pages (sink 1 + recent 2 + topk 2); 400 steps
        // put the crossover near position 320.
        let (ctxSpec, spec) = try makeRunner(dir, maxContext: 512, paged: true,
                                             topK: 2, sink: 1, recent: 2)
        let logitsSpec = try makeLogits(ctxSpec)
        let specStream = try await drive(spec, logitsSpec, steps: 400)

        let (ctxPlain, plain) = try makeRunner(dir, maxContext: 512, paged: true,
                                               topK: 2, sink: 1, recent: 2)
        plain.mtp = nil
        let logitsPlain = try makeLogits(ctxPlain)
        let plainStream = try await drive(plain, logitsPlain, steps: 400)

        #expect(specStream == plainStream)
    }

    /// The exactness gate: speculative rounds run while the selection is
    /// exhaustive (5 pages here) and stop for good once the context
    /// outgrows the budget — the sparse tail decodes as plain paged tokens.
    @Test func pagedSpecDecode_gateStopsRoundsPastCoverage() async throws {
        let dir = try Self.makeAttachedDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (ctx, runner) = try makeRunner(dir, maxContext: 512, paged: true,
                                           topK: 2, sink: 1, recent: 2)
        let mtp = try #require(runner.mtp)
        let logits = try makeLogits(ctx)

        var token: Int32 = 7
        var roundsAtCrossover = -1
        for position in 0..<400 {
            try await runner.produce(token: token, position: position, into: logits)
            if position == 330 { roundsAtCrossover = mtp.stats.rounds }
            token = Int32(runner.lastGreedyToken % UInt32(Self.vocab))
        }
        // Rounds ran in the covered prefix and none started in the sparse
        // tail (coverage ends at position 320; the gate's horizon stops
        // rounds a few positions earlier).
        #expect(roundsAtCrossover > 0)
        #expect(mtp.stats.rounds == roundsAtCrossover)
    }

    /// Tight pool: while the selection is exhaustive, rounds run under pool
    /// pressure; past the coverage gate, plain paged decode fetches sparse
    /// selections from the spill file. Both phases must replay
    /// deterministically.
    @Test func pagedSpecDecode_tightPool_replaysDeterministically() async throws {
        let dir = try Self.makeAttachedDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (ctx, runner) = try makeRunner(dir, maxContext: 512, paged: true,
                                           poolPages: 6, topK: 1,
                                           sink: 1, recent: 2)
        let logits = try makeLogits(ctx)

        let first = try await drive(runner, logits, steps: 300)
        runner.reset()
        let second = try await drive(runner, logits, steps: 300)
        #expect(first == second)
        #expect(first.allSatisfy { $0 < UInt32(Self.vocab) })
    }
}
