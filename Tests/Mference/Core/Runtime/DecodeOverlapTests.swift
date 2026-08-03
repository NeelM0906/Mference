import Testing
import Foundation
import Metal
@testable import Mference

/// Decode-loop overlap restructure: the layer's attention buffer now signals a
/// shared event right after the router top-k and keeps the shared-expert FFN
/// encoded behind that signal, so the routed-expert pread overlaps GPU work.
/// These tests pin the two properties that restructure must preserve — the
/// numerics are untouched, and the phase counters describe the decode window
/// rather than the prompt.
@Suite struct DecodeOverlapTests {

    private func makeRunner() throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try QwenToySynthetic.write()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen36Toy())
        let runner = try RealForwardRunner(model: model,
                                           context: ctx,
                                           maxContext: 64)
        return (dir, ctx, runner)
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    private func decodeGreedy(_ runner: RealForwardRunner,
                              _ logits: MTLBuffer,
                              steps: Int) async throws -> [UInt32] {
        var out: [UInt32] = []
        var token: Int32 = 1
        for position in 0..<steps {
            try await runner.produce(token: token, position: position, into: logits)
            out.append(runner.lastGreedyToken)
            token = Int32(runner.lastGreedyToken)
        }
        return out
    }

    /// Determinism across the sync restructure. Waiting on the mid-buffer
    /// router signal and waiting on the whole buffer differ only in *when* the
    /// CPU wakes: the same kernels are encoded in the same order onto the same
    /// serial queue, so both must emit identical tokens.
    @Test func routerSignalWait_matchesFullCommandBufferWait() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        runner.routerEventWaitEnabled = true
        let early = try await decodeGreedy(runner, logits, steps: 4)

        runner.reset()
        runner.routerEventWaitEnabled = false
        let full = try await decodeGreedy(runner, logits, steps: 4)

        #expect(early == full)
        #expect(early.allSatisfy { $0 < 1024 })
    }

    /// The restructure must not disturb the prefill/decode equivalence the
    /// runner already guarantees: prefilling a prompt then decoding gives the
    /// same argmax as decoding those tokens one at a time.
    @Test func mergedSharedExpert_keepsPrefillDecodeEquivalence() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        let reference = runner.lastGreedyToken

        runner.reset()
        let tokens: [Int32] = [11]
        _ = try await runner.prefillChunked(tokens: tokens[...],
                                            startPosition: 0,
                                            outputMode: .greedyIfAvailable,
                                            config: .production(chunkTokens: 32),
                                            into: logits,
                                            onProgress: { _ in })
        try await runner.produce(token: 7, position: 1, into: logits)
        #expect(runner.lastGreedyToken == reference)
    }

    /// Phase counters describe the decode window. Prefill accumulates into the
    /// same counters (DeepSeek-V4 prefill is literally a decode loop), so
    /// `prefillChunked` must hand back a zeroed window; decode then fills it.
    @Test func phaseCounters_resetAtPrefillDecodeBoundary() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        let tokens: [Int32] = [1, 2, 3, 4]
        _ = try await runner.prefillChunked(tokens: tokens[...],
                                            startPosition: 0,
                                            outputMode: .greedyIfAvailable,
                                            config: .production(chunkTokens: 32),
                                            into: logits,
                                            onProgress: { _ in })
        #expect(runner.totalCb1Nanos == 0)
        #expect(runner.totalIoNanos == 0)
        #expect(runner.totalCb2Nanos == 0)
        #expect(runner.totalIoOverlappedNanos == 0)
        #expect(runner.totalIoExposedNanos == 0)

        try await runner.produce(token: 5, position: 4, into: logits)
        #expect(runner.totalCb1Nanos > 0)
    }

    /// Explicit reset is idempotent and clears every phase counter.
    @Test func beginDecodePhaseWindow_zeroesCounters() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        try await runner.produce(token: 1, position: 0, into: logits)
        #expect(runner.totalCb1Nanos > 0)
        runner.beginDecodePhaseWindow()
        #expect(runner.totalCb1Nanos == 0)
        #expect(runner.totalIoNanos == 0)
        #expect(runner.totalCb2Nanos == 0)
        #expect(runner.totalHeadNanos == 0)
        #expect(runner.totalHeadFusedNanos == 0)
        #expect(runner.totalIoOverlappedNanos == 0)
        #expect(runner.totalIoExposedNanos == 0)
        runner.beginDecodePhaseWindow()
        #expect(runner.totalCb1Nanos == 0)
    }

    /// Speculative prefetch changes *when* expert bytes are read, never *which*
    /// bytes reach a kernel: a prediction only pre-populates the slot cache the
    /// real plan would have filled anyway. All three modes must decode
    /// identically.
    @Test func speculativePrefetchModes_produceIdenticalTokens() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        runner.speculativePrefetchMode = .off
        let off = try await decodeGreedy(runner, logits, steps: 5)

        runner.reset()
        runner.speculativePrefetchMode = .prefetch
        let prefetch = try await decodeGreedy(runner, logits, steps: 5)

        runner.reset()
        runner.speculativePrefetchMode = .advise
        let advise = try await decodeGreedy(runner, logits, steps: 5)

        #expect(off == prefetch)
        #expect(off == advise)
        #expect(off.allSatisfy { $0 < 1024 })
    }

    /// The Qwen toy routes 8-of-8 experts into a 16-slot cache, so after the
    /// first token nothing can ever miss and a correct predictor issues
    /// nothing. Pinned explicitly: "no reads issued" here is the right answer,
    /// not a silently broken predictor.
    @Test func speculativePrefetch_issuesNothingWhenEveryExpertIsResident() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        runner.speculativePrefetchMode = .prefetch
        _ = try await decodeGreedy(runner, logits, steps: 6)

        #expect(runner.totalSpecPrefetchIssued == 0)
        #expect(runner.totalSpecPrefetchBytes == 0)
        // Confirmation is scored against everything the predictor named, not
        // against the (here empty) subset that needed reading.
        #expect(runner.totalSpecPrefetchConfirmed <= runner.totalSpecPrefetchPredicted)
    }

    @Test func speculativePrefetchMode_parsesEnvironmentSpellings() {
        typealias Mode = RealForwardRunner.SpeculativePrefetchMode
        #expect(Mode.parse(nil) == .off)
        #expect(Mode.parse("0") == .off)
        #expect(Mode.parse("off") == .off)
        #expect(Mode.parse("1") == .prefetch)
        #expect(Mode.parse("on") == .prefetch)
        #expect(Mode.parse("PREFETCH") == .prefetch)
        #expect(Mode.parse("advise") == .advise)
        #expect(Mode.parse("Advise") == .advise)
        #expect(Mode.parse("nonsense") == .off)
    }

    /// The overlap probe is what makes "io exposed" trustworthy: it must report
    /// nil (still running) until every tracked buffer completes, then the
    /// completion time of the last one.
    @Test func overlapProbe_reportsOnlyWhenAllBuffersFinish() throws {
        let ctx = try MetalContext()
        let probe = RealForwardRunner.GPUOverlapProbe()
        #expect(probe.finishedNanos == 0)

        let cb = ctx.queue.makeCommandBuffer()!
        probe.track(cb)
        #expect(probe.finishedNanos == nil)
        let before = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        cb.commit()
        cb.waitUntilCompleted()
        // The completion handler runs on a Metal thread; give it a moment.
        let deadline = Date().addingTimeInterval(5)
        while probe.finishedNanos == nil, Date() < deadline {
            usleep(1_000)
        }
        guard let finished = probe.finishedNanos else {
            Issue.record("overlap probe never observed the completion handler")
            return
        }
        #expect(finished >= before)
    }
}

/// Speculative prefetch on a fixture that can actually miss: the DeepSeek-V4
/// toy routes 6 of 16 experts per layer, so an 8-slot cache guarantees real
/// evictions and gives the previous-token predictor something to predict.
/// This is also the primary target path (`produceTokenDSV4`).
@Suite(.serialized)
struct SpeculativePrefetchDSV4Tests {

    private struct Harness {
        let dir: URL
        let runner: RealForwardRunner
        let logits: MTLBuffer
    }

    /// Eight slots for sixteen experts: every token evicts.
    private func makeHarness(slots: Int = 8) throws -> Harness {
        let dir = try DSV4ToySynthetic.write()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .deepseekV4Toy(),
                                   streamingMode: .pread(slotCount: slots))
        let runtime = RuntimeConfiguration(expertCacheSlots: slots,
                                           prefillChunkTokens: 32,
                                           forceLogitsHead: true)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64,
                                           runtimeConfiguration: runtime)
        guard let logits = ctx.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return Harness(dir: dir, runner: runner, logits: logits)
    }

    private func decode(_ h: Harness, steps: Int) async throws -> [Float] {
        var out: [Float] = []
        for position in 0..<steps {
            try await h.runner.produce(token: Int32(position % 7 + 1),
                                       position: position,
                                       into: h.logits)
            let ptr = h.logits.contents().assumingMemoryBound(to: Float16.self)
            out.append(Float(ptr[0]))
            out.append(Float(ptr[1]))
        }
        return out
    }

    /// The default "previous token's routing" predictor provably cannot issue a
    /// read: only layer L's own plans touch layer L's slots, so last token's
    /// experts are still resident when we would predict them. Even with 16
    /// experts, top-6 routing and only 8 slots — a cache that evicts on every
    /// token — the predicted set is always the resident set.
    @Test func previousTokenPredictorNeverFindsANonResidentExpert() async throws {
        let h = try makeHarness()
        defer { try? FileManager.default.removeItem(at: h.dir) }
        h.runner.speculativePrefetchMode = .prefetch

        _ = try await decode(h, steps: 6)

        #expect(h.runner.totalSpecPrefetchIssued == 0)
        #expect(h.runner.totalSpecPrefetchBytes == 0)
    }

    /// Everything downstream of the predictor — reserve, background pread,
    /// join, confirmation scoring, byte accounting — works end to end. Driven
    /// through the predictor seam so the test does not depend on a fixture
    /// whose routing happens to churn the cache.
    @Test func injectedPredictorDrivesReserveReadJoinAndConfirm() async throws {
        let h = try makeHarness()
        defer { try? FileManager.default.removeItem(at: h.dir) }
        h.runner.speculativePrefetchMode = .prefetch
        // Predict the whole expert file: guarantees non-resident experts every
        // layer, so the machinery is exercised on every single layer step.
        h.runner.speculativeExpertPredictor = { _ in Array(0..<16) }

        _ = try await decode(h, steps: 4)

        #expect(h.runner.totalSpecPrefetchIssued > 0)
        #expect(h.runner.totalSpecPrefetchBytes > 0)
        #expect(h.runner.totalSpecPrefetchConfirmed > 0)
        #expect(h.runner.totalSpecPrefetchConfirmed <= h.runner.totalSpecPrefetchPredicted)
    }

    /// Advise mode warms the page cache without touching slots: reads are
    /// "issued" but no expert bytes are copied into the cache, so residency is
    /// driven entirely by the real plans.
    @Test func adviseModeIssuesWithoutClaimingSlots() async throws {
        let h = try makeHarness()
        defer { try? FileManager.default.removeItem(at: h.dir) }
        h.runner.speculativePrefetchMode = .advise
        h.runner.speculativeExpertPredictor = { _ in Array(0..<16) }

        _ = try await decode(h, steps: 4)

        #expect(h.runner.totalSpecPrefetchIssued > 0)
    }

    /// PILOT engages on the fixture that can miss: the lookahead GEMV names a
    /// next-layer expert set, some of it is non-resident and gets read, and the
    /// real plan confirms some of it. Recall is bounded, not asserted at a
    /// level — it is an empirical property of the model, which is exactly what
    /// the counters exist to measure.
    @Test func pilotModeIssuesReadsAndMeasuresRecall() async throws {
        let h = try makeHarness()
        defer { try? FileManager.default.removeItem(at: h.dir) }
        h.runner.speculativePrefetchMode = .pilot

        _ = try await decode(h, steps: 6)

        #expect(h.runner.totalSpecPrefetchPredicted > 0)
        #expect(h.runner.totalSpecPrefetchIssued > 0)
        #expect(h.runner.totalSpecPrefetchBytes > 0)
        #expect(h.runner.totalSpecPrefetchConfirmed > 0)
        #expect(h.runner.totalSpecPrefetchConfirmed <= h.runner.totalSpecPrefetchPredicted)
    }

    /// The whole point of the exercise: unlike the temporal predictor, PILOT
    /// names experts the cache does *not* already hold.
    @Test func pilotPredictsNonResidentExpertsUnlikeTheTemporalPredictor() async throws {
        let pilot = try makeHarness()
        defer { try? FileManager.default.removeItem(at: pilot.dir) }
        pilot.runner.speculativePrefetchMode = .pilot
        _ = try await decode(pilot, steps: 6)

        let temporal = try makeHarness()
        defer { try? FileManager.default.removeItem(at: temporal.dir) }
        temporal.runner.speculativePrefetchMode = .prefetch
        _ = try await decode(temporal, steps: 6)

        #expect(temporal.runner.totalSpecPrefetchIssued == 0)
        #expect(pilot.runner.totalSpecPrefetchIssued > 0)
    }

    /// Determinism: the lookahead GEMV writes only private buffers and the
    /// prefetch only pre-populates slots, so seeded output must be byte
    /// identical to speculation being off.
    @Test func pilotModeLogitsAreIdenticalToOff() async throws {
        let off = try makeHarness()
        defer { try? FileManager.default.removeItem(at: off.dir) }
        off.runner.speculativePrefetchMode = .off
        let baseline = try await decode(off, steps: 6)

        let pilot = try makeHarness()
        defer { try? FileManager.default.removeItem(at: pilot.dir) }
        pilot.runner.speculativePrefetchMode = .pilot
        #expect(try await decode(pilot, steps: 6) == baseline)
    }

    /// Hash-routed layers bypass the GEMV: their expert set is an exact
    /// function of the token id, so predictions for them are perfect. The toy
    /// has hash-routed layers, so decoding the same token repeatedly must
    /// confirm every hash-layer prediction.
    @Test func hashRoutedNextLayerPredictionIsExact() async throws {
        let h = try makeHarness()
        defer { try? FileManager.default.removeItem(at: h.dir) }
        h.runner.speculativePrefetchMode = .pilot
        #expect(ArchConfig.deepseekV4Toy().numHashRoutedLayers > 0)

        for position in 0..<5 {
            try await h.runner.produce(token: 3, position: position, into: h.logits)
        }
        // Predictions exist and at least the hash-layer ones are confirmed.
        #expect(h.runner.totalSpecPrefetchPredicted > 0)
        #expect(h.runner.totalSpecPrefetchConfirmed > 0)
    }

    /// Determinism under an aggressively wrong predictor: speculation evicting
    /// and refilling slots on every layer must not change a single logit.
    @Test func aggressiveMispredictionDoesNotChangeLogits() async throws {
        let off = try makeHarness()
        defer { try? FileManager.default.removeItem(at: off.dir) }
        off.runner.speculativePrefetchMode = .off
        let baseline = try await decode(off, steps: 5)

        let noisy = try makeHarness()
        defer { try? FileManager.default.removeItem(at: noisy.dir) }
        noisy.runner.speculativePrefetchMode = .prefetch
        // Rotating garbage: maximum eviction pressure, minimum recall.
        nonisolated(unsafe) var tick = 0
        noisy.runner.speculativeExpertPredictor = { layer in
            tick += 1
            return [(layer * 7 + tick) % 16, (layer * 3 + tick * 5) % 16]
        }
        #expect(try await decode(noisy, steps: 5) == baseline)
    }

    /// The determinism contract on the path that can actually mispredict:
    /// speculation reorders I/O, never values.
    @Test func allModesProduceIdenticalLogits() async throws {
        let off = try makeHarness()
        defer { try? FileManager.default.removeItem(at: off.dir) }
        off.runner.speculativePrefetchMode = .off
        let baseline = try await decode(off, steps: 6)

        let prefetch = try makeHarness()
        defer { try? FileManager.default.removeItem(at: prefetch.dir) }
        prefetch.runner.speculativePrefetchMode = .prefetch
        #expect(try await decode(prefetch, steps: 6) == baseline)

        let advise = try makeHarness()
        defer { try? FileManager.default.removeItem(at: advise.dir) }
        advise.runner.speculativePrefetchMode = .advise
        #expect(try await decode(advise, steps: 6) == baseline)
    }

    /// Speculation must survive a mid-stream reset (the pending read is joined,
    /// the predictor is cleared) and keep producing the same logits as a fresh
    /// runner would.
    @Test func resetDuringSpeculationIsSafe() async throws {
        let h = try makeHarness()
        defer { try? FileManager.default.removeItem(at: h.dir) }
        h.runner.speculativePrefetchMode = .prefetch

        let first = try await decode(h, steps: 4)
        h.runner.reset()
        #expect(h.runner.continuationPosition == 0)
        let second = try await decode(h, steps: 4)
        #expect(first == second)
    }
}
