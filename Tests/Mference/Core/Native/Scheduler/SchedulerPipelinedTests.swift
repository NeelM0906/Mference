import Foundation
import Metal
import Testing
@testable import Mference

/// The pipelined scheduler against the frozen `ReferenceStepScheduler` on the
/// all-reference tiny synthetic `qwen35` path (R's fixture): byte-identical
/// tokens, stop reasons, events and store state for every decoding path
/// (GPU choice, CPU sampling, verify rows), plus prefix reuse, cancellation,
/// failure handling and the pipelining itself. Model-free.
@Suite("SchedulerPipelinedTests", .serialized)
struct SchedulerPipelinedTests {
    typealias Fixture = ReferenceModelTests.Fixture

    static let greedy = ReferenceModelTests.greedy
    static let sampled = Native.SamplingParams(temperature: 0.9, topK: 40, topP: 0.95, minP: 0.02,
                                               repetitionPenalty: 1.1, presencePenalty: 0, frequencyPenalty: 0,
                                               repeatLastN: 64, seed: 1234)
    static let penalisedGreedy = Native.SamplingParams(temperature: 0, topK: nil, topP: nil, minP: 0,
                                                       repetitionPenalty: 1.3, presencePenalty: 0.2,
                                                       frequencyPenalty: 0.1, repeatLastN: 16, seed: nil)
    static let prompt: [Int32] = [262, 104, 101, 108, 108, 111, 263, 262, 30, 31, 40]

    /// Everything one run leaves behind.
    struct Run: Equatable {
        var tokens: [Int32]
        var reason: Native.StopReason
        var reused: Int
        var length: Int
        var progress: [Int]
        var events: [Int32]
    }

    /// Runs `jobs` in order on one scheduler built by `factory` over a fresh
    /// reference model, returning each run and the store length after it.
    static func runs(_ fx: Fixture, factory: any Native.StepSchedulerFactory, fastSampler: Bool,
                     strategy: @autoclosure () -> any Native.DecodingStrategy = Native.PlainDecoding(),
                     context: Int = 256, chunk: Int = 16, verify: Int = 4,
                     jobs: [Native.GenerationJob]) async throws -> [Run] {
        let probe = ReferenceModelTests.StoreProbe()
        let model = try fx.model(context: context, chunk: chunk, verify: verify)
        probe.kv = model.kv
        let sampler: any Native.Sampler = fastSampler
            ? try Native.ChainSampler(context: fx.ctx) : try Native.ReferenceSampler(context: fx.ctx)
        let scheduler = try factory.make(model: model, sampler: sampler, strategy: strategy(), queue: fx.ctx.queue)
        var out: [Run] = []
        for job in jobs {
            let log = ReferenceModelTests.EventLog()
            let o = try await scheduler.run(job) { log.add($0) }
            var progress: [Int] = []
            var events: [Int32] = []
            for e in log.all {
                switch e {
                case .prefillProgress(let done, _): progress.append(done)
                case .token(let t): events.append(t)
                }
            }
            out.append(Run(tokens: o.tokens, reason: o.stopReason, reused: o.reusedPrefixTokens,
                           length: probe.kv!.length, progress: progress, events: events))
        }
        return out
    }

    static func job(_ prompt: [Int32] = prompt, _ params: Native.SamplingParams = greedy, max: Int = 8,
                    stops: Set<Int32> = [], reuse: Bool = false, chunk: Int = 4) -> Native.GenerationJob {
        Native.GenerationJob(promptTokens: prompt, params: params, maxNewTokens: max, stopTokens: stops,
                             reusePrefix: reuse, prefillChunk: chunk)
    }

    /// The reference scheduler (with the reference sampler) and the pipelined
    /// one (with each sampler) give the same runs.
    static func expectSame(_ fx: Fixture, context: Int = 256, _ jobs: [Native.GenerationJob],
                           sourceLocation: SourceLocation = #_sourceLocation) async throws -> [Run] {
        let expected = try await runs(fx, factory: Native.ReferenceStepSchedulerFactory(), fastSampler: false,
                                      strategy: Native.ReferencePlainDecoding(), context: context, jobs: jobs)
        for fast in [false, true] {
            let got = try await runs(fx, factory: Native.PipelinedStepSchedulerFactory(), fastSampler: fast,
                                     context: context, jobs: jobs)
            #expect(got == expected, "fast sampler \(fast)", sourceLocation: sourceLocation)
        }
        for r in expected { #expect(r.tokens == r.events, sourceLocation: sourceLocation) }
        return expected
    }

    @Test func greedyMatchesTheReferenceScheduler() async throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let runs = try await Self.expectSame(fx, [
            Self.job(max: 12, chunk: 4),
            Self.job(max: 1, chunk: 3),
            Self.job(max: 0, chunk: 16),
            Self.job(max: 9, chunk: 1),
            Self.job([5], max: 5, chunk: 16),
        ])
        #expect(runs[0].tokens.count == 12 && runs[0].reason == .maxTokens)
        #expect(runs[2].tokens.isEmpty)
    }

    @Test func sampledAndPenalisedMatchTheReferenceScheduler() async throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        _ = try await Self.expectSame(fx, [
            Self.job(Self.prompt, Self.sampled, max: 10, chunk: 4),
            Self.job(Self.prompt, Self.penalisedGreedy, max: 10, chunk: 5),
            Self.job(Self.prompt, Self.sampled, max: 3, chunk: 16),
        ])
    }

    @Test func stopTokensMatchTheReferenceScheduler() async throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let plain = try await Self.runs(fx, factory: Native.ReferenceStepSchedulerFactory(), fastSampler: false,
                                        strategy: Native.ReferencePlainDecoding(), jobs: [Self.job(max: 10)])[0]
        var jobs: [Native.GenerationJob] = []
        for k in [0, 1, 4, 9] {
            jobs.append(Self.job(max: 10, stops: [plain.tokens[k]]))
            jobs.append(Self.job(Self.prompt, Self.penalisedGreedy, max: 10, stops: [plain.tokens[k]]))
        }
        let runs = try await Self.expectSame(fx, jobs)
        #expect(runs[0].reason == .stopToken && runs[0].tokens.isEmpty)
    }

    @Test func prefixReuseMatchesTheReferenceScheduler() async throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let followUp = Self.prompt + [44, 45, 46, 263]
        let runs = try await Self.expectSame(fx, [
            Self.job(max: 6, reuse: true),
            Self.job(followUp, max: 6, reuse: true),
            Self.job(followUp, Self.sampled, max: 6, reuse: true, chunk: 3),
            Self.job(followUp, max: 6, reuse: false),
        ])
        #expect(runs[1].reused == Self.prompt.count)
        #expect(runs[1].tokens == runs[3].tokens)
    }

    @Test func contextFullMatchesTheReferenceScheduler() async throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        // Capacity 16: an 11-token prompt has room for a few decode steps.
        let runs = try await Self.expectSame(fx, context: 16, [
            Self.job(max: 40),
            Self.job(Self.prompt, Self.sampled, max: 40),
            Self.job(Array(Self.prompt.prefix(15)), max: 40, chunk: 16),
        ])
        #expect(runs[0].reason == .contextFull)
    }

    // MARK: Verify rows

    /// Drafts the known continuation, with every third draft wrong.
    final class DraftStats: @unchecked Sendable {
        var proposed = 0
        var accepted = 0
    }

    final class ScriptedDrafts: Native.DecodingStrategy {
        let truth: [Int32]
        let promptCount: Int
        let rows: Int
        let stats: DraftStats
        init(truth: [Int32], promptCount: Int, rows: Int, stats: DraftStats = DraftStats()) {
            self.truth = truth
            self.promptCount = promptCount
            self.rows = rows
            self.stats = stats
        }
        let name = "scripted"
        var maxDraftRows: Int { rows }
        func reset(prompt: [Int32]) {}
        func draft(history: [Int32]) -> [Int32] {
            let k = history.count - promptCount
            var out: [Int32] = []
            for i in 0..<rows {
                let j = k + i
                let t = j < truth.count ? truth[j] : 7
                out.append((j % 3 == 2) ? (t &+ 1) % 256 : t)
            }
            return out
        }
        func observe(proposed: Int, accepted: Int) {
            stats.proposed += proposed
            stats.accepted += accepted
        }
    }

    @Test func verifyRowsMatchPlainDecoding() async throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        for params in [Self.greedy, Self.sampled, Self.penalisedGreedy] {
            let plain = try await Self.runs(fx, factory: Native.ReferenceStepSchedulerFactory(), fastSampler: false,
                                            strategy: Native.ReferencePlainDecoding(),
                                            jobs: [Self.job(Self.prompt, params, max: 14)])[0]
            // The continuation after a stop is never needed; drafts follow the
            // plain run (the first token comes from the prefill).
            let truth = Array(plain.tokens.dropFirst())
            for rows in [1, 3] {
                let stats = DraftStats()
                let got = try await Self.runs(
                    fx, factory: Native.PipelinedStepSchedulerFactory(), fastSampler: true,
                    strategy: ScriptedDrafts(truth: truth, promptCount: Self.prompt.count + 1, rows: rows, stats: stats),
                    jobs: [Self.job(Self.prompt, params, max: 14)])[0]
                #expect(got == plain, "rows \(rows) \(params)")
                #expect(stats.proposed > 0 && stats.accepted > 0 && stats.accepted < stats.proposed)
                // Stopping inside a verify step keeps exactly the rows plain decoding ran.
                for k in [2, 5] {
                    let refStop = try await Self.runs(
                        fx, factory: Native.ReferenceStepSchedulerFactory(), fastSampler: false,
                        strategy: Native.ReferencePlainDecoding(),
                        jobs: [Self.job(Self.prompt, params, max: 14, stops: [plain.tokens[k]])])[0]
                    let stop = try await Self.runs(fx, factory: Native.PipelinedStepSchedulerFactory(),
                                                   fastSampler: true,
                                                   strategy: ScriptedDrafts(truth: truth,
                                                                            promptCount: Self.prompt.count + 1,
                                                                            rows: rows),
                                                   jobs: [Self.job(Self.prompt, params, max: 14,
                                                                   stops: [plain.tokens[k]])])[0]
                    #expect(stop == refStop, "stop at \(k), rows \(rows)")
                }
            }
        }
    }

    // MARK: Cancellation and failures

    @Test func cancellationStopsAtAStepBoundary() async throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let probe = ReferenceModelTests.StoreProbe()
        let model = try fx.model(chunk: 4)
        probe.kv = model.kv
        let scheduler = try Native.PipelinedStepSchedulerFactory().make(
            model: model, sampler: try Native.ChainSampler(context: fx.ctx), strategy: Native.PlainDecoding(),
            queue: fx.ctx.queue) as! Native.PipelinedStepScheduler
        // Cancel after the third token: the result keeps three tokens and the
        // store holds exactly the steps that consumed the first two.
        let count = Counter()
        let outcome = try await scheduler.run(Self.job(max: 20)) { e in
            if case .token = e, count.increment() == 3 { scheduler.cancel() }
        }
        #expect(outcome.stopReason == .cancelled && outcome.tokens.count == 3)
        #expect(probe.kv?.length == Self.prompt.count + 2)

        // Cancelled during prefill (between chunks): no tokens.
        let early = try await scheduler.run(Self.job(max: 20, chunk: 2)) { e in
            if case .prefillProgress(let done, _) = e, done == 4 { scheduler.cancel() }
        }
        #expect(early.stopReason == .cancelled && early.tokens.isEmpty && probe.kv?.length == 4)

        // Cancelled before its first step: throws, and does not carry over.
        let gate = DispatchSemaphore(value: 0)
        scheduler.thread.submit { gate.wait() }
        let queued = Task { try await scheduler.run(Self.job(max: 2)) { _ in } }
        while scheduler.pendingCount == 0 { try await Task.sleep(nanoseconds: 1_000_000) }
        scheduler.cancel()
        gate.signal()
        do {
            _ = try await queued.value
            Issue.record("a job cancelled before its first step returned")
        } catch Native.NativeError.cancelled {}
        let next = try await scheduler.run(Self.job(max: 2)) { _ in }
        #expect(next.stopReason == .maxTokens && next.tokens.count == 2)
    }

    final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var n = 0
        func increment() -> Int { lock.withLock { n += 1; return n } }
    }

    /// A model whose encode throws on the decode step at `failAt` (0 = the
    /// first decode step).
    final class FailingDecode: Native.ForwardModel {
        let inner: Native.ReferenceForwardModel
        let failAt: Int
        var decodes = 0
        init(_ inner: Native.ReferenceForwardModel, failAt: Int) {
            self.inner = inner
            self.failAt = failAt
        }
        var config: Native.Qwen35Config { inner.config }
        var kv: any Native.KVStore { inner.kv }
        var maxPrefillChunk: Int { inner.maxPrefillChunk }
        var maxVerifyRows: Int { inner.maxVerifyRows }
        var tap: (any Native.TensorTap)? {
            get { inner.tap }
            set { inner.tap = newValue }
        }
        func encodeStep(_ plan: Native.StepPlan, encoder: MTLComputeCommandEncoder) throws -> Native.TensorRef? {
            if plan.kind == .decode {
                defer { decodes += 1 }
                if decodes >= failAt { throw Native.NativeError.invalidShape("injected") }
            }
            return try inner.encodeStep(plan, encoder: encoder)
        }
    }

    @Test func aFailedEncodeUndoesTheStepOnEveryPath() async throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        for (params, failAt) in [(Self.greedy, 0), (Self.greedy, 3), (Self.sampled, 0), (Self.sampled, 2)] {
            let probe = ReferenceModelTests.StoreProbe()
            let inner = try fx.model(chunk: 8)
            probe.kv = inner.kv
            let scheduler = try Native.PipelinedStepSchedulerFactory().make(
                model: FailingDecode(inner, failAt: failAt), sampler: try Native.ChainSampler(context: fx.ctx),
                strategy: Native.PlainDecoding(), queue: fx.ctx.queue)
            let log = ReferenceModelTests.EventLog()
            do {
                _ = try await scheduler.run(Self.job(Self.prompt, params, max: 10)) { log.add($0) }
                Issue.record("the injected encode failure did not surface")
            } catch Native.NativeError.invalidShape(let message) {
                #expect(message == "injected")
            }
            // The sequential loop emitted failAt + 1 tokens and ran failAt decode steps.
            let tokens = log.all.filter { if case .token = $0 { return true } else { return false } }
            #expect(tokens.count == failAt + 1, "\(params) at \(failAt)")
            #expect(probe.kv?.length == Self.prompt.count + failAt, "\(params) at \(failAt)")
        }
    }

    // MARK: Pipelining

    /// Counts command buffers created while another one is still running.
    final class OverlapQueue: NSObject, MTLCommandQueue, @unchecked Sendable {
        let base: MTLCommandQueue
        private let lock = NSLock()
        private var running = 0
        private(set) var created = 0
        private(set) var createdWhileRunning = 0
        init(_ base: MTLCommandQueue) { self.base = base }
        var label: String? {
            get { base.label }
            set { base.label = newValue }
        }
        var device: MTLDevice { base.device }
        private func track(_ cb: MTLCommandBuffer?) -> MTLCommandBuffer? {
            lock.withLock {
                created += 1
                if running > 0 { createdWhileRunning += 1 }
            }
            cb?.addScheduledHandler { [weak self] _ in self?.lock.withLock { self?.running += 1 } }
            cb?.addCompletedHandler { [weak self] _ in self?.lock.withLock { self?.running -= 1 } }
            return cb
        }
        func makeCommandBuffer() -> MTLCommandBuffer? { track(base.makeCommandBuffer()) }
        func makeCommandBuffer(descriptor: MTLCommandBufferDescriptor) -> MTLCommandBuffer? {
            track(base.makeCommandBuffer(descriptor: descriptor))
        }
        func makeCommandBufferWithUnretainedReferences() -> MTLCommandBuffer? {
            track(base.makeCommandBufferWithUnretainedReferences())
        }
        func insertDebugCaptureBoundary() {}
        func addResidencySet(_ residencySet: MTLResidencySet) { base.addResidencySet(residencySet) }
        func removeResidencySet(_ residencySet: MTLResidencySet) { base.removeResidencySet(residencySet) }
        func __addResidencySets(_ residencySets: UnsafePointer<any MTLResidencySet>, count: Int) {
            base.addResidencySets(Array(UnsafeBufferPointer(start: residencySets, count: count)))
        }
        func __removeResidencySets(_ residencySets: UnsafePointer<any MTLResidencySet>, count: Int) {
            base.removeResidencySets(Array(UnsafeBufferPointer(start: residencySets, count: count)))
        }
    }

    /// With GPU choice, each decode step is created while the previous one
    /// runs (a ~2 ms fixed-cost step), and the GPU-chosen tokens are the
    /// synthetic model's known choices.
    @Test func decodeStepsArePipelined() async throws {
        guard let device = MTLCreateSystemDefaultDevice(), let base = device.makeCommandQueue() else { return }
        let ctx = try MetalContext()
        let kernel = try FixedCostKernel(device: device)
        let iterations = try kernel.calibrate(queue: base, milliseconds: 2)
        let queue = OverlapQueue(base)
        let model = try SyntheticForwardModel(device: device, kernel: kernel, iterations: iterations)
        let scheduler = try Native.PipelinedStepSchedulerFactory().make(
            model: model, sampler: try Native.ChainSampler(context: ctx), strategy: Native.PlainDecoding(),
            queue: queue)
        let prompt: [Int32] = [1, 2, 3, 4, 5]
        let steps = 24
        let outcome = try await scheduler.run(Native.GenerationJob(
            promptTokens: prompt, params: Self.greedy, maxNewTokens: steps, stopTokens: [], reusePrefix: false,
            prefillChunk: 8)) { _ in }
        let expected = (0..<steps).map { SyntheticForwardModel.choice(after: prompt.count - 1 + $0) }
        #expect(outcome.tokens == expected)
        #expect(outcome.timings.decodeSteps == steps - 1)
        // setup + prefill + decode steps; all but the first decode step's
        // successor creation overlap a running buffer.
        #expect(queue.created == 2 + steps - 1)
        #expect(queue.createdWhileRunning >= steps - 3, "\(queue.createdWhileRunning) of \(queue.created)")
    }
}
