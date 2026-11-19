import Foundation
import Metal
import Testing
@testable import Mference

/// Real-model W4 runs on Quick through the reference runner and scheduler
/// with `PagedKVStore` swapped in. Each test is a model run (lock, preflight,
/// swap guard) and is skipped unless all of these are set:
///
/// - `MFERENCE_NATIVE_QUICK_PACK=<quick.mfpack>`
/// - `MFERENCE_NATIVE_GOLDENS=<goldens>/quick-v1/metal` (the `long-32k`
///   prompt supplies real text tokens)
/// - `MFERENCE_W4_RUN=footprint-8k` or `ttft-4k`
///
/// `MFERENCE_W4_OUT=<dir>` writes the results as JSON.
///
/// - `footprint-8k` (A5's measured run): an 8,000-token prompt at an
///   8,192-token plan, then follow-ups reusing it from RAM and from SSD;
///   reports the plan, the live store footprint and the process's peak
///   physical footprint, and checks the reused generations are exact.
/// - `ttft-4k` (a timing run): cold TTFT of a 4,096-token prompt, then warm
///   TTFT of a 64-token follow-up on that cached history (the
///   `docs/NATIVE_BENCHMARKS.md` protocol), after one untimed warm-up.
@Suite("MemoryQuickTests", .serialized)
struct MemoryQuickTests {
    static let env = ProcessInfo.processInfo.environment
    static let mib: UInt64 = 1 << 20

    static func enabled(_ run: String) -> Bool {
        env["MFERENCE_NATIVE_QUICK_PACK"] != nil && env[GoldenStore.environmentKey] != nil && env["MFERENCE_W4_RUN"] == run
    }

    static func log(_ s: String) {
        print("\(ISO8601DateFormatter().string(from: Date())) w4: \(s)")
        fflush(stdout)
    }

    /// Peak physical footprint of this process (`task_vm_info`).
    static func peakFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(info.ledger_phys_footprint_peak) : 0
    }

    struct Setup {
        let ctx: MetalContext
        let plan: Native.MemoryPlan
        let scheduler: any Native.StepScheduler
        let probe: ReferenceModelTests.StoreProbe
    }

    static func setup(context: Int, chunk: Int, prefixBudget: UInt64,
                      ssd: Native.PrefixSSDTierConfiguration?) throws -> Setup {
        let ctx = try MetalContext()
        let pack = URL(fileURLWithPath: try #require(env["MFERENCE_NATIVE_QUICK_PACK"]))
        let weights = try Native.ReferenceWeightStore(directory: pack, device: ctx.device)
        let c = weights.config
        let factory = Native.ReferenceForwardModelFactory(context: ctx)
        let plan = try Native.PagedMemoryPlanner().resolve(
            config: c, weightsMapped: weights.mappedByteCount, contextLength: context, precision: .f16,
            maxVerifyRows: 1, maxPrefillChunk: chunk,
            activationBytes: factory.activationBytes(config: c, maxPrefillChunk: chunk, maxVerifyRows: 1),
            prefixCacheRAMBudget: prefixBudget, budget: 16 << 30)
        let kv = try Native.PagedKVStore(config: c, plan: plan, device: ctx.device, ssdTier: ssd)
        let probe = ReferenceModelTests.StoreProbe()
        probe.kv = kv
        let components = Native.ModelComponents(
            device: ctx.device, weights: weights, gemm: Native.ReferenceQuantGEMM(context: ctx),
            attention: Native.ReferenceAttentionOp(context: ctx), gdn: Native.ReferenceGDNOp(context: ctx),
            glue: Native.ReferenceGlueOps(context: ctx), kv: kv)
        let model = try factory.make(components, plan: plan)
        let scheduler = try Native.ReferenceStepSchedulerFactory().make(
            model: model, sampler: try Native.ReferenceSampler(context: ctx),
            strategy: Native.ReferencePlainDecoding(), queue: ctx.queue)
        return Setup(ctx: ctx, plan: plan, scheduler: scheduler, probe: probe)
    }

    static func job(_ prompt: [Int32], reuse: Bool, n: Int, chunk: Int) -> Native.GenerationJob {
        Native.GenerationJob(promptTokens: prompt, params: ReferenceModelTests.greedy, maxNewTokens: n,
                             stopTokens: [], reusePrefix: reuse, prefillChunk: chunk)
    }

    /// Progress lines for a long GPU-bound run.
    static func progress(_ label: String) -> @Sendable (Native.StepEvent) -> Void {
        { event in
            if case .prefillProgress(let done, let total) = event, done % 1024 < 256 || done == total {
                log("\(label): prefill \(done)/\(total)")
            }
        }
    }

    struct Figures: Codable {
        var plan: [String: UInt64]
        var liveStore: [String: UInt64]
        var peakPhysicalFootprint: UInt64
        var deviceAllocatedSize: Int
        var runs: [Run]
        struct Run: Codable {
            var label: String
            var promptTokens: Int
            var reusedPrefixTokens: Int
            var tokens: [Int32]
            var prefillSeconds: Double
            var timeToFirstTokenSeconds: Double
            var decodeSeconds: Double
        }
    }

    static func footprintFields(_ f: Native.MemoryPlan.Footprint) -> [String: UInt64] {
        ["weightsMapped": f.weightsMapped, "kvArena": f.kvArena, "recurrentState": f.recurrentState,
         "verifyArena": f.verifyArena, "activations": f.activations, "prefixCacheRAM": f.prefixCacheRAM,
         "other": f.other, "total": f.total]
    }

    static func record(_ label: String, _ prompt: [Int32], _ o: Native.GenerationOutcome) -> Figures.Run {
        log("\(label): \(prompt.count) prompt tokens, reused \(o.reusedPrefixTokens), prefill \(o.timings.prefillSeconds) s, "
            + "TTFT \(o.timings.timeToFirstTokenSeconds) s, decode \(o.timings.decodeSeconds) s, tokens \(o.tokens)")
        return Figures.Run(label: label, promptTokens: prompt.count, reusedPrefixTokens: o.reusedPrefixTokens,
                           tokens: o.tokens, prefillSeconds: o.timings.prefillSeconds,
                           timeToFirstTokenSeconds: o.timings.timeToFirstTokenSeconds,
                           decodeSeconds: o.timings.decodeSeconds)
    }

    static func write(_ figures: Figures, _ name: String) throws {
        guard let out = env["MFERENCE_W4_OUT"] else { return }
        let url = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(figures).write(to: url.appendingPathComponent(name))
    }

    /// A5: the measured run at 8K, inside its plan, with exact reuse.
    @Test(.enabled(if: enabled("footprint-8k")))
    func footprint8K() async throws {
        let tokens = try GoldenStore.open("long-32k").promptTokens
        let prompt = Array(tokens.prefix(8000))
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mference-w4-quick-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let s = try Self.setup(context: 8192, chunk: 256, prefixBudget: 512 * Self.mib,
                               ssd: .init(directory: dir, budgetBytes: 2 << 30))
        Self.log("plan: \(Self.footprintFields(s.plan.planned)) budget \(s.plan.budget)")
        var runs: [Figures.Run] = []

        let a = try await s.scheduler.run(Self.job(prompt, reuse: true, n: 16, chunk: 256), onEvent: Self.progress("A"))
        runs.append(Self.record("A cold 8000", prompt, a))
        #expect(a.tokens.count == 16)

        // B: the prompt plus 8 of its own generated tokens, reusing the cached
        // prompt from RAM: greedy continues exactly as A did.
        let followUp = prompt + a.tokens.prefix(8)
        let b = try await s.scheduler.run(Self.job(followUp, reuse: true, n: 8, chunk: 256), onEvent: Self.progress("B"))
        runs.append(Self.record("B warm RAM", followUp, b))
        #expect(b.reusedPrefixTokens == prompt.count)
        #expect(b.tokens == Array(a.tokens[8..<16]))

        // C: the same follow-up with every cached prefix on the SSD tier.
        let kv = try #require(s.probe.kv as? Native.PagedKVStore)
        let spilled = kv.spillPrefixCacheToSSD()
        Self.log("spilled \(spilled) prefixes, \(kv.ssdBytes / Self.mib) MiB on SSD")
        let c = try await s.scheduler.run(Self.job(followUp, reuse: true, n: 8, chunk: 256), onEvent: Self.progress("C"))
        runs.append(Self.record("C warm SSD", followUp, c))
        #expect(c.reusedPrefixTokens == prompt.count && kv.counters.ssdAttaches == 1)
        #expect(c.tokens == b.tokens)

        let peak = Self.peakFootprint()
        let live = kv.footprint
        Self.log("live store \(Self.footprintFields(live)); peak physical footprint \(peak / Self.mib) MiB; "
                 + "device allocated \(s.ctx.device.currentAllocatedSize >> 20) MiB; plan total "
                 + "\(s.plan.planned.total / Self.mib) MiB")
        #expect(live.total <= s.plan.planned.total)
        #expect(peak <= s.plan.planned.total, "peak physical footprint over the plan")
        try Self.write(Figures(plan: Self.footprintFields(s.plan.planned), liveStore: Self.footprintFields(live),
                               peakPhysicalFootprint: peak, deviceAllocatedSize: s.ctx.device.currentAllocatedSize,
                               runs: runs), "footprint-8k.json")
    }

    /// Warm versus cold TTFT on a 4K history (timing run: AC, quiet lock).
    @Test(.enabled(if: enabled("ttft-4k")))
    func ttft4K() async throws {
        let tokens = try GoldenStore.open("long-32k").promptTokens
        let history = Array(tokens.prefix(4096))
        let followUp = history + tokens[4096 ..< 4160]
        let s = try Self.setup(context: 8192, chunk: 256, prefixBudget: 512 * Self.mib, ssd: nil)
        var runs: [Figures.Run] = []

        let warmUp = Array(tokens.prefix(64))
        runs.append(Self.record("warm-up (untimed)", warmUp,
                                try await s.scheduler.run(Self.job(warmUp, reuse: false, n: 1, chunk: 256)) { _ in }))
        let cold = try await s.scheduler.run(Self.job(history, reuse: true, n: 1, chunk: 256),
                                             onEvent: Self.progress("cold"))
        runs.append(Self.record("cold 4096", history, cold))
        var warm: [Double] = []
        for i in 0..<5 {
            let w = try await s.scheduler.run(Self.job(followUp, reuse: true, n: 1, chunk: 256)) { _ in }
            runs.append(Self.record("warm 4096+64 #\(i + 1)", followUp, w))
            #expect(w.reusedPrefixTokens == history.count)
            warm.append(w.timings.timeToFirstTokenSeconds)
        }
        let median = warm.sorted()[warm.count / 2]
        Self.log("TTFT cold \(cold.timings.timeToFirstTokenSeconds) s; warm median \(median) s (min \(warm.min()!), "
                 + "max \(warm.max()!)); warm/cold \(median / cold.timings.timeToFirstTokenSeconds)")
        let kv = try #require(s.probe.kv as? Native.PagedKVStore)
        try Self.write(Figures(plan: Self.footprintFields(s.plan.planned), liveStore: Self.footprintFields(kv.footprint),
                               peakPhysicalFootprint: Self.peakFootprint(),
                               deviceAllocatedSize: s.ctx.device.currentAllocatedSize, runs: runs), "ttft-4k.json")
    }
}
