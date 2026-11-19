import Foundation
import Metal
import Testing
@testable import Mference

/// `PagedMemoryPlanner` arithmetic, its refusals, the store allocating
/// exactly the plan, and the 32K Quick plan checked by arithmetic only
/// (Phase 1 amendment A5: no run above 8K tokens).
@Suite("MemoryPlannerTests", .serialized)
struct MemoryPlannerTests {
    static let quick = ReferenceQuickPackTests.quickConfig
    /// `weights.bin` of the Phase 0 reference Quick pack (`quick-ref.mfpack`).
    static let quickWeightsBytes: UInt64 = 2_901_600_256
    static let mib: UInt64 = 1 << 20

    @Test func tinyPlanArithmeticAndRefusals() throws {
        let c = RefTest.tiny
        let planner = Native.PagedMemoryPlanner()
        let g = Native.PagedMemoryPlanner.geometry(config: c, precision: .f16)
        let l = Native.PagedMemoryPlanner.recurrentLayout(config: c)
        let plan = try planner.resolve(config: c, weightsMapped: 1000, contextLength: 130, precision: .f16,
                                       maxVerifyRows: 3, maxPrefillChunk: 16, activationBytes: 77,
                                       prefixCacheRAMBudget: UInt64(5 * g.pageBytes + 7), budget: 1 << 30)
        #expect(plan.planned.kvArena == UInt64(3 * g.pageBytes))
        #expect(plan.planned.prefixCacheRAM == UInt64(5 * g.pageBytes), "whole pages only")
        #expect(plan.planned.recurrentState == UInt64(2 * 2 * l.layerBytes))
        #expect(plan.planned.verifyArena == UInt64(2 * 3 * l.layerBytes))
        #expect(plan.planned.other == UInt64(2 * 3 * 4))
        #expect(plan.planned.weightsMapped == 1000 && plan.planned.activations == 77)
        // The same geometry, recurrent and verify accounting as the reference planner.
        let ref = try Native.ReferenceMemoryPlanner().resolve(
            config: c, weightsMapped: 1000, contextLength: 130, precision: .int8, maxVerifyRows: 3,
            maxPrefillChunk: 16, activationBytes: 77, prefixCacheRAMBudget: 0, budget: 1 << 30)
        let int8 = try planner.resolve(config: c, weightsMapped: 1000, contextLength: 130, precision: .int8,
                                       maxVerifyRows: 3, maxPrefillChunk: 16, activationBytes: 77,
                                       prefixCacheRAMBudget: 0, budget: 1 << 30)
        #expect(int8.planned.kvArena == ref.planned.kvArena)
        #expect(int8.planned.recurrentState == ref.planned.recurrentState)
        #expect(int8.planned.verifyArena == ref.planned.verifyArena)
        #expect(Native.PagedMemoryPlanner.geometry(config: c, precision: .int8)
                    == Native.ReferenceMemoryPlanner.geometry(config: c, precision: .int8))
        // Over budget by one byte refuses with the required total.
        do {
            _ = try planner.resolve(config: c, weightsMapped: 1000, contextLength: 130, precision: .f16,
                                    maxVerifyRows: 3, maxPrefillChunk: 16, activationBytes: 77,
                                    prefixCacheRAMBudget: UInt64(5 * g.pageBytes), budget: plan.planned.total - 1)
            Issue.record("a plan over budget resolved")
        } catch Native.NativeError.memoryBudgetExceeded(let required, let budget) {
            #expect(required == plan.planned.total && budget == plan.planned.total - 1)
        }
        for (context, verify, chunk) in [(0, 1, 1), (64, 0, 1), (64, 17, 1), (64, 1, 0), (c.contextLength + 1, 1, 1)] {
            #expect(throws: Native.NativeError.self) {
                _ = try planner.resolve(config: c, weightsMapped: 0, contextLength: context, precision: .f16,
                                        maxVerifyRows: verify, maxPrefillChunk: chunk, activationBytes: 0,
                                        prefixCacheRAMBudget: 0, budget: 1 << 40)
            }
        }
    }

    @Test func largestContextIsTheLastPageThatFits() throws {
        let c = RefTest.tiny
        let planner = Native.PagedMemoryPlanner()
        let budget: UInt64 = 1 << 20
        let best = try #require(planner.largestContext(config: c, weightsMapped: 0, precision: .f16, maxVerifyRows: 1,
                                                       activationBytes: 0, prefixCacheRAMBudget: 0, budget: budget))
        #expect(best % 64 == 0 && best > 0)
        _ = try planner.resolve(config: c, weightsMapped: 0, contextLength: best, precision: .f16, maxVerifyRows: 1,
                                maxPrefillChunk: 1, activationBytes: 0, prefixCacheRAMBudget: 0, budget: budget)
        if best < c.contextLength {
            #expect(throws: Native.NativeError.self) {
                _ = try planner.resolve(config: c, weightsMapped: 0, contextLength: best + 64, precision: .f16,
                                        maxVerifyRows: 1, maxPrefillChunk: 1, activationBytes: 0,
                                        prefixCacheRAMBudget: 0, budget: budget)
            }
        }
        #expect(planner.largestContext(config: c, weightsMapped: budget, precision: .f16, maxVerifyRows: 1,
                                       activationBytes: 0, prefixCacheRAMBudget: 0, budget: budget) == nil)
    }

    /// The store allocates exactly what the plan says, and its live
    /// footprint never exceeds the plan.
    @Test func storeAllocatesExactlyThePlan() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let c = RefTest.tiny
        for precision in [Native.KVPrecision.f16, .int8] {
            let plan = try Native.PagedMemoryPlanner().resolve(
                config: c, weightsMapped: 0, contextLength: 200, precision: precision, maxVerifyRows: 4,
                maxPrefillChunk: 64, activationBytes: 0, prefixCacheRAMBudget: 1 << 18, budget: 1 << 30)
            let kv = try Native.PagedKVStore(config: c, plan: plan, device: device)
            #expect(UInt64(kv.arenaBytes) == plan.planned.kvArena + plan.planned.prefixCacheRAM)
            let live = kv.footprint
            #expect(live.kvArena == plan.planned.kvArena)
            #expect(live.recurrentState == plan.planned.recurrentState)
            #expect(live.verifyArena == plan.planned.verifyArena)
            #expect(live.other == plan.planned.other)
            #expect(live.prefixCacheRAM == 0)
            #expect(live.total <= plan.planned.total)
            #expect(kv.contextPages * kv.geometry.tokensPerPage >= kv.capacity)
        }
    }

    struct QuickPlanFigures: Codable {
        var precision: String
        var contextLength: Int
        var pageBytes: Int
        var kvArena: UInt64
        var recurrentState: UInt64
        var verifyArena: UInt64
        var activations: UInt64
        var prefixCacheRAM: UInt64
        var weightsMapped: UInt64
        var other: UInt64
        var total: UInt64
        var budget: UInt64
    }

    /// A5: the 32K Quick plan, by arithmetic only (no model, no run).
    /// Quick: 8 full-attention layers x 2 (K, V) x 4 KV heads x 64 tokens x
    /// 256 dims x 2 B = 2 MiB per FP16 page, 32 KiB per token; 24 GDN layers
    /// of `[3][8192]` F32 conv + `[32][128][128]` F32 S = 2,195,456 B each.
    @Test func quick32KPlanByArithmetic() throws {
        let q = Self.quick
        let ctx = try MetalContext()
        let factory = Native.ReferenceForwardModelFactory(context: ctx)
        let chunk = 512
        let activations = factory.activationBytes(config: q, maxPrefillChunk: chunk, maxVerifyRows: 1)
        let prefixBudget: UInt64 = 512 * Self.mib
        let budget: UInt64 = 16 << 30
        let planner = Native.PagedMemoryPlanner()
        var figures: [QuickPlanFigures] = []

        let g16 = Native.PagedMemoryPlanner.geometry(config: q, precision: .f16)
        let layout = Native.PagedMemoryPlanner.recurrentLayout(config: q)
        #expect(g16.pageBytes == 2 * 1024 * 1024)
        #expect(layout.convBytes == 98_304 && layout.ssmOffset == 98_304 && layout.layerBytes == 2_195_456)
        #expect(Native.PagedMemoryPlanner.snapshotBytes(config: q) == 24 * 2_195_456)
        #expect(Native.PagedMemoryPlanner.snapshotPages(config: q, geometry: g16) == 26)

        let plan32 = try planner.resolve(config: q, weightsMapped: Self.quickWeightsBytes, contextLength: 32_768,
                                         precision: .f16, maxVerifyRows: 1, maxPrefillChunk: chunk,
                                         activationBytes: activations, prefixCacheRAMBudget: prefixBudget,
                                         budget: budget)
        let p = plan32.planned
        #expect(p.kvArena == 1 << 30, "512 pages x 2 MiB")
        #expect(p.recurrentState == 2 * 24 * 2_195_456)
        #expect(p.verifyArena == 24 * 2_195_456)
        #expect(p.prefixCacheRAM == prefixBudget)
        #expect(p.other == 2 * 512 * 4)
        #expect(p.total == Self.quickWeightsBytes + (1 << 30) + 3 * 24 * 2_195_456 + activations + prefixBudget + 4096)

        // INT8 (opt-in): 1 B values plus the F16 scale plane, 1,056,768 B per page.
        let g8 = Native.PagedMemoryPlanner.geometry(config: q, precision: .int8)
        #expect(g8.pageBytes == 8 * 2 * 4 * 64 * 256 + 8 * 2 * 4 * 64 * 2)
        let plan32i8 = try planner.resolve(config: q, weightsMapped: Self.quickWeightsBytes, contextLength: 32_768,
                                           precision: .int8, maxVerifyRows: 1, maxPrefillChunk: chunk,
                                           activationBytes: activations, prefixCacheRAMBudget: prefixBudget,
                                           budget: budget)
        #expect(plan32i8.planned.kvArena == 512 * UInt64(g8.pageBytes))
        #expect(plan32i8.planned.prefixCacheRAM == prefixBudget / UInt64(g8.pageBytes) * UInt64(g8.pageBytes))

        // The 8K plan the measured run uses.
        let plan8 = try planner.resolve(config: q, weightsMapped: Self.quickWeightsBytes, contextLength: 8192,
                                        precision: .f16, maxVerifyRows: 1, maxPrefillChunk: chunk,
                                        activationBytes: activations, prefixCacheRAMBudget: prefixBudget,
                                        budget: budget)
        #expect(plan8.planned.kvArena == 256 << 20)

        // A budget one byte short of the 32K total refuses.
        #expect(throws: Native.NativeError.self) {
            _ = try planner.resolve(config: q, weightsMapped: Self.quickWeightsBytes, contextLength: 32_768,
                                    precision: .f16, maxVerifyRows: 1, maxPrefillChunk: chunk,
                                    activationBytes: activations, prefixCacheRAMBudget: prefixBudget,
                                    budget: p.total - 1)
        }

        for (name, plan) in [("f16", plan32), ("int8", plan32i8), ("f16", plan8)] {
            let f = plan.planned
            figures.append(QuickPlanFigures(
                precision: name, contextLength: plan.contextLength,
                pageBytes: name == "f16" ? g16.pageBytes : g8.pageBytes, kvArena: f.kvArena,
                recurrentState: f.recurrentState, verifyArena: f.verifyArena, activations: f.activations,
                prefixCacheRAM: f.prefixCacheRAM, weightsMapped: f.weightsMapped, other: f.other, total: f.total,
                budget: plan.budget))
            print("quick plan \(name) \(plan.contextLength): kv \(f.kvArena / Self.mib) MiB, recurrent "
                  + "\(f.recurrentState / Self.mib) MiB, verify \(f.verifyArena / Self.mib) MiB, activations "
                  + "\(f.activations / Self.mib) MiB, prefix cache \(f.prefixCacheRAM / Self.mib) MiB, weights "
                  + "\(f.weightsMapped / Self.mib) MiB, total \(f.total / Self.mib) MiB of \(plan.budget / Self.mib) MiB")
        }
        if let out = ProcessInfo.processInfo.environment["MFERENCE_W4_OUT"] {
            let url = URL(fileURLWithPath: out)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(figures).write(to: url.appendingPathComponent("quick-plan-arithmetic.json"))
        }
    }
}
