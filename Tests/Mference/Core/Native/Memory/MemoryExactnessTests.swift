import Foundation
import Metal
import Testing
@testable import Mference

/// W4 acceptance (inside-Mference rule, D1): with the reference ops, the
/// reference `ForwardModelFactory` and the reference `StepSchedulerFactory`,
/// `PagedKVStore` swapped in through `ModelComponents` gives logits
/// byte-identical to a cold run, after prefix reuse (RAM and SSD tiers,
/// partial-page copy-on-write) and after rollback (decode, prefill and
/// partially kept verify steps), in FP16 and INT8; and byte-identical to the
/// reference store on the same steps.
@Suite("MemoryExactnessTests", .serialized)
struct MemoryExactnessTests {
    typealias Fixture = ReferenceModelTests.Fixture

    enum Store {
        case reference
        case paged(ssd: Native.PrefixSSDTierConfiguration?)
    }

    /// The reference runner over the reference ops, with the chosen store.
    static func model(_ fx: Fixture, store kind: Store, context: Int = 256, chunk: Int = 16, verify: Int = 4,
                      precision: Native.KVPrecision = .f16,
                      prefixBudget: UInt64 = 1 << 24) throws -> Native.ReferenceForwardModel {
        let weights = try fx.store()
        let factory = Native.ReferenceForwardModelFactory(context: fx.ctx)
        let c = weights.config
        let activations = factory.activationBytes(config: c, maxPrefillChunk: chunk, maxVerifyRows: verify)
        let kv: any Native.KVStore
        let plan: Native.MemoryPlan
        switch kind {
        case .reference:
            plan = try Native.ReferenceMemoryPlanner().resolve(
                config: c, weightsMapped: weights.mappedByteCount, contextLength: context, precision: precision,
                maxVerifyRows: verify, maxPrefillChunk: chunk, activationBytes: activations,
                prefixCacheRAMBudget: prefixBudget, budget: 1 << 32)
            kv = try Native.ReferenceKVStore(config: c, plan: plan, device: fx.ctx.device)
        case .paged(let ssd):
            plan = try Native.PagedMemoryPlanner().resolve(
                config: c, weightsMapped: weights.mappedByteCount, contextLength: context, precision: precision,
                maxVerifyRows: verify, maxPrefillChunk: chunk, activationBytes: activations,
                prefixCacheRAMBudget: prefixBudget, budget: 1 << 32)
            kv = try Native.PagedKVStore(config: c, plan: plan, device: fx.ctx.device, ssdTier: ssd)
        }
        let components = Native.ModelComponents(
            device: fx.ctx.device, weights: weights, gemm: Native.ReferenceQuantGEMM(context: fx.ctx),
            attention: Native.ReferenceAttentionOp(context: fx.ctx), gdn: Native.ReferenceGDNOp(context: fx.ctx),
            glue: Native.ReferenceGlueOps(context: fx.ctx), kv: kv)
        return try factory.make(components, plan: plan) as! Native.ReferenceForwardModel
    }

    static func paged(_ m: Native.ReferenceForwardModel) -> Native.PagedKVStore { m.kv as! Native.PagedKVStore }

    static func step(_ m: Native.ReferenceForwardModel, _ fx: Fixture, _ kind: Native.StepKind, _ tokens: [Int32],
                     _ logits: Native.LogitRows = .last) throws -> [UInt32] {
        try ReferenceModelTests.step(m, fx.ctx, kind: kind, tokens: tokens, logits: logits).map(\.bitPattern)
    }

    /// Prefill `tokens` in chunks of `chunk` rows; the last chunk's last-row logits.
    static func prefill(_ m: Native.ReferenceForwardModel, _ fx: Fixture, _ tokens: [Int32],
                        chunk: Int = 16) throws -> [UInt32] {
        var out: [UInt32] = []
        var i = 0
        while i < tokens.count {
            let n = min(chunk, tokens.count - i)
            out = try step(m, fx, .prefill, Array(tokens[i ..< i + n]), i + n == tokens.count ? .last : .none)
            i += n
        }
        return out
    }

    static func sync<T>(_ fx: Fixture, _ body: (MTLCommandBuffer) throws -> T) throws -> T {
        try MemoryKVStoreTests.sync(fx.ctx.queue, body)
    }

    static func argmax(_ bits: [UInt32]) -> Int32 { Int32(ReferenceModelTests.argmax(bits.map(Float.init(bitPattern:)))) }

    /// Greedy decode `n` tokens from `logits`; every step's logits.
    static func decode(_ m: Native.ReferenceForwardModel, _ fx: Fixture, from logits: [UInt32],
                       _ n: Int) throws -> [[UInt32]] {
        var current = logits
        var out: [[UInt32]] = []
        for _ in 0..<n {
            current = try step(m, fx, .decode, [argmax(current)])
            out.append(current)
        }
        return out
    }

    static func prompt(_ n: Int, seed: UInt64) -> [Int32] {
        var rng = RefTest.SplitMix64(seed)
        return (0..<n).map { _ in Int32(rng.next() % 300) }
    }

    // MARK: Cold

    /// The same steps on the reference store and on the paged store give
    /// the same bits, decode included (page tables differ, values do not).
    @Test(arguments: [Native.KVPrecision.f16, .int8])
    func coldStepsMatchTheReferenceStore(precision: Native.KVPrecision) throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let tokens = Self.prompt(150, seed: 11)
        let ref = try Self.model(fx, store: .reference, precision: precision)
        let paged = try Self.model(fx, store: .paged(ssd: nil), precision: precision)
        let a = try Self.prefill(ref, fx, tokens)
        let b = try Self.prefill(paged, fx, tokens)
        #expect(a == b)
        let decodedRef = try Self.decode(ref, fx, from: a, 4)
        let decodedPaged = try Self.decode(paged, fx, from: b, 4)
        #expect(decodedRef == decodedPaged)
    }

    // MARK: Prefix reuse

    /// Cold over `prefix + followUp` versus warm: prefill `prefix`, cache
    /// it, reset, attach, prefill `followUp`. Both then decode greedily.
    static func checkReuse(_ fx: Fixture, precision: Native.KVPrecision = .f16, ssd: Native.PrefixSSDTierConfiguration?,
                           spill: Bool, prefix: [Int32], followUps: [[Int32]]) throws {
        let warm = try model(fx, store: .paged(ssd: ssd), precision: precision)
        let kv = paged(warm)
        _ = try Self.prefill(warm, fx, prefix)
        _ = try sync(fx) { try kv.insertPrefix(prefix, commandBuffer: $0) }
        for (i, followUp) in followUps.enumerated() {
            let cold = try model(fx, store: .paged(ssd: nil), precision: precision)
            let coldLogits = try Self.prefill(cold, fx, prefix + followUp)
            let coldDecode = try decode(cold, fx, from: coldLogits, 3)

            try sync(fx) { try kv.reset(commandBuffer: $0) }
            if spill { #expect(kv.spillPrefixCacheToSSD() >= 1) }
            let match = try #require(kv.lookupPrefix(prefix + followUp))
            #expect(match.length == prefix.count && match.tier == (spill ? .ssd : .ram))
            try sync(fx) { try kv.attachPrefix(match, commandBuffer: $0) }
            let warmLogits = try Self.prefill(warm, fx, followUp)
            #expect(warmLogits == coldLogits, "follow-up \(i): warm logits differ from cold")
            let warmDecode = try decode(warm, fx, from: warmLogits, 3)
            #expect(warmDecode == coldDecode, "follow-up \(i): decode differs")
        }
    }

    /// RAM tier, a prefix ending inside a page: the second follow-up proves
    /// the first one wrote only the sequence's copy of the partial page.
    @Test(arguments: [Native.KVPrecision.f16, .int8])
    func prefixReuseFromRAMIsByteIdentical(precision: Native.KVPrecision) throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        try Self.checkReuse(fx, precision: precision, ssd: nil, spill: false, prefix: Self.prompt(100, seed: 1),
                            followUps: [Self.prompt(30, seed: 2), Self.prompt(7, seed: 3)])
    }

    /// A page-aligned prefix (no tail page) and a one-token follow-up.
    @Test func pageAlignedPrefixReuseIsByteIdentical() throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        try Self.checkReuse(fx, ssd: nil, spill: false, prefix: Self.prompt(128, seed: 4),
                            followUps: [Self.prompt(1, seed: 5), Self.prompt(20, seed: 6)])
    }

    /// SSD tier: every attach loads the snapshot from disk.
    @Test(arguments: [Native.KVPrecision.f16, .int8])
    func prefixReuseFromSSDIsByteIdentical(precision: Native.KVPrecision) throws {
        let fx = try ReferenceModelTests.fixture()
        let dir = try MemoryKVStoreTests.ssdDirectory()
        defer {
            try? FileManager.default.removeItem(at: fx.dir)
            try? FileManager.default.removeItem(at: dir)
        }
        try Self.checkReuse(fx, precision: precision, ssd: .init(directory: dir, budgetBytes: 64 << 20), spill: true,
                            prefix: Self.prompt(90, seed: 7), followUps: [Self.prompt(25, seed: 8)])
    }

    /// A prefix that extends a cached one shares its whole pages (attached,
    /// then extended and cached again); the longer one reuses exactly too.
    @Test func chainedPrefixesShareAndReuseExactly() throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let p1 = Self.prompt(70, seed: 21), f1 = Self.prompt(80, seed: 22), f2 = Self.prompt(9, seed: 23)
        let warm = try Self.model(fx, store: .paged(ssd: nil))
        let kv = Self.paged(warm)
        _ = try Self.prefill(warm, fx, p1)
        _ = try Self.sync(fx) { try kv.insertPrefix(p1, commandBuffer: $0) }
        try Self.sync(fx) { try kv.reset(commandBuffer: $0) }
        try Self.sync(fx) { try kv.attachPrefix(kv.lookupPrefix(p1 + f1)!, commandBuffer: $0) }
        let shared = kv.physicalPages[0]
        _ = try Self.prefill(warm, fx, f1)
        _ = try Self.sync(fx) { try kv.insertPrefix(p1 + f1, commandBuffer: $0) }
        try Self.sync(fx) { try kv.reset(commandBuffer: $0) }
        let match = try #require(kv.lookupPrefix(p1 + f1 + f2))
        #expect(match.length == 150)
        try Self.sync(fx) { try kv.attachPrefix(match, commandBuffer: $0) }
        #expect(kv.physicalPages[0] == shared, "page 0 is shared by both prefixes")
        let warmLogits = try Self.prefill(warm, fx, f2)
        let cold = try Self.model(fx, store: .paged(ssd: nil))
        let coldLogits = try Self.prefill(cold, fx, p1 + f1 + f2)
        #expect(warmLogits == coldLogits)
    }

    // MARK: Rollback

    @Test(arguments: [Native.KVPrecision.f16, .int8])
    func rollbackIsByteIdentical(precision: Native.KVPrecision) throws {
        let fx = try ReferenceModelTests.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let p = Self.prompt(60, seed: 31)
        let (x, y) = (Int32(17), Int32(201))
        let m = try Self.model(fx, store: .paged(ssd: nil), precision: precision)
        let kv = Self.paged(m)
        _ = try Self.prefill(m, fx, p)

        // Decode y, undo it, decode x: as a cold run over p + [x].
        _ = try Self.step(m, fx, .decode, [y])
        try Self.sync(fx) { try kv.rollback(to: 60, commandBuffer: $0) }
        let afterX = try Self.step(m, fx, .decode, [x])
        func cold(_ tokens: [Int32]) throws -> [UInt32] {
            try Self.prefill(try Self.model(fx, store: .paged(ssd: nil), precision: precision), fx, tokens)
        }
        let coldX = try cold(p + [x])
        #expect(afterX == coldX)

        // A prefill step that crosses a page boundary, undone, then another.
        let q = Self.prompt(12, seed: 32), q2 = Self.prompt(9, seed: 33)
        _ = try Self.step(m, fx, .prefill, q, .none)
        try Self.sync(fx) { try kv.rollback(to: 61, commandBuffer: $0) }
        let afterQ2 = try Self.step(m, fx, .prefill, q2)
        let coldQ2 = try cold(p + [x] + q2)
        #expect(afterQ2 == coldQ2)

        // Verify four rows, keep two, decode: as a cold run over the kept rows.
        let v = Self.prompt(4, seed: 34)
        let start = kv.length
        let all = try Self.step(m, fx, .verify, v, .all)
        try Self.sync(fx) { try kv.rollback(to: start + 2, commandBuffer: $0) }
        let next = try Self.step(m, fx, .decode, [x])
        let base = p + [x] + q2
        let coldNext = try cold(base + Array(v.prefix(2)) + [x])
        #expect(next == coldNext)
        // And the verify rows themselves were the cold rows.
        let vocab = m.config.vocabSize
        let coldRow1 = try cold(base + Array(v.prefix(2)))
        #expect(Array(all[vocab ..< 2 * vocab]) == coldRow1)
    }

    // MARK: Through the reference scheduler

    actor TokenLog {
        var tokens: [Int32] = []
        func add(_ e: Native.StepEvent) { if case .token(let t) = e { tokens.append(t) } }
    }

    /// The reference scheduler over the paged store: generations equal the
    /// reference store's, and a follow-up reusing the cached prompt (from
    /// RAM, then from SSD) generates what a cold run does.
    @Test func schedulerWithThePagedStoreMatchesAndReusesExactly() async throws {
        let fx = try ReferenceModelTests.fixture()
        let dir = try MemoryKVStoreTests.ssdDirectory()
        defer {
            try? FileManager.default.removeItem(at: fx.dir)
            try? FileManager.default.removeItem(at: dir)
        }
        let greedy = ReferenceModelTests.greedy
        func job(_ prompt: [Int32], reuse: Bool, n: Int = 6) -> Native.GenerationJob {
            Native.GenerationJob(promptTokens: prompt, params: greedy, maxNewTokens: n, stopTokens: [],
                                 reusePrefix: reuse, prefillChunk: 8)
        }
        func scheduler(_ store: Store) throws -> any Native.StepScheduler {
            try Native.ReferenceStepSchedulerFactory().make(
                model: try Self.model(fx, store: store, chunk: 8), sampler: try Native.ReferenceSampler(context: nil),
                strategy: Native.ReferencePlainDecoding(), queue: fx.ctx.queue)
        }
        let probe = ReferenceModelTests.StoreProbe()
        let model = try Self.model(fx, store: .paged(ssd: .init(directory: dir, budgetBytes: 64 << 20)), chunk: 8)
        probe.kv = model.kv
        let paged = try Native.ReferenceStepSchedulerFactory().make(
            model: model, sampler: try Native.ReferenceSampler(context: nil),
            strategy: Native.ReferencePlainDecoding(), queue: fx.ctx.queue)
        let reference = try scheduler(.reference)

        let prompt = Self.prompt(77, seed: 41)
        let a = try await paged.run(job(prompt, reuse: true)) { _ in }
        let b = try await reference.run(job(prompt, reuse: true)) { _ in }
        #expect(a.tokens == b.tokens && a.reusedPrefixTokens == 0)

        let followUp = prompt + a.tokens.prefix(4) + Self.prompt(10, seed: 42)
        let cold = try await reference.run(job(followUp, reuse: false)) { _ in }
        let warm = try await paged.run(job(followUp, reuse: true)) { _ in }
        #expect(warm.reusedPrefixTokens == prompt.count)
        #expect(warm.tokens == cold.tokens)

        // The same follow-up again, its cached prompt now on the SSD tier.
        let kv = try #require(probe.kv as? Native.PagedKVStore)
        #expect(kv.spillPrefixCacheToSSD() >= 1)
        let fromSSD = try await paged.run(job(followUp, reuse: true)) { _ in }
        #expect(fromSSD.reusedPrefixTokens == prompt.count)
        #expect(fromSSD.tokens == cold.tokens)
        #expect(kv.counters.ssdAttaches == 1)
    }
}
