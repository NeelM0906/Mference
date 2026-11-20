import Darwin
import Foundation
import Metal
import Testing
@testable import Mference

/// The server's multi-prefix cache (`PrefixStateCache`). A fake producer
/// whose state is the token history checks the state machine: after every
/// `begin`, the runner holds exactly the rows and recurrent state of the
/// chosen prefix. The toy Qwen 3.6 runner then checks, through the same
/// cache, that a continued turn's logits are bit-identical to the
/// single-prefix control (the same conversation without interleaved requests).
@Suite struct PrefixStateCacheTests {

    // MARK: Fixtures

    private static func tokenizer() async throws -> MFTokenizer {
        let folder = try #require(Bundle.module.url(
            forResource: "ChatMLTokenizer", withExtension: nil, subdirectory: "Fixtures"))
        return try await MFTokenizer.load(from: folder)
    }

    private static let domain = PromptPrefixCacheDomain(
        modelID: "toy", sourceSnapshotHash: nil, runtimeProfileHash: "runtime",
        maximumContext: 4096, kvStorage: "fp16", fp16RingEnabled: false, templateSHA256: "template")

    private static let page = Int(getpagesize())

    /// Exact-prefix hits only: with no messages the continuation rules never
    /// apply, so a turn continues an entry exactly when its rendered prompt
    /// extends the entry's KV-backed tokens.
    private static let turn = PromptPrefixCacheTurn(messages: [])

    private static func entry(kvBacked: [Int32], boundary: Int32) -> PromptPrefixCacheEntry? {
        PromptPrefixCache.makeEntry(
            domain: domain, turn: turn, content: "answer", calls: [],
            result: RawDecodeResult(prefillTokens: kvBacked.count, cachedPromptTokens: 0,
                                    computedPrefillTokens: kvBacked.count, prefillSeconds: 0,
                                    newTokens: 1, decodeSeconds: 0, reason: .endOfTurn,
                                    kvPosition: kvBacked.count, kvBackedTokenIDs: kvBacked,
                                    uncommittedBoundaryTokenIDs: [boundary], prefillExecution: nil))
    }

    // MARK: Fake producer

    /// KV rows are the tokens fed; the recurrent state is an order-dependent
    /// hash of them, which, like gated-DeltaNet state, cannot be rewound.
    final class FakeProducer: PrefixStateSnapshotting, @unchecked Sendable {
        static let seed: UInt64 = 0xcbf2_9ce4_8422_2325
        static func fold(_ state: UInt64, _ token: Int32) -> UInt64 {
            (state ^ UInt64(UInt32(bitPattern: token))) &* 0x0000_0100_0000_01b3
        }
        static func state(of tokens: [Int32]) -> UInt64 { tokens.reduce(seed, fold) }

        private(set) var rows: [Int32] = []
        private(set) var recurrent = seed
        var rowCaptures: [Int] = []
        /// Bytes an image reserves per token; only 4 of them hold the token.
        var rowBytesPerToken = MemoryLayout<Int32>.stride
        var failRestores = false
        var restores: [String] = []

        func feed(_ tokens: [Int32]) {
            for token in tokens {
                rows.append(token)
                recurrent = Self.fold(recurrent, token)
            }
        }

        func reset() {
            rows = []
            recurrent = Self.seed
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            precondition(position == rows.count)
            feed([token])
        }

        var continuationPosition: Int { rows.count }
        func prepareForContinuation(expectedPosition: Int) throws {
            guard rows.count == expectedPosition else { throw PrefixSnapshotError.outOfRange("cursor") }
        }

        var supportsPrefixStateSnapshots: Bool { true }
        func prefixKVRowBytes(tokens: Int) -> Int { tokens * rowBytesPerToken }
        var prefixRecurrentStateBytes: Int { MemoryLayout<UInt64>.stride }

        func capturePrefixKVRows(count: Int, into region: PrefixSnapshotRegion) throws {
            guard count > 0, count <= rows.count else { throw PrefixSnapshotError.outOfRange("rows") }
            rows.withUnsafeBytes { _ = memcpy(region.base, $0.baseAddress!, count * MemoryLayout<Int32>.stride) }
            rowCaptures.append(count)
        }

        func capturePrefixRecurrentState(into region: PrefixSnapshotRegion) throws {
            region.base.storeBytes(of: recurrent, as: UInt64.self)
        }

        func restorePrefixState(rows image: (region: PrefixSnapshotRegion, count: Int)?,
                                position: Int, recurrentState: PrefixSnapshotRegion) throws {
            if failRestores { throw PrefixSnapshotError.allocationFailed(bytes: 1) }
            if let image {
                guard position <= image.count else { throw PrefixSnapshotError.outOfRange("restore") }
                rows = Array(UnsafeBufferPointer(
                    start: image.region.base.assumingMemoryBound(to: Int32.self), count: image.count))
                restores.append("image")
            } else {
                guard position <= rows.count else { throw PrefixSnapshotError.outOfRange("rewind") }
                restores.append("rewind")
            }
            rows = Array(rows.prefix(position))
            recurrent = recurrentState.base.load(as: UInt64.self)
        }
    }

    /// Counts what the cache asks of its memory gate.
    final class CountingGate: PrefixCacheMemoryGate {
        final class Grant: PrefixCacheMemoryGrant {
            let gate: CountingGate
            let bytes: Int
            init(gate: CountingGate, bytes: Int) { self.gate = gate; self.bytes = bytes }
            func commit() { gate.committed += bytes }
            func release() { gate.released += bytes }
        }
        var refuseAbove: Int?
        var reserved = 0
        var committed = 0
        var released = 0
        func reserve(bytes: Int) -> (any PrefixCacheMemoryGrant)? {
            if let refuseAbove, bytes > refuseAbove { return nil }
            reserved += bytes
            return Grant(gate: self, bytes: bytes)
        }
    }

    /// One conversation turn on the fake producer, as the server runs it.
    private struct Driver {
        let cache: PrefixStateCache
        let fake = FakeProducer()
        let tokenizer: MFTokenizer

        /// Starts `prompt`, checks the runner holds the chosen prefix, then
        /// feeds the rest and `generated`; returns the KV-backed tokens.
        @discardableResult
        func run(_ prompt: [Int32], generated: [Int32], boundary: Int32 = 7,
                 expect outcome: PrefixStateCache.Outcome,
                 sourceLocation: SourceLocation = #_sourceLocation) throws -> [Int32] {
            let begun = try cache.begin(domain: PrefixStateCacheTests.domain,
                                        turn: PrefixStateCacheTests.turn,
                                        renderedPromptIDs: prompt, tokenizer: tokenizer,
                                        maxContext: 4096, runner: fake)
            #expect(begun.outcome == outcome, sourceLocation: sourceLocation)
            #expect(begun.effectivePromptIDs == prompt, sourceLocation: sourceLocation)
            switch begun.start {
            case .reset:
                fake.reset()
                fake.feed(prompt)
            case .resume(let cached):
                #expect(fake.rows == Array(prompt.prefix(cached)), sourceLocation: sourceLocation)
                #expect(fake.recurrent == FakeProducer.state(of: fake.rows), sourceLocation: sourceLocation)
                try fake.prepareForContinuation(expectedPosition: cached)
                fake.feed(Array(prompt[cached...]))
            }
            fake.feed(generated)
            let kv = prompt + generated
            cache.didFinish(entry: PrefixStateCacheTests.entry(kvBacked: kv, boundary: boundary),
                            runner: fake)
            return kv
        }

        /// A request that fails after `begin`: the server resets the runner.
        func fail(_ prompt: [Int32], expect outcome: PrefixStateCache.Outcome,
                  sourceLocation: SourceLocation = #_sourceLocation) throws {
            let begun = try cache.begin(domain: PrefixStateCacheTests.domain,
                                        turn: PrefixStateCacheTests.turn,
                                        renderedPromptIDs: prompt, tokenizer: tokenizer,
                                        maxContext: 4096, runner: fake)
            #expect(begun.outcome == outcome, sourceLocation: sourceLocation)
            fake.feed([99, 98])
            fake.reset()
            cache.didFail()
        }
    }

    private func seq(_ start: Int32, _ count: Int) -> [Int32] { (0..<Int32(count)).map { start + $0 } }

    // MARK: State machine

    @Test func oneConversationContinuesLiveWithoutCopyingRows() async throws {
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 1 << 20), tokenizer: try await Self.tokenizer())
        var kv = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        kv = try driver.run(kv + [7] + seq(30, 9), generated: seq(200, 3), expect: .live)
        try driver.run(kv + [7] + seq(40, 5), generated: seq(300, 2), expect: .live)
        #expect(driver.fake.rowCaptures.isEmpty)
        #expect(driver.fake.restores.isEmpty)
        #expect(driver.cache.imageCount == 0)
        #expect(driver.cache.checkpointCount == 3)
    }

    @Test func interleavedConversationsRestoreTheirOwnPrefixes() async throws {
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 1 << 20), tokenizer: try await Self.tokenizer())
        let a1 = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        let b1 = try driver.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        #expect(driver.fake.rowCaptures == [a1.count])
        let a2 = try driver.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .restored)
        #expect(driver.fake.rowCaptures == [a1.count, b1.count])
        let report = driver.cache.lastReport
        #expect(report.outcome == .restored)
        #expect(report.displacedBytes == b1.count * MemoryLayout<Int32>.stride)
        #expect(report.restoredBytes == a1.count * MemoryLayout<Int32>.stride + MemoryLayout<UInt64>.stride)
        #expect(report.capturedBytes == MemoryLayout<UInt64>.stride)
        let diagnostics = PrefixCacheDiagnostics(cache: driver.cache)
        #expect(diagnostics.outcome == "restored")
        #expect(diagnostics.retainedBytes == driver.cache.retainedBytes)
        #expect(diagnostics.images == 1)
        try driver.run(b1 + [7] + seq(700, 4), generated: seq(800, 2), expect: .restored)
        try driver.run(a2 + [7] + seq(40, 2), generated: seq(900, 2), expect: .restored)
        #expect(driver.fake.restores == ["image", "image", "image"])
    }

    @Test func aBranchRewindsAndKeepsTheLaterTurnAsAnImage() async throws {
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 1 << 20), tokenizer: try await Self.tokenizer())
        let a1 = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        let a2 = try driver.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .live)
        // An edited second message: continues A1, not A2.
        try driver.run(a1 + [7] + seq(50, 6), generated: seq(250, 3), expect: .rewound)
        #expect(driver.fake.rowCaptures == [a2.count])
        #expect(driver.cache.imageCount == 1)
        // Back to the original branch.
        try driver.run(a2 + [7] + seq(60, 3), generated: seq(260, 2), expect: .restored)
        #expect(driver.fake.restores == ["rewind", "image"])
    }

    @Test func aRegeneratedIdenticalTurnReplacesTheOlderCopy() async throws {
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 1 << 20), tokenizer: try await Self.tokenizer())
        let a1 = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        let prompt = a1 + [7] + seq(30, 9)
        try driver.run(prompt, generated: seq(200, 3), expect: .live)
        try driver.run(prompt, generated: seq(200, 3), expect: .rewound)
        #expect(driver.cache.checkpointCount == 2)
        #expect(driver.cache.imageCount == 0)
    }

    @Test func zeroBudgetServesExactlyAsSinglePrefix() async throws {
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 0), tokenizer: try await Self.tokenizer())
        let a1 = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        let a2 = try driver.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .live)
        try driver.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        try driver.run(a2 + [7] + seq(40, 2), generated: seq(900, 2), expect: .miss)
        try driver.run(a1 + [7] + seq(50, 6), generated: seq(250, 3), expect: .miss)
        #expect(driver.fake.rowCaptures.isEmpty)
        #expect(driver.fake.restores.isEmpty)
        #expect(driver.cache.retainedBytes == 0)
        #expect(driver.cache.checkpointCount == 1)
    }

    @Test func aFailedRequestLosesOnlyTheLivePrefix() async throws {
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 1 << 20), tokenizer: try await Self.tokenizer())
        let a1 = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        // B starts, which moves A out of the runner first, then fails.
        try driver.fail(seq(500, 15), expect: .miss)
        let a2Prompt = a1 + [7] + seq(30, 9)
        // A failure of A's own resumed request resets the runner (K2.0)...
        try driver.fail(a2Prompt, expect: .restored)
        // ...and with it every live prefix of A.
        try driver.run(a2Prompt, generated: seq(200, 3), expect: .miss)
        #expect(driver.cache.imageCount == 0)
    }

    @Test func theCeilingEvictsTheLeastRecentlyUsedConversation() async throws {
        // Every fake region is one page: an image and a recurrent copy per
        // displaced conversation, a copy for the live one.
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 4 * Self.page),
                            tokenizer: try await Self.tokenizer())
        let a1 = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        let b1 = try driver.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        try driver.run(seq(1000, 10), generated: seq(1100, 2), expect: .miss)
        #expect(driver.cache.retainedBytes <= 4 * Self.page)
        try driver.run(b1 + [7] + seq(700, 4), generated: seq(800, 2), expect: .restored)
        try driver.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .miss)
    }

    @Test func aRefusedGateDropsInsteadOfCopyingAndBalancesItsGrants() async throws {
        // The gate admits one-page recurrent copies and refuses row images.
        let gate = CountingGate()
        gate.refuseAbove = Self.page
        let tokenizer = try await Self.tokenizer()
        do {
            let driver = Driver(cache: PrefixStateCache(ceilingBytes: 1 << 20, gate: gate), tokenizer: tokenizer)
            driver.fake.rowBytesPerToken = 4096
            let a1 = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
            try driver.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
            try driver.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .miss)
            #expect(driver.fake.rowCaptures.isEmpty)
            #expect(gate.committed == gate.reserved)
        }
        #expect(gate.reserved > 0)
        #expect(gate.released == gate.reserved)
    }

    @Test func aFailedRestoreFallsBackToAMissAndKeepsOtherImages() async throws {
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 1 << 20), tokenizer: try await Self.tokenizer())
        let a1 = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        let b1 = try driver.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        driver.fake.failRestores = true
        try driver.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .miss)
        driver.fake.failRestores = false
        // A's image was kept through the failed restore; B was moved out
        // before it and stays restorable.
        try driver.run(b1 + [7] + seq(700, 4), generated: seq(800, 2), expect: .restored)
        try driver.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .restored)
    }

    @Test func aRequestCancelledBeforeItStartsLeavesTheLivePrefix() async throws {
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 1 << 20), tokenizer: try await Self.tokenizer())
        let a1 = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        final class Box: @unchecked Sendable {
            let cache: PrefixStateCache
            let fake: FakeProducer
            let tokenizer: MFTokenizer
            init(_ driver: Driver) { cache = driver.cache; fake = driver.fake; tokenizer = driver.tokenizer }
        }
        let box = Box(driver)
        let prompt = seq(500, 15)
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try box.cache.begin(domain: Self.domain, turn: Self.turn,
                                    renderedPromptIDs: prompt, tokenizer: box.tokenizer,
                                    maxContext: 4096, runner: box.fake)
        }
        await #expect(throws: CancellationError.self) { try await cancelled.value }
        #expect(driver.fake.rowCaptures.isEmpty)
        try driver.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .live)
    }

    @Test func evictionIsSkippedWhenProtectedStateAloneExceedsTheCeiling() async throws {
        // Three live turns hold three one-page copies; their image would need
        // a fourth page under a three-page ceiling. Nothing else is evicted.
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 3 * Self.page),
                            tokenizer: try await Self.tokenizer())
        var kv = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        kv = try driver.run(kv + [7] + seq(30, 2), generated: seq(200, 2), expect: .live)
        try driver.run(kv + [7] + seq(40, 2), generated: seq(300, 2), expect: .live)
        #expect(driver.cache.retainedBytes == 3 * Self.page)
        try driver.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        #expect(driver.fake.rowCaptures.isEmpty)
        #expect(driver.cache.checkpointCount == 1)
    }

    @Test func sheddingKeepsOnlyTheLivePrefix() async throws {
        let driver = Driver(cache: PrefixStateCache(ceilingBytes: 1 << 20), tokenizer: try await Self.tokenizer())
        let a1 = try driver.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        let b1 = try driver.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        driver.cache.shed()
        #expect(driver.cache.retainedBytes == 0)
        #expect(driver.cache.checkpointCount == 1)
        try driver.run(b1 + [7] + seq(700, 4), generated: seq(800, 2), expect: .live)
        try driver.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .miss)
    }

    // MARK: The live-memory governor

    /// A budget with no free memory: full admission would refuse everything,
    /// so any grant shows that the governor only shadows it.
    private static func emptyBudget() -> LiveMemoryBudget {
        LiveMemoryBudget(snapshot: {
            LiveMemorySnapshot(physical: 32 << 30, free: 0, inactive: 0, purgeable: 0, speculative: 0)
        })
    }

    final class Swap: @unchecked Sendable {
        private let lock = NSLock()
        private var value: UInt64 = 1 << 30
        var used: UInt64 { lock.withLock { value } }
        func grow(_ bytes: UInt64) { lock.withLock { value += bytes } }
    }

    final class Clock: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Double = 1000
        var seconds: Double { lock.withLock { value } }
        func advance(_ by: Double) { lock.withLock { value += by } }
    }

    final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func increment() { lock.withLock { value += 1 } }
        var count: Int { lock.withLock { value } }
    }

    private struct Governed {
        let budget = PrefixStateCacheTests.emptyBudget()
        let swap = Swap()
        let clock = Clock()
        let governor: PrefixCacheMemoryGovernor
        let driver: Driver

        init(tokenizer: MFTokenizer) {
            let swap = swap
            let clock = clock
            governor = PrefixCacheMemoryGovernor(budget: budget, swapUsed: { swap.used }, now: { clock.seconds })
            driver = Driver(cache: PrefixStateCache(ceilingBytes: 1 << 20, gate: governor), tokenizer: tokenizer)
        }

        var cache: PrefixStateCache { driver.cache }

        /// One request between the server's two safe points.
        @discardableResult
        func run(_ prompt: [Int32], generated: [Int32], expect outcome: PrefixStateCache.Outcome,
                 sourceLocation: SourceLocation = #_sourceLocation) throws -> [Int32] {
            governor.requestStarted(cache)
            defer { governor.requestEnded(cache) }
            return try driver.run(prompt, generated: generated, expect: outcome, sourceLocation: sourceLocation)
        }
    }

    @Test func aPressureEventShedsAtTheNextSafePointOnly() async throws {
        let g = Governed(tokenizer: try await Self.tokenizer())
        let raised = Counter()
        g.governor.setRaiseAction { raised.increment() }
        let a1 = try g.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        let b1 = try g.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        let retained = g.cache.retainedBytes
        #expect(retained > 0)
        g.budget.deliverPressure(.normal)
        #expect(raised.count == 0)
        #expect(g.governor.idleSafePoint(g.cache) == nil)

        g.budget.deliverPressure(.warning)
        #expect(raised.count == 1)
        // Raising the flag sheds nothing by itself.
        #expect(g.cache.retainedBytes == retained)
        #expect(g.governor.idleSafePoint(g.cache) == .pressure)
        #expect(g.cache.retainedBytes == 0)
        #expect(g.cache.checkpointCount == 1)
        #expect(g.governor.idleSafePoint(g.cache) == nil)
        let report = g.governor.report
        #expect(report.pressureSheds == 1)
        #expect(report.shedBytes == retained)
        #expect(PrefixCacheDiagnostics(cache: g.cache, memory: report).memory?.pressureSheds == 1)
        // The live prefix survives the shed; the displaced one does not.
        g.budget.deliverPressure(.normal)
        try g.run(b1 + [7] + seq(700, 4), generated: seq(800, 2), expect: .live)
        try g.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .miss)
    }

    @Test func noGrowthWhilePressureIsRaised() async throws {
        let g = Governed(tokenizer: try await Self.tokenizer())
        let a1 = try g.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        g.budget.deliverPressure(.critical)
        #expect(!g.governor.admitsGrowth)
        // The start sheds A's copy. B's image and recurrent copies are both
        // refused, and nothing is evicted to make room.
        #expect(g.governor.requestStarted(g.cache) == .pressure)
        try g.driver.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        #expect(g.driver.fake.rowCaptures.isEmpty)
        #expect(g.cache.retainedBytes == 0)
        g.governor.requestEnded(g.cache)
        try g.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .miss)
        // Back to normal, the cache grows again.
        g.budget.deliverPressure(.normal)
        let c1 = try g.run(seq(900, 12), generated: seq(950, 2), expect: .miss)
        try g.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        try g.run(c1 + [7] + seq(960, 3), generated: seq(970, 2), expect: .restored)
    }

    @Test func swapGrowthAfterTheCacheGrewShrinksItBackAndRefusesGrowthUntilSwapSettles() async throws {
        let g = Governed(tokenizer: try await Self.tokenizer())
        let a1 = try g.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        let sizeBeforeB = g.cache.retainedBytes
        #expect(sizeBeforeB > 0)
        // B grows the cache (A's image and B's copy); swap grows beyond the
        // limit during B.
        g.governor.requestStarted(g.cache)
        try g.driver.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        #expect(g.cache.retainedBytes > sizeBeforeB)
        g.swap.grow(PrefixCacheMemoryGovernor.defaultSwapGrowthLimit + 1)
        #expect(g.governor.requestEnded(g.cache) == .swapGrowth)
        #expect(g.cache.retainedBytes <= sizeBeforeB)
        #expect(!g.governor.admitsGrowth)
        #expect(g.governor.report.swapSheds == 1)
        #expect(g.governor.report.refusingForSwap)

        // Swap is still growing 50 s later: growth stays refused, and the
        // settle window restarts.
        g.clock.advance(50)
        g.swap.grow(64 << 20)
        #expect(g.governor.idleSafePoint(g.cache) == nil)
        #expect(!g.governor.admitsGrowth)
        // Flat, but only for 40 s: still refused through a whole request.
        g.clock.advance(40)
        g.governor.requestStarted(g.cache)
        try g.driver.run(a1 + [7] + seq(30, 9), generated: seq(200, 3), expect: .miss)
        #expect(g.cache.imageCount == 0)
        g.governor.requestEnded(g.cache)
        #expect(!g.governor.admitsGrowth)
        // Less than 16 MiB over 60 s: settled, and growth resumes.
        g.clock.advance(20)
        g.swap.grow(1 << 20)
        #expect(g.governor.idleSafePoint(g.cache) == nil)
        #expect(g.governor.admitsGrowth)
        #expect(!g.governor.report.refusingForSwap)
        let c1 = try g.run(seq(900, 12), generated: seq(950, 2), expect: .miss)
        try g.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        try g.run(c1 + [7] + seq(960, 3), generated: seq(970, 2), expect: .restored)
    }

    @Test func swapGrowthWithoutCacheGrowthShedsNothing() async throws {
        let g = Governed(tokenizer: try await Self.tokenizer())
        g.governor.idleSafePoint(g.cache)
        g.swap.grow(1 << 30)
        #expect(g.governor.idleSafePoint(g.cache) == nil)
        #expect(g.governor.admitsGrowth)
    }

    @Test func theIdleHopDefersWhileARequestRuns() async throws {
        let g = Governed(tokenizer: try await Self.tokenizer())
        try g.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        try g.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        let retained = g.cache.retainedBytes
        g.governor.requestStarted(g.cache)
        g.budget.deliverPressure(.warning)
        // The hop arrives mid-request: it must not touch the cache.
        #expect(g.governor.idleSafePoint(g.cache) == nil)
        #expect(g.cache.retainedBytes == retained)
        #expect(g.governor.isPressureRaised)
        // The request's end is the safe point that acts.
        #expect(g.governor.requestEnded(g.cache) == .pressure)
        #expect(g.cache.retainedBytes == 0)
    }

    @Test func shadowAdmissionCountsWhatFullBudgetAdmissionWouldRefuse() async throws {
        let g = Governed(tokenizer: try await Self.tokenizer())
        try g.run(seq(1, 20), generated: seq(100, 4), expect: .miss)
        try g.run(seq(500, 15), generated: seq(600, 3), expect: .miss)
        let report = g.governor.report
        // A's copy, A's image and B's copy were all kept, and an empty budget
        // would have refused each.
        #expect(report.admitted == 3)
        #expect(report.wouldRefuse == 3)
        #expect(report.grantableBytes == 0)
        #expect(g.cache.imageCount == 1)

        let roomy = LiveMemoryBudget(snapshot: {
            LiveMemorySnapshot(physical: 32 << 30, free: 16 << 30, inactive: 0, purgeable: 0, speculative: 0)
        })
        let governor = PrefixCacheMemoryGovernor(budget: roomy, swapUsed: { 0 }, now: { 0 })
        #expect(governor.reserve(bytes: Self.page) != nil)
        #expect(governor.report.wouldRefuse == 0)
        #expect(governor.report.grantableBytes == (16 << 30) - LiveMemoryBudget.defaultHeadroom(physical: 32 << 30))
    }

    @Test func droppingTheGovernorUnregistersItsPressureHandler() {
        let budget = Self.emptyBudget()
        let raised = Counter()
        do {
            let governor = PrefixCacheMemoryGovernor(budget: budget, swapUsed: { 0 }, now: { 0 })
            governor.setRaiseAction { raised.increment() }
            budget.deliverPressure(.warning)
            #expect(governor.isPressureRaised)
        }
        budget.deliverPressure(.warning)
        #expect(raised.count == 1)
    }

    // MARK: Toy Qwen 3.6, full-logit bit identity through the cache

    private struct Toy {
        let dir: URL
        let runner: RealForwardRunner
        let logits: MTLBuffer
        let cache = PrefixStateCache(ceilingBytes: 64 << 20)
        let tokenizer: MFTokenizer
    }

    private func makeToy() async throws -> Toy {
        let dir = try QwenToySynthetic.write()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: context.device, expecting: .qwen36Toy())
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 64,
            runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 32, forceLogitsHead: true,
                                                       kvGrowthTokens: 16))
        let logits = try #require(context.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.stride, options: .storageModeShared))
        return Toy(dir: dir, runner: runner, logits: logits, tokenizer: try await Self.tokenizer())
    }

    private func bits(_ buffer: MTLBuffer) -> [UInt16] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt16.self), count: 1024))
    }

    /// One turn through the cache: prefill what `begin` left, decode
    /// `generated` teacher-forced, publish. Returns the KV-backed tokens and
    /// every logits row (prompt end, then each decode step).
    private func turn(_ toy: Toy, _ prompt: [Int32], generated: [Int32],
                      expect outcome: PrefixStateCache.Outcome,
                      sourceLocation: SourceLocation = #_sourceLocation) async throws
        -> (kv: [Int32], logits: [[UInt16]]) {
        let begun = try toy.cache.begin(domain: Self.domain, turn: Self.turn, renderedPromptIDs: prompt,
                                        tokenizer: toy.tokenizer, maxContext: 64, runner: toy.runner)
        #expect(begun.outcome == outcome, sourceLocation: sourceLocation)
        let start: Int
        switch begun.start {
        case .reset:
            toy.runner.reset()
            start = 0
        case .resume(let cached):
            try toy.runner.prepareForContinuation(expectedPosition: cached)
            start = cached
        }
        _ = try await toy.runner.prefillChunked(
            tokens: prompt[start...], startPosition: start, outputMode: .logits,
            config: .production(chunkTokens: 32), into: toy.logits, onProgress: { _ in })
        var rows = [bits(toy.logits)]
        for (i, token) in generated.enumerated() {
            try await toy.runner.produce(token: token, position: prompt.count + i, into: toy.logits)
            rows.append(bits(toy.logits))
        }
        let kv = prompt + generated
        toy.cache.didFinish(entry: Self.entry(kvBacked: kv, boundary: 7), runner: toy.runner)
        return (kv, rows)
    }

    private let a1Prompt: [Int32] = (0..<19).map { Int32(($0 * 29 + 3) % 1000 + 1) }
    private let a1Generated: [Int32] = [11, 400, 7, 931]
    private let a2User: [Int32] = (0..<9).map { Int32(($0 * 17 + 5) % 1000 + 1) }
    private let a2Generated: [Int32] = [12, 401, 8, 932]
    private let bPrompt: [Int32] = (0..<23).map { Int32(($0 * 41 + 9) % 1000 + 1) }

    /// Control: the single-prefix server's view, A1 then A2 on a live runner.
    private func control() async throws -> [[UInt16]] {
        let toy = try await makeToy()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let a1 = try await turn(toy, a1Prompt, generated: a1Generated, expect: .miss)
        return try await turn(toy, a1.kv + [7] + a2User, generated: a2Generated, expect: .live).logits
    }

    @Test func toyRestoreAfterAnotherConversationMatchesTheControl() async throws {
        let expected = try await control()
        let toy = try await makeToy()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let a1 = try await turn(toy, a1Prompt, generated: a1Generated, expect: .miss)
        _ = try await turn(toy, bPrompt, generated: [3, 5, 8], expect: .miss)
        let a2 = try await turn(toy, a1.kv + [7] + a2User, generated: a2Generated, expect: .restored)
        #expect(a2.logits == expected)
    }

    @Test func toyRewindAfterABranchMatchesTheControl() async throws {
        let expected = try await control()
        let toy = try await makeToy()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let a1 = try await turn(toy, a1Prompt, generated: a1Generated, expect: .miss)
        _ = try await turn(toy, a1.kv + [7] + [44, 45, 46, 47, 48], generated: [3, 5], expect: .live)
        let a2 = try await turn(toy, a1.kv + [7] + a2User, generated: a2Generated, expect: .rewound)
        #expect(a2.logits == expected)
    }

    @Test func toyRestoreAfterACancelledRequestMatchesTheControl() async throws {
        let expected = try await control()
        let toy = try await makeToy()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let a1 = try await turn(toy, a1Prompt, generated: a1Generated, expect: .miss)

        // B is cancelled partway through its prefill; the server resets.
        let begun = try toy.cache.begin(domain: Self.domain, turn: Self.turn, renderedPromptIDs: bPrompt,
                                        tokenizer: toy.tokenizer, maxContext: 64, runner: toy.runner)
        #expect(begun.outcome == .miss)
        toy.runner.reset()
        toy.runner.prefillWillEncodeLayer = { layer in if layer == 2 { throw CancellationError() } }
        await #expect(throws: (any Error).self) {
            _ = try await toy.runner.prefillChunked(
                tokens: bPrompt[...], startPosition: 0, outputMode: .logits,
                config: .production(chunkTokens: 32), into: toy.logits, onProgress: { _ in })
        }
        toy.runner.prefillWillEncodeLayer = nil
        toy.runner.reset()
        toy.cache.didFail()

        let a2 = try await turn(toy, a1.kv + [7] + a2User, generated: a2Generated, expect: .restored)
        #expect(a2.logits == expected)
    }
}
