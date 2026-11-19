import Foundation
import Metal
import Testing
@testable import Mference

/// `PagedKVStore` semantics without a model: the contract's step, bank and
/// rollback rules, page sharing in the radix tree, copy-on-write of partial
/// pages, eviction under the cache reserve, and the SSD tier.
@Suite("MemoryKVStoreTests", .serialized)
struct MemoryKVStoreTests {
    static func store(context: Int = 256, verify: Int = 4, precision: Native.KVPrecision = .f16,
                      cachePages: Int = 64, ssd: Native.PrefixSSDTierConfiguration? = nil,
                      device: MTLDevice) throws -> Native.PagedKVStore {
        let c = RefTest.tiny
        let g = Native.PagedMemoryPlanner.geometry(config: c, precision: precision)
        let plan = try Native.PagedMemoryPlanner().resolve(
            config: c, weightsMapped: 0, contextLength: context, precision: precision, maxVerifyRows: verify,
            maxPrefillChunk: 64, activationBytes: 0, prefixCacheRAMBudget: UInt64(cachePages * g.pageBytes),
            budget: 1 << 32)
        return try Native.PagedKVStore(config: c, plan: plan, device: device, ssdTier: ssd)
    }

    static func sync<T>(_ queue: MTLCommandQueue, _ body: (MTLCommandBuffer) throws -> T) throws -> T {
        try autoreleasepool {
            let cb = queue.makeCommandBuffer()!
            let v = try body(cb)
            cb.commit()
            cb.waitUntilCompleted()
            return v
        }
    }

    static func fill(_ ref: Native.RecurrentStateRef, _ value: Float) {
        let p = ref.buffer.contents().advanced(by: ref.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<ref.layout.layerBytes / 4 { p[i] = value }
    }

    static func first(_ ref: Native.RecurrentStateRef) -> Float {
        ref.buffer.contents().advanced(by: ref.offset + ref.layout.ssmOffset).load(as: Float.self)
    }

    /// Writes byte `value(token)` into every byte of every K/V element of
    /// the sequence's tokens `range`, through the current page table (what
    /// `AttentionOp` does when it appends).
    static func writeKV(_ kv: Native.PagedKVStore, _ range: Range<Int>, _ value: (Int) -> UInt8) {
        let t = kv.pageTable()
        let g = kv.geometry
        let elementBytes = g.precision == .f16 ? 2 : 1
        for pos in range {
            let page = Int(t.pageTable.contents().advanced(by: t.pageTableOffset + pos / g.tokensPerPage * 4)
                .load(as: UInt32.self))
            for layer in 0..<g.attentionLayers {
                for k in 0..<2 {
                    for h in 0..<g.kvHeads {
                        let off = page * g.pageBytes + g.valueOffset(attnLayer: layer, kv: k, kvHead: h,
                                                                     token: pos % g.tokensPerPage, dim: 0)
                        memset(t.arena.contents().advanced(by: off), Int32(value(pos)), g.headDim * elementBytes)
                    }
                }
            }
        }
    }

    /// Reads one byte per token of `range` (last layer, V, last head, last byte).
    static func readKV(_ kv: Native.PagedKVStore, _ range: Range<Int>) -> [UInt8] {
        let g = kv.geometry
        let elementBytes = g.precision == .f16 ? 2 : 1
        return range.map { pos in
            let page = kv.physicalPages[pos / g.tokensPerPage]
            let off = page * g.pageBytes + g.valueOffset(attnLayer: g.attentionLayers - 1, kv: 1, kvHead: g.kvHeads - 1,
                                                         token: pos % g.tokensPerPage, dim: g.headDim - 1)
            return kv.arena.contents().load(fromByteOffset: off + elementBytes - 1, as: UInt8.self)
        }
    }

    static func fillLiveBank(_ kv: Native.PagedKVStore, _ value: Float) {
        for l in 0..<kv.gdnLayerCount { fill(kv.recurrentState(gdnLayer: l).output, value) }
    }

    static func liveBankValues(_ kv: Native.PagedKVStore) -> [Float] {
        let base = kv.recurrent.contents().advanced(by: kv.liveBank * kv.gdnLayerCount * kv.recurrentLayout.layerBytes)
        return Array(UnsafeBufferPointer(start: base.assumingMemoryBound(to: Float.self), count: kv.snapshotBytes / 4))
    }

    /// Runs a "step": begin, write K/V for its rows and fill the output bank.
    static func step(_ kv: Native.PagedKVStore, rows: Int, kind: Native.StepKind = .prefill,
                     kvValue: @escaping (Int) -> UInt8 = { UInt8($0 % 251) }, bank: Float) throws {
        let start = kv.length
        try kv.beginStep(rows: rows, kind: kind)
        writeKV(kv, start ..< start + rows, kvValue)
        for l in 0..<kv.gdnLayerCount { fill(kv.recurrentState(gdnLayer: l).output, bank) }
    }

    static func tokens(_ n: Int, seed: Int32 = 0) -> [Int32] { (0..<Int32(n)).map { ($0 * 7 + seed) % 300 } }

    // MARK: Contract semantics (as the reference store's)

    @Test func stepsBanksAndRollback() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(context: 200, device: ctx.device)
        #expect(throws: Native.NativeError.self) { try kv.rollback(to: 0, commandBuffer: ctx.queue.makeCommandBuffer()!) }
        try kv.beginStep(rows: 10, kind: .prefill)
        let s1 = kv.recurrentState(gdnLayer: 1)
        #expect(s1.input.offset != s1.output.offset)
        Self.fill(s1.output, 1)
        try kv.beginStep(rows: 1, kind: .decode)
        let s2 = kv.recurrentState(gdnLayer: 1)
        #expect(s2.input.offset == s1.output.offset && s2.output.offset == s1.input.offset)
        Self.fill(s2.output, 2)
        try Self.sync(ctx.queue) { try kv.rollback(to: 10, commandBuffer: $0) }
        #expect(kv.length == 10)
        #expect(throws: Native.NativeError.self) { try kv.rollback(to: 10, commandBuffer: ctx.queue.makeCommandBuffer()!) }
        try kv.beginStep(rows: 4, kind: .verify)
        let s3 = kv.recurrentState(gdnLayer: 1)
        #expect(s3.input.offset == s1.output.offset && Self.first(s3.input) == 1)
        let arena = kv.verifyArena(gdnLayer: 1)
        for r in 0..<4 {
            Self.fill(Native.RecurrentStateRef(layout: arena.layout, buffer: arena.buffer,
                                               offset: arena.offset + r * arena.slotStride), Float(10 + r))
        }
        Self.fill(s3.output, 99)
        try Self.sync(ctx.queue) { try kv.rollback(to: 12, commandBuffer: $0) }
        #expect(kv.length == 12 && Self.first(s3.output) == 11)
        try kv.beginStep(rows: 1, kind: .decode)
        #expect(kv.recurrentState(gdnLayer: 1).input.offset == s3.output.offset)
        #expect(throws: Native.NativeError.self) { try kv.beginStep(rows: 200, kind: .prefill) }
        #expect(throws: Native.NativeError.self) { try kv.beginStep(rows: 0, kind: .prefill) }
    }

    @Test func partialRollbackOnlyAfterVerifyAndNoOpAtLength() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(device: ctx.device)
        try kv.beginStep(rows: 4, kind: .prefill)
        #expect(throws: Native.NativeError.self) { try kv.rollback(to: 2, commandBuffer: ctx.queue.makeCommandBuffer()!) }
        try Self.sync(ctx.queue) { try kv.rollback(to: 4, commandBuffer: $0) }
        #expect(kv.length == 4)
        try kv.beginStep(rows: 1, kind: .decode)
        try Self.sync(ctx.queue) { try kv.rollback(to: 5, commandBuffer: $0) }
        #expect(kv.length == 5)
        try kv.beginStep(rows: 3, kind: .verify)
        try Self.sync(ctx.queue) { try kv.rollback(to: 6, commandBuffer: $0) }
        #expect(kv.length == 6)
        try kv.beginStep(rows: 3, kind: .verify)
        #expect(throws: Native.NativeError.self) { try kv.rollback(to: 5, commandBuffer: ctx.queue.makeCommandBuffer()!) }
        #expect(throws: Native.NativeError.self) { try kv.rollback(to: 10, commandBuffer: ctx.queue.makeCommandBuffer()!) }
    }

    @Test func rollbackIsRefusedAfterCacheOperationsAndReset() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(device: ctx.device)
        func refused() {
            #expect(throws: Native.NativeError.self) {
                try kv.rollback(to: kv.length, commandBuffer: ctx.queue.makeCommandBuffer()!)
            }
        }
        try Self.step(kv, rows: 70, bank: 1)
        let handle = try Self.sync(ctx.queue) { try kv.insertPrefix(Self.tokens(70), commandBuffer: $0) }
        refused()
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        refused()
        try Self.sync(ctx.queue) {
            try kv.attachPrefix(Native.PrefixMatch(handle: handle, length: 70, tier: .ram), commandBuffer: $0)
        }
        refused()
        try kv.beginStep(rows: 1, kind: .decode)
        try Self.sync(ctx.queue) { try kv.rollback(to: 70, commandBuffer: $0) }
    }

    @Test func ringSlotsAlternateAndPageTablesAreNotTheIdentity() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(device: ctx.device)
        try kv.beginStep(rows: 130, kind: .prefill)
        let a = kv.pageTable()
        #expect(a.pageCount == 3 && a.tokenCount == 130)
        #expect(kv.physicalPages != [0, 1, 2])
        try kv.beginStep(rows: 1, kind: .decode)
        let b = kv.pageTable().pageTableOffset
        try kv.beginStep(rows: 1, kind: .decode)
        #expect(a.pageTableOffset != b && kv.pageTable().pageTableOffset == a.pageTableOffset)
        // Slot a is not rewritten by the step after it (two in flight).
        let tableA = a.pageTable.contents().advanced(by: a.pageTableOffset).assumingMemoryBound(to: UInt32.self)
        #expect((0..<3).map { Int(tableA[$0]) } == Array(kv.physicalPages.prefix(3)))
    }

    // MARK: Prefix cache

    @Test func resetAttachAndCopyOnWriteOfThePartialPage() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(device: ctx.device)
        let t = Self.tokens(100)
        try Self.step(kv, rows: 100, bank: 3)
        let before = Self.readKV(kv, 0..<100)
        let handle = try Self.sync(ctx.queue) { try kv.insertPrefix(t, commandBuffer: $0) }
        // The cache shares page 0 (whole) and keeps its own copy of page 1 (partial).
        let seqPages = kv.physicalPages
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        #expect(kv.length == 0 && Self.liveBankValues(kv).allSatisfy { $0 == 0 })
        #expect(kv.lookupPrefix(Array(t.prefix(99))) == nil)
        let match = try #require(kv.lookupPrefix(t + [5, 6]))
        #expect(match.handle == handle && match.length == 100 && match.tier == .ram)
        try Self.sync(ctx.queue) { try kv.attachPrefix(match, commandBuffer: $0) }
        #expect(kv.length == 100)
        #expect(kv.physicalPages[0] == seqPages[0], "whole page shared")
        #expect(Self.readKV(kv, 0..<100) == before)
        #expect(Self.liveBankValues(kv).allSatisfy { $0 == 3 })
        #expect(kv.counters.copyOnWritePages == 1)
        // Continue writing into the copy; a second attach still sees the cached bytes.
        try Self.step(kv, rows: 10, kvValue: { _ in 0xEE }, bank: 4)
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        try Self.sync(ctx.queue) { try kv.attachPrefix(match, commandBuffer: $0) }
        #expect(Self.readKV(kv, 0..<100) == before)
        #expect(Self.liveBankValues(kv).allSatisfy { $0 == 3 })
    }

    @Test func radixTreeSharesPagesAndFindsTheLongestPrefix() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(context: 256, device: ctx.device)
        let a = Self.tokens(130)                       // pages 0, 1 whole + tail 2
        let b = Array(a.prefix(128)) + Self.tokens(60, seed: 5)  // shares pages 0, 1
        let c = Array(a.prefix(64)) + Self.tokens(80, seed: 9)   // shares page 0 only
        var handles: [Native.PrefixHandle] = []
        for (i, tokens) in [a, b, c].enumerated() {
            try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
            try Self.step(kv, rows: tokens.count, bank: Float(i + 1))
            handles.append(try Self.sync(ctx.queue) { try kv.insertPrefix(tokens, commandBuffer: $0) })
        }
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        #expect(kv.lookupPrefix(a + [1])?.handle == handles[0])
        #expect(kv.lookupPrefix(b + [1])?.handle == handles[1])
        #expect(kv.lookupPrefix(c)?.handle == handles[2])
        #expect(kv.lookupPrefix(Array(a.prefix(129))) == nil)
        // Unique pages: 3 tree nodes (a0, a1, c1) for b; a's page 1 is shared with b,
        // c's page 1 differs; tails a2, b2, c2; 3 snapshots of 2 pages.
        let pagesHeld = kv.footprint.prefixCacheRAM / UInt64(kv.geometry.pageBytes)
        #expect(pagesHeld == UInt64(3 + 3 + 3 * kv.snapshotPages))
        // Inserting the same tokens again replaces the entry.
        try Self.step(kv, rows: 130, bank: 7)
        let again = try Self.sync(ctx.queue) { try kv.insertPrefix(a, commandBuffer: $0) }
        #expect(again != handles[0] && kv.cachedPrefixCount.ram == 3)
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        try Self.sync(ctx.queue) { try kv.attachPrefix(kv.lookupPrefix(a)!, commandBuffer: $0) }
        #expect(Self.liveBankValues(kv).allSatisfy { $0 == 7 })
        // Everything is returned when the cache is cleared.
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        kv.removeAllPrefixes()
        try kv.beginStep(rows: 1, kind: .decode)   // reclaims the reset's quarantine
        #expect(kv.freePageCount == kv.contextPages + kv.cachePages - 1)
        #expect(kv.footprint.prefixCacheRAM == 0 && kv.cachedPrefixCount.ram == 0)
    }

    @Test func evictionKeepsTheReserveAndDropsTheLeastRecentlyUsed() throws {
        let ctx = try MetalContext()
        // One prefix of 100 tokens holds 2 pages + 2 snapshot pages once unshared: reserve two.
        let probe = try Self.store(device: ctx.device)
        let perPrefix = 2 + probe.snapshotPages
        let kv = try Self.store(cachePages: 2 * perPrefix, device: ctx.device)
        func add(_ i: Int) throws -> Native.PrefixHandle {
            try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
            try Self.step(kv, rows: 100, kvValue: { _ in UInt8(i + 1) }, bank: Float(i))
            return try Self.sync(ctx.queue) { try kv.insertPrefix(Self.tokens(100, seed: Int32(i)), commandBuffer: $0) }
        }
        let h0 = try add(0)
        _ = try add(1)
        // Touch prefix 0, so prefix 1 is the least recently used when prefix 2 needs room.
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        try Self.sync(ctx.queue) { try kv.attachPrefix(kv.lookupPrefix(Self.tokens(100, seed: 0))!, commandBuffer: $0) }
        let h2 = try add(2)
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        try kv.beginStep(rows: 1, kind: .decode)
        #expect(kv.footprint.prefixCacheRAM <= UInt64(kv.cachePages * kv.geometry.pageBytes))
        #expect(kv.lookupPrefix(Self.tokens(100, seed: 1)) == nil)
        #expect(kv.lookupPrefix(Self.tokens(100, seed: 0))?.handle == h0)
        #expect(kv.lookupPrefix(Self.tokens(100, seed: 2))?.handle == h2)
        #expect(kv.counters.drops == 1)
        // A prefix whose private pages exceed the whole reserve is refused.
        let small = try Self.store(cachePages: 1, device: ctx.device)
        try small.beginStep(rows: 3, kind: .prefill)
        do {
            _ = try Self.sync(ctx.queue) { try small.insertPrefix([1, 2, 3], commandBuffer: $0) }
            Issue.record("insert fit a one-page reserve")
        } catch Native.NativeError.memoryBudgetExceeded(_, let budget) {
            #expect(budget == UInt64(small.geometry.pageBytes))
        }
    }

    // MARK: SSD tier

    static func ssdDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mference-w4-ssd-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func ssdSpillAndLoadRoundTripTheBytes() throws {
        let ctx = try MetalContext()
        let dir = try Self.ssdDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        for precision in [Native.KVPrecision.f16, .int8] {
            let kv = try Self.store(precision: precision,
                                    ssd: .init(directory: dir, budgetBytes: 64 << 20), device: ctx.device)
            let t = Self.tokens(150)
            try Self.step(kv, rows: 150, kvValue: { UInt8(($0 * 3) % 256) }, bank: 0.25)
            let kvBefore = Self.readKV(kv, 0..<150)
            let bankBefore = Self.liveBankValues(kv)
            _ = try Self.sync(ctx.queue) { try kv.insertPrefix(t, commandBuffer: $0) }
            try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
            #expect(kv.spillPrefixCacheToSSD() == 1)
            #expect(kv.footprint.prefixCacheRAM == 0 && kv.ssdBytes > 0)
            let match = try #require(kv.lookupPrefix(t + [1]))
            #expect(match.tier == .ssd && match.length == 150)
            try Self.sync(ctx.queue) { try kv.attachPrefix(match, commandBuffer: $0) }
            #expect(kv.length == 150 && kv.counters.ssdAttaches == 1)
            #expect(Self.readKV(kv, 0..<150) == kvBefore)
            #expect(Self.liveBankValues(kv) == bankBefore)
            // The staging pages are released once the attach completes.
            try kv.beginStep(rows: 1, kind: .decode)
            #expect(kv.freePageCount == kv.contextPages + kv.cachePages - kv.physicalPages.count)
        }
    }

    @Test func ssdCorruptionIsDetectedAndBudgetDeletesTheOldest() throws {
        let ctx = try MetalContext()
        let dir = try Self.ssdDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let kv = try Self.store(ssd: .init(directory: dir, budgetBytes: 64 << 20), device: ctx.device)
        let t = Self.tokens(80)
        try Self.step(kv, rows: 80, bank: 1)
        _ = try Self.sync(ctx.queue) { try kv.insertPrefix(t, commandBuffer: $0) }
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        #expect(kv.spillPrefixCacheToSSD() == 1)
        let files = try FileManager.default.subpathsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".mfkv") }
        let file = dir.appendingPathComponent(try #require(files.first))
        let handle = try FileHandle(forUpdating: file)
        try handle.seek(toOffset: 16384 + 100)
        try handle.write(contentsOf: Data([0xAB]))
        try handle.close()
        let match = try #require(kv.lookupPrefix(t))
        do {
            try Self.sync(ctx.queue) { try kv.attachPrefix(match, commandBuffer: $0) }
            Issue.record("a corrupt SSD snapshot attached")
        } catch Native.NativeError.integrity {}
        #expect(kv.length == 0 && kv.lookupPrefix(t) == nil && kv.ssdBytes == 0)

        // A budget of about one file keeps only the most recently spilled one.
        let one = try Self.store(ssd: .init(directory: dir, budgetBytes: 0), device: ctx.device)
        try Self.step(one, rows: 80, bank: 1)
        _ = try Self.sync(ctx.queue) { try one.insertPrefix(t, commandBuffer: $0) }
        #expect(one.spillPrefixCacheToSSD() == 0, "a file over the whole SSD budget is not written")
        let probeBytes: UInt64 = {
            let g = one.geometry
            return UInt64(16384 + 2 * g.pageBytes + one.snapshotBytes)
        }()
        let two = try Self.store(ssd: .init(directory: dir, budgetBytes: probeBytes + 1000), device: ctx.device)
        for seed in [Int32(1), 2] {
            try Self.sync(ctx.queue) { try two.reset(commandBuffer: $0) }
            try Self.step(two, rows: 80, bank: Float(seed))
            _ = try Self.sync(ctx.queue) { try two.insertPrefix(Self.tokens(80, seed: seed), commandBuffer: $0) }
            try Self.sync(ctx.queue) { try two.reset(commandBuffer: $0) }
            #expect(two.spillPrefixCacheToSSD() == 1)
        }
        #expect(two.lookupPrefix(Self.tokens(80, seed: 1)) == nil)
        #expect(two.lookupPrefix(Self.tokens(80, seed: 2))?.tier == .ssd)
        #expect(two.counters.ssdDeletes == 1)
    }

    @Test func ssdDirectoriesAreOwnedAndStaleOnesRemoved() throws {
        let ctx = try MetalContext()
        let dir = try Self.ssdDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        // A directory of a process that no longer exists (pid 999999 is above the default pid_max).
        let stale = dir.appendingPathComponent("mfkv-999999-stale")
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
        var kv: Native.PagedKVStore? = try Self.store(ssd: .init(directory: dir, budgetBytes: 1 << 20),
                                                      device: ctx.device)
        #expect(!FileManager.default.fileExists(atPath: stale.path))
        let owned = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        #expect(owned.count == 1 && owned[0].hasPrefix("mfkv-\(getpid())-"))
        _ = kv
        kv = nil
        #expect(try FileManager.default.contentsOfDirectory(atPath: dir.path).isEmpty)
    }

    // MARK: Command-buffer safety

    /// Pages a rollback releases stay quarantined until its command buffer
    /// completes, so nothing can reuse a page a pending command still uses.
    @Test func releasedPagesWaitForTheirCommandBuffer() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(device: ctx.device)
        try kv.beginStep(rows: 60, kind: .prefill)
        try kv.beginStep(rows: 10, kind: .prefill)   // crosses into page 1
        let total = kv.contextPages + kv.cachePages
        #expect(kv.freePageCount == total - 2)
        let cb = ctx.queue.makeCommandBuffer()!
        try kv.rollback(to: 60, commandBuffer: cb)
        #expect(kv.freePageCount == total - 2, "quarantined while the rollback is not committed")
        cb.commit()
        cb.waitUntilCompleted()
        try kv.beginStep(rows: 1, kind: .decode)
        #expect(kv.freePageCount == total - 1)
    }
}
