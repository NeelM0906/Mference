import Foundation
import Metal

extension Native {
    /// The W4 `KVStore`: paged FP16 KV (INT8 opt-in, D4), double-banked
    /// recurrent state, rollback, and a radix-tree prefix cache with RAM and
    /// SSD tiers (`docs/NATIVE_ENGINE.md`, "Buffer ownership and lifetimes").
    ///
    /// Memory is exactly the `MemoryPlan` (`PagedMemoryPlanner`):
    /// - one arena buffer of `contextPages + cachePages` physical pages, laid
    ///   out per `KVPageGeometry`. The live sequence uses at most
    ///   `contextPages`; the prefix cache keeps its KV pages *and* its
    ///   recurrent snapshots in the rest, so `kvArena + prefixCacheRAM` of the
    ///   plan is the whole arena and nothing is allocated after `init`;
    /// - two recurrent banks per GDN layer, the verify arena, and a two-slot
    ///   page-table ring.
    ///
    /// Sharing: a cached prefix's whole pages are shared, reference-counted,
    /// between the radix tree (`MemoryPrefixNode`) and the sequence that
    /// attached it. Writes only ever go to positions `>= length`, so a shared
    /// whole page is never written; the partial last page is copied on
    /// write, eagerly: `insertPrefix` gives the cache its own copy and
    /// `attachPrefix` gives the sequence its own copy, so neither side writes
    /// the other's page.
    ///
    /// Eviction: the cache may hold at most `cachePages` pages the sequence
    /// does not share. Past that (checked at `beginStep` and `insertPrefix`),
    /// and whenever a page is needed and none is free, the least recently
    /// used RAM prefix is spilled to the SSD tier (when configured, within its
    /// own byte budget) or dropped. Pages the sequence releases may hold the
    /// cache over its reserve until the next `beginStep` or `insertPrefix`;
    /// they are arena pages either way, so the footprint never exceeds the
    /// plan.
    ///
    /// Command-buffer safety: every page released by a call that takes a
    /// command buffer stays quarantined until that buffer completes, and a
    /// prefix is evicted or spilled only after the command buffer that filled
    /// it has completed. So a CPU-side SSD load never writes a page a pending
    /// GPU command still reads or writes.
    ///
    /// Not thread-safe: confined to the scheduler's generation thread.
    public final class PagedKVStore: KVStore {
        public let geometry: KVPageGeometry
        public let recurrentLayout: RecurrentLayout
        public let capacity: Int
        public private(set) var length = 0
        public let gdnLayerCount: Int
        public let verifySlots: Int
        /// Physical pages the sequence can use (`capacity` tokens).
        public let contextPages: Int
        /// Physical pages reserved for the prefix cache (`prefixCacheRAM`).
        public let cachePages: Int
        /// Bytes of one recurrent snapshot, and the arena pages it occupies.
        public let snapshotBytes: Int
        public let snapshotPages: Int

        let arena: MTLBuffer
        let pageTableRing: MTLBuffer
        let recurrent: MTLBuffer
        let verify: MTLBuffer

        private var pool: MemoryPagePool
        /// The sequence's physical pages in logical order.
        private var pages: [Int] = []
        /// Live recurrent bank L.
        private(set) var liveBank = 0
        private var stepCounter = 0
        private var ringSlot = 0
        private struct Step {
            var start: Int
            var rows: Int
            var input: Int
            var output: Int
            var kind: StepKind
        }
        private var lastStep: Step?

        private let root = MemoryPrefixNode(chunk: [], depth: 0, parent: nil)
        private var entries: [UInt64: MemoryPrefixEntry] = [:]
        private var nextPrefixID: UInt64 = 1
        private var clock: UInt64 = 0
        /// Quarantined pages and the command buffer they wait for.
        private var quarantine: [(commandBuffer: MTLCommandBuffer, pages: [Int])] = []
        private let ssd: MemorySSDTier?

        /// Cache activity counters (diagnostics and tests).
        public struct Counters: Sendable, Equatable {
            public var inserts = 0
            public var ramAttaches = 0
            public var ssdAttaches = 0
            public var copyOnWritePages = 0
            public var spills = 0
            public var drops = 0
            public var ssdDeletes = 0
        }
        public private(set) var counters = Counters()

        /// - Parameters:
        ///   - plan: a plan from `PagedMemoryPlanner` (its `kvArena`,
        ///     `prefixCacheRAM`, `contextLength`, `precision` and
        ///     `maxVerifyRows` size every buffer).
        ///   - ssdTier: where to spill evicted prefixes; nil keeps the cache
        ///     in RAM only.
        public init(config: Qwen35Config, plan: MemoryPlan, device: MTLDevice,
                    ssdTier: PrefixSSDTierConfiguration? = nil) throws {
            geometry = PagedMemoryPlanner.geometry(config: config, precision: plan.precision)
            recurrentLayout = PagedMemoryPlanner.recurrentLayout(config: config)
            capacity = plan.contextLength
            gdnLayerCount = config.gatedDeltaNetLayers.count
            verifySlots = max(1, plan.maxVerifyRows)
            contextPages = PagedMemoryPlanner.pages(tokens: capacity, geometry: geometry)
            cachePages = Int(plan.planned.prefixCacheRAM / UInt64(geometry.pageBytes))
            snapshotBytes = PagedMemoryPlanner.snapshotBytes(config: config)
            snapshotPages = PagedMemoryPlanner.snapshotPages(config: config, geometry: geometry)
            guard capacity > 0, geometry.attentionLayers > 0 else {
                throw NativeError.invalidShape("KV store needs a context and attention layers")
            }
            let arenaBytes = (contextPages + cachePages) * geometry.pageBytes
            guard arenaBytes <= device.maxBufferLength else {
                throw NativeError.resource("KV arena of \(arenaBytes) bytes exceeds maxBufferLength \(device.maxBufferLength)")
            }
            func make(_ bytes: Int, _ what: String) throws -> MTLBuffer {
                guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
                    throw NativeError.resource("cannot allocate \(what) (\(bytes) bytes)")
                }
                return b
            }
            arena = try make(arenaBytes, "KV arena")
            pageTableRing = try make(PagedMemoryPlanner.pageTableRingDepth * contextPages * 4, "page table ring")
            recurrent = try make(2 * gdnLayerCount * recurrentLayout.layerBytes, "recurrent banks")
            verify = try make(gdnLayerCount * verifySlots * recurrentLayout.layerBytes, "verify arena")
            memset(recurrent.contents(), 0, recurrent.length)
            pool = MemoryPagePool(total: contextPages + cachePages)
            let shape = MemorySSDTier.Shape(geometry: geometry, layout: recurrentLayout, gdnLayers: gdnLayerCount)
            ssd = try ssdTier.map { try MemorySSDTier(configuration: $0, shape: shape) }
        }

        // MARK: Steps

        public func beginStep(rows: Int, kind: StepKind) throws {
            guard rows >= 1 else { throw NativeError.invalidShape("beginStep rows \(rows)") }
            guard length + rows <= capacity else {
                throw NativeError.stateOutOfRange("step of \(rows) at \(length) exceeds capacity \(capacity)")
            }
            reclaimQuarantine(wait: false)
            let needed = PagedMemoryPlanner.pages(tokens: length + rows, geometry: geometry)
            var added: [Int] = []
            do {
                while pages.count + added.count < needed { added.append(try allocate(forSequence: true)) }
            } catch {
                for p in added { pool.sequenceRelease(p) }
                throw error
            }
            pages += added
            trimCache()
            ringSlot = stepCounter % PagedMemoryPlanner.pageTableRingDepth
            stepCounter += 1
            let table = pageTableRing.contents().advanced(by: ringSlot * contextPages * 4)
                .assumingMemoryBound(to: UInt32.self)
            for (i, p) in pages.enumerated() { table[i] = UInt32(p) }
            let input = liveBank, output = 1 - liveBank
            lastStep = Step(start: length, rows: rows, input: input, output: output, kind: kind)
            liveBank = output
            length += rows
        }

        public func pageTable() -> KVPageTable {
            KVPageTable(geometry: geometry, arena: arena, pageTable: pageTableRing,
                        pageTableOffset: ringSlot * contextPages * 4, pageCount: pages.count, tokenCount: length)
        }

        private func bankOffset(_ bank: Int, gdnLayer: Int) -> Int {
            (bank * gdnLayerCount + gdnLayer) * recurrentLayout.layerBytes
        }

        public func recurrentState(gdnLayer: Int) -> (input: RecurrentStateRef, output: RecurrentStateRef) {
            let (i, o) = lastStep.map { ($0.input, $0.output) } ?? (liveBank, 1 - liveBank)
            return (RecurrentStateRef(layout: recurrentLayout, buffer: recurrent, offset: bankOffset(i, gdnLayer: gdnLayer)),
                    RecurrentStateRef(layout: recurrentLayout, buffer: recurrent, offset: bankOffset(o, gdnLayer: gdnLayer)))
        }

        public func verifyArena(gdnLayer: Int) -> RecurrentVerifyArena {
            RecurrentVerifyArena(layout: recurrentLayout, buffer: verify,
                                 offset: gdnLayer * verifySlots * recurrentLayout.layerBytes,
                                 slotStride: recurrentLayout.layerBytes, slots: verifySlots)
        }

        public func rollback(to position: Int, commandBuffer: MTLCommandBuffer) throws {
            guard let s = lastStep else { throw NativeError.stateOutOfRange("no step to roll back") }
            if position == s.start {
                liveBank = s.input
            } else if position > s.start && position <= s.start + s.rows {
                guard s.kind == .verify || position == s.start + s.rows else {
                    throw NativeError.stateOutOfRange("rollback into a \(s.kind) step: only verify rows can be kept")
                }
                let r = position - s.start - 1
                if r < s.rows - 1 {
                    guard r < verifySlots else { throw NativeError.stateOutOfRange("verify slot \(r)") }
                    guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                        throw NativeError.resource("blit encoder")
                    }
                    for layer in 0..<gdnLayerCount {
                        blit.copy(from: verify, sourceOffset: (layer * verifySlots + r) * recurrentLayout.layerBytes,
                                  to: recurrent, destinationOffset: bankOffset(s.output, gdnLayer: layer),
                                  size: recurrentLayout.layerBytes)
                    }
                    blit.endEncoding()
                }
                liveBank = s.output
            } else {
                throw NativeError.stateOutOfRange(
                    "rollback to \(position): last step is \(s.start)..<\(s.start + s.rows)")
            }
            length = position
            releaseSequencePages(keeping: length, until: commandBuffer)
            lastStep = nil
        }

        public func reset(commandBuffer: MTLCommandBuffer) throws {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw NativeError.resource("blit encoder") }
            zeroLiveBank(blit)
            blit.endEncoding()
            length = 0
            releaseSequencePages(keeping: 0, until: commandBuffer)
            lastStep = nil
        }

        private func zeroLiveBank(_ blit: MTLBlitCommandEncoder) {
            guard snapshotBytes > 0 else { return }
            let start = bankOffset(liveBank, gdnLayer: 0)
            blit.fill(buffer: recurrent, range: start ..< start + snapshotBytes, value: 0)
        }

        // MARK: Pages

        /// One free page for the sequence or the cache, reclaiming quarantine
        /// and evicting the least recently used prefix when none is free.
        private func allocate(forSequence: Bool) throws -> Int {
            while true {
                if let p = forSequence ? pool.allocateForSequence() : pool.allocateForCache() { return p }
                if reclaimQuarantine(wait: true) { continue }
                guard evictOne() else {
                    throw NativeError.stateOutOfRange(
                        "KV arena full: \(pool.total) pages, \(pool.quarantinedCount) waiting on command buffers, "
                            + "no evictable prefix")
                }
            }
        }

        /// Releases the sequence's pages past `tokens`, quarantined until
        /// `commandBuffer` completes.
        private func releaseSequencePages(keeping tokens: Int, until commandBuffer: MTLCommandBuffer) {
            let keep = PagedMemoryPlanner.pages(tokens: tokens, geometry: geometry)
            var held: [Int] = []
            while pages.count > keep {
                let p = pages.removeLast()
                if pool.sequenceRelease(p, quarantine: true) { held.append(p) }
            }
            if !held.isEmpty { quarantine.append((commandBuffer, held)) }
        }

        /// Frees quarantined pages whose command buffer has completed; with
        /// `wait`, waits for committed ones. Returns true if any page was freed.
        @discardableResult
        private func reclaimQuarantine(wait: Bool) -> Bool {
            var freed = false
            quarantine.removeAll { item in
                switch item.commandBuffer.status {
                case .completed, .error:
                    break
                case .committed, .scheduled:
                    guard wait else { return false }
                    item.commandBuffer.waitUntilCompleted()
                default:
                    return false
                }
                for p in item.pages { pool.releaseQuarantined(p) }
                freed = freed || !item.pages.isEmpty
                return true
            }
            return freed
        }

        private func tick() -> UInt64 {
            clock += 1
            return clock
        }

        // MARK: Prefix cache

        public func lookupPrefix(_ tokens: [Int32]) -> PrefixMatch? {
            let tpp = geometry.tokensPerPage
            var node = root
            var best: MemoryPrefixEntry?
            while true {
                let start = node.depth * tpp
                for e in node.entries where !e.fillFailed {
                    let tail = e.tail(tokensPerPage: tpp)
                    guard tail.count <= tokens.count - start,
                          tokens[start ..< start + tail.count].elementsEqual(tail) else { continue }
                    if e.length > (best?.length ?? 0) { best = e }
                }
                let end = start + tpp
                guard end <= tokens.count, let child = node.children[Array(tokens[start ..< end])] else { break }
                node = child
            }
            return best.map { PrefixMatch(handle: $0.handle, length: $0.length, tier: $0.isRAM ? .ram : .ssd) }
        }

        /// The entry for exactly `tokens`, if cached.
        private func entry(exactly tokens: [Int32]) -> MemoryPrefixEntry? {
            let tpp = geometry.tokensPerPage
            var node = root
            for i in 0 ..< tokens.count / tpp {
                guard let child = node.children[Array(tokens[i * tpp ..< (i + 1) * tpp])] else { return nil }
                node = child
            }
            return node.entries.first { $0.tokens == tokens }
        }

        public func attachPrefix(_ match: PrefixMatch, commandBuffer: MTLCommandBuffer) throws {
            reclaimQuarantine(wait: false)
            guard let entry = entries[match.handle.id], entry.length == match.length else {
                throw NativeError.stateOutOfRange("unknown prefix \(match.handle.id) of \(match.length) tokens")
            }
            if entry.settle().failed {
                remove(entry)
                throw NativeError.stateOutOfRange("prefix \(match.handle.id): its snapshot command buffer failed")
            }
            guard entry.length <= capacity else {
                throw NativeError.stateOutOfRange("prefix of \(entry.length) tokens exceeds capacity \(capacity)")
            }
            entry.pinned = true
            defer { entry.pinned = false }
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw NativeError.resource("blit encoder") }
            defer { blit.endEncoding() }
            switch entry.storage {
            case .ram(let tailPage, let snapshot):
                // The entry's pages may be read by the sequence's old pages'
                // pending work only through GPU order, so no wait is needed.
                releaseSequencePages(keeping: 0, until: commandBuffer)
                lastStep = nil
                for node in entry.node.path {
                    let p = node.page!
                    pool.sequenceAcquire(p)
                    pages.append(p)
                }
                if let tailPage {
                    let p: Int
                    do {
                        p = try allocate(forSequence: true)
                    } catch {
                        failAttach(blit, until: commandBuffer)
                        throw error
                    }
                    blit.copy(from: arena, sourceOffset: tailPage * geometry.pageBytes, to: arena,
                              destinationOffset: p * geometry.pageBytes, size: geometry.pageBytes)
                    pages.append(p)
                    counters.copyOnWritePages += 1
                }
                copySnapshot(fromPages: snapshot, toLiveBank: blit)
                counters.ramAttaches += 1
            case .ssd(let url, _):
                guard let ssd else { throw NativeError.stateOutOfRange("SSD prefix without an SSD tier") }
                let header: MemorySSDTier.Header
                do {
                    header = try ssd.readHeader(url, expectedTokens: entry.tokens)
                } catch {
                    remove(entry)
                    throw NativeError.integrity("prefix \(match.handle.id): \(error)")
                }
                releaseSequencePages(keeping: 0, until: commandBuffer)
                lastStep = nil
                var loaded: [Int] = []
                var staging: [Int] = []
                do {
                    for _ in 0..<header.kvPages { loaded.append(try allocate(forSequence: true)) }
                    for _ in 0..<snapshotPages { staging.append(try allocate(forSequence: false)) }
                    try ssd.readData(url, header: header, into: loaded.map(pageChunk) + snapshotChunks(staging))
                } catch {
                    for p in loaded { pool.sequenceRelease(p) }
                    for p in staging { pool.cacheRelease(p) }
                    if case NativeError.stateOutOfRange = error {} else { remove(entry) }
                    failAttach(blit, until: commandBuffer)
                    if let e = error as? NativeError { throw e }
                    throw NativeError.integrity("prefix \(match.handle.id): \(error)")
                }
                pages = loaded
                copySnapshot(fromPages: staging, toLiveBank: blit)
                var held: [Int] = []
                for p in staging where pool.cacheRelease(p, quarantine: true) { held.append(p) }
                if !held.isEmpty { quarantine.append((commandBuffer, held)) }
                counters.ssdAttaches += 1
            }
            length = entry.length
            lastStep = nil
            entry.lastUse = tick()
        }

        /// An attach that cannot complete leaves an empty sequence with a zero bank.
        private func failAttach(_ blit: MTLBlitCommandEncoder, until commandBuffer: MTLCommandBuffer) {
            releaseSequencePages(keeping: 0, until: commandBuffer)
            zeroLiveBank(blit)
            length = 0
            lastStep = nil
        }

        private func pageChunk(_ p: Int) -> MemorySSDTier.Chunk {
            MemorySSDTier.Chunk(base: arena.contents().advanced(by: p * geometry.pageBytes), count: geometry.pageBytes)
        }

        /// Page-sized chunks of a snapshot stored in `pages`.
        private func snapshotChunks(_ pages: [Int]) -> [MemorySSDTier.Chunk] {
            pages.enumerated().map { j, p in
                MemorySSDTier.Chunk(base: arena.contents().advanced(by: p * geometry.pageBytes),
                                    count: min(geometry.pageBytes, snapshotBytes - j * geometry.pageBytes))
            }
        }

        private func copySnapshot(fromPages snapshot: [Int], toLiveBank blit: MTLBlitCommandEncoder) {
            let base = bankOffset(liveBank, gdnLayer: 0)
            for (j, p) in snapshot.enumerated() {
                blit.copy(from: arena, sourceOffset: p * geometry.pageBytes, to: recurrent,
                          destinationOffset: base + j * geometry.pageBytes,
                          size: min(geometry.pageBytes, snapshotBytes - j * geometry.pageBytes))
            }
        }

        public func insertPrefix(_ tokens: [Int32], commandBuffer: MTLCommandBuffer) throws -> PrefixHandle {
            guard tokens.count == length else {
                throw NativeError.invalidShape("prefix of \(tokens.count) tokens, sequence holds \(length)")
            }
            guard length > 0 else { throw NativeError.invalidShape("empty prefix") }
            reclaimQuarantine(wait: false)
            let tpp = geometry.tokensPerPage
            let wholePages = length / tpp
            let hasTail = length % tpp != 0
            let privatePages = (hasTail ? 1 : 0) + snapshotPages
            guard privatePages <= cachePages else {
                throw NativeError.memoryBudgetExceeded(required: UInt64(privatePages * geometry.pageBytes),
                                                       budget: UInt64(cachePages * geometry.pageBytes))
            }
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw NativeError.resource("blit encoder") }
            defer { blit.endEncoding() }
            if let old = entry(exactly: tokens) { remove(old) }
            var fresh: [Int] = []
            do {
                for _ in 0..<privatePages { fresh.append(try allocate(forSequence: false)) }
            } catch {
                for p in fresh { pool.cacheRelease(p) }
                throw error
            }
            let tailPage = hasTail ? fresh[0] : nil
            let snapshot = Array(fresh.dropFirst(hasTail ? 1 : 0))
            var node = root
            for i in 0..<wholePages {
                let chunk = Array(tokens[i * tpp ..< (i + 1) * tpp])
                if let child = node.children[chunk] {
                    node = child
                } else {
                    let child = MemoryPrefixNode(chunk: chunk, depth: i + 1, parent: node)
                    node.children[chunk] = child
                    node = child
                }
                if node.page == nil {
                    pool.cacheRetain(pages[i])
                    node.page = pages[i]
                }
            }
            let handle = PrefixHandle(id: nextPrefixID)
            nextPrefixID += 1
            let entry = MemoryPrefixEntry(handle: handle, tokens: tokens, node: node,
                                          storage: .ram(tailPage: tailPage, snapshotPages: snapshot),
                                          lastUse: tick(), fill: commandBuffer)
            var n: MemoryPrefixNode? = node
            while let x = n {
                x.ramEntriesBelow += 1
                x.entriesBelow += 1
                n = x.parent
            }
            node.entries.append(entry)
            entries[handle.id] = entry
            if let tailPage {
                blit.copy(from: arena, sourceOffset: pages[wholePages] * geometry.pageBytes, to: arena,
                          destinationOffset: tailPage * geometry.pageBytes, size: geometry.pageBytes)
            }
            let base = bankOffset(liveBank, gdnLayer: 0)
            for (j, p) in snapshot.enumerated() {
                blit.copy(from: recurrent, sourceOffset: base + j * geometry.pageBytes, to: arena,
                          destinationOffset: p * geometry.pageBytes,
                          size: min(geometry.pageBytes, snapshotBytes - j * geometry.pageBytes))
            }
            lastStep = nil
            counters.inserts += 1
            trimCache()
            return handle
        }

        // MARK: Eviction

        /// Evicts least recently used RAM prefixes while the cache holds more
        /// unshared pages than its reserve.
        private func trimCache() {
            while pool.cacheOnlyCount > cachePages, evictOne() {}
        }

        /// Spills (or drops) the least recently used evictable RAM prefix.
        /// Returns false when none is evictable.
        private func evictOne() -> Bool {
            var victim: MemoryPrefixEntry?
            for e in entries.values where e.isRAM && !e.pinned {
                let state = e.settle()
                if state.failed {
                    remove(e)
                    return true
                }
                guard state.sealed else { continue }
                if victim == nil || e.lastUse < victim!.lastUse { victim = e }
            }
            guard let victim else { return false }
            if ssd != nil {
                do {
                    try spill(victim)
                    return true
                } catch {}
            }
            remove(victim)
            counters.drops += 1
            return true
        }

        /// Moves a sealed RAM prefix to the SSD tier.
        private func spill(_ entry: MemoryPrefixEntry) throws {
            guard let ssd, case .ram(let tailPage, let snapshot) = entry.storage else { return }
            let kv = entry.node.path.map { $0.page! } + (tailPage.map { [$0] } ?? [])
            let bytes = ssd.fileBytes(length: entry.length, kvPages: kv.count, snapshotBytes: snapshotBytes,
                                      chunks: kv.count + snapshot.count)
            guard bytes <= ssd.budgetBytes else { throw NativeError.memoryBudgetExceeded(required: bytes, budget: ssd.budgetBytes) }
            while ssd.usedBytes + bytes > ssd.budgetBytes {
                let onDisk = entries.values.filter { !$0.isRAM && !$0.pinned }
                guard let oldest = onDisk.min(by: { $0.lastUse < $1.lastUse }) else {
                    throw NativeError.memoryBudgetExceeded(required: ssd.usedBytes + bytes, budget: ssd.budgetBytes)
                }
                remove(oldest)
                counters.ssdDeletes += 1
            }
            let (url, written) = try ssd.write(id: entry.handle.id, tokens: entry.tokens, kv: kv.map(pageChunk),
                                               snapshot: snapshotChunks(snapshot))
            releaseRAM(entry, until: nil)
            entry.storage = .ssd(url: url, bytes: written)
            counters.spills += 1
        }

        /// Drops a RAM prefix's page references. Its private pages are
        /// quarantined until `until` (its own fill when it has not completed).
        private func releaseRAM(_ entry: MemoryPrefixEntry, until: MTLCommandBuffer?) {
            guard case .ram(let tailPage, let snapshot) = entry.storage else { return }
            var held: [Int] = []
            for p in (tailPage.map { [$0] } ?? []) + snapshot where pool.cacheRelease(p, quarantine: until != nil) {
                held.append(p)
            }
            if let until, !held.isEmpty { quarantine.append((until, held)) }
            var n: MemoryPrefixNode? = entry.node
            while let x = n {
                x.ramEntriesBelow -= 1
                if x.ramEntriesBelow == 0, let p = x.page {
                    pool.cacheRelease(p)
                    x.page = nil
                }
                n = x.parent
            }
            entry.storage = .ram(tailPage: nil, snapshotPages: [])
        }

        /// Removes a prefix from the cache, any tier.
        private func remove(_ entry: MemoryPrefixEntry) {
            switch entry.storage {
            case .ram:
                let sealed = entry.settle().sealed
                releaseRAM(entry, until: sealed ? nil : entry.fill)
            case .ssd(let url, let bytes):
                ssd?.remove(url, bytes: bytes)
                // Not RAM: the RAM counts were already dropped at spill.
            }
            entries[entry.handle.id] = nil
            entry.node.entries.removeAll { $0 === entry }
            var n: MemoryPrefixNode? = entry.node
            while let x = n {
                x.entriesBelow -= 1
                n = x.parent
            }
            var node = entry.node
            while node.depth > 0, node.entriesBelow == 0, let parent = node.parent {
                precondition(node.page == nil, "pruned node still holds page")
                parent.children[node.chunk] = nil
                node = parent
            }
        }

        // MARK: Maintenance and accounting

        /// Spills every evictable RAM prefix to the SSD tier (memory pressure,
        /// tests). Requires no step in flight. Returns how many moved.
        @discardableResult
        public func spillPrefixCacheToSSD() -> Int {
            guard ssd != nil else { return 0 }
            reclaimQuarantine(wait: false)
            var moved = 0
            for e in entries.values.sorted(by: { $0.lastUse < $1.lastUse }) where e.isRAM && !e.pinned {
                guard e.settle().sealed, !e.fillFailed, (try? spill(e)) != nil else { continue }
                moved += 1
            }
            return moved
        }

        /// Drops every cached prefix, both tiers. Requires no step in flight.
        public func removeAllPrefixes() {
            for e in entries.values { remove(e) }
            reclaimQuarantine(wait: true)
        }

        /// Cached prefixes (RAM, SSD).
        public var cachedPrefixCount: (ram: Int, ssd: Int) {
            let ram = entries.values.filter(\.isRAM).count
            return (ram, entries.count - ram)
        }

        /// Bytes of SSD snapshot files.
        public var ssdBytes: UInt64 { ssd?.usedBytes ?? 0 }

        /// Pages neither held nor quarantined.
        public var freePageCount: Int { pool.freeCount }

        /// Physical page ids of the sequence, in logical order.
        public var physicalPages: [Int] { pages }

        /// Bytes of the arena buffer (`kvArena + prefixCacheRAM` of the plan).
        public var arenaBytes: Int { arena.length }

        /// Live accounting: `kvArena` is the sequence's share of the arena,
        /// `prefixCacheRAM` the pages only the cache holds right now (KV and
        /// snapshots), `other` the page-table ring. `weightsMapped` and
        /// `activations` are not the store's (zero).
        public var footprint: MemoryPlan.Footprint {
            MemoryPlan.Footprint(kvArena: UInt64(contextPages * geometry.pageBytes),
                                 recurrentState: UInt64(recurrent.length), verifyArena: UInt64(verify.length),
                                 prefixCacheRAM: UInt64(pool.cacheOnlyCount * geometry.pageBytes),
                                 other: UInt64(pageTableRing.length))
        }
    }
}
