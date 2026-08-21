import Darwin
import Foundation
import Metal

public struct ExpertIOAdviceResult: Sendable, Equatable {
    public let requested: Int
    public let failed: Int
    public let calls: Int
    public let bytes: UInt64
    public let skipped: Int
    public let maxCallNanos: UInt64

    public init(requested: Int,
                failed: Int,
                calls: Int? = nil,
                bytes: UInt64 = 0,
                skipped: Int = 0,
                maxCallNanos: UInt64 = 0) {
        self.requested = requested
        self.failed = failed
        self.calls = calls ?? requested
        self.bytes = bytes
        self.skipped = skipped
        self.maxCallNanos = maxCallNanos
    }

    public static func skipped(requested: Int, bytes: UInt64 = 0) -> ExpertIOAdviceResult {
        ExpertIOAdviceResult(requested: requested,
                             failed: 0,
                             calls: 0,
                             bytes: bytes,
                             skipped: requested)
    }

}

public struct ExpertCachePlan: Sendable, Equatable {
    public let experts: [Int]
    public let assignedSlots: [Int]
    public let misses: [Int]
    public let hits: Int

    public init(experts: [Int], assignedSlots: [Int], misses: [Int], hits: Int) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.misses = misses
        self.hits = hits
    }
}

public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
}

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }

    public let layout: StreamLayout
    public let slotCount: Int
    public let cachePolicy: ExpertCachePolicy

    private let fd: Int32
    private let slotPointers: [UnsafeMutableRawPointer]
    /// One contiguous wired allocation holding every slot; slot `n` lives at
    /// byte offset `n * slotAllocationSize`. A single buffer lets the GPU
    /// address slots by offset arithmetic — the substrate the GPU-resident
    /// slot map needs — and halves per-slot bookkeeping.
    private let slotSlabBuffer: MTLBuffer
    private let slotAllocationSize: Int
    /// GPU-resident slot map: 256 `Int16` entries, expert -> slot index or
    /// -1. Mirrors `slotExpert` exactly (updated under `cacheLock` at every
    /// mutation), so a lookup kernel can resolve routed experts to slab
    /// offsets without the CPU. Entries are valid only when the slot's
    /// content is published (reserved/in-flight slots read -1).
    private let slotOfBuffer: MTLBuffer
    private let slotOfPointer: UnsafeMutablePointer<Int16>

    private var nextSlot = 0
    private let cursorLock = NSLock()

    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var expertUseCount: [Int]
    /// Slots a speculative read is currently filling. Their contents are
    /// undefined until the read lands, so they are neither matchable as hits
    /// (`slotExpert` is cleared to -1) nor available as eviction victims.
    private var speculativeInFlight: [Bool]
    private var useClock = 0
    private let cacheLock = NSLock()

    public init(layout: StreamLayout,
                device: MTLDevice,
                slotCount: Int,
                cachePolicy: ExpertCachePolicy = .lfu) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        self.layout = layout
        self.slotCount = slotCount
        self.cachePolicy = cachePolicy
        let pageSize = Int(getpagesize())

        let openedFD = open(layout.path, O_RDONLY)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.fd = openedFD

        var fileStats = stat()
        if fstat(openedFD, &fileStats) == 0 {
            let required = layout.streamOffset + layout.streamSize
            if UInt64(fileStats.st_size) < required {
                close(openedFD)
                throw StreamerError.sizeMismatch(
                    expected: required,
                    actual: UInt64(fileStats.st_size))
            }
        }

        let allocationSize = ((Int(layout.expertStride) + pageSize - 1) / pageSize) * pageSize
        var slabRaw: UnsafeMutableRawPointer?
        let allocResult = posix_memalign(&slabRaw, Self.scratchAlignment,
                                         allocationSize * slotCount)
        guard allocResult == 0, let slab = slabRaw else {
            close(openedFD)
            throw StreamerError.allocFailed(errno: allocResult)
        }
        nonisolated(unsafe) let capturedSlab = slab
        guard let slabBuffer = device.makeBuffer(
            bytesNoCopy: slab,
            length: allocationSize * slotCount,
            options: .storageModeShared,
            deallocator: { _, _ in free(capturedSlab) })
        else {
            free(slab)
            close(openedFD)
            throw StreamerError.bufferWrapFailed
        }

        self.slotPointers = (0..<slotCount).map {
            slab.advanced(by: $0 * allocationSize)
        }
        self.slotSlabBuffer = slabBuffer
        self.slotAllocationSize = allocationSize
        let tableEntries = max(1, layout.expertsPerLayer)
        guard let table = device.makeBuffer(
            length: tableEntries * MemoryLayout<Int16>.stride,
            options: .storageModeShared) else {
            throw StreamerError.bufferWrapFailed
        }
        self.slotOfBuffer = table
        self.slotOfPointer = table.contents()
            .bindMemory(to: Int16.self, capacity: tableEntries)
        for index in 0..<tableEntries { self.slotOfPointer[index] = -1 }
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.speculativeInFlight = [Bool](repeating: false, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
    }

    deinit {
        close(fd)
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        cursorLock.lock()
        let slot = nextSlot
        nextSlot = (nextSlot + 1) % slotCount
        cursorLock.unlock()
        return try loadExpert(layer: layer, expert: expert, slot: slot)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        try readFull(
            into: slotPointers[slot],
            fileOffset: layout.streamOffset + regionOffset,
            count: Int(layout.expertStride))
        return (slotSlabBuffer, UInt64(slot * slotAllocationSize),
                layout.expertStride)
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    public func planExpertsCached(experts: [Int],
                                  avoidingSlots: Set<Int> = []) -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots) else {
            preconditionFailure("expert cache cannot place requested misses")
        }
        return plan
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            avoidingSlots: Set<Int> = []) -> ExpertCachePlan? {
        makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots)
    }

    private func makeExpertCachePlan(experts: [Int],
                                     avoidingSlots rawAvoidingSlots: Set<Int>) -> ExpertCachePlan? {
        precondition(experts.count <= slotCount,
                     "expert cache needs at least \(experts.count) slots")
        let avoidingSlots = Set(rawAvoidingSlots.filter { $0 >= 0 && $0 < slotCount })

        cacheLock.lock()
        defer { cacheLock.unlock() }

        let clock = useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)

        for index in experts.indices {
            for slot in 0..<slotCount
                where !reserved[slot] && slotExpert[slot] == experts[index] {
                assignedSlots[index] = slot
                reserved[slot] = true
                break
            }
        }
        for slot in avoidingSlots where !reserved[slot] {
            reserved[slot] = true
        }
        // A slot a speculative read is still filling must not be handed to the
        // real plan: its buffer is being written from another thread. Callers
        // join outstanding speculation before planning, so in practice this
        // loop reserves nothing — it is the backstop that keeps a skipped join
        // from turning into a data race.
        for slot in 0..<slotCount where speculativeInFlight[slot] && !reserved[slot] {
            reserved[slot] = true
        }

        let misses = experts.indices.filter { assignedSlots[$0] == -1 }
        let evictable = (0..<slotCount)
            .filter { !reserved[$0] }
            .sorted { shouldEvictSlot($0, before: $1) }
        guard misses.count <= evictable.count else { return nil }

        useClock = clock
        for expert in experts where expert >= 0 && expert < expertUseCount.count {
            expertUseCount[expert] &+= 1
        }
        for slot in assignedSlots where slot >= 0 {
            slotLastUse[slot] = clock
        }
        for (offset, index) in misses.enumerated() {
            let slot = evictable[offset]
            assignedSlots[index] = slot
            reserved[slot] = true
            setSlotExpert(slot, to: -1)
            slotLastUse[slot] = clock
        }

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            misses: misses,
            hits: experts.count - misses.count)
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        let missFileOffsets = try plan.misses.map { index in
            try fileOffsetForExpert(plan.experts[index])
        }
        let runs = Self.coalescedReadRuns(offsets: missFileOffsets,
                                          stride: layout.expertStride)
        let errorLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: runs.count) { runIndex in
            let run = runs[runIndex]
            let destinations = run.map {
                self.slotPointers[plan.assignedSlots[plan.misses[$0]]]
            }
            do {
                try self.readScattered(
                    into: destinations,
                    fileOffset: missFileOffsets[run[0]],
                    strideBytes: Int(self.layout.expertStride))
            } catch {
                errorLock.lock()
                if firstError == nil { firstError = error }
                errorLock.unlock()
            }
        }
        if let firstError { throw firstError }

        cacheLock.lock()
        for index in plan.misses {
            setSlotExpert(plan.assignedSlots[index], to: plan.experts[index])
        }
        cacheLock.unlock()

        return expertCachePlanBuffers(plan)
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            (slotSlabBuffer, UInt64(slot * slotAllocationSize),
             layout.expertStride)
        }
    }

    /// Read-only residency probe: which of `experts` are *not* currently in a
    /// slot, deduplicated and in request order. Unlike `planExpertsCached` this
    /// touches neither the LFU counters nor the use clock nor slot assignment,
    /// so a speculative caller cannot make a guessed expert look popular.
    public func nonResidentExperts(_ experts: [Int]) -> [Int] {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        var resident = Set<Int>()
        for slot in 0..<slotCount where slotExpert[slot] >= 0 {
            resident.insert(slotExpert[slot])
        }
        var seen = Set<Int>()
        return experts.filter { expert in
            expert >= 0 && !resident.contains(expert) && seen.insert(expert).inserted
        }
    }

    /// Reserves slots for a speculative read of `experts` (which the caller has
    /// already filtered through `nonResidentExperts`).
    ///
    /// Eviction protection, in order of importance:
    /// * slots holding one of the predicted experts are never victims;
    /// * slots another speculative read is filling are never victims;
    /// * `keepEvictable` slots are left untouched so the real plan that follows
    ///   always has room for its own misses — a wrong guess can slow the cache
    ///   down but can never make the next plan unplaceable.
    ///
    /// Victims are picked with the normal eviction order but *no* LFU or clock
    /// bookkeeping is written: an unconfirmed guess must not shift the policy.
    /// Reserved slots are marked empty for the duration of the read.
    public func reserveSpeculativeSlots(experts: [Int],
                                        keepEvictable: Int) -> [(expert: Int, slot: Int)] {
        guard !experts.isEmpty else { return [] }
        cacheLock.lock()
        defer { cacheLock.unlock() }

        let wanted = Set(experts)
        var reserved = [Bool](repeating: false, count: slotCount)
        for slot in 0..<slotCount
            where speculativeInFlight[slot] || wanted.contains(slotExpert[slot]) {
            reserved[slot] = true
        }
        let evictable = (0..<slotCount)
            .filter { !reserved[$0] }
            .sorted { shouldEvictSlot($0, before: $1) }
        let budget = min(experts.count, max(0, evictable.count - max(0, keepEvictable)))
        guard budget > 0 else { return [] }

        var reservation: [(expert: Int, slot: Int)] = []
        reservation.reserveCapacity(budget)
        for index in 0..<budget {
            let slot = evictable[index]
            speculativeInFlight[slot] = true
            setSlotExpert(slot, to: -1)
            reservation.append((experts[index], slot))
        }
        return reservation
    }

    /// Runs a reservation from `reserveSpeculativeSlots` and publishes the
    /// results. Slots whose read failed stay empty rather than claiming to hold
    /// an expert; the in-flight mark is always cleared. Returns the bytes read.
    @discardableResult
    public func executeSpeculativeReservation(
        _ reservation: [(expert: Int, slot: Int)]
    ) -> UInt64 {
        guard !reservation.isEmpty else { return 0 }
        let fileOffsets = reservation.map { entry in
            (try? fileOffsetForExpert(entry.expert)) ?? UInt64.max
        }
        let readable = reservation.indices.filter { fileOffsets[$0] != UInt64.max }
        let runs = Self.coalescedReadRuns(offsets: readable.map { fileOffsets[$0] },
                                          stride: layout.expertStride)
        let loadedLock = NSLock()
        nonisolated(unsafe) var loaded: [Int] = []
        DispatchQueue.concurrentPerform(iterations: runs.count) { runIndex in
            let entries = runs[runIndex].map { readable[$0] }
            let destinations = entries.map { self.slotPointers[reservation[$0].slot] }
            guard (try? self.readScattered(
                into: destinations,
                fileOffset: fileOffsets[entries[0]],
                strideBytes: Int(self.layout.expertStride))) != nil else { return }
            loadedLock.lock()
            loaded.append(contentsOf: entries)
            loadedLock.unlock()
        }

        cacheLock.lock()
        for index in loaded {
            setSlotExpert(reservation[index].slot, to: reservation[index].expert)
        }
        for entry in reservation {
            speculativeInFlight[entry.slot] = false
        }
        cacheLock.unlock()
        return UInt64(loaded.count) * layout.expertStride
    }

    /// Single point of mutation for slot ownership: keeps the CPU array and
    /// the GPU-visible table in lockstep. Callers hold `cacheLock`.
    private func setSlotExpert(_ slot: Int, to expert: Int) {
        let previous = slotExpert[slot]
        if previous >= 0, previous < layout.expertsPerLayer {
            slotOfPointer[previous] = -1
        }
        slotExpert[slot] = expert
        if expert >= 0, expert < layout.expertsPerLayer {
            slotOfPointer[expert] = Int16(slot)
        }
    }

    /// GPU bindings for the slot-map decode path: the slot slab, the
    /// expert->slot table, and the byte stride between slots.
    public var slotMapBinding: (slab: MTLBuffer, table: MTLBuffer, slotStride: Int) {
        (slotSlabBuffer, slotOfBuffer, slotAllocationSize)
    }

    /// All-hit fast path bookkeeping: bump the LFU counters and use clock
    /// exactly as `planExpertsCached` would, without planning, assigning, or
    /// reading anything. Keeps eviction quality identical when the runner
    /// skips the CPU plan on layers the GPU served entirely from cache.
    public func noteAllHitUse(experts: [Int]) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        useClock += 1
        for expert in experts where expert >= 0 && expert < expertUseCount.count {
            expertUseCount[expert] &+= 1
        }
        for expert in experts {
            for slot in 0..<slotCount where slotExpert[slot] == expert {
                slotLastUse[slot] = useClock
                break
            }
        }
    }

    /// Eager decode path: mark the plan's miss slots in-flight and fill them
    /// on a background queue, invoking `completion` when every read has
    /// landed and been published. The caller encodes GPU work against the
    /// slots immediately and gates it on an event its completion signals;
    /// the in-flight marks keep concurrent plans away exactly as the
    /// speculative path does.
    ///
    /// `completion(false)` means at least one read failed and its slot stays
    /// unpublished: the caller must still unblock its gated GPU work but must
    /// not use the results, because the failed slot holds stale bytes.
    public func beginAsyncFill(_ plan: ExpertCachePlan,
                               qos: DispatchQoS.QoSClass = .userInitiated,
                               completion: @escaping @Sendable (Bool) -> Void) {
        let pairs = plan.misses.map { (expert: plan.experts[$0],
                                       slot: plan.assignedSlots[$0]) }
        guard !pairs.isEmpty else {
            completion(true)
            return
        }
        cacheLock.lock()
        for pair in pairs { speculativeInFlight[pair.slot] = true }
        cacheLock.unlock()
        let expectedBytes = UInt64(pairs.count) * layout.expertStride
        DispatchQueue.global(qos: qos).async { [self] in
            let loadedBytes = executeSpeculativeReservation(pairs)
            completion(loadedBytes == expectedBytes)
        }
    }

    /// Test/diagnostic view of slot residency.
    public func residentExpertsSnapshot() -> [Int] {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return slotExpert
    }

    public func adviseExpertCachePlanMisses(_ plan: ExpertCachePlan) -> ExpertIOAdviceResult {
        let experts = plan.misses.map { plan.experts[$0] }
        return adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExperts(experts: [Int]) -> ExpertIOAdviceResult {
        adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExpertMisses(experts: [Int]) -> ExpertIOAdviceResult {
        cacheLock.lock()
        let misses = experts.filter { !slotExpert.contains($0) }
        cacheLock.unlock()
        return adviseRanges(expertAdviceRanges(experts: misses), requested: misses.count)
    }

    static func coalescedAdjacentAdviceRanges(_ ranges: [(offset: UInt64, count: UInt64)])
        -> [(offset: UInt64, count: UInt64)] {
        let sorted = ranges.filter { $0.count > 0 }.sorted {
            $0.offset == $1.offset ? $0.count < $1.count : $0.offset < $1.offset
        }
        var result: [(offset: UInt64, count: UInt64)] = []
        for range in sorted {
            guard var last = result.popLast() else {
                result.append(range)
                continue
            }
            let lastEnd = last.offset &+ last.count
            let rangeEnd = range.offset &+ range.count
            if range.offset <= lastEnd {
                last.count = max(lastEnd, rangeEnd) - last.offset
                result.append(last)
            } else {
                result.append(last)
                result.append(range)
            }
        }
        return result
    }

    private func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        if cachePolicy == .lru {
            return slotLastUse[lhs] < slotLastUse[rhs]
        }
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        let lhsCount = lhsExpert < expertUseCount.count ? expertUseCount[lhsExpert] : 0
        let rhsCount = rhsExpert < expertUseCount.count ? expertUseCount[rhsExpert] : 0
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    private func expertAdviceRanges(experts: [Int]) -> [(offset: UInt64, count: UInt64)] {
        experts.compactMap { expert in
            let regionOffset = layout.expertOffset(layer: 0, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else { return nil }
            return (layout.streamOffset + regionOffset, layout.expertStride)
        }
    }

    private func adviseRanges(_ ranges: [(offset: UInt64, count: UInt64)],
                              requested: Int) -> ExpertIOAdviceResult {
        let coalesced = Self.coalescedAdjacentAdviceRanges(ranges)
        var failed = 0
        var bytes: UInt64 = 0
        var maxCallNanos: UInt64 = 0
        for range in coalesced {
            let result = RDAdvice.call(fd: fd, offset: range.offset, byteCount: range.count)
            if !result.succeeded { failed += 1 }
            bytes &+= result.requestedBytes
            maxCallNanos = max(maxCallNanos, result.elapsedNanos)
        }
        return ExpertIOAdviceResult(
            requested: requested,
            failed: failed,
            calls: coalesced.count,
            bytes: bytes,
            maxCallNanos: maxCallNanos)
    }

    private func fileOffsetForExpert(_ expert: Int) throws -> UInt64 {
        let regionOffset = layout.expertOffset(layer: 0, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        return layout.streamOffset + regionOffset
    }

    /// Groups reads of `stride` bytes at `offsets` into runs that are exactly
    /// contiguous on disk, so each run can be fetched with one scattered
    /// `preadv` instead of one random `pread` per expert. Returns runs of
    /// indices into `offsets`, each run in ascending disk order. Duplicate
    /// offsets never share a run: their reads would overlap.
    static func coalescedReadRuns(offsets: [UInt64], stride: UInt64) -> [[Int]] {
        let sorted = offsets.indices.sorted { offsets[$0] < offsets[$1] }
        var runs: [[Int]] = []
        for index in sorted {
            if let last = runs.last?.last, offsets[index] == offsets[last] &+ stride {
                runs[runs.count - 1].append(index)
            } else {
                runs.append([index])
            }
        }
        return runs
    }

    /// Reads `destinations.count * strideBytes` contiguous file bytes starting
    /// at `fileOffset`, scattering `strideBytes` into each destination in
    /// order. Single-destination runs use the plain `pread` path.
    private func readScattered(into destinations: [UnsafeMutableRawPointer],
                               fileOffset: UInt64,
                               strideBytes: Int) throws {
        guard destinations.count > 1 else {
            return try readFull(into: destinations[0],
                                fileOffset: fileOffset,
                                count: strideBytes)
        }
        let total = destinations.count * strideBytes
        var filled = 0
        while filled < total {
            let startIndex = filled / strideBytes
            let within = filled % strideBytes
            var vectors = [iovec(
                iov_base: destinations[startIndex].advanced(by: within),
                iov_len: strideBytes - within)]
            for index in (startIndex + 1)..<destinations.count {
                vectors.append(iovec(iov_base: destinations[index],
                                     iov_len: strideBytes))
            }
            let readCount = vectors.withUnsafeBufferPointer { buffer in
                preadv(fd, buffer.baseAddress, Int32(buffer.count),
                       off_t(fileOffset) + off_t(filled))
            }
            if readCount < 0 {
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(total),
                                                 actual: UInt64(filled))
            }
            filled += readCount
        }
    }

    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int) throws {
        var filled = 0
        while filled < count {
            let readCount = pread(
                fd,
                destination.advanced(by: filled),
                count - filled,
                off_t(fileOffset) + off_t(filled))
            if readCount < 0 {
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
        }
    }
}
