import Foundation
import Darwin
import Metal

/// Page geometry for the paged full-attention KV store. Pages are fixed at
/// 64 tokens; per layer a page is one K block and one V block of
/// `tokensPerPage * tokenStrideBytes` each.
public struct KVPageGeometry: Sendable, Equatable {
    /// Fixed page size. 64 tokens * 2048 B/token = 128 KiB per K (or V) page
    /// for Qwen 3.8 — the measured NVMe random-read sweet spot when K and V
    /// are fetched together (256 KiB at ~2.3 GiB/s).
    public static let tokensPerPage = 64

    /// Absolute layer indices with full attention (mask value 1), in order.
    public let fullLayerOrdinals: [Int]
    /// Bytes per token per layer for one of K or V:
    /// `numFullKVHeads * fullHeadDim * sizeof(FP16)`.
    public let tokenStrideBytes: Int
    public let maxContext: Int
    public let numKVHeads: Int
    public let headDim: Int

    public var pagesPerLayer: Int {
        (maxContext + Self.tokensPerPage - 1) / Self.tokensPerPage
    }
    /// Bytes of one K (or V) page for one layer.
    public var kPageBytes: Int { Self.tokensPerPage * tokenStrideBytes }

    /// Spill-file offset of a page's K block. The V block follows at
    /// `+kPageBytes`. Layer-major so a layer's pages stream sequentially.
    public func fileOffset(layerOrdinal: Int, pageIndex: Int) -> Int {
        (layerOrdinal * pagesPerLayer + pageIndex) * 2 * kPageBytes
    }

    /// Quest metadata per page: element-wise min and max over the page's K
    /// rows, per kv-head — 2 * numKVHeads * headDim FP16 values.
    public var metadataBytesPerPage: Int { 2 * numKVHeads * headDim * 2 }

    public func metadataOffset(layerOrdinal: Int, pageIndex: Int) -> Int {
        (layerOrdinal * pagesPerLayer + pageIndex) * metadataBytesPerPage
    }
}

public enum KVPageStoreError: Error, Equatable {
    case allocationFailed(String)
    case poolExhausted(layer: Int, pageIndex: Int)
    case spillFileFailed(String)
    case ioFailed(operation: String, errno: Int32)
    case notAFullAttentionLayer(Int)
    case pageNotSealed(layer: Int, pageIndex: Int)
}

/// Paged FP16 K/V storage for full-attention layers with an SSD spill tier.
///
/// Each full-attention layer owns a K pool and a V pool of
/// `poolPagesPerLayer` page slots. Pages seal on 64-token boundaries as the
/// position cursor advances; sealed pages are written behind to a sparse
/// layer-major spill file and become evictable under LRU pressure. Unsealed
/// (tail) pages and explicitly pinned pages never evict. Fetching a spilled
/// page is a single 2x128 KiB `pread` pair into a free or victim slot.
///
/// Mirrors `KVCacheManager`'s `kSlot`/`vSlot`/`advance`/`reset` surface for
/// the layers it owns; linear/GDN layers remain the runner's concern.
public final class KVPageStore {
    public let geometry: KVPageGeometry
    public let spillFileName = "kvpages.spill"
    /// Per-page Quest min/max metadata, shared storage so score kernels read
    /// it directly and seal kernels write it in the token command buffer.
    public let metadataBuffer: MTLBuffer

    public private(set) var position: Int = 0
    public private(set) var sealedPageCount: Int = 0
    /// True while every page has only ever landed at slot == pageIndex and
    /// nothing has been evicted — the pool region [0, position) is then one
    /// linear cache and the fast (tensor-ops) prefill path applies.
    public private(set) var identityMappingIntact = true

    private let poolPagesPerLayer: Int
    private let kPools: [MTLBuffer]           // [fullLayerOrdinal]
    private let vPools: [MTLBuffer]
    private let ordinalByLayer: [Int: Int]

    private enum PageState: UInt8 { case untouched, unsealed, sealed }

    private struct PageKey: Hashable {
        let ordinal: Int
        let pageIndex: Int
    }

    // All parallel arrays are indexed [fullLayerOrdinal][...].
    private var pageSlot: [[Int32]]           // [ordinal][pageIndex] -> slot or -1
    private var slotPage: [[Int32]]           // [ordinal][slot] -> pageIndex or -1
    private var pageState: [[PageState]]
    /// (ordinal, pageIndex) pairs queued for write-behind but possibly not on
    /// disk yet. Caller-thread-only; the spill queue never touches it — a
    /// `flushSpills()` barrier is the one synchronization point, after which
    /// every queued write has landed and the set empties.
    private var pendingSpills = Set<PageKey>()
    private var pagePinned: [[Bool]]
    private var slotLastUse: [[UInt64]]
    private var useTick: UInt64 = 0

    private let spillFD: Int32
    private let spillQueue = DispatchQueue(label: "mference.kvpage.spill")
    private let spillDirectory: URL
    /// First write-behind failure (ENOSPC, short write, I/O error), recorded
    /// on the spill queue and surfaced by the next spill-file read — pages
    /// past a failed spill are unreliable on disk, so the run must fail
    /// cleanly instead of fetching garbage. A separate box because spill
    /// closures must not retain the store: its deinit synchronizes on the
    /// spill queue, and releasing the last reference on that queue would
    /// deadlock.
    private final class SpillErrorBox {
        private let lock = NSLock()
        private var error: KVPageStoreError?
        func record(_ newError: KVPageStoreError) {
            lock.lock()
            defer { lock.unlock() }
            if error == nil { error = newError }
        }
        var value: KVPageStoreError? {
            lock.lock()
            defer { lock.unlock() }
            return error
        }
        func clear() {
            lock.lock()
            defer { lock.unlock() }
            error = nil
        }
    }
    private let spillError = SpillErrorBox()

    public init(device: MTLDevice,
                config: ArchConfig,
                maxContext: Int,
                poolPagesPerLayer: Int,
                spillDirectory: URL) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(poolPagesPerLayer > 0, "poolPagesPerLayer must be positive")

        let ordinals = config.fullAttentionLayerMask.enumerated()
            .filter { $0.element == 1 }.map(\.offset)
        precondition(!ordinals.isEmpty, "config has no full-attention layers")

        let stride = config.numFullKVHeads * config.fullHeadDim * 2
        self.geometry = KVPageGeometry(fullLayerOrdinals: ordinals,
                                       tokenStrideBytes: stride,
                                       maxContext: maxContext,
                                       numKVHeads: config.numFullKVHeads,
                                       headDim: config.fullHeadDim)
        self.poolPagesPerLayer = poolPagesPerLayer
        self.ordinalByLayer = Dictionary(uniqueKeysWithValues:
            ordinals.enumerated().map { ($0.element, $0.offset) })

        let poolBytes = poolPagesPerLayer * geometry.kPageBytes
        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        ks.reserveCapacity(ordinals.count)
        vs.reserveCapacity(ordinals.count)
        for layer in ordinals {
            guard let k = device.makeBuffer(length: poolBytes, options: .storageModeShared),
                  let v = device.makeBuffer(length: poolBytes, options: .storageModeShared) else {
                throw KVPageStoreError.allocationFailed("kv page pool layer \(layer)")
            }
            k.label = "kvpage.K.layer\(layer)"
            v.label = "kvpage.V.layer\(layer)"
            ks.append(k)
            vs.append(v)
        }
        self.kPools = ks
        self.vPools = vs

        let metadataLength = ordinals.count * geometry.pagesPerLayer
            * geometry.metadataBytesPerPage
        guard let metadata = device.makeBuffer(length: metadataLength,
                                               options: .storageModeShared) else {
            throw KVPageStoreError.allocationFailed("kv page metadata")
        }
        metadata.label = "kvpage.metadata"
        self.metadataBuffer = metadata

        let n = ordinals.count
        let pages = geometry.pagesPerLayer
        self.pageSlot = Array(repeating: Array(repeating: -1, count: pages), count: n)
        self.slotPage = Array(repeating: Array(repeating: -1, count: poolPagesPerLayer), count: n)
        self.pageState = Array(repeating: Array(repeating: .untouched, count: pages), count: n)
        self.pagePinned = Array(repeating: Array(repeating: false, count: pages), count: n)
        self.slotLastUse = Array(repeating: Array(repeating: 0, count: poolPagesPerLayer), count: n)

        self.spillDirectory = spillDirectory
        let path = spillDirectory.appendingPathComponent(spillFileName).path
        let fd = open(path, O_RDWR | O_CREAT, 0o600)
        guard fd >= 0 else {
            throw KVPageStoreError.spillFileFailed("open(\(path)) errno \(errno)")
        }
        // Sparse-extend to full capacity; APFS allocates blocks only on write.
        let fullSize = off_t(ordinals.count * geometry.pagesPerLayer * 2 * geometry.kPageBytes)
        guard ftruncate(fd, fullSize) == 0 else {
            close(fd)
            throw KVPageStoreError.spillFileFailed("ftruncate errno \(errno)")
        }
        // The pool is the cache; keep spill I/O out of the unified page cache.
        _ = fcntl(fd, F_NOCACHE, 1)
        self.spillFD = fd
    }

    deinit {
        spillQueue.sync {}
        close(spillFD)
        try? FileManager.default.removeItem(
            at: spillDirectory.appendingPathComponent(spillFileName))
    }

    // MARK: - Layer mapping

    public func fullLayerOrdinal(forLayer layer: Int) -> Int? { ordinalByLayer[layer] }

    public func kPoolBuffer(layer: Int) -> MTLBuffer { kPools[requireOrdinal(layer)] }
    public func vPoolBuffer(layer: Int) -> MTLBuffer { vPools[requireOrdinal(layer)] }

    // MARK: - Unsealed writes (decode/prefill hot path)

    /// Write target for `layer`'s K projection at `position`. Allocates the
    /// page slot on first touch (possibly evicting an LRU sealed page).
    public func kSlot(layer: Int, position: Int) throws -> (buffer: MTLBuffer, offset: Int) {
        let (ordinal, slot, within) = try writableSlot(layer: layer, position: position)
        return (kPools[ordinal], slot * geometry.kPageBytes + within * geometry.tokenStrideBytes)
    }

    /// Write target for `layer`'s V projection at `position`. Same slot index
    /// as K, in the distinct V pool.
    public func vSlot(layer: Int, position: Int) throws -> (buffer: MTLBuffer, offset: Int) {
        let (ordinal, slot, within) = try writableSlot(layer: layer, position: position)
        return (vPools[ordinal], slot * geometry.kPageBytes + within * geometry.tokenStrideBytes)
    }

    private func writableSlot(layer: Int,
                              position: Int) throws -> (ordinal: Int, slot: Int, within: Int) {
        precondition(position >= 0 && position < geometry.maxContext,
                     "position \(position) out of range 0..<\(geometry.maxContext)")
        let ordinal = requireOrdinal(layer)
        let pageIndex = position / KVPageGeometry.tokensPerPage
        precondition(pageState[ordinal][pageIndex] != .sealed,
                     "write to sealed page \(pageIndex) of layer \(layer)")
        let slot = try residentSlot(ordinal: ordinal, layer: layer, pageIndex: pageIndex,
                                    allocateAs: .unsealed)
        return (ordinal, slot, position % KVPageGeometry.tokensPerPage)
    }

    /// Contiguous K range for a prefill chunk write, valid under the
    /// identity slot mapping (pool sized to the full context, no evictions):
    /// pages are allocated sequentially so page `i` sits at slot `i` and a
    /// multi-page span is one linear region of the pool. The beyond-RAM
    /// prefill path (blocked attention) replaces this.
    public func contiguousKRange(layer: Int, start: Int,
                                 count: Int) throws -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        let ordinal = try contiguousRangeOrdinal(layer: layer, start: start, count: count)
        return (kPools[ordinal], start * geometry.tokenStrideBytes, geometry.tokenStrideBytes)
    }

    public func contiguousVRange(layer: Int, start: Int,
                                 count: Int) throws -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        let ordinal = try contiguousRangeOrdinal(layer: layer, start: start, count: count)
        return (vPools[ordinal], start * geometry.tokenStrideBytes, geometry.tokenStrideBytes)
    }

    private func contiguousRangeOrdinal(layer: Int, start: Int, count: Int) throws -> Int {
        precondition(count > 0 && start >= 0 && start + count <= geometry.maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext")
        let ordinal = requireOrdinal(layer)
        let firstPage = start / KVPageGeometry.tokensPerPage
        let lastPage = (start + count - 1) / KVPageGeometry.tokensPerPage
        for page in firstPage...lastPage {
            let slot = try residentSlot(ordinal: ordinal, layer: layer, pageIndex: page,
                                        allocateAs: .unsealed)
            precondition(slot == page,
                         "contiguous KV range requires the identity slot mapping "
                         + "(page \(page) at slot \(slot)); use the blocked prefill path")
        }
        return ordinal
    }

    // MARK: - Position / sealing

    /// Advance the position cursor. Every page fully crossed by the cursor is
    /// sealed across all full-attention layers and queued for write-behind.
    public func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= geometry.maxContext,
                     "advance would exceed maxContext")
        let sealedBefore = position / KVPageGeometry.tokensPerPage
        position += count
        let sealedAfter = position / KVPageGeometry.tokensPerPage
        for pageIndex in sealedBefore..<sealedAfter {
            seal(pageIndex: pageIndex)
        }
    }

    public func advance() { advance(by: 1) }

    /// Rewind the cursor after a rejected speculative span. Pages the cursor
    /// backs out of revert to unsealed — they are still resident (nothing
    /// allocates between the span's advance and its rewind) and their rows
    /// will simply be rewritten. Spills are flushed first so no in-flight
    /// write races the rows' reuse; a stale spill of an unsealed page is
    /// unreachable (fetches only read sealed pages) and is replaced when the
    /// page seals again.
    public func rewind(to newPosition: Int) {
        precondition(newPosition >= 0 && newPosition <= position,
                     "rewind target \(newPosition) outside 0...\(position)")
        let keepSealed = newPosition / KVPageGeometry.tokensPerPage
        let currentSealed = position / KVPageGeometry.tokensPerPage
        position = newPosition
        guard currentSealed > keepSealed else { return }
        flushSpills()
        for pageIndex in keepSealed..<currentSealed {
            var unsealedAny = false
            for ordinal in 0..<kPools.count {
                guard pageState[ordinal][pageIndex] == .sealed else { continue }
                precondition(pageSlot[ordinal][pageIndex] >= 0,
                             "rewind found an evicted in-span page")
                pageState[ordinal][pageIndex] = .unsealed
                unsealedAny = true
            }
            if unsealedAny { sealedPageCount -= 1 }
        }
    }

    private func seal(pageIndex: Int) {
        var sealedAny = false
        for ordinal in 0..<kPools.count {
            guard pageState[ordinal][pageIndex] == .unsealed else { continue }
            pageState[ordinal][pageIndex] = .sealed
            pendingSpills.insert(PageKey(ordinal: ordinal, pageIndex: pageIndex))
            sealedAny = true
            enqueueSpill(ordinal: ordinal, pageIndex: pageIndex)
        }
        if sealedAny { sealedPageCount += 1 }
    }

    /// Block until all queued write-behind spills have hit the file.
    public func flushSpills() {
        spillQueue.sync {}
        pendingSpills.removeAll(keepingCapacity: true)
    }

    // MARK: - Residency

    public func isResident(layer: Int, pageIndex: Int) -> Bool {
        pageSlot[requireOrdinal(layer)][pageIndex] >= 0
    }

    public func residentPageCount(layer: Int) -> Int {
        slotPage[requireOrdinal(layer)].lazy.filter { $0 >= 0 }.count
    }

    /// Fetch `pageIndex` into the pool if spilled; returns its slot. Sealed
    /// pages only — unsealed pages are always resident.
    @discardableResult
    public func ensureResident(layer: Int, pageIndex: Int) throws -> Int {
        let ordinal = requireOrdinal(layer)
        guard pageState[ordinal][pageIndex] != .untouched else {
            throw KVPageStoreError.pageNotSealed(layer: layer, pageIndex: pageIndex)
        }
        return try residentSlot(ordinal: ordinal, layer: layer, pageIndex: pageIndex,
                                allocateAs: nil)
    }

    public func pin(layer: Int, pageIndex: Int) {
        pagePinned[requireOrdinal(layer)][pageIndex] = true
    }

    public func unpin(layer: Int, pageIndex: Int) {
        pagePinned[requireOrdinal(layer)][pageIndex] = false
    }

    /// Pool-slot table for the paged attention kernel: one `uint32` slot per
    /// selected page, in the caller's (ascending-position) order. Fetches any
    /// spilled selection member. The selection must fit the pool.
    public func pageTable(layer: Int, selectedPages: [Int]) throws -> [UInt32] {
        var table = [UInt32]()
        table.reserveCapacity(selectedPages.count)
        for page in selectedPages {
            table.append(UInt32(try ensureResident(layer: layer, pageIndex: page)))
        }
        return table
    }

    // MARK: - Streamed window reads (blocked prefill)

    /// One sequential `pread` of `pageCount` interleaved [K page | V page]
    /// pairs starting at `firstPage` into `staging` — the blocked-prefill
    /// streaming path. Pages must be sealed and spilled (`flushSpills()`
    /// first). The flash kernel addresses the interleaved layout with a
    /// stride-2 page table and a V base offset of one K page.
    public func readSpilledSpan(layer: Int, firstPage: Int, pageCount: Int,
                                into staging: MTLBuffer) throws {
        precondition(pageCount > 0, "empty span read")
        try throwIfSpillFailed()
        let ordinal = requireOrdinal(layer)
        let bytes = pageCount * 2 * geometry.kPageBytes
        precondition(staging.length >= bytes, "staging buffer too small for span")
        precondition(firstPage + pageCount <= geometry.pagesPerLayer, "span out of range")
        let offset = off_t(geometry.fileOffset(layerOrdinal: ordinal, pageIndex: firstPage))
        var done = 0
        while done < bytes {
            let n = pread(spillFD, staging.contents() + done, bytes - done, offset + off_t(done))
            guard n > 0 else {
                throw KVPageStoreError.ioFailed(operation: "pread span", errno: errno)
            }
            done += n
        }
    }

    /// Current pool slot of a resident page (unsealed or fetched); the
    /// blocked-prefill tail window builds its page table from these.
    public func residentSlot(layer: Int, pageIndex: Int) -> Int? {
        let slot = pageSlot[requireOrdinal(layer)][pageIndex]
        return slot >= 0 ? Int(slot) : nil
    }

    // MARK: - Reset

    /// Drop all pages, rewind the cursor, and return pool pages to the OS.
    public func reset() {
        flushSpills()
        // A fresh run depends on no failed write: the file is re-punched
        // below and every page rewritten before it can be read again.
        spillError.clear()
        position = 0
        sealedPageCount = 0
        identityMappingIntact = true
        let pages = geometry.pagesPerLayer
        for ordinal in 0..<kPools.count {
            pageSlot[ordinal] = Array(repeating: -1, count: pages)
            slotPage[ordinal] = Array(repeating: -1, count: poolPagesPerLayer)
            pageState[ordinal] = Array(repeating: .untouched, count: pages)
            pagePinned[ordinal] = Array(repeating: false, count: pages)
            slotLastUse[ordinal] = Array(repeating: 0, count: poolPagesPerLayer)
        }
        useTick = 0
        let pageSize = Int(getpagesize())
        for buffer in kPools + vPools {
            let len = (buffer.length / pageSize) * pageSize
            if len > 0 { _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED) }
        }
        // Punch the file back to sparse.
        let fullSize = off_t(kPools.count * geometry.pagesPerLayer * 2 * geometry.kPageBytes)
        _ = ftruncate(spillFD, 0)
        _ = ftruncate(spillFD, fullSize)
    }

    // MARK: - Internals

    private func requireOrdinal(_ layer: Int) -> Int {
        guard let ordinal = ordinalByLayer[layer] else {
            preconditionFailure("layer \(layer) is not a full-attention layer")
        }
        return ordinal
    }

    /// Resolve (and if needed allocate or fetch) the slot for a page.
    /// `allocateAs == .unsealed` permits first-touch allocation of a fresh
    /// page; `nil` requires the page to exist already (fetch path).
    private func residentSlot(ordinal: Int, layer: Int, pageIndex: Int,
                              allocateAs: PageState?) throws -> Int {
        useTick += 1
        if pageSlot[ordinal][pageIndex] >= 0 {
            let slot = Int(pageSlot[ordinal][pageIndex])
            slotLastUse[ordinal][slot] = useTick
            return slot
        }
        let slot = try claimSlot(ordinal: ordinal, layer: layer, pageIndex: pageIndex)
        switch pageState[ordinal][pageIndex] {
        case .untouched:
            guard allocateAs == .unsealed else {
                throw KVPageStoreError.pageNotSealed(layer: layer, pageIndex: pageIndex)
            }
            pageState[ordinal][pageIndex] = .unsealed
        case .sealed:
            try fetch(ordinal: ordinal, pageIndex: pageIndex, slot: slot)
        case .unsealed:
            preconditionFailure("unsealed page \(pageIndex) lost residency")
        }
        pageSlot[ordinal][pageIndex] = Int32(slot)
        slotPage[ordinal][slot] = Int32(pageIndex)
        slotLastUse[ordinal][slot] = useTick
        return slot
    }

    private func claimSlot(ordinal: Int, layer: Int, pageIndex: Int) throws -> Int {
        if let free = slotPage[ordinal].firstIndex(of: -1) {
            if free != pageIndex { identityMappingIntact = false }
            return free
        }
        identityMappingIntact = false

        var victim = -1
        var victimUse = UInt64.max
        for slot in 0..<poolPagesPerLayer {
            let page = Int(slotPage[ordinal][slot])
            guard pageState[ordinal][page] == .sealed,
                  !pagePinned[ordinal][page],
                  slotLastUse[ordinal][slot] < victimUse else { continue }
            victim = slot
            victimUse = slotLastUse[ordinal][slot]
        }
        guard victim >= 0 else {
            throw KVPageStoreError.poolExhausted(layer: layer, pageIndex: pageIndex)
        }
        let victimPage = Int(slotPage[ordinal][victim])
        if pendingSpills.contains(PageKey(ordinal: ordinal, pageIndex: victimPage)) {
            // Write-behind has not landed yet; barrier so the eviction cannot
            // outrun its own spill.
            flushSpills()
        }
        pageSlot[ordinal][victimPage] = -1
        slotPage[ordinal][victim] = -1
        return victim
    }

    private func enqueueSpill(ordinal: Int, pageIndex: Int) {
        let slot = Int(pageSlot[ordinal][pageIndex])
        precondition(slot >= 0, "sealing a non-resident page")
        let pageBytes = geometry.kPageBytes
        let kSrc = kPools[ordinal].contents() + slot * pageBytes
        let vSrc = vPools[ordinal].contents() + slot * pageBytes
        let offset = off_t(geometry.fileOffset(layerOrdinal: ordinal, pageIndex: pageIndex))
        let fd = spillFD
        spillQueue.async { [box = spillError] in
            // The slot cannot be reused while its spill is pending (eviction
            // barriers on this queue first), so the pointers stay valid.
            do {
                try Self.writeFully(fd: fd, from: kSrc, count: pageBytes,
                                    offset: offset)
                try Self.writeFully(fd: fd, from: vSrc, count: pageBytes,
                                    offset: offset + off_t(pageBytes))
            } catch let error as KVPageStoreError {
                box.record(error)
            } catch {
                box.record(.ioFailed(operation: "pwrite spill", errno: EIO))
            }
        }
    }

    /// `pwrite` until `count` bytes land, resuming short writes and EINTR.
    static func writeFully(fd: Int32, from source: UnsafeRawPointer,
                           count: Int, offset: off_t) throws {
        var done = 0
        while done < count {
            let n = pwrite(fd, source + done, count - done, offset + off_t(done))
            guard n > 0 else {
                if n < 0 && errno == EINTR { continue }
                throw KVPageStoreError.ioFailed(operation: "pwrite spill",
                                                errno: n < 0 ? errno : ENOSPC)
            }
            done += n
        }
    }

    func recordSpillError(_ error: KVPageStoreError) {
        spillError.record(error)
    }

    /// The first recorded write-behind failure, if any.
    public var spillFailure: KVPageStoreError? { spillError.value }

    private func throwIfSpillFailed() throws {
        if let error = spillFailure { throw error }
    }

    private func fetch(ordinal: Int, pageIndex: Int, slot: Int) throws {
        try throwIfSpillFailed()
        let pageBytes = geometry.kPageBytes
        let kDst = kPools[ordinal].contents() + slot * pageBytes
        let vDst = vPools[ordinal].contents() + slot * pageBytes
        let offset = off_t(geometry.fileOffset(layerOrdinal: ordinal, pageIndex: pageIndex))
        let readK = pread(spillFD, kDst, pageBytes, offset)
        guard readK == pageBytes else {
            throw KVPageStoreError.ioFailed(operation: "pread K", errno: errno)
        }
        let readV = pread(spillFD, vDst, pageBytes, offset + off_t(pageBytes))
        guard readV == pageBytes else {
            throw KVPageStoreError.ioFailed(operation: "pread V", errno: errno)
        }
    }
}
