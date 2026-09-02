import Darwin
import Foundation

/// Validated geometry of one PLE n-gram row pool.
///
/// The pool is the third instantiation of the fixed-aperture idea: weights
/// stream into expert slots, KV streams into pages, and here a 320-million-row
/// embedding table streams into a bounded row cache. Rows are addressed by
/// n-gram hash, a few per token, so the table never needs to be resident.
///
/// Addressing, from `manifest.plePool` and reproduced here so the row reader
/// is one division and two multiplies:
///
/// ```
/// row i of shard s -> shards[s].offset
///                     + (i / rowsPerBlock) * blockStride
///                     + (i % rowsPerBlock) * rowStride
/// ```
///
/// Rows are **not** individually page-aligned — at a 320-byte record that
/// would inflate the ~102 GB table to ~5 TB. Blocks are, and a record never
/// straddles a block, so one row costs one page fault and a cached page serves
/// `rowsPerBlock` neighbours.
public struct PleRowPoolGeometry: Sendable, Equatable {
    /// Zero-indexed layer this pool belongs to.
    public let layer: Int
    /// Pool file path relative to the install directory.
    public let file: String
    /// Total rows across every shard.
    public let rows: Int
    /// Row width in elements (160 in production: 16 n-gram heads x 160 = the
    /// 2560-wide PLE embedding is assembled from 16 separate rows).
    public let rowDim: Int
    /// `bf16` in production — group-64 quantization needs `rowDim % 64 == 0`,
    /// which 160 does not satisfy, so the pool stays BF16.
    public let storage: Storage
    /// Bytes of one complete row record, `[weights | scales | biases]`.
    public let rowStride: Int
    public let rowWeightBytes: Int
    public let rowScaleBytes: Int
    public let rowBiasBytes: Int
    /// Rows packed into one page-aligned block.
    public let rowsPerBlock: Int
    /// Bytes between block starts (one page).
    public let blockStride: Int
    public let fileSize: UInt64
    /// Per-shard regions, in shard order. Global rows are consecutive across
    /// them, so `firstRow` is the running prefix sum.
    public let shards: [Shard]

    public enum Storage: String, Sendable, Equatable {
        case bf16
        case int4AffineG64
    }

    public struct Shard: Sendable, Equatable {
        public let shard: Int
        public let rows: Int
        public let offset: UInt64
        public let size: UInt64
        /// Global index of this shard's first row.
        public let firstRow: Int
    }

    /// How many n-gram head rows are concatenated to form one PLE embedding.
    /// Derived rather than stored: the checkpoint publishes the embedding
    /// width (the model's `hiddenSize`) and the row width, and the installed
    /// `ngram_heads_offsets` / `ngram_heads_vocab_sizes` tables are exactly
    /// this long — `FlashNextResident` cross-checks them at load.
    public let ngramHeads: Int

    /// Validate `manifest.plePool.layers[]` against the arch and precompute the
    /// shard prefix sums. Every assumption the row reader makes is checked
    /// here, once, so a per-row read is pure arithmetic.
    public init(layer entry: ManifestPlePoolLayer, hiddenSize: Int) throws {
        func fail(_ detail: String) -> ModelError {
            .plePoolInvalid(detail: "layer \(entry.layer): \(detail)")
        }
        guard let storage = Storage(rawValue: entry.storage) else {
            throw fail("unknown storage \"\(entry.storage)\"")
        }
        guard entry.rowDim > 0, entry.rows > 0 else {
            throw fail("rowDim \(entry.rowDim) and rows \(entry.rows) must be positive")
        }
        switch storage {
        case .bf16:
            guard entry.weightBits == 16, entry.groupSize == 0,
                  entry.rowScaleBytes == 0, entry.rowBiasBytes == 0,
                  entry.rowWeightBytes == entry.rowDim * MemoryLayout<UInt16>.stride else {
                throw fail("bf16 pool must carry \(entry.rowDim * 2)-byte rows "
                    + "and no scale/bias companions")
            }
        case .int4AffineG64:
            guard entry.weightBits == 4,
                  entry.groupSize == Quantization.groupSize,
                  entry.rowDim % Quantization.groupSize == 0,
                  entry.rowWeightBytes == entry.rowDim / 2 else {
                throw fail("int4 pool geometry does not match group-"
                    + "\(Quantization.groupSize) affine rows")
            }
        }
        let recordBytes = entry.rowWeightBytes + entry.rowScaleBytes + entry.rowBiasBytes
        guard entry.rowStride == recordBytes else {
            throw fail("rowStride \(entry.rowStride) != record \(recordBytes)")
        }
        guard entry.rowsPerBlock > 0, entry.blockStride > 0,
              entry.rowsPerBlock * entry.rowStride <= entry.blockStride else {
            throw fail("\(entry.rowsPerBlock) rows of \(entry.rowStride) B do not "
                + "fit in a \(entry.blockStride) B block")
        }
        guard !entry.shards.isEmpty else { throw fail("no shard regions") }

        var shards: [Shard] = []
        shards.reserveCapacity(entry.shards.count)
        var firstRow = 0
        for (position, shard) in entry.shards.sorted(by: { $0.shard < $1.shard }).enumerated() {
            guard shard.shard == position else {
                throw fail("shard ids are not 0..<\(entry.shards.count)")
            }
            guard shard.rows > 0 else { throw fail("shard \(shard.shard) has no rows") }
            let blocks = (shard.rows + entry.rowsPerBlock - 1) / entry.rowsPerBlock
            let needed = UInt64(blocks) * UInt64(entry.blockStride)
            guard shard.size >= needed else {
                throw fail("shard \(shard.shard) region \(shard.size) B cannot hold "
                    + "\(shard.rows) rows (needs \(needed) B)")
            }
            guard shard.offset % UInt64(entry.blockStride) == 0 else {
                throw fail("shard \(shard.shard) offset \(shard.offset) is not "
                    + "block-aligned")
            }
            guard shard.offset <= entry.fileSize,
                  shard.size <= entry.fileSize - shard.offset else {
                throw fail("shard \(shard.shard) region runs past the \(entry.fileSize) B file")
            }
            shards.append(Shard(shard: shard.shard, rows: shard.rows,
                                offset: shard.offset, size: shard.size,
                                firstRow: firstRow))
            firstRow += shard.rows
        }
        guard firstRow == entry.rows else {
            throw fail("shard rows sum to \(firstRow), not \(entry.rows)")
        }
        guard hiddenSize > 0, hiddenSize % entry.rowDim == 0 else {
            throw fail("rowDim \(entry.rowDim) does not divide hidden \(hiddenSize)")
        }

        self.layer = entry.layer
        self.file = entry.file
        self.rows = entry.rows
        self.rowDim = entry.rowDim
        self.storage = storage
        self.rowStride = entry.rowStride
        self.rowWeightBytes = entry.rowWeightBytes
        self.rowScaleBytes = entry.rowScaleBytes
        self.rowBiasBytes = entry.rowBiasBytes
        self.rowsPerBlock = entry.rowsPerBlock
        self.blockStride = entry.blockStride
        self.fileSize = entry.fileSize
        self.shards = shards
        self.ngramHeads = hiddenSize / entry.rowDim
    }

    /// Byte offset of a global row within the pool file.
    public func fileOffset(row: Int) throws -> UInt64 {
        guard row >= 0, row < rows else {
            throw ModelError.plePoolRowOutOfRange(row: row, rows: rows)
        }
        let shard = shards[shardIndex(containing: row)]
        let local = row - shard.firstRow
        return shard.offset
            + UInt64(local / rowsPerBlock) * UInt64(blockStride)
            + UInt64(local % rowsPerBlock) * UInt64(rowStride)
    }

    /// Index of the shard owning a global row. Binary search over the prefix
    /// sums rather than a division: the last shard is short in general.
    func shardIndex(containing row: Int) -> Int {
        var low = 0
        var high = shards.count - 1
        while low < high {
            let mid = (low + high + 1) / 2
            if shards[mid].firstRow <= row { low = mid } else { high = mid - 1 }
        }
        return low
    }
}

/// Bounded LFU cache over PLE rows.
///
/// Follows the routed-expert slot cache's conventions: a fixed number of
/// equal-sized slots carved out of one contiguous allocation, eviction by use
/// count with a use-clock tiebreak, and no growth under load. The difference
/// is scale — thousands of slots rather than tens — so the eviction victim is
/// found through frequency buckets in O(1) instead of the expert cache's sort
/// over its handful of slots.
///
/// Ties *within* a frequency class are broken by whichever slot the bucket
/// happens to hold last; the order is deterministic for a given call sequence
/// but is not a documented LRU. Callers that need a specific victim should
/// raise the survivor's use count instead of relying on the tiebreak.
final class PleRowCache {
    let capacityRows: Int
    let rowStride: Int

    private let slab: UnsafeMutableRawPointer
    /// slot -> global row index, or -1 when free.
    private var slotRow: [Int]
    /// slot -> use count.
    private var slotUses: [Int]
    /// slot -> position within its frequency bucket.
    private var slotBucketPosition: [Int]
    /// use count -> slots holding rows used that many times.
    private var buckets: [Int: [Int]] = [:]
    private var minimumUses = 0
    private var rowSlot: [Int: Int] = [:]
    private var freeSlots: [Int]

    private(set) var hits = 0
    private(set) var misses = 0
    private(set) var evictions = 0

    init(capacityRows: Int, rowStride: Int) {
        precondition(capacityRows > 0, "PLE row cache needs at least one row")
        precondition(rowStride > 0, "PLE row stride must be positive")
        self.capacityRows = capacityRows
        self.rowStride = rowStride
        self.slab = UnsafeMutableRawPointer.allocate(
            byteCount: capacityRows * rowStride,
            alignment: MemoryLayout<UInt64>.alignment)
        self.slotRow = [Int](repeating: -1, count: capacityRows)
        self.slotUses = [Int](repeating: 0, count: capacityRows)
        self.slotBucketPosition = [Int](repeating: -1, count: capacityRows)
        self.freeSlots = Array((0..<capacityRows).reversed())
        rowSlot.reserveCapacity(capacityRows)
    }

    deinit { slab.deallocate() }

    /// Row count currently held.
    var count: Int { rowSlot.count }

    /// Bytes of one cached row, or nil on a miss. Recording the hit is the
    /// caller's business only in the sense that a hit bumps the use count.
    func withCachedRow<T>(_ row: Int, _ body: (UnsafeRawBufferPointer) -> T) -> T? {
        guard let slot = rowSlot[row] else {
            misses += 1
            return nil
        }
        hits += 1
        bumpUses(slot)
        return body(UnsafeRawBufferPointer(
            start: slab.advanced(by: slot * rowStride), count: rowStride))
    }

    /// Insert a row, evicting the least-frequently-used one when full.
    /// Returns the slot's storage so the caller can fill it in place.
    func insert(_ row: Int, _ fill: (UnsafeMutableRawBufferPointer) -> Void) {
        if let existing = rowSlot[row] {
            fill(UnsafeMutableRawBufferPointer(
                start: slab.advanced(by: existing * rowStride), count: rowStride))
            bumpUses(existing)
            return
        }
        let slot: Int
        if let free = freeSlots.popLast() {
            slot = free
        } else {
            slot = evictVictim()
            evictions += 1
        }
        fill(UnsafeMutableRawBufferPointer(
            start: slab.advanced(by: slot * rowStride), count: rowStride))
        slotRow[slot] = row
        rowSlot[row] = slot
        slotUses[slot] = 1
        addToBucket(slot, uses: 1)
        minimumUses = 1
    }

    /// Diagnostic view: which rows are resident, ascending.
    func residentRows() -> [Int] { rowSlot.keys.sorted() }

    /// Use count of a resident row, for eviction-policy tests.
    func useCount(of row: Int) -> Int? { rowSlot[row].map { slotUses[$0] } }

    private func bumpUses(_ slot: Int) {
        let uses = slotUses[slot]
        removeFromBucket(slot, uses: uses)
        slotUses[slot] = uses + 1
        addToBucket(slot, uses: uses + 1)
        if minimumUses == uses, buckets[uses] == nil { minimumUses = uses + 1 }
    }

    private func addToBucket(_ slot: Int, uses: Int) {
        slotBucketPosition[slot] = buckets[uses]?.count ?? 0
        buckets[uses, default: []].append(slot)
    }

    /// Swap-remove: O(1), at the cost of any order within the frequency class.
    private func removeFromBucket(_ slot: Int, uses: Int) {
        guard var bucket = buckets[uses] else { return }
        let position = slotBucketPosition[slot]
        let last = bucket.count - 1
        if position != last {
            bucket[position] = bucket[last]
            slotBucketPosition[bucket[position]] = position
        }
        bucket.removeLast()
        slotBucketPosition[slot] = -1
        if bucket.isEmpty { buckets[uses] = nil } else { buckets[uses] = bucket }
    }

    private func evictVictim() -> Int {
        while buckets[minimumUses] == nil { minimumUses += 1 }
        let slot = buckets[minimumUses]!.last!
        removeFromBucket(slot, uses: minimumUses)
        rowSlot.removeValue(forKey: slotRow[slot])
        slotRow[slot] = -1
        slotUses[slot] = 0
        return slot
    }
}

/// `pread`-based reader over one PLE n-gram row pool, with a bounded LFU row
/// cache in front of it.
///
/// CPU-only by design at this stage: the rows a token needs are known only
/// after the n-gram hash, and the kernels that would consume them on the GPU
/// do not exist yet. When they land, the cache's slab becomes the upload
/// staging buffer exactly as `PreadExpertStreamer`'s slot slab does.
public final class PleRowPool: @unchecked Sendable {
    /// The only pool format this runtime reads. An install declaring anything
    /// else is refused rather than guessed at.
    public static let supportedKind = "rowLookupPoolV1"

    /// 65 536 rows is ~20 MB at the production 320-byte row: a modest default
    /// beside the expert slot cache's multiple GB, and roughly 4 000 tokens of
    /// distinct n-gram lookups at 16 rows per token.
    public static let defaultCacheRows = 65_536

    public let geometry: PleRowPoolGeometry
    public let path: String

    private let fd: Int32
    private let cache: PleRowCache
    private let lock = NSLock()

    /// Open the pool file described by `geometry`, relative to the install.
    public init(directoryURL: URL,
                geometry: PleRowPoolGeometry,
                cacheRows: Int = PleRowPool.defaultCacheRows) throws {
        let url = directoryURL.appendingPathComponent(geometry.file)
        let opened = open(url.path, O_RDONLY)
        guard opened >= 0 else {
            throw ModelError.posixFailed(call: "open(\(url.path))", errno: errno)
        }
        var fileStats = stat()
        if fstat(opened, &fileStats) == 0,
           UInt64(fileStats.st_size) < geometry.fileSize {
            close(opened)
            throw ModelError.plePoolInvalid(
                detail: "\(geometry.file) is \(fileStats.st_size) B; the manifest "
                    + "declares \(geometry.fileSize) B")
        }
        self.geometry = geometry
        self.path = url.path
        self.fd = opened
        self.cache = PleRowCache(capacityRows: cacheRows,
                                 rowStride: geometry.rowStride)
    }

    deinit { close(fd) }

    /// Raw bytes of one row record, `[weights | scales | biases]`. Served from
    /// the LFU cache when resident, otherwise read with one `pread` and
    /// inserted.
    public func readRowRecord(_ row: Int) throws -> [UInt8] {
        let offset = try geometry.fileOffset(row: row)
        lock.lock()
        defer { lock.unlock() }
        if let cached = cache.withCachedRow(row, { Array($0) }) { return cached }
        var bytes = [UInt8](repeating: 0, count: geometry.rowStride)
        try bytes.withUnsafeMutableBytes { raw in
            try Self.readFull(fd: fd, into: raw.baseAddress!,
                              fileOffset: offset, count: geometry.rowStride)
        }
        cache.insert(row) { destination in
            bytes.withUnsafeBytes { source in
                destination.baseAddress!.copyMemory(from: source.baseAddress!,
                                                    byteCount: geometry.rowStride)
            }
        }
        return bytes
    }

    /// One row widened to float. BF16 pools decode losslessly; an INT4 pool
    /// dequantizes its group-64 affine record.
    public func readRow(_ row: Int) throws -> [Float] {
        let record = try readRowRecord(row)
        switch geometry.storage {
        case .bf16:
            return (0..<geometry.rowDim).map { index in
                let lo = UInt16(record[index * 2])
                let hi = UInt16(record[index * 2 + 1]) << 8
                return Quantization.bf16ToFloat(lo | hi)
            }
        case .int4AffineG64:
            return Self.dequantizeInt4AffineRow(record, geometry: geometry)
        }
    }

    /// The `ngramHeads` rows a token's hashed ids select, concatenated into
    /// the PLE embedding the mixer consumes. Row order is head order.
    ///
    /// The caller computes the row indices. That hash has **zero integer
    /// headroom** — the mixed values are exactly 63 bits wide by construction
    /// — so it must be done in true 64-bit integer arithmetic: a `Double`
    /// intermediate (53-bit significand) or a 32-bit one silently selects a
    /// different row, and nothing downstream can detect it.
    public func readEmbedding(rows: [Int]) throws -> [Float] {
        guard rows.count == geometry.ngramHeads else {
            throw ModelError.plePoolInvalid(
                detail: "expected \(geometry.ngramHeads) n-gram head rows, got \(rows.count)")
        }
        var out: [Float] = []
        out.reserveCapacity(geometry.ngramHeads * geometry.rowDim)
        for row in rows { out.append(contentsOf: try readRow(row)) }
        return out
    }

    /// Cache counters, for tests and diagnostics.
    public var cacheStatistics: (hits: Int, misses: Int, evictions: Int, resident: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (cache.hits, cache.misses, cache.evictions, cache.count)
    }

    /// Test hook: use count of a cached row, or nil when it is not resident.
    func cachedUseCount(of row: Int) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        return cache.useCount(of: row)
    }

    /// Test hook: which rows the cache currently holds.
    func residentRows() -> [Int] {
        lock.lock()
        defer { lock.unlock() }
        return cache.residentRows()
    }

    private static func dequantizeInt4AffineRow(
        _ record: [UInt8], geometry: PleRowPoolGeometry
    ) -> [Float] {
        let groups = geometry.rowDim / Quantization.groupSize
        let scaleBase = geometry.rowWeightBytes
        let biasBase = scaleBase + geometry.rowScaleBytes
        func u16(_ base: Int, _ index: Int) -> UInt16 {
            UInt16(record[base + index * 2]) | (UInt16(record[base + index * 2 + 1]) << 8)
        }
        var out = [Float](repeating: 0, count: geometry.rowDim)
        for group in 0..<groups {
            let scale = Quantization.bf16ToFloat(u16(scaleBase, group))
            let bias = Quantization.bf16ToFloat(u16(biasBase, group))
            for lane in 0..<Quantization.groupSize {
                let index = group * Quantization.groupSize + lane
                let byte = record[index / 2]
                let nibble = index % 2 == 0 ? (byte & 0x0F) : (byte >> 4)
                out[index] = Float(nibble) * scale + bias
            }
        }
        return out
    }

    private static func readFull(fd: Int32,
                                 into destination: UnsafeMutableRawPointer,
                                 fileOffset: UInt64,
                                 count: Int) throws {
        var filled = 0
        while filled < count {
            let got = pread(fd, destination.advanced(by: filled), count - filled,
                            off_t(fileOffset) + off_t(filled))
            if got < 0 { throw ModelError.posixFailed(call: "pread", errno: errno) }
            if got == 0 {
                throw ModelError.plePoolInvalid(
                    detail: "short read at row offset \(fileOffset): "
                        + "\(filled)/\(count) bytes")
            }
            filled += got
        }
    }
}
