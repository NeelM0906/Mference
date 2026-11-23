import Foundation
import Darwin
import Metal

/// The short-convolution tails of one layer visit: FP32 `[3][channels]` each
/// (row 2 the newest pre-convolution input), packed in one block of the
/// state's tail buffer as keys, values, attention output, MoE output.
public struct Inkling2VisitTails {
    public static let keyOffset = 0
    public static let valueOffset = 3 * Inkling2AttentionShape.kvWidth
    public static let attentionOffset = valueOffset + 3 * Inkling2AttentionShape.kvWidth
    public static let moeOffset = attentionOffset + 3 * Inkling2AttentionShape.hidden
    /// Floats per visit (19,968).
    public static let floatsPerVisit = moeOffset + 3 * Inkling2AttentionShape.hidden

    public let buffer: MTLBuffer
    /// Byte offset of this visit's block.
    public let blockOffset: Int

    public var keys: Inkling2BufferRef { .init(buffer, offset: blockOffset + Self.keyOffset * 4) }
    public var values: Inkling2BufferRef { .init(buffer, offset: blockOffset + Self.valueOffset * 4) }
    public var attentionOutput: Inkling2BufferRef { .init(buffer, offset: blockOffset + Self.attentionOffset * 4) }
    public var moeOutput: Inkling2BufferRef { .init(buffer, offset: blockOffset + Self.moeOffset * 4) }

    /// `Inkling2ConvTail` runs that advance the key and value tails from the
    /// projection output, to encode with `tails: keys`.
    public func keyValueRuns(keyColumn: Int, valueColumn: Int) -> [Inkling2ConvTail.Run] {
        [.init(xColumn: keyColumn, tailOffset: 0, channels: Inkling2AttentionShape.kvWidth),
         .init(xColumn: valueColumn, tailOffset: Self.valueOffset - Self.keyOffset,
               channels: Inkling2AttentionShape.kvWidth)]
    }
}

/// What `Inkling2AttentionState.takeSnapshot` recorded; pass it back to restore.
public struct Inkling2StateSnapshot: Sendable, Equatable {
    public let position: Int
    public let includesGlobalRows: Bool
    public let byteCount: Int
    /// The state's rewind sequence when the snapshot was taken.
    let sequence: UInt64
}

/// KV and short-convolution state of all 48 Inkling 2 Mini layer visits.
///
/// - Window visits (40): an FP16 ring of `window + maxChunkTokens` slots for
///   keys and one for values, `[slot][8][128]`; position `p` at slot
///   `p % ringCapacity`, so a chunk's keys and the window behind it coexist.
/// - Global visits (8): linear FP16 storage, position `p` at slot `p`, over
///   address space reserved for `maxContext` and wrapped in steps, so pages
///   cost memory only once written and growing never copies.
/// - Convolution tails (48 x 4): one FP32 buffer, `Inkling2VisitTails`.
///
/// The owner encodes each chunk (or decode step) after `ensureCapacity`, and
/// calls `advance(by:)` for it. Snapshots copy the convolution tails, the last
/// `window - 1` rows of every ring and, optionally, the global rows; a
/// snapshot without global rows restores only while the global rows below
/// its position are untouched (the state tracks every rewind to check that).
/// Snapshots and restores copy on the CPU: drain GPU work first.
public final class Inkling2AttentionState {
    public struct Configuration: Sendable, Equatable {
        public var maxContext: Int
        public var maxChunkTokens: Int
        public var window: Int
        /// Global storage grows in steps of this many tokens.
        public var globalGrowthStep: Int

        public init(maxContext: Int, maxChunkTokens: Int,
                    window: Int = Inkling2AttentionShape.slidingWindow,
                    globalGrowthStep: Int = 16_384) {
            self.maxContext = maxContext
            self.maxChunkTokens = maxChunkTokens
            self.window = window
            self.globalGrowthStep = globalGrowthStep
        }
    }

    /// Bytes of one cached position of one visit's keys (or values): 8 x 128 halves.
    public static let rowBytes = Inkling2AttentionShape.kvWidth * 2
    static let headerBytes = 64
    static let magic: UInt64 = 0x5441_5453_324B_4E49   // "INK2STAT"
    static let version: UInt64 = 1
    static let rewindLogLimit = 4096

    public let configuration: Configuration
    public let ringCapacity: Int
    public private(set) var position = 0
    public private(set) var globalCapacity: Int
    public let tailBuffer: MTLBuffer

    private let device: MTLDevice
    private let ringKeys: [MTLBuffer]
    private let ringValues: [MTLBuffer]
    private var globalKeys: [MTLBuffer]
    private var globalValues: [MTLBuffer]
    private let globalKeyRegions: [Inkling2ReservedRegion]
    private let globalValueRegions: [Inkling2ReservedRegion]
    private let windowIndex: [Int?]
    private let globalIndex: [Int?]
    private var sequence: UInt64 = 0
    private var rewinds: [(sequence: UInt64, position: Int)] = []
    private var trimmedThrough: UInt64 = 0

    public init(device: MTLDevice, configuration: Configuration) throws {
        precondition(configuration.maxContext > 0 && configuration.maxChunkTokens > 0)
        precondition(configuration.window > 0 && configuration.globalGrowthStep > 0)
        self.device = device
        self.configuration = configuration
        ringCapacity = configuration.window + configuration.maxChunkTokens
        globalCapacity = min(configuration.maxContext, configuration.globalGrowthStep)

        var wIndex = [Int?](repeating: nil, count: Inkling2VisitPlan.visitCount)
        var gIndex = [Int?](repeating: nil, count: Inkling2VisitPlan.visitCount)
        for (i, slot) in Inkling2VisitPlan.windowSlots.enumerated() { wIndex[slot] = i }
        for (i, slot) in Inkling2VisitPlan.globalSlots.enumerated() { gIndex[slot] = i }
        windowIndex = wIndex
        globalIndex = gIndex

        func makeBuffer(_ length: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }
        var rk: [MTLBuffer] = [], rv: [MTLBuffer] = []
        for slot in Inkling2VisitPlan.windowSlots {
            rk.append(try makeBuffer(ringCapacity * Self.rowBytes, "ink2.kv.K.visit\(slot)"))
            rv.append(try makeBuffer(ringCapacity * Self.rowBytes, "ink2.kv.V.visit\(slot)"))
        }
        ringKeys = rk
        ringValues = rv

        var gk: [MTLBuffer] = [], gv: [MTLBuffer] = []
        var kr: [Inkling2ReservedRegion] = [], vr: [Inkling2ReservedRegion] = []
        for slot in Inkling2VisitPlan.globalSlots {
            guard let k = Inkling2ReservedRegion(bytes: configuration.maxContext * Self.rowBytes),
                  let v = Inkling2ReservedRegion(bytes: configuration.maxContext * Self.rowBytes),
                  let kb = k.wrap(globalCapacity * Self.rowBytes, device: device, label: "ink2.kv.K.visit\(slot)"),
                  let vb = v.wrap(globalCapacity * Self.rowBytes, device: device, label: "ink2.kv.V.visit\(slot)") else {
                throw ModelError.residentBufferWrapFailed
            }
            kr.append(k)
            vr.append(v)
            gk.append(kb)
            gv.append(vb)
        }
        globalKeyRegions = kr
        globalValueRegions = vr
        globalKeys = gk
        globalValues = gv
        tailBuffer = try makeBuffer(Inkling2VisitPlan.visitCount * Inkling2VisitTails.floatsPerVisit * 4,
                                    "ink2.conv-tails")
        memset(tailBuffer.contents(), 0, tailBuffer.length)
    }

    /// Bytes the rings and tails take, plus the global rows of `tokens` positions.
    public static func memoryBytes(configuration: Configuration, tokens: Int) -> Int {
        let rings = Inkling2VisitPlan.windowSlots.count * 2
            * (configuration.window + configuration.maxChunkTokens) * rowBytes
        let tails = Inkling2VisitPlan.visitCount * Inkling2VisitTails.floatsPerVisit * 4
        return rings + tails + Inkling2VisitPlan.globalSlots.count * 2 * min(tokens, configuration.maxContext) * rowBytes
    }

    // MARK: Views

    /// The cache of the visit in `slot` (`Inkling2VisitPlan.slot`). Global
    /// views wrap the current capacity: fetch them again after `ensureCapacity`.
    public func kv(slot: Int) -> Inkling2KVView {
        if let i = windowIndex[slot] {
            return Inkling2KVView(keys: .init(ringKeys[i]), values: .init(ringValues[i]), ringCapacity: ringCapacity)
        }
        let i = globalIndex[slot]!
        return Inkling2KVView(keys: .init(globalKeys[i]), values: .init(globalValues[i]), ringCapacity: 0)
    }

    public func tails(slot: Int) -> Inkling2VisitTails {
        precondition((0..<Inkling2VisitPlan.visitCount).contains(slot))
        return Inkling2VisitTails(buffer: tailBuffer, blockOffset: slot * Inkling2VisitTails.floatsPerVisit * 4)
    }

    public func mask(slot: Int) -> Inkling2AttentionMask {
        Inkling2VisitPlan.bySlot[slot].isGlobal ? .globalDefault : .window(configuration.window)
    }

    // MARK: Cursor

    /// Makes room in the global storage for positions `0 ..< tokens`; returns
    /// whether the buffers were re-wrapped (fetch views again).
    @discardableResult
    public func ensureCapacity(tokens: Int) throws -> Bool {
        guard tokens <= configuration.maxContext else {
            throw PrefixSnapshotError.outOfRange("\(tokens) tokens exceed maxContext \(configuration.maxContext)")
        }
        guard tokens > globalCapacity else { return false }
        let step = configuration.globalGrowthStep
        let target = min(configuration.maxContext, (tokens + step - 1) / step * step)
        var keys: [MTLBuffer] = [], values: [MTLBuffer] = []
        for (i, slot) in Inkling2VisitPlan.globalSlots.enumerated() {
            guard let k = globalKeyRegions[i].wrap(target * Self.rowBytes, device: device, label: "ink2.kv.K.visit\(slot)"),
                  let v = globalValueRegions[i].wrap(target * Self.rowBytes, device: device, label: "ink2.kv.V.visit\(slot)") else {
                throw ModelError.residentBufferWrapFailed
            }
            keys.append(k)
            values.append(v)
        }
        globalKeys = keys
        globalValues = values
        globalCapacity = target
        return true
    }

    /// Commits `tokens` positions written by the owner's last chunk or step.
    public func advance(by tokens: Int) {
        precondition(tokens > 0 && tokens <= configuration.maxChunkTokens)
        precondition(position + tokens <= globalCapacity, "ensureCapacity before writing past the global capacity")
        position += tokens
    }

    /// Back to an empty sequence: zero tails, cursor 0, global pages returned
    /// to the OS (their rows are rewritten before anything reads them).
    public func reset() {
        memset(tailBuffer.contents(), 0, tailBuffer.length)
        for region in globalKeyRegions + globalValueRegions { region.discard() }
        noteRewind(to: 0)
        position = 0
    }

    private func noteRewind(to target: Int) {
        sequence &+= 1
        rewinds.append((sequence, target))
        if rewinds.count > Self.rewindLogLimit {
            trimmedThrough = rewinds.removeFirst().sequence
        }
    }

    // MARK: Snapshots

    private func ringRows(at position: Int) -> Int { min(position, configuration.window - 1) }

    /// Bytes a snapshot at the current position takes.
    public func snapshotBytes(includeGlobalRows: Bool) -> Int {
        Self.headerBytes
            + Inkling2VisitPlan.visitCount * Inkling2VisitTails.floatsPerVisit * 4
            + Inkling2VisitPlan.windowSlots.count * 2 * ringRows(at: position) * Self.rowBytes
            + (includeGlobalRows ? Inkling2VisitPlan.globalSlots.count * 2 * position * Self.rowBytes : 0)
    }

    /// Copies the state at the current position into `region`. Leaves the
    /// state unchanged. The caller has drained GPU work.
    public func takeSnapshot(into region: PrefixSnapshotRegion, includeGlobalRows: Bool) throws -> Inkling2StateSnapshot {
        guard position > 0 else { throw PrefixSnapshotError.outOfRange("empty state") }
        let needed = snapshotBytes(includeGlobalRows: includeGlobalRows)
        guard region.byteCount >= needed else {
            throw PrefixSnapshotError.regionTooSmall(needed: needed, available: region.byteCount)
        }
        let header: [UInt64] = [Self.magic, Self.version, UInt64(position), UInt64(configuration.window),
                                includeGlobalRows ? 1 : 0, sequence, UInt64(needed), 0]
        header.withUnsafeBytes { _ = memcpy(region.base, $0.baseAddress!, Self.headerBytes) }
        var offset = Self.headerBytes
        memcpy(region.base + offset, tailBuffer.contents(), tailBuffer.length)
        offset += tailBuffer.length
        let rows = ringRows(at: position)
        for i in 0..<ringKeys.count {
            for buffer in [ringKeys[i], ringValues[i]] {
                for p in (position - rows)..<position {
                    memcpy(region.base + offset, buffer.contents() + (p % ringCapacity) * Self.rowBytes, Self.rowBytes)
                    offset += Self.rowBytes
                }
            }
        }
        if includeGlobalRows {
            for i in 0..<globalKeys.count {
                for buffer in [globalKeys[i], globalValues[i]] {
                    memcpy(region.base + offset, buffer.contents(), position * Self.rowBytes)
                    offset += position * Self.rowBytes
                }
            }
        }
        precondition(offset == needed)
        return Inkling2StateSnapshot(position: position, includesGlobalRows: includeGlobalRows,
                                     byteCount: needed, sequence: sequence)
    }

    /// Whether `snapshot` can be restored now: always with global rows;
    /// without them, only while no rewind since the snapshot went below its
    /// position.
    public func canRestore(_ snapshot: Inkling2StateSnapshot) -> Bool {
        if snapshot.includesGlobalRows { return true }
        guard snapshot.sequence >= trimmedThrough, snapshot.position <= globalCapacity else { return false }
        return !rewinds.contains { $0.sequence > snapshot.sequence && $0.position < snapshot.position }
    }

    /// Restores `snapshot` from `region` and moves the cursor to its
    /// position. The caller has drained GPU work. A throw before copying
    /// leaves the state unchanged; nothing throws after copying begins.
    public func restoreSnapshot(_ snapshot: Inkling2StateSnapshot, from region: PrefixSnapshotRegion) throws {
        guard canRestore(snapshot) else {
            throw PrefixSnapshotError.unsupported("global rows below position \(snapshot.position) were rewritten")
        }
        guard region.byteCount >= snapshot.byteCount else {
            throw PrefixSnapshotError.regionTooSmall(needed: snapshot.byteCount, available: region.byteCount)
        }
        var header = [UInt64](repeating: 0, count: 8)
        header.withUnsafeMutableBytes { _ = memcpy($0.baseAddress!, region.base, Self.headerBytes) }
        guard header[0] == Self.magic, header[1] == Self.version, header[2] == UInt64(snapshot.position),
              header[3] == UInt64(configuration.window), header[4] == (snapshot.includesGlobalRows ? 1 : 0),
              header[6] == UInt64(snapshot.byteCount) else {
            throw PrefixSnapshotError.unsupported("region does not hold this snapshot")
        }
        let target = snapshot.position
        if snapshot.includesGlobalRows { try ensureCapacity(tokens: target) }

        var offset = Self.headerBytes
        memcpy(tailBuffer.contents(), region.base + offset, tailBuffer.length)
        offset += tailBuffer.length
        let rows = ringRows(at: target)
        for i in 0..<ringKeys.count {
            for buffer in [ringKeys[i], ringValues[i]] {
                for p in (target - rows)..<target {
                    memcpy(buffer.contents() + (p % ringCapacity) * Self.rowBytes, region.base + offset, Self.rowBytes)
                    offset += Self.rowBytes
                }
            }
        }
        if snapshot.includesGlobalRows {
            for i in 0..<globalKeys.count {
                for buffer in [globalKeys[i], globalValues[i]] {
                    memcpy(buffer.contents(), region.base + offset, target * Self.rowBytes)
                    offset += target * Self.rowBytes
                }
            }
        }
        noteRewind(to: target)
        position = target
    }
}

/// Address space reserved once for one growable global K or V buffer, sized
/// for `maxContext`; pages cost memory only once written. Every Metal wrapper
/// holds a reference, so it is unmapped only after the last wrapper is released.
final class Inkling2ReservedRegion: @unchecked Sendable {
    let base: UnsafeMutableRawPointer
    let length: Int

    init?(bytes: Int) {
        let page = Int(getpagesize())
        let length = max(page, (bytes + page - 1) / page * page)
        let mapped = mmap(nil, length, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0)
        guard mapped != MAP_FAILED, let mapped else { return nil }
        self.base = mapped
        self.length = length
    }

    deinit { munmap(base, length) }

    func wrap(_ bytes: Int, device: MTLDevice, label: String) -> MTLBuffer? {
        let page = Int(getpagesize())
        let wrapped = min(length, max(page, (bytes + page - 1) / page * page))
        let buffer = device.makeBuffer(bytesNoCopy: base, length: wrapped, options: .storageModeShared,
                                       deallocator: { _, _ in _ = self })
        buffer?.label = label
        return buffer
    }

    /// Returns the pages to the OS. Their contents are undefined afterwards;
    /// nothing reads rows at or past the cursor.
    func discard() { _ = posix_madvise(base, length, POSIX_MADV_DONTNEED) }
}
