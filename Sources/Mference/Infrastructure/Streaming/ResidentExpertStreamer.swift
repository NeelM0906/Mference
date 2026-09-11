import Darwin
import Foundation
import Metal

/// All-resident routed-expert backend. Maps the entire layer file once and
/// wraps each expert blob in its own `MTLBuffer` over the shared mapping. It
/// also exposes one complete layer view to checkpoint-specialized GPU routing;
/// only explicit resident mode uses that view. Per-expert views remain the
/// compatibility path for callers that already resolved routes on the CPU.
public final class ResidentExpertStreamer: @unchecked Sendable {

    /// Owns the `mmap` region; captured by every expert buffer's deallocator
    /// so the mapping outlives the last outstanding `MTLBuffer`.
    private final class Mapping: @unchecked Sendable {
        let base: UnsafeMutableRawPointer
        let length: Int
        init(base: UnsafeMutableRawPointer, length: Int) {
            self.base = base
            self.length = length
        }
        deinit { munmap(base, length) }
    }

    public let layout: StreamLayout
    private let mapping: Mapping
    /// Byte shift from the page-aligned mapping start to `streamOffset`.
    private let sliceShift: Int
    /// Per-expert buffer plus the expert's byte offset within it (non-zero
    /// only when the expert's file offset is not page-aligned).
    private let expertViews: [(buffer: MTLBuffer, offset: UInt64)]
    /// One contiguous view used only by checkpoint-specialized GPU routing.
    /// Binding this view makes the complete layer visible to Metal, which is
    /// appropriate for explicit resident mode but deliberately never exposed
    /// by the bounded-memory backend.
    private let layerBuffer: MTLBuffer
    /// Identity expert-to-slot table consumed by the existing GPU slot lookup.
    /// `Int16` matches the bounded slot map and comfortably covers 512 experts.
    private let identitySlotTable: MTLBuffer

    public init(layout: StreamLayout, device: MTLDevice) throws {
        self.layout = layout
        let pageSize = Int(getpagesize())

        let fd = open(layout.path, O_RDONLY)
        guard fd >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        defer { close(fd) }

        var fileStats = stat()
        if fstat(fd, &fileStats) == 0 {
            let required = layout.streamOffset + layout.streamSize
            if UInt64(fileStats.st_size) < required {
                throw StreamerError.sizeMismatch(
                    expected: required,
                    actual: UInt64(fileStats.st_size))
            }
        }

        let alignedOffset = (layout.streamOffset / UInt64(pageSize))
            * UInt64(pageSize)
        let shift = Int(layout.streamOffset - alignedOffset)
        let mappedLength = shift + Int(layout.streamSize)
        let mapped = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE,
                          fd, off_t(alignedOffset))
        guard let base = mapped, base != MAP_FAILED else {
            throw StreamerError.allocFailed(errno: errno)
        }
        let mapping = Mapping(base: base, length: mappedLength)
        self.mapping = mapping
        self.sliceShift = shift

        guard layout.expertsPerLayer <= Int(Int16.max),
              let layerBuffer = device.makeBuffer(
                bytesNoCopy: base,
                length: mappedLength,
                options: .storageModeShared,
                deallocator: { _, _ in _ = mapping }),
              let identitySlotTable = device.makeBuffer(
                length: layout.expertsPerLayer * MemoryLayout<Int16>.stride,
                options: .storageModeShared) else {
            throw StreamerError.bufferWrapFailed
        }
        let identity = identitySlotTable.contents().bindMemory(
            to: Int16.self, capacity: layout.expertsPerLayer)
        for expert in 0..<layout.expertsPerLayer {
            identity[expert] = Int16(expert)
        }
        self.layerBuffer = layerBuffer
        self.identitySlotTable = identitySlotTable

        var views: [(buffer: MTLBuffer, offset: UInt64)] = []
        views.reserveCapacity(layout.expertsPerLayer)
        for expert in 0..<layout.expertsPerLayer {
            let regionOffset = layout.expertOffset(layer: 0, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else {
                throw StreamerError.offsetOutOfRange(regionOffset)
            }
            // Buffers must start page-aligned: align down and carry the
            // remainder as an in-buffer offset.
            let absolute = shift + Int(regionOffset)
            let alignedStart = (absolute / pageSize) * pageSize
            let delta = absolute - alignedStart
            let rawLength = delta + Int(layout.expertStride)
            let bufferLength = ((rawLength + pageSize - 1) / pageSize) * pageSize
            nonisolated(unsafe) let start = base.advanced(by: alignedStart)
            guard let buffer = device.makeBuffer(
                bytesNoCopy: start,
                length: min(bufferLength, mappedLength - alignedStart),
                options: .storageModeShared,
                deallocator: { _, _ in _ = mapping })
            else {
                throw StreamerError.bufferWrapFailed
            }
            views.append((buffer: buffer, offset: UInt64(delta)))
        }
        self.expertViews = views
    }

    /// A direct expert-indexed slab for a resident GPU path. The fast address
    /// calculation is valid only when the file stores experts contiguously at
    /// `expert * expertStride` from a page-aligned stream start.
    public var contiguousSlabBinding:
        (slab: MTLBuffer, table: MTLBuffer, expertStride: Int)? {
        guard sliceShift == 0,
              layout.expertStride <= UInt64(UInt32.max),
              layout.expertOffsets?.enumerated().allSatisfy({ index, offset in
                  offset == UInt64(index) * layout.expertStride
              }) ?? true else {
            return nil
        }
        return (layerBuffer, identitySlotTable, Int(layout.expertStride))
    }

    public func expertBuffer(layer _: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard expert >= 0, expert < expertViews.count else {
            throw StreamerError.slotOutOfRange(expert)
        }
        let view = expertViews[expert]
        return (view.buffer, view.offset, layout.expertStride)
    }

    /// Touch the mapping sequentially so first-token decode does not pay
    /// the page-in cost. Called at load time; counts as model load, not
    /// decode.
    public func warmUp() {
        let pageSize = Int(getpagesize())
        var checksum: UInt8 = 0
        var offset = sliceShift
        let end = sliceShift + Int(layout.streamSize)
        while offset < end {
            checksum ^= mapping.base.load(fromByteOffset: offset, as: UInt8.self)
            offset += pageSize
        }
        _ = checksum
    }
}
