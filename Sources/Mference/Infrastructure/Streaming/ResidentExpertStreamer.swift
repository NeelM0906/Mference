import Darwin
import Foundation
import Metal

/// All-resident routed-expert backend. Maps the entire layer file once and
/// wraps each expert blob in its own `MTLBuffer` over the shared mapping.
/// Per-expert buffers matter: Metal makes a bound buffer resident in full,
/// so one buffer per layer would demand the whole file's pages every token,
/// while per-expert buffers demand only the routed experts' pages.
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

    /// One buffer over the whole mapping, for kernels that address an expert
    /// from a GPU-side index as `baseOffset + expert * expertStride`. Nil when
    /// the layer's experts are not laid out at a uniform stride.
    public struct SlabView {
        public let buffer: MTLBuffer
        /// Byte offset of expert 0 inside `buffer`.
        public let baseOffset: Int
        public let expertStride: Int
    }
    public let slabView: SlabView?

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

        // The slab covers every mapped page; `bytesNoCopy` wants a page
        // multiple, and mmap maps whole pages, so rounding up stays inside
        // the region.
        let uniform = (0..<layout.expertsPerLayer).allSatisfy {
            layout.expertOffset(layer: 0, expert: $0) == UInt64($0) * layout.expertStride
        }
        let slabLength = ((mappedLength + pageSize - 1) / pageSize) * pageSize
        nonisolated(unsafe) let slabBase = base
        if uniform, layout.expertsPerLayer > 0,
           let slab = device.makeBuffer(bytesNoCopy: slabBase, length: slabLength,
                                        options: .storageModeShared,
                                        deallocator: { _, _ in _ = mapping }) {
            self.slabView = SlabView(
                buffer: slab,
                baseOffset: shift + Int(layout.expertOffset(layer: 0, expert: 0)),
                expertStride: Int(layout.expertStride))
        } else {
            self.slabView = nil
        }
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
