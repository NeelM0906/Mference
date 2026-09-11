import Darwin
import Foundation
import Metal

/// All-resident routed-expert backend, in one of two strategies:
///
/// * **`.mapped`** (the original): `mmap` the layer file once and wrap each
///   expert blob in its own `MTLBuffer` over the shared mapping. Per-expert
///   buffers matter here: Metal makes a bound buffer resident in full, so one
///   buffer per layer would demand the whole file's pages every token, while
///   per-expert buffers demand only the routed experts' pages. The right
///   shape for a host whose page cache holds the pool comfortably.
/// * **`.copied`**: one anonymous shared `MTLBuffer` per layer, filled by
///   direct (`F_NOCACHE`) reads at open. The GPU addresses experts by offset
///   inside it, so a kernel can resolve a routed index to an expert with no
///   CPU round trip (`slabView`), and no page of it is ever faulted in by the
///   GPU. Measured 2026-09-11 on the 256 GB M3 Ultra with GLM-5.3-Flash's
///   171 GB expert set: the mapped strategy paged the set in through GPU
///   faults at ~0.7 GB/s (a first token per minute) and pushed ~140 GB of
///   incompressible pages into the compressor; the copied strategy reads the
///   set once at disk speed and holds it as ordinary process memory.
public final class ResidentExpertStreamer: @unchecked Sendable {

    public enum Strategy: Sendable {
        case mapped
        case copied
    }

    /// Owns the `mmap` region (`.mapped` only); captured by every expert
    /// buffer's deallocator so the mapping outlives the last outstanding
    /// `MTLBuffer`.
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
    public let strategy: Strategy
    private let mapping: Mapping?
    /// Byte shift from the page-aligned mapping start to `streamOffset`
    /// (`.mapped`); zero for `.copied`, whose buffer starts at `streamOffset`.
    private let sliceShift: Int
    /// Per-expert buffer plus the expert's byte offset within it. `.mapped`:
    /// one buffer per expert over the mapping (offset non-zero only when the
    /// expert's file offset is not page-aligned). `.copied`: the layer buffer
    /// with the expert's offset inside it.
    private let expertViews: [(buffer: MTLBuffer, offset: UInt64)]

    /// One buffer over the whole layer, for kernels that address an expert
    /// from a GPU-side index as `baseOffset + expert * expertStride`. Nil when
    /// the layer's experts are not laid out at a uniform stride.
    public struct SlabView {
        public let buffer: MTLBuffer
        /// Byte offset of expert 0 inside `buffer`.
        public let baseOffset: Int
        public let expertStride: Int
    }
    public let slabView: SlabView?
    /// Identity expert-to-slot table (`Int16`, `slot_of[e] = e`) for the GPU
    /// slot lookup, so a router's expert ids resolve to `e * expertStride`.
    private let identitySlotTable: MTLBuffer

    /// Bytes per direct read while filling a `.copied` layer.
    static let copyChunkBytes = 64 << 20

    public init(layout: StreamLayout, device: MTLDevice, strategy: Strategy = .mapped) throws {
        self.layout = layout
        self.strategy = strategy
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
        for expert in 0..<layout.expertsPerLayer {
            let regionOffset = layout.expertOffset(layer: 0, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else {
                throw StreamerError.offsetOutOfRange(regionOffset)
            }
        }
        let uniform = (0..<layout.expertsPerLayer).allSatisfy {
            layout.expertOffset(layer: 0, expert: $0) == UInt64($0) * layout.expertStride
        }
        guard layout.expertsPerLayer <= Int(Int16.max),
              let identity = device.makeBuffer(
                length: max(1, layout.expertsPerLayer) * MemoryLayout<Int16>.stride,
                options: .storageModeShared) else {
            throw StreamerError.bufferWrapFailed
        }
        let identityPtr = identity.contents().bindMemory(to: Int16.self, capacity: max(1, layout.expertsPerLayer))
        for expert in 0..<layout.expertsPerLayer { identityPtr[expert] = Int16(expert) }
        self.identitySlotTable = identity

        switch strategy {
        case .copied:
            let length = ((Int(layout.streamSize) + pageSize - 1) / pageSize) * pageSize
            guard length <= device.maxBufferLength,
                  let buffer = device.makeBuffer(length: max(length, pageSize),
                                                 options: .storageModeShared) else {
                throw StreamerError.allocFailed(errno: ENOMEM)
            }
            try Self.readRegion(fd: fd, fileOffset: layout.streamOffset,
                                size: Int(layout.streamSize), into: buffer.contents())
            self.mapping = nil
            self.sliceShift = 0
            self.expertViews = (0..<layout.expertsPerLayer).map {
                (buffer: buffer, offset: layout.expertOffset(layer: 0, expert: $0))
            }
            self.slabView = uniform && layout.expertsPerLayer > 0
                ? SlabView(buffer: buffer,
                           baseOffset: Int(layout.expertOffset(layer: 0, expert: 0)),
                           expertStride: Int(layout.expertStride))
                : nil

        case .mapped:
            let alignedOffset = (layout.streamOffset / UInt64(pageSize)) * UInt64(pageSize)
            let shift = Int(layout.streamOffset - alignedOffset)
            let mappedLength = shift + Int(layout.streamSize)
            let mapped = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, fd, off_t(alignedOffset))
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
            // multiple, and mmap maps whole pages, so rounding up stays
            // inside the region.
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
    }

    /// Direct reads of `[fileOffset, fileOffset + size)` into `destination`,
    /// `copyChunkBytes` at a time across a few workers. `F_NOCACHE` keeps the
    /// bytes out of the page cache: the buffer is the copy that matters.
    private static func readRegion(fd: Int32, fileOffset: UInt64, size: Int,
                                   into destination: UnsafeMutableRawPointer) throws {
        guard size > 0 else { return }
        _ = fcntl(fd, F_NOCACHE, 1)
        let chunks = (size + copyChunkBytes - 1) / copyChunkBytes
        let workers = max(1, min(4, chunks))
        nonisolated(unsafe) var failure: Int32 = 0
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            var chunk = worker
            while chunk < chunks {
                let start = chunk * copyChunkBytes
                let length = min(copyChunkBytes, size - start)
                var done = 0
                while done < length {
                    let n = pread(fd, destination.advanced(by: start + done), length - done,
                                  off_t(fileOffset) + off_t(start + done))
                    if n <= 0 {
                        let code = n == 0 ? EIO : errno
                        if code == EINTR { continue }
                        lock.lock(); failure = code; lock.unlock()
                        return
                    }
                    done += Int(n)
                }
                chunk += workers
            }
        }
        if failure != 0 { throw StreamerError.preadFailed(errno: failure) }
    }


    /// A direct expert-indexed slab for a resident GPU path (Flash-Next's
    /// checkpoint-specialized routing): the layer buffer bound at offset 0, the
    /// identity slot table, and the stride. Valid only when the file stores
    /// experts contiguously at `expert * expertStride` from a page-aligned
    /// stream start, so expert 0 sits at offset 0 of the buffer.
    public var contiguousSlabBinding:
        (slab: MTLBuffer, table: MTLBuffer, expertStride: Int)? {
        guard let view = slabView, view.baseOffset == 0,
              layout.expertStride <= UInt64(UInt32.max) else { return nil }
        return (view.buffer, identitySlotTable, view.expertStride)
    }

    public func expertBuffer(layer _: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard expert >= 0, expert < expertViews.count else {
            throw StreamerError.slotOutOfRange(expert)
        }
        let view = expertViews[expert]
        return (view.buffer, view.offset, layout.expertStride)
    }

    /// `.mapped`: touch the mapping sequentially so first-token decode does
    /// not pay the page-in cost. Called at load time; counts as model load,
    /// not decode. `.copied` already holds every byte and does nothing.
    public func warmUp() {
        guard let mapping else { return }
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
