import Darwin
import Foundation
import Metal
import Testing
@testable import Mference

/// The slot slab is mapped anonymous memory that is unmapped when its buffer
/// is released, so its pages leave the process at once rather than waiting
/// in libmalloc's large-entry cache. The checks read the slab's own pages
/// with `mincore`, not `phys_footprint`, so what the allocator has cached
/// elsewhere in the test process cannot skew them.
@Suite(.serialized) struct PreadExpertStreamerSlabTests {
    static let experts = 4, stride = 1 << 20

    /// Per-page `mincore` bits for `[base, base + length)`, or nil with
    /// `errno` set (ENOMEM when any of the range is unmapped).
    static func pageBits(_ base: UnsafeMutableRawPointer, _ length: Int) -> [Int32]? {
        let pages = length / Int(getpagesize())
        var vec = [CChar](repeating: 0, count: pages)
        let result = vec.withUnsafeMutableBufferPointer { mincore(base, length, $0.baseAddress) }
        return result == 0 ? vec.map { Int32($0) } : nil
    }

    @Test func mappedSlabIsAlignedAndWritable() throws {
        let page = Int(getpagesize())
        let length = 6 * page
        let slab = try #require(PreadExpertStreamer.mapSlab(length: length,
                                                            alignment: PreadExpertStreamer.scratchAlignment))
        defer { munmap(slab, length) }
        #expect(Int(bitPattern: slab) % PreadExpertStreamer.scratchAlignment == 0)
        memset(slab, 0x5A, length)
        let bits = try #require(Self.pageBits(slab, length))
        #expect(bits.allSatisfy { $0 & MINCORE_INCORE != 0 && $0 & MINCORE_MODIFIED != 0 })
    }

    @Test func releasingAStreamerUnmapsItsSlab() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-slab-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 0x33, count: Self.experts * Self.stride).write(to: url)

        weak var weakStreamer: PreadExpertStreamer?
        weak var weakBuffer: MTLBuffer?
        var base: UnsafeMutableRawPointer?
        var length = 0
        try autoreleasepool {
            let streamer = try PreadExpertStreamer(
                layout: StreamLayout(path: url.path, streamOffset: 0,
                                     streamSize: UInt64(Self.experts * Self.stride),
                                     expertsPerLayer: Self.experts, expertStride: UInt64(Self.stride)),
                device: device, slotCount: Self.experts, readStats: nil)
            weakStreamer = streamer
            for expert in 0..<Self.experts {
                let loaded = try streamer.loadExpert(layer: 0, expert: expert, slot: expert)
                weakBuffer = loaded.buffer
                base = loaded.buffer.contents()
                length = loaded.buffer.length
                #expect(UnsafeRawPointer(loaded.buffer.contents().advanced(by: Int(loaded.offset)))
                            .load(as: UInt8.self) == 0x33)
            }
            let slab = try #require(base)
            #expect(Int(bitPattern: slab) % PreadExpertStreamer.scratchAlignment == 0)
            let bits = try #require(Self.pageBits(slab, length), "the filled slab is mapped")
            #expect(bits.allSatisfy { $0 & MINCORE_INCORE != 0 }, "every pread-filled page is resident")
        }
        #expect(weakStreamer == nil && weakBuffer == nil)
        let slab = try #require(base)
        let afterRelease = Self.pageBits(slab, length)
        let releaseErrno = errno
        #expect(afterRelease == nil, "the slab is unmapped on release")
        #expect(releaseErrno == ENOMEM)
    }
}
