import Testing
import Foundation
import Darwin
import Metal
@testable import Mference

/// Unit tests for the all-resident backend: subregion views must alias the
/// exact file bytes at each expert's offset, out-of-range experts must be
/// rejected, and warm-up must be safe to call. No real weights — the same
/// synthetic tagged-blob layer file the `pread` backend tests use.
@Suite struct ResidentExpertStreamerTests {

    static let pageSize = Int(getpagesize())
    static let expertStride = 2 * pageSize
    static let numExperts = 4
    /// A non-zero stream offset exercises `streamOffset` in the mapping math.
    static let headerPages = 1
    static var streamOffset: UInt64 { UInt64(headerPages * pageSize) }
    static var streamSize: UInt64 { UInt64(numExperts * expertStride) }

    static func tagByte(_ expert: Int) -> UInt8 { UInt8(0xB0 + expert) }

    static func writeSyntheticLayer() throws -> URL {
        let total = Int(streamOffset) + Int(streamSize)
        var bytes = [UInt8](repeating: 0, count: total)
        for e in 0..<numExperts {
            let start = Int(streamOffset) + e * expertStride
            for i in 0..<expertStride { bytes[start + i] = tagByte(e) }
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("resident-streamer-\(UUID().uuidString).bin")
        try Data(bytes).write(to: url)
        return url
    }

    static func makeLayout(path: String) -> StreamLayout {
        StreamLayout(path: path,
                     streamOffset: streamOffset,
                     streamSize: streamSize,
                     expertsPerLayer: numExperts,
                     expertStride: UInt64(expertStride))
    }

    static func bytes(of buffer: MTLBuffer, offset: UInt64, count: Int) -> [UInt8] {
        let base = buffer.contents().advanced(by: Int(offset))
        return [UInt8](UnsafeRawBufferPointer(start: base, count: count))
    }

    @Test("Resident streamer serves expert bytes identical to the file")
    func servesFileBytes() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }

        let streamer = try ResidentExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device)
        for expert in 0..<Self.numExperts {
            let view = try streamer.expertBuffer(layer: 0, expert: expert)
            #expect(view.size == UInt64(Self.expertStride))
            let contents = Self.bytes(of: view.buffer,
                                      offset: view.offset,
                                      count: Int(view.size))
            #expect(contents.allSatisfy { $0 == Self.tagByte(expert) },
                    "expert \(expert) bytes must match the file blob")
        }
    }

    @Test("Each expert gets its own page-aligned buffer over the mapping")
    func expertsGetDedicatedBuffers() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }

        let streamer = try ResidentExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device)
        let first = try streamer.expertBuffer(layer: 0, expert: 0)
        let last = try streamer.expertBuffer(layer: 0, expert: Self.numExperts - 1)
        // One buffer per expert keeps Metal residency demands at the routed
        // working set instead of the whole layer file.
        #expect(first.buffer !== last.buffer)
        // Page-aligned expert strides start each buffer exactly at its blob.
        #expect(first.offset == 0)
        #expect(last.offset == 0)
        #expect(first.buffer.length >= Int(Self.expertStride))
    }

    @Test("Out-of-range experts are rejected")
    func rejectsOutOfRange() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }

        let streamer = try ResidentExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device)
        #expect(throws: (any Error).self) {
            _ = try streamer.expertBuffer(layer: 0, expert: Self.numExperts)
        }
        #expect(throws: (any Error).self) {
            _ = try streamer.expertBuffer(layer: 0, expert: -1)
        }
    }

    @Test("Warm-up touches the mapping without changing served bytes")
    func warmUpIsSafe() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }

        let streamer = try ResidentExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device)
        streamer.warmUp()
        let view = try streamer.expertBuffer(layer: 0, expert: 1)
        let contents = Self.bytes(of: view.buffer,
                                  offset: view.offset,
                                  count: Int(view.size))
        #expect(contents.allSatisfy { $0 == Self.tagByte(1) })
    }
}
