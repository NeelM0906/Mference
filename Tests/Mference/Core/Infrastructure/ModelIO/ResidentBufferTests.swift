import Testing
import Foundation
import Metal
@testable import Mference

@Suite struct ResidentBufferTests {

    @Test func wrapsResidentRegionAndReadsBytes() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-resident-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        // 24-byte fake header (zeros) followed by 16-byte fake index region
        // followed by 1 KB of resident payload with a recognizable pattern.
        let preamble = Data(repeating: 0, count: 24 + 16)
        var payload = Data(count: 1024)
        for i in 0..<payload.count { payload[i] = UInt8(i & 0xFF) }
        try (preamble + payload).write(to: url)

        let resident = try ResidentBuffer(
            fileURL: url,
            fileOffset: UInt64(preamble.count),
            residentSize: UInt64(payload.count),
            device: device)
        let chunk = try #require(resident.chunk(containing: 0, UInt64(payload.count)))
        #expect(chunk.bufferOffset == UInt64(preamble.count))
        for i in 0..<payload.count {
            let got = chunk.buffer.contents().load(
                fromByteOffset: Int(chunk.bufferOffset) + i, as: UInt8.self)
            #expect(got == UInt8(i & 0xFF), "byte \(i)")
        }
    }

    @Test func unalignedFileOffsetPreservesTheLogicalPayloadOffset() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-resident-unaligned-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        // Offset that is not a multiple of getpagesize() — say 137.
        let preLen = 137
        let preamble = Data(repeating: 0x55, count: preLen)
        var payload = Data(count: 64)
        for i in 0..<payload.count { payload[i] = UInt8(0x80 | (i & 0x7F)) }
        try (preamble + payload).write(to: url)

        let resident = try ResidentBuffer(
            fileURL: url, fileOffset: UInt64(preLen),
            residentSize: UInt64(payload.count), device: device)
        let chunk = try #require(resident.chunk(containing: 0, UInt64(payload.count)))
        for i in 0..<payload.count {
            let got = chunk.buffer.contents().load(
                fromByteOffset: Int(chunk.bufferOffset) + i, as: UInt8.self)
            #expect(got == UInt8(0x80 | (i & 0x7F)), "byte \(i)")
        }
    }

    @Test(arguments: [0, 137])
    func gpuReadsEveryByteAcrossUnalignedChunkCuts(preambleSize: Int) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-resident-gpu-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        let payload = Data((0..<48_000).map { UInt8(truncatingIfNeeded: $0 * 17 + 29) })
        try (Data(repeating: 0x55, count: preambleSize) + payload).write(to: url)
        let resident = try ResidentBuffer(fileURL: url, fileOffset: UInt64(preambleSize),
            residentSize: UInt64(payload.count), device: device,
            tensorSpans: [(0, 20_000), (20_000, 40_000), (40_000, 48_000)], maximumBufferLength: 32_768)
        #expect(resident.chunks.count == 2)
        let library = try device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            kernel void read_resident(device const uchar* source [[buffer(0)]],
                                      device uchar* target [[buffer(1)]],
                                      constant uint& offset [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) { target[i] = source[offset + i]; }
            """, options: nil)
        let pipeline = try device.makeComputePipelineState(function: #require(library.makeFunction(name: "read_resident")))
        let queue = try #require(device.makeCommandQueue())
        for chunk in resident.chunks {
            #expect(chunk.buffer.length <= 32_768)
            #expect(chunk.buffer.length % 16_384 == 0)
            let size = Int(chunk.end - chunk.start)
            let output = try #require(device.makeBuffer(length: size, options: .storageModeShared))
            let command = try #require(queue.makeCommandBuffer())
            let encoder = try #require(command.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(chunk.buffer, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            var shift = UInt32(chunk.bufferOffset)
            encoder.setBytes(&shift, length: 4, index: 2)
            encoder.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            #expect(command.error == nil)
            #expect(Data(bytes: output.contents(), count: size) == payload[Int(chunk.start)..<Int(chunk.end)])
        }
    }

    @Test func modelViewsIncludeChunkOffsetsForWeightsAndCompanions() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let directory = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = try Model.load(directoryURL: directory, device: device,
                                      expecting: .qwen38FlashNextToy())
        let index = original.residentIndex
        let region = index.header.indexSize
        let spans = index.entries.values.map { entry -> (start: UInt64, end: UInt64) in
            var lo = entry.fileOffset, hi = entry.fileOffset + entry.sizeBytes
            for (offset, length) in [(entry.scaleOffset, entry.scaleSize), (entry.biasOffset, entry.biasSize)] where length > 0 {
                lo = min(lo, offset)
                hi = max(hi, offset + length)
            }
            return (lo - region, hi - region)
        }
        let storage = try ResidentBuffer(fileURL: directory.appendingPathComponent("model_weights.bin"),
            fileOffset: region, residentSize: index.header.residentSize, device: device,
            tensorSpans: spans, maximumBufferLength: 65_536)
        #expect(storage.chunks.count > 1)
        #expect(storage.chunks.contains { $0.bufferOffset > 0 })
        #expect(storage.chunks.allSatisfy { $0.buffer.length <= 65_536 })
        let split = Model(device: device, config: original.config, streamingMode: original.streamingMode,
            expertCachePolicy: original.expertCachePolicy, integrityPolicy: original.integrityPolicy,
            residentBuffer: storage, residentIndex: index, packedExpertsLayout: original.packedExpertsLayout,
            manifest: original.manifest, directoryURL: directory)
        for name in index.entries.keys {
            let a = try original.resident(name: name), b = try split.resident(name: name)
            for (aOffset, bOffset, length) in [(a.offset, b.offset, a.length),
                (a.scaleOffset, b.scaleOffset, a.scaleLength), (a.biasOffset, b.biasOffset, a.biasLength)] where length > 0 {
                let expected = Data(bytes: a.buffer.contents().advanced(by: Int(aOffset)), count: Int(length))
                let actual = Data(bytes: b.buffer.contents().advanced(by: Int(bOffset)), count: Int(length))
                #expect(actual == expected, "\(name): chunked tensor slice")
            }
        }
    }
}
