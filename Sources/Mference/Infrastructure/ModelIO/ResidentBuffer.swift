import Foundation
import Darwin
import Metal

/// `mmap`'d view of `model_weights.bin`'s tensor data region, wrapped in one
/// or more shared `MTLBuffer` chunks. One chunk covers the whole region when
/// it fits under the device's `maxBufferLength`; a larger region (Qwen 3.8's
/// fully-resident 15 GB exceeds the 24 GB M5's 13.3 GB limit) is cut between
/// tensor spans so every tensor and its scale/bias companions stay inside a
/// single buffer. All resident `TensorView`s alias byte offsets inside one
/// chunk's buffer.
final class ResidentBuffer {
    struct Chunk {
        /// Region-relative byte range this chunk's buffer covers.
        let start: UInt64
        let end: UInt64
        let buffer: MTLBuffer
    }

    let chunks: [Chunk]

    /// Backwards-compatible single-chunk accessor for regions that fit in one
    /// buffer (every family before Qwen 3.8, and all test fixtures).
    var buffer: MTLBuffer {
        precondition(chunks.count == 1,
                     "resident region is chunked; resolve through chunk(containing:)")
        return chunks[0].buffer
    }

    /// `tensorSpans` are region-relative `[start, end)` byte spans that must
    /// not be split across chunks (a tensor plus its companions). They are
    /// only consulted when the region exceeds the device's buffer limit.
    init(fileURL: URL,
         fileOffset: UInt64,
         residentSize: UInt64,
         device: MTLDevice,
         tensorSpans: [(start: UInt64, end: UInt64)] = []) throws {
        let limit = UInt64(device.maxBufferLength)
        let ranges: [(start: UInt64, end: UInt64)]
        if residentSize <= limit {
            ranges = [(0, residentSize)]
        } else {
            ranges = try Self.chunkRanges(regionSize: residentSize,
                                          limit: limit,
                                          tensorSpans: tensorSpans)
        }
        self.chunks = try ranges.map { range in
            try Chunk(fileURL: fileURL,
                      regionFileOffset: fileOffset,
                      start: range.start,
                      end: range.end,
                      device: device)
        }
    }

    /// The chunk whose range contains `[start, end)`, or nil when the span
    /// straddles a cut (impossible for spans passed to the initializer).
    func chunk(containing start: UInt64, _ end: UInt64) -> Chunk? {
        // Few chunks (2 for Qwen 3.8): linear scan beats a binary search.
        chunks.first { $0.start <= start && end <= $0.end }
    }

    /// Greedy sweep over the sorted tensor spans: extend the open chunk while
    /// it stays under `limit`, cut at the previous span boundary otherwise.
    /// Trailing region bytes after the last span ride in the final chunk.
    static func chunkRanges(regionSize: UInt64,
                           limit: UInt64,
                           tensorSpans: [(start: UInt64, end: UInt64)])
        throws -> [(start: UInt64, end: UInt64)] {
        guard !tensorSpans.isEmpty else {
            throw ModelError.indexCorrupt(
                detail: "resident region \(regionSize) exceeds the device " +
                        "buffer limit \(limit) and no tensor spans were given")
        }
        let sorted = tensorSpans.sorted { $0.start < $1.start }
        var ranges: [(start: UInt64, end: UInt64)] = []
        var chunkStart: UInt64 = 0
        var chunkEnd: UInt64 = 0
        for span in sorted {
            let extended = max(chunkEnd, span.end)
            if extended - chunkStart > limit {
                guard chunkEnd > chunkStart else {
                    throw ModelError.indexCorrupt(
                        detail: "resident tensor span \(span.start)..<\(span.end) " +
                                "exceeds the device buffer limit \(limit)")
                }
                ranges.append((chunkStart, chunkEnd))
                chunkStart = min(span.start, chunkEnd)
                chunkEnd = span.end
                guard chunkEnd - chunkStart <= limit else {
                    throw ModelError.indexCorrupt(
                        detail: "resident tensor span \(span.start)..<\(span.end) " +
                                "exceeds the device buffer limit \(limit)")
                }
            } else {
                chunkEnd = extended
            }
        }
        // Cover any padding after the last tensor if it still fits.
        let tail = regionSize > chunkEnd && regionSize - chunkStart <= limit
            ? regionSize : chunkEnd
        ranges.append((chunkStart, tail))
        return ranges
    }
}

private extension ResidentBuffer.Chunk {
    /// `mmap` the page-aligned window covering the chunk's file range and
    /// wrap it so the chunk's first byte is byte 0 of the buffer.
    init(fileURL: URL,
         regionFileOffset: UInt64,
         start: UInt64,
         end: UInt64,
         device: MTLDevice) throws {
        let pageSize = Int(getpagesize())

        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }

        let fileOffset = regionFileOffset + start
        let chunkSize = end - start
        let alignedOffset = (fileOffset / UInt64(pageSize)) * UInt64(pageSize)
        let sliceShift = Int(fileOffset - alignedOffset)
        let mappedLen = sliceShift + Int(chunkSize)
        let mapped = mmap(nil, mappedLen, PROT_READ, MAP_PRIVATE,
                          fd, off_t(alignedOffset))
        if mapped == MAP_FAILED {
            throw ModelError.posixFailed(call: "mmap", errno: errno)
        }
        let base = mapped!

        _ = posix_madvise(base, mappedLen, POSIX_MADV_RANDOM)

        let sliceStart = base.advanced(by: sliceShift)

        // Capture pointer + length for the deallocator. Do NOT capture self
        // here — that would create a retain cycle through the MTLBuffer.
        nonisolated(unsafe) let captureBase = base
        let captureLen = mappedLen
        guard let buf = device.makeBuffer(
            bytesNoCopy: sliceStart,
            length: Int(chunkSize),
            options: .storageModeShared,
            deallocator: { _, _ in
                munmap(captureBase, captureLen)
            }
        ) else {
            munmap(base, mappedLen)
            throw ModelError.residentBufferWrapFailed
        }

        self.init(start: start, end: end, buffer: buf)
    }
}
