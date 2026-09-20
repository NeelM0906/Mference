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
        /// The first logical chunk byte within the page-aligned Metal buffer.
        let bufferOffset: UInt64
    }

    let chunks: [Chunk]

    /// `tensorSpans` are region-relative `[start, end)` byte spans that must
    /// not be split across chunks (a tensor plus its companions). They are
    /// only consulted when the region exceeds the device's buffer limit.
    init(fileURL: URL,
         fileOffset: UInt64,
         residentSize: UInt64,
         device: MTLDevice,
         tensorSpans: [(start: UInt64, end: UInt64)] = [],
         maximumBufferLength: UInt64? = nil) throws {
        let limit = min(maximumBufferLength ?? UInt64(device.maxBufferLength), UInt64(device.maxBufferLength))
        let page = UInt64(getpagesize())
        guard residentSize > 0, fileOffset <= UInt64(Int.max) - (page - 1),
              residentSize <= UInt64(Int.max) - fileOffset - (page - 1) else {
            throw ModelError.indexCorrupt(detail: "invalid resident mapped region")
        }
        let ranges: [(start: UInt64, end: UInt64)]
        if Self.mappedSize(fileOffset: fileOffset, start: 0, end: residentSize, page: page) <= limit {
            ranges = [(0, residentSize)]
        } else {
            ranges = try Self.chunkRanges(regionSize: residentSize,
                                          limit: limit,
                                          tensorSpans: tensorSpans,
                                          fileOffset: fileOffset, page: page)
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
                           tensorSpans: [(start: UInt64, end: UInt64)],
                           fileOffset: UInt64 = 0,
                           page: UInt64 = UInt64(getpagesize()))
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
            guard span.start < span.end, span.end <= regionSize else {
                throw ModelError.indexCorrupt(detail: "invalid resident tensor span")
            }
            let extended = max(chunkEnd, span.end)
            if Self.mappedSize(fileOffset: fileOffset, start: chunkStart, end: extended, page: page) > limit {
                guard chunkEnd > chunkStart else {
                    throw ModelError.indexCorrupt(
                        detail: "resident tensor span \(span.start)..<\(span.end) " +
                                "exceeds the device buffer limit \(limit)")
                }
                ranges.append((chunkStart, chunkEnd))
                chunkStart = min(span.start, chunkEnd)
                chunkEnd = span.end
                guard Self.mappedSize(fileOffset: fileOffset, start: chunkStart, end: chunkEnd, page: page) <= limit else {
                    throw ModelError.indexCorrupt(
                        detail: "resident tensor span \(span.start)..<\(span.end) " +
                                "exceeds the device buffer limit \(limit)")
                }
            } else {
                chunkEnd = extended
            }
        }
        // Cover any padding after the last tensor if it still fits.
        let tail = regionSize > chunkEnd && Self.mappedSize(
            fileOffset: fileOffset, start: chunkStart, end: regionSize, page: page) <= limit
            ? regionSize : chunkEnd
        ranges.append((chunkStart, tail))
        return ranges
    }

    private static func mappedSize(fileOffset: UInt64, start: UInt64, end: UInt64, page: UInt64) -> UInt64 {
        let alignedStart = (fileOffset + start) / page * page
        let alignedEnd = (fileOffset + end + page - 1) / page * page
        return alignedEnd - alignedStart
    }
}

private extension ResidentBuffer.Chunk {
    /// `mmap` the page-aligned window covering the chunk's file range and
    /// wrap the aligned base, preserving the logical start as bufferOffset.
    /// Passing base + sliceShift to bytesNoCopy is invalid for Metal: CPU
    /// contents() can appear correct while GPU address translation is wrong.
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
        let mappedLen = (sliceShift + Int(chunkSize) + pageSize - 1) / pageSize * pageSize
        let mapped = mmap(nil, mappedLen, PROT_READ, MAP_PRIVATE,
                          fd, off_t(alignedOffset))
        if mapped == MAP_FAILED {
            throw ModelError.posixFailed(call: "mmap", errno: errno)
        }
        let base = mapped!

        _ = posix_madvise(base, mappedLen, POSIX_MADV_RANDOM)

        // Capture pointer + length for the deallocator. Do NOT capture self
        // here — that would create a retain cycle through the MTLBuffer.
        nonisolated(unsafe) let captureBase = base
        let captureLen = mappedLen
        guard let buf = device.makeBuffer(
            bytesNoCopy: base,
            length: mappedLen,
            options: .storageModeShared,
            deallocator: { _, _ in
                munmap(captureBase, captureLen)
            }
        ) else {
            munmap(base, mappedLen)
            throw ModelError.residentBufferWrapFailed
        }

        self.init(start: start, end: end, buffer: buf, bufferOffset: UInt64(sliceShift))
    }
}
