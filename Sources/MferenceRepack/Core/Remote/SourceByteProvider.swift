import Darwin
import Foundation

public protocol SourceByteProvider {
    func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws
}

public final class HTTPRangeSourceByteProvider: SourceByteProvider {
    private let remote: HuggingFaceRemoteSource
    private let files: [String: RemoteFileInfo]
    private let writeTileBytes: Int

    public init(remote: HuggingFaceRemoteSource,
                files: [String: RemoteFileInfo],
                writeTileBytes: Int = WriterCore.tileBytes) {
        self.remote = remote
        self.files = files
        self.writeTileBytes = writeTileBytes
    }

    public func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        guard writeTileBytes > 0 else {
            throw RepackError.configurationInvalid(
                detail: "writeTileBytes must be positive")
        }
        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Self.validate(copy: copy, scratchBytes: writeTileBytes)
        }
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var outputFDs: [String: Int32] = [:]
        defer { outputFDs.values.forEach { close($0) } }
        var downloaded: UInt64 = 0

        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()
            guard let info = files[copy.shardID] else {
                throw RepackError.configurationInvalid(
                    detail: "missing remote info for \(copy.shardID)")
            }
            if try Posix.entryKind(temporaryPath) != .absent {
                try FileManager.default.removeItem(atPath: temporaryPath)
            }
            let base = downloaded
            let temporary = try await remote.downloadRangeToTempFile(
                filename: copy.shardID,
                info: info,
                offset: copy.sourceOffset,
                length: Int(copy.size),
                targetPath: temporaryPath,
                progress: { bytes in progress(base + bytes) },
                audit: audit)

            audit.remoteRangeRequests += 1
            audit.remoteBytesDownloaded += temporary.byteCount
            audit.largestRemoteTransferBytes = max(
                audit.largestRemoteTransferBytes,
                Int(temporary.byteCount))
            let (nextDownloaded, downloadedOverflow) = downloaded
                .addingReportingOverflow(temporary.byteCount)
            guard !downloadedOverflow else {
                throw RepackError.configurationInvalid(
                    detail: "downloaded byte total overflows UInt64")
            }
            downloaded = nextDownloaded
            progress(downloaded)

            let sourceFD = try Posix.openReadNoFollow(temporary.path)
            var touched = Set<String>()
            do {
                for destination in copy.destinations {
                    let destinationFD: Int32
                    if let existing = outputFDs[destination.destinationPath] {
                        destinationFD = existing
                    } else {
                        destinationFD = try Posix.openExistingRW(
                            destination.destinationPath)
                        outputFDs[destination.destinationPath] = destinationFD
                    }
                    touched.insert(destination.destinationPath)
                    try copyBytes(
                        sourceFD: sourceFD,
                        sourcePath: temporary.path,
                        destinationFD: destinationFD,
                        destinationPath: destination.destinationPath,
                        sourceOffset: destination.sourceOffset - copy.sourceOffset,
                        destinationOffset: destination.destinationOffset,
                        size: destination.size,
                        transform: destination.transform,
                        scratch: scratch,
                        audit: audit)
                }
                close(sourceFD)
            } catch {
                close(sourceFD)
                throw error
            }

            try Task.checkCancellation()
            for path in touched {
                if let descriptor = outputFDs[path] {
                    try Posix.fsync(descriptor, path: path)
                }
            }
            let digest = try Self.destinationDigest(
                copy,
                partialDirectory: partialDirectory,
                scratch: scratch)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: try copy.destinationByteCount()))
            progress(downloaded)
            try? FileManager.default.removeItem(atPath: temporary.path)
            try Task.checkCancellation()
        }
    }

    public static func destinationDigest(
        _ copy: CoalescedRangeCopy,
        partialDirectory: String,
        scratch suppliedScratch: UnsafeMutableRawBufferPointer? = nil
    ) throws -> String {
        let scratch = suppliedScratch ?? UnsafeMutableRawBufferPointer.allocate(
            byteCount: WriterCore.tileBytes,
            alignment: 16_384)
        defer {
            if suppliedScratch == nil { scratch.deallocate() }
        }

        guard scratch.count > 0 else {
            throw RepackError.configurationInvalid(detail: "empty digest scratch")
        }
        let usesTransforms = copy.destinations.contains {
            $0.transform != .identity
        }
        let (copyEnd, copyOverflow) = copy.sourceOffset.addingReportingOverflow(copy.size)
        guard !copyOverflow else {
            throw RepackError.configurationInvalid(
                detail: "remote source range overflows UInt64")
        }
        var digest = DestinationDigest(copy: copy, usesTransforms: usesTransforms)
        for destination in copy.destinations {
            guard destination.sourceOffset >= copy.sourceOffset else {
                throw RepackError.configurationInvalid(
                    detail: "destination source offset precedes its coalesced range")
            }
            let (sourceEnd, sourceOverflow) = destination.sourceOffset
                .addingReportingOverflow(destination.size)
            guard !sourceOverflow, sourceEnd <= copyEnd else {
                throw RepackError.configurationInvalid(
                    detail: "transformed source range exceeds its download")
            }
            let destinationBytes = try destination.destinationByteCount()
            let (_, destinationOverflow) = destination.destinationOffset
                .addingReportingOverflow(destinationBytes)
            guard !destinationOverflow else {
                throw RepackError.configurationInvalid(
                    detail: "transformed destination range overflows UInt64")
            }
            digest.append(try RangeCopyPlanner.normalizedRelativePath(
                destination.destinationPath,
                root: partialDirectory))
            digest.append(destination.destinationOffset)
            digest.append(destination.sourceOffset - copy.sourceOffset)
            digest.append(destination.size)
            if usesTransforms {
                digest.append(destinationBytes)
                digest.append(destination.transform.fingerprintDescription)
            }

            let descriptor = try Posix.openReadNoFollow(destination.destinationPath)
            do {
                var remaining = destinationBytes
                var offset = destination.destinationOffset
                while remaining > 0 {
                    let count = remaining > UInt64(scratch.count)
                        ? scratch.count : Int(remaining)
                    try Posix.preadAll(
                        fd: descriptor,
                        path: destination.destinationPath,
                        buf: scratch.baseAddress!,
                        count: count,
                        offset: offset)
                    digest.append(UnsafeRawBufferPointer(
                        start: scratch.baseAddress,
                        count: count))
                    remaining -= UInt64(count)
                    offset += UInt64(count)
                }
                close(descriptor)
            } catch {
                close(descriptor)
                throw error
            }
        }
        return digest.finalize()
    }

    private func copyBytes(
        sourceFD: Int32,
        sourcePath: String,
        destinationFD: Int32,
        destinationPath: String,
        sourceOffset: UInt64,
        destinationOffset: UInt64,
        size: UInt64,
        transform: RangeCopyTransform,
        scratch: UnsafeMutableRawBufferPointer,
        audit: RepackAudit
    ) throws {
        let sourceCapacity = try Self.sourceTileCapacity(
            for: transform,
            scratchBytes: scratch.count)
        var remaining = size
        var source = sourceOffset
        var destination = destinationOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let count = remaining > UInt64(sourceCapacity)
                ? sourceCapacity : Int(remaining)
            guard count > 0,
                  UInt64(count) % transform.inputUnitBytes == 0 else {
                throw RepackError.configurationInvalid(
                    detail: "unaligned source size for \(transform.fingerprintDescription)")
            }
            try Posix.preadAll(
                fd: sourceFD,
                path: sourcePath,
                buf: scratch.baseAddress!,
                count: count,
                offset: source)
            let outputCount = try Self.transformInPlace(
                transform,
                buffer: scratch,
                sourceCount: count)
            try Posix.pwriteAll(
                fd: destinationFD,
                path: destinationPath,
                buf: scratch.baseAddress!,
                count: outputCount,
                offset: destination)
            audit.recordTile(bytes: max(count, outputCount))
            audit.recordRead(bytes: count)
            audit.recordWrite(bytes: outputCount)
            remaining -= UInt64(count)
            source += UInt64(count)
            destination += UInt64(outputCount)
        }
    }

    private static func validate(copy: CoalescedRangeCopy,
                                 scratchBytes: Int) throws {
        guard copy.size > 0, copy.size <= UInt64(Int.max) else {
            throw RepackError.configurationInvalid(
                detail: "remote range size is outside the supported bound")
        }
        let (copyEnd, copyOverflow) = copy.sourceOffset.addingReportingOverflow(copy.size)
        guard !copyOverflow else {
            throw RepackError.configurationInvalid(
                detail: "remote source range overflows UInt64")
        }
        for destination in copy.destinations {
            let unit = destination.transform.inputUnitBytes
            guard destination.size > 0,
                  destination.size % unit == 0,
                  destination.sourceOffset >= copy.sourceOffset else {
                throw RepackError.configurationInvalid(
                    detail: "invalid transformed source range")
            }
            let (sourceEnd, sourceOverflow) = destination.sourceOffset
                .addingReportingOverflow(destination.size)
            guard !sourceOverflow, sourceEnd <= copyEnd else {
                throw RepackError.configurationInvalid(
                    detail: "transformed source range exceeds its download")
            }
            let destinationBytes = try destination.destinationByteCount()
            let (_, destinationOverflow) = destination.destinationOffset
                .addingReportingOverflow(destinationBytes)
            guard !destinationOverflow else {
                throw RepackError.configurationInvalid(
                    detail: "transformed destination range overflows UInt64")
            }
            _ = try sourceTileCapacity(
                for: destination.transform,
                scratchBytes: scratchBytes)
        }
    }

    private static func sourceTileCapacity(for transform: RangeCopyTransform,
                                           scratchBytes: Int) throws -> Int {
        guard scratchBytes > 0 else {
            throw RepackError.configurationInvalid(detail: "empty write scratch")
        }
        let capacity: Int
        switch transform {
        case .identity:
            capacity = scratchBytes
        case .unpackInt2ToInt4:
            capacity = scratchBytes / 2 / 4 * 4
        case .repeatBF16(let count, _):
            guard count > 0 else {
                throw RepackError.configurationInvalid(
                    detail: "repeat-bf16 count must be positive")
            }
            capacity = scratchBytes / count / 2 * 2
        }
        guard capacity >= Int(transform.inputUnitBytes) else {
            throw RepackError.configurationInvalid(
                detail: "write scratch too small for \(transform.fingerprintDescription)")
        }
        return capacity
    }

    private static func transformInPlace(
        _ transform: RangeCopyTransform,
        buffer: UnsafeMutableRawBufferPointer,
        sourceCount: Int
    ) throws -> Int {
        switch transform {
        case .identity:
            return sourceCount
        case .unpackInt2ToInt4:
            let outputCount = sourceCount * 2
            guard outputCount <= buffer.count else {
                throw RepackError.scratchExceeded(
                    requested: outputCount, limit: buffer.count)
            }
            let bytes = buffer.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for wordIndex in stride(from: sourceCount / 4 - 1, through: 0, by: -1) {
                let source = bytes.advanced(by: wordIndex * 4)
                let packed = UInt32(source[0])
                    | UInt32(source[1]) << 8
                    | UInt32(source[2]) << 16
                    | UInt32(source[3]) << 24
                var low: UInt32 = 0
                var high: UInt32 = 0
                for code in 0..<8 {
                    low |= (packed >> UInt32(code * 2) & 0x3)
                        << UInt32(code * 4)
                    high |= (packed >> UInt32((code + 8) * 2) & 0x3)
                        << UInt32(code * 4)
                }
                writeLittleEndian(low, to: bytes.advanced(by: wordIndex * 8))
                writeLittleEndian(high, to: bytes.advanced(by: wordIndex * 8 + 4))
            }
            return outputCount
        case .repeatBF16(let repetitions, let negated):
            guard repetitions > 0 else {
                throw RepackError.configurationInvalid(
                    detail: "repeat-bf16 count must be positive")
            }
            let (outputCount, overflow) = sourceCount.multipliedReportingOverflow(
                by: repetitions)
            guard !overflow, outputCount <= buffer.count else {
                throw RepackError.scratchExceeded(
                    requested: overflow ? Int.max : outputCount,
                    limit: buffer.count)
            }
            let bytes = buffer.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for valueIndex in stride(from: sourceCount / 2 - 1, through: 0, by: -1) {
                let source = bytes.advanced(by: valueIndex * 2)
                var bits = UInt16(source[0]) | UInt16(source[1]) << 8
                if negated { bits ^= 0x8000 }
                for repetition in 0..<repetitions {
                    let output = bytes.advanced(
                        by: (valueIndex * repetitions + repetition) * 2)
                    output[0] = UInt8(truncatingIfNeeded: bits)
                    output[1] = UInt8(truncatingIfNeeded: bits >> 8)
                }
            }
            return outputCount
        }
    }

    private static func writeLittleEndian(_ value: UInt32,
                                          to bytes: UnsafeMutablePointer<UInt8>) {
        bytes[0] = UInt8(truncatingIfNeeded: value)
        bytes[1] = UInt8(truncatingIfNeeded: value >> 8)
        bytes[2] = UInt8(truncatingIfNeeded: value >> 16)
        bytes[3] = UInt8(truncatingIfNeeded: value >> 24)
    }
}

private struct DestinationDigest {
    private var stream = Sha256Stream()

    init(copy: CoalescedRangeCopy, usesTransforms: Bool) {
        append(usesTransforms
            ? "Mference.RemoteRangeDestination.v2"
            : "Mference.RemoteRangeDestination.v1")
        append(copy.id)
        append(UInt64(copy.destinations.count))
    }

    mutating func append(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { stream.update($0) }
    }

    mutating func append(_ value: String) {
        let data = Data(value.utf8)
        append(UInt64(data.count))
        data.withUnsafeBytes { stream.update($0) }
    }

    mutating func append(_ bytes: UnsafeRawBufferPointer) {
        stream.update(bytes)
    }

    func finalize() -> String {
        stream.finalizeHexString()
    }
}
