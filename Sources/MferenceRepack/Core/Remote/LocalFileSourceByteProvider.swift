import Darwin
import Foundation

/// Serves a range-copy plan from local copies of the pinned source files
/// instead of the Hub: same plan, same checkpoint and commit protocol, same
/// destination digests as `HTTPRangeSourceByteProvider`, so a local install
/// and a remote install of one pin write identical bytes and can resume each
/// other. Identity copies only (the GGUF path); every read and write goes
/// through one bounded scratch tile.
///
/// The caller verifies each local file against its pin (size and SHA-256)
/// before planning; this type only maps a plan's shard IDs to local paths.
public final class LocalFileSourceByteProvider: SourceByteProvider {
    private let files: [String: String]
    private let writeTileBytes: Int

    /// `files` maps a shard ID (the file's path in the pinned repo) to its
    /// local path.
    public init(files: [String: String], writeTileBytes: Int = WriterCore.tileBytes) {
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
            throw RepackError.configurationInvalid(detail: "writeTileBytes must be positive")
        }
        for copy in copies where !completedRangeIDs.contains(copy.id) {
            guard files[copy.shardID] != nil else {
                throw RepackError.configurationInvalid(detail: "no local file for \(copy.shardID)")
            }
            for destination in copy.destinations {
                guard destination.transform == .identity else {
                    throw RepackError.configurationInvalid(
                        detail: "local source copies bytes unchanged; got \(destination.transform.fingerprintDescription)")
                }
                let (end, overflow) = destination.sourceOffset.addingReportingOverflow(destination.size)
                let (copyEnd, copyOverflow) = copy.sourceOffset.addingReportingOverflow(copy.size)
                guard destination.size > 0, !overflow, !copyOverflow,
                      destination.sourceOffset >= copy.sourceOffset, end <= copyEnd else {
                    throw RepackError.configurationInvalid(detail: "invalid local source range")
                }
            }
        }
        let scratch = UnsafeMutableRawBufferPointer.allocate(byteCount: writeTileBytes, alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var sourceFDs: [String: Int32] = [:]
        var outputFDs: [String: Int32] = [:]
        defer {
            sourceFDs.values.forEach { close($0) }
            outputFDs.values.forEach { close($0) }
        }
        var copied: UInt64 = 0
        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()
            let sourcePath = files[copy.shardID]!
            let sourceFD: Int32
            if let open = sourceFDs[copy.shardID] {
                sourceFD = open
            } else {
                sourceFD = try Posix.openReadNoFollow(sourcePath)
                sourceFDs[copy.shardID] = sourceFD
            }
            var touched = Set<String>()
            for destination in copy.destinations {
                let destinationFD: Int32
                if let existing = outputFDs[destination.destinationPath] {
                    destinationFD = existing
                } else {
                    destinationFD = try Posix.openExistingRW(destination.destinationPath)
                    outputFDs[destination.destinationPath] = destinationFD
                }
                touched.insert(destination.destinationPath)
                var remaining = destination.size
                var source = destination.sourceOffset
                var target = destination.destinationOffset
                while remaining > 0 {
                    try Task.checkCancellation()
                    let count = remaining > UInt64(scratch.count) ? scratch.count : Int(remaining)
                    try Posix.preadAll(fd: sourceFD, path: sourcePath, buf: scratch.baseAddress!,
                                       count: count, offset: source)
                    try Posix.pwriteAll(fd: destinationFD, path: destination.destinationPath,
                                        buf: scratch.baseAddress!, count: count, offset: target)
                    audit.recordTile(bytes: count)
                    audit.recordRead(bytes: count)
                    audit.recordWrite(bytes: count)
                    remaining -= UInt64(count)
                    source += UInt64(count)
                    target += UInt64(count)
                }
            }
            for path in touched {
                if let descriptor = outputFDs[path] { try Posix.fsync(descriptor, path: path) }
            }
            let digest = try HTTPRangeSourceByteProvider.destinationDigest(
                copy, partialDirectory: partialDirectory, scratch: scratch)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: try copy.destinationByteCount()))
            copied += copy.size
            progress(copied)
        }
    }
}
