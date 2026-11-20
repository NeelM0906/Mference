import CryptoKit
import Foundation

/// Multi-file pinned packages (Notes spec 7): the speech models are folders
/// of many files, each pinned by size and SHA-256. They download through the
/// same resumable, verified, host-checked transfer as the GGUF models
/// (`RangeDownload`); the package bookkeeping lives in `SpeechModelInstaller`.
extension DownloadManager {
    /// One file of a pinned package.
    public struct PinnedFile: Sendable, Hashable {
        public var url: URL
        /// Where the verified file goes; `<destination>.part` holds bytes so far.
        public var destination: URL
        public var size: UInt64
        /// Lowercase hex.
        public var sha256: String

        public init(url: URL, destination: URL, size: UInt64, sha256: String) {
            self.url = url
            self.destination = destination
            self.size = size
            self.sha256 = sha256.lowercased()
        }

        public var partURL: URL { destination.appendingPathExtension("part") }
    }

    /// Downloads `file` (continuing its `.part` file), checks size and
    /// SHA-256, then moves it into place, so the destination only ever holds
    /// a verified file. A mismatch deletes the bytes and throws
    /// `sizeMismatch` / `checksumMismatch`; cancellation keeps them for a
    /// resume. `progress` reports the bytes of this file on disk.
    public static func fetchPinned(_ file: PinnedFile, allowedHosts: [String],
                                   configuration: URLSessionConfiguration = .ephemeral,
                                   progress: @escaping @Sendable (UInt64) -> Void) async throws {
        let manager = FileManager.default
        if manager.fileExists(atPath: file.destination.path) {
            if fileSize(file.destination) == file.size { return }
            try? manager.removeItem(at: file.destination)
        }
        let part = file.partURL
        do {
            try manager.createDirectory(at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !manager.fileExists(atPath: part.path) {
                guard manager.createFile(atPath: part.path, contents: nil) else {
                    throw DownloadError.file("cannot create \(part.lastPathComponent)")
                }
            }
        } catch let error as DownloadError {
            throw error
        } catch {
            throw DownloadError.file(error.localizedDescription)
        }
        var existing = fileSize(part)
        if existing > file.size {
            try? manager.removeItem(at: part)
            _ = manager.createFile(atPath: part.path, contents: nil)
            existing = 0
        }
        progress(existing)
        // Bytes already on disk are hashed first, off the caller's executor.
        let resumeLength = existing
        let hashing = Task.detached {
            var hasher = SHA256()
            try hash(file: part, length: resumeLength, into: &hasher)
            return hasher
        }
        let hasher = try await withTaskCancellationHandler {
            try await hashing.value
        } onCancel: {
            hashing.cancel()
        }
        try Task.checkCancellation()
        let outcome: RangeDownload.Outcome
        if existing == file.size {
            outcome = RangeDownload.Outcome(length: existing,
                                            sha256: hasher.finalize().map { String(format: "%02x", $0) }.joined())
        } else {
            outcome = try await RangeDownload.run(url: file.url, file: part, offset: existing, hasher: hasher,
                                                  allowedHosts: allowedHosts, configuration: configuration,
                                                  progress: progress)
        }
        guard outcome.length == file.size else {
            try? manager.removeItem(at: part)
            throw DownloadError.sizeMismatch(expected: file.size, actual: outcome.length)
        }
        guard outcome.sha256 == file.sha256 else {
            try? manager.removeItem(at: part)
            throw DownloadError.checksumMismatch
        }
        do {
            try manager.moveItem(at: part, to: file.destination)
        } catch {
            throw DownloadError.file(error.localizedDescription)
        }
    }
}
