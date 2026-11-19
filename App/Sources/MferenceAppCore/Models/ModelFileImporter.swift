import CryptoKit
import Foundation

public enum ModelImportError: Error, Sendable, Hashable, LocalizedError {
    case unreadable(String)
    case notGGUF
    /// A GGUF, but not one of the product files (size and SHA-256 differ).
    case notAProductModel
    case insufficientDisk(neededBytes: UInt64, availableBytes: UInt64)
    /// A different file already has the product name; it is left untouched
    /// (it may be the person's own build). `existing` is for Show in Finder.
    case nameTaken(existing: URL)
    case file(String)

    /// The file to reveal in Finder for this error, if any.
    public var fileToShow: URL? {
        if case .nameTaken(let existing) = self { return existing }
        return nil
    }

    public var errorDescription: String? {
        switch self {
        case .unreadable(let message): "The file could not be read: \(message)"
        case .notGGUF: "This is not a GGUF model file."
        case .notAProductModel:
            "This file is not one of the models Mference offers. Add the exact Quick or Deep file."
        case .insufficientDisk(let needed, let available):
            "Not enough free space: \(DownloadError.gigabytes(needed)) needed, \(DownloadError.gigabytes(available)) free."
        case .nameTaken(let existing):
            "Another file already uses this name in \(existing.deletingLastPathComponent().path). "
                + "It was left as it is; move or rename it, then add the model again."
        case .file(let message): "The file could not be added: \(message)"
        }
    }
}

/// Adds a model file the person picked ("Add a model file" in the library).
///
/// The file must be a GGUF whose size and SHA-256 match one of the pinned
/// product files (Quick or Deep), so the engine never sees a file Mference
/// did not validate. It is copied into the destination (the app passes
/// `~/llm-models/gguf`; tests pass a temporary directory) under the product
/// file name: first to a temporary name in that directory, where the copy
/// itself is hashed against the pinned digest, then renamed atomically
/// without replacing anything, so a half-copied or changed file is never
/// visible and an existing file is never overwritten. On APFS the copy is a
/// clone and takes no extra space on the same volume. The person's original
/// is only read, never moved or deleted.
public struct ModelFileImporter: Sendable {
    public let destination: URL
    public let sources: [DownloadSource]
    private let availableBytes: @Sendable (URL) -> UInt64?

    public init(destination: URL,
                sources: [DownloadSource] = Array(DownloadSource.product.values),
                availableBytes: @escaping @Sendable (URL) -> UInt64? = DownloadManager.availableCapacity) {
        self.destination = destination
        self.sources = sources
        self.availableBytes = availableBytes
    }

    /// Validates and adds the file; returns the model it is. A file already
    /// in place with the same contents is left as it is.
    @discardableResult
    public func importFile(at url: URL) async throws -> AppModel {
        let importer = self
        let task = Task.detached { try importer.run(url) }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    func run(_ url: URL) throws -> AppModel {
        let size = try Self.size(of: url)
        guard try Self.hasGGUFMagic(url) else { throw ModelImportError.notGGUF }
        let candidates = sources.filter { $0.sizeBytes == size }
        guard !candidates.isEmpty else { throw ModelImportError.notAProductModel }

        let manager = FileManager.default
        do {
            try manager.createDirectory(at: destination, withIntermediateDirectories: true)
        } catch {
            throw ModelImportError.file(error.localizedDescription)
        }
        if !Self.sameVolume(url, destination), let available = availableBytes(destination),
           available < size + DownloadManager.diskMarginBytes {
            throw ModelImportError.insufficientDisk(neededBytes: size + DownloadManager.diskMarginBytes,
                                                    availableBytes: available)
        }
        // Copy first and validate the copy: the bytes that are published are
        // the bytes that were hashed, even if the original changes meanwhile.
        let staging = destination.appending(path: ".import-\(UUID().uuidString).gguf")
        defer { try? manager.removeItem(at: staging) }
        do {
            try Task.checkCancellation()
            try manager.copyItem(at: url, to: staging)
            try Task.checkCancellation()
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw ModelImportError.file(error.localizedDescription)
        }
        guard try Self.size(of: staging) == size else { throw ModelImportError.file("the copy is incomplete") }
        let digest = try Self.sha256(of: staging, length: size)
        guard let source = candidates.first(where: { $0.sha256 == digest }) else {
            throw ModelImportError.notAProductModel
        }

        let final = destination.appending(path: source.fileName)
        if manager.fileExists(atPath: final.path) {
            // Already there and identical: nothing to publish.
            if (try? Self.size(of: final)) == size, (try? Self.sha256(of: final, length: size)) == digest {
                return source.model
            }
            // Anything else under that name is never replaced.
            throw ModelImportError.nameTaken(existing: final)
        }
        // Exclusive rename: fails rather than replace a file that appeared
        // under the name meanwhile.
        if renamex_np(staging.path, final.path, UInt32(RENAME_EXCL)) != 0 {
            if errno == EEXIST { throw ModelImportError.nameTaken(existing: final) }
            throw ModelImportError.file(String(cString: strerror(errno)))
        }
        return source.model
    }

    // MARK: Files

    static func size(of url: URL) throws -> UInt64 {
        do {
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            guard values.isRegularFile == true, let size = values.fileSize else {
                throw ModelImportError.unreadable("not a regular file")
            }
            return UInt64(size)
        } catch let error as ModelImportError {
            throw error
        } catch {
            throw ModelImportError.unreadable(error.localizedDescription)
        }
    }

    /// GGUF files start with the four bytes "GGUF".
    static func hasGGUFMagic(_ url: URL) throws -> Bool {
        do {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            return try handle.read(upToCount: 4) == Data("GGUF".utf8)
        } catch {
            throw ModelImportError.unreadable(error.localizedDescription)
        }
    }

    static func sha256(of url: URL, length: UInt64) throws -> String {
        var hasher = SHA256()
        do {
            try DownloadManager.hash(file: url, length: length, into: &hasher)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw ModelImportError.unreadable(error.localizedDescription)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func sameVolume(_ a: URL, _ b: URL) -> Bool {
        let key = URLResourceKey.volumeIdentifierKey
        var probe = b
        while !FileManager.default.fileExists(atPath: probe.path), probe.pathComponents.count > 1 {
            probe = probe.deletingLastPathComponent()
        }
        guard let left = try? a.resourceValues(forKeys: [key]).volumeIdentifier as? NSObject,
              let right = try? probe.resourceValues(forKeys: [key]).volumeIdentifier as? NSObject else { return false }
        return left.isEqual(right)
    }
}
