import CryptoKit
import Foundation

/// The search model Find downloads when it is turned on: the text part of
/// EmbeddingGemma 2 (Apache 2.0), Q8_0 GGUF, pinned by revision, size and
/// SHA-256 (THIRD_PARTY_NOTICES.md).
public enum FindModel {
    public static let repository = "ggml-org/embeddinggemma-2-GGUF"
    public static let revision = "6ea3e3a309d12a4779cfda98b8c19800d2d5da71"
    public static let fileName = "embeddinggemma-2-Q8_0.gguf"
    public static let sizeBytes: UInt64 = 309_855_456
    public static let sha256 = "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135"
    public static let licence = "Apache 2.0"
    public static var url: URL {
        URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(fileName)")!
    }

    /// "310 MB", for the UI.
    public static var sizeLabel: String { FindModelFile.text.sizeLabel }
}

/// One pinned file of the search model.
public struct FindModelFile: Sendable, Hashable {
    public var fileName: String
    public var sizeBytes: UInt64
    public var sha256: String

    public var url: URL {
        URL(string: "https://huggingface.co/\(FindModel.repository)/resolve/\(FindModel.revision)/\(fileName)")!
    }

    /// "310 MB", for the UI.
    public var sizeLabel: String {
        "\(Int((Double(sizeBytes) / 1e6).rounded())) MB"
    }

    /// The text model (always).
    public static let text = FindModelFile(fileName: FindModel.fileName, sizeBytes: FindModel.sizeBytes,
                                           sha256: FindModel.sha256)
    /// The image and audio encoders (Photos, video and audio on), the same
    /// pinned revision.
    public static let media = FindModelFile(
        fileName: "mmproj-embeddinggemma-2-Q8_0.gguf", sizeBytes: 554_821_056,
        sha256: "0531309472a6bc6f4b146c3736eee7795ec9db1903c25490d3c937490cef6dd8")
}

/// Where the search model stands.
public enum FindModelState: Sendable, Hashable {
    case notDownloaded
    case downloading(fraction: Double)
    case verifying
    case ready(URL)
    case failed(String)

    public var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

/// Downloads and checks the search model; resumes a partial download.
public actor FindModelStore {
    public nonisolated let directory: URL
    public nonisolated let file: FindModelFile
    private let allowedHosts: [String]
    private let configuration: URLSessionConfiguration
    private(set) var state: FindModelState
    private var task: Task<URL, Error>?

    public init(directory: URL, file: FindModelFile = .text, allowedHosts: [String] = DownloadSource.allowedHosts,
                configuration: URLSessionConfiguration = .default) {
        self.directory = directory
        self.file = file
        self.allowedHosts = allowedHosts
        self.configuration = configuration
        let final = directory.appending(path: file.fileName)
        state = FileManager.default.fileExists(atPath: final.path) ? .ready(final) : .notDownloaded
    }

    public nonisolated var modelURL: URL { directory.appending(path: file.fileName) }
    private var partURL: URL { directory.appending(path: file.fileName + ".part") }

    public func currentState() -> FindModelState { state }

    /// Downloads the model if it is missing; `progress` gets each new state.
    public func ensureDownloaded(progress: @escaping @Sendable (FindModelState) -> Void) async throws -> URL {
        if case .ready(let url) = state, FileManager.default.fileExists(atPath: url.path) { return url }
        if let task { return try await task.value }
        let work = Task { try await download(progress: progress) }
        task = work
        defer { task = nil }
        do {
            let url = try await work.value
            state = .ready(url)
            progress(state)
            return url
        } catch is CancellationError {
            state = .notDownloaded
            progress(state)
            throw CancellationError()
        } catch {
            let message = (error as? DownloadError)?.userMessage ?? error.localizedDescription
            state = .failed(message)
            progress(state)
            throw error
        }
    }

    public func cancel() {
        task?.cancel()
    }

    /// Removes the model file (Find turned off and its data deleted).
    public func delete() {
        task?.cancel()
        try? FileManager.default.removeItem(at: modelURL)
        try? FileManager.default.removeItem(at: partURL)
        state = .notDownloaded
    }

    private func download(progress: @escaping @Sendable (FindModelState) -> Void) async throws -> URL {
        let manager = FileManager.default
        let final = modelURL
        let part = partURL
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        if !manager.fileExists(atPath: part.path) {
            guard manager.createFile(atPath: part.path, contents: nil) else {
                throw DownloadError.file("cannot create \(part.lastPathComponent)")
            }
        }
        var existing = DownloadManager.fileSize(part)
        if existing > file.sizeBytes {
            try? manager.removeItem(at: part)
            _ = manager.createFile(atPath: part.path, contents: nil)
            existing = 0
        }
        if let available = DownloadManager.availableCapacity(directory),
           available < file.sizeBytes - existing + 200_000_000 {
            throw DownloadError.insufficientDisk(neededBytes: file.sizeBytes - existing,
                                                 availableBytes: available)
        }
        state = .downloading(fraction: Double(existing) / Double(file.sizeBytes))
        progress(state)
        var hasher = SHA256()
        try DownloadManager.hash(file: part, length: existing, into: &hasher)
        let total = file.sizeBytes
        let outcome = try await RangeDownload.run(
            url: file.url, file: part, offset: existing, hasher: hasher, allowedHosts: allowedHosts,
            configuration: configuration,
            progress: { received in progress(.downloading(fraction: min(1, Double(received) / Double(total)))) })
        state = .verifying
        progress(state)
        guard outcome.length == file.sizeBytes else {
            try? manager.removeItem(at: part)
            throw DownloadError.sizeMismatch(expected: file.sizeBytes, actual: outcome.length)
        }
        guard outcome.sha256 == file.sha256 else {
            try? manager.removeItem(at: part)
            throw DownloadError.checksumMismatch
        }
        try manager.moveItem(at: part, to: final)
        return final
    }
}
