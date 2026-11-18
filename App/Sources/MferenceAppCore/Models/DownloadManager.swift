import CryptoKit
import Foundation
import Synchronization

/// Where one model's download stands.
public enum DownloadState: Sendable, Hashable {
    case notStarted
    case downloading(receivedBytes: UInt64, totalBytes: UInt64)
    /// Stopped with the bytes so far kept; `download` continues from here.
    case paused(receivedBytes: UInt64, totalBytes: UInt64)
    /// Every byte is in; the file is being checked.
    case verifying
    case installed(URL)
    case failed(String)

    /// 0...1 while bytes are known.
    public var fraction: Double? {
        switch self {
        case .downloading(let received, let total), .paused(let received, let total):
            total > 0 ? min(1, Double(received) / Double(total)) : nil
        case .verifying, .installed: 1
        case .notStarted, .failed: nil
        }
    }
}

public enum DownloadError: Error, Sendable, Hashable {
    case noSource(AppModel)
    case approvalMismatch
    case offline
    case notOffered(String)
    case insufficientDisk(neededBytes: UInt64, availableBytes: UInt64)
    case hostNotAllowed(String)
    case http(Int)
    case sizeMismatch(expected: UInt64, actual: UInt64)
    case checksumMismatch
    case file(String)
    case network(String)

    public var userMessage: String {
        switch self {
        case .noSource(let model): "\(model.name) cannot be downloaded from the app."
        case .approvalMismatch: "The download you approved does not match this model. Try again."
        case .offline: "Offline mode is on. Turn it off in Settings to download."
        case .notOffered(let reason): reason
        case .insufficientDisk(let needed, let available):
            "Not enough free space: \(Self.gigabytes(needed)) needed, \(Self.gigabytes(available)) free."
        case .hostNotAllowed(let host): "The download was redirected to \(host), which Mference does not contact."
        case .http(let code): "The download server answered \(code). Try again later."
        case .sizeMismatch: "The downloaded file has the wrong size. It was removed; try again."
        case .checksumMismatch: "The downloaded file did not match its checksum. It was removed; try again."
        case .file(let message): "The file could not be saved: \(message)"
        case .network(let message): "The download stopped: \(message)"
        }
    }

    static func gigabytes(_ bytes: UInt64) -> String {
        String(format: "%.1f GB", Double(bytes) / 1e9)
    }
}

/// Downloads the GGUF product models into `~/llm-models/gguf` (spec 4.4,
/// decision L4).
///
/// - **Approval first.** `download` takes a `DownloadApproval` the UI makes
///   from the confirmation dialog; nothing starts without one. Offline mode
///   and the memory gate (Deep only above 16 GB) refuse before any request.
/// - **Resumable.** Bytes go to `<file>.part`; a later `download` (also
///   after a relaunch) asks for the rest with an HTTP `Range` request and
///   hashes the bytes already there first.
/// - **Verified.** Size and SHA-256 are checked before the file is moved to
///   its final name, so the engine never sees a partial or wrong file; a
///   mismatch removes it.
/// - **Bounded memory.** Bytes stream to disk and into the hash as they
///   arrive; existing bytes are hashed in fixed-size chunks.
/// - **Allowed hosts only.** Redirects to any host outside `allowedHosts`
///   are refused (G-A8).
public actor DownloadManager {
    public struct Policy: Sendable {
        /// Offline mode on: no download may start.
        public var isOffline: @Sendable () async -> Bool
        public var memoryFit: MemoryFit

        public init(isOffline: @escaping @Sendable () async -> Bool = { false },
                    memoryFit: MemoryFit = MemoryFit()) {
            self.isOffline = isOffline
            self.memoryFit = memoryFit
        }
    }

    public nonisolated let destination: URL
    private let sources: [AppModel: DownloadSource]
    private let allowedHosts: [String]
    private let policy: Policy
    private let sessionConfiguration: URLSessionConfiguration
    private let availableBytes: @Sendable (URL) -> UInt64?
    private let onInstalled: @Sendable (AppModel) async -> Void

    private var states: [AppModel: DownloadState] = [:]
    private var active: [AppModel: Task<URL, Error>] = [:]
    private var subscribers: [UUID: AsyncStream<[AppModel: DownloadState]>.Continuation] = [:]

    /// Bytes read per step when hashing what is already on disk.
    static let hashChunkBytes = 8 << 20
    /// Kept free beyond the download itself.
    static let diskMarginBytes: UInt64 = 1 << 30

    public init(destination: URL = DownloadSource.defaultDestination,
                sources: [AppModel: DownloadSource] = DownloadSource.product,
                allowedHosts: [String] = DownloadSource.allowedHosts,
                policy: Policy = Policy(),
                sessionConfiguration: URLSessionConfiguration = .ephemeral,
                availableBytes: @escaping @Sendable (URL) -> UInt64? = DownloadManager.availableCapacity,
                onInstalled: @escaping @Sendable (AppModel) async -> Void = { _ in }) {
        self.destination = destination
        self.sources = sources
        self.allowedHosts = allowedHosts
        self.policy = policy
        self.sessionConfiguration = sessionConfiguration
        self.availableBytes = availableBytes
        self.onInstalled = onInstalled
    }

    public nonisolated func source(for model: AppModel) -> DownloadSource? { sources[model] }

    // MARK: State

    /// The state, including a partial file left by an earlier session.
    public func state(of model: AppModel) -> DownloadState {
        if let state = states[model] { return state }
        guard let source = sources[model] else { return .notStarted }
        if FileManager.default.fileExists(atPath: finalURL(source).path) {
            return .installed(finalURL(source))
        }
        let partial = partialBytes(source)
        return partial > 0 ? .paused(receivedBytes: partial, totalBytes: source.sizeBytes) : .notStarted
    }

    /// Every downloadable model's state; yields the current states first.
    public func updates() -> AsyncStream<[AppModel: DownloadState]> {
        let (stream, continuation) = AsyncStream<[AppModel: DownloadState]>.makeStream(
            bufferingPolicy: .bufferingNewest(1))
        let id = UUID()
        subscribers[id] = continuation
        continuation.yield(allStates())
        continuation.onTermination = { [weak self] _ in
            Task { await self?.removeSubscriber(id) }
        }
        return stream
    }

    private func removeSubscriber(_ id: UUID) { subscribers[id] = nil }

    private func allStates() -> [AppModel: DownloadState] {
        Dictionary(uniqueKeysWithValues: sources.keys.map { ($0, state(of: $0)) })
    }

    private func setState(_ state: DownloadState, for model: AppModel) {
        guard states[model] != state else { return }
        states[model] = state
        let snapshot = allStates()
        for continuation in subscribers.values { continuation.yield(snapshot) }
    }

    // MARK: Control

    /// Downloads (or continues) the approved model and returns the installed
    /// file. Throws `CancellationError` when paused or discarded, and a
    /// `DownloadError` otherwise. A second call while one runs joins it: the
    /// running download is reserved before the first suspension, so two
    /// transfers never write one partial file.
    @discardableResult
    public func download(_ approval: DownloadApproval) async throws -> URL {
        let model = approval.source.model
        guard let source = sources[model] else { throw DownloadError.noSource(model) }
        guard approval.source == source else { throw DownloadError.approvalMismatch }
        if let running = active[model] { return try await running.value }

        let task = Task { try await self.checkPolicyAndRun(source) }
        active[model] = task
        defer { active[model] = nil }
        do {
            let url = try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
            // The engine sees the file before the row says installed.
            await onInstalled(model)
            setState(.installed(url), for: model)
            return url
        } catch is CancellationError {
            if case .failed = states[model] {} else {
                let received = partialBytes(source)
                setState(received > 0 ? .paused(receivedBytes: received, totalBytes: source.sizeBytes)
                                      : .notStarted, for: model)
            }
            throw CancellationError()
        } catch {
            let downloadError = error as? DownloadError ?? .network(String(describing: error))
            setState(.failed(downloadError.userMessage), for: model)
            throw downloadError
        }
    }

    private func checkPolicyAndRun(_ source: DownloadSource) async throws -> URL {
        if await policy.isOffline() { throw DownloadError.offline }
        if let reason = policy.memoryFit.notOfferedReason(source.model) { throw DownloadError.notOffered(reason) }
        return try await run(source)
    }

    /// Stops the download and keeps the bytes so far.
    public func pause(_ model: AppModel) {
        active[model]?.cancel()
    }

    /// Pauses every running download, keeping the bytes so far: offline
    /// mode was turned on. Each shows as paused and continues from where it
    /// stopped once the person starts it again.
    public func pauseAll() {
        for task in active.values { task.cancel() }
    }

    /// Stops the download and removes the partial file.
    public func discard(_ model: AppModel) async {
        if let task = active[model] {
            task.cancel()
            _ = try? await task.value
        }
        guard let source = sources[model] else { return }
        try? FileManager.default.removeItem(at: partURL(source))
        try? FileManager.default.removeItem(at: partSourceURL(source))
        states[model] = nil
        setState(state(of: model), for: model)
    }

    // MARK: Download

    private func finalURL(_ source: DownloadSource) -> URL {
        destination.appending(path: source.fileName)
    }

    private func partURL(_ source: DownloadSource) -> URL {
        destination.appending(path: source.fileName + ".part")
    }

    /// Names the source the partial file belongs to (its SHA-256), so bytes
    /// of another revision are never resumed.
    private func partSourceURL(_ source: DownloadSource) -> URL {
        destination.appending(path: source.fileName + ".part.sha256")
    }

    private func partBelongs(to source: DownloadSource) -> Bool {
        let recorded = try? String(contentsOf: partSourceURL(source), encoding: .utf8)
        return recorded?.trimmingCharacters(in: .whitespacesAndNewlines) == source.sha256
    }

    /// Bytes of a partial file of this source, 0 when none or another one.
    private func partialBytes(_ source: DownloadSource) -> UInt64 {
        partBelongs(to: source) ? Self.fileSize(partURL(source)) : 0
    }

    private func run(_ source: DownloadSource) async throws -> URL {
        let final = finalURL(source)
        let part = partURL(source)
        let manager = FileManager.default
        if manager.fileExists(atPath: final.path) { return final }
        do {
            try manager.createDirectory(at: destination, withIntermediateDirectories: true)
            // Bytes left by another revision of this file start over.
            if manager.fileExists(atPath: part.path), !partBelongs(to: source) {
                try manager.removeItem(at: part)
            }
            if !manager.fileExists(atPath: part.path) {
                guard manager.createFile(atPath: part.path, contents: nil) else {
                    throw DownloadError.file("cannot create \(part.lastPathComponent)")
                }
            }
            try Data(source.sha256.utf8).write(to: partSourceURL(source), options: .atomic)
        } catch let error as DownloadError {
            throw error
        } catch {
            throw DownloadError.file(error.localizedDescription)
        }

        var existing = Self.fileSize(part)
        if existing > source.sizeBytes {
            try? manager.removeItem(at: part)
            _ = manager.createFile(atPath: part.path, contents: nil)
            existing = 0
        }
        let needed = source.sizeBytes - existing + Self.diskMarginBytes
        if let available = availableBytes(destination), available < needed {
            throw DownloadError.insufficientDisk(neededBytes: needed, availableBytes: available)
        }

        setState(.downloading(receivedBytes: existing, totalBytes: source.sizeBytes), for: source.model)
        // Hash what is already on disk off the actor, so pause and status
        // stay responsive while a large partial file is read back.
        let resumeLength = existing
        let hashing = Task.detached {
            var hasher = SHA256()
            try Self.hash(file: part, length: resumeLength, into: &hasher)
            return hasher
        }
        let hasher = try await withTaskCancellationHandler {
            try await hashing.value
        } onCancel: {
            hashing.cancel()
        }
        try Task.checkCancellation()

        let model = source.model
        let total = source.sizeBytes
        let outcome = try await RangeDownload.run(
            url: source.url, file: part, offset: existing, hasher: hasher,
            allowedHosts: allowedHosts, configuration: sessionConfiguration,
            progress: { received in
                Task { await self.reportProgress(model: model, received: received, total: total) }
            })

        setState(.verifying, for: source.model)
        guard outcome.length == source.sizeBytes else {
            try? manager.removeItem(at: part)
            throw DownloadError.sizeMismatch(expected: source.sizeBytes, actual: outcome.length)
        }
        guard outcome.sha256 == source.sha256 else {
            try? manager.removeItem(at: part)
            throw DownloadError.checksumMismatch
        }
        do {
            try manager.moveItem(at: part, to: final)
        } catch {
            throw DownloadError.file(error.localizedDescription)
        }
        try? manager.removeItem(at: partSourceURL(source))
        return final
    }

    private func reportProgress(model: AppModel, received: UInt64, total: UInt64) {
        guard case .downloading = states[model] else { return }
        setState(.downloading(receivedBytes: received, totalBytes: total), for: model)
    }

    // MARK: Files

    static func fileSize(_ url: URL) -> UInt64 {
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.uint64Value
        return size ?? 0
    }

    /// Hashes the first `length` bytes of `file` in fixed-size chunks.
    static func hash(file: URL, length: UInt64, into hasher: inout SHA256) throws {
        guard length > 0 else { return }
        do {
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            var remaining = length
            while remaining > 0 {
                try Task.checkCancellation()
                let count = Int(min(UInt64(hashChunkBytes), remaining))
                guard let data = try handle.read(upToCount: count), !data.isEmpty else {
                    throw DownloadError.file("partial file is shorter than expected")
                }
                hasher.update(data: data)
                remaining -= UInt64(data.count)
            }
        } catch let error as DownloadError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw DownloadError.file(error.localizedDescription)
        }
    }

    public static let availableCapacity: @Sendable (URL) -> UInt64? = { url in
        var probe = url
        while !FileManager.default.fileExists(atPath: probe.path), probe.pathComponents.count > 1 {
            probe = probe.deletingLastPathComponent()
        }
        let values = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage.map { UInt64(max(0, $0)) }
    }
}

/// One HTTP GET from `offset` to the end, appended to `file` and hashed as
/// it arrives, on the delegate's serial queue so memory stays bounded by
/// what URLSession buffers.
enum RangeDownload {
    struct Outcome: Sendable {
        /// Bytes in the file at the end.
        var length: UInt64
        var sha256: String
    }

    static func run(url: URL, file: URL, offset: UInt64, hasher: SHA256, allowedHosts: [String],
                    configuration: URLSessionConfiguration,
                    progress: @escaping @Sendable (UInt64) -> Void) async throws -> Outcome {
        guard DownloadSource.isAllowed(host: url.host(), allowedHosts: allowedHosts) else {
            throw DownloadError.hostNotAllowed(url.host() ?? url.absoluteString)
        }
        let handle: FileHandle
        do {
            handle = try FileHandle(forWritingTo: file)
            try handle.seek(toOffset: offset)
        } catch {
            throw DownloadError.file(error.localizedDescription)
        }
        let delegate = Delegate(handle: handle, offset: offset, hasher: hasher, allowedHosts: allowedHosts,
                                progress: progress)
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: queue)
        defer {
            session.invalidateAndCancel()
            try? handle.close()
        }
        var request = URLRequest(url: url)
        if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        let task = session.dataTask(with: request)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                delegate.setContinuation(continuation)
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    /// Mutable state lives in `state` and is touched on the session's serial
    /// delegate queue; the lock makes that explicit for the checker.
    final class Delegate: NSObject, URLSessionDataDelegate, Sendable {
        private struct State {
            var handle: FileHandle
            var length: UInt64
            var hasher: SHA256
            var continuation: CheckedContinuation<Outcome, Error>?
            /// The outcome when the task finished before the continuation was
            /// set (a pause right at the start).
            var pending: Result<Outcome, Error>?
            var failure: (any Error)?
            var lastReported: UInt64 = 0
        }

        private let state: Mutex<State>
        private let allowedHosts: [String]
        private let progress: @Sendable (UInt64) -> Void
        /// Report at most every this many bytes.
        private let reportStride: UInt64 = 4 << 20

        init(handle: FileHandle, offset: UInt64, hasher: SHA256, allowedHosts: [String],
             progress: @escaping @Sendable (UInt64) -> Void) {
            state = Mutex(State(handle: handle, length: offset, hasher: hasher))
            self.allowedHosts = allowedHosts
            self.progress = progress
        }

        func setContinuation(_ continuation: CheckedContinuation<Outcome, Error>) {
            let pending = state.withLock { s -> Result<Outcome, Error>? in
                if let pending = s.pending {
                    s.pending = nil
                    return pending
                }
                s.continuation = continuation
                return nil
            }
            if let pending { continuation.resume(with: pending) }
        }

        /// The first byte of a 206 body, from `Content-Range: bytes a-b/n`.
        static func rangeStart(_ header: String?) -> UInt64? {
            guard let header, header.lowercased().hasPrefix("bytes "),
                  let dash = header.firstIndex(of: "-") else { return nil }
            return UInt64(header[header.index(header.startIndex, offsetBy: 6)..<dash])
        }

        private func fail(_ error: any Error) {
            state.withLock { if $0.failure == nil { $0.failure = error } }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
            let host = request.url?.host()
            if DownloadSource.isAllowed(host: host, allowedHosts: allowedHosts) {
                completionHandler(request)
            } else {
                fail(DownloadError.hostNotAllowed(host ?? "an unknown host"))
                completionHandler(nil)
                task.cancel()
            }
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
            let http = response as? HTTPURLResponse
            let code = http?.statusCode ?? 0
            let contentRange = http?.value(forHTTPHeaderField: "Content-Range")
            let disposition: URLSession.ResponseDisposition = state.withLock { s in
                switch code {
                case 206:
                    // The body must continue exactly where the file ends.
                    guard Self.rangeStart(contentRange) == s.length else {
                        s.failure = DownloadError.network("the server sent a different part of the file")
                        return .cancel
                    }
                    return .allow
                case 200:
                    // The server ignored the range: start the file over.
                    do {
                        try s.handle.truncate(atOffset: 0)
                        s.length = 0
                        s.hasher = SHA256()
                        return .allow
                    } catch {
                        s.failure = DownloadError.file(error.localizedDescription)
                        return .cancel
                    }
                case 416:
                    // Nothing left to send: the file is already complete.
                    return .cancel
                default:
                    s.failure = DownloadError.http(code)
                    return .cancel
                }
            }
            if code == 416 { state.withLock { $0.failure = RangeComplete() } }
            completionHandler(disposition)
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            let report: UInt64? = state.withLock { s in
                guard s.failure == nil else { return nil }
                do {
                    try s.handle.write(contentsOf: data)
                } catch {
                    s.failure = DownloadError.file(error.localizedDescription)
                    dataTask.cancel()
                    return nil
                }
                s.hasher.update(data: data)
                s.length += UInt64(data.count)
                guard s.length - s.lastReported >= reportStride else { return nil }
                s.lastReported = s.length
                return s.length
            }
            if let report { progress(report) }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
            let (continuation, result) = state.withLock { s -> (CheckedContinuation<Outcome, Error>?, Result<Outcome, Error>) in
                defer { s.continuation = nil }
                let finished: Result<Outcome, Error>
                if s.failure is RangeComplete || (s.failure == nil && error == nil) {
                    do {
                        try s.handle.synchronize()
                        let digest = s.hasher.finalize().map { String(format: "%02x", $0) }.joined()
                        finished = .success(Outcome(length: s.length, sha256: digest))
                    } catch {
                        finished = .failure(DownloadError.file(error.localizedDescription))
                    }
                } else if let failure = s.failure {
                    finished = .failure(failure)
                } else if let urlError = error as? URLError, urlError.code == .cancelled {
                    finished = .failure(CancellationError())
                } else {
                    finished = .failure(DownloadError.network(error?.localizedDescription ?? "unknown error"))
                }
                if s.continuation == nil { s.pending = finished }
                return (s.continuation, finished)
            }
            if case .success(let outcome) = result { progress(outcome.length) }
            continuation?.resume(with: result)
        }
    }

    /// The server answered 416: the bytes on disk are the whole file.
    struct RangeComplete: Error {}
}
