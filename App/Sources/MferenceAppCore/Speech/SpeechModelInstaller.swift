import Foundation
import Synchronization

/// Downloads, verifies and removes the speech packages (Notes spec 4.7, 7).
///
/// - **Pinned, file by file.** Every file of a package comes from
///   `SpeechModelPins` (repository, revision, size, SHA-256) and is checked
///   before it is moved into place; a mismatch deletes that file and fails
///   with `SpeechError.verificationFailed(file:)`. A package counts as
///   installed only once every file is in place and its marker is written.
/// - **Resumable.** Partial files stay as `<file>.part` when paused, the app
///   quits, or Offline mode is turned on; the next `install` continues them.
///   `cancel` deletes them.
/// - **Local and allowed hosts only.** Files go under `root`
///   (`~/Library/Application Support/Mference/Speech/<repo>/<revision>/`);
///   transfers reach `SpeechModelPins.allowedHosts` only (G-A8). Nothing
///   here loads a model.
public actor SpeechModelInstaller {
    public struct Policy: Sendable {
        /// Offline mode on: no download may start.
        public var isOffline: @Sendable () async -> Bool
        /// Free bytes on the volume holding a path; nil when unknown.
        public var availableBytes: @Sendable (URL) -> UInt64?

        public init(isOffline: @escaping @Sendable () async -> Bool = { false },
                    availableBytes: @escaping @Sendable (URL) -> UInt64? = DownloadManager.availableCapacity) {
            self.isOffline = isOffline
            self.availableBytes = availableBytes
        }
    }

    /// Kept free beyond the download itself.
    static let diskMarginBytes: UInt64 = 1 << 30

    public nonisolated let root: URL
    private let packages: [SpeechPackageID: SpeechModelPackage]
    private let allowedHosts: [String]
    private let policy: Policy
    private let configuration: URLSessionConfiguration
    private let urlFor: @Sendable (SpeechModelSource, SpeechModelFile) -> URL
    private struct Download {
        var token: UUID
        var task: Task<Void, Error>
        /// Paused or cancelled: a new install waits for it instead of joining.
        var stopped = false
    }
    private var active: [SpeechPackageID: Download] = [:]
    private let observers = ProgressObservers()

    public static var defaultRoot: URL {
        AppStorageLocation.applicationSupport.appending(path: "Speech", directoryHint: .isDirectory)
    }

    /// - Parameter urlFor: where each file is fetched from; tests point it
    ///   at a loopback server.
    public init(root: URL = SpeechModelInstaller.defaultRoot,
                packages: [SpeechPackageID: SpeechModelPackage] = [.speech: SpeechModelPins.speech,
                                                                   .speakerLabels: SpeechModelPins.speakerLabels],
                allowedHosts: [String] = SpeechModelPins.allowedHosts,
                policy: Policy = Policy(),
                configuration: URLSessionConfiguration = .ephemeral,
                urlFor: @escaping @Sendable (SpeechModelSource, SpeechModelFile) -> URL = { $0.url(for: $1) }) {
        self.root = root
        self.packages = packages
        self.allowedHosts = allowedHosts
        self.policy = policy
        self.configuration = configuration
        self.urlFor = urlFor
    }

    public nonisolated func package(_ id: SpeechPackageID) -> SpeechModelPackage? { packages[id] }

    /// The installed folder of one pinned source (`root/<repo>/<revision>`).
    public nonisolated func directory(of source: SpeechModelSource) -> URL {
        root.appending(path: Self.installPrefix(of: source), directoryHint: .isDirectory)
    }

    static func installPrefix(of source: SpeechModelSource) -> String {
        let repo = source.repo.split(separator: "/").last.map(String.init) ?? source.repo
        return "\(repo)/\(source.revision)"
    }

    nonisolated func destination(of file: SpeechModelFile) -> URL {
        root.appending(path: file.installPath)
    }

    nonisolated func marker(_ id: SpeechPackageID) -> URL {
        root.appending(path: ".installed/\(id.rawValue).json")
    }

    // MARK: State

    /// Every file in place with its pinned size, and the marker written.
    public nonisolated func isInstalled(_ id: SpeechPackageID) -> Bool {
        guard let package = packages[id], FileManager.default.fileExists(atPath: marker(id).path) else { return false }
        return package.sources.allSatisfy { source in
            source.files.allSatisfy { DownloadManager.fileSize(destination(of: $0)) == UInt64($0.size) }
        }
    }

    /// Bytes on disk toward the package: finished files plus partial ones.
    public nonisolated func bytesOnDisk(_ id: SpeechPackageID) -> Int64 {
        guard let package = packages[id] else { return 0 }
        var total: Int64 = 0
        for source in package.sources {
            for file in source.files {
                let final = destination(of: file)
                let size = DownloadManager.fileSize(final)
                total += Int64(size == UInt64(file.size) ? size
                               : min(DownloadManager.fileSize(final.appendingPathExtension("part")), UInt64(file.size)))
            }
        }
        return total
    }

    /// 0...1 of the package's bytes on disk.
    public nonisolated func progress(_ id: SpeechPackageID) -> Double {
        guard let total = packages[id]?.totalBytes, total > 0 else { return 0 }
        return min(1, Double(bytesOnDisk(id)) / Double(total))
    }

    // MARK: Control

    /// Downloads every missing file and verifies it, then writes the marker.
    /// A second call while one runs joins it. Throws `SpeechError.offline`,
    /// `.lowDiskSpace`, `.verificationFailed`, `.downloadFailed`, and
    /// `CancellationError` when paused or cancelled.
    public func install(_ id: SpeechPackageID,
                        progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws {
        guard let package = packages[id] else { throw SpeechError.failed("unknown package \(id.rawValue)") }
        let observer = observers.add(id, progress)
        defer { observers.remove(id, observer) }
        // A download that was paused or cancelled may still be winding
        // down: wait for it (two transfers never write one partial file),
        // then start afresh instead of joining it.
        while let running = active[id], running.stopped {
            _ = try? await running.task.value
            // Finished: it no longer counts, whether or not its own caller
            // has run its cleanup yet.
            if active[id]?.token == running.token { active[id] = nil }
        }
        if let running = active[id] { return try await running.task.value }
        let token = UUID()
        let observers = observers
        let task = Task { try await self.run(id, package) { fraction in observers.report(id, fraction) } }
        active[id] = Download(token: token, task: task)
        defer { if active[id]?.token == token { active[id] = nil } }
        try await task.value
    }

    /// Stops the download and keeps the bytes so far.
    public func pause(_ id: SpeechPackageID) {
        stop(id)
    }

    /// Stops every download (Offline mode turned on); bytes are kept.
    public func pauseAll() {
        for id in active.keys { stop(id) }
    }

    private func stop(_ id: SpeechPackageID) {
        guard var running = active[id] else { return }
        running.stopped = true
        running.task.cancel()
        active[id] = running
    }

    /// Stops the download and deletes everything it wrote.
    public func cancel(_ id: SpeechPackageID) async {
        if let running = active[id] {
            stop(id)
            _ = try? await running.task.value
        }
        guard !isInstalled(id) else { return }
        deleteFiles(id)
    }

    /// Deletes an installed (or partial) package.
    public func remove(_ id: SpeechPackageID) async throws {
        if let running = active[id] {
            stop(id)
            _ = try? await running.task.value
        }
        deleteFiles(id)
    }

    private func deleteFiles(_ id: SpeechPackageID) {
        guard let package = packages[id] else { return }
        let manager = FileManager.default
        try? manager.removeItem(at: marker(id))
        for source in package.sources {
            let folder = directory(of: source)
            try? manager.removeItem(at: folder)
            // Drop the repository folder too once no revision is left.
            let parent = folder.deletingLastPathComponent()
            if (try? manager.contentsOfDirectory(atPath: parent.path))?.isEmpty == true {
                try? manager.removeItem(at: parent)
            }
        }
    }

    // MARK: Download

    private func run(_ id: SpeechPackageID, _ package: SpeechModelPackage,
                     progress: @escaping @Sendable (Double) -> Void) async throws {
        if isInstalled(id) {
            progress(1)
            return
        }
        if await policy.isOffline() { throw SpeechError.offline }
        let total = UInt64(package.totalBytes)
        let onDisk = UInt64(bytesOnDisk(id))
        let needed = total - min(total, onDisk) + Self.diskMarginBytes
        if let available = policy.availableBytes(root), available < needed {
            throw SpeechError.lowDiskSpace(availableBytes: Int64(available))
        }
        let meter = ProgressMeter(total: total, report: progress)
        for source in package.sources {
            for file in source.files {
                try Task.checkCancellation()
                let pinned = DownloadManager.PinnedFile(url: urlFor(source, file), destination: destination(of: file),
                                                        size: UInt64(file.size), sha256: file.sha256)
                let key = file.installPath
                do {
                    try await DownloadManager.fetchPinned(pinned, allowedHosts: allowedHosts,
                                                          configuration: configuration) { bytes in
                        meter.update(key, bytes: bytes)
                    }
                } catch is CancellationError {
                    throw CancellationError()
                } catch DownloadError.sizeMismatch, DownloadError.checksumMismatch {
                    throw SpeechError.verificationFailed(file: file.repoPath)
                } catch DownloadError.offline {
                    throw SpeechError.offline
                } catch let error as DownloadError {
                    throw SpeechError.downloadFailed(error.userMessage)
                } catch {
                    throw SpeechError.downloadFailed(error.localizedDescription)
                }
                meter.update(key, bytes: UInt64(file.size))
            }
        }
        // Every file is verified and in place: mark the package installed.
        do {
            let marker = marker(id)
            try FileManager.default.createDirectory(at: marker.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            let revisions = package.sources.map { ["repo": $0.repo, "revision": $0.revision] }
            let data = try JSONSerialization.data(withJSONObject: ["package": id.rawValue, "sources": revisions],
                                                  options: [.sortedKeys])
            try data.write(to: marker, options: .atomic)
        } catch {
            throw SpeechError.downloadFailed(error.localizedDescription)
        }
        guard isInstalled(id) else { throw SpeechError.downloadFailed("files went missing after download") }
        progress(1)
    }

    /// Everyone waiting on a package's download hears its progress, joiners
    /// included, in order and before `install` returns.
    final class ProgressObservers: Sendable {
        private let observers = Mutex<[SpeechPackageID: [UUID: @Sendable (Double) -> Void]]>([:])

        func add(_ id: SpeechPackageID, _ observer: @escaping @Sendable (Double) -> Void) -> UUID {
            let key = UUID()
            observers.withLock { $0[id, default: [:]][key] = observer }
            return key
        }

        func remove(_ id: SpeechPackageID, _ key: UUID) {
            observers.withLock { _ = $0[id]?.removeValue(forKey: key) }
        }

        func report(_ id: SpeechPackageID, _ fraction: Double) {
            for observer in observers.withLock({ Array(($0[id] ?? [:]).values) }) { observer(fraction) }
        }
    }

    /// Sums bytes across files and reports the package fraction.
    final class ProgressMeter: Sendable {
        private let state = Mutex<(bytes: [String: UInt64], lastReported: Double)>(([:], -1))
        private let total: UInt64
        private let report: @Sendable (Double) -> Void

        init(total: UInt64, report: @escaping @Sendable (Double) -> Void) {
            self.total = total
            self.report = report
        }

        func update(_ key: String, bytes: UInt64) {
            let fraction: Double? = state.withLock { s in
                s.bytes[key] = bytes
                let done = s.bytes.values.reduce(0, +)
                let fraction = total > 0 ? min(1, Double(done) / Double(total)) : 1
                // At most every 0.5 %, so the status stream stays quiet.
                guard fraction - s.lastReported >= 0.005 || fraction == 1 else { return nil }
                s.lastReported = fraction
                return fraction
            }
            if let fraction { report(fraction) }
        }
    }
}
