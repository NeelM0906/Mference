import Darwin
import Foundation
import Synchronization

/// Where the bundled SearXNG runtime lives (built by
/// `App/Scripts/searxng/prepare.sh`, copied by `bundle.sh`).
public struct SearXNGRuntime: Sendable, Hashable {
    /// The directory with `python/`, `settings.yml`, `source/`.
    public var root: URL

    public init(root: URL) {
        self.root = root
    }

    public var python: URL { root.appending(path: "python/bin/python3") }
    public var settings: URL { root.appending(path: "settings.yml") }

    public var isComplete: Bool {
        FileManager.default.isExecutableFile(atPath: python.path) && FileManager.default.fileExists(atPath: settings.path)
    }

    /// `MFERENCE_SEARXNG_DIR`, then `Contents/Resources/SearXNG/runtime` in the
    /// app (bundle.sh copies `App/Resources/SearXNG`), then, for development
    /// builds run from `App/.build`, `App/Resources/SearXNG/runtime`.
    public static func locate(bundle: Bundle = .main,
                              environment: [String: String] = ProcessInfo.processInfo.environment) -> SearXNGRuntime? {
        var candidates: [URL] = []
        if let override = environment["MFERENCE_SEARXNG_DIR"] { candidates.append(URL(fileURLWithPath: override)) }
        if let resources = bundle.resourceURL { candidates.append(resources.appending(path: "SearXNG/runtime")) }
        var directory = bundle.executableURL?.deletingLastPathComponent()
        while let current = directory, current.path != "/" {
            if current.lastPathComponent == ".build" {
                candidates.append(current.deletingLastPathComponent().appending(path: "Resources/SearXNG/runtime"))
                break
            }
            directory = current.deletingLastPathComponent()
        }
        return candidates.map(SearXNGRuntime.init(root:)).first(where: \.isComplete)
    }
}

/// Runs the bundled SearXNG as its own process on `127.0.0.1:<random port>`
/// (spec 4.5): JSON output on, limiter off, a new secret per launch. Started
/// only while web search is on, health-checked, restarted when it dies (with
/// a growing pause after failures), stopped when web is turned off and on
/// quit. A stop always wins over a launch still in progress. Each launch gets
/// its own settings file naming a one-time instance token, which is checked
/// through `/config` so the app never talks to some other server that took
/// the port. A pid file lets the next launch clean up after a crash.
public actor SearXNGProcess {
    public enum State: Sendable, Hashable {
        case stopped
        case starting
        case running(URL)
        case failed(String)
    }

    public private(set) var state: State = .stopped
    private let runtime: SearXNGRuntime
    private let pidFile: URL
    private let startTimeout: Duration
    private var process: Process?
    private var launchSettings: URL?
    private var starting: Task<URL, any Error>?
    /// Set by `stop()` (web turned off), cleared by `warmUp()` (web turned
    /// on): while set, nothing starts SearXNG, not even a tool call that was
    /// already running when web was turned off.
    private var stoppedByPerson = false
    private var failures = 0
    private var retryAfter: ContinuousClock.Instant?
    /// The child's pid, readable synchronously for `terminateNow()`.
    private nonisolated let livePID = Mutex<pid_t?>(nil)

    public init(runtime: SearXNGRuntime,
                pidFile: URL = AppStorageLocation.applicationSupport.appending(path: "searxng.pid"),
                startTimeout: Duration = .seconds(20)) {
        self.runtime = runtime
        self.pidFile = pidFile
        self.startTimeout = startTimeout
        // A SearXNG left by a crashed app is cleaned up as soon as the app
        // starts, not only when web is next used, and so are settings files
        // a crash left in the temporary directory.
        Self.killStale(pidFile: pidFile, python: runtime.python)
        Self.removeStaleLaunchSettings()
    }

    /// The base URL of a healthy SearXNG, starting or restarting it if needed.
    /// `resumingAfterStop`: the caller has just checked that web search is
    /// on for it (a chat's toolbox, which re-reads Settings before every
    /// call), so a previous stop does not block this start.
    public func ensureRunning(resumingAfterStop: Bool = false) async throws -> URL {
        if resumingAfterStop { stoppedByPerson = false }
        guard !stoppedByPerson else { throw Self.webOff }
        if case .running(let url) = state, process?.isRunning == true, await Self.isHealthy(url) {
            return url
        }
        if let starting {
            do { return try await starting.value } catch is CancellationError where !Task.isCancelled {
                // The start this call joined was stopped (web turned off).
                throw Self.webOff
            }
        }
        if let retryAfter, ContinuousClock.now < retryAfter {
            throw WebError.searchUnavailable("SearXNG failed to start; trying again shortly.")
        }
        let task = Task { try await self.launch() }
        starting = task
        defer { starting = nil }
        do {
            let url = try await task.value
            failures = 0
            retryAfter = nil
            return url
        } catch is CancellationError where !Task.isCancelled {
            throw Self.webOff
        } catch {
            if !(error is CancellationError) {
                failures += 1
                retryAfter = ContinuousClock.now + .seconds(min(60, 2 << min(failures, 5)))
                state = .failed(Self.describe(error))
            }
            throw error
        }
    }

    static let webOff = WebError.searchUnavailable("web search is off for now.")

    /// Starts in the background (web turned on), ignoring failures; the next
    /// search reports them.
    public func warmUp() {
        stoppedByPerson = false
        Task { _ = try? await self.ensureRunning() }
    }

    /// Stops the process (web turned off, quit). Wins over a launch in
    /// progress, and clears the failure pause.
    public func stop() async {
        stoppedByPerson = true
        failures = 0
        retryAfter = nil
        starting?.cancel()
        if let process {
            self.process = nil
            await terminate(process)
        }
        cleanUpLaunchFiles()
        state = .stopped
    }

    /// Synchronous last resort for app termination, where nothing can be
    /// awaited: sends SIGTERM to the running SearXNG, if any.
    public nonisolated func terminateNow() {
        guard let pid = livePID.withLock({ $0 }) else { return }
        kill(pid, SIGTERM)
    }

    // MARK: Launch

    private func launch() async throws -> URL {
        if let process {
            self.process = nil
            await terminate(process)
        }
        try Task.checkCancellation()
        state = .starting
        let port = try Self.freeLoopbackPort()
        let base = URL(string: "http://127.0.0.1:\(port)")!
        let token = Self.randomSecret()
        let settings = try writeLaunchSettings(token: token)
        let process = Process()
        process.executableURL = runtime.python
        // -I: isolated (no user site, no PYTHON* env, no cwd on the path);
        // -B: never write bytecode into the signed app bundle.
        process.arguments = ["-I", "-B", "-m", "searx.webapp"]
        process.currentDirectoryURL = runtime.root
        process.environment = [
            "SEARXNG_SETTINGS_PATH": settings.path,
            "SEARXNG_PORT": String(port),
            "SEARXNG_BIND_ADDRESS": "127.0.0.1",
            "SEARXNG_SECRET": Self.randomSecret(),
            "SEARXNG_LIMITER": "false",
            "SEARXNG_PUBLIC_INSTANCE": "false",
            "PATH": "/usr/bin:/bin",
            "HOME": NSTemporaryDirectory(),
            "TMPDIR": NSTemporaryDirectory(),
            "LANG": "en_US.UTF-8",
        ]
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        let pid = process.processIdentifier
        process.terminationHandler = { [weak self] _ in
            Task { await self?.processEnded(pid: pid) }
        }
        self.process = process
        livePID.withLock { $0 = pid }
        try? FileManager.default.createDirectory(at: pidFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? String(pid).write(to: pidFile, atomically: true, encoding: .utf8)

        do {
            let deadline = ContinuousClock.now + startTimeout
            while true {
                try Task.checkCancellation()
                guard !stoppedByPerson else { throw CancellationError() }
                guard process.isRunning else {
                    throw WebError.searchUnavailable("SearXNG exited while starting (status \(process.terminationStatus)).")
                }
                if await Self.isOurs(base, token: token) { break }
                guard ContinuousClock.now < deadline else {
                    throw WebError.searchUnavailable("SearXNG did not answer within \(startTimeout).")
                }
                try await Task.sleep(for: .milliseconds(150))
            }
            // A stop may have arrived while the last health check ran.
            guard !stoppedByPerson, !Task.isCancelled else { throw CancellationError() }
        } catch {
            if self.process === process { self.process = nil }
            await terminate(process)
            cleanUpLaunchFiles()
            state = .stopped
            throw error
        }
        state = .running(base)
        return base
    }

    private func processEnded(pid: pid_t) {
        livePID.withLock { if $0 == pid { $0 = nil } }
        guard process?.processIdentifier == pid else { return }
        process = nil
        state = .stopped
        try? FileManager.default.removeItem(at: pidFile)
    }

    private func terminate(_ process: Process) async {
        defer {
            livePID.withLock { if $0 == process.processIdentifier { $0 = nil } }
            try? FileManager.default.removeItem(at: pidFile)
        }
        guard process.isRunning else { return }
        process.terminate()
        // Up to 2 s of grace after SIGTERM, even when the caller was
        // cancelled (Task.sleep would return at once), then SIGKILL.
        for _ in 0..<20 where process.isRunning {
            await Self.pause(milliseconds: 100)
        }
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
    }

    /// Our settings plus this launch's token as the instance name.
    private func writeLaunchSettings(token: String) throws -> URL {
        cleanUpLaunchFiles()
        var text = try String(contentsOf: runtime.settings, encoding: .utf8)
        let line = "instance_name: \"Mference web search\""
        guard text.contains(line) else {
            throw WebError.searchUnavailable("settings.yml has no instance_name line to mark this launch.")
        }
        text = text.replacingOccurrences(of: line, with: "instance_name: \"mference-\(token)\"")
        let url = FileManager.default.temporaryDirectory
            .appending(path: "\(Self.launchSettingsPrefix)\(UUID().uuidString).yml")
        try text.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        launchSettings = url
        return url
    }

    private func cleanUpLaunchFiles() {
        if let launchSettings { try? FileManager.default.removeItem(at: launchSettings) }
        launchSettings = nil
    }

    // MARK: Helpers

    static let launchSettingsPrefix = "mference-searxng-"

    /// Deletes launch settings older than a minute (a running SearXNG has
    /// read its own long before then).
    static func removeStaleLaunchSettings() {
        let directory = FileManager.default.temporaryDirectory
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: directory.path) else { return }
        let cutoff = Date.now.addingTimeInterval(-60)
        for name in names where name.hasPrefix(launchSettingsPrefix) && name.hasSuffix(".yml") {
            let url = directory.appending(path: name)
            let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
            if let modified, modified < cutoff { try? FileManager.default.removeItem(at: url) }
        }
    }

    /// Sleeps without being cut short by cancellation.
    static func pause(milliseconds: Int) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(milliseconds)) {
                continuation.resume()
            }
        }
    }

    static func isHealthy(_ base: URL) async -> Bool {
        await fetch(base.appending(path: "healthz")) != nil
    }

    /// Healthy, and `/config` names this launch's token: the server on the
    /// port is the one this launch started.
    static func isOurs(_ base: URL, token: String) async -> Bool {
        guard await isHealthy(base), let data = await fetch(base.appending(path: "config")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return (object["instance_name"] as? String) == "mference-\(token)"
    }

    static func fetch(_ url: URL) async -> Data? {
        var request = URLRequest(url: url)
        request.timeoutInterval = 1
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        guard let (data, response) = try? await session.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return data
    }

    /// Kills a SearXNG left running by a crashed app, only if the pid still
    /// belongs to our bundled Python.
    static func killStale(pidFile: URL, python: URL) {
        guard let text = try? String(contentsOf: pidFile, encoding: .utf8),
              let pid = pid_t(text.trimmingCharacters(in: .whitespacesAndNewlines)), pid > 0 else { return }
        defer { try? FileManager.default.removeItem(at: pidFile) }
        var buffer = [CChar](repeating: 0, count: Int(4 * MAXPATHLEN))
        guard proc_pidpath(pid, &buffer, UInt32(buffer.count)) > 0 else { return }
        let path = String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        let ours = python.resolvingSymlinksInPath().path
        guard URL(fileURLWithPath: path).resolvingSymlinksInPath().path == ours else { return }
        kill(pid, SIGTERM)
    }

    /// Asks the kernel for a free port on 127.0.0.1.
    static func freeLoopbackPort() throws -> UInt16 {
        let socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard socket >= 0 else { throw WebError.searchUnavailable("no socket") }
        defer { close(socket) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(socket, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { throw WebError.searchUnavailable("no free port") }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(socket, $0, &length) }
        }
        guard named == 0 else { throw WebError.searchUnavailable("no free port") }
        return UInt16(bigEndian: address.sin_port)
    }

    static func randomSecret() -> String {
        (0..<16).map { _ in String(format: "%02x", UInt8.random(in: .min ... .max)) }.joined()
    }

    static func describe(_ error: any Error) -> String {
        (error as? WebError)?.message ?? error.localizedDescription
    }
}
