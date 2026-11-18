import Darwin
import Foundation

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

    /// `MFERENCE_SEARXNG_DIR`, then `Contents/Resources/SearXNG` in the app,
    /// then (development builds run from `App/.build`) `App/.build/searxng/SearXNG`.
    public static func locate(bundle: Bundle = .main,
                              environment: [String: String] = ProcessInfo.processInfo.environment) -> SearXNGRuntime? {
        var candidates: [URL] = []
        if let override = environment["MFERENCE_SEARXNG_DIR"] { candidates.append(URL(fileURLWithPath: override)) }
        if let resources = bundle.resourceURL { candidates.append(resources.appending(path: "SearXNG")) }
        var directory = bundle.executableURL?.deletingLastPathComponent()
        while let current = directory, current.path != "/" {
            if current.lastPathComponent == ".build" {
                candidates.append(current.appending(path: "searxng/SearXNG"))
                break
            }
            directory = current.deletingLastPathComponent()
        }
        return candidates.map(SearXNGRuntime.init(root:)).first(where: \.isComplete)
    }
}

/// Runs the bundled SearXNG as its own process on `127.0.0.1:<random port>`
/// (spec 4.5): JSON output on, limiter off, a new secret per launch. Started
/// only while web search is on, health-checked, restarted when it dies, and
/// stopped on quit. A pid file lets the next launch clean up after a crash.
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
    private var starting: Task<URL, any Error>?
    private var restarts = 0
    /// Launch attempts in a row before giving up until the next request.
    public static let maxRestarts = 3

    public init(runtime: SearXNGRuntime,
                pidFile: URL = AppStorageLocation.applicationSupport.appending(path: "searxng.pid"),
                startTimeout: Duration = .seconds(20)) {
        self.runtime = runtime
        self.pidFile = pidFile
        self.startTimeout = startTimeout
    }

    /// The base URL of a healthy SearXNG, starting or restarting it if needed.
    public func ensureRunning() async throws -> URL {
        if case .running(let url) = state, process?.isRunning == true, await Self.isHealthy(url) {
            return url
        }
        if let starting { return try await starting.value }
        let task = Task { try await self.launch() }
        starting = task
        defer { starting = nil }
        do {
            let url = try await task.value
            restarts = 0
            return url
        } catch {
            restarts += 1
            state = .failed(Self.describe(error))
            throw error
        }
    }

    /// Starts in the background (web turned on), ignoring failures; the next
    /// search reports them.
    public func warmUp() {
        Task { _ = try? await self.ensureRunning() }
    }

    /// Stops the process (web turned off, quit).
    public func stop() async {
        starting?.cancel()
        guard let process else {
            state = .stopped
            return
        }
        self.process = nil
        state = .stopped
        await terminate(process)
        try? FileManager.default.removeItem(at: pidFile)
    }

    // MARK: Launch

    private func launch() async throws -> URL {
        guard restarts < Self.maxRestarts else {
            throw WebError.searchUnavailable("SearXNG did not start after \(Self.maxRestarts) tries.")
        }
        if let process {
            self.process = nil
            await terminate(process)
        }
        Self.killStale(pidFile: pidFile, python: runtime.python)
        state = .starting
        let port = try Self.freeLoopbackPort()
        let base = URL(string: "http://127.0.0.1:\(port)")!
        let process = Process()
        process.executableURL = runtime.python
        // -I: isolated (no user site, no PYTHON* env, no cwd on the path);
        // -B: never write bytecode into the signed app bundle.
        process.arguments = ["-I", "-B", "-m", "searx.webapp"]
        process.currentDirectoryURL = runtime.root
        process.environment = [
            "SEARXNG_SETTINGS_PATH": runtime.settings.path,
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
        try? FileManager.default.createDirectory(at: pidFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? String(process.processIdentifier).write(to: pidFile, atomically: true, encoding: .utf8)

        let deadline = ContinuousClock.now + startTimeout
        while ContinuousClock.now < deadline {
            try Task.checkCancellation()
            guard process.isRunning else {
                throw WebError.searchUnavailable("SearXNG exited while starting (status \(process.terminationStatus)).")
            }
            if await Self.isHealthy(base) {
                state = .running(base)
                return base
            }
            try await Task.sleep(for: .milliseconds(150))
        }
        await terminate(process)
        self.process = nil
        throw WebError.searchUnavailable("SearXNG did not answer within \(startTimeout).")
    }

    private func processEnded(pid: pid_t) {
        guard process?.processIdentifier == pid else { return }
        process = nil
        state = .stopped
        try? FileManager.default.removeItem(at: pidFile)
    }

    // MARK: Helpers

    static func isHealthy(_ base: URL) async -> Bool {
        var request = URLRequest(url: base.appending(path: "healthz"))
        request.timeoutInterval = 1
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        guard let (_, response) = try? await session.data(for: request) else { return false }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    private func terminate(_ process: Process) async {
        guard process.isRunning else { return }
        process.terminate()
        for _ in 0..<20 where process.isRunning {
            try? await Task.sleep(for: .milliseconds(100))
        }
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
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
        (0..<32).map { _ in String(format: "%02x", UInt8.random(in: .min ... .max)) }.joined()
    }

    static func describe(_ error: any Error) -> String {
        (error as? WebError)?.message ?? error.localizedDescription
    }
}
