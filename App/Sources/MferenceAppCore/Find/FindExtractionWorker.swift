import Foundation
import Synchronization

/// Reads files for the indexer.
public protocol FindExtracting: Sendable {
    func extract(_ url: URL, kind: FindKind, options: FindExtractOptions) async -> FindExtraction
    /// Decodes media pieces of `url` into temporary files for the encoders
    /// (`FindMediaData`), in order; nil where a piece could not be decoded.
    /// The caller deletes the files.
    func prepare(_ url: URL, kind: FindKind, pieces: [FindMediaPiece]) async -> [URL?]
    func shutdown() async
    /// Files the reader can work on at once (the indexer keeps this many in
    /// flight).
    var parallelism: Int { get }
    /// Work harder (a large backlog, such as the first index) or go back to
    /// background priority.
    func setHurry(_ hurry: Bool) async
    /// The indexer has nothing left for now: free what is not needed to
    /// start again quickly.
    func relax() async
}

extension FindExtracting {
    public var parallelism: Int { 1 }
    public func setHurry(_ hurry: Bool) async {}
    public func relax() async {}
}

/// Where prepared media waits for the encoders (one folder per user, shared
/// by the app and its helper). `ready` holds pictures prepared while they
/// were read, until they are learned.
public enum FindMediaFolder {
    public static let url = FileManager.default.temporaryDirectory
        .appending(path: "MferenceFindMedia", directoryHint: .isDirectory)
    public static var ready: URL { url.appending(path: "ready", directoryHint: .isDirectory) }
}

/// Several readers behind one `FindExtracting`: each request goes to an
/// idle reader, so `parallelism` files are read at once. Reading is mostly
/// waiting (one file's decode, then Vision, then the reply), so a few
/// helper processes side by side multiply throughput.
public actor FindReaderPool: FindExtracting {
    private let readers: [any FindExtracting]
    private var idle: [Int]
    private var waiters: [CheckedContinuation<Int, Never>] = []
    public nonisolated let parallelism: Int

    public init(_ readers: [any FindExtracting]) {
        precondition(!readers.isEmpty)
        self.readers = readers
        idle = Array(readers.indices.reversed())
        parallelism = readers.count
    }

    /// How many helper processes suit this Mac: one per four cores, two to
    /// three (each holds Vision's models, a few hundred MB).
    public static func size(cores: Int = ProcessInfo.processInfo.activeProcessorCount) -> Int {
        min(3, max(1, cores / 4))
    }

    private func acquire() async -> Int {
        if let reader = idle.popLast() { return reader }
        return await withCheckedContinuation { waiters.append($0) }
    }

    private func release(_ reader: Int) {
        if waiters.isEmpty {
            idle.append(reader)
        } else {
            waiters.removeFirst().resume(returning: reader)
        }
    }

    public func extract(_ url: URL, kind: FindKind, options: FindExtractOptions) async -> FindExtraction {
        let reader = await acquire()
        let result = await readers[reader].extract(url, kind: kind, options: options)
        release(reader)
        return result
    }

    public func prepare(_ url: URL, kind: FindKind, pieces: [FindMediaPiece]) async -> [URL?] {
        let reader = await acquire()
        let result = await readers[reader].prepare(url, kind: kind, pieces: pieces)
        release(reader)
        return result
    }

    public func setHurry(_ hurry: Bool) async {
        for reader in readers { await reader.setHurry(hurry) }
    }

    /// Keeps the first reader (and its warm frameworks); stops the others.
    public func relax() async {
        for reader in readers.dropFirst() { await reader.shutdown() }
    }

    public func shutdown() async {
        for reader in readers { await reader.shutdown() }
    }
}

/// One request at a time on an actor that suspends while it waits.
actor FindTurnstile {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func enter() async {
        if !busy {
            busy = true
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }

    func leave() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}

/// Runs the extractor in this process (tests, and a fallback when the helper
/// cannot start). A file that takes longer than `timeout` is reported as
/// skipped at the timeout; its thread is left to finish on its own.
public struct InProcessFindExtractor: FindExtracting {
    public var timeout: Duration

    public init(timeout: Duration = .seconds(60)) {
        self.timeout = timeout
    }

    public func extract(_ url: URL, kind: FindKind, options: FindExtractOptions) async -> FindExtraction {
        let work = Task.detached(priority: Task.currentPriority) {
            await BlockingWork.run { FindExtractor.extract(url, kind: kind, options: options) }
        }
        return await Self.result(of: work, within: timeout)
    }

    /// `work`'s result, or skipped once `timeout` passes first (or the caller
    /// is cancelled). The caller resumes then; a stuck read keeps its thread
    /// and finishes on its own. (A task group would wait for the read:
    /// awaiting `work.value` does not end when the group is cancelled.)
    static func result(of work: Task<FindExtraction, Never>, within timeout: Duration) async -> FindExtraction {
        let skipped = FindExtraction.skipped(reason: "Took too long to read.")
        // The first value in wins; the stream ends with it.
        let (first, input) = AsyncStream.makeStream(of: FindExtraction.self, bufferingPolicy: .bufferingOldest(1))
        let timer = Task {
            try? await Task.sleep(for: timeout)
            input.yield(skipped)
            input.finish()
        }
        defer { timer.cancel() }
        Task {
            input.yield(await work.value)
            input.finish()
        }
        var results = first.makeAsyncIterator()
        return await results.next() ?? skipped
    }

    public func prepare(_ url: URL, kind: FindKind, pieces: [FindMediaPiece]) async -> [URL?] {
        await BlockingWork.run {
            FindMedia.prepare(url, kind: kind, pieces: pieces, into: FindMediaFolder.url)
        }
    }

    public func shutdown() async {}
}

/// Runs the extractor in a helper process: the app's own executable started
/// with `--find-extract-worker`, one JSON request and reply per line. A
/// crash or a hang costs one file: the helper is killed and started again.
public actor HelperFindExtractor: FindExtracting {
    /// The argument that turns the app's executable into the helper.
    public static let argument = "--find-extract-worker"

    struct Request: Codable {
        var path: String
        var kind: FindKind
        var options: FindExtractOptions
        /// Set for a media preparation request (`prepare`), nil to read text.
        var pieces: [FindMediaPiece]?
        /// Run at user-initiated priority (a large backlog), else utility.
        var hurry: Bool?
    }

    /// The reply to a preparation request: one path per piece, nil where it
    /// could not be decoded.
    struct Prepared: Codable {
        var paths: [String?]
    }

    private let executable: URL
    private var process: Process?
    private var input: FileHandle?
    private var lines: LineChannel?
    /// The kinds of work the running helper has done once: each framework's
    /// first use in a process (Vision, PDFKit, AVFoundation) can take tens
    /// of seconds, so the first request of each kind gets more time.
    private var warmed: Set<String> = []
    private var hurry = false
    /// The helper answers one request at a time; callers may overlap.
    private let turnstile = FindTurnstile()

    public init(executable: URL) {
        self.executable = executable
    }

    public func extract(_ url: URL, kind: FindKind, options: FindExtractOptions) async -> FindExtraction {
        do {
            try ensureRunning()
        } catch {
            return await InProcessFindExtractor().extract(url, kind: kind, options: options)
        }
        // Video and audio are sampled, not read whole: their time goes with
        // their length, which their size only roughly tells.
        let timeout = kind.isMedia ? Duration.seconds(300) : Self.timeout(for: url)
        guard let line = await send(Request(path: url.path, kind: kind, options: options, hurry: hurry),
                                    timeout: timeout, warming: Self.framework(kind)) else {
            return .skipped(reason: "Took too long to read, or made the reader stop.")
        }
        return (try? JSONDecoder().decode(FindExtraction.self, from: line)) ?? .skipped(reason: "Could not be read.")
    }

    public func prepare(_ url: URL, kind: FindKind, pieces: [FindMediaPiece]) async -> [URL?] {
        let none = [URL?](repeating: nil, count: pieces.count)
        guard !pieces.isEmpty else { return [] }
        do {
            try ensureRunning()
        } catch {
            return await InProcessFindExtractor().prepare(url, kind: kind, pieces: pieces)
        }
        let request = Request(path: url.path, kind: kind, options: FindExtractOptions(), pieces: pieces, hurry: hurry)
        guard let line = await send(request, timeout: .seconds(60 + 3 * pieces.count),
                                    warming: kind == .image ? "imageio" : "avfoundation"),
              let prepared = try? JSONDecoder().decode(Prepared.self, from: line),
              prepared.paths.count == pieces.count else { return none }
        return prepared.paths.map { $0.map { URL(filePath: $0) } }
    }

    /// Sends one request and waits for its reply line; nil when the helper
    /// hung or crashed (it is stopped, and the next request starts a new one).
    private func send(_ request: Request, timeout: Duration, warming: String) async -> Data? {
        await turnstile.enter()
        defer { Task { await turnstile.leave() } }
        do {
            try ensureRunning()
        } catch {
            return nil
        }
        guard let data = try? JSONEncoder().encode(request), let input, let lines else { return nil }
        do {
            try input.write(contentsOf: data + Data("\n".utf8))
        } catch {
            stop()
            return nil
        }
        // The first request of each kind also pays its framework's start-up.
        let wait = timeout + (warmed.contains(warming) ? .zero : .seconds(90))
        warmed.insert(warming)
        guard let line = await lines.next(timeout: wait) else {
            stop()
            return nil
        }
        return line
    }

    /// The framework a kind of file is read with, for the start-up allowance.
    static func framework(_ kind: FindKind) -> String {
        switch kind {
        case .image, .pdf: "vision"
        case .video, .audio: "avfoundation"
        default: "text"
        }
    }

    /// 60 s plus 5 s a megabyte, at most 3 minutes.
    static func timeout(for url: URL) -> Duration {
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return .seconds(min(180, 60 + bytes / 200_000))
    }

    public func shutdown() async {
        await turnstile.enter()
        stop()
        await turnstile.leave()
    }

    public func setHurry(_ hurry: Bool) {
        self.hurry = hurry
    }

    private func ensureRunning() throws {
        if let process, process.isRunning { return }
        stop()
        let process = Process()
        process.executableURL = executable
        process.arguments = [Self.argument]
        let stdin = Pipe()
        let stdout = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = FileHandle.nullDevice
        // Each request sets its own thread's priority (`hurry`).
        process.qualityOfService = .userInitiated
        try process.run()
        self.process = process
        warmed = []
        input = stdin.fileHandleForWriting
        lines = LineChannel(reading: stdout.fileHandleForReading)
    }

    private func stop() {
        lines?.close()
        try? input?.close()
        if let process, process.isRunning { process.terminate() }
        process = nil
        input = nil
        lines = nil
    }

    /// The helper's main loop: reads requests from stdin until it closes.
    /// Call from `main` when `argument` is present; never returns.
    public static func runHelper() -> Never {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        while let line = readLine(strippingNewline: true) {
            let request = try? decoder.decode(Request.self, from: Data(line.utf8))
            // Vision and AVFoundation run their work at the caller's priority.
            pthread_set_qos_class_self_np(request?.hurry == true ? QOS_CLASS_USER_INITIATED : QOS_CLASS_UTILITY, 0)
            var data: Data
            if let request, let pieces = request.pieces {
                let paths = autoreleasepool {
                    FindMedia.prepare(URL(filePath: request.path), kind: request.kind, pieces: pieces,
                                      into: FindMediaFolder.url)
                }
                data = (try? encoder.encode(Prepared(paths: paths.map { $0?.path }))) ?? Data("{\"paths\":[]}".utf8)
            } else {
                let reply: FindExtraction
                if let request {
                    reply = autoreleasepool {
                        FindExtractor.extract(URL(filePath: request.path), kind: request.kind, options: request.options)
                    }
                } else {
                    reply = .skipped(reason: "Could not be read.")
                }
                data = (try? encoder.encode(reply)) ?? Data("{\"skipped\":{\"reason\":\"Could not be read.\"}}".utf8)
            }
            data.append(UInt8(ascii: "\n"))
            FileHandle.standardOutput.write(data)
        }
        exit(0)
    }
}

/// Lines from a pipe, read on a background thread; `next(timeout:)` waits for
/// one.
final class LineChannel: Sendable {
    private struct State {
        var lines: [Data] = []
        var buffer = Data()
        var closed = false
        var waiter: CheckedContinuation<Data?, Never>?
    }

    private let state = Mutex(State())

    init(reading handle: FileHandle) {
        let thread = Thread { [weak self] in
            while true {
                let chunk = handle.availableData
                guard let self else { return }
                if chunk.isEmpty {
                    self.close()
                    return
                }
                self.receive(chunk)
            }
        }
        thread.qualityOfService = .utility
        thread.start()
    }

    private func receive(_ chunk: Data) {
        let delivery = state.withLock { state -> (CheckedContinuation<Data?, Never>, Data)? in
            state.buffer.append(chunk)
            while let newline = state.buffer.firstIndex(of: UInt8(ascii: "\n")) {
                state.lines.append(Data(state.buffer[state.buffer.startIndex..<newline]))
                state.buffer.removeSubrange(state.buffer.startIndex...newline)
            }
            guard let waiter = state.waiter, !state.lines.isEmpty else { return nil }
            state.waiter = nil
            return (waiter, state.lines.removeFirst())
        }
        if let (waiter, line) = delivery {
            waiter.resume(returning: line)
        }
    }

    func close() {
        let waiter = state.withLock { state -> CheckedContinuation<Data?, Never>? in
            state.closed = true
            defer { state.waiter = nil }
            return state.waiter
        }
        waiter?.resume(returning: nil)
    }

    /// The next line, or nil when the pipe closed or `timeout` passed.
    func next(timeout: Duration) async -> Data? {
        let timer = Task { [weak self] in
            try? await Task.sleep(for: timeout)
            guard !Task.isCancelled else { return }
            self?.expire()
        }
        defer { timer.cancel() }
        return await withCheckedContinuation { continuation in
            let ready = state.withLock { state -> Data?? in
                if !state.lines.isEmpty { return .some(state.lines.removeFirst()) }
                if state.closed { return .some(nil) }
                state.waiter = continuation
                return nil
            }
            if let ready { continuation.resume(returning: ready) }
        }
    }

    private func expire() {
        let waiter = state.withLock { state -> CheckedContinuation<Data?, Never>? in
            defer { state.waiter = nil }
            return state.waiter
        }
        waiter?.resume(returning: nil)
    }
}
