import CryptoKit
import Foundation
@testable import MferenceAppCore
import MferenceEngine
@preconcurrency import Network
import Synchronization
import Testing

/// A tiny HTTP/1.1 server on 127.0.0.1 for download tests: one request per
/// connection, `Range: bytes=N-` honoured unless told otherwise, optional
/// redirects and a mid-body hold. Never reachable off this Mac.
final class LoopbackHTTPServer: @unchecked Sendable {
    struct Request: Sendable, Hashable {
        var path: String
        var rangeStart: UInt64?
    }

    struct Route: Sendable {
        var body: Data
        var ignoresRange = false
        /// Send this many bytes, then wait for `hold` before the rest.
        var holdAfter: Int?
        var hold: FakeEngineLatch?
        /// Answer with a 302 to this absolute URL instead.
        var redirect: String?
        var status: Int?
        /// Answer a range request with a Content-Range this far off.
        var rangeStartBias: Int = 0
    }

    private let listener: NWListener
    private let queue: DispatchQueue
    private let routes: Mutex<[String: Route]>
    private let log = Mutex<[Request]>([])
    let port: UInt16

    var requests: [Request] { log.withLock { $0 } }

    init(routes: [String: Route]) async throws {
        self.routes = Mutex(routes)
        let queue = DispatchQueue(label: "loopback-http")
        self.queue = queue
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        let listener = try NWListener(using: parameters)
        self.listener = listener
        let ready = Once()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            listener.stateUpdateHandler = { state in
                let resume: Result<Void, Error>? = switch state {
                case .ready: .success(())
                case .failed(let error): .failure(error)
                default: nil
                }
                guard let resume, ready.claim() else { return }
                continuation.resume(with: resume)
            }
            listener.newConnectionHandler = { _ in }
            listener.start(queue: queue)
        }
        port = listener.port?.rawValue ?? 0
        listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
    }

    deinit { listener.cancel() }

    func url(_ path: String) -> URL { URL(string: "http://127.0.0.1:\(port)\(path)")! }

    func setRoute(_ route: Route, for path: String) { routes.withLock { $0[path] = route } }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receiveHeader(connection, buffer: Data())
    }

    private func receiveHeader(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, done, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let end = buffer.range(of: Data("\r\n\r\n".utf8)) {
                self.respond(connection, header: String(decoding: buffer[..<end.lowerBound], as: UTF8.self))
            } else if error == nil, !done {
                self.receiveHeader(connection, buffer: buffer)
            } else {
                connection.cancel()
            }
        }
    }

    private func respond(_ connection: NWConnection, header: String) {
        let lines = header.components(separatedBy: "\r\n")
        let path = lines.first?.split(separator: " ").dropFirst().first.map(String.init) ?? "/"
        var rangeStart: UInt64?
        for line in lines.dropFirst() where line.lowercased().hasPrefix("range:") {
            let value = line.dropFirst("range:".count).trimmingCharacters(in: .whitespaces)
            if value.hasPrefix("bytes="), let dash = value.firstIndex(of: "-") {
                rangeStart = UInt64(value[value.index(value.startIndex, offsetBy: 6)..<dash])
            }
        }
        log.withLock { $0.append(Request(path: path, rangeStart: rangeStart)) }
        guard let route = routes.withLock({ $0[path] }) else {
            return send(connection, head: "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n", body: Data())
        }
        if let redirect = route.redirect {
            return send(connection, head: "HTTP/1.1 302 Found\r\nLocation: \(redirect)\r\nContent-Length: 0\r\n",
                        body: Data())
        }
        if let status = route.status {
            return send(connection, head: "HTTP/1.1 \(status) Error\r\nContent-Length: 0\r\n", body: Data())
        }
        let total = route.body.count
        var head: String
        var body: Data
        if let start = rangeStart, !route.ignoresRange {
            guard Int(start) < total else {
                return send(connection, head: "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */\(total)\r\nContent-Length: 0\r\n",
                            body: Data())
            }
            body = route.body.subdata(in: Int(start)..<total)
            head = "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes \(Int(start) + route.rangeStartBias)-\(total - 1)/\(total)\r\n"
        } else {
            body = route.body
            head = "HTTP/1.1 200 OK\r\n"
        }
        head += "Content-Type: application/octet-stream\r\nContent-Length: \(body.count)\r\n"
        if let holdAfter = route.holdAfter, let hold = route.hold, holdAfter < body.count, rangeStart == nil {
            let first = body.prefix(holdAfter)
            let rest = body.dropFirst(holdAfter)
            connection.send(content: Data((head + "Connection: close\r\n\r\n").utf8) + first,
                            completion: .contentProcessed { _ in })
            Task {
                try? await hold.wait()
                self.queue.async {
                    connection.send(content: Data(rest), isComplete: true,
                                    completion: .contentProcessed { _ in connection.cancel() })
                }
            }
            return
        }
        send(connection, head: head, body: body)
    }

    private func send(_ connection: NWConnection, head: String, body: Data) {
        connection.send(content: Data((head + "Connection: close\r\n\r\n").utf8) + body, isComplete: true,
                        completion: .contentProcessed { _ in connection.cancel() })
    }
}

/// True for the first caller only.
final class Once: Sendable {
    private let done = Mutex(false)

    func claim() -> Bool {
        done.withLock { done in
            defer { done = true }
            return !done
        }
    }
}

/// Deterministic bytes for a fake model file.
func fakeModelBytes(_ count: Int, seed: UInt8 = 7) -> Data {
    var state = UInt32(seed) &* 2_654_435_761 | 1
    return Data((0..<count).map { _ in
        state ^= state << 13
        state ^= state >> 17
        state ^= state << 5
        return UInt8(truncatingIfNeeded: state)
    })
}

func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func temporaryDirectory() -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "mference-downloads-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

func source(_ model: AppModel = .quick, url: URL, body: Data, sha: String? = nil) -> DownloadSource {
    DownloadSource(model: model, repository: "test/test", revision: "abc", fileName: "\(model.rawValue).gguf",
                   sizeBytes: UInt64(body.count), sha256: sha ?? sha256Hex(body), licence: "Apache 2.0", url: url)
}

final class InstalledLog: Sendable {
    private let log = Mutex<[AppModel]>([])
    var models: [AppModel] { log.withLock { $0 } }
    func append(_ model: AppModel) { log.withLock { $0.append(model) } }
}

let bigMac = MemoryFit(physicalMemory: 32 << 30)

func manager(_ sources: [DownloadSource], destination: URL, offline: Bool = false,
             memory: MemoryFit = bigMac, available: UInt64? = nil,
             installed: InstalledLog? = nil) -> DownloadManager {
    DownloadManager(destination: destination,
                    sources: Dictionary(uniqueKeysWithValues: sources.map { ($0.model, $0) }),
                    allowedHosts: ["127.0.0.1"],
                    policy: DownloadManager.Policy(isOffline: { offline }, memoryFit: memory),
                    availableBytes: { _ in available ?? (1 << 40) },
                    onInstalled: { model in installed?.append(model) })
}

@Suite(.serialized) struct ModelsDownloadTests {
    @Test func downloadsVerifiesAndInstalls() async throws {
        let body = fakeModelBytes(300_000)
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let src = source(url: server.url("/q"), body: body)
        let installed = InstalledLog()
        let downloads = manager([src], destination: destination, installed: installed)

        let url = try await downloads.download(DownloadApproval(confirming: src))
        #expect(url == destination.appending(path: "quick.gguf"))
        #expect(try Data(contentsOf: url) == body)
        #expect(!FileManager.default.fileExists(atPath: url.path + ".part"))
        #expect(await downloads.state(of: .quick) == .installed(url))
        #expect(installed.models == [.quick])
    }

    @Test func pauseKeepsBytesAndResumeAsksForTheRest() async throws {
        let body = fakeModelBytes(400_000)
        let hold = FakeEngineLatch()
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body, holdAfter: 150_000, hold: hold)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let src = source(url: server.url("/q"), body: body)
        let downloads = manager([src], destination: destination)
        let part = destination.appending(path: "quick.gguf.part")

        let first = Task { try await downloads.download(DownloadApproval(confirming: src)) }
        let deadline = ContinuousClock.now + .seconds(10)
        while DownloadManager.fileSize(part) < 150_000, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(DownloadManager.fileSize(part) == 150_000)
        await downloads.pause(.quick)
        await #expect(throws: CancellationError.self) { _ = try await first.value }
        #expect(await downloads.state(of: .quick) == .paused(receivedBytes: 150_000, totalBytes: 400_000))
        hold.open()

        // A new manager (a relaunch) sees the partial file and continues it.
        let relaunched = manager([src], destination: destination)
        #expect(await relaunched.state(of: .quick) == .paused(receivedBytes: 150_000, totalBytes: 400_000))
        let url = try await relaunched.download(DownloadApproval(confirming: src))
        #expect(try Data(contentsOf: url) == body)
        #expect(server.requests.last?.rangeStart == 150_000)
    }

    @Test func serverIgnoringTheRangeStartsOver() async throws {
        let body = fakeModelBytes(200_000)
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body, ignoresRange: true)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        // Leave a partial file of the right first bytes, of this source.
        let src = source(url: server.url("/q"), body: body)
        try body.prefix(50_000).write(to: destination.appending(path: "quick.gguf.part"))
        try Data(src.sha256.utf8).write(to: destination.appending(path: "quick.gguf.part.sha256"))
        let url = try await manager([src], destination: destination).download(DownloadApproval(confirming: src))
        #expect(try Data(contentsOf: url) == body)
        #expect(server.requests.last?.rangeStart == 50_000)
    }

    @Test func completePartialFileIsVerifiedWithoutRefetching() async throws {
        let body = fakeModelBytes(100_000)
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let src = source(url: server.url("/q"), body: body)
        try body.write(to: destination.appending(path: "quick.gguf.part"))
        try Data(src.sha256.utf8).write(to: destination.appending(path: "quick.gguf.part.sha256"))
        let url = try await manager([src], destination: destination).download(DownloadApproval(confirming: src))
        #expect(try Data(contentsOf: url) == body)
    }

    @Test func checksumMismatchRemovesTheFile() async throws {
        let body = fakeModelBytes(64_000)
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let src = source(url: server.url("/q"), body: body, sha: String(repeating: "0", count: 64))
        let downloads = manager([src], destination: destination)
        await #expect(throws: DownloadError.checksumMismatch) {
            try await downloads.download(DownloadApproval(confirming: src))
        }
        #expect(!FileManager.default.fileExists(atPath: destination.appending(path: "quick.gguf").path))
        #expect(!FileManager.default.fileExists(atPath: destination.appending(path: "quick.gguf.part").path))
        #expect(await downloads.state(of: .quick) == .failed(DownloadError.checksumMismatch.userMessage))
    }

    @Test func redirectsStayOnAllowedHosts() async throws {
        let body = fakeModelBytes(10_000)
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body)])
        server.setRoute(.init(body: Data(), redirect: server.url("/q").absoluteString), for: "/ok")
        server.setRoute(.init(body: Data(), redirect: "http://example.invalid/q"), for: "/away")
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }

        let allowed = source(url: server.url("/ok"), body: body)
        let url = try await manager([allowed], destination: destination).download(DownloadApproval(confirming: allowed))
        #expect(try Data(contentsOf: url) == body)

        let away = source(.deep, url: server.url("/away"), body: body)
        await #expect(throws: DownloadError.hostNotAllowed("example.invalid")) {
            try await manager([away], destination: destination).download(DownloadApproval(confirming: away))
        }
        // A source on a host outside the list is refused before any request.
        let outside = source(.deep, url: URL(string: "https://example.invalid/q")!, body: body)
        await #expect(throws: DownloadError.hostNotAllowed("example.invalid")) {
            try await manager([outside], destination: destination).download(DownloadApproval(confirming: outside))
        }
    }

    @Test func httpErrorsAreReported() async throws {
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: Data(), status: 503)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let src = source(url: server.url("/q"), body: fakeModelBytes(10))
        await #expect(throws: DownloadError.http(503)) {
            try await manager([src], destination: destination).download(DownloadApproval(confirming: src))
        }
    }

    @Test func concurrentDownloadsShareOneTransfer() async throws {
        let body = fakeModelBytes(300_000)
        let hold = FakeEngineLatch()
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body, holdAfter: 100_000, hold: hold)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let src = source(url: server.url("/q"), body: body)
        let downloads = manager([src], destination: destination)
        async let first = downloads.download(DownloadApproval(confirming: src))
        async let second = downloads.download(DownloadApproval(confirming: src))
        try await Task.sleep(for: .milliseconds(200))
        hold.open()
        let (a, b) = try await (first, second)
        #expect(a == b)
        #expect(try Data(contentsOf: a) == body)
        #expect(server.requests.filter { $0.path == "/q" }.count == 1)
        // A download after install does not touch the installed file.
        let again = try await downloads.download(DownloadApproval(confirming: src))
        #expect(try Data(contentsOf: again) == body)
        #expect(server.requests.filter { $0.path == "/q" }.count == 1)
    }

    @Test func pausingRightAwayDoesNotHang() async throws {
        let body = fakeModelBytes(50_000)
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let src = source(url: server.url("/q"), body: body)
        for _ in 0..<20 {
            let downloads = manager([src], destination: destination)
            let run = Task { try await downloads.download(DownloadApproval(confirming: src)) }
            await downloads.pause(.quick)
            run.cancel()
            let finished = await withTaskGroup(of: Bool.self) { group in
                group.addTask { _ = try? await run.value; return true }
                group.addTask { try? await Task.sleep(for: .seconds(5)); return false }
                let first = await group.next() ?? false
                group.cancelAll()
                return first
            }
            #expect(finished, "a paused download must return")
            await downloads.discard(.quick)
            try? FileManager.default.removeItem(at: destination.appending(path: "quick.gguf"))
        }
    }

    @Test func partialFileOfAnotherRevisionStartsOver() async throws {
        let body = fakeModelBytes(80_000)
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let src = source(url: server.url("/q"), body: body)
        try fakeModelBytes(30_000, seed: 9).write(to: destination.appending(path: "quick.gguf.part"))
        try Data(String(repeating: "a", count: 64).utf8).write(to: destination.appending(path: "quick.gguf.part.sha256"))
        let downloads = manager([src], destination: destination)
        #expect(await downloads.state(of: .quick) == .notStarted)
        let url = try await downloads.download(DownloadApproval(confirming: src))
        #expect(try Data(contentsOf: url) == body)
        #expect(server.requests.last?.rangeStart == nil)
    }

    @Test func wrongContentRangeIsRefused() async throws {
        let body = fakeModelBytes(60_000)
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body, rangeStartBias: 10)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let src = source(url: server.url("/q"), body: body)
        try body.prefix(20_000).write(to: destination.appending(path: "quick.gguf.part"))
        try Data(src.sha256.utf8).write(to: destination.appending(path: "quick.gguf.part.sha256"))
        await #expect(throws: DownloadError.network("the server sent a different part of the file")) {
            try await manager([src], destination: destination).download(DownloadApproval(confirming: src))
        }
    }

    // MARK: Refusals before any request

    @Test func refusalsHappenBeforeTheNetwork() async throws {
        let server = try await LoopbackHTTPServer(routes: [:])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let body = fakeModelBytes(10_000)
        let quick = source(url: server.url("/q"), body: body)
        let deep = source(.deep, url: server.url("/d"), body: body)

        await #expect(throws: DownloadError.offline) {
            try await manager([quick], destination: destination, offline: true)
                .download(DownloadApproval(confirming: quick))
        }
        let sixteen = MemoryFit(physicalMemory: 16 << 30)
        await #expect(throws: DownloadError.notOffered(sixteen.notOfferedReason(.deep)!)) {
            try await manager([deep], destination: destination, memory: sixteen)
                .download(DownloadApproval(confirming: deep))
        }
        await #expect(throws: DownloadError.insufficientDisk(neededBytes: 10_000 + DownloadManager.diskMarginBytes,
                                                             availableBytes: 5)) {
            try await manager([quick], destination: destination, available: 5)
                .download(DownloadApproval(confirming: quick))
        }
        let other = source(url: server.url("/other"), body: fakeModelBytes(5))
        await #expect(throws: DownloadError.approvalMismatch) {
            try await manager([quick], destination: destination).download(DownloadApproval(confirming: other))
        }
        await #expect(throws: DownloadError.noSource(.deep)) {
            try await manager([quick], destination: destination).download(DownloadApproval(confirming: deep))
        }
        #expect(server.requests.isEmpty)
    }

    @Test func productSourcesArePinned() {
        for source in DownloadSource.product.values {
            #expect(source.url.scheme == "https")
            #expect(source.url.host() == "huggingface.co")
            #expect(source.url.path().contains("/resolve/\(source.revision)/"))
            #expect(source.sha256.count == 64)
            #expect(source.revision.count == 40)
            #expect(DownloadSource.isAllowed(host: source.url.host(), allowedHosts: DownloadSource.allowedHosts))
        }
        #expect(DownloadSource.isAllowed(host: "cas-bridge.xethub.hf.co", allowedHosts: DownloadSource.allowedHosts))
        #expect(!DownloadSource.isAllowed(host: "evilhuggingface.co", allowedHosts: DownloadSource.allowedHosts))
        #expect(DownloadSource.defaultDestination.path.hasSuffix("llm-models/gguf"))
    }
}

@Suite struct ModelsMemoryFitTests {
    @Test func deepIsOfferedOnlyAboveSixteenGigabytes() {
        #expect(MemoryFit(physicalMemory: 16 << 30).isOffered(.deep) == false)
        #expect(MemoryFit(physicalMemory: 8 << 30).isOffered(.deep) == false)
        #expect(MemoryFit(physicalMemory: 24 << 30).isOffered(.deep))
        #expect(MemoryFit(physicalMemory: 8 << 30).isOffered(.quick))
    }

    @Test func verdictsFollowSize() {
        let eight = MemoryFit(physicalMemory: 8 << 30)
        #expect(eight.verdict(for: .quick, sizeBytes: DownloadSource.quick.sizeBytes) == .fits)
        #expect(eight.verdict(for: .deep, sizeBytes: DownloadSource.deep.sizeBytes) == .tooLarge)
        let thirtyTwo = MemoryFit(physicalMemory: 32 << 30)
        #expect(thirtyTwo.verdict(for: .deep, sizeBytes: DownloadSource.deep.sizeBytes) == .fits)
        // Qwen 3.6 streams experts, so its install may exceed memory.
        #expect(MemoryFit(physicalMemory: 16 << 30).verdict(for: .qwen36, sizeBytes: 20_000_000_000) == .fits)
        #expect(thirtyTwo.memoryText == "32 GB")
    }
}

@Suite struct ModelsLibraryTests {
    @Test @MainActor func rowsCombineCatalogDownloadsAndMemory() async throws {
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        // A Deep partial file from an earlier session.
        try fakeModelBytes(1_000).write(to: destination.appending(path: DownloadSource.deep.fileName + ".part"))
        try Data(DownloadSource.deep.sha256.utf8)
            .write(to: destination.appending(path: DownloadSource.deep.fileName + ".part.sha256"))
        let downloads = DownloadManager(destination: destination, allowedHosts: [])
        let settings = AppSettings(store: InMemorySettingsStore())
        let library = ModelLibrary(engine: AppServices.fakeEngine(installed: [.quick, .qwen36]),
                                   downloads: downloads, settings: settings,
                                   memoryFit: MemoryFit(physicalMemory: 32 << 30), chipName: "Apple M1 Max")
        await library.refresh()
        #expect(library.row(for: .quick)?.state == .installed)
        #expect(library.row(for: .quick)?.badge == "Installed · default")
        guard case .paused(let fraction, 1_000, DownloadSource.deep.sizeBytes)? = library.row(for: .deep)?.state else {
            Issue.record("Deep should show its paused partial download"); return
        }
        #expect(fraction > 0 && fraction < 0.001)
        #expect(library.row(for: .deep)?.sizeBytes == DownloadSource.deep.sizeBytes)
        #expect(library.row(for: .deep)?.licence == "Apache 2.0")
        guard case .notInstalled = library.row(for: .maple)?.state else {
            Issue.record("Maple is not downloadable from the app"); return
        }
        #expect(library.machineSummary == "This Mac: Apple M1 Max, 32 GB memory.")
        #expect(library.downloadSource(for: .deep) == DownloadSource.deep)
    }

    @Test @MainActor func finishedDownloadShowsInstalled() async throws {
        let body = fakeModelBytes(40_000)
        let server = try await LoopbackHTTPServer(routes: ["/q": .init(body: body)])
        let destination = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: destination) }
        let src = source(url: server.url("/q"), body: body)
        let file = destination.appending(path: src.fileName)
        // The engine discovers Quick once its file exists, as the real one does.
        let engine = EngineService(discover: {
            AppServices.fakeCatalog(FileManager.default.fileExists(atPath: file.path) ? [.quick] : [])
        }, makeHost: { catalog, observer in
            EngineHost(catalog: catalog, backends: [FakeEngineBackend(engine: .llamaCpp)], observer: observer)
        })
        let downloads = DownloadManager(destination: destination, sources: [.quick: src], allowedHosts: ["127.0.0.1"],
                                        policy: .init(memoryFit: bigMac), availableBytes: { _ in 1 << 40 },
                                        onInstalled: { _ in await engine.refreshCatalog() })
        let library = ModelLibrary(engine: engine, downloads: downloads,
                                   settings: AppSettings(store: InMemorySettingsStore()), memoryFit: bigMac, chipName: nil)
        let follow = Task { await library.run() }
        defer { follow.cancel() }
        await library.refresh()
        #expect(library.row(for: .quick)?.state == .downloadable)
        library.startDownload(DownloadApproval(confirming: src))
        await waitUntil { library.row(for: .quick)?.state == .installed }
        #expect(library.row(for: .quick)?.state == .installed)
    }

    @Test @MainActor func deepIsNotOfferedOnSixteenGigabytes() async {
        let library = ModelLibrary(engine: AppServices.fakeEngine(installed: [.quick]),
                                   downloads: DownloadManager(destination: temporaryDirectory(), allowedHosts: []),
                                   settings: AppSettings(store: InMemorySettingsStore()),
                                   memoryFit: MemoryFit(physicalMemory: 16 << 30), chipName: nil)
        await library.refresh()
        guard case .notOffered = library.row(for: .deep)?.state else {
            Issue.record("Deep must not be offered on 16 GB"); return
        }
        #expect(library.downloadSource(for: .deep) == nil)
        #expect(library.machineSummary == "This Mac: 16 GB memory.")
    }
}
