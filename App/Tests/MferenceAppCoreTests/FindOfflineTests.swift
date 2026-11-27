import Foundation
@testable import MferenceAppCore
import Synchronization
import Testing

/// A search model whose download runs until the test finishes it or it is
/// cancelled, like a large download on a slow network.
actor GatedFindModel: FindModelProviding {
    private var state: FindModelState
    private(set) var downloadsStarted = 0
    private(set) var downloadsCancelled = 0
    private var finishing: AsyncStream<Void>.Continuation?
    private var running: Task<URL, any Error>?
    private let url = URL(filePath: "/dev/null")

    init(downloaded: Bool = false) {
        state = downloaded ? .ready(URL(filePath: "/dev/null")) : .notDownloaded
    }

    var isDownloading: Bool { running != nil }

    func currentState() -> FindModelState { state }

    func ensureDownloaded(progress: @escaping @Sendable (FindModelState) -> Void) async throws -> URL {
        if case .ready(let url) = state { return url }
        downloadsStarted += 1
        let (finished, finish) = AsyncStream<Void>.makeStream()
        finishing = finish
        state = .downloading(fraction: 0.4)
        progress(state)
        let url = self.url
        let work = Task<URL, any Error> {
            for await _ in finished {}
            try Task.checkCancellation()
            return url
        }
        running = work
        defer { if running == work { running = nil } }
        do {
            let result = try await work.value
            state = .ready(result)
            progress(state)
            return result
        } catch {
            state = .notDownloaded
            progress(state)
            throw CancellationError()
        }
    }

    /// Lets the running download complete.
    func finish() {
        finishing?.finish()
        finishing = nil
    }

    func cancel() {
        guard let running else { return }
        downloadsCancelled += 1
        running.cancel()
        finishing?.finish()
        finishing = nil
    }

    func delete() {}
}

/// Offline mode for one test: a plain switch the service reads.
final class OfflineSwitch: Sendable {
    private let on: Mutex<Bool>
    init(_ on: Bool) { self.on = Mutex(on) }
    var isOn: Bool { on.withLock { $0 } }
    func set(_ value: Bool) { on.withLock { $0 = value } }
}

@MainActor
@Suite struct FindOfflineTests {
    func service(model: any FindModelProviding, mediaModel: any FindModelProviding = PresentFindModel(),
                 offline: OfflineSwitch) -> FindService {
        let settings = FindSettings(store: InMemorySettingsStore())
        settings.learnOnlyOnPower = false
        return FindService(settings: settings, dependencies: .init(
            indexDirectory: nil, model: model, mediaModel: mediaModel,
            makeEmbedder: { HashingTextEmbedder(media: $1 != nil) },
            extractor: InProcessFindExtractor(), watchFolders: false,
            isOffline: { offline.isOn }))
    }

    /// Polls the main actor until `condition` holds or 10 s pass.
    func eventually(_ condition: () async -> Bool) async -> Bool {
        let deadline = ContinuousClock.now + .seconds(10)
        while ContinuousClock.now < deadline {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return await condition()
    }

    @Test func aDownloadedModelIsUsedOffline() async throws {
        let folder = try FindTempFolder()
        let model = GatedFindModel(downloaded: true)
        let find = service(model: model, offline: OfflineSwitch(true))
        await find.turnOn(roots: [folder.url])
        #expect(await eventually { find.modelState.isReady })
        #expect(await model.downloadsStarted == 0)
        await find.turnOff()
    }

    @Test func offlineDoesNotStartTheDownload() async throws {
        let folder = try FindTempFolder()
        let model = GatedFindModel()
        let find = service(model: model, offline: OfflineSwitch(true))
        await find.turnOn(roots: [folder.url])
        #expect(await eventually { find.modelState == .waitingForOfflineMode })
        try await Task.sleep(for: .milliseconds(100))
        #expect(await model.downloadsStarted == 0)
        await find.turnOff()
    }

    @Test func goingOfflineStopsARunningDownloadAndGoingOnlineStartsItAgain() async throws {
        let folder = try FindTempFolder()
        let model = GatedFindModel()
        let offline = OfflineSwitch(false)
        let find = service(model: model, offline: offline)
        await find.turnOn(roots: [folder.url])
        #expect(await eventually { await model.isDownloading })

        offline.set(true)
        await find.offlineModeChanged(true)
        #expect(await model.downloadsCancelled == 1)
        #expect(await eventually { find.modelState == .waitingForOfflineMode })
        #expect(await eventually { await !model.isDownloading })

        offline.set(false)
        await find.offlineModeChanged(false)
        #expect(await eventually { await model.downloadsStarted == 2 })
        await model.finish()
        #expect(await eventually { find.modelState.isReady })
        await find.turnOff()
    }

    @Test func theMediaModelWaitsTooWhileTheSearchModelWorks() async throws {
        let folder = try FindTempFolder()
        let media = GatedFindModel()
        let find = service(model: GatedFindModel(downloaded: true), mediaModel: media, offline: OfflineSwitch(true))
        await find.turnOn(roots: [folder.url])
        await find.setReadMedia(true)
        #expect(await eventually { find.modelState.isReady })
        #expect(await eventually { find.mediaModelState == .waitingForOfflineMode })
        #expect(await media.downloadsStarted == 0)
        await find.turnOff()
    }

    @Test func offlineChangesLeaveAFindThatIsOffAlone() async throws {
        let folder = try FindTempFolder()
        let model = GatedFindModel()
        let offline = OfflineSwitch(true)
        let find = service(model: model, offline: offline)
        await find.turnOn(roots: [folder.url])
        #expect(await eventually { find.modelState == .waitingForOfflineMode })
        await find.turnOff()
        await find.offlineModeChanged(true)
        offline.set(false)
        await find.offlineModeChanged(false)
        try await Task.sleep(for: .milliseconds(100))
        #expect(await model.downloadsStarted == 0)
    }

    /// Offline on and straight off again: the second change can run while
    /// the first is still stopping the download. Find ends up downloading.
    @Test func offlineOnAndStraightOffAgainKeepsTheDownloadGoing() async throws {
        let folder = try FindTempFolder()
        let model = GatedFindModel()
        let offline = OfflineSwitch(false)
        let find = service(model: model, offline: offline)
        await find.turnOn(roots: [folder.url])
        #expect(await eventually { await model.isDownloading })
        offline.set(true)
        let goingOffline = Task { await find.offlineModeChanged(true) }
        offline.set(false)
        let goingOnline = Task { await find.offlineModeChanged(false) }
        await goingOffline.value
        await goingOnline.value
        #expect(await eventually { await model.isDownloading })
        #expect(find.modelState != .waitingForOfflineMode)
        await model.finish()
        #expect(await eventually { find.modelState.isReady })
        await find.turnOff()
    }

    /// Photos, video and audio turned off while its model waits: nothing is
    /// left waiting (the banner would show it instead of Find's progress).
    @Test func turningMediaOffClearsItsWait() async throws {
        let folder = try FindTempFolder()
        let find = service(model: GatedFindModel(downloaded: true), mediaModel: GatedFindModel(),
                           offline: OfflineSwitch(true))
        await find.turnOn(roots: [folder.url])
        await find.setReadMedia(true)
        #expect(await eventually { find.mediaModelState == .waitingForOfflineMode })
        await find.setReadMedia(false)
        #expect(find.mediaModelState == .notDownloaded)
        await find.turnOff()
    }

    /// The app's own switch reaches Find: AppServices forwards Offline mode.
    @Test func appSettingsOfflineModeReachesFind() async throws {
        let folder = try FindTempFolder()
        let model = GatedFindModel()
        let settings = AppSettings(store: InMemorySettingsStore())
        let findSettings = FindSettings(store: InMemorySettingsStore())
        findSettings.learnOnlyOnPower = false
        let find = FindService(settings: findSettings, dependencies: .init(
            indexDirectory: nil, model: model, mediaModel: PresentFindModel(),
            makeEmbedder: { HashingTextEmbedder(media: $1 != nil) },
            extractor: InProcessFindExtractor(), watchFolders: false,
            isOffline: { settings.offlineMode }))
        let services = AppServices(engine: AppServices.fakeEngine(), settings: settings,
                                   permissions: FakePermissionBroker(),
                                   modelContainer: try AppModelContainer.inMemory(), find: find)
        await services.find?.turnOn(roots: [folder.url])
        #expect(await eventually { await model.isDownloading })
        settings.offlineMode = true
        #expect(await eventually { find.modelState == .waitingForOfflineMode })
        #expect(await model.downloadsCancelled == 1)
        #expect(await eventually { await !model.isDownloading })
        settings.offlineMode = false
        #expect(await eventually { await model.downloadsStarted == 2 })
        await model.cancel()
        await find.turnOff()
    }
}

/// Holds every request open until it is cancelled: a model download in
/// progress. Its own statics, used only by the serialized suite below.
final class HoldingModelURLProtocol: URLProtocol, @unchecked Sendable {
    private static let started = Mutex(0)
    static var requests: Int { started.withLock { $0 } }
    static func reset() { started.withLock { $0 = 0 } }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() { Self.started.withLock { $0 += 1 } }
    override func stopLoading() {}
}

@Suite(.serialized) struct FindModelStoreRestartTests {
    /// A download asked for right after a cancel is a new one, not the
    /// cancelled one still ending (which would fail with CancellationError).
    @Test func aDownloadAskedForRightAfterACancelStartsAgain() async throws {
        HoldingModelURLProtocol.reset()
        let folder = try FindTempFolder()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [HoldingModelURLProtocol.self]
        let store = FindModelStore(directory: folder.url,
                                   file: FindModelFile(fileName: "model.gguf", sizeBytes: 1_000, sha256: "0"),
                                   configuration: configuration)
        let first = Task { try await store.ensureDownloaded { _ in } }
        let deadline = ContinuousClock.now + .seconds(10)
        while HoldingModelURLProtocol.requests < 1, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        await store.cancel()
        let second = Task { try await store.ensureDownloaded { _ in } }
        while HoldingModelURLProtocol.requests < 2, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(HoldingModelURLProtocol.requests == 2, "the second ask joined the cancelled download")
        await store.cancel()
        _ = try? await first.value
        _ = try? await second.value
    }
}
