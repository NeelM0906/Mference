#if canImport(MferenceEngine)
import Foundation
import MferenceEngine
import Observation
import SwiftData

/// The composition root: every service the UI uses, built once at launch
/// (`live()`) or for previews and tests (`preview(...)`).
///
/// The shell injects it with `.environment(services)` and the SwiftData
/// container with `.modelContainer(services.modelContainer)`; views read it
/// with `@Environment(AppServices.self)`.
@MainActor
@Observable
public final class AppServices {
    public let engine: any EngineServicing
    /// What chat talks to: `engine`, plus Apple's on-device model as
    /// `AppModel.apple` while Settings allows it (`AppleChatRouting`). Every
    /// other surface uses `engine`.
    public let chatEngine: any EngineServicing
    public let settings: AppSettings
    public let permissions: any PermissionBroker
    public let modelContainer: ModelContainer
    /// GGUF product model downloads (Quick, Deep); nil where none are offered.
    public let downloads: DownloadManager?
    /// U4's Apple surfaces (Scan, Calendar, Mail, Connections); nil only in
    /// tests that build `AppServices` by hand.
    public let apple: AppleServices?
    /// Set when the on-disk store could not be opened and the app runs on an
    /// in-memory store: nothing is saved this session. The UI says so.
    public let storeError: String?

    @ObservationIgnored private let idleUnload: IdleUnloadControl?

    /// Wires the parts together; `idleUnload` is the control the engine's
    /// host waits on, kept in step with `settings.idleUnloadMinutes`.
    public init(engine: any EngineServicing,
                settings: AppSettings,
                permissions: any PermissionBroker,
                modelContainer: ModelContainer,
                downloads: DownloadManager? = nil,
                apple: AppleServices? = nil,
                appleChat: (any AppleChatModeling)? = nil,
                storeError: String? = nil,
                idleUnload: IdleUnloadControl? = nil) {
        self.engine = engine
        if let appleChat {
            let routing = AppleChatRouting(engine: engine, apple: appleChat,
                                           isEnabled: { await MainActor.run { settings.useAppleModel } })
            chatEngine = routing
            settings.onUseAppleModelChange = { _ in routing.appleModelChanged() }
        } else {
            chatEngine = engine
        }
        self.downloads = downloads
        self.apple = apple
        self.settings = settings
        self.permissions = permissions
        self.modelContainer = modelContainer
        self.storeError = storeError
        self.idleUnload = idleUnload
        if let idleUnload {
            idleUnload.setDelay(settings.idleUnloadDelay)
            settings.onIdleUnloadChange = { idleUnload.setDelay($0) }
        }
        if let downloads {
            // Offline mode stops downloads already running, not only new ones.
            settings.onOfflineModeChange = { offline in
                if offline { Task { await downloads.pauseAll() } }
            }
        }
    }

    /// The app: settings in `UserDefaults`, the store under Application
    /// Support, the real engine and permissions. Does no file-system work for
    /// the engine until something asks for the catalog.
    /// `presentPermissionReason` shows a permission's one-line reason and returns
    /// whether the person went on; the system prompt follows only then (L6).
    public static func live(presentPermissionReason: @escaping LivePermissionBroker.ReasonPresenter) -> AppServices {
        let settings = AppSettings()
        let idleUnload = IdleUnloadControl(delay: settings.idleUnloadDelay)
        let container: ModelContainer
        var storeError: String?
        do {
            container = try AppModelContainer.make()
        } catch {
            storeError = String(describing: error)
            // In-memory creation only fails on a schema bug, which the tests catch.
            container = try! AppModelContainer.inMemory()
        }
        let engine = EngineService.live(idleUnload: idleUnload)
        let downloads = DownloadManager(
            policy: DownloadManager.Policy(isOffline: { await MainActor.run { settings.offlineMode } }),
            onInstalled: { _ in await engine.refreshCatalog() })
        let permissions = LivePermissionBroker(presentReason: presentPermissionReason)
        return AppServices(engine: engine,
                           settings: settings,
                           permissions: permissions,
                           modelContainer: container,
                           downloads: downloads,
                           apple: AppleServices.live(engine: engine, broker: permissions, container: container,
                                                     useAppleModel: { await MainActor.run { settings.useAppleModel } }),
                           appleChat: FoundationChatModel(),
                           storeError: storeError,
                           idleUnload: idleUnload)
    }

    /// Previews and tests: an in-memory store and settings, fake permissions,
    /// and the real `EngineService` over `FakeEngineBackend`s with `installed`
    /// discovered. Nothing touches the disk, the network, TCC or a model.
    public static func preview(installed: [AppModel] = AppModel.engineModels,
                               scripts: [String: FakeEngineScript] = [:],
                               journal: FakeEngineJournal = FakeEngineJournal(),
                               settings: AppSettings? = nil,
                               permissions: any PermissionBroker = FakePermissionBroker(),
                               idleUnload: IdleUnloadControl = IdleUnloadControl(delay: nil)) -> AppServices {
        let settings = settings ?? AppSettings(store: InMemorySettingsStore())
        let engine = fakeEngine(installed: installed, scripts: scripts, journal: journal, idleUnload: idleUnload)
        // Downloads that can never reach the network: no host is allowed.
        let downloads = DownloadManager(
            destination: FileManager.default.temporaryDirectory.appending(path: "mference-preview-downloads"),
            allowedHosts: [])
        let container = try! AppModelContainer.inMemory()
        return AppServices(engine: engine,
                           settings: settings,
                           permissions: permissions,
                           modelContainer: container,
                           downloads: downloads,
                           apple: AppleServices.preview(engine: engine, container: container),
                           idleUnload: idleUnload)
    }

    /// An `EngineService` over fake backends, for previews and tests.
    public nonisolated static func fakeEngine(installed: [AppModel] = AppModel.engineModels,
                                              scripts: [String: FakeEngineScript] = [:],
                                              journal: FakeEngineJournal = FakeEngineJournal(),
                                              idleUnload: IdleUnloadControl = IdleUnloadControl(delay: nil))
        -> EngineService {
        let native = FakeEngineBackend(engine: .mference, journal: journal, scripts: scripts)
        let llama = FakeEngineBackend(engine: .llamaCpp, journal: journal, scripts: scripts)
        let catalog = fakeCatalog(installed)
        return EngineService(discover: { catalog }) { catalog, observer in
            EngineHost(catalog: catalog, backends: [native, llama],
                       idleUnloadAfter: EngineService.defaultIdleUnload,
                       sleep: idleUnload.sleeper, observer: observer)
        }
    }

    /// A catalog in which `installed` are discovered, with plausible sizes.
    public nonisolated static func fakeCatalog(_ installed: [AppModel]) -> EngineCatalog {
        EngineCatalog(models: installed.map { model in
            let slot = EngineProductSlot(modelID: model.modelID)
            let fake = FakeEngineBackend.descriptor(id: model.modelID, engine: slot?.engine ?? .mference,
                                                    product: slot)
            return EngineModelDescriptor(id: fake.id, displayName: slot?.displayName ?? fake.displayName,
                                         engine: fake.engine, format: fake.format, location: fake.location,
                                         sizeBytes: previewSize(model), contextLength: fake.contextLength,
                                         product: slot)
        })
    }

    nonisolated static func previewSize(_ model: AppModel) -> UInt64 {
        switch model {
        case .quick: 2_910_000_000
        case .deep: 12_100_000_000
        case .qwen36: 19_800_000_000
        case .minicpm5: 1_400_000_000
        case .maple: 7_600_000_000
        case .apple: 0
        }
    }
}
#endif
