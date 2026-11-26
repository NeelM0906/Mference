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
    /// GGUF product model downloads (Quick, Deep, Bonsai); nil where none are offered.
    public let downloads: DownloadManager?
    /// The image-understanding add-ons of Quick and Deep (their vision
    /// projectors), into the same folder; nil where none are offered.
    public let imageDownloads: DownloadManager?
    /// U4's Apple surfaces (Scan, Calendar, Mail, Connections); nil only in
    /// tests that build `AppServices` by hand.
    public let apple: AppleServices?
    /// Dictation, meetings and file transcription (Notes spec 6.2). The UI
    /// and Notes talk only to this protocol.
    public let speech: any SpeechServicing
    /// Notes and folders (Notes spec 4.1, 8): the one store the Notes
    /// screens, dictation and meetings write through.
    public let notes: NoteStore
    /// The remembered voice choices (Settings › Speech, the popover, the
    /// dictation bar's chips).
    public let speechPreferences: SpeechPreferences
    /// The one consumer of dictation events (stream ownership, team N3).
    public let dictation: DictationController
    /// The meeting being recorded (Notes spec 4.4): the one consumer of
    /// `MeetingRecording.events`, so a meeting keeps going outside Notes.
    public let meetings: MeetingSession
    /// Summaries, titles, Redo and Add to To-dos for meeting and file notes (spec 4.5, 6.5).
    public let summaries: MeetingSummaries
    /// Audio files being transcribed (spec 4.6): the one consumer of `FileJob.events`.
    public let files: FileTranscriptionQueue
    /// Launch recovery of meetings and files the app quit in (spec 6.3),
    /// started once by `startLaunchRecovery()`. Anything that deletes notes
    /// or recordings at launch (the Recently deleted purge) waits for it.
    public private(set) var launchRecovery: Task<Void, Never>?
    /// Find: content search over the person's folders; nil only in tests
    /// that build `AppServices` by hand.
    public let find: FindService?
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
                imageDownloads: DownloadManager? = nil,
                apple: AppleServices? = nil,
                speech: (any SpeechServicing)? = nil,
                speechPreferences: SpeechPreferences? = nil,
                noteAttachments: NoteAttachments? = nil,
                meetingWriter: (any MeetingNoteWriting)? = nil,
                find: FindService? = nil,
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
        self.imageDownloads = imageDownloads
        self.apple = apple
        let speech = speech ?? UnavailableSpeechService()
        self.speech = speech
        // Without a folder given (tests that build this by hand), attachments
        // go to a temporary folder, never the person's.
        let notes = NoteStore(container: modelContainer, attachments: noteAttachments ?? .temporary(),
                              speech: speech)
        self.notes = notes
        // Scan's Save to Notes: the full text as a new note, titled by the scan.
        apple?.scan.saveToNotes = { [weak notes] title, text in
            guard let notes else { return }
            let body = text.components(separatedBy: .newlines).map { line in
                NoteDocument.Block.paragraph(runs: line.isEmpty ? [] : [TextRun(line)])
            }
            let note = try notes.createNote(title: title)
            try notes.save(.titled(title, body: body), to: note)
        }
        let speechPreferences = speechPreferences ?? SpeechPreferences(store: InMemorySettingsStore())
        self.speechPreferences = speechPreferences
        dictation = DictationController(speech: speech, permissions: permissions, preferences: speechPreferences)
        // Meeting and file notes write through the one Notes store (its
        // context, its revision), so an open editor takes their writes in.
        let writer = meetingWriter ?? notes
        precondition(writer.context === notes.context, "Meeting notes must be written on the Notes store's context")
        let meetings = MeetingSession(speech: speech, writer: writer)
        let summaries = MeetingSummaries(
            engine: engine,
            summarizer: MeetingSummarizer(engine: engine, prompt: try? MeetingSummaryPrompt.load()),
            assistant: apple?.assistant, writer: writer, todos: apple?.calendar)
        let files = FileTranscriptionQueue(speech: speech, writer: writer)
        meetings.afterMeeting = { [weak summaries] noteID, summarize in
            if summarize { summaries?.summarize(noteID) }
        }
        files.afterTranscript = { [weak summaries] noteID in summaries?.summarize(noteID) }
        files.transcriptSaved = { [weak meetings] noteID in meetings?.transcriptDidChange(for: noteID) }
        self.meetings = meetings
        self.summaries = summaries
        self.files = files
        self.find = find
        self.settings = settings
        self.permissions = permissions
        self.modelContainer = modelContainer
        self.storeError = storeError
        self.idleUnload = idleUnload
        if let idleUnload {
            idleUnload.setDelay(settings.idleUnloadDelay)
            settings.onIdleUnloadChange = { idleUnload.setDelay($0) }
        }
        let speechService = self.speech as? SpeechService
        if downloads != nil || imageDownloads != nil || speechService != nil {
            // Offline mode stops downloads already running, not only new
            // ones: the GGUF models and the speech packages.
            settings.onOfflineModeChange = { offline in
                guard offline else { return }
                Task {
                    await downloads?.pauseAll()
                    await imageDownloads?.pauseAll()
                    await speechService?.pauseAllInstalls()
                }
            }
        }
    }

    /// The app: settings in `UserDefaults`, the store under Application
    /// Support, the real engine and permissions. Does no file-system work for
    /// the engine until something asks for the catalog.
    /// `presentPermissionReason` shows a permission's one-line reason and returns
    /// whether the person went on; the system prompt follows only then (L6).
    /// `findHelper` is the executable that serves `--find-extract-worker`
    /// (the app itself); nil reads files in this process.
    public static func live(presentPermissionReason: @escaping LivePermissionBroker.ReasonPresenter,
                            findHelper: URL? = nil) -> AppServices {
        // Demo mode (MFERENCE_DEMO_DATA) keeps every setting in its own suite (DemoData).
        let settings = AppSettings(store: DemoData.settingsStore(demo: DemoData.directory))
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
        let imageDownloads = DownloadManager(
            sources: DownloadSource.imageUnderstanding,
            policy: DownloadManager.Policy(isOffline: { await MainActor.run { settings.offlineMode } }),
            onInstalled: { _ in await engine.refreshCatalog() })
        // Speech (Notes spec 6.2): the call-audio permission is remembered in
        // the app's defaults; models load from the verified install on first use.
        let systemAudio = SystemAudioPermission()
        let speech = SpeechService.live(isOffline: { await MainActor.run { settings.offlineMode } },
                                        systemAudio: systemAudio)
        let permissions = LivePermissionBroker(presentReason: presentPermissionReason,
                                               systemAudio: systemAudio.access)
        let services = AppServices(engine: engine,
                                   settings: settings,
                                   permissions: permissions,
                                   modelContainer: container,
                                   downloads: downloads,
                                   imageDownloads: imageDownloads,
                                   apple: AppleServices.live(engine: engine, broker: permissions, container: container,
                                                             useAppleModel: { await MainActor.run { settings.useAppleModel } }),
                                   speech: speech,
                                   speechPreferences: SpeechPreferences(),
                                   noteAttachments: .live,
                                   find: FindService.live(settings: FindSettings(), engine: engine, helper: findHelper,
                                                          permissions: permissions),
                                   appleChat: FoundationChatModel(),
                                   storeError: storeError,
                                   idleUnload: idleUnload)
        // Close meetings and re-queue files the app quit in first; only then
        // purge Recently deleted (notes kept 30 days, with their audio), so
        // a purge never deletes recordings recovery is about to attach.
        let recovery = services.startLaunchRecovery()
        let notes = services.notes
        Task { @MainActor in
            await recovery.value
            _ = try? await notes.purgeExpired()
        }
        return services
    }

    /// The meeting part of quitting (spec 6.1), after the person confirmed:
    /// ends a meeting through the speech service's fast quit path and saves
    /// it (`MeetingSession.stopForQuit()`). `AppRuntime` then shuts speech
    /// down (a no-op by then), web and the engine.
    public func shutDownVoice() async {
        await meetings.stopForQuit()
    }

    /// Closes meetings and re-queues files the app quit in, once, before any
    /// new session (spec 6.3). `live()` calls it; returns the running task.
    /// Anything that purges notes or recordings awaits `launchRecovery` first.
    @discardableResult
    public func startLaunchRecovery() -> Task<Void, Never> {
        if let launchRecovery { return launchRecovery }
        let task = Task { [meetings, files, summaries] in
            await meetings.recoverInterruptedMeetings()
            await files.recoverInterruptedJobs()
            // A meeting ended at quit gets its summary now.
            summaries.resumePending()
        }
        launchRecovery = task
        return task
    }

    /// Previews and tests: an in-memory store and settings, fake permissions,
    /// and the real `EngineService` over `FakeEngineBackend`s with `installed`
    /// discovered. Nothing touches the disk, the network, TCC or a model.
    public static func preview(installed: [AppModel] = AppModel.engineModels,
                               scripts: [String: FakeEngineScript] = [:],
                               journal: FakeEngineJournal = FakeEngineJournal(),
                               settings: AppSettings? = nil,
                               permissions: any PermissionBroker = FakePermissionBroker(),
                               speech: any SpeechServicing = FakeSpeechService(),
                               seeing: [AppModel] = [],
                               idleUnload: IdleUnloadControl = IdleUnloadControl(delay: nil)) -> AppServices {
        let settings = settings ?? AppSettings(store: InMemorySettingsStore())
        let engine = fakeEngine(installed: installed, seeing: seeing, scripts: scripts, journal: journal,
                                idleUnload: idleUnload)
        // Downloads that can never reach the network: no host is allowed.
        let downloads = DownloadManager(
            destination: FileManager.default.temporaryDirectory.appending(path: "mference-preview-downloads"),
            allowedHosts: [])
        let imageDownloads = DownloadManager(
            destination: FileManager.default.temporaryDirectory.appending(path: "mference-preview-downloads"),
            sources: DownloadSource.imageUnderstanding, allowedHosts: [])
        let container = try! AppModelContainer.inMemory()
        return AppServices(engine: engine,
                           settings: settings,
                           permissions: permissions,
                           modelContainer: container,
                           downloads: downloads,
                           imageDownloads: imageDownloads,
                           apple: AppleServices.preview(engine: engine, container: container),
                           speech: speech,
                           find: FindService.preview(settings: FindSettings(store: InMemorySettingsStore())),
                           idleUnload: idleUnload)
    }

    /// An `EngineService` over fake backends, for previews and tests.
    public nonisolated static func fakeEngine(installed: [AppModel] = AppModel.engineModels,
                                              seeing: [AppModel] = [],
                                              scripts: [String: FakeEngineScript] = [:],
                                              journal: FakeEngineJournal = FakeEngineJournal(),
                                              idleUnload: IdleUnloadControl = IdleUnloadControl(delay: nil))
        -> EngineService {
        let native = FakeEngineBackend(engine: .mference, journal: journal, scripts: scripts)
        let llama = FakeEngineBackend(engine: .llamaCpp, journal: journal, scripts: scripts)
        let catalog = fakeCatalog(installed, seeing: seeing)
        return EngineService(discover: { catalog }) { catalog, observer in
            EngineHost(catalog: catalog, backends: [native, llama],
                       idleUnloadAfter: EngineService.defaultIdleUnload,
                       sleep: idleUnload.sleeper, observer: observer)
        }
    }

    /// A catalog in which `installed` are discovered, with plausible sizes;
    /// those also in `seeing` have their image-understanding add-on paired.
    public nonisolated static func fakeCatalog(_ installed: [AppModel],
                                               seeing: [AppModel] = []) -> EngineCatalog {
        EngineCatalog(models: installed.map { model in
            let slot = EngineProductSlot(modelID: model.modelID)
            let fake = FakeEngineBackend.descriptor(id: model.modelID, engine: slot?.engine ?? .mference,
                                                    product: slot)
            return EngineModelDescriptor(id: fake.id, displayName: slot?.displayName ?? fake.displayName,
                                         engine: fake.engine, format: fake.format, location: fake.location,
                                         sizeBytes: previewSize(model), contextLength: fake.contextLength,
                                         product: slot,
                                         visionProjector: seeing.contains(model) ? EngineVisionProjector(
                                             location: fake.location.appendingPathExtension("mmproj"),
                                             sizeBytes: DownloadSource.imageUnderstanding[model]?.sizeBytes ?? 1)
                                             : nil)
        })
    }

    nonisolated static func previewSize(_ model: AppModel) -> UInt64 {
        switch model {
        case .quick: 2_910_000_000
        case .deep: 12_100_000_000
        case .bonsai27b: 7_910_000_000
        case .inkling2Mini: 17_650_000_000
        case .qwen36: 19_800_000_000
        case .minicpm5: 1_400_000_000
        case .minicpm5_1b: 620_000_000
        case .maple: 7_600_000_000
        case .apple: 0
        }
    }
}
