import Foundation
import IOKit.ps
import Observation
import Synchronization

/// Find's settings (Settings › Find), saved as soon as they change.
@MainActor
@Observable
public final class FindSettings {
    public enum Key {
        public static let enabled = "find.enabled"
        public static let roots = "find.roots"
        public static let includeCode = "find.includeCode"
        public static let readImages = "find.readImages"
        public static let readMedia = "find.readMedia"
        public static let photosLibrary = "find.photosLibrary"
        public static let learnOnlyOnPower = "find.learnOnlyOnPower"
        public static let filesChats = "find.filesChats"
        public static let lastEventID = "find.lastEventID"
    }

    @ObservationIgnored private let store: any SettingsStore

    public var enabled: Bool { didSet { store.set(enabled, forKey: Key.enabled) } }
    public var roots: [URL] {
        didSet { store.set(Self.encode(roots.map(\.path)), forKey: Key.roots) }
    }
    public var includeCode: Bool { didSet { store.set(includeCode, forKey: Key.includeCode) } }
    public var readImages: Bool { didSet { store.set(readImages, forKey: Key.readImages) } }
    /// Learn photos, video and audio by what they show and sound like (the
    /// media encoders, a second download).
    public var readMedia: Bool { didSet { store.set(readMedia, forKey: Key.readMedia) } }
    /// Read the Photos library (with the Photos permission).
    public var photosLibrary: Bool { didSet { store.set(photosLibrary, forKey: Key.photosLibrary) } }
    /// Learn meaning only on the power adapter (words always stay current).
    public var learnOnlyOnPower: Bool { didSet { store.set(learnOnlyOnPower, forKey: Key.learnOnlyOnPower) } }
    /// Chats with Files on.
    public var filesChats: Set<UUID> {
        didSet { store.set(Self.encode(filesChats.map(\.uuidString).sorted()), forKey: Key.filesChats) }
    }

    /// Documents, Desktop and Downloads (locked default).
    public static var defaultRoots: [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return ["Documents", "Desktop", "Downloads"].map { home.appending(path: $0, directoryHint: .isDirectory) }
    }

    public init(store: any SettingsStore = UserDefaults.standard) {
        self.store = store
        enabled = store.object(forKey: Key.enabled) as? Bool ?? false
        roots = Self.decode(store.object(forKey: Key.roots) as? String)?.map { URL(filePath: $0, directoryHint: .isDirectory) }
            ?? Self.defaultRoots
        includeCode = store.object(forKey: Key.includeCode) as? Bool ?? false
        readImages = store.object(forKey: Key.readImages) as? Bool ?? true
        readMedia = store.object(forKey: Key.readMedia) as? Bool ?? false
        photosLibrary = store.object(forKey: Key.photosLibrary) as? Bool ?? false
        learnOnlyOnPower = store.object(forKey: Key.learnOnlyOnPower) as? Bool ?? true
        filesChats = Set((Self.decode(store.object(forKey: Key.filesChats) as? String) ?? []).compactMap(UUID.init))
    }

    var lastEventID: UInt64? {
        get { (store.object(forKey: Key.lastEventID) as? String).flatMap(UInt64.init) }
        set { store.set(newValue.map(String.init), forKey: Key.lastEventID) }
    }

    var rules: FindCrawlRules { FindCrawlRules(includeCode: includeCode, readImages: readImages, readMedia: readMedia) }

    private static func encode(_ values: [String]) -> String {
        (try? String(decoding: JSONEncoder().encode(values), as: UTF8.self)) ?? "[]"
    }

    private static func decode(_ string: String?) -> [String]? {
        guard let string else { return nil }
        return try? JSONDecoder().decode([String].self, from: Data(string.utf8))
    }
}

/// How Find gets its search model.
public protocol FindModelProviding: Sendable {
    func currentState() async -> FindModelState
    func ensureDownloaded(progress: @escaping @Sendable (FindModelState) -> Void) async throws -> URL
    func cancel() async
    func delete() async
}

extension FindModelStore: FindModelProviding {}

/// A model that is already there (tests, previews): no download.
public struct PresentFindModel: FindModelProviding {
    public var url: URL

    public init(url: URL = URL(filePath: "/dev/null")) {
        self.url = url
    }

    public func currentState() async -> FindModelState { .ready(url) }
    public func ensureDownloaded(progress: @escaping @Sendable (FindModelState) -> Void) async throws -> URL {
        progress(.ready(url))
        return url
    }
    public func cancel() async {}
    public func delete() async {}
}

/// Find for the app: settings, the index and its background work, the search
/// model, search itself and chat's Files tool. One per app (`AppServices`).
@MainActor
@Observable
public final class FindService: ChatFilesProviding {
    public struct Dependencies: Sendable {
        /// Where the index lives; nil keeps it in memory (tests).
        public var indexDirectory: URL?
        public var model: any FindModelProviding
        /// The image and audio encoders, fetched when Photos, video and
        /// audio is turned on.
        public var mediaModel: any FindModelProviding
        /// The embedder for the text model and, when there, the media model.
        public var makeEmbedder: @Sendable (URL, URL?) -> any FindTextEmbedding
        public var extractor: any FindExtracting
        /// Watch folders with FSEvents (off in tests).
        public var watchFolders: Bool
        /// Read `kMDItemLastUsedDate` so recently opened files go first.
        public var recentFiles: @Sendable ([URL]) -> [String: Date]
        /// Whether the Mac runs on its battery.
        public var onBattery: @Sendable () -> Bool
        /// The Photos library, read when Settings › Find › Photos library is
        /// on; nil where there is none (tests, previews).
        public var photoLibrary: (any FindPhotoLibrary)?
        /// Asks for the Photos permission.
        public var permissions: (any PermissionBroker)?

        public init(indexDirectory: URL?, model: any FindModelProviding,
                    mediaModel: any FindModelProviding = PresentFindModel(),
                    makeEmbedder: @escaping @Sendable (URL, URL?) -> any FindTextEmbedding,
                    extractor: any FindExtracting, watchFolders: Bool,
                    recentFiles: @escaping @Sendable ([URL]) -> [String: Date] = { _ in [:] },
                    onBattery: @escaping @Sendable () -> Bool = { false },
                    photoLibrary: (any FindPhotoLibrary)? = nil, permissions: (any PermissionBroker)? = nil) {
            self.indexDirectory = indexDirectory
            self.model = model
            self.mediaModel = mediaModel
            self.makeEmbedder = makeEmbedder
            self.extractor = extractor
            self.watchFolders = watchFolders
            self.recentFiles = recentFiles
            self.onBattery = onBattery
            self.photoLibrary = photoLibrary
            self.permissions = permissions
        }
    }

    public let settings: FindSettings
    public private(set) var status = FindIndexStatus()
    public private(set) var modelState: FindModelState = .notDownloaded
    /// The image and audio encoders' download (Photos, video and audio).
    public private(set) var mediaModelState: FindModelState = .notDownloaded
    /// Why Find could not start (the index could not open), shown in Settings.
    public private(set) var lastError: String?
    /// The Photos permission, as last checked (Settings › Find).
    public private(set) var photosAccess: PermissionState = .notDetermined
    /// Photos and videos found in the library, as last synced.
    public private(set) var photosCount: Int?

    @ObservationIgnored private let dependencies: Dependencies
    @ObservationIgnored private var index: FindIndex?
    @ObservationIgnored private var indexer: FindIndexer?
    @ObservationIgnored private var embedder: (any FindTextEmbedding)?
    @ObservationIgnored private var watcher: FindFolderWatcher?
    @ObservationIgnored private var engineTask: Task<Void, Never>?
    @ObservationIgnored private var idleTask: Task<Void, Never>?
    @ObservationIgnored private let engineBusy = BusyFlag()
    /// The last query embedded while typing.
    @ObservationIgnored private var ahead: (text: String, task: Task<[Float]?, Never>)?

    /// Free the model and the vector copy after this long without searching.
    public static let idleAfter: Duration = .seconds(180)

    public init(settings: FindSettings, dependencies: Dependencies, engine: (any EngineServicing)? = nil) {
        self.settings = settings
        self.dependencies = dependencies
        if let engine {
            let updates = engine.statusUpdates()
            let busy = engineBusy
            engineTask = Task { [weak self] in
                for await status in updates {
                    let wasBusy = busy.swap(status.isGenerating)
                    if wasBusy, !status.isGenerating { await self?.indexer?.kick() }
                }
            }
        }
    }

    /// The app's Find: the index and model under Application Support, the
    /// helper process for reading, EmbeddingGemma 2 through the engine.
    public static func live(settings: FindSettings, engine: any EngineServicing, helper: URL?,
                            permissions: (any PermissionBroker)? = nil) -> FindService {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "Mference/Find", directoryHint: .isDirectory)
        // A few helper processes read side by side (one per four cores, up
        // to three); without a helper, this process reads.
        let files: any FindExtracting = helper.map { executable in
            FindReaderPool((0..<FindReaderPool.size()).map { _ in HelperFindExtractor(executable: executable) })
        } ?? InProcessFindExtractor()
        let library = LivePhotoLibrary()
        let extractor = FindLibraryReader(base: files, library: library)
        let models = base.appending(path: "Model", directoryHint: .isDirectory)
        return FindService(settings: settings, dependencies: Dependencies(
            indexDirectory: base.appending(path: "Index", directoryHint: .isDirectory),
            model: FindModelStore(directory: models),
            mediaModel: FindModelStore(directory: models, file: .media),
            makeEmbedder: { FindEmbedderLive(modelPath: $0, mediaModelPath: $1) },
            extractor: extractor, watchFolders: true,
            recentFiles: { FindRecentFiles.lastUsed(under: $0) },
            onBattery: FindPower.isOnBattery, photoLibrary: library, permissions: permissions), engine: engine)
    }

    /// Previews and tests: in memory, the hashing embedder, no download.
    public static func preview(settings: FindSettings = FindSettings(store: InMemorySettingsStore())) -> FindService {
        FindService(settings: settings, dependencies: Dependencies(
            indexDirectory: nil, model: PresentFindModel(), mediaModel: PresentFindModel(),
            makeEmbedder: { HashingTextEmbedder(media: $1 != nil) },
            extractor: InProcessFindExtractor(), watchFolders: false))
    }

    // MARK: Lifecycle

    /// Starts Find if it is on (app launch).
    public func start() async {
        modelState = await dependencies.model.currentState()
        mediaModelState = await dependencies.mediaModel.currentState()
        await refreshPhotosAccess()
        guard settings.enabled else { return }
        await startPipeline(fullSync: true)
    }

    /// "Turn on Find": saves the folders, starts reading at once and fetches
    /// the search model alongside (words work before meaning does).
    public func turnOn(roots: [URL]) async {
        settings.roots = roots.map(\.standardizedFileURL)
        settings.enabled = true
        await startPipeline(fullSync: true)
    }

    /// Stops Find; the index stays on disk until deleted.
    public func turnOff() async {
        settings.enabled = false
        await stopPipeline()
        status = FindIndexStatus()
    }

    /// Quit: stop work, free the model, remember where FSEvents got to.
    public func shutdown() async {
        engineTask?.cancel()
        await stopPipeline()
    }

    /// Deletes the index (Settings › Find › Delete index). Find turns off.
    public func deleteIndex() async {
        let index = self.index
        await turnOff()
        await index?.deleteFiles()
        if self.index == nil, let directory = dependencies.indexDirectory {
            try? FileManager.default.removeItem(at: directory.appending(path: "find.sqlite"))
            try? FileManager.default.removeItem(at: directory.appending(path: "find.sqlite-wal"))
            try? FileManager.default.removeItem(at: directory.appending(path: "find.sqlite-shm"))
        }
        settings.lastEventID = nil
        self.index = nil
    }

    /// Starts over from an empty index.
    public func rebuild() async {
        let roots = settings.roots
        await deleteIndex()
        await turnOn(roots: roots)
    }

    public func addFolder(_ url: URL) async {
        let url = url.standardizedFileURL
        guard !settings.roots.contains(url) else { return }
        settings.roots.append(url)
        await reconfigure()
    }

    public func removeFolder(_ url: URL) async {
        settings.roots.removeAll { $0 == url.standardizedFileURL }
        await reconfigure()
    }

    public func setIncludeCode(_ on: Bool) async {
        settings.includeCode = on
        await reconfigure()
    }

    public func setReadImages(_ on: Bool) async {
        settings.readImages = on
        await reconfigure()
    }

    /// Photos, video and audio by what they show and sound like. On fetches
    /// the media encoders; either way those files are read again.
    public func setReadMedia(_ on: Bool) async {
        guard settings.readMedia != on else { return }
        settings.readMedia = on
        await reconfigure()
        try? await indexer?.requeue(kinds: [.image, .video, .audio])
        if on, settings.enabled { Task { await ensureMediaModel() } }
    }

    /// The Photos library. On asks for the Photos permission first (the
    /// reason, then the system prompt); without it the switch stays off and
    /// `photosAccess` says why.
    public func setPhotosLibrary(_ on: Bool) async {
        guard on else {
            settings.photosLibrary = false
            photosCount = nil
            await reconfigure()
            return
        }
        guard dependencies.photoLibrary != nil, let permissions = dependencies.permissions else { return }
        photosAccess = await permissions.request(.photos)
        guard photosAccess == .granted else { return }
        settings.photosLibrary = true
        await reconfigure()
        watchPhotos()
        await refreshPhotosCount()
    }

    private func refreshPhotosCount() async {
        guard settings.photosLibrary, let index else { return }
        photosCount = try? await index.fileCount(root: FindPhotos.rootPath)
    }

    /// Reads the Photos permission again (Settings › Find appears, or the
    /// person comes back from System Settings).
    public func refreshPhotosAccess() async {
        guard let permissions = dependencies.permissions else { return }
        photosAccess = await permissions.state(of: .photos)
        // Taken away in System Settings: stop reading the library.
        if settings.photosLibrary, photosAccess != .granted, photosAccess != .notDetermined {
            await setPhotosLibrary(false)
        }
        await refreshPhotosCount()
    }

    /// System Settings › Privacy & Security › Photos.
    public func openPhotosSettings() {
        guard let permissions = dependencies.permissions else { return }
        Task { await permissions.openSystemSettings(for: .photos) }
    }

    /// Re-reads the library when it changes (a few seconds after the last
    /// change, so an import is one sync).
    private func watchPhotos() {
        guard settings.photosLibrary, let library = dependencies.photoLibrary else { return }
        library.observeChanges { [weak self] in
            Task { @MainActor in self?.photosChanged() }
        }
    }

    @ObservationIgnored private var photosSync: Task<Void, Never>?

    private func photosChanged() {
        photosSync?.cancel()
        let indexer = self.indexer
        photosSync = Task {
            try? await Task.sleep(for: .seconds(5))
            guard !Task.isCancelled else { return }
            try? await indexer?.syncPhotos()
            await self.refreshPhotosCount()
        }
    }

    /// Retries the media encoders' download after a failure.
    public func retryMediaModelDownload() {
        Task { await ensureMediaModel() }
    }

    public func setLearnOnlyOnPower(_ on: Bool) async {
        settings.learnOnlyOnPower = on
        await indexer?.kick()
    }

    /// Retries the model download after a failure.
    public func retryModelDownload() {
        Task { await ensureModel() }
    }

    public func skippedFiles() async -> [FindSkippedFile] {
        (try? await index?.skippedFiles()) ?? []
    }

    private func reconfigure() async {
        guard let indexer else { return }
        try? await indexer.setConfiguration(configuration)
        startWatching()
    }

    private var configuration: FindIndexer.Configuration {
        FindIndexer.Configuration(roots: settings.roots, rules: settings.rules,
                                  photos: settings.photosLibrary && dependencies.photoLibrary != nil)
    }

    private func startPipeline(fullSync: Bool) async {
        if index == nil {
            do {
                index = try FindIndex(directory: dependencies.indexDirectory)
                lastError = nil
            } catch {
                lastError = "Find could not open its index: \(error)"
                return
            }
        }
        guard let index else { return }
        if indexer == nil {
            // Pictures prepared ahead live in a temporary folder; anything
            // left from an earlier run is removed (and decoded again when it
            // is learned). Files from this run stay.
            Self.removeEarlierPrepared()
            try? await index.forgetPrepared()
            let busy = engineBusy
            let onBattery = dependencies.onBattery
            let settings = self.settings
            let gate = FindIndexer.Gate(
                reading: { busy.value ? "Paused while Chat is answering" : nil },
                learning: {
                    if busy.value { return "Paused while Chat is answering" }
                    if await MainActor.run(body: { settings.learnOnlyOnPower }), onBattery() {
                        return "Waiting for the power adapter"
                    }
                    if ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue {
                        return "Waiting for the Mac to cool down"
                    }
                    return nil
                })
            let indexer = FindIndexer(index: index, extractor: dependencies.extractor, embedder: embedder,
                                      configuration: configuration, gate: gate,
                                      recentFiles: dependencies.recentFiles,
                                      photoLibrary: dependencies.photoLibrary,
                                      onStatus: { [weak self] status in
                                          Task { @MainActor in self?.status = status }
                                      })
            self.indexer = indexer
            await indexer.start()
        }
        startWatching()
        watchPhotos()
        if fullSync {
            let indexer = self.indexer
            Task { try? await indexer?.syncAll() }
        }
        Task { await ensureModel() }
    }

    private func stopPipeline() async {
        watcher?.stop()
        watcher = nil
        idleTask?.cancel()
        await indexer?.stop()
        indexer = nil
        await embedder?.unload()
        await index?.trimMemory()
    }

    private func startWatching() {
        guard dependencies.watchFolders, settings.enabled else { return }
        let indexer = self.indexer
        let settings = self.settings
        let watcher = self.watcher ?? FindFolderWatcher { batch in
            Task {
                if batch.needsRescan {
                    try? await indexer?.syncAll()
                } else {
                    try? await indexer?.update(paths: batch.paths)
                }
                await MainActor.run { settings.lastEventID = batch.lastEventID }
            }
        }
        self.watcher = watcher
        watcher.start(roots: settings.roots, since: settings.lastEventID)
        if settings.lastEventID == nil { settings.lastEventID = FindFolderWatcher.currentEventID }
    }

    private func ensureModel() async {
        do {
            let url = try await dependencies.model.ensureDownloaded { [weak self] state in
                Task { @MainActor in self?.modelState = state }
            }
            modelState = .ready(url)
            if embedder == nil {
                let media: URL? = if case .ready(let mediaURL) = mediaModelState { mediaURL } else { nil }
                await install(dependencies.makeEmbedder(url, settings.readMedia ? media : nil))
            }
            if settings.readMedia { await ensureMediaModel() }
        } catch {
            if case .failed = modelState {} else if !(error is CancellationError) {
                modelState = .failed(error.localizedDescription)
            }
        }
    }

    /// Fetches the media encoders and switches to an embedder that has them.
    private func ensureMediaModel() async {
        do {
            let url = try await dependencies.mediaModel.ensureDownloaded { [weak self] state in
                Task { @MainActor in self?.mediaModelState = state }
            }
            mediaModelState = .ready(url)
            guard settings.readMedia, case .ready(let text) = modelState, embedder?.supportsMedia != true else { return }
            await embedder?.unload()
            await install(dependencies.makeEmbedder(text, url))
        } catch {
            if case .failed = mediaModelState {} else if !(error is CancellationError) {
                mediaModelState = .failed(error.localizedDescription)
            }
        }
    }

    private func install(_ embedder: any FindTextEmbedding) async {
        self.embedder = embedder
        await checkFingerprint(embedder)
        await indexer?.setEmbedder(embedder)
    }

    /// A different model or prompt makes stored vectors meaningless.
    private func checkFingerprint(_ embedder: any FindTextEmbedding) async {
        guard let index else { return }
        let stored = try? await index.meta("embedder")
        if stored != embedder.fingerprint {
            if stored != nil { try? await index.clearVectors() }
            try? await index.setMeta("embedder", embedder.fingerprint)
        }
        // Media pieces planned or embedded another way are read again.
        guard embedder.supportsMedia else { return }
        let storedMedia = try? await index.meta("mediaEmbedder")
        if storedMedia != embedder.mediaFingerprint {
            if storedMedia != nil { try? await indexer?.requeue(kinds: [.image, .video, .audio]) }
            try? await index.setMeta("mediaEmbedder", embedder.mediaFingerprint)
        }
    }

    /// When this process started. Files prepared since then are this run's,
    /// even those prepared before Find first started here (tests run side by
    /// side, each starting its own Find).
    private static let started = processStart() ?? Date()

    /// The process's start time from the kernel; nil if it cannot be read.
    static func processStart() -> Date? {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var name: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        guard sysctl(&name, u_int(name.count), &info, &size, nil, 0) == 0, size > 0 else { return nil }
        let start = info.kp_proc.p_un.__p_starttime
        return Date(timeIntervalSince1970: TimeInterval(start.tv_sec) + TimeInterval(start.tv_usec) / 1_000_000)
    }

    static func removeEarlierPrepared() {
        let cutoff = started
        let manager = FileManager.default
        let files = (try? manager.contentsOfDirectory(at: FindMediaFolder.ready,
                                                      includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        for file in files {
            let modified = (try? file.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
            if (modified ?? .distantPast) < cutoff { try? manager.removeItem(at: file) }
        }
    }

    // MARK: Search

    /// Whether search can run (Find on and its index open).
    public var isSearchable: Bool { settings.enabled && index != nil }

    /// Word matches only, for each keystroke.
    public func searchWords(_ query: FindQuery) async -> FindResults {
        guard let index else { return .empty }
        noteActivity()
        return (try? await FindSearcher(index: index, embedder: nil).searchWords(query)) ?? .empty
    }

    /// Embeds the query while the person types, so `search` finds it ready.
    public func prepare(_ text: String) {
        guard let embedder, !text.trimmingCharacters(in: .whitespaces).isEmpty, ahead?.text != text else { return }
        let prompt = FindEmbeddingPrompts.query(text)
        ahead = (text, Task { try? await embedder.embed(query: prompt) })
    }

    /// Words and meaning.
    public func search(_ query: FindQuery) async -> FindResults {
        guard let index else { return .empty }
        noteActivity()
        var vector: [Float]?
        if let ahead, ahead.text == query.text { vector = await ahead.task.value }
        let searcher = FindSearcher(index: index, embedder: embedder)
        return (try? await searcher.search(query, vector: vector)) ?? .empty
    }

    /// Frees the model and the vector copy after a quiet spell (the indexer
    /// loads the model again when it has passages to learn).
    private func noteActivity() {
        idleTask?.cancel()
        let embedder = self.embedder
        let index = self.index
        let status = self.status
        idleTask = Task {
            try? await Task.sleep(for: Self.idleAfter)
            guard !Task.isCancelled else { return }
            await index?.trimMemory()
            if status.phase != .learning { await embedder?.unload() }
        }
    }

    // MARK: Chat (ChatFilesProviding)

    public var filesUnavailableReason: String? {
        isSearchable ? nil : "Turn on Find first"
    }

    public func filesEnabled(for conversation: UUID) -> Bool {
        settings.filesChats.contains(conversation)
    }

    public func setFilesEnabled(_ enabled: Bool, for conversation: UUID) {
        if enabled { settings.filesChats.insert(conversation) } else { settings.filesChats.remove(conversation) }
    }

    public func makeFilesToolbox() -> (any ChatToolbox)? {
        guard let index else { return nil }
        let embedder = self.embedder
        return FilesToolbox { text in
            try await FindSearcher(index: index, embedder: embedder).search(FindQuery(text, limit: 10))
        }
    }
}

/// Whether Chat is generating, shared with the indexer's gate.
final class BusyFlag: Sendable {
    private let state = Mutex(false)
    var value: Bool { state.withLock { $0 } }
    /// Sets the flag; returns the old value.
    func swap(_ new: Bool) -> Bool {
        state.withLock { value in
            defer { value = new }
            return value
        }
    }
}

/// The power source.
public enum FindPower {
    public static let isOnBattery: @Sendable () -> Bool = {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String? else { return false }
        return type == kIOPMBatteryPowerKey
    }
}
