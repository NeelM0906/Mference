import Foundation

/// The live `SpeechServicing` (Notes spec 6.2, 6.3): dictation, meetings
/// and file transcription on Fluid Speech Pro, all on this Mac.
///
/// `SpeechCoordinator` (an actor) owns the microphone and the models: one
/// live session at a time (`SpeechError.busy` otherwise), file jobs one at
/// a time and paused while a live session runs, models loaded from their
/// verified folders on first use and released after 10 idle minutes. Every
/// stage is injected (`Dependencies`), so tests drive the real pipelines
/// with fixture PCM and scripted stages; `live()` wires the FluidAudio
/// adapters, the AVAudioEngine microphone and the process tap.
public final class SpeechService: SpeechServicing {
    public struct Dependencies: Sendable {
        public var installer: SpeechModelInstaller
        public var loader: any SpeechModelLoading
        public var sources: any AudioSourceFactory
        public var recordings: RecordingStore
        public var power: any PowerEventSource
        public var segmenter: Segmenter.Configuration
        /// Models are released after this long without a session or job; nil keeps them.
        public var idleUnload: Duration?
        /// Told when call audio could not be captured (the tap failed), so
        /// the permission reads as denied from then on.
        public var onCallAudioFailure: @Sendable (SpeechError) async -> Void

        public init(installer: SpeechModelInstaller, loader: any SpeechModelLoading, sources: any AudioSourceFactory,
                    recordings: RecordingStore, power: any PowerEventSource,
                    segmenter: Segmenter.Configuration = Segmenter.Configuration(),
                    idleUnload: Duration? = .seconds(600),
                    onCallAudioFailure: @escaping @Sendable (SpeechError) async -> Void = { _ in }) {
            self.installer = installer
            self.loader = loader
            self.sources = sources
            self.recordings = recordings
            self.power = power
            self.segmenter = segmenter
            self.idleUnload = idleUnload
            self.onCallAudioFailure = onCallAudioFailure
        }
    }

    private let box: SpeechStatusBox
    private let coordinator: SpeechCoordinator
    private let recordings: RecordingStore
    private let installer: SpeechModelInstaller

    public init(_ dependencies: Dependencies) {
        let installer = dependencies.installer
        func initial(_ id: SpeechPackageID) -> SpeechModelState {
            if installer.isInstalled(id) { return .ready }
            let progress = installer.progress(id)
            return progress > 0 ? .paused(progress: progress) : .notInstalled
        }
        box = SpeechStatusBox(SpeechStatus(speech: initial(.speech), speakerLabels: initial(.speakerLabels)))
        coordinator = SpeechCoordinator(dependencies: dependencies, box: box)
        recordings = dependencies.recordings
        self.installer = installer
        let coordinator = coordinator
        Task { await coordinator.listenForPower() }
    }

    // MARK: SpeechServicing

    public var status: SpeechStatus { box.status }
    public func statusUpdates() -> AsyncStream<SpeechStatus> { box.updates() }

    public func downloadSize(of package: SpeechPackageID) -> Int64 { installer.package(package)?.totalBytes ?? 0 }
    public func install(_ package: SpeechPackageID) async throws { try await coordinator.install(package) }
    public func pauseInstall(_ package: SpeechPackageID) async { await coordinator.pauseInstall(package) }
    public func cancelInstall(_ package: SpeechPackageID) async { await coordinator.cancelInstall(package) }
    public func remove(_ package: SpeechPackageID) async throws { try await coordinator.remove(package) }

    /// Offline mode was turned on: downloads stop and keep their bytes.
    public func pauseAllInstalls() async { await coordinator.pauseAllInstalls() }

    public func startDictation(_ options: DictationOptions) async throws -> DictationSession {
        try await coordinator.startDictation(options)
    }

    public func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording {
        try await coordinator.startMeeting(options)
    }

    public func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        try await coordinator.transcribeFile(url, options: options)
    }

    public func cancel(_ job: FileJob.ID) async { await coordinator.cancel(job) }

    public func inputDevices() async -> [AudioInputDevice] { await coordinator.sources.inputDevices() }
    public func runningAudioApps() async -> [RunningAudioApp] { await coordinator.sources.runningAudioApps() }
    public func recordingsSize() async -> Int64 { recordings.totalSize() }

    public func deleteRecordings(for noteID: UUID) async throws {
        do {
            try recordings.delete(noteID: noteID)
        } catch {
            throw SpeechError.failed(error.localizedDescription)
        }
    }

    public func recordingsDirectory(for noteID: UUID) -> URL { recordings.directory(for: noteID) }

    public func recoverInterruptedRecordings() async -> [InterruptedRecording] {
        await recordings.recoverInterrupted()
    }

    public func shutdown() async { await coordinator.shutdown() }
}

/// Owns the microphone, the models and the file queue.
actor SpeechCoordinator {
    private let dependencies: SpeechService.Dependencies
    private let box: SpeechStatusBox
    nonisolated let sources: any AudioSourceFactory
    private let lane = TranscriptionLane()

    private var speech: (recognizer: any SpeechRecognizing, voiceActivity: any VoiceActivityDetecting)?
    private var speechLoad: Task<(recognizer: any SpeechRecognizing, voiceActivity: any VoiceActivityDetecting), Error>?
    private var diarizer: (any SpeakerDiarizing)?

    private var installs: [SpeechPackageID: Task<Void, Error>] = [:]
    private var cancelling: Set<SpeechPackageID> = []

    private enum Live {
        case dictation(UUID, DictationPipeline)
        case meeting(UUID, MeetingPipeline)
        var kind: LiveSessionKind {
            switch self {
            case .dictation: .dictation
            case .meeting: .meeting
            }
        }
        var id: UUID {
            switch self {
            case .dictation(let id, _), .meeting(let id, _): id
            }
        }
    }
    private var live: Live?
    /// Claimed before the first suspension so two starts never both pass.
    private var claiming: LiveSessionKind?
    private var liveWaiters: [CheckedContinuation<Void, Never>] = []

    private struct QueuedJob {
        var job: FileJob
        var file: URL
        var options: FileTranscriptionOptions
        var continuation: AsyncStream<FileJobEvent>.Continuation
    }
    private var queue: [QueuedJob] = []
    private var running: (job: QueuedJob, task: Task<Void, Never>)?

    private var activity = 0
    private var idleTask: Task<Void, Never>?

    init(dependencies: SpeechService.Dependencies, box: SpeechStatusBox) {
        self.dependencies = dependencies
        self.box = box
        sources = dependencies.sources
    }

    // MARK: Install

    func install(_ id: SpeechPackageID) async throws {
        if let running = installs[id] { return try await running.value }
        let task = Task { try await self.runInstall(id) }
        installs[id] = task
        defer { installs[id] = nil }
        try await task.value
    }

    private func runInstall(_ id: SpeechPackageID) async throws {
        let installer = dependencies.installer
        let box = box
        box.setPackage(id, .downloading(progress: installer.progress(id)))
        do {
            try await installer.install(id) { progress in
                box.setPackage(id, .downloading(progress: progress))
            }
        } catch is CancellationError {
            if cancelling.remove(id) != nil {
                box.setPackage(id, .notInstalled)
            } else {
                box.setPackage(id, .paused(progress: installer.progress(id)))
            }
            throw CancellationError()
        } catch let error as SpeechError {
            if error == .offline {
                let progress = installer.progress(id)
                box.setPackage(id, progress > 0 ? .paused(progress: progress) : .notInstalled)
            } else {
                box.setPackage(id, .failed(error))
            }
            throw error
        }
        // The first load: the Neural Engine compile, several seconds.
        box.setPackage(id, .preparing)
        do {
            switch id {
            case .speech: _ = try await loadSpeech()
            case .speakerLabels: _ = try await loadDiarizer()
            }
        } catch {
            let reason = error as? SpeechError ?? .failed(String(describing: error))
            box.setPackage(id, .failed(reason))
            throw reason
        }
        box.setPackage(id, .ready)
        scheduleIdleUnload()
    }

    func pauseInstall(_ id: SpeechPackageID) async {
        await dependencies.installer.pause(id)
    }

    func pauseAllInstalls() async {
        await dependencies.installer.pauseAll()
    }

    func cancelInstall(_ id: SpeechPackageID) async {
        if installs[id] != nil { cancelling.insert(id) }
        await dependencies.installer.cancel(id)
        if let running = installs[id] { _ = try? await running.value }
        cancelling.remove(id)
        if !dependencies.installer.isInstalled(id) { box.setPackage(id, .notInstalled) }
    }

    func remove(_ id: SpeechPackageID) async throws {
        if id == .speech, let live { throw SpeechError.busy(live.kind) }
        if id == .speech, running != nil || !queue.isEmpty {
            throw SpeechError.failed("files are still transcribing")
        }
        switch id {
        case .speech:
            speech = nil
            speechLoad = nil
        case .speakerLabels:
            diarizer = nil
        }
        try await dependencies.installer.remove(id)
        box.setPackage(id, .notInstalled)
    }

    // MARK: Models

    private func loadSpeech() async throws -> (recognizer: any SpeechRecognizing, voiceActivity: any VoiceActivityDetecting) {
        if let speech { return speech }
        if let speechLoad { return try await speechLoad.value }
        guard dependencies.installer.isInstalled(.speech) else { throw SpeechError.notInstalled(.speech) }
        let loader = dependencies.loader
        let root = dependencies.installer.root
        let task = Task { try await loader.loadSpeech(from: root) }
        speechLoad = task
        defer { speechLoad = nil }
        let wasReady = box.status.speech.isReady
        if wasReady { box.setPackage(.speech, .preparing) }
        do {
            let loaded = try await task.value
            speech = loaded
            if wasReady { box.setPackage(.speech, .ready) }
            return loaded
        } catch {
            let reason = error as? SpeechError ?? .failed(String(describing: error))
            box.setPackage(.speech, .failed(reason))
            throw reason
        }
    }

    private func loadDiarizer() async throws -> any SpeakerDiarizing {
        if let diarizer { return diarizer }
        guard dependencies.installer.isInstalled(.speakerLabels) else { throw SpeechError.notInstalled(.speakerLabels) }
        let loaded = try await dependencies.loader.loadSpeakerLabels(from: dependencies.installer.root)
        diarizer = loaded
        return loaded
    }

    /// The diarizer when speaker labels are installed and asked for; a
    /// failed load leaves "You / Others".
    private func optionalDiarizer(_ wanted: Bool) async -> (any SpeakerDiarizing)? {
        guard wanted, dependencies.installer.isInstalled(.speakerLabels) else { return nil }
        return try? await loadDiarizer()
    }

    private func beginActivity() {
        activity += 1
        idleTask?.cancel()
        idleTask = nil
    }

    private func endActivity() {
        activity = max(0, activity - 1)
        scheduleIdleUnload()
    }

    private func scheduleIdleUnload() {
        guard activity == 0, let delay = dependencies.idleUnload else { return }
        idleTask?.cancel()
        idleTask = Task {
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self.unloadIfIdle()
        }
    }

    private func unloadIfIdle() {
        guard activity == 0, live == nil, running == nil else { return }
        speech = nil
        diarizer = nil
    }

    /// Whether the models are in memory (tests of the idle unload).
    var isLoaded: Bool { speech != nil }

    // MARK: Live sessions

    private func claim(_ kind: LiveSessionKind) throws {
        if let live { throw SpeechError.busy(live.kind) }
        if let claiming { throw SpeechError.busy(claiming) }
        claiming = kind
    }

    /// The start finished or failed: the claim ends; waiting files go on
    /// if no session came of it.
    private func endClaim() {
        claiming = nil
        if live == nil { wakeWaiters() }
    }

    private func release(_ id: UUID) {
        guard live?.id == id else { return }
        live = nil
        box.mutate { $0.liveSession = nil }
        wakeWaiters()
        endActivity()
    }

    private func wakeWaiters() {
        let waiters = liveWaiters
        liveWaiters = []
        for waiter in waiters { waiter.resume() }
    }

    /// Suspends while a live session holds the microphone; true if it waited.
    func waitWhileLive(job: FileJob.ID) async -> Bool {
        guard live != nil || claiming != nil else { return false }
        box.updateJob(job) { $0.phase = .paused }
        if running?.job.job.id == job { running?.job.continuation.yield(.paused) }
        while live != nil || claiming != nil, !Task.isCancelled {
            await withTaskCancellationHandler {
                await withCheckedContinuation { liveWaiters.append($0) }
            } onCancel: {
                Task { await self.wakeWaiters() }
            }
        }
        box.updateJob(job) { $0.phase = .running }
        return true
    }

    func startDictation(_ options: DictationOptions) async throws -> DictationSession {
        try claim(.dictation)
        defer { endClaim() }
        let stages = try await loadSpeech()
        let source = try sources.microphone(deviceID: options.microphoneID)
        let id = UUID()
        let (events, continuation) = AsyncStream.makeStream(of: DictationEvent.self)
        var configuration = dependencies.segmenter
        configuration.endOfSpeechSilence = TimeInterval(options.endOfSpeechSilence.components.seconds)
            + Double(options.endOfSpeechSilence.components.attoseconds) / 1e18
        let segmenter = TrackSegmenter(track: .microphone, detector: try await stages.voiceActivity.makeStream(),
                                       frameLength: stages.voiceActivity.frameLength, configuration: configuration)
        let pipeline = DictationPipeline(source: source, segmenter: segmenter, recognizer: stages.recognizer,
                                         lane: lane, options: options, continuation: continuation,
                                         onEnd: { [weak self] in await self?.release(id) })
        try await pipeline.start()
        beginActivity()
        live = .dictation(id, pipeline)
        box.mutate { $0.liveSession = LiveSessionStatus(kind: .dictation, noteID: nil, startedAt: .now) }
        return DictationSession(id: id, events: events,
                                stop: { await pipeline.stop() },
                                setLanguage: { await pipeline.setLanguage($0) })
    }

    func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording {
        try claim(.meeting)
        defer { endClaim() }
        let stages = try await loadSpeech()
        let startedAt = Date.now
        let recording: RecordingSession
        do {
            recording = try dependencies.recordings.begin(noteID: options.noteID, startedAt: startedAt,
                                                          keepAudio: options.keepAudio)
        } catch let error as SpeechError {
            throw error
        } catch {
            throw SpeechError.failed(error.localizedDescription)
        }
        let diarizer = await optionalDiarizer(options.speakerLabels && options.callAudio != .off)
        let id = UUID()
        let (events, continuation) = AsyncStream.makeStream(of: MeetingEvent.self)
        let box = box
        let pipeline = MeetingPipeline(
            options: options, startedAt: startedAt,
            stages: .init(recognizer: stages.recognizer, voiceActivity: stages.voiceActivity, diarizer: diarizer),
            lane: lane, sources: sources, recording: recording, segmenter: dependencies.segmenter,
            continuation: continuation,
            onLiveStatus: { status in box.mutate { if $0.liveSession?.kind == .meeting { $0.liveSession = status } } },
            onCallAudioFailure: dependencies.onCallAudioFailure,
            onEnd: { [weak self] in await self?.release(id) })
        box.mutate { $0.liveSession = LiveSessionStatus(kind: .meeting, noteID: options.noteID, startedAt: startedAt) }
        do {
            try await pipeline.start()
        } catch {
            box.mutate { $0.liveSession = nil }
            _ = try? await recording.finish(keepAudio: false)
            try? dependencies.recordings.delete(noteID: options.noteID)
            throw error as? SpeechError ?? .failed(String(describing: error))
        }
        beginActivity()
        live = .meeting(id, pipeline)
        let controls = MeetingRecording.Controls(pause: { await pipeline.pause() },
                                                 resume: { await pipeline.resume() },
                                                 markMoment: { await pipeline.markMoment() },
                                                 stop: { await pipeline.stop() })
        return MeetingRecording(noteID: options.noteID, startedAt: startedAt, events: events, controls: controls)
    }

    // MARK: Power

    func listenForPower() async {
        for await event in dependencies.power.events() {
            switch (event, live) {
            case (.willSleep, .dictation(_, let pipeline)?): await pipeline.stop()
            case (.willSleep, .meeting(_, let pipeline)?): await pipeline.willSleep()
            case (.didWake, .meeting(_, let pipeline)?): await pipeline.didWake()
            default: break
            }
        }
    }

    // MARK: Files

    func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        guard dependencies.installer.isInstalled(.speech) else { throw SpeechError.notInstalled(.speech) }
        let directory = dependencies.recordings.directory(for: options.noteID)
        let copy = directory.appending(path: url.lastPathComponent)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            if copy.standardizedFileURL != url.standardizedFileURL {
                try? FileManager.default.removeItem(at: copy)
                try FileManager.default.copyItem(at: url, to: copy)
            }
        } catch {
            throw SpeechError.failed(error.localizedDescription)
        }
        let (events, continuation) = AsyncStream.makeStream(of: FileJobEvent.self)
        let job = FileJob(noteID: options.noteID, fileName: url.lastPathComponent, events: events)
        queue.append(QueuedJob(job: job, file: copy, options: options, continuation: continuation))
        box.mutate { $0.fileQueue.append(FileJobStatus(id: job.id, noteID: options.noteID, fileName: job.fileName)) }
        continuation.yield(.queued(position: queue.count - 1 + (running == nil ? 0 : 1)))
        startNextJob()
        return job
    }

    func cancel(_ id: FileJob.ID) async {
        if let index = queue.firstIndex(where: { $0.job.id == id }) {
            finishJob(queue.remove(at: index), .cancelled)
            return
        }
        guard let running, running.job.job.id == id else { return }
        running.task.cancel()
        await running.task.value
    }

    private func startNextJob() {
        guard running == nil, !queue.isEmpty else { return }
        let next = queue.removeFirst()
        let task = Task { await self.runJob(next) }
        running = (next, task)
    }

    private func runJob(_ queued: QueuedJob) async {
        beginActivity()
        defer { endActivity() }
        let id = queued.job.id
        let box = box
        do {
            let stages = try await loadSpeech()
            let transcriber = FileTranscriber(
                recognizer: stages.recognizer, voiceActivity: stages.voiceActivity,
                diarizer: await optionalDiarizer(queued.options.speakerLabels),
                lane: lane, segmenter: dependencies.segmenter,
                waitWhileLive: { [weak self] in await self?.waitWhileLive(job: id) ?? false })
            box.updateJob(id) { $0.phase = .running }
            var result = try await transcriber.run(file: queued.file, options: queued.options,
                                                   events: queued.continuation) { phase, fraction, remaining, language in
                box.updateJob(id) { job in
                    job.phase = phase
                    job.fraction = fraction
                    job.remaining = remaining
                    job.languageCode = language
                }
            }
            box.updateJob(id) { $0.duration = result.duration }
            if queued.options.keepAudio {
                result.tracks = [RecordedTrack(fileName: queued.file.lastPathComponent, kind: .file, startOffset: 0,
                                               duration: result.duration)]
                result.audioDirectory = queued.file.deletingLastPathComponent()
            } else {
                try? FileManager.default.removeItem(at: queued.file)
            }
            finishJob(queued, .finished(result))
        } catch is CancellationError {
            try? FileManager.default.removeItem(at: queued.file)
            finishJob(queued, .cancelled)
        } catch {
            finishJob(queued, .failed(error as? SpeechError ?? .failed(String(describing: error))))
        }
    }

    private func finishJob(_ queued: QueuedJob, _ event: FileJobEvent) {
        let id = queued.job.id
        queue.removeAll { $0.job.id == id }
        if running?.job.job.id == id { running = nil }
        queued.continuation.yield(event)
        queued.continuation.finish()
        box.mutate { status in
            switch event {
            case .finished:
                if let index = status.fileQueue.firstIndex(where: { $0.id == id }) {
                    status.fileQueue[index].phase = .done
                    status.fileQueue[index].fraction = 1
                    status.fileQueue[index].remaining = 0
                }
            case .failed:
                if let index = status.fileQueue.firstIndex(where: { $0.id == id }) { status.fileQueue[index].phase = .failed }
            default:
                status.fileQueue.removeAll { $0.id == id }
            }
        }
        startNextJob()
        let offset = running == nil ? 0 : 1
        for (index, waiting) in queue.enumerated() { waiting.continuation.yield(.queued(position: index + offset)) }
    }

    // MARK: Quit

    func shutdown() async {
        for queued in queue { finishJob(queued, .cancelled) }
        let current = running
        current?.task.cancel()
        switch live {
        case .dictation(_, let pipeline)?: await pipeline.stop()
        case .meeting(_, let pipeline)?: await pipeline.stopAndFinish()
        case nil: break
        }
        await current?.task.value
        idleTask?.cancel()
    }
}
