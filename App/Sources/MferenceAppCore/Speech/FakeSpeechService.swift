import Foundation
import Synchronization

/// A `SpeechServicing` for tests and previews: no model, no microphone, no
/// call tap, no network, no TCC. Package states, dictation phrases, meeting
/// segments and file results are scripted; every call is recorded.
///
/// Live sessions follow the real rules: one at a time (`SpeechError.busy`),
/// and only once `.speech` is ready. Scripts play as soon as a session
/// starts unless `stepped` is set; then `advance()` plays one event at a time.
public final class FakeSpeechService: SpeechServicing {
    public enum Call: Sendable, Hashable {
        case install(SpeechPackageID), pauseInstall(SpeechPackageID), cancelInstall(SpeechPackageID)
        case remove(SpeechPackageID)
        case startDictation(DictationOptions), stopDictation, setLanguage(SpeechLanguage)
        case startMeeting(MeetingOptions), pause, resume, markMoment, stopMeeting
        case transcribeFile(String, FileTranscriptionOptions), cancel(FileJob.ID)
        case deleteRecordings(UUID), shutdown
    }

    /// What a dictation session says: each phrase as started, then finished
    /// (or dropped when `text` is nil).
    public struct DictationScript: Sendable {
        public var phrases: [(text: String?, languageCode: String?)]
        public init(phrases: [(text: String?, languageCode: String?)]) { self.phrases = phrases }
        public static func saying(_ texts: String...) -> DictationScript {
            DictationScript(phrases: texts.map { ($0, "en") })
        }
    }

    /// What a meeting produces while live, and its final result on stop.
    public struct MeetingScript: Sendable {
        public var callAudioAvailable: Bool
        public var live: [MeetingEvent]
        public var result: TranscriptResult
        public init(callAudioAvailable: Bool = true, live: [MeetingEvent] = [], result: TranscriptResult) {
            self.callAudioAvailable = callAudioAvailable
            self.live = live
            self.result = result
        }
    }

    private struct State {
        var status: SpeechStatus
        var calls: [Call] = []
        var observers: [UUID: AsyncStream<SpeechStatus>.Continuation] = [:]
        var dictation: DictationScript
        var meeting: MeetingScript?
        var fileResults: [String: Result<TranscriptResult, SpeechError>] = [:]
        var installError: SpeechError?
        var startError: SpeechError?
        var pending: [@Sendable () -> Void] = []
        var stoppers: [@Sendable () -> Void] = []
        var markTime: TimeInterval = 0
        var recordingsSize: Int64 = 0
    }

    private let state: Mutex<State>
    private let sizes: [SpeechPackageID: Int64]
    private let devices: [AudioInputDevice]
    private let apps: [RunningAudioApp]
    private let recordingsRoot: URL
    /// When true, scripted events wait for `advance()`.
    public let stepped: Bool

    public init(status: SpeechStatus = SpeechStatus(speech: .ready, speakerLabels: .ready),
                dictation: DictationScript = .saying("Hello from dictation."),
                meeting: MeetingScript? = nil,
                sizes: [SpeechPackageID: Int64] = [.speech: 653_632_103, .speakerLabels: 22_000_000],
                devices: [AudioInputDevice] = [AudioInputDevice(id: "builtin", name: "MacBook Pro Microphone",
                                                                isDefault: true)],
                apps: [RunningAudioApp] = [RunningAudioApp(bundleID: "us.zoom.xos", name: "zoom.us")],
                recordingsRoot: URL = FileManager.default.temporaryDirectory.appending(path: "mference-fake-recordings"),
                stepped: Bool = false) {
        state = Mutex(State(status: status, dictation: dictation, meeting: meeting))
        self.sizes = sizes
        self.devices = devices
        self.apps = apps
        self.recordingsRoot = recordingsRoot
        self.stepped = stepped
    }

    /// Not installed: the setup card's starting point.
    public static func notInstalled() -> FakeSpeechService {
        FakeSpeechService(status: SpeechStatus())
    }

    // MARK: Test controls

    public var calls: [Call] { state.withLock { $0.calls } }

    public func setStatus(_ status: SpeechStatus) {
        mutate { $0 = status }
    }

    /// The next `install` fails with this error (once).
    public func failNextInstall(with error: SpeechError) { state.withLock { $0.installError = error } }
    /// The next live start or file job fails with this error (once).
    public func failNextStart(with error: SpeechError) { state.withLock { $0.startError = error } }
    public func setDictationScript(_ script: DictationScript) { state.withLock { $0.dictation = script } }
    public func setMeetingScript(_ script: MeetingScript) { state.withLock { $0.meeting = script } }
    public func setFileResult(_ result: Result<TranscriptResult, SpeechError>, forFileNamed name: String) {
        state.withLock { $0.fileResults[name] = result }
    }
    public func setRecordingsSize(_ bytes: Int64) { state.withLock { $0.recordingsSize = bytes } }

    /// Plays the next scripted event (stepped mode). Returns false when none is left.
    @discardableResult
    public func advance() -> Bool {
        let next: (@Sendable () -> Void)? = state.withLock { s in
            s.pending.isEmpty ? nil : s.pending.removeFirst()
        }
        next?()
        return next != nil
    }

    // MARK: SpeechServicing

    public var status: SpeechStatus { state.withLock { $0.status } }

    public func statusUpdates() -> AsyncStream<SpeechStatus> {
        let (stream, continuation) = AsyncStream.makeStream(of: SpeechStatus.self, bufferingPolicy: .bufferingNewest(16))
        let id = UUID()
        state.withLock { s in
            s.observers[id] = continuation
            continuation.yield(s.status)
        }
        continuation.onTermination = { [weak self] _ in
            self?.state.withLock { _ = $0.observers.removeValue(forKey: id) }
        }
        return stream
    }

    public func downloadSize(of package: SpeechPackageID) -> Int64 { sizes[package] ?? 0 }

    public func install(_ package: SpeechPackageID) async throws {
        record(.install(package))
        if let error = state.withLock({ s -> SpeechError? in defer { s.installError = nil }; return s.installError }) {
            setPackage(package, .failed(error))
            throw error
        }
        for progress in [0.0, 0.5, 1.0] { setPackage(package, .downloading(progress: progress)) }
        setPackage(package, .preparing)
        setPackage(package, .ready)
    }

    public func pauseInstall(_ package: SpeechPackageID) async {
        record(.pauseInstall(package))
        if case .downloading(let progress) = status.state(of: package) { setPackage(package, .paused(progress: progress)) }
    }

    public func cancelInstall(_ package: SpeechPackageID) async {
        record(.cancelInstall(package))
        setPackage(package, .notInstalled)
    }

    public func remove(_ package: SpeechPackageID) async throws {
        record(.remove(package))
        setPackage(package, .notInstalled)
    }

    public func startDictation(_ options: DictationOptions) async throws -> DictationSession {
        record(.startDictation(options))
        try claimLiveSession(.dictation)
        let (events, continuation) = AsyncStream.makeStream(of: DictationEvent.self)
        let script = state.withLock { $0.dictation }
        var steps: [@Sendable () -> Void] = [{ continuation.yield(.level(0.4)) }]
        for phrase in script.phrases {
            let id = UUID()
            steps.append { continuation.yield(.phraseStarted(id)) }
            if let text = phrase.text {
                steps.append { continuation.yield(.phraseFinished(id, text: text, languageCode: phrase.languageCode)) }
            } else {
                steps.append { continuation.yield(.phraseDropped(id)) }
            }
        }
        let end: @Sendable () -> Void = { [weak self] in
            continuation.yield(.ended)
            continuation.finish()
            self?.releaseLiveSession()
        }
        schedule(steps)
        return DictationSession(events: events,
                                stop: { [weak self] in
                                    self?.record(.stopDictation)
                                    self?.flushPending()
                                    end()
                                },
                                setLanguage: { [weak self] language in self?.record(.setLanguage(language)) })
    }

    public func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording {
        record(.startMeeting(options))
        try claimLiveSession(.meeting)
        let script = state.withLock { $0.meeting } ?? MeetingScript(result: TranscriptResult(
            segments: [], speakers: [.you], duration: 0))
        let (events, continuation) = AsyncStream.makeStream(of: MeetingEvent.self)
        let callAudio = options.callAudio != .off && script.callAudioAvailable
        var steps: [@Sendable () -> Void] = [{ continuation.yield(.started(callAudio: callAudio)) }]
        if options.callAudio != .off && !script.callAudioAvailable {
            steps.append { continuation.yield(.callAudioUnavailable(.permissionDenied(.systemAudio))) }
        }
        steps += script.live.map { event in { continuation.yield(event) } }
        schedule(steps)
        let controls = MeetingRecording.Controls(
            pause: { [weak self] in self?.record(.pause); continuation.yield(.paused) },
            resume: { [weak self] in self?.record(.resume); continuation.yield(.resumed) },
            markMoment: { [weak self] in
                self?.record(.markMoment)
                return self?.state.withLock { $0.markTime } ?? 0
            },
            stop: { [weak self] in
                self?.record(.stopMeeting)
                self?.flushPending()
                continuation.yield(.finishing(.transcribingLastPhrases))
                continuation.yield(.finishing(.labellingSpeakers(progress: 1)))
                continuation.yield(.ended(script.result))
                continuation.finish()
                self?.releaseLiveSession()
            })
        return MeetingRecording(noteID: options.noteID, startedAt: .now, events: events, controls: controls)
    }

    /// The recording time `markMoment()` reports (tests set the meeting clock).
    public func setMeetingTime(_ time: TimeInterval) { state.withLock { $0.markTime = time } }

    public func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        let name = url.lastPathComponent
        record(.transcribeFile(name, options))
        try requireReady()
        if let error = takeStartError() { throw error }
        let (events, continuation) = AsyncStream.makeStream(of: FileJobEvent.self)
        let job = FileJob(noteID: options.noteID, fileName: name, events: events)
        let result = state.withLock { $0.fileResults[name] }
            ?? .success(TranscriptResult(segments: [], speakers: [.others], duration: 0))
        mutate { $0.fileQueue.append(FileJobStatus(id: job.id, noteID: options.noteID, fileName: name)) }
        let id = job.id
        schedule([
            { continuation.yield(.queued(position: 0)) },
            { continuation.yield(.started(duration: 60)) },
            { [weak self] in
                continuation.yield(.progress(fraction: 0.5, remaining: 2, languageCode: "en"))
                self?.updateJob(id) { $0.phase = .running; $0.fraction = 0.5; $0.remaining = 2 }
            },
            { [weak self] in
                switch result {
                case .success(let transcript): continuation.yield(.finished(transcript))
                case .failure(let error): continuation.yield(.failed(error))
                }
                continuation.finish()
                self?.updateJob(id) { job in
                    if case .success = result { job.phase = .done; job.fraction = 1 } else { job.phase = .failed }
                }
            },
        ])
        state.withLock { $0.stoppers.append { continuation.yield(.cancelled); continuation.finish() } }
        return job
    }

    public func cancel(_ job: FileJob.ID) async {
        record(.cancel(job))
        mutate { $0.fileQueue.removeAll { $0.id == job } }
    }

    public func inputDevices() async -> [AudioInputDevice] { devices }
    public func runningAudioApps() async -> [RunningAudioApp] { apps }
    public func recordingsSize() async -> Int64 { state.withLock { $0.recordingsSize } }

    public func deleteRecordings(for noteID: UUID) async throws { record(.deleteRecordings(noteID)) }

    public func recordingsDirectory(for noteID: UUID) -> URL {
        recordingsRoot.appending(path: noteID.uuidString, directoryHint: .isDirectory)
    }

    public func shutdown() async {
        record(.shutdown)
        let stoppers = state.withLock { s -> [@Sendable () -> Void] in
            defer { s.stoppers = []; s.pending = [] }
            return s.stoppers
        }
        stoppers.forEach { $0() }
        releaseLiveSession()
    }

    // MARK: Internals

    private func record(_ call: Call) { state.withLock { $0.calls.append(call) } }

    private func mutate(_ change: (inout SpeechStatus) -> Void) {
        let (status, observers) = state.withLock { s in
            change(&s.status)
            return (s.status, Array(s.observers.values))
        }
        observers.forEach { $0.yield(status) }
    }

    private func setPackage(_ package: SpeechPackageID, _ value: SpeechModelState) {
        mutate { status in
            switch package {
            case .speech: status.speech = value
            case .speakerLabels: status.speakerLabels = value
            }
        }
    }

    private func updateJob(_ id: FileJob.ID, _ change: @escaping (inout FileJobStatus) -> Void) {
        mutate { status in
            if let index = status.fileQueue.firstIndex(where: { $0.id == id }) { change(&status.fileQueue[index]) }
        }
    }

    private func requireReady() throws {
        guard status.speech.isReady else { throw SpeechError.notInstalled(.speech) }
    }

    private func takeStartError() -> SpeechError? {
        state.withLock { s in defer { s.startError = nil }; return s.startError }
    }

    private func claimLiveSession(_ kind: LiveSessionKind) throws {
        try requireReady()
        if let error = takeStartError() { throw error }
        if let running = status.liveSession { throw SpeechError.busy(running) }
        mutate { $0.liveSession = kind }
    }

    private func releaseLiveSession() { mutate { $0.liveSession = nil } }

    private func schedule(_ steps: [@Sendable () -> Void]) {
        if stepped {
            state.withLock { $0.pending += steps }
        } else {
            steps.forEach { $0() }
        }
    }

    private func flushPending() {
        while advance() {}
    }
}
