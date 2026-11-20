import Foundation
import Synchronization

/// A `SpeechServicing` for tests and previews: no model, no microphone, no
/// call tap, no network, no TCC. Package states, dictation phrases, meeting
/// segments and file results are scripted; every call is recorded.
///
/// Sessions follow the real rules: `.speech` must be ready, one live session
/// at a time (`SpeechError.busy`), `stop` is idempotent and releases only its
/// own session, `cancel` ends that job's stream with `.cancelled`, and
/// `shutdown` ends everything. Scripts play as soon as a session starts
/// unless `stepped` is set; then `advance()` plays one event at a time, and
/// stopping a session plays the rest of that session's events only.
public final class FakeSpeechService: SpeechServicing {
    public enum Call: Sendable, Hashable {
        case install(SpeechPackageID), pauseInstall(SpeechPackageID), cancelInstall(SpeechPackageID)
        case remove(SpeechPackageID)
        case startDictation(DictationOptions), stopDictation, setLanguage(SpeechLanguage)
        case startMeeting(MeetingOptions), pause, resume, markMoment, stopMeeting
        case transcribeFile(String, FileTranscriptionOptions), cancel(FileJob.ID)
        case deleteRecordings(UUID), recoverInterruptedRecordings, shutdown
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
        /// Sends `.noCallAudioHeard` right after `.started` when the meeting
        /// records call audio (the tap heard only silence).
        public var noCallAudioHeard: Bool
        public var live: [MeetingEvent]
        public var result: TranscriptResult
        public init(callAudioAvailable: Bool = true, noCallAudioHeard: Bool = false, live: [MeetingEvent] = [],
                    result: TranscriptResult) {
            self.callAudioAvailable = callAudioAvailable
            self.noCallAudioHeard = noCallAudioHeard
            self.live = live
            self.result = result
        }
    }

    private typealias Step = @Sendable () -> Void

    private struct State {
        var status: SpeechStatus
        var calls: [Call] = []
        var observers: [UUID: AsyncStream<SpeechStatus>.Continuation] = [:]
        var dictation: DictationScript
        var meeting: MeetingScript?
        var fileResults: [String: Result<TranscriptResult, SpeechError>] = [:]
        var interrupted: [InterruptedRecording] = []
        var installError: SpeechError?
        var startError: SpeechError?
        var hasHeardCallAudio = false
        /// Scripted events not yet played, with the session or job they belong to.
        var pending: [(owner: UUID, step: Step)] = []
        /// How to end each open session or job (for stop, cancel, shutdown).
        var enders: [UUID: Step] = [:]
        /// The session holding the microphone.
        var liveOwner: UUID?
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
                sizes: [SpeechPackageID: Int64] = [.speech: SpeechModelPins.speech.totalBytes,
                                                    .speakerLabels: SpeechModelPins.speakerLabels.totalBytes],
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

    public func setStatus(_ status: SpeechStatus) { mutate { $0 = status } }

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
    /// What `hasHeardCallAudio` reads (false until set).
    public func setHasHeardCallAudio(_ heard: Bool) { state.withLock { $0.hasHeardCallAudio = heard } }
    /// What `recoverInterruptedRecordings()` reports (once).
    public func setInterruptedRecordings(_ recordings: [InterruptedRecording]) {
        state.withLock { $0.interrupted = recordings }
    }
    /// The recording time `markMoment()` reports (tests set the meeting clock).
    public func setMeetingTime(_ time: TimeInterval) { state.withLock { $0.markTime = time } }

    /// Plays the next scripted event (stepped mode). Returns false when none is left.
    @discardableResult
    public func advance() -> Bool {
        let next: Step? = state.withLock { s in s.pending.isEmpty ? nil : s.pending.removeFirst().step }
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
        let owner = UUID()
        try claimLiveSession(LiveSessionStatus(kind: .dictation, noteID: nil, startedAt: .now), owner: owner)
        let (events, continuation) = AsyncStream.makeStream(of: DictationEvent.self)
        let script = state.withLock { $0.dictation }
        var steps: [Step] = [{ continuation.yield(.level(0.4)) }]
        for phrase in script.phrases {
            let id = UUID()
            steps.append { continuation.yield(.phraseStarted(id)) }
            if let text = phrase.text {
                steps.append { continuation.yield(.phraseFinished(id, text: text, languageCode: phrase.languageCode)) }
            } else {
                steps.append { continuation.yield(.phraseDropped(id)) }
            }
        }
        register(owner) {
            continuation.yield(.ended)
            continuation.finish()
        }
        schedule(steps, owner: owner)
        return DictationSession(events: events,
                                stop: { [weak self] in
                                    self?.record(.stopDictation)
                                    self?.end(owner)
                                },
                                setLanguage: { [weak self] language in self?.record(.setLanguage(language)) })
    }

    public func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording {
        record(.startMeeting(options))
        let owner = UUID()
        let startedAt = Date.now
        try claimLiveSession(LiveSessionStatus(kind: .meeting, noteID: options.noteID, startedAt: startedAt), owner: owner)
        let script = state.withLock { $0.meeting } ?? MeetingScript(result: TranscriptResult(
            segments: [], speakers: [.you], duration: 0))
        let (events, continuation) = AsyncStream.makeStream(of: MeetingEvent.self)
        let callAudio = options.callAudio != .off && script.callAudioAvailable
        var steps: [Step] = [{ continuation.yield(.started(callAudio: callAudio)) }]
        if options.callAudio != .off && !script.callAudioAvailable {
            steps.append { continuation.yield(.callAudioUnavailable(.permissionDenied(.systemAudio))) }
        }
        if callAudio && script.noCallAudioHeard {
            steps.append { continuation.yield(.noCallAudioHeard) }
        }
        steps += script.live.map { event in { continuation.yield(event) } }
        register(owner) {
            continuation.yield(.finishing(.transcribingLastPhrases))
            continuation.yield(.finishing(.labellingSpeakers(progress: 1)))
            continuation.yield(.ended(script.result))
            continuation.finish()
        }
        schedule(steps, owner: owner)
        let controls = MeetingRecording.Controls(
            pause: { [weak self] in
                self?.record(.pause)
                self?.setPaused(true, owner: owner)
                continuation.yield(.paused)
            },
            resume: { [weak self] in
                self?.record(.resume)
                self?.setPaused(false, owner: owner)
                continuation.yield(.resumed)
            },
            markMoment: { [weak self] in
                self?.record(.markMoment)
                return self?.state.withLock { $0.markTime } ?? 0
            },
            stop: { [weak self] in
                self?.record(.stopMeeting)
                self?.end(owner)
            })
        return MeetingRecording(noteID: options.noteID, startedAt: startedAt, events: events, controls: controls)
    }

    public func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        let name = url.lastPathComponent
        record(.transcribeFile(name, options))
        guard status.speech.isReady else { throw SpeechError.notInstalled(.speech) }
        if let error = takeStartError() { throw error }
        let (events, continuation) = AsyncStream.makeStream(of: FileJobEvent.self)
        let job = FileJob(noteID: options.noteID, fileName: name, events: events)
        let id = job.id
        let result = state.withLock { $0.fileResults[name] }
            ?? .success(TranscriptResult(segments: [], speakers: [.others], duration: 0))
        mutate { $0.fileQueue.append(FileJobStatus(id: id, noteID: options.noteID, fileName: name)) }
        var steps: [Step] = [{ continuation.yield(.queued(position: 0)) }]
        if status.liveSession != nil {
            // Files wait while dictation or a meeting runs.
            steps.append { [weak self] in
                continuation.yield(.paused)
                self?.updateJob(id) { $0.phase = .paused }
            }
        }
        steps += [
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
                self?.state.withLock { _ = $0.enders.removeValue(forKey: id) }
            },
        ]
        register(id) {
            continuation.yield(.cancelled)
            continuation.finish()
        }
        schedule(steps, owner: id)
        return job
    }

    public func cancel(_ job: FileJob.ID) async {
        record(.cancel(job))
        let ender = state.withLock { s -> Step? in
            s.pending.removeAll { $0.owner == job }
            return s.enders.removeValue(forKey: job)
        }
        ender?()
        mutate { $0.fileQueue.removeAll { $0.id == job } }
    }

    public func inputDevices() async -> [AudioInputDevice] { devices }
    public func runningAudioApps() async -> [RunningAudioApp] { apps }
    public var hasHeardCallAudio: Bool { state.withLock { $0.hasHeardCallAudio } }
    public func recordingsSize() async -> Int64 { state.withLock { $0.recordingsSize } }

    public func deleteRecordings(for noteID: UUID) async throws { record(.deleteRecordings(noteID)) }

    public func recordingsDirectory(for noteID: UUID) -> URL {
        recordingsRoot.appending(path: noteID.uuidString, directoryHint: .isDirectory)
    }

    public func recoverInterruptedRecordings() async -> [InterruptedRecording] {
        record(.recoverInterruptedRecordings)
        return state.withLock { s in defer { s.interrupted = [] }; return s.interrupted }
    }

    public func shutdown() async {
        record(.shutdown)
        // File jobs are cancelled (nothing more plays); a live session
        // finishes its last phrases, then ends.
        let owners = state.withLock { s -> [UUID] in
            s.pending.removeAll { $0.owner != s.liveOwner }
            return Array(s.enders.keys)
        }
        for owner in owners { end(owner) }
    }

    // MARK: Internals

    private func record(_ call: Call) { state.withLock { $0.calls.append(call) } }

    /// Changes the status and tells observers in order (inside the lock, so
    /// two changes can never reach an observer swapped).
    private func mutate(_ change: (inout SpeechStatus) -> Void) {
        state.withLock { s in
            change(&s.status)
            for observer in s.observers.values { observer.yield(s.status) }
        }
    }

    private func setPackage(_ package: SpeechPackageID, _ value: SpeechModelState) {
        mutate { status in
            switch package {
            case .speech: status.speech = value
            case .speakerLabels: status.speakerLabels = value
            }
        }
    }

    private func updateJob(_ id: FileJob.ID, _ change: (inout FileJobStatus) -> Void) {
        mutate { status in
            if let index = status.fileQueue.firstIndex(where: { $0.id == id }) { change(&status.fileQueue[index]) }
        }
    }

    private func takeStartError() -> SpeechError? {
        state.withLock { s in defer { s.startError = nil }; return s.startError }
    }

    /// Checks and claims the microphone in one step.
    private func claimLiveSession(_ live: LiveSessionStatus, owner: UUID) throws {
        try state.withLock { s in
            guard s.status.speech.isReady else { throw SpeechError.notInstalled(.speech) }
            if let error = s.startError {
                s.startError = nil
                throw error
            }
            if let running = s.status.liveSession { throw SpeechError.busy(running.kind) }
            s.liveOwner = owner
            s.status.liveSession = live
            for observer in s.observers.values { observer.yield(s.status) }
        }
    }

    private func setPaused(_ paused: Bool, owner: UUID) {
        mutate { status in
            guard var live = status.liveSession, live.isPaused != paused else { return }
            let now = Date.now
            live.recordedTime = live.recordedTime(at: now)
            live.updatedAt = now
            live.isPaused = paused
            status.liveSession = live
        }
        _ = owner
    }

    private func register(_ owner: UUID, ender: @escaping Step) {
        state.withLock { $0.enders[owner] = ender }
    }

    /// Plays the rest of `owner`'s events, then ends it, once; releases the
    /// microphone only if `owner` holds it.
    private func end(_ owner: UUID) {
        let (rest, ender) = state.withLock { s -> ([Step], Step?) in
            let rest = s.pending.filter { $0.owner == owner }.map { $0.step }
            s.pending.removeAll { $0.owner == owner }
            return (rest, s.enders.removeValue(forKey: owner))
        }
        guard let ender else { return }
        rest.forEach { $0() }
        ender()
        state.withLock { s in
            guard s.liveOwner == owner else { return }
            s.liveOwner = nil
            s.status.liveSession = nil
            for observer in s.observers.values { observer.yield(s.status) }
        }
    }

    private func schedule(_ steps: [Step], owner: UUID) {
        if stepped {
            state.withLock { s in s.pending += steps.map { (owner, $0) } }
        } else {
            steps.forEach { $0() }
        }
    }
}
