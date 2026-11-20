import Foundation
@testable import MferenceAppCore
import Synchronization
import Testing

/// A broker whose microphone request waits until the test answers (the
/// person is looking at the system prompt).
final class GatedBroker: PermissionBroker {
    private let waiter = Mutex<CheckedContinuation<PermissionState, Never>?>(nil)
    var isAsking: Bool { waiter.withLock { $0 != nil } }

    func answer(_ state: PermissionState) {
        let continuation = waiter.withLock { w -> CheckedContinuation<PermissionState, Never>? in defer { w = nil }; return w }
        continuation?.resume(returning: state)
    }

    func state(of permission: AppPermission) async -> PermissionState { .notDetermined }
    func request(_ permission: AppPermission) async -> PermissionState {
        await withCheckedContinuation { continuation in waiter.withLock { $0 = continuation } }
    }
    func openSystemSettings(for permission: AppPermission) async {}
}

/// A speech service whose dictation stream the test drives event by event
/// (to send `.failed`); everything else is the fake's.
final class DrivenDictationSpeech: SpeechServicing {
    let base = FakeSpeechService()
    private let continuation = Mutex<AsyncStream<DictationEvent>.Continuation?>(nil)
    private let stops = Mutex(0)
    var stopCount: Int { stops.withLock { $0 } }

    func send(_ event: DictationEvent) { _ = continuation.withLock { $0?.yield(event) } }
    func end() {
        continuation.withLock { c in
            c?.yield(.ended)
            c?.finish()
            c = nil
        }
    }

    var status: SpeechStatus { base.status }
    func statusUpdates() -> AsyncStream<SpeechStatus> { base.statusUpdates() }
    func downloadSize(of package: SpeechPackageID) -> Int64 { 0 }
    func install(_ package: SpeechPackageID) async throws {}
    func pauseInstall(_ package: SpeechPackageID) async {}
    func cancelInstall(_ package: SpeechPackageID) async {}
    func remove(_ package: SpeechPackageID) async throws {}
    func startDictation(_ options: DictationOptions) async throws -> DictationSession {
        let (events, continuation) = AsyncStream.makeStream(of: DictationEvent.self)
        self.continuation.withLock { $0 = continuation }
        return DictationSession(events: events, stop: { [weak self] in
            self?.stops.withLock { $0 += 1 }
            self?.end()
        }, setLanguage: { _ in })
    }
    func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording { try await base.startMeeting(options) }
    func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        try await base.transcribeFile(url, options: options)
    }
    func cancel(_ job: FileJob.ID) async {}
    func inputDevices() async -> [AudioInputDevice] { [] }
    func runningAudioApps() async -> [RunningAudioApp] { [] }
    var hasHeardCallAudio: Bool { false }
    func recordingsSize() async -> Int64 { 0 }
    func deleteRecordings(for noteID: UUID) async throws {}
    func recordingsDirectory(for noteID: UUID) -> URL { base.recordingsDirectory(for: noteID) }
    func recoverInterruptedRecordings() async -> [InterruptedRecording] { [] }
    func shutdown() async {}
}

/// What a note received, kept outside the target so a test can drop the target.
@MainActor
final class Inbox {
    var phrases: [String] = []
    var dropAllCount = 0
}

@MainActor
final class InboxTarget: DictationInsertionTarget {
    var noteID = UUID()
    var begun: [UUID] = []
    /// What happened, in order ("start", "begin").
    var log: [String] = []
    func dictationWillStart() { log.append("start") }
    let inbox: Inbox
    init(_ inbox: Inbox) { self.inbox = inbox }
    func beginPhrase(_ id: UUID) {
        begun.append(id)
        log.append("begin")
    }
    func finishPhrase(_ id: UUID, text: String) { inbox.phrases.append(text) }
    func dropPhrase(_ id: UUID) {}
    func dropAllPhrases() { inbox.dropAllCount += 1 }
}

@MainActor
@Suite struct DictationEdgeTests {
    private func preferences() -> SpeechPreferences { SpeechPreferences(store: InMemorySettingsStore()) }

    @Test func stopWhileTheMicrophonePromptIsUpStartsNothing() async throws {
        let speech = FakeSpeechService()
        let broker = GatedBroker()
        let dictation = DictationController(speech: speech, permissions: broker, preferences: preferences())
        let start = Task { await dictation.start(in: InboxTarget(Inbox())) }
        await waitUntil("the microphone prompt") { broker.isAsking }
        #expect(dictation.phase == .starting)
        await dictation.stop() // Esc while the prompt shows
        #expect(dictation.phase == .idle)
        broker.answer(.granted)
        await start.value
        #expect(!dictation.isActive, "the overtaken start does nothing more")
        #expect(!speech.calls.contains { if case .startDictation = $0 { true } else { false } })
        #expect(speech.status.liveSession == nil, "the microphone stays free")
    }

    @Test func aFailureMidSessionEndsItAndSaysWhy() async throws {
        let speech = DrivenDictationSpeech()
        let inbox = Inbox()
        let dictation = DictationController(speech: speech, permissions: FakePermissionBroker(), preferences: preferences())
        await dictation.start(in: InboxTarget(inbox))
        #expect(dictation.phase == .listening)
        speech.send(.phraseStarted(UUID()))
        speech.send(.failed(.failed("audio engine stopped")))
        speech.end()
        await waitUntil("dictation to end") { !dictation.isActive }
        #expect(!dictation.isActive && dictation.error == .failed("audio engine stopped"))
        #expect(inbox.dropAllCount >= 1, "the pending marker goes")
        #expect(dictation.pendingPhrases.isEmpty)
    }

    @Test func aFailureKeepsTheOtherPhrasesMarkers() async throws {
        // Codex review on #251: one phrase failing must not send the others
        // to the caret.
        let speech = DrivenDictationSpeech()
        let inbox = Inbox()
        let dictation = DictationController(speech: speech, permissions: FakePermissionBroker(), preferences: preferences())
        await dictation.start(in: InboxTarget(inbox))
        let first = UUID(), second = UUID()
        speech.send(.phraseStarted(first))
        speech.send(.phraseStarted(second))
        speech.send(.failed(.failed("recognition failed")))
        speech.send(.phraseDropped(first))
        speech.send(.phraseFinished(second, text: "Still here.", languageCode: "en"))
        await waitUntil("the second phrase") { inbox.phrases == ["Still here."] }
        #expect(inbox.dropAllCount == 0, "the other markers stay until the end")
        speech.end()
        await waitUntil("dictation to end") { !dictation.isActive }
        #expect(dictation.error == .failed("recognition failed"))
        #expect(inbox.dropAllCount >= 1)
    }

    @Test func theLastPhraseLandsEvenWhenTheEditorIsGone() async throws {
        let speech = FakeSpeechService(dictation: .saying("last words"), stepped: true)
        let inbox = Inbox()
        let dictation = DictationController(speech: speech, permissions: FakePermissionBroker(), preferences: preferences())
        do {
            let target = InboxTarget(inbox)
            await dictation.start(in: target)
        } // The view's own reference is gone (leaving Notes).
        await step(speech, dictation, 2) // level, phrase started
        await dictation.stop()
        #expect(inbox.phrases == ["last words"], "the controller holds the note until stop returns")
    }

    @Test func theEditorBindsTheDictationBeforeTheFirstPhrase() async {
        let speech = FakeSpeechService(dictation: .saying("hello"), stepped: true)
        let target = InboxTarget(Inbox())
        let dictation = DictationController(speech: speech, permissions: FakePermissionBroker(), preferences: preferences())
        await dictation.start(in: target)
        #expect(target.log == ["start"], "dictationWillStart runs as dictation starts")
        await step(speech, dictation, 2) // level, phrase started
        #expect(target.log == ["start", "begin"])
        await dictation.stop()
    }

    @Test func aPhraseNeverBeginsInAnotherNote() async {
        let speech = FakeSpeechService(dictation: .saying("for note A", "would land in B"), stepped: true)
        let inbox = Inbox()
        let target = InboxTarget(inbox)
        let dictation = DictationController(speech: speech, permissions: FakePermissionBroker(), preferences: preferences())
        await dictation.start(in: target)
        await step(speech, dictation, 3) // level, phrase 1 started and finished
        #expect(inbox.phrases == ["for note A"])
        target.noteID = UUID() // the editor now names another note (the binding broke)
        let before = dictation.handledEventCount
        speech.advance() // phrase 2 starts
        await waitUntil("the stray phrase handled") { dictation.handledEventCount > before }
        #expect(target.begun.count == 1, "the second phrase is not begun in the other note")
        await waitUntil("dictation to stop") { !dictation.isActive }
        #expect(inbox.phrases == ["for note A"], "and its text lands nowhere")
        #expect(inbox.dropAllCount >= 1, "the editor's binding ends")
    }

    @Test func theFillersChipSaysItAppliesNextTime() async {
        let speech = FakeSpeechService(stepped: true)
        let prefs = preferences()
        let dictation = DictationController(speech: speech, permissions: FakePermissionBroker(), preferences: prefs)
        await dictation.start(in: InboxTarget(Inbox()))
        #expect(dictation.sessionCleansUpFillers && !dictation.cleanUpChangesNextTime)
        dictation.setCleanUpFillers(false)
        #expect(dictation.cleanUpChangesNextTime)
        await dictation.stop()
        #expect(!dictation.cleanUpChangesNextTime)
        await dictation.start(in: InboxTarget(Inbox()))
        #expect(!dictation.sessionCleansUpFillers)
        #expect(speech.calls.contains(.startDictation(DictationOptions(cleanUpFillers: false))))
        await dictation.stop()
    }
}
