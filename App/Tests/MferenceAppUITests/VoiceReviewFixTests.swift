import Foundation
import Observation
import MferenceAppCore
@testable import MferenceAppUI
import Synchronization
import Testing

/// A speech service whose install blocks like a real download, following
/// the `SpeechServicing` contract: it reports progress, then waits until the
/// test completes it; pause or cancel ends it with `CancellationError`; a
/// second call while one runs joins it (same outcome). A later install
/// picks up where it paused. Everything else is the fake's.
final class GatedInstallSpeech: SpeechServicing {
    private struct Gate {
        var waiters: [CheckedContinuation<Void, any Error>] = []
        var installs = 0
        var removeError: SpeechError?
    }

    let base: FakeSpeechService
    private let gate = Mutex(Gate())

    init(_ base: FakeSpeechService = .notInstalled()) { self.base = base }

    var installCount: Int { gate.withLock { $0.installs } }
    var isWaiting: Bool { gate.withLock { !$0.waiters.isEmpty } }
    func failNextRemove(with error: SpeechError) { gate.withLock { $0.removeError = error } }

    /// The download finishes: verified, prepared, ready.
    func complete() {
        let waiters = gate.withLock { g -> [CheckedContinuation<Void, any Error>] in defer { g.waiters = [] }; return g.waiters }
        base.setStatus(SpeechStatus(speech: .ready, speakerLabels: base.status.speakerLabels))
        waiters.forEach { $0.resume() }
    }

    var status: SpeechStatus { base.status }
    func statusUpdates() -> AsyncStream<SpeechStatus> { base.statusUpdates() }
    func downloadSize(of package: SpeechPackageID) -> Int64 { base.downloadSize(of: package) }

    func install(_ package: SpeechPackageID) async throws {
        let joins = gate.withLock { g -> Bool in
            if g.waiters.isEmpty { g.installs += 1 }
            return !g.waiters.isEmpty
        }
        if !joins {
            let resumeAt: Double = if case .paused(let progress) = base.status.speech { progress } else { 0.1 }
            base.setStatus(SpeechStatus(speech: .downloading(progress: resumeAt), speakerLabels: base.status.speakerLabels))
        }
        try await withCheckedThrowingContinuation { continuation in
            gate.withLock { $0.waiters.append(continuation) }
        }
    }

    func pauseInstall(_ package: SpeechPackageID) async {
        guard case .downloading(let progress) = base.status.speech else { return }
        base.setStatus(SpeechStatus(speech: .paused(progress: max(progress, 0.4)), speakerLabels: base.status.speakerLabels))
        endWaiter()
    }

    func cancelInstall(_ package: SpeechPackageID) async {
        base.setStatus(SpeechStatus(speech: .notInstalled, speakerLabels: base.status.speakerLabels))
        endWaiter()
    }

    private func endWaiter() {
        let waiters = gate.withLock { g -> [CheckedContinuation<Void, any Error>] in defer { g.waiters = [] }; return g.waiters }
        waiters.forEach { $0.resume(throwing: CancellationError()) }
    }

    func remove(_ package: SpeechPackageID) async throws {
        if let error = gate.withLock({ g -> SpeechError? in defer { g.removeError = nil }; return g.removeError }) {
            throw error
        }
        try await base.remove(package)
    }

    func startDictation(_ options: DictationOptions) async throws -> DictationSession {
        try await base.startDictation(options)
    }
    func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording { try await base.startMeeting(options) }
    func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        try await base.transcribeFile(url, options: options)
    }
    func cancel(_ job: FileJob.ID) async { await base.cancel(job) }
    func inputDevices() async -> [AudioInputDevice] { await base.inputDevices() }
    func runningAudioApps() async -> [RunningAudioApp] { await base.runningAudioApps() }
    func recordingsSize() async -> Int64 { await base.recordingsSize() }
    func deleteRecordings(for noteID: UUID) async throws { try await base.deleteRecordings(for: noteID) }
    func recordingsDirectory(for noteID: UUID) -> URL { base.recordingsDirectory(for: noteID) }
    func recoverInterruptedRecordings() async -> [InterruptedRecording] { await base.recoverInterruptedRecordings() }
    func shutdown() async { await base.shutdown() }
}

/// The G-A5 review of #214: errors are shown, pause and resume work on a
/// real blocking install, voice commands are scoped.
@MainActor
@Suite(.serialized) struct VoiceReviewFixTests {
    private func voice(_ speech: any SpeechServicing, offline: Bool = false) -> VoiceModels {
        VoiceModels(speech: speech, permissions: FakePermissionBroker(),
                    preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { offline })
    }

    private func eventually(_ condition: @MainActor () -> Bool) async throws {
        for _ in 0..<400 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        Issue.record("condition never became true")
    }

    @Test func pauseThenResumeInstallsAgainAndContinues() async throws {
        let speech = GatedInstallSpeech()
        let models = voice(speech)
        models.setup.start()
        defer { models.setup.stop() }
        var ran = false
        models.setup.require(.dictate) { ran = true }
        models.setup.download()
        try await eventually { speech.isWaiting && models.setup.display == .downloading(fraction: 0.1) }
        models.setup.pause()
        try await eventually { models.setup.display == .paused(fraction: 0.4) }
        await models.setup.waitForInstall()
        #expect(models.setup.installError == nil, "a pause is not an error")
        models.setup.download() // Resume
        try await eventually { speech.isWaiting && models.setup.display == .downloading(fraction: 0.4) }
        #expect(speech.installCount == 2, "Resume installs again from where it paused")
        speech.complete()
        await models.setup.waitForInstall()
        #expect(ran && models.setup.display == .ready, "the waiting action continues")
    }

    @Test func resumeWhileThePausedDownloadIsStillWindingDown() async throws {
        let speech = GatedInstallSpeech()
        let models = voice(speech)
        models.setup.start()
        defer { models.setup.stop() }
        models.setup.download()
        try await eventually { speech.isWaiting }
        await speech.pauseInstall(.speech)
        models.setup.download() // Resume before the first task saw its CancellationError
        #expect(speech.installCount == 1, "no second call joins the ending install")
        try await eventually { speech.installCount == 2 && speech.isWaiting }
        speech.complete()
        try await eventually { models.setup.display == .ready }
    }

    @Test func cancelReturnsToDownloadWithoutAnError() async throws {
        let speech = GatedInstallSpeech()
        let models = voice(speech)
        models.setup.start()
        defer { models.setup.stop() }
        models.setup.download()
        try await eventually { speech.isWaiting }
        models.setup.cancel()
        try await eventually { models.setup.display == .download }
        await models.setup.waitForInstall()
        #expect(models.setup.display == .download)
    }

    @Test func aThrowWithoutAStatusChangeIsShown() async {
        // No speech service built: install throws and the status stays Not installed.
        let models = voice(UnavailableSpeechService())
        #expect(models.setup.downloadBytes == 0)
        models.setup.download()
        await models.setup.waitForInstall()
        #expect(models.setup.display == .failed("Speech stopped unexpectedly. Try again."))
        let lowDisk = FakeSpeechService.notInstalled()
        lowDisk.failNextInstall(with: .lowDiskSpace(availableBytes: 10))
        let other = voice(lowDisk)
        other.setup.download()
        await other.setup.waitForInstall()
        #expect(other.setup.display == .failed(SpeechError.lowDiskSpace(availableBytes: 10).message))
        other.setup.download() // Try again clears it.
        await other.setup.waitForInstall()
        #expect(other.setup.display == .ready && other.setup.installError == nil)
    }

    @Test func removeErrorAndSpeakerLabelFailuresAreShown() async throws {
        let speech = GatedInstallSpeech(FakeSpeechService())
        speech.failNextRemove(with: .failed("disk busy"))
        let models = voice(speech)
        await models.settings.removeSpeech()
        #expect(models.settings.speechFailure == "Speech stopped unexpectedly. Try again.")
        #expect(SpeechSettingsView.speechSubtitle(models.settings).contains("Speech stopped unexpectedly"))

        let labels = FakeSpeechService(status: SpeechStatus(speech: .ready, speakerLabels: .notInstalled))
        labels.failNextInstall(with: .verificationFailed(file: "Segmentation.mlmodelc"))
        let other = voice(labels)
        other.setup.start()
        defer { other.setup.stop() }
        other.settings.downloadSpeakerLabels()
        await other.settings.waitForSpeakerLabels()
        try await eventually { other.settings.speakerLabelsFailure != nil }
        #expect(other.settings.speakerLabelsFailure == "A downloaded file didn't match. Download again.",
                "a failed speaker-labels download says why, not just Not installed")
        #expect(SpeechSettingsView.speakerLabelsSubtitle(other.settings).hasSuffix("Download again."))
    }

    @Test func offlineIsSaidOnTheRowNotOnlyInHelp() {
        let models = voice(FakeSpeechService.notInstalled(), offline: true)
        #expect(SpeechSettingsView.speechSubtitle(models.settings).hasSuffix(SpeechError.offline.message))
    }

    @Test func modelsRowBadgeFollowsTheDownload() {
        #expect(SettingsView.speechBadge(.ready) == .installed)
        #expect(SettingsView.speechBadge(.downloading(progress: 0.42)) == .custom("Downloading 42%", .accent))
        #expect(SettingsView.speechBadge(.preparing) == .custom("Preparing…", .neutral))
        #expect(SettingsView.speechBadge(.failed(.offline)) == .custom("Needs attention", .preview))
    }

    @Test func voiceCommandsAreScoped() async throws {
        let speech = FakeSpeechService()
        let models = voice(speech)
        models.setup.start()
        defer { models.setup.stop() }
        #expect(!models.canToggleDictation, "⇧⌘D needs a focused note")
        #expect(!models.canMarkMoment, "⇧⌘M only while a meeting records")
        let controls = FakeRecordingControls()
        models.recordingControls = controls
        models.markMoment()
        _ = try await speech.startMeeting(MeetingOptions(noteID: UUID()))
        try await eventually { models.canMarkMoment }
        models.markMoment()
        try await eventually { controls.calls == ["mark"] }
        let note = AppendingTarget()
        models.focusedDictationTarget = note
        #expect(models.canToggleDictation)
        #expect(models.canRecordMeeting == false, "a meeting records: ⇧⌘R waits")
    }

    @Test func focusingAnEditorReEnablesTheDictateMenuItem() {
        let models = voice(FakeSpeechService())
        let changed = Mutex(false)
        withObservationTracking {
            _ = models.canToggleDictation
        } onChange: {
            changed.withLock { $0 = true }
        }
        let note = AppendingTarget()
        models.focusedDictationTarget = note
        #expect(changed.withLock { $0 }, "the Commands body re-evaluates when focus moves")
        #expect(models.canToggleDictation && models.focusedDictationNoteID == note.noteID)
        #expect(models.canRecordMeeting)
    }

    @Test func aFailedRemoveNoLongerShowsOnceTheModelIsGone() async throws {
        let speech = GatedInstallSpeech(FakeSpeechService())
        speech.failNextRemove(with: .failed("disk busy"))
        let models = voice(speech)
        models.setup.start()
        defer { models.setup.stop() }
        await models.settings.removeSpeech()
        #expect(models.settings.speechFailure != nil)
        try await speech.base.remove(.speech) // removed later another way
        try await eventually { models.settings.speechState == .notInstalled }
        #expect(models.settings.speechFailure == nil)
    }
}

/// A speech service whose status stream the test feeds by hand (to deliver
/// a stale value late); everything else is the fake's.
final class ManualStatusSpeech: SpeechServicing {
    let base: FakeSpeechService
    private let continuation = Mutex<AsyncStream<SpeechStatus>.Continuation?>(nil)

    init(_ base: FakeSpeechService) { self.base = base }

    func deliver(_ status: SpeechStatus) { _ = continuation.withLock { $0?.yield(status) } }

    var status: SpeechStatus { base.status }
    func statusUpdates() -> AsyncStream<SpeechStatus> {
        let (stream, continuation) = AsyncStream.makeStream(of: SpeechStatus.self)
        self.continuation.withLock { $0 = continuation }
        return stream
    }
    func downloadSize(of package: SpeechPackageID) -> Int64 { base.downloadSize(of: package) }
    func install(_ package: SpeechPackageID) async throws { try await base.install(package) }
    func pauseInstall(_ package: SpeechPackageID) async { await base.pauseInstall(package) }
    func cancelInstall(_ package: SpeechPackageID) async { await base.cancelInstall(package) }
    func remove(_ package: SpeechPackageID) async throws { try await base.remove(package) }
    func startDictation(_ options: DictationOptions) async throws -> DictationSession {
        try await base.startDictation(options)
    }
    func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording { try await base.startMeeting(options) }
    func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        try await base.transcribeFile(url, options: options)
    }
    func cancel(_ job: FileJob.ID) async { await base.cancel(job) }
    func inputDevices() async -> [AudioInputDevice] { await base.inputDevices() }
    func runningAudioApps() async -> [RunningAudioApp] { await base.runningAudioApps() }
    func recordingsSize() async -> Int64 { await base.recordingsSize() }
    func deleteRecordings(for noteID: UUID) async throws { try await base.deleteRecordings(for: noteID) }
    func recordingsDirectory(for noteID: UUID) -> URL { base.recordingsDirectory(for: noteID) }
    func recoverInterruptedRecordings() async -> [InterruptedRecording] { await base.recoverInterruptedRecordings() }
    func shutdown() async { await base.shutdown() }
}

@MainActor
@Suite struct VoiceStaleStatusTests {
    @Test func aStaleStatusArrivingLateDoesNotHideAFailedDownload() async throws {
        let base = FakeSpeechService.notInstalled()
        base.failNextInstall(with: .verificationFailed(file: "Encoder.mlmodelc/weights/weight.bin"))
        let speech = ManualStatusSpeech(base)
        let models = VoiceModels(speech: speech, permissions: FakePermissionBroker(),
                                 preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
        models.setup.start()
        defer { models.setup.stop() }
        let failed = SetupDisplay.failed("A downloaded file didn't match. Download again.")
        models.setup.download()
        await models.setup.waitForInstall()
        #expect(models.setup.display == failed)
        // The stream's values arrive late, oldest first (load on the main actor).
        speech.deliver(SpeechStatus())
        for _ in 0..<50 { await Task.yield() }
        #expect(models.setup.display == failed, "the stale Not installed doesn't bring Download back")
        speech.deliver(base.status)
        for _ in 0..<50 { await Task.yield() }
        #expect(models.setup.display == failed)
    }
}

private typealias SetupDisplay = SpeechSetupModel.Display
