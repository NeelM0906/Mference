import Foundation
import MferenceAppCore
@testable import MferenceAppUI
@testable import MferenceMac
import MferenceDesign
import SwiftUI
import Testing

/// A double for N4's MeetingSession controls.
@MainActor
final class FakeRecordingControls: LiveRecordingControls {
    var calls: [String] = []
    func pauseRecording() async { calls.append("pause") }
    func resumeRecording() async { calls.append("resume") }
    func stopRecording() async { calls.append("stop") }
    func markMomentInRecording() async { calls.append("mark") }
}

/// The record popover (board 3), the timer chip and menu bar (D7), and quit.
@MainActor
@Suite(.serialized) struct VoiceRecordTests {
    private func voice(_ speech: FakeSpeechService, broker: FakePermissionBroker = FakePermissionBroker(),
                       preferences: SpeechPreferences = SpeechPreferences(store: InMemorySettingsStore()))
        -> VoiceModels {
        VoiceModels(speech: speech, permissions: broker, preferences: preferences, isOffline: { false })
    }

    @Test func startAsksMicrophoneThenCallAudioThenStartsAndOpensTheNote() async {
        let broker = FakePermissionBroker()
        let models = voice(FakeSpeechService(), broker: broker)
        var requests: [RecordMeetingModel.Request] = []
        var opened: [UUID] = []
        let note = UUID()
        models.startMeeting = { request in requests.append(request); return note }
        models.openNote = { opened.append($0) }
        #expect(await models.recordMeetingModel.startRecording())
        #expect(broker.calls == [.request(.microphone), .request(.systemAudio)], "microphone first, then call audio")
        #expect(requests == [RecordMeetingModel.Request(microphoneID: nil, callAudio: .allApps, speakerLabels: true,
                                                        keepAudio: true, summaryWhenYouStop: true,
                                                        language: .automatic)])
        #expect(opened == [note], "the new meeting note opens")
    }

    @Test func deniedMicrophoneStopsHere() async {
        let broker = FakePermissionBroker(answers: [.microphone: .denied])
        let models = voice(FakeSpeechService(), broker: broker)
        var started = false
        models.startMeeting = { _ in started = true; return UUID() }
        #expect(!(await models.recordMeetingModel.startRecording()))
        #expect(!started)
        #expect(models.recordMeetingModel.errorMessage == SpeechError.permissionDenied(.microphone).message)
        #expect(broker.calls == [.request(.microphone)], "call audio is not asked")
    }

    @Test func deniedCallAudioStillStartsWithTheChosenSource() async {
        let broker = FakePermissionBroker(answers: [.systemAudio: .denied])
        let preferences = SpeechPreferences(store: InMemorySettingsStore())
        preferences.callAudio = .app(bundleID: "us.zoom.xos", name: "Zoom")
        let models = voice(FakeSpeechService(), broker: broker, preferences: preferences)
        var requests: [RecordMeetingModel.Request] = []
        models.startMeeting = { request in requests.append(request); return UUID() }
        #expect(await models.recordMeetingModel.startRecording())
        #expect(requests.first?.callAudio == .app(bundleID: "us.zoom.xos", name: "Zoom"),
                "the speech service records the microphone only and the note says so")
    }

    @Test func callAudioOffIsNotAsked() async {
        let broker = FakePermissionBroker()
        let preferences = SpeechPreferences(store: InMemorySettingsStore())
        preferences.callAudio = .off
        let models = voice(FakeSpeechService(), broker: broker, preferences: preferences)
        models.startMeeting = { _ in UUID() }
        #expect(await models.recordMeetingModel.startRecording())
        #expect(broker.calls == [.request(.microphone)])
    }

    @Test func choicesFollowTheDefaultsAndAvailability() async {
        let preferences = SpeechPreferences(store: InMemorySettingsStore())
        preferences.keepAudio = .deleteAfterTranscribing
        preferences.summaryWhenYouStop = false
        let speech = FakeSpeechService(status: SpeechStatus(speech: .ready, speakerLabels: .notInstalled))
        let models = voice(speech, preferences: preferences)
        var request: RecordMeetingModel.Request?
        models.startMeeting = { request = $0; return UUID() }
        await models.recordMeetingModel.load()
        #expect(models.recordMeetingModel.microphoneName == "MacBook Pro Microphone")
        #expect(!models.recordMeetingModel.speakerLabelsAvailable)
        #expect(await models.recordMeetingModel.startRecording())
        #expect(request?.speakerLabels == false, "no speaker labels without their download: You and Others")
        #expect(request?.keepAudio == false && request?.summaryWhenYouStop == false)
    }

    @Test func busyAndUnwiredSayWhy() async {
        let models = voice(FakeSpeechService())
        #expect(!(await models.recordMeetingModel.startRecording()))
        #expect(models.recordMeetingModel.errorMessage == "The recording didn't start. Try again.")
        models.startMeeting = { _ in throw SpeechError.busy(.dictation) }
        #expect(!(await models.recordMeetingModel.startRecording()))
        #expect(models.recordMeetingModel.errorMessage == "Dictation is on. Finish it first.")
        models.startMeeting = { _ in throw SpeechError.lowDiskSpace(availableBytes: 1) }
        #expect(!(await models.recordMeetingModel.startRecording()))
        #expect(models.recordMeetingModel.errorMessage?.hasPrefix("Less than 1 GB") == true)
    }

    @Test func recordMeetingGoesThroughSetup() async throws {
        let speech = FakeSpeechService.notInstalled()
        let models = voice(speech)
        models.recordMeeting()
        #expect(models.setup.pendingAction == .recordMeeting && !models.showsRecordPopover)
        models.setup.download()
        await models.setup.waitForInstall()
        #expect(models.showsRecordPopover, "the popover opens once the model is ready")
    }

    @Test func liveMeetingDrivesChipMenuAndQuit() async throws {
        let speech = FakeSpeechService()
        let models = voice(speech)
        models.setup.start()
        defer { models.setup.stop() }
        let controls = FakeRecordingControls()
        models.recordingControls = controls
        var opened: [UUID] = []
        models.openNote = { opened.append($0) }
        let note = UUID()
        _ = try await speech.startMeeting(MeetingOptions(noteID: note))
        for _ in 0..<200 where models.liveMeeting == nil { try await Task.sleep(for: .milliseconds(5)) }
        #expect(models.liveMeeting?.noteID == note)
        models.recordMeeting()
        #expect(opened == [note] && !models.showsRecordPopover, "Record meeting opens the running meeting")
        await models.stopLiveSessions()
        #expect(controls.calls == ["stop"])
    }

    @Test func dictationIsNotAMeeting() async throws {
        let speech = FakeSpeechService(stepped: true)
        let models = voice(speech)
        models.setup.start()
        defer { models.setup.stop() }
        _ = try await speech.startDictation(DictationOptions())
        for _ in 0..<50 { await Task.yield() }
        #expect(models.setup.status.liveSession?.kind == .dictation)
        #expect(models.liveMeeting == nil, "no timer chip for dictation")
    }

    @Test func clockAndMenuText() {
        #expect(VoiceFormat.clock(768) == "12:48")
        #expect(VoiceFormat.clock(3_725) == "1:02:05")
        #expect(VoiceFormat.spokenClock(768) == "12 minutes, 48 seconds")
        let start = Date(timeIntervalSince1970: 1_000)
        let live = LiveSessionStatus(kind: .meeting, noteID: UUID(), startedAt: start, recordedTime: 60,
                                     updatedAt: start)
        #expect(RecordingMenuContent.title(live, at: start.addingTimeInterval(708)) == "Recording · 12:48")
        var paused = live
        paused.isPaused = true
        #expect(RecordingMenuContent.title(paused, at: start.addingTimeInterval(708)) == "Paused · 01:00",
                "paused time stands still")
    }

    @Test func quitCopy() {
        #expect(QuitConfirmation.title == "A meeting is recording")
        #expect(QuitConfirmation.stopAndQuit == "Stop and Quit")
        #expect(!QuitConfirmation.message.contains("["))
    }

    // MARK: Renders

    @Test(arguments: [ColorScheme.light, .dark])
    func recordPopoverBoard(colorScheme: ColorScheme) async throws {
        let models = voice(FakeSpeechService(),
                           broker: FakePermissionBroker(states: [.microphone: .granted, .systemAudio: .granted]))
        await models.recordMeetingModel.load()
        try Snapshot.render(NotesBoard {
            Color.clear
        }
        .overlay {
            Rectangle().fill(MFColor.ink.opacity(0.1)).padding(.top, 57)
        }
        .overlay(alignment: .topTrailing) {
            RecordMeetingPopover(model: models.recordMeetingModel) {}
                .background(RoundedRectangle(cornerRadius: MFRadius.xl).fill(MFColor.surface))
                .overlay(RoundedRectangle(cornerRadius: MFRadius.xl).strokeBorder(MFColor.hairlineStrong))
                .shadow(color: MFColor.ink.opacity(0.2), radius: 30, y: 20)
                .padding(.top, 50)
                .padding(.trailing, 96)
        }, name: "notes-03-meeting-start", colorScheme: colorScheme)
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func timerChipOnAnotherSection(colorScheme: ColorScheme) async throws {
        let speech = FakeSpeechService()
        let models = voice(speech)
        let start = Date.now.addingTimeInterval(-768)
        speech.setStatus(SpeechStatus(speech: .ready, speakerLabels: .ready,
                                      liveSession: LiveSessionStatus(kind: .meeting, noteID: UUID(), startedAt: start,
                                                                     recordedTime: 768, updatedAt: .now)))
        models.setup.start()
        defer { models.setup.stop() }
        for _ in 0..<200 where models.liveMeeting == nil { try await Task.sleep(for: .milliseconds(5)) }
        let screenModels = ScreenModels()
        try Snapshot.render(ShellBoard(selection: .calendar, screenModels: screenModels) {
            Color.clear
        }.environment(models), name: "recording-chip-calendar", colorScheme: colorScheme)
    }
}
