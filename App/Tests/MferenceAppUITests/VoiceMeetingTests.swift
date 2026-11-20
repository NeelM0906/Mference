import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftData
import SwiftUI
import Testing

/// The meeting layout (board 4) over N4's MeetingSession and FakeSpeechService.
@MainActor
@Suite(.serialized) struct VoiceMeetingTests {
    struct Harness {
        let container: ModelContainer
        let speech: FakeSpeechService
        let session: MeetingSession
        let voice: VoiceModels
    }

    static func segment(_ text: String, _ start: TimeInterval, speaker: String, final: Bool = true) -> TranscriptSegment {
        TranscriptSegment(track: speaker == SpeakerInfo.youID ? .microphone : .call, start: start, end: start + 5,
                          speakerID: speaker, text: text, isFinal: final, languageCode: "en")
    }

    static let speakers = [SpeakerInfo.you, SpeakerInfo(id: "spk2", name: "Speaker 2", colorIndex: 1),
                           SpeakerInfo(id: "spk3", name: "Speaker 3", colorIndex: 2)]

    /// Board 4's conversation.
    static let boardScript = FakeSpeechService.MeetingScript(live: [
        .levels(microphone: 0.7, call: 0.55),
        .speakersUpdated(speakers),
        .segment(segment("For the list column, do we group by day like the mockup, or by folder?", 712, speaker: "spk2")),
        .segment(segment("By day. Folders live in the sidebar, and the list is always newest first.", 725, speaker: "you")),
        .segment(segment("And meetings get the waveform mark, so you can spot them in the list.", 739, speaker: "spk3")),
        .segment(segment("What happens if someone closes the laptop in the middle of a meeting?", 751, speaker: "spk2")),
        .segment(segment("We keep everything transcribed so far, mark the gap, and carry on when it wakes.", 760, speaker: "you")),
        .segment(segment("Okay. And the audio file itself, do we keep it by default or", 766, speaker: "spk3", final: false)),
    ], result: TranscriptResult(segments: [], speakers: speakers, duration: 770))

    func harness(_ script: FakeSpeechService.MeetingScript = boardScript, stepped: Bool = false) throws -> Harness {
        let speech = FakeSpeechService(meeting: script, stepped: stepped)
        let container = try AppModelContainer.inMemory()
        let session = MeetingSession(speech: speech, context: container.mainContext,
                                     writer: InMemoryMeetingNoteWriter(context: container.mainContext))
        let preferences = SpeechPreferences(store: InMemorySettingsStore())
        preferences.callAudio = .app(bundleID: "us.zoom.xos", name: "Zoom")
        let voice = VoiceModels(speech: speech, permissions: FakePermissionBroker(), preferences: preferences,
                                isOffline: { false })
        voice.connect(meetings: session)
        voice.setup.start()
        return Harness(container: container, speech: speech, session: session, voice: voice)
    }

    func settle(until condition: () -> Bool) async {
        for _ in 0..<400 {
            if condition() { return }
            await Task.yield()
        }
    }

    @Test func popoverStartsTheMeetingThroughTheSessionAndOpensItsNote() async throws {
        let h = try harness()
        defer { h.voice.setup.stop() }
        var opened: [UUID] = []
        h.voice.openNote = { opened.append($0) }
        #expect(await h.voice.recordMeetingModel.startRecording())
        #expect(h.session.isActive)
        #expect(opened == [h.session.noteID].compactMap { $0 })
        let started = h.speech.calls.compactMap { call -> MeetingOptions? in
            if case .startMeeting(let options) = call { options } else { nil }
        }
        #expect(started.first?.callAudio == .app(bundleID: "us.zoom.xos", name: "Zoom"))
        await settle { h.session.transcript?.segments.count == 6 }
        #expect(h.session.transcript?.segments.last?.isFinal == false, "the phrase still being spoken")
        await h.voice.stopLiveSessions()
        #expect(h.session.phase == .idle, "quit's stop waits until the transcript is saved")
    }

    @Test func menuBarControlsReachTheSession() async throws {
        let h = try harness()
        defer { h.voice.setup.stop() }
        #expect(await h.voice.recordMeetingModel.startRecording())
        await h.voice.recordingControls?.pauseRecording()
        await settle { h.session.phase == .paused }
        #expect(h.session.phase == .paused)
        await h.voice.recordingControls?.resumeRecording()
        await settle { h.session.phase == .recording }
        #expect(h.speech.calls.contains(.pause) && h.speech.calls.contains(.resume))
        await h.voice.recordingControls?.stopRecording()
        #expect(h.session.phase == .idle)
    }

    @Test func renameInTheTranscriptRenamesEverywhere() async throws {
        let h = try harness()
        defer { h.voice.setup.stop() }
        #expect(await h.voice.recordMeetingModel.startRecording())
        await settle { h.session.transcript?.segments.count == 6 }
        let transcript = try #require(h.session.transcript)
        transcript.rename(speakerID: "spk2", to: "Priya")
        let rows = transcript.segments.filter { $0.speakerID == "spk2" }.map { transcript.speaker(for: $0).name }
        #expect(rows == ["Priya", "Priya"])
        await h.voice.stopLiveSessions()
    }

    @Test func followingPausesAfterAChipJumpThenResumes() {
        let now = Date.now
        #expect(LiveTranscriptView.follows(pausedUntil: nil, now: now))
        #expect(!LiveTranscriptView.follows(pausedUntil: now.addingTimeInterval(LiveTranscriptView.followPause), now: now))
        #expect(LiveTranscriptView.follows(pausedUntil: now, now: now.addingTimeInterval(1)), "auto-follow comes back")
        #expect(TranscriptScrollRequest(time: 10) != TranscriptScrollRequest(time: 10), "the same chip scrolls again")
    }

    @Test func connectReportsActiveAndFinishing() async throws {
        let h = try harness(stepped: true)
        defer { h.voice.setup.stop() }
        #expect(!h.voice.isMeetingActive)
        #expect(await h.voice.recordMeetingModel.startRecording())
        #expect(h.voice.isMeetingActive && !h.voice.isMeetingFinishing)
        #expect(h.voice.recordMeetingModel.lastStarted?.callAudio == .app(bundleID: "us.zoom.xos", name: "Zoom"))
        h.voice.preferences.callAudio = .off
        #expect(MeetingLayoutView<EmptyView>.callSourceName(h.voice.recordMeetingModel.lastStarted!.callAudio) == "Zoom",
                "the bar keeps this meeting's source")
        await h.session.stop()
        await h.session.waitUntilFinished()
        #expect(!h.voice.isMeetingActive)
    }

    @Test func barAndLayoutCopy() {
        #expect(RecordingBar.title(phase: .recording) == "Recording")
        #expect(RecordingBar.title(phase: .paused) == "Paused")
        #expect(RecordingBar.finishingText(.transcribingLastPhrases) == "Transcribing the last phrases…")
        #expect(RecordingBar.finishingText(.labellingSpeakers(progress: 0.4)) == "Labelling speakers · 40%")
        #expect(MeetingLayoutView<EmptyView>.callSourceName(.off) == nil)
        #expect(MeetingLayoutView<EmptyView>.callSourceName(.app(bundleID: "x", name: "Zoom")) == "Zoom")
        #expect(RecordingBar.isLit(0, level: 0), "one bar while capturing silence")
        #expect(!RecordingBar.isLit(3, level: 0.5) && RecordingBar.isLit(2, level: 0.5))
        let segments = [Self.segment("a", 10, speaker: "you"), Self.segment("b", 20, speaker: "you")]
        #expect(LiveTranscriptView.segment(at: 15, in: segments)?.text == "a")
        #expect(LiveTranscriptView.segment(at: 5, in: segments)?.text == "a", "before the first line: the first")
        #expect(SpeakerPalette.color(7) == SpeakerPalette.color(1), "the palette cycles")
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func meetingLiveBoard(colorScheme: ColorScheme) async throws {
        let h = try harness()
        defer { h.voice.setup.stop() }
        #expect(await h.voice.recordMeetingModel.startRecording())
        await h.voice.recordMeetingModel.load()
        await settle { h.session.transcript?.segments.count == 6 && h.session.callAudioAvailable }
        // The board's clock: 12:48 into the meeting.
        var status = h.speech.status
        status.liveSession?.recordedTime = 768
        status.liveSession?.updatedAt = .now
        h.speech.setStatus(status)
        await settle { h.voice.liveMeeting?.recordedTime == 768 }
        let board = VStack(spacing: 0) {
            ShellChromeBar(model: ShellModel(selection: .notes, defaults: nil))
            MeetingLayoutView(session: h.session, voice: h.voice, onShowSidebar: {}) { _ in MeetingNotesStandIn() }
        }
        .mfWindowBackground()
        .environment(h.voice)
        try Snapshot.render(board, name: "notes-04-meeting-live", colorScheme: colorScheme)
        await h.voice.stopLiveSessions()
    }
}

/// A stand-in for N1's editor in meeting mode on board 4 (title, meta, time
/// chips, a Mark moment line, the hint): only for the render.
struct MeetingNotesStandIn: View {
    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            Text("Design review: Notes").font(MFFont.largeTitle).foregroundStyle(MFColor.ink)
            Text("19 November 2026 · started 10:02 · Work").font(MFFont.caption).foregroundStyle(MFColor.tertiary)
                .padding(.bottom, MFSpacing.m)
            line("03:10", "Three panes: folders, list, editor. List grouped by day.")
            line("06:42", "Dictation lands after each pause, not word by word")
            Label("Marked at 09:55", systemImage: "bookmark.fill")
                .font(MFFont.callout).foregroundStyle(MFColor.preview)
                .padding(.horizontal, MFSpacing.compact).padding(.vertical, MFSpacing.snug)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: MFRadius.s).fill(MFColor.previewFill))
            line("11:30", "Ask: what if the lid closes mid-meeting?")
            line("12:46", "Keep audio by default, setting to delete")
            Text("Type here while you listen. Each line is stamped with the meeting time, so it links to that moment in the transcript.")
                .font(MFFont.caption).foregroundStyle(MFColor.tertiary).padding(.top, MFSpacing.l)
            Spacer()
        }
    }

    private func line(_ time: String, _ text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: MFSpacing.compact) {
            Text(time).font(MFFont.keycap).foregroundStyle(MFColor.accent)
                .padding(.horizontal, MFSpacing.snug).padding(.vertical, MFSpacing.xxs)
                .background(RoundedRectangle(cornerRadius: MFRadius.xs).fill(MFColor.accentFillFaint))
            Text(text).font(MFFont.body).foregroundStyle(MFColor.ink)
        }
    }
}
