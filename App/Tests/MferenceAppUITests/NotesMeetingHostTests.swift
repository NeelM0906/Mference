import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import SwiftData
import SwiftUI
import Testing

/// The Notes screen hosting N3's meeting views over N4's session: the live
/// meeting layout (board 4), the finished meeting around its note (board 5),
/// and the purge on opening Notes waiting for launch recovery.
@MainActor
@Suite(.serialized) struct NotesMeetingHostTests {
    static let summary = #"{"summary": ["Notes uses three panes: folders, a list grouped by day, and the editor.", "Dictation inserts text after each pause."], "decisions": [], "actionItems": [{"text": "Draw the empty state for Meetings", "owner": "Sam"}, {"text": "Check call-audio permission copy", "owner": "You"}]}"#

    func services(speech: FakeSpeechService = FakeSpeechService()) throws -> AppServices {
        AppServices(engine: ScriptedEngineService(fallback: .init(content: Self.summary)),
                    settings: AppSettings(store: InMemorySettingsStore()), permissions: FakePermissionBroker(),
                    modelContainer: try AppModelContainer.inMemory(), speech: speech)
    }

    @Test func thePurgeOnOpeningNotesWaitsForLaunchRecovery() async throws {
        let model = try NotesFixtures.model()
        var events: [String] = []
        let (gate, open) = AsyncStream.makeStream(of: Void.self)
        model.waitForLaunchRecovery = {
            events.append("recovery started")
            for await _ in gate { break }
            events.append("recovery done")
        }
        let purge = Task { await model.purgeExpired() }
        #expect(await eventually { events == ["recovery started"] })
        #expect(model.counts.recentlyDeleted == 2, "nothing purged before recovery")
        open.yield()
        await purge.value
        #expect(events == ["recovery started", "recovery done"])
    }

    @Test func theLiveMeetingNoteShowsTheMeetingLayout() async throws {
        let speech = FakeSpeechService(meeting: FakeSpeechService.MeetingScript(
            live: [.segment(TranscriptSegment(track: .microphone, start: 3, end: 6, speakerID: SpeakerInfo.youID,
                                              text: "Let's start.", isFinal: true))],
            result: TranscriptResult(segments: [], speakers: [SpeakerInfo.you], duration: 60)))
        let services = try services(speech: speech)
        let model = NotesModel(services: services)
        let noteID = try await services.meetings.start(MeetingStartOptions())
        model.open(noteID: noteID)
        #expect(model.liveMeetingNoteID == noteID)
        #expect(model.showsMeetingLayout)
        // The sidebar button brings back the panes; another note leaves the layout.
        model.showsPanesDuringMeeting = true
        #expect(!model.showsMeetingLayout)
        model.selectAdjacent(1)
        model.open(noteID: noteID)
        #expect(model.showsMeetingLayout)
        let voice = VoiceModels(speech: speech, permissions: FakePermissionBroker(),
                                preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
        voice.connect(meetings: services.meetings)
        let root = RootView(model: ShellModel(selection: .notes, defaults: nil), chrome: .inline)
            .environment(ScreenModels(notes: model))
            .environment(voice)
        _ = try await HostedSnapshot.render(root, name: "notes-04-meeting-live-hosted")
        await services.meetings.stop()
        await services.meetings.waitUntilFinished()
        #expect(!model.showsMeetingLayout)
    }

    @Test func aFinishedMeetingShowsItsPlayerSummaryAndTranscript() async throws {
        let services = try services()
        let model = NotesModel(services: services, now: { NotesFixtures.now })
        model.locale = NotesFixtures.locale
        model.calendar = NotesFixtures.calendar
        let noteID = UUID()
        let note = try services.notes.createNote(id: noteID, kind: .meeting, title: "Design review: Notes",
                                                 now: NotesFixtures.date(11, 19, 10, 2))
        let record = NoteRecording(startedAt: NotesFixtures.date(11, 19, 10, 2))
        record.segments = [
            TranscriptSegment(track: .call, start: 712, end: 724, speakerID: "spk2",
                              text: "For the list column, do we group by day like the mockup, or by folder?", isFinal: true),
            TranscriptSegment(track: .microphone, start: 725, end: 737, speakerID: SpeakerInfo.youID,
                              text: "By day. Folders live in the sidebar, and the list is always newest first.", isFinal: true),
        ]
        record.speakers = [SpeakerInfo.you, SpeakerInfo(id: "spk2", name: "Priya", colorIndex: 1)]
        record.duration = 2530
        record.endedAt = NotesFixtures.date(11, 19, 10, 44)
        record.transcriptionState = .done
        services.notes.context.insert(record)
        note.recording = record
        try services.notes.context.save()
        services.summaries.summarize(noteID)
        await services.summaries.waitUntilIdle()
        #expect(services.summaries.status(for: noteID) == .written)
        model.open(noteID: noteID)
        #expect(!model.showsMeetingLayout)
        #expect(NoteRecordingHeader.summaryState(.written) == .written(MeetingSummaries.writtenLabel))
        #expect(NoteRecordingHeader.summaryState(.none) == nil)
        for scheme in [ColorScheme.light, .dark] {
            let root = RootView(model: ShellModel(selection: .notes, defaults: nil), chrome: .inline)
                .environment(ScreenModels(notes: model))
            _ = try await HostedSnapshot.render(root, name: "notes-05-meeting-done-hosted", colorScheme: scheme)
        }
        // A time chip in the note plays the recording from there (no audio kept: nothing plays).
        #expect(model.editor.onTimeChipClick != nil)
    }
}
