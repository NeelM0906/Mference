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
        voice.connect(notes: model)
        // Board 4's chrome: the timer chip alone (no Transcribe file, Record meeting or New note).
        #expect(model.isMeetingLive && model.voice.toolbar != nil)
        let root = RootView(model: ShellModel(selection: .notes, defaults: nil), chrome: .inline)
            .environment(ScreenModels(notes: model))
            .environment(voice)
        _ = try await HostedSnapshot.render(root, name: "notes-04-meeting-live-hosted")
        await services.meetings.stop()
        await services.meetings.waitUntilFinished()
        #expect(!model.showsMeetingLayout && !model.isMeetingLive)
    }

    @Test func theTimerChipAndTheRecordingBarReadTheSameSecond() {
        let updated = Date(timeIntervalSinceReferenceDate: 1000)
        let live = LiveSessionStatus(kind: .meeting, noteID: nil, startedAt: updated, recordedTime: 3.7, updatedAt: updated)
        // Two clocks rendering a few milliseconds apart, across the moment the
        // raw time passes 5 s (3.7 + 1.3): both say the same second.
        let chip = VoiceFormat.liveTime(live, at: updated.addingTimeInterval(1.29))
        let bar = VoiceFormat.liveTime(live, at: updated.addingTimeInterval(1.31))
        #expect(VoiceFormat.clock(chip) == VoiceFormat.clock(bar))
        #expect(VoiceFormat.clock(VoiceFormat.liveTime(live, at: updated.addingTimeInterval(2))) == "00:05")
        let paused = LiveSessionStatus(kind: .meeting, noteID: nil, startedAt: updated, isPaused: true, recordedTime: 42,
                                       updatedAt: updated)
        #expect(VoiceFormat.liveTime(paused, at: updated.addingTimeInterval(30)) == 42)
    }

    @Test func aMeetingThatEndsWithThePanesShownGetsItsPlayerAndTranscript() async throws {
        let speech = FakeSpeechService(meeting: FakeSpeechService.MeetingScript(
            live: [.segment(TranscriptSegment(track: .microphone, start: 3, end: 6, speakerID: SpeakerInfo.youID,
                                              text: "Let's start.", isFinal: true))],
            result: TranscriptResult(segments: [TranscriptSegment(track: .microphone, start: 3, end: 6,
                                                                  speakerID: SpeakerInfo.youID, text: "Let's start.",
                                                                  isFinal: true)],
                                     speakers: [SpeakerInfo.you], duration: 60)))
        let services = try services(speech: speech)
        let model = NotesModel(services: services)
        let noteID = try await services.meetings.start(MeetingStartOptions())
        model.open(noteID: noteID)
        model.showsPanesDuringMeeting = true
        let note = try #require(model.selectedNote)
        let during = model.recordingKey(for: note)
        #expect(during.isLive)
        #expect(model.finishedRecording(for: note) == nil, "no player while it records")
        await services.meetings.stop()
        await services.meetings.waitUntilFinished()
        let after = model.recordingKey(for: note)
        #expect(after != during, "the editor loads again")
        let finished = try #require(model.finishedRecording(for: note))
        #expect(finished.transcript.finalSegments.map(\.text) == ["Let's start."])
    }

    @Test func aFileNoteOpenedWhileTranscribingShowsItsTranscriptOnceDone() async throws {
        let services = try services()
        let model = NotesModel(services: services)
        let noteID = UUID()
        let note = try services.notes.createNote(id: noteID, kind: .file, title: "Interview", now: .now)
        let record = NoteRecording(startedAt: .now, audioKept: false, sourceFileName: "interview.m4a")
        record.transcriptionState = .running
        services.notes.context.insert(record)
        note.recording = record
        try services.notes.context.save()
        model.open(noteID: noteID)
        let running = model.recordingKey(for: note)
        #expect(model.finishedRecording(for: note) == nil, "no empty transcript while it runs")
        // Something read the transcript meanwhile: the session caches it empty.
        #expect(services.meetings.transcript(for: noteID)?.segments.isEmpty == true)
        record.segments = [TranscriptSegment(track: .file, start: 1, end: 4, speakerID: "spk1",
                                             text: "Thanks for coming in.", isFinal: true)]
        record.transcriptionState = .done
        record.endedAt = .now
        try services.notes.context.save()
        #expect(model.recordingKey(for: note) != running, "the editor loads again")
        let finished = try #require(model.finishedRecording(for: note))
        #expect(finished.transcript.finalSegments.map(\.text) == ["Thanks for coming in."])
    }

    @Test func aDeletedRecordingKeepsItsPlayerButNotItsSummaryActions() async throws {
        let services = try services()
        let model = NotesModel(services: services)
        let (noteID, note) = try Self.finishedMeeting(in: services)
        services.summaries.summarize(noteID)
        await services.summaries.waitUntilIdle()
        #expect(model.showsSummaryActions(for: note))
        try services.notes.moveToTrash(note)
        #expect(!model.showsSummaryActions(for: note))
        model.addActionItemsToTodos(for: note)
        #expect(!services.summaries.todosAdded(for: noteID), "no to-dos from a deleted note")
        #expect(model.finishedRecording(for: note) != nil, "its transcript still shows")
    }

    @Test func openingNotesSelectsTheFirstNoteBeforeThePurgeWaits() async throws {
        let model = try NotesFixtures.model()
        model.select(nil)
        let (gate, open) = AsyncStream.makeStream(of: Void.self)
        model.waitForLaunchRecovery = { for await _ in gate { break } }
        let opening = Task { await model.opened() }
        #expect(await eventually { model.selectedNoteID != nil }, "a slow recovery leaves a note open")
        open.yield()
        await opening.value
        #expect(model.selectedNoteID != nil)
    }

    @Test func theTranscriptRowPlayingMatchesThePlayersPick() {
        let segments = [0.0, 10, 20].map {
            TranscriptSegment(track: .microphone, start: $0, end: $0 + 5, speakerID: SpeakerInfo.youID, text: "x",
                              isFinal: true)
        }
        for time in [0.0, 5, 9.96, 10, 15, 25, 99] {
            let pick = segments.last { $0.start <= time + 0.05 }?.id
            let rows = segments.indices.filter {
                TranscriptRow.isPlaying(start: segments[$0].start,
                                        nextStart: $0 + 1 < segments.count ? segments[$0 + 1].start : nil,
                                        time: time, playing: true)
            }.map { segments[$0].id }
            #expect(rows == [pick].compactMap { $0 }, "at \(time)")
        }
        #expect(!TranscriptRow.isPlaying(start: 0, nextStart: nil, time: 0, playing: false), "nothing before play")
    }

    /// Board 5's meeting: two lines, two speakers, 42 min, audio kept.
    static func finishedMeeting(in services: AppServices) throws -> (UUID, Note) {
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
        record.tracks = [RecordedTrack(fileName: "mic-1.m4a", kind: .microphone, startOffset: 0, duration: 2530),
                         RecordedTrack(fileName: "call-1.m4a", kind: .call, startOffset: 0, duration: 2530)]
        services.notes.context.insert(record)
        note.recording = record
        try services.notes.context.save()
        return (noteID, note)
    }

    @Test func aFinishedMeetingShowsItsPlayerSummaryAndTranscript() async throws {
        let services = try services()
        let model = NotesModel(services: services, now: { NotesFixtures.now })
        model.locale = NotesFixtures.locale
        model.calendar = NotesFixtures.calendar
        model.makePlayback = { FakePlayback() }
        let (noteID, _) = try Self.finishedMeeting(in: services)
        services.summaries.summarize(noteID)
        await services.summaries.waitUntilIdle()
        #expect(services.summaries.status(for: noteID) == .written)
        model.open(noteID: noteID)
        #expect(!model.showsMeetingLayout)
        let note = try #require(model.selectedNote)
        #expect(model.finishedRecording(for: note)?.audio?.tracks.count == 2, "audio kept: the player shows")
        // One "Summary" label only: the note's own heading; the line above it says who wrote it.
        #expect(NoteRecordingHeader.summaryState(.written) == .written(MeetingSummaries.writtenLabel))
        #expect(NoteRecordingHeader.writtenLine(MeetingSummaries.writtenLabel) == "Summary written by Quick on this Mac")
        #expect(NoteRecordingHeader.summaryState(.none) == nil)
        for scheme in [ColorScheme.light, .dark] {
            let root = RootView(model: ShellModel(selection: .notes, defaults: nil), chrome: .inline)
                .environment(ScreenModels(notes: model))
            _ = try await HostedSnapshot.render(root, name: "notes-05-meeting-done-hosted", colorScheme: scheme) {
                // Board 5: the second line playing (12:05).
                model.editor.onTimeChipClick?(725)
            }
        }
        #expect(model.editor.onTimeChipClick != nil)
    }
}
