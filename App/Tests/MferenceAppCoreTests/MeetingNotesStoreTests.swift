import Foundation
import SwiftData
import Testing
@testable import MferenceAppCore

/// The Notes store as the meeting layer's writer: sections become note
/// blocks after the title line, Redo replaces only them, action items and
/// typed notes read back from the body, and everything survives a fresh
/// read of the store. Model-free.
@MainActor
@Suite struct MeetingNotesStoreTests {
    static let sections = MeetingSections(
        summary: ["Notes uses three panes.", "Dictation inserts text after each pause."],
        decisions: ["Group the list by day."],
        actionItems: [MeetingActionItem(text: "Draw the empty state for Meetings", owner: "Sam"),
                      MeetingActionItem(text: "Check call-audio permission copy")])

    func store() throws -> (NoteStore, ModelContainer) {
        let container = try AppModelContainer.inMemory()
        return (NoteStore(container: container, attachments: .temporary(), speech: FakeSpeechService()), container)
    }

    @Test func aMeetingNoteKeepsItsIDAndTitle() throws {
        let (notes, container) = try store()
        let id = UUID()
        try notes.createNote(id: id, kind: .meeting, title: "Meeting 19 Nov at 10:02", createdAt: .now)
        let note = try #require(try notes.note(id: id))
        #expect(note.kind == .meeting)
        #expect(note.title == "Meeting 19 Nov at 10:02")
        #expect(notes.document(of: note).title == "Meeting 19 Nov at 10:02")
        #expect(try container.mainContext.fetch(FetchDescriptor<Note>()).count == 1)
    }

    @Test func sectionsAreWrittenAfterTheTitleAndReadBack() throws {
        let (notes, _) = try store()
        let id = UUID()
        try notes.createNote(id: id, kind: .meeting, title: "Design review", createdAt: .now)
        try notes.appendParagraph("My own line.", to: id)
        try notes.replaceMeetingSections(Self.sections, in: id)
        let note = try #require(try notes.note(id: id))
        let blocks = notes.document(of: note).blocks
        #expect(blocks.map(\.text) == [
            "Design review", "Summary\u{2060}", "Notes uses three panes.", "Dictation inserts text after each pause.",
            "Decisions", "Group the list by day.", "Action items",
            "Draw the empty state for Meetings · Sam", "Check call-audio permission copy", "My own line.",
        ])
        #expect(notes.actionItems(in: id) == [
            MeetingActionItemRow(item: MeetingActionItem(text: "Draw the empty state for Meetings", owner: "Sam"),
                                 isDone: false),
            MeetingActionItemRow(item: MeetingActionItem(text: "Check call-audio permission copy"), isDone: false),
        ])
        #expect(notes.personalNotes(in: id) == "My own line.")
        #expect(note.searchText.localizedCaseInsensitiveContains("three panes"))
    }

    @Test func redoReplacesOnlyTheSectionsAndTicksAreRead() throws {
        let (notes, _) = try store()
        let id = UUID()
        try notes.createNote(id: id, kind: .meeting, title: "Call", createdAt: .now)
        try notes.replaceMeetingSections(Self.sections, in: id)
        try notes.appendParagraph("Typed after.", to: id)
        // The person ticks the first action item in the editor.
        let note = try #require(try notes.note(id: id))
        var document = notes.document(of: note)
        let index = try #require(document.blocks.firstIndex { $0.text.hasPrefix("Draw the empty state") })
        document.blocks[index] = .check(runs: document.blocks[index].runs, done: true, indent: 0)
        try notes.save(document, to: note)
        #expect(notes.actionItems(in: id).map(\.isDone) == [true, false])

        try notes.replaceMeetingSections(MeetingSections(summary: ["Short."]), in: id)
        #expect(notes.document(of: note).blocks.map(\.text) == ["Call", "Summary\u{2060}", "Short.", "Typed after."])
        #expect(notes.actionItems(in: id).isEmpty)
    }

    @Test func sectionsBelowATypedLineAreStillFoundAndReplacedInPlace() throws {
        // The editor merges a line typed during the summary in above it.
        let blocks: [NoteDocument.Block] = [.heading(level: .title, runs: [TextRun("Call")]),
                                            .paragraph(runs: [TextRun("Typed while it ran.")])]
            + MeetingBlocks.blocks(for: Self.sections)
            + [.paragraph(runs: [TextRun("Typed after.")])]
        #expect(MeetingBlocks.actionItems(in: blocks).count == 2)
        #expect(MeetingBlocks.personalNotes(in: blocks) == "Typed while it ran.\nTyped after.")
        let redone = MeetingBlocks.replacingSections(in: blocks, with: MeetingSections(summary: ["Short."]))
        #expect(redone.map(\.text) == ["Call", "Typed while it ran.", "Summary\u{2060}", "Short.", "Typed after."])
    }

    @Test func markersAndLinesGoAtTheEnd() throws {
        let (notes, _) = try store()
        let id = UUID()
        try notes.createNote(id: id, kind: .meeting, title: "Standup", createdAt: .now)
        try notes.appendMarker(at: 754, to: id)
        try notes.appendParagraph("Recording ended unexpectedly at 10:42", to: id)
        let note = try #require(try notes.note(id: id))
        #expect(notes.document(of: note).blocks.suffix(2) == [
            .marker(time: 754), .paragraph(runs: [TextRun("Recording ended unexpectedly at 10:42")]),
        ])
    }

    @Test func aMissingNoteIsIgnored() throws {
        let (notes, _) = try store()
        let gone = UUID()
        try notes.replaceMeetingSections(Self.sections, in: gone)
        try notes.appendParagraph("x", to: gone)
        notes.transcriptDidChange(in: gone)
        #expect(notes.actionItems(in: gone).isEmpty)
        #expect(notes.personalNotes(in: gone).isEmpty)
    }

    @Test func aStoppedMeetingsSummaryIsInTheNoteBody() async throws {
        let speech = FakeSpeechService()
        let segment = TranscriptSegment(track: .microphone, start: 0, end: 3, speakerID: SpeakerInfo.youID,
                                        text: "Let's group the list by day.", isFinal: true)
        speech.setMeetingScript(FakeSpeechService.MeetingScript(
            live: [.segment(segment)], result: TranscriptResult(segments: [segment], speakers: [.you], duration: 3)))
        let engine = ScriptedEngineService(fallback: .init(content: MeetingSummaryTests.good))
        let services = AppServices(engine: engine, settings: AppSettings(store: InMemorySettingsStore()),
                                   permissions: FakePermissionBroker(),
                                   modelContainer: try AppModelContainer.inMemory(), speech: speech)
        let id = try await services.meetings.start(MeetingStartOptions())
        await services.meetings.stop()
        await services.meetings.waitUntilFinished()
        await services.summaries.waitUntilIdle()
        #expect(services.summaries.status(for: id) == .written)
        let note = try #require(try services.notes.note(id: id))
        // Read back from the stored body, as after a relaunch.
        let stored = NoteDocument.decode(note.bodyJSON)
        #expect(stored.blocks.contains(.heading(level: .heading, runs: [TextRun("Summary\u{2060}")])))
        #expect(services.summaries.uncheckedActionItemCount(for: id) == 2)
    }

    @Test func quittingStopsTheMeetingAndWaitsForItsTranscript() async throws {
        let speech = FakeSpeechService(stepped: true)
        let services = AppServices.preview(speech: speech)
        let id = try await services.meetings.start(MeetingStartOptions(summaryWhenStopped: false))
        await services.shutDownVoice()
        #expect(!services.meetings.isActive)
        let record = try #require(try services.notes.note(id: id)?.recording)
        #expect(record.endedAt != nil && record.transcriptionState == .done)
        #expect(speech.calls.contains(.stopMeeting))
        // AppRuntime shuts speech down next; this step doesn't do it twice.
        #expect(!speech.calls.contains(.shutdown))
    }

    @Test func aSummaryHeadingThePersonTypedIsNeverReplaced() throws {
        let (notes, _) = try store()
        let id = UUID()
        let note = try notes.createNote(id: id, kind: .meeting, title: "Planning", now: .now)
        // The person typed their own Summary section during the meeting.
        try notes.save(NoteDocument(blocks: [.heading(level: .title, runs: [TextRun("Planning")]),
                                             .heading(level: .heading, runs: [TextRun("Summary")]),
                                             .bullet(runs: [TextRun("My own point.")], indent: 0)]), to: note)
        try notes.replaceMeetingSections(Self.sections, in: id)
        try notes.replaceMeetingSections(MeetingSections(summary: ["Redone."]), in: id)
        let texts = notes.document(of: note).blocks.map(\.text)
        #expect(texts == ["Planning", "Summary\u{2060}", "Redone.", "Summary", "My own point."])
        #expect(notes.personalNotes(in: id) == "Summary\nMy own point.")
        #expect(notes.actionItems(in: id).isEmpty)
    }

    @Test func meetingLinesAreNotSentAsThePersonsNotes() throws {
        let (notes, _) = try store()
        let id = UUID()
        try notes.createNote(id: id, kind: .meeting, title: "Call", createdAt: .now)
        try notes.appendParagraph(MeetingCopy.micOnly, to: id)
        try notes.appendParagraph("Recording ended unexpectedly at 10:42", to: id)
        try notes.appendParagraph("Budget is tight.", to: id)
        #expect(notes.personalNotes(in: id) == "Budget is tight.")
    }
}
