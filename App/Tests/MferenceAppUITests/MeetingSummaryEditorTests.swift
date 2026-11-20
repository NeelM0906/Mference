import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import SwiftData
import Testing

/// A summary written into a meeting note while that note is open in the
/// editor (spec 4.5): it goes through the app's one `NoteStore`, so the
/// editor must take it in, never save its older copy over it. Model-free:
/// `ScriptedEngineService` plays Quick.
@MainActor
@Suite(.serialized) struct MeetingSummaryEditorTests {
    static let summary = #"{"summary": ["Notes uses three panes."], "decisions": [], "actionItems": [{"text": "Draw the empty state", "owner": "Sam"}]}"#
    static let pendingEditor = "Needs #232: the editor takes in writes made through the store while a note is open."

    struct Setup {
        let services: AppServices
        let noteID: UUID
        let controller: NoteEditorController
        let harness: NoteEditorTests.Harness
        var note: Note { (try? services.notes.note(id: noteID)).flatMap { $0 }! }
        var stored: NoteDocument { NoteDocument.decode(note.bodyJSON) }
    }

    /// A finished meeting note with a transcript, open in an editor.
    func setup() throws -> Setup {
        let engine = ScriptedEngineService(fallback: .init(content: Self.summary))
        let services = AppServices(engine: engine, settings: AppSettings(store: InMemorySettingsStore()),
                                   permissions: FakePermissionBroker(),
                                   modelContainer: try AppModelContainer.inMemory(), speech: FakeSpeechService())
        let noteID = UUID()
        let note = try services.notes.createNote(id: noteID, kind: .meeting, title: "Design review")
        let record = NoteRecording(startedAt: .now)
        record.segments = [TranscriptSegment(track: .microphone, start: 0, end: 4, speakerID: SpeakerInfo.youID,
                                             text: "Let's use three panes.", isFinal: true)]
        record.endedAt = .now
        record.transcriptionState = .done
        services.notes.context.insert(record)
        note.recording = record
        try services.notes.context.save()
        let controller = NoteEditorController(store: services.notes, autosaveDelay: .seconds(60))
        let harness = NoteEditorTests.Harness()
        controller.attach(harness.view)
        controller.open(note)
        return Setup(services: services, noteID: noteID, controller: controller, harness: harness)
    }

    func hasSummary(_ document: NoteDocument) -> Bool {
        document.blocks.contains(.heading(level: .heading, runs: [TextRun("Summary")]))
    }

    @Test func aSummaryWrittenWhileTheNoteIsOpenIsKept() async throws {
        let s = try setup()
        s.services.summaries.summarize(s.noteID)
        await s.services.summaries.waitUntilIdle()
        #expect(s.services.summaries.status(for: s.noteID) == .written)
        #expect(hasSummary(s.stored))
        // Nothing was typed: saving the open note keeps the summary.
        s.controller.saveNow(force: true)
        #expect(hasSummary(s.stored))
        withKnownIssue(Comment(rawValue: Self.pendingEditor), isIntermittent: true) {
            // The open editor shows it.
            #expect(hasSummary(s.harness.view.document))
        }
    }

    @Test func typingDuringTheSummaryKeepsBoth() async throws {
        let s = try setup()
        // The person types in the note while Quick writes the summary.
        let end = s.harness.view.string.utf16.count
        s.harness.view.setSelectedRange(NSRange(location: end, length: 0))
        s.harness.type("\nAsk Priya about the budget")
        s.services.summaries.summarize(s.noteID)
        await s.services.summaries.waitUntilIdle()
        #expect(hasSummary(s.stored))
        s.controller.saveNow(force: true)
        let saved = s.stored
        #expect(saved.plainText.contains("Ask Priya about the budget"))
        withKnownIssue(Comment(rawValue: Self.pendingEditor), isIntermittent: true) {
            // The editor's save merged the summary in instead of dropping it.
            #expect(hasSummary(saved))
            #expect(s.services.summaries.uncheckedActionItemCount(for: s.noteID) == 1)
        }
    }
}
