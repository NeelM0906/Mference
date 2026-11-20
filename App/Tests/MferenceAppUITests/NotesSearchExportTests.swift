import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import PDFKit
import SwiftData
import SwiftUI
import Testing

/// ⌘K Notes results and export from the Notes screen (Notes spec 4.1, 6.6).
@MainActor
@Suite(.serialized) struct NotesSearchExportTests {
    @Test func aTranscriptHitOpensTheNoteAtItsLine() throws {
        let model = try NotesFixtures.model()
        let meeting = try #require(model.notes.first { $0.kind == .meeting })
        let segment = TranscriptSegment(track: .call, start: 95, end: 99, speakerID: "s2",
                                        text: "Ship the beta on Friday", isFinal: true)
        meeting.recording?.transcriptJSON = try JSONEncoder().encode([segment])
        try model.store.transcriptDidChange(in: meeting)
        var body = model.store.document(of: meeting)
        body.blocks.append(.transcriptAnchor(segmentID: segment.id))
        try model.store.save(body, to: meeting)

        model.scope = .recentlyDeleted
        model.query = "something"
        let hits = model.searchHits("beta friday")
        #expect(hits.count == 1)
        let hit = try #require(hits.first)
        #expect(hit.match == .transcript(segmentID: segment.id, time: 95))
        model.open(hit)
        #expect(model.scope == .all)
        #expect(model.query.isEmpty)
        #expect(model.selectedNote?.id == meeting.id)
        #expect(model.revealedSegmentID == segment.id)
    }

    @Test func theSearchPanelListsNoteResults() {
        let rows = NoteSearchResults(hits: [
            NoteSearchHit(noteID: UUID(), title: "Lisbon trip, March", kind: .note, match: .title, snippet: "Flights"),
            NoteSearchHit(noteID: UUID(), title: "Design review", kind: .meeting,
                          match: .transcript(segmentID: UUID(), time: 125), snippet: "Ship the beta"),
        ]) { _ in }
        #expect(ImageRenderer(content: rows.frame(width: 520)).cgImage != nil)
    }

    @Test func exportWritesMarkdownTextAndPDF() throws {
        let model = try NotesFixtures.model()
        let note = try #require(model.selectedNote)
        let folder = FileManager.default.temporaryDirectory.appending(path: "mference-notes-export-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let md = folder.appending(path: "Lisbon.md")
        try NoteExporter.write(note, store: model.store, format: .markdown, to: md)
        let markdown = try String(contentsOf: md, encoding: .utf8)
        #expect(markdown.hasPrefix("# Lisbon trip, March\n"))
        #expect(markdown.contains("- [x] Book flights\n- [ ] Hotel, four nights in Alfama"))

        let txt = folder.appending(path: "Lisbon.txt")
        try NoteExporter.write(note, store: model.store, format: .plainText, to: txt)
        #expect(try String(contentsOf: txt, encoding: .utf8).contains("☑ Book flights"))

        let pdf = folder.appending(path: "Lisbon.pdf")
        try NoteExporter.write(note, store: model.store, format: .pdf, to: pdf)
        let document = try #require(PDFDocument(url: pdf))
        #expect(document.pageCount >= 1)
        #expect(document.string?.contains("Lisbon trip, March") == true)
        #expect(document.string?.contains("Renew travel insurance") == true)
    }

    @Test func aDamagedNoteIsNotExportedAsAnEmptyFile() throws {
        let model = try NotesFixtures.model()
        let note = try model.store.createNote()
        note.bodyJSON = Data("{\"v\":1,\"blocks\":[".utf8)
        let file = FileManager.default.temporaryDirectory.appending(path: "mference-damaged-\(UUID().uuidString).md")
        #expect(throws: NoteExportError.self) { try NoteExporter.write(note, store: model.store, format: .markdown, to: file) }
        #expect(!FileManager.default.fileExists(atPath: file.path(percentEncoded: false)))
    }

    @Test func aLongNoteMakesSeveralPDFPages() throws {
        let model = try NotesFixtures.model()
        let note = try model.store.createNote()
        let lines = (1...160).map { NoteDocument.Block.paragraph(runs: [TextRun("Line \($0) of the long note")]) }
        try model.store.save(NoteDocument(blocks: [.heading(level: .title, runs: [TextRun("Long")])] + lines), to: note)
        let file = FileManager.default.temporaryDirectory.appending(path: "mference-long-\(UUID().uuidString).pdf")
        try NoteExporter.write(note, store: model.store, format: .pdf, to: file)
        let pdf = try #require(PDFDocument(url: file))
        #expect(pdf.pageCount >= 3)
        #expect(pdf.page(at: pdf.pageCount - 1)?.string?.contains("Line 160 of the long note") == true)
        #expect(pdf.page(at: 0)?.string?.contains("Long") == true)
    }

    @Test func aTranscriptHitWaitsForTheEditorToShowItsNote() throws {
        let store = NoteStore(container: try AppModelContainer.inMemory(), attachments: .temporary(),
                              speech: FakeSpeechService())
        let note = try store.createNote(kind: .meeting)
        let segment = UUID()
        try store.save(NoteDocument(blocks: [.heading(level: .title, runs: [TextRun("Sync")])]
            + (1...40).map { .paragraph(runs: [TextRun("Line \($0)")]) } + [.transcriptAnchor(segmentID: segment)]), to: note)
        let controller = NoteEditorController(store: store)
        // ⌘K from another section: no editor view yet.
        controller.open(note)
        controller.reveal(segmentID: segment)
        let h = NoteEditorTests.Harness()
        controller.attach(h.view)
        #expect(h.view.selectedRange().location == (h.view.string as NSString).length - 1)
    }

    @Test func shareTextIsMarkdown() throws {
        let model = try NotesFixtures.model()
        let h = NoteEditorTests.Harness()
        model.editor.attach(h.view)
        #expect(model.editor.shareText.hasPrefix("# Lisbon trip, March\n\nFlights booked"))
    }
}
