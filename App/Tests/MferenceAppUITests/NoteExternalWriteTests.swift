import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import SwiftData
import Testing

/// Writes to the open note from elsewhere (N4's summary sections, markers,
/// a title) are taken in by the editor, never saved over.
@MainActor
@Suite(.serialized) struct NoteExternalWriteTests {
    typealias Harness = NoteEditorTests.Harness
    func doc(_ blocks: NoteDocument.Block...) -> NoteDocument { NoteDocument(blocks: blocks) }
    let title = NoteDocument.Block.heading(level: .title, runs: [TextRun("Weekly sync")])
    let summary: [NoteDocument.Block] = [.heading(level: .heading, runs: [TextRun("Summary")]),
                                         .bullet(runs: [TextRun("Ship Friday")], indent: 0)]

    func setUp() throws -> (NoteStore, Note, NoteEditorController, Harness) {
        let store = NoteStore(container: try AppModelContainer.inMemory(), attachments: .temporary(),
                              speech: FakeSpeechService())
        let note = try store.createNote(kind: .meeting)
        try store.save(doc(title, .paragraph(runs: [TextRun("My notes")])), to: note)
        let controller = NoteEditorController(store: store, autosaveDelay: .seconds(60))
        let h = Harness()
        controller.attach(h.view)
        controller.open(note)
        return (store, note, controller, h)
    }

    @Test func aWriteWhileIdleIsShownAndTheCaretStaysWithItsText() async throws {
        let (store, note, controller, h) = try setUp()
        h.view.setSelectedRange(NSRange(location: 15, length: 0))     // after "My "
        try store.save(NoteDocument(blocks: [title] + summary + [.paragraph(runs: [TextRun("My notes")])]), to: note)
        #expect(await eventually { h.view.document.blocks.count == 4 })
        let text = h.view.string as NSString
        #expect(h.view.selectedRange().location == text.range(of: "My notes").location + 3)
        #expect(controller.saveState == .saved)
    }

    @Test func unsavedEditsAndAWriteFromElsewhereAreMerged() throws {
        let (store, note, controller, h) = try setUp()
        h.view.setSelectedRange(NSRange(location: (h.view.string as NSString).length, length: 0))
        h.type(" and more\nA new line")
        #expect(controller.saveState == .editing)
        // N4 writes the summary after the title and appends a marker.
        try store.save(NoteDocument(blocks: [title] + summary + [.paragraph(runs: [TextRun("My notes")]),
                                                                  .marker(time: 90)]), to: note)
        controller.saveNow(force: true)
        let expected = doc(title, summary[0], summary[1], .paragraph(runs: [TextRun("My notes and more")]),
                           .paragraph(runs: [TextRun("A new line")]), .marker(time: 90))
        #expect(store.document(of: note) == expected)
        #expect(h.view.document == expected)
    }

    @Test func aPendingPhraseKeepsItsPlaceAcrossAWrite() async throws {
        let (store, note, controller, h) = try setUp()
        h.view.setSelectedRange(NSRange(location: (h.view.string as NSString).length, length: 0))
        let phrase = UUID()
        controller.beginPhrase(phrase)                             // after "My notes"
        try store.setTitle("Weekly sync, 20 Nov", of: note)
        #expect(await eventually { h.view.document.blocks.first == .heading(level: .title, runs: [TextRun("Weekly sync, 20 Nov")]) })
        controller.finishPhrase(phrase, text: "on the roadmap.")
        #expect(h.view.document.blocks.last == .paragraph(runs: [TextRun("My notes on the roadmap.")]))
    }

    @Test func theMergeKeepsBothSides() {
        let a = NoteDocument.Block.paragraph(runs: [TextRun("a")])
        let b = NoteDocument.Block.paragraph(runs: [TextRun("b")])
        let b2 = NoteDocument.Block.paragraph(runs: [TextRun("b!")])
        let c = NoteDocument.Block.paragraph(runs: [TextRun("c")])
        let x = NoteDocument.Block.marker(time: 1)
        // Ours edits b; theirs inserts x after a and removes c.
        #expect(NoteDocumentMerge.merge(base: [a, b, c], ours: [a, b2, c], theirs: [a, x, b]) == [a, x, b2])
        // Both changed the same block: both versions stay.
        let b3 = NoteDocument.Block.paragraph(runs: [TextRun("b?")])
        #expect(NoteDocumentMerge.merge(base: [a, b], ours: [a, b2], theirs: [a, b3]) == [a, b2, b3])
        #expect(NoteDocumentMerge.merge(base: [a], ours: [a], theirs: [a, x]) == [a, x])
    }
}
