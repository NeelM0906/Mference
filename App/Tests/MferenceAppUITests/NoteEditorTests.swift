import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import SwiftData
import SwiftUI
import Testing

/// The note editor (Notes spec 4.1, 8.2): the block round trip, Markdown
/// shortcuts, lists and checklists, run styles, undo, paste, dictation
/// phrases at a moving caret, and autosave. Model-free.
@MainActor
@Suite struct NoteEditorTests {
    static let everyBlock = NoteDocument(blocks: [
        .heading(level: .title, runs: [TextRun("Lisbon trip, March")]),
        .heading(level: .heading, runs: [TextRun("To do")]),
        .heading(level: .subheading, runs: [TextRun("Before"), TextRun(" we go", italic: true)]),
        .paragraph(runs: [TextRun("Plain "), TextRun("bold", bold: true), TextRun(" "),
                          TextRun("italic", italic: true), TextRun("under", underline: true),
                          TextRun("struck", strikethrough: true),
                          TextRun("all", bold: true, italic: true, underline: true, strikethrough: true),
                          TextRun("link", link: URL(string: "https://example.com/a?b=c"))]),
        .heading(level: .heading, runs: [TextRun("Bold heading", bold: true)]),
        .bullet(runs: [TextRun("Sintra")], indent: 0),
        .bullet(runs: [TextRun("Pena palace")], indent: 1),
        .numbered(runs: [TextRun("First")], indent: 0),
        .numbered(runs: [TextRun("Second", bold: true)], indent: 0),
        .check(runs: [TextRun("Book flights")], done: true, indent: 0),
        .check(runs: [TextRun("Hotel "), TextRun("Alfama", strikethrough: true)], done: false, indent: 2),
        .paragraph(runs: []),
        .image(attachmentID: "A.png"),
        .audio(attachmentID: "B.m4a"),
        .transcriptAnchor(segmentID: UUID(uuidString: "6F9619FF-8B86-D011-B42D-00C04FC964FF")!),
        .marker(time: 754.5),
        .paragraph(runs: [TextRun("End")]),
    ])

    final class Harness {
        let window: NSWindow
        let view: NoteTextView
        init() {
            view = NoteTextView(usingTextLayoutManager: true)
            view.configure()
            view.frame = NSRect(x: 0, y: 0, width: 640, height: 900)
            window = NSWindow(contentRect: view.frame, styleMask: [.titled], backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            window.contentView = view
            window.makeFirstResponder(view)
        }

        func type(_ text: String) {
            for character in text {
                if character == "\n" { view.insertNewline(nil) } else {
                    view.insertText(String(character), replacementRange: NSRange(location: NSNotFound, length: 0))
                }
            }
        }

        var undo: UndoManager { view.undoManager! }

        /// Runs `body` as one undo group (no event loop runs in tests).
        func grouped(_ body: () -> Void) {
            undo.groupsByEvent = false
            undo.beginUndoGrouping()
            body()
            undo.endUndoGrouping()
        }
    }

    func doc(_ blocks: NoteDocument.Block...) -> NoteDocument { NoteDocument(blocks: blocks) }

    // MARK: Round trip

    @Test func everyBlockAndRunStyleSurvivesEditorToDocumentToEditor() {
        let h = Harness()
        h.view.load(Self.everyBlock)
        #expect(h.view.document == Self.everyBlock)
        // And a second pass through the editor changes nothing.
        let once = h.view.document
        h.view.load(once)
        #expect(h.view.document == Self.everyBlock)
    }

    @Test func anEmptyNoteKeepsItsTitleLine() {
        let h = Harness()
        h.view.load(.empty)
        #expect(h.view.document == .empty)
        h.type("Trip\nHotel")
        #expect(h.view.document == doc(.heading(level: .title, runs: [TextRun("Trip")]),
                                       .paragraph(runs: [TextRun("Hotel")])))
    }

    @Test func emptyLinesKeepTheirStyleAnywhere() {
        let original = doc(.heading(level: .title, runs: [TextRun("T")]),
                           .check(runs: [], done: false, indent: 0),
                           .heading(level: .heading, runs: []))
        let h = Harness()
        h.view.load(original)
        #expect(h.view.document == original)
    }

    // MARK: Typing

    @Test func markdownShortcutsTurnLinesIntoBlocks() {
        for (prefix, expected) in [("- ", NoteDocument.Block.bullet(runs: [TextRun("Milk")], indent: 0)),
                                   ("* ", .bullet(runs: [TextRun("Milk")], indent: 0)),
                                   ("1. ", .numbered(runs: [TextRun("Milk")], indent: 0)),
                                   ("[] ", .check(runs: [TextRun("Milk")], done: false, indent: 0)),
                                   ("# ", .heading(level: .heading, runs: [TextRun("Milk")]))] {
            let h = Harness()
            h.view.load(doc(.paragraph(runs: [])))
            h.type(prefix + "Milk")
            #expect(h.view.document == doc(expected), "\(prefix)")
        }
    }

    @Test func aShortcutMidLineIsJustText() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [])))
        h.type("a - b")
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("a - b")])))
    }

    @Test func returnContinuesAListAndAnEmptyItemEndsIt() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [])))
        h.type("- A\nB\n\nAfter")
        #expect(h.view.document == doc(.bullet(runs: [TextRun("A")], indent: 0),
                                       .bullet(runs: [TextRun("B")], indent: 0),
                                       .paragraph(runs: [TextRun("After")])))
    }

    @Test func aNewChecklistItemStartsUnticked() {
        let h = Harness()
        h.view.load(doc(.check(runs: [TextRun("Book")], done: true, indent: 0)))
        h.type("\nHotel")
        #expect(h.view.document == doc(.check(runs: [TextRun("Book")], done: true, indent: 0),
                                       .check(runs: [TextRun("Hotel")], done: false, indent: 0)))
    }

    @Test func tabIndentsListItemsAndBackspaceOutdents() {
        let h = Harness()
        h.view.load(doc(.bullet(runs: [TextRun("A")], indent: 0)))
        h.view.insertTab(nil)
        #expect(h.view.document == doc(.bullet(runs: [TextRun("A")], indent: 1)))
        h.view.insertBacktab(nil)
        #expect(h.view.document == doc(.bullet(runs: [TextRun("A")], indent: 0)))
        h.view.setSelectedRange(NSRange(location: 0, length: 0))
        h.view.deleteBackward(nil)
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("A")])))
    }

    // MARK: Checklists and styles

    @Test func tickingABoxAndTheChecklistShortcut() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [TextRun("One")]), .paragraph(runs: [TextRun("Two")])))
        h.view.setSelectedRange(NSRange(location: 0, length: 7))
        h.view.toggleChecklist()
        #expect(h.view.document == doc(.check(runs: [TextRun("One")], done: false, indent: 0),
                                       .check(runs: [TextRun("Two")], done: false, indent: 0)))
        h.view.toggleDone(lineAt: 4)
        #expect(h.view.document.blocks[1] == .check(runs: [TextRun("Two")], done: true, indent: 0))
        h.view.toggleChecklist()
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("One")]), .paragraph(runs: [TextRun("Two")])))
    }

    @Test func textStyleMenuSetsHeadings() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [TextRun("Plan")])))
        h.view.applyBlockKind(.subheading)
        #expect(h.view.currentBlockStyle.kind == .subheading)
        #expect(h.view.document == doc(.heading(level: .subheading, runs: [TextRun("Plan")])))
    }

    @Test func runStylesToggleOnASelectionAndForTyping() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [TextRun("Hello world")])))
        h.view.setSelectedRange(NSRange(location: 6, length: 5))
        h.view.toggle(.bold)
        #expect(h.view.has(.bold))
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Hello "), TextRun("world", bold: true)])))
        h.view.toggle(.bold)
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Hello world")])))

        h.view.setSelectedRange(NSRange(location: 11, length: 0))
        h.view.toggle(.italic)
        h.type("!")
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Hello world"), TextRun("!", italic: true)])))
    }

    @Test func styleChangesUndo() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [TextRun("Hello")])))
        h.view.setSelectedRange(NSRange(location: 0, length: 5))
        h.grouped { h.view.toggle(.underline) }
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Hello", underline: true)])))
        h.undo.undo()
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Hello")])))
        h.undo.redo()
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Hello", underline: true)])))
    }

    @Test func numberedListsCountTheirOwnLevel() {
        let h = Harness()
        h.view.load(doc(.numbered(runs: [TextRun("A")], indent: 0), .numbered(runs: [TextRun("B")], indent: 0),
                        .bullet(runs: [TextRun("C")], indent: 1), .numbered(runs: [TextRun("D")], indent: 0),
                        .paragraph(runs: [TextRun("x")]), .numbered(runs: [TextRun("E")], indent: 0)))
        let text = h.view.textStorage!
        let starts = (h.view.string as NSString).components(separatedBy: "\n").reduce(into: [0]) { $0.append($0.last! + ($1 as NSString).length + 1) }
        let numbered = NoteBlockStyle(.numbered)
        #expect(NoteTextView.ordinal(in: text, lineStart: starts[0], style: numbered) == 1)
        #expect(NoteTextView.ordinal(in: text, lineStart: starts[1], style: numbered) == 2)
        #expect(NoteTextView.ordinal(in: text, lineStart: starts[3], style: numbered) == 3)
        #expect(NoteTextView.ordinal(in: text, lineStart: starts[5], style: numbered) == 1)
    }

    // MARK: Paste

    @Test func pasteKeepsSupportedFormatsOnly() throws {
        let rich = NSMutableAttributedString(string: "Big red bold\nplain", attributes: [
            .font: NSFont.systemFont(ofSize: 30), .foregroundColor: NSColor.red,
        ])
        rich.addAttribute(.font, value: NSFont.boldSystemFont(ofSize: 30), range: NSRange(location: 8, length: 4))
        rich.addAttribute(.link, value: URL(string: "https://example.com")!, range: NSRange(location: 13, length: 5))
        let rtf = try rich.data(from: NSRange(location: 0, length: rich.length),
                                documentAttributes: [.documentType: NSAttributedString.DocumentType.rtf])
        let board = NSPasteboard(name: NSPasteboard.Name("mference-test-\(UUID().uuidString)"))
        board.declareTypes([.rtf], owner: nil)
        board.setData(rtf, forType: .rtf)

        let h = Harness()
        h.view.load(doc(.paragraph(runs: [])))
        #expect(h.view.readSelection(from: board, type: .rtf))
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Big red "), TextRun("bold", bold: true)]),
                                       .paragraph(runs: [TextRun("plain", link: URL(string: "https://example.com"))])))
        // No pasted font size or color survives.
        let font = h.view.textStorage!.attribute(.font, at: 0, effectiveRange: nil) as? NSFont
        #expect(font?.pointSize == NoteEditorMetrics.bodySize)
        board.releaseGlobally()
    }

    @Test func copyAndPasteBetweenNotesKeepsBlocks() {
        let source = Harness()
        source.view.load(doc(.check(runs: [TextRun("Hotel")], done: true, indent: 0),
                             .bullet(runs: [TextRun("Sintra", bold: true)], indent: 1)))
        source.view.selectAll(nil)
        let board = NSPasteboard(name: NSPasteboard.Name("mference-test-\(UUID().uuidString)"))
        board.declareTypes(source.view.writablePasteboardTypes, owner: nil)
        #expect(source.view.writeSelection(to: board, type: NoteTextView.blocksPasteboardType))
        #expect(source.view.writeSelection(to: board, type: .string))
        #expect(board.string(forType: .string) == "Hotel\nSintra")

        let target = Harness()
        target.view.load(doc(.paragraph(runs: [])))
        #expect(target.view.readSelection(from: board, type: NoteTextView.blocksPasteboardType))
        #expect(target.view.document == doc(.check(runs: [TextRun("Hotel")], done: true, indent: 0),
                                             .bullet(runs: [TextRun("Sintra", bold: true)], indent: 1)))
        board.releaseGlobally()
    }

    // MARK: Dictation

    @Test func eachPhraseLandsWhereItsCaretWasWhenItStarted() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [TextRun("Book flights")])))
        let a = UUID()
        h.view.beginPhrase(a)
        // The caret moves and the person types meanwhile.
        h.view.setSelectedRange(NSRange(location: 0, length: 0))
        h.type("Today: ")
        h.view.finishPhrase(a, text: "then the hotel.")
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Today: Book flights then the hotel.")])))
        #expect(h.view.selectedRange().location == 7)

        let b = UUID()
        h.view.beginPhrase(b)
        h.view.finishPhrase(b, text: "first")
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Today: first Book flights then the hotel.")])))
    }

    @Test func phrasesStartedAtOneSpotLandInOrder() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [])))
        let (a, b, c) = (UUID(), UUID(), UUID())
        h.view.beginPhrase(a)
        h.view.beginPhrase(b)
        h.view.beginPhrase(c)
        h.view.finishPhrase(a, text: "one.")
        h.view.dropPhrase(b)
        h.view.finishPhrase(c, text: "two")
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("One. Two")])))
        #expect(h.view.pendingPhraseIDs.isEmpty)
    }

    @Test func typingAtAPhrasesSpotGoesAfterIt() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [TextRun("Note:")])))
        let a = UUID()
        h.view.beginPhrase(a)
        h.type(" later")
        h.view.finishPhrase(a, text: "first")
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Note: first later")])))
    }

    @Test func aDeletedSpotMovesThePhraseToTheDeletionPoint() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [TextRun("abc def")])))
        h.view.setSelectedRange(NSRange(location: 5, length: 0))
        let a = UUID()
        h.view.beginPhrase(a)
        h.view.setSelectedRange(NSRange(location: 3, length: 4))
        h.view.deleteBackward(nil)
        h.view.finishPhrase(a, text: "x")
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("abc x")])))
    }

    @Test func dictatedPhrasesUndoLikeTyping() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [TextRun("Hi")])))
        let a = UUID()
        h.view.beginPhrase(a)
        h.grouped { h.view.finishPhrase(a, text: "there") }
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Hi there")])))
        h.undo.undo()
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Hi")])))
    }

    // MARK: Object lines

    @Test func markersAndAttachmentsGoOnTheirOwnLines() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [TextRun("Agenda")]), .paragraph(runs: [TextRun("Next")])))
        h.view.setSelectedRange(NSRange(location: 2, length: 0))
        h.view.insertMarker(time: 75)
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Agenda")]), .marker(time: 75),
                                       .paragraph(runs: [TextRun("Next")])))
        h.view.setSelectedRange(NSRange(location: (h.view.string as NSString).length, length: 0))
        h.view.insertAttachment(.image(attachmentID: "X.png"))
        #expect(h.view.document.blocks.last == .image(attachmentID: "X.png"))
    }

    // MARK: Autosave

    @Test func autosaveAfterAPauseAndOnSwitchingNotes() async throws {
        let store = NoteStore(container: try AppModelContainer.inMemory(), attachments: .temporary(),
                              speech: FakeSpeechService())
        let first = try store.createNote()
        let second = try store.createNote()
        let controller = NoteEditorController(store: store, autosaveDelay: .milliseconds(30))
        let h = Harness()
        controller.attach(h.view)
        controller.open(first)
        h.type("Groceries\nOat milk")
        #expect(controller.saveState == .editing)
        #expect(await eventually { controller.saveState == .saved })
        #expect(first.title == "Groceries")
        #expect(store.document(of: first).previewLine == "Oat milk")

        // Switching notes saves at once, without waiting.
        controller.open(second)
        h.type("Second")
        controller.open(first)
        #expect(second.title == "Second")
        #expect(controller.wordCount == 3)

        // A dictated phrase marks the note.
        let phrase = UUID()
        controller.beginPhrase(phrase)
        controller.finishPhrase(phrase, text: "and eggs")
        #expect(first.hasDictation)
    }

    // MARK: Drawing

    @Test func theEditorDrawsListsChecklistsAndChips() throws {
        let h = Harness()
        h.window.appearance = NSAppearance(named: .aqua)
        h.view.dateLine = "19 November 2026 at 11:40"
        h.view.load(Self.everyBlock)
        h.view.beginPhrase(UUID())
        h.view.layoutSubtreeIfNeeded()
        let rep = try #require(h.view.bitmapImageRepForCachingDisplay(in: h.view.bounds))
        h.view.cacheDisplay(in: h.view.bounds, to: rep)
        #expect(rep.pixelsWide > 0)
        if let dir = ProcessInfo.processInfo.environment["MFERENCE_EDITOR_PROBE_DIR"] {
            try rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: dir).appending(path: "notes-editor-probe.png"))
        }
    }
}
