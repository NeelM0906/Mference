import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import SwiftData
import Testing

/// Editor data-safety cases (review of #220): nothing typed, pasted or
/// dictated is lost or lands in the wrong note, and paste never keeps
/// formats a note cannot store or reaches the network.
@MainActor
@Suite(.serialized) struct NoteEditorSafetyTests {
    typealias Harness = NoteEditorTests.Harness

    func doc(_ blocks: NoteDocument.Block...) -> NoteDocument { NoteDocument(blocks: blocks) }

    func board(_ types: [NSPasteboard.PasteboardType], _ fill: (NSPasteboard) -> Void) -> NSPasteboard {
        let board = NSPasteboard(name: NSPasteboard.Name("mference-test-\(UUID().uuidString)"))
        board.declareTypes(types, owner: nil)
        fill(board)
        return board
    }

    func store() throws -> NoteStore {
        NoteStore(container: try AppModelContainer.inMemory(), attachments: .temporary(), speech: FakeSpeechService())
    }

    // MARK: Object lines (HIGH 1)

    @Test func textJoinedOntoAnImageLineIsKept() {
        let h = Harness()
        h.view.load(doc(.image(attachmentID: "A.png"), .paragraph(runs: [TextRun("Next")])))
        // Backspace at the start of "Next" joins it onto the image line.
        h.view.setSelectedRange(NSRange(location: 2, length: 0))
        h.view.deleteBackward(nil)
        #expect(h.view.document == doc(.image(attachmentID: "A.png"), .paragraph(runs: [TextRun("Next")])))
        // Typing right of a chip too.
        h.view.load(doc(.marker(time: 5)))
        h.view.setSelectedRange(NSRange(location: 1, length: 0))
        h.type("after")
        #expect(h.view.document == doc(.marker(time: 5), .paragraph(runs: [TextRun("after")])))
    }

    @Test func aBlockFromANewerBuildRoundTripsThroughTheEditor() throws {
        let data = Data(#"{"v":1,"blocks":[{"type":"video","attachment":"clip.mov"},{"type":"paragraph","runs":[{"t":"End"}]}]}"#.utf8)
        let document = NoteDocument.decode(data)
        let h = Harness()
        h.view.load(document)
        #expect(h.view.document == document)
    }

    // MARK: Note switches (HIGH 2, MEDIUM 4, 5)

    @Test func aPhraseFinishingAfterANoteSwitchLandsInItsOwnNote() throws {
        let store = try store()
        let first = try store.createNote()
        try store.save(doc(.heading(level: .title, runs: [TextRun("First")]), .paragraph(runs: [TextRun("Call")])), to: first)
        let second = try store.createNote()
        try store.save(doc(.heading(level: .title, runs: [TextRun("Second")])), to: second)
        let controller = NoteEditorController(store: store)
        let h = Harness()
        controller.attach(h.view)
        controller.open(first)
        let phrase = UUID()
        controller.beginPhrase(phrase)
        controller.open(second)
        controller.finishPhrase(phrase, text: "the dentist.")
        #expect(store.document(of: first).blocks.last == .paragraph(runs: [TextRun("Call the dentist.")]))
        #expect(first.hasDictation)
        #expect(!second.hasDictation)
        #expect(h.view.document == doc(.heading(level: .title, runs: [TextRun("Second")])))
    }

    @Test func editsAreSavedWhenTheAppDeactivatesOrTheViewGoesAway() throws {
        let store = try store()
        let note = try store.createNote()
        let controller = NoteEditorController(store: store, autosaveDelay: .seconds(60))
        let h = Harness()
        controller.attach(h.view)
        controller.open(note)
        h.type("Kept")
        NotificationCenter.default.post(name: NSApplication.didResignActiveNotification, object: nil)
        #expect(note.title == "Kept")

        h.type(" too")
        let scroll = NSScrollView()
        scroll.documentView = h.view
        NoteEditorView.dismantleNSView(scroll, coordinator: NoteEditorView.Coordinator(controller: controller))
        #expect(note.title == "Kept too")
    }

    @Test func reloadKeepsUnsavedTextAndPendingPhrases() throws {
        let store = try store()
        let note = try store.createNote()
        let controller = NoteEditorController(store: store, autosaveDelay: .seconds(60))
        let h = Harness()
        controller.attach(h.view)
        controller.open(note)
        h.type("Plan")
        let phrase = UUID()
        controller.beginPhrase(phrase)
        controller.reload()
        #expect(note.title == "Plan")
        #expect(h.view.pendingPhrases.first?.location == 4)
        controller.finishPhrase(phrase, text: "ahead")
        #expect(h.view.document == doc(.heading(level: .title, runs: [TextRun("Plan ahead")])))
    }

    @Test func aReadOnlyBodyIsShownButNotEditable() throws {
        let store = try store()
        let note = try store.createNote()
        note.bodyJSON = Data(#"{"v":3,"blocks":[]}"#.utf8)
        let controller = NoteEditorController(store: store)
        let h = Harness()
        controller.attach(h.view)
        controller.open(note)
        #expect(!h.view.isEditable)
        #expect(controller.readOnlyReason == .newerVersion)
    }

    // MARK: Paste (HIGH 3, MEDIUM 6, 7)

    @Test func htmlPasteKeepsTextAndStylesWithoutWebKit() {
        let html = #"<html><head><style>p{color:red}</style><script>alert(1)</script></head><body><p>Hello <b>bold</b> &amp; <i>it</i> <a href="https://example.com">link</a> <a href="javascript:x">bad</a></p><img src="https://tracker.example/p.gif"><ul><li>One</li><li>Two</li></ul><ol><li>First</li></ol></body></html>"#
        let pb = board([.html]) { $0.setString(html, forType: .html) }
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [])))
        #expect(h.view.readSelection(from: pb, type: .html))
        #expect(h.view.document == doc(
            .paragraph(runs: [TextRun("Hello "), TextRun("bold", bold: true), TextRun(" & "), TextRun("it", italic: true),
                              TextRun(" "), TextRun("link", link: URL(string: "https://example.com")), TextRun(" bad")]),
            .bullet(runs: [TextRun("One")], indent: 0),
            .bullet(runs: [TextRun("Two")], indent: 0),
            .numbered(runs: [TextRun("First")], indent: 0)))
        pb.releaseGlobally()
    }

    @Test func unsupportedPasteTypesAreRefused() {
        let archive = NSPasteboard.PasteboardType("Apple Web Archive pasteboard type")
        let pb = board([archive]) { $0.setData(Data([1, 2, 3]), forType: archive) }
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [TextRun("Same")])))
        #expect(!h.view.readSelection(from: pb, type: archive))
        #expect(!h.view.readablePasteboardTypes.contains(archive))
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("Same")])))
        pb.releaseGlobally()
    }

    @Test func anImageInPastedRTFDBecomesAnAttachment() throws {
        let dot = NSImage(size: NSSize(width: 4, height: 4), flipped: false) { rect in
            NSColor.red.setFill()
            rect.fill()
            return true
        }
        let tiff = try #require(dot.tiffRepresentation)
        let png = try #require(NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]))
        let wrapper = FileWrapper(regularFileWithContents: png)
        wrapper.preferredFilename = "dot.png"
        let rich = NSMutableAttributedString(string: "Before\n")
        rich.append(NSAttributedString(attachment: NSTextAttachment(fileWrapper: wrapper)))
        rich.append(NSAttributedString(string: "\nAfter"))
        let rtfd = try #require(rich.rtfd(from: NSRange(location: 0, length: rich.length), documentAttributes: [:]))
        let pb = board([.rtfd]) { $0.setData(rtfd, forType: .rtfd) }

        let store = try store()
        let note = try store.createNote()
        let controller = NoteEditorController(store: store)
        let h = Harness()
        controller.attach(h.view)
        controller.open(note)
        h.view.setSelectedRange(NSRange(location: (h.view.string as NSString).length, length: 0))
        h.view.insertNewline(nil)
        #expect(h.view.readSelection(from: pb, type: .rtfd))
        let blocks = h.view.document.blocks
        let image = try #require(blocks.first { if case .image = $0 { true } else { false } })
        #expect(blocks.contains(.paragraph(runs: [TextRun("Before")])))
        #expect(blocks.contains(.paragraph(runs: [TextRun("After")])))
        let id = try #require(image.attachmentID)
        #expect(FileManager.default.fileExists(atPath: store.attachments.url(of: id, in: note.id).path(percentEncoded: false)))
        pb.releaseGlobally()
    }

    @Test func lineSeparatorsBecomePlainLineBreaks() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [])))
        h.view.insertText("One\u{2028}Two\rThree", replacementRange: NSRange(location: NSNotFound, length: 0))
        h.view.insertLineBreak(nil)
        h.type("Four")
        let expected = doc(.paragraph(runs: [TextRun("One")]), .paragraph(runs: [TextRun("Two")]),
                           .paragraph(runs: [TextRun("Three")]), .paragraph(runs: [TextRun("Four")]))
        #expect(h.view.document == expected)
        h.view.load(h.view.document)
        #expect(h.view.document == expected)
        // A plain paste with separators, into a ticked item: new items are unticked.
        let pb = board([.string]) { $0.setString("A\u{2029}B", forType: .string) }
        h.view.load(doc(.check(runs: [TextRun("Done")], done: true, indent: 0)))
        h.view.setSelectedRange(NSRange(location: 4, length: 0))
        h.view.insertNewline(nil)
        h.view.toggleDone(lineAt: 5)
        #expect(h.view.readSelection(from: pb, type: .string))
        #expect(h.view.document.blocks.dropFirst().allSatisfy { if case .check(_, let done, _) = $0 { !done } else { false } })
        #expect(h.view.document.blocks.count == 3)
        pb.releaseGlobally()
    }

    @Test func blocksPastedIntoAnotherNoteBringTheirAttachments() throws {
        let store = try store()
        let source = try store.createNote()
        let id = try store.attachments.write(Data([7, 7]), fileExtension: "png", into: source.id)
        try store.save(doc(.paragraph(runs: [TextRun("Photo")]), .image(attachmentID: id)), to: source)
        let target = try store.createNote()
        let controller = NoteEditorController(store: store)
        let h = Harness()
        controller.attach(h.view)
        controller.open(source)
        h.view.selectAll(nil)
        let pb = board(h.view.writablePasteboardTypes) { _ in }
        #expect(h.view.writeSelection(to: pb, type: NoteTextView.blocksPasteboardType))
        controller.open(target)
        h.view.selectAll(nil)
        #expect(h.view.readSelection(from: pb, type: NoteTextView.blocksPasteboardType))
        let pasted = try #require(h.view.document.blocks.compactMap(\.attachmentID).first)
        #expect(pasted != id)
        #expect(try Data(contentsOf: store.attachments.url(of: pasted, in: target.id)) == Data([7, 7]))
        pb.releaseGlobally()

        // A crafted id that is a path is dropped.
        let evil = NoteBlocksPayload(noteID: nil, document: doc(.image(attachmentID: "../../Mference.store"),
                                                                .paragraph(runs: [TextRun("ok")])))
        let crafted = board([NoteTextView.blocksPasteboardType]) {
            $0.setData(try! JSONEncoder().encode(evil), forType: NoteTextView.blocksPasteboardType)
        }
        h.view.selectAll(nil)
        #expect(h.view.readSelection(from: crafted, type: NoteTextView.blocksPasteboardType))
        #expect(h.view.document.attachmentIDs.isEmpty)
        crafted.releaseGlobally()
    }

    @Test func onlyWebAndMailLinksOpen() {
        #expect(NoteEditorView.opensLink(URL(string: "https://example.com")!))
        #expect(NoteEditorView.opensLink(URL(string: "mailto:a@example.com")!))
        #expect(!NoteEditorView.opensLink(URL(string: "javascript:alert(1)")!))
        #expect(!NoteEditorView.opensLink(URL(string: "file:///etc/passwd")!))
    }

    // MARK: Keyboard and shortcuts (MEDIUM 8, 9)

    @Test func checklistItemsTickFromTheKeyboard() {
        let h = Harness()
        h.view.load(doc(.check(runs: [TextRun("A")], done: false, indent: 0), .check(runs: [TextRun("B")], done: false, indent: 0)))
        h.view.setSelectedRange(NSRange(location: 0, length: 3))
        h.view.toggleDoneForSelection()
        #expect(h.view.document == doc(.check(runs: [TextRun("A")], done: true, indent: 0),
                                       .check(runs: [TextRun("B")], done: true, indent: 0)))
        #expect(h.view.accessibilityCustomActions()?.contains { $0.name == "Uncheck item" } == true)
    }

    @Test func afterUndoingAShortcutALiteralDashSpaceCanBeTyped() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [])))
        h.grouped { h.type("-") }
        h.grouped { h.type(" ") }
        #expect(h.view.document == doc(.bullet(runs: [], indent: 0)))
        h.undo.undo()
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("-")])))
        // The space typed again stays a space: a literal "- ".
        h.grouped { h.type(" x") }
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("- x")])))
        #expect(h.view.textLayoutManager != nil)
    }

    @Test func phrasesFinishingOutOfOrderAtOneSpotKeepTheirOrder() {
        let h = Harness()
        h.view.load(doc(.paragraph(runs: [])))
        let (a, b) = (UUID(), UUID())
        h.view.beginPhrase(a)
        h.view.beginPhrase(b)
        h.view.finishPhrase(b, text: "second")
        h.view.finishPhrase(a, text: "first")
        // "first" was spoken first, so it lands first.
        #expect(h.view.document == doc(.paragraph(runs: [TextRun("First Second")])))
    }

    @Test func aDeletedEarlierItemRenumbersTheList() {
        let h = Harness()
        h.view.load(doc(.numbered(runs: [TextRun("A")], indent: 0), .numbered(runs: [TextRun("B")], indent: 0),
                        .numbered(runs: [TextRun("C")], indent: 0)))
        h.view.setSelectedRange(NSRange(location: 0, length: 2))
        h.view.deleteBackward(nil)
        let storage = h.view.textStorage!
        #expect(NoteTextView.ordinal(in: storage, lineStart: 2, style: NoteBlockStyle(.numbered)) == 2)
        #expect(h.view.document == doc(.numbered(runs: [TextRun("B")], indent: 0), .numbered(runs: [TextRun("C")], indent: 0)))
    }
}
