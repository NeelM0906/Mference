import Foundation
@testable import MferenceAppCore
import Testing

/// Export (Markdown with attachments, plain text) and ⌘K search hits.
@MainActor
@Suite struct NoteExportTests {
    let doc = NoteDocument(blocks: [
        .heading(level: .title, runs: [TextRun("Lisbon trip")]),
        .paragraph(runs: [TextRun("Hotel in "), TextRun("Alfama", bold: true), TextRun(", "),
                          TextRun("maybe", italic: true), TextRun(" "), TextRun("old", strikethrough: true),
                          TextRun(" "), TextRun("site", link: URL(string: "https://example.com")),
                          TextRun(" "), TextRun("key", underline: true), TextRun(" 2*3")]),
        .heading(level: .heading, runs: [TextRun("To do")]),
        .check(runs: [TextRun("Book flights")], done: true, indent: 0),
        .check(runs: [TextRun("Hotel")], done: false, indent: 1),
        .numbered(runs: [TextRun("One")], indent: 0),
        .numbered(runs: [TextRun("Two")], indent: 0),
        .bullet(runs: [TextRun("Sintra")], indent: 0),
        .image(attachmentID: "photo.png"),
        .marker(time: 754),
    ])

    @Test func markdownKeepsEveryFormat() {
        let md = NoteExport.markdown(doc, attachmentsFolder: "Lisbon trip attachments")
        #expect(md == """
        # Lisbon trip

        Hotel in **Alfama**, *maybe* ~~old~~ [site](https://example.com) <u>key</u> 2\\*3

        ## To do

        - [x] Book flights
          - [ ] Hotel
        1. One
        2. Two
        - Sintra

        ![](<Lisbon trip attachments/photo.png>)

        > Marked at 12:34

        """)
    }

    @Test func plainTextUsesBoxesAndBullets() {
        #expect(NoteExport.plainText(doc) == """
        Lisbon trip
        Hotel in Alfama, maybe old site key 2*3
        To do
        ☑ Book flights
          ☐ Hotel
        1. One
        2. Two
        • Sintra
        Marked at 12:34

        """)
    }

    @Test func markdownFileWritesItsAttachmentsBesideIt() throws {
        let attachments = NoteAttachments.temporary()
        let noteID = UUID()
        let id = try attachments.write(Data([1, 2, 3]), fileExtension: "png", into: noteID)
        let document = NoteDocument(blocks: [.heading(level: .title, runs: [TextRun("Trip")]),
                                             .image(attachmentID: id), .audio(attachmentID: "gone.m4a")])
        let folder = FileManager.default.temporaryDirectory.appending(path: "mference-export-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let file = folder.appending(path: "Trip.md")
        let written = try NoteExport.writeMarkdown(document, noteID: noteID, attachments: attachments, to: file)
        #expect(written.count == 2)
        let text = try String(contentsOf: file, encoding: .utf8)
        #expect(text == "# Trip\n\n![](<Trip attachments/\(id)>)\n")
        #expect(try Data(contentsOf: folder.appending(path: "Trip attachments/\(id)")) == Data([1, 2, 3]))
    }

    @Test func contentFromANewerVersionExportsItsText() {
        let document = NoteDocument.decode(Data(#"{"v":1,"blocks":[{"type":"callout","runs":[{"t":"Keep *this*"}]},{"type":"video"}]}"#.utf8))
        #expect(NoteExport.markdown(document) == "Keep \\*this\\*\n")
        #expect(NoteExport.plainText(document) == "Keep *this*\n")
    }

    @Test func fileNamesAreSafe() {
        #expect(NoteExport.fileName(for: "Q3: plan / budget?") == "Q3- plan - budget-")
        #expect(NoteExport.fileName(for: "   ") == "Note")
    }

    @Test func searchHitsRankTitleThenBodyThenTranscript() throws {
        let store = NoteStore(container: try AppModelContainer.inMemory(), attachments: .temporary(),
                              speech: FakeSpeechService())
        let titled = try store.createNote(title: "Budget review")
        let body = try store.createNote()
        try store.save(NoteDocument(blocks: [.heading(level: .title, runs: [TextRun("Planning")]),
                                             .paragraph(runs: [TextRun("Intro")]),
                                             .paragraph(runs: [TextRun("The budget for Q3")])]), to: body)
        let meeting = try store.createNote(kind: .meeting, title: "Weekly sync")
        let recording = NoteRecording()
        store.context.insert(recording)
        meeting.recording = recording
        let segment = TranscriptSegment(track: .call, start: 125, end: 130, speakerID: "s2",
                                        text: "Let's revisit the budget next week", isFinal: true)
        recording.transcriptJSON = try JSONEncoder().encode([segment])
        try store.transcriptDidChange(in: meeting)
        let gone = try store.createNote(title: "Budget draft")
        try store.moveToTrash(gone)

        let hits = try store.searchHits("budget")
        #expect(hits.map(\.noteID) == [titled.id, body.id, meeting.id])
        #expect(hits[1].match == .body)
        #expect(hits[1].snippet == "The budget for Q3")
        #expect(hits[2].match == .transcript(segmentID: segment.id, time: 125))
        #expect(hits[2].snippet == "Let's revisit the budget next week")
        #expect(try store.searchHits("  ").isEmpty)
    }
}
