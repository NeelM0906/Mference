import Foundation
@testable import MferenceAppCore
import Testing

/// NoteDocument (spec 8.2), list grouping, PhraseJoiner and search helpers.
@Suite struct NoteDocumentTests {
    @Test func attachmentIDsStayInsideTheNoteFolder() {
        let attachments = NoteAttachments(root: URL(fileURLWithPath: "/tmp/notes"))
        let note = UUID()
        for bad in ["../../Mference.store", "..", ".hidden", "a/b", ""] {
            #expect(!NoteAttachments.isValidID(bad))
            #expect(attachments.url(of: bad, in: note).lastPathComponent == "invalid-attachment-id")
        }
        #expect(NoteAttachments.isValidID("6F9619FF.png"))
    }

    static let everyBlock = NoteDocument(blocks: [
        .heading(level: .title, runs: [TextRun("Lisbon trip, March")]),
        .heading(level: .heading, runs: [TextRun("To do")]),
        .heading(level: .subheading, runs: [TextRun("Before")]),
        .paragraph(runs: [TextRun("Plain "), TextRun("bold", bold: true), TextRun(" "),
                          TextRun("italic", italic: true), TextRun("under", underline: true),
                          TextRun("struck", strikethrough: true),
                          TextRun("all", bold: true, italic: true, underline: true, strikethrough: true),
                          TextRun("link", link: URL(string: "https://example.com/a?b=c"))]),
        .bullet(runs: [TextRun("Sintra")], indent: 0),
        .bullet(runs: [TextRun("Pena palace")], indent: 1),
        .numbered(runs: [TextRun("First")], indent: 0),
        .check(runs: [TextRun("Book flights")], done: true, indent: 0),
        .check(runs: [TextRun("Hotel")], done: false, indent: 2),
        .image(attachmentID: "A.png"),
        .audio(attachmentID: "B.m4a"),
        .transcriptAnchor(segmentID: UUID(uuidString: "6F9619FF-8B86-D011-B42D-00C04FC964FF")!),
        .marker(time: 754.5),
        .paragraph(runs: []),
    ])

    @Test func everyBlockAndRunStyleRoundTripsThroughJSON() throws {
        let data = try Self.everyBlock.encoded()
        #expect(NoteDocument.decode(data) == Self.everyBlock)
        let json = try #require(String(data: data, encoding: .utf8))
        #expect(json.contains(#""v":1"#))
    }

    @Test func unreadableOrMissingBodiesOpenAsEmptyNotes() {
        #expect(NoteDocument.decode(nil) == .empty)
        // Unreadable: shown empty, but read-only so nothing saves over it.
        let damaged = NoteDocument.decode(Data("not json".utf8))
        #expect(damaged.blocks.isEmpty)
        #expect(damaged.readOnlyReason == .damaged)
    }

    @Test func aBlockThisBuildDoesNotKnowIsKeptExactly() throws {
        let json = #"{"blocks":[{"runs":[{"t":"cell"}],"type":"table","x":[1,true,null]},{"runs":[{"t":"ok"}],"type":"paragraph"}],"v":1}"#
        let document = NoteDocument.decode(Data(json.utf8))
        #expect(document.readOnlyReason == nil)
        #expect(document.blocks.count == 2)
        #expect(document.blocks[0].text == "cell")
        #expect(document.blocks[1] == .paragraph(runs: [TextRun("ok")]))
        #expect(String(data: try document.encoded(), encoding: .utf8) == json)
        // Its text still counts for search and the preview.
        #expect(document.plainText == "cell\nok")
    }

    @Test func newerVersionsAndDamagedBodiesAreReadOnly() {
        let newer = NoteDocument.decode(Data(#"{"v":2,"blocks":[]}"#.utf8))
        #expect(newer.readOnlyReason == .newerVersion)
        #expect(NoteDocument.decode(Data("[1,2".utf8)).readOnlyReason == .damaged)
        #expect(NoteDocument.decode(Data(#"{"blocks":[]}"#.utf8)).readOnlyReason == nil)
    }

    @Test func derivedText() {
        let document = Self.everyBlock
        #expect(document.title == "Lisbon trip, March")
        #expect(document.previewLine == "To do")
        #expect(document.attachmentIDs == ["A.png", "B.m4a"])
        #expect(NoteDocument(blocks: [.paragraph(runs: [TextRun("Four words in here")])]).wordCount == 4)
        #expect(NoteDocument(blocks: [.paragraph(runs: [TextRun("Oat milk")])]).title == "")
    }

    @Test func runsMergeWhenTheirStylesMatch() {
        #expect(normalizedRuns([TextRun("a"), TextRun(""), TextRun("b"), TextRun("c", bold: true)])
                == [TextRun("ab"), TextRun("c", bold: true)])
    }

    // MARK: Search helpers

    @Test func foldingIgnoresCaseAndDiacritics() {
        #expect(NoteSearch.terms("  Café  SÃO ") == ["cafe", "sao"])
        #expect(NoteSearch.matches(unfolded: "São Bento café", terms: ["cafe", "sao"]))
        #expect(!NoteSearch.matches(unfolded: "São Bento", terms: ["cafe", "sao"]))
    }

    @Test func transcriptHitsComeInTimeOrder() {
        let segments = [
            TranscriptSegment(track: .call, start: 40, end: 42, speakerID: "s2", text: "the budget is fine", isFinal: true),
            TranscriptSegment(track: .microphone, start: 10, end: 12, speakerID: "you", text: "Budget first", isFinal: true),
            TranscriptSegment(track: .microphone, start: 20, end: 22, speakerID: "you", text: "other", isFinal: true),
        ]
        #expect(NoteSearch.transcriptHits(segments, terms: ["budget"]).map(\.start) == [10, 40])
    }
}

/// The list's groups (spec 4.1).
@Suite struct NoteListGroupingTests {
    var calendar: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "Europe/Lisbon")!
        return calendar
    }()
    let locale = Locale(identifier: "en_GB")

    func date(_ y: Int, _ m: Int, _ d: Int, _ h: Int = 12, _ min: Int = 0) -> Date {
        calendar.date(from: DateComponents(year: y, month: m, day: d, hour: h, minute: min))!
    }

    @Test func groupsByDay() {
        let now = date(2026, 11, 19, 11, 40)
        #expect(NoteListGrouping.group(for: date(2026, 11, 19, 0, 5), isPinned: false, now: now, calendar: calendar) == .today)
        #expect(NoteListGrouping.group(for: date(2026, 11, 18, 23, 59), isPinned: false, now: now, calendar: calendar) == .yesterday)
        #expect(NoteListGrouping.group(for: date(2026, 11, 17), isPinned: false, now: now, calendar: calendar) == .previous7Days)
        #expect(NoteListGrouping.group(for: date(2026, 11, 12), isPinned: false, now: now, calendar: calendar) == .previous7Days)
        #expect(NoteListGrouping.group(for: date(2026, 11, 11), isPinned: false, now: now, calendar: calendar) == .previous30Days)
        #expect(NoteListGrouping.group(for: date(2026, 10, 20), isPinned: false, now: now, calendar: calendar) == .previous30Days)
        #expect(NoteListGrouping.group(for: date(2026, 10, 19), isPinned: false, now: now, calendar: calendar) == .month(year: 2026, month: 10))
        #expect(NoteListGrouping.group(for: date(2026, 11, 19), isPinned: true, now: now, calendar: calendar) == .pinned)
    }

    @Test func sectionsPutPinnedFirstThenNewestFirst() {
        struct Row { var name: String; var at: Date; var pinned = false }
        let now = date(2026, 11, 19, 11, 40)
        let rows = [
            Row(name: "old", at: date(2025, 3, 2)),
            Row(name: "groceries", at: date(2026, 11, 15), pinned: true),
            Row(name: "review", at: date(2026, 11, 19, 10, 2)),
            Row(name: "lisbon", at: date(2026, 11, 19, 11, 40)),
            Row(name: "interview", at: date(2026, 11, 18, 16, 15)),
            Row(name: "september", at: date(2026, 9, 1)),
        ]
        let sections = NoteListGrouping.sections(rows, now: now, calendar: calendar, locale: locale,
                                                 updatedAt: \.at, isPinned: \.pinned)
        #expect(sections.map(\.title) == ["Pinned", "Today", "Yesterday", "September", "March 2025"])
        #expect(sections.map { $0.items.map(\.name) } == [["groceries"], ["lisbon", "review"], ["interview"],
                                                          ["september"], ["old"]])
    }

    @Test func rowTimeLabels() {
        let now = date(2026, 11, 19, 11, 40)
        #expect(NoteListGrouping.timeLabel(for: date(2026, 11, 19, 11, 40), now: now, calendar: calendar, locale: locale) == "11:40")
        #expect(NoteListGrouping.timeLabel(for: date(2026, 11, 18, 16, 15), now: now, calendar: calendar, locale: locale) == "16:15")
        #expect(NoteListGrouping.timeLabel(for: date(2026, 11, 16), now: now, calendar: calendar, locale: locale) == "Mon")
        // Seven days ago is still in "Previous 7 days": a weekday too.
        #expect(NoteListGrouping.timeLabel(for: date(2026, 11, 12), now: now, calendar: calendar, locale: locale) == "Thu")
        #expect(NoteListGrouping.timeLabel(for: date(2026, 10, 2), now: now, calendar: calendar, locale: locale) == "02/10/2026")
    }
}

/// Joining dictated phrases into the text around them (spec 4.2).
@Suite struct PhraseJoinerTests {
    func join(_ phrase: String, _ before: String, _ after: String = "") -> String {
        PhraseJoiner.text(for: phrase, preceding: before, following: after)
    }

    @Test func oneSpaceBetweenPhrasesNeverTwo() {
        #expect(join("then the hotel", "Book flights") == " then the hotel")
        #expect(join("then the hotel", "Book flights ") == "then the hotel")
        #expect(join("  then   the hotel ", "Book flights") == " then the hotel")
    }

    @Test func noLeadingSpaceAtALineStart() {
        #expect(join("hotel in Alfama", "") == "Hotel in Alfama")
        #expect(join("hotel in Alfama", "To do\n") == "Hotel in Alfama")
        #expect(join("hotel", "To do\n  ") == "Hotel")
    }

    @Test func capitalAfterASentenceEnd() {
        #expect(join("next we go", "Flights booked.") == " Next we go")
        #expect(join("next we go", "Really? ") == "Next we go")
        #expect(join("next we go", "and so") == " next we go")
    }

    @Test func spaceBeforeFollowingWordsNotBeforePunctuation() {
        #expect(join("four nights", "Hotel,", "in Alfama") == " four nights ")
        #expect(join("four nights", "Hotel ", ".") == "four nights")
        #expect(join("four nights", "Hotel ", " in") == "four nights")
        #expect(join("four nights", "(", ")") == "four nights")
    }

    @Test func aPhraseWithNoWordsInsertsNothing() {
        #expect(join("   ", "Hello") == "")
    }

    @Test func aSentenceOpenerContinuingASentenceGoesLowercase() {
        #expect(join("And whether the appointment moved", "Pain in the lower right") == " and whether the appointment moved")
        #expect(join("The second one.", "Ask about") == " the second one.")
        // Names, "I" and sentence starts keep their capitals.
        #expect(join("Marta said so", "Ask whether") == " Marta said so")
        #expect(join("I think so", "and") == " I think so")
        #expect(join("And then", "Done.") == " And then")
        #expect(join("And then", "") == "And then")
    }

    @Test func capitalsOnlyForAPlainLowercaseFirstWord() {
        #expect(join("3 apples", "") == "3 apples")
        #expect(join("iPhone case", "") == "iPhone case")
        #expect(join("then go", #"He said "stop.""#) == " Then go")
        #expect(join("then go", "(See above.)") == " Then go")
    }
}
