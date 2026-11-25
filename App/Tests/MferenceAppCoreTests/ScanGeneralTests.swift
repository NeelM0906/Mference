import AppKit
import CoreGraphics
import CoreText
import Foundation
@testable import MferenceAppCore
import SwiftData
import Testing

/// Returns one fixed reading for every file.
struct ReadingScanner: DocumentRecognizing {
    var reading: ScanReading

    init(_ reading: ScanReading) { self.reading = reading }

    init(lines: [String], tables: [ScanTable] = []) {
        reading = ScanReading(lines: lines.map { ScanLine(text: $0, confidence: 0.95, box: .zero) }, tables: tables)
    }

    func recognize(imageAt url: URL) async throws -> ScanReading { reading }
    func recognize(image: CGImage) async throws -> ScanReading { reading }
}

/// General scanning: any text, its kind told from the text where it is
/// clear, Apple's model only where it is not, and each kind's next step.
@Suite struct ScanGeneralTests {
    static let card = ["Priya Raman", "Design Director", "Northwind Studio", "+1 (415) 555-0134",
                       "priya@northwind.example"]
    static let flyer = ["Autumn Jazz Night", "Saturday, December 5, 2026 at 7:00 PM", "Riverside Hall"]
    static let letter = ["Dear Sam,", "Thank you for the lovely dinner last week.", "Best, Priya"]

    @MainActor static func session(_ scanner: any DocumentRecognizing, apple: ScriptedAssistantTextModel,
                                   quick: ScriptedAssistantTextModel? = nil, container: ModelContainer,
                                   onEvent: @escaping @MainActor (ProposedEvent) -> Void = { _ in },
                                   openFile: @escaping @MainActor (URL) -> Void = { _ in }) -> ScanSession {
        ScanSession(scanner: scanner, assistant: AppleModelService(primary: apple, fallback: quick),
                    context: container.mainContext, imageDirectory: URL.temporaryDirectory,
                    onEventSuggested: onEvent, openFile: openFile)
    }

    static func apple(_ answers: String...) -> ScriptedAssistantTextModel {
        ScriptedAssistantTextModel(kind: .appleOnDevice, answers: answers.map { .success($0) })
    }

    // MARK: Telling the kind

    /// Clear cases need no model: a printed total, a table, a short card
    /// with a phone or email, a short text with a date and a time.
    @Test @MainActor func clearCasesAreToldFromTheTextAlone() async throws {
        let container = try AppModelContainer.inMemory()
        let cases: [(ReadingScanner, ScanKind)] = [
            (ReadingScanner(lines: Self.card), .contact),
            (ReadingScanner(lines: Self.flyer), .event),
            (ReadingScanner(lines: ["Item   Qty", "Pens   4", "Paper   2"],
                            tables: [ScanTable(rows: [["Item", "Qty"], ["Pens", "4"], ["Paper", "2"]])]), .table),
        ]
        for (scanner, kind) in cases {
            let apple = Self.apple("{}")
            let session = Self.session(scanner, apple: apple, container: container)
            session.open(URL(filePath: "/fixtures/scan.png"))
            await session.waitForReading()
            #expect(session.phase == .review)
            #expect(session.kind == kind, "\(scanner.reading.lines.first?.text ?? "")")
            #expect(apple.requests.isEmpty, "\(kind): no model call")
        }

        // A printed total stays a receipt, filled as receipts always were.
        let apple = Self.apple(#"{"merchant": "Corner Hardware", "total": "44.49"}"#)
        let receipt = Self.session(ReadingScanner(lines: ["CORNER HARDWARE", "TOTAL 44.49"]), apple: apple,
                                   container: container)
        receipt.open(URL(filePath: "/fixtures/receipt.png"))
        await receipt.waitForReading()
        #expect(receipt.kind == .receipt)
        #expect(receipt.title == "Receipt from Corner Hardware")
        #expect(apple.requests.count == 1)
    }

    @Test @MainActor func aCardsDetailsComeFromItsLines() async throws {
        let container = try AppModelContainer.inMemory()
        let session = Self.session(ReadingScanner(lines: Self.card), apple: Self.apple("{}"), container: container)
        session.open(URL(filePath: "/fixtures/card.png"))
        await session.waitForReading()
        #expect(session.contact.name == "Priya Raman")
        #expect(session.contact.jobTitle == "Design Director")
        #expect(session.contact.organization == "Northwind Studio")
        #expect(session.contact.emails == ["priya@northwind.example"])
        #expect(session.contact.phones.count == 1)
        #expect(session.title == "Priya Raman")
    }

    @Test @MainActor func aFlyersEventIsReadFromItsDateAndTime() async throws {
        let container = try AppModelContainer.inMemory()
        var proposed: [ProposedEvent] = []
        let session = Self.session(ReadingScanner(lines: Self.flyer), apple: Self.apple("{}"), container: container,
                                   onEvent: { proposed.append($0) })
        session.open(URL(filePath: "/fixtures/flyer.png"))
        await session.waitForReading()
        let event = try #require(session.event)
        #expect(event.title == "Autumn Jazz Night")
        let parts = Calendar.current.dateComponents([.year, .month, .day, .hour], from: event.start)
        #expect(parts.year == 2026 && parts.month == 12 && parts.day == 5 && parts.hour == 19)
        #expect(event.end == event.start.addingTimeInterval(3_600))
        // Handed to Calendar's proposals, from Scan; nothing is added here.
        session.addEventToCalendar()
        #expect(proposed.map(\.title) == ["Autumn Jazz Night"])
        #expect(proposed.first?.source.kind == .scan)
        #expect(session.notice?.contains("Calendar") == true)
    }

    /// Unclear text asks Apple's model, which may name any kind; an answer
    /// outside the list leaves the text's own guess.
    @Test @MainActor func unclearTextAsksApplesModel() async throws {
        let container = try AppModelContainer.inMemory()
        let apple = Self.apple(#"{"kind": "Document"}"#)
        let letter = Self.session(ReadingScanner(lines: Self.letter), apple: apple, container: container)
        letter.open(URL(filePath: "/fixtures/letter.png"))
        await letter.waitForReading()
        #expect(letter.kind == .document)
        #expect(apple.requests.count == 1)
        #expect(letter.title == "Dear Sam,")

        let odd = Self.session(ReadingScanner(lines: Self.letter), apple: Self.apple(#"{"kind": "poster"}"#),
                               container: container)
        odd.open(URL(filePath: "/fixtures/letter.png"))
        await odd.waitForReading()
        #expect(odd.kind == .document)
    }

    /// While reading, only Apple's model runs: with it off, Quick is never
    /// loaded (it would swap out the chat model); the text decides alone.
    @Test @MainActor func readingNeverFallsBackToQuick() async throws {
        let container = try AppModelContainer.inMemory()
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: .unavailable(reason: "off"),
                                               answer: "unused")
        let quick = ScriptedAssistantTextModel(kind: .quick, answer: #"{"kind": "event", "name": "Q"}"#)
        for lines in [Self.letter, ["+1 (415) 555-0134", "hello@example.com"]] {
            let session = Self.session(ReadingScanner(lines: lines), apple: apple, quick: quick, container: container)
            session.open(URL(filePath: "/fixtures/scan.png"))
            await session.waitForReading()
            #expect(session.phase == .review)
        }
        #expect(quick.requests.isEmpty)
    }

    @Test @MainActor func readAsShowsTheScanAsAnotherKind() async throws {
        let container = try AppModelContainer.inMemory()
        let apple = Self.apple(#"{"kind": "document"}"#, #"{"merchant": "Northside Dental", "total": "120.00"}"#)
        let session = Self.session(ReadingScanner(lines: Self.letter), apple: apple, container: container)
        session.open(URL(filePath: "/fixtures/letter.png"))
        await session.waitForReading()
        #expect(session.kind == .document)
        session.read(as: .receipt)
        await session.waitForReading()
        #expect(session.kind == .receipt && !session.isPreparing)
        #expect(apple.requests.count == 2)  // the receipt model filled the fields
        #expect(!session.fields.fields.isEmpty)
        session.read(as: .table)
        await session.waitForReading()
        #expect(session.kind == .table && session.fields.fields.isEmpty)
    }

    /// Only receipts are saved: other scans have no place to be seen again.
    @Test @MainActor func onlyReceiptsAreSaved() async throws {
        let container = try AppModelContainer.inMemory()
        let session = Self.session(ReadingScanner(lines: Self.letter), apple: Self.apple(#"{"kind": "document"}"#),
                                   container: container)
        session.open(URL(filePath: "/fixtures/letter.png"))
        await session.waitForReading()
        #expect(throws: ScanSessionError.nothingToConfirm) { try session.confirm() }
        #expect(try container.mainContext.fetchCount(FetchDescriptor<ScanRecord>()) == 0)
        #expect(try session.exportCSV() == ScanCSVExporter.csv([]))
    }

    @Test @MainActor func saveToNotesAndTextExportUseTheFullText() async throws {
        let container = try AppModelContainer.inMemory()
        let session = Self.session(ReadingScanner(lines: Self.letter), apple: Self.apple(#"{"kind": "document"}"#),
                                   container: container)
        session.open(URL(filePath: "/fixtures/letter.png"))
        await session.waitForReading()
        #expect(!session.canSaveToNotes)  // until the app connects Notes
        var saved: [(String, String)] = []
        session.saveToNotes = { saved.append(($0, $1)) }
        session.saveTextToNotes()
        #expect(saved.map(\.0) == ["Dear Sam,"])
        #expect(saved.first?.1 == Self.letter.joined(separator: "\n"))
        #expect(session.notice == "Saved to Notes as “Dear Sam,”.")
        #expect(String(data: session.textFile, encoding: .utf8) == Self.letter.joined(separator: "\n"))
        #expect(session.exportName == "Dear Sam,")
    }

    /// The app's Save to Notes makes a note titled by the scan, holding its
    /// text, in the one Notes store.
    @Test @MainActor func theAppSavesScansToNotes() async throws {
        let services = AppServices.preview()
        let scan = try #require(services.apple?.scan)
        try scan.saveToNotes?("Lease page 4", "13. ALTERATIONS\nNo painting without consent.")
        let notes = try services.notes.context.fetch(FetchDescriptor<Note>())
        let note = try #require(notes.first { $0.title == "Lease page 4" })
        let body = NoteDocument.decode(note.bodyJSON).plainText
        #expect(body.contains("13. ALTERATIONS") && body.contains("No painting without consent."))
    }

    // MARK: The model jobs

    @Test func theKindAnswerMustNameAKnownKind() async throws {
        let service = AppleModelService(primary: Self.apple(#"Sure! {"kind": "Contact"}"#), fallback: nil)
        #expect(try await service.classifyScan("Priya Raman", allowFallback: false).value == .contact)
        let unknown = AppleModelService(primary: Self.apple(#"{"kind": "poster"}"#), fallback: nil)
        await #expect(throws: AssistantModelError.unreadableAnswer) {
            _ = try await unknown.classifyScan("Jazz night", allowFallback: false)
        }
    }

    @Test func contactFieldsLeaveBlanksForMissingValues() async throws {
        let apple = Self.apple(#"{"name": "Priya Raman", "organization": null, "title": "Design Director"}"#)
        let service = AppleModelService(primary: apple, fallback: nil)
        let draft = try await service.fillContact(from: "PRIYA RAMAN", allowFallback: false).value
        #expect(draft.name == "Priya Raman" && draft.organization.isEmpty && draft.jobTitle == "Design Director")
        // The card's text goes in fenced as untrusted data.
        #expect(apple.requests.first?.prompt.contains("<untrusted>") == true)
    }

    // MARK: Contact card

    @Test @MainActor func theCardOpensInContactsAndGoesWithTheNextScan() async throws {
        let container = try AppModelContainer.inMemory()
        var opened: [URL] = []
        let session = Self.session(ReadingScanner(lines: Self.card), apple: Self.apple("{}"), container: container,
                                   openFile: { opened.append($0) })
        session.open(URL(filePath: "/fixtures/card.png"))
        await session.waitForReading()
        session.addContact()
        let url = try #require(opened.first)
        #expect(url.pathExtension == "vcf")
        let card = try String(contentsOf: url, encoding: .utf8)
        #expect(card.contains("FN:Priya Raman\r\n"))
        #expect(card.contains("N:Raman;Priya;;;\r\n"))
        #expect(card.contains("ORG:Northwind Studio\r\n"))
        #expect(card.contains("EMAIL;TYPE=INTERNET:priya@northwind.example\r\n"))
        #expect(session.notice?.contains("Contacts") == true)
        session.reset()
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test func cardTextCannotAddFieldsOfItsOwn() {
        let card = ContactDraft(name: "Sam; Lee", organization: "A, B\nTEL:999", jobTitle: "Lead\\Dev")
        let text = card.vCard
        #expect(text.contains("ORG:A\\, B\\nTEL:999\r\n"))
        #expect(text.contains("TITLE:Lead\\\\Dev\r\n"))
        #expect(!text.contains("\r\nTEL:999"))
        #expect(text.hasPrefix("BEGIN:VCARD\r\nVERSION:3.0\r\n") && text.hasSuffix("END:VCARD\r\n"))
    }

    @Test func contactWordsMatchWholeWords() {
        #expect(ContactDraft.containsWord("inc.", in: "Northwind Inc."))
        #expect(ContactDraft.containsWord("studio", in: "Northwind Studio"))
        #expect(!ContactDraft.containsWord("ag", in: "Agatha Manager"))
        #expect(!ContactDraft.containsWord("lead", in: "Leaders Forum"))
    }

    // MARK: Tables

    @Test func tablesExportAsCSVWithFormulasDefused() {
        let table = ScanTable(rows: [["Item", "Total"], ["a,b", "=SUM(A1)"], ["c"]])
        #expect(ScanTableExporter.csv(table) == "Item,Total\r\n\"a,b\",'=SUM(A1)\r\nc,\r\n")
        #expect(ScanTableExporter.fileData(table).starts(with: [0xEF, 0xBB, 0xBF]))
    }

    // MARK: Pages

    /// Lines on different pages at the same height are separate rows, and
    /// highlights stay on the page that shows.
    @Test func pagesKeepTheirOwnRowsAndHighlights() {
        func line(_ text: String, x: CGFloat, page: Int) -> ScanLine {
            ScanLine(text: text, confidence: 0.9, box: CGRect(x: x, y: 0.1, width: 0.2, height: 0.03), page: page)
        }
        let reading = ScanReading(lines: [line("Page one", x: 0.1, page: 0), line("TOTAL", x: 0.1, page: 1),
                                          line("44.49", x: 0.7, page: 1)],
                                  pageCount: 2, pageAspects: [0.77, 0.77])
        #expect(reading.rows == ["Page one", "TOTAL   44.49"])
        var fields = ScanFields(fields: [])
        fields[.total] = ScanField(key: .total, value: "44.49", confidence: .sure)
        #expect(ScanHighlights.boxes(fields: fields, reading: reading, page: 0).isEmpty)
        #expect(ScanHighlights.boxes(fields: fields, reading: reading, page: 1)[.total] != nil)
    }

    @Test func aLongPDFIsReadUpToTheCap() throws {
        let url = try Self.pdf(pages: (1...25).map { "Page \($0)" })
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        #expect(VisionScanner.pageCount(at: url) == VisionScanner.maximumPages)
        #expect(VisionScanner.pageCount(at: URL(filePath: "/x/photo.png")) == 1)
        #expect(VisionScanner.loadImage(at: url, page: 24) != nil)
    }

    /// Apple Vision reads every page of a PDF on this Mac (no model).
    @Test func visionReadsEveryPageOfAPDF() async throws {
        let url = try Self.pdf(pages: ["QUARTERLY REPORT FIRST PAGE", "APPENDIX SECOND PAGE"])
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let reading = try await VisionScanner().recognize(imageAt: url)
        #expect(reading.pageCount == 2)
        #expect(Set(reading.lines.map(\.page)) == [0, 1])
        #expect(reading.fullText.localizedCaseInsensitiveContains("first page"))
        #expect(reading.fullText.localizedCaseInsensitiveContains("second page"))
        #expect(reading.lines.first { $0.text.localizedCaseInsensitiveContains("appendix") }?.page == 1)
    }

    static func pdf(pages: [String]) throws -> URL {
        let folder = URL.temporaryDirectory.appending(path: "scan-pdf-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appending(path: "doc.pdf")
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let context = try #require(CGContext(url as CFURL, mediaBox: &box, nil))
        for text in pages {
            context.beginPDFPage(nil)
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: text,
                attributes: [.font: NSFont.systemFont(ofSize: 28, weight: .semibold)]))
            context.textPosition = CGPoint(x: 48, y: 680)
            CTLineDraw(line, context)
            context.endPDFPage()
        }
        context.closePDF()
        return url
    }
}
