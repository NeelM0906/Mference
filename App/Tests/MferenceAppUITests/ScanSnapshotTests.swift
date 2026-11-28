import AppKit
import CoreText
import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftData
import SwiftUI
import Testing

@Suite(.serialized) @MainActor struct ScanSnapshotTests {
    /// Kept for the whole run: a screen rendered in the shell starts its own
    /// load (`.task`), which can outlive the test; SwiftData traps if its
    /// container is gone by then.
    static let container = try! AppModelContainer.inMemory()

    @Test func scan() async throws {
        let container = Self.container
        let model = ScriptedAssistantTextModel(kind: .appleOnDevice, answer: AppleUIFixtures.receiptAnswer)
        let receipt = try AppleUIFixtures.receipt()
        let session = ScanSession(scanner: AppleUIFixtures.FixedScanner(scanLines: receipt.lines),
                                  assistant: AppleModelService(primary: model, fallback: nil),
                                  context: container.mainContext,
                                  imageDirectory: URL.temporaryDirectory.appending(path: "mference-scan-snapshots"))
        session.open(receipt.url)
        await session.waitForReading()
        #expect(session.phase == .review)
        #expect(Set(session.highlights.keys) == [.merchant, .date, .tax, .total])
        for scheme in [ColorScheme.light, .dark] {
            try Snapshot.render(RootView(model: ShellModel(selection: .scan, defaults: nil), chrome: .inline)
                .environment(session), name: "07-scan", colorScheme: scheme)
        }

        let empty = ScanSession(scanner: AppleUIFixtures.FixedScanner(lines: []),
                                assistant: AppleModelService(primary: model, fallback: nil),
                                context: container.mainContext)
        try Snapshot.render(RootView(model: ShellModel(selection: .scan, defaults: nil), chrome: .inline)
                .environment(empty), name: "07-scan-empty")
    }

    /// The top bar holds Open file alone: Capture (Later) and the clear icon
    /// are gone, and closing a scan is the ✕ on the image (or Esc).
    @Test func scanBarIsOpenFileAlone() {
        let model = ScriptedAssistantTextModel(kind: .appleOnDevice, answer: AppleUIFixtures.receiptAnswer)
        let session = ScanSession(scanner: AppleUIFixtures.FixedScanner(lines: []),
                                  assistant: AppleModelService(primary: model, fallback: nil),
                                  context: Self.container.mainContext)
        let width = NSHostingView(rootView: ScanToolbarActions(session: session)).fittingSize.width
        #expect(width > 0 && width < 140)
    }

    /// Esc while a field is being edited only leaves the field (the edits
    /// stay); with nothing in edit it closes the scan. The ✕ always closes.
    @Test func escapeLeavesAFieldBeforeItClosesTheScan() {
        #expect(ScanEscape.action(firstResponder: NSTextView()) == .endEditing)
        #expect(ScanEscape.action(firstResponder: NSDatePicker()) == .endEditing)
        #expect(ScanEscape.action(firstResponder: nil) == .closeScan)
        #expect(ScanEscape.action(firstResponder: NSWindow()) == .closeScan)
        #expect(ScanEscape.action(firstResponder: NSView()) == .closeScan)
    }

    /// General scans: a two-page letter as a document (full text, page
    /// control) and a business card as a contact.
    @Test func generalScans() async throws {
        let container = Self.container
        let folder = URL.temporaryDirectory.appending(path: "mference-scan-general-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let letterPages = [["Northwind Studio", "14 Harbor Lane, Portland", "", "Dear Sam,",
                            "Thank you for joining the studio tour last week.",
                            "The sketches you asked about are attached to this letter."],
                           ["We would love to host your team again in the spring.", "Warm regards,", "Priya Raman"]]
        let pdf = try Self.pdf(pages: letterPages.map { $0.joined(separator: "\n") }, in: folder)
        let lines = letterPages.enumerated().flatMap { page, texts in
            texts.filter { !$0.isEmpty }.map { ScanLine(text: $0, confidence: 0.95, box: .zero, page: page) }
        }
        let reading = ScanReading(lines: lines, pageCount: 2, pageAspects: [612.0 / 792, 612.0 / 792],
                                  heading: "Northwind Studio")
        let off = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: .unavailable(reason: "off"), answer: "")
        let document = ScanSession(scanner: StaticScanner(reading: reading),
                                   assistant: AppleModelService(primary: off, fallback: nil),
                                   context: container.mainContext, openFile: { _ in })
        document.saveToNotes = { _, _ in }
        document.open(pdf)
        await document.waitForReading()
        #expect(document.kind == .document && document.pageCount == 2)
        for scheme in [ColorScheme.light, .dark] {
            try Snapshot.render(RootView(model: ShellModel(selection: .scan, defaults: nil), chrome: .inline)
                .environment(document), name: "07-scan-document", colorScheme: scheme)
        }
        document.showPage(1)
        try Snapshot.render(RootView(model: ShellModel(selection: .scan, defaults: nil), chrome: .inline)
            .environment(document), name: "07-scan-pages")

        let cardLines = ["Priya Raman", "Design Director", "Northwind Studio", "+1 (415) 555-0134",
                         "priya@northwind.example", "northwind.example"]
        let cardPDF = try Self.pdf(pages: [cardLines.joined(separator: "\n")], in: folder, name: "card.pdf",
                                   size: CGSize(width: 252, height: 144), fontSize: 11)
        let card = ScanSession(scanner: StaticScanner(reading: ScanReading(
                                   lines: cardLines.map { ScanLine(text: $0, confidence: 0.95, box: .zero) })),
                               assistant: AppleModelService(primary: off, fallback: nil),
                               context: container.mainContext, openFile: { _ in })
        card.open(cardPDF)
        await card.waitForReading()
        #expect(card.kind == .contact && card.contact.name == "Priya Raman")
        try Snapshot.render(RootView(model: ShellModel(selection: .scan, defaults: nil), chrome: .inline)
            .environment(card), name: "07-scan-contact")
    }

    struct StaticScanner: DocumentRecognizing {
        var reading: ScanReading
        func recognize(imageAt url: URL) async throws -> ScanReading { reading }
        func recognize(image: CGImage) async throws -> ScanReading { reading }
    }

    static func pdf(pages: [String], in folder: URL, name: String = "letter.pdf",
                    size: CGSize = CGSize(width: 612, height: 792), fontSize: CGFloat = 15) throws -> URL {
        let url = folder.appending(path: name)
        var box = CGRect(origin: .zero, size: size)
        let context = try #require(CGContext(url as CFURL, mediaBox: &box, nil))
        for text in pages {
            context.beginPDFPage(nil)
            var y = size.height - fontSize * 3
            for line in text.components(separatedBy: "\n") {
                let drawn = CTLineCreateWithAttributedString(NSAttributedString(string: line,
                    attributes: [.font: NSFont.systemFont(ofSize: fontSize)]))
                context.textPosition = CGPoint(x: fontSize * 2, y: y)
                CTLineDraw(drawn, context)
                y -= fontSize * 1.6
            }
            context.endPDFPage()
        }
        context.closePDF()
        return url
    }

    @Test func shellSlotsBuildWithoutServices() {
        // The no-argument initializers the shell routes to render an honest
        // placeholder until the app injects the models.
        let renderer = ImageRenderer(content: ScanView().frame(width: 600, height: 600))
        #expect(renderer.nsImage != nil)
    }
}
