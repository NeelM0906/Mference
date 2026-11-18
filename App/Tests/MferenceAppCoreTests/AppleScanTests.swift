import CoreGraphics
import Foundation
import ImageIO
@testable import MferenceAppCore
import SwiftData
import Testing

struct FixedScanner: DocumentRecognizing {
    var reading: ScanReading

    init(lines: [String], confidence: Float = 0.95) {
        reading = ScanReading(lines: lines.map { ScanLine(text: $0, confidence: confidence, box: .zero) })
    }

    func recognize(imageAt url: URL) async throws -> ScanReading { reading }
    func recognize(image: CGImage) async throws -> ScanReading { reading }
}

/// Opens once; waiting ignores cancellation, like a slow Vision request.
actor ScanTestLatch {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        waiters.forEach { $0.resume() }
        waiters = []
    }
}

/// Fails "slow-failing" files once `release` opens; reads the rest at once.
struct GatedScanner: DocumentRecognizing {
    var release: ScanTestLatch
    var lines: [String]

    func recognize(imageAt url: URL) async throws -> ScanReading {
        if url.lastPathComponent.hasPrefix("slow-failing") {
            await release.wait()
            throw ScanError.noTextFound
        }
        return ScanReading(lines: lines.map { ScanLine(text: $0, confidence: 0.95, box: .zero) })
    }

    func recognize(image: CGImage) async throws -> ScanReading { throw ScanError.unreadableImage }
}

@Suite struct AppleScanTests {
    static func reading(_ fixture: ReceiptFixture) -> ScanReading {
        ScanReading(lines: fixture.lines.filter { !$0.isEmpty }.map { ScanLine(text: $0, confidence: 0.95, box: .zero) })
    }

    @Test(arguments: ReceiptFixtures.all.map(\.name))
    func printedTextGivesTotalDateMerchantAndCard(name: String) throws {
        let fixture = try #require(ReceiptFixtures.all.first { $0.name == name })
        let draft = ReceiptHeuristics.read(Self.reading(fixture))
        #expect(draft.total == fixture.total)
        #expect(draft.date == fixture.date)
        #expect(draft.merchant == fixture.merchant)
        #expect(draft.paidWith == fixture.paidWith)
        if fixture.name != "03-grocer-eur" { #expect(draft.tax == fixture.tax) }
    }

    @Test func amountsIgnoreSubtotalAndReadTheLastTotal() {
        let lines = ["Subtotal 40.96", "Total before tax 40.96", "Tax 3.53", "TOTAL", "44.49"]
        let draft = ReceiptHeuristics.read(ScanReading(lines: lines.map { ScanLine(text: $0, confidence: 1, box: .zero) }))
        #expect(draft.total == Decimal(string: "44.49"))
        #expect(draft.tax == Decimal(string: "3.53"))
        #expect(ReceiptHeuristics.lastAmount(in: "Qty 2 1234.50") == Decimal(string: "1234.50"))
        #expect(ReceiptHeuristics.lastAmount(in: "Item 12") == nil)
    }

    @Test func refundsKeepTheirSign() {
        let lines = ["REFUND", "Subtotal -40.96", "Tax -$3.53", "TOTAL (44.49)"]
        let draft = ReceiptHeuristics.read(ScanReading(lines: lines.map { ScanLine(text: $0, confidence: 1, box: .zero) }))
        #expect(draft.total == Decimal(string: "-44.49"))
        #expect(draft.tax == Decimal(string: "-3.53"))
        #expect(ReceiptHeuristics.lastAmount(in: "Return 12.50-") == Decimal(string: "-12.50"))
        #expect(ReceiptHeuristics.lastAmount(in: "Total $-6.00") == Decimal(string: "-6.00"))
        #expect(ReceiptHeuristics.lastAmount(in: "Drill bits 1/8 - 12.50") == Decimal(string: "12.50"))
    }

    @Test func rowsJoinLabelsAndAmountsVisionReturnsApart() {
        // Vision order: the labels column, then the amounts column.
        let lines = [
            ScanLine(text: "Subtotal", confidence: 1, box: CGRect(x: 0.1, y: 0.70, width: 0.2, height: 0.03)),
            ScanLine(text: "TOTAL", confidence: 1, box: CGRect(x: 0.1, y: 0.80, width: 0.15, height: 0.03)),
            ScanLine(text: "40.96", confidence: 1, box: CGRect(x: 0.7, y: 0.705, width: 0.1, height: 0.03)),
            ScanLine(text: "44.49", confidence: 1, box: CGRect(x: 0.7, y: 0.795, width: 0.1, height: 0.03)),
        ]
        let reading = ScanReading(lines: lines)
        #expect(reading.rows == ["Subtotal   40.96", "TOTAL   44.49"])
        #expect(ReceiptHeuristics.read(reading).total == Decimal(string: "44.49"))
    }

    @Test func rowsFollowATiltedPage() {
        // Tilted about 3 degrees: the amount on the right sits one line
        // height lower than its label, level with the next label.
        let slope = 0.05
        func line(_ text: String, x: Double, y: Double, width: Double) -> ScanLine {
            ScanLine(text: text, confidence: 1,
                     box: CGRect(x: x, y: y, width: width, height: 0.025 + slope * width), slope: slope)
        }
        let lines = [
            line("Subtotal   ", x: 0.10, y: 0.600, width: 0.30),
            line("Tax", x: 0.10, y: 0.630, width: 0.06),
            line("Total", x: 0.10, y: 0.660, width: 0.10),
            line("75.45", x: 0.70, y: 0.630, width: 0.10),
            line("6.60", x: 0.70, y: 0.660, width: 0.08),
            line("82.05", x: 0.70, y: 0.690, width: 0.10),
        ]
        let draft = ReceiptHeuristics.read(ScanReading(lines: lines))
        #expect(draft.tax == Decimal(string: "6.60"))
        #expect(draft.total == Decimal(string: "82.05"))
    }

    @Test func fieldsAgreeingWithThePrintedTextAreSureTheRestCheck() {
        let reading = Self.reading(ReceiptFixtures.all[0])
        let printed = ReceiptHeuristics.read(reading)
        var model = printed
        model.total = Decimal(string: "45.49")  // disagrees with the print
        model.category = "Home improvement"
        model.currency = "USD"
        let fields = ReceiptFieldMerger.merge(model: model, text: printed, reading: reading, filledBy: .appleOnDevice)
        #expect(fields[.merchant] == ScanField(key: .merchant, value: "Corner Hardware", confidence: .sure))
        #expect(fields[.date] == ScanField(key: .date, value: "Sep 12, 2026", confidence: .sure))
        #expect(fields[.tax] == ScanField(key: .tax, value: "$3.53", confidence: .sure))
        #expect(fields[.total] == ScanField(key: .total, value: "$45.49", confidence: .check))
        #expect(fields[.category]?.confidence == .check)
        #expect(fields[.paidWith]?.value == "Visa")
        #expect(fields.filledBy == .appleOnDevice)
        #expect(fields.fields.map(\.key) == ScanFieldKey.allCases)

        let textOnly = ReceiptFieldMerger.merge(model: nil, text: printed, reading: reading, filledBy: nil)
        #expect(textOnly.fields.allSatisfy { $0.confidence == .check })
        #expect(textOnly[.total]?.value == "44.49")

        let blurry = ScanReading(lines: reading.lines.map { ScanLine(text: $0.text, confidence: 0.3, box: .zero) })
        let lowConfidence = ReceiptFieldMerger.merge(model: printed, text: printed, reading: blurry, filledBy: .quick)
        #expect(lowConfidence.needsChecking == ScanFieldKey.allCases)
    }

    @Test func fieldsRoundTripThroughJSON() throws {
        let fields = ScanFields(fields: [ScanField(key: .total, value: "$1.00", confidence: .sure)],
                                lineItems: [ReceiptLineItem(name: "Tea", amount: 1)], currency: "USD",
                                filledBy: .quick)
        #expect(try ScanFields.decode(fields.encoded()) == fields)
    }

    @Test func csvQuotesAndDefusesFormulas() {
        let fields = ScanFields(fields: [
            ScanField(key: .merchant, value: "=HYPERLINK(\"x\")", confidence: .sure),
            ScanField(key: .date, value: "Sep 12, 2026", confidence: .sure),
            ScanField(key: .total, value: "$1,330.55", confidence: .sure),
            ScanField(key: .tax, value: "$98.56", confidence: .check),
            ScanField(key: .category, value: "Home, garden", confidence: .check),
            ScanField(key: .paidWith, value: "Visa", confidence: .sure),
        ], lineItems: [ReceiptLineItem(name: "Pot", amount: Decimal(string: "1204")),
                       ReceiptLineItem(name: "Soil", amount: nil)], currency: "USD")
        let csv = ScanCSVExporter.csv([ScanExportRow(scannedAt: .now, fields: fields)])
        let lines = csv.components(separatedBy: "\r\n")
        #expect(lines[0] == "Date,Merchant,Category,Total,Tax,Currency,Paid with,Items")
        #expect(lines[1] == #"2026-09-12,"'=HYPERLINK(""x"")","Home, garden",1330.55,98.56,USD,Visa,Pot 1204.00; Soil"#)
        #expect(csv.hasSuffix("\r\n"))
        #expect(ScanCSVExporter.fileData([]).starts(with: [0xEF, 0xBB, 0xBF]))
        #expect(ScanCSVExporter.cell("-12.50") == "-12.50")
        #expect(ScanCSVExporter.cell("-cmd") == "'-cmd")
    }

    @Test @MainActor func sessionReadsFillsConfirmsAndExports() async throws {
        let container = try AppModelContainer.inMemory()
        let directory = URL.temporaryDirectory.appending(path: "scan-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let image = directory.appending(path: "in.png")
        try Data("not really a png".utf8).write(to: image)

        let fixture = ReceiptFixtures.all[0]
        let model = ScriptedAssistantTextModel(kind: .appleOnDevice, answer: """
            {"merchant": "Corner Hardware", "date": "2026-09-12", "total": "44.49", "tax": "3.53",
             "currency": "USD", "category": "Home improvement", "paidWith": "Visa", "lineItems": []}
            """)
        let session = ScanSession(scanner: FixedScanner(lines: fixture.lines.filter { !$0.isEmpty }),
                                  assistant: AppleModelService(primary: model, fallback: nil),
                                  context: container.mainContext,
                                  imageDirectory: directory.appending(path: "Scans"))
        #expect(throws: ScanSessionError.nothingToConfirm) { try session.confirm() }
        session.open(image)
        #expect(session.phase == .reading)
        await session.waitForReading()
        #expect(session.phase == .review)
        #expect(session.title == "Receipt from Corner Hardware")
        #expect(session.filledByLabel == "Apple on-device")
        #expect(session.fields.needsChecking == [.category])

        session.setValue("Tools", for: .category)
        #expect(session.fields.needsChecking.isEmpty)
        let record = try session.confirm()
        #expect(session.phase == .saved)
        #expect(record.isConfirmed)
        #expect(record.imageRelativePath == "Scans/\(record.id.uuidString).png")
        #expect(FileManager.default.fileExists(atPath: directory.appending(path: record.imageRelativePath!).path))
        let stored = try ScanFields.decode(try #require(record.fieldsJSON))
        #expect(stored.value(.category) == "Tools")

        let csv = try session.exportCSV()
        #expect(csv.contains("2026-09-12,Corner Hardware,Tools,44.49,3.53,USD,Visa,"))
        #expect(try session.exportCSVFile().starts(with: [0xEF, 0xBB, 0xBF]))
        session.reset()
        #expect(session.phase == .empty)
    }

    @Test @MainActor func sessionReadsTextWhenNoModelCanRun() async throws {
        let container = try AppModelContainer.inMemory()
        let model = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: .unavailable(reason: "off"),
                                               answer: "unused")
        let session = ScanSession(scanner: FixedScanner(lines: ["CORNER HARDWARE", "TOTAL 44.49"]),
                                  assistant: AppleModelService(primary: model, fallback: nil),
                                  context: container.mainContext, imageDirectory: URL.temporaryDirectory)
        session.open(URL(filePath: "/nonexistent/receipt.png"))
        await session.waitForReading()
        #expect(session.phase == .review)
        #expect(session.fields.value(.total) == "44.49")
        #expect(session.fields.filledBy == nil)
    }

    /// The first file fails only after the second was read: the second's
    /// fields stay on screen.
    @Test @MainActor func aReplacedScanCannotOverwriteTheNewOne() async throws {
        let container = try AppModelContainer.inMemory()
        let gate = ScanTestLatch()
        let scanner = GatedScanner(release: gate, lines: ["CORNER HARDWARE", "TOTAL 44.49"])
        let model = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: .unavailable(reason: "off"),
                                               answer: "unused")
        let session = ScanSession(scanner: scanner, assistant: AppleModelService(primary: model, fallback: nil),
                                  context: container.mainContext, imageDirectory: URL.temporaryDirectory)
        session.open(URL(filePath: "/fixtures/slow-failing.png"))
        session.open(URL(filePath: "/fixtures/receipt.png"))
        let deadline = ContinuousClock.now + .seconds(10)
        while session.phase != .review, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        #expect(session.phase == .review)
        await gate.open()
        await session.waitForReading()
        #expect(session.phase == .review)
        #expect(session.fields.value(.total) == "44.49")
    }

    @Test @MainActor func confirmFailsWithoutSavingWhenTheImageIsGone() async throws {
        let container = try AppModelContainer.inMemory()
        let directory = URL.temporaryDirectory.appending(path: "scan-gone-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let image = directory.appending(path: "receipt.png")
        try Data("png".utf8).write(to: image)
        let model = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: .unavailable(reason: "off"),
                                               answer: "unused")
        let session = ScanSession(scanner: FixedScanner(lines: ["CORNER HARDWARE", "TOTAL 44.49"]),
                                  assistant: AppleModelService(primary: model, fallback: nil),
                                  context: container.mainContext, imageDirectory: directory.appending(path: "Scans"))
        session.open(image)
        await session.waitForReading()
        try FileManager.default.removeItem(at: image)
        #expect(throws: (any Error).self) { try session.confirm() }
        #expect(session.phase == .review)
        #expect(try container.mainContext.fetchCount(FetchDescriptor<ScanRecord>()) == 0)
    }

    @Test @MainActor func exportingNeverConfirmsTheScanInReview() async throws {
        let container = try AppModelContainer.inMemory()
        let model = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: .unavailable(reason: "off"),
                                               answer: "unused")
        let session = ScanSession(scanner: FixedScanner(lines: ["CORNER HARDWARE", "TOTAL 44.49"]),
                                  assistant: AppleModelService(primary: model, fallback: nil),
                                  context: container.mainContext, imageDirectory: URL.temporaryDirectory)
        session.open(URL(filePath: "/fixtures/receipt.png"))
        await session.waitForReading()
        let csv = try session.exportCSV()
        #expect(csv.components(separatedBy: "\r\n").filter { !$0.isEmpty }.count == 1)  // header only
        #expect(session.phase == .review)
        #expect(try container.mainContext.fetchCount(FetchDescriptor<ScanRecord>()) == 0)
    }

    @Test func largeImagesDecodeBounded() throws {
        let directory = URL.temporaryDirectory.appending(path: "scan-big-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let context = try #require(CGContext(data: nil, width: 6_000, height: 1_000, bitsPerComponent: 8, bytesPerRow: 0,
                                             space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        let url = directory.appending(path: "wide.png")
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try #require(context.makeImage()), nil)
        #expect(CGImageDestinationFinalize(destination))
        let image = try #require(VisionScanner.loadImage(at: url))
        #expect(max(image.width, image.height) == VisionScanner.maximumPixelSize)
    }

    @Test @MainActor func aPastedImageIsReadLikeAnOpenedFile() async throws {
        let container = try AppModelContainer.inMemory()
        let model = ScriptedAssistantTextModel(kind: .appleOnDevice, availability: .unavailable(reason: "off"),
                                               answer: "unused")
        let session = ScanSession(scanner: FixedScanner(lines: ["CORNER HARDWARE", "TOTAL 44.49"]),
                                  assistant: AppleModelService(primary: model, fallback: nil),
                                  context: container.mainContext, imageDirectory: URL.temporaryDirectory)
        try session.openPasted(Data("png bytes".utf8))
        await session.waitForReading()
        let url = try #require(session.imageURL)
        #expect(url.pathExtension == "png")
        #expect(FileManager.default.fileExists(atPath: url.path))
        #expect(session.phase == .review)
        #expect(session.fields.value(.total) == "44.49")
    }

    @Test func acceptsImageAndPDFFiles() {
        #expect(ScanSession.accepts(URL(filePath: "/x/a.HEIC")))
        #expect(ScanSession.accepts(URL(filePath: "/x/a.pdf")))
        #expect(!ScanSession.accepts(URL(filePath: "/x/a.txt")))
    }

    /// Apple Vision reads a rendered receipt on this Mac (no network, no
    /// permission).
    @Test func visionReadsARenderedReceipt() async throws {
        let fixture = ReceiptFixtures.all[0]
        let reading = try await VisionScanner().recognize(image: fixture.image())
        let draft = ReceiptHeuristics.read(reading)
        #expect(draft.total == fixture.total)
        #expect(draft.date == fixture.date)
        #expect(reading.transcript.localizedCaseInsensitiveContains("corner hardware"))
    }
}
