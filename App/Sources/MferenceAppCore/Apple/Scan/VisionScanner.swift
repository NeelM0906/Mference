import CoreGraphics
import DataDetection
import Foundation
import Vision

/// One recognized line of text.
public struct ScanLine: Sendable, Hashable, Codable {
    public var text: String
    /// Vision's confidence, 0...1.
    public var confidence: Float
    /// Normalized box on its page, origin at the top left (for highlighting).
    public var box: CGRect
    /// Baseline tilt as dy/dx in top-left coordinates (a photo held askew).
    public var slope: Double
    /// The page it is on, from 0 (a PDF's later pages).
    public var page: Int

    public init(text: String, confidence: Float, box: CGRect, slope: Double = 0, page: Int = 0) {
        self.text = text
        self.confidence = confidence
        self.box = box
        self.slope = slope
        self.page = page
    }
}

/// A table Vision found, as the text of each cell, row by row.
public struct ScanTable: Sendable, Hashable, Codable {
    public var page: Int
    public var rows: [[String]]

    public init(page: Int = 0, rows: [[String]]) {
        self.page = page
        self.rows = rows
    }

    public var columnCount: Int { rows.map(\.count).max() ?? 0 }
}

/// Something the text names that an app can use: found by Vision's data
/// detectors, or by `NSDataDetector` over the text.
public enum ScanDetected: Sendable, Hashable, Codable {
    case phone(String)
    case email(String)
    case link(String)
    case address(String)
    /// A date; `hasTime` false for a day without a time of day.
    case date(start: Date, end: Date?, hasTime: Bool)
}

/// What text recognition read from one image or document.
public struct ScanReading: Sendable, Hashable, Codable {
    /// Lines in reading order, page by page.
    public var lines: [ScanLine]
    /// Vision's transcript of the whole document.
    public var transcript: String
    /// The first page's width over height (normalized boxes hide it; tilt
    /// needs it).
    public var aspect: Double
    /// Pages read (a PDF's first pages, up to `VisionScanner.maximumPages`).
    public var pageCount: Int
    /// Each page's width over height.
    public var pageAspects: [Double]
    public var tables: [ScanTable]
    /// Vision's own detections; `detectedData` falls back to the text.
    public var detected: [ScanDetected]
    /// The document's title line, when Vision recognized one.
    public var heading: String?

    public init(lines: [ScanLine], transcript: String? = nil, aspect: Double = 1, pageCount: Int = 1,
                pageAspects: [Double]? = nil, tables: [ScanTable] = [], detected: [ScanDetected] = [],
                heading: String? = nil) {
        self.lines = lines
        self.transcript = transcript ?? lines.map(\.text).joined(separator: "\n")
        self.aspect = aspect
        self.pageCount = max(pageCount, 1)
        self.pageAspects = pageAspects ?? [aspect]
        self.tables = tables
        self.detected = detected
        self.heading = heading
    }

    public var isEmpty: Bool { lines.isEmpty && transcript.isEmpty }

    func aspect(of page: Int) -> Double { pageAspects.indices.contains(page) ? pageAspects[page] : aspect }

    /// The full text for a person to read, copy or save: the printed rows,
    /// page by page, so a label stays beside its amount.
    public var fullText: String {
        let rows = rows
        return rows.isEmpty ? transcript : rows.joined(separator: "\n")
    }

    /// What a model reads: the printed rows, each label beside its amount.
    /// Vision's transcript lists a right-aligned amount column after all the
    /// labels, which made a model pair them wrongly.
    public var modelText: String { fullText }

    /// The printed rows, page after page: Vision often returns a label
    /// ("TOTAL") and its amount ("44.49") as separate lines, so lines whose
    /// boxes share a baseline on the same page are joined left to right.
    /// Without boxes, the lines as given.
    public var rows: [String] {
        let pages = Array(Set(lines.map(\.page))).sorted()
        return pages.flatMap { page in
            Self.rows(of: lines.filter { $0.page == page }, aspect: aspect(of: page))
        }
    }

    static func rows(of lines: [ScanLine], aspect: Double) -> [String] {
        let boxed = lines.filter { $0.box.height > 0 }
        guard boxed.count == lines.count, !boxed.isEmpty else { return lines.map(\.text) }
        let tilt = Self.tilt(of: lines, aspect: aspect)
        let level = { (line: ScanLine) in Double(line.box.midY) - tilt * Double(line.box.midX) }
        // A tilted line's box is taller than its text by about tilt x width.
        let height = { (line: ScanLine) in
            max(Double(line.box.height) - abs(tilt) * Double(line.box.width), Double(line.box.height) * 0.3)
        }
        var groups: [[ScanLine]] = []
        for line in lines.sorted(by: { level($0) < level($1) }) {
            if let index = groups.indices.last, let anchor = groups[index].first,
               abs(level(line) - level(anchor)) < min(height(line), height(anchor)) * 0.6 {
                groups[index].append(line)
            } else {
                groups.append([line])
            }
        }
        return groups.map { $0.sorted { $0.box.minX < $1.box.minX }.map(\.text).joined(separator: "   ") }
    }

    /// The first page's tilt as dy/dx in normalized top-left coordinates.
    var pageTilt: Double {
        let first = lines.map(\.page).min() ?? 0
        return Self.tilt(of: lines.filter { $0.page == first }, aspect: aspect(of: first))
    }

    /// A page's tilt, so a label and its amount far to the right still share
    /// a row. From the drift of the left text edge down the page (Theil-Sen
    /// over the left-aligned lines), else the median baseline slope.
    static func tilt(of lines: [ScanLine], aspect: Double) -> Double {
        let leftmost = lines.map { Double($0.box.minX) }.min() ?? 0
        let edge = lines.filter { Double($0.box.minX) < leftmost + 0.08 }
        if edge.count >= 4 {
            var drifts: [Double] = []
            for i in edge.indices {
                for j in edge.indices where j > i {
                    let dy = Double(edge[j].box.midY - edge[i].box.midY)
                    if abs(dy) > 0.02 { drifts.append(Double(edge[j].box.minX - edge[i].box.minX) / dy) }
                }
            }
            if !drifts.isEmpty {
                let drift = drifts.sorted()[drifts.count / 2]
                // Rotating by t moves the left edge by tan t per unit of
                // height and rows by -tan t per unit of width, in pixels.
                return max(-0.3, min(0.3, -drift * aspect * aspect))
            }
        }
        let slopes = lines.filter { $0.box.width > 0.15 }.map(\.slope).sorted()
        return slopes.isEmpty ? 0 : slopes[slopes.count / 2]
    }

    /// Phone numbers, addresses, links and dates in the text: Vision's own
    /// detections, else `NSDataDetector` over the full text (Vision gives
    /// none for some documents, and test readings carry only lines).
    public func detectedData(reference: Date = .now) -> [ScanDetected] {
        detected.isEmpty ? Self.detect(in: fullText, reference: reference) : detected
    }

    static func detect(in text: String, reference: Date) -> [ScanDetected] {
        let types: NSTextCheckingResult.CheckingType = [.phoneNumber, .link, .address, .date]
        guard let detector = try? NSDataDetector(types: types.rawValue) else { return [] }
        var found: [ScanDetected] = []
        let range = NSRange(text.startIndex..., in: text)
        for match in detector.matches(in: text, range: range) {
            switch match.resultType {
            case .phoneNumber:
                if let phone = match.phoneNumber { found.append(.phone(phone)) }
            case .link:
                guard let url = match.url else { continue }
                if url.scheme == "mailto" {
                    found.append(.email(String(url.absoluteString.dropFirst("mailto:".count))))
                } else {
                    found.append(.link(url.absoluteString))
                }
            case .address:
                if let span = Range(match.range, in: text) {
                    found.append(.address(String(text[span]).replacingOccurrences(of: "\n", with: ", ")))
                }
            case .date:
                guard let date = match.date, let span = Range(match.range, in: text) else { continue }
                found.append(.date(start: date, end: match.duration > 0 ? date.addingTimeInterval(match.duration) : nil,
                                   hasTime: Self.namesATime(String(text[span]))))
            default:
                break
            }
        }
        return found
    }

    /// "7 pm", "19:30", "noon": the matched words name a time of day.
    static func namesATime(_ text: String) -> Bool {
        let lowered = text.lowercased()
        if lowered.contains("noon") || lowered.contains("midnight") { return true }
        return lowered.range(of: #"\d\s*(:\d{2}|am|pm|a\.m\.|p\.m\.)"#, options: .regularExpression) != nil
    }
}

public enum ScanError: Error, Sendable, Hashable {
    case unreadableImage
    case noTextFound
    case recognitionFailed(String)

    public var userMessage: String {
        switch self {
        case .unreadableImage: "This file is not an image Mference can read."
        case .noTextFound: "No text was found in this image."
        case .recognitionFailed(let detail): "Text recognition failed: \(detail)"
        }
    }
}

/// Reads text from an image on this Mac.
public protocol DocumentRecognizing: Sendable {
    func recognize(imageAt url: URL) async throws -> ScanReading
    func recognize(image: CGImage) async throws -> ScanReading
}

/// Apple Vision's `RecognizeDocumentsRequest` (decision L12). Runs on this
/// Mac with no permission and no network.
public struct VisionScanner: DocumentRecognizing {
    public init() {}

    /// Pages of a PDF read at most; later pages are left out (the screen
    /// says how many were read).
    public static let maximumPages = 20

    /// An image, or a PDF's pages one at a time (each decoded within the
    /// pixel bound and released before the next), merged page by page.
    public func recognize(imageAt url: URL) async throws -> ScanReading {
        let pages = Self.pageCount(at: url)
        guard pages > 1 else {
            guard let image = Self.loadImage(at: url) else { throw ScanError.unreadableImage }
            return try await recognize(image: image)
        }
        var lines: [ScanLine] = [], transcripts: [String] = [], aspects: [Double] = []
        var tables: [ScanTable] = [], detected: [ScanDetected] = []
        var heading: String?
        for page in 0..<pages {
            try Task.checkCancellation()
            guard let image = Self.loadImage(at: url, page: page) else { throw ScanError.unreadableImage }
            aspects.append(Double(image.width) / Double(max(image.height, 1)))
            let reading: ScanReading
            do {
                reading = try await recognize(image: image)
            } catch ScanError.noTextFound {
                continue
            }
            lines += reading.lines.map { var line = $0; line.page = page; return line }
            transcripts.append(reading.transcript)
            tables += reading.tables.map { var table = $0; table.page = page; return table }
            detected += reading.detected
            heading = heading ?? reading.heading
        }
        let reading = ScanReading(lines: lines, transcript: transcripts.joined(separator: "\n\n"),
                                  aspect: aspects.first ?? 1, pageCount: pages, pageAspects: aspects,
                                  tables: tables, detected: detected, heading: heading)
        guard !reading.isEmpty else { throw ScanError.noTextFound }
        return reading
    }

    /// Pages to read at `url`: a PDF's page count (up to `maximumPages`),
    /// else 1.
    public static func pageCount(at url: URL) -> Int {
        guard url.pathExtension.lowercased() == "pdf", let document = CGPDFDocument(url as CFURL) else { return 1 }
        return max(1, min(document.numberOfPages, maximumPages))
    }

    /// Longest side, in pixels, an image is decoded at: enough for text
    /// recognition, bounded for memory (a 100-megapixel photo decodes small).
    public static let maximumPixelSize = 4_096

    /// The image at `url` (decoded at most `maximumPixelSize` on its longest
    /// side, orientation applied), or a PDF's page (from 0) at 200 dpi.
    public static func loadImage(at url: URL, page: Int = 0, maximumPixelSize: Int = maximumPixelSize) -> CGImage? {
        if url.pathExtension.lowercased() == "pdf" {
            return renderPage(page, of: url)
        }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary)
        else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumPixelSize,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    static func renderPage(_ index: Int, of url: URL) -> CGImage? {
        guard let document = CGPDFDocument(url as CFURL), let page = document.page(at: index + 1) else { return nil }
        let bounds = page.getBoxRect(.mediaBox)
        let scale = 200.0 / 72.0
        let width = Int(bounds.width * scale), height = Int(bounds.height * scale)
        guard width > 0, height > 0, width * height <= 60_000_000,
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.scaleBy(x: scale, y: scale)
        context.translateBy(x: -bounds.minX, y: -bounds.minY)
        context.drawPDFPage(page)
        return context.makeImage()
    }

    public func recognize(image: CGImage) async throws -> ScanReading {
        var request = RecognizeDocumentsRequest()
        request.textRecognitionOptions.useLanguageCorrection = true
        request.textRecognitionOptions.automaticallyDetectLanguage = true
        let observations: [DocumentObservation]
        do {
            observations = try await request.perform(on: image)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw ScanError.recognitionFailed(error.localizedDescription)
        }
        var lines: [ScanLine] = []
        var transcripts: [String] = []
        var tables: [ScanTable] = []
        var detected: [ScanDetected] = []
        var heading: String?
        for observation in observations {
            let text = observation.document.text
            transcripts.append(text.transcript)
            heading = heading ?? observation.document.title.map(\.transcript)
            tables += observation.document.tables.map { table in
                ScanTable(rows: table.rows.map { row in
                    row.map { $0.content.text.transcript.trimmingCharacters(in: .whitespacesAndNewlines) }
                })
            }
            detected += text.detectedData.compactMap { Self.detected($0.match) }
            for line in text.lines {
                let candidate = line.topCandidates(1).first
                let string = (candidate?.string ?? line.transcript).trimmingCharacters(in: .whitespaces)
                guard !string.isEmpty else { continue }
                let box = line.boundingBox.cgRect
                let run = line.bottomRight.x - line.bottomLeft.x
                let slope = run > 0.01 ? -Double(line.bottomRight.y - line.bottomLeft.y) / Double(run) : 0
                lines.append(ScanLine(text: string, confidence: candidate?.confidence ?? line.confidence,
                                      box: CGRect(x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height),
                                      slope: slope))
            }
        }
        let reading = ScanReading(lines: lines, transcript: transcripts.joined(separator: "\n"),
                                  aspect: Double(image.width) / Double(max(image.height, 1)),
                                  tables: tables.filter { $0.rows.count >= 2 && $0.columnCount >= 2 },
                                  detected: detected, heading: heading?.trimmingCharacters(in: .whitespacesAndNewlines))
        guard !reading.isEmpty else { throw ScanError.noTextFound }
        return reading
    }
}

extension VisionScanner {
    /// Vision's data-detector match as the kinds Scan uses.
    static func detected(_ match: DataDetector.Match) -> ScanDetected? {
        switch match.details {
        case .phoneNumber(let phone): .phone(phone.phoneNumber)
        case .emailAddress(let email): .email(email.emailAddress)
        case .link(let link): link.url.scheme == "mailto"
            ? .email(String(link.url.absoluteString.dropFirst("mailto:".count))) : .link(link.url.absoluteString)
        case .postalAddress(let address): .address(address.fullAddress.replacingOccurrences(of: "\n", with: ", "))
        case .calendarEvent(let event):
            event.startDate.map { .date(start: $0, end: event.endDate, hasTime: !event.allDay) }
        default: nil
        }
    }
}
