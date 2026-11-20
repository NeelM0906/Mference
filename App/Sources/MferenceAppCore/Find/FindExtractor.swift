import AppKit
import CryptoKit
import Foundation
import PDFKit
import Vision

/// What reading one file produced.
public enum FindExtraction: Sendable, Hashable, Codable {
    /// The file's passages (possibly none: a name-only file such as a Keynote
    /// deck or a photo with no text).
    case text([Passage])
    /// The file could not be read; `reason` is shown in Settings › Find.
    case skipped(reason: String)

    public struct Passage: Sendable, Hashable, Codable {
        public var location: String?
        public var text: String
        public var fromOCR: Bool
        /// A photo, frame or stretch of sound to learn by what it shows or
        /// sounds like (no text).
        public var media: FindMediaPiece?

        public init(location: String?, text: String, fromOCR: Bool = false, media: FindMediaPiece? = nil) {
            self.location = location
            self.text = text
            self.fromOCR = fromOCR
            self.media = media
        }
    }

    /// Passages as index chunks, long passages split.
    public var chunks: [FindIndex.ChunkInput] {
        guard case .text(let passages) = self else { return [] }
        var chunks: [FindIndex.ChunkInput] = []
        for passage in passages {
            if let media = passage.media {
                chunks.append(FindIndex.ChunkInput(location: passage.location, text: "", media: media))
                continue
            }
            for text in FindText.chunks(passage.text) where chunks.count < FindText.maxChunksPerFile {
                chunks.append(FindIndex.ChunkInput(location: passage.location, text: text, fromOCR: passage.fromOCR))
            }
        }
        return chunks
    }

    /// SHA-256 of the text, so copies of one file collapse in results.
    public var contentHash: String? {
        guard case .text(let passages) = self else { return nil }
        let text = passages.filter { $0.media == nil }.map(\.text).joined(separator: "\n")
        guard text.count > 40 else { return nil }
        return SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

/// Options that change what is read.
public struct FindExtractOptions: Sendable, Hashable, Codable {
    /// Read text in images and scanned PDF pages with Vision.
    public var ocr: Bool
    /// Plan photos, video and audio for learning by what they show and
    /// sound like (`FindMedia`).
    public var media: Bool
    /// Also decode a photo for the encoders now, while it is open (learning
    /// follows soon), into `FindMediaFolder.ready`.
    public var prepareMedia: Bool

    public init(ocr: Bool = true, media: Bool = false, prepareMedia: Bool = false) {
        self.ocr = ocr
        self.media = media
        self.prepareMedia = prepareMedia
    }
}

/// Reads the text out of one file. Synchronous; the live app runs it in a
/// helper process (`FindExtractionWorker`) so one bad file cannot take the
/// app down.
public enum FindExtractor {
    /// Files over this size are named only.
    public static let maxFileBytes: Int64 = 200 << 20
    /// Plain text read per file at most.
    static let maxTextBytes = 4 << 20
    /// PDF pages read at most.
    static let maxPDFPages = 2_000
    /// Scanned PDF pages read with OCR per file at most.
    static let maxOCRPagesPerPDF = 20
    /// A PDF page with less text than this has no real text layer and is
    /// treated as a scan (OCR). Short pages with a text layer are read as is.
    static let scannedPageCharacters = 20

    public static func extract(_ url: URL, kind: FindKind, options: FindExtractOptions) -> FindExtraction {
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0
        if kind.isMedia {
            if size > FindMedia.maxFileBytes { return .skipped(reason: "Over 32 GB: searched by name only.") }
        } else if size > maxFileBytes {
            return .skipped(reason: "Over 200 MB: searched by name only.")
        }
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            return .skipped(reason: "macOS did not let Find read it.")
        }
        switch kind {
        case .text: return plainText(url, code: false)
        case .code: return plainText(url, code: true)
        case .pdf: return pdf(url, options: options)
        case .image:
            var passages: [FindExtraction.Passage] = []
            if options.ocr {
                switch image(url) {
                case .text(let read): passages = read
                case .skipped(let reason): return .skipped(reason: reason)
                }
            }
            if options.media, case .text(var pieces) = FindMedia.plan(url, kind: .image) {
                if options.prepareMedia {
                    for index in pieces.indices {
                        guard let piece = pieces[index].media else { continue }
                        pieces[index].media?.prepared = FindMedia.prepare(url, kind: .image, pieces: [piece],
                                                                          into: FindMediaFolder.ready).first??.path
                    }
                }
                passages += pieces
            }
            return .text(passages)
        case .video, .audio: return options.media ? FindMedia.plan(url, kind: kind) : .text([])
        case .document: return document(url)
        case .slides: return slides(url)
        case .sheet: return sheet(url)
        }
    }

    // MARK: Plain text and code

    static func plainText(_ url: URL, code: Bool) -> FindExtraction {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            return .skipped(reason: "Could not be opened.")
        }
        defer { try? handle.close() }
        let data = (try? handle.read(upToCount: maxTextBytes)) ?? Data()
        guard let text = decode(data) else { return .skipped(reason: "Not readable text.") }
        if url.pathExtension.lowercased() == "csv" || url.pathExtension.lowercased() == "tsv" {
            return .text([.init(location: nil, text: text)])
        }
        guard code else { return .text([.init(location: nil, text: text)]) }
        // Code: runs of whole lines, each with its line range.
        var passages: [FindExtraction.Passage] = []
        var current: [Substring] = []
        var startLine = 1
        var length = 0
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        for (index, line) in lines.enumerated() {
            if length + line.count > FindText.chunkCharacters, !current.isEmpty {
                passages.append(.init(location: "Lines \(startLine)–\(startLine + current.count - 1)",
                                      text: current.joined(separator: "\n")))
                current = []
                length = 0
                startLine = index + 1
            }
            current.append(line)
            length += line.count + 1
        }
        if !current.isEmpty, current.contains(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) {
            passages.append(.init(location: "Lines \(startLine)–\(startLine + current.count - 1)",
                                  text: current.joined(separator: "\n")))
        }
        return .text(passages)
    }

    static func decode(_ data: Data) -> String? {
        if data.isEmpty { return "" }
        // Binary files have NUL bytes early on.
        if data.prefix(8_000).contains(0) { return nil }
        if let text = String(data: data, encoding: .utf8) { return text }
        // Cut mid-character at the read limit: drop the last bytes.
        for trim in 1...3 where data.count > trim {
            if let text = String(data: data.dropLast(trim), encoding: .utf8) { return text }
        }
        return String(data: data, encoding: .isoLatin1)
    }

    // MARK: PDF

    static func pdf(_ url: URL, options: FindExtractOptions) -> FindExtraction {
        guard let document = PDFDocument(url: url) else { return .skipped(reason: "Damaged or not a PDF.") }
        if document.isLocked { return .skipped(reason: "Password-protected: searched by name only.") }
        var passages: [FindExtraction.Passage] = []
        var ocrPages = 0
        for index in 0..<min(document.pageCount, maxPDFPages) {
            guard let page = document.page(at: index) else { continue }
            let location = "Page \(index + 1)"
            let text = page.string ?? ""
            if text.trimmingCharacters(in: .whitespacesAndNewlines).count < scannedPageCharacters,
               options.ocr, ocrPages < maxOCRPagesPerPDF,
               let image = render(page), let read = recognize(image), !read.isEmpty {
                ocrPages += 1
                passages.append(.init(location: location, text: read, fromOCR: true))
            } else if !text.isEmpty {
                passages.append(.init(location: location, text: text))
            }
        }
        return .text(passages)
    }

    static func render(_ page: PDFPage) -> CGImage? {
        let bounds = page.bounds(for: .mediaBox)
        guard bounds.width > 0, bounds.height > 0 else { return nil }
        let scale = min(2, 1_800 / max(bounds.width, bounds.height))
        let width = Int(bounds.width * scale), height = Int(bounds.height * scale)
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.scaleBy(x: scale, y: scale)
        page.draw(with: .mediaBox, to: context)
        return context.makeImage()
    }

    // MARK: Images

    static func image(_ url: URL) -> FindExtraction {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            return .skipped(reason: "Not a readable image.")
        }
        // Most photos have no text: a quick look for text at a small size
        // (about 12 ms) decides whether the full read (about 60 ms) runs.
        // Screenshots always get the full read.
        if FindMedia.imageTokens(for: url) != FindMedia.screenshotTokens, !mayHaveText(source) {
            return .text([])
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: 2_400,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            return .skipped(reason: "Not a readable image.")
        }
        guard let text = recognize(image), text.count >= 8 else { return .text([]) }
        return .text([.init(location: nil, text: text, fromOCR: true)])
    }

    /// Whether Vision sees any text in a 768-px version of the image.
    static func mayHaveText(_ source: CGImageSource) -> Bool {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: 768,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        guard let small = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return true }
        let request = VNDetectTextRectanglesRequest()
        do {
            try VNImageRequestHandler(cgImage: small, options: [:]).perform([request])
        } catch {
            return true
        }
        return !(request.results ?? []).isEmpty
    }

    /// Vision's text recognition, run synchronously.
    static func recognize(_ image: CGImage) -> String? {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.automaticallyDetectsLanguage = true
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        let lines = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
        return lines.joined(separator: "\n")
    }

    // MARK: Documents

    static func document(_ url: URL) -> FindExtraction {
        let ext = url.pathExtension.lowercased()
        if ext == "pages" { return .text([]) }
        if ext == "html" || ext == "htm" {
            guard let data = try? Data(contentsOf: url, options: .mappedIfSafe), let html = decode(data) else {
                return .skipped(reason: "Could not be opened.")
            }
            return .text([.init(location: nil, text: stripMarkup(html))])
        }
        do {
            let string = try NSAttributedString(url: url, options: [:], documentAttributes: nil).string
            return .text([.init(location: nil, text: string)])
        } catch {
            return .skipped(reason: "Damaged, encrypted or an unsupported format.")
        }
    }

    /// Text of an HTML page: no scripts, styles or tags; common entities decoded.
    static func stripMarkup(_ html: String) -> String {
        var text = html
        for tag in ["script", "style", "head"] {
            text = text.replacingOccurrences(of: "<\(tag)[^>]*>[\\s\\S]*?</\(tag)>", with: " ",
                                             options: [.regularExpression, .caseInsensitive])
        }
        text = text.replacingOccurrences(of: "<(br|p|div|li|h[1-6]|tr)[^>]*>", with: "\n",
                                         options: [.regularExpression, .caseInsensitive])
        text = text.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
        for (entity, value) in ["&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": "\"", "&#39;": "'", "&nbsp;": " "] {
            text = text.replacingOccurrences(of: entity, with: value)
        }
        return text
    }

    // MARK: Slides and sheets (Office XML)

    static func slides(_ url: URL) -> FindExtraction {
        switch url.pathExtension.lowercased() {
        case "pptx":
            guard let members = Zip.list(url) else { return .skipped(reason: "Damaged or encrypted.") }
            let slides = members.compactMap { name -> (Int, String)? in
                guard name.hasPrefix("ppt/slides/slide"), name.hasSuffix(".xml"),
                      let number = Int(name.dropFirst("ppt/slides/slide".count).dropLast(4)) else { return nil }
                return (number, name)
            }.sorted { $0.0 < $1.0 }
            let passages = slides.compactMap { number, name -> FindExtraction.Passage? in
                guard let data = Zip.read(url, member: name) else { return nil }
                let text = XMLText.collect(data, elements: ["a:t"], separators: ["a:p"])
                return text.isEmpty ? nil : .init(location: "Slide \(number)", text: text)
            }
            return .text(passages)
        case "odp":
            return openDocument(url)
        default:
            // Keynote stores text in its own binary format: name only.
            return .text([])
        }
    }

    static func sheet(_ url: URL) -> FindExtraction {
        switch url.pathExtension.lowercased() {
        case "csv", "tsv": return plainText(url, code: false)
        case "ods": return openDocument(url)
        case "xlsx":
            guard let members = Zip.list(url) else { return .skipped(reason: "Damaged or encrypted.") }
            var shared: [String] = []
            if members.contains("xl/sharedStrings.xml"), let data = Zip.read(url, member: "xl/sharedStrings.xml") {
                shared = XMLText.items(data, item: "si", text: "t")
            }
            var names: [String] = []
            if let data = Zip.read(url, member: "xl/workbook.xml") {
                names = XMLText.attributes(data, element: "sheet", attribute: "name")
            }
            let sheets = members.compactMap { name -> (Int, String)? in
                guard name.hasPrefix("xl/worksheets/sheet"), name.hasSuffix(".xml"),
                      let number = Int(name.dropFirst("xl/worksheets/sheet".count).dropLast(4)) else { return nil }
                return (number, name)
            }.sorted { $0.0 < $1.0 }
            let passages = sheets.compactMap { number, member -> FindExtraction.Passage? in
                guard let data = Zip.read(url, member: member) else { return nil }
                let text = XMLText.sheetRows(data, shared: shared)
                guard !text.isEmpty else { return nil }
                let name = number <= names.count ? names[number - 1] : "Sheet \(number)"
                return .init(location: "Sheet “\(name)”", text: text)
            }
            return .text(passages)
        default:
            // Numbers stores its cells in its own binary format: name only.
            return .text([])
        }
    }

    static func openDocument(_ url: URL) -> FindExtraction {
        guard let data = Zip.read(url, member: "content.xml") else { return .skipped(reason: "Damaged or encrypted.") }
        return .text([.init(location: nil, text: XMLText.collect(data, elements: ["text:p", "text:h", "text:span"],
                                                                separators: ["text:p", "text:h"]))])
    }
}

/// Reads members of a zip file (Office XML) with the system `unzip`, which
/// is part of macOS.
enum Zip {
    static let tool = URL(filePath: "/usr/bin/unzip")

    static func list(_ url: URL) -> [String]? {
        // zipinfo mode: -Z must be the first option.
        guard let data = run(["-Z1", url.path], quiet: false) else { return nil }
        return String(decoding: data, as: UTF8.self).split(separator: "\n").map(String.init)
    }

    static func read(_ url: URL, member: String) -> Data? {
        run(["-p", url.path, member], quiet: true)
    }

    private static func run(_ arguments: [String], quiet: Bool) -> Data? {
        let process = Process()
        process.executableURL = tool
        process.arguments = (quiet ? ["-qq"] : []) + arguments
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return process.terminationStatus == 0 ? data : nil
    }
}

/// Collects text from Office XML with `XMLParser`.
enum XMLText {
    /// Text inside `elements`, with a newline after each of `separators`.
    static func collect(_ data: Data, elements: Set<String>, separators: Set<String>) -> String {
        let delegate = Collector(elements: elements, separators: separators)
        let parser = XMLParser(data: data)
        parser.delegate = delegate
        parser.parse()
        return FindText.normalizeWhitespace(delegate.text)
    }

    /// The text of each `item` element (its `text` children joined).
    static func items(_ data: Data, item: String, text: String) -> [String] {
        let delegate = ItemCollector(item: item, text: text)
        let parser = XMLParser(data: data)
        parser.delegate = delegate
        parser.parse()
        return delegate.items
    }

    static func attributes(_ data: Data, element: String, attribute: String) -> [String] {
        let delegate = AttributeCollector(element: element, attribute: attribute)
        let parser = XMLParser(data: data)
        parser.delegate = delegate
        parser.parse()
        return delegate.values
    }

    /// A worksheet's rows: cells tab-separated, shared strings resolved.
    static func sheetRows(_ data: Data, shared: [String]) -> String {
        let delegate = SheetCollector(shared: shared)
        let parser = XMLParser(data: data)
        parser.delegate = delegate
        parser.parse()
        return delegate.rows.filter { !$0.isEmpty }.map { $0.joined(separator: "\t") }.joined(separator: "\n")
    }

    final class Collector: NSObject, XMLParserDelegate {
        let elements: Set<String>
        let separators: Set<String>
        var depth = 0
        var text = ""

        init(elements: Set<String>, separators: Set<String>) {
            self.elements = elements
            self.separators = separators
        }

        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String] = [:]) {
            if elements.contains(name) { depth += 1 }
        }

        func parser(_ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?) {
            if elements.contains(name) { depth = max(0, depth - 1) }
            if separators.contains(name) { text += "\n" }
        }

        func parser(_ parser: XMLParser, foundCharacters string: String) {
            if depth > 0 { text += string }
        }
    }

    final class ItemCollector: NSObject, XMLParserDelegate {
        let item: String
        let textElement: String
        var items: [String] = []
        var current: String?
        var inText = false

        init(item: String, text: String) {
            self.item = item
            textElement = text
        }

        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String] = [:]) {
            if name == item { current = "" }
            if name == textElement { inText = true }
        }

        func parser(_ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?) {
            if name == textElement { inText = false }
            if name == item, let current {
                items.append(current)
                self.current = nil
            }
        }

        func parser(_ parser: XMLParser, foundCharacters string: String) {
            if inText { current? += string }
        }
    }

    final class AttributeCollector: NSObject, XMLParserDelegate {
        let element: String
        let attribute: String
        var values: [String] = []

        init(element: String, attribute: String) {
            self.element = element
            self.attribute = attribute
        }

        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String] = [:]) {
            if name == element, let value = attributes[attribute] { values.append(value) }
        }
    }

    final class SheetCollector: NSObject, XMLParserDelegate {
        let shared: [String]
        var rows: [[String]] = []
        var row: [String] = []
        var cellType: String?
        var value = ""
        var inValue = false

        init(shared: [String]) {
            self.shared = shared
        }

        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                    qualifiedName: String?, attributes: [String: String] = [:]) {
            switch name {
            case "row": row = []
            case "c": cellType = attributes["t"]; value = ""
            case "v", "t": inValue = true
            default: break
            }
        }

        func parser(_ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?) {
            switch name {
            case "v", "t": inValue = false
            case "c":
                var cell = value
                if cellType == "s", let index = Int(value), index < shared.count { cell = shared[index] }
                if !cell.isEmpty { row.append(cell) }
            case "row": rows.append(row)
            default: break
            }
        }

        func parser(_ parser: XMLParser, foundCharacters string: String) {
            if inValue { value += string }
        }
    }
}
