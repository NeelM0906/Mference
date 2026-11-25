import Foundation
import ImageIO
import PDFKit
import UniformTypeIdentifiers

/// A file attached to a chat message, kept as the text read from it on this
/// Mac. The file itself is not copied: the chat keeps its name, size and
/// text, so deleting the file leaves only that text behind (and deleting the
/// chat removes it too). Stored as JSON in `StoredMessage.attachmentsJSON`.
public struct ChatAttachment: Sendable, Hashable, Codable, Identifiable {
    public enum Kind: String, Sendable, Hashable, Codable {
        /// PDF, Word, RTF, text, Markdown, code, slides or sheets.
        case document
        /// A picture; read for the text in it (Vision OCR).
        case image
    }

    public var id: UUID
    public var filename: String
    public var kind: Kind
    public var byteCount: Int64
    /// The text read from the file; empty for a picture with no text.
    public var text: String
    /// The file had more text than `ChatAttachmentReader.storedCharacterLimit`;
    /// only its start was kept.
    public var truncated: Bool
    /// Pages in a PDF, nil for other files.
    public var pageCount: Int?
    /// A picture's pixels, scaled down to `ChatAttachmentReader.imageSide`
    /// and saved as JPEG, so a model that sees images (Quick or Deep with
    /// image understanding) gets them in this turn and every later one. Nil
    /// for documents, and for pictures attached before Mference kept them.
    public var imageData: Data?

    public init(id: UUID = UUID(), filename: String, kind: Kind, byteCount: Int64, text: String,
                truncated: Bool = false, pageCount: Int? = nil, imageData: Data? = nil) {
        self.id = id
        self.filename = filename
        self.kind = kind
        self.byteCount = byteCount
        self.text = text
        self.truncated = truncated
        self.pageCount = pageCount
        self.imageData = imageData
    }

    /// A picture in which no text was found: the model learns only its name.
    public var hasText: Bool { !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    /// A short line for the chip ("12 pages", "PNG · no text found").
    public var detail: String {
        if kind == .image { return hasText ? "Text read from image" : "No text found" }
        if let pageCount { return pageCount == 1 ? "1 page" : "\(pageCount) pages" }
        return ByteCountFormatter.string(fromByteCount: byteCount, countStyle: .file)
    }
}

// MARK: - What the model reads

extension ChatAttachment {
    /// Characters of attachment text the model reads with the newest
    /// message. Earlier messages' attachments get a quarter of it, so a long
    /// chat with several files still fits the context. Apple's model has a
    /// 4K-token context, so it reads far less.
    public static func characterBudget(for model: AppModel) -> Int {
        model.isEngineModel ? 32_000 : 6_000
    }

    /// The budget for attachments of an earlier message in the chat.
    public static func earlierCharacterBudget(for model: AppModel) -> Int {
        characterBudget(for: model) / 4
    }

    /// Splits `budget` characters among attachments: short files keep all
    /// their text and leave the rest to the long ones.
    public static func shares(for attachments: [ChatAttachment], budget: Int) -> [Int] {
        var shares = Array(repeating: 0, count: attachments.count)
        var remaining = max(0, budget)
        let order = attachments.indices.sorted { attachments[$0].text.count < attachments[$1].text.count }
        for (position, index) in order.enumerated() {
            let fair = remaining / (order.count - position)
            let share = min(attachments[index].text.count, fair)
            shares[index] = share
            remaining -= share
        }
        return shares
    }

    /// The user turn as the model reads it: each attachment's text in an
    /// `<attachment>` block, then what the person typed. The stored message
    /// keeps only what was typed, so the bubble, the title and memory never
    /// see the files' text.
    ///
    /// When the model sees images (`seesImages`), pictures also go to it as
    /// pixels (`images(of:)`), and their blocks say so.
    public static func modelContent(message: String, attachments: [ChatAttachment], budget: Int,
                                    seesImages: Bool = false) -> String {
        guard !attachments.isEmpty else { return message }
        let shares = shares(for: attachments, budget: budget)
        var blocks: [String] = []
        for (attachment, share) in zip(attachments, shares) {
            blocks.append(attachment.block(characters: share, seesImages: seesImages))
        }
        let typed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        let request = typed.isEmpty ? defaultRequest(for: attachments) : typed
        return (blocks + [request]).joined(separator: "\n\n")
    }

    /// What the model is asked when the person sent files without a message.
    static func defaultRequest(for attachments: [ChatAttachment]) -> String {
        attachments.count == 1 ? "Summarize the attached file." : "Summarize the attached files."
    }

    /// The pictures of `attachments` a model that sees images gets, in order.
    public static func images(of attachments: [ChatAttachment]) -> [ChatImage] {
        attachments.filter { $0.kind == .image }.compactMap(\.imageData).map(ChatImage.init(data:))
    }

    /// Whether this picture reaches a model that sees images.
    var isShown: Bool { kind == .image && imageData != nil }

    func block(characters: Int, seesImages: Bool = false) -> String {
        let name = filename.replacingOccurrences(of: "\"", with: "'")
        var attributes = "name=\"\(name)\""
        if let pageCount { attributes += " pages=\"\(pageCount)\"" }
        let shown = seesImages && isShown
        guard hasText else {
            let note = shown ? "(An image, shown to you. No text was found in it.)"
                             : "(An image. No text was found in it, and this model cannot see images.)"
            return "<attachment \(attributes)>\n\(note)\n</attachment>"
        }
        let safe = text.replacingOccurrences(of: "</attachment>", with: "</attachment >")
        let shortened = characters < safe.count || truncated
        let body = characters < safe.count ? String(safe.prefix(characters)) : safe
        if shortened { attributes += " shortened=\"only the start of the file is included\"" }
        let source = kind == .image
            ? (shown ? "\n(An image, shown to you. The text read from it:)" : "\n(Text read from an image.)") : ""
        return "<attachment \(attributes)>\(source)\n\(body)\n</attachment>"
    }
}

extension StoredMessage {
    /// Files attached to a user message.
    public var attachments: [ChatAttachment] {
        get { ChatJSON.decode([ChatAttachment].self, from: attachmentsJSON) ?? [] }
        set { attachmentsJSON = newValue.isEmpty ? nil : ChatJSON.encode(newValue) }
    }
}

// MARK: - Reading files

/// Why a file could not be attached; `message` is shown on its chip.
public struct ChatAttachmentError: Error, Sendable, Hashable {
    public var message: String

    public init(_ message: String) { self.message = message }
}

/// Reads an attached file into text on this Mac, with Find's extractor
/// (PDFKit, rich text, Office files, Vision OCR for pictures).
public struct ChatAttachmentReader: Sendable {
    /// Text kept per file. The model reads less (`characterBudget`); this
    /// bounds what a chat stores.
    public static let storedCharacterLimit = 200_000
    /// Files per message.
    public static let maxAttachments = 10

    private let extractor: any FindExtracting

    public init(extractor: any FindExtracting = InProcessFindExtractor(timeout: .seconds(60))) {
        self.extractor = extractor
    }

    /// File extensions the "+" menu's Add files… offers.
    public static var documentExtensions: [String] {
        (["pdf"] + FindKind.documentExtensions + FindKind.textExtensions + FindKind.codeExtensions
            + FindKind.slideExtensions + FindKind.sheetExtensions)
            .filter { !appleOnlyExtensions.contains($0) }.sorted()
    }

    /// Picture extensions the "+" menu's Add images… offers.
    public static var imageExtensions: [String] { FindKind.imageExtensions.sorted() }

    /// Pages, Keynote and Numbers files hold their text in a format Find
    /// cannot read.
    static let appleOnlyExtensions: Set<String> = ["pages", "key", "numbers"]

    /// A Pages, Keynote or Numbers file.
    public static func isAppleFormat(_ url: URL) -> Bool {
        appleOnlyExtensions.contains(url.pathExtension.lowercased())
    }

    /// The attachment kind of a file, or nil if it cannot be attached.
    public static func kind(of url: URL) -> ChatAttachment.Kind? {
        guard let kind = FindKind.of(url), !kind.isMedia else { return nil }
        if appleOnlyExtensions.contains(url.pathExtension.lowercased()) { return nil }
        return kind == .image ? .image : .document
    }

    public func read(_ url: URL) async -> Result<ChatAttachment, ChatAttachmentError> {
        let ext = url.pathExtension.lowercased()
        if Self.appleOnlyExtensions.contains(ext) {
            return .failure(.init("Export it as PDF to attach it"))
        }
        guard let findKind = FindKind.of(url), !findKind.isMedia, let kind = Self.kind(of: url) else {
            return .failure(.init("This kind of file can’t be read"))
        }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0
        if size > FindExtractor.maxFileBytes { return .failure(.init("Too large (over 200 MB)")) }

        let extraction = await extractor.extract(url, kind: findKind, options: FindExtractOptions(ocr: true))
        let passages: [FindExtraction.Passage]
        switch extraction {
        case .skipped(let reason): return .failure(.init(reason))
        case .text(let read): passages = read.filter { $0.media == nil }
        }
        let joined = passages.map { passage in
            guard let location = passage.location, findKind == .pdf || findKind == .slides || findKind == .sheet
            else { return passage.text }
            return "[\(location)]\n\(passage.text)"
        }.joined(separator: "\n\n")
        let text = joined.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty, kind == .document { return .failure(.init("No text found in this file")) }

        let truncated = text.count > Self.storedCharacterLimit
        let pageCount = findKind == .pdf ? PDFDocument(url: url)?.pageCount : nil
        return .success(ChatAttachment(filename: url.lastPathComponent, kind: kind, byteCount: size,
                                       text: truncated ? String(text.prefix(Self.storedCharacterLimit)) : text,
                                       truncated: truncated, pageCount: pageCount,
                                       imageData: kind == .image ? Self.keptImage(url) : nil))
    }

    /// The longest side a kept picture is scaled to. Qwen3-VL's encoders
    /// work below this; it bounds what a chat stores.
    public static let imageSide = 1_536

    /// A picture scaled to at most `imageSide` (orientation applied) as JPEG,
    /// or nil when ImageIO cannot read it.
    static func keptImage(_ url: URL) -> Data? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: imageSide,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(destination, image,
                                   [kCGImageDestinationLossyCompressionQuality: 0.85] as CFDictionary)
        return CGImageDestinationFinalize(destination) ? data as Data : nil
    }
}
