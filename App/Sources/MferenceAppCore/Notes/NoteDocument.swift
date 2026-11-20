import Foundation

/// A note's body (Notes spec 8.2): a versioned list of blocks, stored as JSON
/// in `Note.bodyJSON`. The editor maps it to and from an attributed string;
/// search, export, the list preview and the word count read it directly.
public struct NoteDocument: Sendable, Hashable, Codable {
    /// The format version written by this build.
    public static let currentVersion = 1

    public var version: Int
    public var blocks: [Block]

    public init(blocks: [Block] = []) {
        self.version = Self.currentVersion
        self.blocks = blocks
    }

    /// An empty note: one empty title line, where the caret starts.
    public static var empty: NoteDocument { NoteDocument(blocks: [.heading(level: .title, runs: [])]) }

    /// A note that starts with `title` (meetings, file transcripts, imports).
    public static func titled(_ title: String, body: [Block] = []) -> NoteDocument {
        NoteDocument(blocks: [.heading(level: .title, runs: title.isEmpty ? [] : [TextRun(title)])] + body)
    }

    // MARK: Blocks

    /// Heading levels: the note's title line, a heading, a subheading.
    public enum HeadingLevel: Int, Sendable, Hashable, Codable, CaseIterable {
        case title = 1, heading = 2, subheading = 3
    }

    public enum Block: Sendable, Hashable {
        case heading(level: HeadingLevel, runs: [TextRun])
        case paragraph(runs: [TextRun])
        case bullet(runs: [TextRun], indent: Int)
        case numbered(runs: [TextRun], indent: Int)
        case check(runs: [TextRun], done: Bool, indent: Int)
        /// A file in the note's attachment folder (`NoteAttachments`).
        case image(attachmentID: String)
        case audio(attachmentID: String)
        /// A meeting note's time chip for one transcript segment.
        case transcriptAnchor(segmentID: UUID)
        /// "Mark moment" during a meeting, at a recording time in seconds.
        case marker(time: TimeInterval)

        /// The text runs of a text block; empty for the others.
        public var runs: [TextRun] {
            switch self {
            case .heading(_, let runs), .paragraph(let runs), .bullet(let runs, _),
                 .numbered(let runs, _), .check(let runs, _, _):
                runs
            case .image, .audio, .transcriptAnchor, .marker:
                []
            }
        }

        /// The block's plain text (empty for attachments and markers).
        public var text: String { runs.map(\.text).joined() }

        public var isText: Bool {
            switch self {
            case .heading, .paragraph, .bullet, .numbered, .check: true
            case .image, .audio, .transcriptAnchor, .marker: false
            }
        }

        /// The attachment file a block shows, if any.
        public var attachmentID: String? {
            switch self {
            case .image(let id), .audio(let id): id
            default: nil
            }
        }
    }

    // MARK: Derived text

    /// The note's title: the first block's text when it is the title line.
    public var title: String {
        guard case .heading(.title, let runs)? = blocks.first else { return "" }
        return runs.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The text blocks, one line each, in order.
    public var plainText: String {
        blocks.filter(\.isText).map(\.text).joined(separator: "\n")
    }

    /// The first non-empty line: the list shows it when the note has no title.
    public var firstLine: String {
        lines.first ?? ""
    }

    /// The list row's preview: the first non-empty line after the title line.
    public var previewLine: String {
        var remaining = blocks.filter(\.isText)
        if case .heading(.title, _)? = blocks.first, !remaining.isEmpty { remaining.removeFirst() }
        return remaining.map { $0.text.trimmingCharacters(in: .whitespaces) }.first { !$0.isEmpty } ?? ""
    }

    /// Non-empty lines of text.
    public var lines: [String] {
        blocks.filter(\.isText)
            .map { $0.text.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// Words in the text blocks (the editor footer's count).
    public var wordCount: Int {
        var count = 0
        let text = plainText
        text.enumerateSubstrings(in: text.startIndex..., options: [.byWords, .substringNotRequired]) { _, _, _, _ in
            count += 1
        }
        return count
    }

    /// Every attachment file the body refers to.
    public var attachmentIDs: [String] { blocks.compactMap(\.attachmentID) }

    /// Replaces the title line's text, inserting a title line when the note
    /// has none.
    public mutating func setTitle(_ title: String) {
        let runs = title.isEmpty ? [] : [TextRun(title)]
        if case .heading(.title, _)? = blocks.first {
            blocks[0] = .heading(level: .title, runs: runs)
        } else {
            blocks.insert(.heading(level: .title, runs: runs), at: 0)
        }
    }

    // MARK: Storage

    public func encoded() -> Data {
        // Every value here encodes; a failure would be a programming error.
        (try? Self.encoder.encode(self)) ?? Data()
    }

    /// Decodes a stored body. Nil or unreadable data is an empty note, so a
    /// damaged body never blocks opening the rest of the store.
    public static func decode(_ data: Data?) -> NoteDocument {
        guard let data, !data.isEmpty,
              let document = try? JSONDecoder().decode(NoteDocument.self, from: data) else { return .empty }
        return document
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    enum CodingKeys: String, CodingKey { case version = "v", blocks }
}

/// A stretch of text with one set of styles.
public struct TextRun: Sendable, Hashable, Codable {
    public var text: String
    public var bold: Bool
    public var italic: Bool
    public var underline: Bool
    public var strikethrough: Bool
    public var link: URL?

    public init(_ text: String, bold: Bool = false, italic: Bool = false, underline: Bool = false,
                strikethrough: Bool = false, link: URL? = nil) {
        self.text = text
        self.bold = bold
        self.italic = italic
        self.underline = underline
        self.strikethrough = strikethrough
        self.link = link
    }

    /// Same styles (so two neighbouring runs can merge).
    public func hasSameStyle(as other: TextRun) -> Bool {
        bold == other.bold && italic == other.italic && underline == other.underline
            && strikethrough == other.strikethrough && link == other.link
    }

    enum CodingKeys: String, CodingKey { case text = "t", bold = "b", italic = "i", underline = "u", strikethrough = "s", link }

    // Flags are written only when set, to keep long bodies small.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decode(String.self, forKey: .text)
        bold = try c.decodeIfPresent(Bool.self, forKey: .bold) ?? false
        italic = try c.decodeIfPresent(Bool.self, forKey: .italic) ?? false
        underline = try c.decodeIfPresent(Bool.self, forKey: .underline) ?? false
        strikethrough = try c.decodeIfPresent(Bool.self, forKey: .strikethrough) ?? false
        link = try c.decodeIfPresent(URL.self, forKey: .link)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(text, forKey: .text)
        if bold { try c.encode(true, forKey: .bold) }
        if italic { try c.encode(true, forKey: .italic) }
        if underline { try c.encode(true, forKey: .underline) }
        if strikethrough { try c.encode(true, forKey: .strikethrough) }
        try c.encodeIfPresent(link, forKey: .link)
    }
}

/// Runs with neighbours of the same style merged and empty runs dropped.
public func normalizedRuns(_ runs: [TextRun]) -> [TextRun] {
    var result: [TextRun] = []
    for run in runs where !run.text.isEmpty {
        if let last = result.last, last.hasSameStyle(as: run) {
            result[result.count - 1].text += run.text
        } else {
            result.append(run)
        }
    }
    return result
}

extension NoteDocument.Block: Codable {
    private enum Kind: String, Codable {
        case heading, paragraph, bullet, numbered, check, image, audio, transcriptAnchor, marker
    }

    private enum CodingKeys: String, CodingKey { case type, level, runs, indent, done, attachment, segment, time }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let runs = { try c.decodeIfPresent([TextRun].self, forKey: .runs) ?? [] }
        let indent = { try c.decodeIfPresent(Int.self, forKey: .indent) ?? 0 }
        // A block written by a newer build is kept as its text, not lost.
        guard let kind = try? c.decode(Kind.self, forKey: .type) else {
            self = .paragraph(runs: (try? runs()) ?? [])
            return
        }
        switch kind {
        case .heading:
            let raw = try c.decodeIfPresent(Int.self, forKey: .level) ?? 1
            self = .heading(level: NoteDocument.HeadingLevel(rawValue: raw) ?? .heading, runs: try runs())
        case .paragraph: self = .paragraph(runs: try runs())
        case .bullet: self = .bullet(runs: try runs(), indent: try indent())
        case .numbered: self = .numbered(runs: try runs(), indent: try indent())
        case .check:
            self = .check(runs: try runs(), done: try c.decodeIfPresent(Bool.self, forKey: .done) ?? false,
                          indent: try indent())
        case .image: self = .image(attachmentID: try c.decode(String.self, forKey: .attachment))
        case .audio: self = .audio(attachmentID: try c.decode(String.self, forKey: .attachment))
        case .transcriptAnchor: self = .transcriptAnchor(segmentID: try c.decode(UUID.self, forKey: .segment))
        case .marker: self = .marker(time: try c.decode(TimeInterval.self, forKey: .time))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        func put(_ runs: [TextRun]) throws { if !runs.isEmpty { try c.encode(runs, forKey: .runs) } }
        func put(indent: Int) throws { if indent != 0 { try c.encode(indent, forKey: .indent) } }
        switch self {
        case .heading(let level, let runs):
            try c.encode(Kind.heading, forKey: .type)
            try c.encode(level.rawValue, forKey: .level)
            try put(runs)
        case .paragraph(let runs):
            try c.encode(Kind.paragraph, forKey: .type)
            try put(runs)
        case .bullet(let runs, let indent):
            try c.encode(Kind.bullet, forKey: .type)
            try put(runs)
            try put(indent: indent)
        case .numbered(let runs, let indent):
            try c.encode(Kind.numbered, forKey: .type)
            try put(runs)
            try put(indent: indent)
        case .check(let runs, let done, let indent):
            try c.encode(Kind.check, forKey: .type)
            try put(runs)
            if done { try c.encode(true, forKey: .done) }
            try put(indent: indent)
        case .image(let id):
            try c.encode(Kind.image, forKey: .type)
            try c.encode(id, forKey: .attachment)
        case .audio(let id):
            try c.encode(Kind.audio, forKey: .type)
            try c.encode(id, forKey: .attachment)
        case .transcriptAnchor(let id):
            try c.encode(Kind.transcriptAnchor, forKey: .type)
            try c.encode(id, forKey: .segment)
        case .marker(let time):
            try c.encode(Kind.marker, forKey: .type)
            try c.encode(time, forKey: .time)
        }
    }
}
