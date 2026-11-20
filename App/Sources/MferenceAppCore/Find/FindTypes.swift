import Foundation

// Find: local, content-based file search (design canvas locked 2026-11-19).
// Everything here runs on this Mac; nothing in Find reaches the network
// except the one pinned download of the search model.

/// What kind of file a result is, for the kind filters and the result tile.
public enum FindKind: String, Sendable, Hashable, Codable, CaseIterable, Identifiable {
    case document, pdf, image, slides, sheet, code, text, video, audio

    public var id: String { rawValue }

    /// The filter a kind belongs to.
    public var filter: FindKindFilter {
        switch self {
        case .document, .pdf, .text: .documents
        case .image: .images
        case .slides, .sheet: .slidesAndSheets
        case .code: .code
        case .video, .audio: .media
        }
    }

    /// Video and audio: searched by name, and by what they show and sound
    /// like when Settings › Find › Photos, video and audio is on.
    public var isMedia: Bool { self == .video || self == .audio }

    /// Kind of a file by its extension; nil for files Find does not read.
    public static func of(_ url: URL) -> FindKind? {
        of(extension: url.pathExtension)
    }

    public static func of(extension ext: String) -> FindKind? {
        let ext = ext.lowercased()
        if ext == "pdf" { return .pdf }
        if documentExtensions.contains(ext) { return .document }
        if imageExtensions.contains(ext) { return .image }
        if slideExtensions.contains(ext) { return .slides }
        if sheetExtensions.contains(ext) { return .sheet }
        if codeExtensions.contains(ext) { return .code }
        if textExtensions.contains(ext) { return .text }
        if videoExtensions.contains(ext) { return .video }
        if audioExtensions.contains(ext) { return .audio }
        return nil
    }

    static let documentExtensions: Set<String> = ["docx", "doc", "rtf", "rtfd", "odt", "html", "htm", "webarchive", "pages"]
    static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "heic", "heif", "tiff", "tif", "gif", "webp", "bmp"]
    static let slideExtensions: Set<String> = ["pptx", "key", "odp"]
    static let sheetExtensions: Set<String> = ["xlsx", "numbers", "ods", "csv", "tsv"]
    static let textExtensions: Set<String> = ["txt", "md", "markdown", "text", "org", "rst", "tex", "log", "eml"]
    /// What AVFoundation plays on a Mac (no MKV, AVI or WebM).
    static let videoExtensions: Set<String> = ["mov", "mp4", "m4v", "3gp"]
    static let audioExtensions: Set<String> = ["mp3", "m4a", "wav", "aiff", "aif", "aac", "caf", "flac", "m4b"]
    static let codeExtensions: Set<String> = [
        "swift", "m", "mm", "h", "c", "cc", "cpp", "hpp", "rs", "go", "py", "js", "jsx", "ts", "tsx", "java",
        "kt", "rb", "php", "cs", "sh", "zsh", "bash", "sql", "lua", "metal", "scala", "r", "dart", "vue",
        "svelte", "css", "scss", "json", "yaml", "yml", "toml", "xml", "gradle", "cmake", "mk",
    ]
}

/// The kind filters in Find (canvas boards 1 and 2).
public enum FindKindFilter: String, Sendable, Hashable, Codable, CaseIterable, Identifiable {
    case all, documents, images, slidesAndSheets, media, code

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .all: "All"
        case .documents: "Documents"
        case .images: "Images"
        case .slidesAndSheets: "Slides & sheets"
        case .media: "Video & audio"
        case .code: "Code"
        }
    }

    public func includes(_ kind: FindKind) -> Bool {
        self == .all || kind.filter == self
    }
}

/// When a file was last changed, for the date filter.
public enum FindDateFilter: String, Sendable, Hashable, Codable, CaseIterable, Identifiable {
    case anyTime, thisYear, last30Days

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .anyTime: "Any time"
        case .thisYear: "This year"
        case .last30Days: "Last 30 days"
        }
    }

    /// The earliest modification date that passes, nil for any time.
    public func earliest(now: Date, calendar: Calendar = .current) -> Date? {
        switch self {
        case .anyTime: nil
        case .thisYear: calendar.date(from: calendar.dateComponents([.year], from: now))
        case .last30Days: calendar.date(byAdding: .day, value: -30, to: now)
        }
    }
}

/// Why a result matched; each result shows its reasons as chips.
public enum FindMatchReason: String, Sendable, Hashable, Codable, CaseIterable {
    /// The passage means what the query asks, whatever its words.
    case meaning
    /// The query's words are in the file.
    case words
    /// The query's words are in the file name.
    case name
    /// The words were read from an image (Vision OCR).
    case textInImage
    /// What the picture (or a video frame) shows matches the query.
    case picture
    /// What the audio (or a video's soundtrack) sounds like matches.
    case sound

    public var title: String {
        switch self {
        case .meaning: "Meaning"
        case .words: "Words"
        case .name: "Name"
        case .textInImage: "Text in image"
        case .picture: "In the picture"
        case .sound: "In the sound"
        }
    }
}

/// One search request.
public struct FindQuery: Sendable, Hashable {
    public var text: String
    public var kind: FindKindFilter
    public var modified: FindDateFilter
    /// Only files under this folder; nil for every indexed folder.
    public var folder: URL?
    public var limit: Int

    public init(_ text: String, kind: FindKindFilter = .all, modified: FindDateFilter = .anyTime,
                folder: URL? = nil, limit: Int = 30) {
        self.text = text
        self.kind = kind
        self.modified = modified
        self.folder = folder
        self.limit = limit
    }
}

/// One file in the results, with the passage that matched.
public struct FindResult: Sendable, Hashable, Identifiable {
    public var url: URL
    public var kind: FindKind
    public var modified: Date
    /// Where in the file: "Page 4", "Slide 7", "Lines 212–221", nil when the
    /// whole file is one passage.
    public var location: String?
    /// The matched passage, split so the hit can be highlighted.
    public var before: String
    public var hit: String
    public var after: String
    public var reasons: [FindMatchReason]
    public var score: Double
    /// Below half the top result's score: shown folded ("weaker matches").
    public var isWeak: Bool
    /// Other paths with exactly the same content (collapsed copies).
    public var copies: Int
    /// The whole chunk the hit came from, for the preview pane.
    public var passage: String
    /// Seconds into a video or audio file where the match is (a frame, or
    /// the start of a stretch of sound); the preview plays from there.
    public var time: Double?

    public var id: URL { url }
    public var name: String { url.lastPathComponent }

    public init(url: URL, kind: FindKind, modified: Date, location: String?, before: String, hit: String,
                after: String, reasons: [FindMatchReason], score: Double, isWeak: Bool = false, copies: Int = 0,
                passage: String = "", time: Double? = nil) {
        self.url = url
        self.kind = kind
        self.modified = modified
        self.location = location
        self.before = before
        self.hit = hit
        self.after = after
        self.reasons = reasons
        self.score = score
        self.isWeak = isWeak
        self.copies = copies
        self.passage = passage
        self.time = time
    }
}

/// What a search returned.
public struct FindResults: Sendable, Hashable {
    public var query: String
    public var results: [FindResult]
    /// Meaning was not used (the model is not ready yet, or still learning
    /// most files); the UI says word matches only.
    public var wordsOnly: Bool
    /// Seconds the search took.
    public var seconds: Double

    public init(query: String, results: [FindResult], wordsOnly: Bool, seconds: Double) {
        self.query = query
        self.results = results
        self.wordsOnly = wordsOnly
        self.seconds = seconds
    }

    public static let empty = FindResults(query: "", results: [], wordsOnly: false, seconds: 0)

    /// Results above the weak fold.
    public var strong: [FindResult] { results.filter { !$0.isWeak } }
    public var weak: [FindResult] { results.filter(\.isWeak) }
}

/// Where indexing stands, for the progress banner and Settings › Find.
public struct FindIndexStatus: Sendable, Hashable {
    public enum Phase: String, Sendable, Hashable {
        /// Find is off.
        case off
        /// Looking through folders for new and changed files.
        case scanning
        /// Reading text out of files (words become searchable).
        case reading
        /// Learning meaning (embedding passages).
        case learning
        /// Waiting: on battery, or Chat is answering.
        case paused
        case upToDate
    }

    public var phase: Phase
    public var totalFiles: Int
    /// Files whose words are searchable.
    public var readFiles: Int
    /// Files whose meaning is fully learned.
    public var learnedFiles: Int
    /// Files Find could not read (encrypted, damaged, too large).
    public var skippedFiles: Int
    /// Why learning is paused, shown in the banner.
    public var pauseReason: String?
    public var indexBytes: UInt64

    public init(phase: Phase = .off, totalFiles: Int = 0, readFiles: Int = 0, learnedFiles: Int = 0,
                skippedFiles: Int = 0, pauseReason: String? = nil, indexBytes: UInt64 = 0) {
        self.phase = phase
        self.totalFiles = totalFiles
        self.readFiles = readFiles
        self.learnedFiles = learnedFiles
        self.skippedFiles = skippedFiles
        self.pauseReason = pauseReason
        self.indexBytes = indexBytes
    }

    /// 0...1 of meaning learned.
    public var learnedFraction: Double {
        totalFiles > 0 ? min(1, Double(learnedFiles) / Double(totalFiles)) : 0
    }

    /// Files still to learn.
    public var remainingFiles: Int { max(0, totalFiles - learnedFiles - skippedFiles) }
}

/// A file Find could not read, and why (Settings › Find, "See the list").
public struct FindSkippedFile: Sendable, Hashable, Identifiable {
    public var url: URL
    public var reason: String
    public var id: URL { url }

    public init(url: URL, reason: String) {
        self.url = url
        self.reason = reason
    }
}
