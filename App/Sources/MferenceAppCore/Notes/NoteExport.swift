import Foundation

/// Exports a note (Notes spec 4.1): Markdown with its attachments in a
/// folder beside the file, or plain text. (PDF is drawn by the editor in
/// the UI layer, from the same document.)
public enum NoteExport {
    /// Times of transcript segments, for meeting notes' time chips.
    public typealias SegmentTimes = [UUID: TimeInterval]

    // MARK: Markdown

    /// The note as Markdown. Attachments link into `attachmentsFolder`
    /// (relative to the file); nil leaves them out (the share sheet).
    public static func markdown(_ document: NoteDocument, attachmentsFolder: String? = nil,
                                segmentTimes: SegmentTimes = [:]) -> String {
        var lines: [String] = []
        var number: [Int: Int] = [:]
        var previousWasList = false
        for block in document.blocks {
            let isList: Bool
            switch block {
            case .bullet, .numbered, .check: isList = true
            default: isList = false
            }
            if !isList { number = [:] }
            // A blank line separates blocks, but not items of one list.
            if !lines.isEmpty, !(isList && previousWasList) { lines.append("") }
            previousWasList = isList
            switch block {
            case .heading(let level, let runs):
                let hashes = String(repeating: "#", count: level.rawValue)
                lines.append("\(hashes) \(inline(runs))")
            case .paragraph(let runs):
                lines.append(inline(runs))
            case .bullet(let runs, let indent):
                lines.append(pad(indent) + "- " + inline(runs))
            case .numbered(let runs, let indent):
                number = number.filter { $0.key <= indent }
                let n = (number[indent] ?? 0) + 1
                number[indent] = n
                lines.append(pad(indent) + "\(n). " + inline(runs))
            case .check(let runs, let done, let indent):
                lines.append(pad(indent) + (done ? "- [x] " : "- [ ] ") + inline(runs))
            case .image(let id):
                if let folder = attachmentsFolder { lines.append("![](\(path(folder, id)))") }
            case .audio(let id):
                if let folder = attachmentsFolder { lines.append("[Audio: \(id)](\(path(folder, id)))") }
            case .transcriptAnchor(let id):
                lines.append("[\(clock(segmentTimes[id]))]")
            case .marker(let time):
                lines.append("> Marked at \(clock(time))")
            }
        }
        // Drop blank lines a left-out attachment leaves behind.
        var result: [String] = []
        for line in lines where !(line.isEmpty && (result.last?.isEmpty ?? true)) { result.append(line) }
        while result.last?.isEmpty == true { result.removeLast() }
        return result.joined(separator: "\n") + "\n"
    }

    /// Runs as Markdown inline text: **bold**, *italic*, <u>underline</u>,
    /// ~~strikethrough~~, [links](url). Markdown's own characters are escaped.
    public static func inline(_ runs: [TextRun]) -> String {
        normalizedRuns(runs).map { run in
            var text = escape(run.text)
            guard !text.trimmingCharacters(in: .whitespaces).isEmpty else { return text }
            // Keep the run's outer spaces outside the markers.
            let leading = String(text.prefix { $0 == " " })
            let trailing = String(text.reversed().prefix { $0 == " " })
            text = String(text.dropFirst(leading.count).dropLast(trailing.count))
            if run.strikethrough { text = "~~\(text)~~" }
            if run.underline { text = "<u>\(text)</u>" }
            if run.italic { text = "*\(text)*" }
            if run.bold { text = "**\(text)**" }
            if let link = run.link { text = "[\(text)](\(link.absoluteString))" }
            return leading + text + trailing
        }.joined()
    }

    // MARK: Plain text

    /// The note as plain text: bullets as "•", numbers, checklists as ☐ / ☑.
    public static func plainText(_ document: NoteDocument, segmentTimes: SegmentTimes = [:]) -> String {
        var lines: [String] = []
        var number: [Int: Int] = [:]
        for block in document.blocks {
            if case .numbered = block {} else { number = [:] }
            switch block {
            case .heading(_, let runs), .paragraph(let runs):
                lines.append(runs.map(\.text).joined())
            case .bullet(let runs, let indent):
                lines.append(pad(indent) + "• " + runs.map(\.text).joined())
            case .numbered(let runs, let indent):
                number = number.filter { $0.key <= indent }
                let n = (number[indent] ?? 0) + 1
                number[indent] = n
                lines.append(pad(indent) + "\(n). " + runs.map(\.text).joined())
            case .check(let runs, let done, let indent):
                lines.append(pad(indent) + (done ? "☑ " : "☐ ") + runs.map(\.text).joined())
            case .image, .audio:
                break
            case .transcriptAnchor(let id):
                lines.append("[\(clock(segmentTimes[id]))]")
            case .marker(let time):
                lines.append("Marked at \(clock(time))")
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    // MARK: Files

    /// A file name for the note: its title with characters a file system
    /// refuses replaced, or "Note".
    public static func fileName(for title: String) -> String {
        let cleaned = title.map { "/:\\?%*|\"<>".contains($0) || $0.isNewline ? "-" : $0 }
        let name = String(cleaned).trimmingCharacters(in: .whitespacesAndNewlines)
        return name.isEmpty ? "Note" : String(name.prefix(120))
    }

    /// Writes `<file>.md` and, when the note has attachments, the folder
    /// "`<file> attachments`" beside it with copies of them. Returns the
    /// files written.
    @discardableResult
    public static func writeMarkdown(_ document: NoteDocument, noteID: UUID, attachments: NoteAttachments,
                                     to file: URL, segmentTimes: SegmentTimes = [:]) throws -> [URL] {
        let base = file.deletingPathExtension().lastPathComponent
        let folderName = "\(base) attachments"
        let ids = document.attachmentIDs.filter {
            FileManager.default.fileExists(atPath: attachments.url(of: $0, in: noteID).path(percentEncoded: false))
        }
        var written: [URL] = []
        if !ids.isEmpty {
            let folder = file.deletingLastPathComponent().appending(path: folderName, directoryHint: .isDirectory)
            if FileManager.default.fileExists(atPath: folder.path(percentEncoded: false)) {
                try FileManager.default.removeItem(at: folder)
            }
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for id in ids {
                let target = folder.appending(path: id)
                try FileManager.default.copyItem(at: attachments.url(of: id, in: noteID), to: target)
                written.append(target)
            }
        }
        var kept = document
        kept.blocks.removeAll { block in block.attachmentID.map { !ids.contains($0) } ?? false }
        let text = markdown(kept, attachmentsFolder: ids.isEmpty ? nil : folderName, segmentTimes: segmentTimes)
        try Data(text.utf8).write(to: file, options: .atomic)
        written.insert(file, at: 0)
        return written
    }

    public static func writePlainText(_ document: NoteDocument, to file: URL, segmentTimes: SegmentTimes = [:]) throws {
        try Data(plainText(document, segmentTimes: segmentTimes).utf8).write(to: file, options: .atomic)
    }

    // MARK: Helpers

    /// "12:34", or "1:02:03" past an hour; "–:––" when unknown.
    public static func clock(_ seconds: TimeInterval?) -> String {
        guard let seconds else { return "–:––" }
        let total = max(0, Int(seconds.rounded(.down)))
        let (h, m, s) = (total / 3600, total / 60 % 60, total % 60)
        return h > 0 ? String(format: "%d:%02d:%02d", h, m, s) : String(format: "%d:%02d", m, s)
    }

    private static func pad(_ indent: Int) -> String { String(repeating: "  ", count: max(0, indent)) }

    private static func path(_ folder: String, _ id: String) -> String {
        "<\(folder)/\(id)>"
    }

    private static func escape(_ text: String) -> String {
        var out = ""
        for character in text {
            if "\\`*_[]<>~#".contains(character) { out.append("\\") }
            out.append(character)
        }
        return out
    }
}
