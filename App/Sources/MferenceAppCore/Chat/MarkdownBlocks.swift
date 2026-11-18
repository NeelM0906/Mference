import Foundation

/// Block-level Markdown for answers: paragraphs, headings, lists, code,
/// quotes and pipe tables (mockup 02 shows a table). Inline styling (bold,
/// links, code spans) is left to `AttributedString(markdown:)` in the view.
public enum MarkdownBlock: Sendable, Hashable {
    case paragraph(String)
    case heading(level: Int, text: String)
    case bulletList([String])
    case orderedList(start: Int, items: [String])
    case code(language: String?, text: String)
    case quote(String)
    case table(header: [String], rows: [[String]])
    case rule

    /// Parses `text`. Unterminated code fences (mid-stream) run to the end.
    public static func parse(_ text: String) -> [MarkdownBlock] {
        let lines = text.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var i = 0

        func flushParagraph() {
            let joined = paragraph.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !joined.isEmpty { blocks.append(.paragraph(joined)) }
            paragraph = []
        }

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.hasPrefix("```") {
                flushParagraph()
                let language = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var code: [String] = []
                i += 1
                while i < lines.count, !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    code.append(lines[i])
                    i += 1
                }
                blocks.append(.code(language: language.isEmpty ? nil : language, text: code.joined(separator: "\n")))
                i += 1
                continue
            }
            if trimmed.isEmpty {
                flushParagraph()
                i += 1
                continue
            }
            if let heading = Self.heading(trimmed) {
                flushParagraph()
                blocks.append(heading)
                i += 1
                continue
            }
            if trimmed == "---" || trimmed == "***" || trimmed == "___" {
                flushParagraph()
                blocks.append(.rule)
                i += 1
                continue
            }
            if trimmed.hasPrefix("|"), i + 1 < lines.count, Self.isTableDivider(lines[i + 1]) {
                flushParagraph()
                let header = Self.cells(trimmed)
                var rows: [[String]] = []
                i += 2
                while i < lines.count, lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("|") {
                    var row = Self.cells(lines[i].trimmingCharacters(in: .whitespaces))
                    if row.count < header.count { row += Array(repeating: "", count: header.count - row.count) }
                    rows.append(Array(row.prefix(header.count)))
                    i += 1
                }
                blocks.append(.table(header: header, rows: rows))
                continue
            }
            if Self.bullet(trimmed) != nil {
                flushParagraph()
                var items: [String] = []
                while i < lines.count, let item = Self.bullet(lines[i].trimmingCharacters(in: .whitespaces)) {
                    items.append(item)
                    i += 1
                }
                blocks.append(.bulletList(items))
                continue
            }
            if let first = Self.ordered(trimmed) {
                flushParagraph()
                var items: [String] = []
                while i < lines.count, let entry = Self.ordered(lines[i].trimmingCharacters(in: .whitespaces)) {
                    items.append(entry.1)
                    i += 1
                }
                blocks.append(.orderedList(start: first.0, items: items))
                continue
            }
            if trimmed.hasPrefix(">") {
                flushParagraph()
                var quoted: [String] = []
                while i < lines.count, lines[i].trimmingCharacters(in: .whitespaces).hasPrefix(">") {
                    quoted.append(String(lines[i].trimmingCharacters(in: .whitespaces).dropFirst())
                        .trimmingCharacters(in: .whitespaces))
                    i += 1
                }
                blocks.append(.quote(quoted.joined(separator: "\n")))
                continue
            }
            paragraph.append(line)
            i += 1
        }
        flushParagraph()
        return blocks
    }

    static func heading(_ line: String) -> MarkdownBlock? {
        let hashes = line.prefix { $0 == "#" }.count
        guard (1...6).contains(hashes), line.dropFirst(hashes).first == " " else { return nil }
        return .heading(level: hashes, text: String(line.dropFirst(hashes + 1)))
    }

    static func bullet(_ line: String) -> String? {
        for marker in ["- ", "* ", "+ ", "• "] where line.hasPrefix(marker) {
            return String(line.dropFirst(marker.count))
        }
        return nil
    }

    static func ordered(_ line: String) -> (Int, String)? {
        let digits = line.prefix { $0.isNumber }
        guard !digits.isEmpty, digits.count <= 3, let number = Int(digits) else { return nil }
        let rest = line.dropFirst(digits.count)
        guard rest.hasPrefix(". ") || rest.hasPrefix(") ") else { return nil }
        return (number, String(rest.dropFirst(2)))
    }

    static func isTableDivider(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("|"), trimmed.contains("-") else { return false }
        return trimmed.allSatisfy { "|-: ".contains($0) }
    }

    static func cells(_ line: String) -> [String] {
        var body = Substring(line)
        if body.hasPrefix("|") { body = body.dropFirst() }
        if body.hasSuffix("|") { body = body.dropLast() }
        return body.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }
}

/// Inline citation markers ("[2]", "[1][3]", "[1, 3]") in answer text.
public enum CitationMarkers {
    /// A run of plain text or one cited source number.
    public enum Segment: Sendable, Hashable {
        case text(String)
        case citation(Int)
    }

    /// Splits `text` into text and citations. Only numbers in `valid` become
    /// citations (a model may write "[5]" for a source it never opened);
    /// others stay as text.
    public static func segments(_ text: String, valid: Set<Int>) -> [Segment] {
        var segments: [Segment] = []
        var buffer = ""
        var index = text.startIndex
        while index < text.endIndex {
            if text[index] == "[", let close = text[index...].firstIndex(of: "]"),
               text.distance(from: index, to: close) <= 16 {
                let inner = text[text.index(after: index)..<close]
                let numbers = inner.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
                let isLink = text.index(after: close) < text.endIndex && text[text.index(after: close)] == "("
                if !numbers.isEmpty, !isLink, numbers.allSatisfy({ $0.map(valid.contains) ?? false }) {
                    if !buffer.isEmpty { segments.append(.text(buffer)); buffer = "" }
                    segments += numbers.compactMap { $0.map(Segment.citation) }
                    index = text.index(after: close)
                    continue
                }
            }
            buffer.append(text[index])
            index = text.index(after: index)
        }
        if !buffer.isEmpty { segments.append(.text(buffer)) }
        return segments
    }
}
