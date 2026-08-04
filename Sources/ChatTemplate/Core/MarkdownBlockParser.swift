import Foundation

public enum MarkdownBlock: Equatable, Sendable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case codeBlock(language: String?, code: String)
    case bulletList(items: [String])
    case orderedList(items: [String])
    case quote(String)
    case table(header: [String], rows: [[String]])
    case divider
}

/// Block-level markdown parser for chat rendering, mirroring the template's
/// custom mdast renderer feature set: fenced code, GFM tables, headings,
/// lists, quotes, and paragraphs. Tolerates an unclosed code fence so
/// streaming output renders sensibly mid-generation.
public enum MarkdownBlockParser {
    public static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        let lines = text.components(separatedBy: "\n")
        var index = 0

        func flushParagraph() {
            if !paragraph.isEmpty {
                blocks.append(.paragraph(paragraph.joined(separator: "\n")))
                paragraph = []
            }
        }

        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.hasPrefix("```") {
                flushParagraph()
                let language = String(trimmed.dropFirst(3))
                    .trimmingCharacters(in: .whitespaces)
                var code: [String] = []
                index += 1
                while index < lines.count,
                      !lines[index].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    code.append(lines[index])
                    index += 1
                }
                index += 1 // skip closing fence (or run past EOF if unclosed)
                blocks.append(.codeBlock(
                    language: language.isEmpty ? nil : language,
                    code: code.joined(separator: "\n")))
                continue
            }

            if trimmed.isEmpty {
                flushParagraph()
                index += 1
                continue
            }

            if let level = headingLevel(of: trimmed) {
                flushParagraph()
                let text = trimmed.drop(while: { $0 == "#" })
                    .trimmingCharacters(in: .whitespaces)
                blocks.append(.heading(level: level, text: text))
                index += 1
                continue
            }

            if isDivider(trimmed) {
                flushParagraph()
                blocks.append(.divider)
                index += 1
                continue
            }

            if trimmed.contains("|"),
               index + 1 < lines.count,
               isTableSeparator(lines[index + 1]) {
                flushParagraph()
                let header = tableCells(trimmed)
                index += 2
                var rows: [[String]] = []
                while index < lines.count {
                    let rowLine = lines[index].trimmingCharacters(in: .whitespaces)
                    guard rowLine.contains("|") else { break }
                    rows.append(tableCells(rowLine))
                    index += 1
                }
                blocks.append(.table(header: header, rows: rows))
                continue
            }

            if let item = bulletItem(trimmed) {
                flushParagraph()
                var items = [item]
                index += 1
                while index < lines.count,
                      let next = bulletItem(lines[index].trimmingCharacters(in: .whitespaces)) {
                    items.append(next)
                    index += 1
                }
                blocks.append(.bulletList(items: items))
                continue
            }

            if let item = orderedItem(trimmed) {
                flushParagraph()
                var items = [item]
                index += 1
                while index < lines.count,
                      let next = orderedItem(lines[index].trimmingCharacters(in: .whitespaces)) {
                    items.append(next)
                    index += 1
                }
                blocks.append(.orderedList(items: items))
                continue
            }

            if trimmed.hasPrefix(">") {
                flushParagraph()
                var quoted: [String] = []
                while index < lines.count {
                    let quoteLine = lines[index].trimmingCharacters(in: .whitespaces)
                    guard quoteLine.hasPrefix(">") else { break }
                    quoted.append(
                        quoteLine.dropFirst()
                            .trimmingCharacters(in: .whitespaces))
                    index += 1
                }
                blocks.append(.quote(quoted.joined(separator: "\n")))
                continue
            }

            paragraph.append(line)
            index += 1
        }

        flushParagraph()
        return blocks
    }

    private static func headingLevel(of line: String) -> Int? {
        let hashes = line.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(hashes) else { return nil }
        let rest = line.dropFirst(hashes)
        guard rest.first == " " else { return nil }
        return hashes
    }

    private static func isDivider(_ line: String) -> Bool {
        guard line.count >= 3 else { return false }
        for marker: Character in ["-", "*", "_"] where line.allSatisfy({ $0 == marker }) {
            return true
        }
        return false
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.contains("-"), trimmed.contains("|") else { return false }
        return trimmed.allSatisfy { "-:| ".contains($0) }
    }

    private static func tableCells(_ line: String) -> [String] {
        var cells = line.components(separatedBy: "|")
            .map { $0.trimmingCharacters(in: .whitespaces) }
        if cells.first?.isEmpty == true { cells.removeFirst() }
        if cells.last?.isEmpty == true { cells.removeLast() }
        return cells
    }

    private static func bulletItem(_ line: String) -> String? {
        for prefix in ["- ", "* ", "+ "] where line.hasPrefix(prefix) {
            return String(line.dropFirst(prefix.count))
        }
        return nil
    }

    private static func orderedItem(_ line: String) -> String? {
        let digits = line.prefix(while: \.isNumber)
        guard !digits.isEmpty else { return nil }
        let rest = line.dropFirst(digits.count)
        guard rest.hasPrefix(". ") || rest.hasPrefix(") ") else { return nil }
        return String(rest.dropFirst(2))
    }
}
