import AppKit
import MferenceAppCore

/// Turns pasted rich text into the editor's canonical form (Notes spec 4.1:
/// "paste keeps the supported formats and drops the rest"): bold, italic,
/// underline, strikethrough and links survive; lists become bullets or
/// numbers; fonts, colors, sizes and everything else go.
nonisolated enum NotePasteSanitizer {
    static func sanitize(_ pboard: NSPasteboard, type: NSPasteboard.PasteboardType, base: NoteBlockStyle) -> NSAttributedString? {
        if type == .string {
            guard let string = pboard.string(forType: .string) else { return nil }
            return sanitize(NSAttributedString(string: string), base: base, plain: true)
        }
        guard let data = pboard.data(forType: type) else { return nil }
        let documentType: NSAttributedString.DocumentType = switch type {
        case .rtfd: .rtfd
        case .html: .html
        default: .rtf
        }
        guard let rich = try? NSAttributedString(data: data, options: [.documentType: documentType],
                                                 documentAttributes: nil) else { return nil }
        return sanitize(rich, base: base, plain: false)
    }

    /// `base` is the style of the line pasted into: plain text lines take it,
    /// so pasting lines into a checklist makes checklist items.
    static func sanitize(_ source: NSAttributedString, base: NoteBlockStyle, plain: Bool) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let string = source.string as NSString
        var location = 0
        while location < string.length {
            let line = string.lineRange(for: NSRange(location: location, length: 0))
            var content = line
            let hasBreak = content.length > 0 && [0x0A, 0x0D, 0x2029].contains(string.character(at: NSMaxRange(content) - 1))
            if hasBreak { content.length -= 1 }
            let paragraph = line.length > 0 ? source.attribute(.paragraphStyle, at: line.location, effectiveRange: nil) as? NSParagraphStyle : nil
            var style = plain ? base : NoteBlockStyle.body
            if let lists = paragraph?.textLists, let list = lists.last {
                let numbered = list.markerFormat.rawValue.contains("decimal") || list.markerFormat.rawValue.contains("roman")
                    || list.markerFormat.rawValue.contains("alpha")
                style = NoteBlockStyle(numbered ? .numbered : .bullet, indent: lists.count - 1)
                content = stripListMarker(string, content)
            }
            let start = result.length
            source.enumerateAttributes(in: content) { attributes, range, _ in
                guard attributes[.attachment] == nil else { return }
                let text = string.substring(with: range).replacingOccurrences(of: "\u{FFFC}", with: "")
                guard !text.isEmpty else { return }
                var clean: [NSAttributedString.Key: Any] = [.noteBlock: style]
                if let font = attributes[.font] as? NSFont {
                    let traits = font.fontDescriptor.symbolicTraits
                    if traits.contains(.bold) { clean[.noteBold] = true }
                    if traits.contains(.italic) { clean[.noteItalic] = true }
                }
                if (attributes[.underlineStyle] as? Int ?? 0) != 0, attributes[.link] == nil {
                    clean[.underlineStyle] = NSUnderlineStyle.single.rawValue
                }
                if (attributes[.strikethroughStyle] as? Int ?? 0) != 0 {
                    clean[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
                }
                if let url = attributes[.link] as? URL { clean[.link] = url }
                if let link = attributes[.link] as? String, let url = URL(string: link) { clean[.link] = url }
                result.append(NSAttributedString(string: text, attributes: clean))
            }
            if hasBreak {
                result.append(NSAttributedString(string: "\n", attributes: [.noteBlock: style]))
            }
            if result.length > start {
                result.addAttribute(.noteBlock, value: style, range: NSRange(location: start, length: result.length - start))
            }
            location = NSMaxRange(line)
        }
        return result
    }

    /// RTF lists carry their markers as text ("\t•\t", "\t1.\t").
    private static func stripListMarker(_ string: NSString, _ range: NSRange) -> NSRange {
        let text = string.substring(with: range)
        guard text.hasPrefix("\t"), let second = text.dropFirst().firstIndex(of: "\t") else { return range }
        let markerLength = text.distance(from: text.startIndex, to: second) + 1
        let utf16 = (String(text[text.startIndex..<text.index(text.startIndex, offsetBy: markerLength)]) as NSString).length
        return NSRange(location: range.location + utf16, length: range.length - utf16)
    }
}
