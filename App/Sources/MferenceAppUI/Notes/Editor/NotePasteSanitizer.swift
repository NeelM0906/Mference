import AppKit
import MferenceAppCore
import UniformTypeIdentifiers

/// Turns pasted rich text into the editor's canonical form (Notes spec 4.1:
/// "paste keeps the supported formats and drops the rest"): bold, italic,
/// underline, strikethrough and links survive; lists become bullets or
/// numbers; images in RTFD become attachments; fonts, colors, sizes and
/// everything else go. HTML is read by `NoteHTMLReader`, never by WebKit,
/// so a paste can never load anything from the network.
nonisolated enum NotePasteSanitizer {
    /// Skips text dragged from the Notes list (note and folder ids).
    static let internalDragPrefixes = ["mference-note:", "mference-folder:"]

    static func sanitize(_ pboard: NSPasteboard, type: NSPasteboard.PasteboardType, base: NoteBlockStyle,
                         importImage: ((Data, String) -> String?)? = nil,
                         context: NoteTextMapping.Context = .init()) -> NSAttributedString? {
        switch type {
        case .string:
            guard let string = pboard.string(forType: .string),
                  !internalDragPrefixes.contains(where: string.hasPrefix) else { return nil }
            return sanitize(NSAttributedString(string: string), base: base, plain: true)
        case .html:
            guard let html = pboard.string(forType: .html)
                    ?? pboard.data(forType: .html).flatMap({ String(data: $0, encoding: .utf8) }) else { return nil }
            return sanitize(NoteHTMLReader.attributedString(html), base: base, plain: false)
        case .rtf, .rtfd:
            guard let data = pboard.data(forType: type) else { return nil }
            let documentType: NSAttributedString.DocumentType = type == .rtfd ? .rtfd : .rtf
            guard let rich = try? NSAttributedString(data: data, options: [.documentType: documentType],
                                                     documentAttributes: nil) else { return nil }
            return sanitize(rich, base: base, plain: false, importImage: importImage, context: context)
        default:
            return nil
        }
    }

    /// `base` is the style of the line pasted into: plain text lines take it
    /// (unticked), so pasting lines into a checklist makes new items.
    static func sanitize(_ source: NSAttributedString, base: NoteBlockStyle, plain: Bool,
                         importImage: ((Data, String) -> String?)? = nil,
                         context: NoteTextMapping.Context = .init()) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let string = source.string as NSString
        let plainStyle = base.with(done: false)
        var location = 0
        while location < string.length {
            let line = string.lineRange(for: NSRange(location: location, length: 0))
            var content = line
            var hasBreak = false
            while content.length > 0, NoteTextMapping.isLineTerminator(string.character(at: NSMaxRange(content) - 1)) {
                content.length -= 1
                hasBreak = true
            }
            let paragraph = line.length > 0 ? source.attribute(.paragraphStyle, at: line.location, effectiveRange: nil) as? NSParagraphStyle : nil
            var style = plain ? plainStyle : NoteBlockStyle.body
            if let lists = paragraph?.textLists, let list = lists.last {
                let format = list.markerFormat.rawValue
                let numbered = format.contains("decimal") || format.contains("roman") || format.contains("alpha")
                style = NoteBlockStyle(numbered ? .numbered : .bullet, indent: lists.count - 1)
                content = stripListMarker(string, content)
            }
            let start = result.length
            source.enumerateAttributes(in: content) { attributes, range, _ in
                if let attachment = attributes[.attachment] as? NSTextAttachment {
                    appendImage(attachment, to: result, importImage: importImage, context: context)
                    return
                }
                let text = string.substring(with: range).replacingOccurrences(of: "\u{FFFC}", with: "")
                guard !text.isEmpty else { return }
                result.append(NSAttributedString(string: text, attributes: clean(attributes, style: style)))
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

    private static func clean(_ attributes: [NSAttributedString.Key: Any], style: NoteBlockStyle) -> [NSAttributedString.Key: Any] {
        var clean: [NSAttributedString.Key: Any] = [.noteBlock: style]
        if let font = attributes[.font] as? NSFont {
            let traits = font.fontDescriptor.symbolicTraits
            if traits.contains(.bold) { clean[.noteBold] = true }
            if traits.contains(.italic) { clean[.noteItalic] = true }
        }
        if attributes[.noteBold] != nil { clean[.noteBold] = true }
        if attributes[.noteItalic] != nil { clean[.noteItalic] = true }
        if (attributes[.underlineStyle] as? Int ?? 0) != 0, attributes[.link] == nil {
            clean[.underlineStyle] = NSUnderlineStyle.single.rawValue
        }
        if (attributes[.strikethroughStyle] as? Int ?? 0) != 0 {
            clean[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
        }
        let link = (attributes[.link] as? URL) ?? (attributes[.link] as? String).flatMap(URL.init(string:))
        if let link, ["http", "https", "mailto"].contains(link.scheme?.lowercased() ?? "") { clean[.link] = link }
        return clean
    }

    /// An image inside pasted RTFD becomes an image line of its own.
    private static func appendImage(_ attachment: NSTextAttachment, to result: NSMutableAttributedString,
                                    importImage: ((Data, String) -> String?)?, context: NoteTextMapping.Context) {
        guard let importImage else { return }
        var data = attachment.fileWrapper?.regularFileContents ?? attachment.contents
        var ext = (attachment.fileWrapper?.preferredFilename as NSString?)?.pathExtension.lowercased() ?? ""
        if let found = data, UTType(filenameExtension: ext)?.conforms(to: .image) != true {
            // Not a known image file: keep it only if it decodes as one.
            guard let png = NSBitmapImageRep(data: found)?.representation(using: .png, properties: [:]) else { return }
            data = png
            ext = "png"
        }
        guard let data, let id = importImage(data, ext.isEmpty ? "png" : ext) else { return }
        if result.length > 0, !(result.string.hasSuffix("\n")) {
            result.append(NSAttributedString(string: "\n", attributes: [.noteBlock: NoteBlockStyle.body]))
        }
        result.append(NoteTextMapping.attributedString(from: NoteDocument(blocks: [.image(attachmentID: id)]),
                                                       context: context).text)
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

/// A small HTML reader for paste: text, line structure, bold, italic,
/// underline, strikethrough, links and lists. It never fetches anything
/// (no WebKit, no images, no styles); scripts and styles are dropped.
nonisolated enum NoteHTMLReader {
    static func attributedString(_ html: String) -> NSAttributedString {
        let result = NSMutableAttributedString()
        var bold = 0, italic = 0, underline = 0, strike = 0
        var links: [URL?] = []
        var lists: [(ordered: Bool, count: Int)] = []
        var skipping: String?
        var pendingBreak = false
        var lineStyle: NoteBlockStyle?

        func breakLine() {
            if result.length > 0, !result.string.hasSuffix("\n") { pendingBreak = true }
        }
        func emit(_ text: String) {
            guard !text.isEmpty else { return }
            var piece = text
            if pendingBreak {
                result.append(NSAttributedString(string: "\n"))
                pendingBreak = false
                piece = String(piece.drop { $0 == " " })
                guard !piece.isEmpty else { return }
            }
            var attributes: [NSAttributedString.Key: Any] = [:]
            if bold > 0 { attributes[.noteBold] = true }
            if italic > 0 { attributes[.noteItalic] = true }
            if underline > 0 { attributes[.underlineStyle] = NSUnderlineStyle.single.rawValue }
            if strike > 0 { attributes[.strikethroughStyle] = NSUnderlineStyle.single.rawValue }
            if let link = links.last(where: { $0 != nil }) ?? nil { attributes[.link] = link }
            if let lineStyle {
                attributes[.noteBlock] = lineStyle
                let paragraph = NSMutableParagraphStyle()
                let list = NSTextList(markerFormat: lineStyle.kind == .numbered ? .decimal : .disc, options: 0)
                paragraph.textLists = Array(repeating: list, count: lineStyle.indent + 1)
                attributes[.paragraphStyle] = paragraph
            }
            result.append(NSAttributedString(string: piece, attributes: attributes))
        }

        var index = html.startIndex
        while index < html.endIndex {
            if html[index] == "<", let close = html[index...].firstIndex(of: ">") {
                let raw = html[html.index(after: index)..<close]
                index = html.index(after: close)
                let isEnd = raw.hasPrefix("/")
                let body = raw.dropFirst(isEnd ? 1 : 0)
                let name = String(body.prefix { $0.isLetter || $0.isNumber }).lowercased()
                if let skip = skipping {
                    if isEnd, name == skip { skipping = nil }
                    continue
                }
                switch (name, isEnd) {
                case ("script", false), ("style", false), ("head", false), ("title", false):
                    skipping = name
                case ("b", _), ("strong", _): bold += isEnd ? -1 : 1
                case ("i", _), ("em", _): italic += isEnd ? -1 : 1
                case ("u", _): underline += isEnd ? -1 : 1
                case ("s", _), ("strike", _), ("del", _): strike += isEnd ? -1 : 1
                case ("a", false): links.append(attribute("href", in: String(body)).flatMap(URL.init(string:)))
                case ("a", true): _ = links.popLast()
                case ("br", _): pendingBreak = true
                case ("ul", false), ("ol", false):
                    lists.append((name == "ol", 0))
                    breakLine()
                case ("ul", true), ("ol", true):
                    _ = lists.popLast()
                    lineStyle = nil
                    breakLine()
                case ("li", false):
                    breakLine()
                    if let last = lists.last {
                        lineStyle = NoteBlockStyle(last.ordered ? .numbered : .bullet, indent: lists.count - 1)
                    }
                case ("li", true):
                    breakLine()
                case ("p", _), ("div", _), ("h1", _), ("h2", _), ("h3", _), ("h4", _), ("h5", _), ("h6", _),
                     ("tr", _), ("blockquote", _), ("pre", _):
                    if lists.isEmpty { lineStyle = nil }
                    breakLine()
                default:
                    break
                }
                bold = max(0, bold); italic = max(0, italic); underline = max(0, underline); strike = max(0, strike)
            } else {
                let next = html[index...].firstIndex(of: "<") ?? html.endIndex
                if skipping == nil {
                    // HTML collapses runs of white space to one space.
                    var collapsed = ""
                    var lastWasSpace = false
                    for character in decodeEntities(String(html[index..<next])) {
                        if character.isWhitespace {
                            if !lastWasSpace { collapsed.append(" ") }
                            lastWasSpace = true
                        } else {
                            collapsed.append(character)
                            lastWasSpace = false
                        }
                    }
                    let atLineStart = result.length == 0 || result.string.hasSuffix("\n") || pendingBreak
                    if atLineStart { collapsed = String(collapsed.drop { $0 == " " }) }
                    if !collapsed.isEmpty, collapsed != " " || !atLineStart { emit(collapsed) }
                }
                index = next
            }
        }
        return result
    }

    private static func attribute(_ name: String, in tag: String) -> String? {
        guard let range = tag.range(of: "\(name)=", options: .caseInsensitive) else { return nil }
        var rest = tag[range.upperBound...]
        guard let quote = rest.first else { return nil }
        if quote == "\"" || quote == "'" {
            rest = rest.dropFirst()
            return rest.firstIndex(of: quote).map { decodeEntities(String(rest[..<$0])) }
        }
        return decodeEntities(String(rest.prefix { !$0.isWhitespace && $0 != ">" }))
    }

    static func decodeEntities(_ text: String) -> String {
        guard text.contains("&") else { return text }
        var out = ""
        var index = text.startIndex
        let named: [String: String] = ["amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": " ",
                                       "ndash": "–", "mdash": "—", "hellip": "…", "rsquo": "’", "lsquo": "‘",
                                       "rdquo": "”", "ldquo": "“"]
        while index < text.endIndex {
            if text[index] == "&", let semi = text[index...].prefix(10).firstIndex(of: ";") {
                let entity = String(text[text.index(after: index)..<semi])
                var replacement: String?
                if entity.hasPrefix("#x") || entity.hasPrefix("#X") {
                    replacement = UInt32(entity.dropFirst(2), radix: 16).flatMap(Unicode.Scalar.init).map { String($0) }
                } else if entity.hasPrefix("#") {
                    replacement = UInt32(entity.dropFirst()).flatMap(Unicode.Scalar.init).map { String($0) }
                } else {
                    replacement = named[entity.lowercased()]
                }
                if let replacement {
                    out += replacement
                    index = text.index(after: semi)
                    continue
                }
            }
            out.append(text[index])
            index = text.index(after: index)
        }
        return out
    }
}
