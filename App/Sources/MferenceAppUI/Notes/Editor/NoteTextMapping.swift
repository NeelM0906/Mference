import AppKit
import MferenceAppCore

/// Maps a `NoteDocument` to the editor's attributed text and back (Notes
/// spec 8.2). Each block is one line. What a line *is* lives in canonical
/// attributes (`.noteBlock`, `.noteBold`, `.noteItalic`, underline,
/// strikethrough, link, block attachments); fonts, colors and paragraph
/// styles are derived from them by `restyle`, so reading the text back never
/// depends on how it is drawn.
nonisolated enum NoteTextMapping {
    /// Context for drawing object lines.
    struct Context {
        /// The file behind an image or audio attachment id.
        var attachmentURL: (String) -> URL? = { _ in nil }
        /// The time a transcript segment starts at.
        var segmentTime: (UUID) -> TimeInterval? = { _ in nil }
    }

    // MARK: Document → text

    /// The text for `document`, and the block style of its last line (which,
    /// when empty, has no character to carry it: it becomes the typing style).
    static func attributedString(from document: NoteDocument, context: Context = Context())
        -> (text: NSAttributedString, lastLineStyle: NoteBlockStyle) {
        let blocks = document.blocks.isEmpty ? NoteDocument.empty.blocks : document.blocks
        let text = NSMutableAttributedString()
        let styles = blocks.map(style(of:))
        for (index, block) in blocks.enumerated() {
            let style = styles[index]
            let next = index + 1 < styles.count ? styles[index + 1] : nil
            let lineStart = text.length
            if let attachment = attachment(for: block, context: context) {
                text.append(NSAttributedString(attachment: attachment))
            } else {
                for run in normalizedRuns(block.runs) {
                    text.append(NSAttributedString(string: run.text,
                                                   attributes: NoteEditorStyle.attributes(block: style, next: next, run: run)))
                }
            }
            if index < blocks.count - 1 {
                text.append(NSAttributedString(string: "\n", attributes: NoteEditorStyle.attributes(block: style, next: next, run: TextRun(""))))
            }
            let range = NSRange(location: lineStart, length: text.length - lineStart)
            if range.length > 0 {
                text.addAttribute(.noteBlock, value: style, range: range)
                text.addAttribute(.paragraphStyle, value: NoteEditorStyle.paragraphStyle(for: style, next: next), range: range)
            }
        }
        return (text, styles.last ?? .body)
    }

    // MARK: Text → document

    /// Reads the document back. `lastLineStyle` is the typing style used for
    /// an empty last line.
    static func document(from text: NSAttributedString, lastLineStyle: NoteBlockStyle) -> NoteDocument {
        var blocks: [NoteDocument.Block] = []
        let string = text.string as NSString
        var location = 0
        repeat {
            let lineRange = string.lineRange(for: NSRange(location: location, length: 0))
            var contentRange = lineRange
            // Any line terminator ends the line: \n, \r, U+2028, U+2029.
            while contentRange.length > 0, isLineTerminator(string.character(at: NSMaxRange(contentRange) - 1)) {
                contentRange.length -= 1
            }
            let style: NoteBlockStyle
            if lineRange.length > 0 {
                style = text.attribute(.noteBlock, at: lineRange.location, effectiveRange: nil) as? NoteBlockStyle ?? .body
            } else {
                style = lastLineStyle
            }
            blocks.append(contentsOf: lineBlocks(in: text, range: contentRange, style: style))
            location = NSMaxRange(lineRange)
        } while location < string.length
        // A text ending in a line break has an empty last line after it.
        if string.length > 0, isLineTerminator(string.character(at: string.length - 1)) {
            blocks.append(contentsOf: lineBlocks(in: text, range: NSRange(location: string.length, length: 0),
                                                 style: lastLineStyle))
        }
        return NoteDocument(blocks: blocks)
    }

    /// The runs of `range` (attachment characters skipped).
    static func runs(in text: NSAttributedString, range: NSRange) -> [TextRun] {
        var runs: [TextRun] = []
        let string = text.string as NSString
        text.enumerateAttributes(in: range) { attributes, subrange, _ in
            guard attributes[.attachment] == nil else { return }
            let piece = string.substring(with: subrange)
                .replacingOccurrences(of: "\u{2028}", with: " ").replacingOccurrences(of: "\u{2029}", with: " ")
                .replacingOccurrences(of: "\r", with: "")
            guard !piece.isEmpty else { return }
            runs.append(TextRun(piece,
                                bold: attributes[.noteBold] != nil,
                                italic: attributes[.noteItalic] != nil,
                                underline: (attributes[.underlineStyle] as? Int ?? 0) != 0,
                                strikethrough: (attributes[.strikethroughStyle] as? Int ?? 0) != 0,
                                link: link(attributes[.link])))
        }
        return normalizedRuns(runs)
    }

    // MARK: Restyling

    /// Re-derives fonts, colors and paragraph styles from the canonical
    /// attributes for every line that touches `range` (and the line before,
    /// whose bottom margin depends on the next line). With `clip`, only the
    /// characters in `range` change: used while the text storage processes
    /// an edit, where touching more would widen the edited range (and move
    /// the caret).
    static func restyle(_ text: NSMutableAttributedString, around range: NSRange, clip: Bool = false) {
        let string = text.string as NSString
        guard string.length > 0 else { return }
        var start = min(range.location, string.length)
        if start > 0 {
            // Include the previous line.
            start = string.lineRange(for: NSRange(location: max(0, start - 1), length: 0)).location
        }
        let endLine = string.lineRange(for: NSRange(location: min(NSMaxRange(range), string.length), length: 0))
        let end = min(string.length, NSMaxRange(endLine))
        var location = start
        while location < end {
            let line = string.lineRange(for: NSRange(location: location, length: 0))
            let style = text.attribute(.noteBlock, at: line.location, effectiveRange: nil) as? NoteBlockStyle ?? .body
            let nextStart = NSMaxRange(line)
            let next = nextStart < string.length
                ? text.attribute(.noteBlock, at: nextStart, effectiveRange: nil) as? NoteBlockStyle ?? .body
                : nil
            let target = clip ? NSIntersectionRange(line, range) : line
            // A line holding an object with text joined onto it keeps each
            // piece's own block style (it is read back as separate blocks).
            var holdsObject = false
            if target.length > 0 {
                text.enumerateAttribute(.attachment, in: line) { value, _, stop in
                    if value is NoteBlockAttachment { holdsObject = true; stop.pointee = true }
                }
            }
            if target.length > 0, !holdsObject {
                text.addAttribute(.noteBlock, value: style, range: target)
            }
            if target.length > 0 {
                text.addAttribute(.paragraphStyle, value: NoteEditorStyle.paragraphStyle(for: style, next: next), range: target)
            }
            text.enumerateAttributes(in: target) { attributes, subrange, _ in
                guard attributes[.attachment] == nil else { return }
                let run = TextRun("", bold: attributes[.noteBold] != nil, italic: attributes[.noteItalic] != nil,
                                  link: link(attributes[.link]))
                let own = holdsObject ? (attributes[.noteBlock] as? NoteBlockStyle ?? style) : style
                var display = NoteEditorStyle.attributes(block: own, next: next, run: run)
                display[.noteBlock] = nil
                display[.paragraphStyle] = nil
                text.addAttributes(display, range: subrange)
            }
            if line.length == 0 { break }
            location = NSMaxRange(line)
        }
    }

    // MARK: Private

    static func style(of block: NoteDocument.Block) -> NoteBlockStyle {
        switch block {
        case .heading(let level, _):
            switch level {
            case .title: .title
            case .heading: NoteBlockStyle(.heading)
            case .subheading: NoteBlockStyle(.subheading)
            }
        case .paragraph, .image, .audio, .transcriptAnchor, .marker, .unknown: .body
        case .bullet(_, let indent): NoteBlockStyle(.bullet, indent: indent)
        case .numbered(_, let indent): NoteBlockStyle(.numbered, indent: indent)
        case .check(_, let done, let indent): NoteBlockStyle(.check, indent: indent, done: done)
        }
    }

    static func isLineTerminator(_ character: unichar) -> Bool {
        character == 0x0A || character == 0x0D || character == 0x2028 || character == 0x2029
    }

    /// The blocks of one line. A line is normally one block; if text was
    /// joined onto an object line (Backspace after an image, typing beside
    /// a chip), the objects and the text around them each become a block,
    /// in order, so nothing is lost.
    private static func lineBlocks(in text: NSAttributedString, range: NSRange,
                                   style: NoteBlockStyle) -> [NoteDocument.Block] {
        var objects: [(range: NSRange, block: NoteDocument.Block)] = []
        if range.length > 0 {
            text.enumerateAttribute(.attachment, in: range) { value, subrange, _ in
                if let attachment = value as? NoteBlockAttachment { objects.append((subrange, attachment.block)) }
            }
        }
        guard !objects.isEmpty else { return [block(in: text, range: range, style: style)] }
        var blocks: [NoteDocument.Block] = []
        var cursor = range.location
        for object in objects {
            let before = NSRange(location: cursor, length: object.range.location - cursor)
            if before.length > 0, !runs(in: text, range: before).isEmpty {
                blocks.append(block(in: text, range: before, style: segmentStyle(text, before, fallback: style)))
            }
            blocks.append(object.block)
            cursor = NSMaxRange(object.range)
        }
        let after = NSRange(location: cursor, length: NSMaxRange(range) - cursor)
        if after.length > 0, !runs(in: text, range: after).isEmpty {
            blocks.append(block(in: text, range: after, style: segmentStyle(text, after, fallback: style)))
        }
        return blocks
    }

    /// Text joined onto an object line keeps the style it had (a checklist
    /// item stays one), not the object line's.
    private static func segmentStyle(_ text: NSAttributedString, _ range: NSRange, fallback: NoteBlockStyle) -> NoteBlockStyle {
        text.attribute(.noteBlock, at: range.location, effectiveRange: nil) as? NoteBlockStyle ?? fallback
    }

    private static func block(in text: NSAttributedString, range: NSRange, style: NoteBlockStyle) -> NoteDocument.Block {
        let runs = runs(in: text, range: range)
        switch style.kind {
        case .title: return .heading(level: .title, runs: runs)
        case .heading: return .heading(level: .heading, runs: runs)
        case .subheading: return .heading(level: .subheading, runs: runs)
        case .body: return .paragraph(runs: runs)
        case .bullet: return .bullet(runs: runs, indent: style.indent)
        case .numbered: return .numbered(runs: runs, indent: style.indent)
        case .check: return .check(runs: runs, done: style.done, indent: style.indent)
        }
    }

    private static func attachment(for block: NoteDocument.Block, context: Context) -> NoteBlockAttachment? {
        switch block {
        case .image(let id), .audio(let id):
            NoteBlockAttachment(block: block, fileURL: context.attachmentURL(id), time: nil)
        case .transcriptAnchor(let id):
            NoteBlockAttachment(block: block, fileURL: nil, time: context.segmentTime(id))
        case .marker(let time):
            NoteBlockAttachment(block: block, fileURL: nil, time: time)
        case .unknown:
            NoteBlockAttachment(block: block, fileURL: nil, time: nil)
        default:
            nil
        }
    }

    private static func link(_ value: Any?) -> URL? {
        if let url = value as? URL { return url }
        if let string = value as? String { return URL(string: string) }
        return nil
    }
}
