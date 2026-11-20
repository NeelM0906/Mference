import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Layout sizes of the note editor (board 1, `Main.dc.html` `.doc`), named
/// per the CONTRACTS deviation for per-screen metrics.
nonisolated enum NoteEditorMetrics {
    static let titleSize: CGFloat = 24
    static let headingSize: CGFloat = 15
    static let subheadingSize: CGFloat = 14
    static let bodySize: CGFloat = 14
    /// CSS line-height 1.6 on 14 pt text.
    static let bodyLineHeight: CGFloat = 22.4
    static let titleLineHeight: CGFloat = 30
    static let headingLineHeight: CGFloat = 22
    static let titleSpacingBefore: CGFloat = 14
    static let titleSpacingAfter: CGFloat = 10
    static let headingSpacingBefore: CGFloat = 18
    static let headingSpacingAfter: CGFloat = 6
    static let paragraphSpacing: CGFloat = 12
    static let listItemSpacing: CGFloat = 0
    static let listEndSpacing: CGFloat = 10
    static let checkItemSpacing: CGFloat = 4
    /// Text inset of a bullet or number per level (`ul` padding-left 20).
    static let listIndent: CGFloat = 20
    /// Text inset of a checklist item: a 16 pt box and a 10 pt gap.
    static let checkIndent: CGFloat = 26
    static let checkBoxSize: CGFloat = 16
    static let checkBoxStroke: CGFloat = 1.5
    static let bulletSize: CGFloat = 5
    /// The bullet sits a little left of the indent's middle, as a browser draws it.
    static let bulletShift: CGFloat = 2
    /// Gap between a list number and its text.
    static let numberGap: CGFloat = 6
    static let tickSize: CGFloat = 10
    static let tickStroke: CGFloat = 1.75
    /// How far above the baseline a done item's strike runs.
    static let strikeRise: CGFloat = 4.5
    static let hairline: CGFloat = 1
    static let chipPadding: CGFloat = 8
    static let chipSymbolWidth: CGFloat = 16
    static let chipSymbolSize: CGFloat = 10
    /// Chips sit a little below the baseline to center on the line.
    static let chipBaselineDrop: CGFloat = 5
    static let imageMaxWidth: CGFloat = 520
    /// Italic in Geist (no italic face): a slant.
    static let italicObliqueness: CGFloat = 0.18
    /// Editor padding (`.doc` padding 22px 56px).
    static let horizontalInset: CGFloat = 56
    static let topInset: CGFloat = 22
    /// The date line's text size and the room it takes above the title.
    static let dateSize: CGFloat = 12
    static let dateLineHeight: CGFloat = 16
    /// The read-only notice under the date (up to two lines).
    static let noticeHeight: CGFloat = 34
    /// The accent capsule marking where a pending dictated phrase lands.
    static let pendingPhraseMinWidth: CGFloat = 18
    static let pendingPhraseHeight: CGFloat = 6
    static let pendingPhraseGap: CGFloat = 3
    static let lineFragmentPadding: CGFloat = 0
    static let maxIndent = 4
    /// Extra room around a checkbox that still ticks it.
    static let checkHitSlop: CGFloat = 4
    static let chipHeight: CGFloat = 22
    static let chipFontSize: CGFloat = 11.5
    static let imageMaxHeight: CGFloat = 360
}

nonisolated extension NSAttributedString.Key {
    /// The paragraph's block (a `NoteBlockStyle`), on every character of it
    /// including its line break.
    static let noteBlock = NSAttributedString.Key("MferenceNoteBlock")
    /// Run styles kept apart from the font, because headings already use a
    /// heavier face.
    static let noteBold = NSAttributedString.Key("MferenceNoteBold")
    static let noteItalic = NSAttributedString.Key("MferenceNoteItalic")
}

/// What kind of paragraph a line is, as the editor stores it.
nonisolated final class NoteBlockStyle: NSObject, @unchecked Sendable {
    enum Kind: Hashable { case title, heading, subheading, body, bullet, numbered, check }

    let kind: Kind
    let indent: Int
    let done: Bool

    init(_ kind: Kind, indent: Int = 0, done: Bool = false) {
        self.kind = kind
        self.indent = max(0, indent)
        self.done = kind == .check && done
    }

    static let body = NoteBlockStyle(.body)
    static let title = NoteBlockStyle(.title)

    var isList: Bool { kind == .bullet || kind == .numbered || kind == .check }
    var isHeading: Bool { kind == .title || kind == .heading || kind == .subheading }

    func with(done: Bool) -> NoteBlockStyle { NoteBlockStyle(kind, indent: indent, done: done) }
    func with(indent: Int) -> NoteBlockStyle { NoteBlockStyle(kind, indent: indent, done: done) }

    override func isEqual(_ object: Any?) -> Bool {
        guard let other = object as? NoteBlockStyle else { return false }
        return kind == other.kind && indent == other.indent && done == other.done
    }

    override var hash: Int {
        var hasher = Hasher()
        hasher.combine(kind)
        hasher.combine(indent)
        hasher.combine(done)
        return hasher.finalize()
    }
}

/// Fonts, colors and paragraph styles the editor draws with.
nonisolated enum NoteEditorStyle {
    static var ink: NSColor { NSColor(MFColor.ink) }
    static var secondary: NSColor { NSColor(MFColor.secondary) }
    static var tertiary: NSColor { NSColor(MFColor.tertiary) }
    static var accent: NSColor { NSColor(MFColor.accent) }
    static var accentFill: NSColor { NSColor(MFColor.accentFill) }
    static var controlBorder: NSColor { NSColor(MFColor.controlBorder) }
    static var onInk: NSColor { NSColor(MFColor.onInk) }
    static var chipFill: NSColor { NSColor(MFColor.fillSubtle) }

    static func font(for style: NoteBlockStyle, bold: Bool, italic: Bool = false) -> NSFont {
        let upright = uprightFont(for: style, bold: bold)
        guard italic else { return upright }
        // Geist has no italic face: slant the upright one.
        // A font matrix is in points: scale, then shear.
        let size = upright.pointSize
        let slant = AffineTransform(m11: size, m12: 0, m21: size * NoteEditorMetrics.italicObliqueness, m22: size,
                                    tX: 0, tY: 0)
        return NSFont(descriptor: upright.fontDescriptor.withMatrix(slant), size: 0) ?? upright
    }

    private static func uprightFont(for style: NoteBlockStyle, bold: Bool) -> NSFont {
        let size: CGFloat
        var weight: NSFont.Weight
        switch style.kind {
        case .title: size = NoteEditorMetrics.titleSize; weight = .semibold
        case .heading: size = NoteEditorMetrics.headingSize; weight = .semibold
        case .subheading: size = NoteEditorMetrics.subheadingSize; weight = .semibold
        case .body, .bullet, .numbered, .check: size = NoteEditorMetrics.bodySize; weight = .regular
        }
        if bold { weight = style.isHeading ? .bold : .semibold }
        return geist(size, weight)
    }

    static func geist(_ size: CGFloat, _ weight: NSFont.Weight) -> NSFont {
        MFFontRegistry.registerBundledFonts()
        let name: String
        switch weight {
        case .bold, .heavy, .black: name = "Geist-Bold"
        case .semibold: name = "Geist-SemiBold"
        case .medium: name = "Geist-Medium"
        default: name = "Geist-Regular"
        }
        return NSFont(name: name, size: size) ?? .systemFont(ofSize: size, weight: weight)
    }

    static func lineHeight(for style: NoteBlockStyle) -> CGFloat {
        switch style.kind {
        case .title: NoteEditorMetrics.titleLineHeight
        case .heading, .subheading: NoteEditorMetrics.headingLineHeight
        default: NoteEditorMetrics.bodyLineHeight
        }
    }

    /// Space above a line of `style` (CSS margin-top on the board).
    static func spacingBefore(_ style: NoteBlockStyle) -> CGFloat {
        switch style.kind {
        case .title: NoteEditorMetrics.titleSpacingBefore
        case .heading, .subheading: NoteEditorMetrics.headingSpacingBefore
        default: 0
        }
    }

    /// Space below a line of `style` followed by `next` (margin-bottom).
    static func spacingAfter(_ style: NoteBlockStyle, next: NoteBlockStyle?) -> CGFloat {
        switch style.kind {
        case .title: NoteEditorMetrics.titleSpacingAfter
        case .heading, .subheading: NoteEditorMetrics.headingSpacingAfter
        case .body: NoteEditorMetrics.paragraphSpacing
        case .bullet, .numbered, .check:
            if let next, next.isList {
                style.kind == .check && next.kind == .check ? NoteEditorMetrics.checkItemSpacing : NoteEditorMetrics.listItemSpacing
            } else {
                NoteEditorMetrics.listEndSpacing
            }
        }
    }

    /// The paragraph style for a line, given the style of the line after it.
    /// Vertical margins collapse as on the board: the gap between two lines
    /// is the larger of the first's bottom and the second's top margin, all
    /// of it below the first line (a first line's top margin is the inset).
    static func paragraphStyle(for style: NoteBlockStyle, next: NoteBlockStyle?) -> NSParagraphStyle {
        let paragraph = NSMutableParagraphStyle()
        let height = lineHeight(for: style)
        paragraph.minimumLineHeight = height
        paragraph.maximumLineHeight = height
        paragraph.paragraphSpacing = max(spacingAfter(style, next: next), next.map(spacingBefore) ?? 0)
        if style.isList {
            let inset = CGFloat(style.indent) * NoteEditorMetrics.listIndent
                + (style.kind == .check ? NoteEditorMetrics.checkIndent : NoteEditorMetrics.listIndent)
            paragraph.firstLineHeadIndent = inset
            paragraph.headIndent = inset
        }
        return paragraph
    }

    /// Display attributes for a run in a line of `style`.
    static func attributes(block style: NoteBlockStyle, next: NoteBlockStyle?, run: TextRun) -> [NSAttributedString.Key: Any] {
        var attributes: [NSAttributedString.Key: Any] = [
            .noteBlock: style,
            .font: font(for: style, bold: run.bold, italic: run.italic),
            .foregroundColor: ink,
            .paragraphStyle: paragraphStyle(for: style, next: next),
        ]
        let lineHeight = lineHeight(for: style)
        let font = attributes[.font] as! NSFont
        attributes[.baselineOffset] = max(0, (lineHeight - (font.ascender - font.descender)) / 2)
        if run.bold { attributes[.noteBold] = true }
        if run.italic { attributes[.noteItalic] = true }
        if run.underline { attributes[.underlineStyle] = NSUnderlineStyle.single.rawValue }
        if run.strikethrough { attributes[.strikethroughStyle] = NSUnderlineStyle.single.rawValue }
        if style.done { attributes[.foregroundColor] = secondary }
        if let link = run.link {
            attributes[.link] = link
            attributes[.foregroundColor] = accent
        }
        return attributes
    }

    /// Typing attributes for an empty line of `style`.
    static func typingAttributes(_ style: NoteBlockStyle) -> [NSAttributedString.Key: Any] {
        attributes(block: style, next: nil, run: TextRun(""))
    }
}
