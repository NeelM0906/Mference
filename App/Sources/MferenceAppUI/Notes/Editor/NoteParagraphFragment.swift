import AppKit

/// Draws a list line's marker (bullet, number, checkbox) in its indent and
/// strikes a done checklist item through, so none of them are characters in
/// the text: the caret never lands before a box and the text reads back clean.
nonisolated final class NoteParagraphFragment: NSTextLayoutFragment {
    let style: NoteBlockStyle
    /// 1-based position among neighbouring numbered lines of the same indent.
    let ordinal: Int

    init(textElement: NSTextElement, range: NSTextRange?, style: NoteBlockStyle, ordinal: Int) {
        self.style = style
        self.ordinal = ordinal
        super.init(textElement: textElement, range: range)
    }

    required init?(coder: NSCoder) { nil }

    /// How far the text starts from the line's left edge in the column.
    private var textInset: CGFloat {
        CGFloat(style.indent) * NoteEditorMetrics.listIndent
            + (style.kind == .check ? NoteEditorMetrics.checkIndent : NoteEditorMetrics.listIndent)
    }

    /// Where the first line's text starts, in fragment coordinates. Markers
    /// are placed from it, whatever offset TextKit gives the fragment frame.
    private var textStartX: CGFloat {
        textLineFragments.first.map { $0.typographicBounds.minX + $0.glyphOrigin.x } ?? textInset
    }

    /// The checkbox's rectangle in fragment coordinates (for hit testing).
    var checkBoxRect: CGRect? {
        guard style.kind == .check, let line = textLineFragments.first else { return nil }
        let size = NoteEditorMetrics.checkBoxSize
        return CGRect(x: textStartX - NoteEditorMetrics.checkIndent, y: line.typographicBounds.midY - size / 2,
                      width: size, height: size)
    }

    /// The marker area, to the left of the text.
    private var markerRect: CGRect {
        let line = textLineFragments.first?.typographicBounds ?? .zero
        let width = style.kind == .check ? NoteEditorMetrics.checkIndent : NoteEditorMetrics.listIndent
        return CGRect(x: textStartX - width - NoteEditorMetrics.numberGap, y: line.minY,
                      width: width + NoteEditorMetrics.numberGap, height: max(line.height, NoteEditorMetrics.checkBoxSize))
    }

    override var renderingSurfaceBounds: CGRect {
        style.isList ? super.renderingSurfaceBounds.union(markerRect) : super.renderingSurfaceBounds
    }

    override func draw(at point: CGPoint, in context: CGContext) {
        super.draw(at: point, in: context)
        guard style.isList, let line = textLineFragments.first else { return }
        NSGraphicsContext.saveGraphicsState()
        defer { NSGraphicsContext.restoreGraphicsState() }
        let textX = point.x + textStartX
        let midY = point.y + line.typographicBounds.midY
        switch style.kind {
        case .bullet:
            let size = NoteEditorMetrics.bulletSize
            NoteEditorStyle.ink.setFill()
            NSBezierPath(ovalIn: CGRect(x: textX - NoteEditorMetrics.listIndent / 2 - size / 2 - NoteEditorMetrics.bulletShift,
                                        y: midY - size / 2, width: size, height: size)).fill()
        case .numbered:
            let font = NoteEditorStyle.geist(NoteEditorMetrics.bodySize, .regular)
            let label = "\(ordinal)." as NSString
            let attributes: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NoteEditorStyle.ink]
            let size = label.size(withAttributes: attributes)
            let baseline = point.y + line.typographicBounds.minY + line.glyphOrigin.y
            label.draw(at: CGPoint(x: textX - size.width - NoteEditorMetrics.numberGap, y: baseline - font.ascender),
                       withAttributes: attributes)
        case .check:
            guard let box = checkBoxRect?.offsetBy(dx: point.x, dy: point.y) else { return }
            drawCheckBox(in: box)
            if style.done { strikeThrough(at: point) }
        default:
            break
        }
    }

    private func drawCheckBox(in rect: CGRect) {
        let inset = NoteEditorMetrics.checkBoxStroke / 2
        let circle = NSBezierPath(ovalIn: rect.insetBy(dx: inset, dy: inset))
        if style.done {
            NoteEditorStyle.ink.setFill()
            circle.fill()
            let tick = NSBezierPath()
            tick.lineWidth = NoteEditorMetrics.tickStroke
            tick.lineCapStyle = .round
            tick.lineJoinStyle = .round
            // The board's tick: m5 12 5 5 9-10 in a 24-unit box, drawn at 10 pt.
            let unit = NoteEditorMetrics.tickSize / 24
            let box = CGRect(x: rect.midX - NoteEditorMetrics.tickSize / 2, y: rect.midY - NoteEditorMetrics.tickSize / 2,
                             width: NoteEditorMetrics.tickSize, height: NoteEditorMetrics.tickSize)
            // Fragment drawing is flipped (y grows down), like the SVG.
            tick.move(to: CGPoint(x: box.minX + 5 * unit, y: box.minY + 12 * unit))
            tick.line(to: CGPoint(x: box.minX + 10 * unit, y: box.minY + 17 * unit))
            tick.line(to: CGPoint(x: box.minX + 19 * unit, y: box.minY + 7 * unit))
            NoteEditorStyle.onInk.setStroke()
            tick.stroke()
        } else {
            circle.lineWidth = NoteEditorMetrics.checkBoxStroke
            NoteEditorStyle.controlBorder.setStroke()
            circle.stroke()
        }
    }

    private func strikeThrough(at origin: CGPoint) {
        NoteEditorStyle.secondary.setStroke()
        for line in textLineFragments {
            let bounds = line.typographicBounds
            let start = bounds.minX + line.glyphOrigin.x
            guard bounds.maxX > start else { continue }
            let y = origin.y + bounds.minY + line.glyphOrigin.y - NoteEditorMetrics.strikeRise
            let path = NSBezierPath()
            path.lineWidth = NoteEditorMetrics.hairline
            path.move(to: CGPoint(x: origin.x + start, y: y))
            path.line(to: CGPoint(x: origin.x + bounds.maxX, y: y))
            path.stroke()
        }
    }
}
