import AppKit
import MferenceAppCore

/// A line that is one object, not text: an image, an audio clip, a
/// transcript time chip or a "Mark moment" marker. It is a single attachment
/// character on its own line; the block it stands for travels with it.
nonisolated final class NoteBlockAttachment: NSTextAttachment {
    let block: NoteDocument.Block
    /// Where image and audio files are (`Notes/<note-id>/`).
    let fileURL: URL?
    /// The time a transcript chip shows, looked up from the recording.
    let time: TimeInterval?

    init(block: NoteDocument.Block, fileURL: URL?, time: TimeInterval?) {
        self.block = block
        self.fileURL = fileURL
        self.time = time
        super.init(data: nil, ofType: nil)
        switch block {
        case .image:
            if let fileURL, let image = NSImage(contentsOf: fileURL) {
                self.image = image
                bounds = Self.fitted(image.size)
            } else {
                image = Self.chipImage(text: "Missing image", symbol: "photo")
                bounds = CGRect(origin: .zero, size: image?.size ?? .zero)
            }
        case .audio:
            image = Self.chipImage(text: fileURL?.lastPathComponent ?? "Audio", symbol: "waveform")
            bounds = CGRect(origin: CGPoint(x: 0, y: -NoteEditorMetrics.chipBaselineDrop), size: image?.size ?? .zero)
        case .transcriptAnchor:
            image = Self.chipImage(text: time.map(Self.clock) ?? "–:––", symbol: nil)
            bounds = CGRect(origin: CGPoint(x: 0, y: -NoteEditorMetrics.chipBaselineDrop), size: image?.size ?? .zero)
        case .marker(let time):
            image = Self.chipImage(text: Self.clock(time), symbol: "bookmark.fill", accent: true)
            bounds = CGRect(origin: CGPoint(x: 0, y: -NoteEditorMetrics.chipBaselineDrop), size: image?.size ?? .zero)
        default:
            break
        }
    }

    required init?(coder: NSCoder) { nil }

    /// "12:34", or "1:02:03" past an hour.
    static func clock(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds.rounded(.down)))
        let (h, m, s) = (total / 3600, total / 60 % 60, total % 60)
        return h > 0 ? String(format: "%d:%02d:%02d", h, m, s) : String(format: "%d:%02d", m, s)
    }

    private static func fitted(_ size: CGSize) -> CGRect {
        let maxWidth = NoteEditorMetrics.imageMaxWidth
        let scale = min(1, maxWidth / max(size.width, 1), NoteEditorMetrics.imageMaxHeight / max(size.height, 1))
        return CGRect(x: 0, y: 0, width: size.width * scale, height: size.height * scale)
    }

    /// A rounded chip with an optional symbol, drawn for the current appearance.
    private static func chipImage(text: String, symbol: String?, accent: Bool = false) -> NSImage {
        let font = NoteEditorStyle.geist(NoteEditorMetrics.chipFontSize, .medium)
        let textSize = (text as NSString).size(withAttributes: [.font: font])
        let symbolWidth = symbol == nil ? 0 : NoteEditorMetrics.chipSymbolWidth
        let size = CGSize(width: ceil(textSize.width) + symbolWidth + 2 * NoteEditorMetrics.chipPadding, height: NoteEditorMetrics.chipHeight)
        return NSImage(size: size, flipped: false) { rect in
            let fill = accent ? NoteEditorStyle.accentFill : NoteEditorStyle.chipFill
            let ink = accent ? NoteEditorStyle.accent : NoteEditorStyle.secondary
            fill.setFill()
            NSBezierPath(roundedRect: rect, xRadius: rect.height / 2, yRadius: rect.height / 2).fill()
            var x = NoteEditorMetrics.chipPadding
            if let symbol, let image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)?
                .withSymbolConfiguration(.init(pointSize: NoteEditorMetrics.chipSymbolSize, weight: .medium)) {
                let tinted = image.tinted(ink)
                let y = (rect.height - tinted.size.height) / 2
                tinted.draw(in: CGRect(x: x, y: y, width: tinted.size.width, height: tinted.size.height))
                x += symbolWidth
            }
            (text as NSString).draw(at: CGPoint(x: x, y: (rect.height - textSize.height) / 2),
                                    withAttributes: [.font: font, .foregroundColor: ink])
            return true
        }
    }
}

nonisolated private extension NSImage {
    func tinted(_ color: NSColor) -> NSImage {
        NSImage(size: size, flipped: false) { rect in
            self.draw(in: rect)
            color.set()
            rect.fill(using: .sourceAtop)
            return true
        }
    }
}
