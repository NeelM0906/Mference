import AppKit
import Foundation
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Synthetic data for the Scan, Calendar and Mail snapshots: invented people
/// and stores, never real mail or calendar data.
enum AppleUIFixtures {
    // MARK: Scan

    nonisolated struct FixedScanner: DocumentRecognizing {
        var lines: [ScanLine]

        init(lines: [String]) {
            self.lines = lines.map { ScanLine(text: $0, confidence: 0.95, box: .zero) }
        }

        init(scanLines: [ScanLine]) {
            lines = scanLines
        }

        func recognize(imageAt url: URL) async throws -> ScanReading { ScanReading(lines: lines) }
        func recognize(image: CGImage) async throws -> ScanReading { ScanReading(lines: lines) }
    }

    static let receiptItems = [("Wood glue 8oz", "6.49"), ("Sandpaper assort", "9.99"), ("Drill bits 1/8", "12.50"),
                               ("Painter tape x2", "11.98")]
    static let receiptTotals = [("Subtotal", "40.96"), ("Tax", "3.53"), ("TOTAL", "44.49")]

    /// The receipt's text as lines (no boxes), for screens that do not draw it.
    static var receiptLines: [String] {
        ["CORNER HARDWARE", "412 Valencia St · San Francisco", "09/12/2026 14:32"]
            + receiptItems.map { "\($0.0)   \($0.1)" } + receiptTotals.map { "\($0.0)   \($0.1)" }
            + ["VISA **** · APPROVED"]
    }

    /// A receipt drawn like mockup 07 (centred header in secondary ink,
    /// monospaced 12 pt at line height 1.7, a dashed rule, TOTAL in bold) at
    /// 2x, and the lines a text reader would return for it, with boxes.
    static func receipt() throws -> (url: URL, lines: [ScanLine]) {
        let scale: CGFloat = 2
        let fontSize: CGFloat = 12 * scale, lineHeight = fontSize * 1.7
        let width: CGFloat = 306 * scale, margin: CGFloat = 24 * scale
        let regular = NSFont.monospacedSystemFont(ofSize: fontSize, weight: .regular)
        let bold = NSFont.monospacedSystemFont(ofSize: fontSize, weight: .semibold)
        let ink = NSColor(srgbRed: 0x17 / 255, green: 0x18 / 255, blue: 0x1C / 255, alpha: 1)
        let secondary = NSColor(srgbRed: 0x5E / 255, green: 0x62 / 255, blue: 0x6C / 255, alpha: 1)
        enum Align { case left, center, right }
        struct Item { var text: String; var font: NSFont; var color: NSColor; var align: Align; var row: Double }
        var items: [Item] = [
            Item(text: "CORNER HARDWARE", font: bold, color: ink, align: .center, row: 0),
            Item(text: "412 Valencia St · San Francisco", font: regular, color: secondary, align: .center, row: 1),
            Item(text: "09/12/2026 14:32", font: regular, color: secondary, align: .center, row: 2),
        ]
        var row = 3.6
        for (name, amount) in receiptItems {
            items.append(Item(text: name, font: regular, color: ink, align: .left, row: row))
            items.append(Item(text: amount, font: regular, color: ink, align: .right, row: row))
            row += 1
        }
        let ruleRow = row + 0.15
        row += 0.6
        for (label, amount) in receiptTotals {
            let font = label == "TOTAL" ? bold : regular
            items.append(Item(text: label, font: font, color: ink, align: .left, row: row))
            items.append(Item(text: amount, font: font, color: ink, align: .right, row: row))
            row += 1
        }
        row += 0.6
        items.append(Item(text: "VISA **** · APPROVED", font: regular, color: secondary, align: .left, row: row))
        let height = margin * 2 + CGFloat(row + 1) * lineHeight
        func top(_ item: Item) -> CGFloat { margin + CGFloat(item.row) * lineHeight }
        var scanLines: [ScanLine] = []
        let image = NSImage(size: NSSize(width: width, height: height), flipped: true) { rect in
            NSColor(srgbRed: 1, green: 0.995, blue: 0.97, alpha: 1).setFill()
            rect.fill()
            let dash = NSBezierPath()
            let y = margin + CGFloat(ruleRow) * lineHeight
            dash.move(to: NSPoint(x: margin, y: y))
            dash.line(to: NSPoint(x: width - margin, y: y))
            dash.lineWidth = scale
            dash.setLineDash([4 * scale, 3 * scale], count: 2, phase: 0)
            NSColor(srgbRed: 0xB8 / 255, green: 0xBB / 255, blue: 0xC3 / 255, alpha: 1).setStroke()
            dash.stroke()
            for item in items {
                let text = NSAttributedString(string: item.text, attributes: [.font: item.font,
                                                                               .foregroundColor: item.color])
                let size = text.size()
                let x: CGFloat = switch item.align {
                case .left: margin
                case .center: (width - size.width) / 2
                case .right: width - margin - size.width
                }
                let y = top(item) + (lineHeight - size.height) / 2
                text.draw(at: NSPoint(x: x, y: y))
            }
            return true
        }
        for item in items {
            let size = NSAttributedString(string: item.text, attributes: [.font: item.font]).size()
            let x: CGFloat = switch item.align {
            case .left: margin
            case .center: (width - size.width) / 2
            case .right: width - margin - size.width
            }
            let y = top(item) + (lineHeight - size.height) / 2
            scanLines.append(ScanLine(text: item.text, confidence: 0.95,
                                      box: CGRect(x: x / width, y: y / height, width: size.width / width,
                                                  height: size.height / height)))
        }
        guard let tiff = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff),
              let png = bitmap.representation(using: .png, properties: [:]) else { throw CocoaError(.fileWriteUnknown) }
        let url = URL.temporaryDirectory.appending(path: "mference-receipt-fixture.png")
        try png.write(to: url)
        return (url, scanLines)
    }

    static let receiptAnswer = """
        {"merchant": "Corner Hardware", "date": "2026-09-12", "total": "44.49", "tax": "3.53", "currency": "USD",
         "category": "Home improvement", "paidWith": "Visa", "lineItems": []}
        """
}
