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
        var lines: [String]
        func recognize(imageAt url: URL) async throws -> ScanReading { reading }
        func recognize(image: CGImage) async throws -> ScanReading { reading }
        private var reading: ScanReading {
            ScanReading(lines: lines.map { ScanLine(text: $0, confidence: 0.95, box: .zero) })
        }
    }

    static let receiptLines = [
        "CORNER HARDWARE", "412 Valencia St · San Francisco", "09/12/2026 14:32",
        row("Wood glue 8oz", "6.49"), row("Sandpaper assort", "9.99"), row("Drill bits 1/8", "12.50"),
        row("Painter tape x2", "11.98"), String(repeating: "-", count: 30), row("Subtotal", "40.96"),
        row("Tax", "3.53"), row("TOTAL", "44.49"), "", "VISA **** · APPROVED",
    ]

    static func row(_ label: String, _ amount: String) -> String {
        label + String(repeating: " ", count: max(1, 30 - label.count - amount.count)) + amount
    }

    /// A receipt image rendered on this Mac.
    static func receiptImage() throws -> URL {
        let font = NSFont.monospacedSystemFont(ofSize: 22, weight: .regular)
        let lineHeight: CGFloat = 32
        let size = CGSize(width: 520, height: CGFloat(receiptLines.count) * lineHeight + 80)
        let image = NSImage(size: size, flipped: true) { rect in
            NSColor(srgbRed: 1, green: 0.995, blue: 0.97, alpha: 1).setFill()
            rect.fill()
            for (index, line) in receiptLines.enumerated() {
                NSAttributedString(string: line, attributes: [.font: font, .foregroundColor: NSColor.black])
                    .draw(at: CGPoint(x: 36, y: 40 + CGFloat(index) * lineHeight))
            }
            return true
        }
        guard let tiff = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff),
              let png = bitmap.representation(using: .png, properties: [:]) else { throw CocoaError(.fileWriteUnknown) }
        let url = URL.temporaryDirectory.appending(path: "mference-receipt-fixture.png")
        try png.write(to: url)
        return url
    }

    static let receiptAnswer = """
        {"merchant": "Corner Hardware", "date": "2026-09-12", "total": "44.49", "tax": "3.53", "currency": "USD",
         "category": "Home improvement", "paidWith": "Visa", "lineItems": []}
        """
}
