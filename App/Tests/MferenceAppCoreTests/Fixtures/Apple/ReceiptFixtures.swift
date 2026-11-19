import AppKit
import CoreGraphics
import Foundation
@testable import MferenceAppCore

/// Synthetic receipts rendered on this Mac: invented stores and amounts, no
/// personal data. Each carries its ground truth for the accuracy record.
struct ReceiptFixture: Sendable {
    var name: String
    var lines: [String]
    var merchant: String
    var date: String       // YYYY-MM-DD
    var totalText: String
    var taxText: String?
    var paidWith: String?
    var currency: String
    /// Degrees; small tilts like a phone photo.
    var rotation: Double = 0
    var fontSize: CGFloat = 26

    var total: Decimal { Decimal(string: totalText)! }
    var tax: Decimal? { taxText.flatMap { Decimal(string: $0) } }

    /// Renders the receipt as a white slip on a grey background.
    func image() -> CGImage {
        let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: .regular)
        let lineHeight = fontSize * 1.45
        let width = 720.0, height = Double(lines.count) * lineHeight + 160
        let canvas = CGSize(width: width + 160, height: height + 160)
        let context = CGContext(data: nil, width: Int(canvas.width), height: Int(canvas.height), bitsPerComponent: 8,
                                bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.setFillColor(CGColor(gray: 0.82, alpha: 1))
        context.fill(CGRect(origin: .zero, size: canvas))
        context.translateBy(x: canvas.width / 2, y: canvas.height / 2)
        context.rotate(by: rotation * .pi / 180)
        context.translateBy(x: -width / 2, y: -height / 2)
        context.setFillColor(CGColor(red: 1, green: 0.995, blue: 0.97, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let graphics = NSGraphicsContext(cgContext: context, flipped: false)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = graphics
        let attributes: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.black]
        for (index, line) in lines.enumerated() {
            let y = height - 80 - Double(index + 1) * lineHeight
            NSAttributedString(string: line, attributes: attributes).draw(at: CGPoint(x: 48, y: y))
        }
        NSGraphicsContext.restoreGraphicsState()
        return context.makeImage()!
    }

    /// Writes the image as PNG and returns its URL.
    func write(to directory: URL) throws -> URL {
        let url = directory.appending(path: "\(name).png")
        let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image(), nil)
        guard CGImageDestinationFinalize(destination) else { throw ScanError.unreadableImage }
        return url
    }

    static func row(_ label: String, _ amount: String, width: Int = 34) -> String {
        label + String(repeating: " ", count: max(1, width - label.count - amount.count)) + amount
    }
}

enum ReceiptFixtures {
    static let all: [ReceiptFixture] = [
        ReceiptFixture(name: "01-hardware", lines: [
            "CORNER HARDWARE", "412 Valencia St, San Francisco", "09/12/2026 14:32", "",
            ReceiptFixture.row("Wood glue 8oz", "6.49"), ReceiptFixture.row("Sandpaper assort", "9.99"),
            ReceiptFixture.row("Drill bits 1/8", "12.50"), ReceiptFixture.row("Painter tape x2", "11.98"),
            "", ReceiptFixture.row("Subtotal", "40.96"), ReceiptFixture.row("Tax", "3.53"),
            ReceiptFixture.row("TOTAL", "44.49"), "", "VISA **** APPROVED",
        ], merchant: "Corner Hardware", date: "2026-09-12", totalText: "44.49", taxText: "3.53", paidWith: "Visa",
           currency: "USD"),
        ReceiptFixture(name: "02-cafe", lines: [
            "BLUE KETTLE CAFE", "18 Harbor Road", "2026-10-03 08:15", "",
            ReceiptFixture.row("Flat white", "4.50"), ReceiptFixture.row("Almond croissant", "3.75"),
            "", ReceiptFixture.row("Subtotal", "8.25"), ReceiptFixture.row("Sales Tax", "0.72"),
            ReceiptFixture.row("Total", "$8.97"), "", "Paid: CASH",
        ], merchant: "Blue Kettle Cafe", date: "2026-10-03", totalText: "8.97", taxText: "0.72", paidWith: "Cash",
           currency: "USD", rotation: 1.5),
        ReceiptFixture(name: "03-grocer-eur", lines: [
            "MARKTHALLE FRISCH", "Hauptstrasse 7", "Datum: 14.08.2026", "",
            ReceiptFixture.row("Brot", "3,20"), ReceiptFixture.row("Kaese", "6,80"),
            ReceiptFixture.row("Aepfel 1kg", "2,99"), "", ReceiptFixture.row("MwSt 7%", "0,85"),
            ReceiptFixture.row("SUMME EUR", "12,99"), ReceiptFixture.row("TOTAL", "12,99 €"), "", "Mastercard",
        ], merchant: "Markthalle Frisch", date: "2026-08-14", totalText: "12.99", taxText: "0.85", paidWith: "Mastercard",
           currency: "EUR"),
        ReceiptFixture(name: "04-books", lines: [
            "HARBORSIDE BOOKS", "Order 55102", "Sep 30, 2026", "",
            ReceiptFixture.row("Paperback novel", "14.99"), ReceiptFixture.row("Notebook A5", "7.50"),
            "", ReceiptFixture.row("Subtotal", "22.49"), ReceiptFixture.row("Tax", "1.91"),
            ReceiptFixture.row("Grand Total", "24.40"), "", "AMEX ending 0000",
        ], merchant: "Harborside Books", date: "2026-09-30", totalText: "24.40", taxText: "1.91",
           paidWith: "American Express", currency: "USD", rotation: -2),
        ReceiptFixture(name: "05-fuel", lines: [
            "NORTHLINE FUEL", "Pump 4", "07/21/2026", "",
            ReceiptFixture.row("Unleaded 11.2 gal", "46.37"), "",
            ReceiptFixture.row("TOTAL", "46.37"), "", "DEBIT CARD",
        ], merchant: "Northline Fuel", date: "2026-07-21", totalText: "46.37", taxText: nil, paidWith: "Debit card",
           currency: "USD", fontSize: 22),
        ReceiptFixture(name: "06-office", lines: [
            "PAPER & PIXEL SUPPLY", "Unit 3, Mill Lane", "2026-11-02", "",
            ReceiptFixture.row("Printer paper x5", "29.95"), ReceiptFixture.row("Gel pens 12pk", "8.40"),
            ReceiptFixture.row("Desk lamp", "34.00"), ReceiptFixture.row("Cable ties", "3.10"),
            "", ReceiptFixture.row("Subtotal", "75.45"), ReceiptFixture.row("Tax", "6.60"),
            ReceiptFixture.row("Total", "82.05"), "", "VISA",
        ], merchant: "Paper & Pixel Supply", date: "2026-11-02", totalText: "82.05", taxText: "6.60", paidWith: "Visa",
           currency: "USD", rotation: 3),
        ReceiptFixture(name: "07-restaurant", lines: [
            "OLIVE & THYME", "Table 12  Server 3", "10/18/2026 19:47", "",
            ReceiptFixture.row("Bruschetta", "11.00"), ReceiptFixture.row("Risotto", "24.00"),
            ReceiptFixture.row("Sparkling water", "5.00"), "", ReceiptFixture.row("Subtotal", "40.00"),
            ReceiptFixture.row("Tax", "3.55"), ReceiptFixture.row("Tip", "8.00"),
            ReceiptFixture.row("TOTAL", "51.55"), "", "MASTERCARD APPROVED",
        ], merchant: "Olive & Thyme", date: "2026-10-18", totalText: "51.55", taxText: "3.55", paidWith: "Mastercard",
           currency: "USD"),
        ReceiptFixture(name: "08-pharmacy", lines: [
            "GREENLEAF PHARMACY", "2 Station Square", "05.06.2026", "",
            ReceiptFixture.row("Vitamin C", "6,49"), ReceiptFixture.row("Plasters", "2,99"),
            "", ReceiptFixture.row("VAT", "1,58"), ReceiptFixture.row("Total", "9,48"), "", "Apple Pay",
        ], merchant: "Greenleaf Pharmacy", date: "2026-06-05", totalText: "9.48", taxText: "1.58", paidWith: "Apple Pay",
           currency: "EUR", rotation: -1),
        ReceiptFixture(name: "09-garden", lines: [
            "SUNNYSIDE GARDEN CENTER", "Route 9", "03/04/2026", "",
            ReceiptFixture.row("Potting soil 20L", "12.99"), ReceiptFixture.row("Tomato seedlings", "15.00"),
            ReceiptFixture.row("Terracotta pot", "1,204.00"), "", ReceiptFixture.row("Subtotal", "1,231.99"),
            ReceiptFixture.row("Tax", "98.56"), ReceiptFixture.row("TOTAL", "1,330.55"), "", "VISA",
        ], merchant: "Sunnyside Garden Center", date: "2026-03-04", totalText: "1330.55", taxText: "98.56", paidWith: "Visa",
           currency: "USD"),
        ReceiptFixture(name: "10-bakery-small", lines: [
            "RISE BAKERY", "12 Mar 2026", "",
            ReceiptFixture.row("Sourdough", "6.00"), ReceiptFixture.row("Total", "6.00"), "Cash",
        ], merchant: "Rise Bakery", date: "2026-03-12", totalText: "6.00", taxText: nil, paidWith: "Cash",
           currency: "USD", fontSize: 20),
    ]
}
