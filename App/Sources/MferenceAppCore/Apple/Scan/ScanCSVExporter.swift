import Foundation

/// One confirmed scan as a CSV row (P8: CSV first, Numbers later).
public struct ScanExportRow: Sendable, Hashable {
    public var scannedAt: Date
    public var fields: ScanFields

    public init(scannedAt: Date, fields: ScanFields) {
        self.scannedAt = scannedAt
        self.fields = fields
    }
}

/// RFC 4180 CSV that Numbers and Excel open as-is.
public enum ScanCSVExporter {
    public static let header = ["Date", "Merchant", "Category", "Total", "Tax", "Currency", "Paid with", "Items"]

    public static func csv(_ rows: [ScanExportRow]) -> String {
        var lines = [header.map(cell).joined(separator: ",")]
        for row in rows {
            let fields = row.fields
            let items = fields.lineItems.map { item in
                item.amount.map { "\(item.name) \(plain($0))" } ?? item.name
            }.joined(separator: "; ")
            let values = [
                fields.value(.date).flatMap(ScanFormat.isoDate(fromDisplay:)) ?? fields.value(.date) ?? "",
                fields.value(.merchant) ?? "",
                fields.value(.category) ?? "",
                fields.value(.total).flatMap(ReceiptValueParser.decimal).map(plain) ?? "",
                fields.value(.tax).flatMap(ReceiptValueParser.decimal).map(plain) ?? "",
                fields.currency ?? "",
                fields.value(.paidWith) ?? "",
                items,
            ]
            lines.append(values.map(cell).joined(separator: ","))
        }
        return lines.joined(separator: "\r\n") + "\r\n"
    }

    /// The file to save: UTF-8 with a byte order mark, so Excel reads accents
    /// and currency signs correctly (Numbers ignores it).
    public static func fileData(_ rows: [ScanExportRow]) -> Data {
        Data([0xEF, 0xBB, 0xBF]) + Data(csv(rows).utf8)
    }

    /// Quotes a cell when needed, and defuses text a spreadsheet would run as
    /// a formula (a leading `=`, `+`, `-`, `@`, tab or return).
    static func cell(_ value: String) -> String {
        var text = value
        if let first = text.first, "=+-@\t\r".contains(first), text.wholeMatch(of: /-?\d+(\.\d+)?/) == nil {
            text = "'" + text
        }
        guard text.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" || $0 == "\r" }) else { return text }
        return "\"" + text.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    static func plain(_ value: Decimal) -> String {
        var rounded = Decimal()
        var input = value
        NSDecimalRound(&rounded, &input, 2, .bankers)
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.minimumFractionDigits = 2
        formatter.maximumFractionDigits = 2
        formatter.usesGroupingSeparator = false
        return formatter.string(from: rounded as NSDecimalNumber) ?? "\(rounded)"
    }
}
