import Foundation

/// The receipt fields the Scan screen shows, in mockup order.
public enum ScanFieldKey: String, Sendable, Hashable, Codable, CaseIterable, Identifiable {
    case merchant, date, total, tax, category, paidWith

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .merchant: "Merchant"
        case .date: "Date"
        case .total: "Total"
        case .tax: "Tax"
        case .category: "Category"
        case .paidWith: "Paid with"
        }
    }
}

/// "Sure" when the model and the printed text agree; "Check" otherwise.
public enum ScanFieldConfidence: String, Sendable, Hashable, Codable {
    case sure, check

    public var label: String {
        switch self {
        case .sure: "Sure"
        case .check: "Check"
        }
    }
}

/// One field: its value as the person sees and edits it.
public struct ScanField: Sendable, Hashable, Codable, Identifiable {
    public var key: ScanFieldKey
    public var value: String
    public var confidence: ScanFieldConfidence

    public var id: ScanFieldKey { key }

    public init(key: ScanFieldKey, value: String, confidence: ScanFieldConfidence) {
        self.key = key
        self.value = value
        self.confidence = confidence
    }
}

/// Everything a scan stores in `ScanRecord.fieldsJSON` (owned here, U4).
public struct ScanFields: Sendable, Hashable, Codable {
    public static let currentVersion = 1

    public var version: Int
    public var fields: [ScanField]
    public var lineItems: [ReceiptLineItem]
    public var currency: String?
    /// The model that filled the fields, nil when only the text was read.
    public var filledBy: AssistantModelKind?

    public init(fields: [ScanField], lineItems: [ReceiptLineItem] = [], currency: String? = nil,
                filledBy: AssistantModelKind? = nil) {
        self.version = Self.currentVersion
        self.fields = fields
        self.lineItems = lineItems
        self.currency = currency
        self.filledBy = filledBy
    }

    public subscript(key: ScanFieldKey) -> ScanField? {
        get { fields.first { $0.key == key } }
        set {
            fields.removeAll { $0.key == key }
            if let newValue { fields.append(newValue) }
            fields.sort { Self.order($0.key) < Self.order($1.key) }
        }
    }

    /// The value of a field, nil when empty.
    public func value(_ key: ScanFieldKey) -> String? {
        guard let value = self[key]?.value.trimmingCharacters(in: .whitespaces), !value.isEmpty else { return nil }
        return value
    }

    public var needsChecking: [ScanFieldKey] {
        fields.filter { $0.confidence == .check }.map(\.key)
    }

    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(self)
    }

    public static func decode(_ data: Data) throws -> ScanFields {
        try JSONDecoder().decode(ScanFields.self, from: data)
    }

    private static func order(_ key: ScanFieldKey) -> Int {
        ScanFieldKey.allCases.firstIndex(of: key) ?? 0
    }
}

/// Merges the model's reading with the printed-text reading into the fields
/// the person confirms.
public enum ReceiptFieldMerger {
    /// Vision lines below this confidence make a field "Check" even when the
    /// two readings agree.
    static let lowRecognitionConfidence: Float = 0.5

    public static func merge(model: ReceiptDraft?, text: ReceiptDraft, reading: ScanReading,
                             filledBy: AssistantModelKind?) -> ScanFields {
        let currency = model?.currency ?? text.currency
        let lowConfidence = reading.lines.contains { $0.confidence < lowRecognitionConfidence }
        func field(_ key: ScanFieldKey, model modelValue: String?, text textValue: String?,
                   inferred: Bool = false) -> ScanField {
            let value = modelValue ?? textValue ?? ""
            var agree = false
            if let modelValue, let textValue {
                agree = modelValue.caseInsensitiveCompare(textValue) == .orderedSame
            }
            let printed = modelValue.map { appearsInText($0, reading: reading) } ?? false
            let sure = !value.isEmpty && !inferred && !lowConfidence && (agree || (textValue == nil && printed))
            return ScanField(key: key, value: value, confidence: sure ? .sure : .check)
        }
        let money = { (amount: Decimal?) in amount.map { ScanFormat.amount($0, currency: currency) } }
        let date = { (iso: String?) in iso.flatMap(ScanFormat.displayDate) }
        return ScanFields(
            fields: [
                field(.merchant, model: model?.merchant, text: text.merchant),
                field(.date, model: date(model?.date), text: date(text.date)),
                field(.total, model: money(model?.total), text: money(text.total)),
                field(.tax, model: money(model?.tax), text: money(text.tax)),
                field(.category, model: model?.category, text: nil, inferred: true),
                field(.paidWith, model: model?.paidWith, text: text.paidWith),
            ],
            lineItems: model?.lineItems ?? [],
            currency: currency,
            filledBy: filledBy)
    }

    /// Whether the model's value is printed on the receipt (the merchant
    /// name, say), ignoring case and spacing.
    static func appearsInText(_ value: String, reading: ScanReading) -> Bool {
        let squash = { (text: String) in text.lowercased().filter { $0.isLetter || $0.isNumber } }
        let needle = squash(value)
        guard needle.count >= 3 else { return false }
        return squash(reading.transcript).contains(needle)
    }
}

/// How scan values are written on screen and in the CSV.
public enum ScanFormat {
    /// "$44.49" for USD, "44.49 EUR" style via the system formatter otherwise.
    public static func amount(_ value: Decimal, currency: String?) -> String {
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US")
        formatter.numberStyle = currency == nil ? .decimal : .currency
        formatter.currencyCode = currency
        formatter.minimumFractionDigits = 2
        formatter.maximumFractionDigits = 2
        return formatter.string(from: value as NSDecimalNumber) ?? "\(value)"
    }

    /// `YYYY-MM-DD` to "Sep 12, 2026".
    public static func displayDate(_ iso: String) -> String? {
        guard let date = isoFormatter.date(from: iso) else { return nil }
        return displayFormatter.string(from: date)
    }

    /// "Sep 12, 2026" (or any form `ReceiptValueParser` reads) to `YYYY-MM-DD`.
    public static func isoDate(fromDisplay text: String) -> String? {
        if let date = displayFormatter.date(from: text) { return isoFormatter.string(from: date) }
        return ReceiptValueParser.isoDate(text)
    }

    private static var isoFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }

    private static var displayFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "MMM d, yyyy"
        return formatter
    }
}
