import Foundation

/// One line of a receipt.
public struct ReceiptLineItem: Sendable, Hashable, Codable {
    public var name: String
    public var amount: Decimal?

    public init(name: String, amount: Decimal?) {
        self.name = name
        self.amount = amount
    }
}

/// The receipt fields a model read from recognized text, before the person
/// confirms them. Nil means the text does not show the field.
public struct ReceiptDraft: Sendable, Hashable, Codable {
    public var merchant: String?
    /// Calendar date, `YYYY-MM-DD`.
    public var date: String?
    public var total: Decimal?
    public var tax: Decimal?
    /// ISO 4217 code ("USD").
    public var currency: String?
    public var category: String?
    public var paidWith: String?
    public var lineItems: [ReceiptLineItem]

    public init(merchant: String? = nil, date: String? = nil, total: Decimal? = nil, tax: Decimal? = nil,
                currency: String? = nil, category: String? = nil, paidWith: String? = nil,
                lineItems: [ReceiptLineItem] = []) {
        self.merchant = merchant
        self.date = date
        self.total = total
        self.tax = tax
        self.currency = currency
        self.category = category
        self.paidWith = paidWith
        self.lineItems = lineItems
    }

    /// Reads the JSON object the receipt prompt asks for.
    init(json object: [String: Any]) {
        let string = AssistantAnswerParser.string
        merchant = string(object["merchant"])
        date = string(object["date"]).flatMap(ReceiptValueParser.isoDate)
        total = string(object["total"]).flatMap(ReceiptValueParser.decimal)
        tax = string(object["tax"]).flatMap(ReceiptValueParser.decimal)
        currency = string(object["currency"]).map { $0.uppercased() }
        category = string(object["category"])
        paidWith = string(object["paidWith"])
        lineItems = (object["lineItems"] as? [[String: Any]] ?? []).compactMap { item in
            guard let name = string(item["name"]) else { return nil }
            return ReceiptLineItem(name: name, amount: string(item["amount"]).flatMap(ReceiptValueParser.decimal))
        }
    }
}

/// Parses amounts and dates as receipts and models write them.
public enum ReceiptValueParser {
    /// "44.49", "$44.49", "1,330.55", "1.330,55", "44,49 €" -> Decimal.
    /// The last separator is the decimal point when the other one also
    /// appears, when one or two digits follow it, or when it is the only dot;
    /// otherwise ("1,234", "1.234.567") separators group thousands.
    /// Look-alike separators a model or OCR may write ("٫", "．", "，")
    /// count as their ASCII forms, so the point is never silently dropped.
    public static func decimal(_ text: String) -> Decimal? {
        var cleaned = ""
        for character in text {
            switch character {
            case "0"..."9", ".", ",", "-": cleaned.append(character)
            case "\u{066B}", "\u{FF0E}", "\u{2024}": cleaned.append(".")
            case "\u{FF0C}", "\u{066C}": cleaned.append(",")
            default: continue
            }
        }
        guard cleaned.contains(where: \.isNumber) else { return nil }
        let separators = cleaned.indices.filter { cleaned[$0] == "." || cleaned[$0] == "," }
        var digits = cleaned
        if let last = separators.last {
            let decimals = cleaned.distance(from: last, to: cleaned.endIndex) - 1
            let mixed = Set(separators.map { cleaned[$0] }).count > 1
            let onlyDot = separators.count == 1 && cleaned[last] == "."
            let isPoint = mixed || (1...2).contains(decimals) || onlyDot
            digits = ""
            for index in cleaned.indices {
                let character = cleaned[index]
                if character == "." || character == "," {
                    if isPoint && index == last { digits.append(".") }
                } else {
                    digits.append(character)
                }
            }
        }
        return Decimal(string: digits, locale: Locale(identifier: "en_US_POSIX"))
    }

    /// Accepts `YYYY-MM-DD`, `MM/DD/YYYY`, `DD.MM.YYYY` and `MM/DD/YY`;
    /// returns `YYYY-MM-DD`, or nil when it is not a real date.
    public static func isoDate(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        let numbers = trimmed.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
        guard numbers.count >= 3 else { return nil }
        var year: Int, month: Int, day: Int
        if numbers[0] > 999 {
            (year, month, day) = (numbers[0], numbers[1], numbers[2])
        } else if trimmed.contains(".") {
            (day, month, year) = (numbers[0], numbers[1], numbers[2])
        } else {
            (month, day, year) = (numbers[0], numbers[1], numbers[2])
        }
        if year < 100 { year += 2000 }
        var components = DateComponents(year: year, month: month, day: day)
        components.calendar = Calendar(identifier: .gregorian)
        guard (1...12).contains(month), (1...31).contains(day), components.isValidDate else { return nil }
        return String(format: "%04d-%02d-%02d", year, month, day)
    }
}
