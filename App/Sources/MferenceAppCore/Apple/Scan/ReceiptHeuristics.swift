import Foundation

/// Reads the plain facts of a receipt (total, tax, date, card) straight from
/// the recognized lines, without a model. Used to check the model's answer:
/// a field both agree on is "Sure"; anything else is "Check".
public enum ReceiptHeuristics {
    public static func read(_ reading: ScanReading) -> ReceiptDraft {
        let lines = reading.rows
        var draft = ReceiptDraft()
        draft.merchant = merchant(in: lines)
        draft.total = amount(in: lines, labelled: ["grand total", "total due", "amount due", "total", "balance due"],
                             excluding: ["subtotal", "sub total", "sub-total", "total tax", "tax total"])
        draft.tax = amount(in: lines, labelled: taxLabels,
                           excluding: ["before tax", "pre-tax", "excl", "incl. tax total"])
        draft.date = date(in: lines)
        draft.paidWith = payment(in: lines)
        draft.currency = currency(in: reading.transcript)
        return draft
    }

    /// The first line with at least three letters that is not an address,
    /// a phone number or a date.
    static func merchant(in lines: [String]) -> String? {
        for line in lines.prefix(4) {
            let letters = line.filter(\.isLetter).count
            let digits = line.filter(\.isNumber).count
            guard letters >= 3, digits <= letters / 2 else { continue }
            let lower = line.lowercased()
            if ["receipt", "invoice", "welcome", "thank you"].contains(where: lower.hasPrefix) { continue }
            return titleCased(line.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters)))
        }
        return nil
    }

    /// The amount on the last line whose label matches (receipts print the
    /// final total last), or on the next line when the amount is printed below
    /// its label.
    static func amount(in lines: [String], labelled labels: [String], excluding: [String]) -> Decimal? {
        var found: Decimal?
        for (index, line) in lines.enumerated() {
            let lower = line.lowercased()
            guard labels.contains(where: { containsWord($0, in: lower) }),
                  !excluding.contains(where: lower.contains) else { continue }
            if let value = lastAmount(in: line) {
                found = value
            } else if index + 1 < lines.count, let value = lastAmount(in: lines[index + 1]),
                      lines[index + 1].filter(\.isLetter).count <= 3 {
                found = value
            }
        }
        return found
    }

    /// The last amount on a line, keeping a refund's sign: `-44.49`,
    /// `-$44.49`, `$-44.49`, `44.49-` and `(44.49)` are all negative; a
    /// spaced dash ("1/8 - 12.50") is a separator, not a sign.
    static func lastAmount(in line: String) -> Decimal? {
        let pattern = /(\()?(-)?[$€£¥]?(-)?(\d+(?:[,.]\d{3})*[.,]\d{2})(?!\d)(\))?(-)?/
        guard let match = line.matches(of: pattern).last,
              let magnitude = ReceiptValueParser.decimal(String(match.output.4)) else { return nil }
        let output = match.output
        let negative = (output.1 != nil && output.5 != nil) || output.2 != nil || output.3 != nil || output.6 != nil
        return negative ? -magnitude : magnitude
    }

    /// Every amount printed on a line, without its sign: what a model's
    /// amount must match to be taken. Amounts with cents anywhere; whole
    /// amounts ("¥1200", "1,200 €", "TOTAL 1200") only beside a currency sign
    /// or as the last figure on a total or tax line, so an address number or
    /// a card's last four digits never vouch for a model's amount.
    static func amounts(in line: String) -> [Decimal] {
        var found = line.matches(of: /\d+(?:[,.]\d{3})*[.,]\d{2}(?!\d)/).compactMap {
            ReceiptValueParser.decimal(String($0.output))
        }
        let whole = { (text: Substring) in
            Decimal(string: String(text.filter(\.isNumber)), locale: Locale(identifier: "en_US_POSIX"))
        }
        // The lookahead refuses any further digit after separators, so the
        // match cannot back off into a prefix of "$44.49" ("4") or "1,200".
        let signed = /(?:[$€£¥]|\b(?:USD|EUR|GBP|JPY|CAD|AUD)\b)\s?(\d{1,3}(?:[,.]\d{3})+|\d+)(?![\d.,]*\d)/
        let trailing = /(?:^|[^\d.,])(\d{1,3}(?:[,.]\d{3})+|\d+)\s?(?:[€¥£円]|\b(?:USD|EUR|GBP|JPY|CAD|AUD)\b)/
        found += line.matches(of: signed).compactMap { whole($0.output.1) }
        found += line.matches(of: trailing).compactMap { whole($0.output.1) }
        let lower = line.lowercased()
        let labelled = (["total", "amount due", "balance due"] + taxLabels).contains { containsWord($0, in: lower) }
        if labelled, let last = line.firstMatch(of: /(?:^|[^\d.,])(\d{1,3}(?:[,.]\d{3})+|\d+)\s*$/) {
            found += [whole(last.output.1)].compactMap { $0 }
        }
        return found
    }

    /// Tax line labels, including Canada's HST, PST and QST (TPS and TVQ in
    /// French), and Dutch BTW and Scandinavian moms.
    static let taxLabels = ["sales tax", "tax", "vat", "gst", "hst", "pst", "qst", "tps", "tvq",
                            "mwst", "ust", "tva", "iva", "btw", "moms"]

    /// Whether the receipt prints a tax line at all.
    static func printsTax(in lines: [String]) -> Bool {
        lines.contains { line in taxLabels.contains { containsWord($0, in: line.lowercased()) } }
    }

    static func date(in lines: [String]) -> String? {
        let patterns: [Regex<Substring>] = [
            /\d{4}-\d{1,2}-\d{1,2}/,
            /\d{1,2}\/\d{1,2}\/\d{2,4}/,
            /\d{1,2}\.\d{1,2}\.\d{2,4}/,
        ]
        for line in lines {
            for pattern in patterns {
                if let match = line.firstMatch(of: pattern),
                   let date = ReceiptValueParser.isoDate(String(match.output)) {
                    return date
                }
            }
            if let date = writtenDate(in: line) { return date }
        }
        return nil
    }

    /// "Sep 12, 2026", "12 September 2026".
    static func writtenDate(in line: String) -> String? {
        let formats = ["MMM d, yyyy", "MMMM d, yyyy", "d MMM yyyy", "d MMMM yyyy"]
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        let candidates = line.matches(of: /[A-Za-z]{3,9}\.? \d{1,2},? \d{4}|\d{1,2} [A-Za-z]{3,9}\.? \d{4}/)
        for candidate in candidates {
            let text = String(candidate.output).replacingOccurrences(of: ".", with: "")
            for format in formats {
                formatter.dateFormat = format
                if let date = formatter.date(from: text) {
                    let parts = Calendar(identifier: .gregorian).dateComponents(in: formatter.timeZone, from: date)
                    return String(format: "%04d-%02d-%02d", parts.year ?? 0, parts.month ?? 0, parts.day ?? 0)
                }
            }
        }
        return nil
    }

    static func payment(in lines: [String]) -> String? {
        let brands = [("visa", "Visa"), ("mastercard", "Mastercard"), ("master card", "Mastercard"),
                      ("amex", "American Express"), ("american express", "American Express"),
                      ("discover", "Discover"), ("apple pay", "Apple Pay"), ("debit", "Debit card"),
                      ("cash", "Cash")]
        for line in lines.reversed() {
            let lower = line.lowercased()
            if let brand = brands.first(where: { containsWord($0.0, in: lower) }) { return brand.1 }
        }
        return nil
    }

    static func currency(in text: String) -> String? {
        if text.contains("€") || text.contains("EUR") { return "EUR" }
        if text.contains("£") || text.contains("GBP") { return "GBP" }
        if text.contains("¥") || text.contains("JPY") { return "JPY" }
        if text.contains("$") || text.contains("USD") { return "USD" }
        return nil
    }

    static func containsWord(_ word: String, in text: String) -> Bool {
        var searchStart = text.startIndex
        while let range = text.range(of: word, range: searchStart..<text.endIndex) {
            let before = range.lowerBound == text.startIndex ? nil : text[text.index(before: range.lowerBound)]
            let after = range.upperBound == text.endIndex ? nil : text[range.upperBound]
            if !(before?.isLetter ?? false), !(after?.isLetter ?? false) { return true }
            searchStart = range.upperBound
        }
        return false
    }

    static func titleCased(_ text: String) -> String {
        guard text == text.uppercased() else { return text }
        return text.lowercased().split(separator: " ", omittingEmptySubsequences: false)
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }
}
