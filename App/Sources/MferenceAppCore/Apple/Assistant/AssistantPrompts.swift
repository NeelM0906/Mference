import Foundation

/// The prompts behind the assistant jobs. Versioned so results can be
/// compared across changes; both models get the same text.
public enum AssistantPrompts {
    public static let version = 1

    static let title = """
        You write short titles. Reply with one title of at most six words that names the topic \
        of the text. No quotes, no punctuation at the end, nothing else.
        """

    static let summary = """
        You summarize text in one or two plain sentences of at most 40 words. Keep names, dates \
        and amounts exact. Reply with the summary only.
        """

    static let mailSort = """
        You sort email into buckets. You get the buckets as `key: description` lines and the \
        messages as numbered blocks. Reply with JSON only: an object that maps each message \
        number (as a string) to one bucket key from the list. Use every message number once.
        """

    static let receiptFields = """
        You read the text of a scanned receipt and fill its fields. Reply with JSON only, with \
        these keys: "merchant" (store name), "date" (YYYY-MM-DD), "total" (the amount paid, digits \
        and a dot, no currency sign), "tax" (same format), "currency" (ISO code such as USD or EUR), \
        "category" (one or two words such as Groceries, Dining, Travel, Home improvement, Office), \
        "paidWith" (card brand or Cash), "lineItems" (array of {"name", "amount"}). Use null for a \
        field the text does not show. Never invent values.
        """

    static let mailEvent = """
        You find meetings in email. If the email proposes or confirms a specific meeting, call or \
        appointment time, reply with JSON only: {"title": short event title, "start": \
        "YYYY-MM-DDTHH:MM", "end": "YYYY-MM-DDTHH:MM" or null}. Resolve weekdays and "tomorrow" \
        against the date the email was received. If there is no specific time, reply {"title": null}.
        """

    static func eventPrompt(subject: String, body: String, received: Date, timeZone: TimeZone) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "EEEE yyyy-MM-dd HH:mm"
        return """
            Received: \(formatter.string(from: received))
            Subject: \(subject)

            \(clip(body, to: 2_000))
            """
    }

    static func mailSortPrompt(messages: [MailSortItem], buckets: [MailSortBucket]) -> String {
        var lines = ["Buckets:"]
        lines += buckets.map { "\($0.key): \($0.description)" }
        lines.append("")
        lines.append("Messages:")
        for (index, message) in messages.enumerated() {
            lines.append("[\(index + 1)]")
            lines.append("From: \(message.sender)")
            lines.append("Subject: \(message.subject)")
            lines.append("Text: \(clip(message.snippet, to: 400))")
        }
        return lines.joined(separator: "\n")
    }

    /// Bounds the text sent to a small model, cutting at a word boundary.
    static func clip(_ text: String, to limit: Int) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > limit else { return trimmed }
        let cut = trimmed.prefix(limit)
        if let space = cut.lastIndex(where: \.isWhitespace) {
            return String(cut[..<space]) + "…"
        }
        return String(cut) + "…"
    }
}

/// Reads the answers: tolerant of code fences and chatter around the JSON.
enum AssistantAnswerParser {
    /// The first balanced JSON object in `text`.
    static func jsonObject(in text: String) -> [String: Any]? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var index = start
        while index < text.endIndex {
            let character = text[index]
            if inString {
                if escaped { escaped = false }
                else if character == "\\" { escaped = true }
                else if character == "\"" { inString = false }
            } else if character == "\"" {
                inString = true
            } else if character == "{" {
                depth += 1
            } else if character == "}" {
                depth -= 1
                if depth == 0 {
                    let slice = text[start...index]
                    let object = try? JSONSerialization.jsonObject(with: Data(slice.utf8))
                    return object as? [String: Any]
                }
            }
            index = text.index(after: index)
        }
        return nil
    }

    /// One line, without wrapping quotes, a "Title:" prefix or a final period.
    static func singleLine(_ text: String, maxWords: Int? = nil) -> String? {
        var line = text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty } ?? ""
        for prefix in ["Title:", "title:", "Summary:", "summary:"] where line.hasPrefix(prefix) {
            line = String(line.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
        }
        line = line.trimmingCharacters(in: CharacterSet(charactersIn: "\"'“”‘’*#` "))
        if line.hasSuffix(".") { line.removeLast() }
        if let maxWords {
            let words = line.split(separator: " ")
            if words.count > maxWords { line = words.prefix(maxWords).joined(separator: " ") }
        }
        return line.isEmpty ? nil : line
    }

    /// Plain text without fences or a leading label.
    static func paragraph(_ text: String) -> String? {
        var body = text.replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["Summary:", "summary:"] where body.hasPrefix(prefix) {
            body = String(body.dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return body.isEmpty ? nil : body
    }

    static func string(_ value: Any?) -> String? {
        switch value {
        case let text as String:
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty || trimmed.lowercased() == "null" ? nil : trimmed
        case let number as NSNumber:
            return number.stringValue
        default:
            return nil
        }
    }
}
