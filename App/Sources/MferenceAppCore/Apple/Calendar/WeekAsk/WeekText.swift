import Foundation

/// How Ask about your week writes days, times and lengths: one place, so the
/// fixed sentences, the phraser's facts and the checker agree on every word.
/// Times follow the calendar's locale (12 or 24 hours).
public struct WeekText: Sendable {
    public var calendar: Calendar
    /// The moment "today" and "tomorrow" are measured from.
    public var now: Date

    public init(calendar: Calendar, now: Date) {
        self.calendar = calendar
        self.now = now
    }

    /// Whether the locale writes times with AM and PM.
    public var usesTwelveHours: Bool {
        (DateFormatter.dateFormat(fromTemplate: "j", options: 0, locale: calendar.locale) ?? "").contains("a")
    }

    private func formatter(_ template: String) -> DateFormatter {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = calendar.locale
        formatter.timeZone = calendar.timeZone
        formatter.setLocalizedDateFormatFromTemplate(template)
        return formatter
    }

    /// "Tuesday".
    public func weekday(_ date: Date) -> String { formatter("EEEE").string(from: date) }

    /// "Tue".
    public func shortWeekday(_ date: Date) -> String { formatter("EEE").string(from: date) }

    /// "Sep 15".
    public func monthDay(_ date: Date) -> String { formatter("MMMd").string(from: date) }

    /// "today", "tomorrow", or the weekday.
    public func dayWord(_ date: Date) -> String {
        if calendar.isDate(date, inSameDayAs: now) { return "today" }
        if let tomorrow = calendar.date(byAdding: .day, value: 1, to: now), calendar.isDate(date, inSameDayAs: tomorrow) {
            return "tomorrow"
        }
        return weekday(date)
    }

    /// "Today", "Tomorrow" or the weekday, to start a sentence.
    public func dayWordCapitalized(_ date: Date) -> String {
        let word = dayWord(date)
        return word.prefix(1).uppercased() + word.dropFirst()
    }

    /// "1 PM", "1:30 PM", "13:00".
    public func time(_ date: Date) -> String {
        let minute = calendar.component(.minute, from: date)
        // Foundation puts a narrow no-break space before AM/PM; plain spaces
        // keep the text the same for the model, the checker and the tests.
        return formatter(usesTwelveHours && minute == 0 ? "j" : "jmm").string(from: date)
            .replacingOccurrences(of: "\u{202F}", with: " ")
            .replacingOccurrences(of: "\u{00A0}", with: " ")
    }

    /// "1 to 5 PM", "11 AM to 1 PM", "13:00 to 17:00". An end at midnight
    /// reads as the end of the day.
    public func range(_ start: Date, _ end: Date) -> String {
        guard usesTwelveHours else { return "\(time(start)) to \(time(end))" }
        let sameHalf = (calendar.component(.hour, from: start) < 12) == (calendar.component(.hour, from: end) < 12)
            && calendar.isDate(start, inSameDayAs: end)
        guard sameHalf else { return "\(time(start)) to \(time(end))" }
        let symbols = formatter("j")
        var bare = time(start)
        for symbol in [symbols.amSymbol, symbols.pmSymbol].compactMap({ $0 }) where !symbol.isEmpty {
            bare = bare.replacingOccurrences(of: symbol, with: "")
        }
        return "\(bare.trimmingCharacters(in: .whitespaces)) to \(time(end))"
    }

    /// "2 h", "2 h 30 min", "45 min".
    public func duration(minutes: Int) -> String {
        let hours = minutes / 60, rest = minutes % 60
        if hours == 0 { return "\(rest) min" }
        return rest == 0 ? "\(hours) h" : "\(hours) h \(rest) min"
    }

    /// "Sep 14 to 20".
    public func weekRange(_ interval: DateInterval) -> String {
        let last = calendar.date(byAdding: .day, value: -1, to: interval.end) ?? interval.start
        if calendar.isDate(interval.start, equalTo: last, toGranularity: .month) {
            return "\(monthDay(interval.start)) to \(calendar.component(.day, from: last))"
        }
        return "\(monthDay(interval.start)) to \(monthDay(last))"
    }

    /// The length a question asks for, in minutes ("2 hours", "1 hour 30
    /// minutes", "2h 15m", "an hour and a half", "half an hour"), or nil.
    /// Block uses it for the event it adds.
    public static func requestedMinutes(in question: String) -> Int? {
        let text = " " + question.lowercased() + " "
        func has(_ pattern: String) -> Bool { text.range(of: pattern, options: .regularExpression) != nil }
        if has(#"\bhalf an? hour\b"#) { return 30 }
        // Digits may touch the unit ("2h"); words need a space ("am" is not
        // "a minute"). Every part counts.
        guard let regex = try? NSRegularExpression(
            pattern: #"\b(?:(\d+(?:\.\d+)?)\s*|(an?|one|two|three|four|five|six)\s+)(hours?|hrs?|h|minutes?|mins?|m)\b"#)
        else { return nil }
        let words: [String: Double] = ["a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6]
        var total = 0.0
        var sawHours = false
        for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            let group = { (index: Int) -> String? in Range(match.range(at: index), in: text).map { String(text[$0]) } }
            guard let unit = group(3), let amount = group(1).flatMap(Double.init) ?? group(2).flatMap({ words[$0] })
            else { continue }
            // A bare "m" only after digits ("15m"), never "a m".
            if unit == "m", group(1) == nil { continue }
            if unit.hasPrefix("h") { sawHours = true }
            total += unit.hasPrefix("h") ? amount * 60 : amount
        }
        if sawHours, has(#"\band a half\b"#) { total += 30 }
        return total > 0 ? Int(total.rounded()) : nil
    }

    /// "A, B and C".
    public static func list(_ items: [String]) -> String {
        switch items.count {
        case 0: ""
        case 1: items[0]
        default: items.dropLast().joined(separator: ", ") + " and " + items[items.count - 1]
        }
    }

    /// "1 event", "2 events".
    public static func count(_ value: Int, _ singular: String, _ plural: String? = nil) -> String {
        "\(value) \(value == 1 ? singular : plural ?? singular + "s")"
    }
}
