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
