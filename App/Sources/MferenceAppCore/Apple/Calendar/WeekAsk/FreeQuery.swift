import Foundation

/// A free-time question, as a chip or the model asks it: Swift answers it
/// exactly, with fixed windows, so the same question always gets the same
/// answer (spec "Free-time rules").
///
/// `{"free_query": {"min_minutes": 90, "days": ["2026-09-16"], "part": "morning"}}`
public struct FreeQuery: Sendable, Hashable {
    /// A part of the day, each with a fixed window.
    public enum Part: String, Sendable, Hashable, CaseIterable {
        case morning, afternoon, evening, night

        /// Minutes after midnight the part covers in a free-time search.
        public var window: Range<Int> {
            switch self {
            case .morning: 9 * 60..<12 * 60
            case .afternoon: 12 * 60..<17 * 60
            case .evening: 17 * 60..<21 * 60
            case .night: 21 * 60..<24 * 60
            }
        }
    }

    /// The shortest opening that counts; nil is the default, 30 minutes.
    public var minMinutes: Int?
    /// ISO dates inside the asked week; nil is every weekday from today on.
    public var days: [String]?
    public var part: Part?
    /// Minutes after midnight ("after 3 PM" is 900).
    public var after: Int?
    /// Minutes after midnight; 1440 is midnight at the end of the day.
    public var before: Int?
    /// Days the question named outside the asked week, as ISO dates. The
    /// answer points to their week instead of saying "no free time".
    public var outside: [String]

    public init(minMinutes: Int? = nil, days: [String]? = nil, part: Part? = nil, after: Int? = nil,
                before: Int? = nil, outside: [String] = []) {
        self.minMinutes = minMinutes
        self.days = days
        self.part = part
        self.after = after
        self.before = before
        self.outside = outside
    }

    /// Shorter gaps are not offered unless the question asks for less.
    public static let defaultMinutes = WeekFacts.shortestOpenMinutes

    /// The shortest opening the search keeps.
    public var minutes: Int { minMinutes ?? Self.defaultMinutes }

    // MARK: The reply, as JSON

    /// The fields, written as the model writes them (only those set), in a
    /// fixed order: the assistant turn a follow-up replays.
    public var json: String {
        var fields: [String] = []
        if let minMinutes { fields.append("\"min_minutes\": \(minMinutes)") }
        let named = (days ?? []) + outside
        if !named.isEmpty { fields.append("\"days\": [" + named.map { "\"\($0)\"" }.joined(separator: ", ") + "]") }
        if let part { fields.append("\"part\": \"\(part.rawValue)\"") }
        if let after { fields.append("\"after\": \"\(Self.hhmm(after))\"") }
        if let before { fields.append("\"before\": \"\(Self.hhmm(before))\"") }
        return "{\"free_query\": {" + fields.joined(separator: ", ") + "}}"
    }

    /// "15:00", "24:00".
    public static func hhmm(_ minutes: Int) -> String { String(format: "%02d:%02d", minutes / 60, minutes % 60) }

    // MARK: Reading the model's fields

    /// The query in a reply's `free_query` value, its days resolved inside
    /// `week` ("Friday", "today", "tomorrow" and ISO dates). Lenient:
    /// numbers may come as strings, and an unreadable field is left at its
    /// default.
    public static func decode(_ value: Any?, in week: WeekSnapshot) -> FreeQuery {
        guard let object = value as? [String: Any] else { return FreeQuery() }
        var query = FreeQuery()
        // A length past a whole day cannot be met; it is unreadable, not
        // a number to multiply.
        if let minutes = integer(object["min_minutes"]), (1...24 * 60).contains(minutes) { query.minMinutes = minutes }
        let named: [String] = switch object["days"] {
        case let list as [Any]: list.compactMap { $0 as? String }
        case let one as String: [one]
        default: []
        }
        var inside: [Date] = []
        for text in named {
            for resolved in resolve(text, in: week) {
                switch resolved {
                case .inside(let day): if !inside.contains(day) { inside.append(day) }
                case .outside(let iso): if !query.outside.contains(iso) { query.outside.append(iso) }
                }
            }
        }
        if !inside.isEmpty {
            query.days = inside.sorted().map { WeekContext.isoDate($0, in: week.calendar) }
        }
        if let text = (object["part"] as? String)?.lowercased().trimmingCharacters(in: .whitespaces) {
            query.part = Part(rawValue: text) ?? (text == "tonight" ? .evening : nil)
        }
        query.after = clockMinutes(object["after"])
        query.before = clockMinutes(object["before"])
        return query
    }

    static func integer(_ value: Any?) -> Int? {
        switch value {
        case let number as NSNumber:
            let value = number.doubleValue
            return value.isFinite && abs(value) < 1_000_000 ? Int(value) : nil
        case let text as String:
            let digits = text.trimmingCharacters(in: .whitespaces).prefix { $0.isNumber }.prefix(7)
            return Int(digits)
        default: return nil
        }
    }

    /// "15:00", "15", "3 PM", "3:30pm", "15:00:00" as minutes after midnight.
    static func clockMinutes(_ value: Any?) -> Int? {
        if let number = value as? NSNumber {
            let hour = Int(truncating: number)
            return (0...24).contains(hour) ? hour * 60 : nil
        }
        guard let text = (value as? String)?.lowercased().trimmingCharacters(in: .whitespaces), !text.isEmpty else {
            return nil
        }
        if text == "noon" { return 12 * 60 }
        if text == "midnight" { return 24 * 60 }
        let range = NSRange(text.startIndex..., in: text)
        guard let match = clockPattern.firstMatch(in: text, range: range),
              let hourRange = Range(match.range(at: 1), in: text), var hour = Int(text[hourRange]) else { return nil }
        let minute = Range(match.range(at: 2), in: text).flatMap { Int(text[$0]) } ?? 0
        if let meridiemRange = Range(match.range(at: 3), in: text) {
            guard (1...12).contains(hour) else { return nil }
            let pm = text[meridiemRange].hasPrefix("p")
            hour = hour % 12 + (pm ? 12 : 0)
        }
        let total = hour * 60 + minute
        return minute < 60 && (0...24 * 60).contains(total) ? total : nil
    }

    private static let clockPattern = WeekText.regex(#"^(\d{1,2})(?::(\d{2}))?(?::\d{2})?\s*(a\.?m\.?|p\.?m\.?)?$"#)

    enum ResolvedDay: Equatable {
        case inside(Date)
        case outside(String)
    }

    /// A day the model named, inside the asked week or not.
    static func resolve(_ text: String, in week: WeekSnapshot) -> [ResolvedDay] {
        let calendar = week.calendar
        let word = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
        func place(_ day: Date) -> ResolvedDay {
            week.contains(day) ? .inside(calendar.startOfDay(for: day)) : .outside(WeekContext.isoDate(day, in: calendar))
        }
        switch word {
        case "today": return [place(week.today)]
        case "tomorrow": return calendar.date(byAdding: .day, value: 1, to: week.today).map { [place($0)] } ?? []
        case "yesterday": return calendar.date(byAdding: .day, value: -1, to: week.today).map { [place($0)] } ?? []
        case "weekend", "this weekend": return week.days.filter { calendar.isDateInWeekend($0) }.map { .inside($0) }
        default: break
        }
        if let day = isoFormatter(calendar).date(from: word) { return [place(day)] }
        // A weekday's name, in English or the Mac's language: that day of
        // the asked week.
        let names = englishWeekdays + localWeekdays(calendar)
        for day in week.days {
            let weekday = calendar.component(.weekday, from: day) - 1
            if names.contains(where: { $0[weekday] == word }) { return [.inside(day)] }
        }
        return []
    }

    private static let englishWeekdays: [[String]] = {
        let english = Calendar(identifier: .gregorian)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = english
        let full = (formatter.weekdaySymbols ?? []).map { $0.lowercased() }
        let short = (formatter.shortWeekdaySymbols ?? []).map { $0.lowercased() }
        return [full, short]
    }()

    private static func localWeekdays(_ calendar: Calendar) -> [[String]] {
        [calendar.weekdaySymbols.map { $0.lowercased() }, calendar.shortWeekdaySymbols.map { $0.lowercased() }]
            .filter { $0.count == 7 }
    }

    /// "2026-09-16" in `calendar`'s time zone, made once per time zone.
    static func isoFormatter(_ calendar: Calendar) -> DateFormatter {
        FormatterCache.shared.formatter("iso|\(calendar.timeZone.identifier)") {
            let formatter = DateFormatter()
            formatter.calendar = Calendar(identifier: .gregorian)
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = calendar.timeZone
            formatter.dateFormat = "yyyy-MM-dd"
            formatter.isLenient = false
            return formatter
        }
    }

    // MARK: The window

    /// The minutes after midnight searched each day, or nil when the
    /// question's times leave no window ("after 4 PM, before 2 PM").
    /// Explicit times replace the part's window; a one-sided time gets a
    /// fixed other end.
    public func window(workdayHours: Range<Int> = 9..<17) -> Range<Int>? {
        let base = part?.window ?? workdayHours.lowerBound * 60..<workdayHours.upperBound * 60
        var start = base.lowerBound, end = base.upperBound
        switch (after, before) {
        case (nil, nil): break
        case (let after?, nil):
            start = after
            end = after < 17 * 60 ? 17 * 60 : after < 21 * 60 ? 21 * 60 : 24 * 60
        case (nil, let before?):
            end = before
            start = before > 9 * 60 ? 9 * 60 : 5 * 60
        case (let after?, let before?):
            start = after
            end = before
        }
        start = max(0, start)
        end = min(24 * 60, end)
        return end > start ? start..<end : nil
    }
}

/// What a free-time search found: the openings, and the sentence over them.
public struct FreeTimeResult: Sendable, Hashable {
    public var query: FreeQuery
    /// The days searched, midnight each.
    public var days: [Date]
    /// Openings at least `query.minutes` long, clipped to the window, in
    /// order.
    public var openings: [DateInterval]
    /// With no opening: the longest gap the window had (15 minutes or more).
    public var longest: DateInterval?
    public var headline: String
}

/// The search over one week.
public enum FreeTimeSearch {
    /// The weekdays from today on (the default days), or the days the query
    /// names inside the week; none when it names only days outside it.
    public static func days(for query: FreeQuery, in week: WeekSnapshot) -> [Date] {
        if query.days == nil, !query.outside.isEmpty { return [] }
        if let named = query.days {
            let formatter = FreeQuery.isoFormatter(week.calendar)
            return named.compactMap { formatter.date(from: $0) }.map(week.calendar.startOfDay)
                .filter { week.contains($0) }
        }
        return week.days.filter { $0 >= week.today && !week.calendar.isDateInWeekend($0) }
    }

    public static func run(_ query: FreeQuery, in week: WeekSnapshot) -> FreeTimeResult {
        let days = days(for: query, in: week)
        guard let window = query.window(workdayHours: week.workdayHours) else {
            return FreeTimeResult(query: query, days: days, openings: [], longest: nil,
                                  headline: Headline(query: query, week: week, days: days).emptyWindow())
        }
        var openings: [DateInterval] = []
        var longest: DateInterval?
        for day in days {
            for gap in WeekFacts.gaps(on: day, in: week, window: window) {
                if gap.duration >= TimeInterval(query.minutes * 60) { openings.append(gap) }
                if gap.duration >= 15 * 60, gap.duration > (longest?.duration ?? 0) { longest = gap }
            }
        }
        let headline = Headline(query: query, week: week, days: days)
        return FreeTimeResult(query: query, days: days, openings: openings, longest: openings.isEmpty ? longest : nil,
                              headline: headline.sentence(openings: openings, longest: longest))
    }

    /// The answer's sentence when a refresh finds every opening passed.
    public static let passedHeadline = "Those openings have passed."
}

/// Swift's sentence over a search: it names the window searched and reads
/// naturally at zero ("No 90-minute opening tomorrow morning. The longest
/// is 1 h, 11 AM to 12 PM.").
struct Headline {
    var query: FreeQuery
    var week: WeekSnapshot
    var days: [Date]

    var text: WeekText { week.text }

    func sentence(openings: [DateInterval], longest: DateInterval?) -> String {
        var sentences: [String] = []
        if days.isEmpty {
            if query.days == nil, query.outside.isEmpty {
                sentences.append("No weekdays are left this week. Name Saturday or Sunday to look there.")
            }
        } else if days.allSatisfy({ $0 < week.today }) {
            let names = days.map { text.weekday($0) }
            sentences.append(names.count == 1 ? "\(names[0]) is over." : "\(WeekText.list(names)) are over.")
        } else if openings.isEmpty {
            var line = "No \(lengthAdjective) \(query.minMinutes == nil ? "open time" : "opening") \(when)."
            if let longest {
                let day = singleDay == nil ? "\(text.shortWeekday(longest.start)) " : ""
                line += " The longest is \(text.duration(minutes: WeekFacts.minutes(longest))), "
                    + "\(day)\(text.range(longest.start, longest.end))."
            }
            sentences.append(line.replacingOccurrences(of: "  ", with: " "))
        } else {
            sentences.append(positive(openings))
        }
        if !query.outside.isEmpty { sentences.append(outsideNote) }
        return sentences.joined(separator: " ")
    }

    func emptyWindow() -> String {
        var line = "There is no time "
        if let after = query.after, let before = query.before {
            line += "after \(clock(after)) and before \(clock(before))."
        } else {
            line += "in that window."
        }
        return query.outside.isEmpty ? line : line + " " + outsideNote
    }

    /// "I'm looking at Sep 14 to 20. Go to that week to check Oct 12."
    var outsideNote: String {
        let formatter = FreeQuery.isoFormatter(week.calendar)
        let named = query.outside.compactMap { formatter.date(from: $0) }.map { text.monthDay($0) }
        let which = named.count == 1 ? "that week" : "those weeks"
        return "I'm looking at \(text.weekRange(week.interval)). Go to \(which) to check \(WeekText.list(named))."
    }

    /// The one day searched, when there is one.
    var singleDay: Date? { days.count == 1 ? days[0] : nil }

    /// "90-minute", "2-hour", or nothing for the default length.
    var lengthAdjective: String {
        guard let minutes = query.minMinutes else { return "" }
        return minutes % 60 == 0 ? "\(minutes / 60)-hour" : "\(minutes)-minute"
    }

    /// "8 AM", "noon".
    func clock(_ minutes: Int) -> String {
        if minutes == 12 * 60 { return "noon" }
        if minutes == 24 * 60 || minutes == 0 { return "midnight" }
        return text.time(week.clock(minutes, on: week.today))
    }

    /// The window in words, or nil for work hours: "in the morning", "after
    /// 3 PM", "before noon", "between 2 and 4 PM".
    var windowPhrase: String? {
        switch (query.after, query.before) {
        case (let after?, nil): return "after \(clock(after))"
        case (nil, let before?): return "before \(clock(before))"
        case (let after?, let before?):
            let start = week.clock(after, on: week.today), end = week.clock(before, on: week.today)
            return before == 24 * 60 ? "between \(clock(after)) and midnight"
                : "between " + text.range(start, end).replacingOccurrences(of: " to ", with: " and ")
        case (nil, nil):
            guard let part = query.part else { return nil }
            return part == .night ? "at night" : "in the \(part.rawValue)"
        }
    }

    /// The day and window together: "tomorrow morning", "this evening",
    /// "tonight", "on Friday after 3 PM", "in the morning this week".
    var when: String {
        guard let day = singleDay else {
            let named = days.count > 1 && query.days != nil
                ? "on " + WeekText.list(days.map { text.shortWeekday($0) }) : "this week"
            return [windowPhrase, named].compactMap { $0 }.joined(separator: " ")
        }
        let word = text.dayWord(day)
        let dayPhrase = word == "today" || word == "tomorrow" ? word : "on \(word)"
        if query.after == nil, query.before == nil, let part = query.part {
            if word == "today" { return part == .night ? "tonight" : "this \(part.rawValue)" }
            return "\(dayPhrase) \(part == .night ? "night" : part.rawValue)"
        }
        return [dayPhrase, windowPhrase].compactMap { $0 }.joined(separator: " ")
    }

    /// "3 openings of 2 h or more: Tue, Wed, Fri." / "2 openings tomorrow
    /// morning: 9 to 10 AM and 11 AM to 12 PM."
    func positive(_ openings: [DateInterval]) -> String {
        var line = WeekText.count(openings.count, "opening")
        if let minutes = query.minMinutes { line += " of \(text.duration(minutes: minutes)) or more" }
        if singleDay != nil {
            return line + " \(when): " + WeekText.list(openings.map { text.range($0.start, $0.end) }) + "."
        }
        if let phrase = windowPhrase { line += " \(phrase)" }
        var seen: [Date] = []
        for opening in openings {
            let day = week.calendar.startOfDay(for: opening.start)
            if !seen.contains(day) { seen.append(day) }
        }
        return line + ": " + seen.map { text.shortWeekday($0) }.joined(separator: ", ") + "."
    }
}
