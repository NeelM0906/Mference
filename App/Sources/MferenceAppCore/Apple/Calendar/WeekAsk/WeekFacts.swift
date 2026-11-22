import Foundation

// Ask about your week: the facts. Everything an answer states (days, times,
// lengths, counts) is worked out here in plain Swift; the model only names
// the question and words the result (WeekAnswerer), and anything it writes
// that these facts do not contain is replaced (WeekSentenceChecker).

/// One week of events and to-dos, frozen for one question.
public struct WeekSnapshot: Sendable, Hashable {
    /// The seven days asked about, by the Mac's first weekday.
    public var interval: DateInterval
    public var now: Date
    public var calendar: Calendar
    /// Events that touch `interval`, all-day ones included.
    public var events: [CalendarEvent]
    public var todos: [TodoItem]
    /// The hours free time is looked for in (9 to 5).
    public var workdayHours: Range<Int>

    public init(interval: DateInterval, now: Date, calendar: Calendar, events: [CalendarEvent], todos: [TodoItem],
                workdayHours: Range<Int> = 9..<17) {
        self.interval = interval
        self.now = now
        self.calendar = calendar
        self.events = events
            .filter { $0.end > interval.start && $0.start < interval.end }
            .sorted { ($0.start, $0.title) < ($1.start, $1.title) }
        self.todos = todos
        self.workdayHours = workdayHours
    }

    public var text: WeekText { WeekText(calendar: calendar, now: now) }

    /// The week's days, midnight each.
    public var days: [Date] {
        var days: [Date] = []
        var day = calendar.startOfDay(for: interval.start)
        while day < interval.end, days.count < 7 {
            days.append(day)
            guard let next = calendar.date(byAdding: .day, value: 1, to: day) else { break }
            day = next
        }
        return days
    }

    /// Events with a start and an end time (all-day ones mark a day, they do
    /// not fill hours).
    public var timedEvents: [CalendarEvent] { events.filter { !$0.isAllDay && $0.end > $0.start } }

    public var today: Date { calendar.startOfDay(for: now) }

    /// Whether `date` falls inside the week.
    public func contains(_ date: Date) -> Bool { interval.contains(date) && date < interval.end }

    /// `hour` o'clock on `day`, right across daylight-saving changes.
    func time(_ hour: Int, on day: Date) -> Date {
        if hour >= 24 { return calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: day)) ?? day }
        return calendar.date(bySettingHour: hour, minute: 0, second: 0, of: day) ?? day
    }
}

/// Part of the day a free-time question can name.
public enum DayPart: String, Sendable, Hashable, Codable, CaseIterable {
    case morning, afternoon, evening

    /// The hours looked in: mornings and afternoons inside the workday,
    /// evenings after it.
    func hours(workday: Range<Int>) -> Range<Int> {
        switch self {
        case .morning: workday.lowerBound..<max(workday.lowerBound, min(12, workday.upperBound))
        case .afternoon: max(12, workday.lowerBound)..<max(12, workday.upperBound)
        case .evening: max(17, workday.upperBound)..<22
        }
    }
}

/// A free-time question, made precise.
public struct FreeTimeRequest: Sendable, Hashable {
    /// The shortest block that counts.
    public var minimumMinutes: Int
    /// Only time after this moment (besides after now).
    public var after: Date?
    /// Only time before this moment.
    public var before: Date?
    public var part: DayPart?
    /// One day only.
    public var day: Date?
    /// Saturdays and Sundays are skipped unless asked for.
    public var includesWeekend: Bool

    public init(minimumMinutes: Int = 60, after: Date? = nil, before: Date? = nil, part: DayPart? = nil,
                day: Date? = nil, includesWeekend: Bool = false) {
        self.minimumMinutes = max(15, minimumMinutes)
        self.after = after
        self.before = before
        self.part = part
        self.day = day
        self.includesWeekend = includesWeekend
    }
}

/// The seven kinds of question Ask answers, plus the two it turns away.
public enum WeekQuery: Sendable, Hashable {
    case freeTime(FreeTimeRequest)
    case findEvent(name: String)
    case compareDays
    case daySummary(day: Date)
    case todosDue
    case overlaps
    /// Names dates outside the week: answered without a model.
    case outsideWeek
    /// Not a question about this week's calendar.
    case other
}

/// How busy one day is.
public struct WeekDayLoad: Sendable, Hashable {
    public var day: Date
    public var bookedMinutes: Int
    public var eventCount: Int
    /// Before today.
    public var isPast: Bool
}

/// One fact an answer may state, with the short ID the phraser cites.
public struct WeekFact: Sendable, Hashable, Identifiable {
    public enum Kind: Sendable, Hashable {
        case openTime(DateInterval)
        case event(CalendarEvent)
        case day(WeekDayLoad)
        case todo(TodoItem)
        case overlap(CalendarEvent, CalendarEvent, DateInterval)
    }

    /// "F1", "E2", "D3", "T1", "O1".
    public var id: String
    public var kind: Kind
    /// The fact written out for the phraser: "F1: today (Tuesday Sep 15), 1 to 5 PM, 4 h".
    public var line: String
}

/// The numbers and day words a sentence about these facts may contain.
public struct WeekTerms: Sendable, Hashable {
    public var numbers: Set<Int>
    public var dayWords: Set<String>

    public init(numbers: Set<Int> = [], dayWords: Set<String> = []) {
        self.numbers = numbers
        self.dayWords = dayWords
    }

    public func union(_ other: WeekTerms) -> WeekTerms {
        WeekTerms(numbers: numbers.union(other.numbers), dayWords: dayWords.union(other.dayWords))
    }

    static let numberWords: [String: Int] = [
        "zero": 0, "no": 0, "one": 1, "a": 1, "an": 1, "single": 1, "two": 2, "both": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        "noon": 12, "midday": 12, "midnight": 0,
    ]
    /// Number words the checker holds a sentence to; "a", "an", "no" and
    /// "both" read too loosely to check, so they only count as allowed.
    static let checkedNumberWords: Set<String> = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    ]

    /// The numbers and day words written in `text`. Day words are weekday
    /// names (full or short, in the calendar's locale and in English) and
    /// today, tomorrow, tonight, yesterday.
    public static func found(in text: String, calendar: Calendar, checkedOnly: Bool = false) -> WeekTerms {
        var terms = WeekTerms()
        let lowered = text.lowercased()
        var digits = ""
        for character in lowered + " " {
            if character.isASCII, character.isNumber { digits.append(character); continue }
            if let value = Int(digits) { terms.numbers.insert(value) }
            digits = ""
        }
        let names = dayNames(calendar)
        for word in lowered.split(whereSeparator: { !$0.isLetter }).map(String.init) {
            if names.contains(word) { terms.dayWords.insert(canonicalDay(word, calendar: calendar)) }
            if let value = numberWords[word], !checkedOnly || checkedNumberWords.contains(word) {
                terms.numbers.insert(value)
            }
        }
        return terms
    }

    static func dayNames(_ calendar: Calendar) -> Set<String> {
        var english = Calendar(identifier: .gregorian)
        english.locale = Locale(identifier: "en_US_POSIX")
        let symbols = calendar.weekdaySymbols + calendar.shortWeekdaySymbols
            + english.weekdaySymbols + english.shortWeekdaySymbols
        return Set(symbols.map { $0.lowercased().trimmingCharacters(in: .punctuationCharacters) })
            .union(["today", "tomorrow", "tonight", "yesterday"])
    }

    /// "tue" and "tuesday" are the same day; "tonight" is today.
    static func canonicalDay(_ word: String, calendar: Calendar) -> String {
        if word == "tonight" { return "today" }
        var english = Calendar(identifier: .gregorian)
        english.locale = Locale(identifier: "en_US_POSIX")
        for symbols in [(calendar.weekdaySymbols, calendar.shortWeekdaySymbols),
                        (english.weekdaySymbols, english.shortWeekdaySymbols)] {
            for (full, short) in zip(symbols.0, symbols.1) {
                let shortWord = short.lowercased().trimmingCharacters(in: .punctuationCharacters)
                if word == full.lowercased() || word == shortWord { return full.lowercased() }
            }
        }
        return word
    }
}

/// The answer to one question, before any model touches it.
public struct WeekAnswerFacts: Sendable, Hashable {
    public var query: WeekQuery
    /// The rows shown under the sentence, in order.
    public var facts: [WeekFact]
    /// Extra lines for the phraser ("count: 8", "earliest: F1").
    public var notes: [String]
    /// A correct sentence written from the facts, used whenever the model's
    /// cannot be used.
    public var sentence: String
    /// What a sentence about these facts may say.
    public var terms: WeekTerms
    /// The answer is "none": no open block long enough, no such event,
    /// nothing due, nothing booked. A worded answer must say so too.
    public var foundNothing = false

    /// The open blocks among the facts.
    public var openTimes: [DateInterval] {
        facts.compactMap { if case .openTime(let interval) = $0.kind { interval } else { nil } }
    }

    /// The events among the facts (found events, a day's events, overlaps).
    public var eventIDs: Set<String> {
        Set(facts.flatMap { fact -> [String] in
            switch fact.kind {
            case .event(let event): [event.id]
            case .overlap(let first, let second, _): [first.id, second.id]
            default: []
            }
        })
    }

    /// Whether a model may word this answer (the two turned-away kinds use
    /// their fixed sentence).
    public var wantsWording: Bool {
        switch query {
        case .outsideWeek, .other: false
        default: true
        }
    }
}

/// Works out the facts for a question.
public enum WeekFacts {
    public static func answer(_ query: WeekQuery, in week: WeekSnapshot) -> WeekAnswerFacts {
        var answer: WeekAnswerFacts
        switch query {
        case .freeTime(let request): answer = freeTime(request, in: week)
        case .findEvent(let name): answer = findEvent(name, in: week)
        case .compareDays: answer = compareDays(in: week)
        case .daySummary(let day): answer = daySummary(day, in: week)
        case .todosDue: answer = todosDue(in: week)
        case .overlaps: answer = overlaps(in: week)
        case .outsideWeek:
            answer = WeekAnswerFacts(query: query, facts: [], notes: [],
                                     sentence: "I can only see \(week.text.weekRange(week.interval)) here. "
                                         + "Go to that week and ask there.", terms: WeekTerms())
        case .other:
            answer = WeekAnswerFacts(query: query, facts: [], notes: [],
                                     sentence: "I can't answer that from this week's events and to-dos.",
                                     terms: WeekTerms())
        }
        answer.terms = terms(for: answer, in: week)
        answer.foundNothing = foundNothing(answer)
        return answer
    }

    static func foundNothing(_ answer: WeekAnswerFacts) -> Bool {
        switch answer.query {
        case .freeTime: answer.notes.first?.hasSuffix(": 0") ?? true
        case .findEvent, .daySummary, .overlaps: answer.facts.isEmpty
        case .todosDue: answer.notes.first == "due this week: 0"
        case .compareDays: !answer.notes.contains { $0.hasPrefix("busiest days left: D") }
        case .outsideWeek, .other: false
        }
    }

    // MARK: Free time

    /// Open blocks of at least `minimumMinutes` (15 when nil), within the
    /// request's window, from now on, by day.
    static func openBlocks(_ request: FreeTimeRequest, minimumMinutes: Int? = nil, in week: WeekSnapshot)
        -> [DateInterval] {
        let calendar = week.calendar
        let shortest = TimeInterval((minimumMinutes ?? request.minimumMinutes) * 60)
        let from = roundedUp(week.now, calendar: calendar)
        var blocks: [DateInterval] = []
        for day in week.days {
            if let only = request.day, !calendar.isDate(day, inSameDayAs: only) { continue }
            if request.day == nil, !request.includesWeekend, calendar.isDateInWeekend(day) { continue }
            let lower: Date, upper: Date
            if request.day != nil, let after = request.after, let before = request.before {
                // A window written out ("6 to 8 PM") is the search, even
                // outside the workday.
                lower = max(after, from)
                upper = before
            } else {
                let hours = request.part?.hours(workday: week.workdayHours) ?? week.workdayHours
                guard !hours.isEmpty else { continue }
                lower = [week.time(hours.lowerBound, on: day), from, request.after ?? .distantPast].max()!
                upper = min(week.time(hours.upperBound, on: day), request.before ?? .distantFuture)
            }
            guard upper > lower else { continue }
            var cursor = lower
            for busy in merged(week.timedEvents, within: DateInterval(start: lower, end: upper)) {
                if busy.start > cursor { blocks.append(DateInterval(start: cursor, end: busy.start)) }
                cursor = max(cursor, busy.end)
            }
            if upper > cursor { blocks.append(DateInterval(start: cursor, end: upper)) }
        }
        return blocks.filter { $0.duration >= shortest }
    }

    /// Now, up to the next quarter hour: "10:30", not "10:17".
    static func roundedUp(_ date: Date, calendar: Calendar) -> Date {
        let minute = calendar.component(.minute, from: date)
        let second = calendar.component(.second, from: date)
        let extra = (15 - minute % 15) % 15
        guard extra > 0 || second > 0 else { return date }
        let base = calendar.date(bySetting: .second, value: 0, of: date) ?? date
        let floored = base.addingTimeInterval(-Double(minute % 15) * 60)
        return floored.addingTimeInterval(15 * 60)
    }

    /// Busy times inside `window`, overlapping events joined.
    static func merged(_ events: [CalendarEvent], within window: DateInterval) -> [DateInterval] {
        let clipped = events.compactMap { event -> DateInterval? in
            let start = max(event.start, window.start), end = min(event.end, window.end)
            return end > start ? DateInterval(start: start, end: end) : nil
        }.sorted { $0.start < $1.start }
        var result: [DateInterval] = []
        for interval in clipped {
            if let last = result.last, interval.start <= last.end {
                result[result.count - 1] = DateInterval(start: last.start, end: max(last.end, interval.end))
            } else {
                result.append(interval)
            }
        }
        return result
    }

    static func minutes(_ interval: DateInterval) -> Int { Int((interval.duration / 60).rounded()) }

    static func dayLabel(_ date: Date, _ text: WeekText) -> String {
        let word = text.dayWord(date)
        let full = text.weekday(date)
        return word == full ? "\(full) \(text.monthDay(date))" : "\(word) (\(full) \(text.monthDay(date)))"
    }

    static func freeTime(_ request: FreeTimeRequest, in week: WeekSnapshot) -> WeekAnswerFacts {
        let text = week.text
        let blocks = openBlocks(request, in: week)
        let facts = blocks.enumerated().map { index, block in
            WeekFact(id: "F\(index + 1)", kind: .openTime(block),
                     line: "F\(index + 1): \(dayLabel(block.start, text)), \(text.range(block.start, block.end)), "
                         + text.duration(minutes: minutes(block)))
        }
        let length = text.duration(minutes: request.minimumMinutes)
        let phrase = windowPhrase(request, in: week)
        guard let earliest = facts.first, let first = blocks.first else {
            let any = openBlocks(request, minimumMinutes: 15, in: week)
            guard let longest = any.max(by: { $0.duration < $1.duration }) else {
                return WeekAnswerFacts(query: .freeTime(request), facts: [], notes: ["count: 0"],
                                       sentence: "Nothing is open \(phrase).", terms: WeekTerms())
            }
            let fact = WeekFact(id: "F1", kind: .openTime(longest),
                                line: "F1: \(dayLabel(longest.start, text)), \(text.range(longest.start, longest.end)), "
                                    + text.duration(minutes: minutes(longest)))
            return WeekAnswerFacts(
                query: .freeTime(request), facts: [fact],
                notes: ["count of \(length) or more: 0", "longest open block instead: F1"],
                sentence: "No open block of \(length) \(phrase). The longest is \(text.dayWord(longest.start)), "
                    + "\(text.range(longest.start, longest.end)) (\(text.duration(minutes: minutes(longest)))).",
                terms: WeekTerms())
        }
        let longestIndex = blocks.indices.max { blocks[$0].duration < blocks[$1].duration } ?? 0
        let count = WeekText.count(blocks.count, "open block")
        let sentence = blocks.count == 1
            ? "You have 1 open block of \(length) or more \(phrase): \(text.dayWord(first.start)), "
                + "\(text.range(first.start, first.end))."
            : "You have \(count) of \(length) or more \(phrase). The earliest is \(text.dayWord(first.start)), "
                + "\(text.range(first.start, first.end))."
        return WeekAnswerFacts(query: .freeTime(request), facts: facts,
                               notes: ["count: \(blocks.count)", "earliest: \(earliest.id)",
                                       "longest: \(facts[longestIndex].id)"],
                               sentence: sentence, terms: WeekTerms())
    }

    static func windowPhrase(_ request: FreeTimeRequest, in week: WeekSnapshot) -> String {
        let text = week.text
        if let day = request.day {
            let word = text.dayWord(day)
            let base = word == text.weekday(day) ? "on \(word)" : word
            if let after = request.after, let before = request.before {
                return "\(base) from \(text.range(after, before))"
            }
            return base
        }
        var phrase = week.interval.contains(week.now) ? "for the rest of the week" : "this week"
        if let part = request.part { phrase = "in the \(part.rawValue)s" }
        if let before = request.before, week.contains(before) {
            phrase += " before \(text.dayWord(before)) \(text.time(before))"
        }
        if let after = request.after, week.contains(after), after > week.now {
            phrase += " after \(text.dayWord(after)) \(text.time(after))"
        }
        return phrase
    }

    // MARK: Events

    /// Events whose title holds the name, or every word of it.
    static func matches(_ name: String, in week: WeekSnapshot) -> [CalendarEvent] {
        let wanted = name.lowercased().trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
        guard !wanted.isEmpty else { return [] }
        let words = wanted.split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init)
        return week.events.filter { event in
            let title = event.title.lowercased()
            return title.contains(wanted) || (!words.isEmpty && words.allSatisfy { title.contains($0) })
        }
    }

    static func eventWhen(_ event: CalendarEvent, _ text: WeekText) -> String {
        event.isAllDay ? "\(text.dayWord(event.start)), all day"
            : "\(text.dayWord(event.start)), \(text.range(event.start, event.end))"
    }

    static func eventFact(_ event: CalendarEvent, id: String, _ text: WeekText) -> WeekFact {
        let when = event.isAllDay ? "all day" : text.range(event.start, event.end)
        return WeekFact(id: id, kind: .event(event), line: "\(id): \(event.title), \(dayLabel(event.start, text)), \(when)")
    }

    static func findEvent(_ name: String, in week: WeekSnapshot) -> WeekAnswerFacts {
        let text = week.text
        let found = matches(name, in: week)
        let facts = found.enumerated().map { eventFact($1, id: "E\($0 + 1)", text) }
        let sentence: String
        switch found.count {
        case 0: sentence = "No event called \u{201C}\(name)\u{201D} this week."
        case 1:
            let event = found[0]
            sentence = "\(event.title) \(event.end <= week.now ? "was" : "is") \(eventWhen(event, text))."
        default:
            sentence = "\(found.count) events match \u{201C}\(name)\u{201D}: "
                + WeekText.list(found.map { eventWhen($0, text) }) + "."
        }
        return WeekAnswerFacts(query: .findEvent(name: name), facts: facts,
                               notes: ["matches: \(found.count)"], sentence: sentence, terms: WeekTerms())
    }

    // MARK: Days

    static func loads(in week: WeekSnapshot) -> [WeekDayLoad] {
        week.days.map { day in
            let end = week.time(24, on: day)
            let window = DateInterval(start: day, end: end)
            let booked = merged(week.timedEvents, within: window).reduce(0) { $0 + minutes($1) }
            let count = week.timedEvents.filter { $0.end > day && $0.start < end }.count
            return WeekDayLoad(day: day, bookedMinutes: booked, eventCount: count, isPast: day < week.today)
        }
    }

    static func compareDays(in week: WeekSnapshot) -> WeekAnswerFacts {
        let text = week.text
        let all = loads(in: week)
        // Weekend days only count when something is booked on them.
        let shown = all.filter { !week.calendar.isDateInWeekend($0.day) || $0.eventCount > 0 }
        let facts = shown.enumerated().map { index, load in
            WeekFact(id: "D\(index + 1)", kind: .day(load),
                     line: "D\(index + 1): \(dayLabel(load.day, text))\(load.isPast ? " (past)" : ""), "
                         + "\(text.duration(minutes: load.bookedMinutes)) booked, "
                         + WeekText.count(load.eventCount, "event"))
        }
        let ahead = shown.filter { !$0.isPast }
        let most = ahead.map(\.bookedMinutes).max() ?? 0
        let busiest = ahead.filter { $0.bookedMinutes == most }
        let sentence: String
        if most == 0 {
            sentence = "Nothing is booked for the rest of this week."
        } else if busiest.count == 1 {
            sentence = "\(text.dayWordCapitalized(busiest[0].day)) is your busiest day left, "
                + "with \(text.duration(minutes: most)) booked."
        } else {
            let names = busiest.map { text.dayWord($0.day) }
            sentence = WeekText.list(names).prefix(1).uppercased() + WeekText.list(names).dropFirst()
                + " tie as your busiest days left, with \(text.duration(minutes: most)) booked each."
        }
        let busiestIDs = facts.filter { fact in
            if case .day(let load) = fact.kind { return !load.isPast && load.bookedMinutes == most && most > 0 }
            return false
        }.map(\.id)
        return WeekAnswerFacts(query: .compareDays, facts: facts,
                               notes: ["busiest days left: \(busiestIDs.joined(separator: ", "))"],
                               sentence: sentence, terms: WeekTerms())
    }

    static func daySummary(_ day: Date, in week: WeekSnapshot) -> WeekAnswerFacts {
        let text = week.text
        let start = week.calendar.startOfDay(for: day)
        let end = week.time(24, on: start)
        let events = week.events.filter { $0.end > start && $0.start < end }
            .sorted { ($0.isAllDay ? 0 : 1, $0.start) < ($1.isAllDay ? 0 : 1, $1.start) }
        let facts = events.enumerated().map { eventFact($1, id: "E\($0 + 1)", text) }
        let name = text.dayWordCapitalized(start)
        let sentence: String
        if events.isEmpty {
            sentence = "\(name) has nothing booked."
        } else {
            let items = events.map { event in
                event.isAllDay ? "\(event.title), all day" : "\(event.title) at \(text.time(event.start))"
            }
            sentence = "\(name) has \(WeekText.count(events.count, "event")): \(WeekText.list(items))."
        }
        return WeekAnswerFacts(query: .daySummary(day: start), facts: facts,
                               notes: ["events: \(events.count)"], sentence: sentence, terms: WeekTerms())
    }

    // MARK: To-dos

    static func todosDue(in week: WeekSnapshot) -> WeekAnswerFacts {
        let text = week.text
        let open = week.todos.filter { !$0.isDone }
        let due = open.filter { $0.due.map(week.contains) ?? false }.sorted { ($0.due!, $0.title) < ($1.due!, $1.title) }
        let overdue = open.filter { ($0.due.map { $0 < week.interval.start }) ?? false }
            .sorted { ($0.due!, $0.title) < ($1.due!, $1.title) }
        let undated = open.filter { $0.due == nil }.count
        var facts: [WeekFact] = []
        for todo in due + overdue {
            let id = "T\(facts.count + 1)"
            let state = todo.due! < week.interval.start ? "overdue since \(text.weekday(todo.due!)) \(text.monthDay(todo.due!))"
                : "due \(dayLabel(todo.due!, text))"
            facts.append(WeekFact(id: id, kind: .todo(todo), line: "\(id): \(todo.title), \(state)"))
        }
        var parts: [String] = []
        if due.isEmpty {
            parts.append("Nothing is due this week.")
        } else {
            let verb = due.count == 1 ? "is" : "are"
            parts.append("\(WeekText.count(due.count, "to-do")) \(verb) due this week: "
                + WeekText.list(due.map { "\($0.title) (\(text.dayWord($0.due!)))" }) + ".")
        }
        if !overdue.isEmpty {
            parts.append("\(overdue.count) \(overdue.count == 1 ? "is" : "are") overdue: "
                + WeekText.list(overdue.map(\.title)) + ".")
        }
        if undated > 0 { parts.append("\(undated) more \(undated == 1 ? "has" : "have") no date.") }
        return WeekAnswerFacts(query: .todosDue, facts: facts,
                               notes: ["due this week: \(due.count)", "overdue: \(overdue.count)",
                                       "no date: \(undated)"],
                               sentence: parts.joined(separator: " "), terms: WeekTerms())
    }

    // MARK: Overlaps

    static func overlaps(in week: WeekSnapshot) -> WeekAnswerFacts {
        let text = week.text
        let events = week.timedEvents
        var facts: [WeekFact] = []
        for (index, first) in events.enumerated() {
            for second in events[(index + 1)...] where second.start < first.end && first.start < second.end {
                let shared = DateInterval(start: max(first.start, second.start), end: min(first.end, second.end))
                let id = "O\(facts.count + 1)"
                facts.append(WeekFact(id: id, kind: .overlap(first, second, shared),
                                      line: "\(id): \(first.title) and \(second.title), \(dayLabel(shared.start, text)), "
                                          + text.range(shared.start, shared.end)))
            }
        }
        let sentence: String
        if let fact = facts.first, case .overlap(let first, let second, let shared) = fact.kind {
            let lead = "\(first.title) and \(second.title) overlap \(text.dayWord(shared.start)), "
                + "\(text.range(shared.start, shared.end))."
            sentence = facts.count == 1 ? lead : "\(facts.count) pairs of events overlap. The first: \(lead)"
        } else {
            sentence = "Nothing overlaps this week."
        }
        return WeekAnswerFacts(query: .overlaps, facts: facts, notes: ["overlapping pairs: \(facts.count)"],
                               sentence: sentence, terms: WeekTerms())
    }

    // MARK: Terms

    /// Everything the fixed sentence, the fact lines and the notes say, plus
    /// each time in 24-hour form, so a model writing "13:00" for "1 PM"
    /// still passes.
    static func terms(for answer: WeekAnswerFacts, in week: WeekSnapshot) -> WeekTerms {
        let calendar = week.calendar
        // Fact IDs ("F6") are labels, not numbers a sentence may use.
        let lines = answer.facts.map { String($0.line.dropFirst($0.id.count + 1)) }
        let notes = answer.notes.map {
            $0.replacingOccurrences(of: #"\b[FEDTO]\d+\b"#, with: "", options: .regularExpression)
        }
        let written = ([answer.sentence] + lines + notes).joined(separator: "\n")
        var terms = WeekTerms.found(in: written, calendar: calendar)
        func addTime(_ date: Date) {
            let hour = calendar.component(.hour, from: date)
            terms.numbers.formUnion([hour, hour % 12 == 0 ? 12 : hour % 12, calendar.component(.minute, from: date)])
        }
        func addDay(_ date: Date) {
            terms.numbers.insert(calendar.component(.day, from: date))
            terms.dayWords.insert(week.text.weekday(date).lowercased())
        }
        for fact in answer.facts {
            switch fact.kind {
            case .openTime(let interval):
                addTime(interval.start); addTime(interval.end); addDay(interval.start)
                let total = minutes(interval)
                terms.numbers.formUnion([total, total / 60, total % 60])
            case .event(let event):
                addTime(event.start); addTime(event.end); addDay(event.start)
            case .overlap(let first, let second, let shared):
                for date in [first.start, first.end, second.start, second.end, shared.start, shared.end] { addTime(date) }
                addDay(shared.start)
            case .day(let load):
                addDay(load.day)
                terms.numbers.formUnion([load.bookedMinutes, load.bookedMinutes / 60, load.bookedMinutes % 60])
            case .todo(let todo):
                if let due = todo.due { addDay(due) }
            }
        }
        return terms
    }
}
