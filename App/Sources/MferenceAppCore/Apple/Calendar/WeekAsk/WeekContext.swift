import Foundation

/// A clock range ("9:30 to 10 AM") as minutes after midnight, for mode B's
/// check of the ranges an answer names. An end at midnight is 0.
public struct ClockRange: Sendable, Hashable {
    public var start: Int
    public var end: Int

    public init(start: Int, end: Int) {
        self.start = start % (24 * 60)
        self.end = end % (24 * 60)
    }
}

/// The schedule a model answers from, as one JSON object (spec "The
/// schedule sent to the model"): the week from today on (`days`, with open
/// time and booked time worked out), the earlier days of the week
/// (`past_days`, events and booked time only), the busiest day left,
/// overlaps (with both titles, so an answer names them) and to-dos. Events
/// and to-dos carry IDs the answer can cite;
/// open time carries none, so highlights and Block only ever come from
/// Swift. Lengths come pre-formatted next to their minutes. The person's own
/// all-day events take a whole day; a subscribed holiday, or an event shown
/// as Free, does not. Built from one snapshot, so every question of a thread
/// reads the same schedule.
public struct WeekContext: Sendable, Hashable {
    /// One day of the schedule.
    public struct Day: Sendable, Hashable {
        /// One event on the day, clipped to it.
        public struct Entry: Sendable, Hashable {
            public var fact: WeekFact
            public var start: Date
            public var end: Date

            public var event: CalendarEvent? {
                if case .event(let event) = fact.kind { event } else { nil }
            }
        }

        public var date: Date
        public var load: WeekDayLoad
        public var entries: [Entry]
        /// Open blocks inside work hours, from now on, 30 minutes or more
        /// (none on a day that is over or taken).
        public var open: [DateInterval]

        public var events: [WeekFact] { entries.map(\.fact) }
        public var openMinutes: Int { open.reduce(0) { $0 + WeekFacts.minutes($1) } }
    }

    /// Two events at the same time.
    public struct Overlap: Sendable, Hashable {
        public var first: WeekFact
        public var second: WeekFact
        /// Clipped to the first listed day.
        public var interval: DateInterval
    }

    /// The busiest days left: the person's all-day ones first, else the most
    /// booked time.
    public struct Busiest: Sendable, Hashable {
        public var days: [Day]
        public var allDay: Bool
    }

    public let week: WeekSnapshot
    /// The day the person is looking at (Day view), if any.
    public let focus: Date?
    /// Today on (empty once the whole week is over).
    public let days: [Day]
    /// The earlier days of the week.
    public let pastDays: [Day]
    public let overlaps: [Overlap]
    public let busiest: Busiest?
    /// To-dos due from the first listed day to the week's end, then overdue
    /// ones.
    public let todos: [WeekFact]
    public let undatedTodos: Int
    /// The schedule the model reads.
    public let json: String
    /// Everything with an ID, in the order the schedule lists it: events
    /// (E1, E2...), then to-dos (T1, T2...).
    public let facts: [WeekFact]
    /// Every time the schedule names, as minutes after midnight (mode B
    /// notes any other time an answer names).
    public let clockTimes: Set<Int>
    /// Every range the schedule names: events, open blocks, overlaps, work
    /// hours.
    public let clockRanges: Set<ClockRange>

    public init(week: WeekSnapshot, focus: Date?) {
        let builder = Builder(week: week, focus: focus.map(week.calendar.startOfDay))
        self.week = week
        self.focus = focus.map(week.calendar.startOfDay)
        self.days = builder.listed
        self.pastDays = builder.past
        self.overlaps = builder.overlaps
        self.busiest = builder.busiest()
        self.todos = builder.todos.map(\.fact)
        self.undatedTodos = builder.undatedTodos
        self.json = builder.jsonText()
        self.facts = builder.facts
        self.clockTimes = builder.clockTimes
        self.clockRanges = builder.clockRanges
    }

    public func fact(_ id: String) -> WeekFact? { facts.first { $0.id == id } }

    /// This schedule's ID for the same event or to-do as `other` (a thread
    /// that moved to a fresh schedule cites by the new IDs).
    public func fact(matching other: WeekFact) -> WeekFact? {
        switch other.kind {
        case .event(let event): facts.first { if case .event(let mine) = $0.kind { mine.id == event.id } else { false } }
        case .todo(let todo): facts.first { if case .todo(let mine) = $0.kind { mine.id == todo.id } else { false } }
        case .openTime: nil
        }
    }

    /// The listed or past day holding `date`.
    public func day(_ date: Date) -> Day? {
        (days + pastDays).first { week.calendar.isDate($0.date, inSameDayAs: date) }
    }

    /// Whether `other` says the same thing apart from the minutes on the
    /// clock: the same day, the same events and to-dos, the same listed days
    /// and open time. A thread keeps its schedule only while this holds, so
    /// a later "today" or an hour that has passed gets a new one.
    public func readsTheSame(as other: WeekContext) -> Bool {
        week.calendar.isDate(week.now, inSameDayAs: other.week.now) && focus == other.focus
            && week.events == other.week.events && week.todos == other.week.todos
            && days.map(\.date) == other.days.map(\.date) && days.map(\.open) == other.days.map(\.open)
            && facts == other.facts
    }

    /// The listed events, in schedule order.
    public var events: [WeekFact] {
        facts.filter { if case .event = $0.kind { true } else { false } }
    }

    /// "2026-09-15" in `calendar`'s time zone.
    static func isoDate(_ date: Date, in calendar: Calendar) -> String {
        let parts = calendar.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", parts.year ?? 0, parts.month ?? 0, parts.day ?? 0)
    }

    private static let fence = WeekText.regex(#"<\s*/?\s*(untrusted|user\s*week|user\s*query)\s*>"#,
                                              options: [.caseInsensitive])
    private static let spaces = WeekText.regex(#"\s{2,}"#)

    /// Text with nothing that could open or close one of the prompt's
    /// blocks (`<USER WEEK>`, `<USER QUERY>`), removed until nothing
    /// changes: "<USER <USER QUERY>QUERY>" does not survive a pass.
    static func unfenced(_ text: String) -> String {
        var current = text
        while true {
            let next = fence.stringByReplacingMatches(in: current, range: NSRange(current.startIndex..., in: current),
                                                      withTemplate: "")
            if next == current { return current }
            current = next
        }
    }

    /// An event or to-do title (or a question) on one line, with nothing
    /// that could close one of the prompt's blocks.
    static func clean(_ title: String, limit: Int = 120) -> String {
        let flat = unfenced(title.components(separatedBy: .newlines).joined(separator: " "))
        let cleaned = spaces.stringByReplacingMatches(in: flat, range: NSRange(flat.startIndex..., in: flat),
                                                      withTemplate: " ")
            .trimmingCharacters(in: .whitespaces)
        return String(cleaned.prefix(limit))
    }

    /// Works the schedule out once and writes it.
    private struct Builder {
        struct Todo {
            var fact: WeekFact
            var title: String
            var due: Date
            var isOverdue: Bool
        }

        let week: WeekSnapshot
        let text: WeekText
        let focus: Date?
        let isoFormatter: DateFormatter
        private(set) var listed: [Day] = []
        private(set) var past: [Day] = []
        private(set) var overlaps: [Overlap] = []
        private(set) var todos: [Todo] = []
        private(set) var undatedTodos = 0
        private(set) var facts: [WeekFact] = []
        private(set) var clockTimes: Set<Int> = []
        private(set) var clockRanges: Set<ClockRange> = []
        private var eventFacts: [String: WeekFact] = [:]

        init(week: WeekSnapshot, focus: Date?) {
            self.week = week
            self.text = week.text
            self.focus = focus
            self.isoFormatter = FreeQuery.isoFormatter(week.calendar)
            let hours = week.workdayHours
            clockTimes.formUnion([minutes(week.now), 0, hours.lowerBound * 60, hours.upperBound * 60 % 1440])
            clockRanges.insert(ClockRange(start: hours.lowerBound * 60, end: hours.upperBound * 60))
            let loads = WeekFacts.loads(in: week)
            let upcoming = week.days.indices.filter { week.days[$0] >= week.today }
            for index in upcoming { listed.append(day(week.days[index], load: loads[index])) }
            for index in week.days.indices where week.days[index] < week.today {
                past.append(day(week.days[index], load: loads[index]))
            }
            findOverlaps()
            findTodos()
        }

        var opens: Date { week.time(week.workdayHours.lowerBound, on: week.today) }
        var closes: Date { week.time(week.workdayHours.upperBound, on: week.today) }

        func minutes(_ date: Date) -> Int {
            week.calendar.component(.hour, from: date) * 60 + week.calendar.component(.minute, from: date)
        }

        mutating func noteRange(_ start: Date, _ end: Date) {
            clockTimes.formUnion([minutes(start), minutes(end)])
            clockRanges.insert(ClockRange(start: minutes(start), end: minutes(end)))
        }

        /// "Tuesday Sep 15".
        func dayName(_ date: Date) -> String { "\(text.weekday(date)) \(text.monthDay(date))" }

        /// "Tuesday 2026-09-15".
        func isoDay(_ date: Date) -> String { "\(text.weekday(date)) \(isoFormatter.string(from: date))" }

        /// "today", "tomorrow", "yesterday" or nil.
        func relative(_ date: Date) -> String? {
            let word = text.dayWord(date)
            if word == "today" || word == "tomorrow" { return word }
            if let yesterday = week.calendar.date(byAdding: .day, value: -1, to: week.today),
               week.calendar.isDate(date, inSameDayAs: yesterday) { return "yesterday" }
            return nil
        }

        // MARK: Working it out

        mutating func eventFact(_ event: CalendarEvent) -> WeekFact {
            if let fact = eventFacts[event.id] { return fact }
            let id = "E\(eventFacts.count + 1)"
            let when = event.isAllDay ? "all day" : text.range(event.start, event.end)
            let fact = WeekFact(id: id, kind: .event(event),
                                line: "\(WeekContext.clean(event.title)), \(dayName(event.start)), \(when)")
            eventFacts[event.id] = fact
            facts.append(fact)
            if !event.isAllDay { noteRange(event.start, event.end) }
            return fact
        }

        mutating func day(_ date: Date, load: WeekDayLoad) -> Day {
            let end = week.time(24, on: date)
            let touching = week.events.filter { $0.end > date && $0.start < end }
                .sorted { ($0.isAllDay ? 0 : 1, $0.start) < ($1.isAllDay ? 0 : 1, $1.start) }
            var entries: [Day.Entry] = []
            for event in touching {
                let entry = Day.Entry(fact: eventFact(event), start: max(event.start, date), end: min(event.end, end))
                // An event clipped at midnight names 12 AM.
                if !event.isAllDay { noteRange(entry.start, entry.end) }
                entries.append(entry)
            }
            let open = load.isPast ? [] : WeekFacts.openBlocks(on: date, in: week)
            for block in open { noteRange(block.start, block.end) }
            return Day(date: date, load: load, entries: entries, open: open)
        }

        mutating func findOverlaps() {
            guard let first = listed.first?.date else { return }
            let events = week.timedEvents
            for (index, one) in events.enumerated() {
                for other in events[(index + 1)...] where other.start < one.end && one.start < other.end {
                    let start = max(one.start, other.start, first), end = min(one.end, other.end)
                    guard end > start, let a = eventFacts[one.id], let b = eventFacts[other.id] else { continue }
                    noteRange(start, end)
                    overlaps.append(Overlap(first: a, second: b, interval: DateInterval(start: start, end: end)))
                }
            }
        }

        mutating func findTodos() {
            let first = listed.first?.date ?? week.interval.start
            let cutoff = min(first, week.today)
            let open = week.todos.filter { !$0.isDone }
            let due = open.filter { $0.due.map { $0 >= first && $0 < week.interval.end } ?? false }
                .sorted { ($0.due!, $0.title) < ($1.due!, $1.title) }
            let overdue = open.filter { $0.due.map { $0 < cutoff } ?? false }
                .sorted { ($0.due!, $0.title) < ($1.due!, $1.title) }
            undatedTodos = open.filter { $0.due == nil }.count
            for (todo, isOverdue) in due.map({ ($0, false) }) + overdue.map({ ($0, true) }) {
                guard let date = todo.due else { continue }
                let title = WeekContext.clean(todo.title)
                let fact = WeekFact(id: "T\(todos.count + 1)", kind: .todo(todo), line: "\(title), due \(dayName(date))")
                facts.append(fact)
                todos.append(Todo(fact: fact, title: title, due: date, isOverdue: isOverdue))
            }
        }

        func busiest() -> Busiest? {
            let blocked = listed.filter(\.load.isBlocked)
            if !blocked.isEmpty { return Busiest(days: blocked, allDay: true) }
            let most = listed.map(\.load.bookedMinutes).max() ?? 0
            guard most > 0 else { return nil }
            return Busiest(days: listed.filter { $0.load.bookedMinutes == most }, allDay: false)
        }

        // MARK: JSON

        /// A JSON string literal.
        func quoted(_ value: String) -> String {
            guard let data = try? JSONSerialization.data(withJSONObject: value,
                                                         options: [.fragmentsAllowed, .withoutEscapingSlashes]),
                  let literal = String(data: data, encoding: .utf8) else { return "\"\"" }
            return literal
        }

        /// `{"a": 1, "b": "x"}` from fields already written as JSON values.
        func object(_ fields: [(String, String)]) -> String {
            "{" + fields.map { "\(quoted($0.0)): \($0.1)" }.joined(separator: ", ") + "}"
        }

        func array(_ values: [String]) -> String { "[" + values.joined(separator: ", ") + "]" }

        func eventObject(_ entry: Day.Entry) -> String {
            guard let event = entry.event else { return "{}" }
            var fields: [(String, String)] = [("id", quoted(entry.fact.id)),
                                              ("title", quoted(WeekContext.clean(event.title)))]
            if event.isAllDay {
                fields.append(("all_day", "true"))
            } else {
                fields += [("from", quoted(text.time(entry.start))), ("to", quoted(text.time(entry.end))),
                           ("length", quoted(text.duration(minutes: WeekFacts.minutes(DateInterval(
                               start: entry.start, end: entry.end)))))]
            }
            if event.isFree { fields.append(("shows_as", quoted("free"))) }
            return object(fields)
        }

        func dayObject(_ day: Day, past: Bool) -> String {
            var fields: [(String, String)] = [("date", quoted(isoFormatter.string(from: day.date))),
                                              ("weekday", quoted(text.weekday(day.date)))]
            if let relative = relative(day.date) { fields.append(("relative", quoted(relative))) }
            fields.append(("events", array(day.entries.map(eventObject))))
            if !past {
                fields.append(("free", array(day.open.map { open in
                    let minutes = WeekFacts.minutes(open)
                    return object([("from", quoted(text.time(open.start))), ("to", quoted(text.time(open.end))),
                                   ("minutes", String(minutes)), ("length", quoted(text.duration(minutes: minutes)))])
                })))
            }
            fields.append(("booked_minutes", String(day.load.bookedMinutes)))
            fields.append(("booked", quoted(text.duration(minutes: day.load.bookedMinutes))))
            if !past {
                fields.append(("open_minutes", String(day.openMinutes)))
                fields.append(("open", quoted(text.duration(minutes: day.openMinutes))))
                if day.load.isBlocked { fields.append(("busy_all_day", "true")) }
            }
            return object(fields)
        }

        func busiestValue() -> String {
            guard let busiest = busiest() else { return "null" }
            let values = busiest.days.map { day in
                object([("date", quoted(isoFormatter.string(from: day.date))), ("weekday", quoted(text.weekday(day.date))),
                        busiest.allDay ? ("busy_all_day", "true")
                            : ("booked", quoted(text.duration(minutes: day.load.bookedMinutes)))])
            }
            return values.count == 1 ? values[0] : array(values)
        }

        func jsonText() -> String {
            let first = week.days.first ?? week.interval.start
            let last = week.days.last ?? first
            var top: [(String, String)] = [
                ("now", quoted("\(isoDay(week.now)), \(text.time(week.now))")),
                ("week", quoted("\(isoDay(first)) to \(isoDay(last))")),
            ]
            if let focus { top.append(("looking_at", quoted(isoDay(focus)))) }
            top.append(("work_hours", quoted("\(text.time(opens)) to \(text.time(closes))")))
            func list(_ days: [Day], past: Bool) -> String {
                days.isEmpty ? "[]" : "[\n" + days.map { "    " + dayObject($0, past: past) }.joined(separator: ",\n") + "\n  ]"
            }
            top.append(("days", list(listed, past: false)))
            top.append(("past_days", list(past, past: true)))
            top.append(("busiest_day_left", busiestValue()))
            top.append(("overlaps", array(overlaps.map { overlap in
                object([("ids", array([quoted(overlap.first.id), quoted(overlap.second.id)])),
                        ("events", array([overlap.first, overlap.second].compactMap { fact in
                            if case .event(let event) = fact.kind { quoted(WeekContext.clean(event.title)) } else { nil }
                        })),
                        ("date", quoted(isoFormatter.string(from: overlap.interval.start))),
                        ("from", quoted(text.time(overlap.interval.start))),
                        ("to", quoted(text.time(overlap.interval.end)))])
            })))
            top.append(("todos", array(todos.map { todo in
                object([("id", quoted(todo.fact.id)), ("title", quoted(todo.title)),
                        ("due", quoted(isoFormatter.string(from: todo.due))),
                        ("status", quoted(todo.isOverdue ? "overdue" : "due"))])
            })))
            top.append(("todos_without_date", String(undatedTodos)))
            return "{\n" + top.map { "  \(quoted($0.0)): \($0.1)" }.joined(separator: ",\n") + "\n}"
        }
    }
}
