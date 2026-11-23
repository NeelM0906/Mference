import Foundation

/// How the schedule is written for the model (internal testing):
/// `defaults write ai.mference.app WeekAskFormat lines|json`; lines when
/// unset.
public enum WeekScheduleFormat: String, Sendable, CaseIterable {
    /// One line per day, with an ID on every event, open block and to-do.
    case lines
    /// The same week as JSON fields, without IDs.
    case json

    public static let defaultsKey = "WeekAskFormat"

    /// The app sends JSON unless `WeekAskFormat` is set to `lines`: on the
    /// 20-question check JSON answered more of the questions correctly.
    public static func current(_ defaults: UserDefaults = .standard) -> WeekScheduleFormat {
        defaults.string(forKey: defaultsKey).flatMap(WeekScheduleFormat.init(rawValue:)) ?? .json
    }
}

/// The schedule a model answers from: the week from today on (earlier days
/// are over and left out), day by day, with the arithmetic already done
/// (open time, booked time, the busiest day left, overlaps, to-dos). The
/// person's own all-day events take a whole day; a subscribed holiday does
/// not. Written both as lines with an ID on every event, open block and
/// to-do the answer can cite, and as JSON. Built from one snapshot, so every
/// question of a thread reads the same schedule.
public struct WeekContext: Sendable, Hashable {
    public let week: WeekSnapshot
    /// The day the person is looking at (Day view), if any.
    public let focus: Date?
    /// The days the schedule lists: today on, or the whole week when all of
    /// it is over.
    public let days: [Date]
    /// The schedule as lines, with IDs.
    public let text: String
    /// The same schedule as JSON, without IDs.
    public let json: String
    /// Everything with an ID, in the order the schedule lists it.
    public let facts: [WeekFact]
    /// Every time the schedule names, as minutes after midnight (mode B
    /// notes any other time an answer names).
    public let clockTimes: Set<Int>

    public init(week: WeekSnapshot, focus: Date?) {
        let builder = Builder(week: week, focus: focus.map(week.calendar.startOfDay))
        self.week = week
        self.focus = focus.map(week.calendar.startOfDay)
        self.days = builder.listed.map(\.date)
        self.text = builder.lines()
        self.json = builder.jsonText()
        self.facts = builder.facts
        self.clockTimes = builder.clockTimes
    }

    /// The schedule in `format`.
    public func schedule(_ format: WeekScheduleFormat) -> String {
        switch format {
        case .lines: text
        case .json: json
        }
    }

    public func fact(_ id: String) -> WeekFact? { facts.first { $0.id == id } }

    /// The listed events, in schedule order.
    public var events: [WeekFact] {
        facts.filter { if case .event = $0.kind { true } else { false } }
    }

    /// The schedule without fact IDs, for a chat that does not cite them.
    public var plainText: String {
        text.replacingOccurrences(of: #"\s?\[[EFT]\d+\]"#, with: "", options: .regularExpression)
    }

    /// "2026-09-15" in `calendar`'s time zone.
    static func isoDate(_ date: Date, in calendar: Calendar) -> String {
        let parts = calendar.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", parts.year ?? 0, parts.month ?? 0, parts.day ?? 0)
    }

    /// Text with nothing that could open or close one of the prompt's
    /// blocks (`<untrusted>`, `<USER WEEK>`, `<USER QUERY>`).
    static func unfenced(_ text: String) -> String {
        text.replacingOccurrences(of: #"(?i)<\s*/?\s*(untrusted|user\s*week|user\s*query)\s*>"#, with: "",
                                  options: .regularExpression)
    }

    /// An event or to-do title on one line, with nothing that could close
    /// one of the prompt's blocks.
    static func clean(_ title: String) -> String {
        let cleaned = unfenced(title.components(separatedBy: .newlines).joined(separator: " "))
            .replacingOccurrences(of: #"\s{2,}"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
        return String(cleaned.prefix(120))
    }

    /// Works the schedule out once and writes it both ways.
    private struct Builder {
        struct Event {
            var event: CalendarEvent
            var id: String
            /// Clipped to the day.
            var start: Date
            var end: Date
            var title: String
        }

        struct Day {
            var date: Date
            var load: WeekDayLoad
            var events: [Event]
            var open: [(interval: DateInterval, id: String)]
        }

        struct Overlap {
            var first: String
            var second: String
            var start: Date
            var end: Date
        }

        struct Todo {
            var title: String
            var due: Date
            var id: String
            var isOverdue: Bool
        }

        let week: WeekSnapshot
        let text: WeekText
        let focus: Date?
        let isoFormatter: DateFormatter
        private(set) var listed: [Day] = []
        /// Earlier days of the week were left out.
        private(set) var droppedPast = false
        private(set) var overlaps: [Overlap] = []
        private(set) var todos: [Todo] = []
        private(set) var undatedTodos = 0
        private(set) var facts: [WeekFact] = []
        private(set) var clockTimes: Set<Int> = []
        private var eventIDs: [String: String] = [:]

        init(week: WeekSnapshot, focus: Date?) {
            self.week = week
            self.text = week.text
            self.focus = focus
            let formatter = DateFormatter()
            formatter.calendar = Calendar(identifier: .gregorian)
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = week.calendar.timeZone
            formatter.dateFormat = "yyyy-MM-dd"
            self.isoFormatter = formatter
            clockTimes.insert(minutes(week.now))
            clockTimes.formUnion([minutes(opens), minutes(closes)])
            let loads = WeekFacts.loads(in: week)
            let upcoming = week.days.indices.filter { week.days[$0] >= week.today }
            droppedPast = !upcoming.isEmpty && upcoming.count < week.days.count
            for index in upcoming.isEmpty ? Array(week.days.indices) : upcoming {
                listed.append(day(week.days[index], load: loads[index]))
            }
            findOverlaps()
            findTodos()
        }

        var opens: Date { week.time(week.workdayHours.lowerBound, on: week.today) }
        var closes: Date { week.time(week.workdayHours.upperBound, on: week.today) }

        func minutes(_ date: Date) -> Int {
            week.calendar.component(.hour, from: date) * 60 + week.calendar.component(.minute, from: date)
        }

        /// "Tuesday Sep 15".
        func dayName(_ date: Date) -> String { "\(text.weekday(date)) \(text.monthDay(date))" }

        /// "Tuesday 2026-09-15".
        func isoDay(_ date: Date) -> String { "\(text.weekday(date)) \(isoFormatter.string(from: date))" }

        /// "today", "tomorrow", "past" or nil.
        func relative(_ date: Date) -> String? {
            let word = text.dayWord(date)
            if word == "today" || word == "tomorrow" { return word }
            return week.calendar.startOfDay(for: date) < week.today ? "past" : nil
        }

        /// "Tuesday Sep 15 (today)".
        func labeled(_ date: Date) -> String {
            relative(date).map { "\(dayName(date)) (\($0))" } ?? dayName(date)
        }

        // MARK: Working it out

        mutating func eventID(_ event: CalendarEvent) -> String {
            if let id = eventIDs[event.id] { return id }
            let id = "E\(eventIDs.count + 1)"
            eventIDs[event.id] = id
            let when = event.isAllDay ? "all day" : text.range(event.start, event.end)
            facts.append(WeekFact(id: id, kind: .event(event),
                                  line: "\(WeekContext.clean(event.title)), \(dayName(event.start)), \(when)"))
            if !event.isAllDay { clockTimes.formUnion([minutes(event.start), minutes(event.end)]) }
            return id
        }

        mutating func day(_ date: Date, load: WeekDayLoad) -> Day {
            let end = week.time(24, on: date)
            let touching = week.events.filter { $0.end > date && $0.start < end }
                .sorted { ($0.isAllDay ? 0 : 1, $0.start) < ($1.isAllDay ? 0 : 1, $1.start) }
            var events: [Event] = []
            for event in touching {
                events.append(Event(event: event, id: eventID(event), start: max(event.start, date),
                                    end: min(event.end, end), title: WeekContext.clean(event.title)))
            }
            var open: [(interval: DateInterval, id: String)] = []
            if !load.isPast {
                for block in WeekFacts.openBlocks(on: date, in: week) {
                    let id = "F\(facts.filter { if case .openTime = $0.kind { true } else { false } }.count + 1)"
                    facts.append(WeekFact(id: id, kind: .openTime(block),
                                          line: "\(dayName(block.start)), \(text.range(block.start, block.end)), "
                                              + text.duration(minutes: WeekFacts.minutes(block))))
                    clockTimes.formUnion([minutes(block.start), minutes(block.end)])
                    open.append((block, id))
                }
            }
            return Day(date: date, load: load, events: events, open: open)
        }

        mutating func findOverlaps() {
            guard let first = listed.first?.date else { return }
            let events = week.timedEvents
            for (index, one) in events.enumerated() {
                for other in events[(index + 1)...] where other.start < one.end && one.start < other.end {
                    let start = max(one.start, other.start), end = min(one.end, other.end)
                    guard end > first else { continue }
                    clockTimes.formUnion([minutes(start), minutes(end)])
                    overlaps.append(Overlap(first: WeekContext.clean(one.title), second: WeekContext.clean(other.title),
                                            start: start, end: end))
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
                let id = "T\(todos.count + 1)"
                let title = WeekContext.clean(todo.title)
                facts.append(WeekFact(id: id, kind: .todo(todo), line: "\(title), due \(dayName(date))"))
                todos.append(Todo(title: title, due: date, id: id, isOverdue: isOverdue))
            }
        }

        /// The days left that are busiest: the person's all-day ones first,
        /// else the most booked time. Nil when nothing is booked.
        func busiest() -> (days: [Day], allDay: Bool)? {
            let ahead = listed.filter { !$0.load.isPast }
            let blocked = ahead.filter(\.load.isBlocked)
            if !blocked.isEmpty { return (blocked, true) }
            let most = ahead.map(\.load.bookedMinutes).max() ?? 0
            guard most > 0 else { return nil }
            return (ahead.filter { $0.load.bookedMinutes == most }, false)
        }

        // MARK: Lines

        func lines() -> String {
            var lines: [String] = []
            let first = week.days.first ?? week.interval.start
            let last = week.days.last ?? first
            lines.append("Now: \(dayName(week.now)), \(text.time(week.now)).")
            lines.append("Week: \(dayName(first)) to \(dayName(last)), \(week.calendar.component(.year, from: last)).")
            if droppedPast, let from = listed.first?.date, let to = listed.last?.date {
                lines.append("Listed: \(dayName(from)) to \(dayName(to)); earlier days are over.")
            }
            if let focus { lines.append("Looking at: \(labeled(focus)).") }
            lines.append("Open time counts \(text.time(opens)) to \(text.time(closes)), from now on.")
            lines += listed.map(dayLine)
            lines.append(busiestLine())
            lines.append("Overlapping events: " + (overlaps.isEmpty ? "none." : overlaps.map {
                "\($0.first) and \($0.second), \(dayName($0.start)), \(text.range($0.start, $0.end))"
            }.joined(separator: "; ") + "."))
            let due = todos.filter { !$0.isOverdue }.map { "\($0.title), \(dayName($0.due)) [\($0.id)]" }
            let overdue = todos.filter(\.isOverdue).map { "\($0.title), due \(dayName($0.due)) [\($0.id)]" }
            lines.append("To-dos due this week: " + (due.isEmpty ? "none." : due.joined(separator: "; ") + "."))
            lines.append("Overdue to-dos: " + (overdue.isEmpty ? "none." : overdue.joined(separator: "; ") + "."))
            lines.append("To-dos with no date: \(undatedTodos).")
            return lines.joined(separator: "\n")
        }

        func dayLine(_ day: Day) -> String {
            var line = "\(labeled(day.date)): "
            if day.events.isEmpty {
                line += "nothing booked."
            } else {
                line += day.events.map { event in
                    "\(event.title), \(event.event.isAllDay ? "all day" : text.range(event.start, event.end)) [\(event.id)]"
                }.joined(separator: "; ") + "."
                if day.load.bookedMinutes > 0 { line += " \(text.duration(minutes: day.load.bookedMinutes)) booked." }
            }
            if !day.load.isPast {
                if day.load.isBlocked { line += " Busy all day." }
                if !day.open.isEmpty {
                    line += " Open: " + day.open.map { "\(text.range($0.interval.start, $0.interval.end)) [\($0.id)]" }
                        .joined(separator: ", ") + "."
                }
            }
            return line
        }

        func busiestLine() -> String {
            guard let busiest = busiest() else { return "Busiest day left: nothing booked." }
            let names = busiest.days.map { dayName($0.date) }
            let how = busiest.allDay ? "busy all day"
                : "\(text.duration(minutes: busiest.days[0].load.bookedMinutes)) booked"
            return names.count == 1 ? "Busiest day left: \(names[0]), \(how)."
                : "Busiest days left: \(WeekText.list(names)), \(how)\(busiest.allDay ? "" : " each")."
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

        func dayObject(_ day: Day) -> String {
            var fields: [(String, String)] = [("date", quoted(isoFormatter.string(from: day.date))),
                                              ("weekday", quoted(text.weekday(day.date)))]
            if let relative = relative(day.date) { fields.append(("relative", quoted(relative))) }
            fields.append(("events", array(day.events.map { event in
                event.event.isAllDay
                    ? object([("title", quoted(event.title)), ("all_day", "true")])
                    : object([("title", quoted(event.title)), ("from", quoted(text.time(event.start))),
                              ("to", quoted(text.time(event.end)))])
            })))
            fields.append(("free", array(day.open.map { open in
                object([("from", quoted(text.time(open.interval.start))), ("to", quoted(text.time(open.interval.end))),
                        ("minutes", String(WeekFacts.minutes(open.interval)))])
            })))
            fields.append(("booked_minutes", String(day.load.bookedMinutes)))
            if day.load.isBlocked, !day.load.isPast { fields.append(("busy_all_day", "true")) }
            return object(fields)
        }

        func busiestValue() -> String {
            guard let busiest = busiest() else { return "null" }
            let values = busiest.days.map { day in
                object([("date", quoted(isoFormatter.string(from: day.date))), ("weekday", quoted(text.weekday(day.date))),
                        busiest.allDay ? ("busy_all_day", "true") : ("booked_minutes", String(day.load.bookedMinutes))])
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
            if let from = listed.first?.date, let to = listed.last?.date {
                top.append(("days_listed", quoted("\(isoDay(from)) to \(isoDay(to))"
                    + (droppedPast ? " (earlier days are over)" : ""))))
            }
            if let focus { top.append(("looking_at", quoted(isoDay(focus)))) }
            top.append(("work_hours", quoted("\(text.time(opens)) to \(text.time(closes))")))
            top.append(("days", "[\n" + listed.map { "    " + dayObject($0) }.joined(separator: ",\n") + "\n  ]"))
            top.append(("busiest_day_left", busiestValue()))
            top.append(("overlaps", array(overlaps.map { overlap in
                object([("events", array([quoted(overlap.first), quoted(overlap.second)])),
                        ("date", quoted(isoFormatter.string(from: overlap.start))),
                        ("from", quoted(text.time(overlap.start))), ("to", quoted(text.time(overlap.end)))])
            })))
            top.append(("todos", array(todos.map { todo in
                object([("title", quoted(todo.title)), ("due", quoted(isoFormatter.string(from: todo.due))),
                        ("status", quoted(todo.isOverdue ? "overdue" : "due"))])
            })))
            top.append(("todos_without_date", String(undatedTodos)))
            return "{\n" + top.map { "  \(quoted($0.0)): \($0.1)" }.joined(separator: ",\n") + "\n}"
        }
    }
}
