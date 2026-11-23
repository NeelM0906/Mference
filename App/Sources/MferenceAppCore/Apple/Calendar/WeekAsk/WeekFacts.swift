import Foundation

// Ask about your week: the arithmetic. Open time, booked time and the
// week's days are worked out here in plain Swift, so the model that answers
// only reads them (WeekContext writes them out, WeekAnswerer asks).

/// One week of events and to-dos, frozen for one question.
public struct WeekSnapshot: Sendable, Hashable {
    /// The seven days asked about, by the Mac's first weekday.
    public var interval: DateInterval
    public var now: Date
    public var calendar: Calendar
    /// Events that touch `interval`, all-day ones included.
    public var events: [CalendarEvent]
    public var todos: [TodoItem]
    /// The hours open time is looked for in (9 to 5).
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

    /// Whether one of the person's own all-day events takes `day` (a
    /// subscribed holiday does not).
    public func isBlocked(_ day: Date) -> Bool {
        let start = calendar.startOfDay(for: day), end = time(24, on: start)
        return events.contains { $0.isAllDay && $0.isUserCalendar && $0.start < end && $0.end > start }
    }

    /// Whether `date` falls inside the week.
    public func contains(_ date: Date) -> Bool { interval.contains(date) && date < interval.end }

    /// `hour` o'clock on `day`, right across daylight-saving changes.
    func time(_ hour: Int, on day: Date) -> Date {
        if hour >= 24 { return calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: day)) ?? day }
        return calendar.date(bySettingHour: hour, minute: 0, second: 0, of: day) ?? day
    }
}

/// How busy one day is.
public struct WeekDayLoad: Sendable, Hashable {
    public var day: Date
    public var bookedMinutes: Int
    public var eventCount: Int
    /// Before today.
    public var isPast: Bool
    /// One of the person's own all-day events takes the whole day.
    public var isBlocked: Bool = false
}

/// One thing the schedule lists with an ID the model can cite: an open
/// block ("F3"), an event ("E2") or a to-do ("T1").
public struct WeekFact: Sendable, Hashable, Identifiable {
    public enum Kind: Sendable, Hashable {
        case openTime(DateInterval)
        case event(CalendarEvent)
        case todo(TodoItem)
    }

    public var id: String
    public var kind: Kind
    /// The fact on its own line, without its ID: "Lunch, Tuesday Sep 15, 12 to 1 PM".
    public var line: String
}

/// The arithmetic the schedule is built from.
public enum WeekFacts {
    /// Shorter gaps are not worth listing as open time.
    public static let shortestOpenMinutes = 30

    /// Open blocks on `day` inside work hours, from the next quarter hour
    /// on, at least `minimumMinutes` long; none on a day the person's own
    /// all-day event takes.
    public static func openBlocks(on day: Date, in week: WeekSnapshot,
                                  minimumMinutes: Int = shortestOpenMinutes) -> [DateInterval] {
        guard !week.isBlocked(day) else { return [] }
        let lower = max(week.time(week.workdayHours.lowerBound, on: day), roundedUp(week.now))
        let upper = week.time(week.workdayHours.upperBound, on: day)
        guard upper > lower else { return [] }
        var blocks: [DateInterval] = []
        var cursor = lower
        for busy in merged(week.timedEvents, within: DateInterval(start: lower, end: upper)) {
            if busy.start > cursor { blocks.append(DateInterval(start: cursor, end: busy.start)) }
            cursor = max(cursor, busy.end)
        }
        if upper > cursor { blocks.append(DateInterval(start: cursor, end: upper)) }
        return blocks.filter { $0.duration >= TimeInterval(minimumMinutes * 60) }
    }

    /// Now, up to the next quarter hour: "10:30", not "10:17".
    static func roundedUp(_ date: Date) -> Date {
        let whole = Date(timeIntervalSinceReferenceDate: date.timeIntervalSinceReferenceDate.rounded(.down))
        let intoQuarter = Int(whole.timeIntervalSinceReferenceDate) % 900
        return intoQuarter == 0 ? whole : whole.addingTimeInterval(TimeInterval(900 - intoQuarter))
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

    /// Booked time and timed events per day.
    public static func loads(in week: WeekSnapshot) -> [WeekDayLoad] {
        week.days.map { day in
            let end = week.time(24, on: day)
            let window = DateInterval(start: day, end: end)
            let booked = merged(week.timedEvents, within: window).reduce(0) { $0 + minutes($1) }
            let count = week.timedEvents.filter { $0.end > day && $0.start < end }.count
            return WeekDayLoad(day: day, bookedMinutes: booked, eventCount: count, isPast: day < week.today,
                               isBlocked: week.isBlocked(day))
        }
    }
}
