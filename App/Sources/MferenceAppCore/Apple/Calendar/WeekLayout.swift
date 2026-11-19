import Foundation

/// Where one block sits in the week grid: pure geometry for the view.
public struct WeekBlock: Sendable, Hashable, Identifiable {
    public enum Kind: Sendable, Hashable {
        case event(CalendarEvent)
        /// Drawn dashed until the person adds it.
        case proposal(ProposedEvent)
        /// An event a pending proposal would move: drawn struck through.
        case moving(CalendarEvent)
    }

    public var kind: Kind
    /// 0 = first day shown.
    public var day: Int
    /// Hours from the top of the grid (fractional).
    public var top: Double
    public var height: Double
    /// Side-by-side lanes for overlapping blocks in the same day.
    public var lane: Int
    public var lanes: Int

    /// Unique per day too: an event crossing midnight has a block each day.
    public var id: String {
        switch kind {
        case .event(let event): "event-\(event.id)-d\(day)"
        case .proposal(let proposal): "proposal-\(proposal.id)-d\(day)"
        case .moving(let event): "moving-\(event.id)-d\(day)"
        }
    }

    public var title: String {
        switch kind {
        case .event(let event), .moving(let event): event.title
        case .proposal(let proposal): proposal.title
        }
    }
}

/// Lays a work week out on an hour grid (mockup 04: Mon to Fri, 9 AM onward).
public struct WeekLayout: Sendable {
    public var calendar: Calendar
    public var days: Int
    public var firstHour: Int
    public var lastHour: Int
    /// A proposal is drawn at least this tall (hours), so its title and a
    /// two-line "11:00 · proposed from Mail" fit; lanes account for it.
    public var minimumProposalHours: Double = 0.9

    public init(calendar: Calendar = .current, days: Int = 5, firstHour: Int = 9, lastHour: Int = 18) {
        self.calendar = calendar
        self.days = days
        self.firstHour = firstHour
        self.lastHour = lastHour
    }

    /// Monday of the week containing `date`.
    public func weekStart(containing date: Date) -> Date {
        var monday = calendar
        monday.firstWeekday = 2
        return monday.dateInterval(of: .weekOfYear, for: date)?.start ?? calendar.startOfDay(for: date)
    }

    public func dayDates(from weekStart: Date) -> [Date] {
        (0..<days).compactMap { calendar.date(byAdding: .day, value: $0, to: weekStart) }
    }

    /// The window the week view fetches.
    public func interval(from weekStart: Date) -> DateInterval {
        let end = calendar.date(byAdding: .day, value: days, to: weekStart) ?? weekStart
        return DateInterval(start: weekStart, end: end)
    }

    /// Timed blocks inside the grid; all-day events and ones outside the
    /// hours shown are left out (the view lists them above the grid).
    public func blocks(weekStart: Date, events: [CalendarEvent], proposals: [ProposedEvent]) -> [WeekBlock] {
        let moving = Set(proposals.compactMap(\.replaces))
        var blocks: [WeekBlock] = []
        for event in events where !event.isAllDay {
            let kind: WeekBlock.Kind = moving.contains(event.reference) ? .moving(event) : .event(event)
            blocks += segments(event.start, event.end, weekStart: weekStart).compactMap { place(kind, $0) }
        }
        for proposal in proposals {
            blocks += segments(proposal.start, proposal.end, weekStart: weekStart).compactMap { place(.proposal(proposal), $0) }
        }
        return assignLanes(blocks)
    }

    public func allDayEvents(weekStart: Date, events: [CalendarEvent]) -> [Int: [CalendarEvent]] {
        var result: [Int: [CalendarEvent]] = [:]
        for event in events where event.isAllDay {
            for (day, date) in dayDates(from: weekStart).enumerated() {
                let dayEnd = calendar.date(byAdding: .day, value: 1, to: date) ?? date
                if event.start < dayEnd && event.end > date { result[day, default: []].append(event) }
            }
        }
        return result
    }

    /// Events the hour grid cannot show, by day: all-day ones and timed
    /// ones entirely before or after the hours shown. The view lists them
    /// above the grid so the week never looks freer than it is.
    public func outsideGrid(weekStart: Date, events: [CalendarEvent]) -> [Int: [CalendarEvent]] {
        var result = allDayEvents(weekStart: weekStart, events: events)
        for event in events where !event.isAllDay {
            // Each day the event touches whose part falls outside the hours.
            for segment in segments(event.start, event.end, weekStart: weekStart)
            where place(.event(event), segment) == nil {
                result[segment.day, default: []].append(event)
            }
        }
        return result.mapValues { $0.sorted { ($0.isAllDay ? 0 : 1, $0.start) < ($1.isAllDay ? 0 : 1, $1.start) } }
    }

    /// One displayed day's part of a span.
    struct Segment {
        var day: Int
        var dayStart: Date
        var start: Date
        var end: Date
    }

    /// The span clipped to each displayed day it touches (an event from
    /// Sunday 10 PM to Monday 2 AM shows on Monday from midnight).
    func segments(_ start: Date, _ end: Date, weekStart: Date) -> [Segment] {
        let first = calendar.startOfDay(for: weekStart)
        var result: [Segment] = []
        for day in 0..<days {
            guard let dayStart = calendar.date(byAdding: .day, value: day, to: first),
                  let dayEnd = calendar.date(byAdding: .day, value: 1, to: dayStart) else { continue }
            let touches = start == end ? (start >= dayStart && start < dayEnd) : (start < dayEnd && end > dayStart)
            if touches {
                result.append(Segment(day: day, dayStart: dayStart, start: max(start, dayStart), end: min(end, dayEnd)))
            }
        }
        return result
    }

    private func place(_ kind: WeekBlock.Kind, _ segment: Segment) -> WeekBlock? {
        let startHours = segment.start.timeIntervalSince(segment.dayStart) / 3_600
        let endHours = min(segment.end.timeIntervalSince(segment.dayStart) / 3_600, 24)
        var top = max(startHours, Double(firstHour)) - Double(firstHour)
        let bottom = min(endHours, Double(lastHour)) - Double(firstHour)
        guard bottom > top || (bottom == top && startHours >= Double(firstHour) && startHours < Double(lastHour))
        else { return nil }
        var height = max(bottom - top, 0.5)
        if case .proposal = kind {
            // A proposal keeps its room even at the grid's lower edge: it
            // rises instead of shrinking, so it never gets a tiny or negative
            // frame, and lanes use the same geometry.
            let span = Double(lastHour - firstHour)
            height = min(max(height, minimumProposalHours), span)
            top = min(top, span - height)
        }
        return WeekBlock(kind: kind, day: segment.day, top: top, height: height, lane: 0, lanes: 1)
    }

    /// Overlapping blocks in one day share the column side by side.
    private func assignLanes(_ blocks: [WeekBlock]) -> [WeekBlock] {
        var result: [WeekBlock] = []
        for day in 0..<days {
            let dayBlocks = blocks.filter { $0.day == day }.sorted { ($0.top, $0.id) < ($1.top, $1.id) }
            var cluster: [WeekBlock] = []
            var clusterEnd = -1.0
            func flush() {
                var laneEnds: [Double] = []
                var placed: [WeekBlock] = []
                for var block in cluster {
                    if let lane = laneEnds.firstIndex(where: { $0 <= block.top }) {
                        block.lane = lane
                        laneEnds[lane] = block.top + block.height
                    } else {
                        block.lane = laneEnds.count
                        laneEnds.append(block.top + block.height)
                    }
                    placed.append(block)
                }
                result += placed.map { var block = $0; block.lanes = laneEnds.count; return block }
                cluster = []
            }
            for block in dayBlocks {
                if block.top >= clusterEnd, !cluster.isEmpty { flush() }
                cluster.append(block)
                clusterEnd = max(clusterEnd, block.top + block.height)
            }
            if !cluster.isEmpty { flush() }
        }
        return result
    }
}

/// The mockup's short times: "9 to 11 AM", "1 to 3 PM", "11:00", "Wed 11:00
/// to 11:45".
public enum CalendarTimeFormat {
    /// "11:00", "9:30": hour and minutes, no AM/PM (the column says when).
    public static func time(_ date: Date, calendar: Calendar = .current) -> String {
        let (hour, minute, _) = parts(date, calendar: calendar)
        return String(format: "%d:%02d", hour, minute)
    }

    /// "9 to 11 AM", "9:30 to 10 AM", "11 AM to 12 PM": minutes only when not
    /// on the hour, the AM/PM once when both ends share it.
    public static func range(_ start: Date, _ end: Date, calendar: Calendar = .current) -> String {
        let (startHour, startMinute, startPM) = parts(start, calendar: calendar)
        let (endHour, endMinute, endPM) = parts(end, calendar: calendar)
        let short = { (hour: Int, minute: Int) in minute == 0 ? "\(hour)" : String(format: "%d:%02d", hour, minute) }
        let suffix = { (pm: Bool) in pm ? "PM" : "AM" }
        if startPM == endPM {
            return "\(short(startHour, startMinute)) to \(short(endHour, endMinute)) \(suffix(endPM))"
        }
        return "\(short(startHour, startMinute)) \(suffix(startPM)) to \(short(endHour, endMinute)) \(suffix(endPM))"
    }

    /// "Wed 11:00 to 11:45" for event cards.
    public static func dayRange(_ start: Date, _ end: Date, calendar: Calendar = .current) -> String {
        "\(dayTime(start, calendar: calendar)) to \(time(end, calendar: calendar))"
    }

    /// "Wed 11:00".
    public static func dayTime(_ date: Date, calendar: Calendar = .current) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "EEE"
        return "\(formatter.string(from: date)) \(time(date, calendar: calendar))"
    }

    static func parts(_ date: Date, calendar: Calendar) -> (hour: Int, minute: Int, pm: Bool) {
        let components = calendar.dateComponents([.hour, .minute], from: date)
        let hour24 = components.hour ?? 0
        let hour12 = hour24 % 12 == 0 ? 12 : hour24 % 12
        return (hour12, components.minute ?? 0, hour24 >= 12)
    }
}
