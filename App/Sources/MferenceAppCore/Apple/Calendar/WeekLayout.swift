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

    public var id: String {
        switch kind {
        case .event(let event): "event-\(event.id)"
        case .proposal(let proposal): "proposal-\(proposal.id)"
        case .moving(let event): "moving-\(event.id)"
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
        let moving = Set(proposals.compactMap(\.replacesEventID))
        var blocks: [WeekBlock] = []
        for event in events where !event.isAllDay {
            let kind: WeekBlock.Kind = moving.contains(event.id) ? .moving(event) : .event(event)
            if let block = place(kind, start: event.start, end: event.end, weekStart: weekStart) { blocks.append(block) }
        }
        for proposal in proposals {
            if let block = place(.proposal(proposal), start: proposal.start, end: proposal.end, weekStart: weekStart) {
                blocks.append(block)
            }
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

    private func place(_ kind: WeekBlock.Kind, start: Date, end: Date, weekStart: Date) -> WeekBlock? {
        let day = calendar.dateComponents([.day], from: calendar.startOfDay(for: weekStart),
                                          to: calendar.startOfDay(for: start)).day ?? -1
        guard (0..<days).contains(day) else { return nil }
        let dayStart = calendar.startOfDay(for: start)
        let startHours = start.timeIntervalSince(dayStart) / 3_600
        let endHours = min(end.timeIntervalSince(dayStart) / 3_600, 24)
        let top = max(startHours, Double(firstHour)) - Double(firstHour)
        let bottom = min(endHours, Double(lastHour)) - Double(firstHour)
        guard bottom > top || (bottom == top && startHours >= Double(firstHour) && startHours < Double(lastHour))
        else { return nil }
        return WeekBlock(kind: kind, day: day, top: top, height: max(bottom - top, 0.5), lane: 0, lanes: 1)
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
