import Foundation

/// The colors an event can take, matching the Mac calendar's event fills.
public enum EventTint: Sendable, Hashable {
    case blue, orange, green, grey, purple
}

/// A calendar event on the phone.
public struct PhoneEvent: Identifiable, Hashable, Sendable {
    public let id: UUID
    public var title: String
    public var start: Date
    public var end: Date
    public var tint: EventTint

    public init(id: UUID = UUID(), title: String, start: Date, end: Date, tint: EventTint) {
        self.id = id
        self.title = title
        self.start = start
        self.end = end
        self.tint = tint
    }
}

/// A to-do, synced with Reminders in the real app.
public struct PhoneTodo: Identifiable, Hashable, Sendable {
    public let id: UUID
    public var title: String
    /// "Mail · due Tue".
    public var detail: String
    public var isDone: Bool

    public init(id: UUID = UUID(), title: String, detail: String, isDone: Bool = false) {
        self.id = id
        self.title = title
        self.detail = detail
        self.isDone = isDone
    }
}

/// A change proposed from mail ("Move the design review to Wed 11:00?").
public struct CalendarSuggestion: Hashable, Sendable {
    public var question: String
    public var detail: String
    public var event: PhoneEvent
}

/// Day, Month or the to-do list (the phone's Calendar views).
public enum CalendarMode: String, CaseIterable, Identifiable, Sendable {
    case day = "Day", month = "Month", todos = "To-dos"
    public var id: String { rawValue }
}

/// Month grid maths, following the device's first weekday.
public struct MonthGrid: Sendable {
    public let calendar: Calendar
    public let month: Date

    public init(month: Date, calendar: Calendar = .current) {
        self.calendar = calendar
        self.month = calendar.dateInterval(of: .month, for: month)?.start ?? month
    }

    /// Weekday symbols starting at the calendar's first weekday ("M", "T", ...).
    public var weekdaySymbols: [String] {
        let symbols = calendar.veryShortStandaloneWeekdaySymbols
        let first = calendar.firstWeekday - 1
        return Array(symbols[first...] + symbols[..<first])
    }

    /// The cells of the grid: nil for padding before the 1st and after the
    /// last day, so the count is a multiple of 7.
    public var cells: [Date?] {
        guard let days = calendar.range(of: .day, in: .month, for: month) else { return [] }
        let weekday = calendar.component(.weekday, from: month)
        let leading = (weekday - calendar.firstWeekday + 7) % 7
        var cells: [Date?] = Array(repeating: nil, count: leading)
        for day in days {
            cells.append(calendar.date(byAdding: .day, value: day - 1, to: month))
        }
        while cells.count % 7 != 0 { cells.append(nil) }
        return cells
    }

    /// The seven days of the week containing `date`.
    public static func week(containing date: Date, calendar: Calendar = .current) -> [Date] {
        let start = calendar.dateInterval(of: .weekOfYear, for: date)?.start ?? calendar.startOfDay(for: date)
        return (0..<7).compactMap { calendar.date(byAdding: .day, value: $0, to: start) }
    }
}

/// What Scan is reading.
public enum ScanKind: String, CaseIterable, Identifiable, Sendable {
    case receipt = "Receipt", document = "Document", card = "Card", whiteboard = "Whiteboard"
    public var id: String { rawValue }
}

/// One field read from a scan, with how sure the model is.
public struct ScanField: Identifiable, Hashable, Sendable {
    public let id: String
    public var label: String
    public var value: String
    /// False shows the amber "Check" badge and field.
    public var isSure: Bool
}

/// Where the scan is in its life.
public enum ScanPhase: Hashable, Sendable {
    case viewfinder, reading, read, added
}
