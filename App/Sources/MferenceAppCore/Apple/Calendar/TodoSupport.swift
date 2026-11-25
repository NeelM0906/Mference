import Foundation

/// The calendar's four views, ⌘1 to ⌘4.
public enum CalendarViewMode: String, CaseIterable, Sendable, Hashable, Identifiable {
    case day, week, month, year

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .day: "Day"
        case .week: "Week"
        case .month: "Month"
        case .year: "Year"
        }
    }
}

/// One group in the to-do panel.
public struct TodoSection: Sendable, Hashable, Identifiable {
    public enum Kind: Sendable, Hashable {
        /// Open to-dos from days before today.
        case overdue
        case day(Date)
        case undated
    }

    public var kind: Kind
    public var todos: [TodoItem]

    public var id: String {
        switch kind {
        case .overdue: "overdue"
        case .day(let date): "day-\(Int(date.timeIntervalSince1970))"
        case .undated: "undated"
        }
    }

    /// Overdue, Today, each later day with to-dos (and the selected day,
    /// even empty, so a typed to-do has a visible home), No date. A to-do
    /// checked off today stays in its group after the open ones, struck
    /// through; one that was overdue moves to Today, so Overdue holds open
    /// to-dos only. To-dos done on an earlier day (or before the check-off
    /// day was recorded) are not shown. Inside a group, oldest first.
    public static func sections(_ todos: [TodoItem], today: Date, selected: Date, calendar: Calendar) -> [TodoSection] {
        let byAge = todos.sorted { ($0.createdAt, $0.title) < ($1.createdAt, $1.title) }
        let open = byAge.filter { !$0.isDone }
        let doneToday = byAge.filter { todo in
            todo.isDone && todo.completedAt.map { calendar.isDate($0, inSameDayAs: today) } == true
        }
        var overdue: [TodoItem] = []
        var days: [Date: [TodoItem]] = [today: []]
        if selected >= today { days[selected] = days[selected] ?? [] }
        var undated: [TodoItem] = []
        for todo in open {
            guard let due = todo.due else { undated.append(todo); continue }
            let day = calendar.startOfDay(for: due)
            if day < today { overdue.append(todo) } else { days[day, default: []].append(todo) }
        }
        for todo in doneToday {
            guard let due = todo.due else { undated.append(todo); continue }
            days[max(calendar.startOfDay(for: due), today), default: []].append(todo)
        }
        var result: [TodoSection] = []
        if !overdue.isEmpty { result.append(TodoSection(kind: .overdue, todos: overdue)) }
        for day in days.keys.sorted() {
            let items = days[day] ?? []
            // Empty days show only for today and the selected day.
            if items.isEmpty, day != today, day != selected { continue }
            result.append(TodoSection(kind: .day(day), todos: items))
        }
        if !undated.isEmpty { result.append(TodoSection(kind: .undated, todos: undated)) }
        return result
    }
}

/// The to-do panel's short dates, in this Mac's region: 11/18/26 in the US,
/// 18/11/26 in the UK.
public enum TodoDateFormat {
    public static func short(_ date: Date, calendar: Calendar = .autoupdatingCurrent) -> String {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.locale = calendar.locale ?? .autoupdatingCurrent
        formatter.dateStyle = .short
        formatter.timeStyle = .none
        return formatter.string(from: date)
    }

    /// "Today", "Tomorrow", "Yesterday", else the weekday ("Fri").
    public static func dayName(_ date: Date, today: Date, calendar: Calendar = .autoupdatingCurrent) -> String {
        let days = calendar.dateComponents([.day], from: calendar.startOfDay(for: today),
                                           to: calendar.startOfDay(for: date)).day ?? 0
        switch days {
        case 0: return "Today"
        case 1: return "Tomorrow"
        case -1: return "Yesterday"
        default:
            let formatter = DateFormatter()
            formatter.calendar = calendar
            formatter.timeZone = calendar.timeZone
            formatter.locale = calendar.locale ?? .autoupdatingCurrent
            formatter.setLocalizedDateFormatFromTemplate("EEE")
            return formatter.string(from: date)
        }
    }
}

/// Reads a date out of a typed to-do: "Call the bank tomorrow" is "Call the
/// bank", due tomorrow (relative to the real clock: `NSDataDetector` takes no
/// reference date). Only a date at the start or end of the text counts,
/// so a title that merely mentions one in the middle stays whole.
public enum TodoInput {
    public static func parse(_ text: String, calendar: Calendar) -> (title: String, due: Date?) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.date.rawValue) else {
            return (trimmed, nil)
        }
        let whole = NSRange(trimmed.startIndex..., in: trimmed)
        let matches = detector.matches(in: trimmed, options: [], range: whole)
        for match in matches {
            guard let date = match.date, let range = Range(match.range, in: trimmed) else { continue }
            let atStart = range.lowerBound == trimmed.startIndex
            let atEnd = range.upperBound == trimmed.endIndex
            guard atStart || atEnd else { continue }
            var rest = trimmed
            rest.removeSubrange(range)
            rest = rest.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
            // A leading "on", "by" or "due" left before an end date goes too.
            for word in ["on", "by", "due", "at"] where rest.lowercased().hasSuffix(" " + word) {
                rest = String(rest.dropLast(word.count + 1))
            }
            guard !rest.isEmpty else { return (trimmed, nil) }
            return (rest, calendar.startOfDay(for: date))
        }
        return (trimmed, nil)
    }
}

/// Text typed in the to-do box that is a question about the schedule, not a
/// task: it goes to Ask about your week and never becomes a to-do. A
/// question mark at the end, or one of a few unambiguous openings ("when am
/// i free", "what do i have thursday"). Openings only, so to-dos such as
/// "Ask Priya about the deck" or "Check flight time" stay to-dos.
public enum TodoQuestion {
    static let openings = [
        "when am i", "when do i", "when's my", "when is my", "what do i", "what's on", "what is on",
        "what does my", "what have i got", "do i have", "am i free", "am i busy", "which day", "how busy",
        "how much free", "how many meetings", "is my week", "is my day", "how does my",
    ]

    public static func isQuestion(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        if trimmed.hasSuffix("?") { return true }
        let lowered = trimmed.lowercased().replacingOccurrences(of: "’", with: "'")
        return openings.contains { opening in
            lowered == opening || lowered.hasPrefix(opening + " ")
        }
    }
}
