import Foundation

/// The note list's groups (Notes spec 4.1): Pinned first, then by the day of
/// the last edit: Today, Yesterday, Previous 7 days, Previous 30 days, then
/// one group per month. Newest edit first inside each group.
public enum NoteListGrouping {
    public enum Group: Sendable, Hashable {
        case pinned, today, yesterday, previous7Days, previous30Days
        /// A calendar month; `year` and `month` as the calendar counts them.
        case month(year: Int, month: Int)
    }

    /// A group and its notes, in display order.
    public struct Section<Item> {
        public var group: Group
        public var title: String
        public var items: [Item]
    }

    /// The group for a note edited at `date`, as seen at `now`.
    public static func group(for date: Date, isPinned: Bool, now: Date, calendar: Calendar = .current) -> Group {
        if isPinned { return .pinned }
        let today = calendar.startOfDay(for: now)
        let day = calendar.startOfDay(for: date)
        let days = calendar.dateComponents([.day], from: day, to: today).day ?? 0
        switch days {
        case ...0: return .today
        case 1: return .yesterday
        case 2...7: return .previous7Days
        case 8...30: return .previous30Days
        default:
            let parts = calendar.dateComponents([.year, .month], from: date)
            return .month(year: parts.year ?? 0, month: parts.month ?? 1)
        }
    }

    /// Sorts and groups `items`. `updatedAt` and `isPinned` read each item.
    public static func sections<Item>(_ items: [Item], now: Date, calendar: Calendar = .current,
                                      locale: Locale = .current,
                                      updatedAt: (Item) -> Date, isPinned: (Item) -> Bool) -> [Section<Item>] {
        let sorted = items.sorted { updatedAt($0) > updatedAt($1) }
        var sections: [Section<Item>] = []
        var index: [Group: Int] = [:]
        // Pinned always leads, whatever its edit dates.
        let ordered = sorted.filter(isPinned) + sorted.filter { !isPinned($0) }
        for item in ordered {
            let group = group(for: updatedAt(item), isPinned: isPinned(item), now: now, calendar: calendar)
            if let at = index[group] {
                sections[at].items.append(item)
            } else {
                index[group] = sections.count
                sections.append(Section(group: group, title: title(group, now: now, calendar: calendar, locale: locale),
                                        items: [item]))
            }
        }
        return sections
    }

    /// The group header ("Today", "October", "October 2025").
    public static func title(_ group: Group, now: Date, calendar: Calendar = .current, locale: Locale = .current) -> String {
        switch group {
        case .pinned: "Pinned"
        case .today: "Today"
        case .yesterday: "Yesterday"
        case .previous7Days: "Previous 7 days"
        case .previous30Days: "Previous 30 days"
        case .month(let year, let month):
            monthTitle(year: year, month: month, sameYear: calendar.component(.year, from: now) == year,
                       calendar: calendar, locale: locale)
        }
    }

    /// The row's time: "11:40" today and yesterday, the weekday within a
    /// week ("Mon"), the date before that.
    public static func timeLabel(for date: Date, now: Date, calendar: Calendar = .current,
                                 locale: Locale = .current) -> String {
        let days = calendar.dateComponents([.day], from: calendar.startOfDay(for: date),
                                           to: calendar.startOfDay(for: now)).day ?? 0
        var style: Date.FormatStyle
        switch days {
        case ...1:
            style = Date.FormatStyle(date: .omitted, time: .shortened)
        case 2...6:
            style = Date.FormatStyle().weekday(.abbreviated)
        default:
            style = Date.FormatStyle(date: .numeric, time: .omitted)
        }
        style.calendar = calendar
        style.timeZone = calendar.timeZone
        style.locale = locale
        return date.formatted(style)
    }

    private static func monthTitle(year: Int, month: Int, sameYear: Bool, calendar: Calendar, locale: Locale) -> String {
        var components = DateComponents()
        components.year = year
        components.month = month
        components.day = 1
        guard let date = calendar.date(from: components) else { return "" }
        var style = sameYear ? Date.FormatStyle().month(.wide) : Date.FormatStyle().month(.wide).year()
        style.calendar = calendar
        style.timeZone = calendar.timeZone
        style.locale = locale
        return date.formatted(style)
    }
}
