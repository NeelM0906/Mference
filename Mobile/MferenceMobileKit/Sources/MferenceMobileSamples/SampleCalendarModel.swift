import Foundation
import MferenceMobileUI
import Observation

/// Calendar on sample events and to-dos (the Mac mockups' examples), for
/// previews, tests and the skeleton build until mobile-apple's EventKit model
/// lands. Nothing here reads Apple Calendar or Reminders.
@Observable
public final class SampleCalendarModel: CalendarScreenModel {
    public let calendar: Calendar
    public var calendarMode: CalendarMode = .day
    public private(set) var selectedDay: Date
    public var monthOffset = 0
    public var events: [PhoneEvent]
    public private(set) var todos: [PhoneTodo]
    public private(set) var suggestion: CalendarSuggestion?

    public init(now: Date = .now, calendar: Calendar = .current) {
        self.calendar = calendar
        selectedDay = calendar.startOfDay(for: now)
        let planner = Self.planner(today: now, calendar: calendar)
        events = planner.events
        todos = planner.todos
        suggestion = planner.suggestion
    }

    public var openTodoCount: Int { todos.filter { !$0.isDone }.count }

    public func events(on day: Date) -> [PhoneEvent] {
        events.filter { calendar.isDate($0.start, inSameDayAs: day) }.sorted { $0.start < $1.start }
    }

    public func hasEvents(on day: Date) -> Bool { !events(on: day).isEmpty }

    public func select(day: Date, switchToDay: Bool) {
        monthOffset = 0
        selectedDay = calendar.startOfDay(for: day)
        if switchToDay { calendarMode = .day }
    }

    public func goToToday() {
        selectedDay = calendar.startOfDay(for: .now)
        monthOffset = 0
    }

    public func acceptSuggestion() {
        guard let suggestion else { return }
        if let index = events.firstIndex(where: { $0.id == suggestion.event.id }) {
            events[index] = suggestion.event
        }
        self.suggestion = nil
    }

    public func dismissSuggestion() { suggestion = nil }

    public func toggle(_ todo: PhoneTodo) {
        guard let index = todos.firstIndex(where: { $0.id == todo.id }) else { return }
        todos[index].isDone.toggle()
    }

    public func addTodo(_ title: String) {
        let trimmed = title.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        todos.insert(PhoneTodo(title: trimmed, detail: "Added on iPhone"), at: 0)
    }

    // MARK: - Sample content

    struct Planner {
        var events: [PhoneEvent]
        var todos: [PhoneTodo]
        var suggestion: CalendarSuggestion?
    }

    static func planner(today now: Date, calendar: Calendar) -> Planner {
        let today = calendar.startOfDay(for: now)
        func at(_ dayOffset: Int, _ hour: Int, _ minute: Int = 0) -> Date {
            let day = calendar.date(byAdding: .day, value: dayOffset, to: today) ?? today
            return calendar.date(bySettingHour: hour, minute: minute, second: 0, of: day) ?? day
        }
        let review = PhoneEvent(title: "Design review", start: at(1, 10), end: at(1, 10, 45), tint: .blue)
        let events = [
            PhoneEvent(title: "Coffee with Tom", start: at(0, 9, 30), end: at(0, 10), tint: .orange),
            PhoneEvent(title: "Focus time", start: at(0, 10), end: at(0, 11), tint: .grey),
            PhoneEvent(title: "Lunch", start: at(0, 12), end: at(0, 13), tint: .green),
            PhoneEvent(title: "Weekly planning", start: at(0, 13), end: at(0, 14), tint: .blue),
            review,
            PhoneEvent(title: "Launch check-in", start: at(2, 11), end: at(2, 12), tint: .purple),
            PhoneEvent(title: "Team sync", start: at(-1, 10), end: at(-1, 10, 30), tint: .blue),
            PhoneEvent(title: "Contract call", start: at(3, 13), end: at(3, 13, 30), tint: .orange),
        ]
        let tomorrow = calendar.date(byAdding: .day, value: 1, to: today) ?? today
        let weekday = tomorrow.formatted(.dateTime.weekday(.abbreviated))
        var moved = review
        moved.start = at(1, 11)
        moved.end = at(1, 11, 45)
        let suggestion = CalendarSuggestion(
            question: "Move the design review to \(weekday) 11:00?",
            detail: "From Priya's email this morning. Priya gets an update.",
            event: moved)
        let todos = [
            PhoneTodo(title: "Send copy decisions to Priya", detail: "Mail · due \(weekday)"),
            PhoneTodo(title: "Review contract section 4", detail: "Mail · due Fri"),
            PhoneTodo(title: "Pick a dentist appointment", detail: "Mail · Northside Dental"),
            PhoneTodo(title: "Share the launch checklist", detail: "Meeting · Weekly product sync"),
            PhoneTodo(title: "Expense the team lunch", detail: "Scan", isDone: true),
        ]
        return Planner(events: events, todos: todos, suggestion: suggestion)
    }
}
