import Foundation
import MferenceAppCore

extension AppleUIFixtures {
    // MARK: Calendar

    static var calendar: Calendar { Calendar.current }

    static func date(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        calendar.date(from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    static let now = date(15, 8)

    static let events: [CalendarEvent] = [
        CalendarEvent(id: "team-sync", title: "Team sync", start: date(14, 10), end: date(14, 10, 30), calendarName: "Work"),
        CalendarEvent(id: "focus-mon", title: "Focus time", start: date(14, 13), end: date(14, 15), calendarName: "Focus"),
        CalendarEvent(id: "coffee", title: "Coffee with Tom", start: date(15, 9, 30), end: date(15, 10), calendarName: "Personal"),
        CalendarEvent(id: "lunch", title: "Lunch", start: date(15, 12), end: date(15, 13), calendarName: "Home"),
        CalendarEvent(id: "design-review", title: "Design review", start: date(16, 10), end: date(16, 11), calendarName: "Work"),
        CalendarEvent(id: "planning", title: "Weekly planning", start: date(16, 13), end: date(16, 14), calendarName: "Work"),
        CalendarEvent(id: "launch", title: "Launch check-in", start: date(17, 11), end: date(17, 12), calendarName: "Work"),
        CalendarEvent(id: "focus-fri", title: "Focus time", start: date(18, 9), end: date(18, 11), calendarName: "Focus"),
        CalendarEvent(id: "contract", title: "Contract call", start: date(18, 13), end: date(18, 13, 30), calendarName: "Personal"),
    ]

    static let proposal = ProposedEvent(
        title: "Design review", start: date(16, 11), end: date(16, 11, 45),
        source: ItemSource(kind: .mail, label: "Priya Raman", reference: "<review@example.com>"),
        replacesEventID: "design-review", note: "From Priya’s email this morning. Priya gets an update.")

    static let todos: [TodoItem] = [
        TodoItem(title: "Send copy decisions to Priya", due: date(15, 17),
                 source: ItemSource(kind: .mail, label: "Priya Raman", reference: "<copy@example.com>"),
                 createdAt: date(14, 9)),
        TodoItem(title: "Review contract section 4", due: date(18, 17),
                 source: ItemSource(kind: .mail, label: "Tom Okafor", reference: "<redlines@example.com>"),
                 createdAt: date(14, 9, 1)),
        TodoItem(title: "Pick a dentist appointment",
                 source: ItemSource(kind: .mail, label: "Northside Dental", reference: "<dental@example.com>"),
                 createdAt: date(14, 9, 2)),
        TodoItem(title: "Share the launch checklist",
                 source: ItemSource(kind: .meeting, label: "Weekly product sync"), createdAt: date(14, 9, 3)),
        TodoItem(title: "Expense the team lunch", isDone: true, createdAt: date(14, 9, 4)),
    ]
}
