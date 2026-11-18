import Foundation
@testable import MferenceAppCore
import Testing

enum CalendarFixtures {
    static var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/Los_Angeles")!
        return calendar
    }

    /// Tuesday, September 15 2026, 08:00 Pacific.
    static let now = calendar.date(from: DateComponents(year: 2026, month: 9, day: 15, hour: 8))!

    static func date(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        calendar.date(from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    /// The mockup's week, invented names only.
    static let events: [CalendarEvent] = [
        CalendarEvent(id: "team-sync", title: "Team sync", start: date(14, 10), end: date(14, 10, 30)),
        CalendarEvent(id: "focus-mon", title: "Focus time", start: date(14, 13), end: date(14, 15)),
        CalendarEvent(id: "coffee", title: "Coffee with Tom", start: date(15, 9, 30), end: date(15, 10)),
        CalendarEvent(id: "lunch", title: "Lunch", start: date(15, 12), end: date(15, 13)),
        CalendarEvent(id: "design-review", title: "Design review", start: date(16, 10), end: date(16, 11)),
        CalendarEvent(id: "planning", title: "Weekly planning", start: date(16, 13), end: date(16, 14)),
        CalendarEvent(id: "launch", title: "Launch check-in", start: date(17, 11), end: date(17, 12)),
        CalendarEvent(id: "focus-fri", title: "Focus time", start: date(18, 9), end: date(18, 11)),
        CalendarEvent(id: "contract", title: "Contract call", start: date(18, 13), end: date(18, 13, 30)),
        CalendarEvent(id: "holiday", title: "Offsite", start: date(17, 0), end: date(18, 0), isAllDay: true),
        CalendarEvent(id: "next-week", title: "Next week", start: date(22, 10), end: date(22, 11)),
    ]

    static let moveProposal = ProposedEvent(
        title: "Design review", start: date(16, 11), end: date(16, 11, 45),
        source: ItemSource(kind: .mail, label: "Priya Raman", reference: "<review@example.com>"),
        replacesEventID: "design-review", note: "From Priya's email this morning. Priya gets an update.")
}

@Suite struct AppleCalendarTests {
    static let layout = WeekLayout(calendar: CalendarFixtures.calendar)

    @MainActor static func model(service: InMemoryCalendarService = InMemoryCalendarService(events: CalendarFixtures.events),
                                 store: PlannerStore = PlannerStore(url: nil),
                                 defaults: UserDefaults = UserDefaults(suiteName: "mference-tests-\(UUID().uuidString)")!)
        -> CalendarWeekModel {
        CalendarWeekModel(service: service, store: store, layout: layout, defaults: defaults,
                          now: { CalendarFixtures.now })
    }

    @Test func weekStartsOnMondayAndLaysOutBlocks() {
        let start = Self.layout.weekStart(containing: CalendarFixtures.now)
        #expect(start == CalendarFixtures.date(14, 0))
        let blocks = Self.layout.blocks(weekStart: start, events: CalendarFixtures.events,
                                        proposals: [CalendarFixtures.moveProposal])
        #expect(!blocks.contains { $0.title == "Next week" || $0.title == "Offsite" })
        let review = blocks.first { if case .moving = $0.kind { true } else { false } }
        #expect(review?.day == 2)
        #expect(review?.top == 1)
        let proposal = blocks.first { if case .proposal = $0.kind { true } else { false } }
        #expect(proposal?.day == 2)
        #expect(proposal?.top == 2)
        #expect(proposal?.height == 0.75)
        let focusFriday = blocks.first { $0.id == "event-focus-fri" }
        #expect(focusFriday?.top == 0 && focusFriday?.height == 2)
        #expect(Self.layout.allDayEvents(weekStart: start, events: CalendarFixtures.events)[3]?.map(\.title) == ["Offsite"])
    }

    @Test func eventsOutsideTheGridAreListedNotHidden() {
        let start = Self.layout.weekStart(containing: CalendarFixtures.now)
        let events = CalendarFixtures.events + [
            CalendarEvent(id: "gym", title: "Gym", start: CalendarFixtures.date(15, 7), end: CalendarFixtures.date(15, 8)),
            CalendarEvent(id: "dinner", title: "Dinner", start: CalendarFixtures.date(16, 19), end: CalendarFixtures.date(16, 21)),
        ]
        let outside = Self.layout.outsideGrid(weekStart: start, events: events)
        #expect(outside[1]?.map(\.title) == ["Gym"])
        #expect(outside[2]?.map(\.title) == ["Dinner"])
        #expect(outside[3]?.map(\.title) == ["Offsite"])
        #expect(outside[0] == nil)
        #expect(!Self.layout.blocks(weekStart: start, events: events, proposals: []).contains { $0.title == "Gym" })
    }

    @Test @MainActor func aDoubleClickOnAddCreatesOneEvent() async {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let model = Self.model(service: service)
        await model.load()
        model.propose(CalendarFixtures.moveProposal)
        async let first: Void = model.add(CalendarFixtures.moveProposal)
        async let second: Void = model.add(CalendarFixtures.moveProposal)
        _ = await (first, second)
        #expect(await service.storedEvents.filter { $0.title == "Design review" }.count == 1)
        #expect(model.addingProposals.isEmpty)
        await model.add(CalendarFixtures.moveProposal)  // already added: nothing happens
        #expect(await service.storedEvents.filter { $0.title == "Design review" }.count == 1)
    }

    @Test func shortTimesReadLikeTheMockup() {
        let calendar = CalendarFixtures.calendar
        let at = CalendarFixtures.date
        #expect(CalendarTimeFormat.range(at(18, 9, 0), at(18, 11, 0), calendar: calendar) == "9 to 11 AM")
        #expect(CalendarTimeFormat.range(at(14, 13, 0), at(14, 15, 0), calendar: calendar) == "1 to 3 PM")
        #expect(CalendarTimeFormat.range(at(15, 9, 30), at(15, 10, 0), calendar: calendar) == "9:30 to 10 AM")
        #expect(CalendarTimeFormat.range(at(17, 11, 0), at(17, 12, 0), calendar: calendar) == "11 AM to 12 PM")
        #expect(CalendarTimeFormat.time(at(16, 11, 0), calendar: calendar) == "11:00")
        #expect(CalendarTimeFormat.dayRange(at(16, 11, 0), at(16, 11, 45), calendar: calendar) == "Wed 11:00 to 11:45")
    }

    /// Every occurrence of a series shares EventKit's identifier: blocks,
    /// moves and deletes must target one occurrence.
    @Test @MainActor func repeatingEventsAreHandledPerOccurrence() async throws {
        let monday = CalendarEvent(id: "standup", occurrenceDate: CalendarFixtures.date(14, 9, 30), title: "Standup",
                                   start: CalendarFixtures.date(14, 9, 30), end: CalendarFixtures.date(14, 9, 45))
        let tuesday = CalendarEvent(id: "standup", occurrenceDate: CalendarFixtures.date(15, 9, 30), title: "Standup",
                                    start: CalendarFixtures.date(15, 9, 30), end: CalendarFixtures.date(15, 9, 45))
        #expect(monday.id != tuesday.id)
        let start = Self.layout.weekStart(containing: CalendarFixtures.now)
        let move = ProposedEvent(title: "Standup", start: CalendarFixtures.date(15, 10), end: CalendarFixtures.date(15, 10, 15),
                                 source: ItemSource(kind: .mail, label: "Team", reference: "<standup@example.com>"),
                                 replacesEventID: "standup", replacesOccurrence: tuesday.occurrenceDate)
        let blocks = Self.layout.blocks(weekStart: start, events: [monday, tuesday], proposals: [move])
        #expect(Set(blocks.map(\.id)).count == blocks.count)
        let moving = blocks.filter { if case .moving = $0.kind { true } else { false } }
        #expect(moving.map(\.day) == [1])

        let service = InMemoryCalendarService(events: [monday, tuesday])
        let model = Self.model(service: service)
        await model.load()
        model.propose(move)
        await model.add(move)
        let stored = await service.storedEvents
        #expect(stored.contains(monday))
        #expect(!stored.contains(tuesday))
        try await service.remove(monday.reference)
        #expect(await service.storedEvents.map(\.title) == ["Standup"])  // the moved one stays
    }

    @Test @MainActor func remindersSyncKeepsEditsMadeInRemindersAndRespectsDeletes() async throws {
        let defaults = UserDefaults(suiteName: "mference-tests-\(UUID().uuidString)")!
        RemindersSyncSetting.setEnabled(true, defaults)
        let service = InMemoryCalendarService(remindersAccess: .granted)
        let model = Self.model(service: service, defaults: defaults)
        let first = TodoItem(title: "Send copy decisions")
        let second = TodoItem(title: "Book the dentist")
        model.addTodo(first)
        model.addTodo(second)
        await model.syncReminders()
        let firstID = try #require(model.todos.first { $0.id == first.id }?.reminderID)
        let secondID = try #require(model.todos.first { $0.id == second.id }?.reminderID)

        // In Reminders: the first is renamed, the second deleted.
        await service.editReminder(firstID, title: "Send copy decisions to Priya")
        await service.deleteReminder(secondID)
        // In Mference: the first is marked done.
        model.setDone(true, for: first.id)
        await model.syncReminders()

        let reminder = try #require(await service.reminders[firstID])
        #expect(reminder.title == "Send copy decisions to Priya")  // Reminders' edit kept
        #expect(reminder.isDone)                                  // Mference's change pushed
        #expect(await service.reminders.count == 1)               // the deleted one is not recreated
        let unlinked = try #require(model.todos.first { $0.id == second.id })
        #expect(unlinked.reminderID == nil && unlinked.unlinkedFromReminders)
        await model.syncReminders()
        #expect(await service.reminders.count == 1)
    }

    @Test @MainActor func newEventFromTheToolbarIsAddedAfterAccess() async {
        let service = InMemoryCalendarService(calendarAccess: .notDetermined)
        let model = Self.model(service: service)
        await model.load()
        #expect(await model.createEvent(title: "  ", start: CalendarFixtures.date(16, 9), end: CalendarFixtures.date(16, 10)) == false)
        #expect(await model.createEvent(title: "Dentist", start: CalendarFixtures.date(16, 9), end: CalendarFixtures.date(16, 9, 30)))
        #expect(await service.requests == [.calendar])
        #expect(model.events.map(\.title) == ["Dentist"])
    }

    @Test func overlappingEventsShareTheColumn() {
        let start = Self.layout.weekStart(containing: CalendarFixtures.now)
        let events = [
            CalendarEvent(id: "a", title: "A", start: CalendarFixtures.date(14, 10), end: CalendarFixtures.date(14, 12)),
            CalendarEvent(id: "b", title: "B", start: CalendarFixtures.date(14, 11), end: CalendarFixtures.date(14, 13)),
            CalendarEvent(id: "c", title: "C", start: CalendarFixtures.date(14, 12), end: CalendarFixtures.date(14, 13)),
            CalendarEvent(id: "d", title: "D", start: CalendarFixtures.date(14, 15), end: CalendarFixtures.date(14, 16)),
        ]
        let blocks = Dictionary(uniqueKeysWithValues: Self.layout.blocks(weekStart: start, events: events, proposals: [])
            .map { ($0.title, $0) })
        #expect(blocks["A"]?.lane == 0 && blocks["A"]?.lanes == 2)
        #expect(blocks["B"]?.lane == 1 && blocks["B"]?.lanes == 2)
        #expect(blocks["C"]?.lane == 0)
        #expect(blocks["D"]?.lanes == 1)
    }

    @Test @MainActor func asksForAccessOnlyWhenThePersonClicks() async {
        let service = InMemoryCalendarService(events: CalendarFixtures.events, calendarAccess: .notDetermined)
        let model = Self.model(service: service)
        await model.load()
        #expect(model.access == .notAsked)
        #expect(model.events.isEmpty)
        #expect(await service.requests.isEmpty)
        await model.requestAccess()
        #expect(model.access == .granted)
        #expect(await service.requests == [.calendar])
        #expect(model.events.count == 10)
        #expect(model.weekTitle == "September 14 to 18")
        #expect(model.isToday(CalendarFixtures.date(15, 0)))
    }

    @Test @MainActor func deniedAccessPointsToSettings() async {
        let service = InMemoryCalendarService(calendarAccess: .notDetermined, grantOnRequest: false)
        let model = Self.model(service: service)
        await model.requestAccess()
        #expect(model.access == .blocked)
    }

    @Test @MainActor func addingAProposalMovesTheEventWithOneClick() async {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let model = Self.model(service: service)
        await model.load()
        model.propose(CalendarFixtures.moveProposal)
        #expect(model.proposalsThisWeek.count == 1)
        #expect(model.blocks.contains { if case .moving = $0.kind { true } else { false } })

        await model.add(CalendarFixtures.moveProposal)
        #expect(model.proposals.isEmpty)
        let stored = await service.storedEvents
        #expect(!stored.contains { $0.id == "design-review" })
        let moved = stored.first { $0.title == "Design review" }
        #expect(moved?.start == CalendarFixtures.date(16, 11))
        #expect(model.events.contains { $0.title == "Design review" && $0.start == CalendarFixtures.date(16, 11) })

        // The same mail does not propose it again.
        model.propose(CalendarFixtures.moveProposal)
        #expect(model.proposals.isEmpty)
    }

    @Test @MainActor func dismissingLeavesCalendarUntouched() async {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let model = Self.model(service: service)
        await model.load()
        model.propose(CalendarFixtures.moveProposal)
        model.dismiss(CalendarFixtures.moveProposal)
        #expect(model.proposals.isEmpty)
        #expect(await service.storedEvents == CalendarFixtures.events)
    }

    @Test @MainActor func weekNavigation() async {
        let model = Self.model()
        await model.load()
        await model.showWeek(offset: 1)
        #expect(model.weekStart == CalendarFixtures.date(21, 0))
        #expect(model.events.map(\.title) == ["Next week"])
        #expect(model.weekTitle == "September 21 to 25")
        await model.showWeek(offset: 2)
        #expect(model.weekTitle == "October 5 to 9")
        await model.showToday()
        #expect(model.weekStart == CalendarFixtures.date(14, 0))
    }

    @Test @MainActor func remindersSyncOnlyWhenTurnedOnAndClicked() async throws {
        let defaults = UserDefaults(suiteName: "mference-tests-\(UUID().uuidString)")!
        let service = InMemoryCalendarService()
        let model = Self.model(service: service, defaults: defaults)
        let todo = TodoItem(title: "Send copy decisions to Priya", due: CalendarFixtures.date(15, 17),
                            source: ItemSource(kind: .mail, label: "Priya Raman", reference: "<copy@example.com>"))
        model.addTodo(todo)
        model.addTodo(todo)  // the same source again is ignored
        #expect(model.todos.count == 1)

        #expect(!model.remindersSyncEnabled)
        await model.syncReminders()
        #expect(await service.requests.isEmpty)
        #expect(await service.reminders.isEmpty)

        RemindersSyncSetting.setEnabled(true, defaults)
        await model.syncReminders()
        #expect(await service.requests == [.reminders])
        #expect(await service.reminders.count == 1)
        let reminderID = try #require(model.todos.first?.reminderID)

        model.setDone(true, for: todo.id)
        await model.syncReminders()
        #expect(await service.reminders[reminderID]?.isDone == true)
        #expect(await service.reminders.count == 1)
    }

    @Test @MainActor func todosPersistAndSortOpenFirst() throws {
        let url = URL.temporaryDirectory.appending(path: "planner-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let model = Self.model(store: PlannerStore(url: url))
        let done = TodoItem(title: "Expense the team lunch", isDone: true)
        let later = TodoItem(title: "Review contract section 4", due: CalendarFixtures.date(18, 17))
        let sooner = TodoItem(title: "Send copy decisions", due: CalendarFixtures.date(15, 17))
        let undated = TodoItem(title: "Share the launch checklist")
        for todo in [done, later, sooner, undated] { model.addTodo(todo) }
        #expect(model.sortedTodos.map(\.title)
            == ["Send copy decisions", "Review contract section 4", "Share the launch checklist", "Expense the team lunch"])

        let reopened = Self.model(store: PlannerStore(url: url))
        #expect(reopened.todos.count == 4)
        reopened.removeTodo(done.id)
        #expect(PlannerStore(url: url).snapshot.todos.count == 3)
    }
}
