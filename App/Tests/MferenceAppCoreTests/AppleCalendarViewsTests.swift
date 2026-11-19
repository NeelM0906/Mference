import Foundation
@testable import MferenceAppCore
import Testing

/// Day, week, month and year views; dated to-dos; two-way Reminders sync.
@Suite struct AppleCalendarViewsTests {
    @MainActor static func model(service: InMemoryCalendarService = InMemoryCalendarService(events: CalendarFixtures.events),
                                 store: PlannerStore = PlannerStore(url: nil),
                                 defaults: UserDefaults = UserDefaults(suiteName: "mference-tests-\(UUID().uuidString)")!)
        -> CalendarWeekModel {
        CalendarWeekModel(service: service, store: store, layout: AppleCalendarTests.layout, defaults: defaults,
                          now: { CalendarFixtures.now }, syncsAutomatically: false)
    }

    static func syncOn() -> UserDefaults {
        let defaults = UserDefaults(suiteName: "mference-tests-\(UUID().uuidString)")!
        RemindersSyncSetting.setEnabled(true, defaults)
        return defaults
    }

    // MARK: Views

    @Test @MainActor func switchesBetweenDayWeekMonthAndYear() async {
        let defaults = UserDefaults(suiteName: "mference-tests-\(UUID().uuidString)")!
        let model = Self.model(defaults: defaults)
        await model.load()
        #expect(model.mode == .week)

        await model.show(.day)
        #expect(model.title == "Tuesday, September 15, 2026")
        #expect(model.days == [CalendarFixtures.date(15, 0)])
        #expect(model.events.map(\.title) == ["Coffee with Tom", "Lunch"])
        await model.step(1)
        #expect(model.selectedDate == CalendarFixtures.date(16, 0))

        await model.show(.month)
        #expect(model.title == "September 2026")
        // Monday-first: Mon Aug 31 to Sun Oct 4, five whole weeks.
        #expect(model.monthGrid.count == 35)
        #expect(model.monthGrid.first == CalendarFixtures.calendar.date(from: DateComponents(year: 2026, month: 8, day: 31)))
        #expect(model.events(on: CalendarFixtures.date(17, 0)).map(\.title).first == "Offsite")  // all-day first
        #expect(model.events.contains { $0.title == "Next week" })
        await model.step(1)
        #expect(model.title == "October 2026")

        await model.show(.year)
        #expect(model.title == "2026")
        #expect(model.daysWithEvents.contains(CalendarFixtures.date(14, 0)))
        #expect(model.daysWithEvents.contains(CalendarFixtures.date(17, 0)))  // the all-day Offsite

        await model.open(CalendarFixtures.date(22, 0), in: .day)
        #expect(model.mode == .day && model.events.map(\.title) == ["Next week"])
        await model.showToday()
        #expect(model.mode == .day && model.selectedDate == CalendarFixtures.date(15, 0))

        // The view is remembered.
        let reopened = Self.model(defaults: defaults)
        #expect(reopened.mode == .day)
    }

    @Test @MainActor func selectingADayInTheWeekKeepsTheWeek() async {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let model = Self.model(service: service)
        await model.load()
        await model.select(CalendarFixtures.date(18, 0))
        #expect(model.selectedDate == CalendarFixtures.date(18, 0))
        #expect(model.weekStart == CalendarFixtures.date(14, 0))
        await model.select(CalendarFixtures.date(22, 0))
        #expect(model.weekStart == CalendarFixtures.date(21, 0))
        #expect(model.events.map(\.title) == ["Next week"])
    }

    // MARK: Dated to-dos

    @Test @MainActor func aTypedToDoIsDatedToTheSelectedDay() async throws {
        let model = Self.model()
        await model.select(CalendarFixtures.date(17, 0))
        let todo = try #require(model.addTodo(text: "Email the landlord", due: model.selectedDate))
        #expect(todo.due == CalendarFixtures.date(17, 0))
        #expect(todo.title == "Email the landlord")
        // The chip changed to "No date".
        let undated = try #require(model.addTodo(text: "Read the post", due: nil, dueChosen: true))
        #expect(undated.due == nil)
        #expect(model.addTodo(text: "   ", due: nil) == nil)

        let kinds = model.todoSections.map(\.kind)
        #expect(kinds == [.day(CalendarFixtures.date(15, 0)), .day(CalendarFixtures.date(17, 0)), .undated])
        #expect(model.todoSections[1].todos.map(\.title) == ["Email the landlord"])
    }

    @Test func sectionsGroupOverdueTodayLaterUndatedAndDone() {
        let calendar = CalendarFixtures.calendar
        func at(_ day: Int, _ hour: Int) -> Date { CalendarFixtures.date(day, hour) }
        let todos = [
            TodoItem(title: "Late", due: at(13, 0), createdAt: at(1, 0)),
            TodoItem(title: "Today", due: at(15, 17), createdAt: at(2, 0)),
            TodoItem(title: "Friday", due: at(18, 0), createdAt: at(3, 0)),
            TodoItem(title: "Someday", createdAt: at(4, 0)),
            TodoItem(title: "Finished", isDone: true, due: at(13, 0), createdAt: at(5, 0)),
        ]
        let sections = TodoSection.sections(todos, today: at(15, 0), selected: at(16, 0), calendar: calendar)
        #expect(sections.map(\.kind) == [.overdue, .day(at(15, 0)), .day(at(16, 0)), .day(at(18, 0)), .undated, .completed])
        #expect(sections.map { $0.todos.map(\.title) } == [["Late"], ["Today"], [], ["Friday"], ["Someday"], ["Finished"]])
    }

    @Test func shortDatesFollowTheRegion() {
        var us = CalendarFixtures.calendar
        us.locale = Locale(identifier: "en_US")
        #expect(TodoDateFormat.short(CalendarFixtures.date(15, 9), calendar: us) == "9/15/26")
        var uk = CalendarFixtures.calendar
        uk.locale = Locale(identifier: "en_GB")
        #expect(TodoDateFormat.short(CalendarFixtures.date(15, 9), calendar: uk).hasPrefix("15/09/"))
        #expect(TodoDateFormat.dayName(CalendarFixtures.date(16, 0), today: CalendarFixtures.date(15, 0), calendar: us)
            == "Tomorrow")
    }

    @Test func aDateAtTheEndOfTheTextBecomesTheDueDate() throws {
        let calendar = Calendar.autoupdatingCurrent
        let parsed = TodoInput.parse("Call the bank tomorrow", calendar: calendar)
        #expect(parsed.title == "Call the bank")
        let tomorrow = try #require(calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: .now)))
        #expect(parsed.due == tomorrow)
        let plain = TodoInput.parse("Buy milk", calendar: calendar)
        #expect(plain.title == "Buy milk" && plain.due == nil)
    }

    @Test func aPlannerFileFromBeforeTheseFieldsStillLoads() throws {
        let url = URL.temporaryDirectory.appending(path: "planner-old-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let old = """
        {"handledSources":[],"proposals":[],"todos":[{"createdAt":0,"id":"\(UUID().uuidString)","isDone":false,\
        "source":{"kind":"manual","label":"Added by you"},"title":"Kept","unlinkedFromReminders":false}],"version":1}
        """
        try Data(old.utf8).write(to: url)
        let store = PlannerStore(url: url)
        #expect(store.snapshot.todos.map(\.title) == ["Kept"])
        #expect(store.snapshot.deletedReminderIDs.isEmpty)
    }

    // MARK: Reminders, both ways

    /// The maintainer added a reminder in Reminders and synced; nothing came
    /// in. Reminders added there now become to-dos here.
    @Test @MainActor func aReminderAddedInRemindersShowsUpHere() async throws {
        let service = InMemoryCalendarService(remindersAccess: .granted)
        let model = Self.model(service: service, defaults: Self.syncOn())
        let id = await service.addReminderInReminders("Pick up dry cleaning", due: CalendarFixtures.date(15, 0))
        await model.syncReminders()  // works with no to-dos at all
        let todo = try #require(model.todos.first)
        #expect(todo.title == "Pick up dry cleaning")
        #expect(todo.reminderID == id)
        #expect(todo.source.kind == .reminders && todo.source.label == "Reminders")
        #expect(model.lastReminderSync == CalendarFixtures.now)

        // Done here: done in Reminders. A second sync imports nothing twice.
        model.setDone(true, for: todo.id)
        await model.syncReminders()
        #expect(await service.reminders[id]?.isDone == true)
        await model.syncReminders()
        #expect(model.todos.count == 1)
    }

    @Test @MainActor func aSyncThatChangesNothingSavesNothing() async {
        let service = InMemoryCalendarService(remindersAccess: .granted)
        let model = Self.model(service: service, defaults: Self.syncOn())
        model.addTodo(TodoItem(title: "Send copy decisions"))
        await model.syncReminders()
        let saves = await service.saves
        await model.syncReminders()
        await model.syncReminders()
        #expect(await service.saves == saves)
    }

    @Test @MainActor func onlyChosenListsShowAndNewToDosGoToTheChosenList() async throws {
        let lists = [ReminderList(id: "home", title: "Home", isDefault: true), ReminderList(id: "work", title: "Work")]
        let service = InMemoryCalendarService(remindersAccess: .granted, lists: lists)
        let defaults = Self.syncOn()
        let model = Self.model(service: service, defaults: defaults)
        await service.addReminderInReminders("Water plants", list: "home")
        await service.addReminderInReminders("File expenses", list: "work")
        await model.loadReminderLists()
        #expect(model.reminderLists.map(\.id) == ["home", "work"])

        model.setShown(false, list: "work")
        #expect(model.shownReminderListIDs == ["home"])
        await model.syncReminders()
        #expect(model.todos.map(\.title) == ["Water plants"])

        model.setTargetReminderList("work")
        let typed = try #require(model.addTodo(text: "Book flights", due: nil, dueChosen: true))
        await model.syncReminders()
        let reminderID = try #require(model.todos.first { $0.id == typed.id }?.reminderID)
        #expect(await service.reminderListOf[reminderID] == "work")
        // A to-do added here stays though its list is not shown; the other
        // Work reminder still does not show.
        #expect(Set(model.todos.map(\.title)) == ["Water plants", "Book flights"])

        // Ticking every list again stores "every list".
        model.setShown(true, list: "work")
        #expect(model.shownReminderListIDs == nil)
        await model.syncReminders()
        #expect(model.todos.contains { $0.title == "File expenses" })
    }

    @Test @MainActor func quietSyncNeverAsks() async {
        let service = InMemoryCalendarService(remindersAccess: .notDetermined)
        let model = Self.model(service: service, defaults: Self.syncOn())
        await model.syncRemindersQuietly()
        #expect(await service.requests.isEmpty)
        await model.calendarChanged()
        #expect(await service.requests.isEmpty)
    }
}
