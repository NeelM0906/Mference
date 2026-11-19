import CoreGraphics
import EventKit
import Foundation
@testable import MferenceAppCore
import Synchronization
import Testing

enum CalendarFixtures {
    static var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/Los_Angeles")!
        calendar.locale = Locale(identifier: "en_US")
        // A Mac set to start the week on Monday (the grid follows the Mac).
        calendar.firstWeekday = 2
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
                          now: { CalendarFixtures.now }, syncsAutomatically: false)
    }

    @Test func weekStartsOnTheMacsFirstWeekdayAndLaysOutBlocks() {
        let start = Self.layout.weekStart(containing: CalendarFixtures.now)
        #expect(start == CalendarFixtures.date(14, 0))
        let blocks = Self.layout.blocks(weekStart: start, events: CalendarFixtures.events,
                                        proposals: [CalendarFixtures.moveProposal])
        #expect(!blocks.contains { $0.title == "Next week" || $0.title == "Offsite" })
        let review = blocks.first { if case .moving = $0.kind { true } else { false } }
        #expect(review?.day == 2)
        #expect(review?.top == 10)  // the grid starts at midnight
        let proposal = blocks.first { if case .proposal = $0.kind { true } else { false } }
        #expect(proposal?.day == 2)
        #expect(proposal?.top == 11)
        #expect(proposal?.height == Self.layout.minimumProposalHours)  // 45 min, drawn at the minimum
        let focusFriday = blocks.first { $0.id == "event-focus-fri-d4" }
        #expect(focusFriday?.top == 9 && focusFriday?.height == 2)
        // A Mac whose week starts on Sunday gets a Sunday-first grid.
        var sundayFirst = CalendarFixtures.calendar
        sundayFirst.firstWeekday = 1
        #expect(WeekLayout(calendar: sundayFirst).weekStart(containing: CalendarFixtures.now) == CalendarFixtures.date(13, 0))
        #expect(Self.layout.days == 7)
        #expect(Self.layout.allDayEvents(weekStart: start, events: CalendarFixtures.events)[3]?.map(\.title) == ["Offsite"])
    }

    /// The maintainer's 3:00 AM event sat in the "All day" row: the grid
    /// now covers the whole day, so early and late events are placed at
    /// their hour and only all-day events sit above the grid.
    @Test func earlyAndLateEventsStayInTheGrid() {
        let start = Self.layout.weekStart(containing: CalendarFixtures.now)
        let events = CalendarFixtures.events + [
            CalendarEvent(id: "testing", title: "Testing", start: CalendarFixtures.date(17, 3), end: CalendarFixtures.date(17, 4)),
            CalendarEvent(id: "dinner", title: "Dinner", start: CalendarFixtures.date(16, 19), end: CalendarFixtures.date(16, 21)),
        ]
        let blocks = Self.layout.blocks(weekStart: start, events: events, proposals: [])
        let early = blocks.first { $0.title == "Testing" }
        #expect(early?.day == 3 && early?.top == 3 && early?.height == 1)
        let late = blocks.first { $0.title == "Dinner" }
        #expect(late?.day == 2 && late?.top == 19 && late?.height == 2)
        let outside = Self.layout.outsideGrid(weekStart: start, events: events)
        #expect(outside.mapValues { $0.map(\.title) } == [3: ["Offsite"]])
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
        #expect(!model.todos.contains { $0.id == second.id })     // and it is deleted here too
        await model.syncReminders()
        #expect(await service.reminders.count == 1)

        // Deleted here: deleted in Reminders at the next sync.
        model.removeTodo(first.id)
        await model.syncReminders()
        #expect(await service.reminders.isEmpty)
        #expect(model.todos.isEmpty)
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

    @Test @MainActor func editAndDeleteFromTheWeek() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let model = Self.model(service: service)
        await model.load()
        let lunch = try #require(model.events.first { $0.title == "Lunch" })
        #expect(await model.updateEvent(lunch, title: "Lunch with Lena", start: CalendarFixtures.date(15, 12, 30),
                                        end: CalendarFixtures.date(15, 13, 30)))
        #expect(model.events.contains { $0.title == "Lunch with Lena" && $0.start == CalendarFixtures.date(15, 12, 30) })
        #expect(await model.updateEvent(lunch, title: " ", start: lunch.start, end: lunch.end) == false)
        let edited = try #require(model.events.first { $0.title == "Lunch with Lena" })
        #expect(await model.deleteEvent(edited))
        #expect(!model.events.contains { $0.title.hasPrefix("Lunch") })
        #expect(await model.deleteEvent(edited) == false)  // already gone
        #expect(model.errorMessage != nil)
    }

    @Test func eventsCrossingMidnightShowOnEveryDayTheyTouch() {
        let start = Self.layout.weekStart(containing: CalendarFixtures.now)
        let overnight = CalendarEvent(id: "deploy", title: "Deploy", start: CalendarFixtures.date(15, 16),
                                      end: CalendarFixtures.date(16, 11))
        let fromLastWeek = CalendarEvent(id: "trip", title: "Trip", start: CalendarFixtures.date(13, 20),
                                         end: CalendarFixtures.date(14, 10))
        let blocks = Self.layout.blocks(weekStart: start, events: [overnight, fromLastWeek], proposals: [])
        #expect(blocks.filter { $0.title == "Deploy" }.map(\.day).sorted() == [1, 2])
        let wednesday = blocks.first { $0.title == "Deploy" && $0.day == 2 }
        #expect(wednesday?.top == 0 && wednesday?.height == 11)
        let tuesday = blocks.first { $0.title == "Deploy" && $0.day == 1 }
        #expect(tuesday?.top == 16 && tuesday?.height == 8)
        #expect(blocks.filter { $0.title == "Trip" }.map(\.day) == [0])
        #expect(Set(blocks.map(\.id)).count == blocks.count)
        // A late-night event is drawn on both days, never listed above.
        let lateOnly = CalendarEvent(id: "late", title: "Late", start: CalendarFixtures.date(16, 21),
                                     end: CalendarFixtures.date(17, 8))
        let late = Self.layout.blocks(weekStart: start, events: [lateOnly], proposals: [])
        #expect(late.map(\.day).sorted() == [2, 3])
        #expect(Self.layout.outsideGrid(weekStart: start, events: [lateOnly]).isEmpty)
    }

    @Test @MainActor func editsMadeInRemindersBecomeTheToDosOwn() async throws {
        let defaults = UserDefaults(suiteName: "mference-tests-\(UUID().uuidString)")!
        RemindersSyncSetting.setEnabled(true, defaults)
        let service = InMemoryCalendarService(remindersAccess: .granted)
        let model = Self.model(service: service, defaults: defaults)
        let todo = TodoItem(title: "Send copy decisions")
        model.addTodo(todo)
        await model.syncReminders()
        let id = try #require(model.todos.first?.reminderID)
        await service.editReminder(id, title: "Send copy decisions to Priya")
        await model.syncReminders()
        #expect(model.todos.first?.title == "Send copy decisions to Priya")
        await model.syncReminders()  // a third sync must not revert the Reminders edit
        #expect(await service.reminders[id]?.title == "Send copy decisions to Priya")
    }

    @Test func aShortProposalKeepsItsRoomAndItsOwnLane() {
        let start = Self.layout.weekStart(containing: CalendarFixtures.now)
        let proposal = ProposedEvent(title: "Quick sync", start: CalendarFixtures.date(16, 11),
                                     end: CalendarFixtures.date(16, 11, 30),
                                     source: ItemSource(kind: .mail, label: "Priya", reference: "<s@example.com>"))
        let next = CalendarEvent(id: "next", title: "Next", start: CalendarFixtures.date(16, 11, 30),
                                 end: CalendarFixtures.date(16, 12))
        let blocks = Self.layout.blocks(weekStart: start, events: [next], proposals: [proposal])
        let drawn = blocks.first { $0.title == "Quick sync" }
        #expect(drawn?.height == Self.layout.minimumProposalHours)
        // The taller proposal now overlaps 11:30, so the two share the column.
        #expect(blocks.allSatisfy { $0.lanes == 2 })
        #expect(Set(blocks.map(\.lane)) == [0, 1])
    }

    /// Codex on #157: a proposal just before the grid's last hour rises to
    /// keep its room instead of getting a tiny or negative height.
    @Test func aProposalAtTheBottomEdgeKeepsItsRoom() throws {
        let start = Self.layout.weekStart(containing: CalendarFixtures.now)
        let late = ProposedEvent(title: "Late call", start: CalendarFixtures.date(16, 23, 59),
                                 end: CalendarFixtures.date(17, 0),
                                 source: ItemSource(kind: .mail, label: "Priya", reference: "<l@example.com>"))
        let drawn = try #require(Self.layout.blocks(weekStart: start, events: [], proposals: [late]).first)
        let span = Double(Self.layout.lastHour - Self.layout.firstHour)
        #expect(drawn.height == Self.layout.minimumProposalHours)
        #expect(drawn.top >= 0 && abs(drawn.top + drawn.height - span) < 1e-9)
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
        #expect(model.title == "September 2026")
        #expect(model.isToday(CalendarFixtures.date(15, 0)))
    }

    @Test @MainActor func deniedAccessPointsToSettings() async {
        let service = InMemoryCalendarService(calendarAccess: .notDetermined, grantOnRequest: false)
        let model = Self.model(service: service)
        await model.requestAccess()
        #expect(model.access == .blocked)
        await model.openCalendarSettings()
        #expect(await service.openedSettings == [.calendar])
    }

    /// The maintainer's loop: Continue under the reason, no macOS prompt, and
    /// the screen asked again. Now it says so, points to System Settings,
    /// and nothing asks again on its own.
    @Test @MainActor func aPromptMacOSNeverShowsEndsTheLoop() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events, calendarAccess: .notDetermined,
                                              promptShown: false)
        let model = Self.model(service: service)
        await model.load()
        await model.requestAccess()
        #expect(model.access == .promptNotShown)
        // Adding a proposal or a new event does not ask again.
        model.propose(CalendarFixtures.moveProposal)
        let proposal = try #require(model.proposals.first)
        await model.add(proposal)
        #expect(await model.createEvent(title: "Dentist", start: CalendarFixtures.date(16, 9),
                                        end: CalendarFixtures.date(16, 9, 30)) == false)
        // Coming back to the app re-reads without asking, and keeps the state.
        await model.recheckAccess()
        #expect(model.access == .promptNotShown)
        #expect(await service.requests == [.calendar])
        await model.openCalendarSettings()
        #expect(await service.openedSettings == [.calendar])
        // Turned on in System Settings: the next activation shows the week.
        await service.setCalendarAccess(.granted)
        await model.recheckAccess()
        #expect(model.access == .granted)
        #expect(model.events.count == 10)
        #expect(await service.requests == [.calendar])
    }

    /// The maintainer granted Full Access in the macOS prompt and the week
    /// stayed empty until a relaunch: a grant (or EventKit's change notice)
    /// now shows the events at once.
    @Test @MainActor func aGrantShowsTheWeekWithoutARelaunch() async {
        let granted = InMemoryCalendarService(events: CalendarFixtures.events, calendarAccess: .notDetermined)
        let model = Self.model(service: granted)
        await model.load()
        await model.requestAccess()
        #expect(model.access == .granted && model.events.count == 10)

        // Turned on outside the prompt: EventKit's change notice reloads.
        let later = InMemoryCalendarService(events: CalendarFixtures.events, calendarAccess: .notDetermined,
                                            promptShown: false)
        let waiting = Self.model(service: later)
        await waiting.requestAccess()
        #expect(waiting.access == .promptNotShown && waiting.events.isEmpty)
        await later.setCalendarAccess(.granted)
        await waiting.calendarChanged()
        #expect(waiting.access == .granted && waiting.events.count == 10)
    }

    /// Codex on #163: the live service passes the broker's answer on while
    /// EventKit still reads "not determined", and trusts a grant even if the
    /// status read lags it.
    @Test func theEventKitServicePassesTheBrokersAnswerOn() async {
        let undetermined: @Sendable (EKEntityType) -> EKAuthorizationStatus = { _ in .notDetermined }
        let noPrompt = EventKitService(broker: RecordingPermissionBroker([.calendar: .needsSystemSettings]),
                                       eventStore: EventStoreBox(), authorization: undetermined)
        #expect(await noPrompt.requestCalendarAccess() == .needsSystemSettings)
        let declined = EventKitService(broker: RecordingPermissionBroker([.calendar: .notDetermined]),
                                       eventStore: EventStoreBox(), authorization: undetermined)
        #expect(await declined.requestCalendarAccess() == .notDetermined)
        let lagging = EventKitService(broker: RecordingPermissionBroker([.calendar: .granted]),
                                      eventStore: EventStoreBox(), authorization: undetermined)
        #expect(await lagging.requestCalendarAccess() == .granted)
    }

    /// The iPhone reads calendar colors through `CGColor` (EventKit has no
    /// `NSColor` there); out-of-gamut channels clamp instead of trapping.
    @Test func calendarColorsReadThroughCGColor() throws {
        let orange = CGColor(srgbRed: 1, green: 0.5, blue: 0, alpha: 1)
        #expect(EventKitService.hex(orange) == 0xFF8000)
        let grey = CGColor(srgbRed: 0.2, green: 0.2, blue: 0.2, alpha: 0.5)
        #expect(EventKitService.hex(grey) == 0x333333)
        let extended = try #require(CGColorSpace(name: CGColorSpace.extendedSRGB))
        let wide = try #require(CGColor(colorSpace: extended, components: [1.4, -0.2, 0.5, 1]))
        #expect(EventKitService.hex(wide) == 0xFF0080)
    }

    /// The iPhone's broker opens the app's Settings page through
    /// `openSettings`, whatever the permission.
    @Test func settingsOpenThroughTheInjectedHook() async {
        let opened = Mutex<[AppPermission]>([])
        let broker = LivePermissionBroker(
            presentReason: { _ in false },
            system: .init(state: { _ in .needsSystemSettings }, prompt: { _ in },
                          openSettings: { permission in opened.withLock { $0.append(permission) } }))
        await broker.openSystemSettings(for: .camera)
        await broker.openSystemSettings(for: .reminders)
        #expect(opened.withLock { $0 } == [.camera, .reminders])
    }

    /// The broker surfaces a refused request: one reason, the error logged,
    /// and System Settings as the answer instead of "not determined".
    @Test func aRefusedSystemPromptAnswersSystemSettings() async {
        struct Refused: Error {}
        let reasons = Mutex(0)
        let broker = LivePermissionBroker(
            presentReason: { _ in reasons.withLock { $0 += 1 }; return true },
            system: .init(state: { _ in .notDetermined },
                          prompt: { _ in throw Refused() },
                          openURL: { _ in }))
        #expect(await broker.request(.calendar) == .needsSystemSettings)
        #expect(await broker.request(.reminders) == .needsSystemSettings)
        #expect(reasons.withLock { $0 } == 2)
        // A declined reason is not a failure: still undecided, nothing logged.
        let declined = LivePermissionBroker(
            presentReason: { _ in false },
            system: .init(state: { _ in .notDetermined }, prompt: { _ in throw Refused() }, openURL: { _ in }))
        #expect(await declined.request(.calendar) == .notDetermined)
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
        await model.step(1)
        #expect(model.weekStart == CalendarFixtures.date(21, 0))
        #expect(model.events.map(\.title) == ["Next week"])
        #expect(model.title == "September 2026")
        await model.step(1)
        // A week across two months names both.
        #expect(model.title.hasPrefix("September") && model.title.hasSuffix("October 2026"))
        await model.step(1)
        #expect(model.title == "October 2026")
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
