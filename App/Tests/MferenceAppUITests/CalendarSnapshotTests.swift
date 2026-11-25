import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftData
import SwiftUI
import Testing

@Suite(.serialized) @MainActor struct CalendarSnapshotTests {
    @Test func calendar() async throws {
        let fixedNow = AppleUIFixtures.now
        let defaults = UserDefaults(suiteName: "mference-snapshots-\(UUID().uuidString)")!
        let store = PlannerStore(url: nil, initial: .init(todos: AppleUIFixtures.todos, proposals: [AppleUIFixtures.proposal]))
        let model = CalendarWeekModel(service: InMemoryCalendarService(events: AppleUIFixtures.events), store: store,
                                      defaults: defaults, now: { fixedNow }, syncsAutomatically: false)
        await model.load()
        #expect(model.title.contains("September"))
        for scheme in [ColorScheme.light, .dark] {
            try Snapshot.render(RootView(model: ShellModel(selection: .calendar, defaults: nil), chrome: .inline)
                .environment(model), name: "04-calendar-todos", colorScheme: scheme)
        }
        // Day, Month and Year (the switch at the top).
        for mode in [CalendarViewMode.day, .month, .year] {
            await model.show(mode)
            try Snapshot.render(RootView(model: ShellModel(selection: .calendar, defaults: nil), chrome: .inline)
                .environment(model), name: "04-calendar-\(mode.rawValue)", colorScheme: .dark)
        }
        await model.show(.week)

        let notAsked = CalendarWeekModel(service: InMemoryCalendarService(calendarAccess: .notDetermined),
                                         store: PlannerStore(url: nil), defaults: defaults,
                                         now: { fixedNow })
        await notAsked.load()
        try Snapshot.render(RootView(model: ShellModel(selection: .calendar, defaults: nil), chrome: .inline)
                .environment(notAsked), name: "04-calendar-todos-access")

        // Asked, but macOS showed no prompt.
        let noPrompt = CalendarWeekModel(service: InMemoryCalendarService(calendarAccess: .notDetermined,
                                                                          promptShown: false),
                                         store: PlannerStore(url: nil), defaults: defaults, now: { fixedNow })
        await noPrompt.requestAccess()
        #expect(noPrompt.access == .promptNotShown)
        try Snapshot.render(RootView(model: ShellModel(selection: .calendar, defaults: nil), chrome: .inline)
                .environment(noPrompt), name: "04-calendar-todos-no-prompt")
    }

    /// Ask about your week: idle chips, a free-time answer drawn on the
    /// grid, a busiest-day answer, Month without Ask, and an out-of-date answer.
    @Test func askAboutYourWeek() async throws {
        let fixedNow = AppleUIFixtures.now
        let defaults = UserDefaults(suiteName: "mference-snapshots-\(UUID().uuidString)")!
        let store = PlannerStore(url: nil, initial: .init(todos: AppleUIFixtures.todos, proposals: []))
        let service = InMemoryCalendarService(events: AppleUIFixtures.events)
        let model = CalendarWeekModel(service: service, store: store, defaults: defaults, now: { fixedNow },
                                      syncsAutomatically: false)
        await model.load()
        // What ships: chips answered in Swift, typed questions as one JSON
        // reply over the JSON schedule.
        let engine = ScriptedEngineService(status: EngineServiceStatus(phase: .ready, residentModel: .quick), replies: [
            .init(content: #"{"answer": "Your busiest day left is the one with the most booked time.", "refs": []}"#),
        ])
        let ask = WeekAskModel(calendar: model, answerer: WeekAnswerer(picker: WeekModelPicker(engine: engine)))
        func render(_ name: String, _ scheme: ColorScheme = .light) throws {
            try Snapshot.render(RootView(model: ShellModel(selection: .calendar, defaults: nil), chrome: .inline)
                .environment(model).environment(ask), name: name, colorScheme: scheme)
        }
        try render("04-calendar-ask-idle")
        await ask.ask(.freeFor(minutes: 120))
        #expect(!ask.highlightedOpenTimes.isEmpty)
        for scheme in [ColorScheme.light, .dark] { try render("04-calendar-ask-free", scheme) }
        await ask.ask("Which day is busiest?")
        #expect(ask.answers.count == 2)
        try render("04-calendar-ask-busiest")
        ask.clear()
        // Month has no Ask about your week.
        await model.show(.month)
        try render("04-calendar-month-no-ask")
        await model.show(.week)
        await ask.ask(.due)
        #expect(!ask.isIdle)
        _ = try await service.add(ProposedEvent(title: "Dentist", start: fixedNow.addingTimeInterval(26 * 3_600),
                                                end: fixedNow.addingTimeInterval(27 * 3_600),
                                                source: ItemSource(kind: .manual, label: "x")))
        await ask.calendarChanged()
        #expect(ask.isStale)
        try render("04-calendar-ask-stale")
        // Day view: the chips are about the selected day.
        ask.clear()
        await model.show(.day)
        try render("04-calendar-ask-day")
    }

    @Test func shellSlotBuildsWithoutServices() {
        let renderer = ImageRenderer(content: CalendarView().frame(width: 600, height: 600))
        #expect(renderer.nsImage != nil)
    }

    @Test func editingKeepsTheExactDurationUnlessTheLengthChanges() {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let twoMinutes = CalendarEvent(id: "short", title: "Stand", start: start, end: start.addingTimeInterval(125))
        let moved = start.addingTimeInterval(3_600)
        #expect(EventEditForm.end(for: twoMinutes, start: moved, minutes: nil) == moved.addingTimeInterval(125))
        #expect(EventEditForm.end(for: twoMinutes, start: moved, minutes: 30) == moved.addingTimeInterval(1_800))
    }
}
