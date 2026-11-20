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
