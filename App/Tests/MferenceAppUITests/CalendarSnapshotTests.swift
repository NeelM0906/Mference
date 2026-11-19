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
                                      defaults: defaults, now: { fixedNow })
        await model.load()
        #expect(model.weekTitle == "September 14 to 18")
        for scheme in [ColorScheme.light, .dark] {
            try Snapshot.render(RootView(model: ShellModel(selection: .calendar, defaults: nil), chrome: .inline)
                .environment(model), name: "04-calendar-todos", colorScheme: scheme)
        }

        let notAsked = CalendarWeekModel(service: InMemoryCalendarService(calendarAccess: .notDetermined),
                                         store: PlannerStore(url: nil), defaults: defaults,
                                         now: { fixedNow })
        await notAsked.load()
        try Snapshot.render(RootView(model: ShellModel(selection: .calendar, defaults: nil), chrome: .inline)
                .environment(notAsked), name: "04-calendar-todos-access")
    }

    @Test func shellSlotBuildsWithoutServices() {
        let renderer = ImageRenderer(content: CalendarView().frame(width: 600, height: 600))
        #expect(renderer.nsImage != nil)
    }
}
