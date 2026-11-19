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
            try AppleSnapshot.render(AppleSnapshot.board(.calendar) { CalendarView(model: model, loadsOnAppear: false) },
                            name: "04-calendar-todos", colorScheme: scheme)
        }

        let notAsked = CalendarWeekModel(service: InMemoryCalendarService(calendarAccess: .notDetermined),
                                         store: PlannerStore(url: nil), defaults: defaults,
                                         now: { fixedNow })
        await notAsked.load()
        try AppleSnapshot.render(AppleSnapshot.board(.calendar) { CalendarView(model: notAsked, loadsOnAppear: false) },
                        name: "04-calendar-todos-access")
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
