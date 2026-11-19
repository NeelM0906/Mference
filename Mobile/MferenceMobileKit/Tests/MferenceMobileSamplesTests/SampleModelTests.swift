import Foundation
import MferenceMobileSamples
import MferenceMobileUI
import Testing

/// The sample screen models behave as the skeleton did before they moved out
/// of MferenceMobileUI.
@Suite("Sample calendar")
struct SampleCalendarModelTests {
    @Test func acceptingTheMailSuggestionMovesTheEvent() throws {
        let model = SampleCalendarModel()
        let suggestion = try #require(model.suggestion)
        model.acceptSuggestion()
        #expect(model.suggestion == nil)
        let moved = try #require(model.events.first { $0.id == suggestion.event.id })
        #expect(Calendar.current.component(.hour, from: moved.start) == 11)
    }

    @Test func todayBringsTheMonthGridBack() {
        let model = SampleCalendarModel()
        model.goToToday()
        model.monthOffset = 3
        model.goToToday()
        #expect(model.monthOffset == 0)
        model.monthOffset = -2
        model.select(day: .now, switchToDay: false)
        #expect(model.monthOffset == 0)
    }

    @Test func selectingFromTheMonthCanSwitchToTheDay() {
        let model = SampleCalendarModel()
        model.calendarMode = .month
        model.select(day: .now, switchToDay: true)
        #expect(model.calendarMode == .day)
        #expect(model.selectedDay == Calendar.current.startOfDay(for: .now))
    }

    @Test func todosToggleAndAdd() throws {
        let model = SampleCalendarModel()
        let open = model.openTodoCount
        let first = try #require(model.todos.first)
        model.toggle(first)
        #expect(model.openTodoCount == open - 1)
        model.addTodo("  Call the bank ")
        #expect(model.todos.first?.title == "Call the bank")
        model.addTodo("   ")
        #expect(model.todos.count == 6)
    }
}

@Suite("Sample scan")
struct SampleScanModelTests {
    @Test func scanAddsARow() {
        let model = SampleScanModel()
        #expect(model.scanPhase == .read)
        #expect(model.scanFields.contains { !$0.isSure })
        model.addScanRow()
        #expect(model.scanPhase == .added)
        model.retake()
        #expect(model.scanPhase == .viewfinder)
    }
}
