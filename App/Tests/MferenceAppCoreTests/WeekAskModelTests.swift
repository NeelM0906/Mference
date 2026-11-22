import Foundation
@testable import MferenceAppCore
import Testing

/// Ask about your week's lifecycle: in memory, one week, three questions,
/// out of date when the calendar changes, Block with Undo, Stop.
@Suite struct WeekAskModelTests {
    @MainActor static func setUp(engine: ScriptedEngineService? = nil,
                                 service: InMemoryCalendarService = InMemoryCalendarService(events: CalendarFixtures.events))
        async -> (calendar: CalendarWeekModel, ask: WeekAskModel) {
        let calendar = AppleCalendarTests.model(service: service)
        await calendar.load()
        let picker = engine.map { WeekModelPicker(engine: $0, appleModel: nil, useAppleModel: { false }) } ?? .none
        return (calendar, WeekAskModel(calendar: calendar, answerer: WeekAnswerer(picker: picker)))
    }

    @Test @MainActor func answersStayForTheWeekUpToThree() async {
        let (_, ask) = await Self.setUp()
        #expect(ask.scopeLabel == "Sep 14 to 20")
        ask.draft = "When am I free for 2 hours?"
        #expect(ask.canAsk)
        await ask.ask()
        #expect(ask.draft.isEmpty)
        #expect(ask.answers.count == 1)
        #expect(ask.answers[0].source == .fixed(.noModel))
        #expect(ask.highlightedOpenTimes.count == 8)
        await ask.ask("What's my busiest day?")
        #expect(ask.highlightedDays == [CalendarFixtures.date(18, 0)])
        #expect(ask.highlightedOpenTimes.isEmpty)
        await ask.ask("When is the contract call?")
        #expect(ask.highlightedEventIDs == ["contract"])
        #expect(ask.isFull)
        await ask.ask("What's due this week?")
        #expect(ask.answers.count == 3)
        #expect(ask.notice?.text.hasPrefix("Three questions is the limit") == true)
    }

    @Test @MainActor func anotherWeekClearsTheAnswersButAnotherViewDoesNot() async {
        let (calendar, ask) = await Self.setUp()
        await ask.ask("What's my busiest day?")
        await calendar.show(.day)
        ask.scopeChanged()
        #expect(ask.answers.count == 1)
        await calendar.show(.month)
        ask.scopeChanged()
        #expect(ask.answers.count == 1)
        await calendar.show(.week)
        await calendar.step(1)
        ask.scopeChanged()
        #expect(ask.answers.isEmpty)
        #expect(ask.scopeLabel == "Sep 21 to 27")
    }

    @Test @MainActor func dayViewStillAsksAboutTheWholeWeek() async {
        let (calendar, ask) = await Self.setUp()
        await calendar.show(.day)
        #expect(!calendar.events.contains { $0.id == "contract" })
        await ask.ask("What's my busiest day?")
        #expect(ask.answers.last?.sentence == "Friday is your busiest day left, with 2 h 30 min booked.")
    }

    @Test @MainActor func aCalendarChangeMarksTheAnswerOutOfDate() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (_, ask) = await Self.setUp(service: service)
        await ask.ask("When am I free for 2 hours?")
        await ask.calendarChanged()
        #expect(!ask.isStale)
        _ = try await service.add(ProposedEvent(title: "Dentist", start: CalendarFixtures.date(16, 15),
                                                end: CalendarFixtures.date(16, 16), source: ItemSource(kind: .manual, label: "x")))
        await ask.calendarChanged()
        #expect(ask.isStale)
        #expect(ask.current == nil)
        #expect(ask.highlightedOpenTimes.isEmpty)
        // Asking again starts over with the new calendar.
        await ask.ask("When am I free for 2 hours?")
        #expect(!ask.isStale)
        #expect(ask.answers.count == 1)
        #expect(ask.highlightedOpenTimes.count == 7)
    }

    @Test @MainActor func blockAddsTheEventItsOwnChangeIsNotStaleAndUndoRemovesIt() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (calendar, ask) = await Self.setUp(service: service)
        await ask.ask("When am I free for 2 hours?")
        let thursday = try #require(ask.highlightedOpenTimes.first { $0.start == CalendarFixtures.date(17, 12) })
        #expect(ask.blockInterval(for: thursday) == DateInterval(start: CalendarFixtures.date(17, 12),
                                                                end: CalendarFixtures.date(17, 14)))
        await ask.block(thursday)
        #expect(ask.notice == WeekAskNotice(text: "Added Focus block to Calendar, Thu 12 to 2 PM.", canUndo: true))
        #expect(ask.blocked[thursday] != nil)
        #expect(calendar.events.contains { $0.title == WeekAskModel.blockTitle })
        await ask.calendarChanged()
        #expect(!ask.isStale)
        await ask.undoBlock()
        #expect(ask.blocked.isEmpty)
        #expect(ask.notice == nil)
        let remaining = try await service.events(from: CalendarFixtures.date(14, 0), to: CalendarFixtures.date(21, 0))
        #expect(!remaining.contains { $0.title == WeekAskModel.blockTitle })
    }

    @Test @MainActor func stopPutsTheQuestionBack() async throws {
        let engine = ScriptedEngineService(status: WeekAnswererTests.deepResident, replies: [
            .init(content: "Today, 10 AM to 12 PM is first, and there are eight. [F1]", chunk: 2, delay: .milliseconds(50)),
        ])
        let (_, ask) = await Self.setUp(engine: engine)
        let running = Task { await ask.ask("When am I free for 2 hours?") }
        for _ in 0..<200 where ask.pending?.step != .wording { try await Task.sleep(for: .milliseconds(10)) }
        #expect(ask.pending?.question == "When am I free for 2 hours?")
        ask.stop()
        await running.value
        #expect(ask.pending == nil)
        #expect(ask.answers.isEmpty)
        #expect(ask.draft == "When am I free for 2 hours?")
    }

    @Test @MainActor func aFailedAnswerSaysWhyAndKeepsTheQuestion() async {
        let engine = ScriptedEngineService(status: WeekAnswererTests.deepResident,
                                           replies: [.init(content: "", error: .busy)])
        let (_, ask) = await Self.setUp(engine: engine)
        await ask.ask("When am I free for 2 hours?")
        #expect(ask.answers.isEmpty)
        #expect(ask.notice?.text == EngineServiceError.busy.userMessage)
        #expect(ask.draft == "When am I free for 2 hours?")
    }

    @Test @MainActor func withoutCalendarAccessNothingIsAsked() async {
        let (_, ask) = await Self.setUp(service: InMemoryCalendarService(events: [], calendarAccess: .denied))
        await ask.ask("When am I free for 2 hours?")
        #expect(ask.answers.isEmpty)
        #expect(ask.notice?.text == "Allow Calendar access to ask about your week.")
    }

    @Test @MainActor func clearEmptiesEverything() async {
        let (_, ask) = await Self.setUp()
        await ask.ask("When am I free for 2 hours?")
        ask.clear()
        #expect(ask.answers.isEmpty)
        #expect(ask.highlightedOpenTimes.isEmpty)
        #expect(!ask.isFull)
    }
}
