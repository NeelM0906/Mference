import Foundation
@testable import MferenceAppCore
import Synchronization
import Testing

/// Ask about your week's lifecycle: in memory, one week, a thread of at most
/// three questions, out of date when the calendar changes, Block with Undo,
/// Stop.
@Suite struct WeekAskModelTests {
    static let fridayAnswer = "Friday 1:30 to 5 PM is open. [F8]"

    /// A resident model that answers every question with `reply`.
    static func engine(_ reply: String = fridayAnswer, delay: Duration = .zero) -> ScriptedEngineService {
        ScriptedEngineService(status: WeekAnswererTests.deepResident,
                              fallback: .init(content: reply, chunk: delay == .zero ? 6 : 2, delay: delay))
    }

    @MainActor static func setUp(engine: ScriptedEngineService? = nil,
                                 service: InMemoryCalendarService = InMemoryCalendarService(events: CalendarFixtures.events))
        async -> (calendar: CalendarWeekModel, ask: WeekAskModel) {
        let calendar = AppleCalendarTests.model(service: service)
        await calendar.load()
        let picker = engine.map { WeekModelPicker(engine: $0, appleModel: nil, useAppleModel: { false }) } ?? .none
        return (calendar, WeekAskModel(calendar: calendar, answerer: WeekAnswerer(picker: picker)))
    }

    @Test @MainActor func answersStayForTheWeekUpToThree() async {
        let (_, ask) = await Self.setUp(engine: Self.engine())
        #expect(ask.scopeLabel == "Sep 14 to 20")
        ask.draft = "When am I free for 2 hours?"
        #expect(ask.canAsk)
        await ask.ask()
        #expect(ask.draft.isEmpty)
        #expect(ask.answers.count == 1)
        #expect(ask.answers[0].source == .model(label: "Deep 27B"))
        #expect(ask.highlightedOpenTimes == [DateInterval(start: CalendarFixtures.date(18, 13, 30),
                                                          end: CalendarFixtures.date(18, 17))])
        await ask.ask("only mornings")
        await ask.ask("and on Friday?")
        #expect(ask.isFull)
        await ask.ask("What's due this week?")
        #expect(ask.answers.count == 3)
        #expect(ask.notice?.text.hasPrefix("Three questions is the limit") == true)
    }

    @Test @MainActor func followUpsShareOneThreadAndOneSchedule() async {
        let engine = Self.engine()
        let (_, ask) = await Self.setUp(engine: engine)
        await ask.ask("When am I free for 2 hours?")
        await ask.ask("only mornings")
        #expect(engine.requests.count == 2)
        let second = engine.requests[1].turns
        #expect(second.map(\.role) == [.system, .user, .assistant, .user])
        #expect(second[2].content == "Friday 1:30 to 5 PM is open.")
        #expect(second[1].content == engine.requests[0].turns[1].content)
    }

    /// A clock the test moves forward.
    final class Clock: Sendable {
        let time = Mutex(CalendarFixtures.now)
        var now: Date { time.withLock { $0 } }
        func set(_ date: Date) { time.withLock { $0 = date } }
    }

    @MainActor static func setUp(engine: ScriptedEngineService, clock: Clock) async -> WeekAskModel {
        let calendar = CalendarWeekModel(service: InMemoryCalendarService(events: CalendarFixtures.events),
                                         store: PlannerStore(url: nil), layout: AppleCalendarTests.layout,
                                         defaults: UserDefaults(suiteName: "mference-tests-\(UUID().uuidString)")!,
                                         now: { clock.now }, syncsAutomatically: false)
        await calendar.load()
        return WeekAskModel(calendar: calendar, answerer: WeekAnswerer(
            picker: WeekModelPicker(engine: engine, appleModel: nil, useAppleModel: { false })))
    }

    @Test @MainActor func aFollowUpTheNextDayReadsANewSchedule() async {
        let engine = Self.engine()
        let clock = Clock()
        let ask = await Self.setUp(engine: engine, clock: clock)
        await ask.ask("What's on today?")
        clock.set(CalendarFixtures.date(15, 8, 5))
        await ask.ask("and tomorrow?")
        // Minutes later the schedule says the same: it is sent word for word.
        #expect(engine.requests[1].turns[1].content == engine.requests[0].turns[1].content)
        clock.set(CalendarFixtures.date(16, 9))
        await ask.ask("What's on today?")
        let third = engine.requests[2].turns
        #expect(third.map(\.role) == [.system, .user, .assistant, .user, .assistant, .user])
        let schedule = third[1].content ?? ""
        #expect(schedule.contains("Now: Wednesday Sep 16"))
        #expect(!schedule.contains("Coffee with Tom"))
    }

    @Test @MainActor func aFollowUpAfterOpenTimeHasPassedReadsANewSchedule() async {
        let engine = Self.engine()
        let clock = Clock()
        let ask = await Self.setUp(engine: engine, clock: clock)
        await ask.ask("When am I free today?")
        clock.set(CalendarFixtures.date(15, 14))
        await ask.ask("anything after that?")
        let first = engine.requests[0].turns[1].content ?? ""
        let second = engine.requests[1].turns[1].content ?? ""
        #expect(first.contains("10 AM to 12 PM"))
        #expect(!second.contains("10 AM to 12 PM"))
        #expect(second.contains("Now: Tuesday Sep 15, 2 PM"))
    }

    @Test @MainActor func anotherWeekClearsTheAnswersButAnotherViewDoesNot() async {
        let (calendar, ask) = await Self.setUp(engine: Self.engine())
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

    @Test @MainActor func dayViewSendsTheWholeWeekAndTheDayInFocus() async {
        let engine = Self.engine()
        let (calendar, ask) = await Self.setUp(engine: engine)
        await calendar.show(.day)
        await calendar.select(CalendarFixtures.date(16, 0))
        #expect(!calendar.events.contains { $0.id == "contract" })
        await ask.ask("What's on today?")
        let schedule = engine.requests.first?.turns[1].content ?? ""
        #expect(schedule.contains("Contract call, 1 to 1:30 PM [E8]"))
        #expect(schedule.contains("Looking at: Wednesday Sep 16 (tomorrow)."))
    }

    @Test @MainActor func aCalendarChangeMarksTheAnswerOutOfDateAndStartsANewThread() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let engine = Self.engine()
        let (_, ask) = await Self.setUp(engine: engine, service: service)
        await ask.ask("When am I free for 2 hours?")
        await ask.calendarChanged()
        #expect(!ask.isStale)
        _ = try await service.add(ProposedEvent(title: "Dentist", start: CalendarFixtures.date(16, 15),
                                                end: CalendarFixtures.date(16, 16), source: ItemSource(kind: .manual, label: "x")))
        await ask.calendarChanged()
        #expect(ask.isStale)
        #expect(ask.current == nil)
        #expect(ask.highlightedOpenTimes.isEmpty)
        // Asking again starts a new thread with the new schedule.
        await ask.ask("When am I free for 2 hours?")
        #expect(!ask.isStale)
        #expect(ask.answers.count == 1)
        let turns = engine.requests[1].turns
        #expect(turns.map(\.role) == [.system, .user])
        #expect(turns[1].content?.contains("Dentist, 3 to 4 PM") == true)
    }

    @Test @MainActor func blockAddsTheEventItsOwnChangeIsNotStaleAndUndoRemovesIt() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (calendar, ask) = await Self.setUp(engine: Self.engine(), service: service)
        await ask.ask("When am I free for 2 hours?")
        let friday = try #require(ask.highlightedOpenTimes.first)
        #expect(ask.blockInterval(for: friday) == DateInterval(start: CalendarFixtures.date(18, 13, 30),
                                                              end: CalendarFixtures.date(18, 15, 30)))
        await ask.block(friday)
        #expect(ask.notice == WeekAskNotice(text: "Added Focus block to Calendar, Fri 1:30 to 3:30 PM.", canUndo: true))
        #expect(ask.blocked[friday] != nil)
        #expect(calendar.events.contains { $0.title == WeekAskModel.blockTitle })
        await ask.calendarChanged()
        #expect(!ask.isStale)
        await ask.undoBlock()
        #expect(ask.blocked.isEmpty)
        #expect(ask.notice == nil)
        let remaining = try await service.events(from: CalendarFixtures.date(14, 0), to: CalendarFixtures.date(21, 0))
        #expect(!remaining.contains { $0.title == WeekAskModel.blockTitle })
    }

    @Test @MainActor func blockUsesTheLengthAskedForElseAnHour() async throws {
        let (_, ask) = await Self.setUp(engine: Self.engine())
        await ask.ask("Where can I fit a 1 hour 30 minutes review?")
        let thursday = try #require(ask.highlightedOpenTimes.first)
        #expect(ask.blockInterval(for: thursday).duration == 90 * 60)
        ask.clear()
        await ask.ask("Anything open Thursday?")
        let open = try #require(ask.highlightedOpenTimes.first)
        #expect(ask.blockInterval(for: open).duration == 60 * 60)
    }

    @Test @MainActor func stopPutsTheQuestionBack() async throws {
        let (_, ask) = await Self.setUp(engine: Self.engine(delay: .milliseconds(50)))
        let running = Task { await ask.ask("When am I free for 2 hours?") }
        for _ in 0..<200 where ask.pending?.step != .answering { try await Task.sleep(for: .milliseconds(10)) }
        #expect(ask.pending?.question == "When am I free for 2 hours?")
        ask.stop()
        await running.value
        #expect(ask.pending == nil)
        #expect(ask.answers.isEmpty)
        #expect(ask.draft == "When am I free for 2 hours?")
    }

    @Test @MainActor func leavingTheWeekDropsTheAnswerBeingWorkedOut() async throws {
        let (calendar, ask) = await Self.setUp(engine: Self.engine(delay: .milliseconds(50)))
        let running = Task { await ask.ask("When am I free for 2 hours?") }
        for _ in 0..<200 where ask.pending == nil { try await Task.sleep(for: .milliseconds(10)) }
        await calendar.step(1)
        ask.scopeChanged()
        await running.value
        #expect(ask.pending == nil)
        #expect(ask.answers.isEmpty)
        #expect(ask.scopeLabel == "Sep 21 to 27")
    }

    @Test @MainActor func aChangeWhileAnsweringMarksTheNewAnswerOutOfDate() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (_, ask) = await Self.setUp(engine: Self.engine(delay: .milliseconds(30)), service: service)
        let running = Task { await ask.ask("When am I free for 2 hours?") }
        for _ in 0..<200 where ask.pending?.step != .answering { try await Task.sleep(for: .milliseconds(10)) }
        _ = try await service.add(ProposedEvent(title: "Dentist", start: CalendarFixtures.date(16, 15),
                                                end: CalendarFixtures.date(16, 16), source: ItemSource(kind: .manual, label: "x")))
        await running.value
        #expect(ask.answers.count == 1)
        #expect(ask.isStale)
        #expect(ask.highlightedOpenTimes.isEmpty)
    }

    @Test @MainActor func aCheckANewerAnswerOvertookLeavesThatAnswerCurrent() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (_, ask) = await Self.setUp(engine: Self.engine(), service: service)
        await ask.ask("When am I free for 2 hours?")
        _ = try await service.add(ProposedEvent(title: "Dentist", start: CalendarFixtures.date(16, 15),
                                                end: CalendarFixtures.date(16, 16), source: ItemSource(kind: .manual, label: "x")))
        // The change check reads Calendar slowly; a new question answers
        // from the changed calendar before it finishes.
        await service.delayNextRead(by: .milliseconds(200))
        let check = Task { await ask.calendarChanged() }
        for _ in 0..<200 where await service.isNextReadDelayed { try await Task.sleep(for: .milliseconds(5)) }
        await ask.ask("When am I free for 2 hours?")
        #expect(ask.answers.count == 1)
        await check.value
        #expect(!ask.isStale)
        #expect(!ask.highlightedOpenTimes.isEmpty)
    }

    @Test @MainActor func movingABlockedEventMakesAnswersStale() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (_, ask) = await Self.setUp(engine: Self.engine(), service: service)
        await ask.ask("When am I free for 2 hours?")
        let slot = try #require(ask.highlightedOpenTimes.first)
        await ask.block(slot)
        await ask.calendarChanged()
        #expect(!ask.isStale)
        let added = try #require(ask.blocked[slot])
        _ = try await service.update(added.reference, title: added.title, start: CalendarFixtures.date(17, 13),
                                     end: CalendarFixtures.date(17, 15))
        await ask.calendarChanged()
        #expect(ask.isStale)
    }

    @Test @MainActor func deletingABlockALaterAnswerReadMakesItStale() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (_, ask) = await Self.setUp(engine: Self.engine(), service: service)
        await ask.ask("When am I free for 2 hours?")
        let slot = try #require(ask.highlightedOpenTimes.first)
        await ask.block(slot)
        let added = try #require(ask.blocked[slot])
        ask.clear()
        await ask.ask("When is the focus block?")
        try await service.remove(added.reference)
        await ask.calendarChanged()
        #expect(ask.isStale)
    }

    @Test @MainActor func aQuestionIsPendingBeforeCalendarIsRead() async {
        let (_, ask) = await Self.setUp(engine: Self.engine())
        let running = Task { await ask.ask("When am I free for 2 hours?") }
        await Task.yield()
        ask.clear()
        await running.value
        #expect(ask.answers.isEmpty)
        #expect(ask.pending == nil)
    }

    @Test @MainActor func twoQuickBlockClicksAddOneEvent() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (_, ask) = await Self.setUp(engine: Self.engine(), service: service)
        await ask.ask("When am I free for 2 hours?")
        let slot = try #require(ask.highlightedOpenTimes.first)
        async let first: Void = ask.block(slot)
        async let second: Void = ask.block(slot)
        _ = await (first, second)
        let events = try await service.events(from: CalendarFixtures.date(14, 0), to: CalendarFixtures.date(21, 0))
        #expect(events.filter { $0.title == WeekAskModel.blockTitle }.count == 1)
        #expect(ask.blocking.isEmpty)
    }

    @Test @MainActor func aToDoOutsideTheWeekDoesNotMakeAnswersStale() async {
        let (calendar, ask) = await Self.setUp(engine: Self.engine())
        await ask.ask("When am I free for 2 hours?")
        calendar.addTodo(text: "Renew the lease", due: CalendarFixtures.date(28, 0), dueChosen: true)
        await ask.calendarChanged()
        #expect(!ask.isStale)
        calendar.addTodo(text: "Call the bank", due: CalendarFixtures.date(16, 0), dueChosen: true)
        await ask.calendarChanged()
        #expect(ask.isStale)
    }

    @Test @MainActor func withoutAModelTheWeekIsStillShown() async {
        let (_, ask) = await Self.setUp()
        await ask.ask("What's on this week?")
        #expect(ask.answers.last?.source == .fixed(.noModel))
        #expect(ask.highlightedEventIDs.contains("contract"))
    }

    @Test @MainActor func withoutCalendarAccessNothingIsAsked() async {
        let (_, ask) = await Self.setUp(service: InMemoryCalendarService(events: [], calendarAccess: .denied))
        await ask.ask("When am I free for 2 hours?")
        #expect(ask.answers.isEmpty)
        #expect(ask.notice?.text == "Allow Calendar access to ask about your week.")
    }

    @Test @MainActor func continueInChatStartsAPrivateChatWithTheThread() async throws {
        let (_, ask) = await Self.setUp(engine: Self.engine())
        await ask.ask("When am I free for 2 hours?")
        await ask.ask("only mornings")
        #expect(ask.chatTitle == "Week of Sep 14 to 20")
        #expect(ask.chatContext.hasPrefix("My calendar for Sep 14 to 20, as Mference read it:\nNow: Tuesday Sep 15"))
        #expect(!ask.chatContext.contains("[F"))

        let container = try AppModelContainer.inMemory()
        let store = ConversationStore(container: container)
        let chat = ChatViewModel(service: ScriptedEngineService(), store: store)
        chat.continueChat(title: ask.chatTitle, context: ask.chatContext, turns: ask.chatTurns)
        let conversation = try #require(try store.conversations().first)
        #expect(conversation.isPrivate)
        #expect(conversation.title == "Week of Sep 14 to 20")
        let stored = conversation.orderedMessages
        #expect(stored.map(\.role) == [.user, .assistant, .user, .assistant])
        #expect(stored[0].content.hasPrefix("My calendar for Sep 14 to 20"))
        #expect(stored[0].content.hasSuffix("When am I free for 2 hours?"))
        #expect(stored[1].content == "Friday 1:30 to 5 PM is open.")
        #expect(chat.isPrivate)
        #expect(ask.answers.count == 2)
    }

    @Test @MainActor func clearEmptiesEverything() async {
        let (_, ask) = await Self.setUp(engine: Self.engine())
        await ask.ask("When am I free for 2 hours?")
        ask.clear()
        #expect(ask.answers.isEmpty)
        #expect(ask.highlightedOpenTimes.isEmpty)
        #expect(!ask.isFull)
    }
}
