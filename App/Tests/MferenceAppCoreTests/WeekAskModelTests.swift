import Foundation
@testable import MferenceAppCore
import Synchronization
import Testing

/// Ask about your week's lifecycle: in memory, one week, a thread of at most
/// three questions, out of date when the calendar changes, Block with Undo,
/// Stop.
@Suite struct WeekAskModelTests {
    /// Friday's openings: 11 AM to 1 PM and 1:30 to 5 PM.
    static let fridayAnswer = #"{"free_query": {"days": ["Friday"]}}"#
    static let fridayOpenings = [DateInterval(start: CalendarFixtures.date(18, 11), end: CalendarFixtures.date(18, 13)),
                                 DateInterval(start: CalendarFixtures.date(18, 13, 30), end: CalendarFixtures.date(18, 17))]

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
        let picker = engine.map { WeekModelPicker(engine: $0) } ?? .none
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
        #expect(ask.answers[0].source == .calendar(askedBy: "Deep 27B"))
        #expect(ask.highlightedOpenTimes == Self.fridayOpenings)
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
        #expect(second[2].content == #"{"free_query": {"days": ["2026-09-18"]}}"#)
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
            picker: WeekModelPicker(engine: engine)))
    }

    @Test @MainActor func aFollowUpTheNextDayReadsANewSchedule() async throws {
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
        #expect(schedule.contains(#""now": "Wednesday 2026-09-16, 9 AM""#))
        // Tuesday is a past day now.
        let past = try #require(schedule.range(of: #""past_days""#))
        let coffee = try #require(schedule.range(of: "Coffee with Tom"))
        #expect(coffee.lowerBound > past.lowerBound)
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
        #expect(first.contains(#"{"from": "10 AM", "to": "12 PM""#))
        #expect(!second.contains(#"{"from": "10 AM", "to": "12 PM""#))
        #expect(second.contains(#""now": "Tuesday 2026-09-15, 2 PM""#))
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
        #expect(schedule.contains(#""id": "E8", "title": "Contract call""#))
        #expect(schedule.contains(#""looking_at": "Wednesday 2026-09-16""#))
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
        #expect(turns[1].content?.contains(#""title": "Dentist", "from": "3 PM", "to": "4 PM""#) == true)
    }

    @Test @MainActor func blockAddsTheEventItsOwnChangeIsNotStaleAndUndoRemovesIt() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (calendar, ask) = await Self.setUp(engine: Self.engine(), service: service)
        await ask.ask("When am I free for 2 hours?")
        let friday = try #require(ask.highlightedOpenTimes.last)
        #expect(ask.blockInterval(for: friday) == DateInterval(start: CalendarFixtures.date(18, 13, 30),
                                                              end: CalendarFixtures.date(18, 14, 30)))
        await ask.block(friday)
        #expect(ask.notice == WeekAskNotice(text: "Added Focus block to Calendar, Fri 1:30 to 2:30 PM.", canUndo: true))
        #expect(ask.blocked[friday] != nil)
        #expect(calendar.events.contains { $0.title == WeekAskModel.blockTitle })
        await ask.calendarChanged()
        #expect(!ask.isStale)
        await ask.undoBlock()
        #expect(ask.blocked.isEmpty)
        #expect(ask.ownEvents.isEmpty)
        #expect(ask.notice == nil)
        let remaining = try await service.events(from: CalendarFixtures.date(14, 0), to: CalendarFixtures.date(21, 0))
        #expect(!remaining.contains { $0.title == WeekAskModel.blockTitle })
    }

    @Test @MainActor func blockUsesTheMinutesAskedForElseAnHour() async throws {
        let (_, ask) = await Self.setUp(engine: Self.engine(#"{"free_query": {"min_minutes": 90, "days": ["Friday"]}}"#))
        await ask.ask("Where can I fit a 90 minute review on Friday?")
        let friday = try #require(ask.highlightedOpenTimes.last)
        #expect(ask.blockInterval(for: friday).duration == 90 * 60)
        // Never more than the open block.
        #expect(ask.blockInterval(for: DateInterval(start: friday.start, duration: 45 * 60)).duration == 45 * 60)
        ask.clear()
        await ask.ask(WeekChip.freeOn(CalendarFixtures.date(18, 0)))
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

    /// Waits until the read `holdNextRead()` holds has started.
    static func untilReadHeld(_ service: InMemoryCalendarService) async throws {
        for _ in 0..<2_000 where !(await service.isReadHeld) { try await Task.sleep(for: .milliseconds(5)) }
        try #require(await service.isReadHeld, "the held read never started")
    }

    @Test @MainActor func aCheckANewerAnswerOvertookLeavesThatAnswerCurrent() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (_, ask) = await Self.setUp(engine: Self.engine(), service: service)
        await ask.ask("When am I free for 2 hours?")
        _ = try await service.add(ProposedEvent(title: "Dentist", start: CalendarFixtures.date(16, 15),
                                                end: CalendarFixtures.date(16, 16), source: ItemSource(kind: .manual, label: "x")))
        // The change check reads Calendar slowly; a new question answers
        // from the changed calendar before it finishes.
        await service.holdNextRead()
        let check = Task { await ask.calendarChanged() }
        try await Self.untilReadHeld(service)
        await ask.ask("When am I free for 2 hours?")
        #expect(ask.answers.count == 1)
        await service.releaseHeldRead()
        await check.value
        #expect(!ask.isStale)
        #expect(!ask.highlightedOpenTimes.isEmpty)
    }

    @Test @MainActor func aChangeNoticeDuringBlockIsCheckedOnceTheBlockIsKnown() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (_, ask) = await Self.setUp(engine: Self.engine(), service: service)
        await ask.ask("When am I free for 2 hours?")
        let slot = try #require(ask.highlightedOpenTimes.first)
        // The grid reload after Block's save is slow; Calendar's change
        // notice for that save arrives before Block has returned.
        await service.holdNextRead()
        let block = Task { await ask.block(slot) }
        try await Self.untilReadHeld(service)
        await ask.calendarChanged()
        await service.releaseHeldRead()
        await block.value
        #expect(ask.blocked[slot] != nil)
        #expect(!ask.isStale)
        // A real change during a Block still counts once the Block is in.
        await service.holdNextRead()
        let second = DateInterval(start: CalendarFixtures.date(18, 9), end: CalendarFixtures.date(18, 10))
        let again = Task { await ask.block(second) }
        try await Self.untilReadHeld(service)
        _ = try await service.add(ProposedEvent(title: "Dentist", start: CalendarFixtures.date(16, 15),
                                                end: CalendarFixtures.date(16, 16), source: ItemSource(kind: .manual, label: "x")))
        await ask.calendarChanged()
        #expect(!ask.isStale)
        await service.releaseHeldRead()
        await again.value
        #expect(ask.isStale)
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

    @Test @MainActor func withoutAModelChipsAnswerAndTypedQuestionsGoBack() async {
        let (_, ask) = await Self.setUp()
        await ask.ask("What's on this week?")
        #expect(ask.answers.isEmpty)
        #expect(ask.notice?.text == "No on-device model can answer right now.")
        #expect(ask.draft == "What's on this week?")
        await ask.ask(.busiest)
        #expect(ask.answers.count == 1)
        #expect(ask.answers[0].source == .calendar(askedBy: nil))
        #expect(ask.highlightedEventIDs == ["holiday", "launch"])
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
        #expect(ask.chatContext.hasPrefix("The person's calendar for Sep 14 to 20, as Mference read it"))
        #expect(ask.chatContext.contains("<USER WEEK>\n{\n  \"now\": \"Tuesday 2026-09-15, 8 AM\""))

        let container = try AppModelContainer.inMemory()
        let store = ConversationStore(container: container)
        let chatEngine = ScriptedEngineService(status: WeekAnswererTests.deepResident,
                                               fallback: .init(content: "Friday is open after 1:30 PM."))
        let chat = ChatViewModel(service: chatEngine, store: store)
        chat.continueChat(title: ask.chatTitle, context: ask.chatContext, turns: ask.chatTurns)
        let conversation = try #require(try store.conversations().first)
        #expect(conversation.isPrivate)
        #expect(conversation.title == "Week of Sep 14 to 20")
        let stored = conversation.orderedMessages
        // The schedule is hidden context: stored, sent, never shown.
        #expect(stored.map(\.role) == [.system, .user, .assistant, .user, .assistant])
        #expect(stored[0].content == ask.chatContext)
        #expect(stored[1].content == "When am I free for 2 hours?")
        #expect(stored[2].content == "2 openings on Friday: 11 AM to 1 PM and 1:30 to 5 PM.")
        #expect(chat.messages.map(\.role) == [.user, .assistant, .user, .assistant])
        #expect(chat.isPrivate)
        #expect(ask.answers.count == 2)
        // The chat model reads the schedule as one system turn, first.
        chat.draft = "and on Friday?"
        chat.send()
        await chat.waitUntilIdle()
        let sent = try #require(chatEngine.requests.last?.turns)
        #expect(sent.filter { $0.role == .system }.count == 1)
        #expect(sent[0].role == .system)
        #expect(sent[0].content?.contains(ask.chatContext) == true)
        #expect(sent.dropFirst().map(\.role) == [.user, .assistant, .user, .assistant, .user])
    }

    @Test @MainActor func clearEmptiesEverything() async {
        let (_, ask) = await Self.setUp(engine: Self.engine())
        await ask.ask("When am I free for 2 hours?")
        ask.clear()
        #expect(ask.answers.isEmpty)
        #expect(ask.highlightedOpenTimes.isEmpty)
        #expect(!ask.isFull)
    }

    // MARK: v1

    @Test @MainActor func chipsAnswerInSwiftAndCountAsQuestions() async {
        let engine = Self.engine()
        let (_, ask) = await Self.setUp(engine: engine)
        #expect(ask.chips == [.freeFor(minutes: 120), .busiest, .due, .overlaps])
        await ask.ask(.freeFor(minutes: 120))
        await ask.ask(.due)
        #expect(engine.requests.isEmpty)
        #expect(ask.answers.map(\.question) == ["When am I free for 2 hours?", "What's due this week?"])
        #expect(ask.answers[0].openTimes.count == 6)
        // A typed follow-up builds on the chip: its query is replayed.
        await ask.ask("only mornings")
        let turns = engine.requests[0].turns
        #expect(turns[2].content == #"{"free_query": {"min_minutes": 120}}"#)
        #expect(ask.isFull)
    }

    @Test @MainActor func aFailureNeverUsesUpAQuestion() async {
        for (reply, message) in [(ScriptedEngineService.Reply(content: " "), "The model couldn't answer. Try again."),
                                 (.init(content: "", error: .busy), EngineServiceError.busy.userMessage),
                                 (.init(content: "", error: .requestTooLarge("Shorten it.")),
                                  EngineServiceError.requestTooLarge("Shorten it.").userMessage)] {
            let engine = ScriptedEngineService(status: WeekAnswererTests.deepResident, replies: [reply])
            let (_, ask) = await Self.setUp(engine: engine)
            await ask.ask("When am I free for 2 hours?")
            #expect(ask.answers.isEmpty)
            #expect(ask.notice?.text == message)
            #expect(ask.draft == "When am I free for 2 hours?")
            #expect(ask.pending == nil)
        }
    }

    @Test @MainActor func aProseReplyIsShownAndCounts() async {
        let (_, ask) = await Self.setUp(engine: Self.engine("Friday afternoon looks open."))
        await ask.ask("anything friday?")
        #expect(ask.answers.count == 1)
        #expect(ask.answers[0].sentence == "Friday afternoon looks open.")
        #expect(ask.answers[0].record == .prose)
        #expect(ask.highlightedOpenTimes.isEmpty)
    }

    @Test @MainActor func afterABlockTheThreadReadsAFreshScheduleAndSaysWhatChanged() async throws {
        let engine = Self.engine()
        let (_, ask) = await Self.setUp(engine: engine)
        await ask.ask("When am I free on Friday?")
        let slot = try #require(ask.highlightedOpenTimes.last)
        await ask.block(slot)
        await ask.ask("and after that?")
        let turns = engine.requests[1].turns
        #expect(turns.map(\.role) == [.system, .user, .assistant, .user])
        #expect(turns[1].content != engine.requests[0].turns[1].content)
        #expect(turns[1].content?.contains(#""title": "Focus block""#) == true)
        #expect(turns[3].content == "Since your last question: Focus block added Friday 1:30 to 2:30 PM.\n\n"
            + "<USER QUERY>\nand after that?\n</USER QUERY>")
        // Said once: the next question has nothing new to report.
        await ask.ask("thanks, anything else?")
        #expect(engine.requests[2].turns.last?.content?.contains("Since your last question") == false)
    }

    @Test @MainActor func anUndoneBlockIsNotReported() async throws {
        let engine = Self.engine()
        let (_, ask) = await Self.setUp(engine: engine)
        await ask.ask("When am I free on Friday?")
        let slot = try #require(ask.highlightedOpenTimes.last)
        await ask.block(slot)
        await ask.undoBlock()
        await ask.ask("and after that?")
        #expect(engine.requests[1].turns.last?.content?.contains("Since your last question") == false)
    }

    @Test @MainActor func clearForgetsTheLastBlockSoUndoDoesNothing() async throws {
        let service = InMemoryCalendarService(events: CalendarFixtures.events)
        let (_, ask) = await Self.setUp(engine: Self.engine(), service: service)
        await ask.ask("When am I free on Friday?")
        let slot = try #require(ask.highlightedOpenTimes.last)
        await ask.block(slot)
        ask.clear()
        await ask.undoBlock()
        let events = try await service.events(from: CalendarFixtures.date(14, 0), to: CalendarFixtures.date(21, 0))
        #expect(events.contains { $0.title == WeekAskModel.blockTitle })
    }

    @Test @MainActor func blockIsRefusedOnceTheOpeningHasStarted() async throws {
        let clock = Clock()
        let ask = await Self.setUp(engine: Self.engine(), clock: clock)
        await ask.ask("When am I free on Friday?")
        let slot = try #require(ask.highlightedOpenTimes.first)
        clock.set(CalendarFixtures.date(18, 11, 5))
        #expect(ask.hasStarted(slot))
        await ask.block(slot)
        #expect(ask.blocked.isEmpty)
        #expect(ask.notice?.text == "That opening has started. Ask again for the time left.")
    }

    @Test @MainActor func freeTimeAnswersFollowTheClock() async throws {
        let clock = Clock()
        let ask = await Self.setUp(engine: Self.engine(#"{"free_query": {"days": ["today"]}}"#), clock: clock)
        await ask.ask("Am I free today?")
        #expect(ask.answers[0].sentence == "3 openings today: 9 to 9:30 AM, 10 AM to 12 PM and 1 to 5 PM.")
        // The next quarter hour comes before the first opening's start.
        #expect(ask.nextRefresh == CalendarFixtures.date(15, 8, 15))
        clock.set(CalendarFixtures.date(15, 10, 40))
        ask.refreshFreeTime()
        #expect(ask.answers[0].sentence == "2 openings today: 10:45 AM to 12 PM and 1 to 5 PM.")
        #expect(ask.highlightedOpenTimes.first?.start == CalendarFixtures.date(15, 10, 45))
        #expect(ask.nextRefresh == CalendarFixtures.date(15, 10, 45))
        clock.set(CalendarFixtures.date(15, 16, 50))
        ask.refreshFreeTime()
        #expect(ask.answers[0].sentence == FreeTimeSearch.passedHeadline)
        #expect(ask.highlightedOpenTimes.isEmpty)
        #expect(ask.nextRefresh == nil)
        #expect(ask.answers.count == 1)
    }

    @Test @MainActor func aRefreshKeepsAnOpeningBlockTookTaken() async throws {
        let clock = Clock()
        let ask = await Self.setUp(engine: Self.engine(#"{"free_query": {"days": ["today"]}}"#), clock: clock)
        await ask.ask("Am I free today?")
        let afternoon = DateInterval(start: CalendarFixtures.date(15, 13), end: CalendarFixtures.date(15, 17))
        #expect(ask.highlightedOpenTimes.contains(afternoon))
        await ask.block(afternoon)
        #expect(ask.blocked[afternoon] != nil)
        // Past the Focus block's start: the opening now starts after it,
        // not at the next quarter hour inside it.
        clock.set(CalendarFixtures.date(15, 13, 10))
        ask.refreshFreeTime()
        #expect(ask.highlightedOpenTimes == [DateInterval(start: CalendarFixtures.date(15, 14),
                                                          end: CalendarFixtures.date(15, 17))])
    }

    @Test @MainActor func availabilityChangesTheFingerprint() {
        let week = DateInterval(start: CalendarFixtures.date(14, 0), end: CalendarFixtures.date(21, 0))
        func print(_ events: [CalendarEvent]) -> Int {
            WeekAskModel.fingerprint(WeekSnapshot(interval: week, now: CalendarFixtures.now,
                                                  calendar: CalendarFixtures.calendar, events: events, todos: []),
                                     ignoring: [:])
        }
        let free = CalendarEvent(id: "hold", title: "Hold", start: CalendarFixtures.date(16, 9),
                                 end: CalendarFixtures.date(16, 17), isFree: true)
        var busy = free
        busy.isFree = false
        #expect(print([free]) != print([busy]))
        let subscribed = CalendarEvent(id: "trip", title: "Trip", start: CalendarFixtures.date(16, 0),
                                       end: CalendarFixtures.date(17, 0), isAllDay: true, isUserCalendar: false)
        var own = subscribed
        own.isUserCalendar = true
        #expect(print([subscribed]) != print([own]))
        // A Focus block Block added, now shown as Free, is a change too.
        #expect(!WeekAskModel.isUnchanged(free, ["hold": busy]))
    }

    @Test @MainActor func lookupAnswersDoNotRefresh() async {
        let clock = Clock()
        let ask = await Self.setUp(engine: Self.engine(#"{"answer": "Lunch is at noon.", "refs": ["E2"]}"#), clock: clock)
        await ask.ask("When is lunch?")
        #expect(ask.nextRefresh == nil)
        clock.set(CalendarFixtures.date(15, 14))
        ask.refreshFreeTime()
        #expect(ask.answers[0].sentence == "Lunch is at noon.")
    }

    @Test @MainActor func blockWaitsWhileAQuestionIsAnswered() async throws {
        let (_, ask) = await Self.setUp(engine: Self.engine(delay: .milliseconds(40)))
        await ask.ask(.freeOn(CalendarFixtures.date(18, 0)))
        let slot = try #require(ask.highlightedOpenTimes.first)
        let running = Task { await ask.ask("and after that?") }
        for _ in 0..<200 where ask.pending == nil { try await Task.sleep(for: .milliseconds(5)) }
        #expect(!ask.canBlock)
        await ask.block(slot)
        #expect(ask.blocked.isEmpty)
        await running.value
        #expect(ask.canBlock)
    }
}
