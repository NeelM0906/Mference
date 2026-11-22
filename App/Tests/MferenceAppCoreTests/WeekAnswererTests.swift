import Foundation
@testable import MferenceAppCore
import Testing

/// Routing, wording, checking and model choice for Ask about your week,
/// over the scripted engine: no model is loaded.
@Suite struct WeekAnswererTests {
    static func week() -> WeekSnapshot { WeekFactsTests.week() }

    static func call(_ json: String) -> ToolCall {
        ToolCall(id: "call-1", name: WeekQuestionRouter.toolName, arguments: .null, argumentsJSON: json)
    }

    static let deepResident = EngineServiceStatus(phase: .ready, residentModel: .deep)

    static func engine(status: EngineServiceStatus = deepResident, replies: [ScriptedEngineService.Reply] = [],
                       quickInstalled: Bool = true) -> ScriptedEngineService {
        let options = AppModel.engineModels.map { model in
            ModelOption(model: model, availability: model == .quick && !quickInstalled
                ? .notInstalled(reason: "Not downloaded.") : .ready)
        }
        return ScriptedEngineService(options: options, status: status, replies: replies)
    }

    static func answerer(_ engine: ScriptedEngineService) -> WeekAnswerer {
        WeekAnswerer(picker: WeekModelPicker(engine: engine, appleModel: nil, useAppleModel: { false }))
    }

    // MARK: Fast path

    @Test func suggestionChipsNeedNoModel() {
        let week = Self.week()
        func route(_ question: String, previous: WeekQuery? = nil) -> WeekQuery? {
            WeekQuestionRouter.fastPath(question, in: week, previous: previous)
        }
        #expect(route("When am I free for 2 hours?") == .freeTime(FreeTimeRequest(minimumMinutes: 120)))
        #expect(route("What's my busiest day?") == .compareDays)
        #expect(route("What's due this week?") == .todosDue)
        #expect(route("Anything overlapping?") == .overlaps)
        #expect(route("When is the contract call?") == .findEvent(name: "contract call"))
        #expect(route("what's on thursday") == .daySummary(day: CalendarFixtures.date(17, 0)))
        #expect(route("any free time tomorrow afternoon for 90 minutes?")
            == .freeTime(FreeTimeRequest(minimumMinutes: 90, part: .afternoon, day: CalendarFixtures.date(16, 0))))
        #expect(route("Am I free next Tuesday?") == .outsideWeek)
        #expect(route("What's happening in October?") == .outsideWeek)
        // A follow-up narrows the last free-time question.
        #expect(route("only mornings", previous: .freeTime(FreeTimeRequest(minimumMinutes: 120)))
            == .freeTime(FreeTimeRequest(minimumMinutes: 120, part: .morning)))
        // Time limits and unknown names go to the model.
        #expect(route("am I free for an hour before noon thursday?") == nil)
        #expect(route("When is the dentist?") == nil)
        #expect(route("should I take the job?") == nil)
    }

    // MARK: Tool call

    @Test func toolCallArgumentsAreCheckedAgainstTheWeek() {
        let week = Self.week()
        func parse(_ json: String) -> WeekQuery? {
            try? WeekQuestionRouter.parse(WeekModelReply(content: "", toolCalls: [Self.call(json)]), in: week).get()
        }
        #expect(parse(#"{"type":"free_time","minutes":120,"before":"2026-09-17T12:00"}"#)
            == .freeTime(FreeTimeRequest(minimumMinutes: 120, before: CalendarFixtures.date(17, 12))))
        #expect(parse(#"{"type":"day_summary","day":"2026-09-18"}"#) == .daySummary(day: CalendarFixtures.date(18, 0)))
        #expect(parse(#"{"type":"free_time","day":"2026-09-24"}"#) == .outsideWeek)
        #expect(parse(#"{"type":"find_event","name":"dentist"}"#) == .findEvent(name: "dentist"))
        #expect(parse(#"{"type":"find_event"}"#) == nil)
        #expect(parse(#"{"type":"weather"}"#) == nil)
        #expect(parse(#"{"type":"free_time","minutes":5000}"#) == nil)
        // Apple's model writes the JSON as text.
        #expect((try? WeekQuestionRouter.parse(WeekModelReply(content: #"Sure: {"type":"overlaps"}"#), in: week).get())
            == .overlaps)
    }

    @Test func aRefusedCallIsRetriedOnceThenTurnedAway() async throws {
        let engine = Self.engine(replies: [
            .init(content: "", toolCalls: [Self.call(#"{"type":"weather"}"#)]),
            .init(content: "", toolCalls: [Self.call(#"{"type":"nonsense"}"#)]),
        ])
        let answer = try await Self.answerer(engine).answer("what should I wear?", in: Self.week())
        #expect(answer.query == .other)
        #expect(answer.source == .fixed(.noModelNeeded))
        #expect(engine.requests.count == 2)
        #expect(engine.requests[1].turns.last?.content?.contains("Your last call was refused: type weather") == true)
        #expect(engine.requests.allSatisfy { $0.model == .deep && $0.options.thinking == false })
    }

    // MARK: Checker

    @Test func checkerKeepsTrueSentencesAndRefusesInventedOnes() {
        let week = Self.week()
        let facts = WeekFacts.answer(.freeTime(FreeTimeRequest(minimumMinutes: 120)), in: week)
        let question = "When am I free for 2 hours?"
        func check(_ sentence: String) -> String? {
            WeekSentenceChecker.check(sentence, against: facts, question: question, calendar: week.calendar)
        }
        #expect(check("Your first 2-hour window is today, 10 AM to 12 PM, and there are eight in all. [F1]")
            == "Your first 2-hour window is today, 10 AM to 12 PM, and there are eight in all.")
        #expect(check("Thursday 12:00 to 17:00 is the longest. [F6]") != nil)
        #expect(check("You're free today at 7 PM. [F1]") == nil)
        #expect(check("You're free Saturday morning. [F1]") == nil)
        #expect(check("You have seven open blocks.") == nil)
        #expect(check("The earliest is today, 10 AM to 12 PM. [F42]") == nil)
        #expect(check("You have no free time this week.") == nil)
        #expect(check("") == nil)

        let none = WeekFacts.answer(.freeTime(FreeTimeRequest(minimumMinutes: 360)), in: week)
        #expect(WeekSentenceChecker.check("You're free Thursday, 12 to 5 PM. [F1]", against: none,
                                          question: "free for 6 hours?", calendar: week.calendar) == nil)
        #expect(WeekSentenceChecker.check("There's no 6-hour gap; the longest is Thursday, 12 to 5 PM. [F1]",
                                          against: none, question: "free for 6 hours?", calendar: week.calendar) != nil)
    }

    // MARK: Which model

    @Test func theResidentModelAnswersAndNothingIsSwapped() async {
        let engine = Self.engine()
        let picked = await WeekModelPicker(engine: engine, appleModel: nil, useAppleModel: { true }).pick()
        #expect(picked?.label == "Deep 27B")
        #expect((picked as? EngineWeekLanguageModel)?.model == .deep)
    }

    @Test func withNothingResidentAppleComesFirstThenQuick() async {
        let engine = Self.engine(status: .empty)
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice, answer: "")
        #expect(await WeekModelPicker(engine: engine, appleModel: apple, useAppleModel: { true }).pick()?.label
            == "Apple on-device")
        #expect(await WeekModelPicker(engine: engine, appleModel: apple, useAppleModel: { false }).pick()?.label
            == "Quick 4B")
        apple.setAvailability(.unavailable(reason: "Not ready."))
        #expect(await WeekModelPicker(engine: engine, appleModel: apple, useAppleModel: { true }).pick()?.label
            == "Quick 4B")
        let noQuick = Self.engine(status: .empty, quickInstalled: false)
        #expect(await WeekModelPicker(engine: noQuick, appleModel: apple, useAppleModel: { true }).pick() == nil)
        #expect(await WeekModelPicker.none.pick() == nil)
    }

    // MARK: End to end

    @Test func aChipIsWordedByTheResidentModel() async throws {
        let engine = Self.engine(replies: [.init(content: "Today, 10 AM to 12 PM is your first 2-hour opening. [F1]")])
        var steps: [WeekAskStep] = []
        let recorder = StepRecorder()
        let answer = try await Self.answerer(engine).answer("When am I free for 2 hours?", in: Self.week()) {
            await recorder.add($0)
        }
        steps = await recorder.steps
        #expect(answer.sentence == "Today, 10 AM to 12 PM is your first 2-hour opening.")
        #expect(answer.source == .model(label: "Deep 27B"))
        #expect(answer.facts.openTimes.count == 8)
        #expect(steps == [.readingQuestion, .working(answer.query), .wording])
        // One call only: the chip skipped routing, and the phraser gets no tools.
        #expect(engine.requests.count == 1)
        #expect(engine.requests[0].options.tools.isEmpty)
        let prompt = engine.requests[0].turns.last?.content ?? ""
        #expect(prompt.contains("F1: today (Tuesday Sep 15), 10 AM to 12 PM, 2 h"))
        #expect(prompt.contains("Correct answer for reference"))
    }

    @Test func anOpenQuestionIsRoutedThenWorded() async throws {
        let engine = Self.engine(replies: [
            .init(content: "", toolCalls: [Self.call(#"{"type":"free_time","minutes":120,"before":"2026-09-17T12:00"}"#)]),
            .init(content: "Yes, five times. The earliest is today, 10 AM to 12 PM. [F1]"),
        ])
        let answer = try await Self.answerer(engine).answer("can i squeeze in 2 hrs before thurs lunch", in: Self.week())
        #expect(answer.facts.openTimes.count == 5)
        #expect(answer.sentence == "Yes, five times. The earliest is today, 10 AM to 12 PM.")
        #expect(engine.requests.count == 2)
        #expect(engine.requests[0].options.tools.map(\.name) == [WeekQuestionRouter.toolName])
    }

    @Test func aWrongSentenceIsReplacedByTheFixedOne() async throws {
        let engine = Self.engine(replies: [.init(content: "Friday is busiest with 4 hours booked. [D5]")])
        let answer = try await Self.answerer(engine).answer("What's my busiest day?", in: Self.week())
        #expect(answer.sentence == "Friday is your busiest day left, with 2 h 30 min booked.")
        #expect(answer.source == .fixed(.failedCheck(label: "Deep 27B")))
    }

    @Test func withoutAModelTheCommonQuestionsStillAnswer() async throws {
        let answerer = WeekAnswerer(picker: .none)
        let free = try await answerer.answer("am I free for an hour before noon?", in: Self.week())
        #expect(free.source == .fixed(.noModel))
        #expect(free.facts.openTimes.first.map { WeekText(calendar: CalendarFixtures.calendar, now: CalendarFixtures.now)
            .range($0.start, $0.end) } == "10 AM to 12 PM")
        let other = try await answerer.answer("write me a poem", in: Self.week())
        #expect(other.query == .other)
    }

    @Test func todayIsOutsideAWeekThatDoesNotHoldIt() {
        let next = WeekFactsTests.week(now: CalendarFixtures.now)
        var later = next
        later.interval = DateInterval(start: CalendarFixtures.date(21, 0), end: CalendarFixtures.date(28, 0))
        #expect(WeekQuestionRouter.fastPath("what's on today?", in: later) == .outsideWeek)
        #expect(WeekQuestionRouter.fastPath("what's my agenda?", in: later) == nil)
    }

    @Test func aFailedModelStillGetsTheFixedAnswer() async throws {
        let engine = Self.engine(replies: [.init(content: "", error: .generationFailed("boom"))])
        let answer = try await Self.answerer(engine).answer("What's my busiest day?", in: Self.week())
        #expect(answer.sentence == "Friday is your busiest day left, with 2 h 30 min booked.")
        #expect(answer.source == .fixed(.modelFailed(label: "Deep 27B")))
        let routed = Self.engine(replies: [.init(content: "", error: .busy)])
        let fallback = try await Self.answerer(routed).answer("am I free for an hour before noon?", in: Self.week())
        #expect(fallback.source == .fixed(.modelFailed(label: "Deep 27B")))
        #expect(!fallback.facts.openTimes.isEmpty)
    }

    @Test func timesMustMatchWithTheirHalfOfTheDay() {
        let week = WeekFactsTests.week(events: [], todos: [])
        let facts = WeekFacts.answer(.freeTime(FreeTimeRequest(minimumMinutes: 60, day: CalendarFixtures.date(16, 0))),
                                     in: week)
        func check(_ sentence: String) -> String? {
            WeekSentenceChecker.check(sentence, against: facts, question: "free wednesday?", calendar: week.calendar)
        }
        #expect(check("You are free tomorrow from 9 AM to 5 PM. [F1]") != nil)
        #expect(check("You are free tomorrow from 9:00 to 17:00. [F1]") != nil)
        #expect(check("You are free tomorrow from 9 PM to 5 PM. [F1]") == nil)
        #expect(check("You are free tomorrow from 9 AM to 12 PM. [F1]") == nil)
        #expect(check("You're free tomorrow after 5 AM. [F1]") == nil)
        #expect(check("Tomorrow is open all 8 hours, 9 to 5 PM. [F1]") != nil)
    }

    @Test func anExplicitWindowIsAnsweredAsThatWindow() {
        let week = Self.week()
        let wednesday = CalendarFixtures.date(16, 0)
        let query = WeekQuestionRouter.fastPath("Am I free Wednesday from 2 PM to 4 PM?", in: week)
        #expect(query == .freeTime(FreeTimeRequest(minimumMinutes: 120, after: CalendarFixtures.date(16, 14),
                                                   before: CalendarFixtures.date(16, 16), day: wednesday,
                                                   includesWeekend: true)))
        #expect(WeekQuestionRouter.fastPath("am I free wednesday 2 to 4 pm", in: week) == query)
        var events = CalendarFixtures.events
        events.append(CalendarEvent(id: "busy", title: "Review", start: CalendarFixtures.date(16, 14),
                                    end: CalendarFixtures.date(16, 16)))
        let answer = WeekFacts.answer(try! #require(query), in: WeekFactsTests.week(events: events))
        #expect(answer.sentence == "Nothing is open tomorrow from 2 to 4 PM.")
        #expect(answer.foundNothing)
        // A single time or a time without a day goes to the model.
        #expect(WeekQuestionRouter.fastPath("am I free at 3 pm thursday?", in: week) == nil)
        #expect(WeekQuestionRouter.fastPath("am I free from 2 to 4 pm?", in: week) == nil)
    }

    @Test func windowsKeepTheirHalfLengthDayAndHours() throws {
        let week = Self.week()
        let empty = WeekFactsTests.week(events: [], todos: [])
        // "09:00" is 24-hour; "9:00 to 11:00" could be either and goes to the model.
        let morning = try #require(WeekQuestionRouter.fastPath("free wednesday from 09:00 to 11:00?", in: week))
        #expect(WeekFacts.answer(morning, in: empty).sentence.hasSuffix("tomorrow from 9 to 11 AM: tomorrow, 9 to 11 AM."))
        #expect(WeekQuestionRouter.fastPath("free wednesday from 9:00 to 11:00?", in: week) == nil)
        // A window after work hours is searched as written.
        let evening = try #require(WeekQuestionRouter.fastPath("Am I free Wednesday from 6 PM to 8 PM?", in: week))
        #expect(WeekFacts.answer(evening, in: empty).openTimes.count == 1)
        // A length asked for is kept even when the window is shorter.
        let long = try #require(WeekQuestionRouter.fastPath("Am I free for 3 hours Wednesday from 2 PM to 4 PM?", in: week))
        let answer = WeekFacts.answer(long, in: empty)
        #expect(answer.foundNothing)
        #expect(answer.sentence.hasPrefix("No open block of 3 h"))
        // "What about Friday?" moves the window to Friday.
        let friday = try #require(WeekQuestionRouter.fastPath("what about friday?", in: week, previous: evening))
        #expect(friday == .freeTime(FreeTimeRequest(minimumMinutes: 120, after: CalendarFixtures.date(18, 18),
                                                    before: CalendarFixtures.date(18, 20),
                                                    day: CalendarFixtures.date(18, 0), includesWeekend: true)))
        #expect(WeekFacts.answer(friday, in: empty).openTimes.count == 1)
    }

    @Test func anEventNamedWithAFreeTimeWordIsFound() {
        var events = CalendarFixtures.events
        events.append(CalendarEvent(id: "house", title: "Open House", start: CalendarFixtures.date(17, 15),
                                    end: CalendarFixtures.date(17, 16)))
        let week = WeekFactsTests.week(events: events)
        #expect(WeekQuestionRouter.fastPath("When is the Open House?", in: week) == .findEvent(name: "open house"))
        #expect(WeekQuestionRouter.fastPath("When am I free for 2 hours?", in: week)
            == .freeTime(FreeTimeRequest(minimumMinutes: 120)))
    }

    @Test func onlyMorningsReplacesAnInheritedWindow() throws {
        let week = Self.week()
        let window = try #require(WeekQuestionRouter.fastPath("Am I free Wednesday from 2 PM to 4 PM?", in: week))
        let mornings = try #require(WeekQuestionRouter.fastPath("only mornings", in: week, previous: window))
        guard case .freeTime(let request) = mornings else { Issue.record("not free time"); return }
        #expect(request.part == .morning && request.after == nil && request.before == nil)
        let answer = WeekFacts.answer(mornings, in: WeekFactsTests.week(events: [], todos: []))
        #expect(answer.openTimes.allSatisfy { CalendarFixtures.calendar.component(.hour, from: $0.end) <= 12 })
    }

    @Test func followUpsKeepTheirOwnIntentAndWindow() throws {
        let week = Self.week()
        let window = try #require(WeekQuestionRouter.fastPath("Am I free Wednesday from 2 PM to 4 PM?", in: week))
        #expect(WeekQuestionRouter.fastPath("What's on Friday?", in: week, previous: window)
            == .daySummary(day: CalendarFixtures.date(18, 0)))
        #expect(WeekQuestionRouter.fastPath("What about Friday from 6 PM to 8 PM?", in: week, previous: window)
            == .freeTime(FreeTimeRequest(minimumMinutes: 120, after: CalendarFixtures.date(18, 18),
                                         before: CalendarFixtures.date(18, 20), day: CalendarFixtures.date(18, 0),
                                         includesWeekend: true)))
        #expect(WeekQuestionRouter.fastPath("and friday at 3 pm?", in: week, previous: window) == nil)
        #expect(WeekQuestionRouter.fastPath("What about Friday before noon?", in: week, previous: window) == nil)
    }

    @Test func compoundLengthsAddUp() {
        #expect(WeekQuestionRouter.durationMinutes(" when am i free for 1 hour 30 minutes? ") == 90)
        #expect(WeekQuestionRouter.durationMinutes(" free for 2h 15m ") == 135)
        #expect(WeekQuestionRouter.durationMinutes(" free for an hour and a half ") == 90)
        #expect(WeekQuestionRouter.durationMinutes(" free for 45 minutes ") == 45)
        #expect(WeekQuestionRouter.durationMinutes(" when am i free? ") == nil)
    }

    @Test func windowsUseWallClockHoursOnAClockChangeDay() throws {
        var calendar = CalendarFixtures.calendar
        calendar.firstWeekday = 1
        let sunday = calendar.date(from: DateComponents(year: 2026, month: 11, day: 1))!
        let now = calendar.date(from: DateComponents(year: 2026, month: 10, day: 31, hour: 20))!
        let layout = WeekLayout(calendar: calendar)
        let week = WeekSnapshot(interval: layout.interval(from: layout.weekStart(containing: sunday)), now: now,
                                calendar: calendar, events: [], todos: [])
        let query = try #require(WeekQuestionRouter.fastPath("am I free sunday from 9 AM to 11 AM?", in: week))
        guard case .freeTime(let request) = query else { Issue.record("not free time"); return }
        #expect(request.after.map { calendar.component(.hour, from: $0) } == 9)
        #expect(request.before.map { calendar.component(.hour, from: $0) } == 11)
    }

    @Test func appleModelGetsTheToolSchemaAsText() async throws {
        let apple = ScriptedAssistantTextModel(kind: .appleOnDevice, answer: #"{"type":"overlaps"}"#)
        let query = try await WeekQuestionRouter.route("do any meetings collide?", in: Self.week(), previous: nil,
                                                       model: AssistantWeekLanguageModel(model: apple))
        #expect(query == .overlaps)
        let prompt = apple.requests.first?.prompt ?? ""
        #expect(prompt.contains(#""enum":["free_time","find_event","#))
        #expect(prompt.contains("YYYY-MM-DDTHH:MM"))
        #expect(prompt.hasSuffix("Reply with only the JSON object of the week_query arguments."))
    }

    @Test func aContrastingNotIsNotANo() {
        let week = Self.week()
        let facts = WeekFacts.answer(.compareDays, in: week)
        #expect(WeekSentenceChecker.check("Friday, not Monday, is your busiest day left with 2 h 30 min booked. [D5]",
                                          against: facts, question: "busiest?", calendar: week.calendar) != nil)
        #expect(WeekSentenceChecker.negative("You aren't free for 6 hours."))
        #expect(WeekSentenceChecker.negative("There's no gap."))
    }

    @Test func anotherWeekRunsNoModel() async throws {
        let engine = Self.engine()
        let answer = try await Self.answerer(engine).answer("what about next week?", in: Self.week())
        #expect(answer.query == .outsideWeek)
        #expect(answer.source == .fixed(.noModelNeeded))
        #expect(engine.requests.isEmpty)
    }
}

actor StepRecorder {
    var steps: [WeekAskStep] = []
    func add(_ step: WeekAskStep) { steps.append(step) }
}
