import Foundation

/// A real-model check of Ask about your week (MferenceSmoke `--week-ask`):
/// 20 questions over two fixed weeks, through the app's own prompt and
/// engine path. Each reply is shown both ways: mode A (as written) and
/// mode B (with its time warnings), since the modes differ only after the
/// reply. Runs only where a model may be loaded (MferenceSmoke, under the
/// machine lock); app tests never call it.
public enum WeekAskCheckpoint {
    public struct Question: Sendable {
        public var week: String
        public var text: String
        /// Continues the previous question's thread.
        public var followUp: Bool
    }

    public struct Result: Codable, Sendable {
        public var index: Int
        public var week: String
        public var question: String
        public var followUp: Bool
        /// The model's reply, exactly.
        public var reply: String
        /// Mode A: what the card shows.
        public var shown: String
        /// Mode B: the notes it adds.
        public var warnings: [String]
        public var cited: [String]
        public var wasCut: Bool
        public var latencyMilliseconds: Int
        public var promptTokens: Int?
        public var completionTokens: Int?
        public var error: String?
    }

    // MARK: The weeks

    static func calendar(firstWeekday: Int) -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/Los_Angeles") ?? .current
        calendar.locale = Locale(identifier: "en_US")
        calendar.firstWeekday = firstWeekday
        return calendar
    }

    static func week(now: Date, calendar: Calendar, events: [CalendarEvent], todos: [TodoItem]) -> WeekSnapshot {
        let layout = WeekLayout(calendar: calendar)
        return WeekSnapshot(interval: layout.interval(from: layout.weekStart(containing: now)), now: now,
                            calendar: calendar, events: events, todos: todos)
    }

    /// The manual-test week: Thanksgiving 2026 from two holiday calendars,
    /// Black Friday, nothing else; Sunday 9 PM; the week starts on Sunday.
    public static var thanksgivingWeek: WeekSnapshot {
        let calendar = calendar(firstWeekday: 1)
        func day(_ day: Int, _ hour: Int = 0) -> Date {
            calendar.date(from: DateComponents(year: 2026, month: 11, day: day, hour: hour)) ?? .now
        }
        return week(now: day(22, 21), calendar: calendar, events: [
            CalendarEvent(id: "tg-1", title: "Thanksgiving", start: day(26), end: day(27), isAllDay: true,
                          calendarName: "Holidays in United States", isUserCalendar: false),
            CalendarEvent(id: "tg-2", title: "Thanksgiving Day", start: day(26), end: day(27), isAllDay: true,
                          calendarName: "US Holidays", isUserCalendar: false),
            CalendarEvent(id: "bf", title: "Black Friday", start: day(27), end: day(28), isAllDay: true,
                          calendarName: "US Holidays", isUserCalendar: false),
        ], todos: [])
    }

    /// A busy work week (the mockup's): Tuesday Sep 15 2026, 8 AM; the week
    /// starts on Monday; one overlap, the person's own all-day offsite
    /// (Thursday is taken), to-dos.
    public static var workWeek: WeekSnapshot {
        let calendar = calendar(firstWeekday: 2)
        func at(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
            calendar.date(from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute)) ?? .now
        }
        return week(now: at(15, 8), calendar: calendar, events: [
            CalendarEvent(id: "sync", title: "Team sync", start: at(14, 10), end: at(14, 10, 30)),
            CalendarEvent(id: "focus-mon", title: "Focus time", start: at(14, 13), end: at(14, 15)),
            CalendarEvent(id: "coffee", title: "Coffee with Tom", start: at(15, 9, 30), end: at(15, 10)),
            CalendarEvent(id: "lunch", title: "Lunch", start: at(15, 12), end: at(15, 13)),
            CalendarEvent(id: "lunch-ana", title: "Lunch with Ana", start: at(15, 12, 30), end: at(15, 13, 30)),
            CalendarEvent(id: "review", title: "Design review", start: at(16, 10), end: at(16, 11)),
            CalendarEvent(id: "planning", title: "Weekly planning", start: at(16, 13), end: at(16, 14)),
            CalendarEvent(id: "offsite", title: "Offsite", start: at(17, 0), end: at(18, 0), isAllDay: true),
            CalendarEvent(id: "launch", title: "Launch check-in", start: at(17, 11), end: at(17, 12)),
            CalendarEvent(id: "focus-fri", title: "Focus time", start: at(18, 9), end: at(18, 11)),
            CalendarEvent(id: "contract", title: "Contract call", start: at(18, 13), end: at(18, 13, 30)),
        ], todos: [
            TodoItem(title: "Send copy decisions to Priya", due: at(15, 0)),
            TodoItem(title: "Review contract section 4", due: at(18, 0)),
            TodoItem(title: "Expense the team lunch", isDone: true, due: at(16, 0)),
            TodoItem(title: "Pick a dentist appointment"),
            TodoItem(title: "Renew passport", due: at(10, 0)),
        ])
    }

    public static let questions: [Question] = [
        Question(week: "thanksgiving", text: "what do i have going on this Week?", followUp: false),
        Question(week: "thanksgiving", text: "what am i doing this week", followUp: false),
        Question(week: "thanksgiving", text: "whats happening on monday", followUp: false),
        Question(week: "thanksgiving", text: "am i free on monday?", followUp: false),
        Question(week: "thanksgiving", text: "When is Thanksgiving?", followUp: false),
        Question(week: "thanksgiving", text: "Is anything happening on Friday?", followUp: false),
        Question(week: "thanksgiving", text: "When am I free for 3 hours this week?", followUp: false),
        Question(week: "thanksgiving", text: "Do I have any to-dos due?", followUp: false),
        Question(week: "work", text: "When am I free for 2 hours?", followUp: false),
        Question(week: "work", text: "only mornings", followUp: true),
        Question(week: "work", text: "what about Friday?", followUp: true),
        Question(week: "work", text: "What's my busiest day?", followUp: false),
        Question(week: "work", text: "When is the contract call?", followUp: false),
        Question(week: "work", text: "What's on Thursday?", followUp: false),
        Question(week: "work", text: "Anything overlapping?", followUp: false),
        Question(week: "work", text: "What's due this week?", followUp: false),
        Question(week: "work", text: "Do I have anything after 3 PM on Wednesday?", followUp: false),
        Question(week: "work", text: "How much time is booked on Monday?", followUp: false),
        Question(week: "work", text: "Can I fit a 90 minute meeting tomorrow morning?", followUp: false),
        Question(week: "work", text: "What's tomorrow like?", followUp: false),
    ]

    static var contexts: [String: WeekContext] {
        ["thanksgiving": WeekContext(week: thanksgivingWeek, focus: nil), "work": WeekContext(week: workWeek, focus: nil)]
    }

    // MARK: Exporting, for a run through MferenceServer

    /// One request to send. A first question carries the whole thread start
    /// (system and user turns); a follow-up carries only its own user turn,
    /// sent after the previous answer as the assistant turn.
    public struct Request: Codable, Sendable {
        public struct Message: Codable, Sendable {
            public var role: String
            public var content: String
        }

        public var format: String
        public var index: Int
        public var week: String
        public var question: String
        public var followUp: Bool
        public var messages: [Message]
    }

    /// Every question in `format`, ready to send.
    public static func requests(format: WeekScheduleFormat) -> [Request] {
        let contexts = contexts
        let placeholder = WeekAnswer(question: "", sentence: "", rows: [], source: .model(label: ""))
        return questions.enumerated().compactMap { index, question in
            guard let context = contexts[question.week] else { return nil }
            var turns = WeekAnswerer.turns(question.text, context: context,
                                           history: question.followUp ? [placeholder] : [], format: format)
            if question.followUp { turns = Array(turns.suffix(1)) }
            return Request(format: format.rawValue, index: index + 1, week: question.week, question: question.text,
                           followUp: question.followUp,
                           messages: turns.map { Request.Message(role: $0.role.rawValue, content: $0.content ?? "") })
        }
    }

    /// One reply from the server.
    public struct Reply: Codable, Sendable {
        public var format: String
        public var index: Int
        public var reply: String
        public var finishReason: String?
        public var latencyMilliseconds: Int
        public var promptTokens: Int?
        public var completionTokens: Int?
        public var error: String?
        /// The assistant turns sent before this follow-up.
        public var history: [String]?
    }

    /// The server's replies as the app would show them (modes A and B).
    public static func review(_ replies: [Reply]) -> [Result] {
        let contexts = contexts
        return replies.compactMap { reply in
            guard questions.indices.contains(reply.index - 1) else { return nil }
            let question = questions[reply.index - 1]
            guard let context = contexts[question.week] else { return nil }
            var result = Result(index: reply.index, week: question.week, question: question.text,
                                followUp: question.followUp, reply: reply.reply, shown: "", warnings: [], cited: [],
                                wasCut: reply.finishReason == "length", latencyMilliseconds: reply.latencyMilliseconds,
                                promptTokens: reply.promptTokens, completionTokens: reply.completionTokens,
                                error: reply.error)
            if reply.error == nil {
                let model = WeekModelReply(content: reply.reply, wasCut: result.wasCut)
                if let shown = WeekAnswerer.shown(model, context: context, question: question.text, review: .timeWarning) {
                    result.shown = shown.text
                    result.warnings = shown.warnings
                    result.cited = shown.cited
                } else {
                    result.error = "empty answer"
                }
            }
            return result
        }
    }

    // MARK: Running

    /// Asks every question with `model` (already resident) and returns the
    /// results; each is also written to `log` as mode B would log it.
    public static func run(model: any WeekLanguageModel, log: WeekAskLog?,
                           format: WeekScheduleFormat = .lines) async -> [Result] {
        let contexts = contexts
        var results: [Result] = []
        var thread: [WeekAnswer] = []
        for (index, question) in questions.enumerated() {
            guard let context = contexts[question.week] else { continue }
            if !question.followUp { thread = [] }
            let historyCount = thread.count
            let started = ContinuousClock.now
            var result = Result(index: index + 1, week: question.week, question: question.text,
                                followUp: question.followUp, reply: "", shown: "", warnings: [], cited: [],
                                wasCut: false, latencyMilliseconds: 0)
            do {
                let reply = try await model.respond(turns: WeekAnswerer.turns(question.text, context: context,
                                                                               history: thread, format: format),
                                                    maximumTokens: WeekAnswerer.maximumTokens)
                result.reply = reply.content
                result.wasCut = reply.wasCut
                result.promptTokens = reply.promptTokens
                result.completionTokens = reply.completionTokens
                if let shown = WeekAnswerer.shown(reply, context: context, question: question.text,
                                                  review: .timeWarning) {
                    result.shown = shown.text
                    result.warnings = shown.warnings
                    result.cited = shown.cited
                    thread.append(WeekAnswer(question: question.text, sentence: shown.text, rows: [],
                                             source: .model(label: model.label)))
                } else {
                    result.error = "empty answer"
                }
            } catch {
                result.error = String(describing: error)
            }
            let elapsed = started.duration(to: .now)
            result.latencyMilliseconds = Int(elapsed.components.seconds) * 1_000
                + Int(elapsed.components.attoseconds / 1_000_000_000_000_000)
            log?.record(WeekAskLogEntry(
                time: ISO8601DateFormatter().string(from: Date()), review: WeekAnswerReview.timeWarning.rawValue,
                model: model.label, question: question.text, historyCount: historyCount,
                format: format.rawValue, schedule: context.schedule(format), reply: result.reply, shown: result.shown, cited: result.cited,
                warnings: result.warnings, source: result.error == nil ? "model" : "modelFailed",
                wasCut: result.wasCut, latencyMilliseconds: result.latencyMilliseconds,
                promptTokens: result.promptTokens, completionTokens: result.completionTokens, error: result.error))
            results.append(result)
        }
        return results
    }
}
