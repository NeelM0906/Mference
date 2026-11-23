import Foundation
import Observation

/// A question being answered.
public struct WeekAskPending: Sendable, Hashable {
    public var question: String
    public var step: WeekAskStep
}

/// The dark line under the box after an action ("Added Focus block…").
public struct WeekAskNotice: Sendable, Hashable {
    public var text: String
    /// Undo removes the event Block just added.
    public var canUndo: Bool
}

/// Ask about your week (canvas board 5). Questions and answers live here,
/// in memory only: never in SwiftData, Memory, Find or logs. They belong to
/// one week, at most three of them, and are gone when the person goes to
/// another week, presses Clear or quits. When the calendar changes under an
/// answer, the answer is marked out of date rather than silently wrong.
/// Two actions make something lasting, both on a click: Block adds a real
/// event (with Undo) and Continue in Chat starts a Private chat.
@MainActor @Observable
public final class WeekAskModel {
    public static let maximumQuestions = 3
    /// The title of an event Block adds.
    public static let blockTitle = "Focus block"

    /// The answers for `week`, oldest first.
    public private(set) var answers: [WeekAnswer] = []
    public private(set) var pending: WeekAskPending?
    /// The calendar changed since the answers were worked out.
    public private(set) var isStale = false
    public private(set) var notice: WeekAskNotice?
    /// Open blocks Block turned into events, with the event added.
    public private(set) var blocked: [DateInterval: CalendarEvent] = [:]
    /// Open blocks whose Block is being written: one click, one event.
    public private(set) var blocking: Set<DateInterval> = []
    /// The text in the box.
    public var draft = ""

    /// The week the answers are about.
    @ObservationIgnored private var week: DateInterval?
    @ObservationIgnored private var fingerprint: Int?
    /// The schedule the thread was answered from.
    @ObservationIgnored private var threadContext: WeekContext?
    /// The events the answers' snapshot held: a Focus block among them is
    /// part of what the answers read, so its edits and deletion count.
    @ObservationIgnored private var snapshotEventIDs: Set<String> = []
    /// Events Block added: they do not make an answer out of date.
    @ObservationIgnored private var ownEvents: [String: CalendarEvent] = [:]
    @ObservationIgnored private var task: Task<WeekAnswer, any Error>?
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var lastBlock: DateInterval?
    /// The week the question being answered is about.
    @ObservationIgnored private var pendingWeek: DateInterval?

    private let calendar: CalendarWeekModel
    private let answerer: WeekAnswerer

    public init(calendar: CalendarWeekModel, answerer: WeekAnswerer) {
        self.calendar = calendar
        self.answerer = answerer
    }

    // MARK: Scope

    /// The week a question asks about now.
    public var scope: DateInterval { calendar.askWeek }

    /// Days and times written the way the answers write them.
    public var text: WeekText { WeekText(calendar: calendar.calendar, now: calendar.currentTime) }

    /// "Sep 14 to 20".
    public var scopeLabel: String { text.weekRange(scope) }

    public var isFull: Bool { answers.count >= Self.maximumQuestions }

    public var canAsk: Bool {
        pending == nil && !isFull && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// The latest answer, unless it is out of date.
    public var current: WeekAnswer? { isStale ? nil : answers.last }

    /// The grid draws these as dashed open blocks.
    public var highlightedOpenTimes: [DateInterval] { current?.openTimes ?? [] }

    /// The grid rings these events and fades the rest.
    public var highlightedEventIDs: Set<String> { current?.eventIDs ?? [] }

    /// The day the person is looking at in Day view; the schedule says so.
    private var focusDay: Date? { calendar.mode == .day ? calendar.selectedDate : nil }

    /// The person moved to another week (or picked a day in another week):
    /// answers about the old week go.
    public func scopeChanged() {
        if let pendingWeek, pendingWeek != scope { stop() }
        guard let week, week != scope else { return }
        clear()
    }

    // MARK: Asking

    public func ask() async {
        let question = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else { return }
        await ask(question)
    }

    public func ask(_ question: String) async {
        let question = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty, pending == nil else { return }
        scopeChanged()
        guard !isFull else {
            notice = WeekAskNotice(text: "Three questions is the limit here. Continue in Chat to keep going with this "
                + "week attached.", canUndo: false)
            return
        }
        guard calendar.access == .granted else {
            notice = WeekAskNotice(text: "Allow Calendar access to ask about your week.", canUndo: false)
            return
        }
        let interval = scope
        // Pending before the first wait, so Stop, Clear and a week change
        // reach a question whose Calendar read is still running.
        draft = ""
        notice = nil
        generation += 1
        let mine = generation
        pending = WeekAskPending(question: question, step: .reading)
        pendingWeek = interval
        let snapshot: WeekSnapshot
        do {
            snapshot = WeekSnapshot(interval: interval, now: calendar.currentTime, calendar: calendar.calendar,
                                    events: try await calendar.events(in: interval), todos: calendar.todos)
        } catch {
            guard mine == generation else { return }
            notice = WeekAskNotice(text: "Calendar could not be read. Try again.", canUndo: false)
            draft = question
            pending = nil
            pendingWeek = nil
            return
        }
        guard mine == generation else { return }
        // A follow-up continues the thread while the calendar is as the
        // thread read it; otherwise this question starts a new one.
        let continuing = !answers.isEmpty && !isStale && week == interval
            && Self.fingerprint(snapshot, ignoring: ignoredOwnEvents) == fingerprint
        let history = continuing ? answers : []
        let fresh = WeekContext(week: snapshot, focus: focusDay)
        // The thread's schedule while it still holds (the engine can reuse
        // it word for word); a new one once the day or the open time moved.
        let context = continuing ? threadContext.flatMap { $0.readsTheSame(as: fresh) ? $0 : nil } ?? fresh : fresh
        let answerer = answerer
        let report: @Sendable (WeekAskStep) async -> Void = { [weak self] step in
            await self?.setStep(step, generation: mine)
        }
        let task = Task {
            try await answerer.answer(question, context: context, history: history, progress: report)
        }
        self.task = task
        do {
            let answer = try await task.value
            guard mine == generation else { return }
            // The person left the week while it was worked out.
            guard interval == scope else {
                pending = nil
                pendingWeek = nil
                self.task = nil
                return
            }
            if !continuing {
                answers = []
                isStale = false
            }
            threadContext = context
            week = interval
            snapshotEventIDs = Set(snapshot.events.map(\.id))
            fingerprint = Self.fingerprint(snapshot, ignoring: ignoredOwnEvents)
            answers.append(answer)
            // The calendar may have changed while the answer was worked
            // out (no change notice reaches an answer not yet shown).
            await revalidate(snapshot: snapshot, generation: mine)
        } catch {
            guard mine == generation else { return }
            if !(error is CancellationError) {
                notice = WeekAskNotice(text: Self.message(for: error), canUndo: false)
                draft = question
            }
        }
        // Stop, Clear or a newer question may have taken over during the
        // last wait: their state is not this question's to clear.
        guard mine == generation else { return }
        pending = nil
        pendingWeek = nil
        self.task = nil
    }

    /// Marks the answers out of date when Calendar no longer matches what
    /// `snapshot` read.
    private func revalidate(snapshot: WeekSnapshot, generation: Int) async {
        guard let events = try? await calendar.events(in: snapshot.interval), generation == self.generation else { return }
        let now = WeekSnapshot(interval: snapshot.interval, now: calendar.currentTime, calendar: calendar.calendar,
                               events: events, todos: calendar.todos)
        if Self.fingerprint(now, ignoring: ignoredOwnEvents) != Self.fingerprint(snapshot, ignoring: ignoredOwnEvents) {
            isStale = true
        }
    }

    private func setStep(_ step: WeekAskStep, generation: Int) {
        guard generation == self.generation, pending != nil else { return }
        pending?.step = step
    }

    static func message(for error: any Error) -> String {
        switch error {
        case let error as EngineServiceError: error.userMessage
        case let error as AssistantModelError: error.userMessage
        default: "The answer could not be worked out. Try again."
        }
    }

    /// Stops the answer being worked out; the question goes back in the box.
    public func stop() {
        guard let pending else { return }
        task?.cancel()
        task = nil
        generation += 1
        draft = pending.question
        self.pending = nil
        pendingWeek = nil
    }

    /// Clear (the x on the card): every answer for this week goes.
    public func clear() {
        if pending != nil { stop() }
        answers = []
        week = nil
        fingerprint = nil
        threadContext = nil
        snapshotEventIDs = []
        ownEvents = [:]
        isStale = false
        blocked = [:]
        notice = nil
    }

    public func dismissNotice() { notice = nil }

    /// Continue in Chat could not start a chat now.
    public func chatUnavailable() {
        notice = WeekAskNotice(text: "Chat is still answering. Try again when it's done.", canUndo: false)
    }

    // MARK: Continue in Chat

    /// The chat's title ("Week of Sep 14 to 20").
    public var chatTitle: String { "Week of \(scopeLabel)" }

    /// What the chat starts from: the schedule the answers read, so the chat
    /// model reads the same calendar.
    public var chatContext: String {
        "My calendar for \(text.weekRange(week ?? scope)), as Mference read it:\n" + (threadContext?.plainText ?? "")
    }

    /// The questions and answers, oldest first.
    public var chatTurns: [(question: String, answer: String)] {
        answers.map { ($0.question, $0.sentence) }
    }

    // MARK: Calendar changes

    /// EventKit or the to-dos changed: an answer worked out before is
    /// marked out of date when what it read is no longer the same.
    public func calendarChanged() async {
        guard let week, let fingerprint, !answers.isEmpty, !isStale else { return }
        guard let events = try? await calendar.events(in: week) else { return }
        let now = WeekSnapshot(interval: week, now: calendar.currentTime, calendar: calendar.calendar,
                               events: events, todos: calendar.todos)
        if Self.fingerprint(now, ignoring: ignoredOwnEvents) != fingerprint { isStale = true }
    }

    /// What an answer read, minus the events Block added.
    /// `own` are the events Block added, skipped only while unchanged: a
    /// Focus block moved or retitled afterwards counts like any change.
    static func fingerprint(_ week: WeekSnapshot, ignoring own: [String: CalendarEvent]) -> Int {
        var hasher = Hasher()
        for event in week.events where !Self.isUnchanged(event, own) {
            hasher.combine(event.id)
            hasher.combine(event.title)
            hasher.combine(event.start)
            hasher.combine(event.end)
            hasher.combine(event.isAllDay)
        }
        // Only to-dos an answer can mention: due before the week ends, or undated.
        let relevant = week.todos.filter { $0.due.map { $0 < week.interval.end } ?? true }
        for todo in relevant.sorted(by: { $0.id.uuidString < $1.id.uuidString }) {
            hasher.combine(todo.id)
            hasher.combine(todo.title)
            hasher.combine(todo.isDone)
            hasher.combine(todo.due)
        }
        return hasher.finalize()
    }

    /// Focus blocks added after the answers' snapshot: only these are left
    /// out of the comparison (while unchanged).
    private var ignoredOwnEvents: [String: CalendarEvent] {
        ownEvents.filter { !snapshotEventIDs.contains($0.key) }
    }

    static func isUnchanged(_ event: CalendarEvent, _ own: [String: CalendarEvent]) -> Bool {
        guard let added = own[event.id] else { return false }
        return added.title == event.title && added.start == event.start && added.end == event.end
            && added.isAllDay == event.isAllDay
    }

    // MARK: Block

    /// The block Block would add for an open time, from its start: the
    /// latest length asked for in the thread (an hour when none), at most
    /// the whole open block.
    public func blockInterval(for openTime: DateInterval) -> DateInterval {
        let asked = answers.reversed().lazy.compactMap { WeekText.requestedMinutes(in: $0.question) }.first ?? 60
        let length = min(openTime.duration, TimeInterval(asked * 60))
        return DateInterval(start: openTime.start, duration: length)
    }

    /// Adds a Focus block event for an open time (a real Calendar write, so
    /// only on the person's click), with Undo.
    public func block(_ openTime: DateInterval) async {
        guard blocked[openTime] == nil, !blocking.contains(openTime) else { return }
        blocking.insert(openTime)
        defer { blocking.remove(openTime) }
        let interval = blockInterval(for: openTime)
        guard let event = await calendar.addEvent(title: Self.blockTitle, start: interval.start, end: interval.end) else {
            notice = WeekAskNotice(text: calendar.errorMessage ?? "The event could not be added.", canUndo: false)
            return
        }
        ownEvents[event.id] = event
        blocked[openTime] = event
        lastBlock = openTime
        notice = WeekAskNotice(text: "Added \(Self.blockTitle) to Calendar, \(text.shortWeekday(interval.start)) "
            + "\(text.range(interval.start, interval.end)).", canUndo: true)
    }

    /// Removes the event the last Block added.
    public func undoBlock() async {
        guard let openTime = lastBlock, let event = blocked[openTime] else { return }
        if await calendar.deleteEvent(event) {
            blocked[openTime] = nil
            lastBlock = nil
            notice = nil
        } else {
            notice = WeekAskNotice(text: calendar.errorMessage ?? "The event could not be removed.", canUndo: false)
        }
    }
}
