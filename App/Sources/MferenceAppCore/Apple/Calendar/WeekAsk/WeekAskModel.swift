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
    /// The text in the box.
    public var draft = ""

    /// The week the answers are about.
    @ObservationIgnored private var week: DateInterval?
    @ObservationIgnored private var fingerprint: Int?
    /// Events Block added: they do not make an answer out of date.
    @ObservationIgnored private var ownEventIDs: Set<String> = []
    @ObservationIgnored private var task: Task<WeekAnswer, any Error>?
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var lastBlock: DateInterval?

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
    public var highlightedOpenTimes: [DateInterval] { current?.facts.openTimes ?? [] }

    /// The grid rings these events and fades the rest.
    public var highlightedEventIDs: Set<String> { current?.facts.eventIDs ?? [] }

    /// The busiest days left, for the "Busiest left" badge.
    public var highlightedDays: [Date] {
        guard let current, case .compareDays = current.query else { return [] }
        let mostBooked = current.facts.facts.compactMap { fact -> WeekDayLoad? in
            if case .day(let load) = fact.kind, !load.isPast { load } else { nil }
        }
        let most = mostBooked.map(\.bookedMinutes).max() ?? 0
        return most > 0 ? mostBooked.filter { $0.bookedMinutes == most }.map(\.day) : []
    }

    /// The person moved to another week (or picked a day in another week):
    /// answers about the old week go.
    public func scopeChanged() {
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
        let snapshot: WeekSnapshot
        do {
            snapshot = WeekSnapshot(interval: interval, now: calendar.currentTime, calendar: calendar.calendar,
                                    events: try await calendar.events(in: interval), todos: calendar.todos)
        } catch {
            notice = WeekAskNotice(text: "Calendar could not be read. Try again.", canUndo: false)
            return
        }
        draft = ""
        notice = nil
        generation += 1
        let mine = generation
        pending = WeekAskPending(question: question, step: .readingQuestion)
        let previous = isStale ? nil : answers.last?.query
        let answerer = answerer
        let report: @Sendable (WeekAskStep) async -> Void = { [weak self] step in
            await self?.setStep(step, generation: mine)
        }
        let task = Task { try await answerer.answer(question, in: snapshot, previous: previous, progress: report) }
        self.task = task
        do {
            let answer = try await task.value
            guard mine == generation else { return }
            if week != interval || isStale {
                answers = []
                isStale = false
            }
            week = interval
            fingerprint = Self.fingerprint(snapshot, ignoring: ownEventIDs)
            answers.append(answer)
        } catch {
            guard mine == generation else { return }
            if !(error is CancellationError) {
                notice = WeekAskNotice(text: Self.message(for: error), canUndo: false)
                draft = question
            }
        }
        pending = nil
        self.task = nil
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
    }

    /// Clear (the x on the card): every answer for this week goes.
    public func clear() {
        if pending != nil { stop() }
        answers = []
        week = nil
        fingerprint = nil
        isStale = false
        blocked = [:]
        notice = nil
    }

    public func dismissNotice() { notice = nil }

    // MARK: Calendar changes

    /// EventKit or the to-dos changed: an answer worked out before is
    /// marked out of date when what it read is no longer the same.
    public func calendarChanged() async {
        guard let week, let fingerprint, !answers.isEmpty, !isStale else { return }
        guard let events = try? await calendar.events(in: week) else { return }
        let now = WeekSnapshot(interval: week, now: calendar.currentTime, calendar: calendar.calendar,
                               events: events, todos: calendar.todos)
        if Self.fingerprint(now, ignoring: ownEventIDs) != fingerprint { isStale = true }
    }

    /// What an answer read, minus the events Block added.
    static func fingerprint(_ week: WeekSnapshot, ignoring ids: Set<String>) -> Int {
        var hasher = Hasher()
        for event in week.events where !ids.contains(event.id) {
            hasher.combine(event.id)
            hasher.combine(event.title)
            hasher.combine(event.start)
            hasher.combine(event.end)
            hasher.combine(event.isAllDay)
        }
        for todo in week.todos.sorted(by: { $0.id.uuidString < $1.id.uuidString }) {
            hasher.combine(todo.id)
            hasher.combine(todo.title)
            hasher.combine(todo.isDone)
            hasher.combine(todo.due)
        }
        return hasher.finalize()
    }

    // MARK: Block

    /// The block Block would add for an open time: the length asked for (at
    /// least the whole block when it is shorter), from its start.
    public func blockInterval(for openTime: DateInterval) -> DateInterval {
        guard case .freeTime(let request) = current?.query else { return openTime }
        let length = min(openTime.duration, TimeInterval(request.minimumMinutes * 60))
        return DateInterval(start: openTime.start, duration: length)
    }

    /// Adds a Focus block event for an open time (a real Calendar write, so
    /// only on the person's click), with Undo.
    public func block(_ openTime: DateInterval) async {
        guard blocked[openTime] == nil else { return }
        let interval = blockInterval(for: openTime)
        guard let event = await calendar.addEvent(title: Self.blockTitle, start: interval.start, end: interval.end) else {
            notice = WeekAskNotice(text: calendar.errorMessage ?? "The event could not be added.", canUndo: false)
            return
        }
        ownEventIDs.insert(event.id)
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
