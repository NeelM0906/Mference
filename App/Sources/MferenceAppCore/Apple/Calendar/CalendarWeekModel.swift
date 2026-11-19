import Foundation
import Observation

/// The Calendar screen's state: the week's events, proposals waiting for one
/// click, and the to-do list (mockup 04).
@MainActor @Observable
public final class CalendarWeekModel {
    public enum Access: Sendable, Hashable {
        /// Not asked yet: the screen shows the one-line reason and a button.
        case notAsked
        case granted
        /// Denied: the screen points to System Settings.
        case blocked
    }

    public private(set) var access: Access = .notAsked
    public private(set) var weekStart: Date
    public private(set) var events: [CalendarEvent] = []
    public private(set) var todos: [TodoItem] = []
    public private(set) var proposals: [ProposedEvent] = []
    /// One user-facing sentence after a failed action.
    public private(set) var errorMessage: String?
    public private(set) var isSyncingReminders = false
    /// Opens Settings > Connections (the "Settings" link in the to-do panel);
    /// set by the app.
    @ObservationIgnored public var openSettings: (@MainActor () -> Void)?
    @ObservationIgnored private var loadGeneration = 0

    public let layout: WeekLayout
    private let service: any CalendarServicing
    private let store: PlannerStore
    private let defaults: UserDefaults
    private let now: @Sendable () -> Date

    public init(service: any CalendarServicing, store: PlannerStore, layout: WeekLayout = WeekLayout(),
                defaults: UserDefaults = .standard, now: @escaping @Sendable () -> Date = { .now }) {
        self.service = service
        self.store = store
        self.layout = layout
        self.defaults = defaults
        self.now = now
        self.weekStart = layout.weekStart(containing: now())
        let contents = store.snapshot
        todos = contents.todos
        proposals = contents.proposals
    }

    // MARK: Week

    public var days: [Date] { layout.dayDates(from: weekStart) }

    public var blocks: [WeekBlock] {
        layout.blocks(weekStart: weekStart, events: events, proposals: proposalsThisWeek)
    }

    /// All-day and off-hours events by day (0 = first day shown).
    public var outsideGrid: [Int: [CalendarEvent]] {
        layout.outsideGrid(weekStart: weekStart, events: events)
    }

    /// Proposals being added right now: a second click does nothing.
    public private(set) var addingProposals: Set<UUID> = []

    public var proposalsThisWeek: [ProposedEvent] {
        let interval = layout.interval(from: weekStart)
        return proposals.filter { interval.contains($0.start) }.sorted { $0.start < $1.start }
    }

    /// "September 14 to 18", or "September 28 to October 2".
    public var weekTitle: String {
        guard let first = days.first, let last = days.last else { return "" }
        let calendar = layout.calendar
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.setLocalizedDateFormatFromTemplate("MMMM")
        let month = { (date: Date) in formatter.string(from: date) }
        let day = { (date: Date) in String(calendar.component(.day, from: date)) }
        if calendar.isDate(first, equalTo: last, toGranularity: .month) {
            return "\(month(first)) \(day(first)) to \(day(last))"
        }
        return "\(month(first)) \(day(first)) to \(month(last)) \(day(last))"
    }

    public func isToday(_ date: Date) -> Bool {
        layout.calendar.isDate(date, inSameDayAs: now())
    }

    /// Checks access without asking, then loads the week if allowed.
    public func load() async {
        switch await service.calendarAccess() {
        case .granted: access = .granted
        case .notDetermined: access = .notAsked
        case .denied, .needsSystemSettings: access = .blocked
        }
        await reload()
    }

    /// The person clicked "Show my calendar": the broker shows the reason,
    /// then the system prompt.
    public func requestAccess() async {
        switch await service.requestCalendarAccess() {
        case .granted: access = .granted
        case .notDetermined: access = .notAsked
        case .denied, .needsSystemSettings: access = .blocked
        }
        await reload()
    }

    public func showWeek(offset: Int) async {
        weekStart = layout.calendar.date(byAdding: .weekOfYear, value: offset, to: weekStart) ?? weekStart
        await reload()
    }

    public func showToday() async {
        weekStart = layout.weekStart(containing: now())
        await reload()
    }

    public func reload() async {
        guard access == .granted else { events = []; return }
        // A slower load for a week the person already left must not
        // overwrite the newer one.
        loadGeneration += 1
        let generation = loadGeneration
        let interval = layout.interval(from: weekStart)
        do {
            let loaded = try await service.events(from: interval.start, to: interval.end)
            guard generation == loadGeneration else { return }
            events = loaded
            errorMessage = nil
        } catch let error as CalendarServiceError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: Proposals

    /// A new suggestion (from Mail, say). Ignored if this source was already
    /// handled, so a dismissed proposal does not come back.
    public func propose(_ proposal: ProposedEvent) {
        let key = proposal.source.reference.map { "\(proposal.source.kind.rawValue):\($0)" }
        guard key.map({ !store.snapshot.handledSources.contains($0) }) ?? true else { return }
        persist {
            $0.proposals.append(proposal)
            if let key { $0.handledSources.insert(key) }
        }
    }

    /// One click: the event goes into Calendar.
    public func add(_ proposal: ProposedEvent) async {
        guard !addingProposals.contains(proposal.id), proposals.contains(where: { $0.id == proposal.id }) else { return }
        addingProposals.insert(proposal.id)
        defer { addingProposals.remove(proposal.id) }
        if access != .granted { await requestAccess() }
        guard access == .granted else { return }
        do {
            _ = try await service.add(proposal)
            persist { $0.proposals.removeAll { $0.id == proposal.id } }
            await reload()
        } catch let error as CalendarServiceError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// "New event" in the toolbar: the person typed it, so it is added at
    /// once (Calendar access asked first if needed). Returns whether it was
    /// added; `errorMessage` says why not.
    @discardableResult
    public func createEvent(title: String, start: Date, end: Date) async -> Bool {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, end > start else {
            errorMessage = "An event needs a title and an end after its start."
            return false
        }
        if access != .granted { await requestAccess() }
        guard access == .granted else { return false }
        do {
            _ = try await service.add(ProposedEvent(title: trimmed, start: start, end: end,
                                                    source: ItemSource(kind: .manual, label: "Added by you")))
            await reload()
            return true
        } catch let error as CalendarServiceError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = error.localizedDescription
        }
        return false
    }

    public func dismiss(_ proposal: ProposedEvent) {
        persist { $0.proposals.removeAll { $0.id == proposal.id } }
    }

    // MARK: To-dos

    public var remindersSyncEnabled: Bool { RemindersSyncSetting.isEnabled(defaults) }

    public func addTodo(_ todo: TodoItem) {
        let key = todo.source.reference.map { "todo-\(todo.source.kind.rawValue):\($0)" }
        guard key.map({ !store.snapshot.handledSources.contains($0) }) ?? true else { return }
        persist {
            $0.todos.append(todo)
            if let key { $0.handledSources.insert(key) }
        }
    }

    public func setDone(_ done: Bool, for id: UUID) {
        persist {
            guard let index = $0.todos.firstIndex(where: { $0.id == id }) else { return }
            $0.todos[index].isDone = done
        }
    }

    public func removeTodo(_ id: UUID) {
        persist { $0.todos.removeAll { $0.id == id } }
    }

    /// Open first, then done ones.
    public var sortedTodos: [TodoItem] {
        todos.sorted { lhs, rhs in
            if lhs.isDone != rhs.isDone { return !lhs.isDone }
            switch (lhs.due, rhs.due) {
            case let (l?, r?) where l != r: return l < r
            case (.some, nil): return true
            case (nil, .some): return false
            default: return lhs.createdAt < rhs.createdAt
            }
        }
    }

    /// The person clicked "Sync to Reminders" (each sync asks first, L9).
    /// Does nothing unless the Settings switch is on.
    public func syncReminders() async {
        guard remindersSyncEnabled else { return }
        isSyncingReminders = true
        defer { isSyncingReminders = false }
        var state = await service.remindersAccess()
        if state != .granted { state = await service.requestRemindersAccess() }
        guard state == .granted else {
            errorMessage = "Mference cannot reach Reminders. Turn it on in System Settings > Privacy & Security."
            return
        }
        do {
            let outcomes = try await service.syncReminders(todos)
            persist {
                for index in $0.todos.indices {
                    switch outcomes[$0.todos[index].id] {
                    case .synced(let reminderID, let fields)?:
                        $0.todos[index].reminderID = reminderID
                        $0.todos[index].lastSynced = fields
                    case .deletedInReminders?:
                        $0.todos[index].reminderID = nil
                        $0.todos[index].lastSynced = nil
                        $0.todos[index].unlinkedFromReminders = true
                    case nil:
                        break
                    }
                }
            }
            errorMessage = nil
        } catch let error as CalendarServiceError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    public func clearError() { errorMessage = nil }

    private func persist(_ change: (inout PlannerStore.Contents) -> Void) {
        do {
            try store.update(change)
        } catch {
            errorMessage = "Mference could not save your to-dos: \(error.localizedDescription)"
        }
        let contents = store.snapshot
        todos = contents.todos
        proposals = contents.proposals
    }
}
