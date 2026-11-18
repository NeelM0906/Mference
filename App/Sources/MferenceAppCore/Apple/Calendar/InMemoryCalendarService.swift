import Foundation

/// A calendar that lives in memory: tests, previews and snapshots. Never
/// touches EventKit.
public actor InMemoryCalendarService: CalendarServicing {
    public private(set) var storedEvents: [CalendarEvent]
    public private(set) var reminders: [String: TodoItem] = [:]
    private var calendarState: PermissionState
    private var remindersState: PermissionState
    private let grantOnRequest: Bool
    public private(set) var requests: [AppPermission] = []
    private var nextID = 1

    public init(events: [CalendarEvent] = [], calendarAccess: PermissionState = .granted,
                remindersAccess: PermissionState = .notDetermined, grantOnRequest: Bool = true) {
        self.storedEvents = events
        self.calendarState = calendarAccess
        self.remindersState = remindersAccess
        self.grantOnRequest = grantOnRequest
    }

    public func calendarAccess() async -> PermissionState { calendarState }

    public func requestCalendarAccess() async -> PermissionState {
        requests.append(.calendar)
        if calendarState == .notDetermined { calendarState = grantOnRequest ? .granted : .denied }
        return calendarState
    }

    public func remindersAccess() async -> PermissionState { remindersState }

    public func requestRemindersAccess() async -> PermissionState {
        requests.append(.reminders)
        if remindersState == .notDetermined { remindersState = grantOnRequest ? .granted : .denied }
        return remindersState
    }

    public func events(from start: Date, to end: Date) async throws -> [CalendarEvent] {
        guard calendarState == .granted else { throw CalendarServiceError.noAccess(calendarState) }
        return storedEvents.filter { $0.end > start && $0.start < end }.sorted { $0.start < $1.start }
    }

    public func add(_ proposal: ProposedEvent) async throws -> CalendarEvent {
        guard calendarState == .granted else { throw CalendarServiceError.noAccess(calendarState) }
        let replaced = proposal.replaces.flatMap { reference in storedEvents.first { $0.reference == reference } }
        let event = CalendarEvent(id: makeID(), title: proposal.title, start: proposal.start, end: proposal.end,
                                  calendarName: replaced?.calendarName ?? "Calendar",
                                  calendarColor: replaced?.calendarColor, location: replaced?.location)
        if let reference = proposal.replaces { storedEvents.removeAll { $0.reference == reference } }
        storedEvents.append(event)
        return event
    }

    public func update(_ reference: EventReference, title: String, start: Date, end: Date) async throws -> CalendarEvent {
        guard let index = storedEvents.firstIndex(where: { $0.reference == reference }) else {
            throw CalendarServiceError.eventNotFound
        }
        storedEvents[index].title = title
        storedEvents[index].start = start
        storedEvents[index].end = end
        return storedEvents[index]
    }

    public func remove(_ reference: EventReference) async throws {
        guard storedEvents.contains(where: { $0.reference == reference }) else { throw CalendarServiceError.eventNotFound }
        storedEvents.removeAll { $0.reference == reference }
    }

    public func syncReminders(_ todos: [TodoItem]) async throws -> [UUID: ReminderSyncOutcome] {
        guard remindersState == .granted else { throw CalendarServiceError.noAccess(remindersState) }
        var outcomes: [UUID: ReminderSyncOutcome] = [:]
        for todo in todos where !todo.unlinkedFromReminders {
            if let id = todo.reminderID {
                guard var stored = reminders[id] else { outcomes[todo.id] = .deletedInReminders; continue }
                let last = todo.lastSynced
                if last?.title != todo.title { stored.title = todo.title }
                if last?.isDone != todo.isDone { stored.isDone = todo.isDone }
                if last?.due != todo.due { stored.due = todo.due }
                reminders[id] = stored
                outcomes[todo.id] = .synced(reminderID: id, fields: stored.reminderFields)
            } else {
                let id = makeID()
                reminders[id] = todo
                outcomes[todo.id] = .synced(reminderID: id, fields: todo.reminderFields)
            }
        }
        return outcomes
    }

    /// Tests: the person edits or deletes a reminder in Reminders.
    public func editReminder(_ id: String, title: String? = nil, isDone: Bool? = nil) {
        if let title { reminders[id]?.title = title }
        if let isDone { reminders[id]?.isDone = isDone }
    }

    public func deleteReminder(_ id: String) {
        reminders[id] = nil
    }

    private func makeID() -> String {
        defer { nextID += 1 }
        return "memory-\(nextID)"
    }
}
