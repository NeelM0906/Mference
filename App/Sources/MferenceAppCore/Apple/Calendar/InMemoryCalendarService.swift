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
        let replaced = storedEvents.first { $0.id == proposal.replacesEventID }
        let event = CalendarEvent(id: makeID(), title: proposal.title, start: proposal.start, end: proposal.end,
                                  calendarName: replaced?.calendarName ?? "Calendar",
                                  calendarColor: replaced?.calendarColor, location: replaced?.location)
        storedEvents.removeAll { $0.id == proposal.replacesEventID }
        storedEvents.append(event)
        return event
    }

    public func update(eventID: String, title: String, start: Date, end: Date) async throws -> CalendarEvent {
        guard let index = storedEvents.firstIndex(where: { $0.id == eventID }) else {
            throw CalendarServiceError.eventNotFound
        }
        storedEvents[index].title = title
        storedEvents[index].start = start
        storedEvents[index].end = end
        return storedEvents[index]
    }

    public func remove(eventID: String) async throws {
        guard storedEvents.contains(where: { $0.id == eventID }) else { throw CalendarServiceError.eventNotFound }
        storedEvents.removeAll { $0.id == eventID }
    }

    public func syncReminders(_ todos: [TodoItem]) async throws -> [UUID: String] {
        guard remindersState == .granted else { throw CalendarServiceError.noAccess(remindersState) }
        var ids: [UUID: String] = [:]
        for todo in todos {
            let id = todo.reminderID ?? makeID()
            reminders[id] = todo
            ids[todo.id] = id
        }
        return ids
    }

    private func makeID() -> String {
        defer { nextID += 1 }
        return "memory-\(nextID)"
    }
}
