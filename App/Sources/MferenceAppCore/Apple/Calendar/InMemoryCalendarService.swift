import Foundation

/// A calendar that lives in memory: tests, previews and snapshots. Never
/// touches EventKit.
public actor InMemoryCalendarService: CalendarServicing {
    public private(set) var storedEvents: [CalendarEvent]
    public private(set) var reminders: [String: TodoItem] = [:]
    /// Which list each reminder is in.
    public private(set) var reminderListOf: [String: String] = [:]
    public private(set) var lists: [ReminderList]
    private var calendarState: PermissionState
    private var remindersState: PermissionState
    private let grantOnRequest: Bool
    /// false: like a Mac that shows no prompt, a request answers
    /// `.needsSystemSettings` and the state stays undecided.
    private let promptShown: Bool
    public private(set) var requests: [AppPermission] = []
    private var nextID = 1
    /// Tests: how long the next events read waits, like a slow EventKit read.
    private var nextReadDelay: Duration?

    public init(events: [CalendarEvent] = [], calendarAccess: PermissionState = .granted,
                remindersAccess: PermissionState = .notDetermined, grantOnRequest: Bool = true,
                promptShown: Bool = true,
                lists: [ReminderList] = [ReminderList(id: "list-reminders", title: "Reminders", isDefault: true)]) {
        self.promptShown = promptShown
        self.lists = lists
        self.storedEvents = events
        self.calendarState = calendarAccess
        self.remindersState = remindersAccess
        self.grantOnRequest = grantOnRequest
    }

    public func calendarAccess() async -> PermissionState { calendarState }

    /// Access granted in System Settings while the app was away.
    public func setCalendarAccess(_ state: PermissionState) {
        calendarState = state
    }

    public func requestCalendarAccess() async -> PermissionState {
        requests.append(.calendar)
        if calendarState == .notDetermined, !promptShown { return .needsSystemSettings }
        if calendarState == .notDetermined { calendarState = grantOnRequest ? .granted : .denied }
        return calendarState
    }

    public func remindersAccess() async -> PermissionState { remindersState }

    public private(set) var openedSettings: [AppPermission] = []

    public func openSystemSettings(for permission: AppPermission) async {
        openedSettings.append(permission)
    }

    public func requestRemindersAccess() async -> PermissionState {
        requests.append(.reminders)
        if remindersState == .notDetermined { remindersState = grantOnRequest ? .granted : .denied }
        return remindersState
    }

    /// Tests: the next events read waits `delay` before reading.
    public func delayNextRead(by delay: Duration) { nextReadDelay = delay }

    /// Tests: the delayed read has not started yet.
    public var isNextReadDelayed: Bool { nextReadDelay != nil }

    public func events(from start: Date, to end: Date) async throws -> [CalendarEvent] {
        if let delay = nextReadDelay {
            nextReadDelay = nil
            try await Task.sleep(for: delay)
        }
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

    public func syncReminders(_ todos: [TodoItem], to listID: String?) async throws -> [UUID: ReminderSyncOutcome] {
        guard remindersState == .granted else { throw CalendarServiceError.noAccess(remindersState) }
        var outcomes: [UUID: ReminderSyncOutcome] = [:]
        for todo in todos where !todo.unlinkedFromReminders {
            if let id = todo.reminderID {
                guard var stored = reminders[id] else { outcomes[todo.id] = .deletedInReminders; continue }
                let last = todo.lastSynced
                if last?.title != todo.title { stored.title = todo.title }
                if last?.isDone != todo.isDone { stored.isDone = todo.isDone }
                if last?.due != todo.due { stored.due = todo.due }
                if stored != reminders[id] { saves += 1 }
                reminders[id] = stored
                outcomes[todo.id] = .synced(reminderID: id, fields: stored.reminderFields)
            } else {
                let id = makeID()
                reminders[id] = todo
                reminderListOf[id] = lists.first { $0.id == listID }?.id ?? lists.first { $0.isDefault }?.id ?? ""
                saves += 1
                outcomes[todo.id] = .synced(reminderID: id, fields: todo.reminderFields)
            }
        }
        return outcomes
    }

    /// Reminders saved by syncs (a sync that changed nothing saves nothing).
    public private(set) var saves = 0

    public func reminderLists() async throws -> [ReminderList] {
        guard remindersState == .granted else { throw CalendarServiceError.noAccess(remindersState) }
        return lists
    }

    /// Tests: runs while a sync reads Reminders (a to-do deleted mid-sync).
    private var onOpenReminders: (@Sendable () async -> Void)?

    public func setOnOpenReminders(_ hook: (@Sendable () async -> Void)?) {
        onOpenReminders = hook
    }

    public func openReminders(in listIDs: Set<String>?) async throws -> [ReminderSnapshot] {
        guard remindersState == .granted else { throw CalendarServiceError.noAccess(remindersState) }
        if let hook = onOpenReminders {
            onOpenReminders = nil
            await hook()
        }
        return reminders.compactMap { id, item -> ReminderSnapshot? in
            let list = reminderListOf[id] ?? ""
            guard !item.isDone, listIDs?.contains(list) ?? true else { return nil }
            return ReminderSnapshot(id: id, title: item.title, isDone: false, due: item.due, listID: list,
                                    listTitle: lists.first { $0.id == list }?.title ?? "Reminders",
                                    created: item.createdAt)
        }.sorted { $0.id < $1.id }
    }

    public func removeReminders(_ ids: [String]) async throws {
        guard remindersState == .granted else { throw CalendarServiceError.noAccess(remindersState) }
        for id in ids {
            reminders[id] = nil
            reminderListOf[id] = nil
        }
    }

    /// Tests: the person adds a reminder in the Reminders app.
    @discardableResult
    public func addReminderInReminders(_ title: String, due: Date? = nil, list: String = "list-reminders") -> String {
        let id = makeID()
        reminders[id] = TodoItem(title: title, due: due, createdAt: .distantPast)
        reminderListOf[id] = list
        return id
    }

    /// Tests: the person edits or deletes a reminder in Reminders.
    public func editReminder(_ id: String, title: String? = nil, isDone: Bool? = nil) {
        if let title { reminders[id]?.title = title }
        if let isDone { reminders[id]?.isDone = isDone }
    }

    public func deleteReminder(_ id: String) {
        reminders[id] = nil
        reminderListOf[id] = nil
    }

    private func makeID() -> String {
        defer { nextID += 1 }
        return "memory-\(nextID)"
    }
}
