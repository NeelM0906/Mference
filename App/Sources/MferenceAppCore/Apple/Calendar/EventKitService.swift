import AppKit
import EventKit
import Foundation

/// Apple Calendar and Reminders through EventKit. Access is asked through the
/// `PermissionBroker` the first time it is needed (L6); Reminders only after
/// the person turned sync on (L9). Never used by the default test suite.
public actor EventKitService: CalendarServicing {
    public static let remindersListName = "Mference"

    private let broker: any PermissionBroker
    private var store: EKEventStore
    /// Entity types that were already allowed when `store` was made.
    private var storeHasAccess: [EKEntityType] = []

    public init(broker: any PermissionBroker) {
        self.broker = broker
        self.store = EKEventStore()
        self.storeHasAccess = [.event, .reminder].filter {
            Self.state(EKEventStore.authorizationStatus(for: $0)) == .granted
        }
    }

    // MARK: Access

    public func calendarAccess() async -> PermissionState {
        Self.state(EKEventStore.authorizationStatus(for: .event))
    }

    public func requestCalendarAccess() async -> PermissionState {
        await request(.calendar, entity: .event)
    }

    public func openSystemSettings(for permission: AppPermission) async {
        await broker.openSystemSettings(for: permission)
    }

    public func remindersAccess() async -> PermissionState {
        Self.state(EKEventStore.authorizationStatus(for: .reminder))
    }

    public func requestRemindersAccess() async -> PermissionState {
        await request(.reminders, entity: .reminder)
    }

    private func request(_ permission: AppPermission, entity: EKEntityType) async -> PermissionState {
        let before = Self.state(EKEventStore.authorizationStatus(for: entity))
        guard before != .granted else { return .granted }
        _ = await broker.request(permission)
        let after = Self.state(EKEventStore.authorizationStatus(for: entity))
        // A store made before access was granted does not see the data.
        if after == .granted, !storeHasAccess.contains(entity) {
            store = EKEventStore()
            storeHasAccess = [.event, .reminder].filter {
                Self.state(EKEventStore.authorizationStatus(for: $0)) == .granted
            }
        }
        return after
    }

    static func state(_ status: EKAuthorizationStatus) -> PermissionState {
        switch status {
        case .fullAccess: .granted
        case .notDetermined: .notDetermined
        case .denied, .restricted, .writeOnly: .needsSystemSettings
        @unknown default: .needsSystemSettings
        }
    }

    private func requireAccess(_ entity: EKEntityType) throws {
        let state = Self.state(EKEventStore.authorizationStatus(for: entity))
        guard state == .granted else { throw CalendarServiceError.noAccess(state) }
        // Access granted later (in System Settings, or by another part of
        // the app): a store made before that does not see the data.
        if !storeHasAccess.contains(entity) {
            store = EKEventStore()
            storeHasAccess = [.event, .reminder].filter {
                Self.state(EKEventStore.authorizationStatus(for: $0)) == .granted
            }
        }
    }

    // MARK: Events

    public func events(from start: Date, to end: Date) async throws -> [CalendarEvent] {
        try requireAccess(.event)
        let predicate = store.predicateForEvents(withStart: start, end: end, calendars: nil)
        return store.events(matching: predicate)
            .map(Self.event)
            .sorted { ($0.start, $0.title) < ($1.start, $1.title) }
    }

    public func add(_ proposal: ProposedEvent) async throws -> CalendarEvent {
        try requireAccess(.event)
        let event = EKEvent(eventStore: store)
        event.title = proposal.title
        event.startDate = proposal.start
        event.endDate = proposal.end
        let replaced = proposal.replaces.flatMap(resolve)
        if let replaced {
            event.calendar = replaced.calendar
            event.location = replaced.location
        } else {
            event.calendar = store.defaultCalendarForNewEvents
        }
        do {
            try store.save(event, span: .thisEvent, commit: false)
            if let replaced { try store.remove(replaced, span: .thisEvent, commit: false) }
            try store.commit()
        } catch {
            store.reset()
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
        return Self.event(event)
    }

    public func update(_ reference: EventReference, title: String, start: Date, end: Date) async throws -> CalendarEvent {
        try requireAccess(.event)
        guard let event = resolve(reference) else { throw CalendarServiceError.eventNotFound }
        event.title = title
        event.startDate = start
        event.endDate = end
        do { try store.save(event, span: .thisEvent, commit: true) } catch {
            store.reset()
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
        return Self.event(event)
    }

    public func remove(_ reference: EventReference) async throws {
        try requireAccess(.event)
        guard let event = resolve(reference) else { throw CalendarServiceError.eventNotFound }
        do { try store.remove(event, span: .thisEvent, commit: true) } catch {
            store.reset()
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
    }

    /// The exact occurrence: `calendarItem(withIdentifier:)` returns a
    /// series' first occurrence, so a repeating event is found by its
    /// occurrence date among that day's events.
    private func resolve(_ reference: EventReference) -> EKEvent? {
        guard let occurrence = reference.occurrenceDate else {
            return store.calendarItem(withIdentifier: reference.eventID) as? EKEvent
        }
        let predicate = store.predicateForEvents(withStart: occurrence.addingTimeInterval(-86_400),
                                                 end: occurrence.addingTimeInterval(86_400), calendars: nil)
        return store.events(matching: predicate).first {
            $0.calendarItemIdentifier == reference.eventID && $0.occurrenceDate == occurrence
        }
    }

    static func event(_ event: EKEvent) -> CalendarEvent {
        let repeats = event.hasRecurrenceRules || event.isDetached
        return CalendarEvent(id: event.calendarItemIdentifier,
                             occurrenceDate: repeats ? event.occurrenceDate : nil,
                             title: event.title ?? "Untitled",
                             start: event.startDate,
                             end: event.endDate,
                             isAllDay: event.isAllDay,
                             calendarName: event.calendar?.title ?? "Calendar",
                             calendarColor: event.calendar?.color.flatMap(hex),
                             location: event.location)
    }

    static func hex(_ color: NSColor) -> UInt32? {
        guard let rgb = color.usingColorSpace(.sRGB) else { return nil }
        let r = UInt32((rgb.redComponent * 255).rounded()), g = UInt32((rgb.greenComponent * 255).rounded())
        let b = UInt32((rgb.blueComponent * 255).rounded())
        return r << 16 | g << 8 | b
    }

    // MARK: Reminders

    public func syncReminders(_ todos: [TodoItem]) async throws -> [UUID: ReminderSyncOutcome] {
        try requireAccess(.reminder)
        var outcomes: [UUID: ReminderSyncOutcome] = [:]
        do {
            for todo in todos where !todo.unlinkedFromReminders {
                if let id = todo.reminderID {
                    guard let reminder = store.calendarItem(withIdentifier: id) as? EKReminder else {
                        // Deleted in Reminders: respect that, do not recreate.
                        outcomes[todo.id] = .deletedInReminders
                        continue
                    }
                    // Push only what changed in Mference since the last sync,
                    // so edits made in Reminders survive; notes are left alone.
                    let last = todo.lastSynced
                    if last?.title != todo.title { reminder.title = todo.title }
                    if last?.isDone != todo.isDone { reminder.isCompleted = todo.isDone }
                    if last?.due != todo.due { reminder.dueDateComponents = Self.dueComponents(todo.due) }
                    try store.save(reminder, commit: false)
                    outcomes[todo.id] = .synced(reminderID: id, fields: Self.fields(reminder))
                } else {
                    let reminder = EKReminder(eventStore: store)
                    reminder.calendar = try remindersList()
                    reminder.title = todo.title
                    reminder.isCompleted = todo.isDone
                    reminder.dueDateComponents = Self.dueComponents(todo.due)
                    reminder.notes = "From Mference · \(todo.source.kindLabel): \(todo.source.label)"
                    try store.save(reminder, commit: false)
                    outcomes[todo.id] = .synced(reminderID: reminder.calendarItemIdentifier, fields: Self.fields(reminder))
                }
            }
            try store.commit()
        } catch {
            store.reset()
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
        return outcomes
    }

    static func dueComponents(_ due: Date?) -> DateComponents? {
        due.map { Calendar.current.dateComponents([.year, .month, .day], from: $0) }
    }

    static func fields(_ reminder: EKReminder) -> TodoItem.ReminderFields {
        TodoItem.ReminderFields(title: reminder.title ?? "", isDone: reminder.isCompleted,
                                due: reminder.dueDateComponents.flatMap { Calendar.current.date(from: $0) })
    }

    private func remindersList() throws -> EKCalendar {
        if let existing = store.calendars(for: .reminder).first(where: { $0.title == Self.remindersListName }) {
            return existing
        }
        let list = EKCalendar(for: .reminder, eventStore: store)
        list.title = Self.remindersListName
        guard let source = store.defaultCalendarForNewReminders()?.source ?? store.sources.first(where: {
            $0.sourceType == .local || $0.sourceType == .calDAV
        }) else { throw CalendarServiceError.saveFailed("No Reminders account is set up on this Mac.") }
        list.source = source
        do { try store.saveCalendar(list, commit: true) } catch {
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
        return list
    }
}
