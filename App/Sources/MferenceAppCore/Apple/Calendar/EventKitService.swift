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

    public init(broker: any PermissionBroker) {
        self.broker = broker
        self.store = EKEventStore()
    }

    // MARK: Access

    public func calendarAccess() async -> PermissionState {
        Self.state(EKEventStore.authorizationStatus(for: .event))
    }

    public func requestCalendarAccess() async -> PermissionState {
        await request(.calendar, entity: .event)
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
        if after == .granted { store = EKEventStore() }
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
        if let replaced = proposal.replacesEventID.flatMap(store.calendarItem(withIdentifier:)) as? EKEvent {
            event.calendar = replaced.calendar
            event.location = replaced.location
        } else {
            event.calendar = store.defaultCalendarForNewEvents
        }
        do {
            try store.save(event, span: .thisEvent, commit: false)
            if let replaced = proposal.replacesEventID.flatMap(store.calendarItem(withIdentifier:)) as? EKEvent {
                try store.remove(replaced, span: .thisEvent, commit: false)
            }
            try store.commit()
        } catch {
            store.reset()
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
        return Self.event(event)
    }

    public func update(eventID: String, title: String, start: Date, end: Date) async throws -> CalendarEvent {
        try requireAccess(.event)
        guard let event = store.calendarItem(withIdentifier: eventID) as? EKEvent else {
            throw CalendarServiceError.eventNotFound
        }
        event.title = title
        event.startDate = start
        event.endDate = end
        do { try store.save(event, span: .thisEvent, commit: true) } catch {
            store.reset()
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
        return Self.event(event)
    }

    public func remove(eventID: String) async throws {
        try requireAccess(.event)
        guard let event = store.calendarItem(withIdentifier: eventID) as? EKEvent else {
            throw CalendarServiceError.eventNotFound
        }
        do { try store.remove(event, span: .thisEvent, commit: true) } catch {
            store.reset()
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
    }

    static func event(_ event: EKEvent) -> CalendarEvent {
        CalendarEvent(id: event.calendarItemIdentifier,
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

    public func syncReminders(_ todos: [TodoItem]) async throws -> [UUID: String] {
        try requireAccess(.reminder)
        let list = try remindersList()
        var ids: [UUID: String] = [:]
        do {
            for todo in todos {
                let reminder = todo.reminderID.flatMap(store.calendarItem(withIdentifier:)) as? EKReminder
                    ?? EKReminder(eventStore: store)
                if reminder.calendar == nil { reminder.calendar = list }
                reminder.title = todo.title
                reminder.isCompleted = todo.isDone
                reminder.dueDateComponents = todo.due.map {
                    Calendar.current.dateComponents([.year, .month, .day], from: $0)
                }
                reminder.notes = "From Mference · \(todo.source.kindLabel): \(todo.source.label)"
                try store.save(reminder, commit: false)
                ids[todo.id] = reminder.calendarItemIdentifier
            }
            try store.commit()
        } catch {
            store.reset()
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
        return ids
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
