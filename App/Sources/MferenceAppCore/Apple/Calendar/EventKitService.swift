import CoreGraphics
import EventKit
import Foundation
#if os(macOS)
import AppKit
#endif

/// Apple Calendar and Reminders through EventKit. Access is asked through the
/// `PermissionBroker` the first time it is needed (L6); Reminders only after
/// the person turned sync on (L9). Never used by the default test suite.
public actor EventKitService: CalendarServicing {
    public static let remindersListName = "Mference"

    private let broker: any PermissionBroker
    private var store: EKEventStore
    /// True when `store` is the broker's (the one that asked): it sees a
    /// grant at once and is reset, never replaced.
    private let sharesBrokerStore: Bool
    /// Entity types that were already allowed when `store` was last made or reset.
    private var storeHasAccess: [EKEntityType] = []
    private let authorization: @Sendable (EKEntityType) -> EKAuthorizationStatus

    /// Uses the live broker's event store (`LivePermissionBroker.eventStore`)
    /// when there is one, so the store that asked is the store that reads.
    /// `authorization` reads EventKit's status; tests pass their own.
    public init(broker: any PermissionBroker, eventStore: EventStoreBox? = nil,
                authorization: @escaping @Sendable (EKEntityType) -> EKAuthorizationStatus = {
                    EKEventStore.authorizationStatus(for: $0)
                }) {
        self.broker = broker
        self.authorization = authorization
        let shared = eventStore?.store ?? (broker as? LivePermissionBroker)?.eventStore
        self.store = shared ?? EKEventStore()
        self.sharesBrokerStore = shared != nil
        self.storeHasAccess = [.event, .reminder].filter { Self.state(authorization($0)) == .granted }
    }

    // MARK: Access

    public func calendarAccess() async -> PermissionState {
        Self.state(authorization(.event))
    }

    public func requestCalendarAccess() async -> PermissionState {
        await request(.calendar, entity: .event)
    }

    public func openSystemSettings(for permission: AppPermission) async {
        await broker.openSystemSettings(for: permission)
    }

    public func remindersAccess() async -> PermissionState {
        Self.state(authorization(.reminder))
    }

    public func requestRemindersAccess() async -> PermissionState {
        await request(.reminders, entity: .reminder)
    }

    func request(_ permission: AppPermission, entity: EKEntityType) async -> PermissionState {
        let before = Self.state(authorization(entity))
        guard before != .granted else { return .granted }
        let answer = await broker.request(permission)
        // The prompt's own answer counts even if the status read lags it.
        let after = answer == .granted ? .granted : Self.state(authorization(entity))
        // Still undecided: the broker's answer stands (.needsSystemSettings
        // when macOS showed no prompt, .notDetermined when the person
        // declined the reason), so the caller points to System Settings.
        if after == .notDetermined { return answer }
        if after == .granted { refreshAfterGrant(entity) }
        return after
    }

    /// A store made before access was granted does not see the data: the
    /// broker's store is reset, a private one replaced.
    private func refreshAfterGrant(_ entity: EKEntityType) {
        guard !storeHasAccess.contains(entity) else { return }
        if sharesBrokerStore { store.reset() } else { store = EKEventStore() }
        storeHasAccess = [.event, .reminder].filter { $0 == entity || Self.state(authorization($0)) == .granted }
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
        let state = Self.state(authorization(entity))
        guard state == .granted else { throw CalendarServiceError.noAccess(state) }
        // Access granted later (in System Settings, or by another part of
        // the app): a store made before that does not see the data.
        refreshAfterGrant(entity)
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
                             calendarColor: color(of: event.calendar),
                             location: event.location)
    }

    /// The calendar's color as 0xRRGGBB: through `NSColor` on the Mac, through
    /// `CGColor` on the iPhone (EventKit has no `color` there).
    static func color(of calendar: EKCalendar?) -> UInt32? {
        #if os(macOS)
        calendar?.color.flatMap(hex)
        #else
        calendar?.cgColor.flatMap(hex)
        #endif
    }

    #if os(macOS)
    static func hex(_ color: NSColor) -> UInt32? {
        guard let rgb = color.usingColorSpace(.sRGB) else { return nil }
        let r = UInt32((rgb.redComponent * 255).rounded()), g = UInt32((rgb.greenComponent * 255).rounded())
        let b = UInt32((rgb.blueComponent * 255).rounded())
        return r << 16 | g << 8 | b
    }
    #endif

    /// sRGB 0xRRGGBB; channels outside sRGB are clamped.
    static func hex(_ color: CGColor) -> UInt32? {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let rgb = color.converted(to: space, intent: .defaultIntent, options: nil),
              let components = rgb.components, components.count >= 3 else { return nil }
        let channel = { (value: CGFloat) in UInt32((min(max(value, 0), 1) * 255).rounded()) }
        return channel(components[0]) << 16 | channel(components[1]) << 8 | channel(components[2])
    }

    // MARK: Reminders

    public func syncReminders(_ todos: [TodoItem], to listID: String?) async throws -> [UUID: ReminderSyncOutcome] {
        try requireAccess(.reminder)
        var outcomes: [UUID: ReminderSyncOutcome] = [:]
        var saved = false
        do {
            for todo in todos where !todo.unlinkedFromReminders {
                if let id = todo.reminderID {
                    guard let reminder = store.calendarItem(withIdentifier: id) as? EKReminder else {
                        // Deleted in Reminders: the to-do goes too.
                        outcomes[todo.id] = .deletedInReminders
                        continue
                    }
                    // Push only what changed in Mference since the last sync,
                    // so edits made in Reminders survive; notes are left alone.
                    // Nothing changed: nothing is saved, so no change notice
                    // comes back to start another sync.
                    let last = todo.lastSynced
                    var changed = false
                    if last?.title != todo.title, reminder.title != todo.title {
                        reminder.title = todo.title
                        changed = true
                    }
                    if last?.isDone != todo.isDone, reminder.isCompleted != todo.isDone {
                        reminder.isCompleted = todo.isDone
                        changed = true
                    }
                    if last?.due != todo.due, Self.fields(reminder).due != todo.due {
                        reminder.dueDateComponents = Self.dueComponents(todo.due)
                        changed = true
                    }
                    if changed {
                        try store.save(reminder, commit: false)
                        saved = true
                    }
                    outcomes[todo.id] = .synced(reminderID: id, fields: Self.fields(reminder))
                } else {
                    let reminder = EKReminder(eventStore: store)
                    reminder.calendar = try remindersList(listID)
                    reminder.title = todo.title
                    reminder.isCompleted = todo.isDone
                    reminder.dueDateComponents = Self.dueComponents(todo.due)
                    if todo.source.kind != .manual {
                        reminder.notes = "From Mference · \(todo.source.kindLabel): \(todo.source.label)"
                    }
                    try store.save(reminder, commit: false)
                    saved = true
                    outcomes[todo.id] = .synced(reminderID: reminder.calendarItemIdentifier, fields: Self.fields(reminder))
                }
            }
            if saved { try store.commit() }
        } catch {
            store.reset()
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
        return outcomes
    }

    public func reminderLists() async throws -> [ReminderList] {
        try requireAccess(.reminder)
        let defaultID = store.defaultCalendarForNewReminders()?.calendarIdentifier
        return store.calendars(for: .reminder)
            .map { ReminderList(id: $0.calendarIdentifier, title: $0.title, color: Self.color(of: $0),
                                isDefault: $0.calendarIdentifier == defaultID) }
            .sorted { ($0.isDefault ? 0 : 1, $0.title) < ($1.isDefault ? 0 : 1, $1.title) }
    }

    public func openReminders(in listIDs: Set<String>?) async throws -> [ReminderSnapshot] {
        try requireAccess(.reminder)
        var lists: [EKCalendar]?
        if let listIDs {
            lists = store.calendars(for: .reminder).filter { listIDs.contains($0.calendarIdentifier) }
            if lists?.isEmpty == true { return [] }
        }
        let predicate = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: lists)
        return await withCheckedContinuation { continuation in
            store.fetchReminders(matching: predicate) { reminders in
                continuation.resume(returning: (reminders ?? []).map(Self.snapshot))
            }
        }
    }

    public func removeReminders(_ ids: [String]) async throws {
        try requireAccess(.reminder)
        var removed = false
        do {
            for id in ids {
                guard let reminder = store.calendarItem(withIdentifier: id) as? EKReminder else { continue }
                try store.remove(reminder, commit: false)
                removed = true
            }
            if removed { try store.commit() }
        } catch {
            store.reset()
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
    }

    static func snapshot(_ reminder: EKReminder) -> ReminderSnapshot {
        let fields = fields(reminder)
        return ReminderSnapshot(id: reminder.calendarItemIdentifier, title: fields.title, isDone: fields.isDone,
                                due: fields.due, listID: reminder.calendar?.calendarIdentifier ?? "",
                                listTitle: reminder.calendar?.title ?? "Reminders", created: reminder.creationDate)
    }

    static func dueComponents(_ due: Date?) -> DateComponents? {
        due.map { Calendar.current.dateComponents([.year, .month, .day], from: $0) }
    }

    static func fields(_ reminder: EKReminder) -> TodoItem.ReminderFields {
        TodoItem.ReminderFields(title: reminder.title ?? "", isDone: reminder.isCompleted,
                                due: reminder.dueDateComponents.flatMap { Calendar.current.date(from: $0) })
    }

    /// The chosen list; else the default Reminders list; else a "Mference"
    /// list made in the default account.
    private func remindersList(_ listID: String?) throws -> EKCalendar {
        if let listID, let chosen = store.calendar(withIdentifier: listID), chosen.allowsContentModifications {
            return chosen
        }
        if let fallback = store.defaultCalendarForNewReminders() { return fallback }
        if let existing = store.calendars(for: .reminder).first(where: { $0.title == Self.remindersListName }) {
            return existing
        }
        let list = EKCalendar(for: .reminder, eventStore: store)
        list.title = Self.remindersListName
        guard let source = store.sources.first(where: {
            $0.sourceType == .local || $0.sourceType == .calDAV
        }) else { throw CalendarServiceError.saveFailed("No Reminders account is set up on this Mac.") }
        list.source = source
        do { try store.saveCalendar(list, commit: true) } catch {
            throw CalendarServiceError.saveFailed(error.localizedDescription)
        }
        return list
    }
}
