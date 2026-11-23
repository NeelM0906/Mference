import Foundation

/// Which event, and for a repeating one which occurrence: every occurrence
/// of a series shares EventKit's identifier.
public struct EventReference: Sendable, Hashable, Codable {
    /// EventKit's `calendarItemIdentifier`.
    public var eventID: String
    /// The occurrence's original start, for repeating events; nil otherwise.
    public var occurrenceDate: Date?

    public init(eventID: String, occurrenceDate: Date? = nil) {
        self.eventID = eventID
        self.occurrenceDate = occurrenceDate
    }
}

/// One Apple Calendar event, as the week view shows it.
public struct CalendarEvent: Sendable, Hashable, Identifiable {
    /// EventKit's `calendarItemIdentifier` (shared by a series' occurrences).
    public var eventID: String
    /// The occurrence's original start, for repeating events; nil otherwise.
    public var occurrenceDate: Date?
    public var title: String
    public var start: Date
    public var end: Date
    public var isAllDay: Bool
    public var calendarName: String
    /// The calendar's color as 0xRRGGBB, if known.
    public var calendarColor: UInt32?
    public var location: String?
    /// From a calendar the person adds events to, not a subscribed holiday or
    /// birthdays calendar. Only these all-day events take a whole day.
    public var isUserCalendar: Bool

    public init(id: String, occurrenceDate: Date? = nil, title: String, start: Date, end: Date, isAllDay: Bool = false,
                calendarName: String = "Calendar", calendarColor: UInt32? = nil, location: String? = nil,
                isUserCalendar: Bool = true) {
        self.eventID = id
        self.occurrenceDate = occurrenceDate
        self.title = title
        self.start = start
        self.end = end
        self.isAllDay = isAllDay
        self.calendarName = calendarName
        self.calendarColor = calendarColor
        self.location = location
        self.isUserCalendar = isUserCalendar
    }

    public var reference: EventReference { EventReference(eventID: eventID, occurrenceDate: occurrenceDate) }

    /// Unique per occurrence: the identifier, plus the occurrence date for
    /// repeating events.
    public var id: String {
        occurrenceDate.map { "\(eventID)@\(Int($0.timeIntervalSince1970))" } ?? eventID
    }
}

/// Where a proposal or a to-do came from, for its source link.
public struct ItemSource: Sendable, Hashable, Codable {
    public enum Kind: String, Sendable, Hashable, Codable { case mail, meeting, scan, manual, reminders }

    public var kind: Kind
    /// "Northside Dental", "Weekly product sync".
    public var label: String
    /// Mail's Message-ID, or an event identifier, to open the source.
    public var reference: String?

    public init(kind: Kind, label: String, reference: String? = nil) {
        self.kind = kind
        self.label = label
        self.reference = reference
    }

    /// "Mail", "Meeting".
    public var kindLabel: String {
        switch kind {
        case .mail: "Mail"
        case .meeting: "Meeting"
        case .scan: "Scan"
        case .manual: "Added by you"
        case .reminders: "Reminders"
        }
    }
}

/// An event Mference suggests (from a mail, say), drawn dashed until the
/// person adds it. Nothing changes in Calendar until they click Add.
public struct ProposedEvent: Sendable, Hashable, Identifiable, Codable {
    public var id: UUID
    public var title: String
    public var start: Date
    public var end: Date
    public var source: ItemSource
    /// The event this one moves, if it reschedules one (shown struck through).
    public var replacesEventID: String?
    /// For a repeating event, the occurrence it moves.
    public var replacesOccurrence: Date?
    /// One line under the question ("From Priya's email this morning.").
    public var note: String?

    public init(id: UUID = UUID(), title: String, start: Date, end: Date, source: ItemSource,
                replacesEventID: String? = nil, replacesOccurrence: Date? = nil, note: String? = nil) {
        self.id = id
        self.title = title
        self.start = start
        self.end = end
        self.source = source
        self.replacesEventID = replacesEventID
        self.replacesOccurrence = replacesOccurrence
        self.note = note
    }

    /// The occurrence this proposal moves, if any.
    public var replaces: EventReference? {
        replacesEventID.map { EventReference(eventID: $0, occurrenceDate: replacesOccurrence) }
    }
}

/// A to-do in the side panel.
public struct TodoItem: Sendable, Hashable, Identifiable, Codable {
    public var id: UUID
    public var title: String
    public var isDone: Bool
    public var due: Date?
    public var source: ItemSource
    /// The matching Apple Reminder once synced.
    public var reminderID: String?
    /// What Mference last pushed to Reminders: only fields changed here since
    /// then are pushed again, so edits made in Reminders survive.
    public var lastSynced: ReminderFields?
    /// The reminder was deleted in Reminders: it is not created again.
    public var unlinkedFromReminders: Bool = false
    public var createdAt: Date

    /// The fields Mference owns in a reminder.
    public struct ReminderFields: Sendable, Hashable, Codable {
        public var title: String
        public var isDone: Bool
        public var due: Date?

        public init(title: String, isDone: Bool, due: Date?) {
            self.title = title
            self.isDone = isDone
            self.due = due
        }
    }

    public var reminderFields: ReminderFields { ReminderFields(title: title, isDone: isDone, due: due) }

    public init(id: UUID = UUID(), title: String, isDone: Bool = false, due: Date? = nil,
                source: ItemSource = ItemSource(kind: .manual, label: "Added by you"),
                reminderID: String? = nil, createdAt: Date = .now) {
        self.id = id
        self.title = title
        self.isDone = isDone
        self.due = due
        self.source = source
        self.reminderID = reminderID
        self.createdAt = createdAt
    }
}

public enum CalendarServiceError: Error, Sendable, Hashable {
    /// Access was not granted; `state` says whether System Settings is needed.
    case noAccess(PermissionState)
    case eventNotFound
    case saveFailed(String)

    public var userMessage: String {
        switch self {
        case .noAccess(.needsSystemSettings), .noAccess(.denied):
            "Mference cannot see your calendar. Turn it on in System Settings > Privacy & Security."
        case .noAccess: "Mference needs access to your calendar first."
        case .eventNotFound: "That event is no longer in Calendar."
        case .saveFailed(let detail): "Calendar could not save the change: \(detail)"
        }
    }
}

/// Apple Calendar and Reminders, behind a protocol so tests never touch the
/// person's data.
public protocol CalendarServicing: Sendable {
    /// Calendar access without prompting.
    func calendarAccess() async -> PermissionState
    /// Asks for Calendar access through the broker (reason first).
    func requestCalendarAccess() async -> PermissionState
    /// Reminders access without prompting.
    func remindersAccess() async -> PermissionState
    /// Asks for Reminders access through the broker; only after the person
    /// turned Reminders sync on.
    func requestRemindersAccess() async -> PermissionState
    /// Opens the Calendars (or Reminders) pane of System Settings, after a
    /// denial.
    func openSystemSettings(for permission: AppPermission) async

    func events(from start: Date, to end: Date) async throws -> [CalendarEvent]
    /// Creates the event (and removes the one it replaces, if any).
    func add(_ proposal: ProposedEvent) async throws -> CalendarEvent
    /// Changes this occurrence only (`.thisEvent`).
    func update(_ event: EventReference, title: String, start: Date, end: Date) async throws -> CalendarEvent
    /// Deletes this occurrence only.
    func remove(_ event: EventReference) async throws

    /// Creates a Reminder for each new to-do in `listID` (nil: the default
    /// Reminders list) and, for linked ones, pushes only the fields Mference
    /// changed since the last sync, saving nothing when nothing changed (a
    /// save would fire `EKEventStoreChanged` and sync again); never touches
    /// notes after creation, never recreates a reminder deleted in Reminders.
    func syncReminders(_ todos: [TodoItem], to listID: String?) async throws -> [UUID: ReminderSyncOutcome]
    /// The Reminders lists on this Mac, the default one first.
    func reminderLists() async throws -> [ReminderList]
    /// Open (not completed) reminders in these lists (nil: every list).
    func openReminders(in listIDs: Set<String>?) async throws -> [ReminderSnapshot]
    /// Deletes these reminders (a to-do deleted in Mference); ones already
    /// gone are skipped.
    func removeReminders(_ ids: [String]) async throws
}

/// One Reminders list, for the Settings choices.
public struct ReminderList: Sendable, Hashable, Identifiable {
    /// EventKit's `calendarIdentifier`.
    public var id: String
    public var title: String
    /// 0xRRGGBB, if known.
    public var color: UInt32?
    public var isDefault: Bool

    public init(id: String, title: String, color: UInt32? = nil, isDefault: Bool = false) {
        self.id = id
        self.title = title
        self.color = color
        self.isDefault = isDefault
    }
}

/// One reminder as read from Reminders.
public struct ReminderSnapshot: Sendable, Hashable, Identifiable {
    /// EventKit's `calendarItemIdentifier`.
    public var id: String
    public var title: String
    public var isDone: Bool
    public var due: Date?
    public var listID: String
    public var listTitle: String
    public var created: Date?

    public init(id: String, title: String, isDone: Bool = false, due: Date? = nil, listID: String,
                listTitle: String, created: Date? = nil) {
        self.id = id
        self.title = title
        self.isDone = isDone
        self.due = due
        self.listID = listID
        self.listTitle = listTitle
        self.created = created
    }

    public var fields: TodoItem.ReminderFields { TodoItem.ReminderFields(title: title, isDone: isDone, due: due) }
}

/// What happened to one to-do in a Reminders sync.
public enum ReminderSyncOutcome: Sendable, Hashable {
    /// Linked to this reminder; these fields are now in Reminders.
    case synced(reminderID: String, fields: TodoItem.ReminderFields)
    /// The reminder was deleted in Reminders; the to-do stays here only.
    case deletedInReminders
}
