import Foundation

/// One Apple Calendar event, as the week view shows it.
public struct CalendarEvent: Sendable, Hashable, Identifiable {
    /// EventKit's `calendarItemIdentifier`.
    public var id: String
    public var title: String
    public var start: Date
    public var end: Date
    public var isAllDay: Bool
    public var calendarName: String
    /// The calendar's color as 0xRRGGBB, if known.
    public var calendarColor: UInt32?
    public var location: String?

    public init(id: String, title: String, start: Date, end: Date, isAllDay: Bool = false,
                calendarName: String = "Calendar", calendarColor: UInt32? = nil, location: String? = nil) {
        self.id = id
        self.title = title
        self.start = start
        self.end = end
        self.isAllDay = isAllDay
        self.calendarName = calendarName
        self.calendarColor = calendarColor
        self.location = location
    }
}

/// Where a proposal or a to-do came from, for its source link.
public struct ItemSource: Sendable, Hashable, Codable {
    public enum Kind: String, Sendable, Hashable, Codable { case mail, meeting, scan, manual }

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
    /// One line under the question ("From Priya's email this morning.").
    public var note: String?

    public init(id: UUID = UUID(), title: String, start: Date, end: Date, source: ItemSource,
                replacesEventID: String? = nil, note: String? = nil) {
        self.id = id
        self.title = title
        self.start = start
        self.end = end
        self.source = source
        self.replacesEventID = replacesEventID
        self.note = note
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
    public var createdAt: Date

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

    func events(from start: Date, to end: Date) async throws -> [CalendarEvent]
    /// Creates the event (and removes the one it replaces, if any).
    func add(_ proposal: ProposedEvent) async throws -> CalendarEvent
    func update(eventID: String, title: String, start: Date, end: Date) async throws -> CalendarEvent
    func remove(eventID: String) async throws

    /// Creates or updates one Reminder per to-do in the "Mference" list;
    /// returns to-do id to reminder id.
    func syncReminders(_ todos: [TodoItem]) async throws -> [UUID: String]
}
