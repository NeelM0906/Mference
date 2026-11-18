import Foundation
import Observation

// The values Settings shows besides the model library. U2 (MemoryStore,
// preferences) and U4 (Apple connections) provide the live ones; proposal
// recorded in $APP/notes/shell.md.

/// One remembered thing (SwiftData `MemoryItem`).
public struct MemoryEntry: Sendable, Hashable, Identifiable {
    public var id: UUID
    public var text: String
    /// Where it was learned: "Learned from 14 sent emails", "From chat, Sep 10".
    public var source: String

    public init(id: UUID, text: String, source: String) {
        self.id = id
        self.text = text
        self.source = source
    }
}

@MainActor
public protocol MemoryListModel: AnyObject, Observable {
    var memories: [MemoryEntry] { get }
    func add(_ text: String)
    func update(_ id: UUID, text: String)
    func delete(_ id: UUID)
}

/// A connection to an Apple app (Phase B, wired by U4).
public enum ConnectionState: Sendable, Hashable {
    case notConnected
    case connected
    /// Access was denied or needs Full Disk Access: open System Settings.
    case needsSystemSettings
    /// Not built yet in this version.
    case later
}

public enum ConnectionKind: String, Sendable, Hashable, CaseIterable {
    case appleMail, appleCalendar
}

/// Switches and connections in Settings > Privacy and Connections.
@MainActor
public protocol AppSettingsModel: AnyObject, Observable {
    /// Off until the person turns it on (L7).
    var webSearchEnabled: Bool { get set }
    /// Turns off web search and mail sync; chat and scanning keep working.
    var offlineMode: Bool { get set }
    /// Remote images and trackers are removed before a message is shown.
    var blockRemoteMailContent: Bool { get set }
    /// Sync to-dos to Reminders; off by default, asks for Reminders when turned on (L9).
    var remindersSyncEnabled: Bool { get set }
    /// Use Apple's on-device model for small jobs when Apple Intelligence is on (L12).
    var useAppleModel: Bool { get set }
    func connectionState(_ kind: ConnectionKind) -> ConnectionState
    func connect(_ kind: ConnectionKind)
}
