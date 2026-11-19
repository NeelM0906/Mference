import Foundation

/// To-dos and pending event proposals, kept in one small JSON file under
/// Application Support (the A0 schema has no to-do model; this file is owned
/// by Apple/Calendar and can move into SwiftData through a schema PR later).
public final class PlannerStore: @unchecked Sendable {
    public struct Contents: Codable, Sendable, Hashable {
        public var version = 1
        public var todos: [TodoItem] = []
        public var proposals: [ProposedEvent] = []
        /// Mail messages already turned into proposals or to-dos, so a
        /// dismissed one does not come back.
        public var handledSources: Set<String> = []
        /// Reminders of to-dos deleted here, deleted in Reminders at the
        /// next sync (a delete in one app deletes in both).
        public var deletedReminderIDs: Set<String> = []
        /// When the last Reminders sync finished.
        public var lastReminderSync: Date?

        public init(todos: [TodoItem] = [], proposals: [ProposedEvent] = []) {
            self.todos = todos
            self.proposals = proposals
        }

        enum CodingKeys: String, CodingKey {
            case version, todos, proposals, handledSources, deletedReminderIDs, lastReminderSync
        }

        /// Fields added later may be missing from an older file: they start
        /// empty instead of failing the whole file (which would drop the
        /// to-dos).
        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            version = try container.decodeIfPresent(Int.self, forKey: .version) ?? 1
            todos = try container.decodeIfPresent([TodoItem].self, forKey: .todos) ?? []
            proposals = try container.decodeIfPresent([ProposedEvent].self, forKey: .proposals) ?? []
            handledSources = try container.decodeIfPresent(Set<String>.self, forKey: .handledSources) ?? []
            deletedReminderIDs = try container.decodeIfPresent(Set<String>.self, forKey: .deletedReminderIDs) ?? []
            lastReminderSync = try container.decodeIfPresent(Date.self, forKey: .lastReminderSync)
        }
    }

    private let url: URL?
    private let lock = NSLock()
    private var contents: Contents

    /// `url` nil keeps everything in memory (tests, previews).
    public init(url: URL?, initial: Contents = Contents()) {
        self.url = url
        if let url, let data = try? Data(contentsOf: url),
           let decoded = try? JSONDecoder().decode(Contents.self, from: data) {
            contents = decoded
        } else {
            contents = initial
        }
    }

    public static func live() -> PlannerStore {
        PlannerStore(url: AppStorageLocation.applicationSupport.appending(path: "Planner.json"))
    }

    public var snapshot: Contents { lock.withLock { contents } }

    public func update(_ change: (inout Contents) -> Void) throws {
        let data: Data? = try lock.withLock {
            change(&contents)
            guard url != nil else { return nil }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            return try encoder.encode(contents)
        }
        if let url, let data {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
        }
    }
}

/// The Settings switch for Reminders sync (L9: off by default). U1's
/// Settings > Connections row binds to the same key. Also which lists are
/// read and which list new to-dos go to.
public enum RemindersSyncSetting {
    public static let defaultsKey = "remindersSyncEnabled"
    public static let listsKey = "remindersSyncListIDs"
    public static let targetKey = "remindersSyncTargetListID"

    /// The lists whose reminders show as to-dos; nil means every list.
    public static func listIDs(_ defaults: UserDefaults = .standard) -> Set<String>? {
        (defaults.array(forKey: listsKey) as? [String]).map(Set.init)
    }

    public static func setListIDs(_ ids: Set<String>?, _ defaults: UserDefaults = .standard) {
        if let ids { defaults.set(ids.sorted(), forKey: listsKey) } else { defaults.removeObject(forKey: listsKey) }
    }

    /// The list new to-dos go to; nil means the default Reminders list.
    public static func targetListID(_ defaults: UserDefaults = .standard) -> String? {
        defaults.string(forKey: targetKey)
    }

    public static func setTargetListID(_ id: String?, _ defaults: UserDefaults = .standard) {
        if let id { defaults.set(id, forKey: targetKey) } else { defaults.removeObject(forKey: targetKey) }
    }

    public static func isEnabled(_ defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: defaultsKey)
    }

    public static func setEnabled(_ enabled: Bool, _ defaults: UserDefaults = .standard) {
        defaults.set(enabled, forKey: defaultsKey)
    }
}
