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

        public init(todos: [TodoItem] = [], proposals: [ProposedEvent] = []) {
            self.todos = todos
            self.proposals = proposals
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
/// Settings > Connections row binds to the same key.
public enum RemindersSyncSetting {
    public static let defaultsKey = "remindersSyncEnabled"

    public static func isEnabled(_ defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: defaultsKey)
    }

    public static func setEnabled(_ enabled: Bool, _ defaults: UserDefaults = .standard) {
        defaults.set(enabled, forKey: defaultsKey)
    }
}
