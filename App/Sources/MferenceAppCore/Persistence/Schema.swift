import Foundation
import SwiftData

// SHARED CONTRACT (App Program spec sections 4.4 and 8). The SwiftData schema
// for everything the app stores under ~/Library/Application Support/Mference/.
// U2 owns this file. Adding, renaming or removing a stored property is a new
// `VersionedSchema` plus a migration stage, through one PR the coordinator
// approves. Rich values (citations, steps, scan fields) are stored as encoded
// `Data` with their Codable types defined next to the feature that owns them.

public enum MferenceSchemaV1: VersionedSchema {
    public static let versionIdentifier = Schema.Version(1, 0, 0)

    public static var models: [any PersistentModel.Type] {
        [Conversation.self, StoredMessage.self, MemoryItem.self,
         ScanRecord.self, MailBucket.self, MailBucketAssignment.self]
    }

    /// One chat in the sidebar.
    @Model
    public final class Conversation {
        @Attribute(.unique) public var id: UUID
        public var title: String
        public var createdAt: Date
        public var updatedAt: Date
        /// `AppModel.rawValue` the chat uses (the picker is per chat, L2).
        public var modelRawValue: String
        /// Web search on for this chat (L7: off until turned on).
        public var webEnabled: Bool
        /// Shown under "Private" in the sidebar: never used for memory.
        public var isPrivate: Bool
        public var isPinned: Bool
        @Relationship(deleteRule: .cascade, inverse: \StoredMessage.conversation)
        public var messages: [StoredMessage] = []

        public init(id: UUID = UUID(), title: String = "New chat", createdAt: Date = .now,
                    modelRawValue: String = AppModel.quick.rawValue, webEnabled: Bool = false,
                    isPrivate: Bool = false, isPinned: Bool = false) {
            self.id = id
            self.title = title
            self.createdAt = createdAt
            self.updatedAt = createdAt
            self.modelRawValue = modelRawValue
            self.webEnabled = webEnabled
            self.isPrivate = isPrivate
            self.isPinned = isPinned
        }
    }

    /// One message. Saved as soon as it is final, so a crash loses at most the
    /// answer being streamed (spec 13).
    @Model
    public final class StoredMessage {
        @Attribute(.unique) public var id: UUID
        public var conversation: Conversation?
        /// Order within the conversation.
        public var index: Int
        /// `ChatTurn.Role.rawValue`.
        public var roleRawValue: String
        public var content: String
        /// Thinking text, replayed in history (engine contract).
        public var reasoning: String?
        /// JSON `[ToolCall]` the assistant made.
        public var toolCallsJSON: Data?
        /// For tool results: the call answered and the tool's name.
        public var toolCallID: String?
        public var toolName: String?
        /// JSON citations shown as numbered sources (owned by Chat/Web, U3).
        public var citationsJSON: Data?
        /// JSON steps for the steps disclosure (owned by Chat, U3).
        public var stepsJSON: Data?
        /// `AppModel.rawValue` that wrote an assistant message.
        public var modelRawValue: String?
        /// `GenerationFinish.rawValue`, or "cancelled" / "failed".
        public var finishRawValue: String?
        public var promptTokens: Int?
        public var completionTokens: Int?
        public var createdAt: Date

        public init(id: UUID = UUID(), index: Int, role: ChatTurn.Role, content: String,
                    reasoning: String? = nil, modelRawValue: String? = nil, createdAt: Date = .now) {
            self.id = id
            self.index = index
            self.roleRawValue = role.rawValue
            self.content = content
            self.reasoning = reasoning
            self.modelRawValue = modelRawValue
            self.createdAt = createdAt
        }
    }

    /// One thing remembered (Settings > Memory: add, edit, delete, where learned).
    @Model
    public final class MemoryItem {
        @Attribute(.unique) public var id: UUID
        public var text: String
        public var createdAt: Date
        public var updatedAt: Date
        /// Where it was learned: "chat", "mail", "scan", "manual".
        public var sourceKind: String
        /// The conversation, mail or scan it came from, if any.
        public var sourceID: UUID?
        /// A short label for "where learned" ("Chat: Lisbon trip").
        public var sourceLabel: String?
        public var isEnabled: Bool

        public init(id: UUID = UUID(), text: String, sourceKind: String = "manual",
                    sourceID: UUID? = nil, sourceLabel: String? = nil, createdAt: Date = .now) {
            self.id = id
            self.text = text
            self.createdAt = createdAt
            self.updatedAt = createdAt
            self.sourceKind = sourceKind
            self.sourceID = sourceID
            self.sourceLabel = sourceLabel
            self.isEnabled = true
        }
    }

    /// One confirmed scan (receipt or document), U4.
    @Model
    public final class ScanRecord {
        @Attribute(.unique) public var id: UUID
        public var createdAt: Date
        /// "receipt" or "document".
        public var kind: String
        public var title: String
        /// Recognized full text.
        public var recognizedText: String
        /// JSON fields the person confirmed (owned by Apple/Scan, U4).
        public var fieldsJSON: Data?
        /// Copy of the scanned image under Application Support, if kept.
        public var imageRelativePath: String?
        public var isConfirmed: Bool

        public init(id: UUID = UUID(), kind: String, title: String, recognizedText: String,
                    createdAt: Date = .now) {
            self.id = id
            self.createdAt = createdAt
            self.kind = kind
            self.title = title
            self.recognizedText = recognizedText
            self.isConfirmed = false
        }
    }

    /// A mail bucket (renamable, L8), U4.
    @Model
    public final class MailBucket {
        @Attribute(.unique) public var id: UUID
        /// Stable key for the built-in buckets ("needs-reply", "updates", ...).
        public var key: String
        public var name: String
        public var sortOrder: Int

        public init(id: UUID = UUID(), key: String, name: String, sortOrder: Int) {
            self.id = id
            self.key = key
            self.name = name
            self.sortOrder = sortOrder
        }
    }

    /// Which bucket a mail message is in. Mail content stays in Mail's store.
    @Model
    public final class MailBucketAssignment {
        /// Mail's Message-ID header.
        @Attribute(.unique) public var messageID: String
        public var bucketKey: String
        /// Set by the person (sticks) or by the sorter (may be redone).
        public var isManual: Bool
        public var assignedAt: Date

        public init(messageID: String, bucketKey: String, isManual: Bool, assignedAt: Date = .now) {
            self.messageID = messageID
            self.bucketKey = bucketKey
            self.isManual = isManual
            self.assignedAt = assignedAt
        }
    }
}

public typealias Conversation = MferenceSchemaV1.Conversation
public typealias StoredMessage = MferenceSchemaV1.StoredMessage
public typealias MemoryItem = MferenceSchemaV1.MemoryItem
public typealias ScanRecord = MferenceSchemaV1.ScanRecord
public typealias MailBucket = MferenceSchemaV1.MailBucket
public typealias MailBucketAssignment = MferenceSchemaV1.MailBucketAssignment

public enum MferenceMigrationPlan: SchemaMigrationPlan {
    public static var schemas: [any VersionedSchema.Type] { [MferenceSchemaV1.self] }
    public static var stages: [MigrationStage] { [] }
}

/// Where the app keeps its data (spec 4.4).
public enum AppStorageLocation {
    public static var applicationSupport: URL {
        URL.applicationSupportDirectory.appending(path: "Mference", directoryHint: .isDirectory)
    }

    public static var storeURL: URL {
        applicationSupport.appending(path: "Mference.store")
    }
}

public enum AppModelContainer {
    /// The on-disk container. Creates the directory if needed.
    public static func make(at url: URL = AppStorageLocation.storeURL) throws -> ModelContainer {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        let configuration = ModelConfiguration(schema: Schema(versionedSchema: MferenceSchemaV1.self), url: url)
        return try ModelContainer(for: Schema(versionedSchema: MferenceSchemaV1.self),
                                  migrationPlan: MferenceMigrationPlan.self,
                                  configurations: configuration)
    }

    /// An in-memory container for tests and previews.
    public static func inMemory() throws -> ModelContainer {
        let configuration = ModelConfiguration(schema: Schema(versionedSchema: MferenceSchemaV1.self),
                                               isStoredInMemoryOnly: true)
        return try ModelContainer(for: Schema(versionedSchema: MferenceSchemaV1.self),
                                  migrationPlan: MferenceMigrationPlan.self,
                                  configurations: configuration)
    }
}
