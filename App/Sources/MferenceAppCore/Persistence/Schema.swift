import Foundation
import SwiftData

// SHARED CONTRACT (App Program spec sections 4.4 and 8; Notes program spec 8
// adds V2). The SwiftData schema
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

/// Version 2 (Notes program spec section 8): V1 unchanged, plus notes,
/// folders and recordings. Migration V1 → V2 is lightweight (models added
/// only). Rich note values (the `NoteDocument` body, transcript segments,
/// speakers, track lists) are Codable types owned by `Notes/`, `Speech/` and
/// `Meetings/`, stored as `Data`.
public enum MferenceSchemaV2: VersionedSchema {
    public static let versionIdentifier = Schema.Version(2, 0, 0)

    public static var models: [any PersistentModel.Type] {
        MferenceSchemaV1.models + [NoteFolder.self, Note.self, NoteRecording.self]
    }

    /// A folder in the Notes sidebar (one level, no nesting in v1).
    @Model
    public final class NoteFolder {
        @Attribute(.unique) public var id: UUID
        public var name: String
        /// Position in the sidebar; the person reorders by dragging.
        public var sortOrder: Int
        public var createdAt: Date
        /// Deleting a folder leaves its notes in All notes.
        @Relationship(deleteRule: .nullify, inverse: \Note.folder)
        public var notes: [Note] = []

        public init(id: UUID = UUID(), name: String, sortOrder: Int, createdAt: Date = .now) {
            self.id = id
            self.name = name
            self.sortOrder = sortOrder
            self.createdAt = createdAt
        }
    }

    /// One note: typed, dictated, a meeting, or a transcribed file.
    @Model
    public final class Note {
        @Attribute(.unique) public var id: UUID
        /// The title the person gave it; empty means the list shows the first line.
        public var title: String
        /// JSON `NoteDocument` (block list, versioned; owned by Notes, N1).
        public var bodyJSON: Data?
        /// Title, body text and transcript text, flattened for search
        /// (spec 6.6); rewritten on every save.
        public var searchText: String
        public var createdAt: Date
        /// Last edit; the list sorts and groups by it.
        public var updatedAt: Date
        public var folder: NoteFolder?
        public var isPinned: Bool
        /// Set while the note is in Recently deleted; purged 30 days later.
        public var deletedAt: Date?
        /// `NoteKind.rawValue`.
        public var kindRawValue: String
        /// Some of the body was dictated (the list's microphone mark).
        public var hasDictation: Bool
        /// The meeting or file recording, for `.meeting` and `.file` notes.
        @Relationship(deleteRule: .cascade, inverse: \NoteRecording.note)
        public var recording: NoteRecording?

        public init(id: UUID = UUID(), title: String = "", kind: NoteKind = .note,
                    createdAt: Date = .now, folder: NoteFolder? = nil) {
            self.id = id
            self.title = title
            self.searchText = title
            self.createdAt = createdAt
            self.updatedAt = createdAt
            self.folder = folder
            self.isPinned = false
            self.kindRawValue = kind.rawValue
            self.hasDictation = false
        }

        public var kind: NoteKind {
            get { NoteKind(rawValue: kindRawValue) ?? .note }
            set { kindRawValue = newValue.rawValue }
        }
    }

    /// The audio and transcript behind a meeting or transcribed-file note.
    /// Audio files live in `Recordings/<note-id>/` (spec 7); this row holds
    /// their names, never the audio.
    @Model
    public final class NoteRecording {
        @Attribute(.unique) public var id: UUID
        public var note: Note?
        public var startedAt: Date
        /// Set when capture ended (stopped, or closed after a crash). Nil
        /// means a meeting is still recording, or the app quit mid-meeting
        /// and the next launch must close it (spec 6.3).
        public var endedAt: Date?
        /// Recorded length in seconds, pauses excluded.
        public var duration: TimeInterval
        /// The language detected (or pinned), BCP 47 ("en", "de").
        public var languageCode: String?
        /// JSON `[RecordedTrack]`: file name, track kind, start offset (Speech, N2).
        public var tracksJSON: Data?
        /// JSON `[TranscriptSegment]` (Speech contract; Meetings, N4).
        public var transcriptJSON: Data?
        /// JSON `[SpeakerInfo]`: id, name, colour index.
        public var speakersJSON: Data?
        /// `RecordingSummaryState.rawValue`.
        public var summaryStateRawValue: String
        /// The audio files are kept (Settings › Speech › Keep audio, D4).
        public var audioKept: Bool
        /// The imported file's original name (file transcripts only).
        public var sourceFileName: String?
        /// When the action items went to To-dos; nil until then.
        public var todosAddedAt: Date?

        public init(id: UUID = UUID(), startedAt: Date = .now, audioKept: Bool = true,
                    sourceFileName: String? = nil) {
            self.id = id
            self.startedAt = startedAt
            self.duration = 0
            self.summaryStateRawValue = RecordingSummaryState.none.rawValue
            self.audioKept = audioKept
            self.sourceFileName = sourceFileName
        }

        public var summaryState: RecordingSummaryState {
            get { RecordingSummaryState(rawValue: summaryStateRawValue) ?? .none }
            set { summaryStateRawValue = newValue.rawValue }
        }
    }
}

/// What a note is, for the sidebar's Library rows and the list's kind marks.
public enum NoteKind: String, Sendable, Hashable, Codable, CaseIterable {
    /// Typed (and maybe dictated).
    case note
    /// Recorded with Record meeting.
    case meeting
    /// Made from an audio or video file.
    case file
}

/// Where a recording's summary stands (spec 4.5, 6.5).
public enum RecordingSummaryState: String, Sendable, Hashable, Codable, CaseIterable {
    case none, pending, written, failed
}

public typealias Conversation = MferenceSchemaV1.Conversation
public typealias StoredMessage = MferenceSchemaV1.StoredMessage
public typealias MemoryItem = MferenceSchemaV1.MemoryItem
public typealias ScanRecord = MferenceSchemaV1.ScanRecord
public typealias MailBucket = MferenceSchemaV1.MailBucket
public typealias MailBucketAssignment = MferenceSchemaV1.MailBucketAssignment
public typealias NoteFolder = MferenceSchemaV2.NoteFolder
public typealias Note = MferenceSchemaV2.Note
public typealias NoteRecording = MferenceSchemaV2.NoteRecording

public enum MferenceMigrationPlan: SchemaMigrationPlan {
    public static var schemas: [any VersionedSchema.Type] { [MferenceSchemaV1.self, MferenceSchemaV2.self] }
    /// V1 → V2 adds the note models only, so it is lightweight.
    public static var stages: [MigrationStage] {
        [.lightweight(fromVersion: MferenceSchemaV1.self, toVersion: MferenceSchemaV2.self)]
    }
}

/// The schema the app opens: always the newest version.
public typealias MferenceCurrentSchema = MferenceSchemaV2

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
        let configuration = ModelConfiguration(schema: Schema(versionedSchema: MferenceCurrentSchema.self), url: url)
        return try ModelContainer(for: Schema(versionedSchema: MferenceCurrentSchema.self),
                                  migrationPlan: MferenceMigrationPlan.self,
                                  configurations: configuration)
    }

    /// An in-memory container for tests and previews.
    public static func inMemory() throws -> ModelContainer {
        let configuration = ModelConfiguration(schema: Schema(versionedSchema: MferenceCurrentSchema.self),
                                               isStoredInMemoryOnly: true)
        return try ModelContainer(for: Schema(versionedSchema: MferenceCurrentSchema.self),
                                  migrationPlan: MferenceMigrationPlan.self,
                                  configurations: configuration)
    }
}
