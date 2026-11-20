import Foundation
import Observation
import SwiftData

/// Which notes a list shows (the sidebar's rows, Notes spec 4.1).
public enum NoteScope: Sendable, Hashable {
    case all
    /// Notes with a meeting recording.
    case meetings
    /// Notes made from an audio or video file.
    case audioTranscripts
    case folder(UUID)
    /// Notes in Recently deleted (30 days, then purged).
    case recentlyDeleted
}

public enum NoteStoreError: Error, Sendable, Equatable {
    /// The note or folder was deleted from the store.
    case gone
}

/// Notes and folders in the SwiftData store: create, edit, pin, folders,
/// Recently deleted and its 30-day purge (Notes spec 4.1, 8). Every change is
/// saved at once and bumps `revision`, which views observe to refetch.
///
/// Purging a note deletes its attachments (`Notes/<note-id>/`) and asks the
/// speech service to delete its recordings (`Recordings/<note-id>/`).
@MainActor
@Observable
public final class NoteStore {
    /// How long a note stays in Recently deleted.
    public static let trashRetention: TimeInterval = 30 * 24 * 60 * 60

    @ObservationIgnored public let context: ModelContext
    /// A context does not keep its container alive; the store holds it.
    @ObservationIgnored public let container: ModelContainer
    @ObservationIgnored public let attachments: NoteAttachments
    @ObservationIgnored private let speech: any SpeechServicing

    /// Bumped on every change to a note or folder; views observe it.
    public private(set) var revision = 0

    public init(container: ModelContainer, attachments: NoteAttachments, speech: any SpeechServicing) {
        self.container = container
        self.context = container.mainContext
        self.attachments = attachments
        self.speech = speech
    }

    // MARK: Reading

    /// The notes `scope` shows, newest edit first, filtered by `query`
    /// (every word must match title, body or transcript; case and
    /// diacritics ignored).
    public func notes(in scope: NoteScope, matching query: String = "") throws -> [Note] {
        let inTrash = scope == .recentlyDeleted
        let descriptor = FetchDescriptor<Note>(
            predicate: inTrash ? #Predicate { $0.deletedAt != nil } : #Predicate { $0.deletedAt == nil },
            sortBy: [SortDescriptor(\.updatedAt, order: .reverse)])
        let terms = NoteSearch.terms(query)
        return try context.fetch(descriptor).filter { note in
            Self.belongs(note, to: scope) && (terms.isEmpty || NoteSearch.matches(note.searchText, terms: terms))
        }
    }

    /// How many notes `scope` holds (the sidebar's counts).
    public func count(in scope: NoteScope) throws -> Int {
        try notes(in: scope).count
    }

    /// Counts for every Library row and folder at once.
    public func counts() throws -> NoteCounts {
        let live = try context.fetch(FetchDescriptor<Note>(predicate: #Predicate { $0.deletedAt == nil }))
        let trash = try context.fetchCount(FetchDescriptor<Note>(predicate: #Predicate { $0.deletedAt != nil }))
        var folders: [UUID: Int] = [:]
        for note in live {
            if let id = note.folder?.id { folders[id, default: 0] += 1 }
        }
        return NoteCounts(all: live.count,
                          meetings: live.count { $0.kind == .meeting },
                          audioTranscripts: live.count { $0.kind == .file },
                          recentlyDeleted: trash,
                          folders: folders)
    }

    public func note(id: UUID) throws -> Note? {
        var descriptor = FetchDescriptor<Note>(predicate: #Predicate { $0.id == id })
        descriptor.fetchLimit = 1
        return try context.fetch(descriptor).first
    }

    /// The note's body.
    public func document(of note: Note) -> NoteDocument {
        NoteDocument.decode(note.bodyJSON)
    }

    /// What the list shows as the note's title: its title, else its first
    /// line, else "New note".
    public func displayTitle(of note: Note) -> String {
        if !note.title.isEmpty { return note.title }
        let first = document(of: note).firstLine
        return first.isEmpty ? "New note" : first
    }

    // MARK: Notes

    /// A new empty note, saved at once, in `folder` if given.
    @discardableResult
    public func createNote(kind: NoteKind = .note, title: String = "", in folder: NoteFolder? = nil,
                           now: Date = .now) throws -> Note {
        let note = Note(title: title, kind: kind, createdAt: now, folder: folder)
        let document = title.isEmpty ? NoteDocument.empty : NoteDocument.titled(title)
        note.bodyJSON = document.encoded()
        note.searchText = NoteSearch.searchText(title: title, document: document, transcript: [])
        context.insert(note)
        try commit()
        return note
    }

    /// Saves an edited body: the title follows the title line, the search
    /// text is rebuilt, and the edit time moves. A body that did not change
    /// saves nothing (switching notes saves without moving the note up).
    public func save(_ document: NoteDocument, to note: Note, now: Date = .now) throws {
        try ensureAlive(note)
        let data = document.encoded()
        guard data != note.bodyJSON else { return }
        note.bodyJSON = data
        note.title = document.title
        note.updatedAt = now
        refreshSearchText(note, document: document)
        try commit()
    }

    /// Sets the title (meetings and file notes get theirs written for them),
    /// keeping the body's title line in step.
    public func setTitle(_ title: String, of note: Note, now: Date = .now) throws {
        try ensureAlive(note)
        var document = document(of: note)
        document.setTitle(title)
        note.bodyJSON = document.encoded()
        note.title = document.title
        note.updatedAt = now
        refreshSearchText(note, document: document)
        try commit()
    }

    /// Rebuilds the search text after the transcript changed (meetings and
    /// file notes); the edit time stays.
    public func transcriptDidChange(in note: Note) throws {
        try ensureAlive(note)
        refreshSearchText(note, document: document(of: note))
        try commit()
    }

    /// Marks that some of the note was dictated (the list's microphone).
    public func markDictated(_ note: Note) throws {
        try ensureAlive(note)
        guard !note.hasDictation else { return }
        note.hasDictation = true
        try commit()
    }

    /// Pinning does not move the edit time.
    public func setPinned(_ pinned: Bool, _ note: Note) throws {
        try ensureAlive(note)
        note.isPinned = pinned
        try commit()
    }

    /// Moves a note into `folder`, or out of every folder with nil.
    public func move(_ note: Note, to folder: NoteFolder?) throws {
        try ensureAlive(note)
        note.folder = folder
        try commit()
    }

    /// A copy of the note's text and attachments, as a plain note in the same
    /// folder (a meeting's recording stays with the original).
    @discardableResult
    public func duplicate(_ note: Note, now: Date = .now) throws -> Note {
        try ensureAlive(note)
        let copy = Note(title: note.title, kind: .note, createdAt: now, folder: note.folder)
        copy.bodyJSON = note.bodyJSON
        copy.hasDictation = note.hasDictation
        let document = document(of: note)
        copy.searchText = NoteSearch.searchText(title: copy.title, document: document, transcript: [])
        try attachments.copyAll(from: note.id, to: copy.id)
        context.insert(copy)
        try commit()
        return copy
    }

    // MARK: Recently deleted

    /// Moves the note to Recently deleted; it is purged 30 days later.
    public func moveToTrash(_ note: Note, now: Date = .now) throws {
        try ensureAlive(note)
        note.deletedAt = now
        note.isPinned = false
        try commit()
    }

    /// Puts a note back from Recently deleted. A folder deleted meanwhile
    /// already left it in All notes.
    public func restore(_ note: Note) throws {
        try ensureAlive(note)
        note.deletedAt = nil
        try commit()
    }

    /// Deletes the note now, with its attachments and recordings. Files go
    /// first: when they cannot be deleted the note stays in Recently deleted,
    /// so the next purge tries again and no audio is orphaned.
    public func deletePermanently(_ note: Note) async throws {
        try ensureAlive(note)
        try await deleteFiles(of: note.id)
        try ensureAlive(note)
        context.delete(note)
        try commit()
    }

    /// Deletes everything in Recently deleted.
    public func emptyTrash() async throws {
        for note in try notes(in: .recentlyDeleted) {
            try await deletePermanently(note)
        }
    }

    /// Purges notes deleted more than 30 days before `now` (at launch and
    /// when Notes opens). Returns how many were purged.
    @discardableResult
    public func purgeExpired(now: Date = .now) async throws -> Int {
        let cutoff = now.addingTimeInterval(-Self.trashRetention)
        let expired = try notes(in: .recentlyDeleted).filter { ($0.deletedAt ?? .distantFuture) <= cutoff }
        for note in expired {
            try await deletePermanently(note)
        }
        return expired.count
    }

    /// The day a note in Recently deleted goes for good.
    public func purgeDate(of note: Note) -> Date? {
        note.deletedAt?.addingTimeInterval(Self.trashRetention)
    }

    // MARK: Folders

    /// Folders in sidebar order.
    public func folders() throws -> [NoteFolder] {
        try context.fetch(FetchDescriptor<NoteFolder>(
            sortBy: [SortDescriptor(\.sortOrder), SortDescriptor(\.createdAt)]))
    }

    public func folder(id: UUID) throws -> NoteFolder? {
        var descriptor = FetchDescriptor<NoteFolder>(predicate: #Predicate { $0.id == id })
        descriptor.fetchLimit = 1
        return try context.fetch(descriptor).first
    }

    /// A new folder at the end of the list. A blank name becomes "New folder"
    /// (numbered when taken).
    @discardableResult
    public func createFolder(named name: String = "", now: Date = .now) throws -> NoteFolder {
        let existing = try folders()
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let folder = NoteFolder(name: trimmed.isEmpty ? Self.freeName("New folder", taken: existing.map(\.name)) : trimmed,
                                sortOrder: (existing.map(\.sortOrder).max() ?? -1) + 1, createdAt: now)
        context.insert(folder)
        try commit()
        return folder
    }

    /// Renames a folder; a blank name keeps the old one.
    public func rename(_ folder: NoteFolder, to name: String) throws {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != folder.name else { return }
        folder.name = trimmed
        try commit()
    }

    /// Deletes a folder; its notes stay, in All notes.
    public func delete(_ folder: NoteFolder) throws {
        for note in folder.notes { note.folder = nil }
        context.delete(folder)
        try commit()
    }

    /// Reorders folders to `ordered` (drag to reorder).
    public func reorderFolders(_ ordered: [NoteFolder]) throws {
        for (index, folder) in ordered.enumerated() where folder.sortOrder != index {
            folder.sortOrder = index
        }
        try commit()
    }

    /// Moves the folders at `source` to `destination`, as `List.onMove` gives them.
    public func moveFolders(fromOffsets source: IndexSet, toOffset destination: Int) throws {
        let all = try folders()
        let moving = source.filter { $0 < all.count }.map { all[$0] }
        var ordered = all.enumerated().filter { !source.contains($0.offset) }.map(\.element)
        let insertAt = destination - source.count { $0 < destination }
        ordered.insert(contentsOf: moving, at: min(max(insertAt, 0), ordered.count))
        try reorderFolders(ordered)
    }

    // MARK: Private

    private static func belongs(_ note: Note, to scope: NoteScope) -> Bool {
        switch scope {
        case .all, .recentlyDeleted: true
        case .meetings: note.kind == .meeting
        case .audioTranscripts: note.kind == .file
        case .folder(let id): note.folder?.id == id
        }
    }

    private func refreshSearchText(_ note: Note, document: NoteDocument) {
        note.searchText = NoteSearch.searchText(title: note.title, document: document,
                                                transcript: NoteSearch.transcript(note.recording?.transcriptJSON))
    }

    private func ensureAlive(_ note: Note) throws {
        guard note.modelContext != nil, !note.isDeleted else { throw NoteStoreError.gone }
    }

    private func commit() throws {
        try context.save()
        revision += 1
    }

    private func deleteFiles(of noteID: UUID) async throws {
        try attachments.deleteAll(for: noteID)
        try await speech.deleteRecordings(for: noteID)
    }

    private static func freeName(_ base: String, taken: [String]) -> String {
        let taken = Set(taken)
        guard taken.contains(base) else { return base }
        var number = 2
        while taken.contains("\(base) \(number)") { number += 1 }
        return "\(base) \(number)"
    }
}

/// The sidebar's counts.
public struct NoteCounts: Sendable, Hashable {
    public var all: Int
    public var meetings: Int
    public var audioTranscripts: Int
    public var recentlyDeleted: Int
    public var folders: [UUID: Int]

    public init(all: Int = 0, meetings: Int = 0, audioTranscripts: Int = 0, recentlyDeleted: Int = 0,
                folders: [UUID: Int] = [:]) {
        self.all = all
        self.meetings = meetings
        self.audioTranscripts = audioTranscripts
        self.recentlyDeleted = recentlyDeleted
        self.folders = folders
    }
}
