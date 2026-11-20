import Foundation
import MferenceAppCore
import Observation

/// Voice actions the Notes screen offers but the voice team (N3) owns: the
/// screen calls these; N3's views and controllers answer them. Nil leaves
/// the button disabled.
@MainActor
public struct NotesVoiceActions {
    /// Opens the Transcribe file sheet.
    public var transcribeFile: (() -> Void)?
    /// Opens the Record meeting popover.
    public var recordMeeting: (() -> Void)?
    /// Starts or stops dictation into the open note (⇧⌘D).
    public var toggleDictation: ((NoteEditorController) -> Void)?

    public init(transcribeFile: (() -> Void)? = nil, recordMeeting: (() -> Void)? = nil,
                toggleDictation: ((NoteEditorController) -> Void)? = nil) {
        self.transcribeFile = transcribeFile
        self.recordMeeting = recordMeeting
        self.toggleDictation = toggleDictation
    }
}

/// The Notes screen's state (Notes spec 4.1): which list shows, the search,
/// the selected note and its editor, the folders and counts, and the speech
/// status card. Every list reads `NoteStore.revision`, so a change anywhere
/// (dictation, a meeting, another window) shows at once.
@MainActor
@Observable
public final class NotesModel {
    public let store: NoteStore
    public let editor: NoteEditorController
    public var voice = NotesVoiceActions()

    /// The sidebar row showing.
    public var scope: NoteScope = .all {
        didSet { if scope != oldValue { selectFirstIfNeeded() } }
    }
    public var query = ""
    public private(set) var selectedNoteID: UUID?
    public private(set) var speechStatus: SpeechStatus
    /// A folder being renamed in place.
    public var renamingFolderID: UUID?
    public private(set) var errorMessage: String?

    /// The calendar and locale the list's groups and times use.
    public var calendar = Calendar.current
    public var locale = Locale.current

    @ObservationIgnored private var statusTask: Task<Void, Never>?
    @ObservationIgnored private let now: () -> Date

    public init(store: NoteStore, speech: any SpeechServicing, now: @escaping () -> Date = Date.init) {
        self.store = store
        self.editor = NoteEditorController(store: store)
        self.speechStatus = speech.status
        self.now = now
        statusTask = Task { [weak self] in
            for await status in speech.statusUpdates() {
                self?.speechStatus = status
            }
        }
    }

    public convenience init(services: AppServices) {
        self.init(store: services.notes, speech: services.speech)
    }

    isolated deinit { statusTask?.cancel() }

    // MARK: Lists

    /// The notes the list shows, newest edit first.
    public var notes: [Note] {
        _ = store.revision
        return (try? store.notes(in: scope, matching: query)) ?? []
    }

    /// A group of the list, as shown.
    public struct ListSection: Identifiable {
        public var id: String
        public var title: String
        public var items: [Note]
    }

    /// The list's groups: Pinned, Today, Yesterday, ... (Recently deleted
    /// is one plain list).
    public var sections: [ListSection] {
        let notes = notes
        if scope == .recentlyDeleted {
            return notes.isEmpty ? [] : [ListSection(id: "deleted", title: "", items: notes)]
        }
        return NoteListGrouping.sections(notes, now: now(), calendar: calendar, locale: locale,
                                         updatedAt: \.updatedAt, isPinned: \.isPinned)
            .map { ListSection(id: "\($0.group)", title: $0.title, items: $0.items) }
    }

    public var counts: NoteCounts {
        _ = store.revision
        return (try? store.counts()) ?? NoteCounts()
    }

    public var folders: [NoteFolder] {
        _ = store.revision
        return (try? store.folders()) ?? []
    }

    public var selectedNote: Note? {
        _ = store.revision
        guard let selectedNoteID, let note = try? store.note(id: selectedNoteID) else { return nil }
        return note
    }

    /// The open folder, when the list is one.
    public var currentFolder: NoteFolder? {
        guard case .folder(let id) = scope else { return nil }
        return folders.first { $0.id == id }
    }

    public var isSearching: Bool { !NoteSearch.terms(query).isEmpty }

    public var scopeTitle: String {
        switch scope {
        case .all: "All notes"
        case .meetings: "Meetings"
        case .audioTranscripts: "Audio transcripts"
        case .recentlyDeleted: "Recently deleted"
        case .folder: currentFolder?.name ?? "Folder"
        }
    }

    // MARK: Rows

    public func title(of note: Note) -> String { store.displayTitle(of: note) }

    /// "11:40", "Mon", or a date.
    public func timeLabel(of note: Note) -> String {
        NoteListGrouping.timeLabel(for: note.updatedAt, now: now(), calendar: calendar, locale: locale)
    }

    /// The row's second line after the time: the body's first line, or for
    /// meetings and files what the recording is.
    public func preview(of note: Note) -> String {
        if scope == .recentlyDeleted, let purge = store.purgeDate(of: note) {
            let days = max(0, calendar.dateComponents([.day], from: now(), to: purge).day ?? 0)
            return days == 1 ? "1 day left" : "\(days) days left"
        }
        let document = store.document(of: note)
        if let recording = note.recording {
            var parts: [String] = []
            if note.kind == .file, let name = recording.sourceFileName { parts.append("From \(name)") }
            if recording.duration >= 60 { parts.append("\(Int(recording.duration / 60)) min") }
            if note.kind == .meeting, recording.summaryState == .written { parts.append("summary") }
            if !parts.isEmpty { return parts.joined(separator: " · ") }
        }
        let preview = note.title.isEmpty ? document.lines.dropFirst().first ?? "" : document.previewLine
        return preview.isEmpty ? "No additional text" : preview
    }

    // MARK: Selection

    public func select(_ note: Note?) {
        selectedNoteID = note?.id
        editor.open(note)
    }

    /// Keeps a selection when the list changes under it.
    public func selectFirstIfNeeded() {
        let notes = notes
        if let selectedNoteID, notes.contains(where: { $0.id == selectedNoteID }) { return }
        select(notes.first)
    }

    // MARK: Note actions

    /// ⌘N: a new note in the open folder, selected, with the caret in its title.
    public func newNote() {
        perform {
            if scope == .recentlyDeleted || scope == .meetings || scope == .audioTranscripts { scope = .all }
            query = ""
            let note = try store.createNote(in: currentFolder, now: now())
            select(note)
            editor.focus()
        }
    }

    public func togglePin(_ note: Note) {
        perform { try store.setPinned(!note.isPinned, note) }
    }

    public func move(_ note: Note, to folder: NoteFolder?) {
        perform { try store.move(note, to: folder) }
    }

    public func duplicate(_ note: Note) {
        perform {
            editor.saveNow()
            select(try store.duplicate(note, now: now()))
        }
    }

    /// Delete moves the note to Recently deleted; the list selects the next one.
    public func delete(_ note: Note) {
        perform {
            if note.id == selectedNoteID { editor.saveNow() }
            let list = notes
            let index = list.firstIndex { $0.id == note.id }
            try store.moveToTrash(note, now: now())
            if note.id == selectedNoteID {
                let remaining = notes
                select(index.flatMap { remaining.indices.contains($0) ? remaining[$0] : remaining.last })
            }
        }
    }

    public func restore(_ note: Note) {
        perform {
            try store.restore(note)
            selectFirstIfNeeded()
        }
    }

    public func deletePermanently(_ note: Note) async {
        await performAsync {
            try await store.deletePermanently(note)
            selectFirstIfNeeded()
        }
    }

    public func emptyTrash() async {
        await performAsync {
            try await store.emptyTrash()
            selectFirstIfNeeded()
        }
    }

    /// Purges notes deleted 30 days ago (when Notes opens).
    public func purgeExpired() async {
        await performAsync { _ = try await store.purgeExpired(now: now()) }
    }

    // MARK: Folders

    public func newFolder() {
        perform {
            let folder = try store.createFolder(now: now())
            scope = .folder(folder.id)
            renamingFolderID = folder.id
        }
    }

    public func rename(_ folder: NoteFolder, to name: String) {
        perform { try store.rename(folder, to: name) }
        renamingFolderID = nil
    }

    public func delete(_ folder: NoteFolder) {
        perform {
            if scope == .folder(folder.id) { scope = .all }
            try store.delete(folder)
        }
    }

    public func moveFolders(fromOffsets source: IndexSet, toOffset destination: Int) {
        perform { try store.moveFolders(fromOffsets: source, toOffset: destination) }
    }

    /// Drop notes (by id) onto a folder.
    public func moveNotes(_ ids: [UUID], to folder: NoteFolder) -> Bool {
        var moved = false
        perform {
            for id in ids {
                guard let note = try store.note(id: id), note.deletedAt == nil else { continue }
                try store.move(note, to: folder)
                moved = true
            }
        }
        return moved
    }

    // MARK: Search (⌘K) and export

    /// The transcript line a ⌘K hit asked to show; the meeting layout (N3)
    /// scrolls to it, and the editor scrolls to its time chip.
    public private(set) var revealedSegmentID: UUID?

    /// ⌘K's Notes results for `query` (title, then body, then transcript lines).
    public func searchHits(_ query: String, limit: Int = 8) -> [NoteSearchHit] {
        (try? store.searchHits(query, limit: limit)) ?? []
    }

    /// Opens a ⌘K hit: All notes, that note, and its transcript line.
    public func open(_ hit: NoteSearchHit) {
        query = ""
        scope = .all
        guard let note = try? store.note(id: hit.noteID) else { return }
        select(note)
        if case .transcript(let segment, _) = hit.match {
            revealedSegmentID = segment
            editor.reveal(segmentID: segment)
        } else {
            revealedSegmentID = nil
        }
    }

    /// Asks where, then exports `note` (Markdown with an attachments folder,
    /// plain text, or PDF).
    public func export(_ note: Note, as format: NoteExportFormat) {
        if note.id == selectedNoteID { editor.saveNow() }
        if let message = NoteExporter.export(note, store: store, format: format) { errorMessage = message }
    }

    // MARK: Errors

    public func dismissError() { errorMessage = nil }

    private func perform(_ body: () throws -> Void) {
        do { try body() } catch { errorMessage = Self.message(for: error) }
    }

    private func performAsync(_ body: () async throws -> Void) async {
        do { try await body() } catch { errorMessage = Self.message(for: error) }
    }

    static func message(for error: any Error) -> String {
        if let error = error as? NoteStoreError, error == .gone { return "That note was deleted." }
        return "Couldn't save the change. \(error.localizedDescription)"
    }
}
