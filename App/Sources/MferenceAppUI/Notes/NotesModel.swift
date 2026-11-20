import Foundation
import MferenceAppCore
import Observation
import SwiftUI

/// Voice actions the Notes screen offers but the voice team (N3) owns: the
/// screen calls these; N3's views and controllers answer them. Nil leaves
/// the button disabled.
@MainActor
public struct NotesVoiceActions {
    /// Opens the Transcribe file sheet.
    public var transcribeFile: (() -> Void)?
    /// Opens the Record meeting popover.
    public var recordMeeting: (() -> Void)?
    /// Starts or stops dictation into the open note. The ⇧⌘D shortcut is
    /// N3's menu command; the Dictate button calls this.
    public var toggleDictation: ((NoteEditorController) -> Void)?
    /// N3's voice toolbar (Transcribe file, Record meeting with its popover),
    /// shown in the shell bar in place of Notes' own two buttons.
    public var toolbar: (() -> AnyView)?
    /// Whether dictation is going into the note (the Dictate button shows
    /// "Dictating").
    public var isDictating: ((UUID) -> Bool)?
    /// Stops dictation and returns once its last phrase is in (switching
    /// notes; the editor keeps that phrase with the note it began in).
    public var stopDictation: (() async -> Void)?
    /// The note editor gained (the controller) or lost (nil) keyboard focus:
    /// ⇧⌘D dictates into the focused one.
    public var editorFocusChanged: ((NoteEditorController?) -> Void)?
    /// Esc in the editor; returns true when it handled it (a dictation
    /// stopped). Otherwise Esc is the text view's own.
    public var escape: (() -> Bool)?
    /// Audio or video files dropped on Notes: queued for transcription.
    public var transcribe: (([URL]) -> Bool)?

    // N1 defines and calls these; N3 sets them (`VoiceModels.connect(notes:)`).
    public init(transcribeFile: (() -> Void)? = nil, recordMeeting: (() -> Void)? = nil,
                toggleDictation: ((NoteEditorController) -> Void)? = nil, toolbar: (() -> AnyView)? = nil,
                isDictating: ((UUID) -> Bool)? = nil, stopDictation: (() async -> Void)? = nil,
                editorFocusChanged: ((NoteEditorController?) -> Void)? = nil, escape: (() -> Bool)? = nil,
                transcribe: (([URL]) -> Bool)? = nil) {
        self.transcribeFile = transcribeFile
        self.recordMeeting = recordMeeting
        self.toggleDictation = toggleDictation
        self.toolbar = toolbar
        self.isDictating = isDictating
        self.stopDictation = stopDictation
        self.editorFocusChanged = editorFocusChanged
        self.escape = escape
        self.transcribe = transcribe
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
    /// The search field's text; the list follows it after a short pause.
    public var query = "" {
        didSet {
            guard query != oldValue else { return }
            scheduleSearch()
        }
    }
    /// The query the list shows (debounced; spec 6.6).
    public private(set) var appliedQuery = ""
    public private(set) var selectedNoteID: UUID?
    /// A permanent deletion waiting for the person to confirm it.
    public private(set) var pendingDeletion: PermanentDeletion?

    public enum PermanentDeletion: Equatable {
        case note(UUID, title: String)
        case trash(count: Int)

        public var title: String {
            switch self {
            case .note(_, let title): "Delete “\(title)” forever?"
            case .trash(let count): count == 1 ? "Delete 1 note forever?" : "Delete \(count) notes forever?"
            }
        }

        public var message: String { "Their recordings are deleted too. This can't be undone." }
    }
    public private(set) var speechStatus: SpeechStatus
    /// A folder being renamed in place.
    public var renamingFolderID: UUID?
    public private(set) var errorMessage: String?

    /// The calendar and locale the list's groups and times use.
    public var calendar = Calendar.current
    public var locale = Locale.current

    /// N4's meeting session and summaries (nil in previews without them):
    /// the live meeting layout and the finished-meeting views read them.
    @ObservationIgnored public var meetings: MeetingSession?
    @ObservationIgnored public var summaries: MeetingSummaries?
    /// Awaited before any purge: launch recovery closes interrupted
    /// meetings and files first (spec 6.3).
    @ObservationIgnored public var waitForLaunchRecovery: (() async -> Void)?
    /// The person opened the sidebar and list during a live meeting.
    public var showsPanesDuringMeeting = false

    @ObservationIgnored private var statusTask: Task<Void, Never>?
    @ObservationIgnored private var searchTask: Task<Void, Never>?
    @ObservationIgnored private let searchDelay: Duration
    @ObservationIgnored private let now: () -> Date

    // Derived values are cached per store revision, scope and query, so
    // typing (each autosave bumps the revision once) does not refetch per
    // view pass, and rows decode a body only when that note changed.
    private struct ListKey: Equatable {
        var revision: Int
        var scope: NoteScope
        var query: String
    }
    @ObservationIgnored private var cachedList: (key: ListKey, notes: [Note], sections: [ListSection])?
    @ObservationIgnored private var cachedCounts: (revision: Int, counts: NoteCounts)?
    @ObservationIgnored private var cachedFolders: (revision: Int, folders: [NoteFolder])?
    @ObservationIgnored private var rowCache: [UUID: (key: RowKey, title: String, preview: String)] = [:]
    private struct RowKey: Equatable {
        var updatedAt: Date
        var title: String
        var duration: TimeInterval?
        var summary: String?
        var deletedAt: Date?
    }

    /// The pause in typing before a search runs.
    public static let searchPause: Duration = .milliseconds(150)

    public init(store: NoteStore, speech: any SpeechServicing, searchDelay: Duration = NotesModel.searchPause,
                now: @escaping () -> Date = Date.init) {
        self.store = store
        self.editor = NoteEditorController(store: store)
        self.speechStatus = speech.status
        self.searchDelay = searchDelay
        self.now = now
        statusTask = Task { [weak self] in
            for await status in speech.statusUpdates() {
                self?.speechStatus = status
            }
        }
        // The editor publishes itself to the voice side when it gains or
        // loses focus, and offers Esc to it first.
        let editor = self.editor
        editor.onFocusChange = { [weak self, weak editor] focused in
            self?.voice.editorFocusChanged?(focused ? editor : nil)
        }
        editor.onCancel = { [weak self] in self?.voice.escape?() ?? false }
        editor.onMediaDropped = { [weak self] urls in self?.voice.transcribe?(urls) ?? false }
    }

    public convenience init(services: AppServices, now: @escaping () -> Date = Date.init) {
        self.init(store: services.notes, speech: services.speech, now: now)
        meetings = services.meetings
        summaries = services.summaries
        waitForLaunchRecovery = { [weak services] in await services?.launchRecovery?.value }
    }

    /// The live meeting's note when it is the one selected (board 4 shows
    /// the meeting layout for it).
    public var liveMeetingNoteID: UUID? {
        guard let meetings, meetings.isActive, let id = meetings.noteID else { return nil }
        return id
    }

    /// Whether the screen shows the meeting layout instead of the panes.
    public var showsMeetingLayout: Bool {
        liveMeetingNoteID != nil && liveMeetingNoteID == selectedNoteID && !showsPanesDuringMeeting
    }

    isolated deinit {
        statusTask?.cancel()
        searchTask?.cancel()
    }

    // MARK: Lists

    /// The notes the list shows, newest edit first.
    public var notes: [Note] { list.notes }

    /// A group of the list, as shown.
    public struct ListSection: Identifiable {
        public var id: String
        public var title: String
        public var items: [Note]
    }

    /// The list's groups: Pinned, Today, Yesterday, ... (Recently deleted
    /// is one plain list).
    public var sections: [ListSection] { list.sections }

    private var list: (notes: [Note], sections: [ListSection]) {
        let key = ListKey(revision: store.revision, scope: scope, query: appliedQuery)
        if let cachedList, cachedList.key == key { return (cachedList.notes, cachedList.sections) }
        let notes = (try? store.notes(in: scope, matching: appliedQuery)) ?? []
        let sections: [ListSection]
        if scope == .recentlyDeleted {
            sections = notes.isEmpty ? [] : [ListSection(id: "deleted", title: "", items: notes)]
        } else {
            sections = NoteListGrouping.sections(notes, now: now(), calendar: calendar, locale: locale,
                                                 updatedAt: \.updatedAt, isPinned: \.isPinned)
                .map { ListSection(id: "\($0.group)", title: $0.title, items: $0.items) }
        }
        cachedList = (key, notes, sections)
        // Forget rows of notes no longer shown or gone.
        if rowCache.count > notes.count * 2 + 64 {
            let shown = Set(notes.map(\.id))
            rowCache = rowCache.filter { shown.contains($0.key) }
        }
        return (notes, sections)
    }

    public var counts: NoteCounts {
        let revision = store.revision
        if let cachedCounts, cachedCounts.revision == revision { return cachedCounts.counts }
        let counts = (try? store.counts()) ?? NoteCounts()
        cachedCounts = (revision, counts)
        return counts
    }

    public var folders: [NoteFolder] {
        let revision = store.revision
        if let cachedFolders, cachedFolders.revision == revision { return cachedFolders.folders }
        let folders = (try? store.folders()) ?? []
        cachedFolders = (revision, folders)
        return folders
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

    public var isSearching: Bool { !NoteSearch.terms(appliedQuery).isEmpty }

    /// Applies the search now (Return in the field; tests).
    public func commitQuery() {
        searchTask?.cancel()
        guard appliedQuery != query else { return }
        appliedQuery = query
        selectFirstIfNeeded()
    }

    private func scheduleSearch() {
        searchTask?.cancel()
        let delay = searchDelay
        searchTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self?.commitQuery()
        }
    }

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

    public func title(of note: Note) -> String { row(note).title }

    private func row(_ note: Note) -> (title: String, preview: String) {
        let key = RowKey(updatedAt: note.updatedAt, title: note.title, duration: note.recording?.duration,
                         summary: note.recording?.summaryStateRawValue, deletedAt: note.deletedAt)
        if let cached = rowCache[note.id], cached.key == key { return (cached.title, cached.preview) }
        let value = (store.displayTitle(of: note), computePreview(of: note))
        rowCache[note.id] = (key, value.0, value.1)
        return value
    }

    /// "11:40", "Mon", or a date.
    public func timeLabel(of note: Note) -> String {
        NoteListGrouping.timeLabel(for: note.updatedAt, now: now(), calendar: calendar, locale: locale)
    }

    /// The row's second line after the time: the body's first line, or for
    /// meetings and files what the recording is.
    public func preview(of note: Note) -> String {
        if note.deletedAt != nil, let purge = store.purgeDate(of: note) {
            let days = max(0, calendar.dateComponents([.day], from: now(), to: purge).day ?? 0)
            return days == 1 ? "1 day left" : "\(days) days left"
        }
        return row(note).preview
    }

    private func computePreview(of note: Note) -> String {
        if note.deletedAt != nil, let purge = store.purgeDate(of: note) {
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
        if note?.id != selectedNoteID {
            showsPanesDuringMeeting = false
            revealedSegmentID = nil
            // Dictating into the note being left: stop; its last phrase still
            // lands there (the editor keeps it with its note).
            if let current = selectedNoteID, voice.isDictating?(current) == true, let stop = voice.stopDictation {
                Task { await stop() }
            }
        }
        selectedNoteID = note?.id
        editor.open(note)
    }

    /// Keeps a selection when the list changes under it: the selected note
    /// if it is still shown, else the first one.
    public func selectFirstIfNeeded() {
        let rows = sections.flatMap(\.items)
        if let selectedNoteID, rows.contains(where: { $0.id == selectedNoteID }) { return }
        select(rows.first)
    }

    /// Shows a note by id (the recording timer chip, Open note in the file
    /// queue, a started meeting): All notes, search cleared, that note.
    public func open(noteID: UUID) {
        guard let note = try? store.note(id: noteID) else { return }
        query = ""
        commitQuery()
        let target: NoteScope = note.deletedAt != nil ? .recentlyDeleted : .all
        if scope != target, !notes.contains(where: { $0.id == noteID }) { scope = target }
        // The timer chip opens the live meeting in its layout.
        showsPanesDuringMeeting = false
        select(note)
    }

    /// ↑ / ↓ in the list.
    public func selectAdjacent(_ offset: Int) {
        let rows = sections.flatMap(\.items)
        guard !rows.isEmpty else { return }
        guard let index = rows.firstIndex(where: { $0.id == selectedNoteID }) else {
            select(offset > 0 ? rows.first : rows.last)
            return
        }
        select(rows[min(max(index + offset, 0), rows.count - 1)])
    }

    /// ⌫ in the list: Recently deleted asks to delete forever; elsewhere the
    /// note moves to Recently deleted.
    public func deleteSelected() {
        guard let note = selectedNote else { return }
        if note.deletedAt != nil { requestDeleteForever(note) } else { delete(note) }
    }

    // MARK: Note actions

    /// ⌘N: a new note in the open folder, selected, with the caret in its title.
    public func newNote() {
        perform {
            if scope == .recentlyDeleted || scope == .meetings || scope == .audioTranscripts { scope = .all }
            query = ""
            commitQuery()
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
        selectFirstIfNeeded()
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
            let index = sections.flatMap(\.items).firstIndex { $0.id == note.id }
            try store.moveToTrash(note, now: now())
            if note.id == selectedNoteID {
                // The note that took its place, else the one above it, else the first.
                let remaining = sections.flatMap(\.items)
                let next = index.flatMap { remaining.indices.contains($0) ? remaining[$0] : remaining.last }
                select(next ?? remaining.first)
            }
        }
    }

    public func restore(_ note: Note) {
        perform {
            try store.restore(note)
            selectFirstIfNeeded()
        }
    }

    /// Asks before deleting a note forever (Delete now in Recently deleted).
    public func requestDeleteForever(_ note: Note) {
        pendingDeletion = .note(note.id, title: title(of: note))
    }

    /// Asks before emptying Recently deleted.
    public func requestEmptyTrash() {
        let ids = ((try? store.notes(in: .recentlyDeleted)) ?? []).map(\.id)
        guard !ids.isEmpty else { return }
        pendingTrashIDs = ids
        pendingDeletion = .trash(count: ids.count)
    }

    /// The notes Empty asked about: only those are deleted, even if more
    /// arrive in Recently deleted while the question is open.
    @ObservationIgnored private var pendingTrashIDs: [UUID] = []

    public func cancelDeletion() { pendingDeletion = nil }

    /// Carries out the confirmed deletion.
    public func confirmDeletion() async {
        guard let pending = pendingDeletion else { return }
        pendingDeletion = nil
        switch pending {
        case .note(let id, _):
            if let note = try? store.note(id: id) { await deletePermanently(note) }
        case .trash:
            var failed = 0
            for id in pendingTrashIDs {
                guard let note = try? store.note(id: id), note.deletedAt != nil else { continue }
                do { try await store.deletePermanently(note) } catch { failed += 1 }
            }
            pendingTrashIDs = []
            if failed > 0 { errorMessage = Self.message(for: NoteStoreError.someNotDeleted(failed)) }
            selectFirstIfNeeded()
        }
    }

    public func deletePermanently(_ note: Note) async {
        let wasSelected = note.id == selectedNoteID
        await performAsync { try await store.deletePermanently(note) }
        if wasSelected { selectedNoteID = nil }
        selectFirstIfNeeded()
    }

    public func emptyTrash() async {
        await performAsync {
            try await store.emptyTrash()
            selectFirstIfNeeded()
        }
    }

    /// Purges notes deleted 30 days ago (when Notes opens).
    public func purgeExpired() async {
        await waitForLaunchRecovery?()
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

    /// Move Up / Move Down (the keyboard path for reordering folders).
    public func moveFolder(_ folder: NoteFolder, by offset: Int) {
        let all = folders
        guard let index = all.firstIndex(where: { $0.id == folder.id }) else { return }
        let target = min(max(index + offset, 0), all.count - 1)
        guard target != index else { return }
        moveFolders(fromOffsets: IndexSet(integer: index), toOffset: target > index ? target + 1 : target)
    }

    /// Where a folder dragged onto `target` lands (before it when dragged up,
    /// after it when dragged down), as `moveFolders` offsets.
    public func dropFolder(_ moving: UUID, onto target: UUID) -> Bool {
        let all = folders
        guard moving != target, let from = all.firstIndex(where: { $0.id == moving }),
              let to = all.firstIndex(where: { $0.id == target }) else { return false }
        moveFolders(fromOffsets: IndexSet(integer: from), toOffset: to > from ? to + 1 : to)
        return true
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
        commitQuery()
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
        if note.id == selectedNoteID { editor.saveNow(force: true) }
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
        if let error = error as? NoteStoreError {
            switch error {
            case .gone: return "That note was deleted."
            case .readOnly: return "This note can't be changed here: it was saved by a newer version of Mference, or it is damaged."
            case .beingDeleted: return "That note is being deleted."
            case .notInTrash: return "Only notes in Recently deleted can be deleted forever."
            case .someNotDeleted(let count):
                return count == 1 ? "1 note couldn't be deleted. It stays in Recently deleted; Mference tries again later."
                    : "\(count) notes couldn't be deleted. They stay in Recently deleted; Mference tries again later."
            }
        }
        return "Couldn't save the change. \(error.localizedDescription)"
    }
}
