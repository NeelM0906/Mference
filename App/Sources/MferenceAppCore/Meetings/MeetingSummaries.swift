import Foundation
import Observation
import SwiftData

/// Where a to-do made from an action item goes: the Calendar to-dos
/// (`CalendarWeekModel`), which sync with Reminders as Calendar decided.
@MainActor
public protocol MeetingTodoSink: AnyObject {
    func addTodo(_ todo: TodoItem)
}

extension CalendarWeekModel: MeetingTodoSink {}

/// The summaries of meeting and file notes (spec 4.5, 6.5): runs them one
/// at a time after waiting for a running chat, writes the sections into the
/// note, names the note, and handles Retry, Redo (after a confirm) and Add
/// to To-dos. `AppServices` holds one; views observe it.
@MainActor
@Observable
public final class MeetingSummaries {
    public enum Status: Sendable, Hashable {
        case none
        /// Behind another note's summary.
        case queued
        /// A chat answer is being written; the summary starts after it.
        case waitingForChat
        /// Quick is writing it; the step shows on long meetings.
        case writing(MeetingSummarizer.Step?)
        case written
        /// "Couldn't write a summary", with Retry.
        case failed(String)
    }

    /// The label next to the Summary heading.
    public static let writtenLabel = "Written by Quick on this Mac"
    public static let waitingLabel = "Waiting for chat to finish"

    /// The note whose Redo asks for a confirm now (the UI shows the dialog).
    public private(set) var redoConfirmation: UUID?
    private var statuses: [UUID: Status] = [:]

    @ObservationIgnored private let engine: any EngineServicing
    @ObservationIgnored private let summarizer: MeetingSummarizer
    @ObservationIgnored private let assistant: (any AppleModelServicing)?
    @ObservationIgnored private let context: ModelContext
    @ObservationIgnored private let writer: any MeetingNoteWriting
    @ObservationIgnored private let todos: (any MeetingTodoSink)?
    @ObservationIgnored private let now: @Sendable () -> Date
    @ObservationIgnored private let locale: Locale
    @ObservationIgnored private let timeZone: TimeZone
    @ObservationIgnored private var queue: [(noteID: UUID, redo: Bool)] = []
    @ObservationIgnored private var runner: Task<Void, Never>?

    public init(engine: any EngineServicing, summarizer: MeetingSummarizer, assistant: (any AppleModelServicing)?,
                writer: any MeetingNoteWriting, todos: (any MeetingTodoSink)?,
                now: @escaping @Sendable () -> Date = { Date.now }, locale: Locale = .current,
                timeZone: TimeZone = .current) {
        self.engine = engine
        self.summarizer = summarizer
        self.assistant = assistant
        self.context = writer.context
        self.writer = writer
        self.todos = todos
        self.now = now
        self.locale = locale
        self.timeZone = timeZone
    }

    // MARK: Status

    public func status(for noteID: UUID) -> Status {
        if let status = statuses[noteID] { return status }
        guard let record = fetchNote(noteID)?.recording else { return .none }
        switch record.summaryState {
        case .written: return .written
        case .failed: return .failed(MeetingSummaryError.unreadableAnswer.message)
        case .pending:
            // Asked for while the file still transcribes: it runs when the
            // transcript is in. Otherwise the app quit while it ran: Retry.
            return Self.isTranscribing(record) ? .queued : .failed(MeetingSummaryError.unreadableAnswer.message)
        case .none: return .none
        }
    }

    // MARK: Running

    /// Queues the note's summary (after a meeting, or Retry).
    public func summarize(_ noteID: UUID) { enqueue(noteID, redo: false) }

    public func retry(_ noteID: UUID) { enqueue(noteID, redo: false) }

    /// Redo replaces the summary and action items only after a confirm.
    public func requestRedo(_ noteID: UUID) { redoConfirmation = noteID }

    public func confirmRedo() {
        guard let noteID = redoConfirmation else { return }
        redoConfirmation = nil
        enqueue(noteID, redo: true)
    }

    public func cancelRedo() { redoConfirmation = nil }

    /// At launch, after recovery: summaries left pending on finished notes
    /// (a meeting ended at quit, or a summary the app quit in) run now.
    /// Notes still transcribing get theirs when the transcript is in.
    @discardableResult
    public func resumePending() -> [UUID] {
        let pending = RecordingSummaryState.pending.rawValue
        let records = (try? context.fetch(FetchDescriptor<NoteRecording>(
            predicate: #Predicate { $0.summaryStateRawValue == pending }))) ?? []
        var resumed: [UUID] = []
        for record in records {
            guard let note = record.note, note.deletedAt == nil, record.endedAt != nil, !Self.isTranscribing(record)
            else { continue }
            summarize(note.id)
            resumed.append(note.id)
        }
        return resumed
    }

    /// Waits until every queued summary has finished (tests, quit).
    public func waitUntilIdle() async {
        while let runner { await runner.value }
    }

    private func enqueue(_ noteID: UUID, redo: Bool) {
        guard !queue.contains(where: { $0.noteID == noteID }) else { return }
        // A written summary is replaced only through Redo, after its confirm.
        if !redo, fetchNote(noteID)?.recording?.summaryState == .written { return }
        if case .writing? = statuses[noteID] { return }
        if statuses[noteID] == .waitingForChat { return }
        queue.append((noteID, redo))
        statuses[noteID] = .queued
        guard runner == nil else { return }
        runner = Task { [weak self] in
            while let self, !self.queue.isEmpty {
                let job = self.queue.removeFirst()
                await self.run(job.noteID, redo: job.redo)
            }
            self?.runner = nil
        }
    }

    private func run(_ noteID: UUID, redo: Bool) async {
        guard let first = liveRecord(noteID) else {
            statuses[noteID] = nil
            return
        }
        // A file still transcribing: its transcript triggers the summary.
        guard !Self.isTranscribing(first) else {
            first.summaryState = .pending
            try? context.save()
            statuses[noteID] = nil
            return
        }
        // A body this build may not write: don't run Quick for nothing.
        guard writer.canWrite(noteID) else {
            first.summaryState = .failed
            try? context.save()
            statuses[noteID] = .failed(MeetingSummaryError.readOnlyNote.message)
            return
        }
        if engine.status.isGenerating {
            statuses[noteID] = .waitingForChat
            for await status in engine.statusUpdates() where !status.isGenerating { break }
        }
        // Every wait can outlast the note (Empty trash): look it up again.
        guard let record = liveRecord(noteID) else {
            statuses[noteID] = nil
            return
        }
        statuses[noteID] = .writing(nil)
        // Stored as failed until it succeeds: if the app dies while Quick
        // runs, the next launch shows Retry instead of trying again by
        // itself every launch (`resumePending()` only resumes pending).
        record.summaryState = .failed
        try? context.save()
        let segments = record.segments
        let speakers = record.speakers
        let notes = writer.personalNotes(in: noteID)
        let outcome: Result<MeetingSections, any Error>
        do {
            outcome = .success(try await summarizer.summarize(segments: segments, speakers: speakers, notes: notes) {
                [weak self] step in
                await MainActor.run { self?.statuses[noteID] = .writing(step) }
            })
        } catch {
            outcome = .failure(error)
        }
        guard let record = liveRecord(noteID) else {
            statuses[noteID] = nil
            return
        }
        do {
            let sections = try outcome.get()
            try writer.replaceMeetingSections(sections, in: noteID)
            if redo { record.todosAddedAt = nil }
            record.summaryState = .written
            try? context.save()
            if !redo { await nameNote(noteID, from: sections) }
            statuses[noteID] = .written
        } catch MeetingSummaryError.emptyTranscript {
            record.summaryState = .none
            try? context.save()
            statuses[noteID] = Status.none
        } catch {
            record.summaryState = .failed
            try? context.save()
            statuses[noteID] = .failed(MeetingSummaryError.unreadableAnswer.message)
        }
    }

    /// The note's recording if the note still exists (it can be deleted for
    /// good while a summary waits or runs).
    private func liveRecord(_ noteID: UUID) -> NoteRecording? {
        guard let note = fetchNote(noteID), !note.isDeleted, let record = note.recording, !record.isDeleted
        else { return nil }
        return record
    }

    static func isTranscribing(_ record: NoteRecording) -> Bool {
        switch record.transcriptionState {
        case .queued, .running, .live: true
        case .none, .done, .failed: false
        }
    }

    /// Apple's model writes the title from the summary (Quick when it is off
    /// or unavailable, inside `AppleModelService`); the time-stamped default
    /// stays if both fail, or if the person already named the note.
    private func nameNote(_ noteID: UUID, from sections: MeetingSections) async {
        guard let assistant, let note = fetchNote(noteID), isDefaultTitle(note) else { return }
        guard let result = try? await assistant.title(for: sections.summary.joined(separator: "\n")) else { return }
        let title = result.value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty, let current = fetchNote(noteID), isDefaultTitle(current) else { return }
        try? writer.setTitle(title, of: noteID)
    }

    private func isDefaultTitle(_ note: Note) -> Bool {
        let title = note.title.trimmingCharacters(in: .whitespacesAndNewlines)
        if title.isEmpty { return true }
        guard let record = note.recording else { return false }
        if let file = record.sourceFileName { return title == MeetingCopy.fileTitle(file) }
        return title == MeetingCopy.defaultTitle(record.startedAt, locale: locale, timeZone: timeZone)
    }

    // MARK: Add to To-dos

    /// The Action items checklist rows as they stand in the note now.
    public func actionItems(for noteID: UUID) -> [MeetingActionItemRow] {
        writer.actionItems(in: noteID)
    }

    /// "Add n to To-dos": the unchecked action items now in the note.
    public func uncheckedActionItemCount(for noteID: UUID) -> Int {
        writer.actionItems(in: noteID).filter { !$0.isDone }.count
    }

    /// The button reads "Added to To-dos".
    public func todosAdded(for noteID: UUID) -> Bool {
        fetchNote(noteID)?.recording?.todosAddedAt != nil
    }

    /// One to-do per unchecked action item, once per summary. Returns how many were added.
    @discardableResult
    public func addActionItemsToTodos(for noteID: UUID) -> Int {
        guard let todos, let note = fetchNote(noteID), let record = note.recording, record.todosAddedAt == nil
        else { return 0 }
        let rows = writer.actionItems(in: noteID).filter { !$0.isDone }
        guard !rows.isEmpty else { return 0 }
        let label = note.title.isEmpty ? "Meeting" : note.title
        let created = now()
        for row in rows {
            todos.addTodo(TodoItem(title: row.item.text,
                                   source: ItemSource(kind: .meeting, label: label,
                                                      reference: "note:\(noteID.uuidString):\(row.item.text)"),
                                   createdAt: created))
        }
        record.todosAddedAt = created
        try? context.save()
        return rows.count
    }

    private func fetchNote(_ id: UUID) -> Note? {
        try? context.fetch(FetchDescriptor<Note>(predicate: #Predicate { $0.id == id })).first
    }
}
