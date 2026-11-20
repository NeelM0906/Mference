import Foundation
import Observation
import SwiftData

/// The import sheet's options (spec 4.6).
public struct FileImportOptions: Sendable, Hashable {
    public var language: SpeechLanguage
    public var speakerLabels: Bool
    public var keepAudio: Bool
    /// "Write a summary".
    public var summarize: Bool

    public init(language: SpeechLanguage = .automatic, speakerLabels: Bool = false, keepAudio: Bool = true,
                summarize: Bool = false) {
        self.language = language
        self.speakerLabels = speakerLabels
        self.keepAudio = keepAudio
        self.summarize = summarize
    }

    /// The import sheet's defaults from the one shared `SpeechPreferences`;
    /// no summary when Settings turned summaries off.
    @MainActor
    public init(preferences: SpeechPreferences) {
        self.init(language: preferences.language, speakerLabels: preferences.fileSpeakerLabels,
                  keepAudio: preferences.keepAudio == .keep,
                  summarize: preferences.fileSummary && preferences.summaryModel != .off)
    }
}

/// Transcribing audio and video files into notes (spec 4.6). `AppServices`
/// holds one; it is the single consumer of every `FileJob.events`, so files
/// keep transcribing with the sheet closed. Each file becomes a `.file`
/// note titled from its name; the transcript, and the summary when asked,
/// are written when the job finishes. The sheet's queue observes `rows`.
@MainActor
@Observable
public final class FileTranscriptionQueue {
    public struct Row: Sendable, Hashable, Identifiable {
        public enum Phase: Sendable, Hashable { case queued, running, paused, done, failed(String) }
        public var id: UUID { noteID }
        public var noteID: UUID
        public var fileName: String
        public var phase: Phase
        public var duration: TimeInterval?
        public var fraction: Double
        public var remaining: TimeInterval?
        public var languageCode: String?
        /// Speakers in the finished transcript; nil until done.
        public var speakerCount: Int?
    }

    /// Queue rows in the order added; Done and failed rows stay until cleared.
    public private(set) var rows: [Row] = []

    /// Runs after a file's transcript is saved, when it asked for a summary.
    @ObservationIgnored public var afterTranscript: (@MainActor (_ noteID: UUID) -> Void)?
    /// Runs after every file's transcript is saved (the meeting session
    /// refreshes a transcript model already shown for that note).
    @ObservationIgnored public var transcriptSaved: (@MainActor (_ noteID: UUID) -> Void)?

    @ObservationIgnored private let speech: any SpeechServicing
    @ObservationIgnored private let context: ModelContext
    @ObservationIgnored private let writer: any MeetingNoteWriting
    @ObservationIgnored private let now: @Sendable () -> Date
    @ObservationIgnored private var jobs: [UUID: FileJob] = [:]
    @ObservationIgnored private var consumers: [UUID: Task<Void, Never>] = [:]
    @ObservationIgnored private var didRecover = false
    /// Files whose job is being created right now, and those cancelled meanwhile.
    @ObservationIgnored private var starting: Set<UUID> = []
    @ObservationIgnored private var cancelledWhileStarting: Set<UUID> = []
    /// Files the person cancelled. Any other `.cancelled` (quit: the speech
    /// service cancels every job) leaves the note queued for the next launch.
    @ObservationIgnored private var cancelledByPerson: Set<UUID> = []

    /// Recordings are read and saved on `writer.context` (the Notes store's).
    public init(speech: any SpeechServicing, writer: any MeetingNoteWriting,
                now: @escaping @Sendable () -> Date = { Date.now }) {
        self.speech = speech
        self.context = writer.context
        self.writer = writer
        self.now = now
    }

    /// Whether any file is waiting or running (the speech status card).
    public var isBusy: Bool {
        rows.contains { row in
            switch row.phase {
            case .queued, .running, .paused: true
            case .done, .failed: false
            }
        }
    }

    // MARK: Adding

    /// Creates one note per file and queues it. Returns the notes made; a
    /// file the speech service refuses gets a failed row and no note.
    @discardableResult
    public func add(_ urls: [URL], options: FileImportOptions) async -> [UUID] {
        var added: [UUID] = []
        for url in urls {
            let noteID = UUID()
            let fileName = url.lastPathComponent
            if let id = await enqueue(url, fileName: fileName, noteID: noteID, options: options, createNote: true) {
                added.append(id)
            }
        }
        return added
    }

    /// Removes a file from the queue; its unfinished note goes too.
    public func cancel(_ noteID: UUID) async {
        guard jobs[noteID] != nil || starting.contains(noteID) else { return }
        cancelledByPerson.insert(noteID)
        guard let job = jobs[noteID] else {
            // The job is still being created: cancel it as soon as it exists.
            if starting.contains(noteID) { cancelledWhileStarting.insert(noteID) }
            return
        }
        await speech.cancel(job.id)
    }

    /// Clears Done and failed rows from the sheet's list.
    public func clearFinished() {
        rows.removeAll { row in
            switch row.phase {
            case .done, .failed: true
            case .queued, .running, .paused: false
            }
        }
    }

    /// Waits until every queued file has finished (tests, quit).
    public func waitUntilIdle() async {
        while let next = consumers.values.first { await next.value }
    }

    private func enqueue(_ url: URL, fileName: String, noteID: UUID, options: FileImportOptions,
                         createNote: Bool) async -> UUID? {
        if createNote {
            do {
                try writer.createNote(id: noteID, kind: .file, title: MeetingCopy.fileTitle(fileName), createdAt: now())
                guard let note = fetchNote(noteID) else { return nil }
                let record = NoteRecording(startedAt: now(), audioKept: options.keepAudio, sourceFileName: fileName)
                record.transcriptionState = .queued
                record.summaryState = options.summarize ? .pending : .none
                context.insert(record)
                note.recording = record
                try context.save()
            } catch {
                rows.append(Row(noteID: noteID, fileName: fileName, phase: .failed("Couldn't create the note."),
                                fraction: 0))
                return nil
            }
        }
        upsertRow(Row(noteID: noteID, fileName: fileName, phase: .queued, fraction: 0))
        let job: FileJob
        starting.insert(noteID)
        defer { starting.remove(noteID) }
        do {
            job = try await speech.transcribeFile(url, options: FileTranscriptionOptions(
                noteID: noteID, language: options.language, speakerLabels: options.speakerLabels,
                keepAudio: options.keepAudio))
        } catch {
            if cancelledWhileStarting.remove(noteID) != nil {
                // Cancelled while starting, and the start failed anyway: still a cancel.
                cancelledByPerson.remove(noteID)
                discard(noteID)
                return nil
            }
            let message = (error as? SpeechError)?.message ?? SpeechError.failed("").message
            fail(noteID, message: message)
            return createNote ? noteID : nil
        }
        jobs[noteID] = job
        let jobID = job.id
        consumers[noteID] = Task { [weak self] in
            for await event in job.events {
                guard let self else { return }
                self.handle(event, noteID: noteID)
            }
            self?.jobEnded(noteID, jobID: jobID)
        }
        if cancelledWhileStarting.remove(noteID) != nil {
            await speech.cancel(job.id)
        }
        return noteID
    }

    // MARK: Events

    private func handle(_ event: FileJobEvent, noteID: UUID) {
        switch event {
        case .queued:
            updateRow(noteID) { $0.phase = .queued }
            setState(.queued, noteID: noteID)
        case .started(let duration):
            updateRow(noteID) { $0.phase = .running; $0.duration = duration }
            setState(.running, noteID: noteID)
        case .progress(let fraction, let remaining, let languageCode):
            updateRow(noteID) { row in
                row.phase = .running
                row.fraction = fraction
                row.remaining = remaining
                row.languageCode = languageCode ?? row.languageCode
            }
        case .paused:
            updateRow(noteID) { $0.phase = .paused }
        case .finished(let result):
            finish(noteID, result: result)
        case .failed(let error):
            fail(noteID, message: error.message)
        case .cancelled:
            // The person's Cancel discards the file. Otherwise the speech
            // service stopped the job (quit): the note stays queued, and
            // `recoverInterruptedJobs()` starts it again at the next launch.
            guard cancelledByPerson.remove(noteID) != nil else {
                updateRow(noteID) { $0.phase = .queued }
                return
            }
            discard(noteID)
        }
    }

    /// Removes a cancelled file: its row, then its note and recordings
    /// through the Notes store. Never left "running": if the note can't be
    /// deleted for good it stays in Recently deleted as failed, not re-queued.
    private func discard(_ noteID: UUID) {
        rows.removeAll { $0.noteID == noteID }
        if let record = fetchNote(noteID)?.recording {
            record.transcriptionState = .failed
            record.summaryState = .none
            try? context.save()
        }
        Task { [writer] in await writer.discardNote(noteID) }
    }

    private func finish(_ noteID: UUID, result: TranscriptResult) {
        var summarize = false
        if let record = fetchNote(noteID)?.recording {
            record.segments = result.segments.filter(\.isFinal)
            record.speakers = result.speakers
            record.tracks = result.tracks
            record.gaps = result.gaps
            record.duration = result.duration
            record.languageCode = result.languageCode
            record.audioKept = record.audioKept && result.audioDirectory != nil && !result.tracks.isEmpty
            record.endedAt = now()
            record.transcriptionState = .done
            summarize = record.summaryState == .pending
            try? context.save()
        }
        writer.transcriptDidChange(in: noteID)
        updateRow(noteID) { row in
            row.phase = .done
            row.fraction = 1
            row.remaining = 0
            row.duration = row.duration ?? result.duration
            row.languageCode = result.languageCode ?? row.languageCode
            row.speakerCount = Set(result.segments.map(\.speakerID)).count
        }
        transcriptSaved?(noteID)
        if summarize { afterTranscript?(noteID) }
    }

    private func fail(_ noteID: UUID, message: String) {
        if let record = fetchNote(noteID)?.recording {
            record.transcriptionState = .failed
            if record.summaryState == .pending { record.summaryState = .none }
            try? context.save()
        }
        updateRow(noteID) { $0.phase = .failed(message) }
    }

    /// Forgets a finished job, unless the note has a newer one by now.
    private func jobEnded(_ noteID: UUID, jobID: FileJob.ID) {
        guard jobs[noteID]?.id == jobID else { return }
        jobs[noteID] = nil
        consumers[noteID] = nil
    }

    // MARK: Launch recovery

    /// Once at launch: files the app quit while transcribing start again
    /// from their copy in the recordings folder, or fail when it is gone.
    /// Returns the notes re-queued.
    @discardableResult
    public func recoverInterruptedJobs() async -> [UUID] {
        guard !didRecover else { return [] }
        didRecover = true
        let queued = TranscriptionState.queued.rawValue, running = TranscriptionState.running.rawValue
        let open = (try? context.fetch(FetchDescriptor<NoteRecording>(predicate: #Predicate {
            $0.transcriptionStateRawValue == queued || $0.transcriptionStateRawValue == running
        }))) ?? []
        var requeued: [UUID] = []
        for record in open {
            // A file added during launch is already queued: leave it.
            guard let note = record.note, note.deletedAt == nil, let fileName = record.sourceFileName,
                  jobs[note.id] == nil else { continue }
            guard let copy = Self.inputCopy(of: fileName, in: speech.recordingsDirectory(for: note.id)) else {
                upsertRow(Row(noteID: note.id, fileName: fileName, phase: .queued, fraction: 0))
                fail(note.id, message: MeetingCopy.fileGone)
                continue
            }
            let options = FileImportOptions(keepAudio: record.audioKept, summarize: record.summaryState == .pending)
            if await enqueue(copy, fileName: fileName, noteID: note.id, options: options, createNote: false) != nil {
                requeued.append(note.id)
                // The speech service copied it again for the new job.
                if copy.lastPathComponent.hasPrefix(Self.importPrefix) { try? FileManager.default.removeItem(at: copy) }
            }
        }
        return requeued
    }

    private nonisolated static let importPrefix = "import-"

    /// The speech service's copy of an imported file in the note's folder:
    /// `import-<8 hex>-<name>` (what `transcribeFile` writes), or the name
    /// itself. The newest wins when there are several.
    nonisolated static func inputCopy(of fileName: String, in folder: URL) -> URL? {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        let copies = names.filter { name in
            guard name.hasPrefix(importPrefix), name.hasSuffix("-" + fileName),
                  name.count == importPrefix.count + 9 + fileName.count else { return name == fileName }
            let tag = name.dropFirst(importPrefix.count).prefix(8)
            return tag.allSatisfy(\.isHexDigit)
        }
        .map { folder.appending(path: $0) }
        func modified(_ url: URL) -> Date {
            (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        }
        return copies.max { modified($0) < modified($1) }
    }

    // MARK: Helpers

    private func setState(_ state: TranscriptionState, noteID: UUID) {
        guard let record = fetchNote(noteID)?.recording, record.transcriptionState != state else { return }
        record.transcriptionState = state
        try? context.save()
    }

    private func upsertRow(_ row: Row) {
        if let index = rows.firstIndex(where: { $0.noteID == row.noteID }) { rows[index] = row } else { rows.append(row) }
    }

    private func updateRow(_ noteID: UUID, _ change: (inout Row) -> Void) {
        guard let index = rows.firstIndex(where: { $0.noteID == noteID }) else { return }
        change(&rows[index])
    }

    private func fetchNote(_ id: UUID) -> Note? {
        try? context.fetch(FetchDescriptor<Note>(predicate: #Predicate { $0.id == id })).first
    }
}
