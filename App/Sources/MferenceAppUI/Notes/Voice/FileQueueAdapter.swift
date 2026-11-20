import Foundation
import MferenceAppCore
import SwiftUI

/// The import sheet's view of N4's `FileTranscriptionQueue` (the one
/// consumer of every `FileJob.events`). Reads its rows as they change and
/// queues with the sheet's options plus the remembered Keep audio.
@MainActor
public final class FileQueueAdapter: FileQueueModel {
    private let queue: FileTranscriptionQueue
    private let preferences: SpeechPreferences

    public init(queue: FileTranscriptionQueue, preferences: SpeechPreferences) {
        self.queue = queue
        self.preferences = preferences
    }

    /// Observed through the queue's own `rows`, so the sheet updates.
    public var rows: [FileQueueRow] { queue.rows.map(Self.row) }

    public func add(_ urls: [URL], language: SpeechLanguage, speakerLabels: Bool, summary: Bool) async {
        let options = FileImportOptions(language: language, speakerLabels: speakerLabels,
                                        keepAudio: preferences.keepAudio == .keep,
                                        summarize: summary && preferences.summaryModel != .off)
        _ = await queue.add(urls, options: options)
    }

    public func cancel(_ id: UUID) {
        Task { [queue] in await queue.cancel(id) }
    }

    static func row(_ row: FileTranscriptionQueue.Row) -> FileQueueRow {
        FileQueueRow(id: row.noteID, noteID: row.noteID, fileName: row.fileName, duration: row.duration,
                     languageCode: row.languageCode, speakerCount: row.speakerCount, fraction: row.fraction,
                     remaining: row.remaining, phase: phase(row.phase))
    }

    static func phase(_ phase: FileTranscriptionQueue.Row.Phase) -> FileQueueRow.Phase {
        switch phase {
        case .queued: .queued
        case .running: .running
        case .paused: .paused
        case .done: .done
        case .failed(let reason): .failed(reason)
        }
    }
}

extension VoiceModels {
    /// Connects the import sheet to N4's file queue (the app calls it once).
    public func connect(files: FileTranscriptionQueue) {
        fileQueue = FileQueueAdapter(queue: files, preferences: preferences)
    }
}

extension VoiceModels {
    /// Connects Notes (N1's `NotesModel`) and the voice models: every
    /// `NotesVoiceActions` field (N1 defines and calls them, N3 sets them),
    /// and opening a note by id for the timer chip, the menu bar and the
    /// import queue's Open note.
    public func connect(notes: NotesModel) {
        notes.voice.transcribeFile = { [weak self] in self?.transcribeFiles() }
        notes.voice.recordMeeting = { [weak self] in self?.recordMeeting() }
        notes.voice.toolbar = { [weak self] in
            guard let self else { return AnyView(EmptyView()) }
            return AnyView(NotesVoiceToolbar(voice: self))
        }
        notes.voice.toggleDictation = { [weak self] editor in self?.toggleDictation(in: editor) }
        notes.voice.isDictating = { [weak self] noteID in self?.dictation.isDictating(into: noteID) ?? false }
        notes.voice.stopDictation = { [weak self] in await self?.stopDictation() }
        notes.voice.editorFocusChanged = { [weak self] editor in self?.focusedDictationTarget = editor }
        notes.voice.escape = { [weak self] in self?.cancelDictation() ?? false }
        notes.voice.transcribe = { [weak self] urls in self?.transcribe(urls) ?? false }
        openNote = { [weak notes] noteID in notes?.open(noteID: noteID) }
    }

    /// Mark moment (⇧⌘M) puts its marker at the caret when the Notes editor
    /// shows the meeting's note (N1's `MeetingNoteTarget`, which checks the
    /// note id), and at the end of the note through the store otherwise.
    public static func connectMarkers(meetings: MeetingSession, notes: NotesModel) {
        meetings.noteTarget = notes.editor
    }
}
