import Foundation
import Observation
import SwiftData

/// What the record popover chose (spec 4.3); remembered in Settings › Speech.
public struct MeetingStartOptions: Sendable, Hashable {
    public var microphoneID: String?
    public var callAudio: CallAudioSource
    public var speakerLabels: Bool
    public var keepAudio: Bool
    public var language: SpeechLanguage
    /// "Summary when you stop" (D5), default on.
    public var summaryWhenStopped: Bool

    public init(microphoneID: String? = nil, callAudio: CallAudioSource = .allApps, speakerLabels: Bool = true,
                keepAudio: Bool = true, language: SpeechLanguage = .automatic, summaryWhenStopped: Bool = true) {
        self.microphoneID = microphoneID
        self.callAudio = callAudio
        self.speakerLabels = speakerLabels
        self.keepAudio = keepAudio
        self.language = language
        self.summaryWhenStopped = summaryWhenStopped
    }
}

/// The live state of the one meeting being recorded (spec 4.4, 4.5, 6.3).
///
/// `AppServices` holds one; it is the single consumer of
/// `MeetingRecording.events`, so a meeting keeps recording and saving when
/// the person leaves Notes (D7). Views observe it. Final segments are saved
/// into the note's `NoteRecording` as they arrive (at most one phrase lost on
/// a crash); on `.ended` the final transcript, speakers (with renames carried
/// over), tracks and gaps are saved and `afterMeeting` runs the summary.
@MainActor
@Observable
public final class MeetingSession {
    public enum Phase: Sendable, Hashable {
        case idle
        case starting
        case recording
        case paused
        /// After Stop: the last phrases, then the final speaker pass.
        case finishing(MeetingFinishStep)
    }

    public private(set) var phase: Phase = .idle
    /// The meeting note being recorded; nil when idle.
    public private(set) var noteID: UUID?
    /// The live transcript; nil when idle.
    public private(set) var transcript: TranscriptModel?
    public private(set) var micLevel: Float = 0
    /// Nil without call audio.
    public private(set) var callLevel: Float?
    /// False when call audio was off, denied or failed: the meeting is mic only.
    public private(set) var callAudioAvailable = false
    /// Recording times of Mark moment, in order.
    public private(set) var marks: [TimeInterval] = []
    /// The last problem, as one line the UI can show.
    public private(set) var lastError: String?
    /// The note of the meeting that ended last (the finished view opens it).
    public private(set) var lastEndedNoteID: UUID?

    public var isActive: Bool { phase != .idle }

    /// Runs after a meeting's transcript is saved: `summarize` is the
    /// meeting's "Summary when you stop". The summary layer sets it.
    @ObservationIgnored public var afterMeeting: (@MainActor (_ noteID: UUID, _ summarize: Bool) -> Void)?

    @ObservationIgnored private let speech: any SpeechServicing
    @ObservationIgnored private let context: ModelContext
    @ObservationIgnored private let writer: any MeetingNoteWriting
    @ObservationIgnored private let now: @Sendable () -> Date
    @ObservationIgnored private let locale: Locale
    @ObservationIgnored private let timeZone: TimeZone
    @ObservationIgnored private var recording: MeetingRecording?
    @ObservationIgnored private var consumer: Task<Void, Never>?
    @ObservationIgnored private var options = MeetingStartOptions()
    @ObservationIgnored private var didRecover = false
    /// Finished notes' transcripts, so a rename shows everywhere at once.
    @ObservationIgnored private var transcripts: [UUID: TranscriptModel] = [:]

    public init(speech: any SpeechServicing, context: ModelContext, writer: any MeetingNoteWriting,
                now: @escaping @Sendable () -> Date = { Date.now }, locale: Locale = .current,
                timeZone: TimeZone = .current) {
        self.speech = speech
        self.context = context
        self.writer = writer
        self.now = now
        self.locale = locale
        self.timeZone = timeZone
    }

    // MARK: Recording

    /// Starts capture, then creates the meeting note with a time-stamped
    /// title. Returns the note's id. Throws `SpeechError` (busy, not
    /// installed, microphone denied, low disk space); no note is left behind.
    @discardableResult
    public func start(_ options: MeetingStartOptions) async throws -> UUID {
        guard phase == .idle else { throw SpeechError.busy(.meeting) }
        phase = .starting
        lastError = nil
        let noteID = UUID()
        let started: MeetingRecording
        do {
            started = try await speech.startMeeting(MeetingOptions(
                noteID: noteID, microphoneID: options.microphoneID, callAudio: options.callAudio,
                speakerLabels: options.speakerLabels, keepAudio: options.keepAudio, language: options.language))
        } catch {
            phase = .idle
            lastError = (error as? SpeechError)?.message
            throw error
        }
        do {
            try writer.createNote(id: noteID, kind: .meeting, title: MeetingCopy.defaultTitle(
                started.startedAt, locale: locale, timeZone: timeZone), createdAt: started.startedAt)
            guard let note = fetchNote(noteID) else { throw SpeechError.failed("The meeting note was not created.") }
            let record = NoteRecording(startedAt: started.startedAt, audioKept: options.keepAudio)
            record.transcriptionState = .live
            context.insert(record)
            note.recording = record
            try context.save()
        } catch {
            await started.stop()
            for await _ in started.events {}
            phase = .idle
            lastError = "Couldn't create the meeting note."
            throw error
        }
        self.options = options
        self.recording = started
        self.noteID = noteID
        let live = TranscriptModel()
        live.onChange = { [weak self] model in self?.saveSpeakers(model, noteID: noteID) }
        transcript = live
        transcripts[noteID] = live
        marks = []
        micLevel = 0
        callLevel = nil
        callAudioAvailable = false
        phase = .recording
        consumer = Task { [weak self] in
            for await event in started.events {
                guard let self else { return }
                self.handle(event, noteID: noteID)
            }
            self?.streamFinished(noteID: noteID)
        }
        return noteID
    }

    public func pause() async {
        guard phase == .recording, let recording else { return }
        await recording.pause()
    }

    public func resume() async {
        guard phase == .paused, let recording else { return }
        await recording.resume()
    }

    /// Marks this moment: a marker line in the note at the recording time.
    @discardableResult
    public func markMoment() async -> TimeInterval? {
        guard phase == .recording || phase == .paused, let recording, let noteID else { return nil }
        let time = await recording.markMoment()
        marks.append(time)
        do { try writer.appendMarker(at: time, to: noteID) } catch { lastError = "Couldn't add the mark." }
        return time
    }

    /// Ends capture; the finishing steps and the saved result follow.
    public func stop() async {
        guard let recording, phase == .recording || phase == .paused else { return }
        await recording.stop()
    }

    /// Waits until the current meeting's events are all handled (tests, quit).
    public func waitUntilFinished() async {
        await consumer?.value
    }

    // MARK: Transcripts of any note

    /// The transcript of a meeting or file note: the live one while it
    /// records, else loaded from the store (one model per note, so a rename
    /// shows everywhere). Nil when the note has no recording.
    public func transcript(for noteID: UUID) -> TranscriptModel? {
        if let model = transcripts[noteID] { return model }
        guard let record = fetchNote(noteID)?.recording else { return nil }
        let model = TranscriptModel(recording: record)
        model.onChange = { [weak self] model in self?.saveSpeakers(model, noteID: noteID) }
        transcripts[noteID] = model
        return model
    }

    /// Forgets a cached transcript (the note was purged or re-transcribed).
    public func forgetTranscript(for noteID: UUID) {
        guard noteID != self.noteID else { return }
        transcripts[noteID] = nil
    }

    // MARK: Launch recovery (spec 6.3)

    /// Once at launch, before any new session: closes meetings the app quit
    /// in with what was saved, and adds "Recording ended unexpectedly at
    /// hh:mm" to each note. Returns the notes closed.
    @discardableResult
    public func recoverInterruptedMeetings() async -> [UUID] {
        guard !didRecover else { return [] }
        didRecover = true
        var closed: [UUID] = []
        for interrupted in await speech.recoverInterruptedRecordings() {
            guard let record = fetchNote(interrupted.noteID)?.recording, record.endedAt == nil else { continue }
            if !interrupted.tracks.isEmpty { record.tracks = interrupted.tracks }
            closeInterrupted(record, noteID: interrupted.noteID, endedAt: interrupted.endedAt)
            closed.append(interrupted.noteID)
        }
        let open = (try? context.fetch(FetchDescriptor<NoteRecording>(
            predicate: #Predicate { $0.endedAt == nil }))) ?? []
        for record in open {
            guard let note = record.note, note.kind == .meeting, note.id != noteID else { continue }
            closeInterrupted(record, noteID: note.id, endedAt: nil)
            closed.append(note.id)
        }
        return closed
    }

    private func closeInterrupted(_ record: NoteRecording, noteID: UUID, endedAt: Date?) {
        let saved = record.segments.filter(\.isFinal)
        let trackLength = record.tracks.map { $0.startOffset + $0.duration }.max() ?? 0
        let length = max(record.duration, trackLength, saved.map(\.end).max() ?? 0)
        let end = endedAt ?? record.startedAt.addingTimeInterval(length)
        record.segments = saved
        record.duration = length
        record.endedAt = end
        record.transcriptionState = .done
        record.audioKept = record.audioKept && !record.tracks.isEmpty
        try? context.save()
        try? writer.appendParagraph(MeetingCopy.endedUnexpectedly(at: end, locale: locale, timeZone: timeZone),
                                    to: noteID)
        writer.transcriptDidChange(in: noteID)
    }

    // MARK: Events

    private func handle(_ event: MeetingEvent, noteID: UUID) {
        guard let transcript else { return }
        switch event {
        case .started(let callAudio):
            callAudioAvailable = callAudio
        case .callAudioUnavailable:
            callAudioAvailable = false
            callLevel = nil
            try? writer.appendParagraph(MeetingCopy.micOnly, to: noteID)
        case .segment(let segment):
            transcript.upsert(segment)
            if segment.isFinal { saveSegments(noteID: noteID) }
        case .segmentRemoved(let id):
            let wasSaved = transcript.segments.contains { $0.id == id && $0.isFinal }
            transcript.remove(id)
            if wasSaved { saveSegments(noteID: noteID) }
        case .levels(let microphone, let call):
            micLevel = microphone
            callLevel = call
        case .speakersUpdated(let speakers):
            transcript.updateSpeakers(speakers)
            saveSpeakers(transcript, noteID: noteID)
        case .paused:
            if phase == .recording { phase = .paused }
        case .resumed:
            if phase == .paused { phase = .recording }
        case .gap(let gap):
            transcript.addGap(gap)
            if let record = fetchNote(noteID)?.recording {
                record.gaps = transcript.gaps
                try? context.save()
            }
        case .finishing(let step):
            phase = .finishing(step)
        case .failed(let error):
            lastError = error.message
        case .ended(let result):
            finish(result, noteID: noteID)
        }
    }

    private func finish(_ result: TranscriptResult, noteID: UUID) {
        guard let transcript else { return }
        transcript.applyFinal(result)
        if let record = fetchNote(noteID)?.recording {
            record.segments = transcript.segments
            record.speakers = transcript.speakers
            record.gaps = transcript.gaps
            record.tracks = result.tracks
            record.duration = result.duration
            record.languageCode = result.languageCode ?? record.languageCode
            record.audioKept = options.keepAudio && result.audioDirectory != nil && !result.tracks.isEmpty
            record.endedAt = now()
            record.transcriptionState = .done
            if options.summaryWhenStopped { record.summaryState = .pending }
            try? context.save()
        }
        writer.transcriptDidChange(in: noteID)
        let summarize = options.summaryWhenStopped
        reset(endedNoteID: noteID)
        afterMeeting?(noteID, summarize)
    }

    /// The stream ended. Without an `.ended` (a failure), close the note
    /// with what was saved.
    private func streamFinished(noteID: UUID) {
        guard self.noteID == noteID else { return }
        if let record = fetchNote(noteID)?.recording, record.endedAt == nil {
            record.segments = transcript?.finalSegments ?? record.segments
            record.duration = max(record.duration, transcript?.duration ?? 0)
            record.endedAt = now()
            record.transcriptionState = .done
            try? context.save()
            writer.transcriptDidChange(in: noteID)
        }
        reset(endedNoteID: noteID)
    }

    private func reset(endedNoteID: UUID) {
        recording = nil
        consumer = nil
        noteID = nil
        transcript = nil
        micLevel = 0
        callLevel = nil
        phase = .idle
        lastEndedNoteID = endedNoteID
    }

    // MARK: Saving

    private func saveSegments(noteID: UUID) {
        guard let transcript, let record = fetchNote(noteID)?.recording else { return }
        record.segments = transcript.finalSegments
        record.speakers = transcript.speakers
        record.duration = max(record.duration, transcript.duration)
        try? context.save()
    }

    private func saveSpeakers(_ model: TranscriptModel, noteID: UUID) {
        guard let record = fetchNote(noteID)?.recording else { return }
        record.speakers = model.speakers
        try? context.save()
    }

    private func fetchNote(_ id: UUID) -> Note? {
        try? context.fetch(FetchDescriptor<Note>(predicate: #Predicate { $0.id == id })).first
    }
}

/// The meeting lines the note shows (spec 4.3, 6.3).
public enum MeetingCopy {
    /// The title a meeting note has until the summary names it: "Meeting 19 Nov at 10:02" (en_GB).
    public static func defaultTitle(_ date: Date, locale: Locale = .current, timeZone: TimeZone = .current) -> String {
        "Meeting " + date.formatted(Date.FormatStyle(locale: locale, timeZone: timeZone)
            .day().month(.abbreviated).hour().minute())
    }

    public static func endedUnexpectedly(at date: Date, locale: Locale = .current,
                                         timeZone: TimeZone = .current) -> String {
        "Recording ended unexpectedly at "
            + date.formatted(Date.FormatStyle(locale: locale, timeZone: timeZone).hour().minute())
    }

    public static let micOnly = "Call audio was off, so only your microphone was recorded."

    /// A transcribed file's title until the summary names it: the file name
    /// without its extension (spec 4.6).
    /// A file the app quit while transcribing, whose copy is gone.
    public static let fileGone = "The file's copy is gone. Add the file again."

    public static func fileTitle(_ fileName: String) -> String {
        let stem = (fileName as NSString).deletingPathExtension
        return stem.isEmpty ? fileName : stem
    }
}
