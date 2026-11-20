import Foundation

// SHARED CONTRACT (Notes program spec 6.2). Only `MferenceAppCore/Speech`
// imports FluidAudio, AVFoundation capture or Core Audio taps; everything
// else (Notes, Meetings, the UI) talks to `SpeechServicing`. Owner: the
// speech team (N2) implements `SpeechService`; `FakeSpeechService` drives
// every test and preview. Changing a requirement is a contract PR.
//
// One coordinator owns the microphone and the model: one live session
// (dictation or meeting) at a time; starting a second throws
// `SpeechError.busy`. File jobs run one at a time and pause while a live
// session runs. No call here downloads anything except `install`.
//
// STREAM OWNERSHIP: `DictationSession.events`, `MeetingRecording.events` and
// `FileJob.events` are single-consumer `AsyncStream`s. Each must be consumed
// by one long-lived `MferenceAppCore` owner held by `AppServices` (the
// meeting session, the file-transcription queue, the dictation controller),
// never by a SwiftUI view's `.task`: leaving the screen cancels the task,
// which ends the stream and loses the rest (a meeting keeps recording when
// you leave Notes, D7; files keep transcribing with the sheet closed, 4.6).
// Views observe those owners instead. Two tasks awaiting one stream trap.

public protocol SpeechServicing: AnyObject, Sendable {
    /// The current status, without waiting.
    var status: SpeechStatus { get }
    /// Every status change, starting with the current one. Each call returns
    /// a new stream; many observers may listen.
    func statusUpdates() -> AsyncStream<SpeechStatus>

    // MARK: Models (spec 4.7, 7)

    /// Bytes the package downloads, from its pinned file list.
    func downloadSize(of package: SpeechPackageID) -> Int64
    /// Downloads (resuming a paused download), verifies every file against
    /// its pinned SHA-256, then runs the first load. Returns when `.ready`.
    /// Throws `SpeechError.offline` in Offline mode, and `CancellationError`
    /// when `pauseInstall` or `cancelInstall` stops it (the status then says
    /// `.paused` or `.notInstalled`). A second call while one runs waits for it.
    func install(_ package: SpeechPackageID) async throws
    func pauseInstall(_ package: SpeechPackageID) async
    /// Stops and deletes the partial download.
    func cancelInstall(_ package: SpeechPackageID) async
    /// Deletes the package's files; voice features go back to setup.
    func remove(_ package: SpeechPackageID) async throws

    // MARK: Live sessions

    /// Starts dictation. The caller asked for the microphone permission
    /// first. Throws `.notInstalled`, `.busy`, `.permissionDenied`.
    func startDictation(_ options: DictationOptions) async throws -> DictationSession
    /// Starts a meeting recording into `Recordings/<options.noteID>/`.
    /// Throws `.notInstalled`, `.busy`, `.permissionDenied(.microphone)`,
    /// `.lowDiskSpace`. Denied call audio is not an error: the recording
    /// continues on the microphone and says so (`MeetingEvent.callAudioUnavailable`).
    func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording

    // MARK: Files (spec 4.6)

    /// Copies the file into `Recordings/<options.noteID>/` and queues it.
    func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob
    func cancel(_ job: FileJob.ID) async

    // MARK: Devices and storage

    func inputDevices() async -> [AudioInputDevice]
    /// Running apps that can be chosen as the call-audio source.
    func runningAudioApps() async -> [RunningAudioApp]
    /// Whether a meeting's call tap has ever heard other apps. The call-audio
    /// permission reads `.granted` once it was asked, because macOS can't be
    /// asked whether it was allowed (a refused tap only hears silence), so
    /// until this is true Settings shows call audio as asked, not allowed.
    /// Never prompts; never means denied.
    var hasHeardCallAudio: Bool { get }
    /// Bytes used by kept recordings (Settings › Speech › Recordings).
    func recordingsSize() async -> Int64
    /// Deletes a note's recordings folder (purge, or "Delete after transcribing").
    func deleteRecordings(for noteID: UUID) async throws
    /// Where a note's audio lives, for playback and Show in Finder.
    func recordingsDirectory(for noteID: UUID) -> URL

    /// Meetings the app quit in, with their audio files closed and usable;
    /// called once at launch, before any new session (spec 6.3).
    func recoverInterruptedRecordings() async -> [InterruptedRecording]

    /// Stops any live session (its stream gets `.ended`) and file jobs, for
    /// quit (spec 6.1). Returns when capture has stopped and files are closed.
    func shutdown() async
}

// MARK: Options

public struct DictationOptions: Sendable, Hashable {
    public var language: SpeechLanguage
    /// Remove "um", "uh" and repeated words (spec 6.4).
    public var cleanUpFillers: Bool
    /// nil uses the system default input.
    public var microphoneID: String?
    /// Silence that ends a phrase; the spec's default is 500 ms.
    public var endOfSpeechSilence: Duration

    public init(language: SpeechLanguage = .automatic, cleanUpFillers: Bool = true,
                microphoneID: String? = nil, endOfSpeechSilence: Duration = .milliseconds(500)) {
        self.language = language
        self.cleanUpFillers = cleanUpFillers
        self.microphoneID = microphoneID
        self.endOfSpeechSilence = endOfSpeechSilence
    }
}

public struct MeetingOptions: Sendable, Hashable {
    /// The meeting note; audio goes to `Recordings/<noteID>/`.
    public var noteID: UUID
    public var microphoneID: String?
    public var callAudio: CallAudioSource
    /// Split the call track into Speaker 2, 3, … (needs `.speakerLabels`).
    public var speakerLabels: Bool
    /// Keep the audio after transcribing (D4).
    public var keepAudio: Bool
    public var language: SpeechLanguage

    public init(noteID: UUID, microphoneID: String? = nil, callAudio: CallAudioSource = .allApps,
                speakerLabels: Bool = true, keepAudio: Bool = true, language: SpeechLanguage = .automatic) {
        self.noteID = noteID
        self.microphoneID = microphoneID
        self.callAudio = callAudio
        self.speakerLabels = speakerLabels
        self.keepAudio = keepAudio
        self.language = language
    }
}

public struct FileTranscriptionOptions: Sendable, Hashable {
    /// The note the transcript lands in; the copy goes to `Recordings/<noteID>/`.
    public var noteID: UUID
    public var language: SpeechLanguage
    public var speakerLabels: Bool
    public var keepAudio: Bool

    public init(noteID: UUID, language: SpeechLanguage = .automatic, speakerLabels: Bool = false,
                keepAudio: Bool = true) {
        self.noteID = noteID
        self.language = language
        self.speakerLabels = speakerLabels
        self.keepAudio = keepAudio
    }
}

// MARK: Dictation

public enum DictationEvent: Sendable, Hashable {
    /// Speech began: hold the caret position for this phrase.
    case phraseStarted(UUID)
    /// The phrase's text, filler-filtered if asked; insert it where its caret was.
    case phraseFinished(UUID, text: String, languageCode: String?)
    /// The phrase had no words (a cough): remove its pending marker.
    case phraseDropped(UUID)
    /// Input level, 0...1, about 20 times a second.
    case level(Float)
    case failed(SpeechError)
    /// The session is over; the stream finishes after this.
    case ended
}

/// A running dictation. `events` is single-consumer: the dictation bar owns it.
public final class DictationSession: Sendable {
    public let id: UUID
    public let events: AsyncStream<DictationEvent>
    private let onStop: @Sendable () async -> Void
    private let onLanguage: @Sendable (SpeechLanguage) async -> Void

    public init(id: UUID = UUID(), events: AsyncStream<DictationEvent>,
                stop: @escaping @Sendable () async -> Void,
                setLanguage: @escaping @Sendable (SpeechLanguage) async -> Void) {
        self.id = id
        self.events = events
        self.onStop = stop
        self.onLanguage = setLanguage
    }

    /// Finishes the phrase in progress, then ends (Done, Esc, ⇧⌘D, switching
    /// notes). Returns after the last `.phraseFinished`/`.phraseDropped` and
    /// `.ended` are in the stream. Calling it again is a no-op.
    public func stop() async { await onStop() }
    /// The language chip: pin one or go back to automatic.
    public func setLanguage(_ language: SpeechLanguage) async { await onLanguage(language) }
}

// MARK: Meetings

public enum MeetingEvent: Sendable, Hashable {
    /// Capture is running; `callAudio` is false when it was denied or off.
    case started(callAudio: Bool)
    /// Call audio was denied or failed: the recording continues on the mic.
    case callAudioUnavailable(SpeechError)
    /// A hint, at most once per meeting: call audio is on, but 30 s of
    /// recording in, the tap has heard only silence while other apps were
    /// playing sound (`noCallAudioHeardMessage`). A refused permission sounds
    /// like this, so does a call that is muted or not started; the meeting
    /// keeps recording both tracks and nothing is stored as denied.
    case noCallAudioHeard
    /// A new or updated segment (the one still being spoken has `isFinal == false`).
    case segment(TranscriptSegment)
    /// A mic segment dropped as an echo of a call segment (spec 6.3).
    case segmentRemoved(UUID)
    /// Levels 0...1; `call` is nil without call audio.
    case levels(microphone: Float, call: Float?)
    /// The live diarizer found a new speaker or relabelled one.
    case speakersUpdated([SpeakerInfo])
    case paused
    case resumed
    /// The Mac slept; capture stopped and resumed in new files.
    case gap(RecordingGap)
    /// After stop: finishing phrases, then the final speaker pass.
    case finishing(MeetingFinishStep)
    case failed(SpeechError)
    /// Final result; the stream finishes after this. Segments keep the ids
    /// they had live (only `speakerID` may change); `speakerMapping` maps
    /// live speaker ids to final ones.
    case ended(TranscriptResult)
}

extension MeetingEvent {
    /// The copy for `.noCallAudioHeard`.
    public static let noCallAudioHeardMessage =
        "No call audio yet. If the other side is talking, check Call audio in System Settings."
}

public enum MeetingFinishStep: Sendable, Hashable {
    case transcribingLastPhrases
    /// 0...1
    case labellingSpeakers(progress: Double)
}

/// A running meeting. `events` is single-consumer: the meeting session model owns it.
public final class MeetingRecording: Sendable {
    public struct Controls: Sendable {
        public var pause: @Sendable () async -> Void
        public var resume: @Sendable () async -> Void
        /// Returns the recording time of the mark.
        public var markMoment: @Sendable () async -> TimeInterval
        /// Ends capture and returns once capture has stopped; the finishing
        /// steps and the result follow as events. Calling it again is a no-op.
        public var stop: @Sendable () async -> Void

        public init(pause: @escaping @Sendable () async -> Void, resume: @escaping @Sendable () async -> Void,
                    markMoment: @escaping @Sendable () async -> TimeInterval,
                    stop: @escaping @Sendable () async -> Void) {
            self.pause = pause
            self.resume = resume
            self.markMoment = markMoment
            self.stop = stop
        }
    }

    public let noteID: UUID
    public let startedAt: Date
    public let events: AsyncStream<MeetingEvent>
    private let controls: Controls

    public init(noteID: UUID, startedAt: Date, events: AsyncStream<MeetingEvent>, controls: Controls) {
        self.noteID = noteID
        self.startedAt = startedAt
        self.events = events
        self.controls = controls
    }

    public func pause() async { await controls.pause() }
    public func resume() async { await controls.resume() }
    public func markMoment() async -> TimeInterval { await controls.markMoment() }
    public func stop() async { await controls.stop() }
}

// MARK: Files

public enum FileJobEvent: Sendable, Hashable {
    /// Waiting; 0 is next.
    case queued(position: Int)
    /// Decoding started; the audio's length is known.
    case started(duration: TimeInterval)
    /// `fraction` 0...1 of audio seconds done; `remaining` from measured speed.
    case progress(fraction: Double, remaining: TimeInterval?, languageCode: String?)
    /// Waiting while dictation or a meeting runs.
    case paused
    case finished(TranscriptResult)
    case failed(SpeechError)
    case cancelled
}

/// One queued file. `events` is single-consumer: the import sheet's queue
/// row owns it; the status card reads `SpeechStatus.fileQueue`.
public final class FileJob: Sendable, Identifiable {
    public typealias ID = UUID
    public let id: ID
    public let noteID: UUID
    public let fileName: String
    public let events: AsyncStream<FileJobEvent>

    public init(id: ID = UUID(), noteID: UUID, fileName: String, events: AsyncStream<FileJobEvent>) {
        self.id = id
        self.noteID = noteID
        self.fileName = fileName
        self.events = events
    }
}

/// A queue row as the status card and the sheet's list show it.
public struct FileJobStatus: Sendable, Hashable, Identifiable {
    public enum Phase: Sendable, Hashable { case queued, running, paused, done, failed }
    public var id: FileJob.ID
    public var noteID: UUID
    public var fileName: String
    public var duration: TimeInterval?
    public var fraction: Double
    public var remaining: TimeInterval?
    public var languageCode: String?
    public var phase: Phase

    public init(id: FileJob.ID, noteID: UUID, fileName: String, duration: TimeInterval? = nil,
                fraction: Double = 0, remaining: TimeInterval? = nil, languageCode: String? = nil,
                phase: Phase = .queued) {
        self.id = id
        self.noteID = noteID
        self.fileName = fileName
        self.duration = duration
        self.fraction = fraction
        self.remaining = remaining
        self.languageCode = languageCode
        self.phase = phase
    }
}
