import Foundation

// SHARED CONTRACT (Notes program spec 6.2): the values that cross
// `SpeechServicing`. Owner: the speech team (N2). Segments, speakers and
// tracks are stored as JSON in `NoteRecording`, so changing a stored field is
// a contract PR; adding an optional field is not.

// MARK: Packages and state

/// A downloadable set of speech model files (spec 7).
public enum SpeechPackageID: String, Sendable, Hashable, Codable, CaseIterable {
    /// Fluid Speech Pro plus the voice-activity model: dictation, meetings, files.
    case speech
    /// The diarization models behind "Speaker 2, 3, …" (D3).
    case speakerLabels
}

/// Where one package stands. The sidebar's speech card and Settings show it.
public enum SpeechModelState: Sendable, Hashable {
    case notInstalled
    /// Downloading; `progress` is 0...1 of the package's bytes.
    case downloading(progress: Double)
    /// A download paused by the person or by Offline mode; resumable.
    case paused(progress: Double)
    /// Verified on disk; the first load (the Neural Engine compile) runs.
    case preparing
    case ready
    /// Needs attention: the message says what happened and what to do.
    case failed(SpeechError)

    public var isReady: Bool { self == .ready }
}

/// Everything the speech status card, Settings and the file queue show.
public struct SpeechStatus: Sendable, Hashable {
    public var speech: SpeechModelState
    public var speakerLabels: SpeechModelState
    /// The live session holding the microphone, if any (the recording chip
    /// and the menu bar item read it).
    public var liveSession: LiveSessionStatus?
    /// Files waiting or running (the queue keeps going with the sheet closed).
    public var fileQueue: [FileJobStatus]

    public init(speech: SpeechModelState = .notInstalled, speakerLabels: SpeechModelState = .notInstalled,
                liveSession: LiveSessionStatus? = nil, fileQueue: [FileJobStatus] = []) {
        self.speech = speech
        self.speakerLabels = speakerLabels
        self.liveSession = liveSession
        self.fileQueue = fileQueue
    }

    public func state(of package: SpeechPackageID) -> SpeechModelState {
        switch package {
        case .speech: speech
        case .speakerLabels: speakerLabels
        }
    }
}

public enum LiveSessionKind: String, Sendable, Hashable, Codable {
    case dictation, meeting
}

/// The session holding the microphone.
public struct LiveSessionStatus: Sendable, Hashable {
    public var kind: LiveSessionKind
    /// The note it writes into (the meeting note, or the note being dictated into).
    public var noteID: UUID?
    public var startedAt: Date
    public var isPaused: Bool
    /// Recording time so far, pauses excluded, as of `updatedAt`.
    public var recordedTime: TimeInterval
    public var updatedAt: Date

    public init(kind: LiveSessionKind, noteID: UUID?, startedAt: Date, isPaused: Bool = false,
                recordedTime: TimeInterval = 0, updatedAt: Date? = nil) {
        self.kind = kind
        self.noteID = noteID
        self.startedAt = startedAt
        self.isPaused = isPaused
        self.recordedTime = recordedTime
        self.updatedAt = updatedAt ?? startedAt
    }

    /// Recording time at `now`: the timer chip's text.
    public func recordedTime(at now: Date) -> TimeInterval {
        isPaused ? recordedTime : recordedTime + max(0, now.timeIntervalSince(updatedAt))
    }
}

/// What a speech call can fail with; `message` is the line the UI shows.
public enum SpeechError: Error, Sendable, Hashable {
    /// The package must be downloaded first (the setup card).
    case notInstalled(SpeechPackageID)
    /// Another live session holds the microphone.
    case busy(LiveSessionKind)
    case permissionDenied(AppPermission)
    /// Under 1 GB free (spec 6.3); `availableBytes` is what is left.
    case lowDiskSpace(availableBytes: Int64)
    /// Offline mode is on, so nothing downloads.
    case offline
    case downloadFailed(String)
    /// A downloaded file did not match its pinned size or SHA-256.
    case verificationFailed(file: String)
    /// The file has no audio track AVFoundation can decode.
    case unsupportedFile(String)
    /// The model or audio pipeline failed; the detail is for the log.
    case failed(String)

    public var message: String {
        switch self {
        case .notInstalled(.speech): "Download Fluid Speech Pro to use voice features."
        case .notInstalled(.speakerLabels): "Download speaker labels to tell speakers apart."
        case .busy(.dictation): "Dictation is on. Finish it first."
        case .busy(.meeting): "A meeting is recording. Stop it first."
        case .permissionDenied(.systemAudio): "Call audio is off. Turn it on in System Settings to record the other side."
        case .permissionDenied: "The microphone is off. Turn it on in System Settings to use voice features."
        case .lowDiskSpace: "Less than 1 GB is free on this Mac. Free some space to record."
        case .offline: "Turn off Offline mode in Settings to download."
        case .downloadFailed: "The download stopped. Try again."
        case .verificationFailed: "A downloaded file didn't match. Download again."
        case .unsupportedFile: "This file has no audio Mference can read."
        case .failed: "Speech stopped unexpectedly. Try again."
        }
    }
}

// MARK: Languages

/// The 25 European languages Fluid Speech Pro transcribes (spec 3).
public enum SpeechLanguage: Sendable, Hashable, Codable {
    /// Detected per phrase.
    case automatic
    /// Pinned by the person; BCP 47 code from `supportedCodes`.
    case fixed(String)

    public static let supportedCodes: [String] = [
        "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it",
        "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
    ]

    /// The language's name in the person's locale ("German").
    public static func displayName(for code: String, locale: Locale = .current) -> String {
        locale.localizedString(forLanguageCode: code) ?? code
    }
}

// MARK: Transcript values (stored in NoteRecording)

/// Which audio a segment came from.
public enum TrackKind: String, Sendable, Hashable, Codable {
    /// The microphone: always "You" in meetings.
    case microphone
    /// Other apps' audio through the process tap.
    case call
    /// An imported audio or video file.
    case file
}

/// One transcribed phrase. Times are in recording time (seconds from the
/// start, pauses excluded), from our own voice-activity boundaries (spec 6.3).
public struct TranscriptSegment: Sendable, Hashable, Codable, Identifiable {
    public var id: UUID
    public var track: TrackKind
    public var start: TimeInterval
    public var end: TimeInterval
    /// A `SpeakerInfo.id`.
    public var speakerID: String
    public var text: String
    /// False while the phrase is still being spoken or transcribed.
    public var isFinal: Bool
    /// BCP 47 code, when detected.
    public var languageCode: String?

    public init(id: UUID = UUID(), track: TrackKind, start: TimeInterval, end: TimeInterval,
                speakerID: String, text: String, isFinal: Bool, languageCode: String? = nil) {
        self.id = id
        self.track = track
        self.start = start
        self.end = end
        self.speakerID = speakerID
        self.text = text
        self.isFinal = isFinal
        self.languageCode = languageCode
    }
}

/// A speaker in one recording. Renames change `name` only; `id` is stable.
public struct SpeakerInfo: Sendable, Hashable, Codable, Identifiable {
    public static let youID = "you"
    public static let othersID = "others"

    public var id: String
    public var name: String
    /// Index into the design's speaker palette.
    public var colorIndex: Int

    public init(id: String, name: String, colorIndex: Int) {
        self.id = id
        self.name = name
        self.colorIndex = colorIndex
    }

    public static let you = SpeakerInfo(id: youID, name: "You", colorIndex: 0)
    public static let others = SpeakerInfo(id: othersID, name: "Others", colorIndex: 1)
}

/// One audio file of a recording, under `Recordings/<note-id>/`.
public struct RecordedTrack: Sendable, Hashable, Codable {
    public var fileName: String
    public var kind: TrackKind
    /// Where this file starts in recording time (a new file after wake).
    public var startOffset: TimeInterval
    public var duration: TimeInterval

    public init(fileName: String, kind: TrackKind, startOffset: TimeInterval, duration: TimeInterval) {
        self.fileName = fileName
        self.kind = kind
        self.startOffset = startOffset
        self.duration = duration
    }
}

/// A stretch with no capture that the transcript marks ("Mac asleep").
public struct RecordingGap: Sendable, Hashable, Codable {
    public enum Reason: String, Sendable, Hashable, Codable { case sleep }
    public var start: Date
    public var end: Date
    /// Recording time where the gap sits.
    public var at: TimeInterval
    public var reason: Reason

    public init(start: Date, end: Date, at: TimeInterval, reason: Reason) {
        self.start = start
        self.end = end
        self.at = at
        self.reason = reason
    }
}

/// A finished meeting or file: what goes into `NoteRecording`.
public struct TranscriptResult: Sendable, Hashable {
    public var segments: [TranscriptSegment]
    public var speakers: [SpeakerInfo]
    public var tracks: [RecordedTrack]
    public var gaps: [RecordingGap]
    public var duration: TimeInterval
    public var languageCode: String?
    /// The folder holding `tracks`; nil when no audio was kept.
    public var audioDirectory: URL?
    /// Live speaker id → final speaker id, from the final diarization pass,
    /// so renames made during the meeting carry over. Segment ids never
    /// change between live and final.
    public var speakerMapping: [String: String]

    public init(segments: [TranscriptSegment], speakers: [SpeakerInfo], tracks: [RecordedTrack] = [],
                gaps: [RecordingGap] = [], duration: TimeInterval, languageCode: String? = nil,
                audioDirectory: URL? = nil, speakerMapping: [String: String] = [:]) {
        self.segments = segments
        self.speakers = speakers
        self.tracks = tracks
        self.gaps = gaps
        self.duration = duration
        self.languageCode = languageCode
        self.audioDirectory = audioDirectory
        self.speakerMapping = speakerMapping
    }
}

/// A meeting the app quit in (crash, power loss), found at the next launch.
/// The speech service has already closed its audio files; the meetings layer
/// closes the note with what was saved.
public struct InterruptedRecording: Sendable, Hashable {
    public var noteID: UUID
    public var tracks: [RecordedTrack]
    /// When the last audio was written.
    public var endedAt: Date

    public init(noteID: UUID, tracks: [RecordedTrack], endedAt: Date) {
        self.noteID = noteID
        self.tracks = tracks
        self.endedAt = endedAt
    }
}

// MARK: Devices

public struct AudioInputDevice: Sendable, Hashable, Identifiable {
    public var id: String
    public var name: String
    public var isDefault: Bool

    public init(id: String, name: String, isDefault: Bool) {
        self.id = id
        self.name = name
        self.isDefault = isDefault
    }
}

/// Whose audio a meeting records besides the microphone.
public enum CallAudioSource: Sendable, Hashable, Codable {
    case off
    /// Every app's output: the safe default (spec 12).
    case allApps
    /// One running app, by bundle identifier, with its name for the UI.
    case app(bundleID: String, name: String)
}

/// A running app that can be chosen as the call-audio source.
public struct RunningAudioApp: Sendable, Hashable, Identifiable {
    public var id: String { bundleID }
    public var bundleID: String
    public var name: String

    public init(bundleID: String, name: String) {
        self.bundleID = bundleID
        self.name = name
    }
}
