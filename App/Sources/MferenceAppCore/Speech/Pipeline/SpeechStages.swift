import Foundation

// The speech service's internal stages (Notes spec 6.3), each behind a small
// protocol so the pipelines run model-free in tests: capture (`AudioSource`),
// voice activity (`VoiceActivityDetecting`), recognition (`SpeechRecognizing`)
// and speaker labels (`SpeakerDiarizing`). The FluidAudio adapters and the
// AVAudioEngine / process-tap sources conform to them; `ScriptedSpeechStages`
// holds the fakes. Only `MferenceAppCore/Speech` uses these.

/// The models' audio format: 16 kHz mono Float32 (spec 3, 6.3).
public enum SpeechAudio {
    public static let sampleRate: Double = 16_000

    public static func seconds(_ samples: Int) -> TimeInterval { TimeInterval(samples) / sampleRate }
    public static func samples(_ seconds: TimeInterval) -> Int { Int((seconds * sampleRate).rounded()) }

    /// Input level 0...1 from a chunk's RMS, on a -50...0 dBFS scale (the
    /// meters' range).
    public static func level(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for sample in samples { sum += sample * sample }
        let rms = (sum / Float(samples.count)).squareRoot()
        guard rms > 0 else { return 0 }
        let decibels = 20 * log10(rms)
        return min(1, max(0, (decibels + 50) / 50))
    }
}

/// A run of 16 kHz mono samples from one capture source.
public struct AudioChunk: Sendable, Hashable {
    public var samples: [Float]

    public init(samples: [Float]) { self.samples = samples }

    public var duration: TimeInterval { SpeechAudio.seconds(samples.count) }
}

// MARK: Capture

/// One capture source: the microphone or the call tap. Both convert to
/// 16 kHz mono before yielding.
public protocol AudioSource: AnyObject, Sendable {
    var track: TrackKind { get }
    /// Starts capture. The stream finishes when `stop()` is called, and
    /// finishes with an error when capture fails (a revoked permission, a
    /// device that went away).
    func start() async throws -> AsyncThrowingStream<AudioChunk, Error>
    /// Stops capture; the stream finishes. Calling it again is a no-op.
    func stop() async
}

/// Makes the sources a session needs, so a meeting can restart them after
/// sleep (spec 6.3: new files on wake).
public protocol AudioSourceFactory: Sendable {
    func microphone(deviceID: String?) throws -> any AudioSource
    /// Throws `SpeechError.permissionDenied(.systemAudio)` when the tap cannot
    /// be created.
    func call(_ source: CallAudioSource) throws -> any AudioSource
    func inputDevices() async -> [AudioInputDevice]
    func runningAudioApps() async -> [RunningAudioApp]
}

// MARK: Voice activity

/// Silero voice activity: a speech probability per fixed-size frame.
public protocol VoiceActivityDetecting: Sendable {
    /// Samples per frame (4,096, 256 ms, for the unified Silero v6.2.1 model).
    var frameLength: Int { get }
    /// A fresh detector state for one stream; each track has its own.
    func makeStream() async throws -> any VoiceActivityStream
}

/// One stream's detector state.
public protocol VoiceActivityStream: AnyObject, Sendable {
    /// The probability, 0...1, that `frame` (exactly `frameLength` samples) holds speech.
    func probability(of frame: [Float]) async throws -> Float
}

// MARK: Recognition

/// What the recogniser heard in one phrase.
public struct RecognizedPhrase: Sendable, Hashable {
    public var text: String
    /// Detected or pinned language, when known (BCP 47).
    public var languageCode: String?

    public init(text: String, languageCode: String? = nil) {
        self.text = text
        self.languageCode = languageCode
    }
}

/// Fluid Speech Pro through FluidAudio: one phrase (at most 15 s) in, text out.
public protocol SpeechRecognizing: Sendable {
    func transcribe(_ samples: [Float]) async throws -> RecognizedPhrase
}

// MARK: Speaker labels

/// A stretch of one speaker's voice, in recording time.
public struct SpeakerTurn: Sendable, Hashable, Codable {
    public var speaker: String
    public var start: TimeInterval
    public var end: TimeInterval

    public init(speaker: String, start: TimeInterval, end: TimeInterval) {
        self.speaker = speaker
        self.start = start
        self.end = end
    }
}

/// The live diarizer of one meeting's call track (LS-EEND).
public protocol LiveDiarizing: AnyObject, Sendable {
    /// Feeds call audio that starts at `time` (recording time) and returns
    /// every turn known so far.
    func feed(_ samples: [Float], at time: TimeInterval) async throws -> [SpeakerTurn]
}

/// Speaker labels: live during a meeting, then the offline pass (spec 6.3).
public protocol SpeakerDiarizing: Sendable {
    func makeLiveSession() async throws -> any LiveDiarizing
    /// The final pass over a recording's call-track files, each placed at
    /// its start offset in recording time. Reads the files in bounded
    /// chunks; `progress` is 0...1.
    func diarize(_ files: [(url: URL, startOffset: TimeInterval)],
                 progress: @escaping @Sendable (Double) -> Void) async throws -> [SpeakerTurn]
}

// MARK: Loading

/// Loads the installed packages' models from their verified folders (never
/// from the network) and releases them when idle.
public protocol SpeechModelLoading: Sendable {
    /// The first load runs the Neural Engine compile (seconds); later loads
    /// are fast. `directory` is the package's install root.
    func loadSpeech(from directory: URL) async throws -> (recognizer: any SpeechRecognizing,
                                                          voiceActivity: any VoiceActivityDetecting)
    func loadSpeakerLabels(from directory: URL) async throws -> any SpeakerDiarizing
}
