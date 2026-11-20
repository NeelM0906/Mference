import Foundation
import Synchronization

// Scripted stages for tests and previews: fixture PCM instead of a
// microphone or call tap, scripted speech probabilities instead of Silero,
// scripted text instead of Fluid Speech Pro, scripted turns instead of the
// diarizers. No model, device, TCC prompt or network is touched.

/// Plays fixture chunks as one capture source. With `paced` false (the
/// default) every chunk is yielded at once; `hold()` / `release()` and
/// `fail(with:)` let tests stop capture mid-stream.
public final class FakeAudioSource: AudioSource {
    public let track: TrackKind
    private struct State {
        var chunks: [AudioChunk]
        var continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation?
        var startCount = 0
        var stopped = false
        var holdsOpen: Bool
    }
    private let state: Mutex<State>
    private let startError: SpeechError?

    /// - Parameters:
    ///   - holdsOpen: after the chunks, the stream stays open until `stop()`
    ///     (a live microphone); false finishes it after the last chunk.
    public init(track: TrackKind, chunks: [AudioChunk], holdsOpen: Bool = true, startError: SpeechError? = nil) {
        self.track = track
        state = Mutex(State(chunks: chunks, holdsOpen: holdsOpen))
        self.startError = startError
    }

    public var startCount: Int { state.withLock { $0.startCount } }
    public var isStopped: Bool { state.withLock { $0.stopped } }

    public func start() async throws -> AsyncThrowingStream<AudioChunk, Error> {
        if let startError { throw startError }
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AudioChunk.self)
        let (chunks, holdsOpen) = state.withLock { s in
            s.startCount += 1
            s.stopped = false
            s.continuation = continuation
            return (s.chunks, s.holdsOpen)
        }
        for chunk in chunks { continuation.yield(chunk) }
        if !holdsOpen { continuation.finish() }
        return stream
    }

    /// Yields more audio on the running stream.
    public func push(_ chunk: AudioChunk) {
        state.withLock { $0.continuation }?.yield(chunk)
    }

    /// Capture fails (a revoked permission, a device that went away).
    public func fail(with error: Error) {
        state.withLock { s in
            let continuation = s.continuation
            s.continuation = nil
            return continuation
        }?.finish(throwing: error)
    }

    public func stop() async {
        state.withLock { s in
            s.stopped = true
            let continuation = s.continuation
            s.continuation = nil
            return continuation
        }?.finish()
    }

    /// Fixture PCM: `seconds` of silence (0) or a 220 Hz tone at `amplitude`.
    public static func tone(seconds: TimeInterval, amplitude: Float = 0.3) -> [Float] {
        let count = SpeechAudio.samples(seconds)
        guard amplitude > 0 else { return [Float](repeating: 0, count: count) }
        return (0..<count).map { amplitude * sin(2 * .pi * 220 * Float($0) / Float(SpeechAudio.sampleRate)) }
    }
}

/// Builds sources from fakes, one per track, reused across restarts.
public final class FakeAudioSourceFactory: AudioSourceFactory {
    public let microphoneSource: FakeAudioSource
    public let callSource: FakeAudioSource?
    /// What `call(_:)` throws when `callSource` is nil.
    public let callError: SpeechError
    /// What `isPlayingOutput(_:)` answers: other apps playing, nil unreadable.
    public let playingOutput: Bool?

    public init(microphone: FakeAudioSource, call: FakeAudioSource? = nil,
                callError: SpeechError = .permissionDenied(.systemAudio), playingOutput: Bool? = true) {
        microphoneSource = microphone
        callSource = call
        self.callError = callError
        self.playingOutput = playingOutput
    }

    public func microphone(deviceID: String?) throws -> any AudioSource { microphoneSource }

    public func call(_ source: CallAudioSource) throws -> any AudioSource {
        guard let callSource else { throw callError }
        return callSource
    }

    public func inputDevices() async -> [AudioInputDevice] {
        [AudioInputDevice(id: "fake-mic", name: "Fixture Microphone", isDefault: true)]
    }

    public func runningAudioApps() async -> [RunningAudioApp] { [] }

    public func isPlayingOutput(_ source: CallAudioSource) async -> Bool? { playingOutput }
}

/// Speech probability from the audio itself: a frame whose peak is above
/// `level` is speech (1), anything else silence (0). Fixture tones make
/// "speech", fixture zeros make "silence", so tests script phrase
/// boundaries by building the PCM.
public struct ScriptedVoiceActivity: VoiceActivityDetecting {
    public let frameLength: Int
    public let level: Float

    public init(frameLength: Int = 1_600, level: Float = 0.05) {
        self.frameLength = frameLength
        self.level = level
    }

    public func makeStream() async throws -> any VoiceActivityStream { Stream(level: level) }

    final class Stream: VoiceActivityStream {
        let level: Float
        init(level: Float) { self.level = level }
        func probability(of frame: [Float]) async throws -> Float {
            frame.contains { abs($0) > level } ? 1 : 0
        }
    }
}

/// Returns scripted phrases in order (then `fallback`), recording how many
/// samples each call got. An optional per-call delay models recognition time.
public final class ScriptedTranscriber: SpeechRecognizing {
    private struct State {
        var phrases: [RecognizedPhrase]
        var received: [Int] = []
    }
    private let state: Mutex<State>
    private let fallback: RecognizedPhrase
    private let delay: Duration?
    private let failure: SpeechError?

    public init(_ phrases: [RecognizedPhrase] = [], fallback: RecognizedPhrase = RecognizedPhrase(text: "Hello."),
                delay: Duration? = nil, failure: SpeechError? = nil) {
        state = Mutex(State(phrases: phrases))
        self.fallback = fallback
        self.delay = delay
        self.failure = failure
    }

    public static func saying(_ texts: String...) -> ScriptedTranscriber {
        ScriptedTranscriber(texts.map { RecognizedPhrase(text: $0) })
    }

    /// Sample counts of every phrase transcribed so far.
    public var received: [Int] { state.withLock { $0.received } }

    public func transcribe(_ samples: [Float]) async throws -> RecognizedPhrase {
        if let delay { try await Task.sleep(for: delay) }
        if let failure { throw failure }
        return state.withLock { s in
            s.received.append(samples.count)
            return s.phrases.isEmpty ? fallback : s.phrases.removeFirst()
        }
    }
}

/// Live turns by recording time and the final pass's turns, scripted.
public final class ScriptedDiarizer: SpeakerDiarizing {
    public let liveTurns: [SpeakerTurn]
    public let finalTurns: [SpeakerTurn]
    private let calls = Mutex<[[URL]]>([])

    public init(live: [SpeakerTurn], final: [SpeakerTurn]) {
        liveTurns = live
        finalTurns = final
    }

    /// The files each final pass read.
    public var finalPassFiles: [[URL]] { calls.withLock { $0 } }

    public func makeLiveSession() async throws -> any LiveDiarizing { Live(turns: liveTurns) }

    public func diarize(_ files: [(url: URL, startOffset: TimeInterval)],
                        progress: @escaping @Sendable (Double) -> Void) async throws -> [SpeakerTurn] {
        calls.withLock { $0.append(files.map(\.url)) }
        progress(0.5)
        progress(1)
        return finalTurns
    }

    final class Live: LiveDiarizing {
        let turns: [SpeakerTurn]
        private let heard = Mutex<TimeInterval>(0)
        init(turns: [SpeakerTurn]) { self.turns = turns }

        /// The turns that started before the end of the audio heard so far.
        func feed(_ samples: [Float], at time: TimeInterval) async throws -> [SpeakerTurn] {
            let end = heard.withLock { heard in
                heard = max(heard, time + SpeechAudio.seconds(samples.count))
                return heard
            }
            return turns.filter { $0.start < end }
        }
    }
}

/// Loads the scripted stages instead of models: `loadSpeech` and
/// `loadSpeakerLabels` count calls and can be made to fail.
public final class ScriptedModelLoader: SpeechModelLoading {
    public let recognizer: any SpeechRecognizing
    public let voiceActivity: any VoiceActivityDetecting
    public let diarizer: any SpeakerDiarizing
    private struct Counts { var speech = 0, speakerLabels = 0 }
    private let counts = Mutex(Counts())
    private let failure = Mutex<SpeechError?>(nil)

    public init(recognizer: any SpeechRecognizing = ScriptedTranscriber(),
                voiceActivity: any VoiceActivityDetecting = ScriptedVoiceActivity(),
                diarizer: any SpeakerDiarizing = ScriptedDiarizer(live: [], final: [])) {
        self.recognizer = recognizer
        self.voiceActivity = voiceActivity
        self.diarizer = diarizer
    }

    public var speechLoads: Int { counts.withLock { $0.speech } }
    public var speakerLabelLoads: Int { counts.withLock { $0.speakerLabels } }
    /// The next load fails (once).
    public func failNextLoad(with error: SpeechError) { failure.withLock { $0 = error } }

    public func loadSpeech(from directory: URL) async throws -> (recognizer: any SpeechRecognizing,
                                                                 voiceActivity: any VoiceActivityDetecting) {
        if let error = failure.withLock({ f -> SpeechError? in defer { f = nil }; return f }) { throw error }
        counts.withLock { $0.speech += 1 }
        return (recognizer, voiceActivity)
    }

    public func loadSpeakerLabels(from directory: URL) async throws -> any SpeakerDiarizing {
        if let error = failure.withLock({ f -> SpeechError? in defer { f = nil }; return f }) { throw error }
        counts.withLock { $0.speakerLabels += 1 }
        return diarizer
    }
}
