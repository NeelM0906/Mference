import Foundation

/// One phrase cut from a track by voice activity: its samples and its place
/// in recording time (spec 6.3: segment times come from our own boundaries).
public struct Phrase: Sendable, Hashable {
    public var id: UUID
    public var track: TrackKind
    public var start: TimeInterval
    public var end: TimeInterval
    public var samples: [Float]

    public init(id: UUID = UUID(), track: TrackKind, start: TimeInterval, end: TimeInterval, samples: [Float]) {
        self.id = id
        self.track = track
        self.start = start
        self.end = end
        self.samples = samples
    }
}

/// What the segmenter saw in a frame.
public enum SegmenterEvent: Sendable, Hashable {
    /// Speech began at `start` (recording time); `id` names the phrase.
    case started(id: UUID, start: TimeInterval)
    /// The phrase ended: after enough silence, at the 15 s cap, or at a flush.
    case finished(Phrase)
    /// Too short to be speech (a click or a cough): forget its start.
    case dropped(id: UUID)
}

/// Splits one track into phrases from per-frame speech probabilities
/// (Silero-style hysteresis): a phrase starts when a frame reaches
/// `threshold`, ends once `endOfSpeechSilence` of frames stay below
/// `negativeThreshold` (spec D2: 500 ms), and is cut at `maxPhrase` (15 s,
/// the model's window) while speech goes on, the next phrase starting at the
/// cut. A pure value: tests feed it scripted probabilities.
public struct Segmenter: Sendable {
    public struct Configuration: Sendable, Hashable {
        public var threshold: Float
        public var negativeThreshold: Float
        public var endOfSpeechSilence: TimeInterval
        public var maxPhrase: TimeInterval
        /// Phrases with less voiced time are dropped.
        public var minSpeech: TimeInterval
        /// Kept before the first voiced frame and after the last one.
        public var padding: TimeInterval

        /// FluidAudio's Silero defaults (threshold 0.85, offset 0.15), with
        /// the spec's 500 ms silence and 15 s cap.
        public init(threshold: Float = 0.85, negativeThreshold: Float = 0.7, endOfSpeechSilence: TimeInterval = 0.5,
                    maxPhrase: TimeInterval = 15, minSpeech: TimeInterval = 0.2, padding: TimeInterval = 0.1) {
            self.threshold = threshold
            self.negativeThreshold = negativeThreshold
            self.endOfSpeechSilence = endOfSpeechSilence
            self.maxPhrase = maxPhrase
            self.minSpeech = minSpeech
            self.padding = padding
        }
    }

    public let track: TrackKind
    public var configuration: Configuration
    /// Recording time of the next sample fed.
    public private(set) var position: TimeInterval

    private var previousFrame: [Float] = []
    private var current: Open?

    private struct Open {
        var id: UUID
        var start: TimeInterval
        var samples: [Float]
        /// Samples of trailing frames below the negative threshold.
        var trailingSilence: Int
        var voiced: Int
    }

    public init(track: TrackKind, configuration: Configuration = Configuration(), startingAt position: TimeInterval = 0) {
        self.track = track
        self.configuration = configuration
        self.position = position
    }

    public var isInSpeech: Bool { current != nil }

    /// Moves the clock without audio (a new file after wake starts at its
    /// own offset). Ends any open phrase first.
    public mutating func jump(to time: TimeInterval) -> [SegmenterEvent] {
        let events = flush()
        position = time
        previousFrame = []
        return events
    }

    /// Feeds one frame and its speech probability.
    public mutating func process(frame: [Float], probability: Float) -> [SegmenterEvent] {
        let frameStart = position
        position += SpeechAudio.seconds(frame.count)
        defer { previousFrame = frame }
        var events: [SegmenterEvent] = []

        guard var open = current else {
            guard probability >= configuration.threshold else { return [] }
            // Speech starts: keep `padding` of the frame before.
            let pad = min(previousFrame.count, SpeechAudio.samples(configuration.padding))
            let id = UUID()
            current = Open(id: id, start: frameStart - SpeechAudio.seconds(pad),
                           samples: Array(previousFrame.suffix(pad)) + frame, trailingSilence: 0,
                           voiced: frame.count)
            return [.started(id: id, start: frameStart - SpeechAudio.seconds(pad))]
        }

        // The 15 s cap (the model's window): cut before the frame that would
        // cross it, so no phrase is ever longer; if the voice goes on, the
        // next phrase starts with this frame.
        if open.samples.count + frame.count > SpeechAudio.samples(configuration.maxPhrase) {
            let stillSpeaking = probability >= configuration.negativeThreshold
            events += finish(keepingTrailingSilence: true)
            if stillSpeaking {
                let id = UUID()
                current = Open(id: id, start: frameStart, samples: frame, trailingSilence: 0, voiced: frame.count)
                events.append(.started(id: id, start: frameStart))
            }
            return events
        }

        open.samples += frame
        if probability < configuration.negativeThreshold {
            open.trailingSilence += frame.count
        } else {
            open.trailingSilence = 0
            open.voiced += frame.count
        }
        current = open

        if SpeechAudio.seconds(open.trailingSilence) >= configuration.endOfSpeechSilence {
            events += finish(keepingTrailingSilence: false)
        }
        return events
    }

    /// Ends the open phrase (stop, sleep, end of file).
    public mutating func flush() -> [SegmenterEvent] {
        guard current != nil else { return [] }
        return finish(keepingTrailingSilence: false)
    }

    private mutating func finish(keepingTrailingSilence: Bool) -> [SegmenterEvent] {
        guard let open = current else { return [] }
        current = nil
        guard SpeechAudio.seconds(open.voiced) >= configuration.minSpeech else { return [.dropped(id: open.id)] }
        var samples = open.samples
        if !keepingTrailingSilence {
            let trim = max(0, open.trailingSilence - SpeechAudio.samples(configuration.padding))
            samples.removeLast(min(trim, samples.count))
        }
        let phrase = Phrase(id: open.id, track: track, start: open.start,
                            end: open.start + SpeechAudio.seconds(samples.count), samples: samples)
        return [.finished(phrase)]
    }
}

/// A segmenter fed with capture chunks of any size: it frames them for the
/// detector, asks for each frame's probability and returns the events.
actor TrackSegmenter {
    private var segmenter: Segmenter
    private let detector: any VoiceActivityStream
    private let frameLength: Int
    private var pending: [Float] = []

    init(track: TrackKind, detector: any VoiceActivityStream, frameLength: Int,
         configuration: Segmenter.Configuration, startingAt position: TimeInterval = 0) {
        segmenter = Segmenter(track: track, configuration: configuration, startingAt: position)
        self.detector = detector
        self.frameLength = frameLength
    }

    /// Recording time of the next sample, counting samples still waiting for
    /// a full frame.
    var position: TimeInterval { segmenter.position + SpeechAudio.seconds(pending.count) }

    func feed(_ samples: [Float]) async throws -> [SegmenterEvent] {
        pending += samples
        var events: [SegmenterEvent] = []
        while pending.count >= frameLength {
            let frame = Array(pending.prefix(frameLength))
            pending.removeFirst(frameLength)
            let probability = try await detector.probability(of: frame)
            events += segmenter.process(frame: frame, probability: probability)
        }
        return events
    }

    /// Ends the open phrase with what is buffered (stop, sleep, end of file).
    func flush() -> [SegmenterEvent] {
        var events: [SegmenterEvent] = []
        if !pending.isEmpty {
            // A partial frame inside a phrase counts as speech so the last
            // words are kept; outside one it is silence, still counted so
            // the clock keeps step with the recording (pause and resume).
            let frame = pending
            pending = []
            let probability: Float = segmenter.isInSpeech ? segmenter.configuration.threshold : 0
            events += segmenter.process(frame: frame, probability: probability)
        }
        return events + segmenter.flush()
    }

    /// Starts again at `time` (new files after wake); ends any open phrase.
    func jump(to time: TimeInterval) -> [SegmenterEvent] {
        let events = flush()
        return events + segmenter.jump(to: time)
    }
}
