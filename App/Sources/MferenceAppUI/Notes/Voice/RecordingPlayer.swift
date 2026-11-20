import AVFoundation
import Foundation
import MferenceAppCore
import Observation

/// Plays a recording's tracks together (playback only; capture stays in
/// `MferenceAppCore/Speech`). The app uses `CompositionPlayback`; tests use
/// a double.
@MainActor
public protocol AudioPlayback: AnyObject {
    /// Loads the tracks, each at its start offset. Returns the length.
    func load(tracks: [RecordedTrack], directory: URL) async throws -> TimeInterval
    func play()
    func pause()
    func seek(to time: TimeInterval)
    var rate: Float { get set }
    var currentTime: TimeInterval { get }
    /// Called about ten times a second while playing, and at the end.
    var onTick: (@MainActor () -> Void)? { get set }
}

/// AVPlayer over one composition of every track (microphone and call files,
/// a new file after each wake), so all of them play in step.
@MainActor
public final class CompositionPlayback: AudioPlayback {
    private let player = AVPlayer()
    private var observer: Any?
    private var desiredRate: Float = 1
    public var onTick: (@MainActor () -> Void)?

    public init() {}

    public func load(tracks: [RecordedTrack], directory: URL) async throws -> TimeInterval {
        let composition = AVMutableComposition()
        for track in tracks {
            let asset = AVURLAsset(url: directory.appending(path: track.fileName))
            guard let source = try await asset.loadTracks(withMediaType: .audio).first else { continue }
            let range = try await source.load(.timeRange)
            let target = composition.addMutableTrack(withMediaType: .audio,
                                                     preferredTrackID: kCMPersistentTrackID_Invalid)
            try target?.insertTimeRange(range, of: source,
                                        at: CMTime(seconds: track.startOffset, preferredTimescale: 600))
        }
        player.replaceCurrentItem(with: AVPlayerItem(asset: composition))
        if observer == nil {
            observer = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: 10),
                                                      queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.onTick?() }
            }
        }
        return composition.duration.seconds.isFinite ? composition.duration.seconds : 0
    }

    public func play() { player.playImmediately(atRate: desiredRate) }
    public func pause() { player.pause() }

    public func seek(to time: TimeInterval) {
        player.seek(to: CMTime(seconds: max(0, time), preferredTimescale: 600),
                    toleranceBefore: .zero, toleranceAfter: .zero)
    }

    public var rate: Float {
        get { desiredRate }
        set {
            desiredRate = newValue
            if player.rate != 0 { player.rate = newValue }
        }
    }

    public var currentTime: TimeInterval {
        let seconds = player.currentTime().seconds
        return seconds.isFinite ? seconds : 0
    }
}

/// The player of a meeting or file note (board 5): play and pause, the
/// time, a scrubber coloured by speaker, the length and the speed. Clicking
/// a transcript time plays from there.
@MainActor
@Observable
public final class RecordingPlayerModel {
    public static let speeds: [Float] = [1, 1.5, 2]

    public private(set) var isPlaying = false
    public private(set) var currentTime: TimeInterval = 0
    public private(set) var duration: TimeInterval
    public private(set) var speed: Float = 1
    /// Nil until loaded; false when the audio was deleted after transcribing.
    public private(set) var hasAudio: Bool?
    public private(set) var loadError: String?

    @ObservationIgnored private let playback: any AudioPlayback
    @ObservationIgnored private let tracks: [RecordedTrack]
    @ObservationIgnored private let directory: URL?

    /// - Parameter directory: where the tracks live; nil when no audio was kept.
    public init(tracks: [RecordedTrack], directory: URL?, duration: TimeInterval,
                playback: any AudioPlayback = CompositionPlayback()) {
        self.tracks = tracks
        self.directory = directory
        self.duration = duration
        self.playback = playback
        playback.onTick = { [weak self] in self?.tick() }
    }

    /// Loads the audio (once).
    public func load() async {
        guard hasAudio == nil else { return }
        guard let directory, !tracks.isEmpty else {
            hasAudio = false
            return
        }
        do {
            let length = try await playback.load(tracks: tracks, directory: directory)
            if length > 0 { duration = max(duration, length) }
            hasAudio = true
        } catch {
            hasAudio = false
            loadError = "The recording could not be opened."
        }
    }

    public func togglePlay() {
        guard hasAudio == true else { return }
        if isPlaying {
            playback.pause()
            isPlaying = false
        } else {
            if currentTime >= duration { seek(to: 0) }
            playback.play()
            isPlaying = true
        }
    }

    /// A transcript time was clicked: play from there.
    public func play(from time: TimeInterval) {
        guard hasAudio == true else { return }
        seek(to: time)
        if !isPlaying {
            playback.play()
            isPlaying = true
        }
    }

    public func seek(to time: TimeInterval) {
        let clamped = min(max(0, time), duration)
        playback.seek(to: clamped)
        currentTime = clamped
    }

    /// 1× → 1.5× → 2× → 1×.
    public func nextSpeed() {
        let index = Self.speeds.firstIndex(of: speed) ?? 0
        speed = Self.speeds[(index + 1) % Self.speeds.count]
        playback.rate = speed
    }

    public static func speedLabel(_ speed: Float) -> String {
        speed == 1.5 ? "1.5×" : "\(Int(speed))×"
    }

    /// The transcript row being played (highlighted).
    public func playingSegment(in segments: [TranscriptSegment]) -> TranscriptSegment.ID? {
        guard isPlaying || currentTime > 0 else { return nil }
        return segments.last { $0.start <= currentTime + 0.05 }?.id
    }

    private func tick() {
        currentTime = playback.currentTime
        if currentTime >= duration - 0.05, isPlaying {
            playback.pause()
            isPlaying = false
            currentTime = duration
        }
    }
}
