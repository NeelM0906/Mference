import AVFoundation
import Foundation
import MferenceAppCore
import Observation

/// Plays a recording's tracks together (playback only; capture stays in
/// `MferenceAppCore/Speech`). The app uses `CompositionPlayback`; tests use
/// a double.
@MainActor
public protocol AudioPlayback: AnyObject {
    /// Loads the tracks that exist, each at its start offset, and returns
    /// the length. Throws `PlaybackError.noAudio` when none of the files is
    /// there (deleted after transcribing).
    func load(tracks: [RecordedTrack], directory: URL) async throws -> TimeInterval
    func play()
    func pause()
    func seek(to time: TimeInterval)
    var rate: Float { get set }
    var currentTime: TimeInterval { get }
    /// Called about ten times a second while playing.
    var onTick: (@MainActor () -> Void)? { get set }
    /// Called when playback reaches the end.
    var onEnded: (@MainActor () -> Void)? { get set }
}

public enum PlaybackError: Error, Equatable {
    /// None of the recording's audio files is on disk.
    case noAudio
}

/// AVPlayer over one composition of every track (microphone and call files,
/// a new file after each wake), so all of them play in step.
@MainActor
public final class CompositionPlayback: AudioPlayback {
    private let player = AVPlayer()
    private var observer: Any?
    private var endObserver: (any NSObjectProtocol)?
    private var desiredRate: Float = 1
    public var onTick: (@MainActor () -> Void)?
    public var onEnded: (@MainActor () -> Void)?

    public init() {}

    isolated deinit {
        if let observer { player.removeTimeObserver(observer) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
    }

    /// The loaded composition's length (tests).
    private(set) var itemDuration: TimeInterval = 0

    /// Silent playback for tests: nothing reaches the speakers.
    var isMuted: Bool {
        get { player.isMuted }
        set { player.isMuted = newValue }
    }

    /// Waits until the item can seek and play (tests).
    func waitUntilReady(timeout: Duration = .seconds(5)) async -> Bool {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if player.currentItem?.status == .readyToPlay { return true }
            try? await Task.sleep(for: .milliseconds(20))
        }
        return false
    }

    public func load(tracks: [RecordedTrack], directory: URL) async throws -> TimeInterval {
        let composition = AVMutableComposition()
        var loaded = 0
        for track in tracks {
            // A missing file (one track deleted, a crash before a file was
            // written) leaves the rest playable.
            guard let url = Self.audioFile(for: track, in: directory) else { continue }
            let asset = AVURLAsset(url: url)
            guard let source = try? await asset.loadTracks(withMediaType: .audio).first else { continue }
            let range = try await source.load(.timeRange)
            let target = composition.addMutableTrack(withMediaType: .audio,
                                                     preferredTrackID: kCMPersistentTrackID_Invalid)
            try target?.insertTimeRange(range, of: source,
                                        at: CMTime(seconds: track.startOffset, preferredTimescale: 600))
            loaded += 1
        }
        guard loaded > 0 else { throw PlaybackError.noAudio }
        let item = AVPlayerItem(asset: composition)
        player.replaceCurrentItem(with: item)
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        endObserver = NotificationCenter.default.addObserver(forName: AVPlayerItem.didPlayToEndTimeNotification,
                                                             object: item, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.onEnded?() }
        }
        if observer == nil {
            observer = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: 10),
                                                      queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.onTick?() }
            }
        }
        itemDuration = composition.duration.seconds.isFinite ? composition.duration.seconds : 0
        return itemDuration
    }

    /// The track's file, or the CAF it was to be converted from: a meeting
    /// ended at quit names its tracks `.m4a` before the conversion at the
    /// next launch, and a failed conversion keeps the CAF.
    nonisolated static func audioFile(for track: RecordedTrack, in directory: URL) -> URL? {
        let url = directory.appending(path: track.fileName)
        if FileManager.default.fileExists(atPath: url.path) { return url }
        guard url.pathExtension == "m4a" else { return nil }
        let caf = url.deletingPathExtension().appendingPathExtension("caf")
        return FileManager.default.fileExists(atPath: caf.path) ? caf : nil
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
    /// Set when the files exist but could not be opened.
    public private(set) var loadError: String?

    @ObservationIgnored private let playback: any AudioPlayback
    @ObservationIgnored private let tracks: [RecordedTrack]
    @ObservationIgnored private let directory: URL?
    /// Ticks right after a seek may still carry the old position: until
    /// this time, only ticks near the seek target count.
    @ObservationIgnored private var ignoreTicksUntil: Date?
    @ObservationIgnored private var seekTarget: TimeInterval = 0

    /// - Parameter directory: where the tracks live; nil when no audio was kept.
    public init(tracks: [RecordedTrack], directory: URL?, duration: TimeInterval,
                playback: any AudioPlayback = CompositionPlayback()) {
        self.tracks = tracks
        self.directory = directory
        self.duration = duration
        self.playback = playback
        playback.onTick = { [weak self] in self?.tick() }
        playback.onEnded = { [weak self] in self?.reachedEnd() }
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
            // The audio's own length: a note can outlast its audio.
            if length > 0 { duration = length }
            hasAudio = true
        } catch PlaybackError.noAudio {
            hasAudio = false
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
        seekTarget = clamped
        ignoreTicksUntil = Date.now.addingTimeInterval(0.3)
    }

    /// Leaving the note: playback stops.
    public func pause() {
        guard isPlaying else { return }
        playback.pause()
        isPlaying = false
    }

    /// The keyboard's arrows on the player: back or ahead by `seconds`.
    public func skip(_ seconds: TimeInterval) {
        guard hasAudio == true else { return }
        seek(to: currentTime + seconds)
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
        if let ignoreTicksUntil, Date.now < ignoreTicksUntil,
           abs(playback.currentTime - seekTarget) > 1 { return }
        ignoreTicksUntil = nil
        currentTime = playback.currentTime
        if currentTime >= duration - 0.05, isPlaying { reachedEnd() }
    }

    /// The end: the button reads Play again, and Play starts over.
    private func reachedEnd() {
        playback.pause()
        isPlaying = false
        currentTime = duration
    }
}
