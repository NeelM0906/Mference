import Foundation
@testable import MferenceAppCore
import Testing

/// Pause, sleep and stop arriving while the pipeline is suspended mid-step
/// (actor reentrancy). Codex review on #251.
@Suite struct MeetingPipelineRaceTests {
    /// A source whose `stop()` waits until the test opens it.
    final class GatedStopSource: AudioSource, @unchecked Sendable {
        let track: TrackKind
        private let lock = NSLock()
        private var continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation?
        private var gate: CheckedContinuation<Void, Never>?
        private var gated = true
        private var starts = 0

        init(track: TrackKind) { self.track = track }

        var startCount: Int { lock.withLock { starts } }
        var isStopping: Bool { lock.withLock { gate != nil } }

        func start() async throws -> AsyncThrowingStream<AudioChunk, Error> {
            let (stream, continuation) = AsyncThrowingStream.makeStream(of: AudioChunk.self)
            lock.withLock {
                starts += 1
                self.continuation = continuation
            }
            return stream
        }

        func stop() async {
            await withCheckedContinuation { (waiter: CheckedContinuation<Void, Never>) in
                let now = lock.withLock { () -> Bool in
                    guard gated else { return true }
                    gate = waiter
                    return false
                }
                if now { waiter.resume() }
            }
            lock.withLock { continuation }?.finish()
        }

        func open() {
            let waiter = lock.withLock { () -> CheckedContinuation<Void, Never>? in
                gated = false
                defer { gate = nil }
                return gate
            }
            waiter?.resume()
        }
    }

    struct Sources: AudioSourceFactory {
        let microphone: any AudioSource
        var callSource: (any AudioSource)?
        func microphone(deviceID: String?) throws -> any AudioSource { microphone }
        func call(_ source: CallAudioSource) throws -> any AudioSource {
            guard let callSource else { throw SpeechError.permissionDenied(.systemAudio) }
            return callSource
        }
        func inputDevices() async -> [AudioInputDevice] { [] }
        func runningAudioApps() async -> [RunningAudioApp] { [] }
        func isPlayingOutput(_ source: CallAudioSource) async -> Bool? { true }
    }

    /// Live speaker labels whose first feed waits until the test opens it.
    final class GatedDiarizer: SpeakerDiarizing, LiveDiarizing, @unchecked Sendable {
        private let lock = NSLock()
        private var gate: CheckedContinuation<Void, Never>?
        private var opened = false

        var isWaiting: Bool { lock.withLock { gate != nil } }

        func open() {
            let waiter = lock.withLock { () -> CheckedContinuation<Void, Never>? in
                opened = true
                defer { gate = nil }
                return gate
            }
            waiter?.resume()
        }

        func makeLiveSession() async throws -> any LiveDiarizing { self }

        func feed(_ samples: [Float], at time: TimeInterval) async throws -> [SpeakerTurn] {
            await withCheckedContinuation { (waiter: CheckedContinuation<Void, Never>) in
                let now = lock.withLock { () -> Bool in
                    if opened { return true }
                    gate = waiter
                    return false
                }
                if now { waiter.resume() }
            }
            return []
        }

        func diarize(_ files: [(url: URL, startOffset: TimeInterval)],
                     progress: @escaping @Sendable (Double) -> Void) async throws -> [SpeakerTurn] { [] }
    }

    static func pipeline(_ sources: Sources, callAudio: CallAudioSource, diarizer: (any SpeakerDiarizing)?,
                         root: URL) throws -> (MeetingPipeline, SpeechServiceTests.EventLog<MeetingEvent>) {
        let note = UUID()
        let recording = try RecordingStore(root: root, availableBytes: { _ in 10 << 30 })
            .begin(noteID: note, startedAt: .now, keepAudio: false)
        let (events, continuation) = AsyncStream.makeStream(of: MeetingEvent.self)
        let pipeline = MeetingPipeline(
            options: MeetingOptions(noteID: note, callAudio: callAudio), startedAt: .now,
            stages: .init(recognizer: ScriptedTranscriber(), voiceActivity: ScriptedVoiceActivity(), diarizer: diarizer),
            lane: TranscriptionLane(), sources: sources, recording: recording, segmenter: .init(),
            continuation: continuation, onLiveStatus: { _ in }, onCallAudioFailure: { _ in }, onEnd: {})
        return (pipeline, SpeechServiceTests.EventLog(events))
    }

    static func root() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "mference-race-\(UUID().uuidString)")
    }

    @Test func stopDuringSleepIsNotUndoneByWake() async throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let mic = GatedStopSource(track: .microphone)
        let (pipeline, log) = try Self.pipeline(Sources(microphone: mic), callAudio: .off, diarizer: nil, root: root)
        try await pipeline.start()
        let sleeping = Task { await pipeline.willSleep() }
        for _ in 0..<1_000 where !mic.isStopping { await Task.yield() }
        #expect(mic.isStopping)
        let stopping = Task { await pipeline.stop() }
        try await Task.sleep(for: .milliseconds(100))
        #expect(!(await log.isFinished), "the meeting does not end before its capture has stopped")
        mic.open()
        await sleeping.value
        await stopping.value
        #expect(await log.waitUntilFinished())
        await pipeline.didWake()
        #expect(mic.startCount == 1, "wake does not restart a meeting that ended")
    }

    @Test func pauseWaitsForCallAudioStillWithTheLiveDiarizer() async throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let mic = FakeAudioSource(track: .microphone, chunks: SpeechServiceTests.chunks([(1, 0)]))
        // One second of speech: the last chunk fills the live diarizer's first feed.
        let call = FakeAudioSource(track: .call, chunks: SpeechServiceTests.chunks([(1, 0.4)]))
        let diarizer = GatedDiarizer()
        let (pipeline, log) = try Self.pipeline(Sources(microphone: mic, callSource: call), callAudio: .allApps,
                                                diarizer: diarizer, root: root)
        try await pipeline.start()
        for _ in 0..<1_000 where !diarizer.isWaiting { try await Task.sleep(for: .milliseconds(1)) }
        #expect(diarizer.isWaiting)
        let pausing = Task { await pipeline.pause() }
        for _ in 0..<50 { await Task.yield() }
        diarizer.open()
        await pausing.value
        await pipeline.stop()
        #expect(await log.waitUntilFinished())
        let events = await log.events
        let paused = try #require(events.firstIndex(of: .paused))
        #expect(!events[paused...].contains { event in
            if case .segment(let segment) = event { !segment.isFinal } else { false }
        }, "no phrase starts after Pause from audio recorded before it")
    }
}
