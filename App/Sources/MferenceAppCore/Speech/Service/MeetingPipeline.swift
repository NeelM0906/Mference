import Foundation
import Synchronization

/// One meeting recording (spec 6.3). Two tracks, each with its own capture,
/// voice activity and file: the microphone ("You") and, when allowed, the
/// call (other apps' audio through the process tap). Phrases from both go
/// through the one recogniser. Call phrases get live speaker labels; on
/// stop the final diarization pass relabels them, and live ids map to the
/// final ones so renames carry over. Microphone phrases that repeat the
/// call (no headphones) are dropped. Sleep closes the files and wake opens
/// new ones, with a gap in the transcript.
actor MeetingPipeline {
    struct Stages: Sendable {
        var recognizer: any SpeechRecognizing
        var voiceActivity: any VoiceActivityDetecting
        /// nil without speaker labels (installed and asked for).
        var diarizer: (any SpeakerDiarizing)?
    }

    enum Phase { case created, running, asleep, stopping, ended }

    let options: MeetingOptions
    let startedAt: Date
    private let stages: Stages
    private let lane: TranscriptionLane
    private let sources: any AudioSourceFactory
    private let recording: RecordingSession
    private let segmenterConfiguration: Segmenter.Configuration
    private let continuation: AsyncStream<MeetingEvent>.Continuation
    private let onLiveStatus: @Sendable (LiveSessionStatus?) -> Void
    private let onCallAudioFailure: @Sendable (SpeechError) async -> Void
    private let callAudioAllowed: @Sendable () async -> Bool
    private let onEnd: @Sendable () async -> Void
    private let now: @Sendable () -> Date

    private var phase = Phase.created
    private var paused = false
    private var callAvailable = false
    private var active: [TrackKind: (source: any AudioSource, task: Task<Void, Never>?)] = [:]
    /// Streams started but not yet read: reading starts once `.started` or `.gap` is out.
    private var unread: [(TrackKind, AsyncThrowingStream<AudioChunk, Error>)] = []
    private var segmenters: [TrackKind: TrackSegmenter] = [:]
    /// Samples written per track (pauses excluded): the recording clock,
    /// counted in samples so it never drifts.
    private var trackSamples: [TrackKind: Int] = [:]
    private var lastTranscription: [TrackKind: Task<Void, Never>] = [:]
    private var finals: [UUID: TranscriptSegment] = [:]
    private var languages: [String?] = []
    private var gaps: [RecordingGap] = []
    private var sleptAt: Date?
    private var levels: (microphone: Float, call: Float?) = (0, nil)
    private var live: (any LiveDiarizing)?
    private var liveBuffer: [Float] = []
    private var liveBufferStart: TimeInterval = 0
    private var liveTurns: [SpeakerTurn] = []
    private var liveSpeakers: [SpeakerInfo] = []
    private var pausedAt: TimeInterval = 0
    private var resumedAt: Date
    private var stopTask: Task<Void, Never>?
    private var quitting = false
    private var quitWaiters: [any QuitWaiter] = []
    private var finishTask: Task<Void, Never>?
    /// Whether the call track has carried a non-zero sample (a refused tap
    /// delivers exact zeros), and whether the hint went out.
    private var callHeard = false
    private var silentCallHinted = false
    /// Call-track samples at which "are other apps playing" is next asked.
    private var nextSilentCallCheck = SpeechAudio.samples(MeetingPipeline.silentCallHintAfter)

    /// Live diarization is fed about once a second.
    static let liveFeedSamples = 16_000
    /// Recording time with only silence on the call track before the hint.
    static let silentCallHintAfter: TimeInterval = 30
    /// While nothing else plays, how often (recording time) to ask again.
    static let silentCallRecheck: TimeInterval = 5

    init(options: MeetingOptions, startedAt: Date, stages: Stages, lane: TranscriptionLane,
         sources: any AudioSourceFactory, recording: RecordingSession, segmenter: Segmenter.Configuration,
         continuation: AsyncStream<MeetingEvent>.Continuation,
         onLiveStatus: @escaping @Sendable (LiveSessionStatus?) -> Void,
         onCallAudioFailure: @escaping @Sendable (SpeechError) async -> Void,
         callAudioAllowed: @escaping @Sendable () async -> Bool = { true },
         onEnd: @escaping @Sendable () async -> Void,
         now: @escaping @Sendable () -> Date = { .now }) {
        self.options = options
        self.startedAt = startedAt
        self.stages = stages
        self.lane = lane
        self.sources = sources
        self.recording = recording
        segmenterConfiguration = segmenter
        self.continuation = continuation
        self.onLiveStatus = onLiveStatus
        self.onCallAudioFailure = onCallAudioFailure
        self.callAudioAllowed = callAudioAllowed
        self.onEnd = onEnd
        self.now = now
        resumedAt = startedAt
    }

    /// The recording time so far: the microphone's clock.
    var recordedTime: TimeInterval { SpeechAudio.seconds(trackSamples[.microphone] ?? 0) }

    // MARK: Start

    /// Starts the microphone (its failure is the meeting's) and, if asked,
    /// the call tap (its failure only drops call audio).
    func start() async throws {
        let microphone = try sources.microphone(deviceID: options.microphoneID)
        try await startTrack(.microphone, source: microphone, at: 0)
        var callFailure: SpeechError?
        var callRefused = false
        if options.callAudio != .off, !(await callAudioAllowed()) {
            // Not granted (never asked, or denied): no tap, so no system
            // prompt without its reason (L6).
            callFailure = .permissionDenied(.systemAudio)
            callRefused = true
        } else if options.callAudio != .off {
            do {
                let call = try sources.call(options.callAudio)
                try await startTrack(.call, source: call, at: 0)
                callAvailable = true
            } catch {
                callFailure = error as? SpeechError ?? .permissionDenied(.systemAudio)
            }
        }
        continuation.yield(.started(callAudio: callAvailable))
        if let callFailure {
            if !callRefused { await onCallAudioFailure(callFailure) }
            continuation.yield(.callAudioUnavailable(callFailure))
        }
        if callAvailable, let diarizer = stages.diarizer {
            live = try? await diarizer.makeLiveSession()
        }
        liveSpeakers = [.you] + (callAvailable && live == nil ? [.others] : [])
        continuation.yield(.speakersUpdated(liveSpeakers))
        phase = .running
        readStartedTracks()
        publishStatus()
    }

    private func startTrack(_ kind: TrackKind, source: any AudioSource, at offset: TimeInterval) async throws {
        if segmenters[kind] == nil {
            segmenters[kind] = TrackSegmenter(track: kind, detector: try await stages.voiceActivity.makeStream(),
                                       frameLength: stages.voiceActivity.frameLength,
                                       configuration: segmenterConfiguration, startingAt: offset)
        }
        try recording.openTrack(kind, at: offset)
        let stream: AsyncThrowingStream<AudioChunk, Error>
        do {
            stream = try await source.start()
        } catch {
            try? recording.closeTrack(kind)
            throw error
        }
        trackSamples[kind] = SpeechAudio.samples(offset)
        active[kind] = (source, nil)
        unread.append((kind, stream))
    }

    private func readStartedTracks() {
        for (kind, stream) in unread {
            let task = Task { await self.consume(kind, stream) }
            active[kind]?.task = task
        }
        unread = []
    }

    // MARK: Capture

    private func consume(_ kind: TrackKind, _ stream: AsyncThrowingStream<AudioChunk, Error>) async {
        do {
            for try await chunk in stream {
                try await handle(kind, chunk)
            }
        } catch {
            let reason = error as? SpeechError ?? .failed(String(describing: error))
            if kind == .call {
                // Call audio failed (a revoked permission): carry on with the mic.
                callAvailable = false
                await onCallAudioFailure(reason)
                continuation.yield(.callAudioUnavailable(reason))
            } else if phase == .running {
                continuation.yield(.failed(reason))
                Task { await self.stop() }
            }
        }
        if let segmenter = segmenters[kind] { await handle(kind, segmenter: await segmenter.flush()) }
    }

    private func handle(_ kind: TrackKind, _ chunk: AudioChunk) async throws {
        guard !paused, phase != .ended else { return }
        try recording.write(chunk.samples, to: kind, now: now())
        let start = SpeechAudio.seconds(trackSamples[kind] ?? 0)
        trackSamples[kind, default: 0] += chunk.samples.count
        let level = SpeechAudio.level(chunk.samples)
        if kind == .microphone {
            levels.microphone = level
            continuation.yield(.levels(microphone: level, call: callAvailable ? levels.call : nil))
        } else {
            levels.call = level
            await feedLiveDiarizer(chunk.samples, at: start)
            await watchForSilentCall(chunk.samples)
        }
        if let segmenter = segmenters[kind] { await handle(kind, segmenter: try await segmenter.feed(chunk.samples)) }
    }

    private func handle(_ kind: TrackKind, segmenter events: [SegmenterEvent]) async {
        for event in events {
            switch event {
            case .started(let id, let start):
                continuation.yield(.segment(TranscriptSegment(id: id, track: kind, start: start, end: start,
                                                              speakerID: speakerID(kind, start, start),
                                                              text: "", isFinal: false)))
            case .dropped(let id):
                continuation.yield(.segmentRemoved(id))
            case .finished(let phrase):
                let previous = lastTranscription[kind]
                lastTranscription[kind] = Task {
                    await previous?.value
                    await self.transcribe(phrase)
                }
            }
        }
    }

    private func transcribe(_ phrase: Phrase) async {
        let recognizer = stages.recognizer
        let samples = phrase.samples
        let heard: RecognizedPhrase
        do {
            heard = try await lane.run { try await recognizer.transcribe(samples) }
        } catch {
            continuation.yield(.segmentRemoved(phrase.id))
            return
        }
        let text = heard.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            continuation.yield(.segmentRemoved(phrase.id))
            return
        }
        let code = heard.languageCode ?? PhraseLanguage.code(for: text, setting: options.language)
        let segment = TranscriptSegment(id: phrase.id, track: phrase.track, start: phrase.start, end: phrase.end,
                                        speakerID: speakerID(phrase.track, phrase.start, phrase.end), text: text,
                                        isFinal: true, languageCode: code)
        languages.append(code)
        let calls = finals.values.filter { $0.track == .call }
        if segment.track == .microphone, EchoDeduplicator.isEcho(segment, of: calls) {
            continuation.yield(.segmentRemoved(segment.id))
            return
        }
        finals[segment.id] = segment
        continuation.yield(.segment(segment))
        if segment.track == .call {
            // A call phrase that finished after the mic heard it as an echo.
            let calls = finals.values.filter { $0.track == .call }
            for mic in finals.values where mic.track == .microphone && EchoDeduplicator.isEcho(mic, of: calls) {
                finals[mic.id] = nil
                continuation.yield(.segmentRemoved(mic.id))
            }
        }
    }

    /// Sends `.noCallAudioHeard` once per meeting: the tap has delivered
    /// only exact silence for `silentCallHintAfter` of recording while the
    /// tapped apps play sound. When Core Audio can't say whether they do,
    /// the silence alone is enough. While they don't, it asks again every
    /// `silentCallRecheck`. Nothing is stored: it is only a hint.
    private func watchForSilentCall(_ samples: [Float]) async {
        guard !callHeard, !silentCallHinted else { return }
        if samples.contains(where: { $0 != 0 }) {
            callHeard = true
            return
        }
        guard let recorded = trackSamples[.call], recorded >= nextSilentCallCheck else { return }
        nextSilentCallCheck = recorded + SpeechAudio.samples(Self.silentCallRecheck)
        let playing = await sources.isPlayingOutput(options.callAudio) ?? true
        guard playing, !callHeard, !silentCallHinted, callAvailable, phase == .running, !paused else { return }
        silentCallHinted = true
        continuation.yield(.noCallAudioHeard)
    }

    // MARK: Speakers

    private func speakerID(_ track: TrackKind, _ start: TimeInterval, _ end: TimeInterval) -> String {
        switch track {
        case .microphone: SpeakerInfo.youID
        case .call, .file:
            live == nil ? SpeakerInfo.othersID
                : SpeakerAlignment.speaker(from: start, to: max(end, start + 0.01), in: liveTurns)
                    ?? liveSpeakers.last(where: { $0.id != SpeakerInfo.youID })?.id ?? SpeakerInfo.othersID
        }
    }

    private func feedLiveDiarizer(_ samples: [Float], at time: TimeInterval) async {
        guard let live else { return }
        if liveBuffer.isEmpty { liveBufferStart = time }
        liveBuffer += samples
        guard liveBuffer.count >= Self.liveFeedSamples else { return }
        let batch = liveBuffer
        let batchStart = liveBufferStart
        liveBuffer = []
        guard let turns = try? await live.feed(batch, at: batchStart) else { return }
        liveTurns = turns.map { SpeakerTurn(speaker: Self.liveID($0.speaker), start: $0.start, end: $0.end) }
        let ids = liveTurns.sorted { $0.start < $1.start }.map(\.speaker)
        var known = liveSpeakers.map(\.id)
        var added = false
        for id in ids where !known.contains(id) {
            known.append(id)
            added = true
        }
        // Relabel call phrases the new turns now attribute differently.
        for segment in finals.values where segment.track == .call {
            let speaker = speakerID(.call, segment.start, segment.end)
            if speaker != segment.speakerID {
                var relabelled = segment
                relabelled.speakerID = speaker
                finals[segment.id] = relabelled
                continuation.yield(.segment(relabelled))
            }
        }
        if added {
            var number = 2
            liveSpeakers = known.map { id in
                if id == SpeakerInfo.youID { return .you }
                defer { number += 1 }
                return SpeakerInfo(id: id, name: "Speaker \(number)", colorIndex: number - 1)
            }
            continuation.yield(.speakersUpdated(liveSpeakers))
        }
    }

    static func liveID(_ raw: String) -> String { "live-\(raw)" }

    // MARK: Controls

    func pause() async {
        guard phase == .running, !paused else { return }
        paused = true
        pausedAt = recordedTime
        // The phrase being spoken ends at the pause.
        for (kind, segmenter) in segmenters { await handle(kind, segmenter: await segmenter.flush()) }
        continuation.yield(.paused)
        publishStatus()
    }

    func resume() {
        guard paused else { return }
        paused = false
        resumedAt = now()
        continuation.yield(.resumed)
        publishStatus()
    }

    func markMoment() -> TimeInterval { recordedTime }

    private func publishStatus() {
        onLiveStatus(LiveSessionStatus(kind: .meeting, noteID: options.noteID, startedAt: startedAt,
                                       isPaused: paused, recordedTime: recordedTime, updatedAt: now()))
    }

    // MARK: Sleep and wake

    func willSleep() async {
        guard phase == .running else { return }
        await stopCapture()
        try? recording.closeAll()
        sleptAt = now()
        phase = .asleep
    }

    func didWake() async {
        guard phase == .asleep else { return }
        let offset = SpeechAudio.seconds(max(trackSamples[.microphone] ?? 0, trackSamples[.call] ?? 0))
        let gap = RecordingGap(start: sleptAt ?? now(), end: now(), at: offset, reason: .sleep)
        gaps.append(gap)
        continuation.yield(.gap(gap))
        for kind in Array(segmenters.keys) {
            if let segmenter = segmenters[kind] { await handle(kind, segmenter: await segmenter.jump(to: offset)) }
        }
        phase = .running
        do {
            try await startTrack(.microphone, source: try sources.microphone(deviceID: options.microphoneID), at: offset)
        } catch {
            continuation.yield(.failed(error as? SpeechError ?? .failed(String(describing: error))))
            await stop()
            return
        }
        if callAvailable, await callAudioAllowed() {
            do {
                try await startTrack(.call, source: try sources.call(options.callAudio), at: offset)
            } catch {
                callAvailable = false
                let reason = error as? SpeechError ?? .permissionDenied(.systemAudio)
                await onCallAudioFailure(reason)
                continuation.yield(.callAudioUnavailable(reason))
            }
        }
        readStartedTracks()
        publishStatus()
    }

    /// Stops the sources and waits until every captured phrase is cut.
    private func stopCapture() async {
        readStartedTracks()
        let running = active
        active = [:]
        for track in running.values { await track.source.stop() }
        for track in running.values { await track.task?.value }
    }

    // MARK: Stop

    /// Ends capture and returns once it has stopped; the last phrases, the
    /// final speaker pass and the result follow as events. Idempotent.
    func stop() async {
        if stopTask == nil {
            phase = .stopping
            let capture = Task { await self.stopCapture() }
            stopTask = capture
            finishTask = Task {
                await capture.value
                await self.finish()
            }
        }
        await stopTask?.value
    }

    /// Quit (spec 6.1): stops capture and closes the files, and returns
    /// within the quit deadline. Phrases already heard get up to
    /// `lastPhrasesAtQuit` to finish; the speaker pass is skipped (live
    /// labels stay) and kept audio converts at the next launch. Anything still
    /// running from an earlier `stop()` is cut short the same way.
    func stopForQuit() async {
        quitting = true
        let waiters = quitWaiters
        quitWaiters = []
        for waiter in waiters { waiter.resume() }
        await stop()
        await finishTask?.value
    }

    var isQuitting: Bool { quitting }

    /// How long quit waits for phrases still being recognized (a phrase is
    /// at most 15 s of audio and takes well under a second): part of the
    /// app's 10 s quit deadline.
    static let lastPhrasesAtQuit: Duration = .seconds(3)

    /// Runs `work` and returns when it ends or after `limit`, whichever is
    /// first.
    private static func waitAtMost(_ limit: Duration, _ work: @escaping @Sendable () async -> Void) async {
        let race = QuitRace<Void>()
        _ = await withCheckedContinuation { (continuation: CheckedContinuation<Void?, Never>) in
            race.begin(continuation)
            Task {
                await work()
                race.resume(())
            }
            Task {
                try? await Task.sleep(for: limit)
                race.resume(nil)
            }
        }
    }

    /// Runs `work`, or gives up as soon as the app quits (the work is left
    /// to end on its own; the process is exiting).
    private func untilQuit<T: Sendable>(_ work: @escaping @Sendable () async throws -> T) async -> T? {
        guard !quitting else { return nil }
        let once = QuitRace<T>()
        quitWaiters.append(once)
        return await withCheckedContinuation { continuation in
            once.begin(continuation)
            Task {
                let value = try? await work()
                once.resume(value)
            }
        }
    }

    private func finish() async {
        continuation.yield(.finishing(.transcribingLastPhrases))
        let pending = Array(lastTranscription.values)
        let lastPhrases: @Sendable () async -> Void = { for task in pending { await task.value } }
        // Quit still gives the phrases already heard a short, bounded wait:
        // nothing transcribes them after the app has gone.
        if await untilQuit(lastPhrases) == nil {
            await Self.waitAtMost(Self.lastPhrasesAtQuit, lastPhrases)
        }
        try? recording.closeAll()
        var segments = finals.values.sorted { $0.start < $1.start || ($0.start == $1.start && $0.track == .microphone) }
        var mapping: [String: String] = [:]
        if !quitting, let diarizer = stages.diarizer, live != nil || segments.contains(where: { $0.track == .call }) {
            continuation.yield(.finishing(.labellingSpeakers(progress: 0)))
            let files = recording.files(of: .call)
            let continuation = continuation
            if !files.isEmpty, let turns = await untilQuit({
                try await diarizer.diarize(files, progress: { progress in
                    continuation.yield(.finishing(.labellingSpeakers(progress: progress)))
                })
            }) {
                let finalTurns = turns.map { SpeakerTurn(speaker: "speaker-\($0.speaker)", start: $0.start, end: $0.end) }
                mapping = SpeakerAlignment.mapping(liveSegments: segments, finalTurns: finalTurns)
                segments = SpeakerAlignment.relabel(segments, finalTurns: finalTurns, mapping: mapping)
            }
            continuation.yield(.finishing(.labellingSpeakers(progress: 1)))
        }
        let (tracks, failure) = await recording.finish(keepAudio: options.keepAudio,
                                                       stopEarly: { [weak self] in await self?.isQuitting ?? true })
        // Conversion errors are reported; the CAF files stay as the tracks.
        if let failure { continuation.yield(.failed(failure)) }
        let result = TranscriptResult(segments: segments, speakers: SpeakerAlignment.speakers(for: segments),
                                      tracks: tracks, gaps: gaps, duration: recordedTime,
                                      languageCode: dominantLanguage(languages),
                                      audioDirectory: options.keepAudio ? recording.directory : nil,
                                      speakerMapping: mapping)
        phase = .ended
        continuation.yield(.ended(result))
        continuation.finish()
        await onEnd()
    }
}

/// One wait that ends with the work's value or with quit, whichever is first.
final class QuitRace<T: Sendable>: Sendable {
    private let state = Mutex<(continuation: CheckedContinuation<T?, Never>?, done: Bool, early: T??)>((nil, false, nil))

    func begin(_ continuation: CheckedContinuation<T?, Never>) {
        let early: T?? = state.withLock { s in
            if let early = s.early { return early }
            s.continuation = continuation
            return nil
        }
        if let early { continuation.resume(returning: early) }
    }

    func resume(_ value: T?) {
        let continuation = state.withLock { s -> CheckedContinuation<T?, Never>? in
            guard !s.done else { return nil }
            s.done = true
            guard let continuation = s.continuation else {
                s.early = .some(value)
                return nil
            }
            s.continuation = nil
            return continuation
        }
        continuation?.resume(returning: value)
    }
}

protocol QuitWaiter: Sendable {
    func resume()
}

extension QuitRace: QuitWaiter {
    func resume() { resume(nil) }
}
