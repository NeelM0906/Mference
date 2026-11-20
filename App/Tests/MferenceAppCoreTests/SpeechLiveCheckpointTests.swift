@preconcurrency import AVFoundation
import Darwin
import Foundation
@testable import MferenceAppCore
import Synchronization
import Testing

private let speechCheckpointEnabled = ProcessInfo.processInfo.environment["MFERENCE_SPEECH_MODELS"] != nil
    && ProcessInfo.processInfo.environment["MFERENCE_SPEECH_FIXTURES"] != nil

/// Checkpoint C-N1 (Notes spec 7, 10): the real speech stages on fixture
/// audio. Skipped unless both are set:
///
/// - `MFERENCE_SPEECH_MODELS`: an installed speech root (the `.installed`
///   markers and the pinned folders, as `SpeechModelInstaller` lays out).
/// - `MFERENCE_SPEECH_FIXTURES`: a folder with `references.txt` (name, tab,
///   reference text), the WAVs it names, `meeting-mix.wav`,
///   `meeting-trackA-mic.wav`, `meeting-trackB-call.wav` and
///   `meeting-timeline.txt` (`A|B start end` per line, seconds).
///
/// It loads Fluid Speech Pro, Silero VAD and the speaker-label models, so it
/// runs only under the machine lock, one model process at a time. Audio
/// comes from the fixtures through `FakeAudioSource`: no microphone, no call
/// tap, no TCC prompt, no network. Each measurement prints one line,
/// `METRIC {json}`, with a `metric` key naming it.
@Suite(.serialized, .enabled(if: speechCheckpointEnabled))
struct SpeechLiveCheckpointTests {
    static let environment = ProcessInfo.processInfo.environment

    @Test(.timeLimit(.minutes(20)))
    func checkpoint() async throws {
        let models = URL(fileURLWithPath: try #require(Self.environment["MFERENCE_SPEECH_MODELS"]), isDirectory: true)
        let fixtures = URL(fileURLWithPath: try #require(Self.environment["MFERENCE_SPEECH_FIXTURES"]), isDirectory: true)
        let scratch = FileManager.default.temporaryDirectory.appending(path: "mference-cn1-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }

        // Nothing is downloaded: no allowed hosts, and the packages must
        // already read as installed.
        let installer = SpeechModelInstaller(root: models, allowedHosts: [])
        #expect(installer.isInstalled(.speech), "the speech package is not installed under \(models.path)")
        #expect(installer.isInstalled(.speakerLabels), "speaker labels are not installed under \(models.path)")
        let loader = TimingLoader()
        let sources = FixtureSources()
        let service = SpeechService(.init(installer: installer, loader: loader, sources: sources,
                                          recordings: RecordingStore(root: scratch, availableBytes: { _ in 100 << 30 }),
                                          power: FakePowerEvents(), idleUnload: nil))
        Self.metric("footprint_start", Self.footprint())

        let timeline = try Self.timeline(fixtures.appending(path: "meeting-timeline.txt"))

        // (2) Files: text, WER and processing time. The first one also loads
        // the speech package, timed in (1).
        try await transcribeReferences(service, fixtures: fixtures)

        // (1) Load times, as the service loaded them.
        if let seconds = loader.speechLoad { Self.metric("load_speech", ["seconds": seconds]) }

        // (3) Dictation at real-time pace: end of speech to text ready.
        try await phraseLatency(service, sources: sources, fixtures: fixtures, timeline: timeline)

        // (4) Diarization: the file path with speaker labels, then LS-EEND live.
        try await fileDiarization(service, fixtures: fixtures, timeline: timeline)
        if let seconds = loader.speakerLabelsLoad { Self.metric("load_speaker_labels", ["seconds": seconds]) }
        // A failed step is recorded and reported, and the rest still runs.
        await Self.step("diarization_live") {
            try await liveDiarization(loader, models: models, fixtures: fixtures, timeline: timeline)
        }

        // (5) The two-track meeting at real-time pace.
        await Self.step("meeting") {
            try await twoTrackMeeting(service, sources: sources, fixtures: fixtures, timeline: timeline)
        }

        await service.shutdown()
        // (6) Peak memory.
        Self.metric("footprint_end", Self.footprint())
    }

    // MARK: (2) Files and WER

    func transcribeReferences(_ service: SpeechService, fixtures: URL) async throws {
        let lines = try String(contentsOf: fixtures.appending(path: "references.txt"), encoding: .utf8)
            .split(whereSeparator: \.isNewline)
        var totalErrors = 0
        var totalWords = 0
        for line in lines {
            let parts = line.split(separator: "\t", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { continue }
            let (name, reference) = (parts[0], parts[1])
            let run = try await transcribe(service, fixtures.appending(path: name), speakerLabels: false)
            let text = run.result.segments.map(\.text).joined(separator: " ")
            let (errors, words) = Self.wordErrors(hypothesis: text, reference: reference)
            totalErrors += errors
            totalWords += words
            Self.metric("file_wer", [
                "file": name, "text": text, "reference": reference,
                "language": run.result.languageCode ?? "",
                "wer": words == 0 ? 0 : Double(errors) / Double(words), "errors": errors, "words": words,
                "audio_seconds": run.result.duration,
                "processing_seconds": run.processing,
                "processing_seconds_per_audio_minute": run.result.duration > 0 ? run.processing / (run.result.duration / 60) : 0,
                "wall_seconds_from_submit": run.wall,
            ])
        }
        Self.metric("file_wer_total", ["wer": totalWords == 0 ? 0 : Double(totalErrors) / Double(totalWords),
                                       "errors": totalErrors, "words": totalWords])
    }

    struct FileRun {
        var result: TranscriptResult
        /// From `.started` to `.finished` (no queueing, no first load).
        var processing: Double
        /// From the call to `.finished`.
        var wall: Double
    }

    func transcribe(_ service: SpeechService, _ url: URL, speakerLabels: Bool) async throws -> FileRun {
        let clock = ContinuousClock()
        let submitted = clock.now
        let job = try await service.transcribeFile(url, options: FileTranscriptionOptions(
            noteID: UUID(), speakerLabels: speakerLabels, keepAudio: false))
        var started: ContinuousClock.Instant?
        for await event in job.events {
            switch event {
            case .started: started = clock.now
            case .finished(let result):
                let end = clock.now
                return FileRun(result: result, processing: Self.seconds(end - (started ?? submitted)),
                               wall: Self.seconds(end - submitted))
            case .failed(let error):
                throw error
            case .cancelled:
                throw CancellationError()
            default:
                break
            }
        }
        throw SpeechError.failed("\(url.lastPathComponent): the job ended without a result")
    }

    // MARK: (3) Phrase latency

    func phraseLatency(_ service: SpeechService, sources: FixtureSources, fixtures: URL,
                       timeline: [Turn]) async throws {
        let samples = try await Self.decode(fixtures.appending(path: "meeting-trackA-mic.wav"))
        let microphone = FakeAudioSource(track: .microphone, chunks: [])
        sources.use(microphone: microphone, call: nil)
        let session = try await service.startDictation(DictationOptions())
        let log = TimedLog(session.events)
        let start = ContinuousClock.now
        await log.setStart(start)
        try await Self.play([(microphone, samples)], from: start)
        // Trailing silence so the last phrase closes, then stop.
        try await Task.sleep(for: .seconds(2))
        await session.stop()
        #expect(await log.waitUntil(seconds: 120) { $0.contains { $0.event == .ended } })

        let events = await log.entries
        var startedAt: [UUID: Double] = [:]
        var latencies: [Double] = []
        let speechTurns = timeline.filter { $0.speaker == "A" }
        for entry in events {
            switch entry.event {
            case .phraseStarted(let id):
                startedAt[id] = entry.time
            case .phraseFinished(let id, let text, let language):
                // The A turn whose start is nearest the phrase start.
                let began = startedAt[id] ?? entry.time
                guard let turn = speechTurns.min(by: { abs($0.start - began) < abs($1.start - began) }) else { continue }
                let latency = entry.time - turn.end
                latencies.append(latency)
                Self.metric("dictation_phrase", ["text": text, "language": language ?? "",
                                                 "turn_start": turn.start, "turn_end": turn.end,
                                                 "phrase_started_at": began, "text_ready_at": entry.time,
                                                 "end_of_speech_to_text_seconds": latency])
            case .failed(let error):
                Self.metric("dictation_failed", ["error": error.message])
            default:
                break
            }
        }
        let sorted = latencies.sorted()
        Self.metric("dictation_latency", [
            "phrases": sorted.count, "a_turns": speechTurns.count,
            "median_seconds": sorted.isEmpty ? 0 : sorted[sorted.count / 2],
            "max_seconds": sorted.last ?? 0,
            "note": "includes the 500 ms end-of-speech silence the segmenter waits for",
        ])
    }

    // MARK: (4) Diarization

    func fileDiarization(_ service: SpeechService, fixtures: URL, timeline: [Turn]) async throws {
        let run = try await transcribe(service, fixtures.appending(path: "meeting-mix.wav"), speakerLabels: true)
        let segments = run.result.segments
        let hypothesis = segments.map { Turn(speaker: $0.speakerID, start: $0.start, end: $0.end) }
        let score = Self.agreement(hypothesis, timeline)
        for segment in segments {
            Self.metric("diarization_file_segment", ["speaker": segment.speakerID, "start": segment.start,
                                                     "end": segment.end, "text": segment.text,
                                                     "reference": Self.majority(segment.start, segment.end, timeline) ?? ""])
        }
        Self.metric("diarization_file", [
            "speakers": Set(segments.map(\.speakerID)).count, "speakers_listed": run.result.speakers.count,
            "reference_speakers": Set(timeline.map(\.speaker)).count, "segments": segments.count,
            "segment_agreement": score.segments, "time_agreement": score.time,
            "mapping": score.mapping, "processing_seconds": run.processing,
        ])
    }

    func liveDiarization(_ loader: TimingLoader, models: URL, fixtures: URL, timeline: [Turn]) async throws {
        let diarizer = try await loader.loadedOrNewDiarizer(from: models)
        let live = try await diarizer.makeLiveSession()
        let samples = try await Self.decode(fixtures.appending(path: "meeting-mix.wav"))
        let chunk = SpeechAudio.samples(0.5)
        let clock = ContinuousClock()
        let began = clock.now
        var turns: [SpeakerTurn] = []
        for offset in stride(from: 0, to: samples.count, by: chunk) {
            let piece = Array(samples[offset..<min(samples.count, offset + chunk)])
            turns = try await live.feed(piece, at: SpeechAudio.seconds(offset))
        }
        let processing = Self.seconds(clock.now - began)
        let hypothesis = turns.map { Turn(speaker: $0.speaker, start: $0.start, end: $0.end) }
        let score = Self.agreement(hypothesis, timeline)
        Self.metric("diarization_live", [
            "speakers": Set(turns.map(\.speaker)).count, "turns": turns.count,
            "segment_agreement": score.segments, "time_agreement": score.time, "mapping": score.mapping,
            "audio_seconds": SpeechAudio.seconds(samples.count), "processing_seconds": processing,
        ])
    }

    // MARK: (5) Two-track meeting

    func twoTrackMeeting(_ service: SpeechService, sources: FixtureSources, fixtures: URL,
                         timeline: [Turn]) async throws {
        let mic = try await Self.decode(fixtures.appending(path: "meeting-trackA-mic.wav"))
        let call = try await Self.decode(fixtures.appending(path: "meeting-trackB-call.wav"))
        let microphone = FakeAudioSource(track: .microphone, chunks: [])
        let callSource = FakeAudioSource(track: .call, chunks: [])
        sources.use(microphone: microphone, call: callSource)
        let recording = try await service.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .allApps,
                                                                      speakerLabels: true, keepAudio: false))
        let log = TimedLog(recording.events)
        let start = ContinuousClock.now
        await log.setStart(start)
        try await Self.play([(microphone, mic), (callSource, call)], from: start)
        try await Task.sleep(for: .seconds(2))
        let stopped = ContinuousClock.now
        await recording.stop()
        #expect(await log.waitUntil(seconds: 600) { entries in
            entries.contains { if case .ended = $0.event { true } else { false } }
        })
        let finishSeconds = Self.seconds(ContinuousClock.now - stopped)

        let entries = await log.entries
        var callAudio = false
        var removed = 0
        var result: TranscriptResult?
        for entry in entries {
            switch entry.event {
            case .started(let available): callAudio = available
            case .segmentRemoved: removed += 1
            case .callAudioUnavailable(let error): Self.metric("meeting_call_unavailable", ["error": error.message])
            case .failed(let error): Self.metric("meeting_failed", ["error": error.message])
            case .ended(let final): result = final
            default: break
            }
        }
        let segments = (result?.segments ?? []).sorted { $0.start < $1.start }
        var trackMismatches = 0
        var orderViolations = 0
        var lastTurn = -1
        for segment in segments {
            let index = Self.bestTurn(segment.start, segment.end, timeline)
            let expected: TrackKind? = index.map { timeline[$0].speaker == "A" ? .microphone : .call }
            if let expected, expected != segment.track { trackMismatches += 1 }
            if let index {
                if index < lastTurn { orderViolations += 1 }
                lastTurn = max(lastTurn, index)
            }
            Self.metric("meeting_segment", ["track": segment.track.rawValue, "speaker": segment.speakerID,
                                            "start": segment.start, "end": segment.end, "text": segment.text,
                                            "turn": index.map { "\(timeline[$0].speaker)\($0)" } ?? ""])
        }
        Self.metric("meeting", [
            "call_audio": callAudio, "segments": segments.count, "timeline_turns": timeline.count,
            "echo_drops": removed, "track_mismatches": trackMismatches, "order_violations": orderViolations,
            "speakers": result?.speakers.map(\.name) ?? [], "seconds_from_stop_to_ended": finishSeconds,
        ])
    }

    // MARK: Fixtures and audio

    struct Turn: Sendable {
        var speaker: String
        var start: Double
        var end: Double
    }

    static func timeline(_ url: URL) throws -> [Turn] {
        try String(contentsOf: url, encoding: .utf8).split(whereSeparator: \.isNewline).compactMap { line in
            let parts = line.split(whereSeparator: \.isWhitespace)
            guard parts.count == 3, let start = Double(parts[1]), let end = Double(parts[2]) else { return nil }
            return Turn(speaker: String(parts[0]), start: start, end: end)
        }
    }

    /// The whole file as 16 kHz mono (fixtures are about a minute at most).
    static func decode(_ url: URL) async throws -> [Float] {
        let decoder = try await FileAudioDecoder(url: url)
        var samples: [Float] = []
        while let chunk = try decoder.next(maxSamples: 16_000) { samples += chunk }
        return samples
    }

    /// Pushes each source's samples in 100 ms chunks at real-time pace.
    static func play(_ tracks: [(FakeAudioSource, [Float])], from start: ContinuousClock.Instant) async throws {
        let chunk = SpeechAudio.samples(0.1)
        let longest = tracks.map(\.1.count).max() ?? 0
        for (index, offset) in stride(from: 0, to: longest, by: chunk).enumerated() {
            try await Task.sleep(until: start + .milliseconds(100 * index), clock: .continuous)
            for (source, samples) in tracks where offset < samples.count {
                source.push(AudioChunk(samples: Array(samples[offset..<min(samples.count, offset + chunk)])))
            }
        }
    }

    // MARK: Scores

    /// Word errors (substitutions, insertions, deletions) after lower-casing
    /// and stripping punctuation, and the reference word count.
    static func wordErrors(hypothesis: String, reference: String) -> (errors: Int, words: Int) {
        let hyp = words(hypothesis)
        let ref = words(reference)
        guard !ref.isEmpty else { return (hyp.count, 0) }
        var previous = Array(0...hyp.count)
        for (i, word) in ref.enumerated() {
            var current = [i + 1] + Array(repeating: 0, count: hyp.count)
            for (j, guess) in hyp.enumerated() {
                current[j + 1] = min(previous[j + 1] + 1, current[j] + 1, previous[j] + (word == guess ? 0 : 1))
            }
            previous = current
        }
        return (previous[hyp.count], ref.count)
    }

    static func words(_ text: String) -> [String] {
        String(text.lowercased().filter { !$0.isPunctuation && !$0.isSymbol })
            .split(whereSeparator: \.isWhitespace).map(String.init)
    }

    static func overlap(_ a: (Double, Double), _ b: (Double, Double)) -> Double {
        max(0, min(a.1, b.1) - max(a.0, b.0))
    }

    /// The timeline turn a span overlaps most.
    static func bestTurn(_ start: Double, _ end: Double, _ timeline: [Turn]) -> Int? {
        let scored = timeline.indices.map { ($0, overlap((start, end), (timeline[$0].start, timeline[$0].end))) }
        guard let best = scored.max(by: { $0.1 < $1.1 }), best.1 > 0 else { return nil }
        return best.0
    }

    static func majority(_ start: Double, _ end: Double, _ timeline: [Turn]) -> String? {
        bestTurn(start, end, timeline).map { timeline[$0].speaker }
    }

    /// Each hypothesis label maps to the reference speaker it overlaps
    /// most. `time`: share of reference speech whose mapped label is right;
    /// `segments`: share of hypothesis spans whose majority reference
    /// speaker is their mapped label.
    static func agreement(_ hypothesis: [Turn], _ reference: [Turn])
        -> (time: Double, segments: Double, mapping: [String: String]) {
        var overlaps: [String: [String: Double]] = [:]
        for h in hypothesis {
            for r in reference {
                overlaps[h.speaker, default: [:]][r.speaker, default: 0] += overlap((h.start, h.end), (r.start, r.end))
            }
        }
        let mapping = overlaps.compactMapValues { $0.max { $0.value < $1.value }?.key }
        let total = reference.reduce(0) { $0 + ($1.end - $1.start) }
        let correct = overlaps.reduce(0.0) { sum, entry in sum + (mapping[entry.key].flatMap { entry.value[$0] } ?? 0) }
        let spans = hypothesis.filter { majority($0.start, $0.end, reference) != nil }
        let right = spans.filter { mapping[$0.speaker] == majority($0.start, $0.end, reference) }.count
        return (total > 0 ? min(1, correct / total) : 0, spans.isEmpty ? 0 : Double(right) / Double(spans.count), mapping)
    }

    // MARK: Output

    /// Runs one step; a failure prints `step_failed` and is recorded as an
    /// issue without stopping the checkpoint.
    static func step(_ name: String, _ body: () async throws -> Void) async {
        do {
            try await body()
        } catch {
            metric("step_failed", ["step": name, "error": (error as? SpeechError)?.message ?? String(describing: error)])
            Issue.record(error, "\(name) failed")
        }
    }

    static func metric(_ name: String, _ fields: [String: Any]) {
        var line = fields
        line["metric"] = name
        guard let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else { return }
        print("METRIC \(json)")
    }

    static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    /// This process's physical footprint now and its lifetime peak, in MB.
    static func footprint() -> [String: Any] {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard status == 0 else { return ["error": "proc_pid_rusage \(status)"] }
        return ["phys_footprint_mb": Double(info.ri_phys_footprint) / 1_048_576,
                "peak_phys_footprint_mb": Double(info.ri_lifetime_max_phys_footprint) / 1_048_576]
    }
}

// MARK: Helpers

/// The real FluidAudio loader, timing each load and keeping the diarizer
/// so the LS-EEND check reuses the one the service loaded.
final class TimingLoader: SpeechModelLoading {
    private let base = FluidModelLoader()
    private let speech = Mutex<Double?>(nil)
    private let labels = Mutex<Double?>(nil)
    private let diarizer = Mutex<(any SpeakerDiarizing)?>(nil)

    var speechLoad: Double? { speech.withLock { $0 } }
    var speakerLabelsLoad: Double? { labels.withLock { $0 } }

    func loadSpeech(from directory: URL) async throws -> (recognizer: any SpeechRecognizing,
                                                          voiceActivity: any VoiceActivityDetecting) {
        let clock = ContinuousClock()
        let began = clock.now
        let loaded = try await base.loadSpeech(from: directory)
        let seconds = SpeechLiveCheckpointTests.seconds(clock.now - began)
        speech.withLock { $0 = $0 ?? seconds }
        return loaded
    }

    func loadSpeakerLabels(from directory: URL) async throws -> any SpeakerDiarizing {
        let clock = ContinuousClock()
        let began = clock.now
        let loaded = try await base.loadSpeakerLabels(from: directory)
        let seconds = SpeechLiveCheckpointTests.seconds(clock.now - began)
        labels.withLock { $0 = $0 ?? seconds }
        diarizer.withLock { $0 = $0 ?? loaded }
        return loaded
    }

    func loadedOrNewDiarizer(from directory: URL) async throws -> any SpeakerDiarizing {
        if let loaded = diarizer.withLock({ $0 }) { return loaded }
        return try await loadSpeakerLabels(from: directory)
    }
}

/// Hands the service whichever fixture sources the current step set.
final class FixtureSources: AudioSourceFactory {
    private struct Current: Sendable {
        var microphone: FakeAudioSource?
        var call: FakeAudioSource?
    }
    private let current = Mutex(Current())

    func use(microphone: FakeAudioSource, call: FakeAudioSource?) {
        current.withLock { $0 = Current(microphone: microphone, call: call) }
    }

    func microphone(deviceID: String?) throws -> any AudioSource {
        guard let source = current.withLock({ $0.microphone }) else { throw SpeechError.failed("no fixture microphone") }
        return source
    }

    func call(_ source: CallAudioSource) throws -> any AudioSource {
        guard let call = current.withLock({ $0.call }) else { throw SpeechError.permissionDenied(.systemAudio) }
        return call
    }

    func inputDevices() async -> [AudioInputDevice] { [] }
    func runningAudioApps() async -> [RunningAudioApp] { [] }
}

/// One stream's events with their arrival time, in seconds from `start`.
actor TimedLog<Event: Sendable> {
    struct Entry: Sendable {
        var time: Double
        var event: Event
    }
    private(set) var entries: [Entry] = []
    private var start = ContinuousClock.now

    init(_ stream: AsyncStream<Event>) {
        Task { await self.consume(stream) }
    }

    func setStart(_ instant: ContinuousClock.Instant) { start = instant }

    private func consume(_ stream: AsyncStream<Event>) async {
        for await event in stream {
            entries.append(Entry(time: SpeechLiveCheckpointTests.seconds(ContinuousClock.now - start), event: event))
        }
    }

    func waitUntil(seconds: Double, _ condition: ([Entry]) -> Bool) async -> Bool {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while !condition(entries) {
            if ContinuousClock.now > deadline { return false }
            try? await Task.sleep(for: .milliseconds(20))
        }
        return true
    }
}
