import Foundation

/// Transcribes one imported file (spec 4.6, 6.3): decode to 16 kHz mono in
/// bounded chunks → voice activity → the recogniser per phrase (15 s
/// windows) → the offline speaker pass if asked → the transcript. Progress
/// comes from audio seconds done; time left from measured speed. Waits
/// between chunks while a live session holds the microphone.
struct FileTranscriber: Sendable {
    /// Decoded per step: one second.
    static let chunkSamples = 16_000

    var recognizer: any SpeechRecognizing
    var voiceActivity: any VoiceActivityDetecting
    var diarizer: (any SpeakerDiarizing)?
    var lane: TranscriptionLane
    var segmenter: Segmenter.Configuration
    /// Suspends while dictation or a meeting runs; returns true if it waited.
    var waitWhileLive: @Sendable () async -> Bool
    var clock: @Sendable () -> ContinuousClock.Instant = { ContinuousClock.now }

    func run(file: URL, options: FileTranscriptionOptions, events: AsyncStream<FileJobEvent>.Continuation,
             status: @escaping @Sendable (FileJobStatus.Phase, Double, TimeInterval?, String?) -> Void) async throws
        -> TranscriptResult {
        let decoder = try await FileAudioDecoder(url: file)
        let duration = decoder.duration
        events.yield(.started(duration: duration))
        let track = TrackSegmenter(track: .file, detector: try await voiceActivity.makeStream(),
                                   frameLength: voiceActivity.frameLength, configuration: segmenter)
        var meter = SpeedMeter()
        var done: TimeInterval = 0
        var segments: [TranscriptSegment] = []
        var languages: [String?] = []
        var lastReport = -1.0

        func transcribe(_ events: [SegmenterEvent]) async throws {
            for case .finished(let phrase) in events {
                let recognizer = recognizer
                let samples = phrase.samples
                let heard = try await lane.run { try await recognizer.transcribe(samples) }
                let text = heard.text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { continue }
                let code = heard.languageCode ?? PhraseLanguage.code(for: text, setting: options.language)
                languages.append(code)
                segments.append(TranscriptSegment(id: phrase.id, track: .file, start: phrase.start, end: phrase.end,
                                                  speakerID: SpeakerInfo.othersID, text: text, isFinal: true,
                                                  languageCode: code))
            }
        }

        while true {
            try Task.checkCancellation()
            if await waitWhileLive() {
                events.yield(.progress(fraction: duration > 0 ? min(1, done / duration) : 0, remaining: nil,
                                       languageCode: dominantLanguage(languages)))
            }
            let started = clock()
            guard let chunk = try decoder.next(maxSamples: Self.chunkSamples) else { break }
            try await transcribe(try await track.feed(chunk))
            let took = clock() - started
            let seconds = SpeechAudio.seconds(chunk.count)
            meter.record(audio: seconds, took: Double(took.components.seconds) + Double(took.components.attoseconds) / 1e18)
            done += seconds
            let fraction = duration > 0 ? min(0.99, done / duration) : 0
            if fraction - lastReport >= 0.01 {
                lastReport = fraction
                let remaining = meter.remaining(for: duration - done)
                let language = dominantLanguage(languages)
                events.yield(.progress(fraction: fraction, remaining: remaining, languageCode: language))
                status(.running, fraction, remaining, language)
            }
        }
        try await transcribe(await track.flush())

        var speakers: [SpeakerInfo] = segments.isEmpty ? [] : [.others]
        if options.speakerLabels, let diarizer, !segments.isEmpty {
            let turns = try await diarizer.diarize([(file, 0)], progress: { _ in })
                .map { SpeakerTurn(speaker: "speaker-\($0.speaker)", start: $0.start, end: $0.end) }
            segments = SpeakerAlignment.relabel(segments, finalTurns: turns, mapping: [:])
            speakers = SpeakerAlignment.speakers(for: segments)
        }
        let language = dominantLanguage(languages)
        events.yield(.progress(fraction: 1, remaining: 0, languageCode: language))
        status(.running, 1, 0, language)
        return TranscriptResult(segments: segments, speakers: speakers, duration: max(duration, done),
                                languageCode: language)
    }
}
