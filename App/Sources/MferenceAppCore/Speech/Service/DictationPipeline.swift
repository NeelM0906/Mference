import Foundation

/// One dictation (spec 6.3): microphone → voice activity → on each end of
/// speech the phrase → the recogniser → FillerFilter (if on) →
/// `.phraseFinished`. Nothing is written to disk.
actor DictationPipeline {
    private let source: any AudioSource
    private let segmenter: TrackSegmenter
    private let recognizer: any SpeechRecognizing
    private let lane: TranscriptionLane
    private let continuation: AsyncStream<DictationEvent>.Continuation
    private let onEnd: @Sendable () async -> Void
    private var language: SpeechLanguage
    private let cleanUp: Bool

    private var captureTask: Task<Void, Never>?
    /// The last phrase sent to the recogniser; each waits for the one
    /// before, so phrases finish in the order they were spoken.
    private var lastTranscription: Task<Void, Never>?
    private var ended = false
    private var stopping: Task<Void, Never>?

    init(source: any AudioSource, segmenter: TrackSegmenter, recognizer: any SpeechRecognizing,
         lane: TranscriptionLane, options: DictationOptions,
         continuation: AsyncStream<DictationEvent>.Continuation, onEnd: @escaping @Sendable () async -> Void) {
        self.source = source
        self.segmenter = segmenter
        self.recognizer = recognizer
        self.lane = lane
        self.continuation = continuation
        self.onEnd = onEnd
        language = options.language
        cleanUp = options.cleanUpFillers
    }

    /// Starts capture; throws what the microphone reports.
    func start() async throws {
        let stream = try await source.start()
        captureTask = Task { await self.consume(stream) }
    }

    func setLanguage(_ language: SpeechLanguage) { self.language = language }

    /// Finishes the phrase in progress, then ends. Idempotent: a second call
    /// waits for the first.
    func stop() async {
        if let stopping { return await stopping.value }
        let task = Task { await self.finish() }
        stopping = task
        await task.value
    }

    private func consume(_ stream: AsyncThrowingStream<AudioChunk, Error>) async {
        do {
            for try await chunk in stream {
                continuation.yield(.level(SpeechAudio.level(chunk.samples)))
                handle(try await segmenter.feed(chunk.samples))
            }
        } catch {
            continuation.yield(.failed(error as? SpeechError ?? .failed(String(describing: error))))
        }
        handle(await segmenter.flush())
        // Capture ended on its own (device gone, permission revoked): end.
        if stopping == nil { Task { await self.stop() } }
    }

    private func handle(_ events: [SegmenterEvent]) {
        for event in events {
            switch event {
            case .started(let id, _):
                continuation.yield(.phraseStarted(id))
            case .dropped(let id):
                continuation.yield(.phraseDropped(id))
            case .finished(let phrase):
                let previous = lastTranscription
                lastTranscription = Task {
                    await previous?.value
                    await self.transcribe(phrase)
                }
            }
        }
    }

    private func transcribe(_ phrase: Phrase) async {
        let recognizer = recognizer
        let samples = phrase.samples
        do {
            let heard = try await lane.run { try await recognizer.transcribe(samples) }
            let code = heard.languageCode ?? PhraseLanguage.code(for: heard.text, setting: language)
            // Only a pinned language picks its own filler list: a guess from a
            // short phrase can be wrong ("Moro em Lisboa" read as Spanish).
            let fillerLanguage: String? = if case .fixed(let pinned) = language { pinned } else { nil }
            let text = cleanUp ? FillerFilter.clean(heard.text, languageCode: fillerLanguage) : heard.text
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            continuation.yield(trimmed.isEmpty ? .phraseDropped(phrase.id)
                                               : .phraseFinished(phrase.id, text: trimmed, languageCode: code))
        } catch {
            continuation.yield(.phraseDropped(phrase.id))
            continuation.yield(.failed(error as? SpeechError ?? .failed(String(describing: error))))
        }
    }

    private func finish() async {
        await source.stop()
        await captureTask?.value
        await lastTranscription?.value
        guard !ended else { return }
        ended = true
        continuation.yield(.ended)
        continuation.finish()
        await onEnd()
    }
}
