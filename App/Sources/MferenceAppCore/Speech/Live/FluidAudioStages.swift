@preconcurrency import AVFoundation
import CoreML
@preconcurrency import FluidAudio
import Foundation

// The FluidAudio adapters behind the speech stages (Notes spec 3, spike S1).
// Every model loads from the verified local folders `SpeechModelInstaller`
// wrote: `AsrModels.loadLocal(from:version: .v3)`, `MLModel(contentsOf:)`
// with `VadManager(config:vadModel:)`, `LSEENDModel(modelURL:)` and the
// offline diarizer's models built by hand. Nothing here downloads
// (`NetworkAllowlistTests`). FluidAudio is never given a `language:` hint:
// Fluid Speech Pro's joint has no top-K outputs (S1).

/// Loads the installed packages from `root` (`Speech/<repo>/<revision>/`).
struct FluidModelLoader: SpeechModelLoading {
    var speech: SpeechModelPackage = SpeechModelPins.speech
    var speakerLabels: SpeechModelPackage = SpeechModelPins.speakerLabels

    static let vadModelName = "silero-vad-unified-256ms-v6.2.1.mlmodelc"
    /// The LS-EEND variant live meetings use (meeting-room audio).
    static let liveVariant = ModelNames.LSEEND.Variant.ami

    func folder(_ root: URL, _ package: SpeechModelPackage, repoSuffix: String) throws -> URL {
        guard let source = package.sources.first(where: { $0.repo.hasSuffix(repoSuffix) }) else {
            throw SpeechError.failed("no pinned source \(repoSuffix)")
        }
        return root.appending(path: SpeechModelInstaller.installPrefix(of: source), directoryHint: .isDirectory)
    }

    func loadSpeech(from root: URL) async throws -> (recognizer: any SpeechRecognizing,
                                                     voiceActivity: any VoiceActivityDetecting) {
        let asrFolder = try folder(root, speech, repoSuffix: "/fluid-speech-pro-coreml")
        let vadURL = try folder(root, speech, repoSuffix: "/silero-vad-coreml").appending(path: Self.vadModelName)
        // The first load compiles for the Neural Engine (seconds): off the
        // cooperative pool, on a thread of its own.
        let loaded = try await Self.offPool {
            let models = try AsrModels.loadLocal(from: asrFolder, version: .v3)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            return UncheckedBox((models, try MLModel(contentsOf: vadURL, configuration: configuration)))
        }
        let manager = AsrManager(config: .default, models: loaded.value.0)
        let vad = VadManager(config: .default, vadModel: loaded.value.1)
        return (FluidRecognizer(manager: manager), FluidVoiceActivity(manager: vad))
    }

    func loadSpeakerLabels(from root: URL) async throws -> any SpeakerDiarizing {
        let offlineFolder = try folder(root, speakerLabels, repoSuffix: "/speaker-diarization-coreml")
        let models = try await Self.offPool { try FluidDiarizer.offlineModels(offlineFolder) }
        return FluidDiarizer(offline: models, liveModelURL: liveDiarizerModel(root))
    }

    /// The pinned LS-EEND compiled model, in the layout FluidAudio 0.18.4's
    /// `LSEENDModel` loads: `<repo subpath>/<step>/<name>_<step>.mlmodelc`
    /// (`optimized/ami/100ms/ls_eend_ami_100ms.mlmodelc`), its config
    /// embedded in the model's metadata. The shortest pinned step wins. Nil
    /// when the pins hold another export (the older `AMI/ls_eend_ami_step`
    /// one has no embedded config and other state tensors): live labels are
    /// then unavailable, and the final speaker pass still runs.
    func liveDiarizerModel(_ root: URL) -> URL? {
        guard let source = speakerLabels.sources.first(where: { $0.repo.hasSuffix("/ls-eend-coreml") }) else { return nil }
        let prefix = Self.liveVariant.repo.subPath.map { "\($0)/" } ?? ""
        for step in ModelNames.LSEEND.StepSize.allCases {
            let model = prefix + Self.liveVariant.fileName(forStep: step)
            if source.files.contains(where: { $0.repoPath.hasPrefix(model + "/") }) {
                return root.appending(path: SpeechModelInstaller.installPrefix(of: source), directoryHint: .isDirectory)
                    .appending(path: model, directoryHint: .isDirectory)
            }
        }
        return nil
    }

    /// The config `LSEENDModel` reads from the model's creator metadata
    /// (`userDefinedMetadata.config` in the compiled model's
    /// `metadata.json`), checked before the model is compiled so a wrong
    /// export fails at once with a clear reason.
    static func liveDiarizerConfig(_ model: URL) throws -> LSEENDMetadata {
        let name = model.lastPathComponent
        guard let data = try? Data(contentsOf: model.appending(path: "metadata.json")),
              let object = try? JSONSerialization.jsonObject(with: data) else {
            throw SpeechError.failed("live speaker labels: \(name) has no readable metadata.json")
        }
        let entry = (object as? [[String: Any]])?.first ?? (object as? [String: Any])
        guard let user = entry?["userDefinedMetadata"] as? [String: Any], let config = user["config"] as? String else {
            throw SpeechError.failed("live speaker labels: \(name) has no embedded config, so it is not an LS-EEND export FluidAudio loads")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        do {
            return try decoder.decode(LSEENDMetadata.self, from: Data(config.utf8))
        } catch {
            throw SpeechError.failed("live speaker labels: the config embedded in \(name) does not decode (\(error))")
        }
    }

    /// Runs blocking work (a Core ML compile) on its own thread.
    static func offPool<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            Thread.detachNewThread {
                continuation.resume(with: Result { try work() })
            }
        }
    }
}

/// Fluid Speech Pro: one phrase, a fresh decoder state, no language hint.
final class FluidRecognizer: SpeechRecognizing {
    let manager: AsrManager
    /// FluidAudio refuses audio under one second; shorter phrases are padded with silence.
    static let minimumSamples = 16_000

    init(manager: AsrManager) { self.manager = manager }

    func transcribe(_ samples: [Float]) async throws -> RecognizedPhrase {
        var audio = samples
        if audio.count < Self.minimumSamples { audio += [Float](repeating: 0, count: Self.minimumSamples - audio.count) }
        var state = TdtDecoderState.make()
        let result = try await manager.transcribe(audio, decoderState: &state)
        return RecognizedPhrase(text: result.text)
    }
}

/// Silero v6.2.1 (unified, 256 ms frames) through `VadManager`.
final class FluidVoiceActivity: VoiceActivityDetecting {
    let manager: VadManager
    var frameLength: Int { VadManager.chunkSize }

    init(manager: VadManager) { self.manager = manager }

    func makeStream() async throws -> any VoiceActivityStream { Stream(manager: manager) }

    actor Stream: VoiceActivityStream {
        let manager: VadManager
        var state = VadStreamState.initial()
        init(manager: VadManager) { self.manager = manager }

        func probability(of frame: [Float]) async throws -> Float {
            let result = try await manager.processStreamingChunk(frame, state: state)
            state = result.state
            return result.probability
        }
    }
}

/// LS-EEND live, the offline community-1 pipeline for the final pass.
final class FluidDiarizer: SpeakerDiarizing, @unchecked Sendable {
    let offline: OfflineDiarizerModels
    /// Nil when no LS-EEND model in FluidAudio's layout is pinned.
    let liveModelURL: URL?

    init(offline: OfflineDiarizerModels, liveModelURL: URL?) {
        self.offline = offline
        self.liveModelURL = liveModelURL
    }

    /// The offline models from their files (FluidAudio's own loader can
    /// download missing files, so it is not used).
    static func offlineModels(_ folder: URL) throws -> OfflineDiarizerModels {
        let started = Date()
        func model(_ name: String, _ units: MLComputeUnits) throws -> MLModel {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = units
            return try MLModel(contentsOf: folder.appending(path: name), configuration: configuration)
        }
        return OfflineDiarizerModels(segmentationModel: try model("Segmentation.mlmodelc", .all),
                                     fbankModel: try model("FBank.mlmodelc", .cpuOnly),
                                     embeddingModel: try model("Embedding.mlmodelc", .all),
                                     pldaRhoModel: try model("PldaRho.mlmodelc", .all),
                                     pldaPsi: try pldaPsi(folder.appending(path: "plda-parameters.json")),
                                     compilationDuration: Date().timeIntervalSince(started))
    }

    /// `tensors.psi.data_base64`: little-endian Float32 values.
    static func pldaPsi(_ url: URL) throws -> [Double] {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        guard let root = object as? [String: Any], let tensors = root["tensors"] as? [String: Any],
              let psi = tensors["psi"] as? [String: Any], let base64 = psi["data_base64"] as? String,
              let data = Data(base64Encoded: base64, options: .ignoreUnknownCharacters), data.count >= 4 else {
            throw SpeechError.failed("plda-parameters.json has no psi tensor")
        }
        var floats = [Float](repeating: 0, count: data.count / 4)
        _ = floats.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return floats.map(Double.init)
    }

    func makeLiveSession() async throws -> any LiveDiarizing {
        guard let url = liveModelURL else {
            throw SpeechError.failed("live speaker labels: no pinned LS-EEND model in the layout FluidAudio loads")
        }
        _ = try FluidModelLoader.liveDiarizerConfig(url)
        let model = try await FluidModelLoader.offPool { try UncheckedBox(LSEENDModel(modelURL: url, computeUnits: .cpuOnly)) }
        return try FluidLiveDiarizer(model: model.value)
    }

    func diarize(_ files: [(url: URL, startOffset: TimeInterval)],
                 progress: @escaping @Sendable (Double) -> Void) async throws -> [SpeakerTurn] {
        guard !files.isEmpty else { return [] }
        // One pass over the whole call track keeps speaker ids consistent:
        // the files (one per capture run) are joined on disk, each at its
        // offset in recording time.
        let joined = FileManager.default.temporaryDirectory.appending(path: "mference-call-\(UUID().uuidString).caf")
        defer { try? FileManager.default.removeItem(at: joined) }
        try await Self.join(files, into: joined)
        let manager = OfflineDiarizerManager(config: .default)
        manager.initialize(models: offline)
        let result = try await manager.process(joined) { done, total in
            progress(total > 0 ? Double(done) / Double(total) : 0)
        }
        return result.segments.map {
            SpeakerTurn(speaker: $0.speakerId, start: TimeInterval($0.startTimeSeconds),
                        end: TimeInterval($0.endTimeSeconds))
        }
    }

    /// Writes `files` into one 16 kHz track, silence between them, in bounded chunks.
    static func join(_ files: [(url: URL, startOffset: TimeInterval)], into output: URL) async throws {
        try await FluidModelLoader.offPool {
            let writer = try TrackWriter(url: output)
            for file in files.sorted(by: { $0.startOffset < $1.startOffset }) {
                let pad = SpeechAudio.samples(file.startOffset) - writer.sampleCount
                if pad > 0 { try writer.write([Float](repeating: 0, count: pad)) }
                let input = try AVAudioFile(forReading: file.url)
                let resampler = try AudioResampler(inputFormat: input.processingFormat)
                guard let buffer = AVAudioPCMBuffer(pcmFormat: input.processingFormat, frameCapacity: 16_000) else {
                    throw SpeechError.failed("cannot allocate a read buffer")
                }
                while input.framePosition < input.length {
                    try input.read(into: buffer, frameCount: 16_000)
                    if buffer.frameLength == 0 { break }
                    try writer.write(try resampler.convert(buffer))
                }
            }
            try writer.close()
        }
    }
}

/// Carries a non-Sendable value across one hand-off it alone owns.
struct UncheckedBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

/// LS-EEND over the call track as it is captured. Its own clock counts the
/// samples fed; each feed's recording time maps turns back after a wake.
actor FluidLiveDiarizer: LiveDiarizing {
    private let diarizer: LSEENDDiarizer
    /// (seconds fed when a run began, its recording time), newest last.
    private var runs: [(fed: TimeInterval, recording: TimeInterval)] = []
    private var fed: TimeInterval = 0

    init(model: LSEENDModel) throws {
        diarizer = try LSEENDDiarizer(model: model)
    }

    func feed(_ samples: [Float], at time: TimeInterval) async throws -> [SpeakerTurn] {
        if let last = runs.last {
            let expected = last.recording + (fed - last.fed)
            if abs(expected - time) > 0.05 { runs.append((fed, time)) }
        } else {
            runs.append((fed, time))
        }
        fed += SpeechAudio.seconds(samples.count)
        _ = try diarizer.process(samples: samples, sourceSampleRate: SpeechAudio.sampleRate)
        var turns: [SpeakerTurn] = []
        for (index, speaker) in diarizer.timeline.speakers {
            for segment in speaker.finalizedSegments + speaker.tentativeSegments {
                let start = recordingTime(TimeInterval(segment.startTime))
                turns.append(SpeakerTurn(speaker: String(index), start: start,
                                         end: start + TimeInterval(segment.duration)))
            }
        }
        return turns.sorted { $0.start < $1.start }
    }

    private func recordingTime(_ fedTime: TimeInterval) -> TimeInterval {
        let run = runs.last { $0.fed <= fedTime } ?? runs.first ?? (0, 0)
        return run.recording + (fedTime - run.fed)
    }
}
