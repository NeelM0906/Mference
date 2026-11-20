@preconcurrency import AVFoundation
import Foundation
@testable import MferenceAppCore
import Testing

/// The live pieces that can be checked without a model, device, tap or TCC
/// prompt (Notes spec 5.2, 6.3): the persisted call-audio permission, the
/// model folders the FluidAudio loader reads, the PLDA parameters, and the
/// call-track join before the final speaker pass.
@Suite struct SpeechLiveWiringTests {
    struct Refused: Error {}

    @Test func callAudioPermissionIsRememberedAcrossLaunches() async throws {
        let store = InMemorySettingsStore()
        let first = SystemAudioPermission(store: store, probe: { true })
        #expect(first.state == .notDetermined)
        try await first.request()
        #expect(first.state == .granted && first.hasHeardCallAudio)
        // A relaunch reads it back; no test tap runs.
        let relaunched = SystemAudioPermission(store: store, probe: { throw Refused() })
        #expect(relaunched.state == .granted)
        #expect(await relaunched.access.state() == .granted)
    }

    @Test func aCoreAudioErrorIsNotADenial() async throws {
        let store = InMemorySettingsStore()
        let failing = SystemAudioPermission(store: store, probe: { throw Refused() })
        await #expect(throws: Refused.self) { try await failing.request() }
        #expect(failing.state == .notDetermined, "nothing recorded")
    }

    @Test func silenceIsNeverRecordedAsADenial() async throws {
        // Review #228 pass 2: an in-person meeting (or a muted call app) is
        // silent too, so a silent test tap is only "asked": meetings keep
        // tapping, and it is not known to work until a tap hears something.
        let store = InMemorySettingsStore()
        let permission = SystemAudioPermission(store: store, probe: { false })
        try await permission.request()
        #expect(permission.stored == "asked")
        #expect(permission.state == .granted, "not asked again; each meeting taps")
        #expect(!permission.hasHeardCallAudio)
        permission.heardCallAudio()
        #expect(permission.stored == "granted" && permission.hasHeardCallAudio)
        // A later silent test tap never takes it back.
        try await permission.request()
        #expect(permission.stored == "granted")
    }

    @Test func anEarlierBuildsDenialReadsAsAsked() {
        let store = InMemorySettingsStore()
        store.set("denied", forKey: SystemAudioPermission.key)
        let permission = SystemAudioPermission(store: store, probe: { false })
        #expect(permission.state == .granted)
        #expect(!permission.hasHeardCallAudio)
    }

    @Test func theBrokerAsksThroughThePersistedAccess() async {
        let store = InMemorySettingsStore()
        let permission = SystemAudioPermission(store: store, probe: { true })
        let broker = LivePermissionBroker(presentReason: { _ in true }, systemAudio: permission.access)
        #expect(await broker.state(of: .systemAudio) == .notDetermined)
        #expect(await broker.request(.systemAudio) == .granted)
        #expect(store.object(forKey: SystemAudioPermission.key) as? String == "granted")
    }

    @Test func aTapHearsAnyNonZeroSample() throws {
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 2))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 480))
        buffer.frameLength = 480
        for channel in 0..<2 { for index in 0..<480 { buffer.floatChannelData![channel][index] = 0 } }
        #expect(!ProcessTapSource.hasSound(buffer), "a denied tap delivers exact zeros")
        buffer.floatChannelData![1][479] = 1e-6
        #expect(ProcessTapSource.hasSound(buffer), "the last sample of the second channel counts")
    }

    @Test func theLoaderReadsThePinnedFolders() throws {
        let loader = FluidModelLoader()
        let root = URL(fileURLWithPath: "/tmp/speech")
        let asr = try loader.folder(root, loader.speech, repoSuffix: "/fluid-speech-pro-coreml")
        #expect(asr.path == "/tmp/speech/fluid-speech-pro-coreml/\(SpeechModelPins.speech.sources[0].revision)")
        let installed = Set(SpeechModelPins.speech.sources.flatMap(\.files).map(\.installPath))
        for name in ["Preprocessor", "Encoder", "Decoder", "JointDecisionv3"] {
            #expect(installed.contains { $0.contains("/\(name).mlmodelc/") }, "the .v3 loader needs \(name)")
        }
        #expect(installed.contains { $0.hasSuffix("/parakeet_vocab.json") })
        #expect(installed.contains { $0.contains("/\(FluidModelLoader.vadModelName)/") })
        let labels = Set(SpeechModelPins.speakerLabels.sources.flatMap(\.files).map(\.installPath))
        for name in ["Segmentation.mlmodelc", "FBank.mlmodelc", "Embedding.mlmodelc", "PldaRho.mlmodelc",
                     "plda-parameters.json"] {
            #expect(labels.contains { $0.contains("/\(name)") }, "speaker labels need \(name)")
        }
    }

    /// An LS-EEND source with one compiled model at `model` (repo path).
    static func lseendPackage(_ model: String) -> SpeechModelPackage {
        let files = ["coremldata.bin", "metadata.json", "model.mil", "weights/weight.bin"].map { name in
            SpeechModelFile(repoPath: "\(model)/\(name)", installPath: "ls-eend-coreml/rev/\(model)/\(name)", size: 1,
                            sha256: String(repeating: "0", count: 64))
        }
        return SpeechModelPackage(id: .speakerLabels, sources: [
            SpeechModelSource(repo: "FluidInference/ls-eend-coreml", revision: "rev", license: "MIT", files: files)])
    }

    @Test func theLiveDiarizerFollowsFluidAudiosLayout() {
        // C-N1: FluidAudio 0.18.4 loads `optimized/ami/<step>/ls_eend_ami_<step>.mlmodelc`.
        let root = URL(fileURLWithPath: "/tmp/speech")
        var loader = FluidModelLoader()
        loader.speakerLabels = Self.lseendPackage("optimized/ami/100ms/ls_eend_ami_100ms.mlmodelc")
        #expect(loader.liveDiarizerModel(root)?.path
                == "/tmp/speech/ls-eend-coreml/rev/optimized/ami/100ms/ls_eend_ami_100ms.mlmodelc")
        loader.speakerLabels = Self.lseendPackage("optimized/ami/500ms/ls_eend_ami_500ms.mlmodelc")
        #expect(loader.liveDiarizerModel(root)?.lastPathComponent == "ls_eend_ami_500ms.mlmodelc")
        // The older export (no embedded config, other state tensors) is not used.
        loader.speakerLabels = Self.lseendPackage("AMI/ls_eend_ami_step.mlmodelc")
        #expect(loader.liveDiarizerModel(root) == nil)
    }

    @Test func thePinnedLiveDiarizerIsOneFluidAudioLoads() {
        // The pin (28ce1b1f) holds the older export; this marks it until the
        // coordinator repins, and fails once the repin lands so it is removed.
        withKnownIssue("LS-EEND pin predates FluidAudio 0.18.4's layout (C-N1 run 1)") {
            #expect(FluidModelLoader().liveDiarizerModel(URL(fileURLWithPath: "/tmp/speech")) != nil)
        }
    }

    @Test func theLiveDiarizerNeedsItsEmbeddedConfig() throws {
        let model = FileManager.default.temporaryDirectory.appending(path: "lseend-\(UUID().uuidString).mlmodelc")
        defer { try? FileManager.default.removeItem(at: model) }
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let metadata = model.appending(path: "metadata.json")
        func write(_ user: [String: String]) throws {
            try JSONSerialization.data(withJSONObject: [["userDefinedMetadata": user, "modelType": ["name": "MLModelType_mlProgram"]]])
                .write(to: metadata)
        }
        // No metadata.json at all.
        #expect(throws: SpeechError.self) { _ = try FluidModelLoader.liveDiarizerConfig(model) }
        // The older export: only coremltools' own keys.
        try write(["com.github.apple.coremltools.version": "8.3.0"])
        #expect(throws: SpeechError.self) { _ = try FluidModelLoader.liveDiarizerConfig(model) }
        // A config of another shape (the older sidecar's keys).
        try write(["config": #"{"max_speakers": 4, "sample_rate": 8000}"#])
        #expect(throws: SpeechError.self) { _ = try FluidModelLoader.liveDiarizerConfig(model) }
        // The config FluidAudio reads, snake-cased as exported.
        let config: [String: Any] = [
            "chunk_size": 100, "frame_duration_seconds": 0.1, "max_speakers": 4, "sample_rate": 8000, "max_nspks": 6,
            "hop_length": 80, "win_length": 200, "n_mels": 23, "context_size": 7, "subsampling": 10, "conv_delay": 9,
            "n_units": 256, "n_heads": 4, "enc_n_layers": 4, "dec_n_layers": 2, "conv_kernel_size": 15,
        ]
        let json = String(decoding: try JSONSerialization.data(withJSONObject: config), as: UTF8.self)
        try write(["config": json, "com.github.apple.coremltools.version": "9.0"])
        _ = try FluidModelLoader.liveDiarizerConfig(model)
    }

    @Test func pldaParametersDecode() throws {
        let floats: [Float] = [1.5, -2.25, 0.125]
        let data = floats.withUnsafeBytes { Data($0) }
        let json = ["tensors": ["psi": ["data_base64": data.base64EncodedString()]]]
        let url = FileManager.default.temporaryDirectory.appending(path: "plda-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        #expect(try FluidDiarizer.pldaPsi(url) == [1.5, -2.25, 0.125])
        try Data("{}".utf8).write(to: url)
        #expect(throws: SpeechError.self) { try FluidDiarizer.pldaPsi(url) }
    }

    @Test func callFilesJoinInRecordingTime() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "join-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        func track(_ name: String, seconds: TimeInterval) throws -> URL {
            let writer = try TrackWriter(url: folder.appending(path: name))
            try writer.write(FakeAudioSource.tone(seconds: seconds))
            try writer.close()
            return writer.url
        }
        // A call file, the Mac slept, a second call file from 5 s.
        let first = try track("call-1.caf", seconds: 2)
        let second = try track("call-2.caf", seconds: 1)
        let joined = folder.appending(path: "joined.caf")
        try await FluidDiarizer.join([(second, 5), (first, 0)], into: joined)
        let file = try AVAudioFile(forReading: joined)
        #expect(abs(Double(file.length) / 16_000 - 6) < 0.05, "silence fills 2-5 s")
    }
}
