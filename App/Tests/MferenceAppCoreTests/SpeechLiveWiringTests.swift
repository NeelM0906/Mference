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
        let first = SystemAudioPermission(store: store, probe: { .granted })
        #expect(first.state == .notDetermined)
        try await first.request()
        #expect(first.state == .granted)
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

    @Test func nothingPlayingLeavesItForTheNextTap() async throws {
        // Granted is never recorded on timing: an undecided test tap is "asked".
        let store = InMemorySettingsStore()
        let permission = SystemAudioPermission(store: store, probe: { .undecided })
        try await permission.request()
        #expect(permission.stored == "asked")
        #expect(permission.state == .granted, "the next meeting may tap and settle it")
        permission.record(.denied)
        #expect(permission.state == .needsSystemSettings)
        permission.record(.undecided)
        #expect(permission.state == .needsSystemSettings)
    }

    @Test func openingSystemSettingsLetsTheNextTapCheckAgain() async throws {
        let store = InMemorySettingsStore()
        let permission = SystemAudioPermission(store: store, probe: { .denied })
        try await permission.request()
        #expect(permission.state == .needsSystemSettings)
        permission.systemSettingsOpened()
        #expect(permission.state == .granted && permission.stored == "asked")
        permission.record(.granted)
        permission.systemSettingsOpened()
        #expect(permission.stored == "granted")
    }

    @Test func theBrokerAsksThroughThePersistedAccess() async {
        let store = InMemorySettingsStore()
        let permission = SystemAudioPermission(store: store, probe: { .granted })
        let broker = LivePermissionBroker(presentReason: { _ in true }, systemAudio: permission.access)
        #expect(await broker.state(of: .systemAudio) == .notDetermined)
        #expect(await broker.request(.systemAudio) == .granted)
        #expect(store.object(forKey: SystemAudioPermission.key) as? String == "granted")
    }

    @Test func aTapIsJudgedByWhatItHears() {
        var verdict = TapAccessVerdict()
        // Silence while nothing plays says nothing.
        for _ in 0..<100 { verdict.tick(heardSound: false, silentSeconds: 0.5, othersPlaying: false) }
        #expect(verdict.result == .undecided)
        // A quiet stretch of a call is not a denial yet.
        for _ in 0..<20 { verdict.tick(heardSound: false, silentSeconds: 0.5, othersPlaying: true) }
        #expect(verdict.result == .undecided)
        for _ in 0..<10 { verdict.tick(heardSound: false, silentSeconds: 0.5, othersPlaying: true) }
        #expect(verdict.result == .denied, "15 s of digital silence while others play")
        verdict.tick(heardSound: true, silentSeconds: 0, othersPlaying: true)
        #expect(verdict.result == .denied, "a verdict is final")

        var heard = TapAccessVerdict()
        heard.tick(heardSound: false, silentSeconds: 0.5, othersPlaying: true)
        heard.tick(heardSound: true, silentSeconds: 0, othersPlaying: true)
        #expect(heard.result == .granted)
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
                     "plda-parameters.json", FluidModelLoader.liveDiarizerModel] {
            #expect(labels.contains { $0.contains("/\(name)") }, "speaker labels need \(name)")
        }
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
