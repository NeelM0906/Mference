import Foundation
@testable import MferenceAppCore
import MferenceEngine
import Synchronization
import Testing

/// The speech package installer (Notes spec 4.7, 7): multi-file pinned
/// downloads verified file by file, resumable, refusing in Offline mode.
/// Files come from a loopback server; nothing reaches the network and no
/// model is loaded.
@Suite(.serialized) struct SpeechInstallerTests {
    struct Fixture {
        let server: LoopbackHTTPServer
        let root: URL
        let package: SpeechModelPackage
        let bodies: [String: Data]

        /// Three files across two sources, one inside an `.mlmodelc` folder.
        static func make(holdAfter: Int? = nil, hold: FakeEngineLatch? = nil,
                         corrupt: [String: Data] = [:]) async throws -> Fixture {
            let bodies = ["Encoder.mlmodelc/weights/weight.bin": fakeModelBytes(300_000, seed: 1),
                          "parakeet_vocab.json": fakeModelBytes(2_000, seed: 2),
                          "vad.mlmodelc/model.mil": fakeModelBytes(5_000, seed: 3)]
            var routes: [String: LoopbackHTTPServer.Route] = [:]
            for (path, body) in bodies {
                var route = LoopbackHTTPServer.Route(body: corrupt[path] ?? body)
                if path.hasSuffix("weight.bin") {
                    route.holdAfter = holdAfter
                    route.hold = hold
                }
                routes["/" + path] = route
            }
            let server = try await LoopbackHTTPServer(routes: routes)
            func file(_ repo: String, _ path: String) -> SpeechModelFile {
                SpeechModelFile(repoPath: path, installPath: "\(repo)/rev1/\(path)",
                                size: Int64(bodies[path]!.count), sha256: sha256Hex(bodies[path]!))
            }
            let package = SpeechModelPackage(id: .speech, sources: [
                SpeechModelSource(repo: "test/asr", revision: "rev1", license: "CC BY 4.0",
                                  files: [file("asr", "Encoder.mlmodelc/weights/weight.bin"),
                                          file("asr", "parakeet_vocab.json")]),
                SpeechModelSource(repo: "test/vad", revision: "rev1", license: "MIT",
                                  files: [file("vad", "vad.mlmodelc/model.mil")]),
            ])
            return Fixture(server: server, root: temporaryDirectory(), package: package, bodies: bodies)
        }

        func installer(offline: Bool = false, available: UInt64 = 1 << 40) -> SpeechModelInstaller {
            let server = server
            return SpeechModelInstaller(root: root, packages: [.speech: package], allowedHosts: ["127.0.0.1"],
                                        policy: .init(isOffline: { offline }, availableBytes: { _ in available }),
                                        urlFor: { _, file in server.url("/" + file.repoPath) })
        }
    }

    @Test func installsEveryFileVerifiedAtItsInstallPath() async throws {
        let fixture = try await Fixture.make()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let installer = fixture.installer()
        #expect(!installer.isInstalled(.speech))
        let reports = ProgressLog()
        try await installer.install(.speech) { reports.append($0) }
        #expect(installer.isInstalled(.speech))
        for (path, body) in fixture.bodies {
            let repo = path.hasPrefix("vad") ? "vad" : "asr"
            #expect(try Data(contentsOf: fixture.root.appending(path: "\(repo)/rev1/\(path)")) == body)
        }
        #expect(reports.values.last == 1)
        #expect(reports.values == reports.values.sorted(), "progress only goes forward")
        #expect(installer.directory(of: fixture.package.sources[0]).path.hasSuffix("asr/rev1"))
        // Installed: a second install fetches nothing.
        let before = fixture.server.requests.count
        try await installer.install(.speech)
        #expect(fixture.server.requests.count == before)
    }

    @Test func aSizeMismatchDeletesTheFileAndFails() async throws {
        let fixture = try await Fixture.make(corrupt: ["parakeet_vocab.json": fakeModelBytes(1_999, seed: 2)])
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let installer = fixture.installer()
        await #expect(throws: SpeechError.verificationFailed(file: "parakeet_vocab.json")) {
            try await installer.install(.speech)
        }
        let vocab = fixture.root.appending(path: "asr/rev1/parakeet_vocab.json")
        #expect(!FileManager.default.fileExists(atPath: vocab.path))
        #expect(!FileManager.default.fileExists(atPath: vocab.appendingPathExtension("part").path))
        #expect(!installer.isInstalled(.speech))
    }

    @Test func aChecksumMismatchDeletesTheFileAndFails() async throws {
        let fixture = try await Fixture.make(corrupt: ["vad.mlmodelc/model.mil": fakeModelBytes(5_000, seed: 99)])
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let installer = fixture.installer()
        await #expect(throws: SpeechError.verificationFailed(file: "vad.mlmodelc/model.mil")) {
            try await installer.install(.speech)
        }
        let model = fixture.root.appending(path: "vad/rev1/vad.mlmodelc/model.mil")
        #expect(!FileManager.default.fileExists(atPath: model.path))
        #expect(!FileManager.default.fileExists(atPath: model.appendingPathExtension("part").path))
        #expect(!installer.isInstalled(.speech))
        // The files that did verify stay, so a retry fetches only the rest.
        #expect(FileManager.default.fileExists(atPath: fixture.root.appending(path: "asr/rev1/parakeet_vocab.json").path))
    }

    @Test func pauseKeepsTheBytesAndAResumeAsksForTheRest() async throws {
        let hold = FakeEngineLatch()
        let fixture = try await Fixture.make(holdAfter: 120_000, hold: hold)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let installer = fixture.installer()
        let part = fixture.root.appending(path: "asr/rev1/Encoder.mlmodelc/weights/weight.bin.part")
        let first = Task { try await installer.install(.speech) }
        let deadline = ContinuousClock.now + .seconds(10)
        while DownloadManager.fileSize(part) < 120_000, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(DownloadManager.fileSize(part) == 120_000)
        await installer.pause(.speech)
        await #expect(throws: CancellationError.self) { try await first.value }
        hold.open()
        #expect(installer.progress(.speech) > 0.3 && installer.progress(.speech) < 1)

        // A relaunch continues from the partial file.
        let relaunched = fixture.installer()
        try await relaunched.install(.speech)
        #expect(relaunched.isInstalled(.speech))
        #expect(fixture.server.requests.contains { $0.path.hasSuffix("weight.bin") && $0.rangeStart == 120_000 })
        #expect(try Data(contentsOf: fixture.root.appending(path: "asr/rev1/Encoder.mlmodelc/weights/weight.bin"))
                == fixture.bodies["Encoder.mlmodelc/weights/weight.bin"])
    }

    @Test func anInstallRightAfterPauseStartsAfresh() async throws {
        // Review #219-1: an install arriving while a paused one winds down must not join it.
        let hold = FakeEngineLatch()
        let fixture = try await Fixture.make(holdAfter: 60_000, hold: hold)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let installer = fixture.installer()
        let part = fixture.root.appending(path: "asr/rev1/Encoder.mlmodelc/weights/weight.bin.part")
        let first = Task { try await installer.install(.speech) }
        let deadline = ContinuousClock.now + .seconds(10)
        while DownloadManager.fileSize(part) < 60_000, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        await installer.pause(.speech)
        let reports = ProgressLog()
        let second = Task { try await installer.install(.speech) { reports.append($0) } }
        hold.open()
        await #expect(throws: CancellationError.self) { try await first.value }
        try await second.value
        #expect(installer.isInstalled(.speech))
        #expect(reports.values.last == 1, "the new caller hears its own progress")
    }

    @Test func cancelDeletesThePartialDownload() async throws {
        let hold = FakeEngineLatch()
        let fixture = try await Fixture.make(holdAfter: 50_000, hold: hold)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let installer = fixture.installer()
        let part = fixture.root.appending(path: "asr/rev1/Encoder.mlmodelc/weights/weight.bin.part")
        let running = Task { try await installer.install(.speech) }
        let deadline = ContinuousClock.now + .seconds(10)
        while DownloadManager.fileSize(part) < 50_000, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        await installer.cancel(.speech)
        hold.open()
        _ = try? await running.value
        #expect(!FileManager.default.fileExists(atPath: fixture.root.appending(path: "asr").path))
        #expect(installer.bytesOnDisk(.speech) == 0)
    }

    @Test func offlineModeAndLowDiskRefuseBeforeAnyRequest() async throws {
        let fixture = try await Fixture.make()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        await #expect(throws: SpeechError.offline) { try await fixture.installer(offline: true).install(.speech) }
        await #expect(throws: SpeechError.lowDiskSpace(availableBytes: 1_000)) {
            try await fixture.installer(available: 1_000).install(.speech)
        }
        #expect(fixture.server.requests.isEmpty)
    }

    @Test func removeDeletesAnInstalledPackage() async throws {
        let fixture = try await Fixture.make()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let installer = fixture.installer()
        try await installer.install(.speech)
        try await installer.remove(.speech)
        #expect(!installer.isInstalled(.speech))
        #expect(!FileManager.default.fileExists(atPath: fixture.root.appending(path: "asr").path))
        #expect(!FileManager.default.fileExists(atPath: fixture.root.appending(path: "vad").path))
    }

    @Test func aFileChangedOnDiskIsNoLongerInstalled() async throws {
        let fixture = try await Fixture.make()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let installer = fixture.installer()
        try await installer.install(.speech)
        try Data("x".utf8).write(to: fixture.root.appending(path: "asr/rev1/parakeet_vocab.json"))
        #expect(!installer.isInstalled(.speech))
        try await installer.install(.speech)
        #expect(installer.isInstalled(.speech))
    }

    @Test func thePinnedPackagesInstallUnderRepoAndRevision() {
        let installer = SpeechModelInstaller(root: URL(fileURLWithPath: "/tmp/speech-root"))
        let pro = SpeechModelPins.speech.sources[0]
        #expect(installer.directory(of: pro).path == "/tmp/speech-root/fluid-speech-pro-coreml/\(pro.revision)")
        for file in pro.files {
            #expect(installer.destination(of: file).path.hasPrefix(installer.directory(of: pro).path + "/"))
        }
    }
}

final class ProgressLog: Sendable {
    private let log = Mutex<[Double]>([])
    var values: [Double] { log.withLock { $0 } }
    func append(_ value: Double) { log.withLock { $0.append(value) } }
}
