@preconcurrency import AVFoundation
import Foundation
import Synchronization
@testable import MferenceAppCore
import Testing

/// Recording files (Notes spec 6.3, 7): crash-safe CAF tracks, AAC on stop,
/// crash recovery, the storage check, and bounded file decoding. Fixture
/// PCM written by the tests; no microphone, tap or model.
@Suite struct SpeechRecordingTests {
    static func root() -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: "mference-recordings-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func aTrackIsReadableWhileStillOpen() throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "microphone-1.caf")
        let writer = try TrackWriter(url: url)
        try writer.write(FakeAudioSource.tone(seconds: 2))
        try writer.flush()
        // As a crash would leave it: open, data size unknown.
        let open = try AVAudioFile(forReading: url)
        #expect(open.processingFormat.sampleRate == 16_000)
        #expect(open.length == 32_000)
        try writer.write(FakeAudioSource.tone(seconds: 1))
        try writer.close()
        let closed = try AVAudioFile(forReading: url)
        #expect(closed.length == 48_000)
        #expect(abs(writer.duration - 3) < 1e-9)
    }

    @Test func repairSetsTheSizeOfAFileACrashLeftOpen() throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "call-1.caf")
        do {
            let writer = try TrackWriter(url: url)
            try writer.write(FakeAudioSource.tone(seconds: 1.5))
            try writer.flush()
            // No close: the app died. Add a torn half-sample too.
        }
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data([0x01]))
        try handle.close()
        let duration = try TrackWriter.repair(url)
        #expect(abs(duration - 1.5) < 1e-9)
        #expect(try AVAudioFile(forReading: url).length == 24_000)
        #expect(throws: SpeechError.self) { try TrackWriter.repair(root.appending(path: "missing.caf")) }
    }

    @Test func aMeetingWritesOneFilePerTrackPerRunThenAAC() async throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RecordingStore(root: root, availableBytes: { _ in 10 << 30 })
        let note = UUID()
        let session = try store.begin(noteID: note, startedAt: .now, keepAudio: true)
        try session.openTrack(.microphone, at: 0)
        try session.openTrack(.call, at: 0)
        try session.write(FakeAudioSource.tone(seconds: 2), to: .microphone)
        try session.write(FakeAudioSource.tone(seconds: 2), to: .call)
        // Sleep: files close; wake: new files at the recording time so far.
        try session.closeAll()
        try session.openTrack(.microphone, at: 2)
        try session.write(FakeAudioSource.tone(seconds: 1), to: .microphone)
        #expect(session.tracks.map(\.fileName) == ["microphone-1.caf", "call-1.caf", "microphone-2.caf"])
        #expect(session.files(of: .call).map(\.startOffset) == [0])
        let (tracks, failure) = await session.finish(keepAudio: true)
        #expect(failure == nil)
        #expect(tracks.map(\.fileName) == ["microphone-1.m4a", "call-1.m4a", "microphone-2.m4a"])
        #expect(tracks.map(\.startOffset) == [0, 0, 2])
        for track in tracks {
            let file = try AVAudioFile(forReading: store.directory(for: note).appending(path: track.fileName))
            #expect(abs(Double(file.length) / file.processingFormat.sampleRate - track.duration) < 0.1)
        }
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: store.directory(for: note).path)
        #expect(!leftovers.contains { $0.hasSuffix(".caf") })
        #expect(store.readManifest(note)?.state == .finished)
        #expect(store.totalSize() > 0)
        try store.delete(noteID: note)
        #expect(store.totalSize() == 0)
    }

    @Test func deleteAfterTranscribingKeepsNoAudio() async throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RecordingStore(root: root, availableBytes: { _ in 10 << 30 })
        let note = UUID()
        let session = try store.begin(noteID: note, startedAt: .now, keepAudio: false)
        try session.openTrack(.microphone, at: 0)
        try session.write(FakeAudioSource.tone(seconds: 1), to: .microphone)
        #expect(await session.finish(keepAudio: false).tracks.isEmpty)
        let files = try FileManager.default.contentsOfDirectory(atPath: store.directory(for: note).path)
        #expect(files == [RecordingStore.manifestName])
    }

    @Test func aMeetingLeftOpenIsRecoveredAtTheNextLaunch() async throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RecordingStore(root: root, availableBytes: { _ in 10 << 30 })
        let note = UUID()
        do {
            let session = try store.begin(noteID: note, startedAt: Date(timeIntervalSince1970: 1_000), keepAudio: true)
            try session.openTrack(.microphone, at: 0)
            try session.openTrack(.call, at: 0)
            try session.write(FakeAudioSource.tone(seconds: 2), to: .microphone, now: Date(timeIntervalSince1970: 1_002))
            try session.write(FakeAudioSource.tone(seconds: 1), to: .call, now: Date(timeIntervalSince1970: 1_002))
            try session.flush(now: Date(timeIntervalSince1970: 1_002))
            // The app quits here without closing anything.
        }
        let finished = UUID()
        let done = try store.begin(noteID: finished, startedAt: .now, keepAudio: true)
        _ = await done.finish(keepAudio: true)

        let recovered = await RecordingStore(root: root).recoverInterrupted()
        #expect(recovered.map(\.noteID) == [note], "only the meeting left open")
        let tracks = try #require(recovered.first?.tracks)
        #expect(tracks.map(\.fileName) == ["microphone-1.m4a", "call-1.m4a"])
        #expect(abs(tracks[0].duration - 2) < 1e-9 && abs(tracks[1].duration - 1) < 1e-9)
        #expect(recovered[0].endedAt >= Date(timeIntervalSince1970: 1_002))
        #expect(try AVAudioFile(forReading: store.directory(for: note).appending(path: "call-1.m4a")).length > 0)
        // Recovered once only.
        #expect(await store.recoverInterrupted().isEmpty)
    }

    @Test func quitClosesTheFilesAndTheNextLaunchConverts() async throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RecordingStore(root: root, availableBytes: { _ in 10 << 30 })
        let note = UUID()
        let session = try store.begin(noteID: note, startedAt: .now, keepAudio: true)
        try session.openTrack(.microphone, at: 0)
        try session.openTrack(.call, at: 0)
        try session.write(FakeAudioSource.tone(seconds: 1), to: .microphone)
        try session.write(FakeAudioSource.tone(seconds: 1), to: .call)
        // Quitting: nothing is converted now.
        let (tracks, failure) = await session.finish(keepAudio: true, stopEarly: { true })
        #expect(failure == nil)
        #expect(tracks.map(\.fileName) == ["microphone-1.m4a", "call-1.m4a"], "the names they will have")
        #expect(store.readManifest(note)?.state == .converting)
        let folder = store.directory(for: note)
        #expect(FileManager.default.fileExists(atPath: folder.appending(path: "call-1.caf").path))
        // Next launch: converted, finished, and not reported as interrupted.
        #expect(await store.recoverInterrupted().isEmpty)
        #expect(store.readManifest(note)?.state == .finished)
        #expect(store.readManifest(note)?.tracks.map(\.fileName) == ["microphone-1.m4a", "call-1.m4a"])
        #expect(try AVAudioFile(forReading: folder.appending(path: "call-1.m4a")).length > 0)
        #expect(!FileManager.default.fileExists(atPath: folder.appending(path: "call-1.caf").path))
    }

    @Test func theManifestFollowsEachConvertedTrack() async throws {
        // Review #223: dying between two conversions must not orphan the first .m4a.
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RecordingStore(root: root, availableBytes: { _ in 10 << 30 })
        let note = UUID()
        let session = try store.begin(noteID: note, startedAt: .now, keepAudio: true)
        try session.openTrack(.microphone, at: 0)
        try session.openTrack(.call, at: 0)
        try session.write(FakeAudioSource.tone(seconds: 1), to: .microphone)
        try session.write(FakeAudioSource.tone(seconds: 1), to: .call)
        let calls = Counter()
        // Stop after the first track, as a quit mid-conversion would.
        _ = await session.finish(keepAudio: true, stopEarly: { calls.next() > 0 })
        let manifest = try #require(store.readManifest(note))
        #expect(manifest.tracks.map(\.fileName) == ["microphone-1.m4a", "call-1.caf"])
        let folder = store.directory(for: note)
        for track in manifest.tracks { #expect(FileManager.default.fileExists(atPath: folder.appending(path: track.fileName).path)) }
    }

    @Test func recordingRefusesUnderOneGigabyteFree() throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RecordingStore(root: root, availableBytes: { _ in 900 << 20 })
        #expect(throws: SpeechError.lowDiskSpace(availableBytes: 900 << 20)) {
            _ = try store.begin(noteID: UUID(), startedAt: .now, keepAudio: true)
        }
    }

    @Test func decodesAFileInBoundedChunks() async throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        // A 44.1 kHz stereo AAC file, as people import them.
        let url = root.appending(path: "interview.m4a")
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 2))
        do {
            let file = try AVAudioFile(forWriting: url, settings: [AVFormatIDKey: kAudioFormatMPEG4AAC,
                                                                   AVSampleRateKey: 44_100, AVNumberOfChannelsKey: 2])
            let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 44_100))
            buffer.frameLength = 44_100
            for second in 0..<5 {
                for channel in 0..<2 {
                    for index in 0..<44_100 {
                        buffer.floatChannelData![channel][index] = second % 2 == 0
                            ? 0.3 * sin(2 * .pi * 220 * Float(index) / 44_100) : 0
                    }
                }
                try file.write(from: buffer)
            }
        }
        let decoder = try await FileAudioDecoder(url: url)
        #expect(abs(decoder.duration - 5) < 0.1)
        var total = 0
        var largest = 0
        while let chunk = try decoder.next(maxSamples: 8_000) {
            total += chunk.count
            largest = max(largest, chunk.count)
        }
        #expect(largest <= 8_000)
        #expect(abs(total - 80_000) < 2_000, "about 5 s at 16 kHz")
    }

    @Test func aFileWithoutAudioIsRefused() async throws {
        let root = Self.root()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "notes.m4a")
        try Data("not audio".utf8).write(to: url)
        await #expect(throws: SpeechError.unsupportedFile("notes.m4a")) { _ = try await FileAudioDecoder(url: url) }
    }
}

final class Counter: Sendable {
    private let value = Mutex(0)
    /// The count before this call.
    func next() -> Int { value.withLock { v in defer { v += 1 }; return v } }
}
