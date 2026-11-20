import AVFoundation
import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import Testing

/// The real player over generated tones (no audio device is used: nothing
/// plays; the composition is built and inspected).
@MainActor
@Suite(.serialized) struct CompositionPlaybackTests {
    /// Writes a sine tone as 16 kHz mono CAF.
    static func writeTone(_ url: URL, seconds: Double, frequency: Double) throws {
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1))
        let frames = AVAudioFrameCount(seconds * 16_000)
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames))
        buffer.frameLength = frames
        let samples = try #require(buffer.floatChannelData?[0])
        for index in 0..<Int(frames) {
            samples[index] = Float(0.2 * sin(2 * .pi * frequency * Double(index) / 16_000))
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }

    func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: "mference-playback-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func tracksPlayTogetherAtTheirOffsetsAndMissingFilesAreSkipped() async throws {
        let folder = try directory()
        defer { try? FileManager.default.removeItem(at: folder) }
        try Self.writeTone(folder.appending(path: "mic-1.caf"), seconds: 1, frequency: 440)
        try Self.writeTone(folder.appending(path: "call-1.caf"), seconds: 1, frequency: 660)
        let playback = CompositionPlayback()
        let length = try await playback.load(tracks: [
            RecordedTrack(fileName: "mic-1.caf", kind: .microphone, startOffset: 0, duration: 1),
            RecordedTrack(fileName: "call-1.caf", kind: .call, startOffset: 0.5, duration: 1),
            RecordedTrack(fileName: "mic-2.caf", kind: .microphone, startOffset: 3, duration: 1), // never written
        ], directory: folder)
        #expect(abs(length - 1.5) < 0.05, "the call track starts at 0.5 s; the missing file is skipped")
        #expect(abs(playback.itemDuration - length) < 0.01)
        playback.rate = 1.5
        #expect(playback.rate == 1.5)
        #expect(playback.currentTime == 0)
    }

    @Test func noFileAtAllIsNoAudio() async throws {
        let folder = try directory()
        defer { try? FileManager.default.removeItem(at: folder) }
        let playback = CompositionPlayback()
        await #expect(throws: PlaybackError.noAudio) {
            _ = try await playback.load(tracks: [RecordedTrack(fileName: "gone.m4a", kind: .file, startOffset: 0,
                                                               duration: 10)], directory: folder)
        }
        let player = RecordingPlayerModel(tracks: [RecordedTrack(fileName: "gone.m4a", kind: .file, startOffset: 0,
                                                                 duration: 10)],
                                          directory: folder, duration: 10, playback: playback)
        await player.load()
        #expect(player.hasAudio == false && player.loadError == nil, "reads as deleted, not as broken")
    }
}

extension CompositionPlaybackTests {
    @Test func seekMovesThePositionAndTheEndIsReported() async throws {
        let folder = try directory()
        defer { try? FileManager.default.removeItem(at: folder) }
        try Self.writeTone(folder.appending(path: "mic-1.caf"), seconds: 1, frequency: 440)
        let playback = CompositionPlayback()
        playback.isMuted = true
        _ = try await playback.load(tracks: [RecordedTrack(fileName: "mic-1.caf", kind: .microphone,
                                                           startOffset: 0, duration: 1)], directory: folder)
        #expect(await playback.waitUntilReady())
        playback.seek(to: 0.5)
        await waitUntil("the seek") { abs(playback.currentTime - 0.5) <= 0.02 }
        #expect(abs(playback.currentTime - 0.5) < 0.02, "seek lands where asked")
        // The real end of the item: play the last 0.3 s silently at 2×.
        playback.seek(to: 0.7)
        var ended = false
        playback.onEnded = { ended = true }
        playback.rate = 2
        playback.play()
        await waitUntil("the end of the item") { ended }
        #expect(ended, "AVPlayerItemDidPlayToEndTime reaches onEnded")
        playback.pause()
    }
}
