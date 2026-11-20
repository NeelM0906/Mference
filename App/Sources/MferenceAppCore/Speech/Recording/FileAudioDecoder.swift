@preconcurrency import AVFoundation
import CoreMedia
import Foundation

/// Decodes an imported audio or video file to 16 kHz mono Float32 a bounded
/// chunk at a time (spec 6.3: never the whole file in memory). AVFoundation
/// resamples and mixes down on the fly.
final class FileAudioDecoder: @unchecked Sendable {
    let duration: TimeInterval
    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    private var carry: [Float] = []
    private var finished = false

    /// Throws `SpeechError.unsupportedFile` when there is no audio track AVFoundation can read.
    init(url: URL) async throws {
        let asset = AVURLAsset(url: url)
        let tracks: [AVAssetTrack]
        let duration: CMTime
        do {
            tracks = try await asset.loadTracks(withMediaType: .audio)
            duration = try await asset.load(.duration)
        } catch {
            throw SpeechError.unsupportedFile(url.lastPathComponent)
        }
        guard let track = tracks.first else { throw SpeechError.unsupportedFile(url.lastPathComponent) }
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM,
                                       AVSampleRateKey: SpeechAudio.sampleRate,
                                       AVNumberOfChannelsKey: 1,
                                       AVLinearPCMBitDepthKey: 32,
                                       AVLinearPCMIsFloatKey: true,
                                       AVLinearPCMIsBigEndianKey: false,
                                       AVLinearPCMIsNonInterleaved: false]
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw SpeechError.unsupportedFile(url.lastPathComponent)
        }
        output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw SpeechError.unsupportedFile(url.lastPathComponent) }
        reader.add(output)
        guard reader.startReading() else { throw SpeechError.unsupportedFile(url.lastPathComponent) }
        self.duration = max(0, duration.seconds.isFinite ? duration.seconds : 0)
    }

    deinit { reader.cancelReading() }

    /// The next `maxSamples` (fewer at the end); nil once the file is done.
    func next(maxSamples: Int) throws -> [Float]? {
        while carry.count < maxSamples, !finished {
            guard let buffer = output.copyNextSampleBuffer() else {
                finished = true
                if reader.status == .failed {
                    throw SpeechError.failed(reader.error?.localizedDescription ?? "decoding failed")
                }
                break
            }
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var samples = [Float](repeating: 0, count: length / MemoryLayout<Float>.size)
            let status = samples.withUnsafeMutableBytes { raw in
                CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: raw.count, destination: raw.baseAddress!)
            }
            guard status == kCMBlockBufferNoErr else { throw SpeechError.failed("cannot read decoded audio") }
            carry += samples
        }
        guard !carry.isEmpty else { return nil }
        let count = min(maxSamples, carry.count)
        let chunk = Array(carry.prefix(count))
        carry.removeFirst(count)
        return chunk
    }
}
