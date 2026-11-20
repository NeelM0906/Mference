@preconcurrency import AVFoundation
import Foundation

/// Converts capture buffers (any rate, any channel count) to the models'
/// 16 kHz mono Float32 (spec 6.3). One instance per source: the converter
/// keeps state between buffers so there are no clicks at buffer edges.
final class AudioResampler: @unchecked Sendable {
    let inputFormat: AVAudioFormat
    private let converter: AVAudioConverter
    static let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: SpeechAudio.sampleRate,
                                            channels: 1, interleaved: false)!

    init(inputFormat: AVAudioFormat) throws {
        guard let converter = AVAudioConverter(from: inputFormat, to: Self.outputFormat) else {
            throw SpeechError.failed("no converter from \(inputFormat) to 16 kHz mono")
        }
        // Mix every channel down, not just the first one.
        if inputFormat.channelCount > 1 {
            converter.channelMap = [0]
            converter.downmix = true
        }
        self.inputFormat = inputFormat
        self.converter = converter
    }

    /// The buffer as 16 kHz mono samples. Called from one thread at a time
    /// (the capture callback).
    func convert(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        let ratio = SpeechAudio.sampleRate / inputFormat.sampleRate
        let capacity = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up)) + 32
        guard let output = AVAudioPCMBuffer(pcmFormat: Self.outputFormat, frameCapacity: capacity) else {
            throw SpeechError.failed("cannot allocate a conversion buffer")
        }
        var supplied = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            if supplied {
                inputStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            inputStatus.pointee = .haveData
            return buffer
        }
        if status == .error { throw SpeechError.failed(error?.localizedDescription ?? "conversion failed") }
        guard let channel = output.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
    }
}
