@preconcurrency import AVFoundation
import Foundation

/// Converts capture buffers (any rate, any channel count, any sample
/// format) to the models' 16 kHz mono Float32 (spec 6.3). Every channel is
/// averaged into one, so a microphone on input 2 of an interface is heard.
/// One instance per source: the rate converter keeps state between buffers,
/// so there are no clicks at buffer edges.
final class AudioResampler: @unchecked Sendable {
    let inputFormat: AVAudioFormat
    /// Input → Float32, non-interleaved, same rate and channels (nil when it already is).
    private let toFloat: AVAudioConverter?
    private let floatFormat: AVAudioFormat
    /// Mono at the input rate → 16 kHz mono.
    private let toModelRate: AVAudioConverter
    private let monoFormat: AVAudioFormat
    static let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: SpeechAudio.sampleRate,
                                            channels: 1, interleaved: false)!

    init(inputFormat: AVAudioFormat) throws {
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0,
              let floatFormat = Self.floatFormat(like: inputFormat),
              let monoFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inputFormat.sampleRate,
                                             channels: 1, interleaved: false),
              let toModelRate = AVAudioConverter(from: monoFormat, to: Self.outputFormat) else {
            throw SpeechError.failed("no converter from \(inputFormat) to 16 kHz mono")
        }
        if inputFormat.commonFormat == .pcmFormatFloat32, !inputFormat.isInterleaved {
            toFloat = nil
        } else {
            guard let converter = AVAudioConverter(from: inputFormat, to: floatFormat) else {
                throw SpeechError.failed("no converter from \(inputFormat) to Float32")
            }
            toFloat = converter
        }
        self.inputFormat = inputFormat
        self.floatFormat = floatFormat
        self.monoFormat = monoFormat
        self.toModelRate = toModelRate
    }

    /// Float32, non-interleaved, same rate and channels. Over two channels
    /// a format needs a layout: the input's own, or discrete channels.
    static func floatFormat(like input: AVAudioFormat) -> AVAudioFormat? {
        if input.channelCount <= 2 {
            return AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: input.sampleRate,
                                 channels: input.channelCount, interleaved: false)
        }
        let layout = input.channelLayout
            ?? AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_DiscreteInOrder | AudioChannelLayoutTag(input.channelCount))
        guard let layout else { return nil }
        return AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: input.sampleRate, interleaved: false,
                             channelLayout: layout)
    }

    /// The buffer as 16 kHz mono samples. Called from one thread at a time
    /// (the capture callback).
    func convert(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        let floats = try floatBuffer(buffer)
        guard let mono = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: max(1, floats.frameLength)),
              let channels = floats.floatChannelData, let out = mono.floatChannelData?[0] else {
            throw SpeechError.failed("cannot allocate a conversion buffer")
        }
        let frames = Int(floats.frameLength)
        let count = Int(floats.format.channelCount)
        mono.frameLength = floats.frameLength
        let scale = 1 / Float(count)
        for frame in 0..<frames {
            var sum: Float = 0
            for channel in 0..<count { sum += channels[channel][frame] }
            out[frame] = sum * scale
        }
        return try resample(mono)
    }

    private func floatBuffer(_ buffer: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        guard let toFloat else { return buffer }
        guard let output = AVAudioPCMBuffer(pcmFormat: floatFormat, frameCapacity: max(1, buffer.frameLength)) else {
            throw SpeechError.failed("cannot allocate a conversion buffer")
        }
        try toFloat.convert(to: output, from: buffer)
        return output
    }

    private func resample(_ mono: AVAudioPCMBuffer) throws -> [Float] {
        let ratio = SpeechAudio.sampleRate / inputFormat.sampleRate
        let capacity = AVAudioFrameCount((Double(mono.frameLength) * ratio).rounded(.up)) + 32
        guard let output = AVAudioPCMBuffer(pcmFormat: Self.outputFormat, frameCapacity: capacity) else {
            throw SpeechError.failed("cannot allocate a conversion buffer")
        }
        var supplied = false
        var error: NSError?
        let status = toModelRate.convert(to: output, error: &error) { _, inputStatus in
            if supplied {
                inputStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            inputStatus.pointee = .haveData
            return mono
        }
        if status == .error { throw SpeechError.failed(error?.localizedDescription ?? "conversion failed") }
        guard let channel = output.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
    }
}
