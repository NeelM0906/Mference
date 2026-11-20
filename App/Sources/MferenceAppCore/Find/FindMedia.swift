import AVFoundation
import CoreGraphics
import Foundation
import ImageIO

/// Which of the model's encoders a passage's vector comes from.
public enum FindModality: Int, Sendable, Hashable, Codable {
    case text = 0
    /// A photo, a screenshot or a video frame (the vision encoder).
    case image = 1
    /// A stretch of sound (the audio encoder).
    case audio = 2
}

/// One piece of a photo, video or audio file that gets its own vector: the
/// whole image, a video frame at `start`, or the sound from `start` to `end`.
public struct FindMediaPiece: Sendable, Hashable, Codable {
    public var modality: FindModality
    /// Seconds into the file (nil for a photo).
    public var start: Double?
    public var end: Double?
    /// Vision tokens for an image (`FindMedia.photoTokens` and friends).
    public var tokens: Int
    /// The piece already decoded for the encoders while it was read (a
    /// `FindMediaData` file in `FindMediaFolder.ready`), so learning need not
    /// decode it again.
    public var prepared: String?

    public init(modality: FindModality, start: Double? = nil, end: Double? = nil, tokens: Int = 0,
                prepared: String? = nil) {
        self.modality = modality
        self.start = start
        self.end = end
        self.tokens = tokens
        self.prepared = prepared
    }

    /// Where in the file, as results show it: "0:42" for a frame,
    /// "1:15–1:45" for sound, nil for a photo.
    public var location: String? {
        guard let start else { return nil }
        guard modality == .audio, let end else { return FindMedia.clock(start) }
        return "\(FindMedia.clock(start))–\(FindMedia.clock(end))"
    }
}

/// Decoded media, ready for the encoders. Written to a temporary file by the
/// reader (the helper process) and read back by the app.
public enum FindMediaData: Sendable, Hashable {
    /// Packed RGB at the size the vision encoder takes.
    case image(width: Int, height: Int, rgb: [UInt8])
    /// Mono samples at `FindMedia.sampleRate`.
    case audio(samples: [Float])

    private static let imageMagic: UInt32 = 0x3149_4D46   // "FMI1"
    private static let audioMagic: UInt32 = 0x3141_4D46   // "FMA1"

    public func write(to url: URL) throws {
        var data = Data()
        func append(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        switch self {
        case .image(let width, let height, let rgb):
            data.reserveCapacity(12 + rgb.count)
            append(Self.imageMagic)
            append(UInt32(width))
            append(UInt32(height))
            data.append(contentsOf: rgb)
        case .audio(let samples):
            data.reserveCapacity(8 + samples.count * 4)
            append(Self.audioMagic)
            append(UInt32(samples.count))
            samples.withUnsafeBytes { data.append(contentsOf: $0) }
        }
        try data.write(to: url)
    }

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        func word(_ offset: Int) -> UInt32? {
            guard data.count >= offset + 4 else { return nil }
            return data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
        }
        switch word(0) {
        case Self.imageMagic?:
            guard let width = word(4).map(Int.init), let height = word(8).map(Int.init),
                  data.count == 12 + width * height * 3 else { throw CocoaError(.fileReadCorruptFile) }
            self = .image(width: width, height: height, rgb: [UInt8](data.dropFirst(12)))
        case Self.audioMagic?:
            guard let count = word(4).map(Int.init), data.count == 8 + count * 4 else {
                throw CocoaError(.fileReadCorruptFile)
            }
            let samples = data.dropFirst(8).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            self = .audio(samples: samples)
        default:
            throw CocoaError(.fileReadCorruptFile)
        }
    }
}

/// Reads photos, video and audio for meaning search (pipeline board, media
/// path), the way DigUp does with the same model: a photo is one vector; a
/// video is a frame every few seconds with near-duplicate frames dropped,
/// plus its soundtrack; audio is 30-second windows with silence skipped.
///
/// `plan` runs when a file is read and says which pieces it has; `prepare`
/// runs when they are learned and decodes them. Both are synchronous and run
/// in the reading helper process, so a damaged file cannot take the app down.
public enum FindMedia {
    /// Vision tokens per kind of picture (more tokens see more detail).
    public static let photoTokens = 140
    public static let screenshotTokens = 280
    public static let frameTokens = 70
    /// Seconds between sampled video frames, at least.
    public static let frameInterval = 3.0
    /// Frames sampled per video at most (long videos are sampled sparser).
    public static let maxFrames = 600
    /// The audio encoder takes 30 s; windows overlap by 5 s.
    public static let audioWindow = 30.0
    public static let audioHop = 25.0
    /// Windows per file at most (long recordings are sampled sparser).
    public static let maxWindows = 400
    /// Sound quieter than this (RMS of full scale) is silence.
    public static let silence: Float = 0.004
    /// 64-bit picture hashes this close are the same shot.
    public static let sameShotBits = 6
    public static let sampleRate = 16_000.0

    /// How media is sampled, for the index's fingerprint: a change reads
    /// photos, video and audio again.
    public static var fingerprint: String {
        "p\(photoTokens)s\(screenshotTokens)f\(frameTokens)/\(frameInterval)a\(audioWindow)/\(audioHop)"
    }

    /// Video and audio are sampled, so large files are fine up to this.
    public static let maxFileBytes: Int64 = 32 << 30

    // MARK: Plan

    /// The pieces of `url`, as passages without text.
    public static func plan(_ url: URL, kind: FindKind) -> FindExtraction {
        switch kind {
        case .image:
            return .text([.init(location: nil, text: "", media: FindMediaPiece(modality: .image,
                                                                               tokens: imageTokens(for: url)))])
        case .video:
            do {
                let asset = AVURLAsset(url: url)
                let (duration, videoTracks, audioTracks) = try blocking {
                    (try await asset.load(.duration).seconds,
                     try await asset.loadTracks(withMediaType: .video).count,
                     try await asset.loadTracks(withMediaType: .audio).count)
                }
                guard duration.isFinite, duration > 0 else { return .skipped(reason: "Not a playable video.") }
                var passages: [FindExtraction.Passage] = []
                if videoTracks > 0 {
                    passages += shots(of: asset, duration: duration).map {
                        .init(location: nil, text: "", media: FindMediaPiece(modality: .image, start: $0,
                                                                             tokens: frameTokens))
                    }
                }
                if audioTracks > 0 {
                    passages += windows(of: asset, duration: duration).map {
                        .init(location: nil, text: "", media: FindMediaPiece(modality: .audio, start: $0.start,
                                                                             end: $0.end))
                    }
                }
                return .text(passages.map { passage in
                    var passage = passage
                    passage.location = passage.media?.location
                    return passage
                })
            } catch {
                return .skipped(reason: "Not a playable video.")
            }
        case .audio:
            do {
                let asset = AVURLAsset(url: url)
                let duration = try blocking { try await asset.load(.duration).seconds }
                guard duration.isFinite, duration > 0 else { return .skipped(reason: "Not playable audio.") }
                return .text(windows(of: asset, duration: duration).map {
                    let piece = FindMediaPiece(modality: .audio, start: $0.start, end: $0.end)
                    return .init(location: piece.location, text: "", media: piece)
                })
            } catch {
                return .skipped(reason: "Not playable audio.")
            }
        default:
            return .text([])
        }
    }

    /// Screenshots get more detail than photos: their text and layout matter.
    static func imageTokens(for url: URL) -> Int {
        let name = url.lastPathComponent.lowercased()
        return name.contains("screenshot") || name.contains("screen shot") || name.hasPrefix("cleanshot")
            ? screenshotTokens : photoTokens
    }

    /// Times to sample (seconds), every `frameInterval` or sparser.
    static func frameTimes(duration: Double) -> [Double] {
        let step = max(frameInterval, duration / Double(maxFrames))
        return Array(stride(from: 0, to: duration, by: step))
    }

    /// The first frame of each shot: a frame every few seconds, minus the
    /// ones that look like the frame kept before them. Times are the frames'
    /// actual times, so `prepare` decodes the same frames.
    static func shots(of asset: AVURLAsset, duration: Double) -> [Double] {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 64, height: 64)
        let tolerance = CMTime(seconds: 0.5, preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = tolerance
        generator.requestedTimeToleranceAfter = tolerance
        let times = frameTimes(duration: duration).map { CMTime(seconds: $0, preferredTimescale: 600) }
        let box = Unchecked(generator)
        let frames: [(Double, UInt64)] = (try? blocking {
            var frames: [(Double, UInt64)] = []
            for await result in box.value.images(for: times) {
                guard let image = try? result.image, let hash = averageHash(image) else { continue }
                frames.append(((try? result.actualTime.seconds) ?? result.requestedTime.seconds, hash))
            }
            return frames
        }) ?? []
        var kept: [Double] = []
        var last: UInt64?
        for (time, hash) in frames.sorted(by: { $0.0 < $1.0 }) {
            if let last, (last ^ hash).nonzeroBitCount <= sameShotBits { continue }
            if let previous = kept.last, abs(previous - time) < 0.01 { continue }
            kept.append(max(0, time))
            last = hash
        }
        return kept
    }

    /// 8 × 8 average hash: one bit per cell, set when it is brighter than
    /// the picture's mean.
    static func averageHash(_ image: CGImage) -> UInt64? {
        var pixels = [UInt8](repeating: 0, count: 64)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: 8, height: 8, bitsPerComponent: 8,
                                          bytesPerRow: 8, space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            context.interpolationQuality = .medium
            context.draw(image, in: CGRect(x: 0, y: 0, width: 8, height: 8))
            return true
        }
        guard drawn else { return nil }
        let mean = pixels.reduce(0) { $0 + Int($1) } / 64
        var hash: UInt64 = 0
        for (index, value) in pixels.enumerated() where Int(value) > mean {
            hash |= 1 << UInt64(index)
        }
        return hash
    }

    /// Window starts and ends (seconds) for a recording.
    static func windowRanges(duration: Double) -> [(start: Double, end: Double)] {
        let hop = max(audioHop, (duration - audioWindow) / Double(maxWindows - 1))
        var ranges: [(Double, Double)] = []
        var start = 0.0
        while start < duration {
            let end = min(start + audioWindow, duration)
            // A last sliver is already covered by the window before it.
            if end - start >= 2 || ranges.isEmpty { ranges.append((start, end)) }
            if end >= duration { break }
            start += hop
        }
        return ranges
    }

    /// The windows of a recording that are not silent. Reads the sound once,
    /// at 16 kHz mono, keeping only a loudness per second.
    static func windows(of asset: AVURLAsset, duration: Double) -> [(start: Double, end: Double)] {
        let ranges = windowRanges(duration: duration)
        guard let loudness = try? secondLoudness(of: asset) else { return ranges }
        return ranges.filter { range in
            let first = Int(range.start), last = max(first, min(loudness.count - 1, Int(range.end.rounded(.up)) - 1))
            guard first < loudness.count else { return false }
            return loudness[first...last].contains { $0 >= silence }
        }
    }

    /// RMS loudness of each second of the sound.
    static func secondLoudness(of asset: AVURLAsset) throws -> [Float] {
        var loudness: [Float] = []
        var sum: Float = 0
        var count = 0
        let perSecond = Int(sampleRate)
        try readSamples(of: asset, range: nil) { samples in
            for sample in samples {
                sum += sample * sample
                count += 1
                if count == perSecond {
                    loudness.append((sum / Float(count)).squareRoot())
                    sum = 0
                    count = 0
                }
            }
        }
        if count > 0 { loudness.append((sum / Float(count)).squareRoot()) }
        return loudness
    }

    // MARK: Prepare

    /// Decodes `pieces` of `url` into temporary files for the encoders, in
    /// order; nil where a piece could not be decoded.
    public static func prepare(_ url: URL, kind: FindKind, pieces: [FindMediaPiece], into folder: URL) -> [URL?] {
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var datas = [FindMediaData?](repeating: nil, count: pieces.count)
        if kind == .image {
            for (index, piece) in pieces.enumerated() where piece.modality == .image {
                datas[index] = photo(url, tokens: piece.tokens)
            }
        } else {
            let asset = AVURLAsset(url: url)
            let frames = pieces.enumerated().filter { $0.element.modality == .image && $0.element.start != nil }
            if !frames.isEmpty {
                let decoded = videoFrames(asset, at: frames.map { $0.element.start! }, tokens: frames[0].element.tokens)
                for (offset, item) in frames.enumerated() { datas[item.offset] = decoded[offset] }
            }
            for (index, piece) in pieces.enumerated() where piece.modality == .audio {
                let start = piece.start ?? 0
                let end = piece.end ?? start + audioWindow
                var samples: [Float] = []
                let range = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 600),
                                        end: CMTime(seconds: min(end, start + audioWindow), preferredTimescale: 600))
                try? readSamples(of: asset, range: range) { samples.append(contentsOf: $0) }
                if !samples.isEmpty { datas[index] = .audio(samples: Array(samples.prefix(Int(audioWindow * sampleRate)))) }
            }
        }
        return datas.map { data in
            guard let data else { return nil }
            let file = folder.appending(path: "\(UUID().uuidString).fmd")
            return (try? data.write(to: file)) == nil ? nil : file
        }
    }

    /// A photo at the size for `tokens`.
    static func photo(_ url: URL, tokens: Int) -> FindMediaData? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: 2_048,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        return rgb(image, tokens: tokens)
    }

    /// Video frames at exactly these times.
    static func videoFrames(_ asset: AVURLAsset, at seconds: [Double], tokens: Int) -> [FindMediaData?] {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1_024, height: 1_024)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.05, preferredTimescale: 600)
        let times = seconds.map { CMTime(seconds: $0, preferredTimescale: 600) }
        let box = Unchecked(generator)
        let images: [Double: FindMediaData] = (try? blocking {
            var images: [Double: FindMediaData] = [:]
            for await result in box.value.images(for: times) {
                guard let image = try? result.image, let data = rgb(image, tokens: tokens) else { continue }
                images[result.requestedTime.seconds] = data
            }
            return images
        }) ?? [:]
        return times.map { images[$0.seconds] }
    }

    /// Packed RGB of `image` at the encoder's size for `tokens`.
    public static func rgb(_ image: CGImage, tokens: Int) -> FindMediaData? {
        guard let size = encoderSize(width: image.width, height: image.height, tokens: tokens),
              let context = CGContext(data: nil, width: size.width, height: size.height, bitsPerComponent: 8,
                                      bytesPerRow: size.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        context.interpolationQuality = .high
        // Transparent pixels become white, as in a viewer.
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        context.draw(image, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        guard let base = context.data else { return nil }
        let count = size.width * size.height
        let pixels = base.bindMemory(to: UInt8.self, capacity: count * 4)
        var bytes = [UInt8](repeating: 0, count: count * 3)
        for index in 0..<count {
            bytes[index * 3] = pixels[index * 4]
            bytes[index * 3 + 1] = pixels[index * 4 + 1]
            bytes[index * 3 + 2] = pixels[index * 4 + 2]
        }
        return .image(width: size.width, height: size.height, rgb: bytes)
    }

    /// The size the vision encoder takes an image at for `tokens` tokens:
    /// filling the budget with the aspect ratio kept, each side a multiple of
    /// 48 px (3 × 3 patches of 16 px). Mirrors `LlamaEmbedder.imageSize` in
    /// the engine, which Find's core does not import.
    public static func encoderSize(width: Int, height: Int, tokens: Int) -> (width: Int, height: Int)? {
        guard width > 0, height > 0, tokens > 0 else { return nil }
        let patch = 16, pooling = 3
        let maxPatches = tokens * pooling * pooling
        let factor = (Double(maxPatches * patch * patch) / Double(height * width)).squareRoot()
        let side = pooling * patch
        var targetHeight = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        var targetWidth = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        if targetHeight == 0 && targetWidth == 0 { return nil }
        let maxSide = (maxPatches / (pooling * pooling)) * side
        if targetHeight == 0 {
            targetHeight = side
            targetWidth = min(width / height * side, maxSide)
        } else if targetWidth == 0 {
            targetWidth = side
            targetHeight = min(height / width * side, maxSide)
        }
        return (targetWidth, targetHeight)
    }

    /// Streams the sound of `asset` (all of it, or `range`) as 16 kHz mono
    /// floats.
    static func readSamples(of asset: AVURLAsset, range: CMTimeRange?, _ body: ([Float]) -> Void) throws {
        guard let track = try blocking({ Unchecked(try await asset.loadTracks(withMediaType: .audio).first) }).value
        else { throw CocoaError(.fileReadCorruptFile) }
        let reader = try AVAssetReader(asset: asset)
        if let range { reader.timeRange = range }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw CocoaError(.fileReadCorruptFile) }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? CocoaError(.fileReadCorruptFile) }
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var samples = [Float](repeating: 0, count: length / 4)
            samples.withUnsafeMutableBytes { raw in
                _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: raw.count, destination: raw.baseAddress!)
            }
            body(samples)
        }
        if reader.status == .failed { throw reader.error ?? CocoaError(.fileReadCorruptFile) }
    }

    /// "0:42", "12:05", "1:02:03".
    public static func clock(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded(.down)))
        let hours = total / 3_600, minutes = (total % 3_600) / 60, rest = total % 60
        return hours > 0 ? String(format: "%d:%02d:%02d", hours, minutes, rest)
            : String(format: "%d:%02d", minutes, rest)
    }

    /// Runs async AVFoundation calls from the synchronous reader (the helper
    /// reads one file at a time on purpose).
    static func blocking<T: Sendable>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
        let box = ResultBox<T>()
        let done = DispatchSemaphore(value: 0)
        Task.detached(priority: .utility) {
            do { box.result = .success(try await operation()) } catch { box.result = .failure(error) }
            done.signal()
        }
        done.wait()
        return try box.result!.get()
    }

    private final class ResultBox<T>: @unchecked Sendable {
        var result: Result<T, Error>?
    }

    /// AVFoundation objects used by one blocking call at a time.
    struct Unchecked<T>: @unchecked Sendable {
        let value: T
        init(_ value: T) { self.value = value }
    }
}
