import CoreGraphics
import CryptoKit
import Foundation
import ImageIO

/// One image for a vision-capable model: packed 8-bit RGB pixels, already
/// decoded (the engine's llama.cpp carries no image decoders), and a stable
/// hash of them.
///
/// The hash keys prompt-prefix reuse: a later request that carries the same
/// image at the same place in the prompt reuses the KV it left, and a
/// different image never does.
public struct EngineMedia: Sendable, Hashable {
    public let width: Int
    public let height: Int
    /// `width * height * 3` bytes, row-major, no padding.
    public let rgb: Data
    /// Lowercase hex SHA-256 of the size and the pixels.
    public let hash: String

    /// Nil when the size is not positive or `rgb` is not `width * height * 3`
    /// bytes.
    public init?(width: Int, height: Int, rgb: Data) {
        guard width > 0, height > 0, rgb.count == width * height * 3 else { return nil }
        self.width = width
        self.height = height
        self.rgb = rgb
        var hasher = SHA256()
        hasher.update(data: Data("rgb8 \(width)x\(height)\n".utf8))
        hasher.update(data: rgb)
        hash = hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// The longest side `image(encoded:)` keeps; larger images are scaled
    /// down to it. The vision encoders resize far below this themselves
    /// (Qwen3-VL's to at most 768 × 768-ish token grids), so this only bounds
    /// the pixels held in memory.
    public static let maximumDecodedSide = 4_096

    /// Decodes a PNG, JPEG, HEIC, TIFF, GIF (first frame) or any other format
    /// ImageIO reads, applies its orientation, and scales it down to
    /// `maximumDecodedSide` if larger. Nil when the data is not an image.
    public static func image(encoded data: Data,
                             maximumSide: Int = EngineMedia.maximumDecodedSide) -> EngineMedia? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              CGImageSourceGetCount(source) > 0 else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumSide,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            return nil
        }
        return Self.image(image)
    }

    /// The pixels of `image`, drawn over white (transparency has no meaning
    /// to the encoders) in sRGB.
    public static func image(_ image: CGImage) -> EngineMedia? {
        let width = image.width, height = image.height
        guard width > 0, height > 0,
              let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            let rect = CGRect(x: 0, y: 0, width: width, height: height)
            context.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
            context.fill(rect)
            context.draw(image, in: rect)
            return true
        }
        guard drawn else { return nil }
        var rgb = Data(count: width * height * 3)
        rgb.withUnsafeMutableBytes { out in
            let bytes = out.bindMemory(to: UInt8.self)
            for pixel in 0..<(width * height) {
                bytes[pixel * 3] = rgba[pixel * 4]
                bytes[pixel * 3 + 1] = rgba[pixel * 4 + 1]
                bytes[pixel * 3 + 2] = rgba[pixel * 4 + 2]
            }
        }
        return EngineMedia(width: width, height: height, rgb: rgb)
    }

    /// How an image's tokens appear in `EnginePreparedPrompt.promptTokens`:
    /// one of these per vision token, so the array's length is the prompt's
    /// length in the model's context. Token ids are never negative.
    public static let placeholderToken: Int32 = -1
}

extension EngineRequest {
    /// Whether any message carries an image.
    public var hasMedia: Bool { media.values.contains { !$0.isEmpty } }

    /// The request's media checked against the model: none on a model without
    /// vision, and only on user messages that exist. Thrown errors are
    /// `EngineError.invalidRequest` with the server's codes.
    public func validateMedia(supportsVision: Bool) throws {
        guard hasMedia else { return }
        guard supportsVision else {
            throw EngineError.invalidRequest(message: "this model does not accept images",
                                             param: "messages", code: "unsupported_content")
        }
        for (index, images) in media where !images.isEmpty {
            guard messages.indices.contains(index), messages[index].role == .user else {
                throw EngineError.invalidRequest(message: "images may only be attached to user messages",
                                                 param: "messages", code: "unsupported_content")
            }
        }
    }

    /// The images of every message, in prompt order.
    var orderedMedia: [EngineMedia] {
        media.keys.sorted().flatMap { media[$0] ?? [] }
    }
}
