import CryptoKit
import Foundation

extension Native {
    /// `verified-install.json` of an `.mfpack` v1: written last, after every
    /// hash check passed. A pack without it is incomplete.
    ///
    /// The first nine fields are the ones every writer records (the reference
    /// writer's receipt has exactly these); the production writer
    /// (`MferenceRepack`, `docs/MFERENCE_PACK.md`) also records the SHA-256 of
    /// `tokenizer.json` and `chat_template.jinja`, which the store then checks.
    public struct PackReceipt: Codable, Equatable, Sendable {
        public var format: String
        public var formatVersion: Int
        public var label: String
        /// `"reference"` or `"MferenceRepack"`.
        public var writer: String
        /// SHA-256 of the whole source GGUF.
        public var sourceSHA256: String
        /// SHA-256 of `manifest.json` as written.
        public var manifestSHA256: String
        public var tensorCount: Int
        /// Size of `weights.bin` in bytes.
        public var weightsByteCount: UInt64
        public var verifiedAt: String
        /// SHA-256 of `tokenizer.json` (production writer only).
        public var tokenizerSHA256: String?
        /// SHA-256 of `chat_template.jinja` (production writer only).
        public var chatTemplateSHA256: String?

        public init(format: String, formatVersion: Int, label: String, writer: String, sourceSHA256: String,
                    manifestSHA256: String, tensorCount: Int, weightsByteCount: UInt64, verifiedAt: String,
                    tokenizerSHA256: String? = nil, chatTemplateSHA256: String? = nil) {
            self.format = format
            self.formatVersion = formatVersion
            self.label = label
            self.writer = writer
            self.sourceSHA256 = sourceSHA256
            self.manifestSHA256 = manifestSHA256
            self.tensorCount = tensorCount
            self.weightsByteCount = weightsByteCount
            self.verifiedAt = verifiedAt
            self.tokenizerSHA256 = tokenizerSHA256
            self.chatTemplateSHA256 = chatTemplateSHA256
        }
    }

    /// SHA-256 helpers shared by the pack store and its tests.
    public enum PackHash {
        static let chunk = 16 << 20

        /// Lowercase hex SHA-256 of `bytes`, hashed in bounded chunks.
        public static func sha256Hex(_ bytes: UnsafeRawBufferPointer) -> String {
            var hasher = SHA256()
            var start = 0
            while start < bytes.count {
                let n = min(chunk, bytes.count - start)
                hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: bytes[start ..< start + n]))
                start += n
            }
            return hex(hasher.finalize())
        }

        public static func sha256Hex(_ data: Data) -> String {
            data.withUnsafeBytes { sha256Hex($0) }
        }

        static func hex(_ digest: SHA256.Digest) -> String {
            var out = ""
            out.reserveCapacity(64)
            let digits = Array("0123456789abcdef".utf8)
            for byte in digest {
                out.unicodeScalars.append(Unicode.Scalar(digits[Int(byte >> 4)]))
                out.unicodeScalars.append(Unicode.Scalar(digits[Int(byte & 0x0F)]))
            }
            return out
        }
    }
}
