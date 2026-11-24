import Foundation

/// Reads the facts the catalog needs from a GGUF file's header: architecture,
/// name, context length, embedding width, tensor count, the
/// next-token-prediction (MTP) tensors, and, for a multimodal projector, its
/// type and projection width. Only the header is parsed, through a bounded 1 MiB read window
/// that may run at most that far past the header; tensor data is never
/// parsed, so listing a 12 GB file costs a few megabytes of reads (the
/// tokenizer vocabulary dominates the header).
public enum EngineGGUFReader {
    public enum ReadError: Error, Equatable, Sendable, CustomStringConvertible {
        case notGGUF
        case unsupportedVersion(UInt32)
        case truncated
        case malformed(String)
        case missingArchitecture

        public var description: String {
            switch self {
            case .notGGUF: "not a GGUF file"
            case .unsupportedVersion(let version): "unsupported GGUF version \(version)"
            case .truncated: "GGUF header is truncated"
            case .malformed(let detail): "malformed GGUF header: \(detail)"
            case .missingArchitecture: "GGUF header has no general.architecture"
            }
        }
    }

    /// Larger than any header seen in practice (the 27B file's is about 10 MB,
    /// almost all vocabulary); bounds a corrupt or hostile file.
    public static let maxHeaderBytes: UInt64 = 256 << 20

    public static func read(url: URL) throws -> EngineGGUFMetadata {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let fileSize = try handle.seekToEnd()
        var reader = Reader(handle: handle, fileSize: fileSize,
                            limit: min(fileSize, maxHeaderBytes))
        return try parse(&reader)
    }

    static func parse(_ reader: inout Reader) throws -> EngineGGUFMetadata {
        guard try reader.bytes(4) == Data("GGUF".utf8) else { throw ReadError.notGGUF }
        let version = try reader.u32()
        guard version == 2 || version == 3 else { throw ReadError.unsupportedVersion(version) }
        let tensorCount = try reader.u64()
        let keyCount = try reader.u64()
        guard tensorCount <= 1 << 20, keyCount <= 1 << 20 else {
            throw ReadError.malformed("implausible counts \(tensorCount) tensors, \(keyCount) keys")
        }

        var architecture: String?
        var name: String?
        var fileType: String?
        var projectorType: String?
        var hasVisionEncoder = false
        var integers: [String: UInt64] = [:]
        for _ in 0..<keyCount {
            let key = try reader.string()
            let type = try reader.u32()
            switch key {
            case "general.architecture":
                architecture = try reader.stringValue(type: type, key: key)
            case "general.name":
                name = try reader.stringValue(type: type, key: key)
            case "general.type" where type == 8:
                fileType = try reader.string()
            case "clip.projector_type" where type == 8:
                projectorType = try reader.string()
            case "clip.has_vision_encoder" where type == 7:
                hasVisionEncoder = try reader.bytes(1).first != 0
            default:
                if key.hasSuffix(".context_length") || key.hasSuffix(".nextn_predict_layers")
                    || key.hasSuffix(".embedding_length") || key == "clip.vision.projection_dim" {
                    // Another architecture's key, or an odd type, is not a
                    // reason to refuse the file: a non-integer is skipped.
                    integers[key] = try reader.integerValueOrSkip(type: type)
                } else {
                    try reader.skipValue(type: type, depth: 0)
                }
            }
        }
        guard let architecture, !architecture.isEmpty else { throw ReadError.missingArchitecture }

        var nextN = 0
        for _ in 0..<tensorCount {
            let tensor = try reader.string()
            let dimensions = try reader.u32()
            guard (1...8).contains(dimensions) else {
                throw ReadError.malformed("tensor \(tensor) has \(dimensions) dimensions")
            }
            try reader.skip(UInt64(dimensions) * 8 + 4 + 8)
            if tensor.hasPrefix("nextn.") || tensor.contains(".nextn.") { nextN += 1 }
        }

        let context = integers["\(architecture).context_length"].map { Int(clamping: $0) }
        // An `mmproj` file: llama.cpp's converters mark it `general.type`
        // `mmproj`; older ones only give the `clip` architecture.
        let projector: EngineGGUFProjector? = (fileType == "mmproj" || architecture == "clip")
            ? EngineGGUFProjector(
                projectorType: projectorType,
                hasVisionEncoder: hasVisionEncoder,
                projectionDimension: integers["clip.vision.projection_dim"].map { Int(clamping: $0) })
            : nil
        return EngineGGUFMetadata(architecture: architecture,
                                  name: name,
                                  contextLength: context,
                                  tensorCount: Int(tensorCount),
                                  nextNTensorCount: nextN,
                                  embeddingLength: integers["\(architecture).embedding_length"]
                                      .map { Int(clamping: $0) },
                                  projector: projector)
    }

    /// Sequential little-endian reads over a file through a 1 MiB window.
    struct Reader {
        static let window = 1 << 20
        private let source: (UInt64, Int) throws -> Data
        let limit: UInt64
        private(set) var position: UInt64 = 0
        private var buffer = Data()
        private var bufferStart: UInt64 = 0

        init(handle: FileHandle, fileSize: UInt64, limit: UInt64) {
            self.limit = min(limit, fileSize)
            self.source = { offset, count in
                try handle.seek(toOffset: offset)
                return try handle.read(upToCount: count) ?? Data()
            }
        }

        /// Reads from an in-memory header, for tests.
        init(data: Data) {
            self.limit = UInt64(data.count)
            self.source = { offset, count in
                let start = Int(offset)
                return data.subdata(in: start..<min(data.count, start + count))
            }
        }

        mutating func bytes(_ count: Int) throws -> Data {
            let end = position + UInt64(count)
            guard end <= limit else { throw ReadError.truncated }
            let bufferEnd = bufferStart + UInt64(buffer.count)
            if position < bufferStart || end > bufferEnd {
                let want = max(count, Self.window)
                let available = Int(min(UInt64(want), limit - position))
                buffer = try source(position, available)
                bufferStart = position
                guard buffer.count >= count else { throw ReadError.truncated }
            }
            let start = Int(position - bufferStart)
            position = end
            return buffer.subdata(in: (buffer.startIndex + start)..<(buffer.startIndex + start + count))
        }

        mutating func skip(_ count: UInt64) throws {
            let (end, overflow) = position.addingReportingOverflow(count)
            guard !overflow, end <= limit else { throw ReadError.truncated }
            position = end
        }

        mutating func u32() throws -> UInt32 {
            try bytes(4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
        }

        mutating func u64() throws -> UInt64 {
            try bytes(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        }

        mutating func string() throws -> String {
            let length = try u64()
            guard length <= 1 << 24 else { throw ReadError.malformed("string of \(length) bytes") }
            let data = try bytes(Int(length))
            guard let text = String(data: data, encoding: .utf8) else {
                throw ReadError.malformed("string is not UTF-8")
            }
            return text
        }

        // GGUF value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32,
        // 7 bool, 8 string, 9 array, 10 u64, 11 i64, 12 f64.
        static func fixedSize(_ type: UInt32) -> UInt64? {
            switch type {
            case 0, 1, 7: 1
            case 2, 3: 2
            case 4, 5, 6: 4
            case 10, 11, 12: 8
            default: nil
            }
        }

        mutating func stringValue(type: UInt32, key: String) throws -> String {
            guard type == 8 else { throw ReadError.malformed("\(key) is not a string") }
            return try string()
        }

        /// A non-negative integer value, or nil (with the value skipped) for
        /// any other type or a negative number.
        mutating func integerValueOrSkip(type: UInt32) throws -> UInt64? {
            func nonNegative(_ value: Int64) -> UInt64? { value >= 0 ? UInt64(value) : nil }
            switch type {
            case 0: return UInt64(try bytes(1).first ?? 0)
            case 1: return nonNegative(Int64(Int8(bitPattern: try bytes(1).first ?? 0)))
            case 2: return UInt64(try u16())
            case 3: return nonNegative(Int64(Int16(bitPattern: try u16())))
            case 4: return UInt64(try u32())
            case 5: return nonNegative(Int64(Int32(bitPattern: try u32())))
            case 10: return try u64()
            case 11: return nonNegative(Int64(bitPattern: try u64()))
            default:
                try skipValue(type: type, depth: 0)
                return nil
            }
        }

        mutating func u16() throws -> UInt16 {
            try bytes(2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }.littleEndian
        }

        mutating func skipValue(type: UInt32, depth: Int) throws {
            if let size = Self.fixedSize(type) {
                try skip(size)
                return
            }
            switch type {
            case 8:
                let length = try u64()
                try skip(length)
            case 9:
                guard depth < 4 else { throw ReadError.malformed("arrays nested too deeply") }
                let elementType = try u32()
                let count = try u64()
                guard count <= 1 << 28 else { throw ReadError.malformed("array of \(count) elements") }
                if let size = Self.fixedSize(elementType) {
                    let (bytes, overflow) = size.multipliedReportingOverflow(by: count)
                    guard !overflow else { throw ReadError.malformed("array too large") }
                    try skip(bytes)
                } else {
                    for _ in 0..<count { try skipValue(type: elementType, depth: depth + 1) }
                }
            default:
                throw ReadError.malformed("unknown value type \(type)")
            }
        }
    }
}
