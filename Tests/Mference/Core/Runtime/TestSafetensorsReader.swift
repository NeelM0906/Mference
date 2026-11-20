import Foundation
@testable import Mference

// MARK: - Minimal safetensors reader

/// Read-only safetensors reader: the header JSON plus byte ranges into an
/// mmapped `Data`. Enough for the toy checkpoints and parity fixtures; no dtype conversion beyond
/// BF16/F32. Moved here from the removed Flash-Next parity fixture.
struct Safetensors {
    struct Entry {
        let name: String
        let dtype: String
        let shape: [Int]
        let begin: Int
        let end: Int
    }

    private let data: Data
    private let payloadBase: Int
    let entries: [String: Entry]
    let names: [String]

    init(url: URL) throws {
        let bytes = try Data(contentsOf: url, options: .mappedIfSafe)
        guard bytes.count >= 8 else { throw CocoaError(.fileReadCorruptFile) }
        var headerLength: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &headerLength) { dst in
            bytes.copyBytes(to: dst, from: 0..<8)
        }
        let headerEnd = 8 + Int(headerLength)
        guard headerEnd <= bytes.count else { throw CocoaError(.fileReadCorruptFile) }
        data = bytes
        payloadBase = headerEnd
        guard let root = try JSONSerialization
            .jsonObject(with: bytes.subdata(in: 8..<headerEnd)) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var parsed: [String: Entry] = [:]
        for (name, value) in root where name != "__metadata__" {
            guard let object = value as? [String: Any],
                  let dtype = object["dtype"] as? String,
                  let shape = object["shape"] as? [Int],
                  let offsets = object["data_offsets"] as? [Int],
                  offsets.count == 2 else { continue }
            parsed[name] = Entry(name: name, dtype: dtype, shape: shape,
                                 begin: offsets[0], end: offsets[1])
        }
        entries = parsed
        names = parsed.keys.sorted()
    }

    func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        return entry
    }

    func rawBytes(_ entry: Entry) -> [UInt8] {
        [UInt8](data[(payloadBase + entry.begin)..<(payloadBase + entry.end)])
    }

    /// BF16 (or F32) tensor widened to `[Float]`, row-major.
    func floats(_ name: String) throws -> [Float] {
        let entry = try self.entry(name)
        let bytes = rawBytes(entry)
        switch entry.dtype {
        case "BF16":
            return stride(from: 0, to: bytes.count, by: 2).map { j -> Float in
                let lo = UInt16(bytes[j])
                let hi = UInt16(bytes[j + 1])
                let bits: UInt16 = lo | (hi << 8)
                return Quantization.bf16ToFloat(bits)
            }
        case "F32":
            return stride(from: 0, to: bytes.count, by: 4).map { i -> Float in
                // Explicitly-typed steps: the fused bit-OR expression times out
                // the Swift type-checker on some toolchains (CI Xcode).
                let b0 = UInt32(bytes[i])
                let b1 = UInt32(bytes[i + 1])
                let b2 = UInt32(bytes[i + 2])
                let b3 = UInt32(bytes[i + 3])
                let bits: UInt32 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                return Float(bitPattern: bits)
            }
        default:
            preconditionFailure("\(name) is \(entry.dtype), not a float tensor")
        }
    }
}
