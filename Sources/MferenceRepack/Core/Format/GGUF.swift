import Foundation

/// GGML tensor types the installer accepts from a GGUF source. Block sizes
/// and byte sizes follow `ggml/src/ggml-common.h` at the pinned llama.cpp
/// v1.0.0 (b1b770d9). Any other type id is rejected: the installer copies
/// GGUF bytes unchanged, so it must know each block's exact size.
enum GGMLType: UInt32, Sendable, Hashable, CaseIterable {
    case f32 = 0
    case f16 = 1
    case q8_0 = 8
    case q6_K = 14
    case iq3_xxs = 18
    case iq4_nl = 20
    case iq2_s = 22
    case iq4_xs = 23
    case bf16 = 30
    case q2_0 = 42

    /// Elements per block (`QK*`).
    var blockElements: UInt64 {
        switch self {
        case .f32, .f16, .bf16: 1
        case .q8_0, .iq4_nl: 32
        case .q2_0: 64
        case .q6_K, .iq3_xxs, .iq2_s, .iq4_xs: 256
        }
    }

    /// Bytes per block (`sizeof(block_*)`).
    var blockBytes: UInt64 {
        switch self {
        case .f32: 4
        case .f16, .bf16: 2
        case .q8_0: 34
        case .iq4_nl, .q2_0: 18
        case .q6_K: 210
        case .iq3_xxs: 98
        case .iq2_s: 82
        case .iq4_xs: 136
        }
    }

    var name: String {
        switch self {
        case .f32: "F32"; case .f16: "F16"; case .q8_0: "Q8_0"; case .q6_K: "Q6_K"
        case .iq3_xxs: "IQ3_XXS"; case .iq4_nl: "IQ4_NL"; case .iq2_s: "IQ2_S"
        case .iq4_xs: "IQ4_XS"; case .bf16: "BF16"; case .q2_0: "Q2_0"
        }
    }

    /// Bytes of one row of `elements` values; nil if the row is not a whole
    /// number of blocks.
    func rowBytes(elements: UInt64) -> UInt64? {
        guard elements % blockElements == 0 else { return nil }
        return elements / blockElements * blockBytes
    }
}

/// Parses a GGUF v3 header (metadata key/values and the tensor-info table)
/// from a bounded prefix of the file. Fails closed on anything it does not
/// fully understand. Tensor coordinates are absolute file offsets so the
/// planner can copy byte ranges without re-reading the header.
enum GGUF {
    static let magic: [UInt8] = Array("GGUF".utf8)
    static let supportedVersion: UInt32 = 3
    static let defaultAlignment: UInt64 = 32
    /// Shard 1 of the Unsloth UD-IQ3_XXS file has a 10.9 MB header (vocab
    /// and merges); 64 MiB bounds a hostile or corrupt header.
    static let maxHeaderBytes: Int = 64 << 20
    static let maxDims = 4
    static let maxStringBytes: UInt64 = 1 << 20
    static let maxArrayCount: UInt64 = 1 << 24

    indirect enum Value: Sendable, Equatable {
        case u8(UInt8), i8(Int8), u16(UInt16), i16(Int16), u32(UInt32), i32(Int32)
        case f32(Float), bool(Bool), string(String), u64(UInt64), i64(Int64), f64(Double)
        case array([Value])

        var uint64: UInt64? {
            switch self {
            case .u8(let v): UInt64(v)
            case .u16(let v): UInt64(v)
            case .u32(let v): UInt64(v)
            case .u64(let v): v
            case .i8(let v): v >= 0 ? UInt64(v) : nil
            case .i16(let v): v >= 0 ? UInt64(v) : nil
            case .i32(let v): v >= 0 ? UInt64(v) : nil
            case .i64(let v): v >= 0 ? UInt64(v) : nil
            default: nil
            }
        }

        var string: String? {
            if case .string(let s) = self { return s }
            return nil
        }
    }

    struct TensorInfo: Sendable, Equatable {
        let name: String
        /// GGML order: `ne[0]` is the contiguous (row) dimension.
        let ne: [UInt64]
        let type: GGMLType
        /// Offset relative to the data section, as stored in the file.
        let relativeOffset: UInt64
        /// Absolute offset in the shard file.
        let absoluteOffset: UInt64
        let sizeBytes: UInt64

        var elementCount: UInt64 { ne.reduce(1, *) }
    }

    struct Header: Sendable {
        let version: UInt32
        /// Key order as stored in the file.
        let keys: [String]
        let metadata: [String: Value]
        let tensors: [TensorInfo]
        let alignment: UInt64
        /// Bytes of header (magic through the last tensor info).
        let headerBytes: UInt64
        /// Absolute offset of the data section (header rounded up to alignment).
        let dataOffset: UInt64
    }

    enum ParseOutcome: Sendable {
        case complete(Header)
        /// The prefix ends inside the header; fetch at least this many bytes and retry.
        case needBytes(Int)
    }

    /// Parses `prefix`, the first bytes of the shard at `path` whose full
    /// size is `fileSize`. Returns `.needBytes` when the prefix is too short.
    static func parseHeader(path: String, fileSize: UInt64, prefix: Data) throws -> ParseOutcome {
        var r = Reader(data: prefix, path: path)
        do {
            let header = try parse(&r, path: path, fileSize: fileSize)
            return .complete(header)
        } catch Reader.Short.needBytes(let n) {
            guard n <= maxHeaderBytes, UInt64(n) <= fileSize else {
                throw RepackError.ggufInvalid(path: path, detail: "header exceeds \(maxHeaderBytes) bytes or the file")
            }
            // Grow geometrically so a 10 MB header takes a handful of reads.
            return .needBytes(min(maxHeaderBytes, max(n, prefix.count * 2), Int(clamping: fileSize)))
        }
    }

    private static func parse(_ r: inout Reader, path: String, fileSize: UInt64) throws -> Header {
        func invalid(_ d: String) -> RepackError { .ggufInvalid(path: path, detail: d) }
        guard try r.bytes(4) == magic else { throw invalid("bad magic") }
        let version = try r.u32()
        guard version == supportedVersion else { throw invalid("unsupported version \(version)") }
        let nTensors = try r.u64()
        let nKV = try r.u64()
        guard nTensors <= 1 << 20, nKV <= 1 << 16 else { throw invalid("implausible counts \(nTensors)/\(nKV)") }

        var keys: [String] = []
        var metadata: [String: Value] = [:]
        for _ in 0..<nKV {
            let key = try r.string()
            let t = try r.u32()
            let v = try r.value(type: t)
            guard metadata[key] == nil else { throw invalid("duplicate key \(key)") }
            keys.append(key)
            metadata[key] = v
        }

        var alignment = defaultAlignment
        if let a = metadata["general.alignment"] {
            guard case .u32(let v) = a, v > 0, v & (v - 1) == 0 else { throw invalid("bad general.alignment") }
            alignment = UInt64(v)
        }

        var raw: [(String, [UInt64], GGMLType, UInt64)] = []
        var names = Set<String>()
        for _ in 0..<nTensors {
            let name = try r.string()
            let nd = try r.u32()
            guard nd >= 1, nd <= maxDims else { throw invalid("tensor \(name) has \(nd) dims") }
            var ne: [UInt64] = []
            for _ in 0..<nd { ne.append(try r.u64()) }
            let typeID = try r.u32()
            let off = try r.u64()
            guard let type = GGMLType(rawValue: typeID) else {
                throw RepackError.ggufUnsupportedType(path: path, tensor: name, typeID: typeID)
            }
            guard names.insert(name).inserted else { throw invalid("duplicate tensor \(name)") }
            raw.append((name, ne, type, off))
        }

        let headerBytes = UInt64(r.offset)
        let dataOffset = (headerBytes + alignment - 1) / alignment * alignment
        var tensors: [TensorInfo] = []
        tensors.reserveCapacity(raw.count)
        for (name, ne, type, off) in raw {
            guard ne.allSatisfy({ $0 > 0 }) else { throw invalid("tensor \(name) has a zero dimension") }
            guard let row = type.rowBytes(elements: ne[0]) else {
                throw invalid("tensor \(name) ne[0]=\(ne[0]) is not a whole number of \(type.name) blocks")
            }
            var size = row
            for d in ne.dropFirst() {
                let (p, o) = size.multipliedReportingOverflow(by: d)
                guard !o else { throw invalid("tensor \(name) size overflows") }
                size = p
            }
            guard off % alignment == 0 else { throw invalid("tensor \(name) offset \(off) not aligned") }
            let (abs, o1) = dataOffset.addingReportingOverflow(off)
            let (end, o2) = abs.addingReportingOverflow(size)
            guard !o1, !o2, end <= fileSize else {
                throw invalid("tensor \(name) [\(abs), \(end)) exceeds file size \(fileSize)")
            }
            tensors.append(TensorInfo(name: name, ne: ne, type: type, relativeOffset: off,
                                      absoluteOffset: abs, sizeBytes: size))
        }
        // Tensor payloads must not overlap.
        let sorted = tensors.sorted { $0.absoluteOffset < $1.absoluteOffset }
        for (a, b) in zip(sorted, sorted.dropFirst()) where a.absoluteOffset + a.sizeBytes > b.absoluteOffset {
            throw invalid("tensors \(a.name) and \(b.name) overlap")
        }
        // A metadata-only shard (shard 1 of a split GGUF) may end exactly at
        // its header, so the aligned data offset can sit past EOF; it has no
        // tensor bytes, and every tensor range above is bounds-checked.
        guard tensors.isEmpty || dataOffset <= fileSize else {
            throw invalid("data section starts past end of file")
        }
        return Header(version: version, keys: keys, metadata: metadata, tensors: tensors,
                      alignment: alignment, headerBytes: headerBytes, dataOffset: dataOffset)
    }

    /// Little-endian cursor over the header prefix.
    struct Reader {
        enum Short: Error { case needBytes(Int) }
        let data: Data
        let path: String
        private(set) var offset = 0

        init(data: Data, path: String) {
            self.data = data
            self.path = path
        }

        mutating func bytes(_ n: Int) throws -> [UInt8] {
            let (end, o) = offset.addingReportingOverflow(n)
            guard !o else { throw RepackError.ggufInvalid(path: path, detail: "length overflow") }
            guard end <= data.count else { throw Short.needBytes(end) }
            let start = data.startIndex + offset
            defer { offset = end }
            return [UInt8](data[start..<(start + n)])
        }

        private mutating func le<T: FixedWidthInteger>(_: T.Type) throws -> T {
            let b = try bytes(MemoryLayout<T>.size)
            var v: T = 0
            for (i, byte) in b.enumerated() { v |= T(truncatingIfNeeded: byte) << (8 * i) }
            return v
        }

        mutating func u32() throws -> UInt32 { try le(UInt32.self) }
        mutating func u64() throws -> UInt64 { try le(UInt64.self) }

        mutating func string() throws -> String {
            let n = try u64()
            guard n <= GGUF.maxStringBytes else {
                throw RepackError.ggufInvalid(path: path, detail: "string of \(n) bytes")
            }
            let b = try bytes(Int(n))
            guard let s = String(validating: b, as: UTF8.self) else {
                throw RepackError.ggufInvalid(path: path, detail: "string is not valid UTF-8")
            }
            return s
        }

        mutating func value(type t: UInt32) throws -> Value {
            switch t {
            case 0: return .u8(try le(UInt8.self))
            case 1: return .i8(try le(Int8.self))
            case 2: return .u16(try le(UInt16.self))
            case 3: return .i16(try le(Int16.self))
            case 4: return .u32(try le(UInt32.self))
            case 5: return .i32(try le(Int32.self))
            case 6: return .f32(Float(bitPattern: try le(UInt32.self)))
            case 7:
                let b = try le(UInt8.self)
                guard b <= 1 else { throw RepackError.ggufInvalid(path: path, detail: "bool byte \(b)") }
                return .bool(b == 1)
            case 8: return .string(try string())
            case 9:
                let et = try u32()
                guard et != 9 else { throw RepackError.ggufInvalid(path: path, detail: "nested array") }
                let n = try u64()
                guard n <= GGUF.maxArrayCount else {
                    throw RepackError.ggufInvalid(path: path, detail: "array of \(n) elements")
                }
                var items: [Value] = []
                items.reserveCapacity(Int(n))
                for _ in 0..<n { items.append(try value(type: et)) }
                return .array(items)
            case 10: return .u64(try u64())
            case 11: return .i64(try le(Int64.self))
            case 12: return .f64(Double(bitPattern: try u64()))
            default: throw RepackError.ggufInvalid(path: path, detail: "unknown value type \(t)")
            }
        }
    }
}
