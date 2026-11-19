import Darwin
import Foundation

/// Errors from the `.mfpack` writer.
public enum MFPackError: Error, Equatable, CustomStringConvertible {
    /// The source is not a GGUF this writer can pack.
    case invalidSource(String)
    /// The source uses something the pack format or this writer does not carry.
    case unsupported(String)
    /// A hash, size or resume-state check failed.
    case integrity(String)
    /// The output already exists, or a partial pack needs `resume` or `discardPartial`.
    case outputState(String)
    /// A tile row group does not fit the bounded scratch.
    case scratch(String)
    /// A system call failed.
    case io(String)

    public var description: String {
        switch self {
        case .invalidSource(let s): return "mfpack: invalid GGUF: \(s)"
        case .unsupported(let s): return "mfpack: unsupported: \(s)"
        case .integrity(let s): return "mfpack: integrity check failed: \(s)"
        case .outputState(let s): return "mfpack: \(s)"
        case .scratch(let s): return "mfpack: scratch: \(s)"
        case .io(let s): return "mfpack: I/O: \(s)"
        }
    }
}

extension MFPack {
    /// A GGUF file's header: metadata (in file order), tensor infos and the
    /// data-section offset. Parsed from a temporary read-only mapping that is
    /// released before `init` returns; only the header values stay in memory.
    ///
    /// Every size, count and offset is checked: hostile input throws
    /// `MFPackError`, never traps. The checks and the value model match the
    /// reference reader (`Native.ReferenceGGUFFile`), so both writers derive
    /// identical manifests from the same file.
    public struct GGUFHeader {
        public enum Value: Equatable, Sendable {
            case uint(UInt64)
            case int(Int64)
            case float(Double)
            case bool(Bool)
            case string(String)
            case array([Value])
        }

        public struct TensorInfo: Equatable, Sendable {
            public var name: String
            /// ggml `ne`.
            public var shape: [Int]
            public var type: GGMLType
            /// Offset from the start of the data section.
            public var offset: UInt64
            public var byteCount: Int
        }

        public let version: Int
        public let keys: [String]
        public let metadata: [String: Value]
        public let tensors: [TensorInfo]
        public let alignment: Int
        /// Absolute offset of the data section.
        public let dataOffset: Int
        public let fileSize: Int

        public init(path: String) throws {
            let fd = open(path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { throw MFPackError.io("open \(path): errno \(errno)") }
            defer { close(fd) }
            var st = stat()
            guard fstat(fd, &st) == 0 else { throw MFPackError.io("stat \(path): errno \(errno)") }
            guard (st.st_mode & S_IFMT) == S_IFREG else { throw MFPackError.invalidSource("\(path) is not a file") }
            let size = Int(st.st_size)
            guard size > 24 else { throw MFPackError.invalidSource("GGUF too small") }
            guard let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0), p != MAP_FAILED else {
                throw MFPackError.io("mmap \(path): errno \(errno)")
            }
            defer { munmap(p, size) }
            try self.init(base: UnsafeRawPointer(p), size: size)
        }

        /// Parses a header from bytes in memory (`size` is the whole file's).
        public init(base: UnsafeRawPointer, size: Int) throws {
            var cursor = Cursor(base: base, size: size)
            guard try cursor.bytes(4) == Array("GGUF".utf8) else { throw MFPackError.invalidSource("not a GGUF file") }
            let v = try cursor.u32()
            guard v == 2 || v == 3 else { throw MFPackError.unsupported("GGUF version \(v)") }
            let tensorCount = try cursor.u64()
            let keyCount = try cursor.u64()
            guard tensorCount < 1 << 20, keyCount < 1 << 20 else { throw MFPackError.invalidSource("GGUF counts") }
            var keys: [String] = []
            var metadata: [String: Value] = [:]
            for _ in 0..<keyCount {
                let key = try cursor.string()
                let type = try cursor.u32()
                guard metadata[key] == nil else { throw MFPackError.invalidSource("GGUF key \(key) appears twice") }
                // ggml reads general.alignment only as a u32.
                if key == "general.alignment" && type != 4 {
                    throw MFPackError.invalidSource("general.alignment must be a u32, not GGUF type \(type)")
                }
                metadata[key] = try cursor.value(type: type)
                keys.append(key)
            }
            var infos: [TensorInfo] = []
            var names: Set<String> = []
            for _ in 0..<tensorCount {
                let name = try cursor.string()
                guard names.insert(name).inserted else { throw MFPackError.invalidSource("tensor \(name) appears twice") }
                let dims = try cursor.int(cursor.u32())
                guard (1...4).contains(dims) else { throw MFPackError.invalidSource("tensor \(name) dims \(dims)") }
                var shape: [Int] = []
                for _ in 0..<dims {
                    let d = try cursor.int(cursor.u64())
                    guard d > 0 else { throw MFPackError.invalidSource("tensor \(name) dimension \(d)") }
                    shape.append(d)
                }
                let rawType = try cursor.u32()
                guard let type = GGMLType(rawValue: rawType) else {
                    throw MFPackError.unsupported("tensor \(name) ggml type \(rawType)")
                }
                let offset = try cursor.u64()
                guard let format = QuantBlockFormat.of(type) else {
                    throw MFPackError.unsupported("tensor \(name) type \(type)")
                }
                var elements = 1
                for d in shape {
                    let (product, overflow) = elements.multipliedReportingOverflow(by: d)
                    guard !overflow else { throw MFPackError.invalidSource("tensor \(name) \(shape) overflows") }
                    elements = product
                }
                guard shape[0] % format.elements == 0 else {
                    throw MFPackError.invalidSource("tensor \(name) \(shape) rows are not whole \(type) blocks")
                }
                let (bytes, overflow) = (elements / format.elements).multipliedReportingOverflow(by: format.bytes)
                guard !overflow else { throw MFPackError.invalidSource("tensor \(name) byte count overflows") }
                infos.append(TensorInfo(name: name, shape: shape, type: type, offset: offset, byteCount: bytes))
            }
            var align = 32
            if let a = metadata["general.alignment"] {
                guard case .uint(let raw) = a, let value = Int(exactly: raw), value > 0, value & (value - 1) == 0,
                      value <= 1 << 20 else {
                    throw MFPackError.invalidSource("general.alignment must be a power of two")
                }
                align = value
            }
            let dataOffset = (cursor.position + align - 1) / align * align
            guard dataOffset <= size else { throw MFPackError.invalidSource("GGUF data section past the end") }
            var spans: [(Int, Int, String)] = []
            for info in infos {
                guard let start = Int(exactly: info.offset), start % align == 0 else {
                    throw MFPackError.invalidSource("tensor \(info.name) offset \(info.offset) is not \(align)-aligned")
                }
                let (end, overflow) = start.addingReportingOverflow(info.byteCount)
                guard !overflow, end <= size - dataOffset else {
                    throw MFPackError.invalidSource("tensor \(info.name) runs past the end of the file")
                }
                spans.append((start, end, info.name))
            }
            spans.sort { $0.0 < $1.0 }
            for (a, b) in zip(spans, spans.dropFirst()) where b.0 < a.1 {
                throw MFPackError.invalidSource("tensors \(a.2) and \(b.2) overlap")
            }
            version = Int(v)
            self.keys = keys
            self.metadata = metadata
            tensors = infos
            alignment = align
            self.dataOffset = dataOffset
            fileSize = size
        }

        // MARK: Typed metadata access

        public func int(_ key: String) -> Int? {
            switch metadata[key] {
            case .uint(let v)?: return Int(exactly: v)
            case .int(let v)?: return Int(exactly: v)
            default: return nil
            }
        }

        public func float(_ key: String) -> Float? {
            switch metadata[key] {
            case .float(let v)?: return Float(v)
            case .uint(let v)?: return Float(v)
            case .int(let v)?: return Float(v)
            default: return nil
            }
        }

        public func string(_ key: String) -> String? {
            if case .string(let s)? = metadata[key] { return s }
            return nil
        }

        public func bool(_ key: String) -> Bool? {
            if case .bool(let b)? = metadata[key] { return b }
            return nil
        }

        public func array(_ key: String) -> [Value]? {
            if case .array(let a)? = metadata[key] { return a }
            return nil
        }

        /// `Qwen35Config` from `qwen35.*` metadata and the tensor list.
        public func qwen35Config() throws -> Qwen35Config {
            let arch = string("general.architecture") ?? ""
            guard arch == "qwen35" else { throw MFPackError.unsupported("architecture \(arch)") }
            func req(_ k: String) throws -> Int {
                guard let v = int("\(arch).\(k)") else { throw MFPackError.invalidSource("missing GGUF key \(arch).\(k)") }
                return v
            }
            func reqF(_ k: String) throws -> Float {
                guard let v = float("\(arch).\(k)") else { throw MFPackError.invalidSource("missing GGUF key \(arch).\(k)") }
                return v
            }
            let nextn = int("\(arch).nextn_predict_layers") ?? 0
            let sections: [Int] = (array("\(arch).rope.dimension_sections") ?? []).compactMap {
                switch $0 {
                case .uint(let v): return Int(exactly: v)
                case .int(let v): return Int(exactly: v)
                default: return nil
                }
            }
            let blocks = try req("block_count")
            guard nextn >= 0, blocks > nextn else {
                throw MFPackError.invalidSource("block_count \(blocks), nextn_predict_layers \(nextn)")
            }
            guard let embd = tensors.first(where: { $0.name == "token_embd.weight" }), embd.shape.count == 2 else {
                throw MFPackError.invalidSource("missing token_embd.weight")
            }
            let keyLength = try req("attention.key_length")
            return Qwen35Config(
                blockCount: blocks - nextn,
                hiddenSize: try req("embedding_length"),
                ffnSize: try req("feed_forward_length"),
                headCount: try req("attention.head_count"),
                kvHeadCount: try req("attention.head_count_kv"),
                keyLength: keyLength,
                valueLength: int("\(arch).attention.value_length") ?? keyLength,
                rmsNormEpsilon: try reqF("attention.layer_norm_rms_epsilon"),
                ropeFreqBase: try reqF("rope.freq_base"),
                ropeDimensionCount: try req("rope.dimension_count"),
                ropeSections: sections,
                fullAttentionInterval: int("\(arch).full_attention_interval") ?? 4,
                ssmConvKernel: try req("ssm.conv_kernel"),
                ssmStateSize: try req("ssm.state_size"),
                ssmGroupCount: try req("ssm.group_count"),
                ssmTimeStepRank: try req("ssm.time_step_rank"),
                ssmInnerSize: try req("ssm.inner_size"),
                contextLength: try req("context_length"),
                nextnPredictLayers: nextn,
                vocabSize: embd.shape[1],
                tiedEmbeddings: !tensors.contains(where: { $0.name == "output.weight" }))
        }

        /// `TokenizerData` from `tokenizer.ggml.*` metadata, verbatim.
        public func tokenizerData() throws -> TokenizerData {
            func strings(_ k: String) throws -> [String] {
                guard let a = array(k) else { throw MFPackError.invalidSource("missing GGUF key \(k)") }
                return a.map { if case .string(let s) = $0 { return s } else { return "" } }
            }
            let types: [Int32] = try (array("tokenizer.ggml.token_type") ?? []).map {
                let t: Int32?
                switch $0 {
                case .int(let v): t = Int32(exactly: v)
                case .uint(let v): t = Int32(exactly: v)
                default: t = nil
                }
                guard let t else { throw MFPackError.invalidSource("tokenizer.ggml.token_type value") }
                return t
            }
            func id(_ k: String) -> Int32? { int(k).flatMap { Int32(exactly: $0) } }
            return TokenizerData(
                model: string("tokenizer.ggml.model") ?? "",
                pre: string("tokenizer.ggml.pre") ?? "",
                tokens: try strings("tokenizer.ggml.tokens"),
                tokenTypes: types,
                merges: (try? strings("tokenizer.ggml.merges")) ?? [],
                bosTokenID: id("tokenizer.ggml.bos_token_id"),
                eosTokenID: id("tokenizer.ggml.eos_token_id"),
                paddingTokenID: id("tokenizer.ggml.padding_token_id"),
                addBOS: bool("tokenizer.ggml.add_bos_token"),
                addEOS: bool("tokenizer.ggml.add_eos_token"),
                specialTokenIDs: specialTokenIDs())
        }

        /// Every `tokenizer.ggml.<name>_token_id` key other than bos, eos and
        /// padding, by `<name>` as spelled in the key; nil when there are none.
        func specialTokenIDs() -> [String: Int32]? {
            var out: [String: Int32] = [:]
            let prefix = "tokenizer.ggml.", suffix = "_token_id"
            for key in keys where key.hasPrefix(prefix) && key.hasSuffix(suffix) {
                let name = String(key.dropFirst(prefix.count).dropLast(suffix.count))
                guard !["bos", "eos", "padding"].contains(name), let v = int(key), let id = Int32(exactly: v) else {
                    continue
                }
                out[name] = id
            }
            return out.isEmpty ? nil : out
        }

        /// Scalar metadata (strings, numbers, booleans) as strings for
        /// `Manifest.metadata`; arrays and the `tokenizer.*` keys are left out
        /// (they have their own files), except the tokenizer model and
        /// pre-tokenizer names.
        public func scalarMetadataStrings() -> [String: String] {
            var out: [String: String] = [:]
            for key in keys where !key.hasPrefix("tokenizer.") {
                switch metadata[key] {
                case .uint(let v)?: out[key] = String(v)
                case .int(let v)?: out[key] = String(v)
                case .float(let v)?: out[key] = String(Float(v))
                case .bool(let v)?: out[key] = v ? "true" : "false"
                case .string(let s)?: out[key] = s
                default: break
                }
            }
            for key in ["tokenizer.ggml.model", "tokenizer.ggml.pre"] {
                if let s = string(key) { out[key] = s }
            }
            return out
        }

        /// Sequential little-endian reader over the mapped header.
        struct Cursor {
            let base: UnsafeRawPointer
            let size: Int
            var position = 0

            mutating func need(_ n: Int) throws {
                guard n >= 0, position + n <= size else { throw MFPackError.invalidSource("GGUF header truncated") }
            }
            mutating func bytes(_ n: Int) throws -> [UInt8] {
                try need(n)
                defer { position += n }
                return Array(UnsafeRawBufferPointer(start: base + position, count: n))
            }
            mutating func load<T: FixedWidthInteger>(_: T.Type) throws -> T {
                try need(MemoryLayout<T>.size)
                defer { position += MemoryLayout<T>.size }
                return T(littleEndian: base.loadUnaligned(fromByteOffset: position, as: T.self))
            }
            mutating func u32() throws -> UInt32 { try load(UInt32.self) }
            mutating func u64() throws -> UInt64 { try load(UInt64.self) }
            func int<T: BinaryInteger>(_ v: T) throws -> Int {
                guard let i = Int(exactly: v) else { throw MFPackError.invalidSource("GGUF value \(v) out of range") }
                return i
            }
            /// The smallest encoding of one element of GGUF value type `type`.
            static func minimumElementSize(_ type: UInt32) -> Int {
                switch type {
                case 0, 1, 7: return 1
                case 2, 3: return 2
                case 4, 5, 6: return 4
                case 8, 10, 11, 12: return 8
                case 9: return 12
                default: return 1
                }
            }

            mutating func string() throws -> String {
                let n = try int(u64())
                guard n < 1 << 26 else { throw MFPackError.invalidSource("GGUF string of \(n) bytes") }
                try need(n)
                defer { position += n }
                return String(decoding: UnsafeRawBufferPointer(start: base + position, count: n), as: UTF8.self)
            }

            // GGUF value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32,
            // 7 bool, 8 string, 9 array, 10 u64, 11 i64, 12 f64.
            mutating func value(type: UInt32, depth: Int = 0) throws -> Value {
                switch type {
                case 0: return .uint(UInt64(try load(UInt8.self)))
                case 1: return .int(Int64(try load(Int8.self)))
                case 2: return .uint(UInt64(try load(UInt16.self)))
                case 3: return .int(Int64(try load(Int16.self)))
                case 4: return .uint(UInt64(try load(UInt32.self)))
                case 5: return .int(Int64(try load(Int32.self)))
                case 6: return .float(Double(Float(bitPattern: try load(UInt32.self))))
                case 7: return .bool(try load(UInt8.self) != 0)
                case 8: return .string(try string())
                case 9:
                    guard depth < 2 else { throw MFPackError.invalidSource("GGUF arrays nested too deeply") }
                    let elementType = try u32()
                    let count = try int(u64())
                    guard count < 1 << 26 else { throw MFPackError.invalidSource("GGUF array of \(count)") }
                    var values: [Value] = []
                    // Never reserve more elements than the remaining bytes can hold.
                    values.reserveCapacity(min(count, max(0, size - position) / Self.minimumElementSize(elementType)))
                    for _ in 0..<count { values.append(try value(type: elementType, depth: depth + 1)) }
                    return .array(values)
                case 10: return .uint(try u64())
                case 11: return .int(Int64(bitPattern: try u64()))
                case 12: return .float(Double(bitPattern: try u64()))
                default: throw MFPackError.invalidSource("GGUF value type \(type)")
                }
            }
        }
    }
}
