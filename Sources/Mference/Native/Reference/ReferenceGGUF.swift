import Foundation

extension Native {
    /// A GGUF file mapped read-only: its metadata, tensor infos and tensor
    /// bytes (the reference pack writer's source). Nothing is copied into the
    /// Swift heap except the metadata values.
    public final class ReferenceGGUFFile {
        public enum Value: Equatable {
            case uint(UInt64)
            case int(Int64)
            case float(Double)
            case bool(Bool)
            case string(String)
            case array([Value])
        }

        public struct TensorInfo: Equatable {
            public var name: String
            /// ggml `ne`.
            public var shape: [Int]
            public var type: GGMLType
            /// Offset from the start of the data section.
            public var offset: UInt64
            public var byteCount: Int
        }

        public let url: URL
        public let version: Int
        /// Metadata in file order.
        public private(set) var keys: [String] = []
        public private(set) var metadata: [String: Value] = [:]
        public private(set) var tensors: [TensorInfo] = []
        public let alignment: Int
        public let dataOffset: Int
        public let fileSize: Int
        private let base: UnsafeMutableRawPointer
        private let mappedLength: Int

        public init(url: URL) throws {
            self.url = url
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else { throw NativeError.missing("GGUF \(url.path)") }
            defer { close(fd) }
            var st = stat()
            guard fstat(fd, &st) == 0, st.st_size > 24 else { throw NativeError.integrity("GGUF too small") }
            fileSize = Int(st.st_size)
            guard let p = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
                throw NativeError.resource("mmap \(url.path)")
            }
            base = p
            mappedLength = fileSize

            var cursor = Cursor(base: UnsafeRawPointer(p), size: fileSize)
            guard try cursor.bytes(4) == Array("GGUF".utf8) else { throw NativeError.integrity("not a GGUF file") }
            let v = try cursor.u32()
            guard v == 2 || v == 3 else { throw NativeError.unsupported("GGUF version \(v)") }
            version = Int(v)
            let tensorCount = try cursor.u64()
            let keyCount = try cursor.u64()
            guard tensorCount < 1 << 20, keyCount < 1 << 20 else { throw NativeError.integrity("GGUF counts") }
            var keys: [String] = []
            var metadata: [String: Value] = [:]
            for _ in 0..<keyCount {
                let key = try cursor.string()
                let type = try cursor.u32()
                metadata[key] = try cursor.value(type: type)
                keys.append(key)
            }
            var infos: [TensorInfo] = []
            for _ in 0..<tensorCount {
                let name = try cursor.string()
                let dims = Int(try cursor.u32())
                guard (1...4).contains(dims) else { throw NativeError.integrity("tensor \(name) dims \(dims)") }
                var shape: [Int] = []
                for _ in 0..<dims { shape.append(Int(try cursor.u64())) }
                let rawType = try cursor.u32()
                guard let type = GGMLType(rawValue: rawType) else {
                    throw NativeError.unsupported("tensor \(name) ggml type \(rawType)")
                }
                let offset = try cursor.u64()
                guard let format = QuantBlockFormat.of(type) else {
                    throw NativeError.unsupported("tensor \(name) type \(type)")
                }
                let elements = shape.reduce(1, *)
                guard elements % format.elements == 0 else {
                    throw NativeError.invalidShape("tensor \(name) \(shape) not whole \(type) blocks")
                }
                infos.append(TensorInfo(name: name, shape: shape, type: type, offset: offset,
                                        byteCount: elements / format.elements * format.bytes))
            }
            var align = 32
            if case .uint(let a)? = metadata["general.alignment"] { align = Int(a) }
            alignment = align
            dataOffset = (cursor.position + align - 1) / align * align
            for info in infos {
                guard dataOffset + Int(info.offset) + info.byteCount <= fileSize else {
                    throw NativeError.integrity("tensor \(info.name) runs past the end of the file")
                }
            }
            self.keys = keys
            self.metadata = metadata
            self.tensors = infos
        }

        deinit { munmap(base, mappedLength) }

        /// The tensor's bytes, in GGUF order (valid for this object's lifetime).
        public func bytes(of info: TensorInfo) -> UnsafeRawBufferPointer {
            UnsafeRawBufferPointer(start: UnsafeRawPointer(base) + dataOffset + Int(info.offset), count: info.byteCount)
        }

        /// The whole file (valid for this object's lifetime).
        public var fileBytes: UnsafeRawBufferPointer {
            UnsafeRawBufferPointer(start: UnsafeRawPointer(base), count: fileSize)
        }

        // MARK: Typed metadata access

        public func int(_ key: String) -> Int? {
            switch metadata[key] {
            case .uint(let v)?: return Int(v)
            case .int(let v)?: return Int(v)
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

        /// Sequential little-endian reader over the mapped header.
        struct Cursor {
            let base: UnsafeRawPointer
            let size: Int
            var position = 0

            mutating func need(_ n: Int) throws {
                guard n >= 0, position + n <= size else { throw NativeError.integrity("GGUF header truncated") }
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
            mutating func string() throws -> String {
                let n = Int(try u64())
                guard n < 1 << 26 else { throw NativeError.integrity("GGUF string of \(n) bytes") }
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
                    guard depth < 2 else { throw NativeError.integrity("GGUF arrays nested too deeply") }
                    let elementType = try u32()
                    let count = Int(try u64())
                    guard count < 1 << 26 else { throw NativeError.integrity("GGUF array of \(count)") }
                    var values: [Value] = []
                    values.reserveCapacity(count)
                    for _ in 0..<count { values.append(try value(type: elementType, depth: depth + 1)) }
                    return .array(values)
                case 10: return .uint(try u64())
                case 11: return .int(Int64(bitPattern: try u64()))
                case 12: return .float(Double(bitPattern: try u64()))
                default: throw NativeError.integrity("GGUF value type \(type)")
                }
            }
        }
    }
}

extension Native.ReferenceGGUFFile {
    /// `Qwen35Config` from `qwen35.*` metadata and the tensor list.
    public func qwen35Config() throws -> Native.Qwen35Config {
        let arch = string("general.architecture") ?? ""
        guard arch == "qwen35" else { throw Native.NativeError.unsupported("architecture \(arch)") }
        func req(_ k: String) throws -> Int {
            guard let v = int("\(arch).\(k)") else { throw Native.NativeError.missing("GGUF key \(arch).\(k)") }
            return v
        }
        func reqF(_ k: String) throws -> Float {
            guard let v = float("\(arch).\(k)") else { throw Native.NativeError.missing("GGUF key \(arch).\(k)") }
            return v
        }
        let nextn = int("\(arch).nextn_predict_layers") ?? 0
        let sections: [Int] = (array("\(arch).rope.dimension_sections") ?? []).compactMap {
            switch $0 {
            case .uint(let v): return Int(v)
            case .int(let v): return Int(v)
            default: return nil
            }
        }
        guard let embd = tensors.first(where: { $0.name == "token_embd.weight" }), embd.shape.count == 2 else {
            throw Native.NativeError.missing("token_embd.weight")
        }
        let keyLength = try req("attention.key_length")
        return Native.Qwen35Config(
            blockCount: try req("block_count") - nextn,
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
    public func tokenizerData() throws -> Native.TokenizerData {
        func strings(_ k: String) throws -> [String] {
            guard let a = array(k) else { throw Native.NativeError.missing("GGUF key \(k)") }
            return a.map { if case .string(let s) = $0 { return s } else { return "" } }
        }
        let types: [Int32] = (array("tokenizer.ggml.token_type") ?? []).map {
            switch $0 {
            case .int(let v): return Int32(v)
            case .uint(let v): return Int32(v)
            default: return 0
            }
        }
        func id(_ k: String) -> Int32? { int(k).map { Int32($0) } }
        return Native.TokenizerData(
            model: string("tokenizer.ggml.model") ?? "",
            pre: string("tokenizer.ggml.pre") ?? "",
            tokens: try strings("tokenizer.ggml.tokens"),
            tokenTypes: types,
            merges: (try? strings("tokenizer.ggml.merges")) ?? [],
            bosTokenID: id("tokenizer.ggml.bos_token_id"),
            eosTokenID: id("tokenizer.ggml.eos_token_id"),
            paddingTokenID: id("tokenizer.ggml.padding_token_id"),
            addBOS: bool("tokenizer.ggml.add_bos_token"),
            addEOS: bool("tokenizer.ggml.add_eos_token"))
    }

    /// Scalar metadata (strings, numbers, booleans) as strings, for
    /// `PackManifest.metadata`; arrays and the tokenizer/template keys are
    /// left out (they have their own files).
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
}
