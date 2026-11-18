import Foundation
import Testing
@testable import MferenceEngine

/// Builds GGUF headers in memory: no tensor data, just what the reader sees.
struct GGUFHeaderBuilder {
    enum Value {
        case string(String)
        case u32(UInt32)
        case u64(UInt64)
        case i32(Int32)
        case f32(Float)
        case bool(Bool)
        case strings([String])
        case u32s([UInt32])
    }

    var version: UInt32 = 3
    var keys: [(String, Value)] = []
    var tensors: [String] = []

    func data() -> Data {
        var out = Data("GGUF".utf8)
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        func string(_ s: String) { u64(UInt64(s.utf8.count)); out.append(contentsOf: s.utf8) }
        u32(version)
        u64(UInt64(tensors.count))
        u64(UInt64(keys.count))
        for (key, value) in keys {
            string(key)
            switch value {
            case .string(let s): u32(8); string(s)
            case .u32(let v): u32(4); u32(v)
            case .u64(let v): u32(10); u64(v)
            case .i32(let v): u32(5); u32(UInt32(bitPattern: v))
            case .f32(let v): u32(6); u32(v.bitPattern)
            case .bool(let v): u32(7); out.append(v ? 1 : 0)
            case .strings(let list): u32(9); u32(8); u64(UInt64(list.count)); list.forEach(string)
            case .u32s(let list): u32(9); u32(4); u64(UInt64(list.count)); list.forEach(u32)
            }
        }
        for name in tensors {
            string(name)
            u32(2); u64(4096); u64(4096)
            u32(12)
            u64(0)
        }
        return out
    }

    static func qwen35(name: String = "Qwen3.8-27B", context: UInt32 = 262_144, mtp: Bool) -> GGUFHeaderBuilder {
        var builder = GGUFHeaderBuilder()
        builder.keys = [
            ("general.architecture", .string("qwen35")),
            ("general.name", .string(name)),
            ("general.sampling.temp", .f32(1.0)),
            ("qwen35.block_count", .u32(65)),
            ("qwen35.context_length", .u32(context)),
            ("qwen35.rope.dimension_sections", .u32s([11, 11, 10, 0])),
            ("tokenizer.ggml.tokens", .strings((0..<5_000).map { "tok\($0)" })),
            ("tokenizer.ggml.add_bos_token", .bool(false)),
        ]
        if mtp { builder.keys.append(("qwen35.nextn_predict_layers", .u32(1))) }
        builder.tensors = ["token_embd.weight", "blk.0.attn_norm.weight", "output.weight"]
        if mtp {
            builder.tensors += ["blk.64.nextn.eh_proj.weight", "blk.64.nextn.enorm.weight",
                                "blk.64.nextn.hnorm.weight", "blk.64.nextn.shared_head_norm.weight"]
        }
        return builder
    }
}

@Suite("EngineGGUFReader")
struct EngineGGUFReaderTests {
    func parse(_ data: Data) throws -> EngineGGUFMetadata {
        var reader = EngineGGUFReader.Reader(data: data)
        return try EngineGGUFReader.parse(&reader)
    }

    @Test func readsArchitectureNameContextAndMTPTensors() throws {
        let deep = try parse(GGUFHeaderBuilder.qwen35(mtp: true).data())
        #expect(deep == EngineGGUFMetadata(architecture: "qwen35", name: "Qwen3.8-27B",
                                           contextLength: 262_144, tensorCount: 7, nextNTensorCount: 4))
        #expect(deep.hasMTP)
        let quick = try parse(GGUFHeaderBuilder.qwen35(name: "Qwen3.5-4B", mtp: false).data())
        #expect(!quick.hasMTP)
        #expect(quick.tensorCount == 3)
    }

    @Test func readsAFileThroughTheBoundedWindow() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("engine-gguf-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }
        // A header bigger than the 1 MiB read window, plus fake tensor bytes.
        var builder = GGUFHeaderBuilder.qwen35(mtp: true)
        builder.keys.append(("tokenizer.ggml.merges", .strings((0..<120_000).map { "merge pair \($0)" })))
        var data = builder.data()
        #expect(data.count > EngineGGUFReader.Reader.window)
        data.append(Data(count: 4_096))
        try data.write(to: url)
        let metadata = try EngineGGUFReader.read(url: url)
        #expect(metadata.nextNTensorCount == 4)
        #expect(metadata.contextLength == 262_144)
    }

    @Test func rejectsWhatIsNotAReadableHeader() throws {
        #expect(throws: EngineGGUFReader.ReadError.notGGUF) { try parse(Data("GGML....".utf8)) }
        var old = GGUFHeaderBuilder.qwen35(mtp: false)
        old.version = 1
        #expect(throws: EngineGGUFReader.ReadError.unsupportedVersion(1)) { try parse(old.data()) }
        let full = GGUFHeaderBuilder.qwen35(mtp: true).data()
        #expect(throws: EngineGGUFReader.ReadError.truncated) { try parse(full.prefix(full.count - 5)) }
        var anonymous = GGUFHeaderBuilder()
        anonymous.keys = [("general.name", .string("x"))]
        #expect(throws: EngineGGUFReader.ReadError.missingArchitecture) { try parse(anonymous.data()) }
    }

    @Test func oddTypesOnSuffixKeysAreSkippedNotFatal() throws {
        var builder = GGUFHeaderBuilder()
        builder.keys = [("general.architecture", .string("qwen35")),
                        ("other.context_length", .f32(4.5)),
                        ("other.nextn_predict_layers", .strings(["x"])),
                        ("qwen35.context_length", .u32(32_768))]
        builder.tensors = ["nextn.pre_projection.weight", "blk.0.attn_norm.weight"]
        let metadata = try parse(builder.data())
        #expect(metadata.contextLength == 32_768)
        #expect(metadata.nextNTensorCount == 1)
        builder.keys = [("general.architecture", .string("qwen35")),
                        ("qwen35.context_length", .i32(-1))]
        #expect(try parse(builder.data()).contextLength == nil)
    }

    @Test func contextIsOptional() throws {
        var builder = GGUFHeaderBuilder()
        builder.keys = [("general.architecture", .string("llama")),
                        ("llama.context_length", .i32(8_192))]
        #expect(try parse(builder.data()).contextLength == 8_192)
        builder.keys = [("general.architecture", .string("llama"))]
        #expect(try parse(builder.data()).contextLength == nil)
    }
}
