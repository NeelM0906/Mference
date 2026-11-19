import Foundation
import Testing
@testable import Mference

/// The reference GGUF reader on well-formed v2/v3 files with non-default
/// alignment, and on hostile headers, which must throw rather than trap.
@Suite("ReferenceGGUFTests")
struct ReferenceGGUFTests {
    static func q8(_ name: String, rows: Int = 2, seed: UInt64 = 1) -> RefSynthetic.Tensor {
        RefSynthetic.Tensor(name: name, type: .q8_0, shape: [32, rows],
                            bytes: RefTest.randomBlocks(.q8_0, blocks: rows, seed: seed))
    }

    static func open(_ b: RefSynthetic.GGUFBuilder) throws -> Native.ReferenceGGUFFile {
        let dir = try RefSynthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("t.gguf")
        try b.data().write(to: url)
        return try Native.ReferenceGGUFFile(url: url)
    }

    static func builder(_ tensors: [RefSynthetic.Tensor]) -> RefSynthetic.GGUFBuilder {
        var b = RefSynthetic.GGUFBuilder()
        b.string("general.architecture", "qwen35")
        b.tensors = tensors
        return b
    }

    @Test(arguments: [(UInt32(2), 32), (3, 64), (3, 4096)])
    func readsVersionsAndAlignments(version: UInt32, alignment: Int) throws {
        let tensors = [Self.q8("a", seed: 1), Self.q8("b", rows: 3, seed: 2)]
        var b = Self.builder(tensors)
        b.version = version
        b.alignment = alignment
        if alignment != 32 { b.u32("general.alignment", UInt32(alignment)) }
        let f = try Self.open(b)
        #expect(f.version == Int(version))
        #expect(f.alignment == alignment && f.dataOffset % alignment == 0)
        for (info, t) in zip(f.tensors, tensors) {
            #expect(Array(f.bytes(of: info)) == t.bytes)
        }
    }

    @Test func readsSpecialTokenKeys() throws {
        var b = Self.builder([Self.q8("a")])
        b.strings("tokenizer.ggml.tokens", ["a", "b", "c"])
        b.u32("tokenizer.ggml.eos_token_id", 1)
        b.u32("tokenizer.ggml.eot_token_id", 2)
        b.u32("tokenizer.ggml.fim_pad_token_id", 0)
        let data = try Self.open(b).tokenizerData()
        #expect(data.eosTokenID == 1)
        #expect(data.specialTokenIDs == ["eot": 2, "fim_pad": 0])
        let none = try Self.open(Self.builder([Self.q8("a")]).withTokens()).tokenizerData()
        #expect(none.specialTokenIDs == nil)
    }

    static func expectRefused(_ b: RefSynthetic.GGUFBuilder, _ what: String) {
        #expect(throws: Native.NativeError.self, "\(what)") { _ = try Self.open(b) }
    }

    @Test func hostileHeadersThrow() throws {
        var b = Self.builder([Self.q8("a")])
        b.version = 4
        Self.expectRefused(b, "version 4")

        for bad in [UInt32(0), 3, 48] {
            b = Self.builder([Self.q8("a")])
            b.u32("general.alignment", bad)
            Self.expectRefused(b, "alignment \(bad)")
        }

        b = Self.builder([Self.q8("a")])
        b.string("general.architecture", "again")
        Self.expectRefused(b, "duplicate key")

        b = Self.builder([Self.q8("a"), Self.q8("a", seed: 2)])
        Self.expectRefused(b, "duplicate tensor")

        b = Self.builder([Self.q8("a")])
        b.dimsOverride[0] = [UInt64.max, 1]
        Self.expectRefused(b, "dimension beyond Int")

        b = Self.builder([Self.q8("a")])
        b.dimsOverride[0] = [1 << 40, 1 << 40]
        Self.expectRefused(b, "element count overflow")

        b = Self.builder([Self.q8("a")])
        b.dimsOverride[0] = [32, 1 << 20]
        Self.expectRefused(b, "tensor past the end")

        b = Self.builder([Self.q8("a"), Self.q8("b")])
        b.offsetOverride[1] = 4
        Self.expectRefused(b, "misaligned offset")

        b = Self.builder([Self.q8("a"), Self.q8("b")])
        b.offsetOverride[1] = 0
        Self.expectRefused(b, "overlapping tensors")

        b = Self.builder([Self.q8("a")])
        b.offsetOverride[0] = UInt64.max - 31
        Self.expectRefused(b, "offset beyond Int")

        // A truncated header and an absurd string length.
        let good = Self.builder([Self.q8("a")]).data()
        for cut in [10, 30, 60] {
            let dir = try RefSynthetic.temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: dir) }
            let url = dir.appendingPathComponent("t.gguf")
            var bytes = good.prefix(cut)
            if cut == 60 {
                bytes = good
                // The first key's length field sits right after the 24-byte header.
                bytes.replaceSubrange(24..<32, with: RefSynthetic.GGUFBuilder.le(UInt64.max))
            }
            try Data(bytes).write(to: url)
            #expect(throws: Native.NativeError.self, "cut \(cut)") { _ = try Native.ReferenceGGUFFile(url: url) }
        }
    }
}

extension RefSynthetic.GGUFBuilder {
    func withTokens() -> Self {
        var b = self
        b.strings("tokenizer.ggml.tokens", ["a", "b", "c"])
        return b
    }
}
