import Foundation
import Metal
import Testing
@testable import Mference

/// The reference pack writer and store on the tiny synthetic GGUF: every
/// tensor round-trips to its GGUF blocks in a tiled layout, the receipt and
/// hashes are checked, and the writer never overwrites.
@Suite("ReferencePackTests", .serialized)
struct ReferencePackTests {
    struct Fixture {
        let ctx: MetalContext
        let dir: URL
        var pack: URL { dir.appendingPathComponent("tiny.mfpack") }
        func store(verifyHashes: Bool = false) throws -> Native.ReferenceWeightStore {
            try Native.ReferenceWeightStore(directory: pack, device: ctx.device, verifyHashes: verifyHashes)
        }
    }

    static func fixture(layout: @escaping (Native.GGMLType, Int, Int) -> Native.TileLayout) throws -> Fixture {
        let ctx = try MetalContext()
        let dir = try RefSynthetic.temporaryDirectory()
        let ggufURL = dir.appendingPathComponent("tiny.gguf")
        try RefSynthetic.gguf().write(to: ggufURL)
        try Native.ReferencePackWriter.write(gguf: ggufURL, to: dir.appendingPathComponent("tiny.mfpack"),
                                             label: "tiny", layout: layout)
        let fx = Fixture(ctx: ctx, dir: dir)
        _ = try fx.store(verifyHashes: true)
        return fx
    }

    static func tiled(_ type: Native.GGMLType, _ rows: Int, _ blocks: Int) -> Native.TileLayout {
        RefSynthetic.tiled(type, rows, blocks)
    }

    @Test func packRoundTripsAndIsVerified() throws {
        let fx = try Self.fixture(layout: Self.tiled)
        let store = try fx.store()
        let gguf = try Native.ReferenceGGUFFile(url: fx.dir.appendingPathComponent("tiny.gguf"))
        let manifest = store.manifest
        #expect(manifest.tensors.count == gguf.tensors.count)
        #expect(manifest.config == RefTest.tiny)
        #expect(manifest.tensors.contains { $0.layout != .gguf && $0.packedSHA256 != $0.sourceSHA256 })
        for info in gguf.tensors {
            let t = try store.tensor(info.name)
            #expect(t.offset % 4096 == 0)
            let format = try #require(Native.QuantBlockFormat.of(info.type))
            let src = gguf.bytes(of: info)
            for row in [0, t.rows - 1] {
                let want = Native.ReferenceDequant.dequantizeRow(info.type, src.baseAddress! + row * t.cols / format.elements * format.bytes,
                                                                 blocks: t.cols / format.elements)
                #expect(Native.ReferenceDequant.row(of: t, row).map(\.bitPattern) == want.map(\.bitPattern), "\(info.name)")
            }
        }
        let tok = try store.tokenizer.makeTokenizer()
        #expect(tok.vocabSize == RefTest.tiny.vocabSize)
        #expect(store.chatTemplate.template.hasPrefix("{% for m in messages %}"))
        #expect(store.chatTemplate.eosToken == "<|im_end|>")
        // Never overwrites, and detects tampering and incomplete packs.
        #expect(throws: Native.NativeError.self) {
            try Native.ReferencePackWriter.write(gguf: fx.dir.appendingPathComponent("tiny.gguf"),
                                                 to: fx.dir.appendingPathComponent("tiny.mfpack"), label: "tiny")
        }
        let weights = fx.dir.appendingPathComponent("tiny.mfpack/weights.bin")
        var bytes = try Data(contentsOf: weights)
        let victim = Int(manifest.tensors[5].offset) + 3
        bytes[victim] ^= 0x55
        try bytes.write(to: weights)
        #expect(throws: Native.NativeError.self) {
            _ = try Native.ReferenceWeightStore(directory: fx.dir.appendingPathComponent("tiny.mfpack"),
                                                device: fx.ctx.device, verifyHashes: true)
        }
        try FileManager.default.removeItem(at: fx.dir.appendingPathComponent("tiny.mfpack/verified-install.json"))
        #expect(throws: Native.NativeError.self) {
            _ = try Native.ReferenceWeightStore(directory: fx.dir.appendingPathComponent("tiny.mfpack"),
                                                device: fx.ctx.device)
        }
        try? FileManager.default.removeItem(at: fx.dir)
    }
}
