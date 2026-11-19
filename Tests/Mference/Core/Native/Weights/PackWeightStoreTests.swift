import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// The production writer (`MferenceRepackCore`) and the pack store on the
/// tiny synthetic GGUF: byte identity with the frozen reference writer, the
/// round trip to GGUF blocks in table and tiled layouts, the store's load
/// checks, and the reference `qwen35` path on the W1 store.
@Suite("PackWeightStoreTests", .serialized)
struct PackWeightStoreTests {
    struct Fixture {
        let dir: URL
        var gguf: URL { dir.appendingPathComponent("tiny.gguf") }
        func pack(_ name: String) -> URL { dir.appendingPathComponent("\(name).mfpack") }
    }

    static func fixture() throws -> Fixture {
        let dir = try RefSynthetic.temporaryDirectory()
        try RefSynthetic.gguf().write(to: dir.appendingPathComponent("tiny.gguf"))
        return Fixture(dir: dir)
    }

    static func tiledMirror(_ t: MFPack.GGMLType, _ rows: Int, _ blocks: Int) -> MFPack.TileLayout {
        PackMirrorPinTests.mirror(RefSynthetic.tiled(Native.GGMLType(rawValue: t.rawValue)!, rows, blocks))
    }

    static func read(_ url: URL, _ name: String) throws -> Data {
        try Data(contentsOf: url.appendingPathComponent(name))
    }

    static func device() throws -> MTLDevice { try #require(MetalContext.makeSystemDefaultDevice()) }

    @Test(arguments: [false, true])
    func productionWriterIsByteIdenticalToTheReference(tiled: Bool) throws {
        let fx = try Self.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        if tiled {
            try Native.ReferencePackWriter.write(gguf: fx.gguf, to: fx.pack("ref"), label: "tiny",
                                                 layout: RefSynthetic.tiled)
        } else {
            try Native.ReferencePackWriter.write(gguf: fx.gguf, to: fx.pack("ref"), label: "tiny")
        }
        let options = tiled
            ? MFPackWriteOptions(label: "tiny", layout: { Self.tiledMirror($0, $1, $2) }, readChunkBytes: 4096, scratchLimitBytes: 1 << 20)
            : MFPackWriteOptions(label: "tiny", readChunkBytes: 4096, scratchLimitBytes: 1 << 20)
        let result = try MFPackWriter.write(gguf: fx.gguf.path, output: fx.pack("w1").path, options: options)
        for file in ["weights.bin", "manifest.json", "tokenizer.json", "chat_template.jinja"] {
            #expect(try Self.read(fx.pack("ref"), file) == Self.read(fx.pack("w1"), file), "\(file)")
        }
        #expect(result.manifest.layoutTableVersion == (tiled ? 0 : 1))
        #expect(result.manifest.tensors.contains { $0.layout != .gguf } == tiled)
        #expect(result.scratchHighWaterBytes <= 1 << 20)
        #expect(result.writtenBytes == result.manifest.tensors.reduce(0) { $0 + $1.byteCount })
        #expect(result.receipt.writer == MFPackWriter.writerName)

        // Byte-exact round trip to the GGUF's blocks, and both stores open it.
        let report = try MFPackVerifier.roundTrip(pack: fx.pack("w1").path, gguf: fx.gguf.path, chunkBytes: 4096)
        #expect(report.tensors == result.manifest.tensors.count && (report.tiledTensors > 0) == tiled)
        let device = try Self.device()
        let ours = try Native.PackWeightStore(directory: fx.pack("w1"), device: device, verification: .packedSHA256)
        let ref = try Native.ReferenceWeightStore(directory: fx.pack("w1"), device: device, verifyHashes: true)
        #expect(ours.manifest == ref.manifest && ours.config == RefTest.tiny)
        #expect(ours.receipt.tokenizerSHA256 != nil && ours.receipt.chatTemplateSHA256 != nil)
        for t in ref.manifest.tensors {
            let a = try ours.tensor(t.name), b = try ref.tensor(t.name)
            #expect(a.offset == b.offset && a.byteCount == b.byteCount && a.layout == b.layout && a.type == b.type
                    && a.shape == b.shape && a.sourceSHA256 == b.sourceSHA256, "\(t.name)")
            #expect(Native.ReferenceDequant.row(of: a, a.rows - 1).map(\.bitPattern)
                    == Native.ReferenceDequant.row(of: b, b.rows - 1).map(\.bitPattern), "\(t.name)")
        }
        #expect(ours.chatTemplate.template == ref.chatTemplate.template)
        #expect(ours.chatTemplate.eosToken == "<|im_end|>" && ours.chatTemplate.bosToken == ref.chatTemplate.bosToken)
        #expect(ours.tokenizer.data == ref.tokenizer.data)
        #expect(!ours.contains("nope") && ours.contains("token_embd.weight"))
        #expect(throws: Native.NativeError.missing("tensor nope")) { _ = try ours.tensor("nope") }
        // The store also reads the reference writer's pack (receipt without the file hashes).
        let refPack = try Native.PackWeightStore(directory: fx.pack("ref"), device: device)
        #expect(refPack.receipt.tokenizerSHA256 == nil && refPack.manifest.tensors.count == ours.manifest.tensors.count)
    }

    @Test func storeRejectsDamagedPacks() throws {
        let fx = try Self.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let device = try Self.device()
        try MFPackWriter.write(gguf: fx.gguf.path, output: fx.pack("good").path, options: .init(label: "tiny"))
        func copy(_ name: String, _ damage: (URL) throws -> Void) throws -> URL {
            let url = fx.pack(name)
            try FileManager.default.copyItem(at: fx.pack("good"), to: url)
            try damage(url)
            return url
        }
        func rejects(_ url: URL, _ comment: Comment,
                     verification: Native.PackWeightStore.Verification = .receipt) {
            #expect(throws: Native.NativeError.self, comment) {
                _ = try Native.PackWeightStore(directory: url, device: device, verification: verification)
            }
        }
        _ = try Native.PackWeightStore(directory: fx.pack("good"), device: device, verification: .packedSHA256)
        rejects(try copy("noreceipt") {
            try FileManager.default.removeItem(at: $0.appendingPathComponent("verified-install.json"))
        }, "no receipt")
        rejects(try copy("manifest") {
            var d = try Self.read($0, "manifest.json")
            d.append(0x20)
            try d.write(to: $0.appendingPathComponent("manifest.json"))
        }, "manifest edited")
        rejects(try copy("tokenizer") {
            var d = try Self.read($0, "tokenizer.json")
            d.append(0x20)
            try d.write(to: $0.appendingPathComponent("tokenizer.json"))
        }, "tokenizer edited")
        rejects(try copy("template") {
            try Data("x".utf8).write(to: $0.appendingPathComponent("chat_template.jinja"))
        }, "template edited")
        rejects(try copy("short") {
            let h = try FileHandle(forWritingTo: $0.appendingPathComponent("weights.bin"))
            try h.truncate(atOffset: 4096)
            try h.close()
        }, "weights truncated")
        let flipped = try copy("flipped") {
            let m = try JSONDecoder().decode(Native.PackManifest.self, from: Self.read($0, "manifest.json"))
            let url = $0.appendingPathComponent("weights.bin")
            var d = try Data(contentsOf: url)
            d[Int(m.tensors[3].offset) + 1] ^= 0x40
            try d.write(to: url)
        }
        // Structural checks pass (the size is unchanged); the hash check catches it.
        _ = try Native.PackWeightStore(directory: flipped, device: device)
        rejects(flipped, "flipped byte", verification: .packedSHA256)
    }

    static func goodManifest() throws -> (Native.PackManifest, UInt64) {
        let fx = try fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let r = try MFPackWriter.write(gguf: fx.gguf.path, output: fx.pack("m").path, options: .init(label: "tiny"))
        let m = try JSONDecoder().decode(Native.PackManifest.self, from: MFPack.encoder().encode(r.manifest))
        return (m, r.receipt.weightsByteCount)
    }

    @Test func manifestValidationCatchesEveryInconsistency() throws {
        let (good, size) = try Self.goodManifest()
        #expect(try Native.PackWeightStore.validate(good, fileSize: size).count == good.tensors.count)
        func rejects(_ comment: Comment, fileSize: UInt64? = nil, _ edit: (inout Native.PackManifest) -> Void) {
            var m = good
            edit(&m)
            #expect(throws: Native.NativeError.self, comment) {
                _ = try Native.PackWeightStore.validate(m, fileSize: fileSize ?? size)
            }
        }
        let embd = good.tensors.firstIndex { $0.name == "token_embd.weight" }!
        let q = good.tensors.firstIndex { $0.name == "blk.1.attn_q.weight" }!
        rejects("format") { $0.formatVersion = 2 }
        rejects("architecture") { $0.architecture = "qwen3" }
        rejects("layout table") { $0.layoutTableVersion = 2 }
        rejects("duplicate") { $0.tensors.append($0.tensors[0]) }
        rejects("byte count") { $0.tensors[embd].byteCount -= 1 }
        rejects("misaligned") { $0.tensors[embd].offset += 64 }
        rejects("past the end", fileSize: size - 1) { _ in }
        rejects("overlap") { $0.tensors[q].offset = $0.tensors[embd].offset }
        rejects("layout does not fit") {
            $0.tensors[embd].layout = Native.TileLayout(rowsPerTile: 7, blocksPerTile: 1, planes: false)
        }
        rejects("not the table's layout") {
            $0.tensors[embd].layout = Native.TileLayout(rowsPerTile: 4, blocksPerTile: 1, planes: false)
            $0.tensors[embd].packedSHA256 = String(repeating: "0", count: 64)
        }
        rejects("identity layout with different hashes") { $0.tensors[embd].packedSHA256 = String(repeating: "0", count: 64) }
        rejects("short hash") { $0.tensors[embd].sourceSHA256 = "abc" }
        rejects("unknown block shape") { $0.tensors[embd].shape = [255, 320] }
        rejects("catalog: missing tensor") { $0.tensors.removeAll { $0.name == "blk.2.ssm_a" } }
        rejects("catalog: shape") { $0.config.ffnSize += 256 }
        rejects("catalog: tied") { $0.config.tiedEmbeddings = false }
        // A tiled layout is accepted under layout table 0.
        var tiled = good
        tiled.layoutTableVersion = 0
        tiled.tensors[embd].layout = Native.TileLayout(rowsPerTile: 4, blocksPerTile: 1, planes: false)
        tiled.tensors[embd].packedSHA256 = String(repeating: "0", count: 64)
        _ = try Native.PackWeightStore.validate(tiled, fileSize: size)
    }

    @Test func tensorCatalogDescribesQuickAndDeep() {
        let quick = Native.Qwen35TensorCatalog.entries(for: PackMirrorPinTests.quickConfig)
        #expect(quick.count == 426 && quick.allSatisfy(\.required))
        #expect(quick.contains { $0.name == "blk.2.attn_qkv.weight" && $0.shape == [2560, 8192] })
        #expect(quick.contains { $0.name == "blk.3.attn_q.weight" && $0.shape == [2560, 8192] })
        #expect(quick.contains { $0.name == "blk.3.attn_output.weight" && $0.shape == [4096, 2560] })
        #expect(quick.contains { $0.name == "blk.2.ssm_conv1d.weight" && $0.shape == [4, 8192] })
        var deep = PackMirrorPinTests.quickConfig
        deep.blockCount = 64
        deep.hiddenSize = 5120
        deep.ffnSize = 17_408
        deep.headCount = 24
        deep.ssmTimeStepRank = 48
        deep.ssmInnerSize = 6144
        deep.nextnPredictLayers = 1
        deep.tiedEmbeddings = false
        let d = Native.Qwen35TensorCatalog.entries(for: deep)
        // 865 required (Deep's GGUF has 866 tensors: + nextn.shared_head_norm).
        #expect(d.filter(\.required).count == 865)
        #expect(d.contains { $0.name == "blk.64.nextn.eh_proj.weight" && $0.shape == [10_240, 5120] })
        #expect(d.contains { $0.name == "blk.64.attn_q.weight" && $0.shape == [5120, 12_288] })
        #expect(d.contains { $0.name == "output.weight" && $0.required })
    }

    /// The reference `qwen35` path with only the weights swapped for the W1
    /// store produces bit-identical logits to the all-reference path.
    @Test func referencePathOnTheW1StoreIsBitIdentical() throws {
        let fx = try Self.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        try Native.ReferencePackWriter.write(gguf: fx.gguf, to: fx.pack("ref"), label: "tiny")
        try MFPackWriter.write(gguf: fx.gguf.path, output: fx.pack("w1").path, options: .init(label: "tiny"))
        let ctx = try MetalContext()
        let tokens: [Int32] = [5, 104, 101, 108, 108, 111, 262, 263, 300, 7]
        func logits(_ makeStore: () throws -> any Native.WeightStore) throws -> [[UInt32]] {
            try autoreleasepool {
                let store = try makeStore()
                let factory = Native.ReferenceForwardModelFactory(context: ctx)
                let c = store.config
                let plan = try Native.ReferenceMemoryPlanner().resolve(
                    config: c, weightsMapped: store.mappedByteCount, contextLength: 64, precision: .f16,
                    maxVerifyRows: 1, maxPrefillChunk: 8,
                    activationBytes: factory.activationBytes(config: c, maxPrefillChunk: 8, maxVerifyRows: 1),
                    prefixCacheRAMBudget: 1 << 20, budget: 1 << 30)
                let kv = try Native.ReferenceKVStore(config: c, plan: plan, device: ctx.device)
                let model = try factory.make(Native.ModelComponents(
                    device: ctx.device, weights: store, gemm: Native.ReferenceQuantGEMM(context: ctx),
                    attention: Native.ReferenceAttentionOp(context: ctx), gdn: Native.ReferenceGDNOp(context: ctx),
                    glue: Native.ReferenceGlueOps(context: ctx), kv: kv), plan: plan)
                var rows: [[UInt32]] = []
                var position = 0
                for (kind, chunk) in [(Native.StepKind.prefill, Array(tokens[0..<8])), (.decode, [tokens[8]]),
                                      (.decode, [tokens[9]])] {
                    try model.kv.beginStep(rows: chunk.count, kind: kind)
                    var out: Native.TensorRef?
                    try RefTest.run(ctx) { enc in
                        out = try model.encodeStep(Native.StepPlan(kind: kind, startPosition: position,
                                                                   rows: chunk.count, input: .host(chunk),
                                                                   logits: .last), encoder: enc)
                    }
                    rows.append(RefTest.floats(try #require(out)).map(\.bitPattern))
                    position += chunk.count
                }
                return rows
            }
        }
        let ours = try logits { try Native.PackWeightStore(directory: fx.pack("w1"), device: ctx.device) }
        let ref = try logits { try Native.ReferenceWeightStore(directory: fx.pack("ref"), device: ctx.device) }
        #expect(ours.count == 3 && ours.allSatisfy { $0.count == RefTest.tiny.vocabSize })
        #expect(ours == ref)
    }
}
