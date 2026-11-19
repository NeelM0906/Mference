import Foundation
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// Pins the production writer's mirror types (`MFPack`, in
/// `MferenceRepackCore`, which cannot import `Mference`) to the frozen
/// contracts: identical JSON encodings, block formats, layout table and
/// `packedOffset` permutation.
@Suite("PackMirrorPinTests")
struct PackMirrorPinTests {
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys, .prettyPrinted]
        return try e.encode(value)
    }

    static func mirror(_ t: Native.GGMLType) -> MFPack.GGMLType { MFPack.GGMLType(rawValue: t.rawValue)! }

    static func mirror(_ l: Native.TileLayout) -> MFPack.TileLayout {
        MFPack.TileLayout(rowsPerTile: l.rowsPerTile, blocksPerTile: l.blocksPerTile, planes: l.planes)
    }

    static let quickConfig = Native.Qwen35Config(
        blockCount: 32, hiddenSize: 2560, ffnSize: 9216, headCount: 16, kvHeadCount: 4, keyLength: 256,
        valueLength: 256, rmsNormEpsilon: 1e-6, ropeFreqBase: 1e7, ropeDimensionCount: 64,
        ropeSections: [11, 11, 10, 0], fullAttentionInterval: 4, ssmConvKernel: 4, ssmStateSize: 128,
        ssmGroupCount: 16, ssmTimeStepRank: 32, ssmInnerSize: 4096, contextLength: 262_144, nextnPredictLayers: 0,
        vocabSize: 248_320, tiedEmbeddings: true)

    @Test func ggmlTypesAndBlockFormatsMatch() {
        #expect(Native.GGMLType.allCases.map(\.rawValue) == MFPack.GGMLType.allCases.map(\.rawValue))
        for t in Native.GGMLType.allCases {
            let a = Native.QuantBlockFormat.of(t)
            let b = MFPack.QuantBlockFormat.of(Self.mirror(t))
            #expect((a == nil) == (b == nil), "\(t)")
            guard let a, let b else { continue }
            #expect(a.elements == b.elements && a.bytes == b.bytes, "\(t)")
            #expect(a.fields.map(\.name) == b.fields.map(\.name) && a.fields.map(\.bytes) == b.fields.map(\.bytes), "\(t)")
        }
    }

    @Test func layoutTableMatches() {
        #expect(Native.PackLayoutTable.version == MFPack.LayoutTable.version)
        for t in Native.GGMLType.allCases {
            #expect(Self.mirror(Native.PackLayoutTable.layout(for: t)) == MFPack.LayoutTable.layout(for: Self.mirror(t)))
            for (rows, blocks) in [(1, 1), (2560, 10), (248_320, 10), (9216, 36), (7, 3), (32, 1), (4096, 16)] {
                #expect(Self.mirror(Native.PackLayoutTable.layout(for: t, rows: rows, blocksPerRow: blocks))
                        == MFPack.LayoutTable.layout(for: Self.mirror(t), rows: rows, blocksPerRow: blocks))
            }
        }
    }

    @Test func packedOffsetMatchesOverManyShapes() {
        let layouts = [Native.TileLayout.gguf,
                       Native.TileLayout(rowsPerTile: 4, blocksPerTile: 2, planes: true),
                       Native.TileLayout(rowsPerTile: 8, blocksPerTile: 1, planes: false),
                       Native.TileLayout(rowsPerTile: 2, blocksPerTile: 4, planes: false),
                       Native.TileLayout(rowsPerTile: 16, blocksPerTile: 2, planes: true),
                       Native.TileLayout(rowsPerTile: 1, blocksPerTile: 8, planes: true)]
        let types: [Native.GGMLType] = [.f32, .f16, .bf16, .q8_0, .q4_K, .q5_K, .q6_K, .iq4_xs, .iq3_s, .iq3_xxs,
                                        .iq2_xxs, .iq2_xs, .iq2_s, .iq1_m, .q2_K]
        var rng = RefTest.SplitMix64(77)
        var checked = 0
        for layout in layouts {
            for t in types {
                let format = Native.QuantBlockFormat.of(t)!
                let mirrorFormat = MFPack.QuantBlockFormat.of(Self.mirror(t))!
                let mirrorLayout = Self.mirror(layout)
                for _ in 0..<6 {
                    let rows = layout.rowsPerTile * Int(1 + rng.next() % 6)
                    let blocksPerRow = layout.blocksPerTile * Int(1 + rng.next() % 5)
                    #expect(layout.fits(rows: rows, blocksPerRow: blocksPerRow)
                            == mirrorLayout.fits(rows: rows, blocksPerRow: blocksPerRow))
                    for _ in 0..<40 {
                        let row = Int(rng.next() % UInt64(rows))
                        let block = Int(rng.next() % UInt64(blocksPerRow))
                        let byte = Int(rng.next() % UInt64(format.bytes))
                        let a = layout.packedOffset(row: row, block: block, byteInBlock: byte,
                                                    blocksPerRow: blocksPerRow, format: format)
                        let b = mirrorLayout.packedOffset(row: row, block: block, byteInBlock: byte,
                                                          blocksPerRow: blocksPerRow, format: mirrorFormat)
                        #expect(a == b, "\(t) \(layout) r\(row) b\(block) byte\(byte)")
                        #expect(Native.TileLayout.ggufOffset(row: row, block: block, byteInBlock: byte,
                                                             blocksPerRow: blocksPerRow, format: format)
                                == MFPack.TileLayout.ggufOffset(row: row, block: block, byteInBlock: byte,
                                                                blocksPerRow: blocksPerRow, format: mirrorFormat))
                        checked += 1
                    }
                }
                // Odd shapes: fits() agrees too.
                for (rows, blocks) in [(3, 5), (layout.rowsPerTile + 1, layout.blocksPerTile), (0, 0)] {
                    #expect(layout.fits(rows: rows, blocksPerRow: blocks)
                            == mirrorLayout.fits(rows: rows, blocksPerRow: blocks))
                }
            }
        }
        #expect(checked == layouts.count * types.count * 6 * 40)
    }

    @Test func manifestEncodingsAreIdentical() throws {
        let tensors: [(String, Native.GGMLType, [Int], Native.TileLayout)] = [
            ("output_norm.weight", .f32, [2560], .gguf),
            ("token_embd.weight", .q6_K, [2560, 248_320], .gguf),
            ("blk.0.attn_qkv.weight", .q5_K, [2560, 8192], Native.TileLayout(rowsPerTile: 4, blocksPerTile: 2, planes: true)),
            ("blk.0.ssm_conv1d.weight", .f32, [4, 8192], .gguf),
            ("blk.3.ffn_gate.weight", .iq4_xs, [2560, 9216], Native.TileLayout(rowsPerTile: 8, blocksPerTile: 1, planes: false)),
        ]
        var offset: UInt64 = 0
        var native: [Native.PackManifest.Tensor] = []
        var mirror: [MFPack.Manifest.Tensor] = []
        for (i, (name, type, shape, layout)) in tensors.enumerated() {
            let format = Native.QuantBlockFormat.of(type)!
            let bytes = UInt64(shape.reduce(1, *) / format.elements * format.bytes)
            let src = String(repeating: String(i), count: 64), packed = String(repeating: "f", count: 64)
            native.append(.init(name: name, type: type, shape: shape, layout: layout, offset: offset, byteCount: bytes,
                                sourceSHA256: src, packedSHA256: packed))
            mirror.append(.init(name: name, type: Self.mirror(type), shape: shape, layout: Self.mirror(layout),
                                offset: offset, byteCount: bytes, sourceSHA256: src, packedSHA256: packed))
            offset = (offset + bytes + 4095) / 4096 * 4096
        }
        let c = Self.quickConfig
        let mc = MFPack.Qwen35Config(
            blockCount: c.blockCount, hiddenSize: c.hiddenSize, ffnSize: c.ffnSize, headCount: c.headCount,
            kvHeadCount: c.kvHeadCount, keyLength: c.keyLength, valueLength: c.valueLength,
            rmsNormEpsilon: c.rmsNormEpsilon, ropeFreqBase: c.ropeFreqBase, ropeDimensionCount: c.ropeDimensionCount,
            ropeSections: c.ropeSections, fullAttentionInterval: c.fullAttentionInterval,
            ssmConvKernel: c.ssmConvKernel, ssmStateSize: c.ssmStateSize, ssmGroupCount: c.ssmGroupCount,
            ssmTimeStepRank: c.ssmTimeStepRank, ssmInnerSize: c.ssmInnerSize, contextLength: c.contextLength,
            nextnPredictLayers: c.nextnPredictLayers, vocabSize: c.vocabSize, tiedEmbeddings: c.tiedEmbeddings)
        let metadata = ["general.name": "Qwen3.5-4B", "qwen35.rope.freq_base": "1e+07", "x/y": "a\"b\\c\u{1}é"]
        let a = Native.PackManifest(
            architecture: "qwen35", label: "quick",
            source: .init(fileName: "Qwen3.5-4B-UD-Q4_K_XL.gguf", byteCount: 2_912_109_728,
                          sha256: String(repeating: "ab", count: 32), ggufVersion: 3),
            config: c, layoutTableVersion: 1, tensors: native, metadata: metadata)
        let b = MFPack.Manifest(
            architecture: "qwen35", label: "quick",
            source: .init(fileName: "Qwen3.5-4B-UD-Q4_K_XL.gguf", byteCount: 2_912_109_728,
                          sha256: String(repeating: "ab", count: 32), ggufVersion: 3),
            config: mc, layoutTableVersion: 1, tensors: mirror, metadata: metadata)
        let ea = try Self.encode(a), eb = try Self.encode(b)
        #expect(ea == eb)
        #expect(try MFPack.encoder().encode(b) == eb)
        // Each side decodes the other's bytes to an equal value.
        #expect(try JSONDecoder().decode(Native.PackManifest.self, from: eb) == a)
        #expect(try JSONDecoder().decode(MFPack.Manifest.self, from: ea) == b)
        // Config alone, including the Deep shape (nextn, untied).
        var deep = c
        deep.blockCount = 64
        deep.nextnPredictLayers = 1
        deep.tiedEmbeddings = false
        deep.rmsNormEpsilon = 9.999999974752427e-07
        var mdeep = mc
        mdeep.blockCount = 64
        mdeep.nextnPredictLayers = 1
        mdeep.tiedEmbeddings = false
        mdeep.rmsNormEpsilon = 9.999999974752427e-07
        #expect(try Self.encode(deep) == Self.encode(mdeep))
    }

    @Test func tokenizerAndReceiptEncodingsAreIdentical() throws {
        for special in [nil, ["eot": Int32(248_046), "fim_pre": 5, "seperator": 9]] as [[String: Int32]?] {
            for flags in [(Bool?.none, Bool?.none), (false, true)] {
                let a = Native.TokenizerData(model: "gpt2", pre: "qwen35", tokens: ["a", "Ġb", "<|im_end|>", "\"", "/"],
                                             tokenTypes: [1, 1, 3, 1, 1], merges: ["Ġ b", "a b"], bosTokenID: nil,
                                             eosTokenID: 2, paddingTokenID: 2, addBOS: flags.0, addEOS: flags.1,
                                             specialTokenIDs: special)
                let b = MFPack.TokenizerData(model: "gpt2", pre: "qwen35", tokens: ["a", "Ġb", "<|im_end|>", "\"", "/"],
                                             tokenTypes: [1, 1, 3, 1, 1], merges: ["Ġ b", "a b"], bosTokenID: nil,
                                             eosTokenID: 2, paddingTokenID: 2, addBOS: flags.0, addEOS: flags.1,
                                             specialTokenIDs: special)
                #expect(try Self.encode(a) == Self.encode(b))
                #expect(try JSONDecoder().decode(Native.TokenizerData.self, from: Self.encode(b)) == a)
            }
        }
        let r = MFPack.Receipt(format: "mfpack", formatVersion: 1, label: "quick", writer: MFPackWriter.writerName,
                               sourceSHA256: "s", manifestSHA256: "m", tensorCount: 3, weightsByteCount: 99,
                               verifiedAt: "2026-11-19T00:00:00Z", tokenizerSHA256: "t", chatTemplateSHA256: "c")
        let n = Native.PackReceipt(format: "mfpack", formatVersion: 1, label: "quick", writer: MFPackWriter.writerName,
                                   sourceSHA256: "s", manifestSHA256: "m", tensorCount: 3, weightsByteCount: 99,
                                   verifiedAt: "2026-11-19T00:00:00Z", tokenizerSHA256: "t", chatTemplateSHA256: "c")
        #expect(try Self.encode(r) == Self.encode(n))
        // The reference store's receipt type reads the production receipt.
        let ref = try JSONDecoder().decode(Native.ReferencePackWriter.Receipt.self, from: Self.encode(r))
        #expect(ref.label == "quick" && ref.tensorCount == 3 && ref.weightsByteCount == 99)
    }
}
