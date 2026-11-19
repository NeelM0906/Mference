import Foundation
import Testing
@testable import Mference

/// Invariants of the normative definitions in `Native/Contracts` (no model,
/// no GPU work).
@Suite("NativeContractTests")
struct NativeContractTests {
    static let quick = Native.Qwen35Config(
        blockCount: 32, hiddenSize: 2560, ffnSize: 9216, headCount: 16, kvHeadCount: 4,
        keyLength: 256, valueLength: 256, rmsNormEpsilon: 1e-6, ropeFreqBase: 1e7,
        ropeDimensionCount: 64, ropeSections: [11, 11, 10, 0], fullAttentionInterval: 4,
        ssmConvKernel: 4, ssmStateSize: 128, ssmGroupCount: 16, ssmTimeStepRank: 32,
        ssmInnerSize: 4096, contextLength: 262_144, nextnPredictLayers: 0,
        vocabSize: 248_320, tiedEmbeddings: true)

    @Test func blockFormatsSumToBlockBytes() throws {
        for type in Native.GGMLType.allCases {
            guard let format = Native.QuantBlockFormat.of(type) else { continue }
            #expect(format.fields.reduce(0) { $0 + $1.bytes } == format.bytes, "\(type)")
        }
        #expect(Native.QuantBlockFormat.of(.q4_K)?.bytes == 144)
        #expect(Native.QuantBlockFormat.of(.q5_K)?.bytes == 176)
        #expect(Native.QuantBlockFormat.of(.q6_K)?.bytes == 210)
        #expect(Native.QuantBlockFormat.of(.q8_0)?.bytes == 34)
        #expect(Native.QuantBlockFormat.of(.iq4_xs)?.bytes == 136)
        #expect(Native.GGMLType.iq4_xs.rawValue == 23)
        #expect(Native.GGMLType.bf16.rawValue == 30)
    }

    @Test(arguments: [
        Native.TileLayout.gguf,
        Native.TileLayout(rowsPerTile: 4, blocksPerTile: 2, planes: false),
        Native.TileLayout(rowsPerTile: 8, blocksPerTile: 1, planes: true),
        Native.TileLayout(rowsPerTile: 2, blocksPerTile: 5, planes: true),
    ])
    func tileLayoutIsABijection(layout: Native.TileLayout) throws {
        for type in [Native.GGMLType.q4_K, .q6_K, .q8_0, .iq4_xs] {
            let format = try #require(Native.QuantBlockFormat.of(type))
            let rows = 8, blocksPerRow = 10
            guard layout.fits(rows: rows, blocksPerRow: blocksPerRow) else { continue }
            let total = rows * blocksPerRow * format.bytes
            var seen = [Bool](repeating: false, count: total)
            for r in 0..<rows {
                for b in 0..<blocksPerRow {
                    for byte in 0..<format.bytes {
                        let o = layout.packedOffset(row: r, block: b, byteInBlock: byte,
                                                    blocksPerRow: blocksPerRow, format: format)
                        #expect(o >= 0 && o < total)
                        #expect(!seen[o], "\(type) \(layout) collides at \(o)")
                        seen[o] = true
                    }
                }
            }
            #expect(seen.allSatisfy { $0 })
        }
    }

    @Test func ggufLayoutIsTheIdentity() throws {
        let format = try #require(Native.QuantBlockFormat.of(.q5_K))
        for r in 0..<3 {
            for b in 0..<4 {
                for byte in [0, 3, 175] {
                    #expect(Native.TileLayout.gguf.packedOffset(
                        row: r, block: b, byteInBlock: byte, blocksPerRow: 4, format: format)
                        == Native.TileLayout.ggufOffset(row: r, block: b, byteInBlock: byte,
                                                        blocksPerRow: 4, format: format))
                }
            }
        }
    }

    @Test func quickLayerKinds() {
        let config = Self.quick
        #expect(config.fullAttentionLayers == [3, 7, 11, 15, 19, 23, 27, 31])
        #expect(config.gatedDeltaNetLayers.count == 24)
        #expect(config.ssmConvChannels == 8192)
    }

    @Test func deepCountsMainBlocksOnly() {
        // Deep's GGUF: block_count 65 with nextn_predict_layers 1 -> 64 main blocks.
        var deep = Self.quick
        deep.blockCount = 65 - 1
        deep.nextnPredictLayers = 1
        #expect(deep.fullAttentionLayers.count == 16)
        #expect(deep.gatedDeltaNetLayers.count == 48)
        #expect(deep.fullAttentionLayers.last == 63)
    }

    @Test func packLayoutTableV1IsIdentity() {
        for type in Native.GGMLType.allCases {
            #expect(Native.PackLayoutTable.layout(for: type) == .gguf)
            #expect(Native.PackLayoutTable.layout(for: type, rows: 32, blocksPerRow: 10) == .gguf)
        }
        #expect(Native.PackLayoutTable.version == 1)
        // The fallback rule rests on `fits`: a tile that does not divide falls back.
        #expect(!Native.TileLayout(rowsPerTile: 4, blocksPerTile: 4, planes: false)
            .fits(rows: 32, blocksPerRow: 10))
    }

    @Test func tokenizerDataDecodesWithAndWithoutSpecialIDs() throws {
        let legacy = #"{"model":"gpt2","pre":"qwen35","tokens":["a"],"tokenTypes":[1],"merges":[]}"#
        let old = try JSONDecoder().decode(Native.TokenizerData.self, from: Data(legacy.utf8))
        #expect(old.specialTokenIDs == nil)
        var data = old
        data.specialTokenIDs = ["eot": 248046, "fim_pre": 248060]
        let round = try JSONDecoder().decode(Native.TokenizerData.self,
                                             from: JSONEncoder().encode(data))
        #expect(round == data)
    }

    @Test func quickKVPageGeometry() {
        let f16 = Native.KVPageGeometry(attentionLayers: 8, kvHeads: 4, headDim: 256)
        #expect(f16.pageBytes == 2 * 1024 * 1024)
        #expect(f16.pageBytes / f16.tokensPerPage == 32 * 1024)
        #expect(f16.valueOffset(attnLayer: 7, kv: 1, kvHead: 3, token: 63, dim: 255)
                == f16.pageBytes - 2)
        let int8 = Native.KVPageGeometry(attentionLayers: 8, kvHeads: 4, headDim: 256, precision: .int8)
        #expect(int8.pageBytes == 1024 * 1024 + 8 * 2 * 4 * 64 * 2)
        #expect(int8.scaleOffset(attnLayer: 7, kv: 1, kvHead: 3, token: 63) == int8.pageBytes - 2)
    }

    @Test func quickRecurrentLayout() {
        let layout = Native.RecurrentLayout(convKernel: 4, convChannels: 8192, valueHeads: 32,
                                            keyDim: 128, valueDim: 128)
        #expect(layout.convBytes == 3 * 8192 * 4)
        #expect(layout.ssmOffset % 256 == 0 && layout.ssmOffset >= layout.convBytes)
        #expect(layout.ssmBytes == 32 * 128 * 128 * 4)
    }

    @Test func manifestRoundTripsThroughJSON() throws {
        let manifest = Native.PackManifest(
            architecture: "qwen35", label: "quick",
            source: .init(fileName: "x.gguf", byteCount: 1, sha256: "00", ggufVersion: 3),
            config: Self.quick, layoutTableVersion: Native.PackLayoutTable.version,
            tensors: [.init(name: "blk.0.attn_qkv.weight", type: .q5_K, shape: [2560, 8192],
                            layout: .init(rowsPerTile: 8, blocksPerTile: 2, planes: true),
                            offset: 4096, byteCount: 8192 * 10 * 176,
                            sourceSHA256: "aa", packedSHA256: "bb")],
            metadata: ["general.name": "Qwen3.5-4B"])
        let data = try JSONEncoder().encode(manifest)
        #expect(try JSONDecoder().decode(Native.PackManifest.self, from: data) == manifest)
    }
}
