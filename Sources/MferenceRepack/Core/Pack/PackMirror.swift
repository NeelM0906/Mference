import Foundation

/// The `.mfpack` v1 format as the production writer sees it
/// (`docs/MFERENCE_PACK.md`).
///
/// `MferenceRepackCore` cannot import `Mference`, so these types mirror the
/// frozen contracts in `Sources/Mference/Native/Contracts/`
/// (`NativeQuantFormat`, `NativeTileLayout`, `NativePackLayout`,
/// `NativePackManifest`, `NativeQwen35Config`) field for field.
/// `PackMirrorPinTests` (in `MferenceTestsCore`, which links both targets)
/// pins them: identical JSON encodings, identical block formats, identical
/// `packedOffset` over many shapes, and an identical layout table. Change the
/// contract and this file together.
public enum MFPack {
    /// ggml tensor types (`enum ggml_type` at the pinned llama.cpp b1b770d9),
    /// as GGUF stores them. Mirrors `Native.GGMLType`.
    public enum GGMLType: UInt32, Codable, Sendable, CaseIterable {
        case f32 = 0
        case f16 = 1
        case q4_0 = 2
        case q4_1 = 3
        case q5_0 = 6
        case q5_1 = 7
        case q8_0 = 8
        case q8_1 = 9
        case q2_K = 10
        case q3_K = 11
        case q4_K = 12
        case q5_K = 13
        case q6_K = 14
        case q8_K = 15
        case iq2_xxs = 16
        case iq2_xs = 17
        case iq3_xxs = 18
        case iq1_s = 19
        case iq4_nl = 20
        case iq3_s = 21
        case iq2_s = 22
        case iq4_xs = 23
        case i8 = 24
        case i16 = 25
        case i32 = 26
        case i64 = 27
        case f64 = 28
        case iq1_m = 29
        case bf16 = 30
    }

    /// One field of a quant block, in ggml struct order. Mirrors `Native.BlockField`.
    public struct BlockField: Equatable, Sendable {
        public let name: String
        public let bytes: Int
        public init(name: String, bytes: Int) {
            self.name = name
            self.bytes = bytes
        }
    }

    /// Block geometry of a ggml type. Mirrors `Native.QuantBlockFormat`.
    public struct QuantBlockFormat: Equatable, Sendable {
        public let type: GGMLType
        public let elements: Int
        public let bytes: Int
        public let fields: [BlockField]

        public init(type: GGMLType, elements: Int, bytes: Int, fields: [BlockField]) {
            self.type = type
            self.elements = elements
            self.bytes = bytes
            self.fields = fields
        }

        public static func of(_ type: GGMLType) -> QuantBlockFormat? {
            func f(_ n: String, _ b: Int) -> BlockField { BlockField(name: n, bytes: b) }
            switch type {
            case .f32: return .init(type: type, elements: 1, bytes: 4, fields: [f("v", 4)])
            case .f16: return .init(type: type, elements: 1, bytes: 2, fields: [f("v", 2)])
            case .bf16: return .init(type: type, elements: 1, bytes: 2, fields: [f("v", 2)])
            case .q8_0: return .init(type: type, elements: 32, bytes: 34,
                                     fields: [f("d", 2), f("qs", 32)])
            case .q2_K: return .init(type: type, elements: 256, bytes: 84,
                                     fields: [f("scales", 16), f("qs", 64), f("d", 2), f("dmin", 2)])
            case .q4_K: return .init(type: type, elements: 256, bytes: 144,
                                     fields: [f("d", 2), f("dmin", 2), f("scales", 12), f("qs", 128)])
            case .q5_K: return .init(type: type, elements: 256, bytes: 176,
                                     fields: [f("d", 2), f("dmin", 2), f("scales", 12), f("qh", 32), f("qs", 128)])
            case .q6_K: return .init(type: type, elements: 256, bytes: 210,
                                     fields: [f("ql", 128), f("qh", 64), f("scales", 16), f("d", 2)])
            case .iq4_xs: return .init(type: type, elements: 256, bytes: 136,
                                       fields: [f("d", 2), f("scales_h", 2), f("scales_l", 4), f("qs", 128)])
            case .iq3_xxs: return .init(type: type, elements: 256, bytes: 98,
                                        fields: [f("d", 2), f("qs", 96)])
            case .iq3_s: return .init(type: type, elements: 256, bytes: 110,
                                      fields: [f("d", 2), f("qs", 64), f("qh", 8), f("signs", 32), f("scales", 4)])
            case .iq2_xxs: return .init(type: type, elements: 256, bytes: 66,
                                        fields: [f("d", 2), f("qs", 64)])
            case .iq2_xs: return .init(type: type, elements: 256, bytes: 74,
                                       fields: [f("d", 2), f("qs", 64), f("scales", 8)])
            case .iq2_s: return .init(type: type, elements: 256, bytes: 82,
                                      fields: [f("d", 2), f("qs", 64), f("qh", 8), f("scales", 8)])
            case .iq1_m: return .init(type: type, elements: 256, bytes: 56,
                                      fields: [f("qs", 32), f("qh", 16), f("scales", 8)])
            default: return nil
            }
        }
    }

    /// How a matrix's GGUF blocks are arranged in the pack. Mirrors
    /// `Native.TileLayout`; `packedOffset` is the normative permutation.
    public struct TileLayout: Codable, Hashable, Sendable {
        public var rowsPerTile: Int
        public var blocksPerTile: Int
        public var planes: Bool

        public init(rowsPerTile: Int, blocksPerTile: Int, planes: Bool) {
            self.rowsPerTile = rowsPerTile
            self.blocksPerTile = blocksPerTile
            self.planes = planes
        }

        public static let gguf = TileLayout(rowsPerTile: 1, blocksPerTile: 1, planes: false)

        public func fits(rows: Int, blocksPerRow: Int) -> Bool {
            rowsPerTile >= 1 && blocksPerTile >= 1
                && rows % rowsPerTile == 0 && blocksPerRow % blocksPerTile == 0
        }

        public func packedOffset(row: Int, block: Int, byteInBlock: Int,
                                 blocksPerRow: Int, format: QuantBlockFormat) -> Int {
            let r = rowsPerTile, k = blocksPerTile, bb = format.bytes
            let chunksPerRow = blocksPerRow / k
            let tileIndex = (row / r) * chunksPerRow + (block / k)
            let tileBytes = r * k * bb
            let blockInTile = (row % r) * k + (block % k)
            let tileBase = tileIndex * tileBytes
            guard planes else { return tileBase + blockInTile * bb + byteInBlock }
            var fieldStart = 0
            var planeBase = 0
            for field in format.fields {
                if byteInBlock < fieldStart + field.bytes {
                    return tileBase + planeBase + blockInTile * field.bytes + (byteInBlock - fieldStart)
                }
                fieldStart += field.bytes
                planeBase += r * k * field.bytes
            }
            preconditionFailure("byteInBlock \(byteInBlock) outside a \(bb)-byte block")
        }

        public static func ggufOffset(row: Int, block: Int, byteInBlock: Int,
                                      blocksPerRow: Int, format: QuantBlockFormat) -> Int {
            (row * blocksPerRow + block) * format.bytes + byteInBlock
        }
    }

    /// The frozen per-type layout table. Mirrors `Native.PackLayoutTable`.
    public enum LayoutTable {
        public static let version = 1

        public static func layout(for type: GGMLType) -> TileLayout {
            switch type {
            default: return .gguf
            }
        }

        public static func layout(for type: GGMLType, rows: Int, blocksPerRow: Int) -> TileLayout {
            let preferred = layout(for: type)
            return preferred.fits(rows: rows, blocksPerRow: blocksPerRow) ? preferred : .gguf
        }
    }

    /// `qwen35.*` hyperparameters. Mirrors `Native.Qwen35Config` (stored
    /// properties, names and types; the encoding is pinned by a test).
    public struct Qwen35Config: Codable, Equatable, Sendable {
        public var blockCount: Int
        public var hiddenSize: Int
        public var ffnSize: Int
        public var headCount: Int
        public var kvHeadCount: Int
        public var keyLength: Int
        public var valueLength: Int
        public var rmsNormEpsilon: Float
        public var ropeFreqBase: Float
        public var ropeDimensionCount: Int
        public var ropeSections: [Int]
        public var fullAttentionInterval: Int
        public var ssmConvKernel: Int
        public var ssmStateSize: Int
        public var ssmGroupCount: Int
        public var ssmTimeStepRank: Int
        public var ssmInnerSize: Int
        public var contextLength: Int
        public var nextnPredictLayers: Int
        public var vocabSize: Int
        public var tiedEmbeddings: Bool

        public init(blockCount: Int, hiddenSize: Int, ffnSize: Int, headCount: Int,
                    kvHeadCount: Int, keyLength: Int, valueLength: Int, rmsNormEpsilon: Float,
                    ropeFreqBase: Float, ropeDimensionCount: Int, ropeSections: [Int],
                    fullAttentionInterval: Int, ssmConvKernel: Int, ssmStateSize: Int,
                    ssmGroupCount: Int, ssmTimeStepRank: Int, ssmInnerSize: Int,
                    contextLength: Int, nextnPredictLayers: Int, vocabSize: Int,
                    tiedEmbeddings: Bool) {
            self.blockCount = blockCount
            self.hiddenSize = hiddenSize
            self.ffnSize = ffnSize
            self.headCount = headCount
            self.kvHeadCount = kvHeadCount
            self.keyLength = keyLength
            self.valueLength = valueLength
            self.rmsNormEpsilon = rmsNormEpsilon
            self.ropeFreqBase = ropeFreqBase
            self.ropeDimensionCount = ropeDimensionCount
            self.ropeSections = ropeSections
            self.fullAttentionInterval = fullAttentionInterval
            self.ssmConvKernel = ssmConvKernel
            self.ssmStateSize = ssmStateSize
            self.ssmGroupCount = ssmGroupCount
            self.ssmTimeStepRank = ssmTimeStepRank
            self.ssmInnerSize = ssmInnerSize
            self.contextLength = contextLength
            self.nextnPredictLayers = nextnPredictLayers
            self.vocabSize = vocabSize
            self.tiedEmbeddings = tiedEmbeddings
        }
    }

    /// `manifest.json`. Mirrors `Native.PackManifest`.
    public struct Manifest: Codable, Equatable, Sendable {
        public var format: String
        public var formatVersion: Int
        public var architecture: String
        public var label: String
        public var source: Source
        public var config: Qwen35Config
        public var layoutTableVersion: Int
        public var tensors: [Tensor]
        public var metadata: [String: String]

        public init(format: String = "mfpack", formatVersion: Int = 1, architecture: String,
                    label: String, source: Source, config: Qwen35Config,
                    layoutTableVersion: Int, tensors: [Tensor], metadata: [String: String]) {
            self.format = format
            self.formatVersion = formatVersion
            self.architecture = architecture
            self.label = label
            self.source = source
            self.config = config
            self.layoutTableVersion = layoutTableVersion
            self.tensors = tensors
            self.metadata = metadata
        }

        public struct Source: Codable, Equatable, Sendable {
            public var fileName: String
            public var byteCount: UInt64
            public var sha256: String
            public var ggufVersion: Int
            public init(fileName: String, byteCount: UInt64, sha256: String, ggufVersion: Int) {
                self.fileName = fileName
                self.byteCount = byteCount
                self.sha256 = sha256
                self.ggufVersion = ggufVersion
            }
        }

        public struct Tensor: Codable, Equatable, Sendable {
            public var name: String
            public var type: GGMLType
            public var shape: [Int]
            public var layout: TileLayout
            public var offset: UInt64
            public var byteCount: UInt64
            public var sourceSHA256: String
            public var packedSHA256: String
            public init(name: String, type: GGMLType, shape: [Int], layout: TileLayout,
                        offset: UInt64, byteCount: UInt64, sourceSHA256: String, packedSHA256: String) {
                self.name = name
                self.type = type
                self.shape = shape
                self.layout = layout
                self.offset = offset
                self.byteCount = byteCount
                self.sourceSHA256 = sourceSHA256
                self.packedSHA256 = packedSHA256
            }
        }
    }

    /// `tokenizer.json`. Mirrors `Native.TokenizerData`.
    public struct TokenizerData: Codable, Equatable, Sendable {
        public var model: String
        public var pre: String
        public var tokens: [String]
        public var tokenTypes: [Int32]
        public var merges: [String]
        public var bosTokenID: Int32?
        public var eosTokenID: Int32?
        public var paddingTokenID: Int32?
        public var addBOS: Bool?
        public var addEOS: Bool?
        public var specialTokenIDs: [String: Int32]?
        public init(model: String, pre: String, tokens: [String], tokenTypes: [Int32],
                    merges: [String], bosTokenID: Int32?, eosTokenID: Int32?,
                    paddingTokenID: Int32?, addBOS: Bool?, addEOS: Bool?,
                    specialTokenIDs: [String: Int32]? = nil) {
            self.model = model
            self.pre = pre
            self.tokens = tokens
            self.tokenTypes = tokenTypes
            self.merges = merges
            self.bosTokenID = bosTokenID
            self.eosTokenID = eosTokenID
            self.paddingTokenID = paddingTokenID
            self.addBOS = addBOS
            self.addEOS = addEOS
            self.specialTokenIDs = specialTokenIDs
        }
    }

    /// `verified-install.json`, written last. The first nine fields are the
    /// reference writer's receipt (so the reference store opens these packs);
    /// the last two let the runtime store check the two metadata files.
    /// Mirrors `Native.PackReceipt`.
    public struct Receipt: Codable, Equatable, Sendable {
        public var format: String
        public var formatVersion: Int
        public var label: String
        public var writer: String
        public var sourceSHA256: String
        public var manifestSHA256: String
        public var tensorCount: Int
        public var weightsByteCount: UInt64
        public var verifiedAt: String
        public var tokenizerSHA256: String?
        public var chatTemplateSHA256: String?

        public init(format: String, formatVersion: Int, label: String, writer: String, sourceSHA256: String,
                    manifestSHA256: String, tensorCount: Int, weightsByteCount: UInt64, verifiedAt: String,
                    tokenizerSHA256: String? = nil, chatTemplateSHA256: String? = nil) {
            self.format = format
            self.formatVersion = formatVersion
            self.label = label
            self.writer = writer
            self.sourceSHA256 = sourceSHA256
            self.manifestSHA256 = manifestSHA256
            self.tensorCount = tensorCount
            self.weightsByteCount = weightsByteCount
            self.verifiedAt = verifiedAt
            self.tokenizerSHA256 = tokenizerSHA256
            self.chatTemplateSHA256 = chatTemplateSHA256
        }
    }

    /// The JSON encoder every pack file uses (sorted keys, pretty printed),
    /// as the contract requires, so the bytes are deterministic.
    public static func encoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys, .prettyPrinted]
        return e
    }

    public static let pageAlignment: UInt64 = 4096
}
