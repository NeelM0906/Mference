import Testing
import Foundation
@testable import Mference

@Suite struct ManifestReaderTests {

    /// Build a manifest dictionary for a 2-layer toy ArchConfig and write it
    /// into a temp directory. Returns the directory URL and the toy config.
    static func writeToyManifest(_ overrides: [String: Any] = [:],
                                 flags: [String: Bool] = ["streamingPresent": true,
                                                          "turboQuantKV": false,
                                                          "aneSharedExpert": false],
                                 archOverrides: [String: Any] = [:],
                                 filesOverride: [String: [String: Any]]? = nil,
                                 config: ArchConfig = .gemma4Toy()) throws
                                 -> (URL, ArchConfig) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        let toy = config
        var archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize,
            "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads,
            "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim,
            "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize,
            "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta,
            "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers,
            "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
        ]
        for (k, v) in archOverrides { archDict[k] = v }

        var files: [String: [String: Any]]
        if let f = filesOverride {
            files = f
        } else {
            files = [
                "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
                "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            ]
            for L in 0..<toy.numLayers {
                files["packed_experts/layer_\(L).bin"] = ["size": 16384, "sha256": String(repeating: "0", count: 64)]
            }
        }

        var root: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": flags,
            "modelID": "toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": 16384,
        ]
        for (k, v) in overrides { root[k] = v }

        let data = try JSONSerialization.data(withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return (dir, toy)
    }

    static func quant(sharedExpertBits: Int = 4,
                      routerBits: Int = 8,
                      routedExpertBits: Int = 4) -> [String: Any] {
        func slot(_ bits: Int) -> [String: Any] {
            [
                "weightBits": bits,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": Quantization.groupSize,
            ]
        }
        return [
            "embedding": slot(4),
            "attention": slot(4),
            "router": slot(routerBits),
            "sharedExpert": slot(sharedExpertBits),
            "routedExpert": slot(routedExpertBits),
        ]
    }

    /// Manifest `arch` values for the DeepSeek-V4 toy baseline. The family
    /// booleans are always written (a dsv4 manifest never relies on the
    /// Gemma defaults); the ca*/hc*/hash/router extension fields can be
    /// withheld to exercise the absent-fields-mean-zeroed-defaults path.
    static func deepseekArchOverrides(includeExtensionFields: Bool = true) -> [String: Any] {
        var overrides: [String: Any] = [
            "family": "deepseekV4Flash",
            "attnOutputGate": false,
            "attentionScale": 0.125,
            "embeddingScaledBySqrtHidden": false,
            "routerScaled": false,
            "ffnSandwichNorms": false,
            "sharedExpertGated": false,
            "ropeNeoxSubdim": false,
        ]
        if includeExtensionFields {
            overrides["caQLoraRank"] = 64
            overrides["caOLoraRank"] = 64
            overrides["caOGroups"] = 2
            overrides["caRopeHeadDim"] = 8
            overrides["caIndexNHeads"] = 2
            overrides["caIndexHeadDim"] = 64
            overrides["caIndexTopK"] = 16
            overrides["caCSACompressRate"] = 4
            overrides["caHCACompressRate"] = 128
            overrides["caCompressRopeTheta"] = 160_000.0
            overrides["hcMult"] = 2
            overrides["hcSinkhornIters"] = 4
            overrides["hcEps"] = 1e-6
            overrides["numHashRoutedLayers"] = 1
            overrides["routerScoringFunc"] = "sqrtsoftplus"
            overrides["routedScalingFactor"] = 1.5
            overrides["swigluLimit"] = 10.0
        }
        return overrides
    }

    @Test func loadsValidManifest() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.magic == "GTURBO")
        #expect(m.numLayers == toy.numLayers)
        #expect(m.expertStride == 16384)
    }

    @Test func missingManifestThrowsPartialInstall() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .gemma4Toy())
        } throws: { error in
            if case ModelError.partialInstall = error { return true }
            return false
        }
    }

    @Test func oversizedManifestRejectsBeforeDecode() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try Data(repeating: 0x20, count: 64).write(to: manifestURL)

        #expect {
            _ = try ManifestReader.load(directoryURL: dir,
                                        expecting: toy,
                                        maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }

    @Test func wrongMagicThrowsNotAGTurboDirectory() throws {
        let (dir, toy) = try Self.writeToyManifest(["magic": "NOT_GTURBO"])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.notAGTurboDirectory) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        }
    }

    @Test func versionTwoThrowsUnsupportedVersion() throws {
        let (dir, toy) = try Self.writeToyManifest(["versionMajor": 2])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unsupportedVersion(let maj, _) = error { return maj == 2 }
            return false
        }
    }

    @Test func unknownFlagThrowsUnknownFlag() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "newFangledOption": true])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unknownFlag(let n) = error { return n == "newFangledOption" }
            return false
        }
    }

    @Test func removedTurboQuantFlagIsRejected() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "turboQuantKV": true,
                                                           "aneSharedExpert": false])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("removed TurboQuant KV")
        }
    }

    @Test func productionManifestRequiresQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("manifest.quant is required")
        }
    }

    @Test func productionManifestAcceptsInt4SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 4)
    }

    @Test func productionManifestAcceptsHistoricalInt8SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 8)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 8)
    }

    @Test func productionManifestAcceptsInt2RoutedExpert() throws {
        // The DeepSeek-V4-Flash dynamic-quant checkpoint ships Q2 routed
        // experts under a Q4 core.
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(routedExpertBits: 2)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.routedExpert.weightBits == 2)
    }

    @Test func productionManifestRejectsInt3RoutedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(routedExpertBits: 3)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization")
        }
    }

    @Test func deepseekArchExtensionFieldsValidate() throws {
        let (dir, config) = try Self.writeToyManifest(
            archOverrides: Self.deepseekArchOverrides(),
            config: .deepseekV4FlashToy())
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.arch.family == "deepseekV4Flash")
        #expect(manifest.arch.caQLoraRank == 64)
        #expect(manifest.arch.caHCACompressRate == 128)
        #expect(manifest.arch.hcMult == 2)
        #expect(manifest.arch.numHashRoutedLayers == 1)
        #expect(manifest.arch.routerScoringFunc == "sqrtsoftplus")
        #expect(manifest.arch.routedScalingFactor == 1.5)
        #expect(manifest.arch.swigluLimit == 10.0)
    }

    @Test func deepseekManifestMissingExtensionFieldsThrowsArchMismatch() throws {
        // Absent extension fields decode as the zeroed defaults, which
        // cannot match a family whose baseline carries real values.
        let (dir, config) = try Self.writeToyManifest(
            archOverrides: Self.deepseekArchOverrides(includeExtensionFields: false),
            config: .deepseekV4FlashToy())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == "caQLoraRank"
        }
    }

    @Test func productionManifestRejectsUnsupportedQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 3, routerBits: 4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization")
        }
    }

    @Test func archMismatchThrowsArchMismatch() throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: ["hiddenSize": 4096])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == "hiddenSize"
        }
    }

    @Test func nonPageAlignedExpertStrideThrows() throws {
        let (dir, toy) = try Self.writeToyManifest(["expertStride": 1024])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.expertStrideNotPageAligned = error { return true }
            return false
        }
    }

    @Test func missingLayerFileThrowsMissingFile() throws {
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            // intentionally do not list layer_0.bin or layer_1.bin
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.missingFile = error { return true }
            return false
        }
    }

    @Test func acceptsZeroPaddedLayerFilenames() throws {
        // Writer emits packed_experts/layer_%02d.bin; loader should accept either form.
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_00.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_01.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.numLayers == toy.numLayers)
    }
}

extension ArchConfig {
    /// Tiny baseline used across the loader tests. 2 layers (both full), hidden 64,
    /// vocab 1024, 8 experts. Numbers are intentionally toy.
    static func gemma4Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 256,
            moeIntermediateSize: 128,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 256,
            finalLogitSoftcap: 30.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 1_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 2,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 1],
            hiddenActivation: "gelu_pytorch_tanh"
        )
    }

    /// Tiny DeepSeek-V4-Flash baseline: 2 layers (CSA then HCA), hidden 128,
    /// 8 experts (layer 0 hash-routed), shared-KV MQA with the low-rank
    /// attention extensions and 2 mHC streams. Numbers are intentionally toy.
    static func deepseekV4FlashToy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 128,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHeads: 2,
            numKVHeads: 1,
            numFullKVHeads: 1,
            headDim: 64,
            fullHeadDim: 64,
            vocabSize: 256,
            slidingWindow: 32,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 10_000.0,
            partialRotaryFactor: 0.125,
            numLayers: 2,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: false,
            attentionKEqV: true,
            fullAttentionLayerMask: [3, 4],
            hiddenActivation: "silu",
            family: .deepseekV4Flash,
            attnOutputGate: false,
            attentionScale: 0.125,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            compressedAttention: CompressedAttentionConfig(
                qLoraRank: 64, oLoraRank: 64, oGroups: 2,
                ropeHeadDim: 8,
                indexNHeads: 2, indexHeadDim: 64, indexTopK: 16,
                csaCompressRate: 4, hcaCompressRate: 128,
                compressRopeTheta: 160_000.0),
            hyperConnections: HyperConnectionConfig(
                mult: 2, sinkhornIters: 4, eps: 1e-6),
            numHashRoutedLayers: 1,
            routerScoringFunc: "sqrtsoftplus",
            routedScalingFactor: 1.5,
            swigluLimit: 10.0
        )
    }
}
