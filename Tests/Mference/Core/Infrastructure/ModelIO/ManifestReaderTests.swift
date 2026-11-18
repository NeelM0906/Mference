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
                                 includesFamily: Bool = true,
                                 filesOverride: [String: [String: Any]]? = nil,
                                 config: ArchConfig = .defaultsToy()) throws
                                 -> (URL, ArchConfig) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        let toy = config
        var archDict: [String: Any] = [
            "family": toy.family.rawValue,
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
        // `arch.family` comes from the config unless the caller writes it
        // (or deliberately omits it) through `archOverrides`.
        if !includesFamily { archDict.removeValue(forKey: "family") }
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
            _ = try ManifestReader.load(directoryURL: dir, expecting: .defaultsToy())
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
        let (dir, config) = try Self.writeToyManifest(config: .productionShapedDefaults())
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
            config: .productionShapedDefaults())
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 4)
    }

    @Test func productionManifestAcceptsHistoricalInt8SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 8)],
            config: .productionShapedDefaults())
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 8)
    }

    @Test func productionManifestAcceptsInt2RoutedExpert() throws {
        // The MoE runtime dispatches on the routed-expert width; Q2 under a
        // Q4 core stays accepted.
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(routedExpertBits: 2)],
            config: .productionShapedDefaults())
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.routedExpert.weightBits == 2)
    }

    @Test func productionManifestRejectsInt3RoutedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(routedExpertBits: 3)],
            config: .productionShapedDefaults())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization")
        }
    }

    @Test func productionManifestRejectsUnsupportedQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 3, routerBits: 4)],
            config: .productionShapedDefaults())
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

    /// A manifest that omits `qkNorm` validates against the default (`true`),
    /// so every install predating the axis loads unchanged; an explicit
    /// `false` against a q/k-norm baseline is an arch mismatch.
    @Test func qkNormAbsentMeansTrueAndExplicitFalseMismatches() throws {
        let (absent, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: absent) }
        _ = try ManifestReader.load(directoryURL: absent, expecting: toy)

        let (explicit, _) = try Self.writeToyManifest(archOverrides: ["qkNorm": true])
        defer { try? FileManager.default.removeItem(at: explicit) }
        _ = try ManifestReader.load(directoryURL: explicit, expecting: toy)

        let (wrong, _) = try Self.writeToyManifest(archOverrides: ["qkNorm": false])
        defer { try? FileManager.default.removeItem(at: wrong) }
        #expect {
            _ = try ManifestReader.load(directoryURL: wrong, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == "qkNorm"
        }
    }

    /// The MiniCPM5 baseline is the one family whose manifest must carry
    /// `qkNorm: false`; omitting it would validate as `true` and mismatch.
    @Test func miniCPM5ManifestCarriesQKNormFalse() throws {
        let toy = ArchConfig.miniCPM5Toy()
        let archOverrides: [String: Any] = [
            "family": "minicpm5",
            "attnOutputGate": false,
            "attentionScale": 0.25,
            "embeddingScaledBySqrtHidden": false,
            "routerScaled": false,
            "ffnSandwichNorms": false,
            "sharedExpertGated": false,
            "ropeNeoxSubdim": true,
            "numSharedExperts": 0,
            "numDenseLayers": 4,
            "denseIntermediateSize": 128,
            "qkNorm": false,
        ]
        let (dir, _) = try Self.writeToyManifest(archOverrides: archOverrides,
                                                 config: toy)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(manifest.arch.family == "minicpm5")
        #expect(manifest.arch.qkNorm == false)

        var withoutAxis = archOverrides
        withoutAxis["qkNorm"] = nil
        let (missing, _) = try Self.writeToyManifest(archOverrides: withoutAxis,
                                                     config: toy)
        defer { try? FileManager.default.removeItem(at: missing) }
        #expect {
            _ = try ManifestReader.load(directoryURL: missing, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == "qkNorm"
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
    /// `defaultsToy` axes at a production baseline's shape (Qwen 3.6's layer
    /// count and width), so `ManifestReader` treats it as a production
    /// architecture and requires quantization metadata.
    static func productionShapedDefaults() -> ArchConfig {
        let q = ArchConfig.qwen36_35B_A3B
        return ArchConfig(
            hiddenSize: q.hiddenSize,
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
            numLayers: q.numLayers,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [UInt8](repeating: 1, count: q.numLayers),
            hiddenActivation: "gelu_pytorch_tanh"
        )
    }

    /// Tiny baseline used across the loader tests, on the `ArchConfig.init`
    /// defaults (the original install format's axes; family `.qwen36`).
    /// 2 layers (sliding then full), hidden 64, vocab 1024, 8 experts.
    /// Numbers are intentionally toy.
    static func defaultsToy(topKExperts: Int = 2) -> ArchConfig {
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
            topKExperts: topKExperts,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 1],
            hiddenActivation: "gelu_pytorch_tanh"
        )
    }

}
