import Foundation
import Metal
import Testing

@testable import Mference

@Suite struct MapleRuntimeSchemaTests {
    private static let maple = ArchConfig.maplePreview

    private static func mapleArchExtensions() -> [String: Any] {
        [
            "family": "maple",
            "attnOutputGate": false,
            "attentionScale": maple.attentionScale,
            "embeddingScaledBySqrtHidden": false,
            "routerScaled": false,
            "ffnSandwichNorms": false,
            "ropeNeoxSubdim": true,
            "routerScoringFunc": "softmax",
            "routedScalingFactor": 1.0,
            "swigluLimit": 7.0,
            "numSharedExperts": 0,
            "numDenseLayers": 0,
            "routerNormAfterTopK": true,
        ]
    }

    private static func mapleQuant() -> [String: Any] {
        func slot(_ weightBits: Int, _ scheme: String, _ scaleType: String,
                  _ biasType: String, _ groupSize: Int) -> [String: Any] {
            [
                "weightBits": weightBits,
                "scheme": scheme,
                "scaleType": scaleType,
                "biasType": biasType,
                "groupSize": groupSize,
            ]
        }
        return [
            "embedding": slot(4, "affine", "bf16", "bf16", Quantization.groupSize),
            "attention": slot(4, "affine", "bf16", "bf16", Quantization.groupSize),
            "router": slot(16, "unquantized", "none", "none", 0),
            "sharedExpert": slot(0, "none", "none", "none", 0),
            "routedExpert": slot(2, "affine", "bf16", "bf16", Quantization.groupSize),
        ]
    }

    private static func writeMapleManifest(
        archOverrides: [String: Any] = mapleArchExtensions(),
        quant: [String: Any] = mapleQuant(),
        flashHead: [String: Any]? = nil
    ) throws -> URL {
        var overrides: [String: Any] = ["quant": quant]
        if let flashHead { overrides["flashHead"] = flashHead }
        return try ManifestReaderTests.writeToyManifest(
            overrides,
            archOverrides: archOverrides,
            config: maple
        ).0
    }

    private static func mapleFlashHead() -> [String: Any] {
        [
            "nClusters": 4_748,
            "clusterSize": 32,
            "nProbes": 512,
            "groupSize": 64,
            "bits": 4,
            "headGroupSize": 64,
            "headBits": 4,
            "scaledCentroids": true,
            "forceTokens": [151_645, 151_668, 151_643],
        ]
    }

    @Test func pinnedMapleArchitectureAndSpecializationsAreRegistered() {
        let config = Self.maple
        #expect(ModelFamily(rawValue: "maple") == .maple)
        #expect(ArchConfig.knownArchitectures[.maple] == config)
        #expect(config.hiddenSize == 2048)
        #expect(config.intermediateSize == 512)
        #expect(config.moeIntermediateSize == 512)
        #expect(config.numHeads == 16)
        #expect(config.numKVHeads == 4)
        #expect(config.headDim == 128)
        #expect(config.vocabSize == 151_936)
        #expect(config.slidingWindow == 512)
        #expect(config.fullRopeTheta == 0.0)
        #expect(config.partialRotaryFactor == 0.5)
        #expect(config.numLayers == 24)
        #expect(config.numExperts == 256)
        #expect(config.topKExperts == 8)
        #expect(config.numSharedExperts == 0)
        #expect(config.routerNormAfterTopK)
        #expect(config.fullAttentionLayerMask == (0..<24).map { $0 % 4 == 3 ? 1 : 0 })
        let expectedInt4Shapes: [(m: Int, n: Int)] = [
            (m: 2048, n: 2048),
            (m: 512, n: 2048),
            (m: 2048, n: 2048),
        ]
        #expect(config.decodeInt4GEMVShapes.count == expectedInt4Shapes.count)
        for (actual, expected) in zip(config.decodeInt4GEMVShapes, expectedInt4Shapes) {
            #expect(actual.m == expected.m)
            #expect(actual.n == expected.n)
        }
        #expect(config.decodeInt8GEMVShapes.isEmpty)
    }

    @Test func generatedCompactMapleManifestValidates() throws {
        let dir = try Self.writeMapleManifest()
        defer { try? FileManager.default.removeItem(at: dir) }

        let manifest = try ManifestReader.load(directoryURL: dir, expecting: Self.maple)
        #expect(manifest.arch.family == "maple")
        #expect(manifest.quant?.router.weightBits == 16)
        #expect(manifest.quant?.sharedExpert.weightBits == 0)
        #expect(try ManifestReader.peekFamily(directoryURL: dir) == .maple)
    }

    @Test func mapleFlashHeadManifestRequiresItsPinnedContract() throws {
        let dir = try Self.writeMapleManifest(flashHead: Self.mapleFlashHead())
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: Self.maple)
        #expect(manifest.flashHead?.nClusters == 4_748)
        #expect(manifest.flashHead?.forceTokens == [151_645, 151_668, 151_643])

        let mutations: [(String, Any)] = [
            ("nClusters", 4_747), ("clusterSize", 31), ("nProbes", 4_749),
            ("groupSize", 32), ("bits", 2), ("headGroupSize", 32),
            ("headBits", 2), ("scaledCentroids", false),
            ("forceTokens", [151_645, 151_645]),
            ("forceTokens", [151_936]),
        ]
        for (field, value) in mutations {
            var flashHead = Self.mapleFlashHead()
            flashHead[field] = value
            let invalid = try Self.writeMapleManifest(flashHead: flashHead)
            defer { try? FileManager.default.removeItem(at: invalid) }
            #expect(throws: ModelError.self) {
                _ = try ManifestReader.load(directoryURL: invalid, expecting: Self.maple)
            }
        }
    }

    @Test func mapleManifestRejectsEveryQuantSlotFieldMutation() throws {
        let mutations: [(slot: String, field: String, value: Any)] = [
            ("embedding", "weightBits", 3), ("embedding", "scheme", "other"),
            ("embedding", "scaleType", "other"), ("embedding", "biasType", "other"),
            ("embedding", "groupSize", 1),
            ("attention", "weightBits", 3), ("attention", "scheme", "other"),
            ("attention", "scaleType", "other"), ("attention", "biasType", "other"),
            ("attention", "groupSize", 1),
            ("router", "weightBits", 8), ("router", "scheme", "other"),
            ("router", "scaleType", "other"), ("router", "biasType", "other"),
            ("router", "groupSize", 1),
            ("sharedExpert", "weightBits", 1), ("sharedExpert", "scheme", "other"),
            ("sharedExpert", "scaleType", "other"), ("sharedExpert", "biasType", "other"),
            ("sharedExpert", "groupSize", 1),
            ("routedExpert", "weightBits", 4), ("routedExpert", "scheme", "other"),
            ("routedExpert", "scaleType", "other"), ("routedExpert", "biasType", "other"),
            ("routedExpert", "groupSize", 1),
        ]

        for mutation in mutations {
            var quant = Self.mapleQuant()
            var slot = try #require(quant[mutation.slot] as? [String: Any])
            slot[mutation.field] = mutation.value
            quant[mutation.slot] = slot
            let dir = try Self.writeMapleManifest(quant: quant)
            defer { try? FileManager.default.removeItem(at: dir) }

            #expect {
                _ = try ManifestReader.load(directoryURL: dir, expecting: Self.maple)
            } throws: { error in
                if case ModelError.indexCorrupt = error { return true }
                return false
            }
        }
    }

    @Test func mapleManifestRequiresEveryBehaviorBearingExtension() throws {
        for field in Self.mapleArchExtensions().keys where field != "numDenseLayers" {
            var arch = Self.mapleArchExtensions()
            arch.removeValue(forKey: field)
            let dir = try Self.writeMapleManifest(archOverrides: arch)
            defer { try? FileManager.default.removeItem(at: dir) }

            #expect {
                _ = try ManifestReader.load(directoryURL: dir, expecting: Self.maple)
            } throws: { error in
                guard case let ModelError.archMismatch(actualField, _, _) = error else {
                    return false
                }
                return actualField == field
            }
        }
    }

    @Test func mapleManifestWithoutNumDenseLayersUsesZeroSemantics() throws {
        var arch = Self.mapleArchExtensions()
        arch.removeValue(forKey: "numDenseLayers")
        let dir = try Self.writeMapleManifest(archOverrides: arch)
        defer { try? FileManager.default.removeItem(at: dir) }

        let manifest = try ManifestReader.load(directoryURL: dir, expecting: Self.maple)
        #expect(manifest.arch.family == ModelFamily.maple.rawValue)
        #expect((manifest.arch.numDenseLayers ?? 0) == Self.maple.numDenseLayers)
    }

    @Test func outOfRangeMapleAttentionMasksThrowWithoutTrapping() throws {
        for invalidValue in [-1, 256] {
            let dir = try Self.writeMapleManifest(
                archOverrides: Self.mapleArchExtensions().merging(
                    ["fullAttentionLayerMask": [invalidValue]],
                    uniquingKeysWith: { _, replacement in replacement })
            )
            defer { try? FileManager.default.removeItem(at: dir) }

            #expect {
                _ = try ManifestReader.load(directoryURL: dir, expecting: Self.maple)
            } throws: { error in
                guard case let ModelError.archMismatch(field, _, _) = error else {
                    return false
                }
                return field == "fullAttentionLayerMask"
            }
        }
    }

    @Test func mapleTensorAccessorsUsePinnedNamesAndRejectSharedExpertPaths() throws {
        let paths = [
            "model.word_embeddings.weight",
            "lm_head.weight",
            "model.layers.7.self_attn.q_proj.weight",
            "model.layers.7.self_attn.k_proj.weight",
            "model.layers.7.self_attn.v_proj.weight",
            "model.layers.7.self_attn.o_proj.weight",
            "model.layers.7.self_attn.q_norm.weight",
            "model.layers.7.self_attn.k_norm.weight",
            "model.layers.7.input_layernorm.weight",
            "model.layers.7.post_attention_layernorm.weight",
            "model.layers.7.mlp.gate.weight",
            "model.norm.weight",
        ]
        let device = try #require(MTLCreateSystemDefaultDevice())
        let (model, dir) = try Self.makeMaplePathModel(device: device, paths: paths)
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(model.embedding.offset == 0)
        #expect(model.lmHead.offset == 1)
        #expect(try model.qProj(layer: 7).offset == 2)
        #expect(try model.kProj(layer: 7).offset == 3)
        #expect(try model.vProj(layer: 7).offset == 4)
        #expect(try model.oProj(layer: 7).offset == 5)
        #expect(try model.qNorm(layer: 7).offset == 6)
        #expect(try model.kNorm(layer: 7).offset == 7)
        #expect(try model.inputNorm(layer: 7).offset == 8)
        #expect(try model.postAttnNorm(layer: 7).offset == 9)
        #expect(try model.router(layer: 7).offset == 10)
        #expect(model.finalNorm.offset == 11)
        #expect {
            _ = try model.sharedExpertGate(layer: 7)
        } throws: { error in
            guard case let ModelError.tensorNotFound(name) = error else { return false }
            return name == "model.layers.7.mlp.shared_expert.gate_proj.weight"
        }
    }

    private static func makeMaplePathModel(
        device: MTLDevice,
        paths: [String]
    ) throws -> (Model, URL) {
        let dir = try writeMapleManifest()
        let residentURL = dir.appendingPathComponent("resident-paths.bin")
        try Data(repeating: 0, count: paths.count).write(to: residentURL)
        let entries = Dictionary(uniqueKeysWithValues: paths.enumerated().map { index, name in
            (name, ResidentIndexEntry(
                name: name,
                dtype: 1,
                fileOffset: UInt64(index),
                sizeBytes: 1,
                shape: (1, 0, 0, 0),
                scaleOffset: 0,
                scaleSize: 0,
                biasOffset: 0,
                biasSize: 0))
        })
        let index = ResidentIndex(
            header: ResidentIndexHeader(
                indexSize: 0,
                residentSize: UInt64(paths.count),
                entryCount: UInt64(paths.count)),
            entries: entries)
        let residentBuffer = try ResidentBuffer(
            fileURL: residentURL,
            fileOffset: 0,
            residentSize: UInt64(paths.count),
            device: device)
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: maple)
        let layout = PackedExpertsLayout(
            expertStride: 0, numLayers: 0, expertsPerLayer: 0, layers: [])
        return (Model(
            device: device,
            config: maple,
            streamingMode: .pread(slotCount: 1),
            expertCachePolicy: .lru,
            integrityPolicy: .fullSha256,
            residentBuffer: residentBuffer,
            residentIndex: index,
            packedExpertsLayout: layout,
            manifest: manifest,
            directoryURL: dir), dir)
    }
}
