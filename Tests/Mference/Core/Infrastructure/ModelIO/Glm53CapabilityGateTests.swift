import Foundation
import Testing
@testable import Mference

/// The Day-0 contract of the `glm53Flash` family: a registered baseline
/// pinned to the checkpoint's `config.json`, layer helpers that encode the
/// vendor's `layer_types` / dense-prefix schedule, manifest validation of the
/// new axes, and — until the runner lands — a capability gate that refuses
/// the family by axis name at the one funnel every load path uses.
@Suite struct Glm53CapabilityGateTests {

    // MARK: - Baseline

    /// Pins the baseline against `zai-org/GLM-5.3-Flash` `config.json ->
    /// text_config` (PipeNetwork's conversion at `d43ea8b4` carries the block
    /// verbatim, plus the per-module `quantization` map).
    @Test func baselineMatchesTheCheckpointConfig() {
        let a = ArchConfig.glm53Flash_320B_A18B
        #expect(a.family == .glm53Flash)
        #expect(a.hiddenSize == 4096)
        #expect(a.numLayers == 45)
        #expect(a.vocabSize == 154_880)
        #expect(a.numHeads == 64)
        #expect(a.numKVHeads == 1)
        #expect(a.headDim == 256)
        #expect(a.fullHeadDim == 256)
        #expect(a.slidingWindow == 0)
        #expect(a.numExperts == 288)
        #expect(a.topKExperts == 8)
        #expect(a.moeIntermediateSize == 2048)
        #expect(a.intermediateSize == 2048)
        #expect(a.numSharedExperts == 1)
        #expect(a.numDenseLayers == 3)
        #expect(a.denseIntermediateSize == 12_288)
        #expect(a.numHashRoutedLayers == 0)
        #expect(a.routerScoringFunc == "sigmoid")
        #expect(a.routedScalingFactor == 2.5)
        #expect(a.routerGateBias == true)
        #expect(a.routerNormAfterTopK == true)
        #expect(a.routerGlobalScale == false)
        #expect(a.swigluLimit == 10.0)
        #expect(a.tieWordEmbeddings == false)
        #expect(a.attentionKEqV == true)
        #expect(a.attentionScale == 0.0625)
        // NoPE by design: no rotary channels anywhere in the text stack.
        #expect(a.ropeTheta == 0.0)
        #expect(a.fullRopeTheta == 0.0)
        #expect(a.partialRotaryFactor == 0.0)
        #expect(a.qkNorm == false)
        #expect(a.finalLogitSoftcap == 0.0)

        #expect(a.linearAttention == LinearAttentionConfig(
            numKHeads: 64, numVHeads: 64, keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4))
        let ca = a.compressedAttention
        #expect(ca.qLoraRank == 1536)
        #expect(ca.oLoraRank == 0)
        #expect(ca.oGroups == 0)
        #expect(ca.ropeHeadDim == 0)
        #expect(ca.indexNHeads == 32)
        #expect(ca.indexHeadDim == 128)
        #expect(ca.indexTopK == 2048)
        #expect(ca.csaCompressRate == 0 && ca.hcaCompressRate == 0)
        #expect(a.hyperConnections == HyperConnectionConfig(mult: 4, sinkhornIters: 20, eps: 1e-6))

        let g = a.glm53
        #expect(g.kvLoraRank == 512)
        #expect(g.qkNopeHeadDim == 256)
        #expect(g.vHeadDim == 256)
        #expect(g.indexKPool == 4)
        #expect(g.indexKPoolAlwaysSelectTail == true)
        #expect(g.indexerKNormEps == 1.0e-6)
        #expect(g.kdaGateLowerBound == -5.0)
        #expect(g.rmsNormEps == 1.0e-5)
        #expect(g.indexerSelectedGroups(indexTopK: ca.indexTopK) == 512)
        #expect(g.indexerMaxSelected(indexTopK: ca.indexTopK) == 2051)
        #expect(a.hasGlm53Axes)
        #expect(!ArchConfig.qwen36_35B_A3B.hasGlm53Axes)
    }

    @Test func baselineIsRegisteredForAutoDetection() {
        #expect(ArchConfig.knownArchitectures[.glm53Flash] == .glm53Flash_320B_A18B)
        #expect(ModelFamily(rawValue: "glm53Flash") == .glm53Flash)
    }

    /// `layer_types`: linear attention everywhere except layers 3, 7, …, 43
    /// (11 of 45), which are sparse; the first 3 layers run the dense FFN.
    @Test func layerMaskAndDenseScheduleFollowTheReference() {
        let a = ArchConfig.glm53Flash_320B_A18B
        #expect(a.fullAttentionLayerMask.count == 45)
        var sparse: [Int] = []
        for L in 0..<45 {
            if a.layerIsLatentSparse(L) { sparse.append(L) }
            #expect(a.layerIsKDA(L) != a.layerIsLatentSparse(L), "layer \(L)")
            #expect(!a.layerIsLinear(L) && !a.layerIsFull(L) && !a.layerIsCompressed(L))
        }
        #expect(sparse == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43])
        // The GDN / CSA helpers must not fire on the new mask values.
        #expect(!a.hasLinearAttentionLayers)
        #expect(!a.hasCompressedAttentionLayers)
        #expect(!a.hasLowRankHyperConnections)
        for L in 0..<3 { #expect(a.layerIsDenseFFN(L)) }
        for L in 3..<45 { #expect(!a.layerIsDenseFFN(L)) }
    }

    /// Every resident projection of this family is INT8 or BF16, so the INT4
    /// GEMV specialization list is empty and the INT8 one carries the KDA,
    /// sparse-attention, indexer, shared / dense FFN and head shapes.
    @Test func gemvSpecializationListsFollowTheInstallWidths() {
        let a = ArchConfig.glm53Flash_320B_A18B
        #expect(a.decodeInt4GEMVShapes.isEmpty)
        let int8 = a.decodeInt8GEMVShapes
        for expected in [(m: 8192, n: 4096), (m: 128, n: 4096), (m: 8192, n: 128),
                         (m: 64, n: 4096), (m: 4096, n: 8192), (m: 1536, n: 4096),
                         (m: 16_384, n: 1536), (m: 512, n: 4096), (m: 4096, n: 16_384),
                         (m: 4096, n: 1536), (m: 32, n: 4096), (m: 2048, n: 4096),
                         (m: 4096, n: 2048), (m: 12_288, n: 4096), (m: 4096, n: 12_288),
                         (m: 154_880, n: 4096)] {
            #expect(int8.contains { $0 == expected }, "\(expected)")
        }
        // The router is BF16: no INT8 router GEMV is listed.
        #expect(!int8.contains { $0 == (m: 288, n: 4096) })
    }

    // MARK: - Capability gate

    @Test func gateTableNamesTheThreeAxes() {
        #expect(ManifestReader.familiesWithoutRunner["glm53Flash"]
                == ManifestReader.glm53RequiredAxes)
        #expect(ManifestReader.glm53RequiredAxes == [
            "kimiDeltaAttention",
            "nopeLatentSparseAttention",
            "pooledLightningIndexer",
        ])
        #expect(ManifestReader.capabilityRefusal(family: "glm53Flash")
                == .familyRunnerNotImplemented(family: "glm53Flash",
                                               missingAxes: ManifestReader.glm53RequiredAxes))
    }

    /// The funnel refuses the family by name off a minimal decode, before the
    /// strict manifest shape is required.
    @Test func peekFamilyRefusesTheFamilyByAxisName() throws {
        let directory = try Self.writeMinimalManifest(family: "glm53Flash")
        defer { try? FileManager.default.removeItem(at: directory) }
        var thrown: Error?
        #expect(throws: (any Error).self) {
            do { _ = try ManifestReader.peekFamily(directoryURL: directory) }
            catch { thrown = error; throw error }
        }
        let error = try #require(thrown as? ModelError)
        #expect(error == .familyRunnerNotImplemented(
            family: "glm53Flash",
            missingAxes: ManifestReader.glm53RequiredAxes))
        let text = error.description
        #expect(text.contains("glm53Flash"))
        #expect(text.contains("runner is not implemented"))
        for axis in ManifestReader.glm53RequiredAxes { #expect(text.contains(axis)) }
    }

    // MARK: - Manifest validation

    /// A manifest that matches the toy baseline field by field — new axes
    /// included — loads; one that drops an axis, or publishes a different
    /// pooling width, is refused by name.
    @Test func manifestValidationCoversTheNewAxes() throws {
        let toy = ArchConfig.glm53Toy()
        let good = Self.manifestDict(for: toy)
        let directory = try Self.write(manifest: good)
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = try ManifestReader.load(directoryURL: directory, expecting: toy)
        #expect(manifest.arch.kvLoraRank == 64)
        #expect(manifest.arch.indexKPool == 2)
        #expect(manifest.arch.kdaGateLowerBound == -5.0)
        #expect(manifest.arch.requiredAxes == ManifestReader.glm53RequiredAxes)

        var droppedAxis = good
        var arch = droppedAxis["arch"] as! [String: Any]
        arch.removeValue(forKey: "kvLoraRank")
        droppedAxis["arch"] = arch
        try Self.expectLoadFailure(droppedAxis, expecting: toy,
                                   equals: .archMismatch(field: "kvLoraRank",
                                                         expected: "64", actual: "0"))

        var wrongPool = good
        arch = wrongPool["arch"] as! [String: Any]
        arch["indexKPool"] = 4
        wrongPool["arch"] = arch
        try Self.expectLoadFailure(wrongPool, expecting: toy,
                                   equals: .archMismatch(field: "indexKPool",
                                                         expected: "2", actual: "4"))

        // The dense-prefix count is validated like every other axis, and the
        // dense layers carry no expert file.
        var wrongDense = good
        arch = wrongDense["arch"] as! [String: Any]
        arch["numDenseLayers"] = 0
        wrongDense["arch"] = arch
        try Self.expectLoadFailure(wrongDense, expecting: toy,
                                   equals: .archMismatch(field: "numDenseLayers",
                                                         expected: "1", actual: "0"))
    }

    /// The quant contract: INT8 attention / embedding / shared expert, INT4
    /// routed experts, an unquantized router slot.
    @Test func quantValidationAcceptsInt8AttentionAndRefusesInt4() throws {
        let toy = ArchConfig.glm53Toy()
        var bad = Self.manifestDict(for: toy)
        var quant = bad["quant"] as! [String: Any]
        quant["attention"] = Self.quantSlot(4)
        bad["quant"] = quant
        var thrown: Error?
        do { _ = try Self.load(bad, expecting: toy) } catch { thrown = error }
        #expect(thrown as? ModelError
                == .indexCorrupt(detail: "unsupported GLM-5.3-Flash quantization for attention"))

        var badRouter = Self.manifestDict(for: toy)
        quant = badRouter["quant"] as! [String: Any]
        quant["router"] = Self.quantSlot(8)
        badRouter["quant"] = quant
        thrown = nil
        do { _ = try Self.load(badRouter, expecting: toy) } catch { thrown = error }
        #expect(thrown as? ModelError
                == .indexCorrupt(detail: "unsupported GLM-5.3-Flash quantization for router"))
    }

    // MARK: - Fixtures

    private static func writeMinimalManifest(family: String) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-glm53-gate-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        let manifest: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "glm-5.3-flash-mlx-mixed-4-8bit",
            "arch": ["family": family,
                     "requiredAxes": ManifestReader.glm53RequiredAxes],
            "files": [:], "expertsPerLayer": 288, "numLayers": 45, "expertStride": 16_384,
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    static func quantSlot(_ bits: Int) -> [String: Any] {
        ["weightBits": bits, "scheme": "affine", "scaleType": "bf16",
         "biasType": "bf16", "groupSize": Quantization.groupSize]
    }

    static var unquantizedSlot: [String: Any] {
        ["weightBits": 16, "scheme": "unquantized", "scaleType": "none",
         "biasType": "none", "groupSize": 0]
    }

    /// A manifest matching `toy` field by field.
    static func manifestDict(for toy: ArchConfig) -> [String: Any] {
        let la = toy.linearAttention
        let ca = toy.compressedAttention
        let hc = toy.hyperConnections
        let g = toy.glm53
        let arch: [String: Any] = [
            "hiddenSize": toy.hiddenSize, "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads, "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim, "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize, "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta, "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers, "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
            "family": toy.family.rawValue,
            "attnOutputGate": toy.attnOutputGate,
            "attentionScale": toy.attentionScale,
            "embeddingScaledBySqrtHidden": toy.embeddingScaledBySqrtHidden,
            "routerScaled": toy.routerScaled,
            "ffnSandwichNorms": toy.ffnSandwichNorms,
            "sharedExpertGated": toy.sharedExpertGated,
            "ropeNeoxSubdim": toy.ropeNeoxSubdim,
            "qkNorm": toy.qkNorm,
            "linearNumKHeads": la.numKHeads, "linearNumVHeads": la.numVHeads,
            "linearKeyHeadDim": la.keyHeadDim, "linearValueHeadDim": la.valueHeadDim,
            "linearConvKernelSize": la.convKernelSize,
            "caQLoraRank": ca.qLoraRank,
            "caIndexNHeads": ca.indexNHeads, "caIndexHeadDim": ca.indexHeadDim,
            "caIndexTopK": ca.indexTopK,
            "hcMult": hc.mult, "hcSinkhornIters": hc.sinkhornIters, "hcEps": hc.eps,
            "numHashRoutedLayers": toy.numHashRoutedLayers,
            "routerScoringFunc": toy.routerScoringFunc,
            "routedScalingFactor": toy.routedScalingFactor,
            "swigluLimit": toy.swigluLimit,
            "numSharedExperts": toy.numSharedExperts,
            "numDenseLayers": toy.numDenseLayers,
            "denseIntermediateSize": toy.denseIntermediateSize,
            "routerGateBias": toy.routerGateBias,
            "routerNormAfterTopK": toy.routerNormAfterTopK,
            "kvLoraRank": g.kvLoraRank,
            "qkNopeHeadDim": g.qkNopeHeadDim,
            "vHeadDim": g.vHeadDim,
            "indexKPool": g.indexKPool,
            "indexKPoolAlwaysSelectTail": g.indexKPoolAlwaysSelectTail,
            "indexerKNormEps": g.indexerKNormEps,
            "kdaGateLowerBound": g.kdaGateLowerBound,
            "rmsNormEps": g.rmsNormEps,
            "requiredAxes": ManifestReader.glm53RequiredAxes,
        ]
        var files: [String: Any] = [
            "model_weights.bin": ["size": 1, "sha256": "0"],
            "packed_experts/layout.json": ["size": 1, "sha256": "0"],
        ]
        for L in toy.numDenseLayers..<toy.numLayers {
            files[String(format: "packed_experts/layer_%02d.bin", L)] = ["size": 1, "sha256": "0"]
        }
        return [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "glm53-toy",
            "arch": arch,
            "quant": [
                "embedding": quantSlot(8), "attention": quantSlot(8),
                "router": unquantizedSlot, "sharedExpert": quantSlot(8),
                "routedExpert": quantSlot(4),
            ],
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": 16_384,
        ]
    }

    private static func write(manifest: [String: Any]) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-glm53-manifest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    private static func load(_ manifest: [String: Any],
                             expecting: ArchConfig) throws -> Manifest {
        let directory = try write(manifest: manifest)
        defer { try? FileManager.default.removeItem(at: directory) }
        return try ManifestReader.load(directoryURL: directory, expecting: expecting)
    }

    private static func expectLoadFailure(_ manifest: [String: Any],
                                          expecting: ArchConfig,
                                          equals expected: ModelError) throws {
        var thrown: Error?
        do { _ = try load(manifest, expecting: expecting) } catch { thrown = error }
        #expect(thrown as? ModelError == expected,
                "expected \(expected), got \(String(describing: thrown))")
    }
}
