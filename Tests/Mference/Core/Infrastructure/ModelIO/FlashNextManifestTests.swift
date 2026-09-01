import Testing
import Foundation
@testable import Mference

/// Manifest reading and arch validation for `qwen38flashnext`: the new axes,
/// the uniform INT4 quantization the W2 repack path writes, and the additive
/// `plePool` / `auxiliaryExpertPools` / `sidecars` blocks.
@Suite struct FlashNextManifestTests {

    // MARK: - The compiled baseline

    @Test func baselinePinsTheProductionArchitecture() {
        let arch = ArchConfig.qwen38FlashNext_180B_A3_5B
        #expect(arch.family == .qwen38flashnext)
        #expect(arch.hiddenSize == 2560)
        #expect(arch.numLayers == 48)
        #expect(arch.numHeads == 24)
        #expect(arch.numFullKVHeads == 2)
        #expect(arch.fullHeadDim == 256)
        #expect(arch.numExperts == 512)
        #expect(arch.topKExperts == 10)
        #expect(arch.moeIntermediateSize == 640)
        #expect(arch.intermediateSize == 640)
        #expect(arch.numSharedExperts == 1)
        #expect(arch.sharedExpertGated)
        #expect(arch.attnOutputGate)
        #expect(arch.ropeNeoxSubdim)
        #expect(!arch.tieWordEmbeddings)
        #expect(arch.vocabSize == 248_320)
        #expect(arch.hiddenActivation == "silu")
        #expect(arch.attentionScale == 0.0625)
        #expect(arch.ropeTheta == 10_000_000.0)
        #expect(arch.partialRotaryFactor == 0.25)

        // The 3:1 hybrid: 12 full-attention layers, 36 gated-DeltaNet.
        #expect(arch.fullAttentionLayerMask.count == 48)
        #expect(arch.fullAttentionLayerMask.filter { $0 == 1 }.count == 12)
        #expect(arch.fullAttentionLayerMask.filter { $0 == 2 }.count == 36)
        #expect((0..<48).allSatisfy { arch.layerIsFull($0) == ($0 % 4 == 3) })

        let la = arch.linearAttention
        #expect((la.numKHeads, la.numVHeads) == (16, 48))
        #expect((la.keyHeadDim, la.valueHeadDim, la.convKernelSize) == (128, 128, 4))

        // The mHC axis is *not* how this family's hyper-connections travel.
        #expect(arch.hyperConnections == .none)
        #expect(arch.compressedAttention == .none)
        #expect(arch.relativePosition == .none)
    }

    @Test func baselinePinsTheNewFlashNextAxes() {
        let fn = ArchConfig.qwen38FlashNext_180B_A3_5B.flashNext
        #expect(fn.hcCount == 4)
        #expect(fn.hcLowRank == 320)
        #expect(fn.indexerNumHeads == 4)
        #expect(fn.indexerHeadDim == 128)
        #expect(fn.indexerNumKVHeads == 1)
        #expect(fn.indexerBudget == 2048)
        #expect(fn.indexerCompressRatio == 4)
        #expect(fn.indexerBlockBudget == 512)
        #expect(fn.pleLayerIDs == [2])
        #expect(fn.pleNgramShardCount == 128)
        #expect(fn.pleNgramVocabSizeBase == 20_000_000)
        #expect(fn.pleConvKernelSize == 4)
        #expect(fn.pleEosTokenID == 248_044)
    }

    /// The residual is four streams wide, and the PLE layer id is one-indexed.
    @Test func derivedHelpersReadTheAxesCorrectly() {
        let arch = ArchConfig.qwen38FlashNext_180B_A3_5B
        #expect(arch.hasLowRankHyperConnections)
        #expect(arch.residualStreamWidth == 10_240)
        #expect(arch.flashNext.pleLayerIndices == [1])
        #expect(arch.layerIsPLE(1))
        #expect(!arch.layerIsPLE(2), "pleLayerIDs is one-indexed: id 2 is layers[1]")
        #expect(arch.layerHasAttentionIndexer(3))
        #expect(!arch.layerHasAttentionIndexer(2))

        // Every other family keeps a single-stream residual and no PLE layer.
        for (family, config) in ArchConfig.knownArchitectures
        where family != .qwen38flashnext {
            #expect(config.flashNext == .none, "\(family.rawValue) grew flashNext axes")
            #expect(!config.hasLowRankHyperConnections)
            #expect(config.residualStreamWidth == config.hiddenSize)
        }
    }

    // MARK: - Arch validation

    @Test func toyManifestValidatesAgainstTheToyBaseline() throws {
        let dir = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir,
                                               expecting: .qwen38FlashNextToy())
        #expect(manifest.arch.family == "qwen38flashnext")
        #expect(manifest.arch.hcCount == 4)
        #expect(manifest.arch.indexerBudget == 32)
        #expect(manifest.arch.pleLayerIDs == [2])
        #expect(manifest.plePool?.kind == "rowLookupPoolV1")
        #expect(manifest.plePool?.layers.count == 1)
        #expect(manifest.sidecars?["vision"]?.carried == false)
        // The install predates the norm bake, so the loader owns it.
        #expect(manifest.zeroCenteredNormsBakedAtInstall == nil)
    }

    /// Each new axis is compared field-by-field, so a manifest that disagrees
    /// on one is refused by that field's name.
    @Test func aDivergentFlashNextAxisIsNamedInTheMismatch() throws {
        for (field, value) in [("hcCount", 8), ("hcLowRank", 640),
                               ("indexerBudget", 4096),
                               ("indexerCompressRatio", 8),
                               ("indexerNumHeads", 8),
                               ("pleNgramShardCount", 64),
                               ("pleConvKernelSize", 3)] {
            let dir = try FlashNextToySynthetic.write()
            defer { try? FileManager.default.removeItem(at: dir) }
            try Self.patchArch(dir, field, value)

            var thrown: Error?
            #expect(throws: (any Error).self) {
                do {
                    _ = try ManifestReader.load(directoryURL: dir,
                                                expecting: .qwen38FlashNextToy())
                } catch { thrown = error; throw error }
            }
            let error = try #require(thrown as? ModelError)
            guard case .archMismatch(let named, _, _) = error else {
                Issue.record("\(field) produced \(error), not an archMismatch")
                continue
            }
            #expect(named == field)
        }
    }

    /// `pleLayerIDs` is a list, and a manifest that moves the PLE block to a
    /// different layer describes a different model.
    @Test func aDivergentPleLayerListIsRefused() throws {
        let dir = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.patchArch(dir, "pleLayerIDs", [3])
        #expect(throws: ModelError.self) {
            _ = try ManifestReader.load(directoryURL: dir,
                                        expecting: .qwen38FlashNextToy())
        }
    }

    /// `pleEosTokenID` is validated only when the manifest carries it: the
    /// shipped repack path does not emit it yet, and defaulting to the
    /// baseline would turn a real divergence into silent agreement. When it
    /// *is* present it must match.
    @Test func pleEosTokenIDIsCheckedOnlyWhenPublished() throws {
        let absent = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: absent) }
        _ = try ManifestReader.load(directoryURL: absent,
                                    expecting: .qwen38FlashNextToy())

        let matching = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: matching) }
        try Self.patchArch(matching, "pleEosTokenID", 1_023)
        _ = try ManifestReader.load(directoryURL: matching,
                                    expecting: .qwen38FlashNextToy())

        let wrong = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: wrong) }
        try Self.patchArch(wrong, "pleEosTokenID", 7)
        #expect(throws: ModelError.archMismatch(field: "pleEosTokenID",
                                                expected: "1023",
                                                actual: "7")) {
            _ = try ManifestReader.load(directoryURL: wrong,
                                        expecting: .qwen38FlashNextToy())
        }
    }

    /// An install whose arch declares a PLE layer but publishes no row pool
    /// cannot be loaded: the layer's embedding table would simply be missing.
    @Test func aMissingPlePoolIsRefusedByLayer() throws {
        let dir = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        var root = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        root.removeValue(forKey: "plePool")
        try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
            .write(to: manifestURL)

        #expect(throws: ModelError.plePoolMissing(
            layer: FlashNextToySynthetic.pleLayer)) {
            _ = try ManifestReader.load(directoryURL: dir,
                                        expecting: .qwen38FlashNextToy())
        }
    }

    /// The additive blocks must not change how any other family decodes: a
    /// manifest without them still reads, with the fields nil.
    @Test func familiesWithoutTheAdditiveBlocksDecodeUnchanged() throws {
        let (dir, toy) = try ManifestReaderTests.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(manifest.plePool == nil)
        #expect(manifest.auxiliaryExpertPools == nil)
        #expect(manifest.sidecars == nil)
        #expect(manifest.zeroCenteredNormsBakedAtInstall == nil)
        #expect(manifest.arch.hcCount == nil)
        #expect(manifest.arch.pleLayerIDs == nil)
    }

    // MARK: - The real install

    /// Integration check against the real 175 GB install. Reads
    /// `manifest.json` only — never the weights, never a `Model.load` — so it
    /// stays cheap and cannot disturb an install that must keep failing the
    /// capability gate. Skipped unless `MFERENCE_FLASHNEXT_GTURBO` points at
    /// one.
    @Test func installedManifestValidatesAgainstTheBaseline() throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_FLASHNEXT_GTURBO"] else { return }
        let url = URL(fileURLWithPath: path)

        // The gate is still the authority, whatever the baseline says.
        #expect(throws: ModelError.familyRunnerNotImplemented(
            family: "qwen38flashnext",
            missingAxes: ["hyperConnectionsLowRank", "attentionIndexer",
                          "pleNgramEmbedding"])) {
            _ = try ManifestReader.peekFamily(directoryURL: url)
        }

        let expected = try #require(ArchConfig.knownArchitectures[.qwen38flashnext])
        let manifest = try ManifestReader.load(directoryURL: url, expecting: expected)

        #expect(manifest.expertsPerLayer == 512)
        #expect(manifest.expertStride == 2_768_896)
        #expect(manifest.numLayers == 48)
        let quant = try #require(manifest.quant)
        for slot in [quant.embedding, quant.attention, quant.router,
                     quant.sharedExpert, quant.routedExpert] {
            #expect(slot.weightBits == 4)
            #expect(slot.groupSize == 64)
            #expect(slot.scheme.lowercased() == "affine")
        }
        let pool = try #require(manifest.plePool)
        #expect(pool.kind == "rowLookupPoolV1")
        let layer = try #require(pool.layers.first)
        #expect(layer.layer == 1)
        #expect(layer.rows == 320_001_536)
        #expect(layer.rowDim == 160)
        #expect(layer.storage == "bf16")
        #expect(layer.rowStride == 320)
        #expect(layer.rowsPerBlock == 51)
        #expect(layer.blockStride == 16_384)
        #expect(layer.shards.count == 128)

        let geometry = try PleRowPoolGeometry(layer: layer,
                                              hiddenSize: expected.hiddenSize)
        #expect(geometry.ngramHeads == 16, "16 heads x 160 = the 2560 embedding")
        #expect(try geometry.fileOffset(row: 0) == 0)
        #expect(try geometry.fileOffset(row: 50) == 50 * 320)
        #expect(try geometry.fileOffset(row: 51) == 16_384)
        #expect(try geometry.fileOffset(row: 2_500_012) == layer.shards[1].offset)

        let mtp = try #require(manifest.auxiliaryExpertPools?.first)
        #expect(mtp.name == "mtp")
        #expect(mtp.directory == "packed_experts_mtp")
        #expect(mtp.expertsPerLayer == 512)
        #expect(manifest.sidecars?["mtp"]?.carried == true)
        #expect(manifest.sidecars?["vision"]?.carried == false)
        // The repacker has not been taught the bake yet, so the loader owns it.
        #expect(manifest.zeroCenteredNormsBakedAtInstall == nil)
    }

    private static func patchArch(_ dir: URL, _ field: String, _ value: Any) throws {
        let manifestURL = dir.appendingPathComponent("manifest.json")
        var root = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        var arch = root["arch"] as! [String: Any]
        arch[field] = value
        root["arch"] = arch
        try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
            .write(to: manifestURL)
    }
}
