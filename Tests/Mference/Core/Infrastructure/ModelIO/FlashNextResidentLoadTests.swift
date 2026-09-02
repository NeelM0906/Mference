import Testing
import Foundation
import Metal
@testable import Mference

/// Resident loading for `qwen38flashnext`: the family's tensor-name contract,
/// the typed I64 hash tables, and the zero-centered `(1 + w)` norm bake.
///
/// The fixture is `FlashNextToySynthetic`, whose norm weights are written
/// zero-centered — scattered around 0, the way the checkpoint stores them —
/// so a bake that silently did nothing would be visible.
@Suite struct FlashNextResidentLoadTests {

    private static func loadToy() throws -> (Model, URL) {
        let dir = try FlashNextToySynthetic.write()
        let device = try #require(MTLCreateSystemDefaultDevice())
        // `expecting:` bypasses the auto-detect path deliberately: this suite
        // exercises the loader the runner will be built on, and the
        // auto-detect path is (correctly) still refusing this family.
        let model = try Model.load(directoryURL: dir,
                                   device: device,
                                   expecting: .qwen38FlashNextToy())
        return (model, dir)
    }

    // MARK: - Tensor-name contract

    @Test func residentTensorsResolveUnderTheVendorTrunkPrefix() throws {
        let (model, dir) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }
        let toy = ArchConfig.qwen38FlashNextToy()

        // The container order is the vendor's, the mirror of the earlier Qwen
        // conversions', and lm_head is bare.
        #expect(model.trunkPrefix == "model.language_model.")
        #expect(model.embedding.shape.0 == UInt32(toy.vocabSize))
        #expect(model.lmHead.shape.0 == UInt32(toy.vocabSize))
        #expect(model.lmHead.offset != model.embedding.offset,
                "lm_head is untied and must be its own tensor")

        for L in 0..<toy.numLayers {
            _ = try model.router(layer: L)
            _ = try model.sharedExpertGate(layer: L)
            _ = try model.sharedExpertUp(layer: L)
            _ = try model.sharedExpertDown(layer: L)
            _ = try model.sharedExpertScalarGate(layer: L)
            for site in [Model.HyperConnectionSite.attention, .mlp] {
                _ = try model.hcMixDown(site: site, layer: L)
                _ = try model.hcMixUp(site: site, layer: L)
                _ = try model.hcInject(site: site, layer: L)
                _ = try model.hcNorm(site: site, layer: L)
            }
            if toy.layerIsLinear(L) {
                _ = try model.linearInProjQKV(layer: L)
                _ = try model.linearConv1d(layer: L)
                _ = try model.linearNorm(layer: L)
            } else {
                _ = try model.qProj(layer: L)
                _ = try model.indexerQKProj(layer: L)
                _ = try model.indexerQNorm(layer: L)
                _ = try model.indexerKNorm(layer: L)
            }
        }
        _ = try model.hcGlobalMixDown
        _ = try model.hcGlobalMixUp
        _ = try model.hcGlobalNorm
    }

    /// The hyper-connection tensors are shaped by the *bundle* width
    /// (`hcCount * hidden`), not by `hiddenSize`, and the inject weight has one
    /// row per stream.
    @Test func hyperConnectionShapesFollowTheBundleWidth() throws {
        let (model, dir) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }
        let toy = ArchConfig.qwen38FlashNextToy()
        let bundle = UInt32(toy.residualStreamWidth)
        #expect(bundle == UInt32(toy.flashNext.hcCount * toy.hiddenSize))

        let down = try model.hcMixDown(site: .attention, layer: 0)
        #expect(down.shape.0 == UInt32(toy.flashNext.hcLowRank))
        #expect(down.shape.1 == bundle)
        let up = try model.hcMixUp(site: .mlp, layer: 0)
        #expect(up.shape.0 == bundle)
        #expect(up.shape.1 == UInt32(toy.flashNext.hcLowRank))
        let inject = try model.hcInject(site: .attention, layer: 2)
        #expect(inject.shape.0 == UInt32(toy.flashNext.hcCount))
        #expect(inject.shape.1 == bundle)
        #expect(try model.hcNorm(site: .mlp, layer: 1).shape.0 == bundle)
    }

    /// This family carries no pre-norms and no final norm: the hyper-connection
    /// sites hold them. Asking for one is a runner that was not written for
    /// this family, and must say so by name rather than reading as corruption.
    @Test func absentPerSublayerNormsRefuseByName() throws {
        let (model, dir) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(!model.hasFinalNorm)
        for accessor in [model.inputNorm, model.postAttnNorm] {
            var thrown: Error?
            #expect(throws: (any Error).self) {
                do { _ = try accessor(0) } catch { thrown = error; throw error }
            }
            let error = try #require(thrown as? ModelError)
            #expect(error == .familyRunnerNotImplemented(
                family: "qwen38flashnext",
                missingAxes: ["hyperConnectionsLowRank",
                              "attentionIndexer",
                              "pleNgramEmbedding"]))
        }
    }

    /// The indexer is a per-full-attention-layer tensor group. A GDN layer has
    /// none, and asking for one must be a plain missing-tensor error, not a
    /// silent fall-through to a neighbouring layer's weights.
    @Test func indexerExistsOnlyOnFullAttentionLayers() throws {
        let (model, dir) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }
        let toy = ArchConfig.qwen38FlashNextToy()

        #expect(toy.layerHasAttentionIndexer(3))
        #expect(!toy.layerHasAttentionIndexer(0))
        let rows = (toy.flashNext.indexerNumHeads + toy.flashNext.indexerNumKVHeads)
            * toy.flashNext.indexerHeadDim
        #expect(try model.indexerQKProj(layer: 3).shape.0 == UInt32(rows))
        #expect(throws: ModelError.self) { _ = try model.indexerQKProj(layer: 0) }
    }

    // MARK: - I64 hash tables

    /// The three PLE tables are integer lookups, not kernel operands: they come
    /// back typed so nothing can bind a 64-bit table as float data, and their
    /// values are loaded verbatim rather than re-derived.
    @Test func pleIntegerTablesLoadAsTypedInt64() throws {
        let (model, dir) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }
        let L = FlashNextToySynthetic.pleLayer

        let multipliers = try model.pleLayerMultipliers(layer: L)
        #expect(multipliers == FlashNextToySynthetic.int64Table(
            named: "layer_multipliers", count: 3))
        #expect(multipliers.allSatisfy { $0 % 2 == 1 },
                "splitmix64-derived n-gram multipliers are always odd")
        #expect(try model.pleNgramSize(layer: L) == 3)

        let offsets = try model.pleNgramHeadOffsets(layer: L)
        let vocabSizes = try model.pleNgramHeadVocabSizes(layer: L)
        #expect(offsets.count == FlashNextToySynthetic.Pool.ngramHeads)
        #expect(vocabSizes.count == FlashNextToySynthetic.Pool.ngramHeads)
        #expect(offsets == FlashNextToySynthetic.int64Table(
            named: "ngram_heads_offsets", count: offsets.count))
        #expect(vocabSizes == FlashNextToySynthetic.int64Table(
            named: "ngram_heads_vocab_sizes", count: vocabSizes.count))
        // The heads tile the pool: every row is reachable by exactly one head.
        #expect(zip(offsets, vocabSizes).map(+).last
                == Int64(FlashNextToySynthetic.Pool.totalRows))
    }

    @Test func readingAFloatTensorAsInt64Refuses() throws {
        let (model, dir) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.self) {
            _ = try model.residentInt64(
                name: "model.language_model.layers.1.ple.norm_key.weight")
        }
    }

    // MARK: - Zero-centered (1 + w) norm bake

    @Test func normBakeAppliesToExactlyTheZeroCenteredNormSet() throws {
        let (model, dir) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(model.zeroCenteredNormPolicy == .bakeAtLoad)

        let baked = [
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",
            "model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight",
            "model.language_model.layers.1.ple.norm_conv.weight",
            "model.language_model.layers.1.ple.norm_key.weight",
            "model.language_model.layers.1.ple.norm_query.weight",
            "model.language_model.layers.3.self_attn.q_norm.weight",
            "model.language_model.layers.3.self_attn.k_norm.weight",
            "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight",
            "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight",
        ]
        for name in baked {
            let raw = try model.resident(name: name)
            let cooked = try model.normWeight(name: name)
            #expect(cooked.length == raw.length)
            #expect(cooked.dtype == 1)
            let count = Int(raw.length) / MemoryLayout<UInt16>.stride
            let rawBits = Self.bf16(raw, count: count)
            let cookedBits = Self.bf16(cooked, count: count)
            // Bit-for-bit the MTP attach's conversion: widen, add one, round.
            let expected = rawBits.map {
                Quantization.bf16Bits(Quantization.bf16ToFloat($0) + 1.0)
            }
            #expect(cookedBits == expected, "\(name) was not baked as (1 + w)")
            // The fixture stores zero-centered weights, so the bake must have
            // moved them: a pass-through would fail here.
            #expect(cookedBits != rawBits, "\(name) came back unbaked")
        }
    }

    /// The bake must not touch weights that are not norms — the depthwise conv
    /// kernels and `A_log` / `dt_bias` are BF16 too, and adding one to them
    /// would corrupt the model silently.
    @Test func normBakeLeavesNonNormBF16TensorsAlone() throws {
        let (model, dir) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }
        let untouched = [
            // The GDN gated norm is ones-initialized (`Qwen3_5RMSNormGated`),
            // not zero-centered: it is the one RMSNorm in this stack the bake
            // must skip. Confirmed against the reference implementation.
            "model.language_model.layers.0.linear_attn.norm.weight",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "model.language_model.layers.0.linear_attn.A_log",
            "model.language_model.layers.0.linear_attn.dt_bias",
            "model.language_model.layers.1.ple.conv1d.weight",
        ]
        for name in untouched {
            #expect(!Model.isZeroCenteredNorm(name))
            let raw = try model.resident(name: name)
            let through = try model.normWeight(name: name)
            #expect(through.buffer === raw.buffer)
            #expect(through.offset == raw.offset)
        }
    }

    /// The baked copy is materialized once and cached: a second read returns
    /// the same buffer, so the loader cannot quietly allocate per call.
    @Test func bakedNormsAreCachedByName() throws {
        let (model, dir) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }
        let name = "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight"
        let first = try model.normWeight(name: name)
        let second = try model.normWeight(name: name)
        #expect(first.buffer === second.buffer)
        #expect(first.offset == second.offset)
    }

    /// The bake is family-gated. A family whose conversion already stores norms
    /// in full form must see byte-identical weights through the same accessor,
    /// or every shipped model changes.
    @Test func otherFamiliesSeeNormWeightsUnchanged() throws {
        let dir = try Qwen38ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwen38Toy())
        #expect(model.zeroCenteredNormPolicy == .storedInFullForm)
        // A name that *is* in the zero-centered set, on a family that is not.
        let name = "language_model.model.layers.3.self_attn.q_norm.weight"
        let raw = try model.resident(name: name)
        let through = try model.normWeight(name: name)
        #expect(through.buffer === raw.buffer)
        #expect(through.offset == raw.offset)
    }

    /// Once the repacker folds the `+1` in at install, the manifest says so and
    /// the loader must stand down — applying it twice would be `2 + w`.
    @Test func installTimeBakeDisablesTheLoadTimeBake() throws {
        let dir = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        var root = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        root["zeroCenteredNormsBakedAtInstall"] = true
        try JSONSerialization.data(withJSONObject: root,
                                   options: [.sortedKeys, .withoutEscapingSlashes])
            .write(to: manifestURL)

        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwen38FlashNextToy())
        #expect(model.zeroCenteredNormPolicy == .storedInFullForm)
        let name = "model.language_model.layers.1.ple.norm_key.weight"
        let raw = try model.resident(name: name)
        let through = try model.normWeight(name: name)
        #expect(through.buffer === raw.buffer)
        #expect(through.offset == raw.offset)
    }

    private static func bf16(_ view: TensorView, count: Int) -> [UInt16] {
        let base = view.buffer.contents().advanced(by: Int(view.offset))
            .assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { base[$0] }
    }
}
