import Darwin
import Foundation
import Testing
@testable import MferenceRepackCore

@Suite
struct DeepseekV4RepackPlannerTests {

    @Test func deepseekArchInfoLoadsFromSyntheticConfig() throws {
        let snapshotDir = temporaryRoot("dsv4-arch")
        defer { try? FileManager.default.removeItem(atPath: snapshotDir) }
        _ = try SyntheticSnapshot.buildDeepseekV4(at: snapshotDir)

        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))

        #expect(arch.family == .deepseekV4Flash)
        #expect(arch.hiddenSize == 128)
        #expect(arch.numLayers == 4)
        #expect(arch.fullAttentionLayerMask == [0, 0, 3, 4])
        #expect(arch.numExperts == 8)
        #expect(arch.topKExperts == 2)
        #expect(arch.moeIntermediateSize == 64)
        #expect(arch.intermediateSize == 64)
        #expect(arch.numHeads == 2)
        #expect(arch.numKVHeads == 1)
        #expect(arch.numFullKVHeads == 1)
        #expect(arch.headDim == 64)
        #expect(arch.fullHeadDim == 64)
        #expect(arch.slidingWindow == 32)
        #expect(arch.tieWordEmbeddings == false)
        #expect(arch.attentionKEqV == true)
        #expect(arch.hiddenActivation == "silu")
        #expect(arch.ropeTheta == 10_000.0)
        #expect(arch.fullRopeTheta == 10_000.0)
        #expect(arch.partialRotaryFactor == 0.125)
        #expect(arch.finalLogitSoftcap == 0.0)
        #expect(arch.attnOutputGate == false)
        #expect(arch.attentionScale == 0.125)   // 64^-0.5
        #expect(arch.embeddingScaledBySqrtHidden == false)
        #expect(arch.routerScaled == false)
        #expect(arch.ffnSandwichNorms == false)
        #expect(arch.sharedExpertGated == false)
        #expect(arch.ropeNeoxSubdim == false)
        #expect(arch.linearNumKHeads == 0)
        #expect(arch.caQLoraRank == 64)
        #expect(arch.caOLoraRank == 64)
        #expect(arch.caOGroups == 2)
        #expect(arch.caRopeHeadDim == 8)   // 64 * 0.125
        #expect(arch.caIndexNHeads == 2)
        #expect(arch.caIndexHeadDim == 64)
        #expect(arch.caIndexTopK == 16)
        #expect(arch.caCSACompressRate == 4)
        #expect(arch.caHCACompressRate == 128)
        #expect(arch.caCompressRopeTheta == 160_000.0)
        #expect(arch.hcMult == 2)
        #expect(arch.hcSinkhornIters == 4)
        #expect(arch.hcEps == 1e-6)
        #expect(arch.numHashRoutedLayers == 1)
        #expect(arch.routerScoringFunc == "sqrtsoftplus")
        #expect(arch.routedScalingFactor == 1.5)
        #expect(arch.swigluLimit == 10.0)
    }

    @Test func productionDeepseekConfigParsesAndCrossChecks() throws {
        let root = temporaryRoot("dsv4-prod")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { _ in })

        let arch = try ArchInfo.load(configPath: configPath)
        #expect(arch.family == .deepseekV4Flash)
        #expect(arch.hiddenSize == 4096)
        #expect(arch.numLayers == 43)
        #expect(arch.vocabSize == 129_280)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 6)
        #expect(arch.attentionScale == 0.044194173824159216)  // 512^-0.5
        #expect(arch.caQLoraRank == 1024)
        #expect(arch.caOLoraRank == 1024)
        #expect(arch.caOGroups == 8)
        #expect(arch.caRopeHeadDim == 64)
        #expect(arch.caIndexNHeads == 64)
        #expect(arch.caIndexHeadDim == 128)
        #expect(arch.caIndexTopK == 512)
        #expect(arch.numHashRoutedLayers == 3)
        #expect(arch.caRopeScalingFactor == 16.0)
        #expect(arch.caRopeScalingOriginalMax == 65_536)
        #expect(arch.caRopeScalingBetaFast == 32.0)
        #expect(arch.caRopeScalingBetaSlow == 1.0)
        #expect(arch.fullAttentionLayerMask.count == 43)
        #expect(arch.fullAttentionLayerMask[0] == 0)
        #expect(arch.fullAttentionLayerMask[1] == 0)
        for i in 2..<43 {
            #expect(arch.fullAttentionLayerMask[i] == (i.isMultiple(of: 2) ? 3 : 4))
        }
    }

    /// The published config carries no `layer_types`; layer kinds ride
    /// `compress_ratios` with one extra trailing entry for the unexported
    /// MTP layer.
    @Test func productionConfigCompressRatiosScheduleParses() throws {
        let root = temporaryRoot("dsv4-prod-ratios")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { c in
            c.removeValue(forKey: "layer_types")
            var ratios: [Int] = [0, 0]
            for i in 2..<43 { ratios.append(i.isMultiple(of: 2) ? 4 : 128) }
            ratios.append(0)  // MTP trailing entry
            c["compress_ratios"] = ratios
            c["num_nextn_predict_layers"] = 1
        })

        let arch = try ArchInfo.load(configPath: configPath)
        #expect(arch.fullAttentionLayerMask.count == 43)
        for i in 2..<43 {
            #expect(arch.fullAttentionLayerMask[i] == (i.isMultiple(of: 2) ? 3 : 4))
        }
    }

    @Test func productionDeepseekConfigMismatchIsRejected() throws {
        let root = temporaryRoot("dsv4-prod-bad")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { c in
            c["num_attention_heads"] = 32
        })

        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: configPath)
        }
    }

    @Test func unknownLayerTypeIsRejected() throws {
        let root = temporaryRoot("dsv4-layer-type")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { c in
            var types = c["layer_types"] as! [String]
            types[0] = "full_attention"
            c["layer_types"] = types
        })

        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: configPath)
        }
    }

    @Test func deepseekSourceIsPinned() {
        // Pinned after first-install verification against
        // 722bf559b7de93575b2320973cf2002e05bfe6c9: the fingerprint table
        // resolves the index hash and the trust-on-first-use path is closed.
        #expect(SourceFingerprint.knownFingerprints["deepseek-v4-flash-2bit-dq"]
            == "d1c2d929ab0a35be32cf18026bb31d6f99dad58d6c93a5a2abbe43791f9d6c30")
        #expect(SourceFingerprint.modelID(forIndexSha256:
            "d1c2d929ab0a35be32cf18026bb31d6f99dad58d6c93a5a2abbe43791f9d6c30")
            == "deepseek-v4-flash-2bit-dq")
        // Pinned repos never resolve through the trust-on-first-use path.
        #expect(SourceFingerprint.trustOnFirstUseModelID(
            forRepoID: "mlx-community/DeepSeek-V4-Flash-2bit-DQ") == nil)
        #expect(SourceFingerprint.trustOnFirstUseModelID(
            forRepoID: "mlx-community/gemma-4-26b-a4b-it-4bit") == nil)
    }

    @Test func deepseekClassificationBucketsNames() {
        let f = RepackModelFamily.deepseekV4Flash
        #expect(RepackPlanner.classify(
            "model.layers.1.ffn.switch_mlp.gate_proj.weight",
            numLayers: 4, family: f)
            == .routedExpert(role: "gate", layer: 1))
        #expect(RepackPlanner.classify(
            "model.layers.2.ffn.switch_mlp.down_proj.weight",
            numLayers: 4, family: f)
            == .routedExpert(role: "down", layer: 2))
        // Shared experts, router sidecars and hc mixes stay resident.
        #expect(RepackPlanner.classify(
            "model.layers.0.ffn.shared_experts.gate_proj.weight",
            numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "model.layers.0.ffn.gate.tid2eid",
            numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "model.layers.2.attn.indexer.compressor.wkv.weight",
            numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "model.hc_head.fn", numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "lm_head.weight", numLayers: 4, family: f) == .lmResident)
        // The Gemma/Qwen prefix is not this family's contract.
        #expect(RepackPlanner.classify(
            "language_model.model.layers.0.ffn.switch_mlp.gate_proj.weight",
            numLayers: 4, family: f) == .unknown)
        // And the plain prefix stays unknown for Qwen.
        #expect(RepackPlanner.classify(
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            numLayers: 4, family: .qwen36) == .unknown)
    }

    @Test func deepseekPlanOrdersResidentsAndSlicesExperts() throws {
        let snapshotDir = temporaryRoot("dsv4-plan")
        let outputDir = temporaryRoot("dsv4-plan-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildDeepseekV4(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)

        let plan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: outputDir)

        let names = plan.resident.entries.map(\.name)
        // Embedding first; HyperHead collapse between the layers and the
        // final norm; untied lm_head last.
        #expect(names.first == "model.embed_tokens.weight")
        #expect(names.last == "lm_head.weight")
        #expect(names.dropLast().last == "model.norm.weight")
        let hcHead = names.filter { $0.hasPrefix("model.hc_head.") }
        #expect(hcHead == ["model.hc_head.base",
                           "model.hc_head.fn",
                           "model.hc_head.scale"])
        let lastLayerIndex = try #require(
            names.lastIndex { $0.contains(".layers.") })
        let hcHeadIndex = try #require(names.firstIndex(of: "model.hc_head.fn"))
        #expect(hcHeadIndex > lastLayerIndex)

        // Layer 2 is the CSA layer: low-rank attention path, compressor,
        // indexer, router (+ correction bias), shared expert, layer norms,
        // then the two mHC mix sites.
        let expectedLayer2 = [
            "attn.wq_a.weight",
            "attn.q_norm.weight",
            "attn.wq_b.weight",
            "attn.wkv.weight",
            "attn.kv_norm.weight",
            "attn.attn_sink",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
            "attn.compressor.wkv.weight",
            "attn.compressor.wgate.weight",
            "attn.compressor.norm.weight",
            "attn.compressor.ape",
            "attn.indexer.compressor.wkv.weight",
            "attn.indexer.compressor.wgate.weight",
            "attn.indexer.compressor.norm.weight",
            "attn.indexer.compressor.ape",
            "attn.indexer.wq_b.weight",
            "attn.indexer.weights_proj.weight",
            "ffn.gate.weight",
            "ffn.gate.e_score_correction_bias",
            "ffn.shared_experts.gate_proj.weight",
            "ffn.shared_experts.up_proj.weight",
            "ffn.shared_experts.down_proj.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
            "attn_hc.fn",
            "attn_hc.base",
            "attn_hc.scale",
            "ffn_hc.fn",
            "ffn_hc.base",
            "ffn_hc.scale",
        ].map { "model.layers.2." + $0 }
        let layer2 = names.filter { $0.contains(".layers.2.") }
        #expect(layer2 == expectedLayer2)

        // Layer 0 is a sliding-window hash-routed layer: no compressor, and
        // the router carries tid2eid instead of the correction bias.
        let layer0 = names.filter { $0.contains(".layers.0.") }
        #expect(!layer0.contains { $0.contains(".compressor.") })
        #expect(layer0.contains("model.layers.0.ffn.gate.tid2eid"))
        #expect(!layer0.contains("model.layers.0.ffn.gate.e_score_correction_bias"))

        // Unquantized pass-through tensors carry no quant companions.
        // sinks / correction bias / hc mixes are FP32, norms and the
        // router gate are BF16, `ape` is BF16, tid2eid rides as raw I64.
        for (suffix, dtype) in [
            ("model.layers.2.attn.attn_sink", UInt8(3)),
            ("model.layers.2.attn.compressor.ape", UInt8(1)),
            ("model.layers.2.attn.indexer.compressor.ape", UInt8(1)),
            ("model.layers.2.ffn.gate.e_score_correction_bias", UInt8(3)),
            ("model.layers.2.ffn.gate.weight", UInt8(1)),
            ("model.layers.2.attn.q_norm.weight", UInt8(1)),
            ("model.layers.2.attn_hc.fn", UInt8(3)),
            ("model.layers.2.attn_hc.base", UInt8(3)),
            ("model.layers.2.attn_hc.scale", UInt8(3)),
            ("model.layers.0.ffn.gate.tid2eid", UInt8(4)),
            ("model.hc_head.fn", UInt8(3)),
        ] {
            let entry = try #require(plan.resident.entries.first { $0.name == suffix })
            #expect(entry.dtype == dtype)
            #expect(entry.quantSpec == nil)
            #expect(entry.sourceScales == nil)
            #expect(entry.sourceBiases == nil)
        }
        // The low-rank attention projections are quantized U32 with companions.
        let qA = try #require(plan.resident.entries.first {
            $0.name == "model.layers.2.attn.wq_a.weight"
        })
        #expect(qA.dtype == 0)
        #expect(qA.quantSpec?.bits == 4)
        #expect(qA.sourceScales != nil)
        #expect(qA.sourceBiases != nil)

        // Every layer slices eight 2-bit experts into the fixed 9-slice
        // blob; the stride comes from the 2-bit (packing factor 16) sizes:
        // 3 x (2048 weight + 256 scales + 256 biases) = 7680 -> one 16 KB page.
        #expect(plan.layers.count == 4)
        for lp in plan.layers {
            #expect(lp.expertsPerLayer == 8)
            #expect(lp.subTensors.count == 9)
            #expect(lp.expertStride == 16_384)
            let order = lp.subTensors.map { "\($0.role).\($0.component)" }
            #expect(order == [
                "gate.weights", "gate.scales", "gate.biases",
                "up.weights", "up.scales", "up.biases",
                "down.weights", "down.scales", "down.biases",
            ])
            for slice in lp.subTensors where slice.component == "weights" {
                #expect(slice.bitsForWeights == 2)
            }
            let gate = try #require(lp.subTensors.first {
                $0.role == "gate" && $0.component == "weights"
            })
            #expect(gate.logicalShape == [64, 128])
            #expect(gate.sizeInExpertBlob == 2048)
            // Mixed gate grouping: group 32 (4 groups/row) everywhere
            // except the last layer's group 64 (2 groups/row).
            let gateScales = try #require(lp.subTensors.first {
                $0.role == "gate" && $0.component == "scales"
            })
            let expectedGroups = lp.layerIndex == 3 ? 2 : 4
            #expect(gateScales.sizeInExpertBlob == UInt64(64 * expectedGroups * 2))
        }

        #expect(plan.excludedMultimodalTensorNames.isEmpty)
    }

    @Test func deepseekManifestCarriesExtensionFields() throws {
        let snapshotDir = temporaryRoot("dsv4-manifest")
        let outputDir = temporaryRoot("dsv4-manifest-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildDeepseekV4(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let plan = try RepackPlanner.plan(
            meta: metadata, arch: arch, shardHeaders: [header], outputDir: outputDir)

        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "deepseek-v4-flash-2bit-dq",
            sourceSnapshotHash: "sha256:0",
            files: [],
            expertsPerLayer: 8,
            numLayers: arch.numLayers,
            expertStride: 16_384,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 4, attention: 4, router: 8,
                sharedExpert: 4, routedExpert: 2))
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let archDict = obj["arch"] as! [String: Any]
        #expect(archDict["family"] as? String == "deepseekV4Flash")
        #expect(archDict["caQLoraRank"] as? Int == 64)
        #expect(archDict["caOLoraRank"] as? Int == 64)
        #expect(archDict["caOGroups"] as? Int == 2)
        #expect(archDict["caRopeHeadDim"] as? Int == 8)
        #expect(archDict["caIndexNHeads"] as? Int == 2)
        #expect(archDict["caIndexHeadDim"] as? Int == 64)
        #expect(archDict["caIndexTopK"] as? Int == 16)
        #expect(archDict["caCSACompressRate"] as? Int == 4)
        #expect(archDict["caHCACompressRate"] as? Int == 128)
        #expect(archDict["caCompressRopeTheta"] as? Double == 160_000.0)
        #expect(archDict["caRopeScalingFactor"] as? Double == 16.0)
        #expect(archDict["caRopeScalingOriginalMax"] as? Int == 65_536)
        #expect(archDict["caRopeScalingBetaFast"] as? Double == 32.0)
        #expect(archDict["caRopeScalingBetaSlow"] as? Double == 1.0)
        #expect(archDict["hcMult"] as? Int == 2)
        #expect(archDict["hcSinkhornIters"] as? Int == 4)
        #expect(archDict["hcEps"] as? Double == 1e-6)
        #expect(archDict["numHashRoutedLayers"] as? Int == 1)
        #expect(archDict["routerScoringFunc"] as? String == "sqrtsoftplus")
        #expect(archDict["routedScalingFactor"] as? Double == 1.5)
        #expect(archDict["swigluLimit"] as? Double == 10.0)
        let quant = obj["quant"] as! [String: Any]
        let routed = quant["routedExpert"] as! [String: Any]
        #expect(routed["weightBits"] as? Int == 2)
    }

    @Test func qwenManifestOmitsDeepseekExtensionFields() throws {
        let snapshotDir = temporaryRoot("dsv4-qwen-manifest")
        let outputDir = temporaryRoot("dsv4-qwen-manifest-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildQwen(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let plan = try RepackPlanner.plan(
            meta: metadata, arch: arch, shardHeaders: [header], outputDir: outputDir)

        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "qwen3.6-35b-a3b-4bit",
            sourceSnapshotHash: "sha256:0",
            files: [],
            expertsPerLayer: 2,
            numLayers: arch.numLayers,
            expertStride: 16_384,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 4, attention: 4, router: 8,
                sharedExpert: 4, routedExpert: 4))
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let archDict = obj["arch"] as! [String: Any]
        #expect(archDict["family"] as? String == "qwen36")
        for key in ["caQLoraRank", "caOLoraRank", "caOGroups", "caRopeHeadDim",
                    "caIndexNHeads", "caIndexHeadDim", "caIndexTopK",
                    "caCSACompressRate", "caHCACompressRate",
                    "caCompressRopeTheta", "hcMult", "hcSinkhornIters", "hcEps",
                    "numHashRoutedLayers", "routerScoringFunc",
                    "routedScalingFactor", "swigluLimit"] {
            #expect(archDict[key] == nil, "qwen manifest must omit \(key)")
        }
    }

    // MARK: - Helpers

    private func writeProductionConfig(
        to path: String,
        mutate: (inout [String: Any]) -> Void) throws {
        var layerTypes: [String] = ["sliding_attention", "sliding_attention"]
        for i in 2..<43 {
            layerTypes.append(i.isMultiple(of: 2)
                ? "compressed_sparse_attention"
                : "heavily_compressed_attention")
        }
        let mlpLayerTypes = [String](repeating: "hash_moe", count: 3)
            + [String](repeating: "moe", count: 40)
        var c: [String: Any] = [
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "quantization": ["bits": 4, "group_size": 64, "mode": "affine"],
            "hidden_size": 4096,
            "moe_intermediate_size": 2048,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "vocab_size": 129_280,
            "num_hidden_layers": 43,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
            "sliding_window": 128,
            "layer_types": layerTypes,
            "mlp_layer_types": mlpLayerTypes,
            "compress_rates": [
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            ],
            "rope_theta": 10_000.0,
            "compress_rope_theta": 160_000.0,
            "rope_scaling": [
                "type": "yarn",
                "factor": 16.0,
                "original_max_position_embeddings": 65_536,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            ],
            "partial_rotary_factor": 0.125,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "tie_word_embeddings": false,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        ]
        mutate(&c)
        let data = try JSONSerialization.data(withJSONObject: c, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-dsv4-plan-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }

    private func parseHeader(path: String) throws -> Safetensors.Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var headerSize: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSize) {
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: $0.baseAddress!,
                count: 8,
                offset: 0)
        }
        headerSize = UInt64(littleEndian: headerSize)
        var headerData = Data(count: Int(headerSize))
        try headerData.withUnsafeMutableBytes {
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: $0.baseAddress!,
                count: $0.count,
                offset: 8)
        }
        return try Safetensors.parseHeaderBytes(
            path: path,
            fileSize: try Posix.fileSize(fd: fd, path: path),
            headerBytes: headerData)
    }
}
