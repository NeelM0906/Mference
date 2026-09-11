import Darwin
import Foundation
import Testing
@testable import MferenceRepackCore

/// The repacker's half of the `glm53Flash` Day-0 contract: the nested
/// `glm5_next` config loads and cross-checks against the pinned production
/// shape, the source is pinned, PipeNetwork's `language_model.` names
/// classify, the generic pre-quantized planner lays the family out (stacked
/// INT4 `switch_mlp` triplets into per-layer expert blobs, everything else
/// resident, the vision tower dropped, no blob for the dense layers), and
/// the manifest carries the new axes, `requiredAxes` and an unquantized
/// router slot.
@Suite struct Glm53RepackPlannerTests {

    @Test func archInfoLoadsFromTheSyntheticNestedConfig() throws {
        let snapshotDir = temporaryRoot("glm53-arch")
        defer { try? FileManager.default.removeItem(atPath: snapshotDir) }
        _ = try SyntheticSnapshot.buildGlm53(at: snapshotDir)

        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        #expect(arch.family == .glm53Flash)
        #expect(arch.hiddenSize == 128)
        #expect(arch.numLayers == 4)
        #expect(arch.fullAttentionLayerMask == [7, 7, 7, 8])
        #expect(arch.numExperts == 8)
        #expect(arch.topKExperts == 2)
        #expect(arch.moeIntermediateSize == 64)
        #expect(arch.intermediateSize == 64)
        #expect(arch.denseIntermediateSize == 128)
        #expect(arch.numDenseLayers == 1)
        #expect(arch.numSharedExperts == 1)
        #expect(arch.numHeads == 2)
        #expect(arch.headDim == 64)
        #expect(arch.attentionKEqV == true)
        #expect(arch.attentionScale == 0.125)
        #expect(arch.ropeTheta == 0.0 && arch.partialRotaryFactor == 0.0)
        #expect(arch.linearNumKHeads == 2 && arch.linearNumVHeads == 2)
        #expect(arch.linearKeyHeadDim == 64 && arch.linearValueHeadDim == 64)
        #expect(arch.linearConvKernelSize == 4)
        #expect(arch.caQLoraRank == 64)
        #expect(arch.caIndexNHeads == 2)
        #expect(arch.caIndexHeadDim == 64)
        #expect(arch.caIndexTopK == 4)
        #expect(arch.hcMult == 4)
        #expect(arch.hcSinkhornIters == 20)
        #expect(arch.routerScoringFunc == "sigmoid")
        #expect(arch.routedScalingFactor == 2.5)
        #expect(arch.routerGateBias && arch.routerNormAfterTopK)
        #expect(arch.swigluLimit == 0.5)
        #expect(arch.qkNorm == false)
        let axes = try #require(arch.glm53)
        #expect(axes.kvLoraRank == 64)
        #expect(axes.qkNopeHeadDim == 64)
        #expect(axes.vHeadDim == 64)
        #expect(axes.indexKPool == 2)
        #expect(axes.indexKPoolAlwaysSelectTail)
        #expect(axes.indexerKNormEps == 1e-6)
        #expect(axes.kdaGateLowerBound == -5.0)
        #expect(axes.rmsNormEps == 1e-5)
    }

    @Test func productionConfigParsesAndCrossChecks() throws {
        let root = temporaryRoot("glm53-prod")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { _ in })

        let arch = try ArchInfo.load(configPath: configPath)
        #expect(arch.family == .glm53Flash)
        #expect(arch.hiddenSize == 4096)
        #expect(arch.numLayers == 45)
        #expect(arch.vocabSize == 154_880)
        #expect(arch.numExperts == 288)
        #expect(arch.topKExperts == 8)
        #expect(arch.numDenseLayers == 3)
        #expect(arch.denseIntermediateSize == 12_288)
        #expect(arch.caQLoraRank == 1536)
        #expect(arch.caIndexTopK == 2048)
        #expect(arch.attentionScale == 0.0625)
        for L in 0..<45 {
            #expect(arch.fullAttentionLayerMask[L] == (L % 4 == 3 ? 8 : 7), "layer \(L)")
        }
        let axes = try #require(arch.glm53)
        #expect(axes.kvLoraRank == 512)
        #expect(axes.indexKPool == 4)
        #expect(axes.rmsNormEps == 1e-5)
    }

    @Test func productionConfigMismatchIsRejected() throws {
        let root = temporaryRoot("glm53-prod-bad")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { tc in
            tc["index_n_heads"] = 64
        })
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: configPath)
        }
    }

    /// The runtime is NoPE-only, sigmoid-router-only, no expert groups; a
    /// config asking for any of those is refused rather than guessed.
    @Test func unsupportedVariantsAreRefused() throws {
        for (key, value) in [("qk_rope_head_dim", 64 as Any),
                             ("n_group", 2 as Any),
                             ("scoring_func", "softmax" as Any),
                             ("norm_topk_prob", false as Any)] {
            let root = temporaryRoot("glm53-variant-\(key)")
            defer { try? FileManager.default.removeItem(atPath: root) }
            let configPath = (root as NSString).appendingPathComponent("config.json")
            try writeProductionConfig(to: configPath, mutate: { tc in tc[key] = value })
            #expect(throws: RepackError.self, "\(key)") {
                _ = try ArchInfo.load(configPath: configPath)
            }
        }
        // `mlp_layer_types` disagreeing with `first_k_dense_replace`.
        let root = temporaryRoot("glm53-variant-mlp")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { tc in
            var types = tc["mlp_layer_types"] as! [String]
            types[3] = "dense"
            tc["mlp_layer_types"] = types
        })
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: configPath)
        }
    }

    @Test func sourceIsPinned() {
        let source = SupportedModelSource.glm53Flash
        #expect(source.name == "glm53flash")
        #expect(source.repoID == "pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit")
        #expect(source.revision == "d43ea8b407ce4e9c25e6ac9baec3feab70d9f5f3")
        #expect(source.isPinned)
        #expect(source.kind == .preQuantized)
        #expect(source.approximateDownloadBytes == 181_944_533_258)
        #expect(SourceFingerprint.knownFingerprints["glm-5.3-flash-mlx-mixed-4-8bit"]
                == "5e0a3768db6fb795c4846bed713785a17d336f4cc4494b4f6b28f75082314383")
        #expect(SourceFingerprint.modelID(forIndexSha256:
            "5e0a3768db6fb795c4846bed713785a17d336f4cc4494b4f6b28f75082314383")
            == "glm-5.3-flash-mlx-mixed-4-8bit")
        #expect(SourceFingerprint.trustOnFirstUseModelID(
            forRepoID: "pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit") == nil)
        #expect(SupportedModelSource.named("glm53flash") == source)
    }

    @Test func classificationBucketsTheConversionNames() {
        let f = RepackModelFamily.glm53Flash
        let lm = "language_model.model.layers."
        #expect(RepackPlanner.classify(lm + "1.mlp.switch_mlp.gate_proj.weight",
                                       numLayers: 4, family: f)
                == .routedExpert(role: "gate", layer: 1))
        #expect(RepackPlanner.classify(lm + "3.mlp.switch_mlp.up_proj.weight",
                                       numLayers: 4, family: f)
                == .routedExpert(role: "up", layer: 3))
        #expect(RepackPlanner.classify(lm + "2.mlp.switch_mlp.down_proj.weight",
                                       numLayers: 4, family: f)
                == .routedExpert(role: "down", layer: 2))
        for resident in [lm + "1.mlp.shared_experts.gate_proj.weight",
                         lm + "0.mlp.gate_proj.weight",
                         lm + "1.mlp.gate.weight",
                         lm + "1.mlp.gate.e_score_correction_bias",
                         lm + "0.self_attn.conv1d.weight",
                         lm + "0.self_attn.forget_gate.A_log",
                         lm + "3.self_attn.embed_q.weight",
                         lm + "3.self_attn.indexer.index_kpool_compress_ape",
                         lm + "3.attn_hc.fn",
                         "language_model.model.embed_tokens.weight",
                         "language_model.lm_head.weight",
                         "language_model.model.norm.weight"] {
            #expect(RepackPlanner.classify(resident, numLayers: 4, family: f) == .lmResident,
                    Comment(rawValue: resident))
        }
        for excluded in ["vision_model.blocks.0.attn.qkv.weight",
                         "vision_model.merger.proj.weight",
                         "language_model.model.mtp.layers.45.eh_proj.weight"] {
            #expect(RepackPlanner.classify(excluded, numLayers: 4, family: f)
                    == .excludedMultimodal, Comment(rawValue: excluded))
        }
        // Other families' prefixes are not this family's contract.
        #expect(RepackPlanner.classify("model.layers.0.mlp.switch_mlp.gate_proj.weight",
                                       numLayers: 4, family: f) == .unknown)
        #expect(RepackPlanner.classify("lm_head.weight", numLayers: 4, family: f) == .unknown)
        #expect(RepackPlanner.layerIndex(in: lm + "17.self_attn.q_proj.weight") == 17)
    }

    /// The generic pre-quantized planner lays the family out on day 0.
    @Test func plannerLaysOutTheSyntheticSnapshot() throws {
        let snapshotDir = temporaryRoot("glm53-plan")
        let outputDir = temporaryRoot("glm53-plan-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildGlm53(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        #expect(metadata.baseBits == 4)
        #expect(metadata.baseGroupSize == 64)
        #expect(!metadata.sourceIsUnquantized)
        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let plan = try RepackPlanner.plan(meta: metadata, arch: arch,
                                          shardHeaders: [header], outputDir: outputDir)

        // The vision tower is excluded, everything under language_model. lands.
        #expect(plan.excludedMultimodalTensorNames.count == 5)
        #expect(plan.excludedMultimodalTensorNames.allSatisfy { $0.hasPrefix("vision_model.") })
        let names = plan.resident.entries.map(\.name)
        #expect(!names.contains { $0.hasPrefix("vision_model.") })
        #expect(!names.contains { $0.contains(".mlp.switch_mlp.") })
        #expect(names.first == "language_model.model.embed_tokens.weight")
        #expect(names.last == "language_model.lm_head.weight")
        #expect(names.contains("language_model.model.norm.weight"))
        #expect(names.contains("language_model.model.layers.0.self_attn.conv1d.weight"))
        #expect(names.contains("language_model.model.layers.3.self_attn.indexer.index_kpool_compress_gate"))
        #expect(names.contains("language_model.model.layers.1.mlp.gate.e_score_correction_bias"))
        #expect(names.contains("language_model.model.layers.0.mlp.gate_proj.weight"))
        #expect(names.contains("language_model.model.layers.1.mlp.shared_experts.down_proj.weight"))

        // Width per tensor follows the per-module overrides: INT8 everywhere
        // the config names, and the routed experts at the INT4 base.
        let embed = try #require(plan.resident.entries.first {
            $0.name == "language_model.model.embed_tokens.weight" })
        #expect(embed.quantSpec?.bits == 8)
        let qProj = try #require(plan.resident.entries.first {
            $0.name == "language_model.model.layers.0.self_attn.q_proj.weight" })
        #expect(qProj.quantSpec?.bits == 8)
        let router = try #require(plan.resident.entries.first {
            $0.name == "language_model.model.layers.1.mlp.gate.weight" })
        #expect(router.quantSpec == nil)

        // The dense layer carries no experts; the MoE layers carry eight INT4
        // triplets, page-aligned.
        #expect(plan.layers.count == 4)
        #expect(plan.layers[0].expertsPerLayer == 0)
        for L in 1..<4 {
            let layer = plan.layers[L]
            #expect(layer.expertsPerLayer == 8, "layer \(L)")
            #expect(layer.subTensors.count == 9, "layer \(L)")
            #expect(layer.expertStride % UInt64(getpagesize()) == 0, "layer \(L)")
            #expect(layer.subTensors.filter { $0.component == "weights" }
                        .allSatisfy { $0.bitsForWeights == 4 }, "layer \(L)")
        }
        #expect(plan.allExpertLayers.count == 3)
    }

    /// The manifest carries the family fields, the three new axes, the
    /// `requiredAxes` list, and records the BF16 router as unquantized.
    @Test func manifestCarriesTheAxesAndTheGate() throws {
        let snapshotDir = temporaryRoot("glm53-manifest")
        let outputDir = temporaryRoot("glm53-manifest-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildGlm53(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let plan = try RepackPlanner.plan(meta: metadata, arch: arch,
                                          shardHeaders: [header], outputDir: outputDir)
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "glm-5.3-flash-mlx-mixed-4-8bit",
            sourceSnapshotHash: "sha256:0",
            files: [],
            expertsPerLayer: 8,
            numLayers: arch.numLayers,
            expertStride: plan.layers[1].expertStride,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 8, attention: 8, router: 8,
                sharedExpert: 8, routedExpert: 4))
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let archDict = obj["arch"] as! [String: Any]
        #expect(archDict["family"] as? String == "glm53Flash")
        #expect(archDict["fullAttentionLayerMask"] as? [Int] == [7, 7, 7, 8])
        #expect(archDict["linearNumKHeads"] as? Int == 2)
        #expect(archDict["linearConvKernelSize"] as? Int == 4)
        #expect(archDict["caQLoraRank"] as? Int == 64)
        #expect(archDict["caIndexTopK"] as? Int == 4)
        #expect(archDict["hcMult"] as? Int == 4)
        #expect(archDict["routerScoringFunc"] as? String == "sigmoid")
        #expect(archDict["routedScalingFactor"] as? Double == 2.5)
        #expect(archDict["swigluLimit"] as? Double == 0.5)
        #expect(archDict["numDenseLayers"] as? Int == 1)
        #expect(archDict["denseIntermediateSize"] as? Int == 128)
        #expect(archDict["routerGateBias"] as? Bool == true)
        #expect(archDict["routerNormAfterTopK"] as? Bool == true)
        #expect(archDict["qkNorm"] as? Bool == false)
        #expect(archDict["kvLoraRank"] as? Int == 64)
        #expect(archDict["qkNopeHeadDim"] as? Int == 64)
        #expect(archDict["vHeadDim"] as? Int == 64)
        #expect(archDict["indexKPool"] as? Int == 2)
        #expect(archDict["indexKPoolAlwaysSelectTail"] as? Bool == true)
        #expect(archDict["indexerKNormEps"] as? Double == 1e-6)
        #expect(archDict["kdaGateLowerBound"] as? Double == -5.0)
        #expect(archDict["rmsNormEps"] as? Double == 1e-5)
        #expect(archDict["requiredAxes"] as? [String]
                == ["kimiDeltaAttention", "nopeLatentSparseAttention", "pooledLightningIndexer"])
        let quant = obj["quant"] as! [String: Any]
        let routerSlot = quant["router"] as! [String: Any]
        #expect(routerSlot["weightBits"] as? Int == 16)
        #expect(routerSlot["scheme"] as? String == "unquantized")
        #expect((quant["routedExpert"] as! [String: Any])["weightBits"] as? Int == 4)
        #expect((quant["attention"] as! [String: Any])["weightBits"] as? Int == 8)
    }

    // MARK: - Helpers

    /// `text_config` of `zai-org/GLM-5.3-Flash` trimmed to the fields the
    /// repacker reads, shaped like the production checkpoint so the
    /// cross-check is exercised.
    private func writeProductionConfig(
        to path: String,
        mutate: (inout [String: Any]) -> Void) throws {
        let layerTypes = (0..<45).map { $0 % 4 == 3 ? "deepseek_sparse_attention" : "linear_attention" }
        let mlpTypes = (0..<45).map { $0 < 3 ? "dense" : "sparse" }
        var tc: [String: Any] = [
            "model_type": "glm5_next_text",
            "vocab_size": 154_880,
            "hidden_size": 4096,
            "intermediate_size": 12_288,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 45,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "n_shared_experts": 1,
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "routed_scaling_factor": 2.5,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 0,
            "qk_nope_head_dim": 256,
            "qk_head_dim": 256,
            "v_head_dim": 256,
            "head_dim": 0,
            "n_group": 1, "topk_group": 1,
            "norm_topk_prob": true,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "hidden_act": "silu",
            "first_k_dense_replace": 3,
            "max_position_embeddings": 1_048_576,
            "rms_norm_eps": 1e-05,
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_kpool": 4,
            "index_kpool_compress": true,
            "index_kpool_always_select_tail": true,
            "indexer_rope_interleave": true,
            "layer_types": layerTypes,
            "mlp_layer_types": mlpTypes,
            "linear_attn_config": [
                "num_heads": 64, "gate_lower_bound": -5.0, "head_dim": 128,
                "short_conv_kernel_size": 4,
            ],
            "swiglu_limit": 10.0,
            "hc_mult": 4, "hc_eps": 1e-06, "hc_sinkhorn_iters": 20, "mhc": true,
            "mla_use_nope": true,
            "moe_router_dtype": "float32",
            "num_nextn_predict_layers": 1,
            "tie_word_embeddings": false,
            "eos_token_id": [154_820, 154_827, 154_829],
        ]
        mutate(&tc)
        let root: [String: Any] = [
            "architectures": ["Glm5NextForConditionalGeneration"],
            "model_type": "glm5_next",
            "text_config": tc,
            "vision_config": ["model_type": "glm5_next_vision", "depth": 24],
            "quantization": ["group_size": 64, "bits": 4],
        ]
        let data = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-glm53-plan-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }

    private func parseHeader(path: String) throws -> Safetensors.Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var headerSize: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSize) {
            try Posix.preadAll(fd: fd, path: path, buf: $0.baseAddress!, count: 8, offset: 0)
        }
        headerSize = UInt64(littleEndian: headerSize)
        var headerData = Data(count: Int(headerSize))
        try headerData.withUnsafeMutableBytes {
            try Posix.preadAll(fd: fd, path: path, buf: $0.baseAddress!, count: $0.count, offset: 8)
        }
        return try Safetensors.parseHeaderBytes(
            path: path, fileSize: try Posix.fileSize(fd: fd, path: path), headerBytes: headerData)
    }
}
