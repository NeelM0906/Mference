import Darwin
import Foundation
import Testing
@testable import MferenceRepackCore

/// The installer's manifest contract for families that stay in
/// `ArchConfig.knownArchitectures` and in the installer but whose planner,
/// runtime and model-run suites were removed: the architecture the installer
/// derives from each family's config and writes into `manifest.json -> arch`
/// (and the quant slots beside it). Restored verbatim from the removed
/// `DeepseekV4RepackPlannerTests`, `Glm53RepackPlannerTests`,
/// `FlintRepackPlannerTests` and `InklingRepackPlannerTests`; Inkling had no
/// manifest case, so its config-to-`ArchInfo` contract stands in.
@Suite
struct RetainedFamilyManifestContractTests {

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

    /// The manifest carries the family fields, the three new axes, the
    /// `requiredAxes` list, and records the BF16 router as unquantized.
    @Test func glm53ManifestCarriesTheAxesAndTheGate() throws {
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

    @Test func flintManifestCarriesNoRopeSinkAndQuantContracts() throws {
        let path = try writeConfig(Self.flintConfigJSON(), label: "flint")
        defer { try? FileManager.default.removeItem(
            atPath: (path as NSString).deletingLastPathComponent) }
        let arch = try ArchInfo.load(configPath: path)
        let resident = ResidentFilePlan(
            path: "/tmp/model_weights.bin",
            entries: [],
            stringTable: [],
            stringTableOffsets: [],
            indexSize: 0,
            residentSize: 0)
        let plan = RepackPlan(
            arch: arch,
            baseMode: "affine",
            baseGroupSize: 64,
            bitsOverrideCount: 142,
            resident: resident,
            layers: [],
            matchedModelID: SupportedModelSource.flint.modelID,
            excludedMultimodalTensorNames: [],
            flashHead: nil)
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: SupportedModelSource.flint.modelID,
            sourceSnapshotHash: "sha256:test",
            files: [],
            expertsPerLayer: 256,
            numLayers: 49,
            expertStride: 1_769_472,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 8,
                attention: 8,
                router: 16,
                sharedExpert: 8,
                routedExpert: 4))
        let root = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let archJSON = try #require(root["arch"] as? [String: Any])
        #expect(archJSON["family"] as? String == "flint")
        #expect(archJSON["numDenseLayers"] as? Int == 2)
        #expect(archJSON["denseIntermediateSize"] as? Int == 6_144)
        #expect(archJSON["routerScoringFunc"] as? String == "sigmoid")
        #expect(archJSON["routerNormAfterTopK"] as? Bool == true)
        #expect(archJSON["attentionSinks"] as? Bool == true)
        #expect(archJSON["noRopeLayerMask"] as? [Int]
            == arch.noRopeLayerMask.map(Int.init))
        let quant = try #require(root["quant"] as? [String: [String: Any]])
        #expect(quant["embedding"]?["weightBits"] as? Int == 8)
        #expect(quant["attention"]?["weightBits"] as? Int == 8)
        #expect(quant["sharedExpert"]?["weightBits"] as? Int == 8)
        #expect(quant["routedExpert"]?["weightBits"] as? Int == 4)
        #expect(quant["router"]?["weightBits"] as? Int == 16)
        #expect(quant["router"]?["scheme"] as? String == "unquantized")
        #expect(quant["router"]?["groupSize"] as? Int == 0)
    }

    @Test func inklingArchInfoLoadsFromProductionConfig() throws {
        let path = try writeConfig(Self.inklingConfigJSON(), label: "inkling")
        defer { try? FileManager.default.removeItem(
            atPath: (path as NSString).deletingLastPathComponent) }

        let arch = try ArchInfo.load(configPath: path)

        #expect(arch.family == .inklingSmall)
        #expect(arch.hiddenSize == 4096)
        #expect(arch.numLayers == 42)
        #expect(arch.vocabSize == 201_024)
        #expect(arch.numHeads == 32)
        #expect(arch.numKVHeads == 8)
        #expect(arch.headDim == 128)
        #expect(arch.fullHeadDim == 128)
        #expect(arch.slidingWindow == 512)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 6)
        #expect(arch.intermediateSize == 2048)
        #expect(arch.moeIntermediateSize == 2048)
        #expect(arch.hiddenActivation == "silu")
        // q/k are per-head RMS-normalized, so the reference scales by 1/d.
        #expect(arch.attentionScale == 1.0 / 128.0)
        #expect(arch.unpaddedVocabSize == 200_058)
        #expect(arch.tieWordEmbeddings == false)

        // Position comes entirely from the relative bias; no RoPE.
        #expect(arch.ropeTheta == 0.0)
        #expect(arch.fullRopeTheta == 0.0)
        #expect(arch.partialRotaryFactor == 0.0)
        #expect(arch.relDRel == 16)
        #expect(arch.relExtent == 1024)
        #expect(arch.relProjDim == 512)      // num_attention_heads * d_rel
        #expect(arch.relLogScalingFloor == 128_000)
        #expect(arch.relLogScalingAlpha == 0.1)

        #expect(arch.sconvKernelSize == 4)
        #expect(arch.numSharedExperts == 2)
        #expect(arch.sharedExpertSink == true)
        #expect(arch.numDenseLayers == 2)
        #expect(arch.denseIntermediateSize == 16_384)
        #expect(arch.embedNormEnabled == true)
        #expect(arch.logitsWidthMultiplier == 16.0)
        #expect(arch.routerScoringFunc == "sigmoid")
        #expect(arch.routedScalingFactor == 8.0)
        #expect(arch.routerGateBias == true)
        #expect(arch.routerNormAfterTopK == true)
        #expect(arch.routerGlobalScale == true)

        // Layers 5, 11, 17, 23, 29, 35, 41 are full attention; rest sliding.
        #expect(arch.fullAttentionLayerMask.count == 42)
        #expect(arch.fullAttentionLayerMask.reduce(0) { $0 + Int($1) } == 7)
        for L in [5, 11, 17, 23, 29, 35, 41] {
            #expect(arch.fullAttentionLayerMask[L] == 1, "layer \(L) full")
        }
        for L in [0, 4, 6, 34, 40] {
            #expect(arch.fullAttentionLayerMask[L] == 0, "layer \(L) sliding")
        }
    }

    // MARK: - Helpers

    private static func flintConfigJSON(overrides: [String: Any] = [:]) -> String {
        let fullLayers = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
        let fullSet = Set(fullLayers)
        var config: [String: Any] = [
            "architectures": ["FlintForCausalLM"],
            "model_type": "flint",
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 150_000,
            "decoder_sparse_step": 1,
            "eos_token_id": [150_001, 150_003, 150_005, 150_018],
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2_048,
            "intermediate_size": 6_144,
            "layer_types": (0..<49).map {
                fullSet.contains($0) ? "full_attention" : "sliding_attention"
            },
            "max_position_embeddings": 262_144,
            "mlp_only_layers": [0, 1],
            "moe_intermediate_size": 512,
            "norm_topk_prob": true,
            "num_attention_heads": 32,
            "num_experts": 256,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 49,
            "num_key_value_heads": 4,
            "no_rope_layer_indices": fullLayers,
            "output_router_logits": false,
            "pad_token_id": 150_001,
            "rms_norm_eps": 0.000_001,
            "rope_theta": 10_000.0,
            "router_aux_loss_coef": 0.000_5,
            "router_score_function": "sigmoid",
            "shared_expert_intermediate_size": 1_024,
            "sliding_window": 513,
            "softmax_type": "learnable",
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "use_cache": true,
            "use_sliding_window": true,
            "vocab_size": 150_144,
        ]
        for (key, value) in overrides { config[key] = value }
        return String(decoding: try! JSONSerialization.data(withJSONObject: config),
                      as: UTF8.self)
    }

    /// `text_config` of `pipenetwork/Inkling-Small-MLX-4bit` revision
    /// `9d6e4720`, trimmed to the fields the repacker reads. Shaped like the
    /// production checkpoint so the cross-check is exercised.
    private static func inklingConfigJSON(
        overrides: [String: Any] = [:]) -> String {
        var tc: [String: Any] = [
            "hidden_size": 4096,
            "num_hidden_layers": 42,
            "vocab_size": 201_024,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "swa_head_dim": 128,
            "sliding_window_size": 512,
            "d_rel": 16,
            "rel_extent": 1024,
            "log_scaling_n_floor": 128_000,
            "log_scaling_alpha": 0.1,
            "use_sconv": true,
            "sconv_kernel_size": 4,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "n_shared_experts": 2,
            "shared_expert_sink": true,
            "dense_mlp_idx": 2,
            "dense_intermediate_size": 16_384,
            "intermediate_size": 2048,
            "route_scale": 8.0,
            "use_gate_bias": true,
            "gate_activation": "sigmoid",
            "norm_after_topk": true,
            "use_global_scale": true,
            "use_embed_norm": true,
            "logits_mup_width_multiplier": 16.0,
            "unpadded_vocab_size": 200_058,
            "local_layer_ids": (0..<42).filter { $0 % 6 != 5 },
        ]
        for (k, v) in overrides { tc[k] = v }
        let root: [String: Any] = [
            "model_type": "inkling_mm_model",
            "text_config": tc,
            "quantization": ["group_size": 64, "bits": 4, "recipe": "uniform"],
        ]
        let data = try! JSONSerialization.data(withJSONObject: root)
        return String(decoding: data, as: UTF8.self)
    }

    private func writeConfig(_ json: String, label: String) throws -> String {
        let dir = NSTemporaryDirectory() + "retained-\(label)-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        let path = (dir as NSString).appendingPathComponent("config.json")
        try json.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-retained-plan-\(tag)-\(UUID().uuidString)")
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
