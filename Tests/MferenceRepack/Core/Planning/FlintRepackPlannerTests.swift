import Foundation
import Testing
@testable import MferenceRepackCore

@Suite
struct FlintRepackPlannerTests {
    private static func configJSON(overrides: [String: Any] = [:]) -> String {
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

    private func writeConfig(_ json: String) throws -> String {
        let dir = NSTemporaryDirectory() + "flint-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: dir), withIntermediateDirectories: true)
        let path = (dir as NSString).appendingPathComponent("config.json")
        try json.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    @Test func productionConfigParsesPinnedGeometry() throws {
        let path = try writeConfig(Self.configJSON())
        defer { try? FileManager.default.removeItem(
            atPath: (path as NSString).deletingLastPathComponent) }

        let arch = try ArchInfo.load(configPath: path)
        #expect(arch.family == .flint)
        #expect(arch.hiddenSize == 2_048)
        #expect(arch.intermediateSize == 1_024)
        #expect(arch.moeIntermediateSize == 512)
        #expect(arch.denseIntermediateSize == 6_144)
        #expect(arch.numLayers == 49)
        #expect(arch.numDenseLayers == 2)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 6)
        #expect(arch.numHeads == 32)
        #expect(arch.numKVHeads == 4)
        #expect(arch.headDim == 128)
        #expect(arch.slidingWindow == 513)
        #expect(arch.ropeTheta == 10_000)
        #expect(arch.fullRopeTheta == 10_000)
        #expect(arch.partialRotaryFactor == 1.0)
        #expect(arch.attentionScale == 1.0 / Double(128).squareRoot())
        #expect(arch.routerScoringFunc == "sigmoid")
        #expect(arch.routerNormAfterTopK)
        #expect(arch.attentionSinks)
        #expect(arch.noRopeLayerMask == arch.fullAttentionLayerMask)
        #expect(arch.noRopeLayerMask.reduce(0) { $0 + Int($1) } == 14)
        for layer in [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48] {
            #expect(arch.fullAttentionLayerMask[layer] == 1)
            #expect(arch.noRopeLayerMask[layer] == 1)
        }
    }

    @Test func productionConfigRejectsDivergentSchedules() throws {
        for overrides in [
            ["num_experts_per_tok": 8],
            ["sliding_window": 512],
            ["router_score_function": "softmax"],
            ["mlp_only_layers": [0]],
            ["no_rope_layer_indices": [0, 1, 5]],
        ] as [[String: Any]] {
            let path = try writeConfig(Self.configJSON(overrides: overrides))
            defer { try? FileManager.default.removeItem(
                atPath: (path as NSString).deletingLastPathComponent) }
            #expect(throws: RepackError.self) {
                _ = try ArchInfo.load(configPath: path)
            }
        }
    }

    @Test func classificationUsesSwitchMLPContainer() {
        let family = RepackModelFamily.flint
        #expect(RepackPlanner.classify(
            "model.layers.2.mlp.switch_mlp.gate_proj.weight",
            numLayers: 49, family: family)
            == .routedExpert(role: "gate", layer: 2))
        #expect(RepackPlanner.classify(
            "model.layers.48.mlp.switch_mlp.down_proj.weight",
            numLayers: 49, family: family)
            == .routedExpert(role: "down", layer: 48))
        #expect(RepackPlanner.classify(
            "model.layers.2.mlp.shared_expert.gate_proj.weight",
            numLayers: 49, family: family) == .lmResident)
        #expect(RepackPlanner.classify(
            "model.layers.0.mlp.gate_proj.weight",
            numLayers: 49, family: family) == .lmResident)
        #expect(RepackPlanner.classify(
            "model.layers.2.self_attn.softmax_offset",
            numLayers: 49, family: family) == .lmResident)
    }

    @Test func sourceIsPinnedAndRegistered() throws {
        let source = try #require(SupportedModelSource.named("flint"))
        #expect(source.repoID == "OsaurusAI/Flint-40B-A2.5B-JANG_4M")
        #expect(source.revision == "2cc33f8d9d211217febf00e1cbf503c3eac7b16e")
        #expect(source.sourceIndexSHA256
            == "df25783e939be39f41a68a4ae8fd0ad95ab19245363e599d4235f9be897d4dcb")
        #expect(source.approximateDownloadBytes == 23_370_683_168)
        #expect(source.isPinned)
        #expect(SourceFingerprint.modelID(
            forIndexSha256: source.sourceIndexSHA256!) == source.modelID)
    }

    @Test func manifestCarriesNoRopeSinkAndQuantContracts() throws {
        let path = try writeConfig(Self.configJSON())
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
}
