import Foundation
import Testing
@testable import MferenceRepackCore

@Suite
struct MapleRepackPlannerTests {

    @Test func pinnedMapleConfigParsesExactly() throws {
        let root = try makeSnapshot("arch")
        defer { try? FileManager.default.removeItem(atPath: root) }

        let arch = try ArchInfo.load(configPath: configPath(in: root))

        #expect(arch.family == .maple)
        #expect(arch.hiddenSize == 2_048)
        #expect(arch.intermediateSize == 512)
        #expect(arch.moeIntermediateSize == 512)
        #expect(arch.numHeads == 16)
        #expect(arch.numKVHeads == 4)
        #expect(arch.headDim == 128)
        #expect(arch.vocabSize == 151_936)
        #expect(arch.slidingWindow == 512)
        #expect(arch.numLayers == 24)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 8)
        #expect(arch.ropeTheta == 10_000.0)
        #expect(arch.fullRopeTheta == 0.0)
        #expect(arch.partialRotaryFactor == 0.5)
        #expect(arch.fullAttentionLayerMask ==
            (0..<24).map { $0 % 4 == 3 ? 1 : 0 })
        #expect(arch.attentionScale == 1.0 / Double(128).squareRoot())
        #expect(arch.routerScoringFunc == "softmax")
        #expect(arch.routedScalingFactor == 1.0)
        #expect(arch.swigluLimit == 7.0)
        #expect(arch.numSharedExperts == 0)
        #expect(arch.numDenseLayers == 0)
        #expect(arch.routerNormAfterTopK)
    }

    @Test func mapleConfigRejectsOneFieldMutations() throws {
        for (key, value) in [
            ("hidden_size", NSNumber(value: 2_048.5) as Any),
            ("moe_shared_expert_intermediate_size", 511 as Any),
            ("moe_intermediate_size", 511 as Any),
            ("num_attention_heads", 15 as Any),
            ("num_key_value_heads", 3 as Any),
            ("head_dim", 127 as Any),
            ("vocab_size", 151_935 as Any),
            ("sliding_window", 511 as Any),
            ("rope_theta", 10_001.0 as Any),
            ("partial_rotary_factor", 0.25 as Any),
            ("num_hidden_layers", 23 as Any),
            ("num_experts", true as Any),
            ("num_experts_per_tok", 7 as Any),
            ("num_shared_experts", 1 as Any),
            ("first_k_dense_replace", 1 as Any),
            ("rms_norm_eps", 0.000_01 as Any),
            ("use_qk_norm", false as Any),
            ("norm_topk_prob", false as Any),
            ("tie_word_embeddings", true as Any),
            ("use_rmsnorm", false as Any),
            ("use_bias", true as Any),
            ("router_dtype", "bf16" as Any),
            ("hidden_act", "gelu" as Any),
            ("layer_types", (0..<24).map {
                $0 % 4 == 0 ? "full_attention" : "sliding_attention"
            } as Any),
        ] {
            let root = try makeSnapshot("arch-mutation") { config in
                config[key] = value
            }
            defer { try? FileManager.default.removeItem(atPath: root) }

            #expect(throws: RepackError.self) {
                _ = try ArchInfo.load(configPath: configPath(in: root))
            }
        }
    }

    @Test func mapleQuantizationIsExact() throws {
        let root = try makeSnapshot("quant")
        defer { try? FileManager.default.removeItem(atPath: root) }

        let meta = try IndexLoader.load(snapshotDir: root)
        #expect(meta.baseBits == 2)
        #expect(meta.baseGroupSize == 128)
        #expect(meta.baseMode == "affine")
        #expect(meta.bitsOverrides == [
            "model.word_embeddings": QuantSpec(bits: 4, groupSize: 64),
            "lm_head": QuantSpec(bits: 4, groupSize: 64),
        ])
    }

    @Test func mapleQuantizationRejectsSchemaMutations() throws {
        for kind in ["missing", "extra", "fractionalBase", "fractionalOverride"] {
            let root = try makeSnapshot("quant-\(kind)") { config in
                var quant = config["quantization"] as! [String: Any]
                switch kind {
                case "missing": quant.removeValue(forKey: "lm_head")
                case "extra": quant["model.layers.0.self_attn.q_proj"] = ["bits": 4, "group_size": 64]
                case "fractionalBase": quant["group_size"] = NSNumber(value: 128.5)
                default:
                    quant["lm_head"] = ["bits": NSNumber(value: 4.5), "group_size": 64]
                }
                config["quantization"] = quant
            }
            defer { try? FileManager.default.removeItem(atPath: root) }

            #expect(throws: RepackError.self) {
                _ = try IndexLoader.load(snapshotDir: root)
            }
        }
    }

    @Test func mapleSourceIsPinnedAndRegistered() throws {
        let source = try #require(SupportedModelSource.named("maple"))
        #expect(source.repoID == "deepgrove/maple-preview-2bit-mlx")
        #expect(source.revision == "361db5da5e74ff6fcdd852d478e1f266ce11013a")
        #expect(source.sourceIndexSHA256 ==
            "56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95")
        #expect(source.modelID == "maple-preview-2bit-mlx")
        #expect(source.isPinned)
        #expect(SourceFingerprint.modelID(forIndexSha256: source.sourceIndexSHA256!) == source.modelID)
    }

    @Test func mapleFlashHeadI32TensorIsParsedAndExcluded() throws {
        let headerData = try JSONSerialization.data(withJSONObject: [
            "lm_head_flash.token_map": [
                "dtype": "I32", "shape": [2], "data_offsets": [0, 8],
            ],
        ])
        let header = try Safetensors.parseHeaderBytes(
            path: "maple.safetensors", fileSize: UInt64(8 + headerData.count + 8),
            headerBytes: headerData)

        #expect(header.tensors == [SourceTensor(
            name: "lm_head_flash.token_map", shardPath: "maple.safetensors",
            dtype: .i32, shape: [2], absoluteOffset: UInt64(8 + headerData.count),
            sizeBytes: 8)])
        #expect(RepackPlanner.classify(
            "lm_head_flash.token_map", numLayers: 24, family: .maple) == .excludedMultimodal)
        #expect(RepackPlanner.classify(
            "lm_head_flash.token_map", numLayers: 24, family: .deepseekV4Flash) == .unknown)
    }

    @Test func mapleManifestUsesFixedQuantSlotsAndBehavior() throws {
        let root = try makeSnapshot("manifest")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let arch = try ArchInfo.load(configPath: configPath(in: root))
        let plan = RepackPlan(
            arch: arch, baseMode: "bogus", baseGroupSize: 999, bitsOverrideCount: 17,
            resident: ResidentFilePlan(
                path: "weights.bin", entries: [], stringTable: [], stringTableOffsets: [],
                indexSize: 0, residentSize: 0),
            layers: [], matchedModelID: nil, excludedMultimodalTensorNames: [])

        let data = try GTurboJSON.encodeManifest(
            plan: plan, modelID: "maple-preview-2bit-mlx", sourceSnapshotHash: "sha256:0",
            files: [], expertsPerLayer: 256, numLayers: 24, expertStride: 16_384,
            bitWidths: .init(embedding: 7, attention: 8, router: 9, sharedExpert: 10, routedExpert: 11))
        let manifest = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let archDict = manifest["arch"] as! [String: Any]
        let quant = manifest["quant"] as! [String: [String: Any]]

        #expect(archDict["family"] as? String == "maple")
        #expect(archDict["routerScoringFunc"] as? String == "softmax")
        #expect(archDict["routedScalingFactor"] as? Double == 1.0)
        #expect(archDict["swigluLimit"] as? Double == 7.0)
        #expect(archDict["numSharedExperts"] as? Int == 0)
        #expect(archDict["numDenseLayers"] as? Int == 0)
        #expect(archDict["routerNormAfterTopK"] as? Bool == true)
        #expect(quant.count == 5)
        expectQuant(quant["embedding"], bits: 4, scheme: "affine", scale: "BF16", bias: "BF16", group: 64)
        expectQuant(quant["attention"], bits: 4, scheme: "affine", scale: "BF16", bias: "BF16", group: 64)
        expectQuant(quant["router"], bits: 16, scheme: "unquantized", scale: "none", bias: "none", group: 0)
        expectQuant(quant["sharedExpert"], bits: 0, scheme: "none", scale: "none", bias: "none", group: 0)
        expectQuant(quant["routedExpert"], bits: 2, scheme: "affine", scale: "BF16", bias: "BF16", group: 64)
    }

    private func makeSnapshot(
        _ label: String,
        mutate: (inout [String: Any]) -> Void = { _ in }) throws -> String {
        let root = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-maple-\(label)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        var config: [String: Any] = [
            "model_type": "maple",
            "hidden_size": 2_048,
            "moe_shared_expert_intermediate_size": 512,
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 151_936,
            "sliding_window": 512,
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
            "num_hidden_layers": 24,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_shared_experts": 0,
            "first_k_dense_replace": 0,
            "rms_norm_eps": 0.000_001,
            "layer_types": (0..<24).map {
                $0 % 4 == 3 ? "full_attention" : "sliding_attention"
            },
            "use_qk_norm": true,
            "norm_topk_prob": true,
            "tie_word_embeddings": false,
            "use_rmsnorm": true,
            "use_bias": false,
            "router_dtype": "fp32",
            "hidden_act": "silu",
            "quantization": [
                "bits": 2,
                "group_size": 128,
                "mode": "affine",
                "model.word_embeddings": ["bits": 4, "group_size": 64],
                "lm_head": ["bits": 4, "group_size": 64],
            ],
        ]
        mutate(&config)
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: configPath(in: root)))
        try JSONSerialization.data(withJSONObject: ["weight_map": [String: String]()])
            .write(to: URL(fileURLWithPath: (root as NSString)
                .appendingPathComponent("model.safetensors.index.json")))
        return root
    }

    private func configPath(in root: String) -> String {
        (root as NSString).appendingPathComponent("config.json")
    }

    private func expectQuant(_ quant: [String: Any]?, bits: Int, scheme: String,
                             scale: String, bias: String, group: Int) {
        #expect(quant?["weightBits"] as? Int == bits)
        #expect(quant?["scheme"] as? String == scheme)
        #expect(quant?["scaleType"] as? String == scale)
        #expect(quant?["biasType"] as? String == bias)
        #expect(quant?["groupSize"] as? Int == group)
    }
}
