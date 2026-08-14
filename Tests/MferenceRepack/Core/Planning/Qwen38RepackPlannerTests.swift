import Darwin
import Foundation
import Testing
@testable import MferenceRepackCore

@Suite
struct Qwen38RepackPlannerTests {

    @Test func qwen38ArchInfoLoadsFromSyntheticConfig() throws {
        let snapshotDir = temporaryRoot("qwen38-arch")
        defer { try? FileManager.default.removeItem(atPath: snapshotDir) }
        _ = try SyntheticSnapshot.buildQwen38(at: snapshotDir)

        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))

        #expect(arch.family == .qwen38)
        #expect(arch.hiddenSize == 128)
        #expect(arch.numLayers == 4)
        #expect(arch.fullAttentionLayerMask == [2, 2, 2, 1])
        #expect(arch.numExperts == 0)
        #expect(arch.topKExperts == 0)
        #expect(arch.moeIntermediateSize == 0)
        #expect(arch.intermediateSize == 64)
        #expect(arch.numSharedExperts == 0)
        #expect(arch.numDenseLayers == 4)
        #expect(arch.denseIntermediateSize == 64)
        #expect(arch.tieWordEmbeddings == false)
        #expect(arch.attentionKEqV == false)
        #expect(arch.hiddenActivation == "silu")
        #expect(arch.ropeTheta == 10_000_000.0)
        #expect(arch.fullRopeTheta == 10_000_000.0)
        #expect(arch.partialRotaryFactor == 0.25)
        #expect(arch.finalLogitSoftcap == 0.0)
        #expect(arch.attnOutputGate == true)
        #expect(arch.attentionScale == 0.125)   // 64^-0.5
        #expect(arch.embeddingScaledBySqrtHidden == false)
        #expect(arch.routerScaled == false)
        #expect(arch.ffnSandwichNorms == false)
        #expect(arch.sharedExpertGated == false)
        #expect(arch.ropeNeoxSubdim == true)
        #expect(arch.linearNumKHeads == 2)
        #expect(arch.linearNumVHeads == 4)
        #expect(arch.linearKeyHeadDim == 32)
        #expect(arch.linearValueHeadDim == 32)
        #expect(arch.linearConvKernelSize == 4)
    }

    @Test func productionQwen38ConfigParsesAndCrossChecks() throws {
        let root = temporaryRoot("qwen38-prod")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { _ in })

        let arch = try ArchInfo.load(configPath: configPath)
        #expect(arch.family == .qwen38)
        #expect(arch.hiddenSize == 5120)
        #expect(arch.numLayers == 64)
        #expect(arch.vocabSize == 248_320)
        #expect(arch.numExperts == 0)
        #expect(arch.topKExperts == 0)
        #expect(arch.intermediateSize == 17_408)
        #expect(arch.denseIntermediateSize == 17_408)
        #expect(arch.numDenseLayers == 64)
        #expect(arch.attentionScale == 0.0625) // 256^-0.5
        #expect(arch.linearNumKHeads == 16)
        #expect(arch.linearNumVHeads == 48)
        #expect(arch.linearKeyHeadDim == 128)
        #expect(arch.linearValueHeadDim == 128)
        #expect(arch.linearConvKernelSize == 4)
        #expect(arch.fullAttentionLayerMask.count == 64)
        #expect(arch.fullAttentionLayerMask.filter { $0 == 1 }.count == 16)
        for (i, v) in arch.fullAttentionLayerMask.enumerated() {
            #expect(v == ((i + 1) % 4 == 0 ? 1 : 2))
        }
    }

    @Test func productionQwen38ConfigMismatchIsRejected() throws {
        let root = temporaryRoot("qwen38-prod-bad")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { tc in
            tc["num_attention_heads"] = 8
        })

        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: configPath)
        }
    }

    @Test func qwen38SourceFingerprintIsKnown() {
        #expect(SourceFingerprint.modelID(forIndexSha256:
            "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8")
            == "qwen3.8-27b-4bit")
    }

    @Test func qwen38ClassificationBucketsNames() {
        let f = RepackModelFamily.qwen38
        // Dense: the layer's own MLP is resident, never a routed expert.
        #expect(RepackPlanner.classify(
            "language_model.model.layers.1.mlp.gate_proj.weight",
            numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "language_model.model.layers.2.mlp.down_proj.weight",
            numLayers: 4, family: f) == .lmResident)
        // Untied head and the DeltaNet bundle are resident.
        #expect(RepackPlanner.classify(
            "language_model.lm_head.weight", numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            numLayers: 4, family: f) == .lmResident)
        // Vision is excluded; unknown prefixes stay unknown.
        #expect(RepackPlanner.classify(
            "vision_tower.blocks.0.norm1.weight", numLayers: 4, family: f)
            == .excludedMultimodal)
        #expect(RepackPlanner.classify(
            "model.layers.0.mlp.gate_proj.weight",
            numLayers: 4, family: f) == .unknown)
    }

    @Test func qwen38PlanOrdersResidentsWithZeroExpertLayers() throws {
        let snapshotDir = temporaryRoot("qwen38-plan")
        let outputDir = temporaryRoot("qwen38-plan-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildQwen38(at: snapshotDir)
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
        // Embedding first; final norm then the untied lm_head last.
        #expect(names.first == "language_model.model.embed_tokens.weight")
        #expect(names.last == "language_model.lm_head.weight")
        #expect(names.dropLast().last == "language_model.model.norm.weight")

        // Layer 0 is a linear-attention layer: DeltaNet bundle, then the
        // layer's own MLP, then the two layer norms.
        let expectedLayer0 = [
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.conv1d.weight",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ].map { "language_model.model.layers.0." + $0 }
        let layer0 = names.filter { $0.contains(".layers.0.") }
        #expect(layer0 == expectedLayer0)

        // Layer 3 is the full-attention layer.
        let expectedLayer3 = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ].map { "language_model.model.layers.3." + $0 }
        let layer3 = names.filter { $0.contains(".layers.3.") }
        #expect(layer3 == expectedLayer3)

        // The dense MLP is a quantized U32 entry with companions.
        let gateProj = try #require(plan.resident.entries.first {
            $0.name == "language_model.model.layers.0.mlp.gate_proj.weight"
        })
        #expect(gateProj.dtype == 0)
        #expect(gateProj.quantSpec?.bits == 4)
        #expect(gateProj.sourceScales != nil)
        #expect(gateProj.sourceBiases != nil)

        // Dense: every layer plan is empty — no experts, no stride, no blob.
        #expect(plan.layers.count == 4)
        for lp in plan.layers {
            #expect(lp.expertsPerLayer == 0)
            #expect(lp.expertStride == 0)
            #expect(lp.subTensors.isEmpty)
        }

        // Vision-tower tensors are dropped, not planned.
        #expect(plan.excludedMultimodalTensorNames.sorted() == [
            "vision_tower.blocks.0.norm1.weight",
            "vision_tower.patch_embed.proj.weight",
        ])
        #expect(!names.contains { $0.hasPrefix("vision_tower.") })
    }

    @Test func qwen38ManifestDeclaresDenseFieldsAndAbsentSlots() throws {
        let snapshotDir = temporaryRoot("qwen38-manifest")
        let outputDir = temporaryRoot("qwen38-manifest-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildQwen38(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let plan = try RepackPlanner.plan(
            meta: metadata, arch: arch, shardHeaders: [header], outputDir: outputDir)

        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "unknown/snapshot",
            sourceSnapshotHash: "sha256:0",
            files: [],
            expertsPerLayer: 0,
            numLayers: arch.numLayers,
            expertStride: 0,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 4, attention: 4, router: 0,
                sharedExpert: 0, routedExpert: 0))
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let archDict = obj["arch"] as! [String: Any]
        #expect(archDict["family"] as? String == "qwen38")
        #expect(archDict["numExperts"] as? Int == 0)
        #expect(archDict["topKExperts"] as? Int == 0)
        #expect(archDict["numSharedExperts"] as? Int == 0)
        #expect(archDict["numDenseLayers"] as? Int == 4)
        #expect(archDict["denseIntermediateSize"] as? Int == 64)
        #expect(archDict["linearNumKHeads"] as? Int == 2)
        #expect(archDict["linearNumVHeads"] as? Int == 4)

        let quant = obj["quant"] as! [String: [String: Any]]
        for slot in ["embedding", "attention"] {
            #expect(quant[slot]?["weightBits"] as? Int == 4)
            #expect(quant[slot]?["scheme"] as? String == "affine")
            #expect(quant[slot]?["groupSize"] as? Int == 64)
        }
        for slot in ["router", "sharedExpert", "routedExpert"] {
            #expect(quant[slot]?["weightBits"] as? Int == 0)
            #expect(quant[slot]?["scheme"] as? String == "none")
            #expect(quant[slot]?["scaleType"] as? String == "none")
            #expect(quant[slot]?["biasType"] as? String == "none")
            #expect(quant[slot]?["groupSize"] as? Int == 0)
        }
    }

    // MARK: - Helpers

    private func writeProductionConfig(
        to path: String,
        mutate: (inout [String: Any]) -> Void) throws {
        var layerTypes: [String] = []
        for i in 0..<64 {
            layerTypes.append((i + 1) % 4 == 0 ? "full_attention" : "linear_attention")
        }
        var tc: [String: Any] = [
            "hidden_size": 5120,
            "intermediate_size": 17_408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248_320,
            "num_hidden_layers": 64,
            "layer_types": layerTypes,
            "full_attention_interval": 4,
            "rope_parameters": [
                "rope_theta": 10_000_000.0,
                "rope_type": "default",
                "partial_rotary_factor": 0.25
            ],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "attn_output_gate": true,
            "tie_word_embeddings": false,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        ]
        mutate(&tc)
        let config: [String: Any] = [
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "quantization": ["bits": 4, "group_size": 64, "mode": "affine"],
            "text_config": tc
        ]
        let data = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-qwen38-plan-\(tag)-\(UUID().uuidString)")
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
