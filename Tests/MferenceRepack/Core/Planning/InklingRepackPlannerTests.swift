import Foundation
import Testing
@testable import MferenceRepackCore

@Suite
struct InklingRepackPlannerTests {

    /// `text_config` of `pipenetwork/Inkling-Small-MLX-4bit` revision
    /// `9d6e4720`, trimmed to the fields the repacker reads. Shaped like the
    /// production checkpoint so the cross-check is exercised.
    private static func productionConfigJSON(
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
        let dir = NSTemporaryDirectory() + "inkling-\(label)-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        let path = (dir as NSString).appendingPathComponent("config.json")
        try json.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    @Test func inklingArchInfoLoadsFromProductionConfig() throws {
        let path = try writeConfig(Self.productionConfigJSON(), label: "arch")
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

    /// The runtime pins every hyperparameter, so a checkpoint that claims the
    /// production shape but disagrees must fail at convert time rather than
    /// producing a `.gturbo` the loader will reject much later.
    @Test func productionCrossCheckRejectsDivergentConfig() throws {
        let path = try writeConfig(
            Self.productionConfigJSON(overrides: ["num_experts_per_tok": 8]),
            label: "topk")
        defer { try? FileManager.default.removeItem(
            atPath: (path as NSString).deletingLastPathComponent) }

        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: path)
        }
    }

    @Test func nonProductionShapeSkipsCrossCheck() throws {
        // A toy config keeps its own (non-production) values.
        let path = try writeConfig(
            Self.productionConfigJSON(overrides: [
                "hidden_size": 256,
                "num_hidden_layers": 6,
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "local_layer_ids": [0, 1, 2, 3, 4],
            ]),
            label: "toy")
        defer { try? FileManager.default.removeItem(
            atPath: (path as NSString).deletingLastPathComponent) }

        let arch = try ArchInfo.load(configPath: path)
        #expect(arch.hiddenSize == 256)
        #expect(arch.numLayers == 6)
        #expect(arch.fullAttentionLayerMask == [0, 0, 0, 0, 0, 1])
    }

    @Test func rejectsOutOfRangeLocalLayerIDs() throws {
        let path = try writeConfig(
            Self.productionConfigJSON(overrides: [
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "local_layer_ids": [0, 9],
            ]),
            label: "badlocal")
        defer { try? FileManager.default.removeItem(
            atPath: (path as NSString).deletingLastPathComponent) }

        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: path)
        }
    }

    @Test func inklingClassificationBucketsNames() {
        let f = RepackModelFamily.inklingSmall
        // Routed experts stream; everything else in the text tower is resident.
        #expect(RepackPlanner.classify(
            "model.llm.layers.3.mlp.experts.gate_proj.weight",
            numLayers: 42, family: f)
            == .routedExpert(role: "gate", layer: 3))
        #expect(RepackPlanner.classify(
            "model.llm.layers.40.mlp.experts.down_proj.weight",
            numLayers: 42, family: f)
            == .routedExpert(role: "down", layer: 40))

        // Shared experts sit under `.mlp.shared_experts.` and must NOT be
        // mistaken for routed experts.
        #expect(RepackPlanner.classify(
            "model.llm.layers.3.mlp.shared_experts.gate_proj.weight",
            numLayers: 42, family: f) == .lmResident)
        // Dense-FFN layers use the bare `.mlp.` names.
        #expect(RepackPlanner.classify(
            "model.llm.layers.0.mlp.gate_proj.weight",
            numLayers: 42, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "model.llm.layers.7.attn.wr_du.weight",
            numLayers: 42, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "model.llm.layers.7.attn.rel_logits_proj.proj",
            numLayers: 42, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "model.llm.layers.7.attn_sconv.weight",
            numLayers: 42, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "model.llm.embed.weight", numLayers: 42, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "model.llm.unembed.weight", numLayers: 42, family: f) == .lmResident)

        // The vision and audio towers are excluded from the text path.
        #expect(RepackPlanner.classify(
            "model.visual.layers.linear_0.weight",
            numLayers: 42, family: f) == .excludedMultimodal)
        #expect(RepackPlanner.classify(
            "model.audio.encoder.weight",
            numLayers: 42, family: f) == .excludedMultimodal)

        // Other families' prefixes are not this family's contract.
        #expect(RepackPlanner.classify(
            "model.layers.0.ffn.switch_mlp.gate_proj.weight",
            numLayers: 42, family: f) == .unknown)
        #expect(RepackPlanner.classify(
            "model.llm.layers.0.mlp.experts.gate_proj.weight",
            numLayers: 42, family: .deepseekV4Flash) == .lmResident)
    }

    @Test func inklingSourceIsPinnedAndRegistered() throws {
        let s = try #require(SupportedModelSource.named("inklingsmall"))
        #expect(s.repoID == "pipenetwork/Inkling-Small-MLX-4bit")
        #expect(s.revision == "9d6e4720ab7002af25d6129c88ccea6cd9f19372")
        #expect(s.sourceIndexSHA256
                == "fe16aec3cef12438f1d0ff657f7e785781b61271528a66b3b7160fcf1aaca30c")
        #expect(SupportedModelSource.all.contains { $0.name == "inklingsmall" })
    }
}

extension InklingRepackPlannerTests {
    /// `--verify-install` accepts empty expert layouts ONLY inside the
    /// manifest's declared dense-layer prefix; an empty ROUTED layer is a
    /// broken install the runtime would reject, and verification must not
    /// certify it (PR #4 review, Codex P2).
    @Test func verifyRejectsEmptyLayoutOutsideDenseLayers() throws {
        func makeInstall(numDenseLayers: Int, emptyLayer: Int) throws -> String {
            let dir = NSTemporaryDirectory() + "vfy-\(UUID().uuidString)"
            let pe = dir + "/packed_experts"
            try FileManager.default.createDirectory(
                atPath: pe, withIntermediateDirectories: true)
            let stride: UInt64 = 16384
            let numLayers = 3
            var layers: [[String: Any]] = []
            var files: [String: Any] = [
                "packed_experts/layout.json": ["size": 1, "sha256": "00"],
            ]
            for L in 0..<numLayers {
                let file = String(format: "layer_%02d.bin", L)
                if L == emptyLayer || L < numDenseLayers {
                    layers.append(["layer": L, "file": file, "experts": []])
                    continue
                }
                layers.append(["layer": L, "file": file,
                               "experts": [["expert": 0, "offset": 0,
                                            "size": stride,
                                            "tensors": [String: Any]()]]])
                let data = Data(count: Int(stride))
                try data.write(to: URL(fileURLWithPath: "\(pe)/\(file)"))
                files["packed_experts/\(file)"] = ["size": stride, "sha256": "00"]
            }
            let layout: [String: Any] = [
                "expertStride": stride, "numLayers": numLayers,
                "expertsPerLayer": 1, "layers": layers,
            ]
            try JSONSerialization.data(withJSONObject: layout)
                .write(to: URL(fileURLWithPath: "\(pe)/layout.json"))
            let manifest: [String: Any] = [
                "files": files, "expertsPerLayer": 1, "numLayers": numLayers,
                "expertStride": stride,
                "arch": ["numDenseLayers": numDenseLayers],
            ]
            try JSONSerialization.data(withJSONObject: manifest)
                .write(to: URL(fileURLWithPath: "\(dir)/manifest.json"))
            return dir
        }

        // Empty layout confined to the dense prefix: accepted.
        let ok = try makeInstall(numDenseLayers: 2, emptyLayer: 1)
        defer { try? FileManager.default.removeItem(atPath: ok) }
        try VerifiedInstallTool.validatePackedExpertLayout(inputGTurbo: ok)

        // Empty layout on a ROUTED layer: rejected.
        let bad = try makeInstall(numDenseLayers: 1, emptyLayer: 2)
        defer { try? FileManager.default.removeItem(atPath: bad) }
        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.validatePackedExpertLayout(inputGTurbo: bad)
        }
    }
}
