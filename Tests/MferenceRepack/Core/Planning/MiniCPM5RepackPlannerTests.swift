import Darwin
import Foundation
import Testing
@testable import MferenceRepackCore

/// The repacker's contract for the `minicpm5` family: the flat llama config
/// loader with its production cross-check, the two source entries, the
/// classifiers and orderings both install paths share, and the quantization
/// policy (uniform INT4, no norm fold) read from the vendor's own control.
@Suite
struct MiniCPM5RepackPlannerTests {

    @Test func miniCPM5ArchInfoLoadsFromSyntheticConfig() throws {
        let snapshotDir = temporaryRoot("arch")
        defer { try? FileManager.default.removeItem(atPath: snapshotDir) }
        _ = try SyntheticSnapshot.buildMiniCPM5(at: snapshotDir)

        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))

        #expect(arch.family == .minicpm5)
        #expect(arch.hiddenSize == 128)
        #expect(arch.intermediateSize == 64)
        #expect(arch.denseIntermediateSize == 64)
        #expect(arch.numLayers == 4)
        #expect(arch.numDenseLayers == 4)
        #expect(arch.fullAttentionLayerMask == [1, 1, 1, 1])
        #expect(arch.numHeads == 2)
        #expect(arch.numKVHeads == 2)
        #expect(arch.numFullKVHeads == 2)
        #expect(arch.headDim == 64)
        #expect(arch.fullHeadDim == 64)
        #expect(arch.vocabSize == 256)
        #expect(arch.numExperts == 0)
        #expect(arch.topKExperts == 0)
        #expect(arch.numSharedExperts == 0)
        #expect(arch.moeIntermediateSize == 0)
        #expect(arch.tieWordEmbeddings == false)
        #expect(arch.attentionKEqV == false)
        #expect(arch.hiddenActivation == "silu")
        #expect(arch.ropeTheta == 5_000_000.0)
        #expect(arch.fullRopeTheta == 5_000_000.0)
        #expect(arch.partialRotaryFactor == 1.0)
        #expect(arch.ropeNeoxSubdim == true)
        #expect(arch.finalLogitSoftcap == 0.0)
        #expect(arch.attnOutputGate == false)
        #expect(arch.attentionScale == 0.125)   // pow(64, -0.5), exact
        #expect(arch.qkNorm == false)
        #expect(arch.embeddingScaledBySqrtHidden == false)
        #expect(arch.routerScaled == false)
        #expect(arch.ffnSandwichNorms == false)
        #expect(arch.sharedExpertGated == false)
        #expect(arch.linearNumKHeads == 0)
        #expect(arch.linearConvKernelSize == 0)
        #expect(arch.slidingWindow == 0)
    }

    /// The real `config.json` of `openbmb/MiniCPM5-2B` @ `cd199ce3`, verbatim,
    /// parses and matches the pinned baseline field by field.
    @Test func productionMiniCPM5ConfigParsesAndCrossChecks() throws {
        let root = temporaryRoot("prod")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { _ in })

        let arch = try ArchInfo.load(configPath: configPath)
        #expect(arch.family == .minicpm5)
        #expect(arch.hiddenSize == 2048)
        #expect(arch.intermediateSize == 6144)
        #expect(arch.numLayers == 42)
        #expect(arch.numHeads == 16)
        #expect(arch.numKVHeads == 2)
        #expect(arch.headDim == 128)
        #expect(arch.vocabSize == 130_560)
        #expect(arch.ropeTheta == 5_000_000.0)
        #expect(arch.fullAttentionLayerMask.count == 42)
        #expect(arch.fullAttentionLayerMask.allSatisfy { $0 == 1 })
        // `head_dim ** -0.5` in transformers; the runtime baseline carries the
        // same bits, so the manifest's exact Double check will pass.
        #expect(arch.attentionScale == 0.08838834764831845)
        #expect(arch.attentionScale == pow(128.0, -0.5))
        #expect(arch.qkNorm == false)
    }

    /// One JSON mutation of the production config, as a `Sendable` test
    /// argument: the key and a JSON-encoded replacement value.
    private struct Mutation: Sendable, CustomTestStringConvertible {
        let key: String
        let json: String
        var testDescription: String { "\(key) = \(json)" }
        var value: Any {
            try! JSONSerialization.jsonObject(with: Data(json.utf8),
                                              options: [.fragmentsAllowed])
        }
    }

    @Test("a production-shaped llama config that is not MiniCPM5-2B is refused",
          arguments: [
            Mutation(key: "num_attention_heads", json: "8"),
            Mutation(key: "intermediate_size", json: "8192"),
            Mutation(key: "rope_theta", json: "10000.0"),
            Mutation(key: "tie_word_embeddings", json: "true"),
            Mutation(key: "vocab_size", json: "130000"),
          ])
    private func productionMiniCPM5ConfigMismatchIsRejected(mutation: Mutation) throws {
        let root = temporaryRoot("prod-bad")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { config in
            config[mutation.key] = mutation.value
        })
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: configPath)
        }
    }

    /// The things the runner hard-codes are refused at install when a config
    /// disagrees, rather than silently running a different model.
    @Test("hard-coded runtime facts are checked, not assumed",
          arguments: [
            Mutation(key: "rms_norm_eps", json: "1e-5"),
            Mutation(key: "rope_scaling", json: #"{"rope_type": "linear", "factor": 2.0}"#),
            Mutation(key: "attention_bias", json: "true"),
            Mutation(key: "mlp_bias", json: "true"),
            Mutation(key: "architectures", json: #"["MiniCPMForCausalLM"]"#),
          ])
    private func runtimeAssumptionsAreCheckedAtLoad(mutation: Mutation) throws {
        let root = temporaryRoot("assume")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { config in
            config[mutation.key] = mutation.value
        })
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: configPath)
        }
    }

    @Test func miniCPM5SourceEntriesArePinned() {
        let bf16 = try? #require(SupportedModelSource.named("minicpm5"))
        #expect(bf16?.repoID == "openbmb/MiniCPM5-2B")
        #expect(bf16?.kind == .originalRepoQuantize)
        #expect(bf16?.isPinned == true)
        #expect(bf16?.revision == "cd199ce3ee67549c42ef7372f809f2c63599a3e9")
        #expect(SourceFingerprint.modelID(forIndexSha256:
            "6d839cd76e8395de548a0e6cc310386f66d1ecbb2c75d198a8dfd3d70892b756")
            == "minicpm5-2b-int4g64")

        let control = try? #require(SupportedModelSource.named("minicpm5mlx"))
        #expect(control?.repoID == "openbmb/MiniCPM5-2B-MLX")
        #expect(control?.kind == .preQuantized)
        #expect(control?.isPinned == true)
        #expect(control?.revision == "35ac38ee7bdb0bf7fa748d0700eeb6d6675760a3")
        #expect(SourceFingerprint.modelID(forIndexSha256:
            "ccf202e0a06fe3c7eb8f354cfb29412a5e64956ad895413d4d9267ae4b3a6045")
            == "minicpm5-2b-mlx-4bit")
        // Two installs of one checkpoint must stay tellable apart.
        #expect(bf16?.modelID != control?.modelID)
    }

    /// A missing `quantization` block is accepted only when the caller vouches
    /// for the source; `"llama"` alone is never enough.
    @Test func bf16LlamaSnapshotNeedsAVouchedSource() throws {
        let snapshotDir = temporaryRoot("vouch")
        defer { try? FileManager.default.removeItem(atPath: snapshotDir) }
        _ = try SyntheticSnapshot.buildMiniCPM5(at: snapshotDir)
        #expect(!IndexLoader.unquantizedSourceModelTypes.contains("llama"))
        #expect(throws: RepackError.self) {
            _ = try IndexLoader.load(snapshotDir: snapshotDir)
        }
        let meta = try IndexLoader.load(snapshotDir: snapshotDir,
                                        acceptsUnquantizedSource: true)
        #expect(meta.sourceIsUnquantized)
        #expect(meta.baseBits == 4)
        #expect(meta.baseGroupSize == 64)
        #expect(meta.bitsOverrides.isEmpty)
    }

    @Test func miniCPM5ClassificationBucketsNames() {
        let f = RepackModelFamily.minicpm5
        #expect(RepackPlanner.classify("model.layers.1.mlp.gate_proj.weight",
                                       numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify("model.layers.2.mlp.down_proj.weight",
                                       numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify("model.layers.0.self_attn.q_proj.weight",
                                       numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify("model.embed_tokens.weight",
                                       numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify("lm_head.weight",
                                       numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify("model.visual.patch_embed.proj.weight",
                                       numLayers: 4, family: f) == .excludedMultimodal)
        #expect(RepackPlanner.classify("language_model.model.layers.0.mlp.gate_proj.weight",
                                       numLayers: 4, family: f) == .unknown)
    }

    /// Both install paths lay a layer out the same way: the four attention
    /// projections, the SwiGLU MLP, then the two layer norms.
    @Test func bothPathsShareTheLayerOrder() {
        let layer = [
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight",
        ].map { "model.layers.0." + $0 }
        let all = ["model.embed_tokens.weight"] + layer
            + ["model.layers.1.self_attn.q_proj.weight", "model.norm.weight", "lm_head.weight"]
        let shuffled = all.reversed()
        let inFlight = shuffled.sorted {
            FlashNextPlanner.residentOrdering($0, $1, family: .minicpm5)
        }
        #expect(inFlight == all)
        for (rank, name) in layer.enumerated() {
            #expect(RepackPlanner.miniCPM5SlotRank(in: name) == rank, Comment(rawValue: name))
        }
    }

    @Test func miniCPM5QuantizationPolicyMirrorsTheVendorControl() {
        #expect(QuantBitPolicy.originalRepo(family: .minicpm5) == .uniformInt4)
        for name in ["model.embed_tokens.weight", "lm_head.weight",
                     "model.layers.3.self_attn.k_proj.weight",
                     "model.layers.3.mlp.down_proj.weight"] {
            #expect(QuantBitPolicy.originalRepo(family: .minicpm5)
                .bits(forTensorNamed: name) == 4, Comment(rawValue: name))
            #expect(FlashNextPlanner.residentName(for: name, family: .minicpm5) == name)
        }
        // LlamaRMSNorm is plain w * x_hat: no (1 + w) fold on any norm.
        for name in ["model.norm.weight",
                     "model.layers.0.input_layernorm.weight",
                     "model.layers.41.post_attention_layernorm.weight"] {
            #expect(!FlashNextPlanner.foldsNormBias(name, family: .minicpm5), Comment(rawValue: name))
        }
    }

    // MARK: - Helpers

    /// `openbmb/MiniCPM5-2B` `config.json` @ `cd199ce3` (704 bytes on the Hub).
    private func writeProductionConfig(
        to path: String,
        mutate: (inout [String: Any]) -> Void) throws {
        var config: [String: Any] = [
            "_name_or_path": "openbmb/MiniCPM5-2B",
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 0,
            "eos_token_id": [1, 130_073],
            "pad_token_id": 1,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "max_position_embeddings": 131_072,
            "model_type": "llama",
            "num_attention_heads": 16,
            "num_hidden_layers": 42,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "rms_norm_eps": 1e-06,
            "rope_theta": 5_000_000,
            "rope_scaling": NSNull(),
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "transformers_version": "5.6.2",
            "use_cache": true,
            "vocab_size": 130_560,
        ]
        mutate(&config)
        let data = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-minicpm5-plan-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: path,
                                                 withIntermediateDirectories: true)
        return path
    }
}
