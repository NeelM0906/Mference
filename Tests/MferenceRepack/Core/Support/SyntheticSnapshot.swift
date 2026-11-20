import Foundation
@testable import MferenceRepackCore

/// Synthesises a tiny MLX-affine-quantized safetensors snapshot inside a
/// temporary directory. The remote repack tests need only deterministic bytes
/// plus matching `config.json` and `model.safetensors.index.json` metadata.
enum SyntheticSnapshot {

    struct Snapshot {
        /// First (or only) shard. Single-shard fixtures use this directly.
        let shardPath: String
        /// Every shard the fixture wrote, in index order.
        let shardPaths: [String]

        init(shardPath: String, shardPaths: [String]? = nil) {
            self.shardPath = shardPath
            self.shardPaths = shardPaths ?? [shardPath]
        }
    }

    enum MapleTensorMutation {
        case remove(String)
        case replaceDtype(String, String)
        case replaceShape(String, [Int])
        case addAffineCompanions(String)
    }

    // MARK: - Maple variant

    /// Pinned Maple metadata with the smallest tensor inventory that exercises
    /// ternary resident widening and routed-expert planning.
    static func buildMaple(
        at dir: String,
        seed: UInt64 = 0x4D41_504C_455F_0002,
        includeFlashHead: Bool = false,
        mutations: [MapleTensorMutation] = []) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        var rng = SplitMix64(seed: seed)
        var tensors: [(String, String, [Int], [UInt8])] = []
        let input = 128
        let rows = 2
        let experts = 256

        appendQuantizedWeight(name: "model.word_embeddings", outerShape: [rows],
                              innerLogical: input, bits: 4, groupSize: 64,
                              into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "lm_head", outerShape: [rows],
                              innerLogical: input, bits: 4, groupSize: 64,
                              into: &tensors, rng: &rng)
        appendMapleTernaryWeight(name: "model.layers.0.self_attn.q_proj",
                                 outerShape: [rows], innerLogical: input,
                                 into: &tensors, rng: &rng)
        for layer in 0..<24 {
            for role in ["gate", "up", "down"] {
                appendMapleTernaryWeight(
                    name: "model.layers.\(layer).mlp.switch_mlp.\(role)_proj",
                    outerShape: [experts, rows], innerLogical: input,
                    into: &tensors, rng: &rng)
            }
        }
        if includeFlashHead {
            appendMapleFlashHead(into: &tensors, rng: &rng)
        }

        for mutation in mutations {
            switch mutation {
            case .remove(let name):
                tensors.removeAll { $0.0 == name }
            case .replaceDtype(let name, let dtype):
                guard let index = tensors.firstIndex(where: { $0.0 == name }) else {
                    preconditionFailure("unknown Maple fixture tensor \(name)")
                }
                tensors[index].1 = dtype
                tensors[index].3 = maplePayload(dtype: dtype, shape: tensors[index].2,
                                                 rng: &rng)
            case .replaceShape(let name, let shape):
                guard let index = tensors.firstIndex(where: { $0.0 == name }) else {
                    preconditionFailure("unknown Maple fixture tensor \(name)")
                }
                tensors[index].2 = shape
                tensors[index].3 = maplePayload(dtype: tensors[index].1, shape: shape,
                                                 rng: &rng)
            case .addAffineCompanions(let base):
                tensors.append((base + ".scales", "BF16", [1], maplePayload(
                    dtype: "BF16", shape: [1], rng: &rng)))
                tensors.append((base + ".biases", "BF16", [1], maplePayload(
                    dtype: "BF16", shape: [1], rng: &rng)))
            }
        }

        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        var config: [String: Any] = [
            "model_type": "maple",
            "hidden_size": 2_048,
            "moe_shared_expert_intermediate_size": 512,
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 151_936,
            "max_position_embeddings": 128_000,
            "sliding_window": 512,
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
            "num_hidden_layers": 24,
            "num_experts": experts,
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
        if includeFlashHead {
            config["flash_head"] = [
                "n_clusters": 4_748,
                "cluster_size": 32,
                "n_probes": 512,
                "group_size": 64,
                "bits": 4,
                "head_group_size": 64,
                "head_bits": 4,
                "scaled_centroids": true,
                "force_tokens": [151_645, 151_668, 151_643],
            ]
        }
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString)
                .appendingPathComponent("config.json")))

        let weightMap = Dictionary(uniqueKeysWithValues: tensors.map { ($0.0, shardName) })
        try JSONSerialization.data(withJSONObject: ["weight_map": weightMap],
                                   options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString)
                .appendingPathComponent("model.safetensors.index.json")))
        return Snapshot(shardPath: shardPath)
    }

    // MARK: - Qwen 3.6 variant

    /// Tiny qwen3_5_moe-shaped architecture: a hybrid of three gated-DeltaNet
    /// linear-attention layers and one full-attention layer, two routed
    /// experts, and an untied lm_head.
    struct QwenArch {
        let hidden: Int = 128
        let moeIntermediate: Int = 64
        let sharedIntermediate: Int = 64
        let numHeads: Int = 2
        let numKVHeads: Int = 2
        let headDim: Int = 64
        let vocab: Int = 256
        let numLayers: Int = 4
        let numExperts: Int = 2
        let topK: Int = 2
        let groupSize: Int = 64
        let linearNumKHeads: Int = 2
        let linearNumVHeads: Int = 4
        let linearKeyHeadDim: Int = 32
        let linearValueHeadDim: Int = 32
        let linearConvKernelSize: Int = 4
        // layers 0-2 = linear attention, layer 3 = full attention
        let layerTypes: [String] = ["linear_attention", "linear_attention",
                                    "linear_attention", "full_attention"]
        /// Fused qkv rows: 2 * K-dim + V-dim. Also the conv1d channel count.
        var qkvDim: Int { 2 * linearNumKHeads * linearKeyHeadDim + linearNumVHeads * linearValueHeadDim }
        var valueDim: Int { linearNumVHeads * linearValueHeadDim }
    }

    static func buildQwen(at dir: String,
                          seed: UInt64 = 0xC0FF_EE00_9A11_AB1E) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        let arch = QwenArch()
        var rng = SplitMix64(seed: seed)

        var tensors: [(String, String, [Int], [UInt8])] = []
        tensors.reserveCapacity(96)

        // -- Embedding + untied lm_head (4-bit, group=64)
        appendQuantizedWeight(name: "language_model.model.embed_tokens",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "language_model.lm_head",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        for li in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            if arch.layerTypes[li] == "full_attention" {
                // q_proj emits per-head [query ; gate] halves (attn_output_gate).
                appendQuantizedWeight(name: prefix + ".self_attn.q_proj",
                                      outerShape: [2 * arch.numHeads * arch.headDim],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.k_proj",
                                      outerShape: [arch.numKVHeads * arch.headDim],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.v_proj",
                                      outerShape: [arch.numKVHeads * arch.headDim],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.o_proj",
                                      outerShape: [arch.hidden],
                                      innerLogical: arch.numHeads * arch.headDim, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.q_norm.weight",
                                      shape: [arch.headDim], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.k_norm.weight",
                                      shape: [arch.headDim], into: &tensors, rng: &rng)
            } else {
                // Gated-DeltaNet linear attention bundle.
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_qkv",
                                      outerShape: [arch.qkvDim],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_z",
                                      outerShape: [arch.valueDim],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_a",
                                      outerShape: [arch.linearNumVHeads],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_b",
                                      outerShape: [arch.linearNumVHeads],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.conv1d.weight",
                                      shape: [arch.qkvDim, arch.linearConvKernelSize, 1],
                                      into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.A_log",
                                      shape: [arch.linearNumVHeads], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.dt_bias",
                                      shape: [arch.linearNumVHeads], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.norm.weight",
                                      shape: [arch.linearValueHeadDim], into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.out_proj",
                                      outerShape: [arch.hidden],
                                      innerLogical: arch.valueDim, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
            }

            // Router + sigmoid-gated shared expert — 8-bit gates, 4-bit MLP.
            appendQuantizedWeight(name: prefix + ".mlp.gate",
                                  outerShape: [arch.numExperts], innerLogical: arch.hidden,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert_gate",
                                  outerShape: [1], innerLogical: arch.hidden,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert.gate_proj",
                                  outerShape: [arch.sharedIntermediate], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert.up_proj",
                                  outerShape: [arch.sharedIntermediate], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert.down_proj",
                                  outerShape: [arch.hidden], innerLogical: arch.sharedIntermediate,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Routed experts — stacked expert-major, 4-bit.
            appendQuantizedWeight(name: prefix + ".mlp.switch_mlp.gate_proj",
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.switch_mlp.up_proj",
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.switch_mlp.down_proj",
                                  outerShape: [arch.numExperts, arch.hidden],
                                  innerLogical: arch.moeIntermediate, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Plain pre-norm layer norms (no sandwich norms).
            appendUnquantizedBF16(name: prefix + ".input_layernorm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".post_attention_layernorm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
        }
        // Final norm
        appendUnquantizedBF16(name: "language_model.model.norm.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)

        // Vision-tower tensors included to prove the text-only repacker drops them.
        appendUnquantizedBF16(name: "vision_tower.blocks.0.norm1.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)
        appendUnquantizedBF16(name: "vision_tower.patch_embed.proj.weight",
                              shape: [arch.hidden, arch.hidden], into: &tensors, rng: &rng)

        // -- Encode safetensors.
        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        // -- Write config.json with 8-bit overrides for the router + gate.
        var quant: [String: Any] = [
            "bits": 4, "group_size": arch.groupSize, "mode": "affine"
        ]
        for li in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            for k in ["mlp.gate", "mlp.shared_expert_gate"] {
                quant[prefix + "." + k] = ["bits": 8, "group_size": arch.groupSize]
            }
        }

        let textConfig: [String: Any] = [
            "hidden_size": arch.hidden,
            "moe_intermediate_size": arch.moeIntermediate,
            "shared_expert_intermediate_size": arch.sharedIntermediate,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "head_dim": arch.headDim,
            "vocab_size": arch.vocab,
            "num_hidden_layers": arch.numLayers,
            "num_experts": arch.numExperts,
            "num_experts_per_tok": arch.topK,
            "layer_types": arch.layerTypes,
            "rope_parameters": [
                "rope_theta": 10_000_000.0,
                "rope_type": "default",
                "partial_rotary_factor": 0.25
            ],
            "linear_num_key_heads": arch.linearNumKHeads,
            "linear_num_value_heads": arch.linearNumVHeads,
            "linear_key_head_dim": arch.linearKeyHeadDim,
            "linear_value_head_dim": arch.linearValueHeadDim,
            "linear_conv_kernel_dim": arch.linearConvKernelSize,
            "attn_output_gate": true,
            "tie_word_embeddings": false,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        ]
        let config: [String: Any] = [
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "quantization": quant,
            "text_config": textConfig
        ]
        let configData = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try configData.write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))

        // -- Write model.safetensors.index.json.
        var weightMap: [String: String] = [:]
        for (name, _, _, _) in tensors { weightMap[name] = shardName }
        let indexObj: [String: Any] = [
            "metadata": ["format": "mlx"],
            "weight_map": weightMap
        ]
        let indexData = try JSONSerialization.data(withJSONObject: indexObj, options: [.sortedKeys])
        let indexPath = (dir as NSString).appendingPathComponent("model.safetensors.index.json")
        try indexData.write(to: URL(fileURLWithPath: indexPath))
        return Snapshot(shardPath: shardPath)
    }

    // MARK: - Qwen 3.8 variant

    /// Tiny qwen3_5-shaped architecture: the Qwen 3.6 hybrid attention
    /// schedule (three gated-DeltaNet layers, one full-attention layer) but
    /// dense — one SwiGLU MLP per layer, no router, no shared expert, no
    /// routed experts — with an untied lm_head and a nested `text_config`.
    struct Qwen38Arch {
        let hidden: Int = 128
        let intermediate: Int = 64
        let numHeads: Int = 2
        let numKVHeads: Int = 2
        let headDim: Int = 64
        let vocab: Int = 256
        let numLayers: Int = 4
        let groupSize: Int = 64
        let linearNumKHeads: Int = 2
        let linearNumVHeads: Int = 4
        let linearKeyHeadDim: Int = 32
        let linearValueHeadDim: Int = 32
        let linearConvKernelSize: Int = 4
        // layers 0-2 = linear attention, layer 3 = full attention
        let layerTypes: [String] = ["linear_attention", "linear_attention",
                                    "linear_attention", "full_attention"]
        /// Fused qkv rows: 2 * K-dim + V-dim. Also the conv1d channel count.
        var qkvDim: Int { 2 * linearNumKHeads * linearKeyHeadDim + linearNumVHeads * linearValueHeadDim }
        var valueDim: Int { linearNumVHeads * linearValueHeadDim }
    }

    static func buildQwen38(at dir: String,
                            seed: UInt64 = 0x0038_D0C5_E11A_B1E5) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        let arch = Qwen38Arch()
        var rng = SplitMix64(seed: seed)

        var tensors: [(String, String, [Int], [UInt8])] = []
        tensors.reserveCapacity(80)

        // -- Embedding + untied lm_head (4-bit, group=64)
        appendQuantizedWeight(name: "language_model.model.embed_tokens",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "language_model.lm_head",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        for li in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            if arch.layerTypes[li] == "full_attention" {
                // q_proj emits per-head [query ; gate] halves (attn_output_gate).
                appendQuantizedWeight(name: prefix + ".self_attn.q_proj",
                                      outerShape: [2 * arch.numHeads * arch.headDim],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.k_proj",
                                      outerShape: [arch.numKVHeads * arch.headDim],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.v_proj",
                                      outerShape: [arch.numKVHeads * arch.headDim],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.o_proj",
                                      outerShape: [arch.hidden],
                                      innerLogical: arch.numHeads * arch.headDim, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.q_norm.weight",
                                      shape: [arch.headDim], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.k_norm.weight",
                                      shape: [arch.headDim], into: &tensors, rng: &rng)
            } else {
                // Gated-DeltaNet linear attention bundle.
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_qkv",
                                      outerShape: [arch.qkvDim],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_z",
                                      outerShape: [arch.valueDim],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_a",
                                      outerShape: [arch.linearNumVHeads],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_b",
                                      outerShape: [arch.linearNumVHeads],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.conv1d.weight",
                                      shape: [arch.qkvDim, arch.linearConvKernelSize, 1],
                                      into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.A_log",
                                      shape: [arch.linearNumVHeads], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.dt_bias",
                                      shape: [arch.linearNumVHeads], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.norm.weight",
                                      shape: [arch.linearValueHeadDim], into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.out_proj",
                                      outerShape: [arch.hidden],
                                      innerLogical: arch.valueDim, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
            }

            // The layer's own SwiGLU MLP — 4-bit, no router in front of it.
            appendQuantizedWeight(name: prefix + ".mlp.gate_proj",
                                  outerShape: [arch.intermediate], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.up_proj",
                                  outerShape: [arch.intermediate], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.down_proj",
                                  outerShape: [arch.hidden], innerLogical: arch.intermediate,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Plain pre-norm layer norms.
            appendUnquantizedBF16(name: prefix + ".input_layernorm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".post_attention_layernorm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
        }
        // Final norm
        appendUnquantizedBF16(name: "language_model.model.norm.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)

        // Vision-tower tensors included to prove the text-only repacker drops them.
        appendUnquantizedBF16(name: "vision_tower.blocks.0.norm1.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)
        appendUnquantizedBF16(name: "vision_tower.patch_embed.proj.weight",
                              shape: [arch.hidden, arch.hidden], into: &tensors, rng: &rng)

        // -- Encode safetensors.
        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        // -- Write config.json. Everything is 4-bit affine: no overrides.
        let textConfig: [String: Any] = [
            "hidden_size": arch.hidden,
            "intermediate_size": arch.intermediate,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "head_dim": arch.headDim,
            "vocab_size": arch.vocab,
            "num_hidden_layers": arch.numLayers,
            "layer_types": arch.layerTypes,
            "full_attention_interval": 4,
            "rope_parameters": [
                "rope_theta": 10_000_000.0,
                "rope_type": "default",
                "partial_rotary_factor": 0.25
            ],
            "linear_num_key_heads": arch.linearNumKHeads,
            "linear_num_value_heads": arch.linearNumVHeads,
            "linear_key_head_dim": arch.linearKeyHeadDim,
            "linear_value_head_dim": arch.linearValueHeadDim,
            "linear_conv_kernel_dim": arch.linearConvKernelSize,
            "attn_output_gate": true,
            "tie_word_embeddings": false,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        ]
        let config: [String: Any] = [
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "quantization": ["bits": 4, "group_size": arch.groupSize, "mode": "affine"],
            "text_config": textConfig
        ]
        let configData = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try configData.write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))

        // -- Write model.safetensors.index.json.
        var weightMap: [String: String] = [:]
        for (name, _, _, _) in tensors { weightMap[name] = shardName }
        let indexObj: [String: Any] = [
            "metadata": ["format": "mlx"],
            "weight_map": weightMap
        ]
        let indexData = try JSONSerialization.data(withJSONObject: indexObj, options: [.sortedKeys])
        let indexPath = (dir as NSString).appendingPathComponent("model.safetensors.index.json")
        try indexData.write(to: URL(fileURLWithPath: indexPath))
        return Snapshot(shardPath: shardPath)
    }

    /// Finite BF16 payload. Random *bit patterns* would inject NaN and Inf,
    /// which have no defined group min/max, so values are drawn in [-2, 2] and
    /// rounded to BF16 — exactly what the reference quantizer will later see.
    static func appendBF16(name: String, shape: [Int],
                           into tensors: inout [(String, String, [Int], [UInt8])],
                           rng: inout SplitMix64) {
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 2)
        for index in 0..<elements {
            let unit = Float(rng.next() >> 40) / Float(1 << 24)
            let value = (unit - 0.5) * 4.0
            let bits = value.bitPattern
            let lsb = (bits >> 16) & 1
            let rounded = UInt16(truncatingIfNeeded: (bits &+ (0x7FFF &+ lsb)) >> 16)
            bytes[index * 2] = UInt8(truncatingIfNeeded: rounded)
            bytes[index * 2 + 1] = UInt8(truncatingIfNeeded: rounded >> 8)
        }
        tensors.append((name, "BF16", shape, bytes))
    }

    // MARK: - MiniCPM5 variants

    /// Tiny plain-llama shape: four full-attention layers, 2 query heads over
    /// 2 KV heads of 64, one SwiGLU MLP per layer, untied `lm_head`, no q/k
    /// norm, no experts. `attentionScale` is exactly 0.125 (= pow(64, -0.5)).
    struct MiniCPM5Arch {
        let hidden: Int = 128
        let intermediate: Int = 64
        let numHeads: Int = 2
        let numKVHeads: Int = 2
        let headDim: Int = 64
        let vocab: Int = 256
        let numLayers: Int = 4
        let groupSize: Int = 64
        let ropeTheta: Double = 5_000_000.0
    }

    /// The flat `LlamaForCausalLM` config both MiniCPM5 repos share, minus
    /// the `quantization` block the MLX control adds.
    private static func miniCPM5Config(_ arch: MiniCPM5Arch) -> [String: Any] {
        [
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": arch.hidden,
            "intermediate_size": arch.intermediate,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "head_dim": arch.headDim,
            "vocab_size": arch.vocab,
            "num_hidden_layers": arch.numLayers,
            "rms_norm_eps": 1e-6,
            "rope_theta": arch.ropeTheta,
            "rope_scaling": NSNull(),
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "attention_bias": false,
            "mlp_bias": false,
            "bos_token_id": 0,
            "eos_token_id": [1, 3],
        ]
    }

    /// The vendor's BF16 upload shape: bare `model.` trunk, top-level
    /// `lm_head.weight`, one shard named as the real repo names it, no
    /// `quantization` block. Every tensor is finite BF16 so the in-flight
    /// quantizer sees the same value range the real checkpoint has.
    static func buildMiniCPM5(at dir: String,
                              seed: UInt64 = 0x5C9B_0001_2B00_CD19) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)
        let arch = MiniCPM5Arch()
        var rng = SplitMix64(seed: seed)
        var tensors: [(String, String, [Int], [UInt8])] = []

        appendBF16(name: "model.embed_tokens.weight",
                   shape: [arch.vocab, arch.hidden], into: &tensors, rng: &rng)
        for li in 0..<arch.numLayers {
            let prefix = "model.layers.\(li)"
            appendBF16(name: prefix + ".self_attn.q_proj.weight",
                       shape: [arch.numHeads * arch.headDim, arch.hidden],
                       into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".self_attn.k_proj.weight",
                       shape: [arch.numKVHeads * arch.headDim, arch.hidden],
                       into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".self_attn.v_proj.weight",
                       shape: [arch.numKVHeads * arch.headDim, arch.hidden],
                       into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".self_attn.o_proj.weight",
                       shape: [arch.hidden, arch.numHeads * arch.headDim],
                       into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".mlp.gate_proj.weight",
                       shape: [arch.intermediate, arch.hidden], into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".mlp.up_proj.weight",
                       shape: [arch.intermediate, arch.hidden], into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".mlp.down_proj.weight",
                       shape: [arch.hidden, arch.intermediate], into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".input_layernorm.weight",
                       shape: [arch.hidden], into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".post_attention_layernorm.weight",
                       shape: [arch.hidden], into: &tensors, rng: &rng)
        }
        appendBF16(name: "model.norm.weight", shape: [arch.hidden],
                   into: &tensors, rng: &rng)
        appendBF16(name: "lm_head.weight", shape: [arch.vocab, arch.hidden],
                   into: &tensors, rng: &rng)

        let shardName = "model-00000-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        try JSONSerialization.data(withJSONObject: miniCPM5Config(arch),
                                   options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString)
                .appendingPathComponent("config.json")))
        var weightMap: [String: String] = [:]
        for (name, _, _, _) in tensors { weightMap[name] = shardName }
        try JSONSerialization.data(
            withJSONObject: ["metadata": ["total_size": tensors.reduce(0) { $0 + $1.3.count }],
                             "weight_map": weightMap],
            options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString)
                .appendingPathComponent("model.safetensors.index.json")))
        return Snapshot(shardPath: shardPath)
    }

    /// The vendor's MLX control shape: the same names, every projection
    /// (embedding and head included) packed U32 INT4 group-64 with BF16
    /// `.scales` / `.biases` companions, BF16 norms, and the base
    /// `quantization` block with no per-tensor overrides.
    static func buildMiniCPM5MLX(at dir: String,
                                 seed: UInt64 = 0x5C9B_0002_2B00_35AC) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)
        let arch = MiniCPM5Arch()
        var rng = SplitMix64(seed: seed)
        var tensors: [(String, String, [Int], [UInt8])] = []

        appendQuantizedWeight(name: "model.embed_tokens",
                              outerShape: [arch.vocab], innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        for li in 0..<arch.numLayers {
            let prefix = "model.layers.\(li)"
            appendQuantizedWeight(name: prefix + ".self_attn.q_proj",
                                  outerShape: [arch.numHeads * arch.headDim],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".self_attn.k_proj",
                                  outerShape: [arch.numKVHeads * arch.headDim],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".self_attn.v_proj",
                                  outerShape: [arch.numKVHeads * arch.headDim],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".self_attn.o_proj",
                                  outerShape: [arch.hidden],
                                  innerLogical: arch.numHeads * arch.headDim, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.gate_proj",
                                  outerShape: [arch.intermediate], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.up_proj",
                                  outerShape: [arch.intermediate], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.down_proj",
                                  outerShape: [arch.hidden], innerLogical: arch.intermediate,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".input_layernorm.weight",
                       shape: [arch.hidden], into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".post_attention_layernorm.weight",
                       shape: [arch.hidden], into: &tensors, rng: &rng)
        }
        appendBF16(name: "model.norm.weight", shape: [arch.hidden],
                   into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "lm_head",
                              outerShape: [arch.vocab], innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        let shardName = "model.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        var config = miniCPM5Config(arch)
        config["quantization"] = ["bits": 4, "group_size": arch.groupSize, "mode": "affine"]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString)
                .appendingPathComponent("config.json")))
        var weightMap: [String: String] = [:]
        for (name, _, _, _) in tensors { weightMap[name] = shardName }
        try JSONSerialization.data(
            withJSONObject: ["metadata": ["format": "mlx"], "weight_map": weightMap],
            options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString)
                .appendingPathComponent("model.safetensors.index.json")))
        return Snapshot(shardPath: shardPath)
    }

    // MARK: - Tensor builders

    private static func appendQuantizedWeight(name: String,
                                              outerShape: [Int],
                                              innerLogical: Int,
                                              bits: Int,
                                              groupSize: Int,
                                              into tensors: inout [(String, String, [Int], [UInt8])],
                                              rng: inout SplitMix64) {
        precondition(innerLogical % groupSize == 0)
        let factor = 32 / bits
        precondition(innerLogical % factor == 0)
        let innerSource = innerLogical / factor
        let shape = outerShape + [innerSource]
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 4)
        for i in 0..<bytes.count { bytes[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name + ".weight", "U32", shape, bytes))

        let groups = innerLogical / groupSize
        let companionShape = outerShape + [groups]
        let companionElems = companionShape.reduce(1, *)
        var sb = [UInt8](repeating: 0, count: companionElems * 2)
        for i in 0..<sb.count { sb[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name + ".scales", "BF16", companionShape, sb))
        var bb = [UInt8](repeating: 0, count: companionElems * 2)
        for i in 0..<bb.count { bb[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name + ".biases", "BF16", companionShape, bb))
    }

    private static func appendMapleTernaryWeight(
        name: String,
        outerShape: [Int],
        innerLogical: Int,
        into tensors: inout [(String, String, [Int], [UInt8])],
        rng: inout SplitMix64) {
        precondition(innerLogical % 64 == 0 && innerLogical % 16 == 0)
        let packedShape = outerShape + [innerLogical / 16]
        tensors.append((name + ".weight", "U32", packedShape,
                        maplePayload(dtype: "U32", shape: packedShape, rng: &rng)))
        tensors.append((name + ".row_alpha", "BF16", outerShape,
                        maplePayload(dtype: "BF16", shape: outerShape, rng: &rng)))
    }

    private static func appendMapleFlashHead(
        into tensors: inout [(String, String, [Int], [UInt8])],
        rng: inout SplitMix64) {
        appendQuantizedWeight(name: "lm_head_flash.centroids", outerShape: [4_748],
                              innerLogical: 2_048, bits: 4, groupSize: 64,
                              into: &tensors, rng: &rng)
        let mapShape = [4_748, 32]
        tensors.append(("lm_head_flash.token_map", "I32", mapShape,
                        mapleFlashTokenMapPayload(count: mapShape[0] * mapShape[1])))
        tensors.append(("lm_head_flash.cluster_scale", "BF16", [4_748],
                        maplePayload(dtype: "BF16", shape: [4_748], rng: &rng)))
        // The source carries this redundant cluster-order copy of lm_head.
        // The repacker must retain neither it nor its companions.
        appendQuantizedWeight(name: "lm_head_flash.head", outerShape: [2],
                              innerLogical: 128, bits: 4, groupSize: 64,
                              into: &tensors, rng: &rng)
    }

    private static func mapleFlashTokenMapPayload(count: Int) -> [UInt8] {
        var payload = [UInt8](repeating: 0, count: count * MemoryLayout<UInt32>.stride)
        for token in 0..<count {
            let value = UInt32(token)
            let offset = token * MemoryLayout<UInt32>.stride
            payload[offset] = UInt8(truncatingIfNeeded: value)
            payload[offset + 1] = UInt8(truncatingIfNeeded: value >> 8)
            payload[offset + 2] = UInt8(truncatingIfNeeded: value >> 16)
            payload[offset + 3] = UInt8(truncatingIfNeeded: value >> 24)
        }
        return payload
    }

    private static func maplePayload(dtype: String, shape: [Int],
                                     rng: inout SplitMix64) -> [UInt8] {
        let elementBytes: Int
        switch dtype {
        case "U32", "F32", "I32": elementBytes = 4
        case "BF16", "F16": elementBytes = 2
        case "I64": elementBytes = 8
        default: preconditionFailure("unsupported synthetic dtype \(dtype)")
        }
        var bytes = [UInt8](repeating: 0, count: shape.reduce(1, *) * elementBytes)
        for index in bytes.indices { bytes[index] = UInt8(rng.next() & 0xFF) }
        return bytes
    }

    private static func appendUnquantizedBF16(name: String, shape: [Int],
                                              into tensors: inout [(String, String, [Int], [UInt8])],
                                              rng: inout SplitMix64) {
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 2)
        for i in 0..<bytes.count { bytes[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name, "BF16", shape, bytes))
    }

    // MARK: - Safetensors writer

    private static func writeShard(path: String,
                                   tensors: [(String, String, [Int], [UInt8])]) throws {
        var off: UInt64 = 0
        var headerEntries: [(String, [String: Any])] = []
        for (name, dtype, shape, bytes) in tensors {
            let begin = off
            let end = begin + UInt64(bytes.count)
            headerEntries.append((name, [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, end]
            ]))
            off = end
        }
        var headerDict: [String: Any] = [:]
        for (n, e) in headerEntries { headerDict[n] = e }
        headerDict["__metadata__"] = ["format": "mlx"]
        // Ensure deterministic key ordering for the header — JSONSerialization
        // sortedKeys handles that for us.
        let headerData = try JSONSerialization.data(withJSONObject: headerDict,
                                                    options: [.sortedKeys])
        // Pad header so payload starts on an 8-byte boundary (matches MLX
        // convention and trips fewer downstream surprises).
        var padded = headerData
        while padded.count % 8 != 0 { padded.append(0x20) } // space pad

        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        precondition(fd >= 0, "open failed for \(path)")
        defer { close(fd) }
        var headerLenLE = UInt64(padded.count).littleEndian
        withUnsafeBytes(of: &headerLenLE) { raw in
            _ = write(fd, raw.baseAddress, 8)
        }
        padded.withUnsafeBytes { raw in
            _ = write(fd, raw.baseAddress, padded.count)
        }
        for (_, _, _, bytes) in tensors {
            bytes.withUnsafeBufferPointer { ptr in
                _ = write(fd, ptr.baseAddress, ptr.count)
            }
        }
    }
}

/// Tiny deterministic PRNG. We do not need crypto quality — just stable
/// byte streams across test runs.
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
