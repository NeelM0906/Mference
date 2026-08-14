import Foundation

/// Synthesises a tiny MLX-affine-quantized safetensors snapshot inside a
/// temporary directory. The remote repack tests need only deterministic bytes
/// plus matching `config.json` and `model.safetensors.index.json` metadata.
enum SyntheticSnapshot {

    struct Snapshot {
        let shardPath: String
    }

    enum MapleTensorMutation {
        case remove(String)
        case replaceDtype(String, String)
        case replaceShape(String, [Int])
        case addAffineCompanions(String)
    }

    struct Arch {
        let hidden: Int = 128
        let moeIntermediate: Int = 64
        let intermediate: Int = 192
        let numHeads: Int = 2
        let numKVHeads: Int = 2
        let numGlobalKVHeads: Int = 2
        let headDim: Int = 32
        let globalHeadDim: Int = 64
        let vocab: Int = 512
        let numLayers: Int = 2
        let numExperts: Int = 2
        let topK: Int = 2
        let slidingWindow: Int = 128
        let groupSize: Int = 64
        // layer 0 = sliding, layer 1 = full
        let layerTypes: [String] = ["sliding_attention", "full_attention"]
    }

    /// Build the snapshot. `seed` controls the pseudo-random payload bytes so
    /// tests can pre-compute byte-fidelity expectations.
    static func build(at dir: String, seed: UInt64 = 0xA17B_EEF1_5FAC_E202) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        let arch = Arch()
        var rng = SplitMix64(seed: seed)

        // Build the inventory: (name, dtype, shape, payload)
        var tensors: [(String, String, [Int], [UInt8])] = []
        tensors.reserveCapacity(64)

        // -- LM resident: embedding (4-bit, group=64)
        appendQuantizedWeight(name: "language_model.model.embed_tokens",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        for li in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            // q/k/v/o projections — 4-bit attention
            // SWA layer: heads*head_dim out; full: globalHeadDim heads (here equal kv configs).
            let isFull = arch.layerTypes[li] == "full_attention"
            let qOut = arch.numHeads * (isFull ? arch.globalHeadDim : arch.headDim)
            let kOut = (isFull ? arch.numGlobalKVHeads : arch.numKVHeads) * (isFull ? arch.globalHeadDim : arch.headDim)
            let vOut = kOut
            let oIn  = qOut
            appendQuantizedWeight(name: prefix + ".self_attn.q_proj",
                                  outerShape: [qOut], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".self_attn.k_proj",
                                  outerShape: [kOut], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            if !isFull {
                appendQuantizedWeight(name: prefix + ".self_attn.v_proj",
                                      outerShape: [vOut], innerLogical: arch.hidden,
                                      bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            }
            appendQuantizedWeight(name: prefix + ".self_attn.o_proj",
                                  outerShape: [arch.hidden], innerLogical: oIn,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Per-head norms (BF16, no scale/bias)
            appendUnquantizedBF16(name: prefix + ".self_attn.q_norm.weight",
                                  shape: [isFull ? arch.globalHeadDim : arch.headDim],
                                  into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".self_attn.k_norm.weight",
                                  shape: [isFull ? arch.globalHeadDim : arch.headDim],
                                  into: &tensors, rng: &rng)

            // Router proj — 8-bit affine
            appendQuantizedWeight(name: prefix + ".router.proj",
                                  outerShape: [arch.numExperts], innerLogical: arch.hidden,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".router.scale",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".router.per_expert_scale",
                                  shape: [arch.numExperts], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".layer_scalar",
                                  shape: [1], into: &tensors, rng: &rng)

            // Shared-expert mlp — 8-bit affine
            appendQuantizedWeight(name: prefix + ".mlp.gate_proj",
                                  outerShape: [arch.intermediate], innerLogical: arch.hidden,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.up_proj",
                                  outerShape: [arch.intermediate], innerLogical: arch.hidden,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.down_proj",
                                  outerShape: [arch.hidden], innerLogical: arch.intermediate,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Routed experts — 4-bit affine, leading dim = numExperts
            appendQuantizedWeight(name: prefix + ".experts.switch_glu.gate_proj",
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".experts.switch_glu.up_proj",
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".experts.switch_glu.down_proj",
                                  outerShape: [arch.numExperts, arch.hidden],
                                  innerLogical: arch.moeIntermediate, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Per-layer norms
            for norm in ["input_layernorm","post_attention_layernorm",
                         "pre_feedforward_layernorm","pre_feedforward_layernorm_2",
                         "post_feedforward_layernorm","post_feedforward_layernorm_1",
                         "post_feedforward_layernorm_2"] {
                appendUnquantizedBF16(name: prefix + "." + norm + ".weight",
                                      shape: [arch.hidden], into: &tensors, rng: &rng)
            }
        }
        // Final norm
        appendUnquantizedBF16(name: "language_model.model.norm.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)

        // Multimodal tensors included to prove the text-only repacker drops them.
        appendUnquantizedBF16(name: "vision_tower.encoder.layers.0.input_layernorm.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "vision_tower.encoder.layers.0.self_attn.q_proj.linear",
                              outerShape: [arch.hidden], innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "embed_vision.embedding_projection",
                              outerShape: [arch.hidden], innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        // -- Encode safetensors.
        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        // -- Write config.json with bit-width overrides for mlp + router.
        var overrides: [String: [String: Any]] = [:]
        for li in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            for k in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"] {
                overrides[prefix + "." + k] = ["bits": 8, "group_size": arch.groupSize]
            }
        }
        var quant: [String: Any] = [
            "bits": 4, "group_size": arch.groupSize, "mode": "affine"
        ]
        for (k, v) in overrides { quant[k] = v }

        let textConfig: [String: Any] = [
            "hidden_size": arch.hidden,
            "intermediate_size": arch.intermediate,
            "moe_intermediate_size": arch.moeIntermediate,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "num_global_key_value_heads": arch.numGlobalKVHeads,
            "head_dim": arch.headDim,
            "global_head_dim": arch.globalHeadDim,
            "vocab_size": arch.vocab,
            "num_hidden_layers": arch.numLayers,
            "num_experts": arch.numExperts,
            "top_k_experts": arch.topK,
            "sliding_window": arch.slidingWindow,
            "final_logit_softcapping": 30.0,
            "rope_parameters": [
                "sliding_attention": ["rope_theta": 10000.0, "rope_type": "default"],
                "full_attention":   ["rope_theta": 1000000.0, "rope_type": "proportional",
                                      "partial_rotary_factor": 0.25]
            ],
            "layer_types": arch.layerTypes,
            "tie_word_embeddings": true,
            "attention_k_eq_v": true,
            "hidden_activation": "gelu_pytorch_tanh"
        ]
        let config: [String: Any] = [
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
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

            // Plain pre-norm layer norms (no Gemma sandwich norms).
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

    // MARK: - DeepSeek-V4-Flash variant

    /// Tiny deepseek_v4-shaped architecture: two sliding-window bootstrap
    /// layers, one CSA layer (with lightning indexer), one HCA layer; eight
    /// routed 2-bit experts (top-2, layer 0 hash-routed) plus a shared
    /// expert; low-rank shared-KV MQA attention with sinks; two mHC mix
    /// sites per layer plus the top-level HyperHead; untied lm_head. Plain
    /// `model.` prefix (no `language_model.` — mlx-lm conversion naming).
    struct DeepseekArch {
        let hidden: Int = 128
        let moeIntermediate: Int = 64
        let sharedIntermediate: Int = 64
        let numHeads: Int = 2
        let numKVHeads: Int = 1
        let headDim: Int = 64
        let vocab: Int = 256
        let numLayers: Int = 4
        let numExperts: Int = 8
        let topK: Int = 2
        let groupSize: Int = 64
        let qLoraRank: Int = 64
        let oLoraRank: Int = 64
        let oGroups: Int = 2
        let indexNHeads: Int = 2
        let indexHeadDim: Int = 64
        let indexTopK: Int = 16
        let slidingWindow: Int = 32
        let csaCompressRate: Int = 4
        let hcaCompressRate: Int = 128
        let hcMult: Int = 2
        let hcSinkhornIters: Int = 4
        let numHashLayers: Int = 1
        // layers 0-1 = sliding-window, layer 2 = CSA, layer 3 = HCA
        let layerTypes: [String] = ["sliding_attention", "sliding_attention",
                                    "compressed_sparse_attention",
                                    "heavily_compressed_attention"]
        /// mHC mix rows: (2 + mult) * mult.
        var hcFnRows: Int { (2 + hcMult) * hcMult }
    }

    static func buildDeepseekV4(at dir: String,
                                seed: UInt64 = 0xD5EE_C4F1_A5B0_0757) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        let arch = DeepseekArch()
        var rng = SplitMix64(seed: seed)

        var tensors: [(String, String, [Int], [UInt8])] = []
        tensors.reserveCapacity(160)

        // -- Embedding + untied lm_head (4-bit, group=64)
        appendQuantizedWeight(name: "model.embed_tokens",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "lm_head",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        for li in 0..<arch.numLayers {
            let prefix = "model.layers.\(li)"
            let layerType = arch.layerTypes[li]

            // Low-rank shared-KV MQA attention path, every layer.
            appendQuantizedWeight(name: prefix + ".attn.wq_a",
                                  outerShape: [arch.qLoraRank],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".attn.q_norm.weight",
                                  shape: [arch.qLoraRank], into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".attn.wq_b",
                                  outerShape: [arch.numHeads * arch.headDim],
                                  innerLogical: arch.qLoraRank, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".attn.wkv",
                                  outerShape: [arch.headDim],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".attn.kv_norm.weight",
                                  shape: [arch.headDim], into: &tensors, rng: &rng)
            appendUnquantizedFP32(name: prefix + ".attn.attn_sink",
                                  shape: [arch.numHeads], into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".attn.wo_a",
                                  outerShape: [arch.oGroups * arch.oLoraRank],
                                  innerLogical: arch.numHeads * arch.headDim / arch.oGroups,
                                  bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".attn.wo_b",
                                  outerShape: [arch.hidden],
                                  innerLogical: arch.oGroups * arch.oLoraRank, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // CSA/HCA compressor branch. CSA pools two overlapping series
            // (2 * headDim rows); HCA one non-overlapping series.
            if layerType != "sliding_attention" {
                let isCSA = layerType == "compressed_sparse_attention"
                let rows = isCSA ? 2 * arch.headDim : arch.headDim
                let rate = isCSA ? arch.csaCompressRate : arch.hcaCompressRate
                appendQuantizedWeight(name: prefix + ".attn.compressor.wkv",
                                      outerShape: [rows],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".attn.compressor.wgate",
                                      outerShape: [rows],
                                      innerLogical: arch.hidden, bits: 4,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".attn.compressor.norm.weight",
                                      shape: [arch.headDim], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".attn.compressor.ape",
                                      shape: [min(rate, 8)], into: &tensors, rng: &rng)
                if isCSA {
                    let ip = prefix + ".attn.indexer"
                    appendQuantizedWeight(name: ip + ".compressor.wkv",
                                          outerShape: [2 * arch.indexHeadDim],
                                          innerLogical: arch.hidden, bits: 4,
                                          groupSize: arch.groupSize, into: &tensors, rng: &rng)
                    appendQuantizedWeight(name: ip + ".compressor.wgate",
                                          outerShape: [2 * arch.indexHeadDim],
                                          innerLogical: arch.hidden, bits: 4,
                                          groupSize: arch.groupSize, into: &tensors, rng: &rng)
                    appendUnquantizedBF16(name: ip + ".compressor.norm.weight",
                                          shape: [arch.indexHeadDim], into: &tensors, rng: &rng)
                    appendUnquantizedBF16(name: ip + ".compressor.ape",
                                          shape: [arch.csaCompressRate], into: &tensors, rng: &rng)
                    appendQuantizedWeight(name: ip + ".wq_b",
                                          outerShape: [arch.indexNHeads * arch.indexHeadDim],
                                          innerLogical: arch.qLoraRank, bits: 4,
                                          groupSize: arch.groupSize, into: &tensors, rng: &rng)
                    appendQuantizedWeight(name: ip + ".weights_proj",
                                          outerShape: [arch.indexNHeads],
                                          innerLogical: arch.hidden, bits: 4,
                                          groupSize: arch.groupSize, into: &tensors, rng: &rng)
                }
            }

            // Router — unquantized BF16 (the real conversion keeps the gate
            // in BF16); hash layers carry the I64 tid2eid table, learned
            // layers the selection correction bias.
            appendUnquantizedBF16(name: prefix + ".ffn.gate.weight",
                                  shape: [arch.numExperts, arch.hidden],
                                  into: &tensors, rng: &rng)
            if li < arch.numHashLayers {
                appendUnquantizedI64(name: prefix + ".ffn.gate.tid2eid",
                                     shape: [arch.vocab, arch.topK],
                                     into: &tensors, rng: &rng)
            } else {
                appendUnquantizedFP32(name: prefix + ".ffn.gate.e_score_correction_bias",
                                      shape: [arch.numExperts], into: &tensors, rng: &rng)
            }

            // Shared expert — 4-bit.
            appendQuantizedWeight(name: prefix + ".ffn.shared_experts.gate_proj",
                                  outerShape: [arch.sharedIntermediate], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".ffn.shared_experts.up_proj",
                                  outerShape: [arch.sharedIntermediate], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".ffn.shared_experts.down_proj",
                                  outerShape: [arch.hidden], innerLogical: arch.sharedIntermediate,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Routed experts — stacked expert-major, 2-bit (U32 packing
            // factor 16). gate_proj mirrors the real checkpoint's mixed
            // grouping: group 32 on every layer except the last (group 64).
            let gateGroup = li == arch.numLayers - 1 ? arch.groupSize
                                                     : arch.groupSize / 2
            appendQuantizedWeight(name: prefix + ".ffn.switch_mlp.gate_proj",
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: 2,
                                  groupSize: gateGroup, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".ffn.switch_mlp.up_proj",
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: 2,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".ffn.switch_mlp.down_proj",
                                  outerShape: [arch.numExperts, arch.hidden],
                                  innerLogical: arch.moeIntermediate, bits: 2,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Plain pre-norm layer norms plus the two mHC mix sites.
            appendUnquantizedBF16(name: prefix + ".attn_norm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".ffn_norm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
            for site in ["attn_hc", "ffn_hc"] {
                appendUnquantizedFP32(name: prefix + ".\(site).fn",
                                      shape: [arch.hcFnRows, arch.hcMult * arch.hidden],
                                      into: &tensors, rng: &rng)
                appendUnquantizedFP32(name: prefix + ".\(site).base",
                                      shape: [arch.hcFnRows], into: &tensors, rng: &rng)
                appendUnquantizedFP32(name: prefix + ".\(site).scale",
                                      shape: [3], into: &tensors, rng: &rng)
            }
        }
        // HyperHead stream collapse + final norm.
        appendUnquantizedFP32(name: "model.hc_head.fn",
                              shape: [arch.hcFnRows, arch.hcMult * arch.hidden],
                              into: &tensors, rng: &rng)
        appendUnquantizedFP32(name: "model.hc_head.base",
                              shape: [arch.hcFnRows], into: &tensors, rng: &rng)
        appendUnquantizedFP32(name: "model.hc_head.scale",
                              shape: [3], into: &tensors, rng: &rng)
        appendUnquantizedBF16(name: "model.norm.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)

        // -- Encode safetensors.
        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        // -- Write config.json: 4-bit base, 2-bit routed experts with the
        // real checkpoint's mixed gate grouping (32 everywhere except the
        // last layer's 64). The router gate is unquantized (no override).
        var quant: [String: Any] = [
            "bits": 4, "group_size": arch.groupSize, "mode": "affine"
        ]
        for li in 0..<arch.numLayers {
            let prefix = "model.layers.\(li)"
            let gateGroup = li == arch.numLayers - 1 ? arch.groupSize
                                                     : arch.groupSize / 2
            quant[prefix + ".ffn.switch_mlp.gate_proj"] =
                ["bits": 2, "group_size": gateGroup]
            for k in ["ffn.switch_mlp.up_proj", "ffn.switch_mlp.down_proj"] {
                quant[prefix + "." + k] = ["bits": 2, "group_size": arch.groupSize]
            }
        }

        var mlpLayerTypes = [String](repeating: "moe", count: arch.numLayers)
        for li in 0..<arch.numHashLayers { mlpLayerTypes[li] = "hash_moe" }
        // deepseek_v4 configs are flat: no `text_config` wrapper.
        let config: [String: Any] = [
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "quantization": quant,
            "hidden_size": arch.hidden,
            "moe_intermediate_size": arch.moeIntermediate,
            "intermediate_size": arch.sharedIntermediate,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "head_dim": arch.headDim,
            "vocab_size": arch.vocab,
            "num_hidden_layers": arch.numLayers,
            "n_routed_experts": arch.numExperts,
            "n_shared_experts": 1,
            "num_experts_per_tok": arch.topK,
            "q_lora_rank": arch.qLoraRank,
            "o_lora_rank": arch.oLoraRank,
            "o_groups": arch.oGroups,
            "index_n_heads": arch.indexNHeads,
            "index_head_dim": arch.indexHeadDim,
            "index_topk": arch.indexTopK,
            "sliding_window": arch.slidingWindow,
            "layer_types": arch.layerTypes,
            "mlp_layer_types": mlpLayerTypes,
            "compress_rates": [
                "compressed_sparse_attention": arch.csaCompressRate,
                "heavily_compressed_attention": arch.hcaCompressRate,
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
            "hc_mult": arch.hcMult,
            "hc_sinkhorn_iters": arch.hcSinkhornIters,
            "hc_eps": 1e-6,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "tie_word_embeddings": false,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
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

    private static func appendUnquantizedFP32(name: String, shape: [Int],
                                              into tensors: inout [(String, String, [Int], [UInt8])],
                                              rng: inout SplitMix64) {
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 4)
        for i in 0..<bytes.count { bytes[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name, "F32", shape, bytes))
    }

    /// Raw (unpacked) integer table riding as U32 — no `.weight` suffix, so
    /// the planner takes the no-companion path.
    private static func appendUnquantizedU32(name: String, shape: [Int],
                                             into tensors: inout [(String, String, [Int], [UInt8])],
                                             rng: inout SplitMix64) {
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 4)
        for i in 0..<bytes.count { bytes[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name, "U32", shape, bytes))
    }

    /// I64 lookup table (the real `tid2eid` dtype). Values are kept small
    /// and positive so a U32/I64-agnostic reader sees valid expert ids.
    private static func appendUnquantizedI64(name: String, shape: [Int],
                                             into tensors: inout [(String, String, [Int], [UInt8])],
                                             rng: inout SplitMix64) {
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 8)
        for i in stride(from: 0, to: bytes.count, by: 8) {
            bytes[i] = UInt8(rng.next() & 0x7)
        }
        tensors.append((name, "I64", shape, bytes))
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
