import Foundation

/// The native aligned checkpoint's storage contract, independent of execution.
enum GemmaQATSource {
    static let requiredAssets = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "generation_config.json",
    ]

    static let assetHashes: [String: String] = [
        "config.json": "d563ee897015b7826949f5b9c545494ba6098ef210efdbbf0c65e03f4fa5c776",
        "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        "tokenizer_config.json": "8155d42e19a25623e76fc0460ce24ddfd34a7c85b496456684249c68e5f93fc5",
        "chat_template.jinja": "94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413",
        "generation_config.json": "b69207f9be617e982d13cc273cce6fd88c98dda99a4bdc5e2d52ffe0a0d9f0a9",
    ]

    static func applies(arch: ArchInfo, metadata: IndexLoader.SourceMetadata) -> Bool {
        arch.family == .gemma4 && metadata.baseGroupSize == 32
    }

    static func validatePin(repoID: String, commit: String,
                            metadata: IndexLoader.SourceMetadata) throws {
        let source = SupportedModelSource.gemma4QAT
        guard repoID == source.repoID else { return }
        guard commit == source.revision, metadata.indexSha256Hex == source.sourceIndexSHA256 else {
            throw RepackError.configurationInvalid(detail: "Gemma QAT source revision/index does not match the pinned aligned checkpoint")
        }
        try validateAsset(name: "config.json", path: metadata.configPath, pinned: true)
    }

    static func validateAsset(name: String, path: String, pinned: Bool) throws {
        if pinned {
            guard try Sha256Stream.hashFile(path: path) == assetHashes[name] else {
                throw RepackError.configurationInvalid(detail: "Gemma QAT required asset \(name) differs from the pinned source")
            }
        }
        let bytes = try Data(contentsOf: URL(fileURLWithPath: path))
        if name == "chat_template.jinja" {
            guard let template = String(data: bytes, encoding: .utf8),
                  !template.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw RepackError.configurationInvalid(detail: "Gemma QAT chat template is empty or invalid UTF-8")
            }
            return
        }
        guard let json = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
            throw RepackError.configurationInvalid(detail: "Gemma QAT \(name) must be a JSON object")
        }
        if name == "generation_config.json" {
            guard json["do_sample"] as? Bool == true,
                  json["temperature"] as? Double == 1,
                  json["top_k"] as? Int == 64, json["top_p"] as? Double == 0.95,
                  json["bos_token_id"] as? Int == 2,
                  json["eos_token_id"] as? [Int] == [1, 106, 50],
                  json["pad_token_id"] as? Int == 0 else {
                throw RepackError.configurationInvalid(detail: "Gemma QAT generation settings do not match its native profile")
            }
        }
    }

    /// Exact text tensor inventory derived from architecture. This also covers
    /// small test architectures; the production source config is SHA pinned.
    static func validateInventory(arch: ArchInfo, metadata: IndexLoader.SourceMetadata,
                                  registry: [String: SourceTensor]) throws {
        guard applies(arch: arch, metadata: metadata) else { return }
        guard metadata.baseBits == 4, metadata.baseMode == "affine",
              !metadata.sourceIsUnquantized, metadata.bitsOverrides.isEmpty else {
            throw RepackError.configurationInvalid(detail: "Gemma QAT requires native INT4/group-32 affine weights and BF16 routers")
        }
        var expected: [String: (SourceTensor.Dtype, [UInt64])] = [:]
        func bf16(_ name: String, _ shape: [Int]) {
            expected[name] = (.bf16, shape.map(UInt64.init))
        }
        func packed(_ base: String, _ shape: [Int]) throws {
            guard let width = shape.last, width > 0, width.isMultiple(of: 32) else {
                throw RepackError.configurationInvalid(detail: "Gemma QAT invalid group-32 shape for \(base)")
            }
            expected[base + ".weight"] = (.u32, (Array(shape.dropLast()) + [width / 8]).map(UInt64.init))
            for suffix in [".scales", ".biases"] {
                bf16(base + suffix, Array(shape.dropLast()) + [width / 32])
            }
        }
        let root = "language_model.model"
        let d = arch.hiddenSize
        try packed(root + ".embed_tokens", [arch.vocabSize, d])
        bf16(root + ".norm.weight", [d])
        for layer in 0..<arch.numLayers {
            let p = root + ".layers.\(layer)"
            let full = arch.fullAttentionLayerMask[layer] == 1
            let head = full ? arch.fullHeadDim : arch.headDim
            let kv = full ? arch.numFullKVHeads : arch.numKVHeads
            try packed(p + ".self_attn.q_proj", [arch.numHeads * head, d])
            try packed(p + ".self_attn.k_proj", [kv * head, d])
            if !full || !arch.attentionKEqV {
                try packed(p + ".self_attn.v_proj", [kv * head, d])
            }
            try packed(p + ".self_attn.o_proj", [d, arch.numHeads * head])
            bf16(p + ".self_attn.q_norm.weight", [head])
            bf16(p + ".self_attn.k_norm.weight", [head])
            bf16(p + ".router.proj.weight", [arch.numExperts, d])
            bf16(p + ".router.scale", [d])
            bf16(p + ".router.per_expert_scale", [arch.numExperts])
            bf16(p + ".layer_scalar", [1])
            for norm in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "pre_feedforward_layernorm_2",
                         "post_feedforward_layernorm", "post_feedforward_layernorm_1",
                         "post_feedforward_layernorm_2"] {
                bf16(p + "." + norm + ".weight", [d])
            }
            for role in ["gate", "up", "down"] {
                let shared = role == "down" ? [d, arch.intermediateSize] : [arch.intermediateSize, d]
                let routed = role == "down" ? [d, arch.moeIntermediateSize] : [arch.moeIntermediateSize, d]
                try packed(p + ".mlp.\(role)_proj", shared)
                try packed(p + ".experts.switch_glu.\(role)_proj", [arch.numExperts] + routed)
            }
        }
        guard Set(registry.keys) == Set(expected.keys),
              Set(metadata.weightMap.keys) == Set(expected.keys) else {
            throw RepackError.configurationInvalid(detail: "Gemma QAT tensor inventory differs from its text architecture")
        }
        for (name, (dtype, shape)) in expected {
            guard let tensor = registry[name], tensor.dtype == dtype, tensor.shape == shape else {
                throw RepackError.configurationInvalid(detail: "Gemma QAT invalid dtype/shape for \(name)")
            }
        }
    }
}
