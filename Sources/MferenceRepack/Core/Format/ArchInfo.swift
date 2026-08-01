import Foundation

/// Model family discriminator, mirrored into `manifest.json -> arch.family`
/// for non-Gemma families (Gemma manifests omit it — the format's original
/// architecture). Raw values match the runtime's `ModelFamily`.
enum RepackModelFamily: String, Sendable, Equatable {
    case gemma4 = "gemma4"
    case qwen36 = "qwen36"
    case deepseekV4Flash = "deepseekV4Flash"
}

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention,
/// 3 = compressed sparse attention (CSA), 4 = heavily compressed attention
/// (HCA).
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`, 2 if `linear_attention`.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String

    // Family-dependent extensions. Defaults describe Gemma 4 so the Gemma
    // path (and its manifest output) is unchanged.
    let family: RepackModelFamily
    let attnOutputGate: Bool
    let attentionScale: Double
    let embeddingScaledBySqrtHidden: Bool
    let routerScaled: Bool
    let ffnSandwichNorms: Bool
    let sharedExpertGated: Bool
    let ropeNeoxSubdim: Bool
    let linearNumKHeads: Int
    let linearNumVHeads: Int
    let linearKeyHeadDim: Int
    let linearValueHeadDim: Int
    let linearConvKernelSize: Int

    // DeepSeek-V4 compressed-attention / mHC / router extensions. Zeroed
    // defaults keep the Gemma and Qwen constructors (and their manifest
    // output) unchanged. Field names match the manifest JSON keys.
    let caQLoraRank: Int
    let caOLoraRank: Int
    let caOGroups: Int
    let caRopeHeadDim: Int
    let caIndexNHeads: Int
    let caIndexHeadDim: Int
    let caIndexTopK: Int
    let caCSACompressRate: Int
    let caHCACompressRate: Int
    let caCompressRopeTheta: Double
    let hcMult: Int
    let hcSinkhornIters: Int
    let hcEps: Double
    let numHashRoutedLayers: Int
    let routerScoringFunc: String
    let routedScalingFactor: Double
    let swigluLimit: Double

    init(hiddenSize: Int,
         intermediateSize: Int,
         moeIntermediateSize: Int,
         numHeads: Int,
         numKVHeads: Int,
         numFullKVHeads: Int,
         headDim: Int,
         fullHeadDim: Int,
         vocabSize: Int,
         slidingWindow: Int,
         finalLogitSoftcap: Double,
         ropeTheta: Double,
         fullRopeTheta: Double,
         partialRotaryFactor: Double,
         numLayers: Int,
         numExperts: Int,
         topKExperts: Int,
         tieWordEmbeddings: Bool,
         attentionKEqV: Bool,
         fullAttentionLayerMask: [UInt8],
         hiddenActivation: String,
         family: RepackModelFamily,
         attnOutputGate: Bool,
         attentionScale: Double,
         embeddingScaledBySqrtHidden: Bool,
         routerScaled: Bool,
         ffnSandwichNorms: Bool,
         sharedExpertGated: Bool,
         ropeNeoxSubdim: Bool,
         linearNumKHeads: Int,
         linearNumVHeads: Int,
         linearKeyHeadDim: Int,
         linearValueHeadDim: Int,
         linearConvKernelSize: Int,
         caQLoraRank: Int = 0,
         caOLoraRank: Int = 0,
         caOGroups: Int = 0,
         caRopeHeadDim: Int = 0,
         caIndexNHeads: Int = 0,
         caIndexHeadDim: Int = 0,
         caIndexTopK: Int = 0,
         caCSACompressRate: Int = 0,
         caHCACompressRate: Int = 0,
         caCompressRopeTheta: Double = 0,
         hcMult: Int = 0,
         hcSinkhornIters: Int = 0,
         hcEps: Double = 0,
         numHashRoutedLayers: Int = 0,
         routerScoringFunc: String = "softmax",
         routedScalingFactor: Double = 1.0,
         swigluLimit: Double = 0.0) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.family = family
        self.attnOutputGate = attnOutputGate
        self.attentionScale = attentionScale
        self.embeddingScaledBySqrtHidden = embeddingScaledBySqrtHidden
        self.routerScaled = routerScaled
        self.ffnSandwichNorms = ffnSandwichNorms
        self.sharedExpertGated = sharedExpertGated
        self.ropeNeoxSubdim = ropeNeoxSubdim
        self.linearNumKHeads = linearNumKHeads
        self.linearNumVHeads = linearNumVHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelSize = linearConvKernelSize
        self.caQLoraRank = caQLoraRank
        self.caOLoraRank = caOLoraRank
        self.caOGroups = caOGroups
        self.caRopeHeadDim = caRopeHeadDim
        self.caIndexNHeads = caIndexNHeads
        self.caIndexHeadDim = caIndexHeadDim
        self.caIndexTopK = caIndexTopK
        self.caCSACompressRate = caCSACompressRate
        self.caHCACompressRate = caHCACompressRate
        self.caCompressRopeTheta = caCompressRopeTheta
        self.hcMult = hcMult
        self.hcSinkhornIters = hcSinkhornIters
        self.hcEps = hcEps
        self.numHashRoutedLayers = numHashRoutedLayers
        self.routerScoringFunc = routerScoringFunc
        self.routedScalingFactor = routedScalingFactor
        self.swigluLimit = swigluLimit
    }

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        // DeepSeek V4 is text-only; its config is flat (no `text_config`
        // wrapper), so dispatch on model_type before the wrapper guard.
        if (root["model_type"] as? String) == "deepseek_v4" {
            return try loadDeepseekV4Flash(configPath: configPath, tc: root)
        }
        guard let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        if (root["model_type"] as? String) == "qwen3_5_moe" {
            return try loadQwen36(configPath: configPath, tc: tc)
        }
        return try loadGemma4(configPath: configPath, tc: tc)
    }

    // MARK: - Gemma 4

    private static func loadGemma4(configPath: String,
                                   tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = (ropeFull["partial_rotary_factor"] as? Double)
            ?? (ropeFull["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let fullTheta = (ropeFull["rope_theta"] as? Double)
            ?? (ropeFull["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let swaTheta = (ropeSWA["rope_theta"] as? Double)
            ?? (ropeSWA["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000.0
        let kEqV = (tc["attention_k_eq_v"] as? Bool) ?? false
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let act = (tc["hidden_activation"] as? String) ?? "gelu_pytorch_tanh"
        return ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_global_key_value_heads"),
            headDim: try i("head_dim"),
            fullHeadDim: try i("global_head_dim"),
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: try d("final_logit_softcapping"),
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .gemma4,
            attnOutputGate: false,
            attentionScale: 1.0,
            embeddingScaledBySqrtHidden: true,
            routerScaled: true,
            ffnSandwichNorms: true,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0)
    }

    // MARK: - Qwen 3.6 MoE (`model_type == "qwen3_5_moe"`)

    private static func loadQwen36(configPath: String,
                                   tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "linear_attention": mask.append(2)
            case "full_attention":   mask.append(1)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        guard let theta = (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.rope_theta")
        }
        guard let prf = (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.partial_rotary_factor")
        }
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let gate = (tc["attn_output_gate"] as? Bool) ?? false
        let act = (tc["hidden_act"] as? String) ?? "silu"
        let headDim = try i("head_dim")

        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("shared_expert_intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: tie,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .qwen36,
            attnOutputGate: gate,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: try i("linear_num_key_heads"),
            linearNumVHeads: try i("linear_num_value_heads"),
            linearKeyHeadDim: try i("linear_key_head_dim"),
            linearValueHeadDim: try i("linear_value_head_dim"),
            linearConvKernelSize: try i("linear_conv_kernel_dim"))
        try crossCheckProductionQwen36(arch, configPath: configPath)
        return arch
    }

    /// Production Qwen3.6-35B-A3B baseline (mirrors the runtime's
    /// `ArchConfig.qwen36_35B_A3B`; the repack target has no dependency on the
    /// runtime module). A config that matches the production shape
    /// (hidden 2048, 40 layers) must agree on every field; toy/synthetic
    /// configs are exempt.
    private static func crossCheckProductionQwen36(_ a: ArchInfo,
                                                   configPath: String) throws {
        guard a.hiddenSize == 2048, a.numLayers == 40 else { return }
        var expectedMask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { expectedMask[i] = 1 }
        let expected = ArchInfo(
            hiddenSize: 2048,
            intermediateSize: 512,
            moeIntermediateSize: 512,
            numHeads: 16,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 256,
            fullHeadDim: 256,
            vocabSize: 248_320,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 40,
            numExperts: 256,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .qwen36,
            attnOutputGate: true,
            attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 16,
            linearNumVHeads: 32,
            linearKeyHeadDim: 128,
            linearValueHeadDim: 128,
            linearConvKernelSize: 4)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen3_5_moe config does not match the pinned "
                    + "Qwen3.6-35B-A3B architecture baseline")
        }
    }

    // MARK: - DeepSeek-V4-Flash (`model_type == "deepseek_v4"`)

    /// `tc` is the config root: DeepSeek V4 configs are flat.
    private static func loadDeepseekV4Flash(configPath: String,
                                            tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "sliding_attention":            mask.append(0)
            case "compressed_sparse_attention":  mask.append(3)
            case "heavily_compressed_attention": mask.append(4)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        // MoE schedule: the leading `hash_moe` run routes by the frozen
        // tid2eid table. `mlp_layer_types` wins; legacy configs ship
        // `num_hash_layers` (upstream default 3).
        let numHashLayers: Int
        if let mlpTypes = tc["mlp_layer_types"] as? [String] {
            var hash = 0
            var seenLearned = false
            for t in mlpTypes {
                switch t {
                case "hash_moe":
                    guard !seenLearned else {
                        throw RepackError.configJsonInvalid(
                            path: configPath,
                            detail: "mlp_layer_types has hash_moe after moe")
                    }
                    hash += 1
                case "moe":
                    seenLearned = true
                default:
                    throw RepackError.configJsonInvalid(
                        path: configPath, detail: "unknown mlp_layer_types entry \"\(t)\"")
                }
            }
            numHashLayers = hash
        } else {
            numHashLayers = (tc["num_hash_layers"] as? Int)
                ?? (tc["num_hash_layers"] as? NSNumber)?.intValue ?? 3
        }
        // Per-layer-type compression rates; legacy scalar keys fold in.
        let rates = (tc["compress_rates"] as? [String: Any]) ?? [:]
        func rate(_ key: String, legacy: String, fallback: Int) -> Int {
            if let n = (rates[key] as? Int) ?? (rates[key] as? NSNumber)?.intValue { return n }
            if let n = (tc[legacy] as? Int) ?? (tc[legacy] as? NSNumber)?.intValue { return n }
            return fallback
        }
        let csaRate = rate("compressed_sparse_attention",
                           legacy: "compress_rate_csa", fallback: 4)
        let hcaRate = rate("heavily_compressed_attention",
                           legacy: "compress_rate_hca", fallback: 128)
        let headDim = try i("head_dim")
        // Rope head dim = partial_rotary_factor * head_dim; legacy configs
        // ship qk_rope_head_dim instead (upstream default 64/512).
        let prf: Double
        if let p = (tc["partial_rotary_factor"] as? Double)
            ?? (tc["partial_rotary_factor"] as? NSNumber)?.doubleValue {
            prf = p
        } else if let ropeDim = (tc["qk_rope_head_dim"] as? Int)
            ?? (tc["qk_rope_head_dim"] as? NSNumber)?.intValue {
            prf = Double(ropeDim) / Double(headDim)
        } else {
            prf = 64.0 / 512.0
        }
        let ropeHeadDim = Int((Double(headDim) * prf).rounded())
        let theta = try d("rope_theta")
        let moeIntermediate = try i("moe_intermediate_size")
        // V4 ships only moe_intermediate_size; the shared expert reads it
        // through `intermediate_size` when a config carries one explicitly.
        let sharedIntermediate = (tc["intermediate_size"] as? Int)
            ?? (tc["intermediate_size"] as? NSNumber)?.intValue ?? moeIntermediate
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let act = (tc["hidden_act"] as? String) ?? "silu"
        let scoring = (tc["scoring_func"] as? String) ?? "sqrtsoftplus"

        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: sharedIntermediate,
            moeIntermediateSize: moeIntermediate,
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("n_routed_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: tie,
            // Shared-KV MQA: K and V are the same cache entry.
            attentionKEqV: true,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .deepseekV4Flash,
            attnOutputGate: false,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0,
            caQLoraRank: try i("q_lora_rank"),
            caOLoraRank: try i("o_lora_rank"),
            caOGroups: try i("o_groups"),
            caRopeHeadDim: ropeHeadDim,
            caIndexNHeads: try i("index_n_heads"),
            caIndexHeadDim: try i("index_head_dim"),
            caIndexTopK: try i("index_topk"),
            caCSACompressRate: csaRate,
            caHCACompressRate: hcaRate,
            caCompressRopeTheta: try d("compress_rope_theta"),
            hcMult: try i("hc_mult"),
            hcSinkhornIters: try i("hc_sinkhorn_iters"),
            hcEps: try d("hc_eps"),
            numHashRoutedLayers: numHashLayers,
            routerScoringFunc: scoring,
            routedScalingFactor: try d("routed_scaling_factor"),
            swigluLimit: try d("swiglu_limit"))
        try crossCheckProductionDeepseekV4Flash(arch, configPath: configPath)
        return arch
    }

    /// Production DeepSeek-V4-Flash 284B-A13B baseline (mirrors the runtime's
    /// `ArchConfig.deepseekV4Flash_284B_A13B`; the repack target has no
    /// dependency on the runtime module). A config that matches the
    /// production shape (hidden 4096, 43 layers) must agree on every field;
    /// toy/synthetic configs are exempt.
    private static func crossCheckProductionDeepseekV4Flash(_ a: ArchInfo,
                                                            configPath: String) throws {
        guard a.hiddenSize == 4096, a.numLayers == 43 else { return }
        var expectedMask = [UInt8](repeating: 0, count: 43)
        for i in 2..<43 { expectedMask[i] = i.isMultiple(of: 2) ? 3 : 4 }
        let expected = ArchInfo(
            hiddenSize: 4096,
            intermediateSize: 2048,
            moeIntermediateSize: 2048,
            numHeads: 64,
            numKVHeads: 1,
            numFullKVHeads: 1,
            headDim: 512,
            fullHeadDim: 512,
            vocabSize: 129_280,
            slidingWindow: 128,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 10_000.0,
            partialRotaryFactor: 0.125,
            numLayers: 43,
            numExperts: 256,
            topKExperts: 6,
            tieWordEmbeddings: false,
            attentionKEqV: true,
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .deepseekV4Flash,
            attnOutputGate: false,
            attentionScale: 0.044194173824159216,   // 512^-0.5
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0,
            caQLoraRank: 1024,
            caOLoraRank: 1024,
            caOGroups: 8,
            caRopeHeadDim: 64,
            caIndexNHeads: 64,
            caIndexHeadDim: 128,
            caIndexTopK: 512,
            caCSACompressRate: 4,
            caHCACompressRate: 128,
            caCompressRopeTheta: 160_000.0,
            hcMult: 4,
            hcSinkhornIters: 20,
            hcEps: 1.0e-6,
            numHashRoutedLayers: 3,
            routerScoringFunc: "sqrtsoftplus",
            routedScalingFactor: 1.5,
            swigluLimit: 10.0)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "deepseek_v4 config does not match the pinned "
                    + "DeepSeek-V4-Flash-284B-A13B architecture baseline")
        }
    }
}
