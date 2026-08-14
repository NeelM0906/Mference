import CoreFoundation
import Foundation

/// Model family discriminator, mirrored into `manifest.json -> arch.family`
/// for non-Gemma families (Gemma manifests omit it — the format's original
/// architecture). Raw values match the runtime's `ModelFamily`.
enum RepackModelFamily: String, Sendable, Equatable {
    case gemma4 = "gemma4"
    case qwen36 = "qwen36"
    case qwen38 = "qwen38"
    case deepseekV4Flash = "deepseekV4Flash"
    case inklingSmall = "inklingSmall"
    case maple = "maple"
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
    let caRopeScalingFactor: Double
    let caRopeScalingOriginalMax: Int
    let caRopeScalingBetaFast: Double
    let caRopeScalingBetaSlow: Double
    let hcMult: Int
    let hcSinkhornIters: Int
    let hcEps: Double
    let numHashRoutedLayers: Int
    let routerScoringFunc: String
    let routedScalingFactor: Double
    let swigluLimit: Double

    // Inkling extensions: learned relative-attention position encoding (no
    // RoPE), depthwise short convolutions, multiple shared experts that sink
    // their own router logits, and leading dense-FFN layers. Zeroed defaults
    // (one shared expert, unit logit scale) keep the other three families'
    // manifest output unchanged.
    let relDRel: Int
    let relExtent: Int
    let relProjDim: Int
    let relLogScalingFloor: Int
    let relLogScalingAlpha: Double
    let sconvKernelSize: Int
    let numSharedExperts: Int
    let numDenseLayers: Int
    let denseIntermediateSize: Int
    let sharedExpertSink: Bool
    let embedNormEnabled: Bool
    let logitsWidthMultiplier: Double
    let routerGateBias: Bool
    let routerNormAfterTopK: Bool
    let routerGlobalScale: Bool
    let unpaddedVocabSize: Int

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
         caRopeScalingFactor: Double = 0,
         caRopeScalingOriginalMax: Int = 0,
         caRopeScalingBetaFast: Double = 0,
         caRopeScalingBetaSlow: Double = 0,
         hcMult: Int = 0,
         hcSinkhornIters: Int = 0,
         hcEps: Double = 0,
         numHashRoutedLayers: Int = 0,
         routerScoringFunc: String = "softmax",
         routedScalingFactor: Double = 1.0,
         swigluLimit: Double = 0.0,
         relDRel: Int = 0,
         relExtent: Int = 0,
         relProjDim: Int = 0,
         relLogScalingFloor: Int = 0,
         relLogScalingAlpha: Double = 0,
         sconvKernelSize: Int = 0,
         numSharedExperts: Int = 1,
         numDenseLayers: Int = 0,
         denseIntermediateSize: Int = 0,
         sharedExpertSink: Bool = false,
         embedNormEnabled: Bool = false,
         logitsWidthMultiplier: Double = 1.0,
         routerGateBias: Bool = false,
         routerNormAfterTopK: Bool = false,
         routerGlobalScale: Bool = false,
         unpaddedVocabSize: Int = 0) {
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
        self.caRopeScalingFactor = caRopeScalingFactor
        self.caRopeScalingOriginalMax = caRopeScalingOriginalMax
        self.caRopeScalingBetaFast = caRopeScalingBetaFast
        self.caRopeScalingBetaSlow = caRopeScalingBetaSlow
        self.hcMult = hcMult
        self.hcSinkhornIters = hcSinkhornIters
        self.hcEps = hcEps
        self.numHashRoutedLayers = numHashRoutedLayers
        self.routerScoringFunc = routerScoringFunc
        self.routedScalingFactor = routedScalingFactor
        self.swigluLimit = swigluLimit
        self.relDRel = relDRel
        self.relExtent = relExtent
        self.relProjDim = relProjDim
        self.relLogScalingFloor = relLogScalingFloor
        self.relLogScalingAlpha = relLogScalingAlpha
        self.sconvKernelSize = sconvKernelSize
        self.numSharedExperts = numSharedExperts
        self.numDenseLayers = numDenseLayers
        self.denseIntermediateSize = denseIntermediateSize
        self.sharedExpertSink = sharedExpertSink
        self.embedNormEnabled = embedNormEnabled
        self.logitsWidthMultiplier = logitsWidthMultiplier
        self.routerGateBias = routerGateBias
        self.routerNormAfterTopK = routerNormAfterTopK
        self.routerGlobalScale = routerGlobalScale
        self.unpaddedVocabSize = unpaddedVocabSize
    }

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        // DeepSeek V4 and Maple are text-only; their configs are flat (no
        // `text_config` wrapper), so dispatch before the wrapper guard.
        if (root["model_type"] as? String) == "deepseek_v4" {
            return try loadDeepseekV4Flash(configPath: configPath, tc: root)
        }
        if (root["model_type"] as? String) == "maple" {
            return try loadMaple(configPath: configPath, tc: root)
        }
        guard let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        if (root["model_type"] as? String) == "qwen3_5_moe" {
            return try loadQwen36(configPath: configPath, tc: tc)
        }
        if (root["model_type"] as? String) == "qwen3_5" {
            return try loadQwen38(configPath: configPath, tc: tc)
        }
        if (root["model_type"] as? String) == "inkling_mm_model" {
            return try loadInklingSmall(configPath: configPath, tc: tc)
        }
        return try loadGemma4(configPath: configPath, tc: tc)
    }

    // MARK: - Maple

    private static func loadMaple(configPath: String,
                                  tc: [String: Any]) throws -> ArchInfo {
        func integer(_ key: String) throws -> Int {
            guard let number = tc[key] as? NSNumber,
                  CFGetTypeID(number) != CFBooleanGetTypeID(),
                  NSNumber(value: number.int64Value).compare(number) == .orderedSame,
                  let value = Int(exactly: number.int64Value) else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "\(key) must be an exact integer")
            }
            return value
        }
        func decimal(_ key: String) throws -> Double {
            guard let value = (tc[key] as? Double) ?? (tc[key] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(key)")
            }
            return value
        }
        func require(_ key: String, equals expected: Bool) throws {
            guard tc[key] as? Bool == expected else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "Maple requires \(key)=\(expected)")
            }
        }

        let expectedMask: [UInt8] = (0..<24).map { $0 % 4 == 3 ? 1 : 0 }
        let expectedLayerTypes = (0..<24).map { $0 % 4 == 3 ? "full_attention" : "sliding_attention" }
        guard let layerTypes = tc["layer_types"] as? [String], layerTypes == expectedLayerTypes else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "Maple requires the pinned 3-sliding/1-full layer schedule")
        }
        try require("use_qk_norm", equals: true)
        try require("norm_topk_prob", equals: true)
        try require("tie_word_embeddings", equals: false)
        try require("use_rmsnorm", equals: true)
        try require("use_bias", equals: false)
        guard tc["router_dtype"] as? String == "fp32" else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "Maple requires router_dtype=fp32")
        }
        guard tc["hidden_act"] as? String == "silu" else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "Maple requires hidden_act=silu")
        }
        guard try integer("max_position_embeddings") == 128_000 else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "Maple requires max_position_embeddings=128000")
        }

        let arch = ArchInfo(
            hiddenSize: try integer("hidden_size"),
            intermediateSize: try integer("moe_shared_expert_intermediate_size"),
            moeIntermediateSize: try integer("moe_intermediate_size"),
            numHeads: try integer("num_attention_heads"),
            numKVHeads: try integer("num_key_value_heads"),
            numFullKVHeads: try integer("num_key_value_heads"),
            headDim: try integer("head_dim"),
            fullHeadDim: try integer("head_dim"),
            vocabSize: try integer("vocab_size"),
            slidingWindow: try integer("sliding_window"),
            finalLogitSoftcap: 0.0,
            ropeTheta: try decimal("rope_theta"),
            // Maple's full layers are NoPE. Do not derive this from the
            // checkpoint's unrelated `nope_on_global_attention` flag.
            fullRopeTheta: 0.0,
            partialRotaryFactor: try decimal("partial_rotary_factor"),
            numLayers: try integer("num_hidden_layers"),
            numExperts: try integer("num_experts"),
            topKExperts: try integer("num_experts_per_tok"),
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .maple,
            attnOutputGate: false,
            attentionScale: 1.0 / Double(128).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0,
            routerScoringFunc: "softmax",
            routedScalingFactor: 1.0,
            swigluLimit: 7.0,
            numSharedExperts: try integer("num_shared_experts"),
            numDenseLayers: try integer("first_k_dense_replace"),
            routerNormAfterTopK: true)
        try crossCheckMaple(arch, configPath: configPath, rmsNormEpsilon: try decimal("rms_norm_eps"))
        return arch
    }

    private static func crossCheckMaple(_ actual: ArchInfo,
                                        configPath: String,
                                        rmsNormEpsilon: Double) throws {
        let expected = ArchInfo(
            hiddenSize: 2_048,
            intermediateSize: 512,
            moeIntermediateSize: 512,
            numHeads: 16,
            numKVHeads: 4,
            numFullKVHeads: 4,
            headDim: 128,
            fullHeadDim: 128,
            vocabSize: 151_936,
            slidingWindow: 512,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 0.0,
            partialRotaryFactor: 0.5,
            numLayers: 24,
            numExperts: 256,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: (0..<24).map { $0 % 4 == 3 ? 1 : 0 },
            hiddenActivation: "silu",
            family: .maple,
            attnOutputGate: false,
            attentionScale: 1.0 / Double(128).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0,
            routerScoringFunc: "softmax",
            routedScalingFactor: 1.0,
            swigluLimit: 7.0,
            numSharedExperts: 0,
            numDenseLayers: 0,
            routerNormAfterTopK: true)
        guard actual == expected, rmsNormEpsilon == 0.000_001 else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "maple config does not match the pinned Maple Preview architecture baseline")
        }
    }

    // MARK: - Inkling Small

    /// Inkling carries no RoPE at all — position rides on a learned
    /// relative-attention bias — so `ropeTheta` and `partialRotaryFactor` are
    /// deliberately zero rather than parsed. Attention layers are named by
    /// exclusion: `local_layer_ids` lists the sliding-window layers and every
    /// other layer is full attention.
    ///
    /// `relExtent` is the *global*-layer bias width; sliding layers use
    /// `sliding_window_size` instead (the reference picks
    /// `sliding_window_size if is_sliding else rel_extent`), which is why the
    /// checkpoint ships `rel_logits_proj.proj` as `[16, 512]` on local layers
    /// and `[16, 1024]` on layers 5, 11, 17, 23, 29, 35 and 41. `relProjDim`
    /// is `num_attention_heads * d_rel`, the output width of `attn.wr_du`.
    private static func loadInklingSmall(configPath: String,
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
        let numLayers = try i("num_hidden_layers")
        guard let localIDs = tc["local_layer_ids"] as? [Int] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing local_layer_ids")
        }
        let localSet = Set(localIDs)
        if let bad = localSet.first(where: { $0 < 0 || $0 >= numLayers }) {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "local_layer_ids entry \(bad) out of range for \(numLayers) layers")
        }
        // 0 = sliding-window, 1 = full attention.
        let mask: [UInt8] = (0..<numLayers).map { localSet.contains($0) ? 0 : 1 }

        guard (tc["use_sconv"] as? Bool) == true else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "use_sconv must be true for Inkling")
        }
        guard let gateActivation = tc["gate_activation"] as? String else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing gate_activation")
        }
        guard gateActivation == "sigmoid" else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "unsupported gate_activation \"\(gateActivation)\"")
        }
        let headDim = try i("head_dim")
        let swaHeadDim = try i("swa_head_dim")
        guard swaHeadDim == headDim else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "swa_head_dim \(swaHeadDim) != head_dim \(headDim); "
                    + "the runtime carries a single head dimension")
        }

        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window_size"),
            finalLogitSoftcap: 0.0,
            ropeTheta: 0.0,
            fullRopeTheta: 0.0,
            partialRotaryFactor: 0.0,
            numLayers: numLayers,
            numExperts: try i("n_routed_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: "silu",
            family: .inklingSmall,
            attnOutputGate: false,
            // Inkling RMS-normalizes q and k per head, so the reference uses
            // `1 / head_dim`, NOT the usual `1 / sqrt(head_dim)`
            // (`inkling_mlx/attention.py`: "q/k are per-head RMS-normalized,
            // hence 1/d rather than 1/sqrt(d)").
            attentionScale: 1.0 / Double(headDim),
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
            routerScoringFunc: gateActivation,
            routedScalingFactor: try d("route_scale"),
            relDRel: try i("d_rel"),
            relExtent: try i("rel_extent"),
            relProjDim: try i("num_attention_heads") * (try i("d_rel")),
            relLogScalingFloor: try i("log_scaling_n_floor"),
            relLogScalingAlpha: try d("log_scaling_alpha"),
            sconvKernelSize: try i("sconv_kernel_size"),
            numSharedExperts: try i("n_shared_experts"),
            numDenseLayers: try i("dense_mlp_idx"),
            denseIntermediateSize: try i("dense_intermediate_size"),
            sharedExpertSink: (tc["shared_expert_sink"] as? Bool) ?? false,
            embedNormEnabled: (tc["use_embed_norm"] as? Bool) ?? false,
            logitsWidthMultiplier: try d("logits_mup_width_multiplier"),
            routerGateBias: (tc["use_gate_bias"] as? Bool) ?? false,
            routerNormAfterTopK: (tc["norm_after_topk"] as? Bool) ?? false,
            routerGlobalScale: (tc["use_global_scale"] as? Bool) ?? false,
            unpaddedVocabSize: (tc["unpadded_vocab_size"] as? Int)
                ?? (tc["unpadded_vocab_size"] as? NSNumber)?.intValue ?? 0)
        try crossCheckProductionInklingSmall(arch, configPath: configPath)
        return arch
    }

    /// Production Inkling-Small 276B-A12B baseline (mirrors the runtime's
    /// `ArchConfig.inklingSmall_276B_A12B`). A config matching the production
    /// shape (hidden 4096, 42 layers, 256 experts) must agree on every field;
    /// toy/synthetic configs are exempt.
    private static func crossCheckProductionInklingSmall(
        _ a: ArchInfo, configPath: String) throws {
        guard a.hiddenSize == 4096, a.numLayers == 42, a.numExperts == 256 else {
            return
        }
        let expectedMask: [UInt8] = (0..<42).map { $0 % 6 == 5 ? 1 : 0 }
        func fail(_ field: String, _ actual: String, _ expected: String) -> RepackError {
            .configJsonInvalid(
                path: configPath,
                detail: "production Inkling-Small \(field) is \(actual), expected \(expected)")
        }
        guard a.fullAttentionLayerMask == expectedMask else {
            throw fail("local_layer_ids",
                       a.fullAttentionLayerMask.description,
                       expectedMask.description)
        }
        guard a.vocabSize == 201_024 else {
            throw fail("vocabSize", "\(a.vocabSize)", "201024")
        }
        guard a.numHeads == 32, a.numKVHeads == 8, a.headDim == 128 else {
            throw fail("attention shape",
                       "\(a.numHeads)/\(a.numKVHeads)/\(a.headDim)", "32/8/128")
        }
        guard a.topKExperts == 6, a.numSharedExperts == 2 else {
            throw fail("routing", "\(a.topKExperts)/\(a.numSharedExperts)", "6/2")
        }
        guard a.moeIntermediateSize == 2048, a.denseIntermediateSize == 16_384 else {
            throw fail("FFN widths",
                       "\(a.moeIntermediateSize)/\(a.denseIntermediateSize)",
                       "2048/16384")
        }
        guard a.slidingWindow == 512 else {
            throw fail("sliding_window_size", "\(a.slidingWindow)", "512")
        }
        guard a.numDenseLayers == 2 else {
            throw fail("dense_mlp_idx", "\(a.numDenseLayers)", "2")
        }
        guard a.sconvKernelSize == 4 else {
            throw fail("sconv_kernel_size", "\(a.sconvKernelSize)", "4")
        }
        guard a.relDRel == 16, a.relExtent == 1024 else {
            throw fail("relative position", "\(a.relDRel)/\(a.relExtent)", "16/1024")
        }
        guard a.routedScalingFactor == 8.0 else {
            throw fail("route_scale", "\(a.routedScalingFactor)", "8.0")
        }
        guard a.logitsWidthMultiplier == 16.0 else {
            throw fail("logits_mup_width_multiplier",
                       "\(a.logitsWidthMultiplier)", "16.0")
        }
        guard a.ropeTheta == 0, a.partialRotaryFactor == 0 else {
            throw fail("rope", "non-zero", "zero (relative position only)")
        }
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

    // MARK: - Qwen 3.8 dense (`model_type == "qwen3_5"`)

    /// Text stack of the multimodal Qwen3.8 checkpoint (the vision tower is
    /// excluded by the planner). Same hybrid linear/full attention schedule as
    /// Qwen 3.6, but dense: one SwiGLU MLP per layer, no router, no shared
    /// expert, no routed experts — so the MoE slots are zeroed and every layer
    /// counts as dense (`numDenseLayers == numLayers`).
    private static func loadQwen38(configPath: String,
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
        let intermediate = try i("intermediate_size")
        let numLayers = try i("num_hidden_layers")

        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: intermediate,
            moeIntermediateSize: 0,
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
            numLayers: numLayers,
            numExperts: 0,
            topKExperts: 0,
            tieWordEmbeddings: tie,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .qwen38,
            attnOutputGate: gate,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true,
            linearNumKHeads: try i("linear_num_key_heads"),
            linearNumVHeads: try i("linear_num_value_heads"),
            linearKeyHeadDim: try i("linear_key_head_dim"),
            linearValueHeadDim: try i("linear_value_head_dim"),
            linearConvKernelSize: try i("linear_conv_kernel_dim"),
            numSharedExperts: 0,
            numDenseLayers: numLayers,
            denseIntermediateSize: intermediate)
        try crossCheckProductionQwen38(arch, configPath: configPath)
        return arch
    }

    /// Production Qwen3.8-27B baseline (mirrors the runtime's
    /// `ArchConfig.qwen38_27B`; the repack target has no dependency on the
    /// runtime module). A config that matches the production shape
    /// (hidden 5120, 64 layers) must agree on every field; toy/synthetic
    /// configs are exempt.
    private static func crossCheckProductionQwen38(_ a: ArchInfo,
                                                   configPath: String) throws {
        guard a.hiddenSize == 5120, a.numLayers == 64 else { return }
        var expectedMask = [UInt8](repeating: 2, count: 64)
        for i in stride(from: 3, to: 64, by: 4) { expectedMask[i] = 1 }
        let expected = ArchInfo(
            hiddenSize: 5120,
            intermediateSize: 17_408,
            moeIntermediateSize: 0,
            numHeads: 24,
            numKVHeads: 4,
            numFullKVHeads: 4,
            headDim: 256,
            fullHeadDim: 256,
            vocabSize: 248_320,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 64,
            numExperts: 0,
            topKExperts: 0,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .qwen38,
            attnOutputGate: true,
            attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true,
            linearNumKHeads: 16,
            linearNumVHeads: 48,
            linearKeyHeadDim: 128,
            linearValueHeadDim: 128,
            linearConvKernelSize: 4,
            numSharedExperts: 0,
            numDenseLayers: 64,
            denseIntermediateSize: 17_408)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen3_5 config does not match the pinned "
                    + "Qwen3.8-27B architecture baseline")
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
        // Per-layer-type compression rates; legacy scalar keys fold in.
        // Parsed before the layer mask so `compress_ratios` entries can be
        // matched against them.
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

        var mask: [UInt8] = []
        if let layerTypes = tc["layer_types"] as? [String] {
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
        } else if let allRatios = tc["compress_ratios"] as? [Any] {
            // The published config encodes layer kinds as per-layer
            // compression ratios: 0 = sliding-window, csaRate = CSA,
            // hcaRate = HCA. It carries one extra trailing entry per
            // `num_nextn_predict_layers` MTP layer, which the conversion
            // does not export — keep only the decoder layers.
            let numLayers = try i("num_hidden_layers")
            let ratios = allRatios.prefix(numLayers)
            mask.reserveCapacity(ratios.count)
            for r in ratios {
                guard let n = (r as? Int) ?? (r as? NSNumber)?.intValue else {
                    throw RepackError.configJsonInvalid(
                        path: configPath, detail: "non-integer compress_ratios entry")
                }
                switch n {
                case 0:       mask.append(0)
                case csaRate: mask.append(3)
                case hcaRate: mask.append(4)
                default:
                    throw RepackError.configJsonInvalid(
                        path: configPath,
                        detail: "compress_ratios entry \(n) matches neither "
                            + "the CSA rate \(csaRate) nor the HCA rate \(hcaRate)")
                }
            }
        } else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing layer_types / compress_ratios")
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

        // YaRN applies to the compress rope only (upstream folds top-level
        // `rope_scaling` into the `compress` rope-type parameters and keeps
        // `main` unscaled; attention_factor is forced to 1.0).
        var yarnFactor = 0.0
        var yarnOriginalMax = 0
        var yarnBetaFast = 0.0
        var yarnBetaSlow = 0.0
        if let scaling = tc["rope_scaling"] as? [String: Any] {
            let kind = (scaling["rope_type"] as? String)
                ?? (scaling["type"] as? String) ?? "default"
            guard kind == "yarn" else {
                throw RepackError.configJsonInvalid(
                    path: configPath,
                    detail: "unsupported rope_scaling type \"\(kind)\"")
            }
            guard let f = (scaling["factor"] as? Double)
                    ?? (scaling["factor"] as? NSNumber)?.doubleValue,
                  let om = (scaling["original_max_position_embeddings"] as? Int)
                    ?? (scaling["original_max_position_embeddings"] as? NSNumber)?.intValue
            else {
                throw RepackError.configJsonInvalid(
                    path: configPath,
                    detail: "yarn rope_scaling missing factor / original_max_position_embeddings")
            }
            yarnFactor = f
            yarnOriginalMax = om
            yarnBetaFast = (scaling["beta_fast"] as? Double)
                ?? (scaling["beta_fast"] as? NSNumber)?.doubleValue ?? 32.0
            yarnBetaSlow = (scaling["beta_slow"] as? Double)
                ?? (scaling["beta_slow"] as? NSNumber)?.doubleValue ?? 1.0
        }

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
            caRopeScalingFactor: yarnFactor,
            caRopeScalingOriginalMax: yarnOriginalMax,
            caRopeScalingBetaFast: yarnBetaFast,
            caRopeScalingBetaSlow: yarnBetaSlow,
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
            caRopeScalingFactor: 16.0,
            caRopeScalingOriginalMax: 65_536,
            caRopeScalingBetaFast: 32.0,
            caRopeScalingBetaSlow: 1.0,
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
