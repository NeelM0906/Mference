import CoreFoundation
import Foundation

/// Model family discriminator, mirrored into `manifest.json -> arch.family`.
/// Raw values match the runtime's `ModelFamily`.
enum RepackModelFamily: String, Sendable, Equatable {
    case qwen36 = "qwen36"
    case qwen38 = "qwen38"
    case maple = "maple"
    /// MiniCPM5-2B: plain-llama dense. Installed from the vendor's BF16 repo
    /// (quantize-in-flight) and, as the W2.1b control, from the vendor's own
    /// MLX INT4 conversion (pre-quantized path). Bare `model.` prefix and a
    /// top-level `lm_head.weight` on both.
    case minicpm5 = "minicpm5"
    /// Inkling 2 Mini, copied byte for byte from Unsloth's UD-Q4_K_XL GGUF
    /// (`Inkling2GGUFPlanner`); never planned from safetensors.
    case inkling2Loop = "inkling2Loop"
}

/// Inkling 2's axes, mirrored from the runtime's `ArchConfig`: the
/// relative-position, short-convolution, router, vocabulary and sink fields it
/// shares with the retired Inkling-Small baseline, and the runtime's
/// `Inkling2Config`. Nil for every other family; `GTurboJSON` emits these keys
/// only when it is set, so every other manifest stays byte-identical.
struct Inkling2Axes: Sendable, Equatable {
    let relDRel: Int
    let relExtent: Int
    let relProjDim: Int
    let relLogScalingFloor: Int
    let relLogScalingAlpha: Double
    let sconvKernelSize: Int
    let embedNormEnabled: Bool
    let routerGateBias: Bool
    let unpaddedVocabSize: Int
    let attentionSinks: Bool
    let loopCoreStart: Int
    let loopCoreEnd: Int
    let loopNumPasses: Int
    let loopGateRank: Int
    let attnResBlockSize: Int
    let qkClipThreshold: Double
    let relClipThreshold: Double
    let attentionSinkProjected: Bool
    let expertGateSoftcap: Double
    let expertUpSoftcap: Double
    /// One GGML block format name per stored layer.
    let routedExpertFormats: [String]
}

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention.
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

    // Family-dependent extensions, emitted into every manifest.
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

    // Router and FFN-shape extensions. The defaults (softmax routing, unit
    // scaling, no clamp, one shared expert, no dense layers) describe the
    // families that do not emit these fields; `GTurboJSON` writes each one only
    // for the families whose manifest carries it.
    let routerScoringFunc: String
    let routedScalingFactor: Double
    let swigluLimit: Double
    let numSharedExperts: Int
    let numDenseLayers: Int
    let denseIntermediateSize: Int
    let routerNormAfterTopK: Bool

    /// Per-head q/k RMSNorm gains on full-attention layers. Only plain-llama
    /// families (MiniCPM5) set it false.
    let qkNorm: Bool

    /// Inkling 2's axes; nil for every other family.
    let inkling2: Inkling2Axes?

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
         routerScoringFunc: String = "softmax",
         routedScalingFactor: Double = 1.0,
         swigluLimit: Double = 0.0,
         numSharedExperts: Int = 1,
         numDenseLayers: Int = 0,
         denseIntermediateSize: Int = 0,
         routerNormAfterTopK: Bool = false,
         qkNorm: Bool = true,
         inkling2: Inkling2Axes? = nil) {
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
        self.routerScoringFunc = routerScoringFunc
        self.routedScalingFactor = routedScalingFactor
        self.swigluLimit = swigluLimit
        self.numSharedExperts = numSharedExperts
        self.numDenseLayers = numDenseLayers
        self.denseIntermediateSize = denseIntermediateSize
        self.routerNormAfterTopK = routerNormAfterTopK
        self.qkNorm = qkNorm
        self.inkling2 = inkling2
    }

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        let modelType = root["model_type"] as? String
        // Maple is text-only; its config is flat (no `text_config` wrapper),
        // so dispatch before the wrapper guard.
        if modelType == "maple" {
            return try loadMaple(configPath: configPath, tc: root)
        }
        // MiniCPM5 is a flat `LlamaForCausalLM` config. The loader is keyed on
        // the llama model type because that is all a config carries; the
        // production cross-check below refuses any llama config that is not
        // the pinned MiniCPM5-2B shape, and the installer's source
        // fingerprint refuses any repo that is not the pinned one.
        if modelType == "llama" {
            return try loadMiniCPM5(configPath: configPath, root: root)
        }
        guard let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        if modelType == "qwen3_5_moe" {
            return try loadQwen36(configPath: configPath, tc: tc)
        }
        if modelType == "qwen3_5" {
            return try loadQwen38(configPath: configPath, tc: tc)
        }
        throw RepackError.configJsonInvalid(
            path: configPath,
            detail: "unsupported model_type \"\(modelType ?? "<missing>")\"")
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

    // MARK: - MiniCPM5 (`model_type == "llama"`, `LlamaForCausalLM`)

    /// Plain-llama dense stack read from a flat `config.json`: every layer is
    /// full attention (no `layer_types`), one SwiGLU MLP per layer, no
    /// experts, no q/k norm, no output gate, full-head NeoX RoPE. Everything
    /// the runner hard-codes is checked here rather than assumed:
    /// `rms_norm_eps` must be the runtime's compile-time 1e-6, `rope_scaling`
    /// must be null, and `attention_bias` / `mlp_bias` must be off (the
    /// tensor contract has no bias tensors).
    private static func loadMiniCPM5(configPath: String,
                                     root: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (root[k] as? Int) ?? (root[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (root[k] as? Double) ?? (root[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        guard (root["architectures"] as? [String]) == ["LlamaForCausalLM"] else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "llama config must declare architectures [\"LlamaForCausalLM\"]")
        }
        guard root["rope_scaling"] is NSNull || root["rope_scaling"] == nil else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "rope_scaling must be null for minicpm5")
        }
        guard try d("rms_norm_eps") == 1e-6 else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "rms_norm_eps must be 1e-6: the runtime hard-codes that epsilon")
        }
        for bias in ["attention_bias", "mlp_bias"] {
            guard (root[bias] as? Bool ?? false) == false else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "\(bias) must be false for minicpm5")
            }
        }
        let numLayers = try i("num_hidden_layers")
        let hidden = try i("hidden_size")
        let numHeads = try i("num_attention_heads")
        let numKV = try i("num_key_value_heads")
        let headDim = (root["head_dim"] as? Int) ?? hidden / numHeads
        let intermediate = try i("intermediate_size")
        let theta = try d("rope_theta")
        let tie = (root["tie_word_embeddings"] as? Bool) ?? false
        let act = (root["hidden_act"] as? String) ?? "silu"

        let arch = ArchInfo(
            hiddenSize: hidden,
            intermediateSize: intermediate,
            moeIntermediateSize: 0,
            numHeads: numHeads,
            numKVHeads: numKV,
            numFullKVHeads: numKV,
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: 1.0,
            numLayers: numLayers,
            numExperts: 0,
            topKExperts: 0,
            tieWordEmbeddings: tie,
            attentionKEqV: false,
            fullAttentionLayerMask: [UInt8](repeating: 1, count: numLayers),
            hiddenActivation: act,
            family: .minicpm5,
            attnOutputGate: false,
            // `LlamaAttention.scaling = head_dim ** -0.5`; `pow` reproduces it
            // bit-exactly (1 / sqrt is one ulp off, and the manifest check is
            // an exact Double comparison against the runtime baseline).
            attentionScale: pow(Double(headDim), -0.5),
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
            numSharedExperts: 0,
            numDenseLayers: numLayers,
            denseIntermediateSize: intermediate,
            qkNorm: false)
        try crossCheckProductionMiniCPM5(arch, configPath: configPath)
        return arch
    }

    /// Production MiniCPM5-2B baseline (mirrors the runtime's
    /// `ArchConfig.miniCPM5_2B`). A config that matches the production shape
    /// (hidden 2048, 42 layers) must agree on every field; toy configs are
    /// exempt.
    private static func crossCheckProductionMiniCPM5(_ a: ArchInfo,
                                                     configPath: String) throws {
        guard a.hiddenSize == 2048, a.numLayers == 42 else { return }
        let expected = ArchInfo(
            hiddenSize: 2048,
            intermediateSize: 6144,
            moeIntermediateSize: 0,
            numHeads: 16,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 128,
            fullHeadDim: 128,
            vocabSize: 130_560,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 5_000_000.0,
            fullRopeTheta: 5_000_000.0,
            partialRotaryFactor: 1.0,
            numLayers: 42,
            numExperts: 0,
            topKExperts: 0,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [UInt8](repeating: 1, count: 42),
            hiddenActivation: "silu",
            family: .minicpm5,
            attnOutputGate: false,
            attentionScale: 0.08838834764831845,   // pow(128, -0.5)
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
            numSharedExperts: 0,
            numDenseLayers: 42,
            denseIntermediateSize: 6144,
            qkNorm: false)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "llama config does not match the pinned MiniCPM5-2B "
                    + "architecture baseline")
        }
    }
}
