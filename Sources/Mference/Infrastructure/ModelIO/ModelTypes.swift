import Foundation
import Metal

/// Model family discriminator. Selects the tensor-name contract, the layer
/// graph shape (norm sandwich vs plain pre-norm), and family-specific kernel
/// behavior. Stored in `manifest.json -> arch.family`; absent means Gemma 4
/// (the format's original architecture).
public enum ModelFamily: String, Sendable, Hashable {
    case gemma4 = "gemma4"
    case qwen36 = "qwen36"
    case qwen38 = "qwen38"
    case deepseekV4Flash = "deepseekV4Flash"
    case inklingSmall = "inklingSmall"
    case maple = "maple"
    /// Qwen3.8-Flash-Next. The repacker installs it and the runtime carries a
    /// compiled baseline so the install validates, but no runner executes it:
    /// `ManifestReader.familiesWithoutRunner` refuses it by name at every
    /// entry point until the kernels for its three new axes land.
    case qwen38flashnext = "qwen38flashnext"
}

/// Gated-DeltaNet (linear attention) dimensions. Zeroed for architectures
/// without linear-attention layers.
public struct LinearAttentionConfig: Sendable, Equatable {
    public let numKHeads: Int
    public let numVHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let convKernelSize: Int

    public init(numKHeads: Int, numVHeads: Int,
                keyHeadDim: Int, valueHeadDim: Int,
                convKernelSize: Int) {
        self.numKHeads = numKHeads
        self.numVHeads = numVHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convKernelSize = convKernelSize
    }

    public static let none = LinearAttentionConfig(
        numKHeads: 0, numVHeads: 0, keyHeadDim: 0, valueHeadDim: 0,
        convKernelSize: 0)

    /// Fused qkv projection rows: 2 * K-dim + V-dim. Also the depthwise conv
    /// channel count.
    public var qkvDim: Int { 2 * numKHeads * keyHeadDim + numVHeads * valueHeadDim }
    /// Value dim, also the z-gate projection rows and out_proj columns.
    public var valueDim: Int { numVHeads * valueHeadDim }
}

/// DeepSeek-V4 compressed-attention dimensions. Zeroed for architectures
/// without CSA/HCA layers.
///
/// V4 attention is shared-KV MQA (one 512-dim KV head read as both K and V)
/// with a low-rank query path, a grouped low-rank output projection, per-head
/// learnable attention sinks, and interleaved partial RoPE on the *trailing*
/// `ropeHeadDim` channels of each head. CSA layers pool every `csaCompressRate`
/// source tokens into one compressed KV entry (two overlapping series) and
/// gather the top `indexTopK` entries per query with a lightning indexer; HCA
/// layers pool every `hcaCompressRate` tokens non-overlapping and attend
/// densely over the result. Sliding layers rope at `mainRopeTheta` (the
/// ArchConfig `ropeTheta`); CSA/HCA layers and their compressors rope at
/// `compressRopeTheta`.
public struct CompressedAttentionConfig: Sendable, Equatable {
    public let qLoraRank: Int
    public let oLoraRank: Int
    public let oGroups: Int
    public let ropeHeadDim: Int
    public let indexNHeads: Int
    public let indexHeadDim: Int
    public let indexTopK: Int
    public let csaCompressRate: Int
    public let hcaCompressRate: Int
    public let compressRopeTheta: Double
    /// YaRN scaling on the compress rope only (the upstream reference
    /// applies `rope_scaling` to the `compress` rope type and leaves the
    /// sliding-window `main` rope unscaled, with attention_factor forced
    /// to 1.0). `ropeScalingFactor == 0` disables scaling.
    public let ropeScalingFactor: Double
    public let ropeScalingOriginalMax: Int
    public let ropeScalingBetaFast: Double
    public let ropeScalingBetaSlow: Double

    public init(qLoraRank: Int, oLoraRank: Int, oGroups: Int,
                ropeHeadDim: Int,
                indexNHeads: Int, indexHeadDim: Int, indexTopK: Int,
                csaCompressRate: Int, hcaCompressRate: Int,
                compressRopeTheta: Double,
                ropeScalingFactor: Double = 0,
                ropeScalingOriginalMax: Int = 0,
                ropeScalingBetaFast: Double = 0,
                ropeScalingBetaSlow: Double = 0) {
        self.qLoraRank = qLoraRank
        self.oLoraRank = oLoraRank
        self.oGroups = oGroups
        self.ropeHeadDim = ropeHeadDim
        self.indexNHeads = indexNHeads
        self.indexHeadDim = indexHeadDim
        self.indexTopK = indexTopK
        self.csaCompressRate = csaCompressRate
        self.hcaCompressRate = hcaCompressRate
        self.compressRopeTheta = compressRopeTheta
        self.ropeScalingFactor = ropeScalingFactor
        self.ropeScalingOriginalMax = ropeScalingOriginalMax
        self.ropeScalingBetaFast = ropeScalingBetaFast
        self.ropeScalingBetaSlow = ropeScalingBetaSlow
    }

    public static let none = CompressedAttentionConfig(
        qLoraRank: 0, oLoraRank: 0, oGroups: 0, ropeHeadDim: 0,
        indexNHeads: 0, indexHeadDim: 0, indexTopK: 0,
        csaCompressRate: 0, hcaCompressRate: 0, compressRopeTheta: 0)
}

/// Manifold-Constrained Hyper-Connection (mHC) residual dimensions. Zeroed
/// for architectures with a plain single-stream residual.
///
/// The residual is `mult` parallel streams. Each sublayer site owns a learned
/// mix `fn: [(2 + mult) * mult, mult * hiddenSize]` (plus per-output `base`
/// biases and 3 scales) producing sigmoid `pre` collapse weights, sigmoid
/// `post` placement weights (range [0, 2]), and a `mult × mult` combine
/// matrix projected onto the doubly-stochastic manifold by `sinkhornIters`
/// alternating row/column normalizations with floor `eps`.
public struct HyperConnectionConfig: Sendable, Equatable {
    public let mult: Int
    public let sinkhornIters: Int
    public let eps: Double

    public init(mult: Int, sinkhornIters: Int, eps: Double) {
        self.mult = mult
        self.sinkhornIters = sinkhornIters
        self.eps = eps
    }

    public static let none = HyperConnectionConfig(mult: 0, sinkhornIters: 0, eps: 0)
}

/// Learned relative-attention position encoding, used by architectures that
/// carry no RoPE at all. Inkling projects the residual to `projDim`
/// (`attn.wr_du`, width `numHeads * dRel`) and reshapes it to a per-head
/// `dRel` relative-state vector, which mixes a bank of bias-vs-distance
/// profiles (`attn.rel_logits_proj.proj`, `[dRel, extent]`) into one bias per
/// backward distance. The bias is zero outside `0 ..< extent`.
///
/// `extent` here is the **full-attention** width. Sliding layers use
/// `ArchConfig.slidingWindow` instead, so the two layer kinds ship differently
/// shaped `proj` tensors (Inkling: `[16, 512]` local, `[16, 1024]` global).
/// The `logScaling*` correction likewise applies only to full-attention
/// layers. Zeroed for RoPE architectures.
public struct RelativePositionConfig: Sendable, Equatable {
    public let dRel: Int
    /// Bias width on full-attention layers; sliding layers use the window.
    public let extent: Int
    /// Output width of `attn.wr_du`, i.e. `numHeads * dRel`.
    public let projDim: Int
    public let logScalingFloor: Int
    public let logScalingAlpha: Double

    public init(dRel: Int, extent: Int, projDim: Int,
                logScalingFloor: Int, logScalingAlpha: Double) {
        self.dRel = dRel
        self.extent = extent
        self.projDim = projDim
        self.logScalingFloor = logScalingFloor
        self.logScalingAlpha = logScalingAlpha
    }

    public static let none = RelativePositionConfig(
        dRel: 0, extent: 0, projDim: 0, logScalingFloor: 0, logScalingAlpha: 0)
}

/// Qwen3.8-Flash-Next's three new axes: low-rank hyper-connections, the QSA
/// attention indexer, and the per-layer n-gram embedding (PLE). Zeroed
/// (`.none`) for every family that carries none of them.
///
/// **Low-rank hyper-connections.** The residual stream is `hcCount` parallel
/// copies of `hiddenSize`, i.e. `hcCount * hiddenSize` wide from the embedding
/// through the last layer. Each sub-block site owns a `GatedResidual`:
/// `input_mix_weight_down` `[hcLowRank, hcCount * hidden]` and
/// `input_mix_weight_up` `[hcCount * hidden, hcLowRank]` factorize the mix,
/// `block_inject_weight` `[hcCount, hcCount * hidden]` places the block output
/// back into all streams, and `hc_norm` is a group RMSNorm with group size
/// `hiddenSize`. This is a *different* mechanism from
/// `HyperConnectionConfig` (DeepSeek V4's manifold-constrained mHC, which
/// carries a Sinkhorn-projected combine matrix and no low-rank factorization),
/// so the two configs are separate axes rather than one shared struct.
///
/// **QSA indexer.** Per full-attention layer, `index_qk_proj`
/// `[indexerNumHeads * indexerHeadDim + indexerNumKVHeads * indexerHeadDim,
/// hidden]` scores the visible prefix in blocks of `indexerCompressRatio`
/// tokens; the top `indexerBudget / indexerCompressRatio` blocks (plus the
/// incomplete tail, always selected) form the attention mask for that query.
/// Selection sparsity is applied through the mask only — KV entries are never
/// dropped.
///
/// **PLE.** At each layer id in `pleLayerIDs` (**one-indexed**, so id 2 is
/// `layers[1]`) a hashed n-gram embedding is added to the residual stream
/// before attention. Rows come from the streamed row pool described by
/// `manifest.plePool`, not from the resident buffer. `pleEosTokenID` delimits
/// the segments the n-gram shift must not cross.
///
/// Facts established against the reference implementation (2026-09-01 parity
/// harness) that constrain any kernel built on these axes:
///   * **RMSNorm upcasts.** Every `Qwen4ExpTextRMSNorm` computes
///     `_norm(x.float()) * (1 + w.float())` and casts back, so the reduction
///     and the scale are fp32 even when the operands are BF16.
///   * **The n-gram mix has zero integer headroom.** The hash values are
///     exactly 63 bits by construction, so the mix must be done in true
///     64-bit integer arithmetic — never `Double` (53-bit significand) and
///     never a 32-bit intermediate.
///   * **The indexer does not guarantee a query selects its own block.**
///     Nothing in this plumbing may assume a "keep self" entry; the visible
///     set is whatever the top-k over block scores plus the incomplete tail
///     produces.
///   * **Gated `q_proj` packs per head.** The `2 * numHeads * fullHeadDim`
///     rows are `[heads, 2 * headDim]` split on the last dimension, not a
///     global query-half / gate-half row split — which is already the
///     convention the shipped Qwen path implements (`split_q_gate_fp16`), so
///     `attnOutputGate` is reusable here unchanged.
public struct FlashNextConfig: Sendable, Equatable {
    /// Parallel residual streams. The residual is `hcCount * hiddenSize` wide.
    public let hcCount: Int
    /// Rank of the hyper-connection mix factorization (`input_mix_weight_*`).
    public let hcLowRank: Int
    /// Indexer query heads.
    public let indexerNumHeads: Int
    /// Indexer head width, shared by the query heads and the single key head.
    public let indexerHeadDim: Int
    /// Indexer key heads; 1 in production (one pooled key per block).
    public let indexerNumKVHeads: Int
    /// Tokens the indexer keeps visible per query, before the always-selected
    /// tail. `indexerBudget / indexerCompressRatio` blocks are selected.
    public let indexerBudget: Int
    /// Consecutive tokens pooled into one indexer block key.
    public let indexerCompressRatio: Int
    /// **One-indexed** layer ids carrying a PLE block: id `n` is `layers[n-1]`.
    public let pleLayerIDs: [Int]
    /// `split_ngram_parts`: how many shards the source n-gram table arrives
    /// in, and how many page-aligned regions the installed row pool holds.
    public let pleNgramShardCount: Int
    /// `ngram_vocab_size_base` verbatim — the PER-HEAD base vocab, *not* the
    /// table's row count. The true row count comes from the shard headers and
    /// is published in `manifest.plePool.layers[].rows`; nothing validates one
    /// against the other.
    public let pleNgramVocabSizeBase: Int
    /// Depthwise causal conv width in the PLE mixer (dilation is the n-gram
    /// size, which is read from the installed `layer_multipliers` length).
    public let pleConvKernelSize: Int
    /// Token id that delimits PLE n-gram segments: a shifted token stream must
    /// not read across it (`_shift_right_ignore_eos`, EOS inclusive).
    ///
    /// The installed manifests predating this axis do not publish it, so
    /// `validateArch` checks it only when present and otherwise trusts this
    /// compiled constant. Emitting `arch.pleEosTokenID` from the repacker is
    /// an open follow-up; until it lands, a checkpoint with a different EOS
    /// would violate the constant silently.
    public let pleEosTokenID: Int

    public init(hcCount: Int, hcLowRank: Int,
                indexerNumHeads: Int, indexerHeadDim: Int,
                indexerNumKVHeads: Int, indexerBudget: Int,
                indexerCompressRatio: Int,
                pleLayerIDs: [Int],
                pleNgramShardCount: Int,
                pleNgramVocabSizeBase: Int,
                pleConvKernelSize: Int,
                pleEosTokenID: Int) {
        self.hcCount = hcCount
        self.hcLowRank = hcLowRank
        self.indexerNumHeads = indexerNumHeads
        self.indexerHeadDim = indexerHeadDim
        self.indexerNumKVHeads = indexerNumKVHeads
        self.indexerBudget = indexerBudget
        self.indexerCompressRatio = indexerCompressRatio
        self.pleLayerIDs = pleLayerIDs
        self.pleNgramShardCount = pleNgramShardCount
        self.pleNgramVocabSizeBase = pleNgramVocabSizeBase
        self.pleConvKernelSize = pleConvKernelSize
        self.pleEosTokenID = pleEosTokenID
    }

    public static let none = FlashNextConfig(
        hcCount: 0, hcLowRank: 0,
        indexerNumHeads: 0, indexerHeadDim: 0, indexerNumKVHeads: 0,
        indexerBudget: 0, indexerCompressRatio: 0,
        pleLayerIDs: [], pleNgramShardCount: 0, pleNgramVocabSizeBase: 0,
        pleConvKernelSize: 0, pleEosTokenID: 0)

    /// Zero-indexed layer indices carrying a PLE block, derived from the
    /// one-indexed `pleLayerIDs` the checkpoint publishes.
    public var pleLayerIndices: [Int] { pleLayerIDs.map { $0 - 1 } }

    /// Indexer blocks selected per query: `indexerBudget / compressRatio`.
    public var indexerBlockBudget: Int {
        indexerCompressRatio > 0 ? indexerBudget / indexerCompressRatio : 0
    }
}

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention,
/// 3 = compressed sparse attention (CSA), 4 = heavily compressed attention
/// (HCA). Values 3/4 additionally include the sliding-window branch (DeepSeek
/// V4 concatenates compressed entries onto the window KV).
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String

    // Family-dependent extensions. Defaults describe Gemma 4 so that legacy
    // manifests (which omit them) validate unchanged.
    public let family: ModelFamily
    /// Full-attention q_proj emits `2 * numHeads * fullHeadDim` rows: per-head
    /// [query ; gate] halves. Attention output is multiplied by sigmoid(gate)
    /// before o_proj.
    public let attnOutputGate: Bool
    /// Softmax scale for full attention. Gemma 4 uses 1.0.
    public let attentionScale: Double
    /// Embedding lookup is multiplied by sqrt(hiddenSize) (Gemma) or not (Qwen).
    public let embeddingScaledBySqrtHidden: Bool
    /// Router has `router.scale` (input multiplier) and `per_expert_scale`
    /// tensors (Gemma). False: plain quantized linear router with renormalized
    /// top-k softmax weights and no auxiliary scale tensors.
    public let routerScaled: Bool
    /// Gemma's dual-branch FFN sandwich: pre/post feedforward norms plus a
    /// per-layer residual scalar. False = plain pre-norm residual block.
    public let ffnSandwichNorms: Bool
    /// Shared expert output is gated by sigmoid(shared_expert_gate(x)) (Qwen).
    public let sharedExpertGated: Bool
    /// Partial RoPE convention. False (Gemma): pairs (i, headDim/2 + i) for
    /// i < rotatedPairs with frequency divisor = headDim. True (Qwen/NeoX
    /// sub-dim): rotation confined to the first `rotaryDim` elements, pairing
    /// (i, rotaryDim/2 + i), frequency divisor = rotaryDim.
    public let ropeNeoxSubdim: Bool
    /// Gated-DeltaNet dimensions for layers with mask value 2.
    public let linearAttention: LinearAttentionConfig
    /// DeepSeek-V4 compressed-attention dimensions for layers with mask
    /// values 3/4 (and the family's shared-KV MQA sliding layers).
    public let compressedAttention: CompressedAttentionConfig
    /// Manifold-Constrained Hyper-Connection residual streams. `.none` means
    /// a plain single-stream residual.
    public let hyperConnections: HyperConnectionConfig
    /// Leading MoE layers whose expert selection is a frozen token-id lookup
    /// (`tid2eid`) instead of a learned argmax. 0 for Gemma/Qwen.
    public let numHashRoutedLayers: Int
    /// Router score activation applied to the logits before top-k selection:
    /// "softmax" (Gemma/Qwen behavior) or "sqrtsoftplus" (DeepSeek V4).
    public let routerScoringFunc: String
    /// Multiplier applied to the renormalized top-k routing weights.
    public let routedScalingFactor: Double
    /// Clamp for expert gate (max) and up (±) pre-activations. 0 = no clamp.
    public let swigluLimit: Double
    /// Learned relative-attention bias; `.none` for RoPE architectures.
    public let relativePosition: RelativePositionConfig
    /// Depthwise short-convolution width applied to the block inputs and to
    /// the K/V streams. 0 disables the short-conv path entirely.
    public let sconvKernelSize: Int
    /// Shared experts active on every token (Gemma/Qwen/DeepSeek ship 1,
    /// Inkling 2).
    public let numSharedExperts: Int
    /// Leading layers that use a plain dense FFN instead of the MoE block.
    public let numDenseLayers: Int
    /// FFN width of those dense layers; 0 when `numDenseLayers` is 0.
    public let denseIntermediateSize: Int
    /// Whether the shared experts occupy their own router logits as sinks, so
    /// the gate emits `numExperts + numSharedExperts` scores.
    public let sharedExpertSink: Bool
    /// RMS norm applied to the token embeddings before the first layer.
    public let embedNormEnabled: Bool
    /// muP output scaling divided into the logits. 1.0 disables.
    public let logitsWidthMultiplier: Double
    /// Real vocabulary size when the embedding matrix is padded for alignment.
    /// Logits beyond this are padding and must be dropped before sampling, or
    /// the model can emit ids the tokenizer cannot decode. 0 means "no
    /// padding": `vocabSize` is the real vocabulary.
    public let unpaddedVocabSize: Int
    /// Learned additive bias on the router logits, used for selection only.
    public let routerGateBias: Bool
    /// Renormalize the top-k router weights after selection rather than before.
    public let routerNormAfterTopK: Bool
    /// Per-layer learned scalar multiplying the router weights.
    public let routerGlobalScale: Bool
    /// Qwen3.8-Flash-Next's low-rank hyper-connections, QSA indexer and
    /// per-layer n-gram embedding. `.none` for every other family.
    public let flashNext: FlashNextConfig

    public init(
        hiddenSize: Int,
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
        family: ModelFamily = .gemma4,
        attnOutputGate: Bool = false,
        attentionScale: Double = 1.0,
        embeddingScaledBySqrtHidden: Bool = true,
        routerScaled: Bool = true,
        ffnSandwichNorms: Bool = true,
        sharedExpertGated: Bool = false,
        ropeNeoxSubdim: Bool = false,
        linearAttention: LinearAttentionConfig = .none,
        compressedAttention: CompressedAttentionConfig = .none,
        hyperConnections: HyperConnectionConfig = .none,
        numHashRoutedLayers: Int = 0,
        routerScoringFunc: String = "softmax",
        routedScalingFactor: Double = 1.0,
        swigluLimit: Double = 0.0,
        relativePosition: RelativePositionConfig = .none,
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
        unpaddedVocabSize: Int = 0,
        flashNext: FlashNextConfig = .none
    ) {
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
        self.linearAttention = linearAttention
        self.compressedAttention = compressedAttention
        self.hyperConnections = hyperConnections
        self.numHashRoutedLayers = numHashRoutedLayers
        self.routerScoringFunc = routerScoringFunc
        self.routedScalingFactor = routedScalingFactor
        self.swigluLimit = swigluLimit
        self.relativePosition = relativePosition
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
        self.flashNext = flashNext
    }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh"
    )

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }

    /// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
    /// gated-DeltaNet linear-attention layers and 10 full-attention layers
    /// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
    /// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
    ///
    /// The sliding-window slots (`numKVHeads`/`headDim`/`slidingWindow`/
    /// `ropeTheta`) mirror the full-attention values; the architecture has no
    /// sliding-window layers so they are never used to size storage.
    public static let qwen36_35B_A3B = ArchConfig(
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
        fullAttentionLayerMask: Self.qwen36LayerMask(),
        hiddenActivation: "silu",
        family: .qwen36,
        attnOutputGate: true,
        attentionScale: 0.0625,   // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 32,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4)
    )

    private static func qwen36LayerMask() -> [UInt8] {
        // Layer kinds: 2 = gated-DeltaNet linear, 1 = full attention on every
        // 4th layer ((i + 1) % 4 == 0).
        var mask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { mask[i] = 1 }
        return mask
    }

    /// Canonical Qwen3.8-27B baseline (text stack of the multimodal
    /// checkpoint; the vision tower is excluded at repack). A 64-layer dense
    /// hybrid: 48 gated-DeltaNet linear-attention layers and 16 full-attention
    /// layers (every 4th layer), one SwiGLU MLP per layer — no routed experts,
    /// no shared expert, no router. Sigmoid attention output gate, QK norms,
    /// partial RoPE over 64 of 256 dims, untied 248k-row lm_head. Everything
    /// is resident: with zero experts per layer the expert streamer never
    /// opens a pool.
    public static let qwen38_27B = ArchConfig(
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
        fullAttentionLayerMask: Self.qwen38LayerMask(),
        hiddenActivation: "silu",
        family: .qwen38,
        attnOutputGate: true,
        attentionScale: 0.0625,   // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 48,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4),
        numSharedExperts: 0,
        numDenseLayers: 64,
        denseIntermediateSize: 17_408
    )

    private static func qwen38LayerMask() -> [UInt8] {
        // Same 3:1 hybrid shape as Qwen 3.6 (full_attention_interval = 4).
        var mask = [UInt8](repeating: 2, count: 64)
        for i in stride(from: 3, to: 64, by: 4) { mask[i] = 1 }
        return mask
    }

    /// Canonical Maple Preview baseline: 24 plain pre-norm MoE layers with
    /// 256 routed experts (top-8) and no shared expert. Three sliding layers
    /// are followed by one global NoPE layer.
    public static let maplePreview = ArchConfig(
        hiddenSize: 2048,
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
        fullAttentionLayerMask: Self.mapleLayerMask(),
        hiddenActivation: "silu",
        family: .maple,
        attnOutputGate: false,
        attentionScale: 1.0 / Double(128).squareRoot(),
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: true,
        routerScoringFunc: "softmax",
        routedScalingFactor: 1.0,
        swigluLimit: 7.0,
        numSharedExperts: 0,
        routerNormAfterTopK: true
    )

    private static func mapleLayerMask() -> [UInt8] {
        (0..<24).map { $0 % 4 == 3 ? 1 : 0 }
    }

    /// Canonical DeepSeek-V4-Flash 284B-A13B baseline: 43 all-MoE layers
    /// (1 shared + 256 routed experts of width 2048, top-6; layers 0–2 route
    /// by frozen `tid2eid` hash), shared-KV MQA attention (64 query heads over
    /// one 512-dim K=V head, low-rank Q, grouped low-rank output projection,
    /// per-head attention sinks), sliding window 128 on every layer, and
    /// compressed long-range KV: CSA (rate 4 + lightning indexer) on even
    /// layers from 2, HCA (rate 128, dense) on odd layers from 3. The
    /// residual is 4 mHC streams. Untied lm_head, no logit softcap.
    ///
    /// `numKVHeads`/`numFullKVHeads` are 1 (shared KV). `attentionKEqV` is
    /// true in the strongest sense: K and V are the same cache entry.
    /// `partialRotaryFactor = 64/512` with V4's interleaved-trailing RoPE
    /// convention (neither Gemma's proportional nor Qwen's NeoX sub-dim
    /// layout; the family's own kernels implement it).
    public static let deepseekV4Flash_284B_A13B = ArchConfig(
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
        fullAttentionLayerMask: Self.deepseekV4FlashLayerMask(),
        hiddenActivation: "silu",
        family: .deepseekV4Flash,
        attnOutputGate: false,
        attentionScale: 0.044194173824159216,   // 512^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: false,
        linearAttention: .none,
        compressedAttention: CompressedAttentionConfig(
            qLoraRank: 1024, oLoraRank: 1024, oGroups: 8,
            ropeHeadDim: 64,
            indexNHeads: 64, indexHeadDim: 128, indexTopK: 512,
            csaCompressRate: 4, hcaCompressRate: 128,
            compressRopeTheta: 160_000.0,
            ropeScalingFactor: 16.0,
            ropeScalingOriginalMax: 65_536,
            ropeScalingBetaFast: 32.0,
            ropeScalingBetaSlow: 1.0),
        hyperConnections: HyperConnectionConfig(
            mult: 4, sinkhornIters: 20, eps: 1.0e-6),
        numHashRoutedLayers: 3,
        routerScoringFunc: "sqrtsoftplus",
        routedScalingFactor: 1.5,
        swigluLimit: 10.0
    )

    private static func deepseekV4FlashLayerMask() -> [UInt8] {
        // Layer kinds: 0 = sliding-window only (layers 0-1), then 3 = CSA on
        // even layers and 4 = HCA on odd layers (compress_ratios
        // [0, 0, 4, 128, 4, 128, ...]).
        var mask = [UInt8](repeating: 0, count: 43)
        for i in 2..<43 { mask[i] = i.isMultiple(of: 2) ? 3 : 4 }
        return mask
    }

    /// Canonical Inkling-Small 276B-A12B baseline, checked against the
    /// installed model manifest. Source checkpoint
    /// `pipenetwork/Inkling-Small-MLX-4bit` revision `9d6e4720` (MLX affine
    /// 4-bit, group 64, uniform across embeddings/attention/experts; the
    /// router gate stays BF16). See `docs/INKLING_SMALL.md`.
    ///
    /// The distinguishing features versus the other three families: no RoPE at
    /// all (a learned relative-attention bias carries position, so
    /// `ropeTheta`/`partialRotaryFactor` are zero), depthwise short
    /// convolutions on the block inputs and the K/V streams, two shared
    /// experts that sink their own router logits, and two leading dense-FFN
    /// layers at 8× the expert width.
    ///
    /// `intermediateSize` is the shared-expert FFN width, which equals the
    /// routed width here (2048); `denseIntermediateSize` (16 384) applies only
    /// to layers 0–1.
    public static let inklingSmall_276B_A12B = ArchConfig(
        hiddenSize: 4096,
        intermediateSize: 2048,
        moeIntermediateSize: 2048,
        numHeads: 32,
        numKVHeads: 8,
        numFullKVHeads: 8,
        headDim: 128,
        fullHeadDim: 128,
        vocabSize: 201_024,
        slidingWindow: 512,
        finalLogitSoftcap: 0.0,
        ropeTheta: 0.0,
        fullRopeTheta: 0.0,
        partialRotaryFactor: 0.0,
        numLayers: 42,
        numExperts: 256,
        topKExperts: 6,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.inklingSmallLayerMask(),
        hiddenActivation: "silu",
        family: .inklingSmall,
        attnOutputGate: false,
        // Inkling RMS-normalizes q and k per head, so attention scales by
        // `1 / head_dim`, NOT `1 / sqrt(head_dim)` — see
        // `inkling_mlx/attention.py`. Kept as the same expression the
        // converter evaluates, since `validateArch` compares exactly.
        attentionScale: 1.0 / Double(128),
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: false,
        routerScoringFunc: "sigmoid",
        routedScalingFactor: 8.0,              // route_scale
        swigluLimit: 0.0,
        relativePosition: RelativePositionConfig(
            dRel: 16, extent: 1024, projDim: 512,
            logScalingFloor: 128_000, logScalingAlpha: 0.1),
        sconvKernelSize: 4,
        numSharedExperts: 2,
        numDenseLayers: 2,
        denseIntermediateSize: 16_384,
        sharedExpertSink: true,
        embedNormEnabled: true,
        logitsWidthMultiplier: 16.0,
        routerGateBias: true,
        routerNormAfterTopK: true,
        routerGlobalScale: true,
        unpaddedVocabSize: 200_058
    )

    private static func inklingSmallLayerMask() -> [UInt8] {
        // `local_layer_ids` covers every layer except 5, 11, 17, 23, 29, 35
        // and 41 — i.e. one global layer at the end of each group of six.
        // 0 = sliding-window (512), 1 = full attention.
        (0..<42).map { $0 % 6 == 5 ? 1 : 0 }
    }

    /// Canonical Qwen3.8-Flash-Next 180B-A3.5B baseline (text stack of the
    /// multimodal `qwen4_exp` checkpoint; the vision tower is excluded at
    /// repack). 48 layers in the same 3 : 1 hybrid as Qwen 3.6/3.8 — 36
    /// gated-DeltaNet layers and 12 full-attention layers — with 512 routed
    /// experts (top-10) of width 640 plus one sigmoid-gated shared expert,
    /// gated attention output, per-head q/k norms, partial NeoX RoPE over 64
    /// of 256 dims at θ 1e7, and an untied 248 320-row head.
    ///
    /// Three axes are new and carried in `flashNext`: the residual is 4
    /// low-rank hyper-connected streams (so the stream is 10 240 wide and
    /// there is **no** final norm — the global mixer collapses it before
    /// `lm_head`), each full-attention layer carries a QSA indexer, and
    /// layer id 2 (`layers[1]`) carries a PLE n-gram embedding whose 320M-row
    /// table streams from `manifest.plePool` rather than sitting resident.
    ///
    /// Values are read from the installed manifest at
    /// `scratch/qwen38flashnext.gturbo` (revision `de4b8e4d`); see
    /// `docs/families/QWEN38_FLASH_NEXT.md`. The runner does not exist yet:
    /// `ManifestReader.familiesWithoutRunner` refuses this family by name at
    /// every load path. The baseline exists so the manifest can be validated
    /// and toy fixtures built while the kernels are written.
    public static let qwen38FlashNext_180B_A3_5B = ArchConfig(
        hiddenSize: 2560,
        intermediateSize: 640,
        moeIntermediateSize: 640,
        numHeads: 24,
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
        numLayers: 48,
        numExperts: 512,
        topKExperts: 10,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwen38FlashNextLayerMask(),
        hiddenActivation: "silu",
        family: .qwen38flashnext,
        attnOutputGate: true,          // output_gate_type: sigmoid
        attentionScale: 0.0625,        // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 48,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4),
        // The router path is Qwen 3.6's: softmax over the logits, top-k of the
        // probabilities, then renormalize the selected k (`norm_topk_prob`).
        // `routerNormAfterTopK` stays false because that flag selects
        // Inkling's and Maple's *different* ordering, not this one.
        numSharedExperts: 1,
        flashNext: FlashNextConfig(
            hcCount: 4,
            hcLowRank: 320,
            indexerNumHeads: 4,
            indexerHeadDim: 128,
            indexerNumKVHeads: 1,
            indexerBudget: 2048,
            indexerCompressRatio: 4,
            pleLayerIDs: [2],
            pleNgramShardCount: 128,
            pleNgramVocabSizeBase: 20_000_000,
            pleConvKernelSize: 4,
            // config.json text_config.eos_token_id @ de4b8e4d.
            pleEosTokenID: 248_044)
    )

    private static func qwen38FlashNextLayerMask() -> [UInt8] {
        // Same 3:1 hybrid shape as Qwen 3.6/3.8 (full_attention_interval = 4).
        var mask = [UInt8](repeating: 2, count: 48)
        for i in stride(from: 3, to: 48, by: 4) { mask[i] = 1 }
        return mask
    }

    /// Registry keyed by `manifest.arch.family` for auto-detection at load.
    ///
    /// A family here has a validated baseline, not necessarily a runner:
    /// `ManifestReader.familiesWithoutRunner` is the separate, authoritative
    /// gate over whether the runtime can execute one.
    public static let knownArchitectures: [ModelFamily: ArchConfig] = [
        .gemma4: .gemma4_26B_A4B,
        .qwen36: .qwen36_35B_A3B,
        .qwen38: .qwen38_27B,
        .deepseekV4Flash: .deepseekV4Flash_284B_A13B,
        .inklingSmall: .inklingSmall_276B_A12B,
        .maple: .maplePreview,
        .qwen38flashnext: .qwen38FlashNext_180B_A3_5B,
    ]

    /// Resident INT4 GEMV shapes this architecture issues during decode, for
    /// pipeline specialization. Constant-folding the loop bounds measurably
    /// raises achieved bandwidth on the narrower projections.
    public var decodeInt4GEMVShapes: [(m: Int, n: Int)] {
        var shapes: [(m: Int, n: Int)] = []
        if hasCompressedAttentionLayers {
            // DeepSeek V4 low-rank attention path: q_a, q_b, kv, o_a (as
            // oGroups separate group GEMVs), o_b, plus the compressor /
            // indexer projections.
            let ca = compressedAttention
            shapes.append((m: ca.qLoraRank, n: hiddenSize))
            shapes.append((m: numHeads * fullHeadDim, n: ca.qLoraRank))
            shapes.append((m: fullHeadDim, n: hiddenSize))
            shapes.append((m: ca.oLoraRank, n: numHeads * fullHeadDim / ca.oGroups))
            shapes.append((m: hiddenSize, n: ca.oGroups * ca.oLoraRank))
            shapes.append((m: 2 * fullHeadDim, n: hiddenSize))              // CSA compressor kv/gate
            shapes.append((m: fullHeadDim, n: hiddenSize))                  // HCA compressor kv/gate
            shapes.append((m: 2 * ca.indexHeadDim, n: hiddenSize))          // indexer kv/gate
            shapes.append((m: ca.indexNHeads * ca.indexHeadDim, n: ca.qLoraRank))
        } else if attnOutputGate {
            shapes.append((m: 2 * numHeads * fullHeadDim, n: hiddenSize))
        } else {
            shapes.append((m: numHeads * fullHeadDim, n: hiddenSize))
        }
        if !hasCompressedAttentionLayers {
            shapes.append((m: numFullKVHeads * fullHeadDim, n: hiddenSize))
            shapes.append((m: hiddenSize, n: numHeads * fullHeadDim))
        }
        if hasLinearAttentionLayers {
            let la = linearAttention
            shapes.append((m: la.qkvDim, n: hiddenSize))
            shapes.append((m: la.valueDim, n: hiddenSize))
            shapes.append((m: hiddenSize, n: la.valueDim))
        }
        if numSharedExperts > 0 {
            shapes.append((m: intermediateSize, n: hiddenSize))
            shapes.append((m: hiddenSize, n: intermediateSize))
        }
        if numDenseLayers == numLayers, denseIntermediateSize > 0 {
            // Fully dense family: every layer runs one resident SwiGLU MLP.
            shapes.append((m: denseIntermediateSize, n: hiddenSize))
            shapes.append((m: hiddenSize, n: denseIntermediateSize))
        }
        return shapes
    }

    /// Resident INT8 GEMV shapes issued during decode (router and, when the
    /// architecture has one, the shared-expert scalar gate).
    public var decodeInt8GEMVShapes: [(m: Int, n: Int)] {
        var shapes: [(m: Int, n: Int)] = family == .maple
            ? [] : [(m: numExperts, n: hiddenSize)]
        if sharedExpertGated { shapes.append((m: 1, n: hiddenSize)) }
        return shapes
    }

    /// Layer kind helpers over the mask encoding.
    public func layerIsFull(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 1 }
    public func layerIsLinear(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 2 }
    public func layerIsCSA(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 3 }
    public func layerIsHCA(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 4 }
    /// True for any layer carrying a compressed long-range KV branch.
    public func layerIsCompressed(_ layer: Int) -> Bool {
        let v = fullAttentionLayerMask[layer]
        return v == 3 || v == 4
    }
    public var hasLinearAttentionLayers: Bool { fullAttentionLayerMask.contains(2) }
    public var hasCompressedAttentionLayers: Bool {
        fullAttentionLayerMask.contains(where: { $0 == 3 || $0 == 4 })
    }
    /// Hash-routed MoE layer: expert selection is `tid2eid[token]`.
    public func layerIsHashRouted(_ layer: Int) -> Bool { layer < numHashRoutedLayers }

    /// True when the residual is Flash-Next's low-rank hyper-connected bundle
    /// of `flashNext.hcCount` streams rather than a single stream.
    public var hasLowRankHyperConnections: Bool { flashNext.hcCount > 0 }
    /// Width of the residual the layer graph carries between blocks. Blocks
    /// themselves always run at `hiddenSize`; only the residual widens.
    public var residualStreamWidth: Int {
        hasLowRankHyperConnections ? flashNext.hcCount * hiddenSize : hiddenSize
    }
    /// True when this layer carries a PLE n-gram embedding block. `pleLayerIDs`
    /// is one-indexed in the checkpoint, so id 2 answers true for layer 1.
    public func layerIsPLE(_ layer: Int) -> Bool {
        flashNext.pleLayerIDs.contains(layer + 1)
    }
    /// True when this layer carries a QSA indexer: full-attention layers of a
    /// family that configures one.
    public func layerHasAttentionIndexer(_ layer: Int) -> Bool {
        flashNext.indexerNumHeads > 0 && layerIsFull(layer)
    }
}

/// Failure modes for the validation gates in `Model.load`.
enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)
    case routedExpertPlanUnavailable(layer: Int)
    case eagerExpertFillFailed(layer: Int)
    /// The repacker can install this family but no runner executes it yet.
    /// Named rather than "unknown family" so the failure points at the missing
    /// kernels instead of reading as a corrupt install.
    case familyRunnerNotImplemented(family: String, missingAxes: [String])
    /// The install declares no `manifest.plePool` block, or none for the layer
    /// the arch says carries a PLE n-gram embedding.
    case plePoolMissing(layer: Int)
    /// The `manifest.plePool` block is present but self-inconsistent (unknown
    /// kind, geometry that does not tile the file, shard rows that do not sum
    /// to the declared total).
    case plePoolInvalid(detail: String)
    /// A row index outside the pool's declared row count.
    case plePoolRowOutOfRange(row: Int, rows: Int)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        case .routedExpertPlanUnavailable(let layer):
            return "routed expert fetch plan unavailable for layer \(layer)"
        case .eagerExpertFillFailed(let layer):
            return "eager routed expert read failed for layer \(layer); decode step aborted"
        case .familyRunnerNotImplemented(let family, let missingAxes):
            return "family \(family) is installed but its runner is not implemented; "
                + "missing axes: \(missingAxes.joined(separator: ", "))"
        case .plePoolMissing(let layer):
            return "manifest.plePool has no row-lookup pool for PLE layer \(layer)"
        case .plePoolInvalid(let detail):
            return "manifest.plePool is invalid: \(detail)"
        case .plePoolRowOutOfRange(let row, let rows):
            return "PLE row \(row) is outside the pool's \(rows) rows"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32, 4 = I64
    /// (integer lookup tables, read CPU-side only); 5 = I32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
