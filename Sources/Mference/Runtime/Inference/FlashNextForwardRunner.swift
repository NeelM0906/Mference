import Foundation
import Metal

public enum FlashNextForwardRunnerError: Error, CustomStringConvertible {
    case invalidConfiguration(String)
    case invalidInput(String)
    case commandFailed(String)

    public var description: String {
        switch self {
        case .invalidConfiguration(let reason),
             .invalidInput(let reason),
             .commandFailed(let reason):
            return reason
        }
    }
}

/// The production forward runner for `qwen38flashnext` (upstream `qwen4_exp`).
///
/// # Shape of the model
///
/// The residual stream is `hc_count(4) x hidden(2560) = 10240` wide from the
/// embedding tile to the very end. Blocks run at 2560; the hyper-connections mix
/// the four streams down to a block input and inject the block output back into
/// all four. **There is no final norm** — the global `hyper_connection_mixer`
/// produces `last_hidden_state` and `lm_head` applies straight to it.
///
/// ```
/// embed -> tile x4
/// for L in 0..<48:
///     if L == PLE layer:  hyper += PLE(hyper, token)
///     mixed, inj = attn_hyper_connection(hyper)
///     block      = GDN(mixed)                     if L % 4 != 3
///                | Attention(mixed, indexer(mixed)) otherwise
///     hyper     += inject(block, inj)
///     mixed, inj = mlp_hyper_connection(hyper)
///     block      = SparseMoE(mixed)               // top-10 streamed + shared
///     hyper     += inject(block, inj)
/// last = hyper_connection_mixer(hyper)
/// logits = lm_head . last
/// ```
///
/// # Precision
///
/// Activations are FP16 in memory, matching the rest of the runtime, with three
/// deliberate exceptions, each because the value feeds a nonlinearity where FP16
/// rounding is amplified rather than absorbed:
///
/// * the hyper-connections' pre-sigmoid mix gate, low-rank vector and four
///   injection scalars (FP32 — `FlashNextHyperConnections`);
/// * the shared expert's pre-sigmoid scalar gate (FP32);
/// * **the whole QSA indexer** — projection, query heads, raw-key and pooled
///   block-key caches, and scores (FP32 — `FlashNextIndexer`). Its output is a
///   selection, not a tensor: a score perturbed below the FP16 floor reorders the
///   top-k boundary and changes which KV a layer may read.
///
/// Norms upcast to FP32 internally, as the reference does. The KV cache is FP16.
///
/// # Both install dtypes
///
/// Every projection goes through `FlashNextWeightMatrix`, which carries the
/// stored dtype with the buffer: INT4 affine group-64 for the production install,
/// dense BF16 for the parity install (whose `moe_intermediate_size` of 32
/// group-64 cannot quantize at all). The routed experts, the embedding and
/// `lm_head` split the same way. Nothing in this file assumes a quantization.
///
/// # Prefill
///
/// Sequential, one token per call, through `produceWithoutLogits` for all but the
/// last prompt token. This runner does not implement `ChunkedPrefillRunner`, and
/// `ForwardRunnerFactory` hands it a `.off` prefill config accordingly. That is a
/// correctness-first choice with a real cost — a prompt token re-reads its ten
/// expert blobs per layer rather than amortizing them over a chunk — and it is
/// the first thing the perf pass should take.
///
/// PERF, not correctness: the layer loop takes two CPU round trips per attention
/// layer (indexer scores out, selection in) and one per layer for the router's
/// expert ids, ~60 per token before any expert I/O. `RealForwardRunner`'s
/// event-signalled overlap, its eager fill and its GPU slot map all apply here
/// and none of them are wired yet.
public final class FlashNextForwardRunner: ContinuableLogitProducer,
                                           ContextWindowReporting,
                                           HeadlessSequentialPrefillRunner,
                                           @unchecked Sendable {

    // MARK: - Capture

    /// Per-forward record of the same tensors and integer sets
    /// `FlashNextReferenceRunner.Capture` records, under the same keys, so a
    /// parity test can diff them point for point.
    ///
    /// Off unless a caller sets `capture`, and it costs a queue drain and a
    /// readback at every capture point — this is gate machinery, not a
    /// production path. Nothing in the forward pass changes when it is nil.
    public struct Capture {
        public var floats: [String: [Float]] = [:]
        public var integers: [String: [[Int]]] = [:]
        public init() {}
    }

    /// Set to a fresh `Capture` before a `produce` call to record it.
    public var capture: Capture?

    // MARK: - Per-layer weights

    private struct GDNTensors {
        let qkv: FlashNextWeightMatrix
        let z: FlashNextWeightMatrix
        let a: FlashNextWeightMatrix
        let b: FlashNextWeightMatrix
        let out: FlashNextWeightMatrix
        let conv: TensorView
        let aLog: TensorView
        let dtBias: TensorView
        let norm: TensorView
        /// The four input projections as views, for the fused INT4 GEMV. Nil
        /// when the install is not INT4 and the fused kernel does not apply.
        let fusedInProj: (qkv: TensorView, z: TensorView, a: TensorView, b: TensorView)?
    }

    private struct IndexerTensors {
        let qkProj: FlashNextWeightMatrix
        let qNorm: TensorView
        let kNorm: TensorView
    }

    private struct MoETensors {
        let router: FlashNextWeightMatrix
        let sharedGate: FlashNextWeightMatrix       // [1, hidden]
        let sharedGateProj: FlashNextWeightMatrix   // [intermediate, hidden]
        let sharedUp: FlashNextWeightMatrix
        let sharedDown: FlashNextWeightMatrix       // [hidden, intermediate]
        let expertOffsets: MoEExpertOffsets
        /// Routed experts stored dense BF16 rather than INT4 affine g64 — read
        /// from the layout's sub-tensors, not assumed.
        let expertsAreBF16: Bool
    }

    private struct LayerTensors {
        let attnHC: FlashNextHyperConnections.Weights
        let mlpHC: FlashNextHyperConnections.Weights
        let isLinear: Bool
        let gdn: GDNTensors?
        let attention: FlashNextAttention.Weights?
        let indexer: IndexerTensors?
        let moe: MoETensors
    }

    // MARK: - Stored state

    private let model: Model
    private let ctx: MetalContext
    private let cfg: ArchConfig
    public let maxContext: Int

    private let hidden: Int
    private let bundle: Int
    private let topK: Int
    private let numExperts: Int
    private let moeIntermediate: Int
    private let sharedIntermediate: Int
    private let pleLayer: Int
    private static let epsilon: Float = 1e-6

    // Kernels
    private let matVec: FlashNextMatVec
    private let rms: RMSNorm
    private let elementwise: Elementwise
    private let hc: FlashNextHyperConnections
    private let indexer: FlashNextIndexer
    private let attention: FlashNextAttention
    /// Built only when the install stores routed experts as INT4 affine g64 —
    /// the parity install stores them dense BF16 at top-2, a width the shipped
    /// INT4 reduce does not implement and never needs to.
    private let moeInt4: MoE?
    private let moeBF16: FlashNextMoE
    /// The shipped Qwen 3.8 GDN kernels, with the gated norm on SIGMOID. Nil
    /// when the install's GDN geometry is below their 32-lane floor — only the
    /// parity toy, whose Dk is 8.
    private let gdn: GDN?
    private let gdnState: GDNStateManager?
    /// The dimension-generic fallback, built only when `gdn` is nil.
    private let genericGDN: FlashNextGDN?
    private let genericGDNScratch: FlashNextGDN.Scratch?
    private let genericGDNState: [Int: FlashNextGDN.LayerState]
    private let embedInt4: EmbedLookupInt4
    private let embedBF16PSO: MTLComputePipelineState
    private let siluMulPSO: MTLComputePipelineState

    private let layers: [LayerTensors]
    private let embedding: TensorView
    private let embeddingMatrix: FlashNextWeightMatrix
    private let lmHead: FlashNextWeightMatrix
    private let mixer: FlashNextHyperConnections.Weights

    // PLE
    private let ple: FlashNextPLE?
    private let pleWeights: FlashNextPLE.Weights?
    private let pleScratch: FlashNextPLE.Scratch?
    private let pleHash: FlashNextPleHash?
    private let pleRowPool: PleRowPool?
    private let pleStaging: MTLBuffer?
    private var pleHistory: [Int] = []

    // Scratch
    private let hcScratch: FlashNextHyperConnections.Scratch
    private let indexerScratch: FlashNextIndexer.Scratch
    private let indexerCaches: [Int: FlashNextIndexer.LayerCache]
    private let attnScratch: FlashNextAttention.Scratch
    private let kvCaches: [Int: FlashNextAttention.KVCache]

    private let hyper: MTLBuffer
    private let embedRow: MTLBuffer
    private let mixed: MTLBuffer
    private let blockOut: MTLBuffer
    private let moeOut: MTLBuffer
    private let moeActs: MTLBuffer
    private let zeroResidual: MTLBuffer
    private let routerLogits: MTLBuffer
    private let routerExpertScale: MTLBuffer
    private let routerIndices: MTLBuffer
    private let routerWeights: MTLBuffer
    private let sharedGateScratch: MTLBuffer
    private let sharedUpScratch: MTLBuffer
    private let sharedActScratch: MTLBuffer
    private let sharedOut: MTLBuffer
    private let sharedGateScalar: MTLBuffer
    private let gdnQKVRaw: MTLBuffer
    private let gdnConvOut: MTLBuffer
    private let gdnZ: MTLBuffer
    private let gdnA: MTLBuffer
    private let gdnB: MTLBuffer
    private let gdnY: MTLBuffer
    private let gdnOut: MTLBuffer

    private var position = 0
    /// The MoE sub-block's command buffer, committed without a wait so the CPU
    /// can start the next layer's work while it runs. Joined before anything
    /// else touches `hyper`.
    ///
    /// Command buffers on one queue execute in commit order, so this is belt and
    /// braces — but it is the pattern `RealForwardRunner` uses for exactly the
    /// same reason, and a residual-stream race would be silent in the output
    /// rather than loud.
    private var pendingMoECommand: MTLCommandBuffer?
    /// Checked lazily: the LFU slot cache must hold at least `topK` experts or
    /// `PreadExpertStreamer.planExpertsCached` traps rather than degrading.
    private var slotBudgetChecked = false

    public var continuationPosition: Int { position }

    // MARK: - Init

    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        let cfg = model.config
        try Self.validate(config: cfg, maxContext: maxContext)
        self.model = model
        self.ctx = context
        self.cfg = cfg
        self.maxContext = maxContext
        self.hidden = cfg.hiddenSize
        self.bundle = cfg.flashNext.hcCount * cfg.hiddenSize
        self.topK = cfg.topKExperts
        self.numExperts = cfg.numExperts
        self.moeIntermediate = cfg.moeIntermediateSize
        self.sharedIntermediate = cfg.intermediateSize
        self.pleLayer = cfg.flashNext.pleLayerIndices.first ?? -1

        let rotaryDim = Int(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor)
        let theta = Float(cfg.fullRopeTheta)
        let eps = Self.epsilon

        let int4 = try DequantInt4GEMV(context: context,
                                       additionalShapes: cfg.decodeInt4GEMVShapes)
        self.matVec = try FlashNextMatVec(context: context, int4: int4)
        self.rms = try RMSNorm(context: context)
        self.elementwise = try Elementwise(context: context)
        self.hc = try FlashNextHyperConnections(
            context: context, rms: rms, matVec: matVec,
            hidden: hidden, hcCount: cfg.flashNext.hcCount,
            lowRank: cfg.flashNext.hcLowRank, eps: eps)
        self.indexer = try FlashNextIndexer(
            context: context, matVec: matVec,
            geometry: .init(numHeads: cfg.flashNext.indexerNumHeads,
                            numKVHeads: cfg.flashNext.indexerNumKVHeads,
                            headDim: cfg.flashNext.indexerHeadDim,
                            compressRatio: cfg.flashNext.indexerCompressRatio,
                            blockBudget: cfg.flashNext.indexerBlockBudget,
                            rotaryDim: rotaryDim, theta: theta, eps: eps))
        self.attention = FlashNextAttention(
            context: context, matVec: matVec, elementwise: elementwise,
            epilogue: try PrefillQKVEpilogue(context: context),
            attention: try Attention(context: context),
            geometry: .init(hidden: hidden,
                            numHeads: cfg.numHeads,
                            numKVHeads: cfg.numFullKVHeads,
                            headDim: cfg.fullHeadDim,
                            rotaryDim: rotaryDim, theta: theta, eps: eps,
                            scale: 1 / Float(cfg.fullHeadDim).squareRoot()))
        // Which expert compute path the install needs is read from the layout,
        // not assumed: a sub-tensor with no `_scales` companion is dense BF16.
        let expert0 = model.packedExpertsLayout.expert(layer: 0, expert: 0)
        let expertsAreBF16 = expert0.subTensors["gate_scales"] == nil
        if expertsAreBF16 {
            self.moeInt4 = nil
        } else {
            guard MoE.routedComputeWidths.contains(UInt32(cfg.topKExperts)) else {
                throw FlashNextForwardRunnerError.invalidConfiguration(
                    "INT4 routed experts at top-\(cfg.topKExperts); the reduce "
                    + "implements \(MoE.routedComputeWidths.sorted())")
            }
            self.moeInt4 = try MoE(context: context, siluActivation: true,
                                   specializedD: UInt32(hidden),
                                   specializedF: UInt32(moeIntermediate),
                                   specializedNumExperts: UInt32(numExperts),
                                   specializedTopK: UInt32(topK))
        }
        self.moeBF16 = try FlashNextMoE(context: context)
        // The gated norm's activation is SIGMOID for this family
        // (`output_gate_type`), where Qwen 3.6 and Qwen 3.8 use silu. Everything
        // else about the GDN block is the Qwen 3.8 geometry, fused Hv=48 decode
        // kernel included.
        //
        // `GDN.init` preconditions on `key_head_dim % 32 == 0` and
        // `value_head_dim % 4 == 0`; every real install satisfies both (Dk = Dv =
        // 128) and the parity toy (Dk = Dv = 8) satisfies neither. Which path
        // applies is decided from the geometry, once, here.
        let la = cfg.linearAttention
        let shippedGDNApplies = la.keyHeadDim % 32 == 0
            && la.keyHeadDim / 32 <= 8
            && la.valueHeadDim % 4 == 0
        if shippedGDNApplies {
            self.gdn = try GDN(context: context, config: la,
                               specializedHiddenSize: hidden,
                               outputGate: .sigmoid)
            self.gdnState = try GDNStateManager(device: context.device, config: cfg)
            self.genericGDN = nil
            self.genericGDNScratch = nil
            self.genericGDNState = [:]
        } else {
            self.gdn = nil
            self.gdnState = nil
            let generic = try FlashNextGDN(
                context: context,
                geometry: .init(numKHeads: la.numKHeads, numVHeads: la.numVHeads,
                                keyHeadDim: la.keyHeadDim,
                                valueHeadDim: la.valueHeadDim,
                                convKernel: la.convKernelSize, eps: eps))
            var states: [Int: FlashNextGDN.LayerState] = [:]
            for L in 0..<cfg.numLayers where cfg.layerIsLinear(L) {
                states[L] = try generic.makeState(device: context.device)
            }
            self.genericGDN = generic
            self.genericGDNScratch = try generic.makeScratch(
                device: context.device, rows: 1)
            self.genericGDNState = states
        }
        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.embedBF16PSO = try context.pipeline("flashnext_embed_row_bf16")
        self.siluMulPSO = try context.pipeline("silu_mul_fp16")

        self.embedding = model.embedding
        self.embeddingMatrix = FlashNextWeightMatrix.from(model.embedding)
        self.lmHead = FlashNextWeightMatrix.from(model.lmHead)
        let globalNorm = try model.hcGlobalNorm
        self.mixer = FlashNextHyperConnections.Weights(
            norm: globalNorm.buffer,
            normOffset: Int(globalNorm.offset),
            mixDown: .from(try model.hcGlobalMixDown),
            mixUp: .from(try model.hcGlobalMixUp),
            inject: nil)

        // Per-layer tensors, resolved once.
        var built: [LayerTensors] = []
        built.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let isLinear = cfg.layerIsLinear(L)
            func hcWeights(_ site: Model.HyperConnectionSite) throws
                -> FlashNextHyperConnections.Weights {
                let norm = try model.hcNorm(site: site, layer: L)
                return .init(norm: norm.buffer, normOffset: Int(norm.offset),
                             mixDown: .from(try model.hcMixDown(site: site, layer: L)),
                             mixUp: .from(try model.hcMixUp(site: site, layer: L)),
                             inject: .from(try model.hcInject(site: site, layer: L)))
            }
            var gdnTensors: GDNTensors?
            var attnWeights: FlashNextAttention.Weights?
            var indexerTensors: IndexerTensors?
            if isLinear {
                let qkv = try model.linearInProjQKV(layer: L)
                let z = try model.linearInProjZ(layer: L)
                let a = try model.linearInProjA(layer: L)
                let b = try model.linearInProjB(layer: L)
                let allInt4 = [qkv, z, a, b].allSatisfy { $0.dtype == 0 }
                gdnTensors = GDNTensors(
                    qkv: .from(qkv), z: .from(z), a: .from(a), b: .from(b),
                    out: .from(try model.linearOutProj(layer: L)),
                    conv: try model.linearConv1d(layer: L),
                    aLog: try model.linearALog(layer: L),
                    dtBias: try model.linearDtBias(layer: L),
                    // `linear_attn.norm` is ones-initialized, not zero-centered:
                    // the `(1 + w)` bake must NOT be applied to it.
                    norm: try model.linearNorm(layer: L),
                    fusedInProj: allInt4 ? (qkv, z, a, b) : nil)
            } else {
                let qNorm = try model.normWeight(
                    name: "\(model.trunkPrefix)layers.\(L).self_attn.q_norm.weight")
                let kNorm = try model.normWeight(
                    name: "\(model.trunkPrefix)layers.\(L).self_attn.k_norm.weight")
                attnWeights = .init(q: .from(try model.qProj(layer: L)),
                                    k: .from(try model.kProj(layer: L)),
                                    v: .from(try model.vProj(layer: L)),
                                    o: .from(try model.oProj(layer: L)),
                                    qNorm: qNorm.buffer,
                                    qNormOffset: Int(qNorm.offset),
                                    kNorm: kNorm.buffer,
                                    kNormOffset: Int(kNorm.offset))
                indexerTensors = IndexerTensors(
                    qkProj: .from(try model.indexerQKProj(layer: L)),
                    qNorm: try model.indexerQNorm(layer: L),
                    kNorm: try model.indexerKNorm(layer: L))
            }
            let entry = model.packedExpertsLayout.expert(layer: L, expert: 0)
            built.append(LayerTensors(
                attnHC: try hcWeights(.attention),
                mlpHC: try hcWeights(.mlp),
                isLinear: isLinear,
                gdn: gdnTensors,
                attention: attnWeights,
                indexer: indexerTensors,
                moe: MoETensors(
                    router: .from(try model.router(layer: L)),
                    sharedGate: .from(try model.sharedExpertScalarGate(layer: L)),
                    sharedGateProj: .from(try model.sharedExpertGate(layer: L)),
                    sharedUp: .from(try model.sharedExpertUp(layer: L)),
                    sharedDown: .from(try model.sharedExpertDown(layer: L)),
                    expertOffsets: model.routedExpertOffsets(layer: L),
                    expertsAreBF16: entry.subTensors["gate_scales"] == nil)))
        }
        self.layers = built

        // PLE.
        if pleLayer >= 0 {
            let p = try FlashNextPLE(
                context: context, rms: rms, matVec: matVec, elementwise: elementwise,
                hidden: hidden, hcCount: cfg.flashNext.hcCount,
                convKernel: cfg.flashNext.pleConvKernelSize,
                dilation: try model.pleNgramSize(layer: pleLayer),
                eps: eps)
            let normConv = try model.pleNormConv(layer: pleLayer)
            let normKey = try model.pleNormKey(layer: pleLayer)
            let normQuery = try model.pleNormQuery(layer: pleLayer)
            let conv = try model.pleConv1d(layer: pleLayer)
            self.ple = p
            self.pleWeights = .init(
                keyProj: .from(try model.pleKeyProj(layer: pleLayer)),
                valueProj: .from(try model.pleValueProj(layer: pleLayer)),
                conv: conv.buffer, convOffset: Int(conv.offset),
                normKey: normKey.buffer, normKeyOffset: Int(normKey.offset),
                normQuery: normQuery.buffer, normQueryOffset: Int(normQuery.offset),
                normConv: normConv.buffer, normConvOffset: Int(normConv.offset))
            self.pleScratch = try p.makeScratch(device: context.device, rows: 1)
            self.pleHash = FlashNextPleHash(
                multipliers: try model.pleLayerMultipliers(layer: pleLayer),
                headOffsets: try model.pleNgramHeadOffsets(layer: pleLayer),
                headVocabSizes: try model.pleNgramHeadVocabSizes(layer: pleLayer),
                eosTokenID: cfg.flashNext.pleEosTokenID)
            self.pleRowPool = try model.openPleRowPool(layer: pleLayer)
            guard let staging = context.device.makeBuffer(
                length: hidden * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            self.pleStaging = staging
        } else {
            self.ple = nil
            self.pleWeights = nil
            self.pleScratch = nil
            self.pleHash = nil
            self.pleRowPool = nil
            self.pleStaging = nil
        }

        // Scratch. Every buffer is one token wide: prefill is sequential.
        self.hcScratch = try hc.makeScratch(device: context.device, rows: 1)
        self.indexerScratch = try indexer.makeScratch(device: context.device,
                                                      rows: 1, maxTokens: maxContext)
        self.attnScratch = try attention.makeScratch(
            device: context.device, rows: 1,
            maxSelected: indexer.maxSelected, gatherSlots: 1)
        var indexCaches: [Int: FlashNextIndexer.LayerCache] = [:]
        var kv: [Int: FlashNextAttention.KVCache] = [:]
        for L in 0..<cfg.numLayers where !cfg.layerIsLinear(L) {
            indexCaches[L] = try indexer.makeLayerCache(device: context.device,
                                                        maxTokens: maxContext)
            kv[L] = try attention.makeKVCache(device: context.device,
                                              maxTokens: maxContext)
        }
        self.indexerCaches = indexCaches
        self.kvCaches = kv

        let device = context.device
        func half(_ count: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(
                length: max(1, count) * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else { throw MetalError.noDevice }
            return b
        }
        func float(_ count: Int, shared: Bool = true) throws -> MTLBuffer {
            guard let b = device.makeBuffer(
                length: max(1, count) * MemoryLayout<Float>.stride,
                options: shared ? .storageModeShared : .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return b
        }
        self.hyper = try half(bundle)
        self.embedRow = try half(hidden)
        self.mixed = try half(hidden)
        self.blockOut = try half(hidden)
        self.moeOut = try half(hidden)
        self.moeActs = try half(topK * moeIntermediate)
        self.zeroResidual = try half(hidden)
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.routerLogits = try float(numExperts, shared: false)
        guard let scale = device.makeBuffer(
                bytes: [UInt16](repeating: Quantization.bf16Bits(1),
                                count: numExperts),
                length: numExperts * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
              let indices = device.makeBuffer(
                length: topK * MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else { throw MetalError.noDevice }
        self.routerExpertScale = scale
        self.routerIndices = indices
        self.routerWeights = try half(topK)
        self.sharedGateScratch = try half(sharedIntermediate)
        self.sharedUpScratch = try half(sharedIntermediate)
        self.sharedActScratch = try half(sharedIntermediate)
        self.sharedOut = try half(hidden)
        self.sharedGateScalar = try float(1)
        self.gdnQKVRaw = try half(la.qkvDim)
        self.gdnConvOut = try half(la.qkvDim)
        self.gdnZ = try half(la.valueDim)
        self.gdnA = try half(la.numVHeads)
        self.gdnB = try half(la.numVHeads)
        self.gdnY = try half(la.valueDim)
        self.gdnOut = try half(la.valueDim)

        reset()
    }

    private static func validate(config: ArchConfig, maxContext: Int) throws {
        guard config.family == .qwen38flashnext else {
            throw FlashNextForwardRunnerError.invalidConfiguration(
                "FlashNextForwardRunner requires the qwen38flashnext family")
        }
        guard config.hasLowRankHyperConnections else {
            throw FlashNextForwardRunnerError.invalidConfiguration(
                "this family's residual stream is hc_count x hidden; hcCount is 0")
        }
        guard config.numExperts > 0, config.topKExperts > 0,
              config.topKExperts <= MoE.routedBlobSlots else {
            throw FlashNextForwardRunnerError.invalidConfiguration(
                "routed expert compute supports top-k up to \(MoE.routedBlobSlots)")
        }
        guard config.hasLinearAttentionLayers else {
            throw FlashNextForwardRunnerError.invalidConfiguration(
                "this family is a 3:1 GDN/attention hybrid; the layer mask has no "
                + "linear-attention layers")
        }
        guard config.attnOutputGate, config.ropeNeoxSubdim, config.sharedExpertGated
        else {
            throw FlashNextForwardRunnerError.invalidConfiguration(
                "expected a gated attention output, NeoX sub-dim RoPE and a gated "
                + "shared expert")
        }
        guard maxContext > 0 else {
            throw FlashNextForwardRunnerError.invalidConfiguration(
                "maxContext must be positive")
        }
    }

    // MARK: - Lifecycle

    public func reset() {
        position = 0
        try? joinPendingMoE()
        gdnState?.reset()
        if let generic = genericGDN, let cb = ctx.queue.makeCommandBuffer() {
            for state in genericGDNState.values {
                generic.encodeReset(commandBuffer: cb, state: state)
            }
            cb.commit()
            cb.waitUntilCompleted()
        }
        pleHistory = pleHash?.initialHistory() ?? []
        if let ple, let pleScratch, let cb = ctx.queue.makeCommandBuffer() {
            ple.encodeResetState(commandBuffer: cb, scratch: pleScratch)
            cb.commit()
            cb.waitUntilCompleted()
        }
        // The indexer and KV caches are append-only and addressed by absolute
        // position, so rewinding the cursor is what clears them.
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition == position else {
            throw FlashNextForwardRunnerError.invalidInput(
                "continuation expects position \(expectedPosition) but the runner "
                + "is at \(position)")
        }
    }

    // MARK: - Entry points

    public func produce(token: Int32, position p: Int,
                        into logits: MTLBuffer) async throws {
        try await produceToken(token: token, position: p, into: logits)
    }

    func produceWithoutLogits(token: Int32, position p: Int) async throws {
        try await produceToken(token: token, position: p, into: nil)
    }

    // MARK: - The forward pass

    private func produceToken(token: Int32, position p: Int,
                              into logits: MTLBuffer?) async throws {
        guard p == position else {
            throw FlashNextForwardRunnerError.invalidInput(
                "expected position \(position), got \(p)")
        }
        guard p < maxContext else {
            throw FlashNextForwardRunnerError.invalidInput(
                "position \(p) exceeds maxContext \(maxContext)")
        }
        guard token >= 0, Int(token) < cfg.vocabSize else {
            throw FlashNextForwardRunnerError.invalidInput(
                "token \(token) outside the \(cfg.vocabSize)-entry vocabulary")
        }
        try Task.checkCancellation()

        // Embed, then tile across the four streams. `repeat(1, 1, hc_count)` is a
        // TILE — stream j is an exact copy of the row, not an interleave — and
        // there is no sqrt(hidden) scaling.
        guard var head = ctx.queue.makeCommandBuffer() else {
            throw FlashNextForwardRunnerError.commandFailed("no command buffer")
        }
        encodeEmbedRow(commandBuffer: head, token: UInt32(token))
        try captureFloats(&head, "embed_out", embedRow, count: hidden)
        hc.encodeTileEmbedding(commandBuffer: head, embedding: embedRow,
                               hyper: hyper, rows: 1)
        head.commit()

        for L in 0..<cfg.numLayers {
            try Task.checkCancellation()
            try await encodeLayer(L, token: Int(token), position: p)
        }

        try joinPendingMoE()
        guard let tail = ctx.queue.makeCommandBuffer() else {
            throw FlashNextForwardRunnerError.commandFailed("no command buffer")
        }
        // No final norm: the global mixer stands in its place, and `lm_head`
        // applies straight to its 2560-wide output.
        var tailCB = tail
        hc.encodeMix(commandBuffer: tailCB, weights: mixer, scratch: hcScratch,
                     hyper: hyper, mixed: mixed, rows: 1)
        try captureFloats(&tailCB, "last_hidden_state", mixed, count: hidden)
        if let logits {
            matVec.encode(commandBuffer: tailCB, matrix: lmHead,
                          x: mixed, y: logits,
                          rows: cfg.vocabSize, cols: hidden)
        }
        try finish(tailCB)
        if capture != nil, let logits {
            capture?.floats["logits"] = Self.readFP16(logits, count: cfg.vocabSize)
        }
        position += 1
    }

    private func encodeLayer(_ L: Int, token: Int, position p: Int) async throws {
        try joinPendingMoE()
        let layer = layers[L]
        let key = String(format: "layer%02d.", L)

        if L == pleLayer { try encodePLE(token: token, key: key) }

        // --- Attention / GDN sub-block -------------------------------------
        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw FlashNextForwardRunnerError.commandFailed("no command buffer")
        }
        try captureFloats(&cb, key + "attn_hc_stream_in", hyper, count: bundle)
        hc.encodeMix(commandBuffer: cb, weights: layer.attnHC, scratch: hcScratch,
                     hyper: hyper, mixed: mixed, rows: 1)
        hc.encodeInjectGate(commandBuffer: cb, weights: layer.attnHC,
                            scratch: hcScratch, rows: 1)
        try captureFloats(&cb, key + "attn_hc_mixed", mixed, count: hidden)
        if layer.isLinear {
            encodeGDN(commandBuffer: cb, layer: layer, index: L)
        } else {
            guard let idx = layer.indexer, let cache = indexerCaches[L] else {
                throw FlashNextForwardRunnerError.invalidConfiguration(
                    "layer \(L) is a full-attention layer with no indexer")
            }
            indexer.encodeProjection(commandBuffer: cb, weight: idx.qkProj,
                                     x: mixed, xOffset: 0, hidden: hidden,
                                     scratch: indexerScratch, rows: 1)
            indexer.encodePrepare(commandBuffer: cb,
                                  qNorm: idx.qNorm.buffer,
                                  qNormOffset: Int(idx.qNorm.offset),
                                  kNorm: idx.kNorm.buffer,
                                  kNormOffset: Int(idx.kNorm.offset),
                                  scratch: indexerScratch, cache: cache,
                                  rows: 1, startPosition: p)
            indexer.encodeScores(commandBuffer: cb, scratch: indexerScratch,
                                 cache: cache, rows: 1, startPosition: p)
            try finish(cb)

            // Selection is CPU work by design: exact `torch.topk` ordering over
            // a few thousand FP32 scores, read back while the attention it gates
            // is still much larger.
            let selected = indexer.selections(scratch: indexerScratch, rows: 1,
                                              startPosition: p)[0]
            if capture != nil {
                capture?.integers[key + "indexer_selected"] = [selected]
                capture?.integers[key + "indexer_visible"] = [Array(0...p)]
            }
            guard let attnWeights = layer.attention, let kv = kvCaches[L] else {
                throw FlashNextForwardRunnerError.invalidConfiguration(
                    "layer \(L) is a full-attention layer with no projections")
            }
            guard let next = ctx.queue.makeCommandBuffer() else {
                throw FlashNextForwardRunnerError.commandFailed("no command buffer")
            }
            cb = next
            attention.encodeProjectAndCache(
                commandBuffer: cb, weights: attnWeights, scratch: attnScratch,
                cache: kv, x: mixed, xOffset: 0, rows: 1, startPosition: p)
            let count = indexer.writeSelection(selected, row: 0,
                                               into: indexerScratch)
            indexer.encodeGatherKV(
                commandBuffer: cb,
                kCache: kv.keys, kCacheOffset: 0,
                vCache: kv.values, vCacheOffset: 0,
                scratch: indexerScratch, selectionRow: 0,
                kOut: attnScratch.gatheredK, kOutOffset: 0,
                vOut: attnScratch.gatheredV, vOutOffset: 0,
                kvDim: attention.geometry.kvDim, count: count)
            attention.encodeAttendRow(commandBuffer: cb, scratch: attnScratch,
                                      row: 0, slot: 0, selectedCount: count)
            attention.encodeGateAndProject(commandBuffer: cb, weights: attnWeights,
                                           scratch: attnScratch,
                                           out: blockOut, outOffset: 0, rows: 1)
        }
        try captureFloats(&cb, key + "block_out", blockOut, count: hidden)
        hc.encodeInjectAccumulate(commandBuffer: cb, scratch: hcScratch,
                                  hyper: hyper, block: blockOut, rows: 1)

        // --- MoE sub-block --------------------------------------------------
        try captureFloats(&cb, key + "mlp_hc_stream_in", hyper, count: bundle)
        hc.encodeMix(commandBuffer: cb, weights: layer.mlpHC, scratch: hcScratch,
                     hyper: hyper, mixed: mixed, rows: 1)
        hc.encodeInjectGate(commandBuffer: cb, weights: layer.mlpHC,
                            scratch: hcScratch, rows: 1)
        try captureFloats(&cb, key + "mlp_hc_mixed", mixed, count: hidden)
        matVec.encode(commandBuffer: cb, matrix: layer.moe.router,
                      x: mixed, y: routerLogits,
                      rows: numExperts, cols: hidden, outputFloat32: true)
        moeBF16.encodeRouterSelect(commandBuffer: cb, logits: routerLogits,
                                   perExpertScale: routerExpertScale,
                                   outIndices: routerIndices,
                                   outWeights: routerWeights,
                                   numExperts: UInt32(numExperts))
        // The shared expert reads the same `mixed` and does not depend on the
        // routing, so it rides in this command buffer rather than waiting for
        // the expert blobs.
        encodeSharedExpert(commandBuffer: cb, layer: layer)
        try finish(cb)

        let indexPointer = routerIndices.contents()
            .bindMemory(to: UInt32.self, capacity: topK)
        let experts = (0..<topK).map { min(Int(indexPointer[$0]), numExperts - 1) }
        if capture != nil {
            capture?.integers[key + "router_indices"] = [experts]
            capture?.floats[key + "router_weights"] =
                Self.readFP16(routerWeights, count: topK)
        }
        try checkSlotBudget(layer: L)
        let blobs = try await fetchExperts(layer: L, experts: experts)

        guard var moeCB = ctx.queue.makeCommandBuffer() else {
            throw FlashNextForwardRunnerError.commandFailed("no command buffer")
        }
        encodeRoutedExperts(commandBuffer: moeCB, layer: layer, blobs: blobs)
        // `out = sum_rank w_rank * expert_rank(x)` then `+ sigmoid(g) * shared`,
        // the reference's order: the routed reduce runs against a zero residual
        // and the gated shared expert is added afterwards.
        elementwise.encodeResidualAdd(commandBuffer: moeCB, hidden: moeOut,
                                      delta: sharedOut, count: hidden)
        try captureFloats(&moeCB, key + "moe_out", moeOut, count: hidden)
        hc.encodeInjectAccumulate(commandBuffer: moeCB, scratch: hcScratch,
                                  hyper: hyper, block: moeOut, rows: 1)
        moeCB.commit()
        pendingMoECommand = moeCB
        try captureAfterDrain(key + "stream_out", hyper, count: bundle)
    }

    /// Wait for the previous layer's MoE command buffer, if one is still in
    /// flight, and surface its error rather than letting it vanish.
    private func joinPendingMoE() throws {
        guard let pending = pendingMoECommand else { return }
        pendingMoECommand = nil
        pending.waitUntilCompleted()
        if let error = pending.error {
            throw FlashNextForwardRunnerError.commandFailed("\(error)")
        }
    }

    // MARK: - Blocks

    private func encodeEmbedRow(commandBuffer cb: MTLCommandBuffer, token: UInt32) {
        switch embeddingMatrix {
        case .int4:
            embedInt4.encode(commandBuffer: cb,
                             table: embedding.buffer,
                             tableOffset: Int(embedding.offset),
                             scales: embedding.buffer,
                             scalesOffset: Int(embedding.scaleOffset),
                             biases: embedding.buffer,
                             biasesOffset: Int(embedding.biasOffset),
                             out: embedRow, tokenId: token,
                             d: UInt32(hidden), outScale: 1.0)
        case let .bf16(buffer, offset):
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(embedBF16PSO)
            enc.setBuffer(buffer, offset: offset, index: 0)
            enc.setBuffer(embedRow, offset: 0, index: 1)
            var row = token
            var d = UInt32(hidden)
            enc.setBytes(&row, length: MemoryLayout<UInt32>.size, index: 2)
            enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
            let width = min(Int(embedBF16PSO.maxTotalThreadsPerThreadgroup), 256)
            enc.dispatchThreads(MTLSize(width: hidden, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(width, hidden),
                                                               height: 1, depth: 1))
            enc.endEncoding()
        }
    }

    /// The n-gram hash and the 16 row reads are CPU work by design: the table is
    /// 102 GB on disk and the reads go through `PleRowPool`'s LFU cache. What
    /// lands on the GPU is the mixing.
    private func encodePLE(token: Int, key: String) throws {
        guard let ple, let weights = pleWeights, let scratch = pleScratch,
              let hash = pleHash, let pool = pleRowPool, let staging = pleStaging
        else { return }
        // The decode path reads the cached history BEFORE overwriting it, and the
        // EOS-aware shift runs over that window only.
        let window = pleHistory + [token]
        guard let rowIDs = hash.rowIDs(window: window).last else { return }
        if capture != nil { capture?.integers[key + "ple_ngram_row_ids"] = [rowIDs] }
        let embedding = try pool.readEmbedding(rows: rowIDs)
        precondition(embedding.count == hidden,
                     "PLE gather produced \(embedding.count) values, expected \(hidden)")
        let base = staging.contents().bindMemory(to: Float16.self, capacity: hidden)
        for i in 0..<hidden { base[i] = Float16(embedding[i]) }
        pleHistory = Array(window.suffix(hash.historyLength))

        guard let cb = ctx.queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else {
            throw FlashNextForwardRunnerError.commandFailed("no command buffer")
        }
        blit.copy(from: staging, sourceOffset: 0,
                  to: scratch.embeds, destinationOffset: 0,
                  size: hidden * MemoryLayout<Float16>.stride)
        blit.endEncoding()
        ple.encode(commandBuffer: cb, weights: weights, scratch: scratch,
                   hyper: hyper, rows: 1)
        cb.commit()
    }

    private func encodeGDN(commandBuffer cb: MTLCommandBuffer,
                           layer: LayerTensors, index L: Int) {
        guard let w = layer.gdn else { return }
        let la = cfg.linearAttention
        encodeGDNInputProjections(commandBuffer: cb, w: w, la: la)
        if let gdn, let gdnState {
            encodeShippedGDN(commandBuffer: cb, w: w, gdn: gdn, state: gdnState,
                             index: L)
        } else if let generic = genericGDN, let scratch = genericGDNScratch,
                  let state = genericGDNState[L] {
            generic.encodeConv(commandBuffer: cb, qkv: gdnQKVRaw,
                               convWeight: w.conv.buffer,
                               convWeightOffset: Int(w.conv.offset),
                               scratch: scratch, state: state, rows: 1)
            generic.encodeRecurrence(commandBuffer: cb, scratch: scratch,
                                     state: state, z: gdnZ, a: gdnA, b: gdnB,
                                     aLog: w.aLog.buffer,
                                     aLogOffset: Int(w.aLog.offset),
                                     dtBias: w.dtBias.buffer,
                                     dtBiasOffset: Int(w.dtBias.offset),
                                     normWeight: w.norm.buffer,
                                     normWeightOffset: Int(w.norm.offset),
                                     out: gdnOut)
        }
        matVec.encode(commandBuffer: cb, matrix: w.out, x: gdnOut, y: blockOut,
                      rows: hidden, cols: la.valueDim)
    }

    private func encodeGDNInputProjections(commandBuffer cb: MTLCommandBuffer,
                                           w: GDNTensors,
                                           la: LinearAttentionConfig) {
        if let gdn, let fused = w.fusedInProj {
            gdn.encodeInputProjections(commandBuffer: cb, x: mixed,
                                       qkv: fused.qkv, qkvOut: gdnQKVRaw,
                                       z: fused.z, zOut: gdnZ,
                                       a: fused.a, aOut: gdnA,
                                       b: fused.b, bOut: gdnB,
                                       hiddenSize: hidden)
        } else {
            matVec.encode(commandBuffer: cb, matrix: w.qkv, x: mixed, y: gdnQKVRaw,
                          rows: la.qkvDim, cols: hidden)
            matVec.encode(commandBuffer: cb, matrix: w.z, x: mixed, y: gdnZ,
                          rows: la.valueDim, cols: hidden)
            matVec.encode(commandBuffer: cb, matrix: w.a, x: mixed, y: gdnA,
                          rows: la.numVHeads, cols: hidden)
            matVec.encode(commandBuffer: cb, matrix: w.b, x: mixed, y: gdnB,
                          rows: la.numVHeads, cols: hidden)
        }
    }

    private func encodeShippedGDN(commandBuffer cb: MTLCommandBuffer,
                                  w: GDNTensors, gdn: GDN,
                                  state gdnState: GDNStateManager, index L: Int) {
        gdn.encodeConvDecode(commandBuffer: cb,
                             tail: gdnState.convTailBuffer(layer: L),
                             qkv: gdnQKVRaw,
                             convWeight: w.conv.buffer,
                             convWeightOffset: Int(w.conv.offset),
                             out: gdnConvOut)
        gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        let fusedDecode = gdn.encodeDeltaGatedDecode(
            commandBuffer: cb,
            convOut: gdnConvOut, aProj: gdnA, bProj: gdnB,
            aLog: w.aLog.buffer, aLogOffset: Int(w.aLog.offset),
            dtBias: w.dtBias.buffer, dtBiasOffset: Int(w.dtBias.offset),
            state: gdnState.stateBuffer(layer: L),
            z: gdnZ,
            weight: w.norm.buffer, weightOffset: Int(w.norm.offset),
            out: gdnOut)
        if !fusedDecode {
            gdn.encodeDeltaStepDecode(commandBuffer: cb,
                                      convOut: gdnConvOut, aProj: gdnA, bProj: gdnB,
                                      aLog: w.aLog.buffer,
                                      aLogOffset: Int(w.aLog.offset),
                                      dtBias: w.dtBias.buffer,
                                      dtBiasOffset: Int(w.dtBias.offset),
                                      state: gdnState.stateBuffer(layer: L),
                                      y: gdnY)
            gdn.encodeGatedNorm(commandBuffer: cb, y: gdnY, z: gdnZ,
                                weight: w.norm.buffer,
                                weightOffset: Int(w.norm.offset),
                                out: gdnOut)
        }
    }

    /// A plain SwiGLU MLP at `intermediate_size`, scaled by
    /// `sigmoid(shared_expert_gate . x)`. The gate is a single `[1, hidden]` row,
    /// kept FP32 to the sigmoid.
    private func encodeSharedExpert(commandBuffer cb: MTLCommandBuffer,
                                    layer: LayerTensors) {
        matVec.encode(commandBuffer: cb, matrix: layer.moe.sharedGateProj,
                      x: mixed, y: sharedGateScratch,
                      rows: sharedIntermediate, cols: hidden)
        matVec.encode(commandBuffer: cb, matrix: layer.moe.sharedUp,
                      x: mixed, y: sharedUpScratch,
                      rows: sharedIntermediate, cols: hidden)
        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(siluMulPSO)
            enc.setBuffer(sharedGateScratch, offset: 0, index: 0)
            enc.setBuffer(sharedUpScratch, offset: 0, index: 1)
            enc.setBuffer(sharedActScratch, offset: 0, index: 2)
            var count = UInt32(sharedIntermediate)
            enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            let width = min(Int(siluMulPSO.maxTotalThreadsPerThreadgroup), 256)
            enc.dispatchThreads(
                MTLSize(width: sharedIntermediate, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(width, sharedIntermediate),
                                               height: 1, depth: 1))
            enc.endEncoding()
        }
        matVec.encode(commandBuffer: cb, matrix: layer.moe.sharedDown,
                      x: sharedActScratch, y: sharedOut,
                      rows: hidden, cols: sharedIntermediate)
        matVec.encode(commandBuffer: cb, matrix: layer.moe.sharedGate,
                      x: mixed, y: sharedGateScalar,
                      rows: 1, cols: hidden, outputFloat32: true)
        moeBF16.encodeSharedGateScale(commandBuffer: cb, out: sharedOut,
                                      scalar: sharedGateScalar, count: hidden)
    }

    private func encodeRoutedExperts(commandBuffer cb: MTLCommandBuffer,
                                     layer: LayerTensors,
                                     blobs: [(buffer: MTLBuffer, offset: Int)]) {
        if layer.moe.expertsAreBF16 {
            let argBuffer = moeBF16.makeRoutedArgumentBuffer(routedBlobs: blobs)
            moeBF16.encodePhase1(commandBuffer: cb, routedArgBuffer: argBuffer,
                                 routedBlobs: blobs,
                                 routedOffsets: layer.moe.expertOffsets,
                                 x: mixed, acts: moeActs,
                                 d: UInt32(hidden), f: UInt32(moeIntermediate),
                                 topK: UInt32(topK))
            moeBF16.encodePhase2(commandBuffer: cb, routedArgBuffer: argBuffer,
                                 routedBlobs: blobs,
                                 routedOffsets: layer.moe.expertOffsets,
                                 acts: moeActs, routingWeights: routerWeights,
                                 residual: zeroResidual, y: moeOut,
                                 d: UInt32(hidden), f: UInt32(moeIntermediate),
                                 topK: UInt32(topK))
        } else {
            guard let moeInt4 else {
                preconditionFailure(
                    "layer stores INT4 experts but the INT4 path was not built")
            }
            let argBuffer = moeInt4.makeReusedRoutedArgumentBuffer(
                routedBlobs: blobs, topK: UInt32(topK))
            moeInt4.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: blobs,
                routedOffsets: layer.moe.expertOffsets, x: mixed, acts: moeActs,
                d: UInt32(hidden), f: UInt32(moeIntermediate), topK: UInt32(topK))
            moeInt4.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: blobs,
                routedOffsets: layer.moe.expertOffsets, acts: moeActs,
                routingWeights: routerWeights, residual: zeroResidual, y: moeOut,
                d: UInt32(hidden), f: UInt32(moeIntermediate), topK: UInt32(topK))
        }
    }

    // MARK: - Expert streaming

    /// The LFU slot cache must hold at least `topK` experts: the planner
    /// preconditions on it rather than degrading, and 8 is still an offered rung.
    private func checkSlotBudget(layer L: Int) throws {
        guard !slotBudgetChecked else { return }
        slotBudgetChecked = true
        guard let slots = model.routedExpertCacheSlotCount(layer: L) else { return }
        guard slots >= topK else {
            throw FlashNextForwardRunnerError.invalidConfiguration(
                "this family routes top-\(topK) experts per layer but the expert "
                + "cache has only \(slots) slots; use --expert-cache-slots 16 or more")
        }
    }

    private func fetchExperts(layer L: Int, experts: [Int]) async throws
        -> [(buffer: MTLBuffer, offset: Int)] {
        let views: [TensorView]
        if let plan = try model.planRoutedExpertsIfPossible(layer: L,
                                                            experts: experts) {
            views = try await model.fetchRoutedExperts(plan: plan)
        } else {
            views = try await model.fetchRoutedExperts(layer: L, experts: experts)
        }
        return views.map { (buffer: $0.buffer, offset: Int($0.offset)) }
    }

    // MARK: - Capture helpers

    /// Commit `cb`, read `buffer` into the capture under `key`, and hand back a
    /// fresh command buffer. A no-op when capture is off, which is why the
    /// forward pass can call it unconditionally.
    private func captureFloats(_ cb: inout MTLCommandBuffer, _ key: String,
                               _ buffer: MTLBuffer, count: Int) throws {
        guard capture != nil else { return }
        try finish(cb)
        capture?.floats[key] = Self.readFP16(buffer, count: count)
        guard let next = ctx.queue.makeCommandBuffer() else {
            throw FlashNextForwardRunnerError.commandFailed("no command buffer")
        }
        cb = next
    }

    /// Drain the queue — command buffers on one queue execute in commit order,
    /// so an empty one completing means every earlier one has — then read.
    private func captureAfterDrain(_ key: String, _ buffer: MTLBuffer,
                                   count: Int) throws {
        guard capture != nil else { return }
        guard let drain = ctx.queue.makeCommandBuffer() else {
            throw FlashNextForwardRunnerError.commandFailed("no command buffer")
        }
        try finish(drain)
        capture?.floats[key] = Self.readFP16(buffer, count: count)
    }

    static func readFP16(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let base = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(base[$0]) }
    }

    // MARK: - Command helpers

    private func finish(_ cb: MTLCommandBuffer) throws {
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            throw FlashNextForwardRunnerError.commandFailed("\(error)")
        }
        guard cb.status == .completed else {
            throw FlashNextForwardRunnerError.commandFailed(
                "command buffer status \(cb.status.rawValue)")
        }
    }
}
