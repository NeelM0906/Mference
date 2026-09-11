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
/// Reached through `ForwardRunnerFactory.make` for every caller: the family's
/// capability gate was lifted on 2026-09-10 (owner decision), so
/// `ManifestReader.familiesWithoutRunner` no longer lists it and the CLI and the
/// loopback server (and therefore the UI behind it) load this family like any
/// other.
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
/// # Every install dtype
///
/// Every projection goes through `FlashNextWeightMatrix`, which carries the
/// stored dtype with the buffer: affine group-64 for the production install —
/// INT4 for everything but the two MoE gating tensors, which the install policy
/// keeps at INT8 — and dense BF16 for the parity install (whose
/// `moe_intermediate_size` of 32 group-64 cannot quantize at all). The routed
/// experts, the embedding and `lm_head` split the same way. Nothing in this file
/// assumes a quantization, and in particular nothing assumes a *uniform* one:
/// each tensor's width comes from its own resident index entry, so the
/// uniform-INT4 install predating the router change loads unchanged.
///
/// # Prefill
///
/// Production prefill is chunked and layer-major. Resident projections operate
/// over the token block, GDN carries its exact recurrence inside the prefill
/// kernel, QSA ranks all rows at one layer boundary, and routed experts are
/// grouped expert-major so each unique blob is fetched once per chunk-layer.
/// `produceWithoutLogits` remains only as the explicit sequential reference
/// seam used by parity/debug callers.
///
/// PERF, not correctness: the layer loop still takes two CPU round trips per
/// attention layer (indexer scores out, selection in). Resident routed experts
/// stay GPU-directed; the bounded-memory backend still reads route ids to plan
/// cache fills. Device-side indexer selection and bounded-mode I/O overlap are
/// the remaining synchronization work.
///
/// `MFERENCE_PHASES=1` splits a decode window into the four costs that pass
/// touches — expert I/O (all of it exposed, since nothing overlaps it yet), the
/// indexer's CPU top-k, the PLE row-pool gather, and GPU busy versus span. See
/// the counters under "Phase counters" below and the report in
/// `MferenceCLI/Run.swift`. Unset, none of it is wired and every counter is
/// zero.
public final class FlashNextForwardRunner: ContinuableLogitProducer,
                                           ContextWindowReporting,
                                           ChunkedPrefillRunner,
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

    private final class PrefillScratch {
        let chunkTokens: Int
        let hc: FlashNextHyperConnections.Scratch
        let indexer: FlashNextIndexer.Scratch
        let attention: FlashNextAttention.Scratch
        let ple: FlashNextPLE.Scratch?
        let genericGDN: FlashNextGDN.Scratch?
        let tokens: MTLBuffer
        let pleStaging: MTLBuffer?
        let hyper: MTLBuffer
        let embed: MTLBuffer
        let mixed: MTLBuffer
        let blockOut: MTLBuffer
        let moeOut: MTLBuffer
        let routerLogits: MTLBuffer
        let routeIDs: MTLBuffer
        let routeWeights: MTLBuffer
        let routePartials: MTLBuffer
        let routedGateUpAct: MTLBuffer
        let routedDown: MTLBuffer
        let routedMatrixAct: MTLBuffer
        let sharedGate: MTLBuffer
        let sharedUp: MTLBuffer
        let sharedAct: MTLBuffer
        let sharedOut: MTLBuffer
        let sharedScalar: MTLBuffer
        let gdnQKV: MTLBuffer
        let gdnConvOut: MTLBuffer
        let gdnZ: MTLBuffer
        let gdnA: MTLBuffer
        let gdnB: MTLBuffer
        let gdnY: MTLBuffer
        let gdnOut: MTLBuffer

        init(device: MTLDevice, chunkTokens: Int, maxContext: Int,
             cfg: ArchConfig, bundle: Int,
             hcEncoder: FlashNextHyperConnections,
             indexerEncoder: FlashNextIndexer,
             attentionEncoder: FlashNextAttention,
             pleEncoder: FlashNextPLE?,
             genericGDNEncoder: FlashNextGDN?) throws {
            self.chunkTokens = chunkTokens
            self.hc = try hcEncoder.makeScratch(device: device, rows: chunkTokens)
            self.indexer = try indexerEncoder.makeScratch(
                device: device, rows: chunkTokens, maxTokens: maxContext)
            self.attention = try attentionEncoder.makeScratch(
                device: device, rows: chunkTokens,
                maxSelected: indexerEncoder.maxSelected, gatherSlots: 1)
            self.ple = try pleEncoder?.makeScratch(device: device, rows: chunkTokens)
            self.genericGDN = try genericGDNEncoder?.makeScratch(
                device: device, rows: chunkTokens)

            func make(_ elements: Int, stride: Int,
                      mode: MTLResourceOptions = .storageModePrivate,
                      label: String) throws -> MTLBuffer {
                guard let value = device.makeBuffer(
                    length: max(1, elements) * stride, options: mode) else {
                    throw FlashNextForwardRunnerError.commandFailed(
                        "unable to allocate Flash-Next prefill \(label)")
                }
                value.label = label
                return value
            }
            let h = MemoryLayout<Float16>.stride
            let f = MemoryLayout<Float>.stride
            let u = MemoryLayout<UInt32>.stride
            let d = cfg.hiddenSize
            let la = cfg.linearAttention
            let pairRows = min(256, chunkTokens * cfg.topKExperts)
            self.tokens = try make(chunkTokens, stride: u, mode: .storageModeShared,
                                   label: "flashnext.prefill.tokens")
            self.pleStaging = pleEncoder == nil ? nil : try make(
                chunkTokens * d, stride: h, mode: .storageModeShared,
                label: "flashnext.prefill.pleStaging")
            self.hyper = try make(chunkTokens * bundle, stride: h,
                                  label: "flashnext.prefill.hyper")
            self.embed = try make(chunkTokens * d, stride: h,
                                  label: "flashnext.prefill.embed")
            self.mixed = try make(chunkTokens * d, stride: h,
                                  label: "flashnext.prefill.mixed")
            self.blockOut = try make(chunkTokens * d, stride: h,
                                     label: "flashnext.prefill.blockOut")
            self.moeOut = try make(chunkTokens * d, stride: h,
                                   label: "flashnext.prefill.moeOut")
            self.routerLogits = try make(chunkTokens * cfg.numExperts, stride: f,
                                         label: "flashnext.prefill.routerLogits")
            self.routeIDs = try make(chunkTokens * cfg.topKExperts, stride: u,
                                     mode: .storageModeShared,
                                     label: "flashnext.prefill.routeIDs")
            self.routeWeights = try make(chunkTokens * cfg.topKExperts, stride: h,
                                         mode: .storageModeShared,
                                         label: "flashnext.prefill.routeWeights")
            self.routePartials = try make(
                chunkTokens * cfg.topKExperts * d, stride: h,
                label: "flashnext.prefill.routePartials")
            self.routedGateUpAct = try make(
                3 * pairRows * cfg.moeIntermediateSize, stride: h,
                label: "flashnext.prefill.routedGateUpAct")
            self.routedDown = try make(pairRows * d, stride: h,
                                       label: "flashnext.prefill.routedDown")
            self.routedMatrixAct = try make(
                chunkTokens * cfg.topKExperts * cfg.moeIntermediateSize,
                stride: h, label: "flashnext.prefill.routedMatrixAct")
            self.sharedGate = try make(chunkTokens * cfg.intermediateSize, stride: h,
                                       label: "flashnext.prefill.sharedGate")
            self.sharedUp = try make(chunkTokens * cfg.intermediateSize, stride: h,
                                     label: "flashnext.prefill.sharedUp")
            self.sharedAct = try make(chunkTokens * cfg.intermediateSize, stride: h,
                                      label: "flashnext.prefill.sharedAct")
            self.sharedOut = try make(chunkTokens * d, stride: h,
                                      label: "flashnext.prefill.sharedOut")
            self.sharedScalar = try make(chunkTokens, stride: f,
                                         label: "flashnext.prefill.sharedScalar")
            self.gdnQKV = try make(chunkTokens * la.qkvDim, stride: h,
                                   label: "flashnext.prefill.gdnQKV")
            self.gdnConvOut = try make(chunkTokens * la.qkvDim, stride: h,
                                       label: "flashnext.prefill.gdnConvOut")
            self.gdnZ = try make(chunkTokens * la.valueDim, stride: h,
                                 label: "flashnext.prefill.gdnZ")
            self.gdnA = try make(chunkTokens * la.numVHeads, stride: h,
                                 label: "flashnext.prefill.gdnA")
            self.gdnB = try make(chunkTokens * la.numVHeads, stride: h,
                                 label: "flashnext.prefill.gdnB")
            self.gdnY = try make(chunkTokens * la.valueDim, stride: h,
                                 label: "flashnext.prefill.gdnY")
            self.gdnOut = try make(chunkTokens * la.valueDim, stride: h,
                                   label: "flashnext.prefill.gdnOut")
        }
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
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillMPPGroupedMoE: MPPGroupedRoutedMoE

    private let layers: [LayerTensors]
    private let embedding: TensorView
    private let embeddingMatrix: FlashNextWeightMatrix
    private let lmHead: FlashNextWeightMatrix
    private let mixer: FlashNextHyperConnections.Weights

    // PLE
    private let ple: FlashNextPLE?
    private let pleWeights: FlashNextPLE.Weights?
    private var pleScratch: FlashNextPLE.Scratch?
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
    private let prefillEffectiveScale: MTLBuffer
    private let routerIndices: MTLBuffer
    private let routerWeights: MTLBuffer
    /// Resident-mode GPU routing scratch. The router writes expert ids, the
    /// slot lookup converts them to stable offsets in the mapped layer slab,
    /// and expert compute consumes those offsets without a CPU readback.
    private let residentSlotOffsets: MTLBuffer
    private let residentAllHit: MTLBuffer
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
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillScratch?
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

    // MARK: - Phase counters (MFERENCE_PHASES=1)

    /// Phase probes cost a completion handler per command buffer and two clock
    /// reads per accounted region, so every counter below is wired up only when
    /// the phase report is going to be printed. Same gate, same reason, as
    /// `RealForwardRunner.phaseInstrumentationEnabled`: with the variable unset
    /// the forward pass does no timing work at all and every counter stays zero.
    private static let phaseInstrumentationEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_PHASES"] == "1"

    /// Wall time inside `fetchExperts` — the top-10 routed expert blobs read
    /// from SSD through the LFU slot cache, per layer, per token. Resident
    /// GPU-directed decode does not enter this path, so its value remains zero.
    public private(set) var totalIoNanos: UInt64 = 0
    /// Wall time gathering one PLE n-gram row set through `PleRowPool` (its LFU
    /// row cache, then the FP16 staging copy). One layer per token, and the
    /// pool it reads from is ~102 GB on disk.
    public private(set) var totalPleRowNanos: UInt64 = 0
    /// Wall time in the indexer's CPU top-k: the score readback plus the exact
    /// `torch.topk` ordering in `FlashNextDescendingTopK`. One round trip per
    /// full-attention layer (12 of 48), and it is a hard CPU/GPU serialization
    /// point — the attention it gates cannot be encoded until it lands.
    public private(set) var totalIndexerTopKNanos: UInt64 = 0
    private let gpuTimeLock = NSLock()
    /// GPU busy time summed over tracked command buffers, and the wall span
    /// they cover. `span - busy` is scheduling gap: this runner commits ~4
    /// buffers per layer with two CPU round trips per attention layer, so the
    /// gap is the headline number for a consolidation pass.
    public private(set) var totalGpuBusyNanos: UInt64 = 0
    private var gpuSpanFirstStart: Double = .infinity
    private var gpuSpanLastEnd: Double = 0

    public var totalGpuSpanNanos: UInt64 {
        gpuTimeLock.lock()
        defer { gpuTimeLock.unlock() }
        guard gpuSpanLastEnd > gpuSpanFirstStart else { return 0 }
        return UInt64((gpuSpanLastEnd - gpuSpanFirstStart) * 1e9)
    }

    /// Set by `produceWithoutLogits`, cleared by the `produce` that follows it.
    /// That `produce` is sequential prefill's final prompt token, so clearing
    /// the flag is also where the phase window resets — see
    /// `beginDecodePhaseWindow`.
    private var inSequentialPrefill = false

    /// Zeroes the per-phase counters. Prompt prefill runs through the same
    /// per-token code path as decode here (`PrefillRuntimeConfig.off`), so
    /// without a reset at the prefill/decode boundary the phase report prints
    /// prompt-time nanoseconds against a decode-only wall clock and the
    /// unaccounted remainder goes negative.
    ///
    /// The runner resets itself at that boundary. A one-token prompt is the
    /// exception: nothing calls `produceWithoutLogits`, so that single prefill
    /// token stays in the window.
    public func beginDecodePhaseWindow() {
        totalIoNanos = 0
        totalPleRowNanos = 0
        totalIndexerTopKNanos = 0
        gpuTimeLock.lock()
        totalGpuBusyNanos = 0
        gpuSpanFirstStart = .infinity
        gpuSpanLastEnd = 0
        gpuTimeLock.unlock()
    }

    /// `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)` when the phase report is on,
    /// 0 otherwise — so a disabled counter costs one static Bool read.
    @inline(__always)
    private static func phaseClock() -> UInt64 {
        guard phaseInstrumentationEnabled else { return 0 }
        return clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }

    /// Attribute one command buffer's GPU interval. Must be called before the
    /// buffer is committed.
    @inline(__always)
    private func trackGpuInterval(_ cb: MTLCommandBuffer) {
        guard Self.phaseInstrumentationEnabled else { return }
        cb.addCompletedHandler { [self] done in
            let start = done.gpuStartTime
            let end = done.gpuEndTime
            guard end > start else { return }
            gpuTimeLock.lock()
            totalGpuBusyNanos &+= UInt64((end - start) * 1e9)
            gpuSpanFirstStart = min(gpuSpanFirstStart, start)
            gpuSpanLastEnd = max(gpuSpanLastEnd, end)
            gpuTimeLock.unlock()
        }
    }

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
        // Both tensors the install policy may keep at INT8 — the router
        // `[numExperts, hidden]` and the shared-expert scalar gate
        // `[1, hidden]` — are `hidden` wide, so that is the widest INT8 row
        // this runner can ask for.
        self.matVec = try FlashNextMatVec(context: context, int4: int4,
                                          int8Columns: cfg.hiddenSize)
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
            prefillAttention: try PrefillAttention(context: context),
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
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRouter = try PrefillRouter(context: context)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context, weightBits: 4, siluActivation: true)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(
            context: context, siluActivation: true)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillMPPGroupedMoE = MPPGroupedRoutedMoE(context: context)

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

        // Decode scratch is one token wide. Chunked prompt ingestion owns its
        // separate, multi-row `PrefillScratch` allocation.
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
              let effectiveScale = device.makeBuffer(
                bytes: [UInt16](repeating: Quantization.bf16Bits(1),
                                count: hidden),
                length: hidden * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
              let indices = device.makeBuffer(
                length: topK * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let residentOffsets = device.makeBuffer(
                length: topK * MemoryLayout<UInt32>.stride,
                options: .storageModePrivate),
              let residentAllHit = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModePrivate) else { throw MetalError.noDevice }
        self.routerExpertScale = scale
        self.prefillEffectiveScale = effectiveScale
        self.routerIndices = indices
        self.routerWeights = try half(topK)
        self.residentSlotOffsets = residentOffsets
        self.residentAllHit = residentAllHit
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
        prefillChunkState.reset()
        inSequentialPrefill = false
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
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token, position: p, into: logits)
        // Sequential prefill runs `produceWithoutLogits` for every prompt token
        // but the last, so the first `produce` after one of those *is* that last
        // prompt token — the prefill/decode boundary, and where the phase window
        // has to start over. Every later `produce` is a decode step.
        if inSequentialPrefill {
            inSequentialPrefill = false
            beginDecodePhaseWindow()
        }
    }

    func produceWithoutLogits(token: Int32, position p: Int) async throws {
        try prefillChunkState.requireClean(operation: "produceWithoutLogits")
        inSequentialPrefill = true
        try await produceToken(token: token, position: p, into: nil)
    }

    // MARK: - Prefill (chunked, layer-major)

    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "Flash-Next prefillChunked requires chunked mode")
        }
        guard startPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "Flash-Next prefill cursor \(position) != startPosition \(startPosition)")
        }
        guard startPosition >= 0,
              tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "Flash-Next prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < cfg.vocabSize }) else {
            throw FlashNextForwardRunnerError.invalidInput(
                "Flash-Next prefill token is outside the vocabulary")
        }
        guard capture == nil else {
            throw PrefillError.chunkedUnsupported(
                "Flash-Next tensor capture is a sequential reference/debug facility")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        guard logits.length >= cfg.vocabSize * MemoryLayout<Float16>.stride else {
            throw FlashNextForwardRunnerError.invalidInput(
                "Flash-Next logits buffer is too small")
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        for (spanIndex, span) in spans.enumerated() {
            try Task.checkCancellation()
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                logits: logits,
                scratch: scratch,
                writeFinalHead: spanIndex == spans.count - 1)
            onProgress(span.completedCount)
        }
        beginDecodePhaseWindow()
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws
        -> PrefillScratch {
        let chunkTokens = max(1, min(config.chunkTokens,
                                     PrefillRuntimeConfig.maxChunkTokens))
        if let prefillScratch, prefillScratch.chunkTokens == chunkTokens {
            return prefillScratch
        }
        let scratch = try PrefillScratch(
            device: ctx.device, chunkTokens: chunkTokens, maxContext: maxContext,
            cfg: cfg, bundle: bundle, hcEncoder: hc,
            indexerEncoder: indexer, attentionEncoder: attention,
            pleEncoder: ple, genericGDNEncoder: genericGDN)
        if let ple, let chunkPLE = scratch.ple,
           let cb = ctx.queue.makeCommandBuffer() {
            if let prior = pleScratch, prior.convState !== chunkPLE.convState,
               let blit = cb.makeBlitCommandEncoder() {
                blit.copy(from: prior.convState, sourceOffset: 0,
                          to: chunkPLE.convState, destinationOffset: 0,
                          size: min(prior.convState.length,
                                    chunkPLE.convState.length))
                blit.endEncoding()
            } else {
                ple.encodeResetState(commandBuffer: cb, scratch: chunkPLE)
            }
            try finish(cb)
            // Decode continues in the same carried conv state. A PLE scratch
            // sized for a chunk is also valid for a one-row decode call.
            pleScratch = chunkPLE
        }
        prefillScratch = scratch
        return scratch
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     logits: MTLBuffer,
                                     scratch: PrefillScratch,
                                     writeFinalHead: Bool) async throws {
        let t = tokens.count
        precondition(t > 0 && t <= scratch.chunkTokens)
        let half = MemoryLayout<Float16>.stride
        let tokenBase = scratch.tokens.contents()
            .bindMemory(to: UInt32.self, capacity: scratch.chunkTokens)
        for (row, token) in tokens.enumerated() {
            tokenBase[row] = UInt32(bitPattern: token)
        }

        // PLE table I/O depends only on token IDs. Gather the entire chunk up
        // front so its one special layer does not turn into per-token GPU/CPU
        // synchronization.
        if let hash = pleHash, let pool = pleRowPool,
           let staging = scratch.pleStaging {
            let ids = tokens.map(Int.init)
            let window = pleHistory + ids
            let rows = Array(hash.rowIDs(window: window).suffix(t))
            let base = staging.contents().bindMemory(
                to: Float16.self, capacity: scratch.chunkTokens * hidden)
            let clock = Self.phaseClock()
            for row in 0..<t {
                let embedding = try pool.readEmbedding(rows: rows[row])
                precondition(embedding.count == hidden)
                for column in 0..<hidden {
                    base[row * hidden + column] = Float16(embedding[column])
                }
            }
            if Self.phaseInstrumentationEnabled {
                totalPleRowNanos &+= Self.phaseClock() - clock
            }
            pleHistory = Array(window.suffix(hash.historyLength))
        }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: t)
        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw FlashNextForwardRunnerError.commandFailed("no command buffer")
        }
        switch embeddingMatrix {
        case .int4:
            prefillEmbed.encode(
                commandBuffer: cb,
                table: embedding.buffer, tableOffset: Int(embedding.offset),
                scales: embedding.buffer, scalesOffset: Int(embedding.scaleOffset),
                biases: embedding.buffer, biasesOffset: Int(embedding.biasOffset),
                tokens: scratch.tokens, out: scratch.embed,
                t: UInt32(t), d: UInt32(hidden), outScale: 1)
        case .int8:
            preconditionFailure("Flash-Next embedding policy never emits INT8")
        case let .bf16(buffer, offset):
            for row in 0..<t {
                guard let enc = cb.makeComputeCommandEncoder() else { continue }
                enc.setComputePipelineState(embedBF16PSO)
                enc.setBuffer(buffer, offset: offset, index: 0)
                enc.setBuffer(scratch.embed, offset: row * hidden * half, index: 1)
                var token = tokenBase[row]
                var d = UInt32(hidden)
                enc.setBytes(&token, length: 4, index: 2)
                enc.setBytes(&d, length: 4, index: 3)
                let width = min(Int(embedBF16PSO.maxTotalThreadsPerThreadgroup), 256)
                enc.dispatchThreads(
                    MTLSize(width: hidden, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(width, hidden),
                                                   height: 1, depth: 1))
                enc.endEncoding()
            }
        }
        hc.encodeTileEmbedding(commandBuffer: cb, embedding: scratch.embed,
                               hyper: scratch.hyper, rows: t)

        for (layerIndex, layer) in layers.enumerated() {
            try Task.checkCancellation()
            if layerIndex == pleLayer, let ple, let weights = pleWeights,
               let pleScratch = scratch.ple, let staging = scratch.pleStaging {
                if let blit = cb.makeBlitCommandEncoder() {
                    blit.copy(from: staging, sourceOffset: 0,
                              to: pleScratch.embeds, destinationOffset: 0,
                              size: t * hidden * half)
                    blit.endEncoding()
                }
                ple.encode(commandBuffer: cb, weights: weights,
                           scratch: pleScratch, hyper: scratch.hyper, rows: t)
            }

            hc.encodeMix(commandBuffer: cb, weights: layer.attnHC,
                         scratch: scratch.hc, hyper: scratch.hyper,
                         mixed: scratch.mixed, rows: t)
            hc.encodeInjectGate(commandBuffer: cb, weights: layer.attnHC,
                                scratch: scratch.hc, rows: t)
            if layer.isLinear {
                try encodeGDNPrefill(commandBuffer: cb, layer: layer,
                                     index: layerIndex, scratch: scratch,
                                     rows: t)
            } else {
                cb = try encodeAttentionPrefill(
                    commandBuffer: cb, layer: layer, index: layerIndex,
                    scratch: scratch, rows: t, startPosition: startPosition)
            }
            hc.encodeInjectAccumulate(commandBuffer: cb, scratch: scratch.hc,
                                      hyper: scratch.hyper,
                                      block: scratch.blockOut, rows: t)

            hc.encodeMix(commandBuffer: cb, weights: layer.mlpHC,
                         scratch: scratch.hc, hyper: scratch.hyper,
                         mixed: scratch.mixed, rows: t)
            hc.encodeInjectGate(commandBuffer: cb, weights: layer.mlpHC,
                                scratch: scratch.hc, rows: t)
            try encodeRouterPrefill(commandBuffer: cb, layer: layer,
                                    scratch: scratch, rows: t)
            try encodeSharedExpertPrefill(commandBuffer: cb, layer: layer,
                                          scratch: scratch, rows: t)
            try finish(cb)

            let routeCount = t * topK
            let idPointer = scratch.routeIDs.contents()
                .bindMemory(to: UInt32.self, capacity: routeCount)
            let weightPointer = scratch.routeWeights.contents()
                .bindMemory(to: Float16.self, capacity: routeCount)
            let routeIDs = (0..<routeCount).map {
                min(idPointer[$0], UInt32(numExperts - 1))
            }
            let routeWeights = (0..<routeCount).map { weightPointer[$0] }
            if layer.moe.expertsAreBF16 {
                try await encodeBF16RoutedPrefill(
                    layerIndex: layerIndex, layer: layer, scratch: scratch,
                    routeIDs: routeIDs, rows: t)
            } else {
                try await encodeINT4RoutedPrefill(
                    layerIndex: layerIndex, layer: layer, scratch: scratch,
                    routeIDs: routeIDs, routeWeights: routeWeights, rows: t)
            }

            guard let tail = ctx.queue.makeCommandBuffer() else {
                throw FlashNextForwardRunnerError.commandFailed("no command buffer")
            }
            cb = tail
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                           hidden: scratch.moeOut,
                                           delta: scratch.sharedOut,
                                           count: t * hidden)
            hc.encodeInjectAccumulate(commandBuffer: cb, scratch: scratch.hc,
                                      hyper: scratch.hyper,
                                      block: scratch.moeOut, rows: t)
        }

        hc.encodeMix(commandBuffer: cb, weights: mixer, scratch: scratch.hc,
                     hyper: scratch.hyper, mixed: scratch.mixed, rows: t)
        if writeFinalHead {
            matVec.encode(commandBuffer: cb, matrix: lmHead,
                          x: scratch.mixed, xOffset: (t - 1) * hidden * half,
                          y: logits, rows: cfg.vocabSize, cols: hidden)
        }
        try finish(cb)
        position += t
        prefillChunkState.markCommitted()
    }

    private func encodeGDNPrefill(commandBuffer cb: MTLCommandBuffer,
                                  layer: LayerTensors, index: Int,
                                  scratch: PrefillScratch, rows: Int) throws {
        guard let weights = layer.gdn else { return }
        guard let gdn, let state = gdnState else {
            throw PrefillError.chunkedUnsupported(
                "Flash-Next chunked GDN requires the production 32-lane geometry")
        }
        let la = cfg.linearAttention
        matVec.encodeBatched(commandBuffer: cb, matrix: weights.qkv,
                             x: scratch.mixed, y: scratch.gdnQKV,
                             matrixRows: la.qkvDim, matrixColumns: hidden,
                             tokens: rows)
        matVec.encodeBatched(commandBuffer: cb, matrix: weights.z,
                             x: scratch.mixed, y: scratch.gdnZ,
                             matrixRows: la.valueDim, matrixColumns: hidden,
                             tokens: rows)
        matVec.encodeBatched(commandBuffer: cb, matrix: weights.a,
                             x: scratch.mixed, y: scratch.gdnA,
                             matrixRows: la.numVHeads, matrixColumns: hidden,
                             tokens: rows)
        matVec.encodeBatched(commandBuffer: cb, matrix: weights.b,
                             x: scratch.mixed, y: scratch.gdnB,
                             matrixRows: la.numVHeads, matrixColumns: hidden,
                             tokens: rows)
        let tail = state.convTailBuffer(layer: index)
        gdn.encodeConvPrefill(commandBuffer: cb, tail: tail,
                              qkvRows: scratch.gdnQKV,
                              convWeight: weights.conv.buffer,
                              convWeightOffset: Int(weights.conv.offset),
                              out: scratch.gdnConvOut, rows: rows)
        gdn.encodeConvTailUpdate(commandBuffer: cb, tail: tail,
                                 qkvRows: scratch.gdnQKV, rows: rows)
        gdn.encodeQKNorm(commandBuffer: cb, convOut: scratch.gdnConvOut,
                         rows: rows)
        gdn.encodeDeltaStepPrefill(
            commandBuffer: cb, convOut: scratch.gdnConvOut,
            aProj: scratch.gdnA, bProj: scratch.gdnB,
            aLog: weights.aLog.buffer, aLogOffset: Int(weights.aLog.offset),
            dtBias: weights.dtBias.buffer, dtBiasOffset: Int(weights.dtBias.offset),
            state: state.stateBuffer(layer: index), y: scratch.gdnY, rows: rows)
        gdn.encodeGatedNorm(commandBuffer: cb, y: scratch.gdnY,
                            z: scratch.gdnZ,
                            weight: weights.norm.buffer,
                            weightOffset: Int(weights.norm.offset),
                            out: scratch.gdnOut, rows: rows)
        matVec.encodeBatched(commandBuffer: cb, matrix: weights.out,
                             x: scratch.gdnOut, y: scratch.blockOut,
                             matrixRows: hidden, matrixColumns: la.valueDim,
                             tokens: rows)
    }

    private func encodeAttentionPrefill(
        commandBuffer cb: MTLCommandBuffer,
        layer: LayerTensors, index: Int,
        scratch: PrefillScratch, rows: Int, startPosition: Int
    ) throws -> MTLCommandBuffer {
        guard let indexWeights = layer.indexer,
              let indexCache = indexerCaches[index],
              let attentionWeights = layer.attention,
              let kv = kvCaches[index] else {
            throw FlashNextForwardRunnerError.invalidConfiguration(
                "Flash-Next full-attention layer \(index) is incomplete")
        }
        indexer.encodeProjection(commandBuffer: cb,
                                 weight: indexWeights.qkProj,
                                 x: scratch.mixed, xOffset: 0,
                                 hidden: hidden, scratch: scratch.indexer,
                                 rows: rows)
        indexer.encodePrepare(
            commandBuffer: cb,
            qNorm: indexWeights.qNorm.buffer,
            qNormOffset: Int(indexWeights.qNorm.offset),
            kNorm: indexWeights.kNorm.buffer,
            kNormOffset: Int(indexWeights.kNorm.offset),
            scratch: scratch.indexer, cache: indexCache,
            rows: rows, startPosition: startPosition)
        let allRowsContiguous = indexer.selectsAllVisible(
            row: rows - 1, startPosition: startPosition)
        if !allRowsContiguous {
            indexer.encodeScores(commandBuffer: cb, scratch: scratch.indexer,
                                 cache: indexCache, rows: rows,
                                 startPosition: startPosition)
            indexer.encodeSelection(commandBuffer: cb, scratch: scratch.indexer,
                                    rows: rows, startPosition: startPosition)
        }
        attention.encodeProjectAndCache(
            commandBuffer: cb, weights: attentionWeights,
            scratch: scratch.attention, cache: kv,
            x: scratch.mixed, xOffset: 0, rows: rows,
            startPosition: startPosition)
        var contiguousRows = 0
        while contiguousRows < rows,
              indexer.selectsAllVisible(row: contiguousRows,
                                        startPosition: startPosition) {
            contiguousRows += 1
        }
        if contiguousRows > 0 {
            attention.encodeAttendContiguousPrefix(
                commandBuffer: cb, scratch: scratch.attention, cache: kv,
                rows: contiguousRows, startPosition: startPosition)
        }
        for row in contiguousRows..<rows {
            let count = indexer.selectionCount(row: row,
                                               startPosition: startPosition)
            indexer.encodeGatherKV(
                commandBuffer: cb,
                kCache: kv.keys, kCacheOffset: 0,
                vCache: kv.values, vCacheOffset: 0,
                scratch: scratch.indexer, selectionRow: row,
                kOut: scratch.attention.gatheredK, kOutOffset: 0,
                vOut: scratch.attention.gatheredV, vOutOffset: 0,
                kvDim: attention.geometry.kvDim, count: count)
            attention.encodeAttendRow(commandBuffer: cb,
                                      scratch: scratch.attention,
                                      row: row, slot: 0,
                                      selectedCount: count)
        }
        attention.encodeGateAndProject(
            commandBuffer: cb, weights: attentionWeights,
            scratch: scratch.attention, out: scratch.blockOut,
            outOffset: 0, rows: rows)
        return cb
    }

    private func encodeRouterPrefill(commandBuffer cb: MTLCommandBuffer,
                                     layer: LayerTensors,
                                     scratch: PrefillScratch, rows: Int) throws {
        let half = MemoryLayout<Float16>.stride
        switch layer.moe.router {
        case let .int8(weights, weightsOffset, scales, scalesOffset,
                       biases, biasesOffset):
            prefillRouter.encodeGemma4Block(
                commandBuffer: cb,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                hidden: scratch.mixed,
                effectiveScale: prefillEffectiveScale,
                perExpertScale: routerExpertScale,
                outIndices: scratch.routeIDs,
                outWeights: scratch.routeWeights,
                queryCount: UInt32(rows), numExperts: UInt32(numExperts),
                d: UInt32(hidden), topK: UInt32(topK),
                hiddenStrideElements: UInt32(hidden))
        case .int4, .bf16:
            matVec.encodeBatched(commandBuffer: cb, matrix: layer.moe.router,
                                 x: scratch.mixed, y: scratch.routerLogits,
                                 matrixRows: numExperts, matrixColumns: hidden,
                                 tokens: rows, outputFloat32: true)
            for row in 0..<rows {
                moeBF16.encodeRouterSelect(
                    commandBuffer: cb,
                    logits: scratch.routerLogits,
                    logitsOffset: row * numExperts * MemoryLayout<Float>.stride,
                    perExpertScale: routerExpertScale,
                    outIndices: scratch.routeIDs,
                    outIndicesOffset: row * topK * MemoryLayout<UInt32>.stride,
                    outWeights: scratch.routeWeights,
                    outWeightsOffset: row * topK * half,
                    numExperts: UInt32(numExperts))
            }
        }
    }

    private func encodeSharedExpertPrefill(commandBuffer cb: MTLCommandBuffer,
                                           layer: LayerTensors,
                                           scratch: PrefillScratch,
                                           rows: Int) throws {
        func projection(_ matrix: FlashNextWeightMatrix,
                        rows: Int, columns: Int) -> SharedExpertProjection? {
            guard case let .int4(weights, weightsOffset, scales, scalesOffset,
                                 biases, biasesOffset) = matrix else { return nil }
            return SharedExpertProjection(
                weights: weights, scales: scales, biases: biases,
                weightsOffset: weightsOffset, scalesOffset: scalesOffset,
                biasesOffset: biasesOffset,
                rows: UInt32(rows), cols: UInt32(columns))
        }
        if let gate = projection(layer.moe.sharedGateProj,
                                 rows: sharedIntermediate, columns: hidden),
           let up = projection(layer.moe.sharedUp,
                               rows: sharedIntermediate, columns: hidden),
           let down = projection(layer.moe.sharedDown,
                                 rows: hidden, columns: sharedIntermediate) {
            _ = try prefillSharedExpert.encodeBlock(
                commandBuffer: cb, x: scratch.mixed, y: scratch.sharedOut,
                gate: gate, up: up, down: down,
                scratchGate: scratch.sharedGate,
                scratchUp: scratch.sharedUp,
                scratchAct: scratch.sharedAct,
                queryCount: rows, d: hidden,
                intermediate: sharedIntermediate,
                xStrideElements: hidden, yStrideElements: hidden)
            if case let .int8(weights, weightsOffset, scales, scalesOffset,
                              biases, biasesOffset) = layer.moe.sharedGate {
                let scalar = SharedExpertProjection(
                    weights: weights, scales: scales, biases: biases,
                    weightsOffset: weightsOffset, scalesOffset: scalesOffset,
                    biasesOffset: biasesOffset, rows: 1, cols: UInt32(hidden))
                try prefillSharedExpert.encodeQwenScalarGate(
                    commandBuffer: cb, x: scratch.mixed, gate: scalar,
                    y: scratch.sharedOut, queryCount: rows, d: hidden,
                    xStrideElements: hidden, yStrideElements: hidden)
            } else {
                matVec.encodeBatched(commandBuffer: cb,
                                     matrix: layer.moe.sharedGate,
                                     x: scratch.mixed, y: scratch.sharedScalar,
                                     matrixRows: 1, matrixColumns: hidden,
                                     tokens: rows, outputFloat32: true)
                for row in 0..<rows {
                    moeBF16.encodeSharedGateScale(
                        commandBuffer: cb, out: scratch.sharedOut,
                        outOffset: row * hidden * MemoryLayout<Float16>.stride,
                        scalar: scratch.sharedScalar,
                        scalarOffset: row * MemoryLayout<Float>.stride,
                        count: hidden)
                }
            }
            return
        }

        matVec.encodeBatched(commandBuffer: cb,
                             matrix: layer.moe.sharedGateProj,
                             x: scratch.mixed, y: scratch.sharedGate,
                             matrixRows: sharedIntermediate,
                             matrixColumns: hidden, tokens: rows)
        matVec.encodeBatched(commandBuffer: cb, matrix: layer.moe.sharedUp,
                             x: scratch.mixed, y: scratch.sharedUp,
                             matrixRows: sharedIntermediate,
                             matrixColumns: hidden, tokens: rows)
        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(siluMulPSO)
            enc.setBuffer(scratch.sharedGate, offset: 0, index: 0)
            enc.setBuffer(scratch.sharedUp, offset: 0, index: 1)
            enc.setBuffer(scratch.sharedAct, offset: 0, index: 2)
            var count = UInt32(rows * sharedIntermediate)
            enc.setBytes(&count, length: 4, index: 3)
            enc.dispatchThreads(
                MTLSize(width: Int(count), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(Int(siluMulPSO.maxTotalThreadsPerThreadgroup), 256),
                    height: 1, depth: 1))
            enc.endEncoding()
        }
        matVec.encodeBatched(commandBuffer: cb, matrix: layer.moe.sharedDown,
                             x: scratch.sharedAct, y: scratch.sharedOut,
                             matrixRows: hidden,
                             matrixColumns: sharedIntermediate, tokens: rows)
        matVec.encodeBatched(commandBuffer: cb, matrix: layer.moe.sharedGate,
                             x: scratch.mixed, y: scratch.sharedScalar,
                             matrixRows: 1, matrixColumns: hidden,
                             tokens: rows, outputFloat32: true)
        for row in 0..<rows {
            moeBF16.encodeSharedGateScale(
                commandBuffer: cb, out: scratch.sharedOut,
                outOffset: row * hidden * MemoryLayout<Float16>.stride,
                scalar: scratch.sharedScalar,
                scalarOffset: row * MemoryLayout<Float>.stride,
                count: hidden)
        }
    }

    private func encodeINT4RoutedPrefill(
        layerIndex: Int, layer: LayerTensors,
        scratch: PrefillScratch,
        routeIDs: [UInt32], routeWeights: [Float16], rows: Int
    ) async throws {
        let pairs = PrefillRouter.makeTokenExpertPairs(
            indices: routeIDs, weights: routeWeights,
            queryCount: rows, topK: topK)
        let cacheSlotCount = model.routedExpertCacheSlotCount(layer: layerIndex)
        let slotCount = cacheSlotCount ?? 16
        guard slotCount >= topK else {
            throw PrefillError.chunkedUnsupported(
                "Flash-Next top-\(topK) routing needs at least \(topK) expert slots")
        }
        let tileExperts = min(16, slotCount)
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: rows, topK: topK, numExperts: numExperts,
            tileExpertCount: tileExperts,
            expertSortKeys: model.routedExpertPhysicalOffsets(layer: layerIndex))
        let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
            device: ctx.device, routes: routes)

        // A resident backend cannot overwrite expert slots because it has no
        // slot cache: every view points at its own immutable mapped region.
        // Encode all expert tiles into one command buffer so a high-memory Mac
        // pays one GPU drain per layer rather than one per 16 experts.
        if cacheSlotCount == nil, hidden >= 1_024,
           scratch.chunkTokens >= 1_024, prefillMPPGroupedMoE.isAvailable,
           let resident = try model.routedResidentSlabBinding(layer: layerIndex) {
            guard resident.slotStride <= Int(UInt32.max),
                  let cb = ctx.queue.makeCommandBuffer() else {
                throw FlashNextForwardRunnerError.commandFailed(
                    "invalid resident expert slab binding")
            }
            let params = PrefillGroupedRoutedMoEStreamedParams(
                groupStart: 0, groupCount: UInt32(routes.groups.count),
                d: UInt32(hidden), routedIntermediate: UInt32(moeIntermediate),
                topK: UInt32(topK), hiddenStrideElements: UInt32(hidden),
                offsets: layer.moe.expertOffsets)
            let encoded = prefillMPPGroupedMoE.encodeResident(
                commandBuffer: cb, hidden: scratch.mixed,
                sortedPairs: metadata.sortedPairs, groups: metadata.groups,
                activation: scratch.routedMatrixAct,
                routePartials: scratch.routePartials,
                slab: resident.slab, params: params,
                residentExpertStride: UInt32(resident.slotStride),
                maxPairsPerGroup: routes.maxPairsPerExpert)
            precondition(encoded, "resident grouped TensorOps encoding failed")
            try finish(cb)
        } else if cacheSlotCount == nil {
            guard let cb = ctx.queue.makeCommandBuffer() else {
                throw FlashNextForwardRunnerError.commandFailed("no command buffer")
            }
            var retainedFetches: [PrefillStreamedTileFetchResult] = []
            var retainedArguments: [MTLBuffer] = []
            retainedFetches.reserveCapacity(routes.tiles.count)
            retainedArguments.reserveCapacity(routes.tiles.count)
            for (tileIndex, tile) in routes.tiles.enumerated() {
                let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                    model: model, layer: layerIndex, tileIndex: tileIndex,
                    routes: routes)
                let argument = try encodeINT4RoutedPrefillTile(
                    commandBuffer: cb, tile: tile, routes: routes,
                    binding: fetch.binding, offsets: layer.moe.expertOffsets,
                    metadata: metadata, scratch: scratch)
                retainedFetches.append(fetch)
                retainedArguments.append(argument)
            }
            try withExtendedLifetime((retainedFetches, retainedArguments)) {
                try finish(cb)
            }
        } else {
            for (tileIndex, tile) in routes.tiles.enumerated() {
                let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                    model: model, layer: layerIndex, tileIndex: tileIndex,
                    routes: routes)
                guard let cb = ctx.queue.makeCommandBuffer() else {
                    throw FlashNextForwardRunnerError.commandFailed("no command buffer")
                }
                let argument = try encodeINT4RoutedPrefillTile(
                    commandBuffer: cb, tile: tile, routes: routes,
                    binding: fetch.binding, offsets: layer.moe.expertOffsets,
                    metadata: metadata, scratch: scratch)
                try withExtendedLifetime((fetch, argument)) { try finish(cb) }
            }
        }
        guard let reduce = ctx.queue.makeCommandBuffer() else {
            throw FlashNextForwardRunnerError.commandFailed("no command buffer")
        }
        prefillMoE.encodeReduceTokenMajor(
            commandBuffer: reduce, routePartials: scratch.routePartials,
            routeWeights: scratch.routeWeights, h2: scratch.moeOut,
            queryCount: UInt32(rows), topK: UInt32(topK), d: UInt32(hidden))
        try finish(reduce)
    }

    /// One expert tile through the 8-row grouped TensorOps kernel when the
    /// production geometry supports it, with the original exact GEMV path as
    /// the portable and synthetic-fixture fallback.
    private func encodeINT4RoutedPrefillTile(
        commandBuffer cb: MTLCommandBuffer,
        tile: PrefillMoETile,
        routes: PrefillMoEGroupedRoutes,
        binding: PrefillStreamedTileBinding,
        offsets: MoEExpertOffsets,
        metadata: PrefillGroupedRoutedMoEStreamedMetadataBuffers,
        scratch: PrefillScratch
    ) throws -> MTLBuffer {
        let params = PrefillGroupedRoutedMoEStreamedParams(
            pairStart: tile.pairStart, pairCount: tile.pairCount,
            d: UInt32(hidden), routedIntermediate: UInt32(moeIntermediate),
            topK: UInt32(topK), hiddenStrideElements: UInt32(hidden),
            binding: binding, offsets: offsets)
        if hidden >= 1_024, scratch.chunkTokens >= 1_024,
           prefillMPPGroupedMoE.isAvailable {
            let argument = try prefillMPPGroupedMoE.makeArgumentBuffer(
                device: ctx.device, binding: binding)
            let first = Int(tile.groupStart)
            let end = first + Int(tile.groupCount)
            let maxPairs = routes.groups[first..<end]
                .map { Int($0.pairCount) }.max() ?? 0
            var mppParams = params
            mppParams.pairStart = tile.groupStart
            mppParams.pairCount = tile.groupCount
            if prefillMPPGroupedMoE.encode(
                commandBuffer: cb, hidden: scratch.mixed,
                sortedPairs: metadata.sortedPairs, groups: metadata.groups,
                activation: scratch.routedMatrixAct,
                routePartials: scratch.routePartials,
                argumentBuffer: argument, binding: binding,
                params: mppParams, maxPairsPerGroup: maxPairs) {
                return argument.buffer
            }
        }

        let argument = try prefillGroupedMoE.makeStreamedArgumentBuffer(
            device: ctx.device, binding: binding)
        _ = prefillGroupedMoE.encodeStreamedBatched(
            commandBuffer: cb, hidden: scratch.mixed,
            sortedPairs: metadata.sortedPairs,
            routePartials: scratch.routePartials,
            gateUpActScratch: scratch.routedGateUpAct,
            downScratch: scratch.routedDown,
            argumentBuffer: argument, binding: binding,
            params: params, pairMicrobatchRows: 256)
        return argument.buffer
    }

    /// Dense-BF16 experts exist only in the tiny parity fixture. Keep them on
    /// the same layer-major chunk path, while the shipped INT4 checkpoint takes
    /// the expert-major grouped implementation above.
    private func encodeBF16RoutedPrefill(
        layerIndex: Int, layer: LayerTensors,
        scratch: PrefillScratch, routeIDs: [UInt32], rows: Int
    ) async throws {
        let half = MemoryLayout<Float16>.stride
        for row in 0..<rows {
            let experts = (0..<topK).map {
                Int(routeIDs[row * topK + $0])
            }
            let blobs = try await fetchExperts(layer: layerIndex, experts: experts)
            guard let cb = ctx.queue.makeCommandBuffer() else {
                throw FlashNextForwardRunnerError.commandFailed("no command buffer")
            }
            let arg = moeBF16.makeRoutedArgumentBuffer(routedBlobs: blobs)
            moeBF16.encodePhase1(
                commandBuffer: cb, routedArgBuffer: arg, routedBlobs: blobs,
                routedOffsets: layer.moe.expertOffsets,
                x: scratch.mixed, xOffset: row * hidden * half,
                acts: moeActs, d: UInt32(hidden), f: UInt32(moeIntermediate),
                topK: UInt32(topK))
            moeBF16.encodePhase2(
                commandBuffer: cb, routedArgBuffer: arg, routedBlobs: blobs,
                routedOffsets: layer.moe.expertOffsets, acts: moeActs,
                routingWeights: scratch.routeWeights,
                routingWeightsOffset: row * topK * half,
                residual: zeroResidual,
                y: scratch.moeOut, yOffset: row * hidden * half,
                d: UInt32(hidden), f: UInt32(moeIntermediate),
                topK: UInt32(topK))
            try withExtendedLifetime((arg, blobs)) { try finish(cb) }
        }
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
        trackGpuInterval(head)
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
            guard let attnWeights = layer.attention, let kv = kvCaches[L] else {
                throw FlashNextForwardRunnerError.invalidConfiguration(
                    "layer \(L) is a full-attention layer with no projections")
            }
            let contiguous = indexer.selectsAllVisible(row: 0,
                                                       startPosition: p)
            let count: Int
            if capture != nil {
                // Capture keeps the CPU oracle as an independent reference for
                // the production device selector.
                let tTopKStart = Self.phaseClock()
                let selected: [Int]
                if contiguous {
                    selected = Array(0...p)
                } else {
                    indexer.encodeScores(commandBuffer: cb,
                                         scratch: indexerScratch,
                                         cache: cache, rows: 1,
                                         startPosition: p)
                    try finish(cb)
                    selected = indexer.selections(
                        scratch: indexerScratch, rows: 1, startPosition: p)[0]
                }
                if Self.phaseInstrumentationEnabled {
                    totalIndexerTopKNanos &+= Self.phaseClock() - tTopKStart
                }
                capture?.integers[key + "indexer_selected"] = [selected]
                capture?.integers[key + "indexer_visible"] = [Array(0...p)]
                count = selected.count
                if !contiguous {
                    _ = indexer.writeSelection(selected, row: 0,
                                               into: indexerScratch)
                    guard let next = ctx.queue.makeCommandBuffer() else {
                        throw FlashNextForwardRunnerError.commandFailed(
                            "no command buffer")
                    }
                    cb = next
                }
            } else {
                if !contiguous {
                    indexer.encodeScores(commandBuffer: cb,
                                         scratch: indexerScratch,
                                         cache: cache, rows: 1,
                                         startPosition: p)
                    indexer.encodeSelection(commandBuffer: cb,
                                            scratch: indexerScratch,
                                            rows: 1, startPosition: p)
                }
                count = indexer.selectionCount(row: 0, startPosition: p)
            }
            attention.encodeProjectAndCache(
                commandBuffer: cb, weights: attnWeights, scratch: attnScratch,
                cache: kv, x: mixed, xOffset: 0, rows: 1, startPosition: p)
            if contiguous {
                attention.encodeAttendContiguousRow(
                    commandBuffer: cb, scratch: attnScratch, cache: kv,
                    row: 0, visibleCount: count)
            } else {
                indexer.encodeGatherKV(
                    commandBuffer: cb,
                    kCache: kv.keys, kCacheOffset: 0,
                    vCache: kv.values, vCacheOffset: 0,
                    scratch: indexerScratch, selectionRow: 0,
                    kOut: attnScratch.gatheredK, kOutOffset: 0,
                    vOut: attnScratch.gatheredV, vOutOffset: 0,
                    kvDim: attention.geometry.kvDim, count: count)
                attention.encodeAttendRow(commandBuffer: cb,
                                          scratch: attnScratch,
                                          row: 0, slot: 0,
                                          selectedCount: count)
            }
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

        // In resident mode the complete layer is already a stable Metal slab.
        // Keep router ids on the GPU and address the selected expert records
        // there directly. Capture deliberately retains the reference path,
        // because its contract includes the selected ids and weights.
        if capture == nil, let moeInt4,
           let resident = try model.routedResidentSlabBinding(layer: L) {
            moeInt4.encodeResidentSlabRouted(
                commandBuffer: cb,
                slab: resident.slab,
                table: resident.table,
                slotStride: resident.slotStride,
                indices: routerIndices,
                slotOffsets: residentSlotOffsets,
                allHit: residentAllHit,
                routedOffsets: layer.moe.expertOffsets,
                x: mixed,
                acts: moeActs,
                routingWeights: routerWeights,
                residual: zeroResidual,
                y: moeOut,
                d: UInt32(hidden),
                f: UInt32(moeIntermediate),
                numExperts: UInt32(numExperts),
                topK: UInt32(topK))
            elementwise.encodeResidualAdd(commandBuffer: cb, hidden: moeOut,
                                           delta: sharedOut, count: hidden)
            hc.encodeInjectAccumulate(commandBuffer: cb, scratch: hcScratch,
                                      hyper: hyper, block: moeOut, rows: 1)
            trackGpuInterval(cb)
            cb.commit()
            pendingMoECommand = cb
            return
        }
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
        let tIoStart = Self.phaseClock()
        let blobs = try await fetchExperts(layer: L, experts: experts)
        if Self.phaseInstrumentationEnabled {
            totalIoNanos &+= Self.phaseClock() - tIoStart
        }

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
        trackGpuInterval(moeCB)
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
        case .int8:
            // The bit policy overrides two MoE gating suffixes and nothing
            // else, so no install reaches here with an INT8 embedding table.
            // There is no INT8 row-gather kernel to fall back to, and quietly
            // decoding the row as INT4 would emit a plausible wrong embedding
            // for every token, so this refuses instead.
            preconditionFailure(
                "the embedding table is INT8; Flash-Next has no INT8 embedding "
                    + "gather and the install policy never produces one")
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
        let tRowStart = Self.phaseClock()
        let embedding = try pool.readEmbedding(rows: rowIDs)
        precondition(embedding.count == hidden,
                     "PLE gather produced \(embedding.count) values, expected \(hidden)")
        let base = staging.contents().bindMemory(to: Float16.self, capacity: hidden)
        for i in 0..<hidden { base[i] = Float16(embedding[i]) }
        if Self.phaseInstrumentationEnabled {
            totalPleRowNanos &+= Self.phaseClock() - tRowStart
        }
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
        trackGpuInterval(cb)
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
        trackGpuInterval(cb)
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
