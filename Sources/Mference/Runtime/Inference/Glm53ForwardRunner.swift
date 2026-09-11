import Foundation
import Metal

public enum Glm53ForwardRunnerError: Error, CustomStringConvertible {
    case invalidConfiguration(String)
    case invalidInput(String)
    case commandFailed(String)

    public var description: String {
        switch self {
        case .invalidConfiguration(let reason), .invalidInput(let reason), .commandFailed(let reason):
            return reason
        }
    }
}

/// The forward runner for `glm53Flash` (GLM-5.3-Flash, text stack).
///
/// Built against the family's runtime design spec
/// (`docs/superpowers/specs/2026-09-11-glm53-flash-runtime-design.md`) and
/// gated by `Glm53ReferenceParityTests` (the fp32 CPU oracle against
/// PipeNetwork's goldens) and `Glm53ForwardRunnerTests` (this runner against
/// the oracle). Reached through `ForwardRunnerFactory.make`.
///
/// # Shape of the model
///
/// ```
/// streams = tile(embed(token), hc=4)
/// for L in 0..<45:
///     (pre, post, comb) = mixes(streams, attn_hc)
///     x = input_layernorm(collapse(streams, pre))
///     x = KDA(x)  (mask 7)  |  NoPE latent sparse attention(x)  (mask 8)
///     streams = post * x + combᵀ streams
///     (pre, post, comb) = mixes(streams, ffn_hc)
///     x = post_attention_layernorm(collapse(streams, pre))
///     x = dense FFN(x)  (L < 3)  |  shared expert + top-8 of 288 routed experts
///     streams = post * x + combᵀ streams
/// logits = head(norm(mean over streams))
/// ```
///
/// # Precision
///
/// FP16 activations with FP32 accumulation, as the rest of the runtime. The
/// mHC coefficients, the KDA decay and state, the indexer scores and the
/// router logits stay FP32 because each feeds a nonlinearity, a recurrence or
/// a selection. Norm eps is the model's 1e-5 (indexer key LayerNorm 1e-6).
///
/// # One command stream per token
///
/// A token is encoded into one command buffer that is committed only where a
/// CPU decision genuinely needs GPU data: the pooled indexer's top-k (sparse
/// layers past `index_topk` tokens) and, under the slot cache, the routed
/// expert fetch. The router runs on the GPU in both expert-streaming modes
/// (`glm53_router_select_k8`), so with the experts **resident** (the whole
/// set fits a 256 GB host) a MoE layer needs no round trip at all: the
/// router's indices resolve to slab offsets on the GPU and the slot-map
/// FFN bodies run in the same stream. The two modes are byte-identical.
///
/// # Prefill
///
/// Chunked prefill and sequential decode share one per-token path, so the two
/// are equal by construction: `prefillChunked` walks the tokens through
/// `produceToken`, carrying the conv tails, the KDA state, the latent and
/// indexer caches across chunk boundaries. Prompt tokens need no readback
/// (below `index_topk`, with resident experts), so each token's stream is
/// committed without waiting and the CPU encodes the next token while the
/// GPU runs this one; the chunk waits once at its end. PERF, not
/// correctness: a prompt token still streams every weight once per token
/// rather than once per chunk. A batched (GEMM) prefill is the next perf
/// item after first light.
public final class Glm53ForwardRunner: ContinuableLogitProducer,
                                       ContextWindowReporting,
                                       HeadlessSequentialPrefillRunner,
                                       ExactPrefillLogitProducer,
                                       ChunkedPrefillRunner,
                                       @unchecked Sendable {

    // MARK: - Capture

    /// Per-forward record under the goldens' local keys (`layerNN.attn_out`,
    /// ...), so `Glm53ForwardRunnerTests` can diff this runner against the
    /// fp32 oracle point for point. Off unless set; every capture point costs
    /// a queue drain and a readback.
    public struct Capture {
        public var floats: [String: [Float]] = [:]
        public var integers: [String: [Int]] = [:]
        /// `nil` inside the optional means the dense bypass.
        public var selections: [String: [Int]?] = [:]
        public init() {}
    }

    public var capture: Capture?

    /// **Debug A/B knob, not a production path.** When set, every sparse layer
    /// attends all cached latents instead of the indexer's pooled selection.
    /// While the cache holds at most `index_topk` tokens the model's own
    /// selection is exhaustive, so the two arms must generate the same tokens;
    /// past that the knob is refused, where dense would no longer be the model.
    public var denseSelectionForAB = false

    /// True when every MoE layer's experts are served from a resident slab and
    /// the routed FFN runs GPU-indexed with no CPU round trip.
    public let expertsResident: Bool

    // MARK: - Weights

    struct LayerTensors {
        let attnNorm: TensorView
        let ffnNorm: TensorView
        let hcAttnFn: TensorView
        let hcAttnBase: TensorView
        let hcAttnScale: TensorView
        let hcFFNFn: TensorView
        let hcFFNBase: TensorView
        let hcFFNScale: TensorView
        let oProj: TensorView
        // Kimi Delta Attention
        let qProj: TensorView?
        let kProj: TensorView?
        let vProj: TensorView?
        let conv: TensorView?
        let fA: TensorView?
        let fB: TensorView?
        let gA: TensorView?
        let gB: TensorView?
        let bProj: TensorView?
        let aLog: TensorView?
        let dtBias: TensorView?
        let oNorm: TensorView?
        // NoPE latent sparse attention and its indexer
        let qA: TensorView?
        let qANorm: TensorView?
        let qB: TensorView?
        let kvA: TensorView?
        let kvANorm: TensorView?
        let embedQ: TensorView?
        let unembedOut: TensorView?
        let idxQB: TensorView?
        let idxK: TensorView?
        let idxKNormWeight: TensorView?
        let idxKNormBias: TensorView?
        let idxWeights: TensorView?
        let idxPoolGate: FlashNextWeightMatrix?
        let idxApe: TensorView?
        // FFN
        let router: FlashNextWeightMatrix?
        let routerBias: TensorView?
        let sharedGate: TensorView?
        let sharedUp: TensorView?
        let sharedDown: TensorView?
        let expertOffsets: MoEExpertOffsets?
        let slab: ResidentExpertSlab?
        let denseGate: TensorView?
        let denseUp: TensorView?
        let denseDown: TensorView?
    }

    // MARK: - Stored state

    let model: Model
    let ctx: MetalContext
    let cfg: ArchConfig
    let g53: Glm53Config
    public let maxContext: Int

    let hidden: Int
    let hc: Int
    let kdaHeads: Int
    let kdaDim: Int
    let convTaps: Int
    let numHeads: Int
    let qkDim: Int
    let vDim: Int
    let kvRank: Int
    let qRank: Int
    let idxHeads: Int
    let idxDim: Int
    let idxTopK: Int
    let kPool: Int
    let topK: Int
    let numExperts: Int
    let moeF: Int
    let sharedF: Int
    let denseF: Int
    let eps: Float
    let hcEps: Float

    let kernels: Glm53Kernels
    let rms: RMSNorm
    let int8: DequantInt8GEMV
    let matVec: FlashNextMatVec
    let moe: MoE
    let state: Glm53StateManager

    let layers: [LayerTensors]
    let embedding: TensorView
    let finalNorm: TensorView
    let lmHead: TensorView

    // Scratch (FP16 unless noted)
    let streams: MTLBuffer
    let streamsAlt: MTLBuffer
    let hiddenBuf: MTLBuffer
    let normed: MTLBuffer
    let meanPre: MTLBuffer               // fp32 [hc] = 1/hc
    let hcPreA: MTLBuffer                // fp32
    let hcPostA: MTLBuffer
    let hcCombA: MTLBuffer
    let hcPreF: MTLBuffer
    let hcPostF: MTLBuffer
    let hcCombF: MTLBuffer
    let mixed: MTLBuffer                 // [3 * H * D]
    let convOut: MTLBuffer
    let kdaA: MTLBuffer                  // [H * D]
    let kdaB: MTLBuffer                  // [H]
    let kdaGate: MTLBuffer               // [H * D]
    let kdaLow: MTLBuffer                // [D]
    let kdaY: MTLBuffer                  // [H * D]
    let attnHeads: MTLBuffer             // [max(H*D, heads*vDim)]
    let attnOut: MTLBuffer               // [hidden]
    let qr: MTLBuffer                    // [qRank]
    let q: MTLBuffer                     // [heads * qkDim]
    let qLat: MTLBuffer                  // [heads * kvRank]
    let oLat: MTLBuffer                  // [heads * kvRank]
    let idxKRaw: MTLBuffer               // [idxDim]
    let idxQ: MTLBuffer                  // [idxHeads * idxDim]
    let idxW: MTLBuffer                  // [idxHeads]
    let idxScores: MTLBuffer             // fp32 [pools]
    let selected: MTLBuffer              // uint32
    let routerLogits: MTLBuffer          // fp32 [numExperts]
    let routerIndices: MTLBuffer         // uint32 [topK]
    let routerWeights: MTLBuffer         // fp16 [topK]
    let identityTable: MTLBuffer         // int16 [numExperts], slot_of[e] = e
    let slotOffsets: MTLBuffer           // uint32 [topK]
    let allHit: MTLBuffer                // uint32 [1]
    let ffnGateScratch: MTLBuffer        // [max(sharedF, denseF)]
    let ffnUpScratch: MTLBuffer
    let ffnActScratch: MTLBuffer
    let sharedOut: MTLBuffer
    let moeActs: MTLBuffer
    let mlpOut: MTLBuffer

    /// The open command buffer of the token being produced.
    var stream: MTLCommandBuffer?

    /// Resident experts: every layer's slab is pinned into the GPU's working
    /// set once, here, so no command buffer pays residency for ~171 GB of
    /// buffers on the way in (macOS 15 residency sets).
    var residencySet: (any MTLResidencySet)?

    /// The batched prefill (`Glm53PrefillEngine`), built on first use when
    /// the experts are resident. `MFERENCE_GLM53_BATCHED_PREFILL=0` keeps
    /// every prompt token on the per-token path (the exactness reference).
    var batchedPrefill: Glm53PrefillEngine?
    public var batchedPrefillEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_GLM53_BATCHED_PREFILL"] != "0"

    var position = 0
    var inSequentialPrefill = false
    var slotBudgetChecked = false

    public var continuationPosition: Int { position }

    // MARK: - Phase counters (MFERENCE_PHASES=1)

    private static let phaseInstrumentationEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_PHASES"] == "1"

    /// Wall time inside `fetchExperts` (slot cache only): the eight routed
    /// expert blobs per MoE layer, nothing overlapping them. Zero when resident.
    public private(set) var totalIoNanos: UInt64 = 0
    /// Wall time in the indexer's CPU selection (readback, sort, expand).
    public private(set) var totalIndexerTopKNanos: UInt64 = 0
    /// Wall time waiting on the router readback before an expert fetch (slot
    /// cache only); zero when resident, where the router never leaves the GPU.
    public private(set) var totalRouterNanos: UInt64 = 0
    /// Command buffers committed in the window.
    public private(set) var totalCommandBuffers = 0
    let gpuTimeLock = NSLock()
    public private(set) var totalGpuBusyNanos: UInt64 = 0
    var gpuSpanFirstStart: Double = .infinity
    var gpuSpanLastEnd: Double = 0

    public var totalGpuSpanNanos: UInt64 {
        gpuTimeLock.lock()
        defer { gpuTimeLock.unlock() }
        guard gpuSpanLastEnd > gpuSpanFirstStart else { return 0 }
        return UInt64((gpuSpanLastEnd - gpuSpanFirstStart) * 1e9)
    }

    public func beginDecodePhaseWindow() {
        totalIoNanos = 0
        totalIndexerTopKNanos = 0
        totalRouterNanos = 0
        totalCommandBuffers = 0
        gpuTimeLock.lock()
        totalGpuBusyNanos = 0
        gpuSpanFirstStart = .infinity
        gpuSpanLastEnd = 0
        gpuTimeLock.unlock()
    }

    @inline(__always)
    private static func phaseClock() -> UInt64 {
        guard phaseInstrumentationEnabled else { return 0 }
        return clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }

    @inline(__always)
    private func trackGpuInterval(_ cb: MTLCommandBuffer) {
        guard Self.phaseInstrumentationEnabled else { return }
        totalCommandBuffers += 1
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
        self.g53 = cfg.glm53
        self.maxContext = maxContext
        hidden = cfg.hiddenSize
        hc = cfg.hyperConnections.mult
        let la = cfg.linearAttention
        kdaHeads = la.numVHeads
        kdaDim = la.valueHeadDim
        convTaps = la.convKernelSize
        numHeads = cfg.numHeads
        qkDim = cfg.glm53.qkNopeHeadDim
        vDim = cfg.glm53.vHeadDim
        kvRank = cfg.glm53.kvLoraRank
        let ca = cfg.compressedAttention
        qRank = ca.qLoraRank
        idxHeads = ca.indexNHeads
        idxDim = ca.indexHeadDim
        idxTopK = ca.indexTopK
        kPool = cfg.glm53.indexKPool
        topK = cfg.topKExperts
        numExperts = cfg.numExperts
        moeF = cfg.moeIntermediateSize
        sharedF = cfg.intermediateSize
        denseF = cfg.denseIntermediateSize
        eps = Float(cfg.glm53.rmsNormEps)
        hcEps = Float(cfg.hyperConnections.eps)

        kernels = try Glm53Kernels(context: context)
        rms = try RMSNorm(context: context)
        int8 = try DequantInt8GEMV(context: context, additionalShapes: cfg.decodeInt8GEMVShapes)
        let int4 = try DequantInt4GEMV(context: context, additionalShapes: cfg.decodeInt4GEMVShapes)
        matVec = try FlashNextMatVec(context: context, int4: int4)
        moe = try MoE(context: context, siluActivation: true,
                      specializedD: UInt32(hidden), specializedF: UInt32(moeF),
                      specializedNumExperts: UInt32(numExperts),
                      specializedTopK: UInt32(topK),
                      swigluLimit: Float(cfg.swigluLimit))
        state = try Glm53StateManager(device: context.device, config: cfg, maxContext: maxContext)

        embedding = model.embedding
        finalNorm = model.finalNorm
        lmHead = model.lmHead

        let device = context.device
        func half(_ count: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(1, count) * MemoryLayout<Float16>.stride,
                                            options: .storageModeShared) else { throw MetalError.noDevice }
            return b
        }
        func float(_ count: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(1, count) * MemoryLayout<Float>.stride,
                                            options: .storageModeShared) else { throw MetalError.noDevice }
            return b
        }
        func words(_ count: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(1, count) * MemoryLayout<UInt32>.stride,
                                            options: .storageModeShared) else { throw MetalError.noDevice }
            return b
        }

        // Resident experts: every MoE layer must offer a slab, else the slot
        // cache path serves all of them (one rule for the whole model).
        var slabs: [Int: ResidentExpertSlab] = [:]
        for L in 0..<cfg.numLayers where !cfg.layerIsDenseFFN(L) {
            if let slab = try model.residentExpertSlab(layer: L) { slabs[L] = slab }
        }
        let moeLayerCount = (0..<cfg.numLayers).filter { !cfg.layerIsDenseFFN($0) }.count
        expertsResident = moeLayerCount > 0 && slabs.count == moeLayerCount

        var built: [LayerTensors] = []
        for L in 0..<cfg.numLayers {
            let isKDA = cfg.layerIsKDA(L)
            let isDense = cfg.layerIsDenseFFN(L)
            let prefix = "language_model.model.layers.\(L)."
            built.append(LayerTensors(
                attnNorm: try model.inputNorm(layer: L),
                ffnNorm: try model.postAttnNorm(layer: L),
                // The mHC `fn` is BF16 in the conversion; the kernel reads fp32.
                hcAttnFn: try model.residentAsF32(name: prefix + "attn_hc.fn"),
                hcAttnBase: try model.glm53AttnHCBase(layer: L),
                hcAttnScale: try model.glm53AttnHCScale(layer: L),
                hcFFNFn: try model.residentAsF32(name: prefix + "ffn_hc.fn"),
                hcFFNBase: try model.glm53FFNHCBase(layer: L),
                hcFFNScale: try model.glm53FFNHCScale(layer: L),
                oProj: try model.glm53OProj(layer: L),
                qProj: isKDA ? try model.glm53KDAQProj(layer: L) : nil,
                kProj: isKDA ? try model.glm53KDAKProj(layer: L) : nil,
                vProj: isKDA ? try model.glm53KDAVProj(layer: L) : nil,
                conv: isKDA ? try model.glm53KDAConv1d(layer: L) : nil,
                fA: isKDA ? try model.glm53KDAForgetAProj(layer: L) : nil,
                fB: isKDA ? try model.glm53KDAForgetBProj(layer: L) : nil,
                gA: isKDA ? try model.glm53KDAGateAProj(layer: L) : nil,
                gB: isKDA ? try model.glm53KDAGateBProj(layer: L) : nil,
                bProj: isKDA ? try model.glm53KDABetaProj(layer: L) : nil,
                aLog: isKDA ? try model.glm53KDAALog(layer: L) : nil,
                dtBias: isKDA ? try model.glm53KDADtBias(layer: L) : nil,
                oNorm: isKDA ? try model.glm53KDAOutNorm(layer: L) : nil,
                qA: isKDA ? nil : try model.glm53QAProj(layer: L),
                qANorm: isKDA ? nil : try model.glm53QANorm(layer: L),
                qB: isKDA ? nil : try model.glm53QBProj(layer: L),
                kvA: isKDA ? nil : try model.glm53KVAProj(layer: L),
                kvANorm: isKDA ? nil : try model.glm53KVANorm(layer: L),
                embedQ: isKDA ? nil : try model.glm53EmbedQ(layer: L),
                unembedOut: isKDA ? nil : try model.glm53UnembedOut(layer: L),
                idxQB: isKDA ? nil : try model.glm53IndexerQBProj(layer: L),
                idxK: isKDA ? nil : try model.glm53IndexerKProj(layer: L),
                idxKNormWeight: isKDA ? nil : try model.glm53IndexerKNormWeight(layer: L),
                idxKNormBias: isKDA ? nil : try model.glm53IndexerKNormBias(layer: L),
                idxWeights: isKDA ? nil : try model.glm53IndexerWeightsProj(layer: L),
                idxPoolGate: isKDA ? nil : .from(try model.glm53IndexerPoolGate(layer: L)),
                idxApe: isKDA ? nil : try model.glm53IndexerPoolAPE(layer: L),
                router: isDense ? nil : .from(try model.router(layer: L)),
                routerBias: isDense ? nil : try model.glm53RouterCorrectionBias(layer: L),
                sharedGate: isDense ? nil : try model.sharedExpertGate(layer: L),
                sharedUp: isDense ? nil : try model.sharedExpertUp(layer: L),
                sharedDown: isDense ? nil : try model.sharedExpertDown(layer: L),
                expertOffsets: isDense ? nil : model.routedExpertOffsets(layer: L),
                slab: expertsResident ? slabs[L] : nil,
                denseGate: isDense ? try model.glm53DenseFFN("gate_proj", layer: L) : nil,
                denseUp: isDense ? try model.glm53DenseFFN("up_proj", layer: L) : nil,
                denseDown: isDense ? try model.glm53DenseFFN("down_proj", layer: L) : nil))
        }
        layers = built

        if expertsResident {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.label = "glm53 resident experts"
            descriptor.initialCapacity = slabs.count + 1
            let set = try context.device.makeResidencySet(descriptor: descriptor)
            var seen = Set<ObjectIdentifier>()
            for slab in slabs.values where seen.insert(ObjectIdentifier(slab.buffer)).inserted {
                set.addAllocation(slab.buffer)
            }
            set.commit()
            set.requestResidency()
            context.queue.addResidencySet(set)
            residencySet = set
        }

        streams = try half(hc * hidden)
        streamsAlt = try half(hc * hidden)
        hiddenBuf = try half(hidden)
        normed = try half(hidden)
        meanPre = try float(hc)
        let mp = meanPre.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<hc { mp[i] = 1 / Float(hc) }
        hcPreA = try float(hc); hcPostA = try float(hc); hcCombA = try float(hc * hc)
        hcPreF = try float(hc); hcPostF = try float(hc); hcCombF = try float(hc * hc)
        mixed = try half(3 * kdaHeads * kdaDim)
        convOut = try half(3 * kdaHeads * kdaDim)
        kdaA = try half(kdaHeads * kdaDim)
        kdaB = try half(kdaHeads)
        kdaGate = try half(kdaHeads * kdaDim)
        kdaLow = try half(kdaDim)
        kdaY = try half(kdaHeads * kdaDim)
        attnHeads = try half(max(kdaHeads * kdaDim, numHeads * vDim))
        attnOut = try half(hidden)
        qr = try half(qRank)
        q = try half(numHeads * qkDim)
        qLat = try half(numHeads * kvRank)
        oLat = try half(numHeads * kvRank)
        idxKRaw = try half(idxDim)
        idxQ = try half(idxHeads * idxDim)
        idxW = try half(idxHeads)
        idxScores = try float(maxContext / max(kPool, 1) + 1)
        selected = try words(idxTopK + kPool)
        routerLogits = try float(numExperts)
        routerIndices = try words(topK)
        routerWeights = try half(topK)
        guard let table = device.makeBuffer(length: numExperts * MemoryLayout<Int16>.stride,
                                            options: .storageModeShared) else { throw MetalError.noDevice }
        let tablePtr = table.contents().assumingMemoryBound(to: Int16.self)
        for e in 0..<numExperts { tablePtr[e] = Int16(e) }
        identityTable = table
        slotOffsets = try words(topK)
        allHit = try words(1)
        let ffnWidth = max(sharedF, denseF)
        ffnGateScratch = try half(ffnWidth)
        ffnUpScratch = try half(ffnWidth)
        ffnActScratch = try half(ffnWidth)
        sharedOut = try half(hidden)
        moeActs = try half(topK * moeF)
        mlpOut = try half(hidden)
        reset()
    }

    private static func validate(config: ArchConfig, maxContext: Int) throws {
        guard config.family == .glm53Flash, config.hasGlm53Axes else {
            throw Glm53ForwardRunnerError.invalidConfiguration(
                "Glm53ForwardRunner requires the glm53Flash family")
        }
        let la = config.linearAttention
        guard la.numKHeads == la.numVHeads, la.keyHeadDim == la.valueHeadDim,
              la.keyHeadDim % 32 == 0, la.keyHeadDim <= Glm53Kernels.maxKDAHeadDim else {
            throw Glm53ForwardRunnerError.invalidConfiguration(
                "glm53_kda_decode is written for square KDA heads of at most \(Glm53Kernels.maxKDAHeadDim), got \(la)")
        }
        guard la.convKernelSize >= 2 else {
            throw Glm53ForwardRunnerError.invalidConfiguration("the depthwise conv needs at least 2 taps")
        }
        guard config.glm53.kvLoraRank % 32 == 0, config.glm53.kvLoraRank <= Glm53Kernels.maxLatentDim else {
            throw Glm53ForwardRunnerError.invalidConfiguration(
                "glm53_latent_attention is written for latents of at most \(Glm53Kernels.maxLatentDim), multiples of 32")
        }
        guard config.glm53.indexKPool >= 1, config.glm53.indexKPool <= 8 else {
            throw Glm53ForwardRunnerError.invalidConfiguration("index_kpool must be 1...8")
        }
        guard config.hyperConnections.mult <= 4 else {
            throw Glm53ForwardRunnerError.invalidConfiguration("hc_mult above 4 is not supported")
        }
        guard config.topKExperts == MoE.maxStreamedExperts, config.numExperts <= Int(Int16.max) else {
            throw Glm53ForwardRunnerError.invalidConfiguration(
                "the GLM-5.3 router and expert path are written for top-\(MoE.maxStreamedExperts) "
                + "over at most \(Int16.max) experts; got top-\(config.topKExperts) of \(config.numExperts)")
        }
        guard maxContext > 0 else {
            throw Glm53ForwardRunnerError.invalidConfiguration("maxContext must be positive")
        }
    }

    // MARK: - Lifecycle

    public func reset() {
        // Anything still executing writes the state this is about to clear.
        try? sync()
        position = 0
        inSequentialPrefill = false
        state.reset()
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition == position else {
            throw Glm53ForwardRunnerError.invalidInput(
                "continuation expects position \(expectedPosition) but the runner is at \(position)")
        }
    }

    // MARK: - Entry points

    public func produce(token: Int32, position p: Int, into logits: MTLBuffer) async throws {
        try await produceToken(token: token, position: p, into: logits)
        if inSequentialPrefill {
            inSequentialPrefill = false
            beginDecodePhaseWindow()
        }
    }

    func produceWithoutLogits(token: Int32, position p: Int) async throws {
        inSequentialPrefill = true
        try await produceToken(token: token, position: p, into: nil)
        try waitForCommitted()
    }

    func produceExactPrefill(token: Int32, position p: Int, into logits: MTLBuffer) async throws {
        try await produceToken(token: token, position: p, into: logits)
    }

    /// Sequential by construction: the same per-token path as decode.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        guard startPosition == position else {
            throw PrefillError.chunkedUnsupported(
                "GLM-5.3 prefill cursor \(position) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "GLM-5.3 prefill range starting at \(startPosition) with \(tokens.count) "
                + "tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        var done = 0
        var remaining = tokens
        // Batched chunks over the whole prompt (dense attention below
        // index_topk, the indexer's per-query selection past it); the
        // per-token loop below only runs when the batched path is off.
        if batchedPrefillEnabled, expertsResident, capture == nil, !denseSelectionForAB {
            if batchedPrefill == nil { batchedPrefill = try Glm53PrefillEngine(runner: self) }
            if let engine = batchedPrefill {
                // `MFERENCE_GLM53_PREFILL_CHUNK` caps the chunk (diagnostics: 1 isolates
                // kernel arithmetic from anything chunk-size dependent).
                let chunkCap = ProcessInfo.processInfo.environment["MFERENCE_GLM53_PREFILL_CHUNK"]
                    .flatMap { Int($0) }.map { max(1, min($0, Glm53PrefillEngine.capacity)) }
                    ?? Glm53PrefillEngine.capacity
                while !remaining.isEmpty {
                    let n = min(remaining.count, chunkCap)
                    try Task.checkCancellation()
                    let chunk = remaining.prefix(n)
                    let last = n == remaining.count
                    try engine.run(tokens: chunk, startPosition: position, into: last ? logits : nil)
                    position += n
                    remaining = remaining.dropFirst(n)
                    done += n
                    onProgress(done)
                }
            }
        }
        for (i, token) in remaining.enumerated() {
            try Task.checkCancellation()
            let last = i == remaining.count - 1
            try await produceToken(token: token, position: position, into: last ? logits : nil)
            done += 1
            if done % max(1, config.chunkTokens) == 0 || last {
                try waitForCommitted()
                onProgress(done)
            }
        }
        try waitForCommitted()
        beginDecodePhaseWindow()
        return PrefillResult(newPosition: startPosition + tokens.count, seed: .logitsWritten)
    }

    // MARK: - The forward pass

    func produceToken(token: Int32, position p: Int, into logits: MTLBuffer?) async throws {
        guard p == position else {
            throw Glm53ForwardRunnerError.invalidInput("expected position \(position), got \(p)")
        }
        guard p < maxContext else {
            throw Glm53ForwardRunnerError.invalidInput("position \(p) exceeds maxContext \(maxContext)")
        }
        guard token >= 0, Int(token) < cfg.vocabSize else {
            throw Glm53ForwardRunnerError.invalidInput(
                "token \(token) outside the \(cfg.vocabSize)-entry vocabulary")
        }
        try Task.checkCancellation()

        let cb = try open()
        kernels.encodeEmbedLookupInt8(commandBuffer: cb, table: embedding, out: hiddenBuf,
                                      tokenId: UInt32(token), d: hidden)
        try captureHalf("embed_out", hiddenBuf, count: hidden)
        kernels.encodeBroadcastStreams(commandBuffer: try open(), x: hiddenBuf, streams: streams,
                                       hcMult: hc, hidden: hidden)

        for L in 0..<cfg.numLayers {
            try Task.checkCancellation()
            try await encodeLayer(L, position: p)
        }

        let tail = try open()
        kernels.encodeHCCollapse(commandBuffer: tail, streams: streams, pre: meanPre, x: hiddenBuf,
                                 hcMult: hc, hidden: hidden)
        rms.encodeBF16W(commandBuffer: tail, x: hiddenBuf,
                        weight: finalNorm.buffer, weightOffset: Int(finalNorm.offset),
                        out: normed, d: UInt32(hidden), eps: eps)
        try captureHalf("final_norm_out", normed, count: hidden)
        if let logits {
            gemvInt8(try open(), lmHead, x: normed, y: logits, m: cfg.vocabSize, n: hidden)
            try sync()
            if capture != nil {
                capture?.floats["logits"] = Self.readFP16(logits, count: cfg.vocabSize)
            }
        } else {
            // Headless prefill: nothing is read back, so the token's stream is
            // committed without waiting and the next token encodes while it
            // runs. One queue executes in order, and every token reuses the
            // same scratch through GPU-ordered dispatches, so the result is
            // the sequential one bit for bit. The chunk waits at its end.
            try flush()
        }
        position += 1
    }

    private func encodeLayer(_ L: Int, position p: Int) async throws {
        let layer = layers[L]
        let key = String(format: "layer%02d.", L)
        if capture != nil {
            try sync()
            capture?.floats[key + "stream_in"] = Self.readFP16(streams, count: hc * hidden)
        }

        // ---- attention site: mixes, collapse, norm ----
        var cb = try open()
        kernels.encodeHCWeights(commandBuffer: cb, streams: streams,
                                fn: layer.hcAttnFn, base: layer.hcAttnBase, scale: layer.hcAttnScale,
                                outPre: hcPreA, outPost: hcPostA, outComb: hcCombA,
                                hcMult: hc, hidden: hidden,
                                sinkhornIters: cfg.hyperConnections.sinkhornIters,
                                hcEps: hcEps, rmsEps: eps)
        try captureFloat(key + "attn_hc_pre", hcPreA, count: hc)
        try captureFloat(key + "attn_hc_post", hcPostA, count: hc)
        try captureFloat(key + "attn_hc_comb", hcCombA, count: hc * hc)
        cb = try open()
        kernels.encodeHCCollapse(commandBuffer: cb, streams: streams, pre: hcPreA, x: hiddenBuf,
                                 hcMult: hc, hidden: hidden)
        try captureHalf(key + "attn_collapsed", hiddenBuf, count: hidden)
        cb = try open()
        rms.encodeBF16W(commandBuffer: cb, x: hiddenBuf,
                        weight: layer.attnNorm.buffer, weightOffset: Int(layer.attnNorm.offset),
                        out: normed, d: UInt32(hidden), eps: eps)
        try captureHalf(key + "input_layernorm_out", normed, count: hidden)

        if cfg.layerIsKDA(L) {
            try encodeKDA(layer: layer, index: L, key: key)
        } else {
            try encodeSparseAttention(layer: layer, index: L, position: p, key: key)
        }
        try captureHalf(key + "attn_out", attnOut, count: hidden)
        cb = try open()
        kernels.encodeHCPlaceMix(commandBuffer: cb, streams: streams, sub: attnOut,
                                 post: hcPostA, comb: hcCombA, outStreams: streamsAlt,
                                 hcMult: hc, hidden: hidden)

        // ---- FFN site ----
        kernels.encodeHCWeights(commandBuffer: cb, streams: streamsAlt,
                                fn: layer.hcFFNFn, base: layer.hcFFNBase, scale: layer.hcFFNScale,
                                outPre: hcPreF, outPost: hcPostF, outComb: hcCombF,
                                hcMult: hc, hidden: hidden,
                                sinkhornIters: cfg.hyperConnections.sinkhornIters,
                                hcEps: hcEps, rmsEps: eps)
        try captureFloat(key + "ffn_hc_pre", hcPreF, count: hc)
        try captureFloat(key + "ffn_hc_post", hcPostF, count: hc)
        try captureFloat(key + "ffn_hc_comb", hcCombF, count: hc * hc)
        cb = try open()
        kernels.encodeHCCollapse(commandBuffer: cb, streams: streamsAlt, pre: hcPreF, x: hiddenBuf,
                                 hcMult: hc, hidden: hidden)
        try captureHalf(key + "ffn_collapsed", hiddenBuf, count: hidden)
        cb = try open()
        rms.encodeBF16W(commandBuffer: cb, x: hiddenBuf,
                        weight: layer.ffnNorm.buffer, weightOffset: Int(layer.ffnNorm.offset),
                        out: normed, d: UInt32(hidden), eps: eps)
        try captureHalf(key + "post_attention_layernorm_out", normed, count: hidden)

        if cfg.layerIsDenseFFN(L) {
            guard let g = layer.denseGate, let u = layer.denseUp, let d = layer.denseDown else {
                throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) has no dense FFN")
            }
            cb = try open()
            gemvInt8(cb, g, x: normed, y: ffnGateScratch, m: denseF, n: hidden)
            gemvInt8(cb, u, x: normed, y: ffnUpScratch, m: denseF, n: hidden)
            kernels.encodeSwigluClampMul(commandBuffer: cb, gate: ffnGateScratch, up: ffnUpScratch,
                                         out: ffnActScratch, n: denseF, limit: Float(cfg.swigluLimit))
            gemvInt8(cb, d, x: ffnActScratch, y: mlpOut, m: hidden, n: denseF)
        } else {
            try await encodeMoE(layer: layer, index: L, key: key)
        }
        try captureHalf(key + "mlp_out", mlpOut, count: hidden)
        cb = try open()
        kernels.encodeHCPlaceMix(commandBuffer: cb, streams: streamsAlt, sub: mlpOut,
                                 post: hcPostF, comb: hcCombF, outStreams: streams,
                                 hcMult: hc, hidden: hidden)
        if capture != nil {
            try sync()
            capture?.floats[key + "stream_out"] = Self.readFP16(streams, count: hc * hidden)
        }
    }

    // MARK: - Kimi Delta Attention

    private func encodeKDA(layer: LayerTensors, index L: Int, key: String) throws {
        guard let qP = layer.qProj, let kP = layer.kProj, let vP = layer.vProj, let conv = layer.conv,
              let fA = layer.fA, let fB = layer.fB, let gA = layer.gA, let gB = layer.gB,
              let bP = layer.bProj, let aLog = layer.aLog, let dtBias = layer.dtBias,
              let oNorm = layer.oNorm, let tail = state.convTail[L], let st = state.kdaState[L] else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) is not a KDA layer")
        }
        let qkv = kdaHeads * kdaDim
        let rowBytes = qkv * MemoryLayout<Float16>.stride
        var cb = try open()
        gemvInt8(cb, qP, x: normed, y: mixed, m: qkv, n: hidden)
        gemvInt8(cb, kP, x: normed, y: mixed, yOffset: rowBytes, m: qkv, n: hidden)
        gemvInt8(cb, vP, x: normed, y: mixed, yOffset: 2 * rowBytes, m: qkv, n: hidden)
        try captureHalf(key + "kda_mixed", mixed, count: 3 * qkv)
        cb = try open()
        kernels.encodeConvDecode(commandBuffer: cb, tail: tail, mixed: mixed,
                                 convWeight: conv.buffer, convWeightOffset: Int(conv.offset),
                                 out: convOut, channels: 3 * qkv, taps: convTaps)
        try captureHalf(key + "kda_conv_out", convOut, count: 3 * qkv)
        cb = try open()
        gemvInt8(cb, fA, x: normed, y: kdaLow, m: kdaDim, n: hidden)
        gemvInt8(cb, fB, x: kdaLow, y: kdaA, m: qkv, n: kdaDim)
        gemvInt8(cb, gA, x: normed, y: kdaLow, m: kdaDim, n: hidden)
        gemvInt8(cb, gB, x: kdaLow, y: kdaGate, m: qkv, n: kdaDim)
        gemvInt8(cb, bP, x: normed, y: kdaB, m: kdaHeads, n: hidden)
        try captureHalf(key + "kda_gate", kdaGate, count: qkv)
        cb = try open()
        kernels.encodeKDADecode(commandBuffer: cb, convOut: convOut, a: kdaA, b: kdaB, gate: kdaGate,
                                aLog: aLog, dtBias: dtBias, oNorm: oNorm, state: st,
                                out: attnHeads, yOut: kdaY, heads: kdaHeads, headDim: kdaDim,
                                lowerBound: Float(g53.kdaGateLowerBound), eps: eps)
        try captureHalf(key + "kda_y", kdaY, count: qkv)
        gemvInt8(try open(), layer.oProj, x: attnHeads, y: attnOut, m: hidden, n: qkv)
    }

    // MARK: - NoPE latent sparse attention

    private func encodeSparseAttention(layer: LayerTensors, index L: Int, position p: Int,
                                       key: String) throws {
        guard let qA = layer.qA, let qANorm = layer.qANorm, let qB = layer.qB, let kvA = layer.kvA,
              let kvANorm = layer.kvANorm, let embedQ = layer.embedQ, let unembed = layer.unembedOut,
              let latents = state.latents[L] else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) is not a sparse layer")
        }
        let T = p + 1
        var cb = try open()
        gemvInt8(cb, qA, x: normed, y: qr, m: qRank, n: hidden)
        rms.encodeBF16W(commandBuffer: cb, x: qr, weight: qANorm.buffer, weightOffset: Int(qANorm.offset),
                        out: qr, d: UInt32(qRank), eps: eps)
        try captureHalf(key + "dsa_qr", qr, count: qRank)
        cb = try open()
        gemvInt8(cb, qB, x: qr, y: q, m: numHeads * qkDim, n: qRank)
        try captureHalf(key + "dsa_q", q, count: numHeads * qkDim)
        let latentOffset = p * kvRank * MemoryLayout<Float16>.stride
        cb = try open()
        gemvInt8(cb, kvA, x: normed, y: latents, yOffset: latentOffset, m: kvRank, n: hidden)
        rms.encodeBF16W(commandBuffer: cb, x: latents, xOffset: latentOffset,
                        weight: kvANorm.buffer, weightOffset: Int(kvANorm.offset),
                        out: latents, outOffset: latentOffset, d: UInt32(kvRank), eps: eps)
        try captureHalf(key + "dsa_latent_new", latents, offset: latentOffset, count: kvRank)

        let selectedCount = try encodeIndexer(layer: layer, index: L, position: p, key: key)

        cb = try open()
        kernels.encodeHeadedInt8GEMV(commandBuffer: cb, weights: embedQ, x: q, y: qLat,
                                     heads: numHeads, m: kvRank, n: qkDim)
        try captureHalf(key + "dsa_q_latent", qLat, count: numHeads * kvRank)
        cb = try open()
        kernels.encodeLatentAttention(commandBuffer: cb, qLatent: qLat, latents: latents, selected: selected,
                                      out: oLat, heads: numHeads, latentDim: kvRank,
                                      cachedRows: T, selectedCount: selectedCount,
                                      scale: Float(cfg.attentionScale))
        kernels.encodeHeadedInt8GEMV(commandBuffer: cb, weights: unembed, x: oLat, y: attnHeads,
                                     heads: numHeads, m: vDim, n: kvRank)
        gemvInt8(cb, layer.oProj, x: attnHeads, y: attnOut, m: hidden, n: numHeads * vDim)
    }

    /// Appends this token's indexer key and pooling gate, pools a completed
    /// group, and — once the cache holds more than `index_topk` tokens —
    /// scores the complete pools and selects on the CPU. Returns the count of
    /// rows in `selected`, or `Glm53Kernels.attendAll` for the dense bypass.
    private func encodeIndexer(layer: LayerTensors, index L: Int, position p: Int,
                               key: String) throws -> UInt32 {
        guard let wk = layer.idxK, let kw = layer.idxKNormWeight, let kb = layer.idxKNormBias,
              let gate = layer.idxPoolGate, let ape = layer.idxApe, let wqB = layer.idxQB,
              let wproj = layer.idxWeights,
              let keys = state.indexKeys[L], let gates = state.indexGates[L], let pooled = state.pooledKeys[L] else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) has no indexer")
        }
        let T = p + 1
        let rowOffset = p * idxDim * MemoryLayout<Float16>.stride
        var cb = try open()
        gemvInt8(cb, wk, x: normed, y: idxKRaw, m: idxDim, n: hidden)
        kernels.encodeLayerNormBias(commandBuffer: cb, x: idxKRaw, weight: kw, bias: kb,
                                    out: keys, outOffset: rowOffset, d: idxDim,
                                    eps: Float(g53.indexerKNormEps))
        matVec.encode(commandBuffer: cb, matrix: gate, x: normed, y: gates, yOffset: rowOffset,
                      rows: idxDim, cols: hidden)
        try captureHalf(key + "idx_k_new", keys, offset: rowOffset, count: idxDim)
        try captureHalf(key + "idx_gate_new", gates, offset: rowOffset, count: idxDim)
        if T % kPool == 0 {
            cb = try open()
            kernels.encodePoolKeys(commandBuffer: cb, keys: keys, gates: gates, ape: ape, pooled: pooled,
                                   pool: T / kPool - 1, kPool: kPool, dim: idxDim)
        }
        if capture != nil { capture?.integers[key + "idx_visible"] = [T] }
        guard T > idxTopK else {
            if capture != nil { capture?.selections[key + "idx_selected"] = .some(nil) }
            return Glm53Kernels.attendAll
        }
        if denseSelectionForAB {
            throw Glm53ForwardRunnerError.invalidInput(
                "dense A/B: \(T) cached tokens exceed index_topk \(idxTopK); dense attention is only the model below that")
        }
        let completePools = T / kPool
        cb = try open()
        gemvInt8(cb, wqB, x: qr, y: idxQ, m: idxHeads * idxDim, n: qRank)
        gemvInt8(cb, wproj, x: normed, y: idxW, m: idxHeads, n: hidden)
        kernels.encodeIndexerScore(commandBuffer: cb, q: idxQ, keys: pooled, weights: idxW, scores: idxScores,
                                   numHeads: idxHeads, indexDim: idxDim, entryCount: completePools,
                                   headScale: 1 / Float(idxDim).squareRoot(),
                                   weightScale: 1 / Float(idxHeads).squareRoot())
        try sync()

        let tSel = Self.phaseClock()
        let scoresPtr = idxScores.contents().assumingMemoryBound(to: Float.self)
        let scores = (0..<completePools).map { scoresPtr[$0] }
        let picks = Glm53Selection.selectTokens(poolScores: scores, cached: T, kPool: kPool,
                                                indexTopK: idxTopK,
                                                alwaysSelectTail: g53.indexKPoolAlwaysSelectTail)
        let selPtr = selected.contents().assumingMemoryBound(to: UInt32.self)
        for (i, t) in picks.enumerated() { selPtr[i] = UInt32(t) }
        if Self.phaseInstrumentationEnabled { totalIndexerTopKNanos &+= Self.phaseClock() - tSel }
        if capture != nil {
            capture?.floats[key + "idx_scores"] = scores
            capture?.selections[key + "idx_selected"] = .some(picks)
        }
        return UInt32(picks.count)
    }

    // MARK: - MoE

    private func encodeMoE(layer: LayerTensors, index L: Int, key: String) async throws {
        guard let router = layer.router, let bias = layer.routerBias, let sg = layer.sharedGate,
              let su = layer.sharedUp, let sd = layer.sharedDown, let offsets = layer.expertOffsets else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) has no MoE block")
        }
        // Router logits in fp32 from the BF16 gate, the top-8 on the GPU, and
        // the shared expert alongside.
        var cb = try open()
        matVec.encode(commandBuffer: cb, matrix: router, x: normed, y: routerLogits,
                      rows: numExperts, cols: hidden, outputFloat32: true)
        kernels.encodeRouterSelect(commandBuffer: cb, logits: routerLogits, bias: bias,
                                   outIndices: routerIndices, outWeights: routerWeights,
                                   numExperts: numExperts, routeScale: Float(cfg.routedScalingFactor))
        gemvInt8(cb, sg, x: normed, y: ffnGateScratch, m: sharedF, n: hidden)
        gemvInt8(cb, su, x: normed, y: ffnUpScratch, m: sharedF, n: hidden)
        kernels.encodeSwigluClampMul(commandBuffer: cb, gate: ffnGateScratch, up: ffnUpScratch,
                                     out: ffnActScratch, n: sharedF, limit: Float(cfg.swigluLimit))
        gemvInt8(cb, sd, x: ffnActScratch, y: sharedOut, m: hidden, n: sharedF)
        try captureHalf(key + "shared_out", sharedOut, count: hidden)
        if capture != nil {
            try sync()
            let logitsPtr = routerLogits.contents().assumingMemoryBound(to: Float.self)
            let logitsRow = (0..<numExperts).map { logitsPtr[$0] }
            capture?.floats[key + "router_logits"] = logitsRow
            capture?.floats[key + "router_scores"] = logitsRow.map { 1 / (1 + expf(-$0)) }
            capture?.integers[key + "router_indices"] = routedExpertIndices()
            capture?.floats[key + "router_weights"] = Self.readFP16(routerWeights, count: topK)
        }

        if let slab = layer.slab {
            // Resident: the indices resolve to slab offsets on the GPU and the
            // routed FFN runs in the same stream. `y = shared + routed`.
            cb = try open()
            moe.encodeRoutedResidentFFN(
                commandBuffer: cb, slab: slab.buffer, slabOffset: slab.baseOffset,
                expertStride: slab.expertStride, indices: routerIndices, identityTable: identityTable,
                slotOffsets: slotOffsets, allHit: allHit, routedOffsets: offsets,
                x: normed, acts: moeActs, routingWeights: routerWeights,
                residual: sharedOut, y: mlpOut,
                numExperts: UInt32(numExperts), d: UInt32(hidden), f: UInt32(moeF), topK: UInt32(topK))
            return
        }

        // Slot cache: the fetch needs the indices on the CPU.
        let tRoute = Self.phaseClock()
        try sync()
        let experts = routedExpertIndices()
        if Self.phaseInstrumentationEnabled { totalRouterNanos &+= Self.phaseClock() - tRoute }
        try checkSlotBudget(layer: L)
        let tIo = Self.phaseClock()
        let blobs = try await fetchExperts(layer: L, experts: experts)
        if Self.phaseInstrumentationEnabled { totalIoNanos &+= Self.phaseClock() - tIo }
        cb = try open()
        let argBuffer = moe.makeReusedRoutedArgumentBuffer(routedBlobs: blobs, topK: UInt32(topK))
        moe.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: blobs,
            routedOffsets: offsets, x: normed, acts: moeActs,
            d: UInt32(hidden), f: UInt32(moeF), topK: UInt32(topK))
        // The reduce seeds with the shared-expert output: routed + shared.
        moe.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: blobs,
            routedOffsets: offsets, acts: moeActs, routingWeights: routerWeights,
            residual: sharedOut, y: mlpOut, d: UInt32(hidden), f: UInt32(moeF), topK: UInt32(topK))
    }

    /// The router's chosen experts, read after a `sync()`.
    func routedExpertIndices() -> [Int] {
        let ptr = routerIndices.contents().assumingMemoryBound(to: UInt32.self)
        return (0..<topK).map { min(Int(ptr[$0]), numExperts - 1) }
    }

    // MARK: - Expert streaming (slot cache)

    private func checkSlotBudget(layer L: Int) throws {
        guard !slotBudgetChecked else { return }
        slotBudgetChecked = true
        guard let slots = model.routedExpertCacheSlotCount(layer: L) else { return }
        guard slots >= topK else {
            throw Glm53ForwardRunnerError.invalidConfiguration(
                "this family routes top-\(topK) experts per layer but the expert cache has only "
                + "\(slots) slots; use --expert-cache-slots 16 or more")
        }
    }

    private func fetchExperts(layer L: Int, experts: [Int]) async throws -> [(buffer: MTLBuffer, offset: Int)] {
        let views: [TensorView]
        if let plan = try model.planRoutedExpertsIfPossible(layer: L, experts: experts) {
            views = try await model.fetchRoutedExperts(plan: plan)
        } else {
            views = try await model.fetchRoutedExperts(layer: L, experts: experts)
        }
        return views.map { (buffer: $0.buffer, offset: Int($0.offset)) }
    }

    // MARK: - Command stream

    /// The token's open command buffer, opened on demand.
    func open() throws -> MTLCommandBuffer {
        if let stream { return stream }
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw Glm53ForwardRunnerError.commandFailed("no command buffer")
        }
        stream = cb
        return cb
    }

    /// Streams committed without waiting (headless prefill), oldest first.
    var inFlight: [MTLCommandBuffer] = []

    /// Commit the open stream and wait for it and everything committed before
    /// it; the next `open()` starts a new one.
    func sync() throws {
        if let cb = stream {
            stream = nil
            trackGpuInterval(cb)
            cb.commit()
            inFlight.append(cb)
        }
        try waitForCommitted()
    }

    /// Commit the open stream without waiting; the next `open()` starts a new
    /// one that the queue executes after it.
    func flush() throws {
        guard let cb = stream else { return }
        stream = nil
        trackGpuInterval(cb)
        cb.commit()
        inFlight.append(cb)
    }

    /// Wait for every committed stream and surface its error.
    func waitForCommitted() throws {
        let pending = inFlight
        inFlight.removeAll()
        for cb in pending {
            cb.waitUntilCompleted()
            if let error = cb.error {
                throw Glm53ForwardRunnerError.commandFailed("\(error)")
            }
            guard cb.status == .completed else {
                throw Glm53ForwardRunnerError.commandFailed("command buffer status \(cb.status.rawValue)")
            }
        }
    }

    // MARK: - Encoding helpers

    func gemvInt8(_ cb: MTLCommandBuffer, _ view: TensorView,
                          x: MTLBuffer, xOffset: Int = 0,
                          y: MTLBuffer, yOffset: Int = 0, m: Int, n: Int) {
        int8.encode(commandBuffer: cb,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                    m: UInt32(m), n: UInt32(n))
    }

    private func captureHalf(_ key: String, _ buffer: MTLBuffer, offset: Int = 0, count: Int) throws {
        guard capture != nil else { return }
        try sync()
        capture?.floats[key] = Self.readFP16(buffer, offset: offset, count: count)
    }

    private func captureFloat(_ key: String, _ buffer: MTLBuffer, count: Int) throws {
        guard capture != nil else { return }
        try sync()
        let base = buffer.contents().assumingMemoryBound(to: Float.self)
        capture?.floats[key] = (0..<count).map { base[$0] }
    }

    static func readFP16(_ buffer: MTLBuffer, offset: Int = 0, count: Int) -> [Float] {
        let base = buffer.contents().advanced(by: offset).bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(base[$0]) }
    }

    // MARK: - State readbacks (tests)

    func kdaState(layer L: Int) -> [Float] {
        guard let buf = state.kdaState[L] else { return [] }
        let count = kdaHeads * kdaDim * kdaDim
        let base = buf.contents().assumingMemoryBound(to: Float.self)
        return (0..<count).map { base[$0] }
    }
}

/// The CPU-side selection of the GLM-5.3 forward: the pooled indexer's
/// attended set, an exact transcription of the reference (`Glm5NextIndexer`).
/// Ties break toward the lower index, as MLX's stable `argsort` does. The
/// router's selection lives on the GPU (`glm53_router_select_k8`).
enum Glm53Selection {
    /// The indexer's attended token set for the newest query: the top
    /// `indexTopK / kPool` complete pools by score (stable descending order),
    /// expanded to their tokens, plus the incomplete tail. Ascending.
    static func selectTokens(poolScores: [Float], cached T: Int, kPool: Int, indexTopK: Int,
                             alwaysSelectTail: Bool) -> [Int] {
        let complete = T / kPool
        let scores = Array(poolScores.prefix(complete))
        let selectK = min(indexTopK / kPool, complete)
        let order = (0..<complete).sorted { a, b in
            scores[a] != scores[b] ? scores[a] > scores[b] : a < b
        }
        // Marks instead of a set: the result is the ascending token list, and
        // this runs once per query of a prefill chunk.
        var chosen = [Bool](repeating: false, count: complete)
        for j in order.prefix(selectK) { chosen[j] = true }
        var tokens: [Int] = []
        tokens.reserveCapacity(selectK * kPool + kPool)
        for j in 0..<complete where chosen[j] { for c in 0..<kPool { tokens.append(j * kPool + c) } }
        if alwaysSelectTail { for t in (complete * kPool)..<T { tokens.append(t) } }
        return tokens
    }
}
