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
/// # Selection on the CPU
///
/// The pooled indexer's top-k and the router's top-8 are computed on the CPU
/// from FP32 scores read back at the sync the expert fetch already needs
/// (`Glm53Selection`, exact transcriptions of the reference including the
/// stable ascending-index order among equal scores).
///
/// # Prefill
///
/// Chunked prefill and sequential decode share one per-token path, so the two
/// are equal by construction: `prefillChunked` walks the tokens through
/// `produceToken`, carrying the conv tails, the KDA state, the latent and
/// indexer caches across chunk boundaries. PERF, not correctness: a prompt
/// token re-reads its eight expert blobs per layer rather than amortizing them
/// over a chunk. A layer-major expert pass is the first perf item.
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

    // MARK: - Weights

    private struct LayerTensors {
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
        let routerBias: [Float]
        let sharedGate: TensorView?
        let sharedUp: TensorView?
        let sharedDown: TensorView?
        let expertOffsets: MoEExpertOffsets?
        let denseGate: TensorView?
        let denseUp: TensorView?
        let denseDown: TensorView?
    }

    // MARK: - Stored state

    private let model: Model
    private let ctx: MetalContext
    private let cfg: ArchConfig
    private let g53: Glm53Config
    public let maxContext: Int

    private let hidden: Int
    private let hc: Int
    private let kdaHeads: Int
    private let kdaDim: Int
    private let convTaps: Int
    private let numHeads: Int
    private let qkDim: Int
    private let vDim: Int
    private let kvRank: Int
    private let qRank: Int
    private let idxHeads: Int
    private let idxDim: Int
    private let idxTopK: Int
    private let kPool: Int
    private let topK: Int
    private let numExperts: Int
    private let moeF: Int
    private let sharedF: Int
    private let denseF: Int
    private let eps: Float
    private let hcEps: Float

    private let kernels: Glm53Kernels
    private let rms: RMSNorm
    private let int8: DequantInt8GEMV
    private let matVec: FlashNextMatVec
    private let moe: MoE
    private let state: Glm53StateManager

    private let layers: [LayerTensors]
    private let embedding: TensorView
    private let finalNorm: TensorView
    private let lmHead: TensorView

    // Scratch (FP16 unless noted)
    private let streams: MTLBuffer
    private let streamsAlt: MTLBuffer
    private let hiddenBuf: MTLBuffer
    private let normed: MTLBuffer
    private let meanPre: MTLBuffer               // fp32 [hc] = 1/hc
    private let hcPreA: MTLBuffer                // fp32
    private let hcPostA: MTLBuffer
    private let hcCombA: MTLBuffer
    private let hcPreF: MTLBuffer
    private let hcPostF: MTLBuffer
    private let hcCombF: MTLBuffer
    private let mixed: MTLBuffer                 // [3 * H * D]
    private let convOut: MTLBuffer
    private let kdaA: MTLBuffer                  // [H * D]
    private let kdaB: MTLBuffer                  // [H]
    private let kdaGate: MTLBuffer               // [H * D]
    private let kdaLow: MTLBuffer                // [D]
    private let kdaY: MTLBuffer                  // [H * D]
    private let attnHeads: MTLBuffer             // [max(H*D, heads*vDim)]
    private let attnOut: MTLBuffer               // [hidden]
    private let qr: MTLBuffer                    // [qRank]
    private let q: MTLBuffer                     // [heads * qkDim]
    private let qLat: MTLBuffer                  // [heads * kvRank]
    private let oLat: MTLBuffer                  // [heads * kvRank]
    private let idxKRaw: MTLBuffer               // [idxDim]
    private let idxQ: MTLBuffer                  // [idxHeads * idxDim]
    private let idxW: MTLBuffer                  // [idxHeads]
    private let idxScores: MTLBuffer             // fp32 [pools]
    private let selected: MTLBuffer              // uint32
    private let routerLogits: MTLBuffer          // fp32 [numExperts]
    private let routerWeights: MTLBuffer         // [topK]
    private let ffnGateScratch: MTLBuffer        // [max(sharedF, denseF)]
    private let ffnUpScratch: MTLBuffer
    private let ffnActScratch: MTLBuffer
    private let sharedOut: MTLBuffer
    private let moeActs: MTLBuffer
    private let mlpOut: MTLBuffer

    private var position = 0
    private var inSequentialPrefill = false
    private var slotBudgetChecked = false

    public var continuationPosition: Int { position }

    // MARK: - Phase counters (MFERENCE_PHASES=1)

    private static let phaseInstrumentationEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_PHASES"] == "1"

    /// Wall time inside `fetchExperts`: the eight routed expert blobs per MoE
    /// layer, nothing overlapping them.
    public private(set) var totalIoNanos: UInt64 = 0
    /// Wall time in the indexer's CPU selection (readback, sort, expand).
    public private(set) var totalIndexerTopKNanos: UInt64 = 0
    /// Wall time in the router's CPU top-8 (logit readback included).
    public private(set) var totalRouterNanos: UInt64 = 0
    /// Command buffers committed in the window.
    public private(set) var totalCommandBuffers = 0
    private let gpuTimeLock = NSLock()
    public private(set) var totalGpuBusyNanos: UInt64 = 0
    private var gpuSpanFirstStart: Double = .infinity
    private var gpuSpanLastEnd: Double = 0

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

        var built: [LayerTensors] = []
        for L in 0..<cfg.numLayers {
            let isKDA = cfg.layerIsKDA(L)
            let isDense = cfg.layerIsDenseFFN(L)
            var bias: [Float] = []
            if !isDense {
                let biasView = try model.glm53RouterCorrectionBias(layer: L)
                let ptr = biasView.buffer.contents().advanced(by: Int(biasView.offset))
                    .assumingMemoryBound(to: Float.self)
                bias = (0..<cfg.numExperts).map { ptr[$0] }
            }
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
                routerBias: bias,
                sharedGate: isDense ? nil : try model.sharedExpertGate(layer: L),
                sharedUp: isDense ? nil : try model.sharedExpertUp(layer: L),
                sharedDown: isDense ? nil : try model.sharedExpertDown(layer: L),
                expertOffsets: isDense ? nil : model.routedExpertOffsets(layer: L),
                denseGate: isDense ? try model.glm53DenseFFN("gate_proj", layer: L) : nil,
                denseUp: isDense ? try model.glm53DenseFFN("up_proj", layer: L) : nil,
                denseDown: isDense ? try model.glm53DenseFFN("down_proj", layer: L) : nil))
        }
        layers = built

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
        guard let sel = device.makeBuffer(length: (idxTopK + kPool) * MemoryLayout<UInt32>.stride,
                                          options: .storageModeShared) else { throw MetalError.noDevice }
        selected = sel
        routerLogits = try float(numExperts)
        routerWeights = try half(topK)
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
        guard MoE.routedComputeWidths.contains(UInt32(config.topKExperts)) else {
            throw Glm53ForwardRunnerError.invalidConfiguration(
                "INT4 routed experts at top-\(config.topKExperts); the reduce implements "
                + "\(MoE.routedComputeWidths.sorted())")
        }
        guard maxContext > 0 else {
            throw Glm53ForwardRunnerError.invalidConfiguration("maxContext must be positive")
        }
    }

    // MARK: - Lifecycle

    public func reset() {
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
        for (i, token) in tokens.enumerated() {
            try Task.checkCancellation()
            let last = i == tokens.count - 1
            try await produceToken(token: token, position: startPosition + i, into: last ? logits : nil)
            done += 1
            if done % max(1, config.chunkTokens) == 0 || last { onProgress(done) }
        }
        beginDecodePhaseWindow()
        return PrefillResult(newPosition: startPosition + tokens.count, seed: .logitsWritten)
    }

    // MARK: - The forward pass

    private func produceToken(token: Int32, position p: Int, into logits: MTLBuffer?) async throws {
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

        var cb = try makeCommandBuffer()
        kernels.encodeEmbedLookupInt8(commandBuffer: cb, table: embedding, out: hiddenBuf,
                                      tokenId: UInt32(token), d: hidden)
        try captureHalf(&cb, "embed_out", hiddenBuf, count: hidden)
        kernels.encodeBroadcastStreams(commandBuffer: cb, x: hiddenBuf, streams: streams,
                                       hcMult: hc, hidden: hidden)
        try finish(cb)

        for L in 0..<cfg.numLayers {
            try Task.checkCancellation()
            try await encodeLayer(L, position: p)
        }

        cb = try makeCommandBuffer()
        kernels.encodeHCCollapse(commandBuffer: cb, streams: streams, pre: meanPre, x: hiddenBuf,
                                 hcMult: hc, hidden: hidden)
        rms.encodeBF16W(commandBuffer: cb, x: hiddenBuf,
                        weight: finalNorm.buffer, weightOffset: Int(finalNorm.offset),
                        out: normed, d: UInt32(hidden), eps: eps)
        try captureHalf(&cb, "final_norm_out", normed, count: hidden)
        if let logits {
            gemvInt8(cb, lmHead, x: normed, y: logits, m: cfg.vocabSize, n: hidden)
        }
        try finish(cb)
        if capture != nil, let logits {
            capture?.floats["logits"] = Self.readFP16(logits, count: cfg.vocabSize)
        }
        position += 1
    }

    private func encodeLayer(_ L: Int, position p: Int) async throws {
        let layer = layers[L]
        let key = String(format: "layer%02d.", L)
        var cb = try makeCommandBuffer()
        if capture != nil {
            capture?.floats[key + "stream_in"] = Self.readFP16(streams, count: hc * hidden)
        }

        // ---- attention site: mixes, collapse, norm ----
        kernels.encodeHCWeights(commandBuffer: cb, streams: streams,
                                fn: layer.hcAttnFn, base: layer.hcAttnBase, scale: layer.hcAttnScale,
                                outPre: hcPreA, outPost: hcPostA, outComb: hcCombA,
                                hcMult: hc, hidden: hidden,
                                sinkhornIters: cfg.hyperConnections.sinkhornIters,
                                hcEps: hcEps, rmsEps: eps)
        try captureFloat(&cb, key + "attn_hc_pre", hcPreA, count: hc)
        try captureFloat(&cb, key + "attn_hc_post", hcPostA, count: hc)
        try captureFloat(&cb, key + "attn_hc_comb", hcCombA, count: hc * hc)
        kernels.encodeHCCollapse(commandBuffer: cb, streams: streams, pre: hcPreA, x: hiddenBuf,
                                 hcMult: hc, hidden: hidden)
        try captureHalf(&cb, key + "attn_collapsed", hiddenBuf, count: hidden)
        rms.encodeBF16W(commandBuffer: cb, x: hiddenBuf,
                        weight: layer.attnNorm.buffer, weightOffset: Int(layer.attnNorm.offset),
                        out: normed, d: UInt32(hidden), eps: eps)
        try captureHalf(&cb, key + "input_layernorm_out", normed, count: hidden)

        if cfg.layerIsKDA(L) {
            try encodeKDA(&cb, layer: layer, index: L, key: key)
        } else {
            try encodeSparseAttention(&cb, layer: layer, index: L, position: p, key: key)
        }
        try captureHalf(&cb, key + "attn_out", attnOut, count: hidden)
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
        try captureFloat(&cb, key + "ffn_hc_pre", hcPreF, count: hc)
        try captureFloat(&cb, key + "ffn_hc_post", hcPostF, count: hc)
        try captureFloat(&cb, key + "ffn_hc_comb", hcCombF, count: hc * hc)
        kernels.encodeHCCollapse(commandBuffer: cb, streams: streamsAlt, pre: hcPreF, x: hiddenBuf,
                                 hcMult: hc, hidden: hidden)
        try captureHalf(&cb, key + "ffn_collapsed", hiddenBuf, count: hidden)
        rms.encodeBF16W(commandBuffer: cb, x: hiddenBuf,
                        weight: layer.ffnNorm.buffer, weightOffset: Int(layer.ffnNorm.offset),
                        out: normed, d: UInt32(hidden), eps: eps)
        try captureHalf(&cb, key + "post_attention_layernorm_out", normed, count: hidden)

        if cfg.layerIsDenseFFN(L) {
            guard let g = layer.denseGate, let u = layer.denseUp, let d = layer.denseDown else {
                throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) has no dense FFN")
            }
            gemvInt8(cb, g, x: normed, y: ffnGateScratch, m: denseF, n: hidden)
            gemvInt8(cb, u, x: normed, y: ffnUpScratch, m: denseF, n: hidden)
            kernels.encodeSwigluClampMul(commandBuffer: cb, gate: ffnGateScratch, up: ffnUpScratch,
                                         out: ffnActScratch, n: denseF, limit: Float(cfg.swigluLimit))
            gemvInt8(cb, d, x: ffnActScratch, y: mlpOut, m: hidden, n: denseF)
            try captureHalf(&cb, key + "mlp_out", mlpOut, count: hidden)
            kernels.encodeHCPlaceMix(commandBuffer: cb, streams: streamsAlt, sub: mlpOut,
                                     post: hcPostF, comb: hcCombF, outStreams: streams,
                                     hcMult: hc, hidden: hidden)
            try finish(cb)
        } else {
            try await encodeMoE(cb, layer: layer, index: L, key: key)
        }
        if capture != nil {
            capture?.floats[key + "stream_out"] = Self.readFP16(streams, count: hc * hidden)
        }
    }

    // MARK: - Kimi Delta Attention

    private func encodeKDA(_ cb: inout MTLCommandBuffer, layer: LayerTensors, index L: Int,
                           key: String) throws {
        guard let qP = layer.qProj, let kP = layer.kProj, let vP = layer.vProj, let conv = layer.conv,
              let fA = layer.fA, let fB = layer.fB, let gA = layer.gA, let gB = layer.gB,
              let bP = layer.bProj, let aLog = layer.aLog, let dtBias = layer.dtBias,
              let oNorm = layer.oNorm, let tail = state.convTail[L], let st = state.kdaState[L] else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) is not a KDA layer")
        }
        let qkv = kdaHeads * kdaDim
        let rowBytes = qkv * MemoryLayout<Float16>.stride
        gemvInt8(cb, qP, x: normed, y: mixed, m: qkv, n: hidden)
        gemvInt8(cb, kP, x: normed, y: mixed, yOffset: rowBytes, m: qkv, n: hidden)
        gemvInt8(cb, vP, x: normed, y: mixed, yOffset: 2 * rowBytes, m: qkv, n: hidden)
        try captureHalf(&cb, key + "kda_mixed", mixed, count: 3 * qkv)
        kernels.encodeConvDecode(commandBuffer: cb, tail: tail, mixed: mixed,
                                 convWeight: conv.buffer, convWeightOffset: Int(conv.offset),
                                 out: convOut, channels: 3 * qkv, taps: convTaps)
        try captureHalf(&cb, key + "kda_conv_out", convOut, count: 3 * qkv)
        gemvInt8(cb, fA, x: normed, y: kdaLow, m: kdaDim, n: hidden)
        gemvInt8(cb, fB, x: kdaLow, y: kdaA, m: qkv, n: kdaDim)
        gemvInt8(cb, gA, x: normed, y: kdaLow, m: kdaDim, n: hidden)
        gemvInt8(cb, gB, x: kdaLow, y: kdaGate, m: qkv, n: kdaDim)
        gemvInt8(cb, bP, x: normed, y: kdaB, m: kdaHeads, n: hidden)
        try captureHalf(&cb, key + "kda_gate", kdaGate, count: qkv)
        kernels.encodeKDADecode(commandBuffer: cb, convOut: convOut, a: kdaA, b: kdaB, gate: kdaGate,
                                aLog: aLog, dtBias: dtBias, oNorm: oNorm, state: st,
                                out: attnHeads, yOut: kdaY, heads: kdaHeads, headDim: kdaDim,
                                lowerBound: Float(g53.kdaGateLowerBound), eps: eps)
        try captureHalf(&cb, key + "kda_y", kdaY, count: qkv)
        gemvInt8(cb, layer.oProj, x: attnHeads, y: attnOut, m: hidden, n: qkv)
    }

    // MARK: - NoPE latent sparse attention

    private func encodeSparseAttention(_ cb: inout MTLCommandBuffer, layer: LayerTensors, index L: Int,
                                       position p: Int, key: String) throws {
        guard let qA = layer.qA, let qANorm = layer.qANorm, let qB = layer.qB, let kvA = layer.kvA,
              let kvANorm = layer.kvANorm, let embedQ = layer.embedQ, let unembed = layer.unembedOut,
              let latents = state.latents[L] else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) is not a sparse layer")
        }
        let T = p + 1
        gemvInt8(cb, qA, x: normed, y: qr, m: qRank, n: hidden)
        rms.encodeBF16W(commandBuffer: cb, x: qr, weight: qANorm.buffer, weightOffset: Int(qANorm.offset),
                        out: qr, d: UInt32(qRank), eps: eps)
        try captureHalf(&cb, key + "dsa_qr", qr, count: qRank)
        gemvInt8(cb, qB, x: qr, y: q, m: numHeads * qkDim, n: qRank)
        try captureHalf(&cb, key + "dsa_q", q, count: numHeads * qkDim)
        let latentOffset = p * kvRank * MemoryLayout<Float16>.stride
        gemvInt8(cb, kvA, x: normed, y: latents, yOffset: latentOffset, m: kvRank, n: hidden)
        rms.encodeBF16W(commandBuffer: cb, x: latents, xOffset: latentOffset,
                        weight: kvANorm.buffer, weightOffset: Int(kvANorm.offset),
                        out: latents, outOffset: latentOffset, d: UInt32(kvRank), eps: eps)
        try captureHalf(&cb, key + "dsa_latent_new", latents, offset: latentOffset, count: kvRank)

        let selectedCount = try encodeIndexer(&cb, layer: layer, index: L, position: p, key: key)

        kernels.encodeHeadedInt8GEMV(commandBuffer: cb, weights: embedQ, x: q, y: qLat,
                                     heads: numHeads, m: kvRank, n: qkDim)
        try captureHalf(&cb, key + "dsa_q_latent", qLat, count: numHeads * kvRank)
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
    private func encodeIndexer(_ cb: inout MTLCommandBuffer, layer: LayerTensors, index L: Int,
                               position p: Int, key: String) throws -> UInt32 {
        guard let wk = layer.idxK, let kw = layer.idxKNormWeight, let kb = layer.idxKNormBias,
              let gate = layer.idxPoolGate, let ape = layer.idxApe, let wqB = layer.idxQB,
              let wproj = layer.idxWeights,
              let keys = state.indexKeys[L], let gates = state.indexGates[L], let pooled = state.pooledKeys[L] else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) has no indexer")
        }
        let T = p + 1
        let rowOffset = p * idxDim * MemoryLayout<Float16>.stride
        gemvInt8(cb, wk, x: normed, y: idxKRaw, m: idxDim, n: hidden)
        kernels.encodeLayerNormBias(commandBuffer: cb, x: idxKRaw, weight: kw, bias: kb,
                                    out: keys, outOffset: rowOffset, d: idxDim,
                                    eps: Float(g53.indexerKNormEps))
        matVec.encode(commandBuffer: cb, matrix: gate, x: normed, y: gates, yOffset: rowOffset,
                      rows: idxDim, cols: hidden)
        try captureHalf(&cb, key + "idx_k_new", keys, offset: rowOffset, count: idxDim)
        try captureHalf(&cb, key + "idx_gate_new", gates, offset: rowOffset, count: idxDim)
        if T % kPool == 0 {
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
        gemvInt8(cb, wqB, x: qr, y: idxQ, m: idxHeads * idxDim, n: qRank)
        gemvInt8(cb, wproj, x: normed, y: idxW, m: idxHeads, n: hidden)
        kernels.encodeIndexerScore(commandBuffer: cb, q: idxQ, keys: pooled, weights: idxW, scores: idxScores,
                                   numHeads: idxHeads, indexDim: idxDim, entryCount: completePools,
                                   headScale: 1 / Float(idxDim).squareRoot(),
                                   weightScale: 1 / Float(idxHeads).squareRoot())
        try finish(cb)
        cb = try makeCommandBuffer()

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

    private func encodeMoE(_ cbIn: MTLCommandBuffer, layer: LayerTensors, index L: Int, key: String) async throws {
        var cb = cbIn
        guard let router = layer.router, let sg = layer.sharedGate, let su = layer.sharedUp,
              let sd = layer.sharedDown, let offsets = layer.expertOffsets else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) has no MoE block")
        }
        // Router logits in fp32 from the BF16 gate; the shared expert alongside.
        matVec.encode(commandBuffer: cb, matrix: router, x: normed, y: routerLogits,
                      rows: numExperts, cols: hidden, outputFloat32: true)
        gemvInt8(cb, sg, x: normed, y: ffnGateScratch, m: sharedF, n: hidden)
        gemvInt8(cb, su, x: normed, y: ffnUpScratch, m: sharedF, n: hidden)
        kernels.encodeSwigluClampMul(commandBuffer: cb, gate: ffnGateScratch, up: ffnUpScratch,
                                     out: ffnActScratch, n: sharedF, limit: Float(cfg.swigluLimit))
        gemvInt8(cb, sd, x: ffnActScratch, y: sharedOut, m: hidden, n: sharedF)
        try captureHalf(&cb, key + "shared_out", sharedOut, count: hidden)
        try finish(cb)

        let tRoute = Self.phaseClock()
        let logitsPtr = routerLogits.contents().assumingMemoryBound(to: Float.self)
        let logitsRow = (0..<numExperts).map { logitsPtr[$0] }
        let route = Glm53Selection.route(logits: logitsRow, bias: layer.routerBias, topK: topK,
                                         routeScale: Float(cfg.routedScalingFactor))
        let weightsPtr = routerWeights.contents().assumingMemoryBound(to: Float16.self)
        for (i, w) in route.weights.enumerated() { weightsPtr[i] = Float16(w) }
        if Self.phaseInstrumentationEnabled { totalRouterNanos &+= Self.phaseClock() - tRoute }
        if capture != nil {
            capture?.floats[key + "router_logits"] = logitsRow
            capture?.floats[key + "router_scores"] = route.scores
            capture?.integers[key + "router_indices"] = route.experts
            capture?.floats[key + "router_weights"] = route.weights
        }

        try checkSlotBudget(layer: L)
        let tIo = Self.phaseClock()
        let blobs = try await fetchExperts(layer: L, experts: route.experts)
        if Self.phaseInstrumentationEnabled { totalIoNanos &+= Self.phaseClock() - tIo }
        var moeCB = try makeCommandBuffer()
        let argBuffer = moe.makeReusedRoutedArgumentBuffer(routedBlobs: blobs, topK: UInt32(topK))
        moe.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: moeCB, routedArgBuffer: argBuffer, routedBlobs: blobs,
            routedOffsets: offsets, x: normed, acts: moeActs,
            d: UInt32(hidden), f: UInt32(moeF), topK: UInt32(topK))
        // The reduce seeds with the shared-expert output: routed + shared.
        moe.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: moeCB, routedArgBuffer: argBuffer, routedBlobs: blobs,
            routedOffsets: offsets, acts: moeActs, routingWeights: routerWeights,
            residual: sharedOut, y: mlpOut, d: UInt32(hidden), f: UInt32(moeF), topK: UInt32(topK))
        try captureHalf(&moeCB, key + "mlp_out", mlpOut, count: hidden)
        kernels.encodeHCPlaceMix(commandBuffer: moeCB, streams: streamsAlt, sub: mlpOut,
                                 post: hcPostF, comb: hcCombF, outStreams: streams,
                                 hcMult: hc, hidden: hidden)
        try finish(moeCB)
    }

    // MARK: - Expert streaming

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

    // MARK: - Encoding helpers

    private func gemvInt8(_ cb: MTLCommandBuffer, _ view: TensorView,
                          x: MTLBuffer, xOffset: Int = 0,
                          y: MTLBuffer, yOffset: Int = 0, m: Int, n: Int) {
        int8.encode(commandBuffer: cb,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                    m: UInt32(m), n: UInt32(n))
    }

    private func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw Glm53ForwardRunnerError.commandFailed("no command buffer")
        }
        return cb
    }

    private func finish(_ cb: MTLCommandBuffer) throws {
        trackGpuInterval(cb)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            throw Glm53ForwardRunnerError.commandFailed("\(error)")
        }
        guard cb.status == .completed else {
            throw Glm53ForwardRunnerError.commandFailed("command buffer status \(cb.status.rawValue)")
        }
    }

    private func captureHalf(_ cb: inout MTLCommandBuffer, _ key: String, _ buffer: MTLBuffer,
                             offset: Int = 0, count: Int) throws {
        guard capture != nil else { return }
        try finish(cb)
        capture?.floats[key] = Self.readFP16(buffer, offset: offset, count: count)
        cb = try makeCommandBuffer()
    }

    private func captureFloat(_ cb: inout MTLCommandBuffer, _ key: String, _ buffer: MTLBuffer,
                              count: Int) throws {
        guard capture != nil else { return }
        try finish(cb)
        let base = buffer.contents().assumingMemoryBound(to: Float.self)
        capture?.floats[key] = (0..<count).map { base[$0] }
        cb = try makeCommandBuffer()
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

/// The CPU-side selections of the GLM-5.3 forward, exact transcriptions of
/// the reference (`Glm5NextIndexer`, `group_expert_select`). Ties break toward
/// the lower index, as MLX's stable `argsort` does.
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
        var tokens = Set<Int>()
        for j in order.prefix(selectK) { for c in 0..<kPool { tokens.insert(j * kPool + c) } }
        if alwaysSelectTail { for t in (complete * kPool)..<T { tokens.insert(t) } }
        return tokens.sorted()
    }

    struct Route {
        let experts: [Int]
        let weights: [Float]
        let scores: [Float]
    }

    /// Sigmoid scores; selection on `score + bias`; weights are the unbiased
    /// scores of the chosen experts renormalized to one and scaled.
    static func route(logits: [Float], bias: [Float], topK: Int, routeScale: Float) -> Route {
        let scores = logits.map { 1 / (1 + expf(-$0)) }
        let biased = (0..<scores.count).map { scores[$0] + bias[$0] }
        let order = (0..<scores.count).sorted { a, b in
            biased[a] != biased[b] ? biased[a] > biased[b] : a < b
        }
        let chosen = Array(order.prefix(topK))
        var sum: Float = 0
        for e in chosen { sum += scores[e] }
        let weights = chosen.map { scores[$0] / sum * routeScale }
        return Route(experts: chosen, weights: weights, scores: scores)
    }
}
