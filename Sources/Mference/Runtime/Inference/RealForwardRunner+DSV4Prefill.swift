import Foundation
import Metal

// ============================================================================
// DeepSeek-V4 chunked prefill.
//
// The v1 DSV4 prefill replayed `produceTokenDSV4` once per prompt token. That
// is correct but pathologically slow for one reason: routed-expert I/O. Every
// token streams 43 layers x 6 experts x 8 MB, so a 629-token prompt reads
// ~1.3 TB off the expert file and prefill runs at decode speed (measured
// 3.4 tok/s on an M5).
//
// This path keeps the token loop but turns it inside out: **layer-major**.
// For one chunk of T tokens it walks layers on the outside and tokens on the
// inside, which buys two things:
//
//  1. All T tokens' attention-site dispatches for a layer go into ONE command
//     buffer, so the per-layer CPU/GPU round trip is paid once per chunk
//     instead of once per token.
//  2. The layer's routed MoE becomes **pair-major**: the chunk's (token, rank)
//     routes are sorted by expert, split into tiles of at most
//     `kMaxStreamedExperts`, and each tile is streamed in once and applied to
//     every route that lands on it. An expert blob is read once per chunk
//     instead of once per token.
//
// Correctness contract: prefill(T) must equal T sequential decode steps,
// bit-for-bit. That is achieved by construction, not by tolerance:
//
//  * Every dispatch is the *same* kernel the decode path encodes, with the
//    same arguments. Chunk-wide buffers are addressed with byte offsets, which
//    only shifts a base pointer.
//  * Metal executes dispatches within a command buffer in submission order, so
//    replaying token t's chain into a shared scratch buffer before token t+1's
//    is equivalent to running them in separate command buffers. Only state that
//    outlives a token (residual streams, routed input, routing decisions,
//    shared-expert output) is stored per token.
//  * The compressor / window-ring / indexer bookkeeping is the decode
//    bookkeeping, stepped on the CPU while encoding, in token order.
//  * The pair-major MoE kernels reuse `moe.metal`'s INT2 helpers verbatim and
//    reduce residual-first in rank order, matching
//    `moe_phase2_down_reduce_int2_k6` exactly. The one structural change --
//    the down projection lands in an FP32 per-route partial instead of
//    threadgroup memory -- preserves the value bit-for-bit.
//
// Above the lightning-indexer threshold, GPU top-k selects the same ascending
// entry set as decode's CPU reference, keeping each query in the command stream.
// ============================================================================

/// Everything the chunked DSV4 prefill needs from `RealForwardRunner`. Built
/// at the call site inside `prefillChunked`, which is the only place the
/// runner's private members are visible.
struct DSV4PrefillBindings {
    let model: Model
    let ctx: MetalContext
    let cfg: ArchConfig
    let dsv4: DSV4Kernels
    let moeDSV4: MoEDeepseekV4
    let state: DSV4StateManager
    let int4: DequantInt4GEMV
    let rms: RMSNorm
    let embed: EmbedLookupInt4
    let fusionHead: LMHeadChainInt4
    let sharedExperts: [DSV4PrefillSharedExpert]
    let effectiveScales: [MTLBuffer]
    let useFusedGreedyHead: Bool
}

struct DSV4PrefillSharedExpert {
    let gate: SharedExpertInt8Proj
    let up: SharedExpertInt8Proj
    let down: SharedExpertInt8Proj
}

/// A routed tile whose GPU work is still executing while the next tile's
/// experts are being read.
private struct PendingRoutedTile {
    let cb: MTLCommandBuffer
    let assignedSlots: [Int]
}

/// One (token, routing rank) route bound to a tile-local expert slot. Must
/// match `DSV4PrefillRoute` in `moe.metal`.
private struct DSV4PrefillRoute {
    var token: UInt32
    var rank: UInt32
    var localSlot: UInt32
    var reserved: UInt32 = 0
}

final class DSV4ChunkedPrefill {
    /// Experts bound by one routed tile. Matches `kMaxStreamedExperts` in
    /// `moe.metal` (the `RoutedBlobs` argument-buffer struct).
    static let tileExpertCapacity = 8

    /// Tokens whose state-free projections share one weight read.
    static let projectionBatch = DequantInt4GEMVMultiX.maxTokens

    /// Routes per simdgroup in the routed pair kernels
    /// (`kDSV4PrefillGroupRoutes` in `moe.metal`): each expert's route list is
    /// padded to a multiple of this with invalid entries.
    static let groupRoutes = 4

    private let b: DSV4PrefillBindings
    private let chunkTokens: Int

    /// Largest chunk this instance's scratch can serve.
    var chunkCapacity: Int { chunkTokens }

    private let phase1PSO: MTLComputePipelineState
    private let downPSO: MTLComputePipelineState
    private let reducePSO: MTLComputePipelineState
    private let routerGemvPSO: MTLComputePipelineState
    private let routerSelectPSO: MTLComputePipelineState
    private let routerHashPSO: MTLComputePipelineState
    private let argEncoder: MTLArgumentEncoder
    /// One argument buffer per in-flight routed tile. Its contents are read at
    /// GPU execution time, so a tile still running must keep its own.
    private let argBuffers: [MTLBuffer]
    private let tileScheduler: PrefillRoutedTileScheduler
    /// Depth-1 lookahead needs `(depth + 1) * tileExperts` cache slots. Below
    /// that the tiles run serially.
    private let pipelineTiles: Bool

    // Chunk-wide state (one slice per token).
    private let streams: MTLBuffer          // [T, mult * D] FP16
    private let streamsAlt: MTLBuffer       // [T, mult * D] FP16
    private let routedX: MTLBuffer          // [T, D] FP16
    private let h1: MTLBuffer               // [T, D] FP16 shared-expert output
    private let h2: MTLBuffer               // [T, D] FP16 routed + shared
    private let routeIndices: MTLBuffer     // [T, topK] UInt32
    private let routeWeights: MTLBuffer     // [T, topK] FP16
    private let acts: MTLBuffer             // [T * topK, FmoE] FP16
    private let partials: MTLBuffer         // [T * topK, D] FP32
    private let routesBuffer: MTLBuffer     // [T * topK] DSV4PrefillRoute
    private let routerLogits: MTLBuffer     // [B, numExperts] FP32
    private let hcPostF: MTLBuffer          // [T, mult] FP32
    private let hcCombF: MTLBuffer          // [T, mult * mult] FP32

    // Single-token scratch. Safe to share across the chunk's tokens because
    // dispatches inside a command buffer run in submission order.
    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let attnOut: MTLBuffer

    // Projection-batch scratch, one row per token of a batch of up to
    // `projectionBatch` tokens: the projections that need no other token's
    // state read their weights once per batch (`DequantInt4GEMVMultiX`, bit
    // for bit the decode GEMV per token).
    private let multiX: DequantInt4GEMVMultiX
    private let normedB: MTLBuffer          // [B, D]
    private let qaB: MTLBuffer              // [B, qLoraRank]
    private let qB: MTLBuffer               // [B, numHeads * headDim]
    private let kvB: MTLBuffer              // [B, headDim]
    private let hiddenB: MTLBuffer          // [B, D] collapsed streams
    private let hcPreAB: MTLBuffer          // [B, mult] FP32
    private let oGroupedB: MTLBuffer        // [B, oGroups * oLoraRank]
    private let oOutB: MTLBuffer            // [B, D]
    private let hcPostAB: MTLBuffer         // [B, mult] FP32
    private let hcCombAB: MTLBuffer         // [B, mult * mult] FP32
    private let denseGateB: MTLBuffer       // [B, sharedIntermediate]
    private let denseUpB: MTLBuffer
    private let denseActB: MTLBuffer
    private let selected: MTLBuffer
    private let indexerQ: MTLBuffer
    private let indexerW: MTLBuffer
    private let indexerScores: MTLBuffer

    private static let fp16 = MemoryLayout<Float16>.stride

    /// Byte strides of the per-token slices.
    private let streamStride: Int
    private let hiddenStride: Int
    private let actStride: Int
    private let partialStride: Int
    private let hcPostStride: Int
    private let hcCombStride: Int

    // MARK: - Setup

    init(bindings: DSV4PrefillBindings, chunkTokens: Int) throws {
        precondition(chunkTokens > 0)
        self.b = bindings
        self.chunkTokens = chunkTokens
        let cfg = bindings.cfg
        let ctx = bindings.ctx
        let device = ctx.device
        let D = cfg.hiddenSize
        let mult = cfg.hyperConnections.mult
        let topK = cfg.topKExperts
        let fp16 = Self.fp16

        let moeConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 0, value: .uint32(UInt32(D))),
            MetalFunctionConstant(index: 1, value: .uint32(UInt32(cfg.moeIntermediateSize))),
            MetalFunctionConstant(index: 2, value: .uint32(UInt32(MoEDeepseekV4.topK))),
            MetalFunctionConstant(index: 3, value: .bool(true)),
            MetalFunctionConstant(index: 4, value: .bool(cfg.hiddenActivation == "silu")),
            MetalFunctionConstant(index: 5, value: .float(Float(cfg.swigluLimit))),
        ]
        let routerConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 40, value: .uint32(UInt32(cfg.numExperts))),
            MetalFunctionConstant(index: 41, value: .uint32(UInt32(D))),
            MetalFunctionConstant(index: 42, value: .uint32(UInt32(MoEDeepseekV4.topK))),
            MetalFunctionConstant(index: 43, value: .bool(true)),
        ]
        self.phase1PSO = try ctx.pipeline("dsv4_prefill_moe_phase1_groups_int2",
                                          constants: moeConstants)
        self.downPSO = try ctx.pipeline("dsv4_prefill_moe_down_groups_int2",
                                        constants: moeConstants)
        self.reducePSO = try ctx.pipeline("dsv4_prefill_moe_reduce_pairs_k6",
                                          constants: moeConstants)
        // The router kernels are the decode ones; only the buffer offsets
        // differ, so selection and weighting stay bit-identical.
        self.routerGemvPSO = try ctx.pipeline("router_gemv_bf16_r4",
                                              constants: routerConstants,
                                              maxTotalThreadsPerThreadgroup: 512)
        self.routerSelectPSO = try ctx.pipeline(
            "router_topk_select_sqrtsoftplus_k6_par", constants: routerConstants)
        self.routerHashPSO = try ctx.pipeline("router_hash_weights_k6_par")

        guard let phase1Fn = ctx.library.makeFunction(
                name: "dsv4_prefill_moe_phase1_groups_int2") else {
            throw MetalError.noDevice
        }
        self.argEncoder = phase1Fn.makeArgumentEncoder(bufferIndex: 0)
        let slotCount = bindings.model.routedExpertCacheSlotCount(layer: 0)
        let tileWidth = max(1, min(Self.tileExpertCapacity, slotCount ?? Self.tileExpertCapacity))
        let schedulerConfig = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 1,
                                                               tileExperts: tileWidth)
        self.tileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
        self.pipelineTiles = Self.pipeliningEnabled
            && schedulerConfig.fitsSlotBudget(slotCount: slotCount ?? 0)
        let argBufferCount = pipelineTiles ? schedulerConfig.maxPendingDepth + 1 : 1
        var argPool: [MTLBuffer] = []
        for index in 0..<argBufferCount {
            guard let buf = device.makeBuffer(length: max(argEncoder.encodedLength, 16),
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buf.label = "dsv4.prefill.routedArgumentBuffer.\(index)"
            argPool.append(buf)
        }
        self.argBuffers = argPool

        func buf(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(bytes, 16),
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = label
            return b
        }

        self.streamStride = mult * D * fp16
        self.hiddenStride = D * fp16
        self.actStride = cfg.moeIntermediateSize * fp16
        self.partialStride = D * MemoryLayout<Float>.stride
        self.hcPostStride = mult * MemoryLayout<Float>.stride
        self.hcCombStride = mult * mult * MemoryLayout<Float>.stride

        self.streams = try buf(chunkTokens * streamStride, "dsv4.prefill.streams")
        self.streamsAlt = try buf(chunkTokens * streamStride, "dsv4.prefill.streamsAlt")
        self.routedX = try buf(chunkTokens * hiddenStride, "dsv4.prefill.routedX")
        self.h1 = try buf(chunkTokens * hiddenStride, "dsv4.prefill.h1")
        self.h2 = try buf(chunkTokens * hiddenStride, "dsv4.prefill.h2")
        self.routeIndices = try buf(chunkTokens * topK * MemoryLayout<UInt32>.stride,
                                    "dsv4.prefill.routeIndices")
        self.routeWeights = try buf(chunkTokens * topK * fp16,
                                    "dsv4.prefill.routeWeights")
        self.acts = try buf(chunkTokens * topK * actStride, "dsv4.prefill.acts")
        self.partials = try buf(chunkTokens * topK * partialStride,
                                "dsv4.prefill.partials")
        let paddedRoutes = chunkTokens * topK + cfg.numExperts * (Self.groupRoutes - 1)
        self.routesBuffer = try buf(paddedRoutes * MemoryLayout<DSV4PrefillRoute>.stride,
                                    "dsv4.prefill.routes")
        self.routerLogits = try buf(Self.projectionBatch * cfg.numExperts * MemoryLayout<Float>.stride,
                                    "dsv4.prefill.routerLogits")
        self.hcPostF = try buf(chunkTokens * hcPostStride, "dsv4.prefill.hcPostF")
        self.hcCombF = try buf(chunkTokens * hcCombStride, "dsv4.prefill.hcCombF")

        let ca = cfg.compressedAttention
        self.hidden = try buf(hiddenStride, "dsv4.prefill.hidden")
        self.normed = try buf(hiddenStride, "dsv4.prefill.normed")
        self.attnOut = try buf(Self.projectionBatch * cfg.numHeads * cfg.fullHeadDim * fp16,
                               "dsv4.prefill.attnOut")

        let batch = Self.projectionBatch
        self.multiX = try DequantInt4GEMVMultiX(context: ctx)
        self.normedB = try buf(batch * hiddenStride, "dsv4.prefill.normedB")
        self.qaB = try buf(batch * ca.qLoraRank * fp16, "dsv4.prefill.qaB")
        self.qB = try buf(batch * cfg.numHeads * cfg.fullHeadDim * fp16, "dsv4.prefill.qB")
        self.kvB = try buf(batch * cfg.fullHeadDim * fp16, "dsv4.prefill.kvB")
        self.hiddenB = try buf(batch * hiddenStride, "dsv4.prefill.hiddenB")
        self.hcPreAB = try buf(batch * hcPostStride, "dsv4.prefill.hcPreAB")
        self.oGroupedB = try buf(batch * max(ca.oGroups, 1) * ca.oLoraRank * fp16,
                                 "dsv4.prefill.oGroupedB")
        self.oOutB = try buf(batch * hiddenStride, "dsv4.prefill.oOutB")
        self.hcPostAB = try buf(batch * hcPostStride, "dsv4.prefill.hcPostAB")
        self.hcCombAB = try buf(batch * hcCombStride, "dsv4.prefill.hcCombAB")
        self.denseGateB = try buf(batch * cfg.intermediateSize * fp16, "dsv4.prefill.denseGateB")
        self.denseUpB = try buf(batch * cfg.intermediateSize * fp16, "dsv4.prefill.denseUpB")
        self.denseActB = try buf(batch * cfg.intermediateSize * fp16, "dsv4.prefill.denseActB")
        self.selected = try buf(max(ca.indexTopK, 1) * MemoryLayout<UInt32>.stride,
                                "dsv4.prefill.selected")
        self.indexerQ = try buf(ca.indexNHeads * ca.indexHeadDim * fp16, "dsv4.prefill.indexerQ")
        self.indexerW = try buf(ca.indexNHeads * fp16, "dsv4.prefill.indexerW")
        let maxEntries = (bindings.state.maxContext + max(ca.csaCompressRate, 1) - 1)
            / max(ca.csaCompressRate, 1)
        self.indexerScores = try buf(maxEntries * MemoryLayout<Float>.stride, "dsv4.prefill.indexerScores")
    }

    /// Escape hatch for A/B-ing the batched path against the token-by-token
    /// decode replay on a real checkpoint: `MFERENCE_DSV4_PREFILL=off` forces
    /// every chunk down the fallback. Any other value (or unset) keeps the
    /// batched path.
    static let batchedPathEnabled: Bool = flag("MFERENCE_DSV4_PREFILL")

    /// Depth-1 tile pipelining: `MFERENCE_DSV4_PREFILL_PIPELINE=off` disables.
    ///
    /// Forward `F_RDADVISE` readahead over the tile sweep was tried here and
    /// removed: the tiles are already fetched by
    /// `executeExpertCachePlan`'s 8-way `concurrentPerform` of 8 MB `pread`s,
    /// which is itself a strong sequential-access signal, and advising one to
    /// two tiles ahead measured flat-to-slightly-worse on both a 33-token and
    /// an 802-token prompt.
    static let pipeliningEnabled: Bool = flag("MFERENCE_DSV4_PREFILL_PIPELINE")

    private static func flag(_ name: String) -> Bool {
        let raw = ProcessInfo.processInfo.environment[name]?.lowercased()
        return !(raw == "off" || raw == "0" || raw == "false")
    }

    /// How many tokens from the start of a chunk can run batched.
    ///
    /// GPU sparse selection serves every position, including chunks crossing
    /// the cutover and warm appends above it. Only reference/geometry controls
    /// can decline the production path.
    static func batchedTokenPrefix(config: ArchConfig,
                                   startPosition: Int,
                                   tokenCount: Int,
                                   expertCacheSlots: Int?) -> Int {
        guard batchedPathEnabled else { return 0 }
        guard config.routerScoringFunc == "sqrtsoftplus",
              config.topKExperts == MoEDeepseekV4.topK else { return 0 }
        if let expertCacheSlots, expertCacheSlots < 1 { return 0 }
        let ca = config.compressedAttention
        guard ca.csaCompressRate > 0 else { return 0 }
        return tokenCount
    }

    /// Whether a whole chunk starting at `startPosition` can run batched.
    static func supports(config: ArchConfig,
                         startPosition: Int,
                         tokenCount: Int,
                         expertCacheSlots: Int?) -> Bool {
        batchedTokenPrefix(config: config,
                           startPosition: startPosition,
                           tokenCount: tokenCount,
                           expertCacheSlots: expertCacheSlots) >= tokenCount
    }

    // MARK: - Entry point

    /// Runs one chunk. Returns the greedy token when the fused head ran.
    @discardableResult
    func run(tokens: ArraySlice<Int32>,
             startPosition: Int,
             emitHead: Bool,
             outputMode: PrefillOutputMode,
             logits: MTLBuffer,
             didCompleteLayer: ((Int) throws -> Void)? = nil) async throws -> UInt32? {
        let tokenList = Array(tokens)
        let T = tokenList.count
        guard T > 0 else { return nil }
        guard T <= chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "DSV4 chunk of \(T) tokens exceeds scratch capacity \(chunkTokens)")
        }
        guard b.model.routedExpertWeightBits == 2 else {
            throw PrefillError.prefillCursorMismatch(
                "DeepSeek-V4 runtime supports 2-bit routed experts; manifest says \(b.model.routedExpertWeightBits)")
        }

        try Task.checkCancellation()
        try encodeEmbedding(tokens: tokenList)
        // A chunk this long routes to nearly every expert of every layer, so
        // each layer's sweep reads its whole expert file. Read the next
        // layer's file into the page cache while this layer's tail and the
        // next attention site run, when the SSD would otherwise sit idle.
        let warmAhead = T * b.cfg.topKExperts >= 4 * b.cfg.numExperts
        var warming: (stop: StopFlag, done: DispatchGroup)?
        func stopWarming() {
            warming?.stop.set()
            warming?.done.wait()
            warming = nil
        }
        func startWarming(layer: Int) {
            guard warmAhead, layer < b.cfg.numLayers,
                  let streamer = try? b.model.routedExpertStreamer(layer: layer) else { return }
            let stop = StopFlag()
            let done = DispatchGroup()
            DispatchQueue.global(qos: .userInitiated).async(group: done) {
                streamer.warmPageCache { stop.isSet }
            }
            warming = (stop, done)
        }
        defer { stopWarming() }
        startWarming(layer: 0)
        for layer in 0..<b.cfg.numLayers {
            try Task.checkCancellation()
            try encodeAttentionSite(layer: layer, tokens: tokenList,
                                    startPosition: startPosition, count: T)
            try await runRoutedMoE(layer: layer, count: T)
            stopWarming()
            startWarming(layer: layer + 1)
            try encodeLayerTail(layer: layer, count: T)
            try didCompleteLayer?(layer)
        }
        guard emitHead else { return nil }
        return try encodeHead(lastToken: T - 1, outputMode: outputMode, logits: logits)
    }

    // MARK: - Stages

    private func encodeEmbedding(tokens: [Int32]) throws {
        let cfg = b.cfg
        let emb = b.model.embedding
        let cb = try makeCommandBuffer()
        for (t, token) in tokens.enumerated() {
            b.embed.encode(commandBuffer: cb,
                           table: emb.buffer, tableOffset: Int(emb.offset),
                           scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                           biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                           out: hidden,
                           tokenId: UInt32(bitPattern: token),
                           d: UInt32(cfg.hiddenSize),
                           outScale: 1.0)
            b.dsv4.encodeBroadcastStreams(commandBuffer: cb,
                                          x: hidden,
                                          streams: streams,
                                          streamsOffset: t * streamStride,
                                          hcMult: cfg.hyperConnections.mult,
                                          hidden: cfg.hiddenSize)
        }
        cb.commit()
        try wait(cb)
    }

    /// Everything from the attention-site mHC through the router and the
    /// shared expert, for every token of the chunk, in one command buffer.
    private func encodeAttentionSite(layer L: Int,
                                     tokens: [Int32],
                                     startPosition: Int,
                                     count T: Int) throws {
        let cfg = b.cfg
        let ca = cfg.compressedAttention
        let hc = cfg.hyperConnections
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let headDim = cfg.fullHeadDim
        let numHeads = cfg.numHeads
        let fp16 = Self.fp16
        let isCSA = cfg.layerIsCSA(L)
        let isHCA = cfg.layerIsHCA(L)
        let isCompressed = isCSA || isHCA
        let ropeKind: DSV4Kernels.RopeKind = isCompressed ? .compress : .main
        let isHash = cfg.layerIsHashRouted(L)

        let inNorm = try b.model.inputNorm(layer: L)
        let postAttn = try b.model.postAttnNorm(layer: L)
        let qaView = try b.model.dsv4QAProj(layer: L)
        let qaNormView = try b.model.dsv4QANorm(layer: L)
        let qbView = try b.model.dsv4QBProj(layer: L)
        let kvView = try b.model.dsv4KVProj(layer: L)
        let kvNormView = try b.model.dsv4KVNorm(layer: L)
        let oaView = try b.model.dsv4OAProj(layer: L)
        let obView = try b.model.dsv4OBProj(layer: L)
        let sinksView = try b.model.dsv4Sinks(layer: L)
        let attnFn = try b.model.dsv4AttnHCFn(layer: L)
        let attnBase = try b.model.dsv4AttnHCBase(layer: L)
        let attnScale3 = try b.model.dsv4AttnHCScale(layer: L)
        let ffnFn = try b.model.dsv4FFNHCFn(layer: L)
        let ffnBase = try b.model.dsv4FFNHCBase(layer: L)
        let ffnScale3 = try b.model.dsv4FFNHCScale(layer: L)
        let routerW = try b.model.router(layer: L)
        let sharedProj = b.sharedExperts[L]
        let correctionBias = isHash ? nil : try b.model.dsv4RouterCorrectionBias(layer: L)
        let hashTable = isHash ? try b.model.dsv4HashTable(layer: L) : nil

        var counters = b.state.counters[L]
        var cb = try makeCommandBuffer()
        let qaStride = ca.qLoraRank * fp16
        let qStride = numHeads * headDim * fp16

        // Tokens run in batches. Within a batch, every step that reads only the
        // token's own residual streams (mHC, norms, q projections) runs for the
        // whole batch before the stateful steps (K=V window, compressor,
        // attention), which still go one token at a time in order; then the
        // output side, which again depends only on the token's own attention
        // output. Each token sees the same kernels on the same inputs as decode.
        for t0 in stride(from: 0, to: T, by: Self.projectionBatch) {
            let n = min(Self.projectionBatch, T - t0)

            let batchStreamOff = t0 * streamStride
            b.dsv4.encodeHCWeights(commandBuffer: cb,
                                   streams: streams, streamsOffset: batchStreamOff,
                                   fn: attnFn.buffer, fnOffset: Int(attnFn.offset),
                                   base: attnBase.buffer, baseOffset: Int(attnBase.offset),
                                   scale: attnScale3.buffer, scaleOffset: Int(attnScale3.offset),
                                   outPre: hcPreAB, outPost: hcPostAB, outComb: hcCombAB,
                                   hcMult: hc.mult, hidden: cfg.hiddenSize,
                                   sinkhornIters: hc.sinkhornIters,
                                   hcEps: Float(hc.eps), rmsEps: eps, tokens: n)
            b.dsv4.encodeHCCollapse(commandBuffer: cb,
                                    streams: streams, streamsOffset: batchStreamOff,
                                    pre: hcPreAB, x: hiddenB,
                                    hcMult: hc.mult, hidden: cfg.hiddenSize, tokens: n)
            // One row per token: the per-head kernel is the row kernel at an offset.
            b.rms.encodeBF16WPerHead(commandBuffer: cb, x: hiddenB,
                                     weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                     out: normedB, headDim: D, numHeads: n, eps: eps)

            multiGemv(cb, qaView, x: normedB, y: qaB,
                      m: ca.qLoraRank, n: cfg.hiddenSize, tokens: n)
            b.rms.encodeBF16WPerHead(commandBuffer: cb, x: qaB,
                                     weight: qaNormView.buffer,
                                     weightOffset: Int(qaNormView.offset),
                                     out: qaB, headDim: UInt32(ca.qLoraRank),
                                     numHeads: n, eps: eps)
            multiGemv(cb, qbView, x: qaB, y: qB,
                      m: numHeads * headDim, n: ca.qLoraRank, tokens: n)
            b.rms.encodeNoScalePerHead(commandBuffer: cb, x: qB, out: qB,
                                       headDim: UInt32(headDim),
                                       numHeads: numHeads * n, eps: eps)
            b.dsv4.encodeRope(commandBuffer: cb, x: qB,
                              numHeads: numHeads, headDim: headDim,
                              ropeDim: ca.ropeHeadDim,
                              position: startPosition + t0,
                              rope: ropeKind, direction: 1, tokens: n)

            multiGemv(cb, kvView, x: normedB, y: kvB,
                      m: headDim, n: cfg.hiddenSize, tokens: n)
            for j in 0..<n {
                let t = t0 + j
                let position = startPosition + t
                let normedOff = j * hiddenStride

                let slot = b.state.windowSlot(position: position)
                let ring = b.state.windowKV[L]
                let slotOffset = slot * headDim * fp16
                b.rms.encodeBF16W(commandBuffer: cb, x: kvB, xOffset: j * headDim * fp16,
                                  weight: kvNormView.buffer,
                                  weightOffset: Int(kvNormView.offset),
                                  out: ring, outOffset: slotOffset,
                                  d: UInt32(headDim), eps: eps)
                b.dsv4.encodeRope(commandBuffer: cb, x: ring, xOffset: slotOffset,
                                  numHeads: 1, headDim: headDim,
                                  ropeDim: ca.ropeHeadDim,
                                  position: position, rope: ropeKind, direction: 1)

                var willEmit = false
                var willEmitIndexer = false
                if isCompressed {
                    let rate = isCSA ? ca.csaCompressRate : ca.hcaCompressRate
                    let rowWidth = isCSA ? 2 * headDim : headDim
                    let compKV = try b.model.dsv4CompressorKVProj(layer: L)
                    let compGate = try b.model.dsv4CompressorGateProj(layer: L)
                    let compNorm = try b.model.dsv4CompressorKVNorm(layer: L)
                    let compBias = try b.model.dsv4CompressorPositionBias(layer: L)
                    // Pending rows for the run of tokens up to the next emit
                    // (or the batch end), encoded after the previous emit.
                    if j == 0 || counters.pendingRows == 0 {
                        let run = min(rate - counters.pendingRows, n - j)
                        let rowOffset = counters.pendingRows * rowWidth * fp16
                        multiGemv(cb, compKV, x: normedB, xOffset: normedOff,
                                  y: b.state.pendingKV[L]!, yOffset: rowOffset,
                                  m: rowWidth, n: cfg.hiddenSize, tokens: run)
                        multiGemv(cb, compGate, x: normedB, xOffset: normedOff,
                                  y: b.state.pendingGate[L]!, yOffset: rowOffset,
                                  m: rowWidth, n: cfg.hiddenSize, tokens: run)
                    }
                    willEmit = counters.pendingRows + 1 == rate
                    if willEmit {
                        let entry = counters.compressedEntries
                        b.dsv4.encodeCompressEmit(
                            commandBuffer: cb,
                            pendingKV: b.state.pendingKV[L]!,
                            pendingGate: b.state.pendingGate[L]!,
                            priorCaKV: b.state.priorCaKV[L] ?? b.state.pendingKV[L]!,
                            priorCaGate: b.state.priorCaGate[L] ?? b.state.pendingGate[L]!,
                            positionBias: compBias.buffer,
                            positionBiasOffset: Int(compBias.offset),
                            normWeight: compNorm.buffer,
                            normWeightOffset: Int(compNorm.offset),
                            outEntry: b.state.compressedKV[L]!,
                            outEntryOffset: entry * headDim * fp16,
                            nextPriorCaKV: b.state.priorCaKV[L] ?? b.state.pendingKV[L]!,
                            nextPriorCaGate: b.state.priorCaGate[L] ?? b.state.pendingGate[L]!,
                            rate: rate, dim: headDim, dual: isCSA,
                            hasPrior: counters.hasPrior, eps: eps)
                        b.dsv4.encodeRope(commandBuffer: cb,
                                          x: b.state.compressedKV[L]!,
                                          xOffset: entry * headDim * fp16,
                                          numHeads: 1, headDim: headDim,
                                          ropeDim: ca.ropeHeadDim,
                                          position: entry * rate,
                                          rope: .compress, direction: 1)
                    }
                    if isCSA {
                        let idxRate = ca.csaCompressRate
                        let idxDim = ca.indexHeadDim
                        let idxKV = try b.model.dsv4IndexerKVProj(layer: L)
                        let idxGate = try b.model.dsv4IndexerGateProj(layer: L)
                        let idxNorm = try b.model.dsv4IndexerKVNorm(layer: L)
                        let idxBias = try b.model.dsv4IndexerPositionBias(layer: L)
                        if j == 0 || counters.indexerPendingRows == 0 {
                            let run = min(idxRate - counters.indexerPendingRows, n - j)
                            let idxRowOffset = counters.indexerPendingRows * 2 * idxDim * fp16
                            multiGemv(cb, idxKV, x: normedB, xOffset: normedOff,
                                      y: b.state.indexerPendingKV[L]!, yOffset: idxRowOffset,
                                      m: 2 * idxDim, n: cfg.hiddenSize, tokens: run)
                            multiGemv(cb, idxGate, x: normedB, xOffset: normedOff,
                                      y: b.state.indexerPendingGate[L]!, yOffset: idxRowOffset,
                                      m: 2 * idxDim, n: cfg.hiddenSize, tokens: run)
                        }
                        willEmitIndexer = counters.indexerPendingRows + 1 == idxRate
                        if willEmitIndexer {
                            let entry = counters.indexerEntries
                            b.dsv4.encodeCompressEmit(
                                commandBuffer: cb,
                                pendingKV: b.state.indexerPendingKV[L]!,
                                pendingGate: b.state.indexerPendingGate[L]!,
                                priorCaKV: b.state.indexerPriorCaKV[L]!,
                                priorCaGate: b.state.indexerPriorCaGate[L]!,
                                positionBias: idxBias.buffer,
                                positionBiasOffset: Int(idxBias.offset),
                                normWeight: idxNorm.buffer,
                                normWeightOffset: Int(idxNorm.offset),
                                outEntry: b.state.indexerKeys[L]!,
                                outEntryOffset: entry * idxDim * fp16,
                                nextPriorCaKV: b.state.indexerPriorCaKV[L]!,
                                nextPriorCaGate: b.state.indexerPriorCaGate[L]!,
                                rate: idxRate, dim: idxDim, dual: true,
                                hasPrior: counters.indexerHasPrior, eps: eps)
                            b.dsv4.encodeRope(commandBuffer: cb,
                                              x: b.state.indexerKeys[L]!,
                                              xOffset: entry * idxDim * fp16,
                                              numHeads: 1, headDim: idxDim,
                                              ropeDim: ca.ropeHeadDim,
                                              position: entry * idxRate,
                                              rope: .compress, direction: 1)
                        }
                    }
                }

                let compressedCount = isCompressed
                    ? counters.compressedEntries + (willEmit ? 1 : 0)
                    : 0
                var selectedCount = DSV4Kernels.selectAll
                if isCSA && compressedCount > ca.indexTopK {
                    let idxQB = try b.model.dsv4IndexerQBProj(layer: L)
                    let idxWProj = try b.model.dsv4IndexerWeightsProj(layer: L)
                    let idxEntries = counters.indexerEntries + (willEmitIndexer ? 1 : 0)
                    gemv(cb, idxQB, x: qaB, xOffset: j * qaStride, y: indexerQ,
                         m: ca.indexNHeads * ca.indexHeadDim, n: ca.qLoraRank)
                    b.dsv4.encodeRope(commandBuffer: cb, x: indexerQ,
                        numHeads: ca.indexNHeads, headDim: ca.indexHeadDim,
                        ropeDim: ca.ropeHeadDim, position: position, rope: .compress, direction: 1)
                    gemv(cb, idxWProj, x: normedB, xOffset: normedOff, y: indexerW,
                         m: ca.indexNHeads, n: cfg.hiddenSize)
                    b.dsv4.encodeIndexerScore(commandBuffer: cb, q: indexerQ,
                        keys: b.state.indexerKeys[L]!, weights: indexerW, scores: indexerScores,
                        numHeads: ca.indexNHeads, indexDim: ca.indexHeadDim, entryCount: idxEntries,
                        headScale: 1.0 / Float(ca.indexHeadDim).squareRoot(),
                        weightScale: 1.0 / Float(ca.indexNHeads).squareRoot())
                    b.dsv4.encodeIndexerSelect(commandBuffer: cb, scores: indexerScores,
                        selected: selected, entryCount: compressedCount, topK: ca.indexTopK)
                    selectedCount = UInt32(min(ca.indexTopK, compressedCount))
                }

                b.dsv4.encodeAttention(
                    commandBuffer: cb, q: qB, qOffset: j * qStride,
                    windowKV: ring,
                    compressedKV: b.state.compressedKV[L] ?? ring,
                    selected: selected,
                    sinks: sinksView.buffer, sinksOffset: Int(sinksView.offset),
                    out: attnOut, outOffset: j * qStride,
                    headDim: headDim, numHeads: numHeads,
                    windowCount: b.state.windowCount(position: position),
                    windowStartPos: b.state.windowStartPosition(position: position),
                    ringCapacity: b.state.ringCapacity,
                    compressedCount: compressedCount,
                    selectedCount: selectedCount,
                    scale: Float(cfg.attentionScale))

                counters.tokens += 1
                if isCompressed {
                    if willEmit {
                        counters.pendingRows = 0
                        counters.compressedEntries += 1
                        if isCSA { counters.hasPrior = true }
                    } else {
                        counters.pendingRows += 1
                    }
                    if isCSA {
                        if willEmitIndexer {
                            counters.indexerPendingRows = 0
                            counters.indexerEntries += 1
                            counters.indexerHasPrior = true
                        } else {
                            counters.indexerPendingRows += 1
                        }
                    }
                }
            }

            b.dsv4.encodeRope(commandBuffer: cb, x: attnOut,
                              numHeads: numHeads, headDim: headDim,
                              ropeDim: ca.ropeHeadDim,
                              position: startPosition + t0, rope: ropeKind,
                              direction: -1, tokens: n)
            // Grouped o_a for the whole batch: one weight read per batch
            // instead of one per token.
            multiX.encodeGrouped(commandBuffer: cb,
                weights: oaView.buffer, weightsOffset: Int(oaView.offset),
                scales: oaView.buffer, scalesOffset: Int(oaView.scaleOffset),
                biases: oaView.buffer, biasesOffset: Int(oaView.biasOffset),
                x: attnOut, y: oGroupedB,
                m: ca.oGroups * ca.oLoraRank, n: numHeads * headDim / ca.oGroups, tokens: n,
                rowsPerGroup: ca.oLoraRank, xStride: numHeads * headDim)
            multiGemv(cb, obView, x: oGroupedB, y: oOutB,
                      m: cfg.hiddenSize, n: ca.oGroups * ca.oLoraRank, tokens: n)

            b.dsv4.encodeHCPlaceMix(commandBuffer: cb,
                                    streams: streams, streamsOffset: batchStreamOff,
                                    sub: oOutB, post: hcPostAB, comb: hcCombAB,
                                    outStreams: streamsAlt, outStreamsOffset: batchStreamOff,
                                    hcMult: hc.mult, hidden: cfg.hiddenSize, tokens: n)
            b.dsv4.encodeHCWeights(commandBuffer: cb,
                                   streams: streamsAlt, streamsOffset: batchStreamOff,
                                   fn: ffnFn.buffer, fnOffset: Int(ffnFn.offset),
                                   base: ffnBase.buffer, baseOffset: Int(ffnBase.offset),
                                   scale: ffnScale3.buffer, scaleOffset: Int(ffnScale3.offset),
                                   outPre: hcPreAB,
                                   outPost: hcPostF, outPostOffset: t0 * hcPostStride,
                                   outComb: hcCombF, outCombOffset: t0 * hcCombStride,
                                   hcMult: hc.mult, hidden: cfg.hiddenSize,
                                   sinkhornIters: hc.sinkhornIters,
                                   hcEps: Float(hc.eps), rmsEps: eps, tokens: n)
            b.dsv4.encodeHCCollapse(commandBuffer: cb,
                                    streams: streamsAlt, streamsOffset: batchStreamOff,
                                    pre: hcPreAB, x: hiddenB,
                                    hcMult: hc.mult, hidden: cfg.hiddenSize, tokens: n)
            b.rms.encodeBF16WPerHead(commandBuffer: cb, x: hiddenB,
                                     weight: postAttn.buffer,
                                     weightOffset: Int(postAttn.offset),
                                     out: routedX, outOffset: t0 * hiddenStride,
                                     headDim: D, numHeads: n, eps: eps)
            encodeRouterGemv(cb, weights: routerW, hiddenOffset: t0 * hiddenStride,
                             onesScale: b.effectiveScales[L], tokens: n)
            for j in 0..<n {
                let t = t0 + j
                if let hashTable {
                    writeHashRoute(table: hashTable, token: tokens[t], tokenIndex: t)
                    encodeRouterHashWeights(cb, tokenIndex: t, batchRow: j)
                } else {
                    encodeRouterSelect(cb, correctionBias: correctionBias!, tokenIndex: t, batchRow: j)
                }
            }

            encodeSharedExpert(cb, proj: sharedProj, firstToken: t0, tokens: n)
            // Commit each batch as it is encoded so the GPU starts while the
            // CPU encodes the next one; queue order keeps the batches serial.
            if t0 + n < T {
                cb.commit()
                cb = try makeCommandBuffer()
            }
        }
        cb.commit()
        try wait(cb)
        b.state.counters[L] = counters
    }

    /// Pair-major routed MoE: sort the chunk's routes by expert, stream each
    /// expert tile once, and apply it to every route that lands on it.
    private func runRoutedMoE(layer L: Int, count T: Int) async throws {
        let cfg = b.cfg
        let topK = cfg.topKExperts
        let routeCount = T * topK
        let idPtr = routeIndices.contents().bindMemory(to: UInt32.self,
                                                       capacity: routeCount)
        var experts = [Int](repeating: 0, count: routeCount)
        for i in 0..<routeCount {
            experts[i] = min(Int(idPtr[i]), cfg.numExperts - 1)
        }

        // Group routes by expert, expert order following the on-disk layout so
        // each tile is a forward sweep of the expert file.
        let physical = b.model.routedExpertPhysicalOffsets(layer: L)
        var byExpert = [Int: [Int]]()
        byExpert.reserveCapacity(min(cfg.numExperts, routeCount))
        for i in 0..<routeCount { byExpert[experts[i], default: []].append(i) }
        let liveExperts = byExpert.keys.sorted {
            let lhs = $0 < physical.count ? physical[$0] : UInt64($0)
            let rhs = $1 < physical.count ? physical[$1] : UInt64($1)
            return lhs == rhs ? $0 < $1 : lhs < rhs
        }

        let routesPtr = routesBuffer.contents()
            .bindMemory(to: DSV4PrefillRoute.self,
                        capacity: routeCount + cfg.numExperts * (Self.groupRoutes - 1))
        let perTile = tileScheduler.config.tileExperts
        var tiles: [(experts: [Int], routeStart: Int, routeCount: Int)] = []
        var written = 0
        var index = 0
        while index < liveExperts.count {
            let slice = Array(liveExperts[index..<min(index + perTile, liveExperts.count)])
            let start = written
            for (localSlot, expert) in slice.enumerated() {
                for route in byExpert[expert]! {
                    routesPtr[written] = DSV4PrefillRoute(
                        token: UInt32(route / topK),
                        rank: UInt32(route % topK),
                        localSlot: UInt32(localSlot))
                    written += 1
                }
                while written % Self.groupRoutes != 0 {
                    routesPtr[written] = DSV4PrefillRoute(token: 0, rank: 0, localSlot: .max)
                    written += 1
                }
            }
            tiles.append((slice, start, written - start))
            index += perTile
        }

        let routedOffsets = b.model.routedExpertOffsets(layer: L)
        let gateGroupSize = UInt32(b.model.routedGateGroupSize(layer: L))

        let liveTiles = tiles.filter { $0.routeCount > 0 }
        var pending: PendingRoutedTile?
        // A failed fetch/cancellation must not leave GPU readers live while
        // reset reuses cache slots and scratch. Queue order drains older tiles.
        defer { if let pending { try? wait(pending.cb) } }
        for (index, tile) in liveTiles.enumerated() {
            try Task.checkCancellation()
            // Plan around the in-flight tile's slots so its GPU work keeps
            // reading valid blobs while this tile's pread lands.
            var plan: RoutedExpertFetchPlan?
            if let inFlight = pending {
                let candidate = inFlight.assignedSlots.isEmpty
                    ? nil
                    : try b.model.planRoutedExpertsIfPossible(
                        layer: L, experts: tile.experts,
                        avoidingSlots: Set(inFlight.assignedSlots))
                let decision = tileScheduler.decide(PrefillRoutedTileSchedulerInput(
                    hasPendingTile: true,
                    pendingAssignedSlots: inFlight.assignedSlots,
                    avoidingSlotPlanAvailable: candidate != nil))
                switch decision {
                case .prefetchNext:
                    plan = candidate
                case .drainBeforeIssue, .issueWithoutPending:
                    try wait(inFlight.cb)
                    pending = nil
                }
            }
            if plan == nil {
                plan = try b.model.planRoutedExperts(layer: L, experts: tile.experts)
            }
            guard let tilePlan = plan else {
                throw ModelError.routedExpertPlanUnavailable(layer: L)
            }

            let blobs = try await b.model.fetchRoutedExperts(plan: tilePlan)
                .map { (buffer: $0.buffer, offset: Int($0.offset)) }
            let argBuf = argBuffers[index % argBuffers.count]
            encodeArguments(into: argBuf, blobs: blobs)
            let cb = try makeCommandBuffer()
            encodeRoutedPairs(cb, pso: phase1PSO, argBuffer: argBuf, blobs: blobs,
                              offsets: routedOffsets, gateGroupSize: gateGroupSize,
                              routeStart: tile.routeStart, routeCount: tile.routeCount,
                              rowsPerRoute: cfg.moeIntermediateSize, isPhase1: true)
            encodeRoutedPairs(cb, pso: downPSO, argBuffer: argBuf, blobs: blobs,
                              offsets: routedOffsets, gateGroupSize: gateGroupSize,
                              routeStart: tile.routeStart, routeCount: tile.routeCount,
                              rowsPerRoute: cfg.hiddenSize, isPhase1: false)
            cb.commit()
            let previous = pending
            pending = PendingRoutedTile(cb: cb, assignedSlots: tilePlan.assignedSlots)
            if let previous { try wait(previous.cb) }
            if !pipelineTiles {
                try wait(cb)
                pending = nil
            }
        }
        if let inFlight = pending { try wait(inFlight.cb) }

        // Residual-first, rank-ordered reduce: h2 = shared + sum_k w_k * down_k.
        let reduceCB = try makeCommandBuffer()
        if let enc = reduceCB.makeComputeCommandEncoder() {
            enc.setComputePipelineState(reducePSO)
            enc.setBuffer(partials, offset: 0, index: 0)
            enc.setBuffer(h1, offset: 0, index: 1)
            enc.setBuffer(h2, offset: 0, index: 2)
            var d = UInt32(cfg.hiddenSize)
            enc.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.dispatchThreads(MTLSize(width: cfg.hiddenSize, height: T, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        reduceCB.commit()
        try wait(reduceCB)
    }

    /// FFN-site placement + residual mix: streamsAlt -> streams.
    private func encodeLayerTail(layer _: Int, count T: Int) throws {
        let cfg = b.cfg
        let hc = cfg.hyperConnections
        let cb = try makeCommandBuffer()
        b.dsv4.encodeHCPlaceMix(commandBuffer: cb,
                                streams: streamsAlt, sub: h2,
                                post: hcPostF, comb: hcCombF, outStreams: streams,
                                hcMult: hc.mult, hidden: cfg.hiddenSize, tokens: T)
        cb.commit()
        try wait(cb)
    }

    private func encodeHead(lastToken t: Int,
                            outputMode: PrefillOutputMode,
                            logits: MTLBuffer) throws -> UInt32? {
        let cfg = b.cfg
        let hc = cfg.hyperConnections
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let hhFn = try b.model.dsv4HyperHeadFn
        let hhBase = try b.model.dsv4HyperHeadBase
        let hhScale = try b.model.dsv4HyperHeadScale
        let fNorm = b.model.finalNorm
        let lm = b.model.lmHead
        let cb = try makeCommandBuffer()
        b.dsv4.encodeHyperHead(commandBuffer: cb,
                               streams: streams, streamsOffset: t * streamStride,
                               fn: hhFn.buffer, fnOffset: Int(hhFn.offset),
                               base: hhBase.buffer, baseOffset: Int(hhBase.offset),
                               scale: hhScale.buffer, scaleOffset: Int(hhScale.offset),
                               x: hidden,
                               hcMult: hc.mult, hidden: cfg.hiddenSize,
                               hcEps: Float(hc.eps), rmsEps: eps)
        let fused = b.useFusedGreedyHead && outputMode == .greedyIfAvailable
        var greedyBuf: MTLBuffer?
        if fused {
            guard let out = b.ctx.device.makeBuffer(length: MemoryLayout<UInt32>.stride,
                                                    options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            greedyBuf = out
            b.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: out,
                d: D, vocab: UInt32(cfg.vocabSize),
                rmsEps: eps)
        } else {
            b.rms.encodeBF16W(commandBuffer: cb, x: hidden,
                              weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                              out: normed, d: D, eps: eps)
            b.int4.encode(commandBuffer: cb,
                          weights: lm.buffer, weightsOffset: Int(lm.offset),
                          scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                          biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                          x: normed, y: logits,
                          m: UInt32(cfg.vocabSize), n: D)
        }
        cb.commit()
        try wait(cb)
        return greedyBuf.map { $0.contents().load(as: UInt32.self) }
    }

    // MARK: - Encoding helpers

    private func gemv(_ cb: MTLCommandBuffer, _ view: TensorView,
                      x: MTLBuffer, xOffset: Int = 0,
                      y: MTLBuffer, yOffset: Int = 0,
                      m: Int, n: Int) {
        b.int4.encode(commandBuffer: cb,
                      weights: view.buffer, weightsOffset: Int(view.offset),
                      scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                      biases: view.buffer, biasesOffset: Int(view.biasOffset),
                      x: x, xOffset: xOffset,
                      y: y, yOffset: yOffset,
                      m: UInt32(m), n: UInt32(n))
    }

    /// Shared expert for `tokens` consecutive tokens from `firstToken`: their
    /// `routedX` rows are contiguous, and so are their `h1` rows.
    private func encodeSharedExpert(_ cb: MTLCommandBuffer,
                                    proj: DSV4PrefillSharedExpert,
                                    firstToken t0: Int,
                                    tokens n: Int) {
        let cfg = b.cfg
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        multiX.encode(commandBuffer: cb,
                      weights: proj.gate.weights, weightsOffset: proj.gate.weightsOffset,
                      scales: proj.gate.scales, scalesOffset: proj.gate.scalesOffset,
                      biases: proj.gate.biases, biasesOffset: proj.gate.biasesOffset,
                      x: routedX, xOffset: t0 * hiddenStride, y: denseGateB,
                      m: F, n: D, tokens: n)
        multiX.encode(commandBuffer: cb,
                      weights: proj.up.weights, weightsOffset: proj.up.weightsOffset,
                      scales: proj.up.scales, scalesOffset: proj.up.scalesOffset,
                      biases: proj.up.biases, biasesOffset: proj.up.biasesOffset,
                      x: routedX, xOffset: t0 * hiddenStride, y: denseUpB,
                      m: F, n: D, tokens: n)
        b.dsv4.encodeSwigluClampMul(commandBuffer: cb,
                                    gate: denseGateB, up: denseUpB, out: denseActB,
                                    n: n * F,
                                    limit: Float(cfg.swigluLimit))
        multiX.encode(commandBuffer: cb,
                      weights: proj.down.weights, weightsOffset: proj.down.weightsOffset,
                      scales: proj.down.scales, scalesOffset: proj.down.scalesOffset,
                      biases: proj.down.biases, biasesOffset: proj.down.biasesOffset,
                      x: denseActB, y: h1, yOffset: t0 * hiddenStride,
                      m: D, n: F, tokens: n)
    }

    /// `gemv` for `tokens` consecutive rows of `x` into consecutive rows of
    /// `y`, reading the weights once.
    private func multiGemv(_ cb: MTLCommandBuffer, _ view: TensorView,
                           x: MTLBuffer, xOffset: Int = 0,
                           y: MTLBuffer, yOffset: Int = 0,
                           m: Int, n: Int, tokens: Int) {
        multiX.encode(commandBuffer: cb,
                      weights: view.buffer, weightsOffset: Int(view.offset),
                      scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                      biases: view.buffer, biasesOffset: Int(view.biasOffset),
                      x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                      m: m, n: n, tokens: tokens)
    }

    private func writeHashRoute(table: TensorView, token: Int32, tokenIndex t: Int) {
        let cfg = b.cfg
        let base = table.buffer.contents().advanced(by: Int(table.offset))
        let row = min(max(Int(token), 0), cfg.vocabSize - 1) * cfg.topKExperts
        let idxPtr = routeIndices.contents().assumingMemoryBound(to: UInt32.self)
        let expertCap = UInt32(cfg.numExperts - 1)
        let out = t * cfg.topKExperts
        if table.dtype == 4 {
            let tPtr = base.assumingMemoryBound(to: Int64.self)
            for i in 0..<cfg.topKExperts {
                idxPtr[out + i] = min(UInt32(clamping: max(0, tPtr[row + i])), expertCap)
            }
        } else {
            let tPtr = base.assumingMemoryBound(to: UInt32.self)
            for i in 0..<cfg.topKExperts {
                idxPtr[out + i] = min(tPtr[row + i], expertCap)
            }
        }
    }

    private func encodeRouterGemv(_ cb: MTLCommandBuffer,
                                  weights: TensorView,
                                  hiddenOffset: Int,
                                  onesScale: MTLBuffer,
                                  tokens: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var expertCount = UInt32(b.cfg.numExperts)
        var dimension = UInt32(b.cfg.hiddenSize)
        enc.setComputePipelineState(routerGemvPSO)
        enc.setBuffer(weights.buffer, offset: Int(weights.offset), index: 0)
        enc.setBuffer(routedX, offset: hiddenOffset, index: 1)
        enc.setBuffer(onesScale, offset: 0, index: 2)
        enc.setBuffer(routerLogits, offset: 0, index: 3)
        enc.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (b.cfg.numExperts + 3) / 4, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeRouterSelect(_ cb: MTLCommandBuffer,
                                    correctionBias: TensorView,
                                    tokenIndex t: Int, batchRow j: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var expertCount = UInt32(b.cfg.numExperts)
        var scale = Float(b.cfg.routedScalingFactor)
        enc.setComputePipelineState(routerSelectPSO)
        enc.setBuffer(routerLogits, offset: j * b.cfg.numExperts * MemoryLayout<Float>.stride,
                      index: 0)
        enc.setBuffer(correctionBias.buffer, offset: Int(correctionBias.offset), index: 1)
        enc.setBuffer(routeIndices,
                      offset: t * b.cfg.topKExperts * MemoryLayout<UInt32>.stride, index: 2)
        enc.setBuffer(routeWeights, offset: t * b.cfg.topKExperts * Self.fp16, index: 3)
        enc.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeRouterHashWeights(_ cb: MTLCommandBuffer, tokenIndex t: Int, batchRow j: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var scale = Float(b.cfg.routedScalingFactor)
        enc.setComputePipelineState(routerHashPSO)
        enc.setBuffer(routerLogits, offset: j * b.cfg.numExperts * MemoryLayout<Float>.stride,
                      index: 0)
        enc.setBuffer(routeIndices,
                      offset: t * b.cfg.topKExperts * MemoryLayout<UInt32>.stride, index: 1)
        enc.setBuffer(routeWeights, offset: t * b.cfg.topKExperts * Self.fp16, index: 2)
        enc.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeArguments(into argBuffer: MTLBuffer,
                                 blobs: [(buffer: MTLBuffer, offset: Int)]) {
        argEncoder.setArgumentBuffer(argBuffer, offset: 0)
        for (index, blob) in blobs.enumerated() where index < Self.tileExpertCapacity {
            argEncoder.setBuffer(blob.buffer, offset: blob.offset, index: index)
        }
        // Never leave a dangling pointer in an unused slot.
        if let first = blobs.first {
            for index in blobs.count..<Self.tileExpertCapacity {
                argEncoder.setBuffer(first.buffer, offset: first.offset, index: index)
            }
        }
    }

    private func encodeRoutedPairs(_ cb: MTLCommandBuffer,
                                   pso: MTLComputePipelineState,
                                   argBuffer: MTLBuffer,
                                   blobs: [(buffer: MTLBuffer, offset: Int)],
                                   offsets: MoEExpertOffsets,
                                   gateGroupSize: UInt32,
                                   routeStart: Int,
                                   routeCount: Int,
                                   rowsPerRoute: Int,
                                   isPhase1: Bool) {
        guard routeCount > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(argBuffer, offset: 0, index: 0)
        for blob in blobs { enc.useResource(blob.buffer, usage: .read) }
        var routedOffsets = offsets
        enc.setBytes(&routedOffsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        var d = UInt32(b.cfg.hiddenSize)
        var f = UInt32(b.cfg.moeIntermediateSize)
        var topK = UInt32(b.cfg.topKExperts)
        var start = UInt32(routeStart)
        var count = UInt32(routeCount)
        if isPhase1 {
            enc.setBuffer(routedX, offset: 0, index: 2)
            enc.setBuffer(acts, offset: 0, index: 3)
            enc.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&f, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBytes(&topK, length: MemoryLayout<UInt32>.stride, index: 6)
            var gateGroup = gateGroupSize
            enc.setBytes(&gateGroup, length: MemoryLayout<UInt32>.stride, index: 7)
        } else {
            enc.setBuffer(acts, offset: 0, index: 2)
            enc.setBuffer(routeWeights, offset: 0, index: 3)
            enc.setBuffer(partials, offset: 0, index: 4)
            enc.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBytes(&f, length: MemoryLayout<UInt32>.stride, index: 6)
            enc.setBytes(&topK, length: MemoryLayout<UInt32>.stride, index: 7)
        }
        enc.setBuffer(routesBuffer, offset: 0, index: 8)
        enc.setBytes(&start, length: MemoryLayout<UInt32>.stride, index: 9)
        enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 10)
        let rows = routeCount / Self.groupRoutes * rowsPerRoute
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let cb = b.ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        return cb
    }

    private func wait(_ cb: MTLCommandBuffer) throws {
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
    }
}

/// One-way stop signal shared with a background page-cache warm.
private final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    func set() {
        lock.lock()
        value = true
        lock.unlock()
    }

    var isSet: Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}
