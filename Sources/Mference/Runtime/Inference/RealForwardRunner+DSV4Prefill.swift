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
// Falls back to the token-by-token path (see `supports`) when the chunk would
// need lightning-indexer selection, which requires a mid-layer CPU readback
// per token.
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
    private let routerLogits: MTLBuffer     // [numExperts] FP32
    private let hcPostF: MTLBuffer          // [T, mult] FP32
    private let hcCombF: MTLBuffer          // [T, mult * mult] FP32

    // Single-token scratch. Safe to share across the chunk's tokens because
    // dispatches inside a command buffer run in submission order.
    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let qaBuf: MTLBuffer
    private let qScratch: MTLBuffer
    private let attnOut: MTLBuffer
    private let oGrouped: MTLBuffer
    private let oOut: MTLBuffer
    private let hcPreA: MTLBuffer
    private let hcPostA: MTLBuffer
    private let hcCombA: MTLBuffer
    private let hcPreF: MTLBuffer
    private let denseGate: MTLBuffer
    private let denseUp: MTLBuffer
    private let denseAct: MTLBuffer
    private let selected: MTLBuffer

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
        self.phase1PSO = try ctx.pipeline("dsv4_prefill_moe_phase1_pairs_int2",
                                          constants: moeConstants)
        self.downPSO = try ctx.pipeline("dsv4_prefill_moe_down_pairs_int2",
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
                name: "dsv4_prefill_moe_phase1_pairs_int2") else {
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
        self.routesBuffer = try buf(chunkTokens * topK * MemoryLayout<DSV4PrefillRoute>.stride,
                                    "dsv4.prefill.routes")
        self.routerLogits = try buf(cfg.numExperts * MemoryLayout<Float>.stride,
                                    "dsv4.prefill.routerLogits")
        self.hcPostF = try buf(chunkTokens * hcPostStride, "dsv4.prefill.hcPostF")
        self.hcCombF = try buf(chunkTokens * hcCombStride, "dsv4.prefill.hcCombF")

        let ca = cfg.compressedAttention
        self.hidden = try buf(hiddenStride, "dsv4.prefill.hidden")
        self.normed = try buf(hiddenStride, "dsv4.prefill.normed")
        self.qaBuf = try buf(ca.qLoraRank * fp16, "dsv4.prefill.qa")
        self.qScratch = try buf(cfg.numHeads * cfg.fullHeadDim * fp16, "dsv4.prefill.q")
        self.attnOut = try buf(cfg.numHeads * cfg.fullHeadDim * fp16, "dsv4.prefill.attnOut")
        self.oGrouped = try buf(max(ca.oGroups, 1) * ca.oLoraRank * fp16,
                                "dsv4.prefill.oGrouped")
        self.oOut = try buf(hiddenStride, "dsv4.prefill.oOut")
        self.hcPreA = try buf(hcPostStride, "dsv4.prefill.hcPreA")
        self.hcPostA = try buf(hcPostStride, "dsv4.prefill.hcPostA")
        self.hcCombA = try buf(hcCombStride, "dsv4.prefill.hcCombA")
        self.hcPreF = try buf(hcPostStride, "dsv4.prefill.hcPreF")
        self.denseGate = try buf(cfg.intermediateSize * fp16, "dsv4.prefill.denseGate")
        self.denseUp = try buf(cfg.intermediateSize * fp16, "dsv4.prefill.denseUp")
        self.denseAct = try buf(cfg.intermediateSize * fp16, "dsv4.prefill.denseAct")
        self.selected = try buf(max(ca.indexTopK, 1) * MemoryLayout<UInt32>.stride,
                                "dsv4.prefill.selected")
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
    /// The lightning indexer selects a top-`indexTopK` subset of compressed
    /// entries once a CSA layer holds more than `indexTopK` of them, and the
    /// selection is a CPU top-k over a GPU readback taken *in the middle* of
    /// the layer. Batching a whole chunk into one command buffer cannot host a
    /// per-token mid-layer readback, so positions past that context length
    /// fall back to the token-by-token path. A chunk that crosses the cutover
    /// keeps its eligible prefix batched; only the remainder falls back.
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
        let hasCSA = (0..<config.numLayers).contains { config.layerIsCSA($0) }
        guard hasCSA else { return tokenCount }
        // A position is batchable while the compressed-entry count after it
        // stays within the selection threshold:
        //   (lastPosition + 1) / csaCompressRate <= indexTopK
        // which holds for lastPosition + 1 <= (indexTopK + 1) * rate - 1.
        let batchablePositions = (ca.indexTopK + 1) * ca.csaCompressRate - 1
        return max(0, min(tokenCount, batchablePositions - startPosition))
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
             logits: MTLBuffer) async throws -> UInt32? {
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

        try encodeEmbedding(tokens: tokenList)
        for layer in 0..<b.cfg.numLayers {
            try encodeAttentionSite(layer: layer, tokens: tokenList,
                                    startPosition: startPosition, count: T)
            try await runRoutedMoE(layer: layer, count: T)
            try encodeLayerTail(layer: layer, count: T)
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
        let cb = try makeCommandBuffer()

        for t in 0..<T {
            let position = startPosition + t
            let streamOff = t * streamStride

            b.dsv4.encodeHCWeights(commandBuffer: cb,
                                   streams: streams, streamsOffset: streamOff,
                                   fn: attnFn.buffer, fnOffset: Int(attnFn.offset),
                                   base: attnBase.buffer, baseOffset: Int(attnBase.offset),
                                   scale: attnScale3.buffer, scaleOffset: Int(attnScale3.offset),
                                   outPre: hcPreA, outPost: hcPostA, outComb: hcCombA,
                                   hcMult: hc.mult, hidden: cfg.hiddenSize,
                                   sinkhornIters: hc.sinkhornIters,
                                   hcEps: Float(hc.eps), rmsEps: eps)
            b.dsv4.encodeHCCollapse(commandBuffer: cb,
                                    streams: streams, streamsOffset: streamOff,
                                    pre: hcPreA, x: hidden,
                                    hcMult: hc.mult, hidden: cfg.hiddenSize)
            b.rms.encodeBF16W(commandBuffer: cb, x: hidden,
                              weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                              out: normed, d: D, eps: eps)

            gemv(cb, qaView, x: normed, y: qaBuf, m: ca.qLoraRank, n: cfg.hiddenSize)
            b.rms.encodeBF16W(commandBuffer: cb, x: qaBuf,
                              weight: qaNormView.buffer,
                              weightOffset: Int(qaNormView.offset),
                              out: qaBuf, d: UInt32(ca.qLoraRank), eps: eps)
            gemv(cb, qbView, x: qaBuf, y: qScratch,
                 m: numHeads * headDim, n: ca.qLoraRank)
            b.rms.encodeNoScalePerHead(commandBuffer: cb, x: qScratch, out: qScratch,
                                       headDim: UInt32(headDim),
                                       numHeads: numHeads, eps: eps)
            b.dsv4.encodeRope(commandBuffer: cb, x: qScratch,
                              numHeads: numHeads, headDim: headDim,
                              ropeDim: ca.ropeHeadDim,
                              position: position, rope: ropeKind, direction: 1)

            let slot = b.state.windowSlot(position: position)
            let ring = b.state.windowKV[L]
            let slotOffset = slot * headDim * fp16
            gemv(cb, kvView, x: normed, y: ring, yOffset: slotOffset,
                 m: headDim, n: cfg.hiddenSize)
            b.rms.encodeBF16W(commandBuffer: cb, x: ring, xOffset: slotOffset,
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
                let rowOffset = counters.pendingRows * rowWidth * fp16
                gemv(cb, compKV, x: normed,
                     y: b.state.pendingKV[L]!, yOffset: rowOffset,
                     m: rowWidth, n: cfg.hiddenSize)
                gemv(cb, compGate, x: normed,
                     y: b.state.pendingGate[L]!, yOffset: rowOffset,
                     m: rowWidth, n: cfg.hiddenSize)
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
                    let idxRowOffset = counters.indexerPendingRows * 2 * idxDim * fp16
                    gemv(cb, idxKV, x: normed,
                         y: b.state.indexerPendingKV[L]!, yOffset: idxRowOffset,
                         m: 2 * idxDim, n: cfg.hiddenSize)
                    gemv(cb, idxGate, x: normed,
                         y: b.state.indexerPendingGate[L]!, yOffset: idxRowOffset,
                         m: 2 * idxDim, n: cfg.hiddenSize)
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
            // `supports` guarantees no token in this chunk needs lightning
            // selection, so every compressed entry is attended.
            precondition(!(isCSA && compressedCount > ca.indexTopK),
                         "DSV4 chunked prefill reached lightning-indexer selection")

            b.dsv4.encodeAttention(
                commandBuffer: cb, q: qScratch,
                windowKV: ring,
                compressedKV: b.state.compressedKV[L] ?? ring,
                selected: selected,
                sinks: sinksView.buffer, sinksOffset: Int(sinksView.offset),
                out: attnOut,
                headDim: headDim, numHeads: numHeads,
                windowCount: b.state.windowCount(position: position),
                windowStartPos: b.state.windowStartPosition(position: position),
                ringCapacity: b.state.ringCapacity,
                compressedCount: compressedCount,
                selectedCount: DSV4Kernels.selectAll,
                scale: Float(cfg.attentionScale))
            b.dsv4.encodeRope(commandBuffer: cb, x: attnOut,
                              numHeads: numHeads, headDim: headDim,
                              ropeDim: ca.ropeHeadDim,
                              position: position, rope: ropeKind, direction: -1)

            b.dsv4.encodeOGroupProjection(
                commandBuffer: cb,
                weights: oaView.buffer, weightsOffset: Int(oaView.offset),
                scales: oaView.buffer, scalesOffset: Int(oaView.scaleOffset),
                biases: oaView.buffer, biasesOffset: Int(oaView.biasOffset),
                x: attnOut, y: oGrouped,
                rank: ca.oLoraRank,
                groupIn: numHeads * headDim / ca.oGroups,
                groups: ca.oGroups)
            gemv(cb, obView, x: oGrouped, y: oOut,
                 m: cfg.hiddenSize, n: ca.oGroups * ca.oLoraRank)

            b.dsv4.encodeHCPlaceMix(commandBuffer: cb,
                                    streams: streams, streamsOffset: streamOff,
                                    sub: oOut, post: hcPostA, comb: hcCombA,
                                    outStreams: streamsAlt, outStreamsOffset: streamOff,
                                    hcMult: hc.mult, hidden: cfg.hiddenSize)

            b.dsv4.encodeHCWeights(commandBuffer: cb,
                                   streams: streamsAlt, streamsOffset: streamOff,
                                   fn: ffnFn.buffer, fnOffset: Int(ffnFn.offset),
                                   base: ffnBase.buffer, baseOffset: Int(ffnBase.offset),
                                   scale: ffnScale3.buffer, scaleOffset: Int(ffnScale3.offset),
                                   outPre: hcPreF,
                                   outPost: hcPostF, outPostOffset: t * hcPostStride,
                                   outComb: hcCombF, outCombOffset: t * hcCombStride,
                                   hcMult: hc.mult, hidden: cfg.hiddenSize,
                                   sinkhornIters: hc.sinkhornIters,
                                   hcEps: Float(hc.eps), rmsEps: eps)
            b.dsv4.encodeHCCollapse(commandBuffer: cb,
                                    streams: streamsAlt, streamsOffset: streamOff,
                                    pre: hcPreF, x: hidden,
                                    hcMult: hc.mult, hidden: cfg.hiddenSize)
            b.rms.encodeBF16W(commandBuffer: cb, x: hidden,
                              weight: postAttn.buffer,
                              weightOffset: Int(postAttn.offset),
                              out: routedX, outOffset: t * hiddenStride,
                              d: D, eps: eps)

            if let hashTable {
                writeHashRoute(table: hashTable, token: tokens[t], tokenIndex: t)
                encodeRouterGemv(cb, weights: routerW, hiddenOffset: t * hiddenStride,
                                 onesScale: b.effectiveScales[L])
                encodeRouterHashWeights(cb, tokenIndex: t)
            } else {
                encodeRouterGemv(cb, weights: routerW, hiddenOffset: t * hiddenStride,
                                 onesScale: b.effectiveScales[L])
                encodeRouterSelect(cb, correctionBias: correctionBias!, tokenIndex: t)
            }

            encodeSharedExpert(cb, proj: sharedProj, tokenIndex: t)

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
            .bindMemory(to: DSV4PrefillRoute.self, capacity: routeCount)
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
            }
            tiles.append((slice, start, written - start))
            index += perTile
        }

        let routedOffsets = b.model.routedExpertOffsets(layer: L)
        let gateGroupSize = UInt32(b.model.routedGateGroupSize(layer: L))
        let liveTiles = tiles.filter { $0.routeCount > 0 }

        var pending: PendingRoutedTile?
        for (index, tile) in liveTiles.enumerated() {
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
            if let inFlight = pending { try wait(inFlight.cb) }
            if pipelineTiles {
                pending = PendingRoutedTile(cb: cb, assignedSlots: tilePlan.assignedSlots)
            } else {
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
        for t in 0..<T {
            b.dsv4.encodeHCPlaceMix(commandBuffer: cb,
                                    streams: streamsAlt, streamsOffset: t * streamStride,
                                    sub: h2, subOffset: t * hiddenStride,
                                    post: hcPostF, postOffset: t * hcPostStride,
                                    comb: hcCombF, combOffset: t * hcCombStride,
                                    outStreams: streams, outStreamsOffset: t * streamStride,
                                    hcMult: hc.mult, hidden: cfg.hiddenSize)
        }
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

    private func encodeSharedExpert(_ cb: MTLCommandBuffer,
                                    proj: DSV4PrefillSharedExpert,
                                    tokenIndex t: Int) {
        let cfg = b.cfg
        let D = UInt32(cfg.hiddenSize)
        let F = UInt32(cfg.intermediateSize)
        let xOff = t * hiddenStride
        b.int4.encode(commandBuffer: cb,
                      weights: proj.gate.weights, weightsOffset: proj.gate.weightsOffset,
                      scales: proj.gate.scales, scalesOffset: proj.gate.scalesOffset,
                      biases: proj.gate.biases, biasesOffset: proj.gate.biasesOffset,
                      x: routedX, xOffset: xOff, y: denseGate, m: F, n: D)
        b.int4.encode(commandBuffer: cb,
                      weights: proj.up.weights, weightsOffset: proj.up.weightsOffset,
                      scales: proj.up.scales, scalesOffset: proj.up.scalesOffset,
                      biases: proj.up.biases, biasesOffset: proj.up.biasesOffset,
                      x: routedX, xOffset: xOff, y: denseUp, m: F, n: D)
        b.dsv4.encodeSwigluClampMul(commandBuffer: cb,
                                    gate: denseGate, up: denseUp, out: denseAct,
                                    n: cfg.intermediateSize,
                                    limit: Float(cfg.swigluLimit))
        b.int4.encode(commandBuffer: cb,
                      weights: proj.down.weights, weightsOffset: proj.down.weightsOffset,
                      scales: proj.down.scales, scalesOffset: proj.down.scalesOffset,
                      biases: proj.down.biases, biasesOffset: proj.down.biasesOffset,
                      x: denseAct, y: h1, yOffset: t * hiddenStride, m: D, n: F)
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
                                  onesScale: MTLBuffer) {
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
            MTLSize(width: (b.cfg.numExperts + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeRouterSelect(_ cb: MTLCommandBuffer,
                                    correctionBias: TensorView,
                                    tokenIndex t: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var expertCount = UInt32(b.cfg.numExperts)
        var scale = Float(b.cfg.routedScalingFactor)
        enc.setComputePipelineState(routerSelectPSO)
        enc.setBuffer(routerLogits, offset: 0, index: 0)
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

    private func encodeRouterHashWeights(_ cb: MTLCommandBuffer, tokenIndex t: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var scale = Float(b.cfg.routedScalingFactor)
        enc.setComputePipelineState(routerHashPSO)
        enc.setBuffer(routerLogits, offset: 0, index: 0)
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
        let rows = routeCount * rowsPerRoute
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
