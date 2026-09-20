import Foundation
import Metal

/// Internal one-layer native MTP execution, not an enabled speculator.
/// The caller supplies already-aligned FP16 embeddings and full target/draft
/// HC rows. Every preceding draft KV row must have been appended: accepting
/// a last-target-row snapshot does NOT authorize jumping over prompt priming.
/// No GDN, PLE, target verification, token sampling or default-path caller.
final class FlashNextMTPDraftRunner {
    enum ExpertPolicy {
        case bounded(slots: Int)
        case resident
    }

    struct Output {
        let processedRows: Int
        /// Owned full, unmixed FP16 HC row; safe as the next draft's input.
        let hidden: MTLBuffer
    }

    struct Checkpoint {
        fileprivate let owner: UUID
        fileprivate let epoch: UInt64
        fileprivate let position: Int
    }

    private let model: Model
    private let context: MetalContext
    private let weights: FlashNextMTPWeights
    private let maxContext: Int
    private let matvec: FlashNextMatVec
    private let fusion: FlashNextMTPInputFusion
    private let hc: FlashNextHyperConnections
    private let hcScratch: FlashNextHyperConnections.Scratch
    private let indexer: FlashNextIndexer
    private let indexScratch: FlashNextIndexer.Scratch
    private let indexCache: FlashNextIndexer.LayerCache
    private let attention: FlashNextAttention
    private let attentionScratch: FlashNextAttention.Scratch
    private let kv: FlashNextAttention.KVCache
    private let moe: MoE
    private let router: FlashNextMoE
    private let elementwise: Elementwise
    private let silu: MTLComputePipelineState
    private let resident: ResidentExpertStreamer?
    private let streamed: PreadExpertStreamer?
    private let hyper: MTLBuffer
    private let mixed: MTLBuffer
    private let block: MTLBuffer
    private let routed: MTLBuffer
    private let acts: MTLBuffer
    private let zero: MTLBuffer
    private let routerLogits: MTLBuffer
    private let routerScale: MTLBuffer
    private let routeIDs: MTLBuffer
    private let routeWeights: MTLBuffer
    private let sharedGate: MTLBuffer
    private let sharedUp: MTLBuffer
    private let sharedAct: MTLBuffer
    private let sharedOutput: MTLBuffer
    private let sharedScalar: MTLBuffer
    private let head: FlashNextWeightMatrix
    private let owner = UUID()
    private var epoch: UInt64 = 0
    private var dirty = false
    private(set) var position = 0
    /// Correctness-only failure seam after attention/indexer GPU writes.
    var didPrepareAttention: (() throws -> Void)?

    init(model: Model, context: MetalContext, maxContext: Int, policy: ExpertPolicy) throws {
        let cfg = model.config
        let fn = cfg.flashNext
        let d = cfg.hiddenSize
        let rotary = Int(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor)
        guard cfg.family == .qwen38flashnext, maxContext > 0,
              fn.hcCount > 1, fn.hcLowRank > 0,
              MoE.routedComputeWidths.contains(UInt32(cfg.topKExperts)),
              cfg.numExperts >= cfg.topKExperts, cfg.numExperts <= 512,
              fn.indexerNumKVHeads == 1, fn.indexerCompressRatio > 0,
              fn.indexerBlockBudget > 0, fn.indexerBlockBudget <= FlashNextIndexer.maxBlockBudget,
              fn.indexerHeadDim <= FlashNextIndexer.maxHeadDim,
              rotary >= 0, rotary.isMultiple(of: 2), rotary <= fn.indexerHeadDim,
              rotary <= cfg.fullHeadDim, cfg.numFullKVHeads > 0,
              cfg.numHeads.isMultiple(of: cfg.numFullKVHeads) else {
            throw FlashNextForwardRunnerError.invalidConfiguration("unsupported Flash-Next native MTP geometry")
        }
        if case .bounded(let slots) = policy, slots < cfg.topKExperts {
            throw FlashNextForwardRunnerError.invalidConfiguration("MTP expert cache must hold every routed expert")
        }
        self.model = model
        self.context = context
        self.maxContext = maxContext
        weights = try FlashNextMTPWeights(model: model)
        // INT8 kernels here produce FP32 gating logits, not FP16 block
        // activations. Reject an unsupported conversion at construction rather
        // than reaching a matvec precondition after draft state has advanced.
        var blockMatrices = [weights.embeddingProjection, weights.hiddenProjection,
            weights.attentionHC.mixDown, weights.attentionHC.mixUp,
            weights.mlpHC.mixDown, weights.mlpHC.mixUp, weights.mixer.mixDown, weights.mixer.mixUp,
            weights.attention.q, weights.attention.k, weights.attention.v, weights.attention.o,
            weights.sharedGateProjection, weights.sharedUp, weights.sharedDown]
        blockMatrices += [weights.attentionHC.inject, weights.mlpHC.inject].compactMap { $0 }
        guard blockMatrices.allSatisfy({ matrix in
            if case .int8 = matrix { return false }; return true
        }) else {
            throw FlashNextForwardRunnerError.invalidConfiguration("MTP block activations require BF16 or INT4 projections")
        }
        try FlashNextMTPWeights.validate(model.lmHead, name: "lm_head.weight", rows: cfg.vocabSize, columns: d)
        let headMatrix = FlashNextWeightMatrix.from(model.lmHead)
        if case .int8 = headMatrix {
            throw FlashNextForwardRunnerError.invalidConfiguration("MTP output head requires BF16 or INT4 weights")
        }
        let int4 = try DequantInt4GEMV(context: context, additionalShapes: cfg.decodeInt4GEMVShapes)
        matvec = try FlashNextMatVec(context: context, int4: int4, int8Columns: d)
        fusion = try FlashNextMTPInputFusion(context: context, hidden: d, streams: fn.hcCount)
        elementwise = try Elementwise(context: context)
        hc = try FlashNextHyperConnections(context: context, rms: RMSNorm(context: context),
            matVec: matvec, hidden: d, hcCount: fn.hcCount, lowRank: fn.hcLowRank, eps: 1e-6)
        hcScratch = try hc.makeScratch(device: context.device, rows: 1)
        indexer = try FlashNextIndexer(context: context, matVec: matvec,
            geometry: .init(numHeads: fn.indexerNumHeads, numKVHeads: fn.indexerNumKVHeads,
                headDim: fn.indexerHeadDim, compressRatio: fn.indexerCompressRatio,
                blockBudget: fn.indexerBlockBudget, rotaryDim: rotary, theta: Float(cfg.fullRopeTheta), eps: 1e-6))
        indexScratch = try indexer.makeScratch(device: context.device, rows: 1, maxTokens: maxContext)
        indexCache = try indexer.makeLayerCache(device: context.device, maxTokens: maxContext)
        attention = FlashNextAttention(context: context, matVec: matvec, elementwise: elementwise,
            epilogue: try PrefillQKVEpilogue(context: context), attention: try Attention(context: context),
            prefillAttention: try PrefillAttention(context: context),
            geometry: .init(hidden: d, numHeads: cfg.numHeads, numKVHeads: cfg.numFullKVHeads,
                headDim: cfg.fullHeadDim, rotaryDim: rotary, theta: Float(cfg.fullRopeTheta),
                eps: 1e-6, scale: 1 / Float(cfg.fullHeadDim).squareRoot()))
        attentionScratch = try attention.makeScratch(device: context.device, rows: 1,
            maxSelected: indexer.maxSelected, gatherSlots: 1)
        kv = try attention.makeKVCache(device: context.device, maxTokens: maxContext)
        moe = try MoE(context: context, siluActivation: true, specializedD: UInt32(d),
            specializedF: UInt32(cfg.moeIntermediateSize), specializedNumExperts: UInt32(cfg.numExperts),
            specializedTopK: UInt32(cfg.topKExperts))
        router = try FlashNextMoE(context: context, routerTopK: cfg.topKExperts)
        silu = try context.pipeline("silu_mul_fp16")
        head = headMatrix
        switch policy {
        case .bounded(let slots):
            streamed = try PreadExpertStreamer(layout: weights.expertLayout, device: context.device, slotCount: slots)
            resident = nil
        case .resident:
            resident = try ResidentExpertStreamer(layout: weights.expertLayout, device: context.device)
            streamed = nil
        }
        func buffer(_ elements: Int, stride: Int = 2, shared: Bool = false) throws -> MTLBuffer {
            guard let b = context.device.makeBuffer(length: elements * stride,
                options: shared ? .storageModeShared : .storageModePrivate) else { throw MetalError.noDevice }
            return b
        }
        hyper = try buffer(cfg.residualStreamWidth)
        mixed = try buffer(d)
        block = try buffer(d)
        routed = try buffer(d)
        acts = try buffer(cfg.topKExperts * cfg.moeIntermediateSize)
        zero = try buffer(d, shared: true)
        memset(zero.contents(), 0, zero.length)
        routerLogits = try buffer(cfg.numExperts, stride: 4)
        routerScale = try buffer(cfg.numExperts, shared: true)
        routerScale.contents().assumingMemoryBound(to: UInt16.self)
            .update(repeating: Quantization.bf16Bits(1), count: cfg.numExperts)
        routeIDs = try buffer(cfg.topKExperts, stride: 4, shared: true)
        routeWeights = try buffer(cfg.topKExperts)
        sharedGate = try buffer(cfg.intermediateSize)
        sharedUp = try buffer(cfg.intermediateSize)
        sharedAct = try buffer(cfg.intermediateSize)
        sharedOutput = try buffer(d)
        sharedScalar = try buffer(1, stride: 4)
    }

    func reset() {
        // All state is append-only KV/indexer storage. No recurrence or PLE;
        // newly visible rows are overwritten before selection can read them.
        position = 0
        dirty = false
        epoch &+= 1
    }

    func checkpoint() throws -> Checkpoint {
        guard !dirty else { throw failure("cannot checkpoint dirty draft state") }
        return Checkpoint(owner: owner, epoch: epoch, position: position)
    }

    func restore(_ checkpoint: Checkpoint) throws {
        guard checkpoint.owner == owner, checkpoint.epoch == epoch, checkpoint.position <= position else {
            throw failure("foreign or stale draft checkpoint")
        }
        position = checkpoint.position
        dirty = false
        epoch &+= 1
    }

    /// Append exactly one aligned row. This primitive never synthesizes missing
    /// prompt KV, shifts target positions, samples a token, or advances target state.
    func append(embedding: MTLBuffer, targetHidden: MTLBuffer,
                at expectedPosition: Int, into logits: MTLBuffer) throws -> Output {
        let cfg = model.config
        let d = cfg.hiddenSize, k = cfg.topKExperts
        guard !dirty, expectedPosition == position, position < maxContext,
              embedding.length >= d * 2, targetHidden.length >= cfg.residualStreamWidth * 2,
              logits.length >= cfg.vocabSize * 2,
              logits !== embedding, logits !== targetHidden else {
            throw failure("dirty draft, missing prefix rows, exhausted context or invalid buffers")
        }
        try Task.checkCancellation()
        guard let owned = context.device.makeBuffer(length: cfg.residualStreamWidth * 2,
                                                     options: .storageModePrivate) else { throw MetalError.noDevice }
        let cb = try command()
        dirty = true
        fusion.encode(commandBuffer: cb, embedding: embedding, targetHidden: targetHidden,
            embeddingNorm: weights.embeddingNorm.buffer, embeddingNormOffset: Int(weights.embeddingNorm.offset),
            hiddenNorm: weights.hiddenNorm.buffer, hiddenNormOffset: Int(weights.hiddenNorm.offset),
            embeddingProjection: weights.embeddingProjection, hiddenProjection: weights.hiddenProjection, output: hyper)
        hc.encodeMix(commandBuffer: cb, weights: weights.attentionHC, scratch: hcScratch, hyper: hyper, mixed: mixed, rows: 1)
        hc.encodeInjectGate(commandBuffer: cb, weights: weights.attentionHC, scratch: hcScratch, rows: 1)
        indexer.encodeProjection(commandBuffer: cb, weight: weights.indexerProjection,
            x: mixed, xOffset: 0, hidden: d, scratch: indexScratch, rows: 1)
        indexer.encodePrepare(commandBuffer: cb,
            qNorm: weights.indexerQNorm.buffer, qNormOffset: Int(weights.indexerQNorm.offset),
            kNorm: weights.indexerKNorm.buffer, kNormOffset: Int(weights.indexerKNorm.offset),
            scratch: indexScratch, cache: indexCache, rows: 1, startPosition: position)
        let contiguous = indexer.selectsAllVisible(row: 0, startPosition: position)
        let count = indexer.selectionCount(row: 0, startPosition: position)
        if !contiguous {
            indexer.encodeScores(commandBuffer: cb, scratch: indexScratch, cache: indexCache, rows: 1, startPosition: position)
            indexer.encodeSelection(commandBuffer: cb, scratch: indexScratch, rows: 1, startPosition: position)
        }
        attention.encodeProjectAndCache(commandBuffer: cb, weights: weights.attention, scratch: attentionScratch,
            cache: kv, x: mixed, xOffset: 0, rows: 1, startPosition: position)
        if contiguous {
            attention.encodeAttendContiguousRow(commandBuffer: cb, scratch: attentionScratch, cache: kv, row: 0, visibleCount: count)
        } else {
            indexer.encodeGatherKV(commandBuffer: cb, kCache: kv.keys, kCacheOffset: 0, vCache: kv.values, vCacheOffset: 0,
                scratch: indexScratch, selectionRow: 0, kOut: attentionScratch.gatheredK, kOutOffset: 0,
                vOut: attentionScratch.gatheredV, vOutOffset: 0, kvDim: attention.geometry.kvDim, count: count)
            attention.encodeAttendRow(commandBuffer: cb, scratch: attentionScratch, row: 0, slot: 0, selectedCount: count)
        }
        attention.encodeGateAndProject(commandBuffer: cb, weights: weights.attention, scratch: attentionScratch,
            out: block, outOffset: 0, rows: 1)
        hc.encodeInjectAccumulate(commandBuffer: cb, scratch: hcScratch, hyper: hyper, block: block, rows: 1)
        hc.encodeMix(commandBuffer: cb, weights: weights.mlpHC, scratch: hcScratch, hyper: hyper, mixed: mixed, rows: 1)
        hc.encodeInjectGate(commandBuffer: cb, weights: weights.mlpHC, scratch: hcScratch, rows: 1)
        matvec.encode(commandBuffer: cb, matrix: weights.router, x: mixed, y: routerLogits,
            rows: cfg.numExperts, cols: d, outputFloat32: true)
        router.encodeRouterSelect(commandBuffer: cb, logits: routerLogits, perExpertScale: routerScale,
            outIndices: routeIDs, outWeights: routeWeights, numExperts: UInt32(cfg.numExperts))
        encodeShared(commandBuffer: cb)
        try finish(cb)
        try didPrepareAttention?()
        try Task.checkCancellation()
        let ids = routeIDs.contents().assumingMemoryBound(to: UInt32.self)
        let experts = (0..<k).map { Int(ids[$0]) }
        guard experts.allSatisfy({ $0 < cfg.numExperts }), Set(experts).count == k else {
            throw failure("invalid draft router indices")
        }
        let blobs: [(buffer: MTLBuffer, offset: Int)]
        if let resident {
            blobs = try experts.map { let b = try resident.expertBuffer(layer: 0, expert: $0); return (b.buffer, Int(b.offset)) }
        } else if let streamed {
            blobs = try streamed.loadExpertsCached(experts: experts).map { ($0.buffer, Int($0.offset)) }
        } else { throw failure("missing draft expert backend") }
        let tail = try command()
        let args = moe.makeReusedRoutedArgumentBuffer(routedBlobs: blobs, topK: UInt32(k))
        moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: tail, routedArgBuffer: args, routedBlobs: blobs,
            routedOffsets: weights.expertOffsets, x: mixed, acts: acts,
            d: UInt32(d), f: UInt32(cfg.moeIntermediateSize), topK: UInt32(k))
        moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: tail, routedArgBuffer: args, routedBlobs: blobs,
            routedOffsets: weights.expertOffsets, acts: acts, routingWeights: routeWeights,
            residual: zero, y: routed, d: UInt32(d), f: UInt32(cfg.moeIntermediateSize), topK: UInt32(k))
        elementwise.encodeResidualAdd(commandBuffer: tail, hidden: routed, delta: sharedOutput, count: d)
        hc.encodeInjectAccumulate(commandBuffer: tail, scratch: hcScratch, hyper: hyper, block: routed, rows: 1)
        hc.encodeMix(commandBuffer: tail, weights: weights.mixer, scratch: hcScratch, hyper: hyper, mixed: mixed, rows: 1)
        matvec.encode(commandBuffer: tail, matrix: head, x: mixed, y: logits, rows: cfg.vocabSize, cols: d)
        guard let blit = tail.makeBlitCommandEncoder() else { throw failure("cannot capture draft hidden row") }
        blit.copy(from: hyper, sourceOffset: 0, to: owned, destinationOffset: 0, size: owned.length)
        blit.endEncoding()
        try finish(tail)
        position += 1
        dirty = false
        return Output(processedRows: position, hidden: owned)
    }

    private func encodeShared(commandBuffer cb: MTLCommandBuffer) {
        let d = model.config.hiddenSize, f = model.config.intermediateSize
        matvec.encode(commandBuffer: cb, matrix: weights.sharedGateProjection, x: mixed, y: sharedGate, rows: f, cols: d)
        matvec.encode(commandBuffer: cb, matrix: weights.sharedUp, x: mixed, y: sharedUp, rows: f, cols: d)
        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(silu)
            enc.setBuffer(sharedGate, offset: 0, index: 0)
            enc.setBuffer(sharedUp, offset: 0, index: 1)
            enc.setBuffer(sharedAct, offset: 0, index: 2)
            var count = UInt32(f)
            enc.setBytes(&count, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: f, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(f, min(silu.maxTotalThreadsPerThreadgroup, 256)), height: 1, depth: 1))
            enc.endEncoding()
        }
        matvec.encode(commandBuffer: cb, matrix: weights.sharedDown, x: sharedAct, y: sharedOutput, rows: d, cols: f)
        matvec.encode(commandBuffer: cb, matrix: weights.sharedGate, x: mixed, y: sharedScalar, rows: 1, cols: d, outputFloat32: true)
        router.encodeSharedGateScale(commandBuffer: cb, out: sharedOutput, scalar: sharedScalar, count: d)
    }

    private func command() throws -> MTLCommandBuffer {
        guard let cb = context.queue.makeCommandBuffer() else { throw failure("no draft command buffer") }
        return cb
    }

    private func finish(_ cb: MTLCommandBuffer) throws {
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw failure("draft GPU command failed: \(error)") }
    }

    private func failure(_ detail: String) -> FlashNextForwardRunnerError { .invalidInput(detail) }
}
