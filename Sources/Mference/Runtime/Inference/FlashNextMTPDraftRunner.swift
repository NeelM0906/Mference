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
    private let indexer: FlashNextIndexer
    private let indexScratch: FlashNextIndexer.Scratch
    private let indexCache: FlashNextIndexer.LayerCache
    private let attention: FlashNextMTPAttention
    private let mixer: FlashNextMTPMixer
    private let moe: FlashNextMTPMoE
    private let projection: FlashNextMTPFloat32Projection
    private let narrow: MTLComputePipelineState
    private let indexInput: MTLBuffer
    private let router: FlashNextMoE
    private let resident: ResidentExpertStreamer?
    private let streamed: PreadExpertStreamer?
    private let hyper: MTLBuffer
    private let mixed: MTLBuffer
    private let block: MTLBuffer
    private let routed: MTLBuffer
    private let routerLogits: MTLBuffer
    private let routerScale: MTLBuffer
    private let routeIDs: MTLBuffer
    private let routeWeights: MTLBuffer
    private let head: FlashNextWeightMatrix
    private let embedInt4: EmbedLookupInt4
    private let embedBF16: MTLComputePipelineState
    private let embeddingRow: MTLBuffer
    private let owner = UUID()
    private var epoch: UInt64 = 0
    private var dirty = false
    private(set) var position = 0
    /// Correctness-only failure seam after attention/indexer GPU writes.
    var didPrepareAttention: (() throws -> Void)?
    /// Correctness-only stage readback; nil performs no allocation or copies.
    var didCaptureStages: (([String: [Float]]) -> Void)?

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
              rotary <= cfg.fullHeadDim, cfg.fullHeadDim <= 256, cfg.numFullKVHeads > 0,
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
        try FlashNextMTPWeights.validate(model.embedding, name: "embed_tokens.weight", rows: cfg.vocabSize, columns: d)
        if case .int8 = FlashNextWeightMatrix.from(model.embedding) {
            throw FlashNextForwardRunnerError.invalidConfiguration("MTP embedding requires BF16 or INT4 weights")
        }
        let headMatrix = FlashNextWeightMatrix.from(model.lmHead)
        if case .int8 = headMatrix {
            throw FlashNextForwardRunnerError.invalidConfiguration("MTP output head requires BF16 or INT4 weights")
        }
        let int4 = try DequantInt4GEMV(context: context, additionalShapes: cfg.decodeInt4GEMVShapes)
        matvec = try FlashNextMatVec(context: context, int4: int4, int8Columns: d)
        fusion = try FlashNextMTPInputFusion(context: context, hidden: d, streams: fn.hcCount,
            normWeightsFloat32: true, outputFloat32: true)
        projection = try FlashNextMTPFloat32Projection(context: context)
        narrow = try context.pipeline("flashnext_mtp_float_to_half")
        indexer = try FlashNextIndexer(context: context, matVec: matvec,
            geometry: .init(numHeads: fn.indexerNumHeads, numKVHeads: fn.indexerNumKVHeads,
                headDim: fn.indexerHeadDim, compressRatio: fn.indexerCompressRatio,
                blockBudget: fn.indexerBlockBudget, rotaryDim: rotary, theta: Float(cfg.fullRopeTheta), eps: 1e-6), normWeightsFloat32: true)
        indexScratch = try indexer.makeScratch(device: context.device, rows: 1, maxTokens: maxContext)
        indexCache = try indexer.makeLayerCache(device: context.device, maxTokens: maxContext)
        attention = try FlashNextMTPAttention(context: context,
            geometry: .init(hidden: d, numHeads: cfg.numHeads, numKVHeads: cfg.numFullKVHeads,
                headDim: cfg.fullHeadDim, rotaryDim: rotary, theta: Float(cfg.fullRopeTheta),
                eps: 1e-6, scale: 1 / Float(cfg.fullHeadDim).squareRoot()), maxContext: maxContext, float32IO: true)
        mixer = try FlashNextMTPMixer(context: context, hidden: d, streams: fn.hcCount,
            rank: fn.hcLowRank, vocab: cfg.vocabSize)
        moe = try FlashNextMTPMoE(context: context, hidden: d, intermediate: cfg.moeIntermediateSize,
            sharedIntermediate: cfg.intermediateSize, topK: cfg.topKExperts)
        router = try FlashNextMoE(context: context, routerTopK: cfg.topKExperts)
        head = headMatrix
        embedInt4 = try EmbedLookupInt4(context: context)
        embedBF16 = try context.pipeline("flashnext_embed_row_bf16")
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
        hyper = try buffer(cfg.residualStreamWidth, stride: 4)
        embeddingRow = try buffer(d)
        indexInput = try buffer(d)
        mixed = try buffer(d, stride: 4)
        block = try buffer(d, stride: 4)
        routed = try buffer(d, stride: 4)
        routerLogits = try buffer(cfg.numExperts, stride: 4)
        routerScale = try buffer(cfg.numExperts, shared: true)
        routerScale.contents().assumingMemoryBound(to: UInt16.self)
            .update(repeating: Quantization.bf16Bits(1), count: cfg.numExperts)
        routeIDs = try buffer(cfg.topKExperts, stride: 4, shared: true)
        routeWeights = try buffer(cfg.topKExperts)
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

    /// The shifted token is explicit: at target position i the native draft
    /// consumes embedding(token[i + 1]), not embedding(token[i]).
    func append(token: Int32, targetHidden: MTLBuffer,
                at expectedPosition: Int, into logits: MTLBuffer) throws -> Output {
        guard token >= 0, Int(token) < model.config.vocabSize,
              !dirty, expectedPosition == position, position < maxContext else {
            throw failure("invalid shifted token or draft position")
        }
        try Task.checkCancellation()
        let cb = try command()
        let tensor = model.embedding
        let d = model.config.hiddenSize
        switch FlashNextWeightMatrix.from(tensor) {
        case .int4:
            embedInt4.encode(commandBuffer: cb, table: tensor.buffer,
                tableOffset: Int(tensor.offset), scales: tensor.buffer,
                scalesOffset: Int(tensor.scaleOffset), biases: tensor.buffer,
                biasesOffset: Int(tensor.biasOffset), out: embeddingRow,
                tokenId: UInt32(token), d: UInt32(d), outScale: 1)
        case .bf16(let buffer, let offset):
            guard let enc = cb.makeComputeCommandEncoder() else { throw failure("cannot encode draft embedding") }
            enc.setComputePipelineState(embedBF16)
            enc.setBuffer(buffer, offset: offset, index: 0)
            enc.setBuffer(embeddingRow, offset: 0, index: 1)
            var row = UInt32(token), width = UInt32(d)
            enc.setBytes(&row, length: 4, index: 2)
            enc.setBytes(&width, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: d, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(d, min(embedBF16.maxTotalThreadsPerThreadgroup, 256)), height: 1, depth: 1))
            enc.endEncoding()
        case .int8:
            throw failure("unsupported draft embedding dtype")
        }
        try finish(cb)
        return try append(embedding: embeddingRow, targetHidden: targetHidden, at: expectedPosition, into: logits)
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
        var captures: [String: (MTLBuffer, Bool)] = [:]
        func capture(_ name: String, _ buffer: MTLBuffer, on command: MTLCommandBuffer, fp32: Bool = false) throws {
            guard didCaptureStages != nil else { return }
            guard let copy = context.device.makeBuffer(length: buffer.length, options: .storageModeShared),
                  let blit = command.makeBlitCommandEncoder() else { throw failure("cannot capture draft stage") }
            blit.copy(from: buffer, sourceOffset: 0, to: copy, destinationOffset: 0, size: buffer.length)
            blit.endEncoding()
            captures[name] = (copy, fp32)
        }
        dirty = true
        try capture("embedding", embedding, on: cb)
        try capture("target_hidden", targetHidden, on: cb)
        fusion.encode(commandBuffer: cb, embedding: embedding, targetHidden: targetHidden,
            embeddingNorm: weights.embeddingNorm.buffer, embeddingNormOffset: Int(weights.embeddingNorm.offset),
            hiddenNorm: weights.hiddenNorm.buffer, hiddenNormOffset: Int(weights.hiddenNorm.offset),
            embeddingProjection: weights.embeddingProjection, hiddenProjection: weights.hiddenProjection, output: hyper)
        try capture("fusion", hyper, on: cb, fp32: true)
        mixer.encodeMix(commandBuffer: cb, weights: weights.attentionHC, hyper: hyper, output: mixed)
        try capture("attention_mix", mixed, on: cb, fp32: true)
        encodeNarrow(commandBuffer: cb, input: mixed, output: indexInput, count: d)
        indexer.encodeProjection(commandBuffer: cb, weight: weights.indexerProjection,
            x: indexInput, xOffset: 0, hidden: d, scratch: indexScratch, rows: 1)
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
        attention.encode(commandBuffer: cb, weights: weights.attention, x: mixed, output: block,
            position: position, selected: indexScratch.selection, selectedCount: count, contiguous: contiguous)
        try capture("attention", block, on: cb, fp32: true)
        mixer.encodeInject(commandBuffer: cb, hyper: hyper, block: block)
        try capture("post_attention", hyper, on: cb, fp32: true)
        mixer.encodeMix(commandBuffer: cb, weights: weights.mlpHC, hyper: hyper, output: mixed)
        try capture("mlp_mix", mixed, on: cb, fp32: true)
        projection.encode(commandBuffer: cb, matrix: weights.router, x: mixed, out: routerLogits,
            rows: cfg.numExperts, columns: d)
        router.encodeRouterSelect(commandBuffer: cb, logits: routerLogits, perExpertScale: routerScale,
            outIndices: routeIDs, outWeights: routeWeights, numExperts: UInt32(cfg.numExperts))
        try capture("router", routerLogits, on: cb, fp32: true)
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
        moe.encode(commandBuffer: tail, weights: weights, blobs: blobs, x: mixed,
            routerLogits: routerLogits, routeIDs: routeIDs, output: routed)
        try capture("moe", routed, on: tail, fp32: true)
        mixer.encodeInject(commandBuffer: tail, hyper: hyper, block: routed)
        mixer.encode(commandBuffer: tail, weights: weights.mixer, head: head, hyper: hyper, output: logits)
        encodeNarrow(commandBuffer: tail, input: hyper, output: owned, count: cfg.residualStreamWidth)
        try finish(tail)
        position += 1
        dirty = false
        didCaptureStages?(captures.mapValues { buffer, fp32 in
            if fp32 {
                return Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: buffer.length / 4))
            }
            return UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float16.self), count: buffer.length / 2).map(Float.init)
        })
        return Output(processedRows: position, hidden: owned)
    }

    private func encodeNarrow(commandBuffer cb: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer, count: Int) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(narrow)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        enc.dispatchThreads(.init(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: .init(width: min(32, count), height: 1, depth: 1))
        enc.endEncoding()
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
