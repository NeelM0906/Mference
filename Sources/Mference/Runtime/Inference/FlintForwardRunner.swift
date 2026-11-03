import Metal

public enum FlintForwardRunnerError: Error, CustomStringConvertible {
    case invalidConfiguration(String)
    case invalidInput(String)
    case commandFailed(String)

    public var description: String {
        switch self {
        case .invalidConfiguration(let message),
             .invalidInput(let message),
             .commandFailed(let message):
            return message
        }
    }
}

/// Exact Flint decode and sequential prompt prefill. Resident INT8 core
/// weights stay mapped; the 47 routed layers fetch six INT4 expert blobs from
/// their per-layer streaming files.
public final class FlintForwardRunner: ContinuableLogitProducer,
    ContextWindowReporting, ChunkedPrefillRunner,
    HeadlessSequentialPrefillRunner, ExactPrefillLogitProducer, @unchecked Sendable {
    private static let hiddenSize = 2_048
    private static let vocabularySize = 150_144
    private static let layerCount = 49
    private static let denseLayerCount = 2
    private static let numQHeads = 32
    private static let numKVHeads = 4
    private static let headDim = 128
    private static let moeIntermediate = 512
    private static let sharedIntermediate = 1_024
    private static let denseIntermediate = 6_144
    private static let topK = 6
    private static let slidingWindow = 513
    private static let epsilon: Float = 1e-6
    private static let sharedExpertOverlapEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_FLINT_SHARED_OVERLAP"] != "0"
    private static let batchedPrefillEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_FLINT_BATCH_PREFILL"] != "0"
    private static let maxBatchPrefillTokens = 32

    private struct LayerTensors {
        let inputNorm: TensorView
        let postAttentionNorm: TensorView
        let q: TensorView
        let k: TensorView
        let v: TensorView
        let o: TensorView
        let qNorm: TensorView
        let kNorm: TensorView
        let sink: TensorView
        let router: TensorView?
        let sharedGate: SharedExpertProjection?
        let sharedUp: SharedExpertProjection?
        let sharedDown: SharedExpertProjection?
        let denseGate: SharedExpertProjection?
        let denseUp: SharedExpertProjection?
        let denseDown: SharedExpertProjection?
        let expertOffsets: MoEExpertOffsets?
        let sliding: Bool
        let noRope: Bool
    }

    private let model: Model
    private let context: MetalContext
    private let kv: KVCacheManager
    private let pagedKV: FlintPagedKVRuntime?
    private let norm: RMSNorm
    private let int8: DequantInt8GEMV
    private let embeddingLookup: EmbedLookupInt8
    private let attention: FlintAttention
    private let pagedAttention: Attention?
    private let router: FlintRouter
    private let rope: RoPE
    private let moe: MoE
    private let sharedExpert: SharedExpertRuntime
    private let prefillAttention: PrefillAttention
    private let groupedRoutedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let elementwise: Elementwise
    private let embedding: TensorView
    private let finalNorm: TensorView
    private let lmHead: TensorView
    private let layers: [LayerTensors]

    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let q: MTLBuffer
    private let attentionOutput: MTLBuffer
    private let attentionDelta: MTLBuffer
    private let routedInput: MTLBuffer
    private let sharedDelta: MTLBuffer
    private let ffnDelta: MTLBuffer
    private let finalNormed: MTLBuffer
    private let routerIndices: MTLBuffer
    private let routerWeights: MTLBuffer
    private let moeActs: MTLBuffer
    private let sharedActs: MTLBuffer
    private var prefillChunkState = PrefillChunkCommitState()
    private var forceExactPagedSelection = true
    private var batchPrefillScratch: FlintBatchPrefillScratch?

    public let maxContext: Int

    public init(model: Model,
                context: MetalContext,
                maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        try Self.validate(config: model.config, maxContext: maxContext)
        guard model.routedExpertCacheSlotCount(layer: Self.denseLayerCount) ?? 0
                >= Self.topK else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint requires at least six routed-expert cache slots")
        }
        self.model = model
        self.context = context
        self.maxContext = maxContext
        let paged = runtimeConfiguration.kvPagedPolicy == .on
        self.kv = try KVCacheManager(
            device: context.device,
            config: model.config,
            maxContext: maxContext,
            fp16RingEnabled: true,
            maxPrefillChunkTokens: 1,
            fp16RingCapacityOverride: Self.slidingWindow,
            pagedFullAttention: paged)
        if paged {
            let pagesPerLayer = (maxContext + KVPageGeometry.tokensPerPage - 1)
                / KVPageGeometry.tokensPerPage
            self.pagedKV = try FlintPagedKVRuntime(
                context: context,
                config: model.config,
                maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration,
                poolPagesOverride: pagesPerLayer)
            self.pagedAttention = try Attention(context: context)
        } else {
            self.pagedKV = nil
            self.pagedAttention = nil
        }
        self.norm = try RMSNorm(context: context)
        self.int8 = try DequantInt8GEMV(context: context)
        self.embeddingLookup = try EmbedLookupInt8(context: context)
        self.attention = try FlintAttention(context: context)
        self.router = try FlintRouter(context: context)
        self.rope = try RoPE(context: context)
        self.moe = try MoE(
            context: context,
            siluActivation: true,
            specializedD: UInt32(Self.hiddenSize),
            specializedF: UInt32(Self.moeIntermediate),
            specializedNumExperts: 256,
            specializedTopK: UInt32(Self.topK))
        self.sharedExpert = try SharedExpertRuntime(
            context: context,
            weightBits: 8,
            siluActivation: true)
        self.prefillAttention = try PrefillAttention(context: context)
        self.groupedRoutedMoE = try PrefillGroupedRoutedMoE(
            context: context,
            siluActivation: true)
        self.prefillMoE = try PrefillMoE(context: context)
        self.elementwise = try Elementwise(context: context)

        self.embedding = try model.resident(name: "model.embed_tokens.weight")
        self.finalNorm = try model.resident(name: "model.norm.weight")
        self.lmHead = try model.resident(name: "lm_head.weight")
        try Self.validateInt8(embedding, name: "embedding",
                              rows: Self.vocabularySize,
                              columns: Self.hiddenSize)
        try Self.validateNorm(finalNorm, name: "final norm")
        try Self.validateInt8(lmHead, name: "lm head",
                              rows: Self.vocabularySize,
                              columns: Self.hiddenSize)
        self.layers = try Self.validateLayers(model)

        func buffer(_ elements: Int) throws -> MTLBuffer {
            guard let made = context.device.makeBuffer(
                length: elements * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else {
                throw FlintForwardRunnerError.invalidConfiguration(
                    "unable to allocate Flint runtime scratch")
            }
            return made
        }
        self.hidden = try buffer(Self.hiddenSize)
        self.normed = try buffer(Self.hiddenSize)
        self.q = try buffer(Self.numQHeads * Self.headDim)
        self.attentionOutput = try buffer(Self.numQHeads * Self.headDim)
        self.attentionDelta = try buffer(Self.hiddenSize)
        self.routedInput = try buffer(Self.hiddenSize)
        self.sharedDelta = try buffer(Self.hiddenSize)
        self.ffnDelta = try buffer(Self.hiddenSize)
        self.finalNormed = try buffer(Self.hiddenSize)
        self.routerIndices = try Self.indexBuffer(context: context, count: Self.topK)
        self.routerWeights = try Self.halfBuffer(context: context, count: Self.topK)
        self.moeActs = try Self.halfBuffer(
            context: context, count: Self.topK * Self.moeIntermediate)
        self.sharedActs = try Self.halfBuffer(
            context: context, count: Self.denseIntermediate)
    }

    public func reset() {
        prefillChunkState.reset()
        pagedKV?.resetState()
        forceExactPagedSelection = true
        kv.reset()
    }

    public var continuationPosition: Int { kv.position }

    public func prepareForContinuation(expectedPosition: Int) throws {
        try prefillChunkState.requireClean(operation: "prepareForContinuation")
        guard expectedPosition > 0, expectedPosition == kv.position else {
            throw PrefillError.prefillCursorMismatch(
                "Flint continuation cursor \(expectedPosition) does not match \(kv.position)")
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        forceExactPagedSelection = false
        try await run(token: token, position: position, logits: logits)
    }

    func produceWithoutLogits(token: Int32, position: Int) async throws {
        try await run(token: token, position: position, logits: nil)
    }

    func produceExactPrefill(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await run(token: token, position: position, logits: logits)
    }

    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "Flint prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0, startPosition == kv.position else {
            throw PrefillError.chunkedUnsupported(
                "Flint chunked prefill cursor \(kv.position) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "Flint chunked prefill range exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        if Self.batchedPrefillEnabled {
            return try await prefillChunkedBatched(
                tokens: tokens,
                startPosition: startPosition,
                outputMode: outputMode,
                config: config,
                into: logits,
                onProgress: onProgress)
        }
        _ = outputMode
        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)
        forceExactPagedSelection = true
        for (offset, token) in tokens.enumerated() {
            try Task.checkCancellation()
            let isLast = offset + 1 == tokens.count
            try await run(token: token,
                          position: startPosition + offset,
                          logits: isLast ? logits : nil,
                          checkCommitState: false)
            onProgress(offset + 1)
        }
        prefillChunkState.markCommitted()
        forceExactPagedSelection = true
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    private func prefillChunkedBatched(tokens: ArraySlice<Int32>,
                                       startPosition: Int,
                                       outputMode: PrefillOutputMode,
                                       config: PrefillRuntimeConfig,
                                       into logits: MTLBuffer,
                                       onProgress: (Int) -> Void) async throws
        -> PrefillResult {
        _ = outputMode
        prefillChunkState.markDirty(startPosition: startPosition,
                                    tokenCount: tokens.count)
        forceExactPagedSelection = true
        let capacity = Self.maxBatchPrefillTokens
        let scratch: FlintBatchPrefillScratch
        if let existing = batchPrefillScratch, existing.maxTokens >= capacity {
            scratch = existing
        } else {
            scratch = try FlintBatchPrefillScratch(context: context,
                                                   maxTokens: capacity)
            batchPrefillScratch = scratch
        }

        let chunkTokens = min(capacity, config.chunkTokens)
        var cursor = 0
        while cursor < tokens.count {
            try Task.checkCancellation()
            let count = min(chunkTokens, tokens.count - cursor)
            let slice = tokens[(tokens.startIndex + cursor)..<(tokens.startIndex + cursor + count)]
            let isLast = cursor + count == tokens.count
            try await prefillBatch(slice, startPosition: startPosition + cursor,
                                   logits: isLast ? logits : nil,
                                   scratch: scratch)
            cursor += count
            onProgress(cursor)
        }
        prefillChunkState.markCommitted()
        forceExactPagedSelection = true
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    private func prefillBatch(_ tokens: ArraySlice<Int32>,
                              startPosition: Int,
                              logits: MTLBuffer?,
                              scratch: FlintBatchPrefillScratch) async throws {
        let tokenCount = tokens.count
        precondition(tokenCount > 0 && tokenCount <= scratch.maxTokens)
        let D = Self.hiddenSize
        let qStride = Self.numQHeads * Self.headDim
        let kvStride = Self.numKVHeads * Self.headDim
        let halfStride = MemoryLayout<Float16>.stride

        if let pagedKV {
            try pagedKV.prepareSelections(
                position: startPosition + tokenCount - 1,
                mode: .exact)
        }

        let embed = try commandBuffer()
        for (offset, token) in tokens.enumerated() {
            embeddingLookup.encode(
                commandBuffer: embed,
                table: embedding.buffer, tableOffset: Int(embedding.offset),
                scales: embedding.buffer, scalesOffset: Int(embedding.scaleOffset),
                biases: embedding.buffer, biasesOffset: Int(embedding.biasOffset),
                out: scratch.hidden, outOffset: offset * D * halfStride,
                tokenId: UInt32(token), d: UInt32(D))
        }
        try finish(embed)

        for (layerIndex, layer) in layers.enumerated() {
            try Task.checkCancellation()
            let block = try commandBuffer()
            if layerIndex == 0, let pagedKV {
                try pagedKV.encodePendingMetadata(commandBuffer: block)
            }

            for offset in 0..<tokenCount {
                norm.encodeBF16W(
                    commandBuffer: block,
                    x: scratch.hidden, xOffset: offset * D * halfStride,
                    weight: layer.inputNorm.buffer,
                    weightOffset: Int(layer.inputNorm.offset),
                    out: scratch.normed, outOffset: offset * D * halfStride,
                    d: UInt32(D), eps: Self.epsilon)
            }

            var keySlots: [(buffer: MTLBuffer, offset: Int)] = []
            var valueSlots: [(buffer: MTLBuffer, offset: Int)] = []
            keySlots.reserveCapacity(tokenCount)
            valueSlots.reserveCapacity(tokenCount)
            for offset in 0..<tokenCount {
                let position = startPosition + offset
                let kSlot: (buffer: MTLBuffer, offset: Int)
                let vSlot: (buffer: MTLBuffer, offset: Int)
                if let pagedKV, !layer.sliding {
                    kSlot = try pagedKV.store.kSlot(layer: layerIndex,
                                                    position: position)
                    vSlot = try pagedKV.store.vSlot(layer: layerIndex,
                                                    position: position)
                } else {
                    kSlot = kv.kSlot(layer: layerIndex, position: position)
                    vSlot = kv.vSlot(layer: layerIndex, position: position)
                }
                keySlots.append(kSlot)
                valueSlots.append(vSlot)
                encode(layer.q,
                       x: scratch.normed, xOffset: offset * D * halfStride,
                       y: scratch.q, yOffset: offset * qStride * halfStride,
                       rows: Self.numQHeads * Self.headDim,
                       columns: D, on: block)
                encode(layer.k,
                       x: scratch.normed, xOffset: offset * D * halfStride,
                       y: kSlot.buffer, yOffset: kSlot.offset,
                       rows: kvStride, columns: D, on: block)
                encode(layer.v,
                       x: scratch.normed, xOffset: offset * D * halfStride,
                       y: vSlot.buffer, yOffset: vSlot.offset,
                       rows: kvStride, columns: D, on: block)
            }

            for offset in 0..<tokenCount {
                let qOffset = offset * qStride * halfStride
                norm.encodeBF16WPerHead(
                    commandBuffer: block,
                    x: scratch.q, xOffset: qOffset,
                    weight: layer.qNorm.buffer,
                    weightOffset: Int(layer.qNorm.offset),
                    out: scratch.q, outOffset: qOffset,
                    headDim: UInt32(Self.headDim),
                    numHeads: Self.numQHeads,
                    eps: Self.epsilon)
                let kSlot = keySlots[offset]
                norm.encodeBF16WPerHead(
                    commandBuffer: block,
                    x: kSlot.buffer, xOffset: kSlot.offset,
                    weight: layer.kNorm.buffer,
                    weightOffset: Int(layer.kNorm.offset),
                    out: kSlot.buffer, outOffset: kSlot.offset,
                    headDim: UInt32(Self.headDim),
                    numHeads: Self.numKVHeads,
                    eps: Self.epsilon)
                if !layer.noRope {
                    rope.encodeNeoxSubdim(
                        commandBuffer: block,
                        data: scratch.q, dataOffset: qOffset,
                        position: UInt32(startPosition + offset),
                        headDim: UInt32(Self.headDim),
                        numHeads: UInt32(Self.numQHeads),
                        rotaryDim: UInt32(Self.headDim),
                        theta: Float(model.config.ropeTheta))
                    rope.encodeNeoxSubdim(
                        commandBuffer: block,
                        data: kSlot.buffer, dataOffset: kSlot.offset,
                        position: UInt32(startPosition + offset),
                        headDim: UInt32(Self.headDim),
                        numHeads: UInt32(Self.numKVHeads),
                        rotaryDim: UInt32(Self.headDim),
                        theta: Float(model.config.ropeTheta))
                }
            }

            let kvBuffer: MTLBuffer
            let vBuffer: MTLBuffer
            if layer.sliding {
                kvBuffer = keySlots[0].buffer
                vBuffer = valueSlots[0].buffer
            } else if let pagedKV {
                kvBuffer = pagedKV.store.kPoolBuffer(layer: layerIndex)
                vBuffer = pagedKV.store.vPoolBuffer(layer: layerIndex)
            } else {
                kvBuffer = keySlots[0].buffer
                vBuffer = valueSlots[0].buffer
            }
            prefillAttention.encodeCausal(
                commandBuffer: block,
                q: scratch.q,
                k: kvBuffer,
                v: vBuffer,
                out: scratch.attentionOut,
                params: PrefillAttentionParams(
                    startPosition: UInt32(startPosition),
                    queryCount: UInt32(tokenCount),
                    headDim: UInt32(Self.headDim),
                    numQHeads: UInt32(Self.numQHeads),
                    numKVHeads: UInt32(Self.numKVHeads),
                    kvValidCount: UInt32(startPosition + tokenCount),
                    slidingWindow: layer.sliding ? UInt32(Self.slidingWindow) : 0,
                    kvTokenStrideElements: UInt32(kvStride),
                    qTokenStrideElements: UInt32(qStride),
                    oTokenStrideElements: UInt32(qStride),
                    scale: Float(model.config.attentionScale)),
                kvRingCapacity: layer.sliding
                    ? UInt32(kv.ringCapacity(layer: layerIndex)) : 0,
                sinks: layer.sink.buffer,
                sinksOffset: Int(layer.sink.offset),
                numSinks: UInt32(Self.numQHeads))

            for offset in 0..<tokenCount {
                let row = offset * D * halfStride
                encode(layer.o,
                       x: scratch.attentionOut,
                       xOffset: offset * qStride * halfStride,
                       y: scratch.attentionDelta, yOffset: row,
                       rows: D, columns: qStride, on: block)
                elementwise.encodeResidualAdd(
                    commandBuffer: block,
                    hidden: scratch.hidden, hiddenOffset: row,
                    delta: scratch.attentionDelta, deltaOffset: row,
                    count: D)
                norm.encodeBF16W(
                    commandBuffer: block,
                    x: scratch.hidden, xOffset: row,
                    weight: layer.postAttentionNorm.buffer,
                    weightOffset: Int(layer.postAttentionNorm.offset),
                    out: scratch.routedInput, outOffset: row,
                    d: UInt32(D), eps: Self.epsilon)
            }
            if let routerView = layer.router {
                scratch.router.encode(
                    commandBuffer: block,
                    weights: routerView.buffer,
                    weightsOffset: Int(routerView.offset),
                    hidden: scratch.routedInput,
                    hiddenStrideElements: D,
                    indices: scratch.routerIndices,
                    routingWeights: scratch.routerWeights,
                    tokens: tokenCount)
            }
            try finish(block)

            if layer.router == nil {
                try encodeBatchedDenseFFN(layer: layer,
                                          layerIndex: layerIndex,
                                          tokenCount: tokenCount,
                                          scratch: scratch)
            } else {
                try await encodeBatchedMoEFFN(layer: layer,
                                              layerIndex: layerIndex,
                                              tokenCount: tokenCount,
                                              scratch: scratch)
            }
        }

        if let logits {
            let head = try commandBuffer()
            let row = (tokenCount - 1) * D * halfStride
            norm.encodeBF16W(
                commandBuffer: head,
                x: scratch.hidden, xOffset: row,
                weight: finalNorm.buffer,
                weightOffset: Int(finalNorm.offset),
                out: finalNormed, d: UInt32(D), eps: Self.epsilon)
            int8.encode(
                commandBuffer: head,
                weights: lmHead.buffer, weightsOffset: Int(lmHead.offset),
                scales: lmHead.buffer, scalesOffset: Int(lmHead.scaleOffset),
                biases: lmHead.buffer, biasesOffset: Int(lmHead.biasOffset),
                x: finalNormed, y: logits,
                m: UInt32(Self.vocabularySize), n: UInt32(D))
            try finish(head)
        }

        if let pagedKV {
            pagedKV.store.advance(by: tokenCount)
            pagedKV.noteAdvance(from: startPosition,
                                to: startPosition + tokenCount)
            let score = try commandBuffer()
            try pagedKV.encodePendingMetadata(commandBuffer: score)
            let sealedPages = (startPosition + tokenCount)
                / KVPageGeometry.tokensPerPage
            for ordinal in 0..<pagedKV.store.geometry.fullLayerOrdinals.count {
                pagedKV.encodeScores(
                    commandBuffer: score,
                    ordinal: ordinal,
                    q: scratch.q,
                    qOffset: (tokenCount - 1) * qStride * halfStride,
                    sealedPages: sealedPages)
            }
            try finish(score)
            pagedKV.readBackScores(sealedPages: sealedPages)
        }
        kv.advance(by: tokenCount)
    }

    private func encodeBatchedDenseFFN(layer: LayerTensors,
                                       layerIndex: Int,
                                       tokenCount: Int,
                                       scratch: FlintBatchPrefillScratch) throws {
        guard let gate = layer.denseGate,
              let up = layer.denseUp,
              let down = layer.denseDown else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint dense layer \(layerIndex) is missing projections")
        }
        let command = try commandBuffer()
        let D = Self.hiddenSize
        let halfStride = MemoryLayout<Float16>.stride
        let actStride = Self.denseIntermediate * halfStride
        for offset in 0..<tokenCount {
            try sharedExpert.encode(
                commandBuffer: command,
                x: scratch.routedInput, xOffset: offset * D * halfStride,
                gate: gate, up: up, down: down,
                y: scratch.ffnDelta, yOffset: offset * D * halfStride,
                scratchGate: scratch.sharedActs,
                scratchGateOffset: offset * actStride,
                scratchUp: scratch.sharedActs,
                scratchUpOffset: offset * actStride,
                scratchAct: scratch.sharedActs,
                scratchActOffset: offset * actStride)
        }
        for offset in 0..<tokenCount {
            let row = offset * D * halfStride
            elementwise.encodeResidualAdd(
                commandBuffer: command,
                hidden: scratch.hidden, hiddenOffset: row,
                delta: scratch.ffnDelta, deltaOffset: row,
                count: D)
        }
        try finish(command)
    }

    private func encodeBatchedMoEFFN(layer: LayerTensors,
                                     layerIndex: Int,
                                     tokenCount: Int,
                                     scratch: FlintBatchPrefillScratch) async throws {
        guard let gate = layer.sharedGate,
              let up = layer.sharedUp,
              let down = layer.sharedDown,
              let offsets = layer.expertOffsets else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint MoE layer \(layerIndex) is missing projections")
        }
        let D = Self.hiddenSize
        let F = Self.moeIntermediate
        let halfStride = MemoryLayout<Float16>.stride
        let actStride = Self.denseIntermediate * halfStride

        let shared = try commandBuffer()
        for offset in 0..<tokenCount {
            try sharedExpert.encode(
                commandBuffer: shared,
                x: scratch.routedInput, xOffset: offset * D * halfStride,
                gate: gate, up: up, down: down,
                y: scratch.ffnDelta, yOffset: offset * D * halfStride,
                scratchGate: scratch.sharedActs,
                scratchGateOffset: offset * actStride,
                scratchUp: scratch.sharedActs,
                scratchUpOffset: offset * actStride,
                scratchAct: scratch.sharedActs,
                scratchActOffset: offset * actStride)
        }
        try finish(shared)

        let indices = scratch.routerIndices.contents()
            .bindMemory(to: UInt32.self,
                        capacity: tokenCount * Self.topK)
        let weights = scratch.routerWeights.contents()
            .bindMemory(to: Float16.self,
                        capacity: tokenCount * Self.topK)
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(tokenCount * Self.topK)
        for token in 0..<tokenCount {
            for rank in 0..<Self.topK {
                let i = token * Self.topK + rank
                pairs.append(PrefillTokenExpertPair(
                    token: UInt32(token),
                    expert: indices[i],
                    rank: UInt32(rank),
                    weight: weights[i]))
            }
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: tokenCount,
            topK: Self.topK,
            numExperts: 256,
            tileExpertCount: 16)
        let metadata = try groupedRoutedMoE.makeStreamedMetadataBuffers(
            device: context.device,
            routes: routes)

        for tileIndex in routes.tiles.indices {
            let tile = routes.tiles[tileIndex]
            let fetched = try await PrefillStreamedTileBinding.fetchBindingForTile(
                model: model,
                layer: layerIndex,
                tileIndex: tileIndex,
                routes: routes)
            let argument = try groupedRoutedMoE.makeStreamedArgumentBuffer(
                device: context.device,
                binding: fetched.binding)
            let params = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(D),
                routedIntermediate: UInt32(F),
                topK: UInt32(Self.topK),
                hiddenStrideElements: UInt32(D),
                binding: fetched.binding,
                offsets: offsets)
            let command = try commandBuffer()
            _ = groupedRoutedMoE.encodeStreamedBatched(
                commandBuffer: command,
                hidden: scratch.routedInput,
                sortedPairs: metadata.sortedPairs,
                routePartials: scratch.routePartials,
                gateUpActScratch: scratch.gateUpAct,
                downScratch: scratch.downScratch,
                argumentBuffer: argument,
                binding: fetched.binding,
                params: params)
            try finish(command)
        }

        let reduce = try commandBuffer()
        prefillMoE.encodeReduceTokenMajor(
            commandBuffer: reduce,
            routePartials: scratch.routePartials,
            routeWeights: scratch.routerWeights,
            h2: scratch.h2,
            queryCount: UInt32(tokenCount),
            topK: UInt32(Self.topK),
            d: UInt32(D))
        for offset in 0..<tokenCount {
            let row = offset * D * halfStride
            elementwise.encodeResidualAdd(
                commandBuffer: reduce,
                hidden: scratch.ffnDelta, hiddenOffset: row,
                delta: scratch.h2, deltaOffset: row,
                count: D)
            elementwise.encodeResidualAdd(
                commandBuffer: reduce,
                hidden: scratch.hidden, hiddenOffset: row,
                delta: scratch.ffnDelta, deltaOffset: row,
                count: D)
        }
        try finish(reduce)
    }

    private func run(token: Int32,
                     position: Int,
                     logits: MTLBuffer?,
                     selectionMode: FlintPagedKVRuntime.SelectionMode? = nil,
                     checkCommitState: Bool = true) async throws {
        if checkCommitState {
            try prefillChunkState.requireClean(operation: "produce")
        }
        try Task.checkCancellation()
        guard position == kv.position, position >= 0, position < maxContext else {
            throw FlintForwardRunnerError.invalidInput(
                "Flint position does not match its KV cursor")
        }
        guard token >= 0, token < Int32(Self.vocabularySize) else {
            throw FlintForwardRunnerError.invalidInput(
                "Flint token is outside the vocabulary")
        }
        if let logits,
           logits.length < Self.vocabularySize * MemoryLayout<Float16>.stride {
            throw FlintForwardRunnerError.invalidInput(
                "Flint logits buffer is too small")
        }

        if let pagedKV {
            let mode = selectionMode
                ?? (forceExactPagedSelection ? .exact : .sparse)
            try pagedKV.prepareSelections(position: position, mode: mode)
        }
        let embedCB = try commandBuffer()
        if let pagedKV {
            try pagedKV.encodePendingMetadata(commandBuffer: embedCB)
        }
        embeddingLookup.encode(
            commandBuffer: embedCB,
            table: embedding.buffer, tableOffset: Int(embedding.offset),
            scales: embedding.buffer, scalesOffset: Int(embedding.scaleOffset),
            biases: embedding.buffer, biasesOffset: Int(embedding.biasOffset),
            out: hidden, tokenId: UInt32(token), d: UInt32(Self.hiddenSize))
        try finish(embedCB)

        for layerIndex in 0..<Self.layerCount {
            try Task.checkCancellation()
            let layer = layers[layerIndex]
            let kSlot: (buffer: MTLBuffer, offset: Int)
            let vSlot: (buffer: MTLBuffer, offset: Int)
            if let pagedKV, !layer.sliding {
                kSlot = try pagedKV.store.kSlot(layer: layerIndex,
                                                position: position)
                vSlot = try pagedKV.store.vSlot(layer: layerIndex,
                                                position: position)
            } else {
                kSlot = kv.kSlot(layer: layerIndex, position: position)
                vSlot = kv.vSlot(layer: layerIndex, position: position)
            }

            let block = try commandBuffer()
            norm.encodeBF16W(
                commandBuffer: block,
                x: hidden, weight: layer.inputNorm.buffer,
                weightOffset: Int(layer.inputNorm.offset),
                out: normed, d: UInt32(Self.hiddenSize), eps: Self.epsilon)
            encode(layer.q, x: normed, y: q,
                   rows: Self.numQHeads * Self.headDim,
                   on: block)
            encode(layer.k, x: normed, y: kSlot.buffer,
                   yOffset: kSlot.offset,
                   rows: Self.numKVHeads * Self.headDim,
                   on: block)
            encode(layer.v, x: normed, y: vSlot.buffer,
                   yOffset: vSlot.offset,
                   rows: Self.numKVHeads * Self.headDim,
                   on: block)
            norm.encodeBF16WPerHead(
                commandBuffer: block,
                x: q,
                weight: layer.qNorm.buffer,
                weightOffset: Int(layer.qNorm.offset),
                out: q,
                headDim: UInt32(Self.headDim),
                numHeads: Self.numQHeads,
                eps: Self.epsilon)
            norm.encodeBF16WPerHead(
                commandBuffer: block,
                x: kSlot.buffer, xOffset: kSlot.offset,
                weight: layer.kNorm.buffer,
                weightOffset: Int(layer.kNorm.offset),
                out: kSlot.buffer, outOffset: kSlot.offset,
                headDim: UInt32(Self.headDim),
                numHeads: Self.numKVHeads,
                eps: Self.epsilon)
            if !layer.noRope {
                rope.encodeNeoxSubdim(
                    commandBuffer: block,
                    data: q, position: UInt32(position),
                    headDim: UInt32(Self.headDim),
                    numHeads: UInt32(Self.numQHeads),
                    rotaryDim: UInt32(Self.headDim),
                    theta: Float(model.config.ropeTheta))
                rope.encodeNeoxSubdim(
                    commandBuffer: block,
                    data: kSlot.buffer, dataOffset: kSlot.offset,
                    position: UInt32(position),
                    headDim: UInt32(Self.headDim),
                    numHeads: UInt32(Self.numKVHeads),
                    rotaryDim: UInt32(Self.headDim),
                    theta: Float(model.config.ropeTheta))
            }
            if layer.sliding {
                let end = position + 1
                let start = max(0, end - Self.slidingWindow)
                attention.encode(
                    commandBuffer: block,
                    q: q,
                    k: kSlot.buffer,
                    v: vSlot.buffer,
                    sinks: layer.sink.buffer,
                    sinksOffset: Int(layer.sink.offset),
                    out: attentionOutput,
                    start: UInt32(start),
                    end: UInt32(end),
                    ringCapacity: UInt32(kv.ringCapacity(layer: layerIndex)),
                    scale: Float(model.config.attentionScale))
            } else if let pagedKV, let pagedAttention {
                guard let ordinal = pagedKV.store.fullLayerOrdinal(
                    forLayer: layerIndex) else {
                    throw FlintForwardRunnerError.invalidConfiguration(
                        "Flint paged attention on a non-full-attention layer")
                }
                let selection = pagedKV.selections[ordinal]
                pagedAttention.encodeFullPaged(
                    commandBuffer: block,
                    q: q,
                    kPool: pagedKV.store.kPoolBuffer(layer: layerIndex),
                    vPool: pagedKV.store.vPoolBuffer(layer: layerIndex),
                    pageTable: pagedKV.tablesBuf,
                    pageTableOffset: ordinal * pagedKV.store.geometry.pagesPerLayer
                        * MemoryLayout<UInt32>.stride,
                    out: attentionOutput,
                    headDim: UInt32(Self.headDim),
                    numQHeads: UInt32(Self.numQHeads),
                    numKVHeads: UInt32(Self.numKVHeads),
                    selTokens: UInt32(selection.selTokens),
                    scale: Float(model.config.attentionScale),
                    sinks: layer.sink.buffer,
                    sinksOffset: Int(layer.sink.offset),
                    numSinks: UInt32(Self.numQHeads))
                pagedKV.encodeScores(
                    commandBuffer: block,
                    ordinal: ordinal,
                    q: q,
                    qOffset: 0,
                    sealedPages: position / KVPageGeometry.tokensPerPage)
            } else {
                attention.encode(
                    commandBuffer: block,
                    q: q,
                    k: kSlot.buffer,
                    v: vSlot.buffer,
                    sinks: layer.sink.buffer,
                    sinksOffset: Int(layer.sink.offset),
                    out: attentionOutput,
                    start: 0,
                    end: UInt32(position + 1),
                    ringCapacity: 0,
                    scale: Float(model.config.attentionScale))
            }
            encode(layer.o, x: attentionOutput, y: attentionDelta,
                   rows: Self.hiddenSize,
                   columns: Self.numQHeads * Self.headDim,
                   on: block)
            elementwise.encodeResidualAdd(
                commandBuffer: block,
                hidden: hidden, delta: attentionDelta,
                count: Self.hiddenSize)
            norm.encodeBF16W(
                commandBuffer: block,
                x: hidden, weight: layer.postAttentionNorm.buffer,
                weightOffset: Int(layer.postAttentionNorm.offset),
                out: routedInput, d: UInt32(Self.hiddenSize), eps: Self.epsilon)
            if let routerView = layer.router {
                router.encode(
                    commandBuffer: block,
                    weights: routerView.buffer,
                    weightsOffset: Int(routerView.offset),
                    hidden: routedInput,
                    indices: routerIndices,
                    routingWeights: routerWeights)
            }
            try finish(block)
            try Task.checkCancellation()
            try await encodeFeedForward(layer: layer, index: layerIndex)
        }

        if let logits {
            let head = try commandBuffer()
            norm.encodeBF16W(
                commandBuffer: head,
                x: hidden, weight: finalNorm.buffer,
                weightOffset: Int(finalNorm.offset),
                out: finalNormed, d: UInt32(Self.hiddenSize), eps: Self.epsilon)
            int8.encode(
                commandBuffer: head,
                weights: lmHead.buffer, weightsOffset: Int(lmHead.offset),
                scales: lmHead.buffer, scalesOffset: Int(lmHead.scaleOffset),
                biases: lmHead.buffer, biasesOffset: Int(lmHead.biasOffset),
                x: finalNormed, y: logits,
                m: UInt32(Self.vocabularySize), n: UInt32(Self.hiddenSize))
            try finish(head)
        }
        if let pagedKV {
            pagedKV.readBackScores(
                sealedPages: position / KVPageGeometry.tokensPerPage)
            pagedKV.store.advance()
            pagedKV.noteAdvance(from: position, to: position + 1)
        }
        kv.advance()
    }

    private func encodeFeedForward(layer: LayerTensors, index: Int) async throws {
        if index < Self.denseLayerCount {
            let command = try commandBuffer()
            guard let gate = layer.denseGate,
                  let up = layer.denseUp,
                  let down = layer.denseDown else {
                throw FlintForwardRunnerError.invalidConfiguration(
                    "Flint dense layer \(index) is missing projections")
            }
            try sharedExpert.encode(
                commandBuffer: command,
                x: routedInput, gate: gate, up: up, down: down,
                y: ffnDelta,
                scratchGate: sharedActs, scratchUp: sharedActs,
                scratchAct: sharedActs)
            elementwise.encodeResidualAdd(
                commandBuffer: command,
                hidden: hidden, delta: ffnDelta, count: Self.hiddenSize)
            try finish(command)
            return
        }

        guard let gate = layer.sharedGate,
              let up = layer.sharedUp,
              let down = layer.sharedDown,
              let offsets = layer.expertOffsets else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint MoE layer \(index) is missing projections")
        }

        if !Self.sharedExpertOverlapEnabled {
            let command = try commandBuffer()
            try sharedExpert.encode(
                commandBuffer: command,
                x: routedInput, gate: gate, up: up, down: down,
                y: sharedDelta,
                scratchGate: sharedActs, scratchUp: sharedActs,
                scratchAct: sharedActs)
            let ranks = try routedRanks()
            guard let plan = try model.planRoutedExperts(layer: index,
                                                          experts: ranks) else {
                throw ModelError.routedExpertPlanUnavailable(layer: index)
            }
            try await encodeRoutedExperts(
                plan: plan,
                offsets: offsets,
                commandBuffer: command)
            elementwise.encodeResidualAdd(
                commandBuffer: command,
                hidden: hidden, delta: ffnDelta, count: Self.hiddenSize)
            try finish(command)
            return
        }

        // Commit the shared expert before waiting for routed-expert I/O. Its
        // GPU work then overlaps the six expert preads instead of starting
        // only after they land.
        let sharedCommand = try commandBuffer()
        try sharedExpert.encode(
            commandBuffer: sharedCommand,
            x: routedInput, gate: gate, up: up, down: down,
            y: sharedDelta,
            scratchGate: sharedActs, scratchUp: sharedActs,
            scratchAct: sharedActs)
        sharedCommand.commit()

        let ranks = try routedRanks()
        guard let plan = try model.planRoutedExperts(layer: index, experts: ranks) else {
            _ = try? wait(sharedCommand)
            throw ModelError.routedExpertPlanUnavailable(layer: index)
        }
        let routedCommand = try commandBuffer()
        do {
            try await encodeRoutedExperts(
                plan: plan,
                offsets: offsets,
                commandBuffer: routedCommand,
                beforeEncode: { try self.wait(sharedCommand) })
        } catch {
            _ = try? wait(sharedCommand)
            throw error
        }
        elementwise.encodeResidualAdd(
            commandBuffer: routedCommand,
            hidden: hidden, delta: ffnDelta, count: Self.hiddenSize)
        try finish(routedCommand)
    }

    private func encodeRoutedExperts(
        plan: RoutedExpertFetchPlan,
        offsets: MoEExpertOffsets,
        commandBuffer: MTLCommandBuffer,
        beforeEncode: () throws -> Void = {}) async throws {
        let plannedViews = try model.routedExpertBuffers(for: plan)
        let plannedBlobs = try expertBlobs(plannedViews)
        if !plan.misses.isEmpty {
            let loadedViews = try await model.fetchRoutedExperts(plan: plan)
            let loadedBlobs = try expertBlobs(loadedViews)
            guard zip(plannedBlobs, loadedBlobs).allSatisfy({
                $0.0.buffer === $0.1.buffer
                    && $0.0.offset == $0.1.offset
                    && $0.0.length == $0.1.length
            }) else {
                throw FlintForwardRunnerError.invalidConfiguration(
                    "Flint expert fetch changed the planned cache-slot binding")
            }
        }
        try beforeEncode()
        let argument = moe.makeReusedRoutedArgumentBuffer(
            routedBlobs: plannedBlobs.map { (buffer: $0.buffer, offset: $0.offset) },
            topK: UInt32(Self.topK))
        moe.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: commandBuffer,
            routedArgBuffer: argument,
            routedBlobs: plannedBlobs.map { (buffer: $0.buffer, offset: $0.offset) },
            routedOffsets: offsets,
            x: routedInput,
            acts: moeActs,
            d: UInt32(Self.hiddenSize),
            f: UInt32(Self.moeIntermediate),
            topK: UInt32(Self.topK))
        moe.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: commandBuffer,
            routedArgBuffer: argument,
            routedBlobs: plannedBlobs.map { (buffer: $0.buffer, offset: $0.offset) },
            routedOffsets: offsets,
            acts: moeActs,
            routingWeights: routerWeights,
            residual: sharedDelta,
            y: ffnDelta,
            d: UInt32(Self.hiddenSize),
            f: UInt32(Self.moeIntermediate),
            topK: UInt32(Self.topK))
    }

    private func routedRanks() throws -> [Int] {
        let values = routerIndices.contents()
            .bindMemory(to: UInt32.self, capacity: Self.topK)
        let ranks = (0..<Self.topK).map { Int(values[$0]) }
        guard Set(ranks).count == Self.topK,
              ranks.allSatisfy({ $0 >= 0 && $0 < 256 }) else {
            throw FlintForwardRunnerError.invalidInput(
                "Flint router returned invalid top-6 ranks")
        }
        return ranks
    }

    private func expertBlobs(_ views: [TensorView]) throws
        -> [(buffer: MTLBuffer, offset: Int, length: Int)] {
        guard views.count == Self.topK,
              views.allSatisfy({
                  $0.offset + $0.length <= UInt64($0.buffer.length)
              }) else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint routed expert views must stay within their cache buffers")
        }
        return views.map {
            (buffer: $0.buffer, offset: Int($0.offset), length: Int($0.length))
        }
    }

    private func encode(_ view: TensorView,
                        x: MTLBuffer, xOffset: Int = 0,
                        y: MTLBuffer, yOffset: Int = 0,
                        rows: Int,
                        columns: Int = hiddenSize,
                        on commandBuffer: MTLCommandBuffer) {
        int8.encode(
            commandBuffer: commandBuffer,
            weights: view.buffer, weightsOffset: Int(view.offset),
            scales: view.buffer, scalesOffset: Int(view.scaleOffset),
            biases: view.buffer, biasesOffset: Int(view.biasOffset),
            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
            m: UInt32(rows), n: UInt32(columns))
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let buffer = context.queue.makeCommandBuffer() else {
            throw FlintForwardRunnerError.commandFailed(
                "unable to create Flint command buffer")
        }
        return buffer
    }

    private func finish(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        try wait(commandBuffer)
    }

    private func wait(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw FlintForwardRunnerError.commandFailed(
                commandBuffer.error?.localizedDescription
                    ?? "Flint command buffer did not complete")
        }
    }

    private static func validate(config: ArchConfig, maxContext: Int) throws {
        guard config == .flint40B_A2_5B_JANG_4M else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "model is not the pinned Flint-40B-A2.5B-JANG_4M geometry")
        }
        guard maxContext > 0, maxContext <= 262_144 else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint runtime context must be 1...262,144")
        }
    }

    private static func validateLayers(_ model: Model) throws -> [LayerTensors] {
        let layout = model.packedExpertsLayout
        let projectionWeight = UInt64(moeIntermediate * hiddenSize / 2)
        let projectionCompanion = UInt64(
            moeIntermediate * (hiddenSize / Quantization.groupSize) * 2)
        let downWeight = UInt64(hiddenSize * moeIntermediate / 2)
        let downCompanion = UInt64(
            hiddenSize * (moeIntermediate / Quantization.groupSize) * 2)
        let expectedStride = 3 * (projectionWeight + 2 * projectionCompanion)
        guard downWeight == projectionWeight,
              downCompanion == projectionCompanion,
              layout.numLayers == layerCount,
              layout.layers.count == layerCount,
              layout.expertsPerLayer == 256,
              layout.expertStride == expectedStride else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint routed-expert layout geometry is invalid")
        }
        let expectedSlices: [(String, UInt64, UInt64)] = [
            ("gate", 0, projectionWeight),
            ("gate_scales", projectionWeight, projectionCompanion),
            ("gate_biases", projectionWeight + projectionCompanion, projectionCompanion),
            ("up", projectionWeight + 2 * projectionCompanion, projectionWeight),
            ("up_scales", 2 * projectionWeight + 2 * projectionCompanion,
             projectionCompanion),
            ("up_biases", 2 * projectionWeight + 3 * projectionCompanion,
             projectionCompanion),
            ("down", 2 * (projectionWeight + 2 * projectionCompanion), downWeight),
            ("down_scales", 3 * projectionWeight + 4 * projectionCompanion,
             downCompanion),
            ("down_biases", 3 * projectionWeight + 5 * projectionCompanion,
             downCompanion),
        ]
        for layer in 0..<layerCount {
            let entry = layout.layers[layer]
            let expectedName = String(format: "layer_%02d.bin", layer)
            guard entry.layer == layer, entry.file == expectedName else {
                throw FlintForwardRunnerError.invalidConfiguration(
                    "Flint routed-expert layer \(layer) file is invalid")
            }
            if layer < denseLayerCount {
                guard entry.experts.isEmpty else {
                    throw FlintForwardRunnerError.invalidConfiguration(
                        "Flint dense layer \(layer) unexpectedly carries routed experts")
                }
                continue
            }
            guard entry.experts.count == 256 else {
                throw FlintForwardRunnerError.invalidConfiguration(
                    "Flint routed-expert layer \(layer) must contain 256 experts")
            }
            for expert in entry.experts {
                guard expert.size == expectedStride,
                      expert.subTensors.count == expectedSlices.count else {
                    throw FlintForwardRunnerError.invalidConfiguration(
                        "Flint routed-expert blob \(layer)/\(expert.expert) is invalid")
                }
                for (name, offset, size) in expectedSlices {
                    guard let tensor = expert.subTensors[name],
                          tensor.offset == offset,
                          tensor.size == size else {
                        throw FlintForwardRunnerError.invalidConfiguration(
                            "Flint routed-expert tensor \(name) is invalid")
                    }
                }
            }
        }

        return try (0..<layerCount).map { layer in
            let prefix = "model.layers.\(layer)"
            let inputNorm = try model.resident(name: prefix + ".input_layernorm.weight")
            let postNorm = try model.resident(
                name: prefix + ".post_attention_layernorm.weight")
            let q = try model.resident(name: prefix + ".self_attn.q_proj.weight")
            let k = try model.resident(name: prefix + ".self_attn.k_proj.weight")
            let v = try model.resident(name: prefix + ".self_attn.v_proj.weight")
            let o = try model.resident(name: prefix + ".self_attn.o_proj.weight")
            let qNorm = try model.resident(name: prefix + ".self_attn.q_norm.weight")
            let kNorm = try model.resident(name: prefix + ".self_attn.k_norm.weight")
            let sink = try model.resident(name: prefix + ".self_attn.softmax_offset")
            try validateNorm(inputNorm, name: "layer \(layer) input norm")
            try validateNorm(postNorm, name: "layer \(layer) post-attn norm")
            try validateNorm(qNorm, name: "layer \(layer) Q norm", rows: headDim)
            try validateNorm(kNorm, name: "layer \(layer) K norm", rows: headDim)
            try validateNorm(sink, name: "layer \(layer) attention sink",
                              rows: numQHeads)
            try validateInt8(q, name: "layer \(layer) Q",
                             rows: numQHeads * headDim, columns: hiddenSize)
            try validateInt8(k, name: "layer \(layer) K",
                             rows: numKVHeads * headDim, columns: hiddenSize)
            try validateInt8(v, name: "layer \(layer) V",
                             rows: numKVHeads * headDim, columns: hiddenSize)
            try validateInt8(o, name: "layer \(layer) O",
                             rows: hiddenSize, columns: numQHeads * headDim)

            var router: TensorView?
            var sharedGate: SharedExpertProjection?
            var sharedUp: SharedExpertProjection?
            var sharedDown: SharedExpertProjection?
            var denseGate: SharedExpertProjection?
            var denseUp: SharedExpertProjection?
            var denseDown: SharedExpertProjection?
            var offsets: MoEExpertOffsets?
            if layer < denseLayerCount {
                let g = try model.resident(name: prefix + ".mlp.gate_proj.weight")
                let u = try model.resident(name: prefix + ".mlp.up_proj.weight")
                let d = try model.resident(name: prefix + ".mlp.down_proj.weight")
                try validateInt8(g, name: "layer \(layer) dense gate",
                                 rows: denseIntermediate, columns: hiddenSize)
                try validateInt8(u, name: "layer \(layer) dense up",
                                 rows: denseIntermediate, columns: hiddenSize)
                try validateInt8(d, name: "layer \(layer) dense down",
                                 rows: hiddenSize, columns: denseIntermediate)
                denseGate = projection(g, rows: denseIntermediate, columns: hiddenSize)
                denseUp = projection(u, rows: denseIntermediate, columns: hiddenSize)
                denseDown = projection(d, rows: hiddenSize, columns: denseIntermediate)
            } else {
                let r = try model.resident(name: prefix + ".mlp.gate.weight")
                try validateBF16(r, name: "layer \(layer) router",
                                 rows: 256, columns: hiddenSize)
                router = r
                let g = try model.resident(
                    name: prefix + ".mlp.shared_expert.gate_proj.weight")
                let u = try model.resident(
                    name: prefix + ".mlp.shared_expert.up_proj.weight")
                let d = try model.resident(
                    name: prefix + ".mlp.shared_expert.down_proj.weight")
                try validateInt8(g, name: "layer \(layer) shared gate",
                                 rows: sharedIntermediate, columns: hiddenSize)
                try validateInt8(u, name: "layer \(layer) shared up",
                                 rows: sharedIntermediate, columns: hiddenSize)
                try validateInt8(d, name: "layer \(layer) shared down",
                                 rows: hiddenSize, columns: sharedIntermediate)
                sharedGate = projection(g, rows: sharedIntermediate, columns: hiddenSize)
                sharedUp = projection(u, rows: sharedIntermediate, columns: hiddenSize)
                sharedDown = projection(d, rows: hiddenSize, columns: sharedIntermediate)
                offsets = model.routedExpertOffsets(layer: layer)
            }
            return LayerTensors(
                inputNorm: inputNorm,
                postAttentionNorm: postNorm,
                q: q, k: k, v: v, o: o,
                qNorm: qNorm, kNorm: kNorm, sink: sink,
                router: router,
                sharedGate: sharedGate, sharedUp: sharedUp, sharedDown: sharedDown,
                denseGate: denseGate, denseUp: denseUp, denseDown: denseDown,
                expertOffsets: offsets,
                sliding: model.config.fullAttentionLayerMask[layer] == 0,
                noRope: model.config.noRopeLayerMask[layer] != 0)
        }
    }

    private static func projection(_ view: TensorView,
                                   rows: Int,
                                   columns: Int) -> SharedExpertProjection {
        SharedExpertProjection(
            weights: view.buffer,
            scales: view.buffer,
            biases: view.buffer,
            weightsOffset: Int(view.offset),
            scalesOffset: Int(view.scaleOffset),
            biasesOffset: Int(view.biasOffset),
            rows: UInt32(rows),
            cols: UInt32(columns))
    }

    private static func validateInt8(_ view: TensorView,
                                     name: String,
                                     rows: Int,
                                     columns: Int) throws {
        let weights = UInt64(rows * columns)
        let groups = UInt64(columns / Quantization.groupSize)
        let companions = UInt64(rows) * groups * 2
        guard view.dtype == 0, shape2D(view, rows: rows, columns: columns),
              view.length == weights,
              view.scaleLength == companions,
              view.biasLength == companions else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint \(name) INT8 metadata is invalid")
        }
        try validateRanges(view, name: name, requireCompanions: true)
    }

    private static func validateNorm(_ view: TensorView,
                                     name: String,
                                     rows: Int = hiddenSize) throws {
        guard view.dtype == 1, shape1D(view, elements: rows),
              view.length == UInt64(rows * 2),
              view.scaleLength == 0, view.biasLength == 0 else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint \(name) metadata is invalid")
        }
        try validateRanges(view, name: name, requireCompanions: false)
    }

    private static func validateBF16(_ view: TensorView,
                                     name: String,
                                     rows: Int,
                                     columns: Int) throws {
        guard view.dtype == 1, shape2D(view, rows: rows, columns: columns),
              view.length == UInt64(rows * columns * 2),
              view.scaleLength == 0, view.biasLength == 0 else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint \(name) BF16 metadata is invalid")
        }
        try validateRanges(view, name: name, requireCompanions: false)
    }

    private static func shape1D(_ view: TensorView, elements: Int) -> Bool {
        view.shape.0 == UInt32(elements) && view.shape.1 == 0
            && view.shape.2 == 0 && view.shape.3 == 0
    }

    private static func shape2D(_ view: TensorView,
                                rows: Int,
                                columns: Int) -> Bool {
        view.shape.0 == UInt32(rows) && view.shape.1 == UInt32(columns)
            && view.shape.2 == 0 && view.shape.3 == 0
    }

    private static func validateRanges(_ view: TensorView,
                                       name: String,
                                       requireCompanions: Bool) throws {
        let length = UInt64(view.buffer.length)
        func contains(_ offset: UInt64, _ count: UInt64) -> Bool {
            offset <= length && count <= length - offset
        }
        let ranges = requireCompanions
            ? [(view.offset, view.length), (view.scaleOffset, view.scaleLength),
               (view.biasOffset, view.biasLength)]
            : [(view.offset, view.length)]
        guard ranges.allSatisfy({ contains($0.0, $0.1) }),
              ranges.allSatisfy({ $0.0.isMultiple(of: 2) }) else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "Flint \(name) ranges are invalid")
        }
        for left in ranges.indices {
            for right in ranges.indices where right > left {
                let a = ranges[left], b = ranges[right]
                guard a.0 + a.1 <= b.0 || b.0 + b.1 <= a.0 else {
                    throw FlintForwardRunnerError.invalidConfiguration(
                        "Flint \(name) ranges overlap")
                }
            }
        }
    }

    private static func halfBuffer(context: MetalContext, count: Int) throws -> MTLBuffer {
        guard let buffer = context.device.makeBuffer(
            length: count * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "unable to allocate Flint scratch")
        }
        return buffer
    }

    private static func indexBuffer(context: MetalContext, count: Int) throws -> MTLBuffer {
        guard let buffer = context.device.makeBuffer(
            length: count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared) else {
            throw FlintForwardRunnerError.invalidConfiguration(
                "unable to allocate Flint router indices")
        }
        return buffer
    }
}
