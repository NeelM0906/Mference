import Metal

public enum MapleForwardRunnerError: Error, CustomStringConvertible {
    case invalidConfiguration(String)
    case invalidInput(String)
    case commandFailed(String)

    public var description: String {
        switch self {
        case .invalidConfiguration(let message), .invalidInput(let message), .commandFailed(let message):
            return message
        }
    }
}

/// Exact sequential Maple decode. One instance owns mutable BF16/KV/expert scratch and is serial.
public final class MapleForwardRunner: ContinuableLogitProducer, ContextWindowReporting,
    HeadlessSequentialPrefillRunner, @unchecked Sendable {
    private static let hiddenSize = 2_048
    private static let vocabularySize = 151_936
    private static let layerCount = 24
    private static let topK = 8
    private static let epsilon: Float = 1e-6

    private struct LayerTensors {
        let inputNorm: TensorView
        let postAttentionNorm: TensorView
        let q: TensorView
        let k: TensorView
        let v: TensorView
        let o: TensorView
        let qNorm: TensorView
        let kNorm: TensorView
        let router: TensorView
        let expertOffsets: MoEExpertOffsets
        let sliding: Bool
    }

    private let model: Model
    private let context: MetalContext
    private let kv: KVCacheManager
    private let addRMSNorm: MapleAddRMSNorm
    private let ternary: MapleTernaryGEMV
    private let qkNormRoPE: MapleQKNormRoPE
    private let attention: MapleDecodeAttention
    private let moe: MapleMoE
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
    private let moeDelta: MTLBuffer
    private let finalNormed: MTLBuffer
    private let zero: MTLBuffer
    private let routerIndices: MTLBuffer
    private let routerWeights: MTLBuffer
    private let moeActs: MTLBuffer

    public let maxContext: Int

    public init(model: Model, context: MetalContext, maxContext: Int) throws {
        try Self.validate(config: model.config, maxContext: maxContext)
        guard model.routedExpertCacheSlotCount(layer: 0) ?? 0 >= Self.topK else {
            throw MapleForwardRunnerError.invalidConfiguration(
                "Maple requires at least eight routed-expert cache slots")
        }
        self.model = model
        self.context = context
        self.maxContext = maxContext
        self.kv = try KVCacheManager(device: context.device,
                                     config: model.config,
                                     maxContext: maxContext,
                                     fp16RingEnabled: true,
                                     maxPrefillChunkTokens: 1,
                                     fp16RingCapacityOverride: MapleDecodeAttention.slidingCapacity)
        self.addRMSNorm = try MapleAddRMSNorm(context: context)
        self.ternary = try MapleTernaryGEMV(context: context)
        self.qkNormRoPE = try MapleQKNormRoPE(context: context)
        self.attention = try MapleDecodeAttention(context: context)
        self.moe = try MapleMoE(context: context)
        let embedding = try model.resident(name: "model.word_embeddings.weight")
        let finalNorm = try model.resident(name: "model.norm.weight")
        let lmHead = try model.resident(name: "lm_head.weight")
        try Self.validateAffineInt4(embedding, name: "embedding", rows: Self.vocabularySize)
        try Self.validateNorm(finalNorm, name: "final norm")
        try Self.validateAffineInt4(lmHead, name: "lm head", rows: Self.vocabularySize)
        self.embedding = embedding
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.layers = try Self.validateExpertLayout(model)

        func buffer(_ elements: Int, _ stride: Int) throws -> MTLBuffer {
            guard let buffer = context.device.makeBuffer(length: elements * stride,
                                                         options: .storageModeShared) else {
                throw MapleForwardRunnerError.invalidConfiguration("unable to allocate Maple runtime scratch")
            }
            return buffer
        }
        self.hidden = try buffer(Self.hiddenSize, MemoryLayout<UInt16>.stride)
        self.normed = try buffer(Self.hiddenSize, MemoryLayout<UInt16>.stride)
        self.q = try buffer(Self.hiddenSize, MemoryLayout<UInt16>.stride)
        self.attentionOutput = try buffer(Self.hiddenSize, MemoryLayout<UInt16>.stride)
        self.attentionDelta = try buffer(Self.hiddenSize, MemoryLayout<UInt16>.stride)
        self.routedInput = try buffer(Self.hiddenSize, MemoryLayout<UInt16>.stride)
        self.moeDelta = try buffer(Self.hiddenSize, MemoryLayout<UInt16>.stride)
        self.finalNormed = try buffer(Self.hiddenSize, MemoryLayout<UInt16>.stride)
        self.zero = try buffer(Self.hiddenSize, MemoryLayout<UInt16>.stride)
        self.routerIndices = try buffer(Self.topK, MemoryLayout<UInt32>.stride)
        self.routerWeights = try buffer(Self.topK, MemoryLayout<Float>.stride)
        self.moeActs = try buffer(Self.topK * MapleMoE.intermediate, MemoryLayout<UInt16>.stride)
        memset(zero.contents(), 0, zero.length)
    }

    public func reset() {
        kv.reset()
    }

    public var continuationPosition: Int { kv.position }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition > 0, expectedPosition == kv.position else {
            throw PrefillError.prefillCursorMismatch(
                "Maple continuation cursor \(expectedPosition) does not match \(kv.position)")
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await produce(token: token, position: position, logits: logits)
    }

    func produceWithoutLogits(token: Int32, position: Int) async throws {
        try await produce(token: token, position: position, logits: nil)
    }

    private func produce(token: Int32, position: Int, logits: MTLBuffer?) async throws {
        try Task.checkCancellation()
        guard position == kv.position, position >= 0, position < maxContext else {
            throw MapleForwardRunnerError.invalidInput("Maple position does not match its KV cursor")
        }
        guard token >= 0, token < Int32(Self.vocabularySize) else {
            throw MapleForwardRunnerError.invalidInput("Maple token is outside the vocabulary")
        }
        if let logits, logits.length < Self.vocabularySize * MemoryLayout<Float16>.stride {
            throw MapleForwardRunnerError.invalidInput("Maple logits buffer is too small")
        }

        let embeddingCB = try commandBuffer()
        ternary.encodeEmbedding(commandBuffer: embeddingCB,
                                table: embedding.buffer, tableOffset: Int(embedding.offset),
                                scales: embedding.buffer, scalesOffset: Int(embedding.scaleOffset),
                                biases: embedding.buffer, biasesOffset: Int(embedding.biasOffset),
                                out: hidden, tokenID: UInt32(token))
        try finish(embeddingCB)

        for (index, layer) in layers.enumerated() {
            try Task.checkCancellation()
            let beforeRouter = try commandBuffer()
            addRMSNorm.encode(commandBuffer: beforeRouter,
                              hidden: hidden,
                              delta: index == 0 ? zero : moeDelta,
                              weight: layer.inputNorm.buffer,
                              weightOffset: Int(layer.inputNorm.offset),
                              normed: normed,
                              eps: Self.epsilon)
            encodeProjection(beforeRouter, layer.q, x: normed, y: q, rows: UInt32(Self.hiddenSize))
            let kSlot = kv.kSlot(layer: index, position: position)
            let vSlot = kv.vSlot(layer: index, position: position)
            encodeProjection(beforeRouter, layer.k, x: normed, y: kSlot.buffer,
                             yOffset: kSlot.offset, rows: UInt32(MapleQKNormRoPE.numKVHeads * MapleQKNormRoPE.headDim))
            encodeProjection(beforeRouter, layer.v, x: normed, y: vSlot.buffer,
                             yOffset: vSlot.offset, rows: UInt32(MapleQKNormRoPE.numKVHeads * MapleQKNormRoPE.headDim))
            qkNormRoPE.encode(commandBuffer: beforeRouter,
                              q: q,
                              k: kSlot.buffer, kOffset: kSlot.offset,
                              qWeight: layer.qNorm.buffer, qWeightOffset: Int(layer.qNorm.offset),
                              kWeight: layer.kNorm.buffer, kWeightOffset: Int(layer.kNorm.offset),
                              position: UInt32(position),
                              sliding: layer.sliding)
            if layer.sliding {
                attention.encodeSliding(commandBuffer: beforeRouter,
                                        q: q,
                                        k: kSlot.buffer,
                                        v: vSlot.buffer,
                                        out: attentionOutput,
                                        physicalCount: UInt32(min(position + 1, MapleDecodeAttention.slidingCapacity)))
            } else {
                attention.encodeFull(commandBuffer: beforeRouter,
                                     q: q,
                                     k: kSlot.buffer,
                                     v: vSlot.buffer,
                                     out: attentionOutput,
                                     sequenceLength: UInt32(position + 1))
            }
            encodeProjection(beforeRouter, layer.o, x: attentionOutput, y: attentionDelta,
                             rows: UInt32(Self.hiddenSize))
            addRMSNorm.encode(commandBuffer: beforeRouter,
                              hidden: hidden,
                              delta: attentionDelta,
                              weight: layer.postAttentionNorm.buffer,
                              weightOffset: Int(layer.postAttentionNorm.offset),
                              normed: routedInput,
                              eps: Self.epsilon)
            moe.encodeRouterTop8(commandBuffer: beforeRouter,
                                 weights: layer.router.buffer, weightsOffset: Int(layer.router.offset),
                                 hidden: routedInput,
                                 indices: routerIndices,
                                 routingWeights: routerWeights)
            try finish(beforeRouter)
            try Task.checkCancellation()

            let ranks = try routedRanks()
            guard let plan = try model.planRoutedExperts(layer: index, experts: ranks) else {
                throw ModelError.routedExpertPlanUnavailable(layer: index)
            }
            try await encodeExperts(plan: plan, offsets: layer.expertOffsets)
            try Task.checkCancellation()
        }

        if let logits {
            let head = try commandBuffer()
            addRMSNorm.encode(commandBuffer: head,
                              hidden: hidden,
                              delta: moeDelta,
                              weight: finalNorm.buffer,
                              weightOffset: Int(finalNorm.offset),
                              normed: finalNormed,
                              eps: Self.epsilon)
            ternary.encodeInt4(commandBuffer: head,
                               weights: lmHead.buffer, weightsOffset: Int(lmHead.offset),
                               scales: lmHead.buffer, scalesOffset: Int(lmHead.scaleOffset),
                               biases: lmHead.buffer, biasesOffset: Int(lmHead.biasOffset),
                               x: finalNormed,
                               y: logits,
                               rows: UInt32(Self.vocabularySize))
            try finish(head)
        }
        kv.advance()
    }

    private func encodeExperts(plan: RoutedExpertFetchPlan, offsets: MoEExpertOffsets) async throws {
        let plannedViews = try model.routedExpertBuffers(for: plan)
        let blobs = try expertBlobs(plannedViews)

        if !plan.misses.isEmpty {
            let loadedViews = try await model.fetchRoutedExperts(plan: plan)
            try requireSameExpertBuffers(loadedViews, blobs)
        }

        let argument = moe.makeRoutedArgumentBuffer(routedBlobs: blobs, offsets: offsets)
        let commandBuffer = try commandBuffer()
        moe.encodePhase1(commandBuffer: commandBuffer,
                         routedArgumentBuffer: argument, routedBlobs: blobs, offsets: offsets,
                         x: routedInput, acts: moeActs)
        moe.encodePhase2(commandBuffer: commandBuffer,
                         routedArgumentBuffer: argument, routedBlobs: blobs, offsets: offsets,
                         acts: moeActs, routingWeights: routerWeights, output: moeDelta)
        try finish(commandBuffer)
    }

    private func routedRanks() throws -> [Int] {
        let values = routerIndices.contents().bindMemory(to: UInt32.self, capacity: Self.topK)
        let ranks = (0..<Self.topK).map { Int(values[$0]) }
        guard Set(ranks).count == Self.topK, ranks.allSatisfy({ $0 >= 0 && $0 < MapleMoE.expertCount }) else {
            throw MapleForwardRunnerError.invalidInput("Maple router returned invalid top-8 ranks")
        }
        return ranks
    }

    private func expertBlobs(_ views: [TensorView]) throws -> [MTLBuffer] {
        guard views.count == Self.topK,
              views.allSatisfy({ $0.offset == 0 && $0.length == UInt64($0.buffer.length) }) else {
            throw MapleForwardRunnerError.invalidConfiguration("Maple routed experts must occupy whole cache buffers")
        }
        return views.map(\.buffer)
    }

    private func requireSameExpertBuffers(_ views: [TensorView], _ blobs: [MTLBuffer]) throws {
        let loaded = try expertBlobs(views)
        guard loaded.count == blobs.count, zip(loaded, blobs).allSatisfy({ $0 === $1 }) else {
            throw MapleForwardRunnerError.invalidConfiguration(
                "Maple expert fetch changed the planned cache-slot binding")
        }
    }

    private func encodeProjection(_ commandBuffer: MTLCommandBuffer,
                                  _ view: TensorView,
                                  x: MTLBuffer,
                                  y: MTLBuffer,
                                  yOffset: Int = 0,
                                  rows: UInt32) {
        ternary.encode(commandBuffer: commandBuffer,
                       weights: view.buffer, weightsOffset: Int(view.offset),
                       scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                       biases: view.buffer, biasesOffset: Int(view.biasOffset),
                       x: x, y: y, yOffset: yOffset, rows: rows)
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MapleForwardRunnerError.commandFailed("unable to create Maple command buffer")
        }
        return commandBuffer
    }

    private func finish(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        try requireCompletion(commandBuffer)
    }

    private func requireCompletion(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw MapleForwardRunnerError.commandFailed(
                commandBuffer.error?.localizedDescription ?? "Maple command buffer did not complete")
        }
    }

    private static func validate(config: ArchConfig, maxContext: Int) throws {
        guard config == .maplePreview else {
            throw MapleForwardRunnerError.invalidConfiguration("model is not the pinned Maple geometry")
        }
        guard maxContext > 0 && maxContext <= Int(MapleDecodeAttention.maximumSequenceLength) else {
            throw MapleForwardRunnerError.invalidConfiguration("Maple runtime context must be 1...128,000")
        }
    }

    private static func validateExpertLayout(_ model: Model) throws -> [LayerTensors] {
        let layout = model.packedExpertsLayout
        let rawBytes = UInt64(MapleMoE.intermediate * hiddenSize / 4)
        let companionBytes = UInt64(MapleMoE.intermediate * (hiddenSize / 64)
            * MemoryLayout<UInt16>.stride)
        let expectedStride = 3 * (rawBytes + 2 * companionBytes)
        let expectedSlices: [(String, UInt64, UInt64)] = [
            ("gate", 0, rawBytes),
            ("gate_scales", rawBytes, companionBytes),
            ("gate_biases", rawBytes + companionBytes, companionBytes),
            ("up", rawBytes + 2 * companionBytes, rawBytes),
            ("up_scales", 2 * rawBytes + 2 * companionBytes, companionBytes),
            ("up_biases", 2 * rawBytes + 3 * companionBytes, companionBytes),
            ("down", 2 * (rawBytes + 2 * companionBytes), rawBytes),
            ("down_scales", 3 * rawBytes + 4 * companionBytes, companionBytes),
            ("down_biases", 3 * rawBytes + 5 * companionBytes, companionBytes),
        ]
        let expectedNames = Set(expectedSlices.map(\.0))
        guard layout.numLayers == layerCount, layout.layers.count == layerCount,
              layout.expertsPerLayer == MapleMoE.expertCount,
              layout.expertStride == expectedStride else {
            throw MapleForwardRunnerError.invalidConfiguration("Maple routed-expert layout geometry is invalid")
        }
        for layer in 0..<layerCount {
            let entry = layout.layers[layer]
            let expectedFile = "layer_\(layer < 10 ? "0" : "")\(layer).bin"
            guard entry.layer == layer, entry.file == expectedFile,
                  entry.experts.count == MapleMoE.expertCount else {
                throw MapleForwardRunnerError.invalidConfiguration("Maple routed-expert layer \(layer) is invalid")
            }
            for expert in 0..<MapleMoE.expertCount {
                let blob = entry.experts[expert]
                guard blob.expert == expert, blob.size == expectedStride,
                      blob.offset == UInt64(expert) * expectedStride,
                      Set(blob.subTensors.keys) == expectedNames else {
                    throw MapleForwardRunnerError.invalidConfiguration(
                        "Maple routed-expert blob \(layer)/\(expert) is invalid")
                }
                for (name, offset, size) in expectedSlices {
                    guard let tensor = blob.subTensors[name], tensor.offset == offset,
                          tensor.size == size, tensor.offset <= expectedStride,
                          tensor.size <= expectedStride - tensor.offset,
                          (name.hasSuffix("scales") || name.hasSuffix("biases"))
                              ? tensor.offset.isMultiple(of: UInt64(MemoryLayout<UInt16>.stride))
                              : true else {
                        throw MapleForwardRunnerError.invalidConfiguration(
                            "Maple routed-expert tensor \(name) is invalid")
                    }
                }
                guard expectedSlices.last!.1 + expectedSlices.last!.2 == expectedStride else {
                    throw MapleForwardRunnerError.invalidConfiguration(
                        "Maple routed-expert stride is invalid")
                }
            }
        }
        return try (0..<layerCount).map { layer in
            let inputNorm = try model.resident(name: "model.layers.\(layer).input_layernorm.weight")
            let postAttentionNorm = try model.resident(
                name: "model.layers.\(layer).post_attention_layernorm.weight")
            let q = try model.resident(name: "model.layers.\(layer).self_attn.q_proj.weight")
            let k = try model.resident(name: "model.layers.\(layer).self_attn.k_proj.weight")
            let v = try model.resident(name: "model.layers.\(layer).self_attn.v_proj.weight")
            let o = try model.resident(name: "model.layers.\(layer).self_attn.o_proj.weight")
            let qNorm = try model.resident(name: "model.layers.\(layer).self_attn.q_norm.weight")
            let kNorm = try model.resident(name: "model.layers.\(layer).self_attn.k_norm.weight")
            let router = try model.resident(name: "model.layers.\(layer).mlp.gate.weight")
            try validateNorm(inputNorm, name: "input norm")
            try validateNorm(postAttentionNorm, name: "post-attention norm")
            try validateNorm(qNorm, name: "Q norm", rows: MapleQKNormRoPE.headDim)
            try validateNorm(kNorm, name: "K norm", rows: MapleQKNormRoPE.headDim)
            try validateAffineInt4(q, name: "Q projection", rows: hiddenSize)
            try validateAffineInt4(k, name: "K projection", rows: MapleQKNormRoPE.numKVHeads * MapleQKNormRoPE.headDim)
            try validateAffineInt4(v, name: "V projection", rows: MapleQKNormRoPE.numKVHeads * MapleQKNormRoPE.headDim)
            try validateAffineInt4(o, name: "O projection", rows: hiddenSize)
            try validateBF16(router, name: "router", rows: MapleMoE.expertCount, columns: hiddenSize)
            let offsets = Dictionary(uniqueKeysWithValues: expectedSlices.map { ($0.0, $0.1) })
            guard let gate = UInt32(exactly: offsets["gate"]!),
                  let gateScales = UInt32(exactly: offsets["gate_scales"]!),
                  let gateBiases = UInt32(exactly: offsets["gate_biases"]!),
                  let up = UInt32(exactly: offsets["up"]!),
                  let upScales = UInt32(exactly: offsets["up_scales"]!),
                  let upBiases = UInt32(exactly: offsets["up_biases"]!),
                  let down = UInt32(exactly: offsets["down"]!),
                  let downScales = UInt32(exactly: offsets["down_scales"]!),
                  let downBiases = UInt32(exactly: offsets["down_biases"]!) else {
                throw MapleForwardRunnerError.invalidConfiguration("Maple routed-expert offset is invalid")
            }
            return LayerTensors(
                inputNorm: inputNorm, postAttentionNorm: postAttentionNorm,
                q: q, k: k, v: v, o: o, qNorm: qNorm, kNorm: kNorm, router: router,
                expertOffsets: MoEExpertOffsets(
                    gateWOff: gate, gateSOff: gateScales, gateBOff: gateBiases,
                    upWOff: up, upSOff: upScales, upBOff: upBiases,
                    downWOff: down, downSOff: downScales, downBOff: downBiases),
                sliding: layer % 4 != 3)
        }
    }

    private static func validateAffineInt4(_ view: TensorView, name: String, rows: Int) throws {
        let weights = UInt64(rows * hiddenSize / 2)
        let companions = UInt64(rows * (hiddenSize / 64) * MemoryLayout<UInt16>.stride)
        guard view.dtype == 0, shape2D(view, rows: rows, columns: hiddenSize),
              view.length == weights, view.scaleLength == companions, view.biasLength == companions else {
            throw MapleForwardRunnerError.invalidConfiguration("Maple \(name) metadata is invalid")
        }
        try validateRanges(view, name: name, requireCompanions: true)
    }

    private static func validateNorm(_ view: TensorView, name: String, rows: Int = hiddenSize) throws {
        guard view.dtype == 1, shape1D(view, elements: rows),
              view.length == UInt64(rows * MemoryLayout<UInt16>.stride),
              view.scaleLength == 0, view.biasLength == 0 else {
            throw MapleForwardRunnerError.invalidConfiguration("Maple \(name) metadata is invalid")
        }
        try validateRanges(view, name: name, requireCompanions: false)
    }

    private static func validateBF16(_ view: TensorView, name: String, rows: Int, columns: Int) throws {
        guard view.dtype == 1, shape2D(view, rows: rows, columns: columns),
              view.length == UInt64(rows * columns * MemoryLayout<UInt16>.stride),
              view.scaleLength == 0, view.biasLength == 0 else {
            throw MapleForwardRunnerError.invalidConfiguration("Maple \(name) metadata is invalid")
        }
        try validateRanges(view, name: name, requireCompanions: false)
    }

    private static func shape1D(_ view: TensorView, elements: Int) -> Bool {
        view.shape.0 == UInt32(elements) && view.shape.1 == 0 &&
        view.shape.2 == 0 && view.shape.3 == 0
    }

    private static func shape2D(_ view: TensorView, rows: Int, columns: Int) -> Bool {
        view.shape.0 == UInt32(rows) && view.shape.1 == UInt32(columns) &&
        view.shape.2 == 0 && view.shape.3 == 0
    }

    private static func validateRanges(_ view: TensorView, name: String, requireCompanions: Bool) throws {
        let length = UInt64(view.buffer.length)
        func contains(_ offset: UInt64, _ count: UInt64) -> Bool {
            offset <= length && count <= length - offset
        }
        guard contains(view.offset, view.length),
              (!requireCompanions || (contains(view.scaleOffset, view.scaleLength) &&
                                      contains(view.biasOffset, view.biasLength))),
              view.offset.isMultiple(of: UInt64(MemoryLayout<UInt16>.stride)),
              (!requireCompanions || (view.scaleOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.stride)) &&
                                      view.biasOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.stride)))) else {
            throw MapleForwardRunnerError.invalidConfiguration("Maple \(name) ranges are invalid")
        }
        let ranges = requireCompanions
            ? [(view.offset, view.length), (view.scaleOffset, view.scaleLength),
               (view.biasOffset, view.biasLength)]
            : [(view.offset, view.length)]
        for left in ranges.indices {
            for right in ranges.indices where right > left {
                let a = ranges[left], b = ranges[right]
                guard a.0 + a.1 <= b.0 || b.0 + b.1 <= a.0 else {
                    throw MapleForwardRunnerError.invalidConfiguration("Maple \(name) ranges overlap")
                }
            }
        }
    }
}
