import Metal

struct PrefillChunkScratchLayout: Sendable, Equatable {
    let chunkTokens: Int
    let hiddenSize: Int
    let maxQElementsPerToken: Int
    let maxKVElementsPerToken: Int
    let sharedIntermediate: Int
    let routedIntermediate: Int
    let topK: Int
    let routedPairMicrobatchRows: Int
    /// Rows of the widest per-token projection written into `q`: the packed
    /// [query ; gate] q_proj when `attnOutputGate`, and the gated-DeltaNet
    /// `in_proj_qkv` rows on architectures with linear-attention layers.
    let qProjElementsPerToken: Int
    /// Non-zero when the architecture gates attention output (Qwen): sizes
    /// the split q / gate buffers.
    let attnGateElementsPerToken: Int
    /// Gated-DeltaNet dims (all zero when the mask has no linear layers).
    let gdnQKVDim: Int
    let gdnValueDim: Int
    let gdnVHeads: Int
    /// Non-zero when the shared expert output is scalar-gated (Qwen).
    let sharedScalarGateElements: Int
    /// The INT4 shared expert runs as one batched block, so each
    /// intermediate retains one row per token (Gemma).
    let batchedSharedExpert: Bool
    /// Routed tiles may run as grouped GEMM and need an activation row per pair.
    let groupedExperts: Bool

    init(config: ArchConfig,
                chunkTokens: Int,
                routedPairMicrobatchRows: Int = 32,
                batchedSharedExpert: Bool = false,
                groupedExperts: Bool = false) {
        self.chunkTokens = max(1, min(chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        self.hiddenSize = config.hiddenSize
        self.maxQElementsPerToken = config.numHeads * max(config.headDim, config.fullHeadDim)
        self.maxKVElementsPerToken = max(config.numKVHeads * config.headDim,
                                         config.numFullKVHeads * config.fullHeadDim)
        self.sharedIntermediate = config.intermediateSize
        self.routedIntermediate = config.moeIntermediateSize
        self.topK = config.topKExperts
        self.routedPairMicrobatchRows = max(1, min(routedPairMicrobatchRows, 128))
        let qDim = config.numHeads * max(config.headDim, config.fullHeadDim)
        let hasLinear = config.hasLinearAttentionLayers
        self.qProjElementsPerToken = max(config.attnOutputGate ? 2 * qDim : qDim,
                                         hasLinear ? config.linearAttention.qkvDim : 0)
        self.attnGateElementsPerToken = config.attnOutputGate ? qDim : 0
        self.gdnQKVDim = hasLinear ? config.linearAttention.qkvDim : 0
        self.gdnValueDim = hasLinear ? config.linearAttention.valueDim : 0
        self.gdnVHeads = hasLinear ? config.linearAttention.numVHeads : 0
        self.sharedScalarGateElements = config.sharedExpertGated ? 1 : 0
        self.batchedSharedExpert = batchedSharedExpert
        self.groupedExperts = groupedExperts
    }

    init(config: ArchConfig, runtime: PrefillRuntimeConfig,
         batchedSharedExpert: Bool = false, groupedExperts: Bool = false) {
        self.init(config: config,
                  chunkTokens: runtime.chunkTokens,
                  batchedSharedExpert: batchedSharedExpert,
                  groupedExperts: groupedExperts)
    }

    var hiddenElements: Int { chunkTokens * hiddenSize }
    var normedElements: Int { hiddenElements }
    var qElements: Int { chunkTokens * qProjElementsPerToken }
    var attnElements: Int { chunkTokens * maxQElementsPerToken }
    var attnQElements: Int { chunkTokens * attnGateElementsPerToken }
    var attnGateElements: Int { attnQElements }
    var gdnConvOutElements: Int { chunkTokens * gdnQKVDim }
    var gdnZElements: Int { chunkTokens * gdnValueDim }
    var gdnAElements: Int { chunkTokens * gdnVHeads }
    var gdnBElements: Int { gdnAElements }
    var gdnYElements: Int { gdnZElements }
    var sharedScalarGateBufferElements: Int { chunkTokens * sharedScalarGateElements }
    var kStageElements: Int { chunkTokens * maxKVElementsPerToken }
    var vStageElements: Int { kStageElements }
    var attentionOutputElements: Int { attnElements }
    var denseXElements: Int { hiddenElements }
    var routedXElements: Int { hiddenElements }
    var routerXElements: Int { hiddenElements }
    var h1Elements: Int { hiddenElements }
    var h2Elements: Int { hiddenElements }
    var routePartialElements: Int { chunkTokens * topK * hiddenSize }
    var routeIDElements: Int { chunkTokens * topK }
    var routeWeightElements: Int { routeIDElements }
    /// Qwen and Gemma batch-dispatch their INT4 shared expert over the
    /// complete chunk, so each intermediate must retain one row per token.
    /// Other families keep the scalar-row scratch footprint.
    var sharedExpertScratchElements: Int {
        (sharedScalarGateElements > 0 || batchedSharedExpert ? chunkTokens : 1) * sharedIntermediate
    }
    /// Grouped-GEMM routed experts keep one activation row per routed pair of
    /// a tile; a tile can hold every pair of the chunk.
    var groupedExpertActivationElements: Int { chunkTokens * topK * routedIntermediate }
    /// Attention output is consumed by the O projection before any routed
    /// tile of the layer runs and is rewritten only by the next layer, so
    /// grouped experts borrow it when it is large enough (every shipped
    /// Gemma shape) and the arena grows only otherwise.
    var dedicatedGroupedExpertActivationElements: Int {
        groupedExperts && groupedExpertActivationElements > attentionOutputElements
            ? groupedExpertActivationElements : 0
    }
    var routedGateUpActElements: Int { 3 * routedPairMicrobatchRows * routedIntermediate }
    var routedDownOutputElements: Int { routedPairMicrobatchRows * hiddenSize }

    var devicePrivateBytes: Int {
        let fp16Elements = hiddenElements
            + normedElements
            + qElements
            + kStageElements
            + vStageElements
            + attentionOutputElements
            + denseXElements
            + routedXElements
            + routerXElements
            + h1Elements
            + h2Elements
            + routePartialElements
            + 3 * sharedExpertScratchElements
            + dedicatedGroupedExpertActivationElements
            + routedGateUpActElements
            + routedDownOutputElements
            + attnQElements
            + attnGateElements
            + gdnConvOutElements
            + gdnZElements
            + gdnAElements
            + gdnBElements
            + gdnYElements
            + sharedScalarGateBufferElements
        return fp16Elements * MemoryLayout<Float16>.stride
    }

    var sharedMetadataBytes: Int {
        routeIDElements * MemoryLayout<UInt32>.stride
            + routeWeightElements * MemoryLayout<Float16>.stride
    }

    var totalPersistentBytes: Int {
        devicePrivateBytes + sharedMetadataBytes
    }
}

struct PrefillChunkScratchBuffers {
    let layout: PrefillChunkScratchLayout
    let hidden: MTLBuffer
    let normed: MTLBuffer
    let q: MTLBuffer
    let kStage: MTLBuffer
    let vStage: MTLBuffer
    let attentionOutput: MTLBuffer
    let denseX: MTLBuffer
    let routedX: MTLBuffer
    let routerX: MTLBuffer
    let h1: MTLBuffer
    let h2: MTLBuffer
    let routePartials: MTLBuffer
    let routeIDs: MTLBuffer
    let routeWeights: MTLBuffer
    let sharedGateScratch: MTLBuffer
    let sharedUpScratch: MTLBuffer
    let sharedActScratch: MTLBuffer
    let routedGateUpActScratch: MTLBuffer
    let routedDownScratch: MTLBuffer
    // Qwen 3.6 additions. Placeholder-sized (1 element) when the arch does
    // not use them, so the struct stays non-optional.
    let attnQ: MTLBuffer
    let attnGate: MTLBuffer
    let gdnConvOut: MTLBuffer
    let gdnZ: MTLBuffer
    let gdnA: MTLBuffer
    let gdnB: MTLBuffer
    let gdnY: MTLBuffer
    let sharedScalarGate: MTLBuffer

    /// Nil unless the layout admits grouped experts; otherwise the borrowed
    /// attention output or a dedicated buffer (see the layout).
    let groupedExpertActivation: MTLBuffer?

    static func allocate(device: MTLDevice,
                         layout: PrefillChunkScratchLayout) throws -> PrefillChunkScratchBuffers {
        func privateBuffer(_ elements: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * MemoryLayout<Float16>.stride,
                options: .storageModePrivate)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        func sharedBuffer(_ bytes: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(bytes, 1),
                                                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        let attentionOutput = try privateBuffer(layout.attentionOutputElements, label: "prefill.attnOut")
        let groupedExpertActivation: MTLBuffer? = !layout.groupedExperts ? nil
            : layout.dedicatedGroupedExpertActivationElements == 0 ? attentionOutput
            : try privateBuffer(layout.dedicatedGroupedExpertActivationElements,
                                label: "prefill.groupedExpertActivation")
        return PrefillChunkScratchBuffers(
            layout: layout,
            hidden: try privateBuffer(layout.hiddenElements, label: "prefill.hidden"),
            normed: try privateBuffer(layout.normedElements, label: "prefill.normed"),
            q: try privateBuffer(layout.qElements, label: "prefill.q"),
            kStage: try privateBuffer(layout.kStageElements, label: "prefill.kStage"),
            vStage: try privateBuffer(layout.vStageElements, label: "prefill.vStage"),
            attentionOutput: attentionOutput,
            denseX: try privateBuffer(layout.denseXElements, label: "prefill.denseX"),
            routedX: try privateBuffer(layout.routedXElements, label: "prefill.routedX"),
            routerX: try privateBuffer(layout.routerXElements, label: "prefill.routerX"),
            h1: try privateBuffer(layout.h1Elements, label: "prefill.h1"),
            h2: try privateBuffer(layout.h2Elements, label: "prefill.h2"),
            routePartials: try privateBuffer(layout.routePartialElements, label: "prefill.routePartials"),
            routeIDs: try sharedBuffer(layout.routeIDElements * MemoryLayout<UInt32>.stride,
                                       label: "prefill.routeIDs"),
            routeWeights: try sharedBuffer(layout.routeWeightElements * MemoryLayout<Float16>.stride,
                                           label: "prefill.routeWeights"),
            sharedGateScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                 label: "prefill.sharedGateScratch"),
            sharedUpScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                               label: "prefill.sharedUpScratch"),
            sharedActScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                label: "prefill.sharedActScratch"),
            routedGateUpActScratch: try privateBuffer(layout.routedGateUpActElements,
                                                      label: "prefill.routedGateUpActScratch"),
            routedDownScratch: try privateBuffer(layout.routedDownOutputElements,
                                                 label: "prefill.routedDownScratch"),
            attnQ: try privateBuffer(layout.attnQElements, label: "prefill.attnQ"),
            attnGate: try privateBuffer(layout.attnGateElements, label: "prefill.attnGate"),
            gdnConvOut: try privateBuffer(layout.gdnConvOutElements, label: "prefill.gdnConvOut"),
            gdnZ: try privateBuffer(layout.gdnZElements, label: "prefill.gdnZ"),
            gdnA: try privateBuffer(layout.gdnAElements, label: "prefill.gdnA"),
            gdnB: try privateBuffer(layout.gdnBElements, label: "prefill.gdnB"),
            gdnY: try privateBuffer(layout.gdnYElements, label: "prefill.gdnY"),
            sharedScalarGate: try privateBuffer(layout.sharedScalarGateBufferElements,
                                                label: "prefill.sharedScalarGate"),
            groupedExpertActivation: groupedExpertActivation)
    }
}
