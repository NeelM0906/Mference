import Foundation
import Metal

public enum MiniCPM5ForwardRunnerError: Error, CustomStringConvertible {
    case invalidConfiguration(String)
    case invalidInput(String)
    case commandFailed(String)

    public var description: String {
        switch self {
        case .invalidConfiguration(let message), .invalidInput(let message),
             .commandFailed(let message):
            return message
        }
    }
}

/// MiniCPM5 (plain-llama dense) decode pass. One instance owns mutable KV and
/// scratch state and is serial.
///
/// The layer graph is `LlamaForCausalLM`, every layer full attention:
///
///   embed_lookup_int4(token)                       // out_scale 1.0
///   for L in 0..<42:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     attn = full attention(a): q/k/v projections, NeoX RoPE over the
///            whole head (no q/k norm, no output gate), GQA softmax
///            attention at head_dim^-0.5, o_proj
///     h += attn
///     m = rmsnorm_bf16w(h, post_attention_layernorm)
///     h += SwiGLU(m)                               // one dense MLP per layer
///   head = greedy fused lm_head | full-logits GEMV over rmsnorm(h)
///
/// Derived from `Qwen38ForwardRunner` with the gated-DeltaNet branch, the
/// attention output gate, the per-head q/k norms and the MTP speculator
/// removed. The `qkNorm == false` axis is what selects this runner's attention
/// epilogue: the fused QKV epilogues require the norm gains, so the standalone
/// `RoPE` / `PrefillRoPE` kernels run instead. Chunked layer-major prefill and
/// the paged long-context KV (`Qwen38PagedKVRuntime`) are kept unchanged.
public final class MiniCPM5ForwardRunner: ContinuableLogitProducer, ContextWindowReporting,
    ChunkedPrefillRunner, HeadlessSequentialPrefillRunner, ExactPrefillLogitProducer,
    FusedHeadLogitProducer, @unchecked Sendable {

    struct LayerTensors {
        let inputNorm: TensorView
        let postAttnNorm: TensorView
        let q: TensorView
        let k: TensorView
        let v: TensorView
        let o: TensorView
        let mlpGate: SharedExpertProjection
        let mlpUp: SharedExpertProjection
        let mlpDown: SharedExpertProjection
    }

    /// Chunk-sized prefill scratch: one FP16 row per token, device-private.
    private struct PrefillScratch {
        let chunkTokens: Int
        let hidden: MTLBuffer     // [T, D]
        let normed: MTLBuffer     // [T, D] input_layernorm output
        let mlpX: MTLBuffer       // [T, D] post_attention_layernorm output
        let h1: MTLBuffer         // [T, D] attention-branch (o_proj) output
        let mlpOut: MTLBuffer     // [T, D] dense MLP output
        let attnQ: MTLBuffer      // [T, qDim]
        let kStage: MTLBuffer     // [T, kvDim] pre-cache K staging
        let vStage: MTLBuffer     // [T, kvDim] pre-cache V staging
        let attnOut: MTLBuffer    // [T, qDim]
        let mlpGate: MTLBuffer    // [T, F]
        let mlpUp: MTLBuffer      // [T, F]
        let mlpAct: MTLBuffer     // [T, F]

        init(device: MTLDevice, config: ArchConfig, chunkTokens: Int) throws {
            precondition(chunkTokens > 0, "prefill scratch chunk size must be positive")
            func buf(_ elementsPerToken: Int, _ label: String) throws -> MTLBuffer {
                guard let made = device.makeBuffer(
                    length: max(chunkTokens * elementsPerToken, 1) * MemoryLayout<Float16>.stride,
                    options: .storageModePrivate) else {
                    throw MiniCPM5ForwardRunnerError.invalidConfiguration(
                        "unable to allocate MiniCPM5 prefill scratch")
                }
                made.label = "minicpm5.prefill.\(label)"
                return made
            }
            let D = config.hiddenSize
            let F = config.intermediateSize
            let qDim = config.numHeads * config.fullHeadDim
            let kvDim = config.numFullKVHeads * config.fullHeadDim
            self.chunkTokens = chunkTokens
            self.hidden = try buf(D, "hidden")
            self.normed = try buf(D, "normed")
            self.mlpX = try buf(D, "mlpX")
            self.h1 = try buf(D, "h1")
            self.mlpOut = try buf(D, "mlpOut")
            self.attnQ = try buf(qDim, "attnQ")
            self.kStage = try buf(kvDim, "kStage")
            self.vStage = try buf(kvDim, "vStage")
            self.attnOut = try buf(qDim, "attnOut")
            self.mlpGate = try buf(F, "mlpGate")
            self.mlpUp = try buf(F, "mlpUp")
            self.mlpAct = try buf(F, "mlpAct")
        }
    }

    private let model: Model
    private let ctx: MetalContext
    private let cfg: ArchConfig
    private let kv: KVCacheManager

    /// Paged long-context state (kvPagedPolicy == .on).
    private var pagedKV: Qwen38PagedKVRuntime?

    /// Scratch for the blocked (streamed) prefill attention: FP32 running
    /// online-softmax state per (query, q-head) plus two staging buffers the
    /// past KV windows stream through.
    private struct BlockedPrefillScratch {
        static let windowPages = 128            // 8k tokens per stage
        let chunkTokens: Int
        let mState: MTLBuffer
        let dState: MTLBuffer
        let oState: MTLBuffer
        let stages: [MTLBuffer]
        let stagingTable: MTLBuffer
        let tailTable: MTLBuffer

        init(device: MTLDevice, config: ArchConfig, chunkTokens: Int,
             pagesPerLayer: Int, kPageBytes: Int) throws {
            self.chunkTokens = chunkTokens
            let rows = chunkTokens * config.numHeads
            let headDim = config.fullHeadDim
            guard let m = device.makeBuffer(length: rows * 4, options: .storageModeShared),
                  let d = device.makeBuffer(length: rows * 4, options: .storageModeShared),
                  let o = device.makeBuffer(length: rows * headDim * 4,
                                            options: .storageModeShared) else {
                throw KVPageStoreError.allocationFailed("blocked prefill state")
            }
            m.label = "kvpage.flash.m"; d.label = "kvpage.flash.d"; o.label = "kvpage.flash.o"
            self.mState = m; self.dState = d; self.oState = o

            var stages: [MTLBuffer] = []
            for i in 0..<2 {
                guard let s = device.makeBuffer(length: Self.windowPages * 2 * kPageBytes,
                                                options: .storageModeShared) else {
                    throw KVPageStoreError.allocationFailed("blocked prefill staging")
                }
                s.label = "kvpage.flash.stage\(i)"
                stages.append(s)
            }
            self.stages = stages

            var identity = (0..<Self.windowPages).map { UInt32(2 * $0) }
            guard let table = device.makeBuffer(bytes: &identity,
                                                length: identity.count * 4,
                                                options: .storageModeShared),
                  let tail = device.makeBuffer(
                      length: (chunkTokens / KVPageGeometry.tokensPerPage + 2) * 4,
                      options: .storageModeShared) else {
                throw KVPageStoreError.allocationFailed("blocked prefill tables")
            }
            table.label = "kvpage.flash.stagingTable"
            tail.label = "kvpage.flash.tailTable"
            self.stagingTable = table
            self.tailTable = tail
        }
    }

    private var blockedPrefillScratch: BlockedPrefillScratch?

    private func ensureBlockedPrefillScratch(chunkTokens: Int,
                                             paged: Qwen38PagedKVRuntime) throws -> BlockedPrefillScratch {
        if let scratch = blockedPrefillScratch, scratch.chunkTokens >= chunkTokens {
            return scratch
        }
        let scratch = try BlockedPrefillScratch(device: ctx.device,
                                                config: cfg,
                                                chunkTokens: chunkTokens,
                                                pagesPerLayer: paged.store.geometry.pagesPerLayer,
                                                kPageBytes: paged.store.geometry.kPageBytes)
        blockedPrefillScratch = scratch
        return scratch
    }

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let elementwise: Elementwise
    private let rope: RoPE
    private let mlp: SharedExpertRuntime
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let layers: [LayerTensors]

    // Chunked-prefill kernels.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPInt4: MPPPrefillInt4QMM
    private let prefillRoPE: PrefillRoPE
    private let prefillAttention: PrefillAttention
    private let prefillMLP: PrefillSharedExpert
    private let prefillMLPActivation: MTLComputePipelineState
    private let mlpWeightBits: Int
    private var prefillScratch: PrefillScratch?
    private var prefillChunkState = PrefillChunkCommitState()

    // Decode scratch, allocated once. FP16.
    private let hidden: MTLBuffer         // [D]
    private let normed: MTLBuffer         // [D] input_layernorm output
    private let mlpX: MTLBuffer           // [D] post_attention_layernorm output
    private let oOut: MTLBuffer           // [D] attention-branch output
    private let mlpOut: MTLBuffer         // [D] dense MLP output
    private let qScratch: MTLBuffer       // [qDim]
    private let attnOut: MTLBuffer        // [qDim]
    private let mlpScratchGate: MTLBuffer // [F]
    private let mlpScratchUp: MTLBuffer   // [F]
    private let mlpScratchAct: MTLBuffer  // [F]
    private let greedyTokenBuf: MTLBuffer // [1] UInt32 fused-head output

    public let maxContext: Int
    private let useFusedGreedyHead: Bool
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }

    /// Decode attribution for `MFERENCE_PHASES=1`. A dense family has no
    /// expert I/O and one command buffer per token, so the only phases are
    /// CPU encode + commit and the GPU's own execution span (from the command
    /// buffer's `gpuStartTime` / `gpuEndTime`); everything else is the wait.
    public struct PhaseStats: Sendable {
        public var decodeSteps: Int = 0
        public var encodeNanos: UInt64 = 0
        public var gpuNanos: UInt64 = 0
        public var waitNanos: UInt64 = 0
    }
    public private(set) var phaseStats = PhaseStats()

    /// Reference-parity capture for the sequential decode path. When set, the
    /// decode step blits, per layer, the attention-branch output (after
    /// `o_proj`), the MLP-branch output and the layer's residual output — the
    /// three tensors the goldens hold — into this buffer, followed by the
    /// embedding row and the final-norm row. Layout: `captureSlot(...)`.
    var parityCapture: MTLBuffer?

    static let captureSlotsPerLayer = 3
    /// FP16 element offset of one captured row.
    static func captureSlot(layer: Int, kind: Int, config: ArchConfig) -> Int {
        (layer * captureSlotsPerLayer + kind) * config.hiddenSize
    }
    static func captureEmbedSlot(config: ArchConfig) -> Int {
        config.numLayers * captureSlotsPerLayer * config.hiddenSize
    }
    static func captureFinalNormSlot(config: ArchConfig) -> Int {
        (config.numLayers * captureSlotsPerLayer + 1) * config.hiddenSize
    }
    static func captureElements(config: ArchConfig) -> Int {
        (config.numLayers * captureSlotsPerLayer + 2) * config.hiddenSize
    }

    private static let epsilon: Float = 1e-6

    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        let cfg = model.config
        try Self.validate(config: cfg, maxContext: maxContext)
        self.model = model
        self.ctx = context
        self.cfg = cfg
        self.maxContext = maxContext
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        let paged = runtimeConfiguration.kvPagedPolicy == .on
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: runtimeConfiguration.fp16RingEnabled,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: runtimeConfiguration.prefillConfig.chunkTokens,
                                     pagedFullAttention: paged)
        if paged {
            self.pagedKV = try Qwen38PagedKVRuntime(context: context,
                                                    config: cfg,
                                                    maxContext: maxContext,
                                                    runtimeConfiguration: runtimeConfiguration)
        }

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms = try RMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context,
                                        additionalShapes: cfg.decodeInt4GEMVShapes)
        self.attention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.rope = try RoPE(context: context)
        // The dense MLP shares the attention quant slot (the manifest's
        // sharedExpert slot is absent for this family).
        let mlpWeightBits = model.manifest.quant?.attention.weightBits ?? 8
        self.mlpWeightBits = mlpWeightBits
        self.mlp = try SharedExpertRuntime(context: context,
                                           weightBits: mlpWeightBits,
                                           siluActivation: cfg.hiddenActivation == "silu",
                                           specializedD: cfg.hiddenSize,
                                           specializedF: cfg.intermediateSize)
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillMPPInt4 = MPPPrefillInt4QMM(context: context)
        self.prefillRoPE = try PrefillRoPE(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillMLP = try PrefillSharedExpert(
            context: context,
            weightBits: mlpWeightBits,
            siluActivation: cfg.hiddenActivation == "silu")
        self.prefillMLPActivation = try context.pipeline(
            cfg.hiddenActivation == "silu" ? "silu_mul_fp16" : "gelu_mul_fp16")

        let device = context.device
        func buf(_ elements: Int, _ stride: Int = MemoryLayout<Float16>.stride) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: max(elements, 1) * stride,
                                               options: .storageModeShared) else {
                throw MiniCPM5ForwardRunnerError.invalidConfiguration(
                    "unable to allocate MiniCPM5 runtime scratch")
            }
            return made
        }
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let qDim = cfg.numHeads * cfg.fullHeadDim
        self.hidden = try buf(D)
        self.normed = try buf(D)
        self.mlpX = try buf(D)
        self.oOut = try buf(D)
        self.mlpOut = try buf(D)
        self.qScratch = try buf(qDim)
        self.attnOut = try buf(qDim)
        self.mlpScratchGate = try buf(F)
        self.mlpScratchUp = try buf(F)
        self.mlpScratchAct = try buf(F)
        self.greedyTokenBuf = try buf(1, MemoryLayout<UInt32>.stride)

        func projection(_ view: TensorView, rows: Int, cols: Int) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                   scales: view.buffer,
                                   biases: view.buffer,
                                   weightsOffset: Int(view.offset),
                                   scalesOffset: Int(view.scaleOffset),
                                   biasesOffset: Int(view.biasOffset),
                                   rows: UInt32(rows),
                                   cols: UInt32(cols))
        }
        self.layers = try (0..<cfg.numLayers).map { L in
            LayerTensors(
                inputNorm: try model.inputNorm(layer: L),
                postAttnNorm: try model.postAttnNorm(layer: L),
                q: try model.qProj(layer: L),
                k: try model.kProj(layer: L),
                v: try model.vProj(layer: L),
                o: try model.oProj(layer: L),
                mlpGate: projection(try model.sharedExpertGate(layer: L), rows: F, cols: D),
                mlpUp: projection(try model.sharedExpertUp(layer: L), rows: F, cols: D),
                mlpDown: projection(try model.sharedExpertDown(layer: L), rows: D, cols: F))
        }
    }

    private static func validate(config: ArchConfig, maxContext: Int) throws {
        guard config.family == .minicpm5 else {
            throw MiniCPM5ForwardRunnerError.invalidConfiguration(
                "MiniCPM5ForwardRunner requires the minicpm5 family")
        }
        guard config.numExperts == 0, !config.attnOutputGate, !config.qkNorm,
              !config.hasLinearAttentionLayers, !config.hasCompressedAttentionLayers,
              config.fullAttentionLayerMask.allSatisfy({ $0 == 1 }),
              config.ropeNeoxSubdim, !config.ffnSandwichNorms, !config.sharedExpertGated,
              !config.embeddingScaledBySqrtHidden, config.finalLogitSoftcap == 0,
              config.intermediateSize > 0,
              config.numHeads % config.numFullKVHeads == 0 else {
            throw MiniCPM5ForwardRunnerError.invalidConfiguration(
                "model does not match the plain-llama MiniCPM5 layer graph")
        }
        guard maxContext > 0 else {
            throw MiniCPM5ForwardRunnerError.invalidConfiguration(
                "MiniCPM5 runtime context must be positive")
        }
    }

    // MARK: - LogitProducer

    public func reset() {
        prefillChunkState.reset()
        kv.reset()
        pagedKV?.resetState()
    }

    public var continuationPosition: Int { kv.position }

    public func prepareForContinuation(expectedPosition: Int) throws {
        try prefillChunkState.requireClean(operation: "prepareForContinuation")
        guard expectedPosition > 0, expectedPosition == kv.position else {
            throw PrefillError.prefillCursorMismatch(
                "MiniCPM5 continuation cursor \(expectedPosition) does not match \(kv.position)")
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await produceToken(token: token, position: position, into: logits,
                               emitHead: true, outputMode: .greedyIfAvailable)
    }

    func produceWithoutLogits(token: Int32, position: Int) async throws {
        try await produceToken(token: token, position: position, into: nil,
                               emitHead: false, outputMode: .logits)
    }

    func produceExactPrefill(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await produceToken(token: token, position: position, into: logits,
                               emitHead: true, outputMode: .logits)
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
                "MiniCPM5 prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0, startPosition == kv.position else {
            throw PrefillError.chunkedUnsupported(
                "MiniCPM5 prefill cursor \(kv.position) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "MiniCPM5 prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        guard tokens.allSatisfy({ $0 >= 0 && $0 < Int32(cfg.vocabSize) }) else {
            throw MiniCPM5ForwardRunnerError.invalidInput(
                "MiniCPM5 prefill token is outside the vocabulary")
        }
        let emitLogitsHead = !(outputMode == .greedyIfAvailable && useFusedGreedyHead)
        if emitLogitsHead {
            guard logits.length >= cfg.vocabSize * MemoryLayout<Float16>.stride else {
                throw MiniCPM5ForwardRunnerError.invalidInput(
                    "MiniCPM5 logits buffer is too small")
            }
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        for (spanIndex, span) in spans.enumerated() {
            try Task.checkCancellation()
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try executePrefillChunk(tokens: tokens[lower..<upper],
                                    startPosition: span.startPosition,
                                    outputMode: outputMode,
                                    logits: logits,
                                    scratch: scratch,
                                    writeFinalHead: spanIndex == spans.count - 1)
            onProgress(span.completedCount)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillScratch {
        let chunkTokens = max(1, min(config.chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        if let scratch = prefillScratch, scratch.chunkTokens == chunkTokens {
            return scratch
        }
        let scratch = try PrefillScratch(device: ctx.device,
                                         config: cfg,
                                         chunkTokens: chunkTokens)
        prefillScratch = scratch
        return scratch
    }

    /// One prefill chunk: embed the chunk, run all layers token-parallel,
    /// and (on the final chunk) emit the last token's head through the decode
    /// head kernels, so prefill-then-decode continues bit-identically to pure
    /// sequential decode.
    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillScratch,
                                     writeFinalHead: Bool) throws {
        let t = tokens.count
        precondition(t > 0 && t <= scratch.chunkTokens,
                     "prefill chunk exceeds its scratch capacity")
        let D = cfg.hiddenSize
        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = ctx.device.makeBuffer(
            bytes: tokenIDs,
            length: tokenIDs.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared) else {
            throw MiniCPM5ForwardRunnerError.commandFailed(
                "unable to allocate MiniCPM5 prefill token buffer")
        }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: t)
        var cb = try commandBuffer()

        let emb = model.embedding
        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer, tableOffset: Int(emb.offset),
                            scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: scratch.hidden,
                            t: UInt32(t), d: UInt32(D),
                            outScale: 1.0)

        for (index, layer) in layers.enumerated() {
            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: layer.inputNorm.buffer,
                                   weightOffset: Int(layer.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t), d: UInt32(D),
                                   eps: Self.epsilon)
            cb = try encodeAttentionPrefill(cb, layer: layer,
                                            layerIndex: index,
                                            scratch: scratch,
                                            tokenCount: t,
                                            startPosition: startPosition)
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                          hidden: scratch.hidden,
                                          delta: scratch.h1,
                                          count: t * D)
            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: layer.postAttnNorm.buffer,
                                   weightOffset: Int(layer.postAttnNorm.offset),
                                   out: scratch.mlpX,
                                   t: UInt32(t), d: UInt32(D),
                                   eps: Self.epsilon)
            try encodeDenseMLPPrefill(cb, layer: layer, scratch: scratch,
                                      tokenCount: t)
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                          hidden: scratch.hidden,
                                          delta: scratch.mlpOut,
                                          count: t * D)
        }

        let emitGreedyHead = outputMode == .greedyIfAvailable && useFusedGreedyHead
        if writeFinalHead {
            let lastRowOffset = (t - 1) * D * MemoryLayout<Float16>.stride
            let fNorm = model.finalNorm
            let lm = model.lmHead
            if emitGreedyHead {
                fusionHead.encodeGreedyDecode(commandBuffer: cb,
                                              hidden: scratch.hidden,
                                              hiddenOffset: lastRowOffset,
                                              normWeight: fNorm.buffer,
                                              normOffset: Int(fNorm.offset),
                                              weights: lm.buffer,
                                              weightsOffset: Int(lm.offset),
                                              scales: lm.buffer,
                                              scalesOffset: Int(lm.scaleOffset),
                                              biases: lm.buffer,
                                              biasesOffset: Int(lm.biasOffset),
                                              outToken: greedyTokenBuf,
                                              d: UInt32(D), vocab: UInt32(cfg.vocabSize),
                                              rmsEps: Self.epsilon)
            } else {
                rms.encodeBF16W(commandBuffer: cb,
                                x: scratch.hidden, xOffset: lastRowOffset,
                                weight: fNorm.buffer,
                                weightOffset: Int(fNorm.offset),
                                out: normed,
                                d: UInt32(D), eps: Self.epsilon)
                int4.encode(commandBuffer: cb,
                            weights: lm.buffer, weightsOffset: Int(lm.offset),
                            scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                            biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                            x: normed, y: logits,
                            m: UInt32(cfg.vocabSize), n: UInt32(D))
            }
        }

        try withExtendedLifetime(tokenBuffer) {
            try finish(cb)
        }
        if writeFinalHead, emitGreedyHead {
            lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
        }
        if let paged = pagedKV {
            paged.store.advance(by: t)
            for i in 0..<paged.lastScores.count { paged.lastScores[i] = [] }
        }
        kv.advance(by: t)
        prefillChunkState.markCommitted()
    }

    /// Full attention over one chunk: batched q/k/v projections, NeoX RoPE
    /// over the whole head on q and k (no norms), KV-cache append, causal
    /// tiled prefill attention, then o_proj into `h1`.
    /// Returns the command buffer later encoders must continue on.
    private func encodeAttentionPrefill(_ cb: MTLCommandBuffer,
                                        layer: LayerTensors,
                                        layerIndex: Int,
                                        scratch: PrefillScratch,
                                        tokenCount t: Int,
                                        startPosition: Int) throws -> MTLCommandBuffer {
        let D = cfg.hiddenSize
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = cfg.numHeads * headDim
        let kvDim = numKV * headDim
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

        encodePrefillInt4Projection(cb,
                                    weights: layer.q,
                                    x: scratch.normed, y: scratch.attnQ,
                                    rows: qDim, columns: D,
                                    tokenCount: t,
                                    xStrideElements: D,
                                    yStrideElements: qDim)
        encodePrefillInt4Projection(cb,
                                    weights: layer.k,
                                    x: scratch.normed, y: scratch.kStage,
                                    rows: kvDim, columns: D,
                                    tokenCount: t,
                                    xStrideElements: D,
                                    yStrideElements: kvDim)
        encodePrefillInt4Projection(cb,
                                    weights: layer.v,
                                    x: scratch.normed, y: scratch.vStage,
                                    rows: kvDim, columns: D,
                                    tokenCount: t,
                                    xStrideElements: D,
                                    yStrideElements: kvDim)
        // qkNorm == false: straight to RoPE, no per-head norm epilogue.
        prefillRoPE.encodeNeoxSubdim(commandBuffer: cb,
                                     data: scratch.attnQ,
                                     startPosition: UInt32(startPosition),
                                     queryCount: UInt32(t),
                                     headDim: UInt32(headDim),
                                     numHeads: UInt32(cfg.numHeads),
                                     rotaryDim: rotaryDim,
                                     tokenStrideElements: UInt32(qDim),
                                     theta: Float(cfg.fullRopeTheta))
        prefillRoPE.encodeNeoxSubdim(commandBuffer: cb,
                                     data: scratch.kStage,
                                     startPosition: UInt32(startPosition),
                                     queryCount: UInt32(t),
                                     headDim: UInt32(headDim),
                                     numHeads: UInt32(numKV),
                                     rotaryDim: rotaryDim,
                                     tokenStrideElements: UInt32(kvDim),
                                     theta: Float(cfg.fullRopeTheta))
        var activeCB = cb
        let bytesPerToken = kvDim * MemoryLayout<Float16>.stride
        let endPages = (startPosition + t + KVPageGeometry.tokensPerPage - 1)
            / KVPageGeometry.tokensPerPage
        if let paged = pagedKV,
           !(paged.store.identityMappingIntact && endPages <= paged.poolPagesPerLayer) {
            try encodePagedChunkKVScatter(cb, paged: paged, layerIndex: layerIndex,
                                          startPosition: startPosition, tokenCount: t,
                                          keySource: scratch.kStage,
                                          valueSource: scratch.vStage,
                                          bytesPerToken: bytesPerToken)
            try encodePagedChunkMinMax(cb, paged: paged, layerIndex: layerIndex,
                                       startPosition: startPosition, tokenCount: t)
            try finish(cb)
            let blocked = try ensureBlockedPrefillScratch(chunkTokens: scratch.chunkTokens,
                                                          paged: paged)
            try runBlockedAttention(paged: paged, blocked: blocked,
                                    layerIndex: layerIndex,
                                    queryCount: t, startPosition: startPosition,
                                    scratch: scratch)
            activeCB = try commandBuffer()
        } else {
            try copyStagedKVToCache(cb, layer: layerIndex,
                                    startPosition: startPosition,
                                    tokenCount: t,
                                    keySource: scratch.kStage,
                                    valueSource: scratch.vStage,
                                    bytesPerToken: bytesPerToken)
            if let paged = pagedKV {
                try encodePagedChunkMinMax(cb, paged: paged, layerIndex: layerIndex,
                                           startPosition: startPosition, tokenCount: t)
            }
            let params = PrefillAttentionParams(
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKV),
                kvValidCount: UInt32(startPosition + t),
                slidingWindow: UInt32(startPosition + t),
                kvTokenStrideElements: UInt32(kvDim),
                qTokenStrideElements: UInt32(qDim),
                oTokenStrideElements: UInt32(qDim),
                scale: Float(cfg.attentionScale))
            prefillAttention.encodeCausal(
                commandBuffer: cb,
                q: scratch.attnQ,
                k: pagedKV?.store.kPoolBuffer(layer: layerIndex)
                    ?? kv.keyBuffer(layer: layerIndex, validTokenCount: startPosition + t),
                v: pagedKV?.store.vPoolBuffer(layer: layerIndex)
                    ?? kv.valueBuffer(layer: layerIndex, validTokenCount: startPosition + t),
                out: scratch.attnOut,
                params: params)
        }
        encodePrefillInt4Projection(activeCB,
                                    weights: layer.o,
                                    x: scratch.attnOut, y: scratch.h1,
                                    rows: D, columns: qDim,
                                    tokenCount: t,
                                    xStrideElements: qDim,
                                    yStrideElements: D)
        return activeCB
    }

    private func encodePagedChunkKVScatter(_ cb: MTLCommandBuffer,
                                           paged: Qwen38PagedKVRuntime,
                                           layerIndex: Int,
                                           startPosition: Int,
                                           tokenCount: Int,
                                           keySource: MTLBuffer,
                                           valueSource: MTLBuffer,
                                           bytesPerToken: Int) throws {
        let pageTokens = KVPageGeometry.tokensPerPage
        let firstPage = startPosition / pageTokens
        let lastPage = (startPosition + tokenCount - 1) / pageTokens
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw MiniCPM5ForwardRunnerError.commandFailed(
                "unable to create MiniCPM5 paged KV scatter encoder")
        }
        for page in firstPage...lastPage {
            let writeStart = max(page * pageTokens, startPosition)
            let writeEnd = min((page + 1) * pageTokens, startPosition + tokenCount)
            let count = writeEnd - writeStart
            guard count > 0 else { continue }
            let kDst = try paged.store.kSlot(layer: layerIndex, position: writeStart)
            let vDst = try paged.store.vSlot(layer: layerIndex, position: writeStart)
            let srcOffset = (writeStart - startPosition) * bytesPerToken
            blit.copy(from: keySource, sourceOffset: srcOffset,
                      to: kDst.buffer, destinationOffset: kDst.offset,
                      size: count * bytesPerToken)
            blit.copy(from: valueSource, sourceOffset: srcOffset,
                      to: vDst.buffer, destinationOffset: vDst.offset,
                      size: count * bytesPerToken)
        }
        blit.endEncoding()
    }

    private func encodePagedChunkMinMax(_ cb: MTLCommandBuffer,
                                        paged: Qwen38PagedKVRuntime,
                                        layerIndex: Int,
                                        startPosition: Int,
                                        tokenCount: Int) throws {
        let pageTokens = KVPageGeometry.tokensPerPage
        let firstSealed = startPosition / pageTokens
        let sealedEnd = (startPosition + tokenCount) / pageTokens
        guard sealedEnd > firstSealed,
              let ordinal = paged.store.fullLayerOrdinal(forLayer: layerIndex) else { return }
        let g = paged.store.geometry
        for page in firstSealed..<sealedEnd {
            guard let slot = paged.store.residentSlot(layer: layerIndex, pageIndex: page) else {
                throw KVPageStoreError.pageNotSealed(layer: layerIndex, pageIndex: page)
            }
            paged.kernels.encodePageMinMax(
                commandBuffer: cb,
                kPool: paged.store.kPoolBuffer(layer: layerIndex),
                slot: UInt32(slot),
                validTokens: UInt32(pageTokens),
                metadata: paged.store.metadataBuffer,
                metadataOffset: g.metadataOffset(layerOrdinal: ordinal, pageIndex: page),
                numKVHeads: UInt32(cfg.numFullKVHeads),
                headDim: UInt32(cfg.fullHeadDim))
        }
    }

    private func runBlockedAttention(paged: Qwen38PagedKVRuntime,
                                     blocked: BlockedPrefillScratch,
                                     layerIndex: Int,
                                     queryCount t: Int,
                                     startPosition: Int,
                                     scratch: PrefillScratch) throws {
        let g = paged.store.geometry
        let pageTokens = KVPageGeometry.tokensPerPage
        let headDim = UInt32(cfg.fullHeadDim)
        let numQ = UInt32(cfg.numHeads)
        let numKV = UInt32(cfg.numFullKVHeads)
        let qStride = UInt32(cfg.numHeads * cfg.fullHeadDim)
        let rows = UInt32(t * cfg.numHeads)
        let scale = Float(cfg.attentionScale)

        let initCB = try commandBuffer()
        paged.kernels.encodeFlashInit(commandBuffer: initCB,
                                      mState: blocked.mState,
                                      dState: blocked.dState,
                                      oState: blocked.oState,
                                      rows: rows, headDim: headDim)
        initCB.commit()

        let pastPages = startPosition / pageTokens
        paged.store.flushSpills()
        var stageCBs: [MTLCommandBuffer?] = [nil, nil]
        var page = 0
        var windowIndex = 0
        while page < pastPages {
            let count = min(BlockedPrefillScratch.windowPages, pastPages - page)
            let stageIndex = windowIndex % 2
            if let previous = stageCBs[stageIndex] {
                previous.waitUntilCompleted()
            }
            try paged.store.readSpilledSpan(layer: layerIndex,
                                            firstPage: page, pageCount: count,
                                            into: blocked.stages[stageIndex])
            let windowCB = try commandBuffer()
            paged.kernels.encodeFlashUpdate(
                commandBuffer: windowCB,
                q: scratch.attnQ,
                kPool: blocked.stages[stageIndex],
                vPool: blocked.stages[stageIndex], vPoolOffset: g.kPageBytes,
                pageTable: blocked.stagingTable,
                mState: blocked.mState, dState: blocked.dState, oState: blocked.oState,
                queryCount: UInt32(t),
                qStartPosition: UInt32(startPosition),
                headDim: headDim, numQHeads: numQ, numKVHeads: numKV,
                windowStartPosition: UInt32(page * pageTokens),
                windowTokens: UInt32(count * pageTokens),
                qStrideElements: qStride,
                scale: scale,
                causal: false)
            windowCB.commit()
            stageCBs[stageIndex] = windowCB
            page += count
            windowIndex += 1
        }

        let endPage = (startPosition + t - 1) / pageTokens
        var tailSlots: [UInt32] = []
        for tailPage in pastPages...endPage {
            guard let slot = paged.store.residentSlot(layer: layerIndex, pageIndex: tailPage) else {
                throw KVPageStoreError.pageNotSealed(layer: layerIndex, pageIndex: tailPage)
            }
            tailSlots.append(UInt32(slot))
        }
        precondition(tailSlots.count * 4 <= blocked.tailTable.length,
                     "tail window exceeds its table")
        tailSlots.withUnsafeBytes { raw in
            blocked.tailTable.contents().copyMemory(from: raw.baseAddress!,
                                                    byteCount: raw.count)
        }

        let tailCB = try commandBuffer()
        paged.kernels.encodeFlashUpdate(
            commandBuffer: tailCB,
            q: scratch.attnQ,
            kPool: paged.store.kPoolBuffer(layer: layerIndex),
            vPool: paged.store.vPoolBuffer(layer: layerIndex),
            pageTable: blocked.tailTable,
            mState: blocked.mState, dState: blocked.dState, oState: blocked.oState,
            queryCount: UInt32(t),
            qStartPosition: UInt32(startPosition),
            headDim: headDim, numQHeads: numQ, numKVHeads: numKV,
            windowStartPosition: UInt32(pastPages * pageTokens),
            windowTokens: UInt32(startPosition + t - pastPages * pageTokens),
            qStrideElements: qStride,
            scale: scale,
            causal: true)
        paged.kernels.encodeFlashFinalize(commandBuffer: tailCB,
                                          mState: blocked.mState,
                                          dState: blocked.dState,
                                          oState: blocked.oState,
                                          out: scratch.attnOut,
                                          queryCount: UInt32(t),
                                          headDim: headDim,
                                          numQHeads: numQ,
                                          oStrideElements: qStride)
        try finish(tailCB)
    }

    private func encodeDenseMLPPrefill(_ cb: MTLCommandBuffer,
                                       layer: LayerTensors,
                                       scratch: PrefillScratch,
                                       tokenCount t: Int) throws {
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        guard mlpWeightBits == 4, t >= 32 else {
            try prefillMLP.encodeBlock(commandBuffer: cb,
                                       x: scratch.mlpX,
                                       y: scratch.mlpOut,
                                       gate: layer.mlpGate,
                                       up: layer.mlpUp,
                                       down: layer.mlpDown,
                                       scratchGate: scratch.mlpGate,
                                       scratchUp: scratch.mlpUp,
                                       scratchAct: scratch.mlpAct,
                                       queryCount: t,
                                       d: D,
                                       intermediate: F,
                                       xStrideElements: D,
                                       yStrideElements: D)
            return
        }
        func qmm(_ proj: SharedExpertProjection, x: MTLBuffer, y: MTLBuffer,
                 n: Int, k: Int) {
            encodeQMMOrMPP(cb,
                           weights: proj.weights, weightsOffset: proj.weightsOffset,
                           scales: proj.scales, scalesOffset: proj.scalesOffset,
                           biases: proj.biases, biasesOffset: proj.biasesOffset,
                           x: x, y: y, t: t, n: n, k: k)
        }
        qmm(layer.mlpGate, x: scratch.mlpX, y: scratch.mlpGate, n: F, k: D)
        qmm(layer.mlpUp, x: scratch.mlpX, y: scratch.mlpUp, n: F, k: D)
        guard let activation = cb.makeComputeCommandEncoder() else {
            throw MiniCPM5ForwardRunnerError.commandFailed(
                "unable to create MiniCPM5 prefill MLP activation encoder")
        }
        activation.setComputePipelineState(prefillMLPActivation)
        activation.setBuffer(scratch.mlpGate, offset: 0, index: 0)
        activation.setBuffer(scratch.mlpUp, offset: 0, index: 1)
        activation.setBuffer(scratch.mlpAct, offset: 0, index: 2)
        var count = UInt32(t * F)
        activation.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(prefillMLPActivation.maxTotalThreadsPerThreadgroup, 256)
        activation.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        activation.endEncoding()
        qmm(layer.mlpDown, x: scratch.mlpAct, y: scratch.mlpOut, n: D, k: F)
    }

    private func encodePrefillInt4Projection(_ cb: MTLCommandBuffer,
                                             weights: TensorView,
                                             x: MTLBuffer,
                                             y: MTLBuffer,
                                             rows: Int,
                                             columns: Int,
                                             tokenCount: Int,
                                             xStrideElements: Int,
                                             yStrideElements: Int) {
        if tokenCount >= 32 {
            encodeQMMOrMPP(cb,
                           weights: weights.buffer, weightsOffset: Int(weights.offset),
                           scales: weights.buffer, scalesOffset: Int(weights.scaleOffset),
                           biases: weights.buffer, biasesOffset: Int(weights.biasOffset),
                           x: x, y: y,
                           t: tokenCount, n: rows, k: columns)
            return
        }
        for row in 0..<tokenCount {
            int4.encode(commandBuffer: cb,
                        weights: weights.buffer, weightsOffset: Int(weights.offset),
                        scales: weights.buffer, scalesOffset: Int(weights.scaleOffset),
                        biases: weights.buffer, biasesOffset: Int(weights.biasOffset),
                        x: x,
                        xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                        y: y,
                        yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                        m: UInt32(rows), n: UInt32(columns))
        }
    }

    private func encodeQMMOrMPP(_ cb: MTLCommandBuffer,
                                weights: MTLBuffer, weightsOffset: Int,
                                scales: MTLBuffer, scalesOffset: Int,
                                biases: MTLBuffer, biasesOffset: Int,
                                x: MTLBuffer, y: MTLBuffer,
                                t: Int, n: Int, k: Int) {
        if prefillMPPInt4.isAvailable {
            let path = prefillMPPInt4.encode(commandBuffer: cb,
                                             weights: weights, weightsOffset: weightsOffset,
                                             scales: scales, scalesOffset: scalesOffset,
                                             biases: biases, biasesOffset: biasesOffset,
                                             x: x, y: y,
                                             m: t, n: n, k: k)
            if path == .affineThreadgroupF16 {
                return
            }
        }
        prefillQMM.encode(commandBuffer: cb,
                          weights: weights, weightsOffset: weightsOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          x: x, y: y,
                          t: t, n: n, k: k)
    }

    private func copyStagedKVToCache(_ cb: MTLCommandBuffer,
                                     layer: Int,
                                     startPosition: Int,
                                     tokenCount: Int,
                                     keySource: MTLBuffer,
                                     valueSource: MTLBuffer,
                                     bytesPerToken: Int) throws {
        func copy(_ source: MTLBuffer,
                  to destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                  sourceTokenOffset: Int,
                  count: Int) throws {
            guard count > 0 else { return }
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw MiniCPM5ForwardRunnerError.commandFailed(
                    "unable to create MiniCPM5 prefill KV blit encoder")
            }
            blit.copy(from: source,
                      sourceOffset: sourceTokenOffset * bytesPerToken,
                      to: destination.buffer,
                      destinationOffset: destination.offset,
                      size: count * bytesPerToken)
            blit.endEncoding()
        }
        if let paged = pagedKV {
            try copy(keySource,
                     to: paged.store.contiguousKRange(layer: layer,
                                                      start: startPosition,
                                                      count: tokenCount),
                     sourceTokenOffset: 0, count: tokenCount)
            try copy(valueSource,
                     to: paged.store.contiguousVRange(layer: layer,
                                                      start: startPosition,
                                                      count: tokenCount),
                     sourceTokenOffset: 0, count: tokenCount)
            return
        }
        let capacity = kv.capacity(layer: layer)
        let physicalStart = startPosition % capacity
        let firstSpan = min(tokenCount, capacity - physicalStart)
        try copy(keySource,
                 to: kv.kRange(layer: layer, start: startPosition, count: firstSpan),
                 sourceTokenOffset: 0, count: firstSpan)
        try copy(valueSource,
                 to: kv.vRange(layer: layer, start: startPosition, count: firstSpan),
                 sourceTokenOffset: 0, count: firstSpan)
        guard firstSpan < tokenCount else { return }
        let secondCount = tokenCount - firstSpan
        let secondStart = startPosition + firstSpan
        try copy(keySource,
                 to: kv.kRange(layer: layer, start: secondStart, count: secondCount),
                 sourceTokenOffset: firstSpan, count: secondCount)
        try copy(valueSource,
                 to: kv.vRange(layer: layer, start: secondStart, count: secondCount),
                 sourceTokenOffset: firstSpan, count: secondCount)
    }

    // MARK: - Decode step

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer?,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try Task.checkCancellation()
        guard position == kv.position, position >= 0, position < maxContext else {
            throw MiniCPM5ForwardRunnerError.invalidInput(
                "MiniCPM5 position \(position) does not match its KV cursor \(kv.position)")
        }
        guard token >= 0, token < Int32(cfg.vocabSize) else {
            throw MiniCPM5ForwardRunnerError.invalidInput(
                "MiniCPM5 token is outside the vocabulary")
        }
        let emitLogitsHead = emitHead
            && !(useFusedGreedyHead && outputMode == .greedyIfAvailable)
        if emitLogitsHead {
            guard let logits,
                  logits.length >= cfg.vocabSize * MemoryLayout<Float16>.stride else {
                throw MiniCPM5ForwardRunnerError.invalidInput(
                    "MiniCPM5 logits buffer is too small")
            }
        }

        let stepStart = DispatchTime.now().uptimeNanoseconds
        if let paged = pagedKV {
            try paged.prepareSelections(position: position)
        }

        let D = UInt32(cfg.hiddenSize)
        let cb = try commandBuffer()

        if let paged = pagedKV {
            try paged.encodePendingMetadata(commandBuffer: cb)
        }

        let emb = model.embedding
        embedInt4.encode(commandBuffer: cb,
                         table: emb.buffer, tableOffset: Int(emb.offset),
                         scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                         biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                         out: hidden,
                         tokenId: UInt32(bitPattern: token),
                         d: D,
                         outScale: 1.0)
        capture(cb, hidden, elementOffset: Self.captureEmbedSlot(config: cfg))

        for (index, layer) in layers.enumerated() {
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: layer.inputNorm.buffer,
                            weightOffset: Int(layer.inputNorm.offset),
                            out: normed,
                            d: D, eps: Self.epsilon)
            try encodeAttentionDecode(cb, layer: layer,
                                      layerIndex: index,
                                      position: position,
                                      seqLen: UInt32(position + 1))
            capture(cb, oOut, elementOffset: Self.captureSlot(layer: index, kind: 0, config: cfg))
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                          hidden: hidden,
                                          delta: oOut,
                                          count: cfg.hiddenSize)
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: layer.postAttnNorm.buffer,
                            weightOffset: Int(layer.postAttnNorm.offset),
                            out: mlpX,
                            d: D, eps: Self.epsilon)
            try mlp.encode(commandBuffer: cb,
                           x: mlpX,
                           gate: layer.mlpGate,
                           up: layer.mlpUp,
                           down: layer.mlpDown,
                           y: mlpOut,
                           scratchGate: mlpScratchGate,
                           scratchUp: mlpScratchUp,
                           scratchAct: mlpScratchAct)
            capture(cb, mlpOut, elementOffset: Self.captureSlot(layer: index, kind: 1, config: cfg))
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                          hidden: hidden,
                                          delta: mlpOut,
                                          count: cfg.hiddenSize)
            capture(cb, hidden, elementOffset: Self.captureSlot(layer: index, kind: 2, config: cfg))
        }

        if emitHead {
            let fNorm = model.finalNorm
            let lm = model.lmHead
            if emitLogitsHead, let logits {
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: fNorm.buffer,
                                weightOffset: Int(fNorm.offset),
                                out: normed,
                                d: D, eps: Self.epsilon)
                capture(cb, normed, elementOffset: Self.captureFinalNormSlot(config: cfg))
                int4.encode(commandBuffer: cb,
                            weights: lm.buffer, weightsOffset: Int(lm.offset),
                            scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                            biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                            x: normed, y: logits,
                            m: UInt32(cfg.vocabSize), n: D)
            } else {
                fusionHead.encodeGreedyDecode(commandBuffer: cb,
                                              hidden: hidden,
                                              normWeight: fNorm.buffer,
                                              normOffset: Int(fNorm.offset),
                                              weights: lm.buffer,
                                              weightsOffset: Int(lm.offset),
                                              scales: lm.buffer,
                                              scalesOffset: Int(lm.scaleOffset),
                                              biases: lm.buffer,
                                              biasesOffset: Int(lm.biasOffset),
                                              outToken: greedyTokenBuf,
                                              d: D, vocab: UInt32(cfg.vocabSize),
                                              rmsEps: Self.epsilon)
            }
        }

        let encodeStart = DispatchTime.now().uptimeNanoseconds
        try finish(cb)
        let finished = DispatchTime.now().uptimeNanoseconds
        let gpuSeconds = max(0, cb.gpuEndTime - cb.gpuStartTime)
        let gpuNanos = UInt64(gpuSeconds * 1e9)
        let wall = finished - stepStart
        phaseStats.decodeSteps += 1
        phaseStats.encodeNanos += encodeStart - stepStart
        phaseStats.gpuNanos += gpuNanos
        phaseStats.waitNanos += wall > (encodeStart - stepStart) + gpuNanos
            ? wall - (encodeStart - stepStart) - gpuNanos : 0
        if emitHead, !emitLogitsHead {
            lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
        }
        if let paged = pagedKV {
            paged.readBackScores(sealedPages: position / KVPageGeometry.tokensPerPage)
            paged.store.advance()
            paged.noteAdvance(from: position, to: position + 1)
        }
        kv.advance()
    }

    /// Full attention, one decode step: fused q/k/v GEMV (K and V straight
    /// into their cache slots), NeoX RoPE over the whole head on q and the
    /// new k row, GQA attention at the configured scale, then o_proj into
    /// `oOut`. No norms, no gate: this is the `qkNorm == false` branch.
    private func encodeAttentionDecode(_ cb: MTLCommandBuffer,
                                       layer: LayerTensors,
                                       layerIndex: Int,
                                       position: Int,
                                       seqLen: UInt32) throws {
        let D = UInt32(cfg.hiddenSize)
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot: (buffer: MTLBuffer, offset: Int)
        let vSlot: (buffer: MTLBuffer, offset: Int)
        if let paged = pagedKV {
            kSlot = try paged.store.kSlot(layer: layerIndex, position: position)
            vSlot = try paged.store.vSlot(layer: layerIndex, position: position)
        } else {
            kSlot = kv.kSlot(layer: layerIndex, position: position)
            vSlot = kv.vSlot(layer: layerIndex, position: position)
        }
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

        fusedQKVGEMV.encode(commandBuffer: cb,
                            qWeights: layer.q.buffer, qWeightsOffset: Int(layer.q.offset),
                            qScales: layer.q.buffer, qScalesOffset: Int(layer.q.scaleOffset),
                            qBiases: layer.q.buffer, qBiasesOffset: Int(layer.q.biasOffset),
                            kWeights: layer.k.buffer, kWeightsOffset: Int(layer.k.offset),
                            kScales: layer.k.buffer, kScalesOffset: Int(layer.k.scaleOffset),
                            kBiases: layer.k.buffer, kBiasesOffset: Int(layer.k.biasOffset),
                            vWeights: layer.v.buffer, vWeightsOffset: Int(layer.v.offset),
                            vScales: layer.v.buffer, vScalesOffset: Int(layer.v.scaleOffset),
                            vBiases: layer.v.buffer, vBiasesOffset: Int(layer.v.biasOffset),
                            x: normed,
                            qOut: qScratch,
                            kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                            vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                            qRows: qDim,
                            kvRows: kvDim,
                            n: D)
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: qScratch,
                              position: UInt32(position),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(cfg.numHeads),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: kSlot.buffer,
                              dataOffset: kSlot.offset,
                              position: UInt32(position),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(numKV),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        if let paged = pagedKV {
            let g = paged.store.geometry
            guard let ordinal = paged.store.fullLayerOrdinal(forLayer: layerIndex) else {
                preconditionFailure("paged decode on a non-full-attention layer")
            }
            let selection = paged.selections[ordinal]
            attention.encodeFullPaged(commandBuffer: cb,
                                      q: qScratch,
                                      kPool: paged.store.kPoolBuffer(layer: layerIndex),
                                      vPool: paged.store.vPoolBuffer(layer: layerIndex),
                                      pageTable: paged.tablesBuf,
                                      pageTableOffset: ordinal * g.pagesPerLayer
                                          * MemoryLayout<UInt32>.stride,
                                      out: attnOut,
                                      headDim: UInt32(headDim),
                                      numQHeads: UInt32(cfg.numHeads),
                                      numKVHeads: UInt32(numKV),
                                      selTokens: UInt32(selection.selTokens),
                                      scale: Float(cfg.attentionScale))
            paged.encodeScores(commandBuffer: cb, ordinal: ordinal,
                               q: qScratch, qOffset: 0,
                               sealedPages: position / KVPageGeometry.tokensPerPage)
        } else {
            attention.encodeFull(commandBuffer: cb,
                                 q: qScratch,
                                 k: kSlot.buffer, kOffset: 0,
                                 v: vSlot.buffer, vOffset: 0,
                                 out: attnOut,
                                 headDim: UInt32(headDim),
                                 numQHeads: UInt32(cfg.numHeads),
                                 numKVHeads: UInt32(numKV),
                                 seqLen: seqLen,
                                 scale: Float(cfg.attentionScale))
        }
        int4.encode(commandBuffer: cb,
                    weights: layer.o.buffer, weightsOffset: Int(layer.o.offset),
                    scales: layer.o.buffer, scalesOffset: Int(layer.o.scaleOffset),
                    biases: layer.o.buffer, biasesOffset: Int(layer.o.biasOffset),
                    x: attnOut, y: oOut, m: D, n: qDim)
    }

    /// Blit one FP16 `[D]` row into the parity capture buffer, when one is set.
    private func capture(_ cb: MTLCommandBuffer, _ source: MTLBuffer, elementOffset: Int) {
        guard let target = parityCapture,
              let blit = cb.makeBlitCommandEncoder() else { return }
        let bytes = cfg.hiddenSize * MemoryLayout<Float16>.stride
        blit.copy(from: source, sourceOffset: 0,
                  to: target, destinationOffset: elementOffset * MemoryLayout<Float16>.stride,
                  size: bytes)
        blit.endEncoding()
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw MiniCPM5ForwardRunnerError.commandFailed(
                "unable to create MiniCPM5 command buffer")
        }
        return cb
    }

    private func finish(_ cb: MTLCommandBuffer) throws {
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw MiniCPM5ForwardRunnerError.commandFailed(
                cb.error?.localizedDescription ?? "MiniCPM5 command buffer did not complete")
        }
    }
}
