import Foundation
import Metal

public enum Qwen38ForwardRunnerError: Error, CustomStringConvertible {
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

/// Qwen 3.8 dense decode pass. One instance owns mutable KV / GDN / scratch
/// state and is serial.
///
/// The architecture is the Qwen 3.6 layer graph with the MoE branch deleted:
///
///   embed_lookup_int4(token)                       // out_scale 1.0
///   for L in 0..<64:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     attn = (mask 2) gated-DeltaNet linear attention(a)      // FP32 state
///          | (mask 1) gated full attention(a)                 // packed
///            [query ; gate] q_proj, per-head q/k norm, NeoX sub-dim RoPE
///            over 64 of 256 dims, sigmoid(gate) before o_proj
///     h += attn
///     m = rmsnorm_bf16w(h, post_attention_layernorm)
///     h += SwiGLU(m)                               // one dense MLP per layer
///   head = greedy fused lm_head | full-logits GEMV over rmsnorm(h)
///
/// Every weight is resident (`numExperts == 0`: the expert streamer never
/// opens), so a decode step encodes into a single command buffer with no
/// CPU readback between layers. Prefill is chunked and layer-major: each
/// chunk of prompt tokens runs token-parallel projections, attention, and
/// MLP kernels (the qwen36 prefill kernel set), with the inherently
/// sequential DeltaNet recurrence handled by its dedicated in-kernel scan
/// (`gdn_delta_step_prefill`), which produces the exact FP32 state of
/// token-by-token decode. There is no router readback, so a whole chunk —
/// all 64 layers — encodes into one command buffer.
public final class Qwen38ForwardRunner: ContinuableLogitProducer, ContextWindowReporting,
    ChunkedPrefillRunner, HeadlessSequentialPrefillRunner, ExactPrefillLogitProducer,
    FusedHeadLogitProducer, @unchecked Sendable {

    /// Every per-layer TensorView, resolved once at init so the decode hot
    /// path never touches the resident-index dictionary. Internal so the MTP
    /// speculator's verify pass can reuse the resolved views.
    struct LayerTensors {
        let inputNorm: TensorView
        let postAttnNorm: TensorView
        // Full-attention layers (mask 1) only.
        let q: TensorView?
        let k: TensorView?
        let v: TensorView?
        let o: TensorView?
        let qNorm: TensorView?
        let kNorm: TensorView?
        // Gated-DeltaNet layers (mask 2) only.
        let linQKV: TensorView?
        let linZ: TensorView?
        let linA: TensorView?
        let linB: TensorView?
        let linOut: TensorView?
        let linConv: TensorView?
        let linALog: TensorView?
        let linDtBias: TensorView?
        let linNorm: TensorView?
        // Dense SwiGLU MLP (every layer; served by the sharedExpert accessors).
        let mlpGate: SharedExpertProjection
        let mlpUp: SharedExpertProjection
        let mlpDown: SharedExpertProjection
        let isLinear: Bool
    }

    /// Chunk-sized prefill scratch: one FP16 row per token, device-private.
    /// Allocated on the first chunked prefill from the resolved chunk size
    /// and reused; reallocated only when that size changes. `attnOut` is
    /// shared by the two attention branches, so its per-token width covers
    /// both the full-attention output (qDim) and the gated-norm output of
    /// the DeltaNet branch (valueDim).
    private struct PrefillScratch {
        let chunkTokens: Int
        let hidden: MTLBuffer     // [T, D]
        let normed: MTLBuffer     // [T, D] input_layernorm output
        let mlpX: MTLBuffer       // [T, D] post_attention_layernorm output
        let h1: MTLBuffer         // [T, D] attention-branch (o_proj) output
        let mlpOut: MTLBuffer     // [T, D] dense MLP output
        let qPacked: MTLBuffer    // [T, 2*qDim] packed [query ; gate]
        let attnQ: MTLBuffer      // [T, qDim]
        let attnGate: MTLBuffer   // [T, qDim]
        let kStage: MTLBuffer     // [T, kvDim] pre-cache K staging
        let vStage: MTLBuffer     // [T, kvDim] pre-cache V staging
        let attnOut: MTLBuffer    // [T, max(qDim, valueDim)]
        let mlpGate: MTLBuffer    // [T, F]
        let mlpUp: MTLBuffer      // [T, F]
        let mlpAct: MTLBuffer     // [T, F]
        let gdnQKV: MTLBuffer     // [T, qkvDim] raw in_proj_qkv rows
        let gdnConvOut: MTLBuffer // [T, qkvDim] conv + SiLU output
        let gdnZ: MTLBuffer       // [T, valueDim]
        let gdnA: MTLBuffer       // [T, numVHeads]
        let gdnB: MTLBuffer       // [T, numVHeads]
        let gdnY: MTLBuffer       // [T, valueDim] delta-rule output

        init(device: MTLDevice, config: ArchConfig, chunkTokens: Int) throws {
            precondition(chunkTokens > 0, "prefill scratch chunk size must be positive")
            func buf(_ elementsPerToken: Int, _ label: String) throws -> MTLBuffer {
                guard let made = device.makeBuffer(
                    length: max(chunkTokens * elementsPerToken, 1) * MemoryLayout<Float16>.stride,
                    options: .storageModePrivate) else {
                    throw Qwen38ForwardRunnerError.invalidConfiguration(
                        "unable to allocate Qwen 3.8 prefill scratch")
                }
                made.label = "qwen38.prefill.\(label)"
                return made
            }
            let D = config.hiddenSize
            let F = config.intermediateSize
            let qDim = config.numHeads * config.fullHeadDim
            let kvDim = config.numFullKVHeads * config.fullHeadDim
            let la = config.linearAttention
            self.chunkTokens = chunkTokens
            self.hidden = try buf(D, "hidden")
            self.normed = try buf(D, "normed")
            self.mlpX = try buf(D, "mlpX")
            self.h1 = try buf(D, "h1")
            self.mlpOut = try buf(D, "mlpOut")
            self.qPacked = try buf(2 * qDim, "qPacked")
            self.attnQ = try buf(qDim, "attnQ")
            self.attnGate = try buf(qDim, "attnGate")
            self.kStage = try buf(kvDim, "kStage")
            self.vStage = try buf(kvDim, "vStage")
            self.attnOut = try buf(max(qDim, la.valueDim), "attnOut")
            self.mlpGate = try buf(F, "mlpGate")
            self.mlpUp = try buf(F, "mlpUp")
            self.mlpAct = try buf(F, "mlpAct")
            self.gdnQKV = try buf(la.qkvDim, "gdnQKV")
            self.gdnConvOut = try buf(la.qkvDim, "gdnConvOut")
            self.gdnZ = try buf(la.valueDim, "gdnZ")
            self.gdnA = try buf(la.numVHeads, "gdnA")
            self.gdnB = try buf(la.numVHeads, "gdnB")
            self.gdnY = try buf(la.valueDim, "gdnY")
        }
    }

    private let model: Model
    private let ctx: MetalContext
    private let cfg: ArchConfig
    private let kv: KVCacheManager
    private let gdnState: GDNStateManager

    /// Paged long-context state (kvPagedPolicy == .on), shared with the
    /// MTP speculator so round and plain tokens drive one cursor.
    private var pagedKV: Qwen38PagedKVRuntime?

    /// Scratch for the blocked (streamed) prefill attention: FP32 running
    /// online-softmax state per (query, q-head) plus two staging buffers the
    /// past KV windows stream through (ring of two so a window's pread can
    /// overlap the previous window's GPU pass).
    private struct BlockedPrefillScratch {
        static let windowPages = 128            // 8k tokens, 32 MiB per stage
        let chunkTokens: Int
        let mState: MTLBuffer
        let dState: MTLBuffer
        let oState: MTLBuffer
        let stages: [MTLBuffer]
        /// Stride-2 identity table addressing the interleaved [K|V] staging
        /// layout (K page i at slot 2i, its V page at +1 K-page offset).
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
    private let gdn: GDN
    private let mlp: SharedExpertRuntime
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let layers: [LayerTensors]

    // Chunked-prefill kernels (token-parallel variants of the decode set).
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPInt4: MPPPrefillInt4QMM
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillMLP: PrefillSharedExpert
    private let prefillMLPActivation: MTLComputePipelineState
    private let mlpWeightBits: Int
    private var prefillScratch: PrefillScratch?
    private var prefillChunkState = PrefillChunkCommitState()

    // Decode scratch, allocated once. FP16 unless noted. At production shape
    // (D 5120, F 17408, qDim 24*256 = 6144, gdn qkvDim 10240, valueDim 6144)
    // the whole set is ~293 KB:
    //   5 x D vectors (hidden/normed/mlpX/oOut/mlpOut)          50 KB
    //   packed q + q + gate + attn out (2*qDim + 3*qDim)        60 KB
    //   3 x F MLP intermediates                                102 KB
    //   gdn qkv + conv (2 x qkvDim) + z/y/out (3 x valueDim)
    //     + a/b (2 x numVHeads)                                 76 KB
    private let hidden: MTLBuffer         // [D]
    private let normed: MTLBuffer         // [D] input_layernorm output
    private let mlpX: MTLBuffer           // [D] post_attention_layernorm output
    private let oOut: MTLBuffer           // [D] attention-branch output
    private let mlpOut: MTLBuffer         // [D] dense MLP output
    private let qPackedScratch: MTLBuffer // [2 * qDim] packed [query ; gate]
    private let qScratch: MTLBuffer       // [qDim]
    private let attnGateScratch: MTLBuffer // [qDim]
    private let attnOut: MTLBuffer        // [qDim]
    private let mlpScratchGate: MTLBuffer // [F]
    private let mlpScratchUp: MTLBuffer   // [F]
    private let mlpScratchAct: MTLBuffer  // [F]
    private let gdnQKVRaw: MTLBuffer      // [qkvDim] raw in_proj_qkv output
    private let gdnConvOut: MTLBuffer     // [qkvDim] conv + SiLU output
    private let gdnZ: MTLBuffer           // [valueDim]
    private let gdnA: MTLBuffer           // [numVHeads]
    private let gdnB: MTLBuffer           // [numVHeads]
    private let gdnY: MTLBuffer           // [valueDim] delta-rule output
    private let gdnOut: MTLBuffer         // [valueDim] gated-norm output
    private let greedyTokenBuf: MTLBuffer // [1] UInt32 fused-head output

    public let maxContext: Int
    private let useFusedGreedyHead: Bool
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }

    /// MTP speculative decoding, present when the install carries the
    /// `mtp.*` draft tensors and `MFERENCE_MTP` is not "0". Greedy-only:
    /// rounds run only through the fused-greedy `produce` path, which the
    /// generation loop already restricts to temperature 0 with no
    /// repetition penalty. Internal var so tests can disable it per-instance.
    var mtp: Qwen38MTPSpeculator?

    public struct MTPSpecStats: Sendable {
        public let rounds: Int
        public let draftedTokens: Int
        public let acceptedTokens: Int
        public let emittedTokens: Int
        public let rollbacks: Int
        public let draftNanos: UInt64
        public let verifyNanos: UInt64
        public let acceptNanos: UInt64
        public let positionTrials: [Int]
        public let positionAccepts: [Int]
    }

    public var mtpSpecStats: MTPSpecStats? {
        mtp.map {
            MTPSpecStats(rounds: $0.stats.rounds,
                         draftedTokens: $0.stats.draftedTokens,
                         acceptedTokens: $0.stats.acceptedTokens,
                         emittedTokens: $0.stats.emittedTokens,
                         rollbacks: $0.stats.rollbacks,
                         draftNanos: $0.stats.draftNanos,
                         verifyNanos: $0.stats.verifyNanos,
                         acceptNanos: $0.stats.acceptNanos,
                         positionTrials: $0.stats.positionTrials,
                         positionAccepts: $0.stats.positionAccepts)
        }
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
        self.gdnState = try GDNStateManager(device: context.device, config: cfg)

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms = try RMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context,
                                        additionalShapes: cfg.decodeInt4GEMVShapes)
        self.attention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.rope = try RoPE(context: context)
        self.gdn = try GDN(context: context, config: cfg.linearAttention,
                           specializedHiddenSize: cfg.hiddenSize)
        // The dense MLP shares the attention quant (the manifest's
        // sharedExpert slot is deliberately absent for this family); the
        // quant-less toy manifest keeps the INT8 default.
        self.mlp = try SharedExpertRuntime(context: context,
                                           weightBits: model.manifest.quant?.attention.weightBits ?? 8,
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
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        let mlpWeightBits = model.manifest.quant?.attention.weightBits ?? 8
        self.mlpWeightBits = mlpWeightBits
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
                throw Qwen38ForwardRunnerError.invalidConfiguration(
                    "unable to allocate Qwen 3.8 runtime scratch")
            }
            return made
        }
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let qDim = cfg.numHeads * cfg.fullHeadDim
        let la = cfg.linearAttention
        self.hidden = try buf(D)
        self.normed = try buf(D)
        self.mlpX = try buf(D)
        self.oOut = try buf(D)
        self.mlpOut = try buf(D)
        self.qPackedScratch = try buf(2 * qDim)
        self.qScratch = try buf(qDim)
        self.attnGateScratch = try buf(qDim)
        self.attnOut = try buf(qDim)
        self.mlpScratchGate = try buf(F)
        self.mlpScratchUp = try buf(F)
        self.mlpScratchAct = try buf(F)
        self.gdnQKVRaw = try buf(la.qkvDim)
        self.gdnConvOut = try buf(la.qkvDim)
        self.gdnZ = try buf(la.valueDim)
        self.gdnA = try buf(la.numVHeads)
        self.gdnB = try buf(la.numVHeads)
        self.gdnY = try buf(la.valueDim)
        self.gdnOut = try buf(la.valueDim)
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
            let isLinear = cfg.layerIsLinear(L)
            return LayerTensors(
                inputNorm: try model.inputNorm(layer: L),
                postAttnNorm: try model.postAttnNorm(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: isLinear ? nil : try model.kProj(layer: L),
                v: isLinear ? nil : try model.vProj(layer: L),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: isLinear ? nil : try model.qNorm(layer: L),
                kNorm: isLinear ? nil : try model.kNorm(layer: L),
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: isLinear ? try model.linearInProjZ(layer: L) : nil,
                linA: isLinear ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear ? try model.linearOutProj(layer: L) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear ? try model.linearALog(layer: L) : nil,
                linDtBias: isLinear ? try model.linearDtBias(layer: L) : nil,
                linNorm: isLinear ? try model.linearNorm(layer: L) : nil,
                mlpGate: projection(try model.sharedExpertGate(layer: L), rows: F, cols: D),
                mlpUp: projection(try model.sharedExpertUp(layer: L), rows: F, cols: D),
                mlpDown: projection(try model.sharedExpertDown(layer: L), rows: D, cols: F),
                isLinear: isLinear)
        }

        if ProcessInfo.processInfo.environment["MFERENCE_MTP"] != "0" {
            self.mtp = try Qwen38MTPSpeculator.probe(model: model,
                                                     context: context,
                                                     config: cfg,
                                                     kv: kv,
                                                     gdnState: gdnState,
                                                     layers: layers,
                                                     maxContext: maxContext,
                                                     mlpWeightBits: mlpWeightBits,
                                                     paged: pagedKV)
            // MFERENCE_DFLASH2_DIR swaps the round's draft source to the
            // DFlash2 block-diffusion drafter; the verify/accept machinery
            // (and therefore emitted bytes) is unchanged.
            if let mtp {
                mtp.dflash2 = try Qwen38DFlash2Drafter.probe(
                    context: context, model: model, targetConfig: cfg)
            }
        }
    }

    private static func validate(config: ArchConfig, maxContext: Int) throws {
        guard config.family == .qwen38 else {
            throw Qwen38ForwardRunnerError.invalidConfiguration(
                "Qwen38ForwardRunner requires the qwen38 family")
        }
        guard config.numExperts == 0, config.attnOutputGate,
              config.hasLinearAttentionLayers, config.ropeNeoxSubdim,
              !config.ffnSandwichNorms, !config.sharedExpertGated,
              config.intermediateSize > 0 else {
            throw Qwen38ForwardRunnerError.invalidConfiguration(
                "model does not match the dense Qwen 3.8 layer graph")
        }
        guard maxContext > 0 else {
            throw Qwen38ForwardRunnerError.invalidConfiguration(
                "Qwen 3.8 runtime context must be positive")
        }
    }

    // MARK: - LogitProducer

    public func reset() {
        prefillChunkState.reset()
        kv.reset()
        gdnState.reset()
        mtp?.reset()
        pagedKV?.resetState()
    }

    public var continuationPosition: Int { kv.position }

    public func prepareForContinuation(expectedPosition: Int) throws {
        try prefillChunkState.requireClean(operation: "prepareForContinuation")
        // A speculative round may have committed verified tokens past the
        // consumed stream; the speculator rewinds the KV/GDN state when the
        // requested cursor falls inside its last verify span.
        if let mtp { try mtp.prepareForContinuation(expectedPosition: expectedPosition) }
        guard expectedPosition > 0, expectedPosition == kv.position else {
            throw PrefillError.prefillCursorMismatch(
                "Qwen 3.8 continuation cursor \(expectedPosition) does not match \(kv.position)")
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        if let mtp, useFusedGreedyHead {
            if let queued = try mtp.consumePending(token: token, position: position) {
                lastGreedyToken = queued
                return
            }
            if mtp.canRunRound(position: position) {
                try prefillChunkState.requireClean(operation: "produce")
                try Task.checkCancellation()
                guard token >= 0, token < Int32(cfg.vocabSize) else {
                    throw Qwen38ForwardRunnerError.invalidInput(
                        "Qwen 3.8 token is outside the vocabulary")
                }
                lastGreedyToken = try mtp.runRound(bonus: token, position: position)
                return
            }
        }
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

    /// Chunk-at-a-time prefill: each chunk of prompt tokens runs the whole
    /// layer stack with token-parallel kernels, and only the DeltaNet
    /// recurrence walks the chunk sequentially — inside its prefill kernel,
    /// producing exactly the FP32 state of token-by-token decode. The last
    /// chunk emits the final token's head through the decode head kernels
    /// (same buffers, same kernels), so prefill-then-decode continues
    /// bit-identically to pure sequential decode.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "Qwen 3.8 prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0, startPosition == kv.position else {
            throw PrefillError.chunkedUnsupported(
                "Qwen 3.8 prefill cursor \(kv.position) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "Qwen 3.8 prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        guard tokens.allSatisfy({ $0 >= 0 && $0 < Int32(cfg.vocabSize) }) else {
            throw Qwen38ForwardRunnerError.invalidInput(
                "Qwen 3.8 prefill token is outside the vocabulary")
        }
        let emitLogitsHead = !(outputMode == .greedyIfAvailable && useFusedGreedyHead)
        if emitLogitsHead {
            guard logits.length >= cfg.vocabSize * MemoryLayout<Float16>.stride else {
                throw Qwen38ForwardRunnerError.invalidInput(
                    "Qwen 3.8 logits buffer is too small")
            }
        }

        let scratch = try ensurePrefillScratch(config: config)
        // DFlash2 conditioning: the prompt's trailing tap rows (one drafter
        // context window) prime the first draft rounds — without them the
        // drafter starts blind and early acceptance craters. Restart the
        // drafter at the window base; each chunk stages its in-window rows.
        var dflash2TapBase = Int.max
        if let drafter = mtp?.dflash2 {
            drafter.reset()
            let finalPosition = startPosition + tokens.count
            dflash2TapBase = max(startPosition,
                                 finalPosition - drafter.contextWindowRows)
            drafter.alignPositionBase(dflash2TapBase)
        }
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
                                    writeFinalHead: spanIndex == spans.count - 1,
                                    dflash2TapBase: dflash2TapBase)
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
    /// and (on the final chunk) emit the last token's head. Everything
    /// encodes into a single command buffer — this family has no router
    /// readback — and the KV cursor advances only after it completes.
    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillScratch,
                                     writeFinalHead: Bool,
                                     dflash2TapBase: Int = .max) throws {
        let t = tokens.count
        precondition(t > 0 && t <= scratch.chunkTokens,
                     "prefill chunk exceeds its scratch capacity")
        let D = cfg.hiddenSize
        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = ctx.device.makeBuffer(
            bytes: tokenIDs,
            length: tokenIDs.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared) else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to allocate Qwen 3.8 prefill token buffer")
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
            if layer.isLinear {
                encodeLinearAttentionPrefill(cb, layer: layer, layerIndex: index,
                                             scratch: scratch, tokenCount: t)
            } else {
                cb = try encodeGatedFullAttentionPrefill(cb, layer: layer,
                                                         layerIndex: index,
                                                         scratch: scratch,
                                                         tokenCount: t,
                                                         startPosition: startPosition)
            }
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
            if let drafter = mtp?.dflash2, drafter.isTapLayer(index),
               dflash2TapBase < startPosition + t {
                let local = max(0, dflash2TapBase - startPosition)
                drafter.encodeTapCapture(
                    commandBuffer: cb,
                    layerIndex: index,
                    src: scratch.hidden,
                    srcOffset: local * D * MemoryLayout<Float16>.stride,
                    rows: t - local)
            }
        }
        if let drafter = mtp?.dflash2, dflash2TapBase < startPosition + t {
            drafter.commitTapRows(t - max(0, dflash2TapBase - startPosition))
        }

        let emitGreedyHead = outputMode == .greedyIfAvailable && useFusedGreedyHead
        if writeFinalHead {
            // The decode head kernels over the last token's row, so the
            // emitted logits are bit-identical to a sequential-decode head.
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
            if let mtp {
                // Seed the drafter's hidden input with the last prompt
                // position's final-norm row.
                rms.encodeBF16W(commandBuffer: cb,
                                x: scratch.hidden, xOffset: lastRowOffset,
                                weight: fNorm.buffer,
                                weightOffset: Int(fNorm.offset),
                                out: mtp.lastHiddenBuf,
                                d: UInt32(D), eps: Self.epsilon)
            }
        }

        try withExtendedLifetime(tokenBuffer) {
            try finish(cb)
        }
        if writeFinalHead, emitGreedyHead {
            lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
        }
        if let paged = pagedKV {
            // Page summaries for chunk-sealed pages were encoded in the chunk
            // command buffer itself, so nothing is pending here. The chunk
            // moved the frontier; any scores from an earlier token are stale
            // and the next decode token takes the warmup selection.
            paged.store.advance(by: t)
            for i in 0..<paged.lastScores.count { paged.lastScores[i] = [] }
        }
        kv.advance(by: t)
        prefillChunkState.markCommitted()
    }

    /// Gated-DeltaNet linear attention over one chunk: batched qkv/z/a/b
    /// projections, causal conv over [tail | chunk] with the tail carried
    /// forward, per-head q/k norm, the in-kernel sequential delta scan
    /// (exact FP32 state), gated norm, then out_proj into `h1`.
    private func encodeLinearAttentionPrefill(_ cb: MTLCommandBuffer,
                                              layer: LayerTensors,
                                              layerIndex: Int,
                                              scratch: PrefillScratch,
                                              tokenCount t: Int) {
        guard let qkvW = layer.linQKV, let zW = layer.linZ,
              let aW = layer.linA, let bW = layer.linB,
              let outW = layer.linOut, let convW = layer.linConv,
              let aLog = layer.linALog, let dtBias = layer.linDtBias,
              let gatedNormW = layer.linNorm else {
            preconditionFailure("linear-attention layer without GDN tensors")
        }
        let la = cfg.linearAttention
        let D = cfg.hiddenSize
        encodePrefillInt4Projection(cb,
                                    weights: qkvW,
                                    x: scratch.normed, y: scratch.gdnQKV,
                                    rows: la.qkvDim, columns: D,
                                    tokenCount: t,
                                    xStrideElements: D,
                                    yStrideElements: la.qkvDim)
        encodePrefillInt4Projection(cb,
                                    weights: zW,
                                    x: scratch.normed, y: scratch.gdnZ,
                                    rows: la.valueDim, columns: D,
                                    tokenCount: t,
                                    xStrideElements: D,
                                    yStrideElements: la.valueDim)
        encodePrefillInt4Projection(cb,
                                    weights: aW,
                                    x: scratch.normed, y: scratch.gdnA,
                                    rows: la.numVHeads, columns: D,
                                    tokenCount: t,
                                    xStrideElements: D,
                                    yStrideElements: la.numVHeads)
        encodePrefillInt4Projection(cb,
                                    weights: bW,
                                    x: scratch.normed, y: scratch.gdnB,
                                    rows: la.numVHeads, columns: D,
                                    tokenCount: t,
                                    xStrideElements: D,
                                    yStrideElements: la.numVHeads)
        let tail = gdnState.convTailBuffer(layer: layerIndex)
        gdn.encodeConvPrefill(commandBuffer: cb,
                              tail: tail,
                              qkvRows: scratch.gdnQKV,
                              convWeight: convW.buffer,
                              convWeightOffset: Int(convW.offset),
                              out: scratch.gdnConvOut,
                              rows: t)
        gdn.encodeConvTailUpdate(commandBuffer: cb,
                                 tail: tail,
                                 qkvRows: scratch.gdnQKV,
                                 rows: t)
        gdn.encodeQKNorm(commandBuffer: cb,
                         convOut: scratch.gdnConvOut,
                         rows: t)
        gdn.encodeDeltaStepPrefill(commandBuffer: cb,
                                   convOut: scratch.gdnConvOut,
                                   aProj: scratch.gdnA,
                                   bProj: scratch.gdnB,
                                   aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                   dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                                   state: gdnState.stateBuffer(layer: layerIndex),
                                   y: scratch.gdnY,
                                   rows: t)
        gdn.encodeGatedNorm(commandBuffer: cb,
                            y: scratch.gdnY,
                            z: scratch.gdnZ,
                            weight: gatedNormW.buffer,
                            weightOffset: Int(gatedNormW.offset),
                            out: scratch.attnOut,
                            rows: t)
        encodePrefillInt4Projection(cb,
                                    weights: outW,
                                    x: scratch.attnOut, y: scratch.h1,
                                    rows: D, columns: la.valueDim,
                                    tokenCount: t,
                                    xStrideElements: la.valueDim,
                                    yStrideElements: D)
    }

    /// Gated full attention over one chunk: batched packed-[query ; gate]
    /// q_proj plus K/V projections, per-head q/k norm + NeoX sub-dim RoPE
    /// over the chunk, KV-cache append, causal tiled prefill attention,
    /// sigmoid output gate, then o_proj into `h1`.
    /// Returns the command buffer that later encoders must continue on: the
    /// incoming one on the resident paths, or a fresh one after the blocked
    /// streamed path completed the incoming buffer mid-layer.
    private func encodeGatedFullAttentionPrefill(_ cb: MTLCommandBuffer,
                                                 layer: LayerTensors,
                                                 layerIndex: Int,
                                                 scratch: PrefillScratch,
                                                 tokenCount t: Int,
                                                 startPosition: Int) throws -> MTLCommandBuffer {
        guard let q = layer.q, let k = layer.k, let v = layer.v, let o = layer.o,
              let qNormW = layer.qNorm, let kNormW = layer.kNorm else {
            preconditionFailure("full-attention layer without attention tensors")
        }
        let D = cfg.hiddenSize
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = cfg.numHeads * headDim
        let kvDim = numKV * headDim
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

        encodePrefillInt4Projection(cb,
                                    weights: q,
                                    x: scratch.normed, y: scratch.qPacked,
                                    rows: 2 * qDim, columns: D,
                                    tokenCount: t,
                                    xStrideElements: D,
                                    yStrideElements: 2 * qDim)
        encodePrefillInt4Projection(cb,
                                    weights: k,
                                    x: scratch.normed, y: scratch.kStage,
                                    rows: kvDim, columns: D,
                                    tokenCount: t,
                                    xStrideElements: D,
                                    yStrideElements: kvDim)
        encodePrefillInt4Projection(cb,
                                    weights: v,
                                    x: scratch.normed, y: scratch.vStage,
                                    rows: kvDim, columns: D,
                                    tokenCount: t,
                                    xStrideElements: D,
                                    yStrideElements: kvDim)
        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: scratch.qPacked,
                                     q: scratch.attnQ,
                                     gate: scratch.attnGate,
                                     heads: cfg.numHeads,
                                     dim: headDim,
                                     rows: t)
        prefillQKVEpilogue.encodeNeoxSubdimNoVNorm(
            commandBuffer: cb,
            q: scratch.attnQ,
            k: scratch.kStage,
            qWeight: qNormW.buffer,
            qWeightOffset: Int(qNormW.offset),
            kWeight: kNormW.buffer,
            kWeightOffset: Int(kNormW.offset),
            startPosition: UInt32(startPosition),
            queryCount: UInt32(t),
            headDim: UInt32(headDim),
            numQHeads: UInt32(cfg.numHeads),
            numKVHeads: UInt32(numKV),
            qTokenStrideElements: UInt32(qDim),
            kvTokenStrideElements: UInt32(kvDim),
            theta: Float(cfg.fullRopeTheta),
            rotaryDim: rotaryDim,
            eps: Self.epsilon)
        var activeCB = cb
        let bytesPerToken = kvDim * MemoryLayout<Float16>.stride
        let endPages = (startPosition + t + KVPageGeometry.tokensPerPage - 1)
            / KVPageGeometry.tokensPerPage
        if let paged = pagedKV,
           !(paged.store.identityMappingIntact && endPages <= paged.poolPagesPerLayer) {
            // Blocked (streamed) path: the chunk's KV scatters into whatever
            // pool slots are free while the sealed past streams from the
            // spill file through staging windows, folded into FP32 running
            // softmax state. Exact — only the summation order differs from
            // the tensor-ops path.
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
        elementwise.encodeSigmoidGateMul(commandBuffer: activeCB,
                                         out: scratch.attnOut,
                                         gate: scratch.attnGate,
                                         count: t * qDim)
        encodePrefillInt4Projection(activeCB,
                                    weights: o,
                                    x: scratch.attnOut, y: scratch.h1,
                                    rows: D, columns: qDim,
                                    tokenCount: t,
                                    xStrideElements: qDim,
                                    yStrideElements: D)
        return activeCB
    }

    /// Scatter a chunk's staged K/V rows into their (possibly non-identity)
    /// pool page slots — one blit per touched page per stream.
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
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 paged KV scatter encoder")
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

    /// Quest min/max summaries for every page this chunk finishes filling —
    /// encoded in the chunk's own command buffer while the rows are
    /// guaranteed resident, so a long prefill never triggers a refetch storm
    /// at first decode.
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

    /// Stream the sealed past through staging windows and fold it, then the
    /// resident tail (the chunk's own pages plus any unsealed prefix) with
    /// the causal predicate, into the chunk's attention output. Synchronous:
    /// window N+1's pread overlaps window N's GPU pass via the stage ring.
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

        // Sealed past pages stream from the spill file.
        let pastPages = startPosition / pageTokens
        paged.store.flushSpills()
        var stageCBs: [MTLCommandBuffer?] = [nil, nil]
        var page = 0
        var windowIndex = 0
        while page < pastPages {
            let count = min(BlockedPrefillScratch.windowPages, pastPages - page)
            let stageIndex = windowIndex % 2
            if let previous = stageCBs[stageIndex] {
                previous.waitUntilCompleted()   // the stage is about to be rewritten
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

        // Resident tail: the partially-filled page at the prefill frontier
        // (if any) plus the chunk's own pages, causal-masked per query.
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

    /// Dense SwiGLU MLP over the chunk. With INT4 weights and a chunk tall
    /// enough to feed the tiled QMM, all three projections go batched (gate
    /// and up over [t, F], SiLU-mul, down over [t, D]) — this branch is the
    /// bulk of the prefill FLOPs, D 5120 x F 17408 on all 64 layers. The
    /// toy fixture's INT8 MLP and sub-tile chunks fall back to
    /// `PrefillSharedExpert`, whose per-row path is the decode kernel and
    /// therefore exact.
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
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 prefill MLP activation encoder")
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

    /// Batched INT4 projection over the chunk: the MPP TensorOps QMM when
    /// available and the chunk is at least one tile tall, the threadgroup
    /// QMM for tall chunks otherwise, and the decode GEMV repeated per row
    /// for short chunks. Unlike qwen36's dispatch policy, the q family also
    /// goes through the QMM: this family's q-side matrices (packed q_proj,
    /// GDN in_proj_qkv) are its largest resident projections, and a per-row
    /// replay of them would collapse the chunk back to sequential cost.
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

    /// Blit the chunk's staged K/V rows into the layer's cache, split into
    /// two spans when the physical layout wraps (never for the full-attention
    /// layers of this family, whose capacity is maxContext).
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
                throw Qwen38ForwardRunnerError.commandFailed(
                    "unable to create Qwen 3.8 prefill KV blit encoder")
            }
            blit.copy(from: source,
                      sourceOffset: sourceTokenOffset * bytesPerToken,
                      to: destination.buffer,
                      destinationOffset: destination.offset,
                      size: count * bytesPerToken)
            blit.endEncoding()
        }
        if let paged = pagedKV {
            // Identity slot mapping (pool == full context) keeps the pool
            // linear, so a chunk lands as one span per stream.
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

    /// One full decode step, encoded into a single command buffer: no router
    /// readback exists in this architecture, so nothing forces a mid-layer
    /// CPU round-trip.
    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer?,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try Task.checkCancellation()
        if let mtp, mtp.hasPending {
            throw Qwen38ForwardRunnerError.invalidInput(
                "Qwen 3.8 decode step with unconsumed speculative tokens; reset or continue first")
        }
        guard position == kv.position, position >= 0, position < maxContext else {
            throw Qwen38ForwardRunnerError.invalidInput(
                "Qwen 3.8 position \(position) does not match its KV cursor \(kv.position)")
        }
        guard token >= 0, token < Int32(cfg.vocabSize) else {
            throw Qwen38ForwardRunnerError.invalidInput(
                "Qwen 3.8 token is outside the vocabulary")
        }
        let emitLogitsHead = emitHead
            && !(useFusedGreedyHead && outputMode == .greedyIfAvailable)
        if emitLogitsHead {
            guard let logits,
                  logits.length >= cfg.vocabSize * MemoryLayout<Float16>.stride else {
                throw Qwen38ForwardRunnerError.invalidInput(
                    "Qwen 3.8 logits buffer is too small")
            }
        }

        // Paged mode: pick this token's page selection per layer (lag-one
        // scores from the previous token), fetch any spilled members, and
        // build the page tables — all before the command buffer encodes.
        if let paged = pagedKV {
            try paged.prepareSelections(position: position)
        }

        let D = UInt32(cfg.hiddenSize)
        let cb = try commandBuffer()

        // Sealed pages from the previous token need their Quest min/max
        // summaries before this token's score pass reads them.
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

        for (index, layer) in layers.enumerated() {
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: layer.inputNorm.buffer,
                            weightOffset: Int(layer.inputNorm.offset),
                            out: normed,
                            d: D, eps: Self.epsilon)
            if layer.isLinear {
                encodeLinearAttentionDecode(cb, layer: layer, layerIndex: index)
            } else {
                try encodeGatedFullAttentionDecode(cb, layer: layer,
                                               layerIndex: index,
                                               position: position,
                                               seqLen: UInt32(position + 1))
            }
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
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                          hidden: hidden,
                                          delta: mlpOut,
                                          count: cfg.hiddenSize)
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
            if let mtp {
                // The drafter consumes the target's final-norm hidden of the
                // last processed position; keep it current so a spec round
                // can start after any plain decode step.
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: fNorm.buffer,
                                weightOffset: Int(fNorm.offset),
                                out: mtp.lastHiddenBuf,
                                d: D, eps: Self.epsilon)
            }
        }

        try finish(cb)
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

    // MARK: - Paged decode support

    /// Gated-DeltaNet linear attention (layer mask 2), one decode step. Reads
    /// `normed`, updates the layer's recurrent state + conv tail in place,
    /// and leaves the attention-branch output in `oOut`. Hv 48 != the fused
    /// `_qwen` kernels' Hv 32, so `encodeDeltaGatedDecode` declines and the
    /// generic delta step + gated norm run instead.
    private func encodeLinearAttentionDecode(_ cb: MTLCommandBuffer,
                                             layer: LayerTensors,
                                             layerIndex: Int) {
        guard let qkvW = layer.linQKV, let zW = layer.linZ,
              let aW = layer.linA, let bW = layer.linB,
              let outW = layer.linOut, let convW = layer.linConv,
              let aLog = layer.linALog, let dtBias = layer.linDtBias,
              let gatedNormW = layer.linNorm else {
            preconditionFailure("linear-attention layer without GDN tensors")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)

        // One dispatch over the concatenated qkv/z/a/b row space instead of
        // four separate GEMVs.
        gdn.encodeInputProjections(commandBuffer: cb,
                                   x: normed,
                                   qkv: qkvW, qkvOut: gdnQKVRaw,
                                   z: zW, zOut: gdnZ,
                                   a: aW, aOut: gdnA,
                                   b: bW, bOut: gdnB,
                                   hiddenSize: cfg.hiddenSize)
        gdn.encodeConvDecode(commandBuffer: cb,
                             tail: gdnState.convTailBuffer(layer: layerIndex),
                             qkv: gdnQKVRaw,
                             convWeight: convW.buffer,
                             convWeightOffset: Int(convW.offset),
                             out: gdnConvOut)
        gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        let usedFusedDeltaNorm = gdn.encodeDeltaGatedDecode(
            commandBuffer: cb,
            convOut: gdnConvOut,
            aProj: gdnA,
            bProj: gdnB,
            aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
            dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
            state: gdnState.stateBuffer(layer: layerIndex),
            z: gdnZ,
            weight: gatedNormW.buffer, weightOffset: Int(gatedNormW.offset),
            out: gdnOut)
        if !usedFusedDeltaNorm {
            gdn.encodeDeltaStepDecode(commandBuffer: cb,
                                      convOut: gdnConvOut,
                                      aProj: gdnA,
                                      bProj: gdnB,
                                      aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                      dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                                      state: gdnState.stateBuffer(layer: layerIndex),
                                      y: gdnY)
            gdn.encodeGatedNorm(commandBuffer: cb,
                                y: gdnY,
                                z: gdnZ,
                                weight: gatedNormW.buffer,
                                weightOffset: Int(gatedNormW.offset),
                                out: gdnOut)
        }
        int4.encode(commandBuffer: cb,
                    weights: outW.buffer, weightsOffset: Int(outW.offset),
                    scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                    biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                    x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim))
    }

    /// Qwen full attention (attn_output_gate), one decode step: packed
    /// [query ; gate] q_proj split per head, weighted per-head q/k norms
    /// (no V norm), NeoX sub-dim RoPE over headDim * partialRotaryFactor
    /// dims, full attention at the configured scale, sigmoid output gate,
    /// then o_proj into `oOut`.
    private func encodeGatedFullAttentionDecode(_ cb: MTLCommandBuffer,
                                                layer: LayerTensors,
                                                layerIndex: Int,
                                                position: Int,
                                                seqLen: UInt32) throws {
        guard let q = layer.q, let k = layer.k, let v = layer.v, let o = layer.o,
              let qNormW = layer.qNorm, let kNormW = layer.kNorm else {
            preconditionFailure("full-attention layer without attention tensors")
        }
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
                            qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                            qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                            qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                            kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                            kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                            kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                            vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                            vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                            vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                            x: normed,
                            qOut: qPackedScratch,
                            kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                            vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                            qRows: 2 * qDim,
                            kvRows: kvDim,
                            n: D)
        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: qPackedScratch,
                                     q: qScratch,
                                     gate: attnGateScratch,
                                     heads: cfg.numHeads,
                                     dim: headDim)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: qScratch,
                               weight: qNormW.buffer,
                               weightOffset: Int(qNormW.offset),
                               out: qScratch,
                               headDim: UInt32(headDim),
                               numHeads: cfg.numHeads,
                               eps: Self.epsilon)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: kSlot.buffer, xOffset: kSlot.offset,
                               weight: kNormW.buffer,
                               weightOffset: Int(kNormW.offset),
                               out: kSlot.buffer, outOffset: kSlot.offset,
                               headDim: UInt32(headDim),
                               numHeads: numKV,
                               eps: Self.epsilon)
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
            // Score every sealed page against this token's query — the
            // selection input for the *next* token (lag-one policy).
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
        elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: attnOut,
                                         gate: attnGateScratch,
                                         count: Int(qDim))
        int4.encode(commandBuffer: cb,
                    weights: o.buffer, weightsOffset: Int(o.offset),
                    scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                    biases: o.buffer, biasesOffset: Int(o.biasOffset),
                    x: attnOut, y: oOut, m: D, n: qDim)
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 command buffer")
        }
        return cb
    }

    private func finish(_ cb: MTLCommandBuffer) throws {
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw Qwen38ForwardRunnerError.commandFailed(
                cb.error?.localizedDescription ?? "Qwen 3.8 command buffer did not complete")
        }
    }
}
