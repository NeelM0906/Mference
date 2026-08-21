import Foundation
import Metal

/// Greedy speculative decoding for the Qwen 3.8 dense family using the
/// checkpoint's native MTP (multi-token-prediction) draft layer.
///
/// Per round, with the last committed token `b` at position `P`:
///
///  1. **Draft** `k` tokens by iterating the MTP layer (one full-attention
///     block that consumes `fc(concat(norm_e(embed(tok)), norm_h(hidden)))`,
///     sharing the target's embedding and lm_head — mlx-vlm
///     `Qwen3_5MTPDraftModel` semantics).
///  2. **Verify** all `k+1` tokens `[b, d0..dk-1]` in ONE target pass built
///     from decode-exact kernels: every weight-consuming projection runs the
///     multi-token GEMV (`DequantInt4GEMVMultiX`, weights read once,
///     per-token math bit-identical to the decode GEMV) and every stateful
///     or reduction-order-sensitive step (attention, RoPE, per-head norms,
///     the GDN conv/delta recurrence) runs the decode kernels per token.
///     The per-position greedy head is `LMHeadGreedyMultiX`, bit-identical
///     to the fused decode head. Emitted tokens are therefore byte-identical
///     to plain decode for any draft quality.
///  3. **Accept** the longest draft prefix matching the target argmaxes; the
///     following token is the target's own argmax (standard bonus token).
///  4. **Roll back** on partial acceptance: the GDN FP32 states and conv
///     tails are captured per verified position into an arena during the
///     verify pass, so rollback is a blit restore of the state after the
///     last accepted row plus a KV cursor rewind. No replay pass.
///
/// The drafter's own KV cache is trimmed to the committed pair count each
/// round and re-fed with exact (token, target-hidden) pairs, which also
/// yields the next round's free "seed" draft token (mlx's
/// `accept_verified_tokens` flow, simplified to always re-feed).
final class Qwen38MTPSpeculator {

    struct Stats {
        var rounds = 0
        var draftedTokens = 0
        var acceptedTokens = 0
        var emittedTokens = 0
        var rollbacks = 0
        var draftNanos: UInt64 = 0
        var verifyNanos: UInt64 = 0
        var acceptNanos: UInt64 = 0
        /// Per-draft-position accept/trial counts (index = draft ordinal):
        /// separates "weak first draft" from "decay along the block".
        var positionTrials = [Int](repeating: 0, count: 8)
        var positionAccepts = [Int](repeating: 0, count: 8)
    }

    /// Resolved MTP tensor views (`mtp.*` names in the resident index).
    private struct Weights {
        let fc: TensorView
        let preFcNormEmbedding: TensorView
        let preFcNormHidden: TensorView
        let finalNorm: TensorView
        let inputNorm: TensorView
        let postAttnNorm: TensorView
        let q: TensorView
        let k: TensorView
        let v: TensorView
        let o: TensorView
        let qNorm: TensorView
        let kNorm: TensorView
        let mlpGate: SharedExpertProjection
        let mlpUp: SharedExpertProjection
        let mlpDown: SharedExpertProjection
    }

    private let model: Model
    private let ctx: MetalContext
    private let cfg: ArchConfig
    private let kv: KVCacheManager
    /// Paged long-context state shared with the runner; nil in linear mode.
    /// The verify pass writes draft rows through the page store and runs
    /// per-position paged sparse attention over the round's pinned selection.
    private let paged: Qwen38PagedKVRuntime?
    private let gdnState: GDNStateManager
    private let layers: [Qwen38ForwardRunner.LayerTensors]
    private let maxContext: Int
    private let mlpWeightBits: Int
    private let weights: Weights

    /// Scratch/arena capacity in draft tokens (`MFERENCE_MTP_K` at init).
    let draftCapacity: Int
    /// Number of draft tokens per round; test-adjustable up to the capacity.
    var draftCount: Int {
        didSet {
            precondition(draftCount >= 1 && draftCount <= draftCapacity,
                         "draftCount outside 1...\(draftCapacity)")
        }
    }

    // Kernels (pipeline states are shared through the MetalContext cache).
    private let embedInt4: EmbedLookupInt4
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let rms: RMSNorm
    private let prefillRMS: PrefillRMSNorm
    private let int4: DequantInt4GEMV
    private let multix: DequantInt4GEMVMultiX
    private let attention: Attention
    private let mtpAttention: Attention
    private let elementwise: Elementwise
    private let rope: RoPE
    private let gdn: GDN
    private let mtpMLP: SharedExpertRuntime
    private let verifyMLPInt8: SharedExpertRuntime?
    private let siluMulPSO: MTLComputePipelineState
    private let verifyHead: LMHeadGreedyMultiX
    private let draftHead: LMHeadChainInt4

    // Verify scratch, `vT = draftCount + 1` rows.
    private let vT: Int
    private let vHidden: MTLBuffer      // [T, D]
    private let vNormed: MTLBuffer      // [T, D]
    private let vMlpX: MTLBuffer        // [T, D]
    private let vOut1: MTLBuffer        // [T, D]
    private let vMlpOut: MTLBuffer      // [T, D]
    private let vQPacked: MTLBuffer     // [T, 2*qDim]
    private let vQ: MTLBuffer           // [T, qDim]
    private let vGate: MTLBuffer        // [T, qDim]
    private let vAttnOut: MTLBuffer     // [T, qDim]
    private let vKStage: MTLBuffer      // [T, kvDim]
    private let vVStage: MTLBuffer      // [T, kvDim]
    private let vMlpGate: MTLBuffer     // [T, F]
    private let vMlpUp: MTLBuffer       // [T, F]
    private let vMlpAct: MTLBuffer      // [T, F]
    private let vGdnQKV: MTLBuffer      // [T, qkvDim]
    private let vGdnConv: MTLBuffer     // [T, qkvDim]
    private let vGdnZ: MTLBuffer        // [T, valueDim]
    private let vGdnA: MTLBuffer        // [T, Hv]
    private let vGdnB: MTLBuffer        // [T, Hv]
    private let vGdnY: MTLBuffer        // [T, valueDim]
    private let vGdnOut: MTLBuffer      // [T, valueDim]
    private let normedHidden: MTLBuffer // [T, D] final-norm rows (verify)
    private let verifyTokensBuf: MTLBuffer // [T] UInt32
    private let tokenIdsBuf: MTLBuffer  // [max(T, R)] UInt32

    // GDN rollback arena: `vT - 1` capture slots (state after verify rows
    // 0..vT-2), each holding every linear layer's FP32 state + conv tail.
    private let linearLayerIndex: [Int?] // layer -> dense linear ordinal
    private let numLinearLayers: Int
    private let stateArena: MTLBuffer
    private let tailArena: MTLBuffer
    private var arenaRoundBase = -1     // kv position of the last round's row 0
    private var arenaCaptured = 0

    // MTP layer state.
    private let mtpKVDim: Int
    private let mtpK: MTLBuffer         // [maxContext, kvDim]
    private let mtpV: MTLBuffer         // [maxContext, kvDim]
    private var mtpLen = 0              // committed pair entries
    private var basePair = 0            // pair index of slot 0
    private var started = false
    private var haveSeed = false
    private var seedToken: Int32 = 0
    private let seedHiddenBuf: MTLBuffer // [D] post-mtp-norm hidden of the seed pair
    let lastHiddenBuf: MTLBuffer         // [D] target final-norm hidden of the last processed position

    // Draft scratch (single token).
    private let dEmb: MTLBuffer         // [D]
    private let dFcIn: MTLBuffer        // [2D]
    private let dH: MTLBuffer           // [D]
    private let dNormed: MTLBuffer      // [D]
    private let dQPacked: MTLBuffer     // [2*qDim]
    private let dQ: MTLBuffer           // [qDim]
    private let dGate: MTLBuffer        // [qDim]
    private let dAttnOut: MTLBuffer     // [qDim]
    private let dOut1: MTLBuffer        // [D]
    private let dMlpX: MTLBuffer        // [D]
    private let dMlpGate: MTLBuffer     // [F]
    private let dMlpUp: MTLBuffer       // [F]
    private let dMlpAct: MTLBuffer      // [F]
    private let dMlpOut: MTLBuffer      // [D]
    private let dHOutA: MTLBuffer       // [D] ping
    private let dHOutB: MTLBuffer       // [D] pong
    private let dTokenBuf: MTLBuffer    // [1] UInt32

    // Accept scratch, `aR = draftCount + 2` rows.
    private let aR: Int
    private let aEmb: MTLBuffer         // [R, D]
    private let aEmbNorm: MTLBuffer     // [R, D]
    private let aHiddenSrc: MTLBuffer   // [R, D]
    private let aHidNorm: MTLBuffer     // [R, D]
    private let aFcIn: MTLBuffer        // [R, 2D]
    private let aH: MTLBuffer           // [R, D]
    private let aNormed: MTLBuffer      // [R, D]
    private let aQPacked: MTLBuffer     // [R, 2*qDim]
    private let aQ: MTLBuffer           // [R, qDim]
    private let aGate: MTLBuffer        // [R, qDim]
    private let aAttnOut: MTLBuffer     // [R, qDim]
    private let aOut1: MTLBuffer        // [R, D]
    private let aMlpX: MTLBuffer        // [R, D]
    private let aMlpGate: MTLBuffer     // [R, F]
    private let aMlpUp: MTLBuffer       // [R, F]
    private let aMlpAct: MTLBuffer      // [R, F]
    private let aMlpOut: MTLBuffer      // [R, D]

    // Pending verified tokens: produce(expectInput, position) -> emit.
    private var pending: [(expectInput: Int32, emit: UInt32)] = []
    private var pendingPosition = 0

    private(set) var stats = Stats()

    /// Test hook: supplies the draft tokens for a round instead of the MTP
    /// layer (still `draftCount`-clamped). Byte-identity must hold for any
    /// value this returns.
    var draftOverride: ((_ round: Int, _ k: Int) -> [Int32])?

    /// Alternative draft source: the DFlash2 block-diffusion drafter. When
    /// set, rounds draft through it (the MTP layer sits idle), the verify
    /// pass stages target-tap rows for it, and the accept pass shrinks to
    /// the GDN restore. Emitted bytes stay identical either way.
    var dflash2: Qwen38DFlash2Drafter?

    private static let epsilon: Float = 1e-6

    /// Probe the resident index for MTP tensors; returns nil when the
    /// install carries none (spec decode silently unavailable).
    static func probe(model: Model,
                      context: MetalContext,
                      config: ArchConfig,
                      kv: KVCacheManager,
                      gdnState: GDNStateManager,
                      layers: [Qwen38ForwardRunner.LayerTensors],
                      maxContext: Int,
                      mlpWeightBits: Int,
                      paged: Qwen38PagedKVRuntime? = nil) throws -> Qwen38MTPSpeculator? {
        guard (try? model.resident(name: "mtp.fc.weight")) != nil else { return nil }
        return try Qwen38MTPSpeculator(model: model, context: context, config: config,
                                       kv: kv, gdnState: gdnState, layers: layers,
                                       maxContext: maxContext,
                                       mlpWeightBits: mlpWeightBits,
                                       paged: paged)
    }

    private init(model: Model,
                 context: MetalContext,
                 config: ArchConfig,
                 kv: KVCacheManager,
                 gdnState: GDNStateManager,
                 layers: [Qwen38ForwardRunner.LayerTensors],
                 maxContext: Int,
                 mlpWeightBits: Int,
                 paged: Qwen38PagedKVRuntime?) throws {
        self.model = model
        self.ctx = context
        self.cfg = config
        self.kv = kv
        self.paged = paged
        self.gdnState = gdnState
        self.layers = layers
        self.maxContext = maxContext
        self.mlpWeightBits = mlpWeightBits

        let D = config.hiddenSize
        let F = config.intermediateSize
        let qDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        self.mtpKVDim = kvDim

        func view(_ name: String) throws -> TensorView {
            try model.resident(name: "mtp.\(name)")
        }
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
        self.weights = Weights(
            fc: try view("fc.weight"),
            preFcNormEmbedding: try view("pre_fc_norm_embedding.weight"),
            preFcNormHidden: try view("pre_fc_norm_hidden.weight"),
            finalNorm: try view("norm.weight"),
            inputNorm: try view("layers.0.input_layernorm.weight"),
            postAttnNorm: try view("layers.0.post_attention_layernorm.weight"),
            q: try view("layers.0.self_attn.q_proj.weight"),
            k: try view("layers.0.self_attn.k_proj.weight"),
            v: try view("layers.0.self_attn.v_proj.weight"),
            o: try view("layers.0.self_attn.o_proj.weight"),
            qNorm: try view("layers.0.self_attn.q_norm.weight"),
            kNorm: try view("layers.0.self_attn.k_norm.weight"),
            mlpGate: projection(try view("layers.0.mlp.gate_proj.weight"), rows: F, cols: D),
            mlpUp: projection(try view("layers.0.mlp.up_proj.weight"), rows: F, cols: D),
            mlpDown: projection(try view("layers.0.mlp.down_proj.weight"), rows: D, cols: F))

        let requestedK = ProcessInfo.processInfo.environment["MFERENCE_MTP_K"].flatMap(Int.init)
        // Upper bound 6 keeps the GDN capture arena bounded (k slots of every
        // linear layer's FP32 state) and the verify batch within the multi-x
        // kernels' 8-token limit. The DFlash2 drafter's acceptance length
        // justifies the full width by default; MTP stays at its measured 3.
        let dflash2Requested = (ProcessInfo.processInfo
            .environment["MFERENCE_DFLASH2_DIR"]?.isEmpty == false)
        let capacity = min(max(requestedK ?? (dflash2Requested ? 6 : 3), 1), 6)
        self.draftCapacity = capacity
        self.draftCount = capacity
        let vT = capacity + 1
        let aR = capacity + 2
        self.vT = vT
        self.aR = aR

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.rms = try RMSNorm(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context)
        self.multix = try DequantInt4GEMVMultiX(context: context)
        self.attention = try Attention(context: context)
        self.mtpAttention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.rope = try RoPE(context: context)
        self.gdn = try GDN(context: context, config: config.linearAttention,
                           specializedHiddenSize: config.hiddenSize)
        // MTP projections are always INT4 (the attach step quantizes them);
        // the target MLP follows the install's attention quant.
        self.mtpMLP = try SharedExpertRuntime(context: context, weightBits: 4,
                                              siluActivation: config.hiddenActivation == "silu",
                                              specializedD: D, specializedF: F)
        self.verifyMLPInt8 = mlpWeightBits == 8
            ? try SharedExpertRuntime(context: context, weightBits: 8,
                                      siluActivation: config.hiddenActivation == "silu")
            : nil
        self.siluMulPSO = try context.pipeline(
            config.hiddenActivation == "silu" ? "silu_mul_fp16" : "gelu_mul_fp16")
        self.verifyHead = try LMHeadGreedyMultiX(context: context, maxVocab: config.vocabSize)
        self.draftHead = try LMHeadChainInt4(context: context, maxD: D,
                                             maxVocab: config.vocabSize)

        let la = config.linearAttention
        let device = context.device
        func buf(_ elements: Int, _ label: String,
                 stride: Int = MemoryLayout<Float16>.stride,
                 options: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: max(elements, 1) * stride,
                                               options: options) else {
                throw Qwen38ForwardRunnerError.invalidConfiguration(
                    "unable to allocate Qwen 3.8 MTP scratch (\(label))")
            }
            made.label = "qwen38.mtp.\(label)"
            return made
        }

        self.vHidden = try buf(vT * D, "v.hidden")
        self.vNormed = try buf(vT * D, "v.normed")
        self.vMlpX = try buf(vT * D, "v.mlpX")
        self.vOut1 = try buf(vT * D, "v.out1")
        self.vMlpOut = try buf(vT * D, "v.mlpOut")
        self.vQPacked = try buf(vT * 2 * qDim, "v.qPacked")
        self.vQ = try buf(vT * qDim, "v.q")
        self.vGate = try buf(vT * qDim, "v.gate")
        self.vAttnOut = try buf(vT * qDim, "v.attnOut")
        self.vKStage = try buf(vT * kvDim, "v.kStage")
        self.vVStage = try buf(vT * kvDim, "v.vStage")
        self.vMlpGate = try buf(vT * F, "v.mlpGate")
        self.vMlpUp = try buf(vT * F, "v.mlpUp")
        self.vMlpAct = try buf(vT * F, "v.mlpAct")
        self.vGdnQKV = try buf(vT * la.qkvDim, "v.gdnQKV")
        self.vGdnConv = try buf(vT * la.qkvDim, "v.gdnConv")
        self.vGdnZ = try buf(vT * la.valueDim, "v.gdnZ")
        self.vGdnA = try buf(vT * la.numVHeads, "v.gdnA")
        self.vGdnB = try buf(vT * la.numVHeads, "v.gdnB")
        self.vGdnY = try buf(vT * la.valueDim, "v.gdnY")
        self.vGdnOut = try buf(vT * la.valueDim, "v.gdnOut")
        self.normedHidden = try buf(vT * D, "v.normedHidden")
        self.verifyTokensBuf = try buf(vT, "v.tokens", stride: MemoryLayout<UInt32>.stride)
        self.tokenIdsBuf = try buf(max(vT, aR), "tokenIds", stride: MemoryLayout<UInt32>.stride)

        var linearIndex: [Int?] = []
        var linearCount = 0
        for layer in 0..<config.numLayers {
            if config.layerIsLinear(layer) {
                linearIndex.append(linearCount)
                linearCount += 1
            } else {
                linearIndex.append(nil)
            }
        }
        self.linearLayerIndex = linearIndex
        self.numLinearLayers = linearCount
        let stateBytes = gdnState.stateBytesPerLayer
        let tailBytes = gdnState.convTailBytesPerLayer
        let captureSlots = max(vT - 1, 1)
        self.stateArena = try buf(captureSlots * linearCount * stateBytes, "gdn.stateArena",
                                  stride: 1, options: .storageModePrivate)
        self.tailArena = try buf(captureSlots * linearCount * tailBytes, "gdn.tailArena",
                                 stride: 1, options: .storageModePrivate)

        self.mtpK = try buf(maxContext * kvDim, "mtp.K")
        self.mtpV = try buf(maxContext * kvDim, "mtp.V")
        self.seedHiddenBuf = try buf(D, "mtp.seedHidden")
        self.lastHiddenBuf = try buf(D, "mtp.lastHidden")

        self.dEmb = try buf(D, "d.emb")
        self.dFcIn = try buf(2 * D, "d.fcIn")
        self.dH = try buf(D, "d.h")
        self.dNormed = try buf(D, "d.normed")
        self.dQPacked = try buf(2 * qDim, "d.qPacked")
        self.dQ = try buf(qDim, "d.q")
        self.dGate = try buf(qDim, "d.gate")
        self.dAttnOut = try buf(qDim, "d.attnOut")
        self.dOut1 = try buf(D, "d.out1")
        self.dMlpX = try buf(D, "d.mlpX")
        self.dMlpGate = try buf(F, "d.mlpGate")
        self.dMlpUp = try buf(F, "d.mlpUp")
        self.dMlpAct = try buf(F, "d.mlpAct")
        self.dMlpOut = try buf(D, "d.mlpOut")
        self.dHOutA = try buf(D, "d.hOutA")
        self.dHOutB = try buf(D, "d.hOutB")
        self.dTokenBuf = try buf(1, "d.token", stride: MemoryLayout<UInt32>.stride)

        self.aEmb = try buf(aR * D, "a.emb")
        self.aEmbNorm = try buf(aR * D, "a.embNorm")
        self.aHiddenSrc = try buf(aR * D, "a.hiddenSrc")
        self.aHidNorm = try buf(aR * D, "a.hidNorm")
        self.aFcIn = try buf(aR * 2 * D, "a.fcIn")
        self.aH = try buf(aR * D, "a.h")
        self.aNormed = try buf(aR * D, "a.normed")
        self.aQPacked = try buf(aR * 2 * qDim, "a.qPacked")
        self.aQ = try buf(aR * qDim, "a.q")
        self.aGate = try buf(aR * qDim, "a.gate")
        self.aAttnOut = try buf(aR * qDim, "a.attnOut")
        self.aOut1 = try buf(aR * D, "a.out1")
        self.aMlpX = try buf(aR * D, "a.mlpX")
        self.aMlpGate = try buf(aR * F, "a.mlpGate")
        self.aMlpUp = try buf(aR * F, "a.mlpUp")
        self.aMlpAct = try buf(aR * F, "a.mlpAct")
        self.aMlpOut = try buf(aR * D, "a.mlpOut")
    }

    // MARK: - Session state

    func reset() {
        pending.removeAll(keepingCapacity: true)
        pendingPosition = 0
        started = false
        haveSeed = false
        mtpLen = 0
        basePair = 0
        arenaRoundBase = -1
        arenaCaptured = 0
        dflash2?.reset()
    }

    var hasPending: Bool { !pending.isEmpty }

    /// Continuation support after a generation that ended mid-round: the KV
    /// and GDN state may have run ahead of the tokens the caller consumed.
    /// When the requested cursor falls inside the last verify span, restore
    /// the captured GDN state for that position and rewind the KV cursor;
    /// otherwise (and for an exact match) just clear speculative state.
    func prepareForContinuation(expectedPosition: Int) throws {
        defer {
            pending.removeAll(keepingCapacity: true)
            started = false
            haveSeed = false
        }
        if expectedPosition == kv.position { return }
        guard expectedPosition < kv.position else { return } // main guard throws
        let row = expectedPosition - 1 - arenaRoundBase
        guard arenaRoundBase >= 0, row >= 0, row < arenaCaptured else {
            throw PrefillError.prefillCursorMismatch(
                "Qwen 3.8 continuation cursor \(expectedPosition) is behind the "
                + "speculative KV cursor \(kv.position) and outside the last verify span")
        }
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 MTP continuation command buffer")
        }
        try encodeGDNRestore(cb, slot: row)
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "Qwen 3.8 MTP continuation restore did not complete")
        }
        kv.rewind(to: expectedPosition)
        paged?.store.rewind(to: expectedPosition)
        arenaRoundBase = -1
        arenaCaptured = 0
    }

    // MARK: - Round orchestration

    /// Serve a queued verified token if this call matches the expectation.
    func consumePending(token: Int32, position: Int) throws -> UInt32? {
        guard !pending.isEmpty else { return nil }
        let next = pending.removeFirst()
        guard position == pendingPosition, token == next.expectInput else {
            throw Qwen38ForwardRunnerError.invalidInput(
                "Qwen 3.8 speculative stream desync: produce(\(token), \(position)) "
                + "does not match the pending verified continuation "
                + "(\(next.expectInput), \(pendingPosition))")
        }
        pendingPosition += 1
        stats.emittedTokens += 1
        return next.emit
    }

    /// A round needs the cursor caught up and room for at least one draft.
    func canRunRound(position: Int) -> Bool {
        // A round needs a prior decoded position: the drafter seeds from the
        // previous token's final-norm hidden, and `basePair = position - 1`
        // anchors its RoPE — position 0 decodes plainly.
        guard position >= 1, position == kv.position,
              position + 2 <= maxContext else { return false }
        guard let paged else { return true }
        // Sparse selections break the byte-identity contract: a round reuses
        // one page table across its verify rows where plain decode reselects
        // per token, and accepted rows are emitted without their own score
        // pass. Rounds therefore run only while the selection through the
        // round's span — plus one position of margin, so the token feeding
        // the first sparse selection's lag-one scores is always
        // plain-decoded — covers the entire context. Past that point decode
        // falls back to plain paged tokens.
        let kRound = min(draftCount, maxContext - position - 1)
        let horizon = min(position + kRound + 1, maxContext - 1)
        return paged.selectionIsExhaustive(at: horizon, maxSpanTokens: kRound + 1)
    }

    /// Run one draft/verify/accept round for `produce(bonus, position)`.
    /// Returns the first verified token; the rest are queued.
    func runRound(bonus: Int32, position: Int) throws -> UInt32 {
        precondition(pending.isEmpty, "spec round with pending tokens")
        let P = position
        let kRound = min(draftCount, maxContext - P - 1)
        precondition(kRound >= 1, "runRound caller must guarantee draft room")

        // (Re)initialize the drafter when starting or after a cursor gap
        // (plain decode steps or a resumed prefill advanced the target).
        if !started || basePair + mtpLen < P - 1 {
            started = true
            basePair = P - 1
            mtpLen = 0
            haveSeed = false
        }

        // 1. Draft.
        let draftStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var drafts: [Int32] = []
        if let draftOverride {
            drafts = Array(draftOverride(stats.rounds, kRound).prefix(kRound))
            precondition(!drafts.isEmpty, "draft override returned no tokens")
            precondition(drafts.allSatisfy { $0 >= 0 && $0 < Int32(cfg.vocabSize) },
                         "draft override token outside the vocabulary")
        } else if let dflash2 {
            // Continuity invariant: the drafter's context rows plus its
            // pending taps must land exactly at P. Plain-decode gaps or a
            // continuation rewind break it; the drafter restarts with empty
            // context at the right positions (quality-only, never bytes).
            if dflash2.ctxTotal + dflash2.pendingTapRows != P {
                dflash2.reset()
                dflash2.alignPositionBase(P)
            }
            drafts = try dflash2.propose(anchor: bonus, maxDrafts: kRound)
        } else {
            var slot = mtpLen
            var tok: Int32
            var hPrev: MTLBuffer
            if haveSeed {
                drafts.append(seedToken)
                tok = seedToken
                hPrev = seedHiddenBuf
            } else {
                tok = bonus
                hPrev = lastHiddenBuf
            }
            var useA = true
            while drafts.count < kRound {
                let hOut = useA ? dHOutA : dHOutB
                let next = try runDraftForward(token: tok, hPrev: hPrev,
                                               hOut: hOut, slot: slot)
                slot += 1
                tok = next
                hPrev = hOut
                useA.toggle()
                drafts.append(next)
            }
        }
        stats.draftNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - draftStart
        stats.draftedTokens += drafts.count

        // 2. Verify.
        let verifyStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let verifyTokens = [bonus] + drafts
        if let paged {
            // Round selection: lag-one scores as in plain decode, with the
            // table extended over the pages the verify span writes into.
            try paged.prepareSelections(position: P)
            try paged.appendVerifySpan(position: P, count: verifyTokens.count)
        }
        try runVerifyPass(tokens: verifyTokens, startPosition: P)
        kv.advance(by: verifyTokens.count)
        paged?.store.advance(by: verifyTokens.count)
        arenaRoundBase = P
        arenaCaptured = verifyTokens.count - 1

        let targets = verifyTokensBuf.contents()
            .bindMemory(to: UInt32.self, capacity: verifyTokens.count)
        var accepted = 0
        while accepted < drafts.count,
              targets[accepted] == UInt32(bitPattern: drafts[accepted]) {
            accepted += 1
        }
        let emitted = (0...accepted).map { targets[$0] } // t1..t_{accepted+1}
        stats.verifyNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - verifyStart
        stats.rounds += 1
        stats.acceptedTokens += accepted
        for i in 0..<min(drafts.count, stats.positionTrials.count) {
            stats.positionTrials[i] += 1
            if i < accepted { stats.positionAccepts[i] += 1 }
        }

        // 3. Accept + rollback.
        let acceptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let rolledBack = accepted < drafts.count
        if rolledBack {
            stats.rollbacks += 1
            kv.rewind(to: P + accepted + 1)
            paged?.store.rewind(to: P + accepted + 1)
        }
        if let paged {
            // Net-sealed pages need their summaries next command buffer, and
            // the round's score pass (bonus-position query) feeds the next
            // selection.
            paged.noteAdvance(from: P, to: P + accepted + 1)
            paged.readBackScores(sealedPages: P / KVPageGeometry.tokensPerPage)
        }
        if let dflash2 {
            // DFlash2 keeps its own context; the accept pass reduces to the
            // GDN rollback restore plus committing the accepted tap rows.
            if rolledBack {
                let restoreCB = try commandBuffer()
                try encodeGDNRestore(restoreCB, slot: accepted)
                try finish(restoreCB)
            }
            dflash2.commitTapRows(accepted + 1)
        } else {
            try runAcceptPass(bonus: bonus, emitted: emitted, position: P,
                              restoreSlot: rolledBack ? accepted : nil)
        }
        stats.acceptNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - acceptStart

        // 4. Queue.
        pending = (1..<emitted.count).map {
            (expectInput: Int32(bitPattern: emitted[$0 - 1]), emit: emitted[$0])
        }
        pendingPosition = P + 1
        stats.emittedTokens += 1
        return emitted[0]
    }

    // MARK: - Target verify pass (decode-exact, weights read once)

    private func runVerifyPass(tokens: [Int32], startPosition: Int) throws {
        let t = tokens.count
        precondition(t >= 2 && t <= vT, "verify batch \(t) outside 2...\(vT)")
        let D = cfg.hiddenSize
        let ids = tokenIdsBuf.contents().bindMemory(to: UInt32.self, capacity: t)
        for (i, token) in tokens.enumerated() { ids[i] = UInt32(bitPattern: token) }

        let cb = try commandBuffer()
        try paged?.encodePendingMetadata(commandBuffer: cb)
        let emb = model.embedding
        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer, tableOffset: Int(emb.offset),
                            scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                            tokens: tokenIdsBuf,
                            out: vHidden,
                            t: UInt32(t), d: UInt32(D),
                            outScale: 1.0)

        for (index, layer) in layers.enumerated() {
            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: vHidden,
                                   weight: layer.inputNorm.buffer,
                                   weightOffset: Int(layer.inputNorm.offset),
                                   out: vNormed,
                                   t: UInt32(t), d: UInt32(D),
                                   eps: Self.epsilon)
            if layer.isLinear {
                try encodeVerifyLinear(cb, layer: layer, layerIndex: index, tokenCount: t)
            } else {
                try encodeVerifyFullAttention(cb, layer: layer, layerIndex: index,
                                              tokenCount: t, startPosition: startPosition)
            }
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                          hidden: vHidden, delta: vOut1,
                                          count: t * D)
            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: vHidden,
                                   weight: layer.postAttnNorm.buffer,
                                   weightOffset: Int(layer.postAttnNorm.offset),
                                   out: vMlpX,
                                   t: UInt32(t), d: UInt32(D),
                                   eps: Self.epsilon)
            try encodeVerifyMLP(cb, layer: layer, tokenCount: t)
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                          hidden: vHidden, delta: vMlpOut,
                                          count: t * D)
            // DFlash2 conditioning: stage this layer's output rows for the
            // drafter's target-hidden feature. Rows beyond the accepted
            // prefix are overwritten by the next verify (commitTapRows only
            // advances past the committed ones).
            if let dflash2, dflash2.isTapLayer(index) {
                dflash2.encodeTapCapture(commandBuffer: cb,
                                         layerIndex: index,
                                         src: vHidden, rows: t)
            }
        }

        let fNorm = model.finalNorm
        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: vHidden,
                               weight: fNorm.buffer,
                               weightOffset: Int(fNorm.offset),
                               out: normedHidden,
                               t: UInt32(t), d: UInt32(D),
                               eps: Self.epsilon)
        let lm = model.lmHead
        verifyHead.encode(commandBuffer: cb,
                          xNormed: normedHidden,
                          weights: lm.buffer, weightsOffset: Int(lm.offset),
                          scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                          biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                          outTokens: verifyTokensBuf,
                          d: D, vocab: cfg.vocabSize, tokens: t)
        try finish(cb)
    }

    private func encodeVerifyFullAttention(_ cb: MTLCommandBuffer,
                                           layer: Qwen38ForwardRunner.LayerTensors,
                                           layerIndex: Int,
                                           tokenCount t: Int,
                                           startPosition: Int) throws {
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
        let h = MemoryLayout<Float16>.stride

        func mx(_ w: TensorView, y: MTLBuffer, m: Int, n: Int) {
            multix.encode(commandBuffer: cb,
                          weights: w.buffer, weightsOffset: Int(w.offset),
                          scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                          biases: w.buffer, biasesOffset: Int(w.biasOffset),
                          x: vNormed, y: y, m: m, n: n, tokens: t)
        }
        mx(q, y: vQPacked, m: 2 * qDim, n: D)
        mx(k, y: vKStage, m: kvDim, n: D)
        mx(v, y: vVStage, m: kvDim, n: D)
        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: vQPacked, q: vQ, gate: vGate,
                                     heads: cfg.numHeads, dim: headDim, rows: t)

        // Stage -> cache: contiguous slots in linear mode, per-page scatter
        // through the page store in paged mode (the span may cross a page
        // boundary into a freshly allocated slot).
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 MTP verify KV blit encoder")
        }
        if let paged {
            let pageTokens = KVPageGeometry.tokensPerPage
            let firstPage = startPosition / pageTokens
            let lastPage = (startPosition + t - 1) / pageTokens
            for page in firstPage...lastPage {
                let writeStart = max(page * pageTokens, startPosition)
                let writeEnd = min((page + 1) * pageTokens, startPosition + t)
                let kDst = try paged.store.kSlot(layer: layerIndex, position: writeStart)
                let vDst = try paged.store.vSlot(layer: layerIndex, position: writeStart)
                let srcOffset = (writeStart - startPosition) * kvDim * h
                blit.copy(from: vKStage, sourceOffset: srcOffset,
                          to: kDst.buffer, destinationOffset: kDst.offset,
                          size: (writeEnd - writeStart) * kvDim * h)
                blit.copy(from: vVStage, sourceOffset: srcOffset,
                          to: vDst.buffer, destinationOffset: vDst.offset,
                          size: (writeEnd - writeStart) * kvDim * h)
            }
        } else {
            let kRange = kv.kRange(layer: layerIndex, start: startPosition, count: t)
            let vRange = kv.vRange(layer: layerIndex, start: startPosition, count: t)
            blit.copy(from: vKStage, sourceOffset: 0,
                      to: kRange.buffer, destinationOffset: kRange.offset,
                      size: t * kvDim * h)
            blit.copy(from: vVStage, sourceOffset: 0,
                      to: vRange.buffer, destinationOffset: vRange.offset,
                      size: t * kvDim * h)
        }
        blit.endEncoding()

        for i in 0..<t {
            let kSlot = try paged?.store.kSlot(layer: layerIndex,
                                               position: startPosition + i)
                ?? kv.kSlot(layer: layerIndex, position: startPosition + i)
            rms.encodeBF16WPerHead(commandBuffer: cb,
                                   x: vQ, xOffset: i * qDim * h,
                                   weight: qNormW.buffer,
                                   weightOffset: Int(qNormW.offset),
                                   out: vQ, outOffset: i * qDim * h,
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
                                  data: vQ, dataOffset: i * qDim * h,
                                  position: UInt32(startPosition + i),
                                  headDim: UInt32(headDim),
                                  numHeads: UInt32(cfg.numHeads),
                                  rotaryDim: rotaryDim,
                                  theta: Float(cfg.fullRopeTheta))
            rope.encodeNeoxSubdim(commandBuffer: cb,
                                  data: kSlot.buffer, dataOffset: kSlot.offset,
                                  position: UInt32(startPosition + i),
                                  headDim: UInt32(headDim),
                                  numHeads: UInt32(numKV),
                                  rotaryDim: rotaryDim,
                                  theta: Float(cfg.fullRopeTheta))
        }
        if let paged {
            let g = paged.store.geometry
            guard let ordinal = paged.store.fullLayerOrdinal(forLayer: layerIndex) else {
                preconditionFailure("paged verify on a non-full-attention layer")
            }
            // Verify position i attends the round's selected pages plus the
            // span's rows through i — the tail page fills logically in place,
            // so the selected-token count just grows by one per position.
            let base = paged.verifyBaseTokens(ordinal: ordinal)
                + startPosition % KVPageGeometry.tokensPerPage
            for i in 0..<t {
                attention.encodeFullPaged(
                    commandBuffer: cb,
                    q: vQ, qOffset: i * qDim * h,
                    kPool: paged.store.kPoolBuffer(layer: layerIndex),
                    vPool: paged.store.vPoolBuffer(layer: layerIndex),
                    pageTable: paged.tablesBuf,
                    pageTableOffset: ordinal * g.pagesPerLayer
                        * MemoryLayout<UInt32>.stride,
                    out: vAttnOut, outOffset: i * qDim * h,
                    headDim: UInt32(headDim),
                    numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(numKV),
                    selTokens: UInt32(base + i + 1),
                    scale: Float(cfg.attentionScale))
            }
            // Bonus-position query scores every sealed page for the next
            // selection — the bonus token is always committed, so its query
            // is always the right one to rank against.
            paged.encodeScores(commandBuffer: cb, ordinal: ordinal,
                               q: vQ, qOffset: 0,
                               sealedPages: startPosition / KVPageGeometry.tokensPerPage)
        } else {
            for i in 0..<t {
                attention.encodeFull(commandBuffer: cb,
                                     q: vQ, qOffset: i * qDim * h,
                                     k: kv.keyBuffer(layer: layerIndex,
                                                     validTokenCount: startPosition + i + 1),
                                     v: kv.valueBuffer(layer: layerIndex,
                                                       validTokenCount: startPosition + i + 1),
                                     out: vAttnOut, outOffset: i * qDim * h,
                                     headDim: UInt32(headDim),
                                     numQHeads: UInt32(cfg.numHeads),
                                     numKVHeads: UInt32(numKV),
                                     seqLen: UInt32(startPosition + i + 1),
                                     scale: Float(cfg.attentionScale))
            }
        }
        elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: vAttnOut, gate: vGate,
                                         count: t * qDim)
        multix.encode(commandBuffer: cb,
                      weights: o.buffer, weightsOffset: Int(o.offset),
                      scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                      biases: o.buffer, biasesOffset: Int(o.biasOffset),
                      x: vAttnOut, y: vOut1, m: D, n: qDim, tokens: t)
    }

    private func encodeVerifyLinear(_ cb: MTLCommandBuffer,
                                    layer: Qwen38ForwardRunner.LayerTensors,
                                    layerIndex: Int,
                                    tokenCount t: Int) throws {
        guard let qkvW = layer.linQKV, let zW = layer.linZ,
              let aW = layer.linA, let bW = layer.linB,
              let outW = layer.linOut, let convW = layer.linConv,
              let aLog = layer.linALog, let dtBias = layer.linDtBias,
              let gatedNormW = layer.linNorm else {
            preconditionFailure("linear-attention layer without GDN tensors")
        }
        let la = cfg.linearAttention
        let D = cfg.hiddenSize
        let h = MemoryLayout<Float16>.stride
        guard let linearOrdinal = linearLayerIndex[layerIndex] else {
            preconditionFailure("linear layer without ordinal")
        }

        func mx(_ w: TensorView, y: MTLBuffer, m: Int) {
            multix.encode(commandBuffer: cb,
                          weights: w.buffer, weightsOffset: Int(w.offset),
                          scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                          biases: w.buffer, biasesOffset: Int(w.biasOffset),
                          x: vNormed, y: y, m: m, n: D, tokens: t)
        }
        // The decode step fuses these four into one dispatch
        // (`gdn_in_proj_gemv_simd`), documented bit-identical to the four
        // separate GEMVs; the multi-x rows match that per-row body.
        mx(qkvW, y: vGdnQKV, m: la.qkvDim)
        mx(zW, y: vGdnZ, m: la.valueDim)
        mx(aW, y: vGdnA, m: la.numVHeads)
        mx(bW, y: vGdnB, m: la.numVHeads)

        let tail = gdnState.convTailBuffer(layer: layerIndex)
        let state = gdnState.stateBuffer(layer: layerIndex)
        let stateBytes = gdnState.stateBytesPerLayer
        let tailBytes = gdnState.convTailBytesPerLayer
        for i in 0..<t {
            gdn.encodeConvDecode(commandBuffer: cb,
                                 tail: tail,
                                 qkv: vGdnQKV, qkvOffset: i * la.qkvDim * h,
                                 convWeight: convW.buffer,
                                 convWeightOffset: Int(convW.offset),
                                 out: vGdnConv, outOffset: i * la.qkvDim * h)
            if i < t - 1 {
                guard let blit = cb.makeBlitCommandEncoder() else {
                    throw Qwen38ForwardRunnerError.commandFailed(
                        "unable to create Qwen 3.8 MTP tail-capture blit encoder")
                }
                blit.copy(from: tail, sourceOffset: 0,
                          to: tailArena,
                          destinationOffset: (i * numLinearLayers + linearOrdinal) * tailBytes,
                          size: tailBytes)
                blit.endEncoding()
            }
        }
        gdn.encodeQKNorm(commandBuffer: cb, convOut: vGdnConv, rows: t)
        for i in 0..<t {
            let usedFused = gdn.encodeDeltaGatedDecode(
                commandBuffer: cb,
                convOut: vGdnConv, convOutOffset: i * la.qkvDim * h,
                aProj: vGdnA, aProjOffset: i * la.numVHeads * h,
                bProj: vGdnB, bProjOffset: i * la.numVHeads * h,
                aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                state: state,
                z: vGdnZ, zOffset: i * la.valueDim * h,
                weight: gatedNormW.buffer, weightOffset: Int(gatedNormW.offset),
                out: vGdnOut, outOffset: i * la.valueDim * h)
            if !usedFused {
                gdn.encodeDeltaStepDecode(commandBuffer: cb,
                                          convOut: vGdnConv,
                                          convOutOffset: i * la.qkvDim * h,
                                          aProj: vGdnA, aProjOffset: i * la.numVHeads * h,
                                          bProj: vGdnB, bProjOffset: i * la.numVHeads * h,
                                          aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                          dtBias: dtBias.buffer,
                                          dtBiasOffset: Int(dtBias.offset),
                                          state: state,
                                          y: vGdnY, yOffset: i * la.valueDim * h)
                gdn.encodeGatedNorm(commandBuffer: cb,
                                    y: vGdnY, yOffset: i * la.valueDim * h,
                                    z: vGdnZ, zOffset: i * la.valueDim * h,
                                    weight: gatedNormW.buffer,
                                    weightOffset: Int(gatedNormW.offset),
                                    out: vGdnOut, outOffset: i * la.valueDim * h)
            }
            if i < t - 1 {
                guard let blit = cb.makeBlitCommandEncoder() else {
                    throw Qwen38ForwardRunnerError.commandFailed(
                        "unable to create Qwen 3.8 MTP state-capture blit encoder")
                }
                blit.copy(from: state, sourceOffset: 0,
                          to: stateArena,
                          destinationOffset: (i * numLinearLayers + linearOrdinal) * stateBytes,
                          size: stateBytes)
                blit.endEncoding()
            }
        }
        multix.encode(commandBuffer: cb,
                      weights: outW.buffer, weightsOffset: Int(outW.offset),
                      scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                      biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                      x: vGdnOut, y: vOut1, m: D, n: la.valueDim, tokens: t)
    }

    private func encodeVerifyMLP(_ cb: MTLCommandBuffer,
                                 layer: Qwen38ForwardRunner.LayerTensors,
                                 tokenCount t: Int) throws {
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        if let int8Runtime = verifyMLPInt8 {
            // Toy fixtures carry an INT8 MLP: replay the decode kernel per
            // row (exact by construction; toy performance is irrelevant).
            let h = MemoryLayout<Float16>.stride
            for i in 0..<t {
                try int8Runtime.encode(commandBuffer: cb,
                                       x: vMlpX, xOffset: i * D * h,
                                       gate: layer.mlpGate, up: layer.mlpUp,
                                       down: layer.mlpDown,
                                       y: vMlpOut, yOffset: i * D * h,
                                       scratchGate: vMlpGate,
                                       scratchUp: vMlpUp,
                                       scratchAct: vMlpAct)
            }
            return
        }
        // INT4: the decode step's fused gate/up/act kernel preserves the
        // exact three-dispatch boundary (gate and up round through FP16, the
        // activation multiplies in FP32), so multi-x gate/up + the shared
        // silu-mul + multi-x down reproduce it bit for bit.
        func mx(_ proj: SharedExpertProjection, x: MTLBuffer, y: MTLBuffer,
                m: Int, n: Int) {
            multix.encode(commandBuffer: cb,
                          weights: proj.weights, weightsOffset: proj.weightsOffset,
                          scales: proj.scales, scalesOffset: proj.scalesOffset,
                          biases: proj.biases, biasesOffset: proj.biasesOffset,
                          x: x, y: y, m: m, n: n, tokens: t)
        }
        mx(layer.mlpGate, x: vMlpX, y: vMlpGate, m: F, n: D)
        mx(layer.mlpUp, x: vMlpX, y: vMlpUp, m: F, n: D)
        try encodeSiluMul(cb, gate: vMlpGate, up: vMlpUp, out: vMlpAct, count: t * F)
        mx(layer.mlpDown, x: vMlpAct, y: vMlpOut, m: D, n: F)
    }

    private func encodeSiluMul(_ cb: MTLCommandBuffer,
                               gate: MTLBuffer, up: MTLBuffer, out: MTLBuffer,
                               count: Int) throws {
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 MTP activation encoder")
        }
        encoder.setComputePipelineState(siluMulPSO)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(siluMulPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeGDNRestore(_ cb: MTLCommandBuffer, slot: Int) throws {
        precondition(slot >= 0 && slot < vT - 1, "GDN restore slot out of range")
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 MTP restore blit encoder")
        }
        let stateBytes = gdnState.stateBytesPerLayer
        let tailBytes = gdnState.convTailBytesPerLayer
        for layer in 0..<cfg.numLayers {
            guard let ordinal = linearLayerIndex[layer] else { continue }
            blit.copy(from: stateArena,
                      sourceOffset: (slot * numLinearLayers + ordinal) * stateBytes,
                      to: gdnState.stateBuffer(layer: layer),
                      destinationOffset: 0,
                      size: stateBytes)
            blit.copy(from: tailArena,
                      sourceOffset: (slot * numLinearLayers + ordinal) * tailBytes,
                      to: gdnState.convTailBuffer(layer: layer),
                      destinationOffset: 0,
                      size: tailBytes)
        }
        blit.endEncoding()
    }

    // MARK: - MTP draft layer

    /// One MTP forward: pair (`token`, `hPrev`) appended at `slot`, argmax of
    /// the shared lm_head over the drafter's output, and the post-norm hidden
    /// written to `hOut` for the next chained step.
    private func runDraftForward(token: Int32, hPrev: MTLBuffer,
                                 hOut: MTLBuffer, slot: Int) throws -> Int32 {
        precondition(slot >= 0 && slot < maxContext, "MTP slot out of range")
        let D = UInt32(cfg.hiddenSize)
        let cb = try commandBuffer()

        let emb = model.embedding
        embedInt4.encode(commandBuffer: cb,
                         table: emb.buffer, tableOffset: Int(emb.offset),
                         scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                         biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                         out: dEmb,
                         tokenId: UInt32(bitPattern: token),
                         d: D, outScale: 1.0)
        rms.encodeBF16W(commandBuffer: cb,
                        x: dEmb,
                        weight: weights.preFcNormEmbedding.buffer,
                        weightOffset: Int(weights.preFcNormEmbedding.offset),
                        out: dFcIn,
                        d: D, eps: Self.epsilon)
        rms.encodeBF16W(commandBuffer: cb,
                        x: hPrev,
                        weight: weights.preFcNormHidden.buffer,
                        weightOffset: Int(weights.preFcNormHidden.offset),
                        out: dFcIn, outOffset: cfg.hiddenSize * MemoryLayout<Float16>.stride,
                        d: D, eps: Self.epsilon)
        int4.encode(commandBuffer: cb,
                    weights: weights.fc.buffer, weightsOffset: Int(weights.fc.offset),
                    scales: weights.fc.buffer, scalesOffset: Int(weights.fc.scaleOffset),
                    biases: weights.fc.buffer, biasesOffset: Int(weights.fc.biasOffset),
                    x: dFcIn, y: dH,
                    m: D, n: 2 * D)
        try encodeMTPLayerSingle(cb, slot: slot)
        rms.encodeBF16W(commandBuffer: cb,
                        x: dH,
                        weight: weights.finalNorm.buffer,
                        weightOffset: Int(weights.finalNorm.offset),
                        out: hOut,
                        d: D, eps: Self.epsilon)
        let lm = model.lmHead
        draftHead.encodeGreedyDecode(commandBuffer: cb,
                                     hidden: dH,
                                     normWeight: weights.finalNorm.buffer,
                                     normOffset: Int(weights.finalNorm.offset),
                                     weights: lm.buffer, weightsOffset: Int(lm.offset),
                                     scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                     biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                     outToken: dTokenBuf,
                                     d: D, vocab: UInt32(cfg.vocabSize),
                                     rmsEps: Self.epsilon)
        try finish(cb)
        return Int32(bitPattern: dTokenBuf.contents().load(as: UInt32.self))
    }

    /// The MTP decoder layer over the single-token scratch: gated full
    /// attention against the drafter's own KV cache plus the dense SwiGLU
    /// MLP, with residuals — the target's full-attention block shape.
    private func encodeMTPLayerSingle(_ cb: MTLCommandBuffer, slot: Int) throws {
        let D = UInt32(cfg.hiddenSize)
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
        let h = MemoryLayout<Float16>.stride
        let slotOffset = slot * mtpKVDim * h

        rms.encodeBF16W(commandBuffer: cb,
                        x: dH,
                        weight: weights.inputNorm.buffer,
                        weightOffset: Int(weights.inputNorm.offset),
                        out: dNormed,
                        d: D, eps: Self.epsilon)
        func gemv(_ w: TensorView, x: MTLBuffer, xOffset: Int = 0,
                  y: MTLBuffer, yOffset: Int = 0, m: UInt32, n: UInt32) {
            int4.encode(commandBuffer: cb,
                        weights: w.buffer, weightsOffset: Int(w.offset),
                        scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                        biases: w.buffer, biasesOffset: Int(w.biasOffset),
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n)
        }
        gemv(weights.q, x: dNormed, y: dQPacked, m: 2 * qDim, n: D)
        gemv(weights.k, x: dNormed, y: mtpK, yOffset: slotOffset,
             m: UInt32(mtpKVDim), n: D)
        gemv(weights.v, x: dNormed, y: mtpV, yOffset: slotOffset,
             m: UInt32(mtpKVDim), n: D)
        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: dQPacked, q: dQ, gate: dGate,
                                     heads: cfg.numHeads, dim: headDim)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: dQ,
                               weight: weights.qNorm.buffer,
                               weightOffset: Int(weights.qNorm.offset),
                               out: dQ,
                               headDim: UInt32(headDim), numHeads: cfg.numHeads,
                               eps: Self.epsilon)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: mtpK, xOffset: slotOffset,
                               weight: weights.kNorm.buffer,
                               weightOffset: Int(weights.kNorm.offset),
                               out: mtpK, outOffset: slotOffset,
                               headDim: UInt32(headDim), numHeads: numKV,
                               eps: Self.epsilon)
        let ropePosition = UInt32(basePair + slot)
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: dQ,
                              position: ropePosition,
                              headDim: UInt32(headDim),
                              numHeads: UInt32(cfg.numHeads),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: mtpK, dataOffset: slotOffset,
                              position: ropePosition,
                              headDim: UInt32(headDim),
                              numHeads: UInt32(numKV),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        mtpAttention.encodeFull(commandBuffer: cb,
                                q: dQ,
                                k: mtpK, v: mtpV,
                                out: dAttnOut,
                                headDim: UInt32(headDim),
                                numQHeads: UInt32(cfg.numHeads),
                                numKVHeads: UInt32(numKV),
                                seqLen: UInt32(slot + 1),
                                scale: Float(cfg.attentionScale))
        elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: dAttnOut, gate: dGate,
                                         count: Int(qDim))
        gemv(weights.o, x: dAttnOut, y: dOut1, m: D, n: qDim)
        elementwise.encodeResidualAdd(commandBuffer: cb,
                                      hidden: dH, delta: dOut1,
                                      count: cfg.hiddenSize)
        rms.encodeBF16W(commandBuffer: cb,
                        x: dH,
                        weight: weights.postAttnNorm.buffer,
                        weightOffset: Int(weights.postAttnNorm.offset),
                        out: dMlpX,
                        d: D, eps: Self.epsilon)
        try mtpMLP.encode(commandBuffer: cb,
                          x: dMlpX,
                          gate: weights.mlpGate, up: weights.mlpUp,
                          down: weights.mlpDown,
                          y: dMlpOut,
                          scratchGate: dMlpGate, scratchUp: dMlpUp,
                          scratchAct: dMlpAct)
        elementwise.encodeResidualAdd(commandBuffer: cb,
                                      hidden: dH, delta: dMlpOut,
                                      count: cfg.hiddenSize)
    }

    // MARK: - MTP accept pass

    /// Trim the drafter cache to the committed pair count and re-feed the
    /// exact (token, target-hidden) pairs the round committed; the last row
    /// (the new bonus pair) yields the next round's seed token and hidden.
    /// Also restores the target's GDN state on partial acceptance and
    /// persists the accepted position's target hidden for the next round.
    private func runAcceptPass(bonus: Int32, emitted: [UInt32], position P: Int,
                               restoreSlot: Int?) throws {
        let accepted = emitted.count - 1
        let firstMissingPair = basePair + mtpLen
        let lastPair = P + accepted
        precondition(firstMissingPair >= P - 1 && firstMissingPair <= lastPair,
                     "MTP pair bookkeeping out of range")
        let R = lastPair - firstMissingPair + 1
        precondition(R >= 1 && R <= aR, "MTP accept batch \(R) outside 1...\(aR)")
        let D = cfg.hiddenSize
        let h = MemoryLayout<Float16>.stride

        // Pair j consumes (token at j+1, target hidden at j).
        var tokens: [Int32] = []
        tokens.reserveCapacity(R)
        for pair in firstMissingPair...lastPair {
            let tokenPosition = pair + 1
            if tokenPosition == P {
                tokens.append(bonus)
            } else {
                tokens.append(Int32(bitPattern: emitted[tokenPosition - P - 1]))
            }
        }
        let ids = tokenIdsBuf.contents().bindMemory(to: UInt32.self, capacity: R)
        for (i, token) in tokens.enumerated() { ids[i] = UInt32(bitPattern: token) }

        let cb = try commandBuffer()
        if let restoreSlot {
            try encodeGDNRestore(cb, slot: restoreSlot)
        }

        // Assemble the hidden rows: lastHiddenBuf for pair P-1, verify
        // normed-hidden rows for pairs P..P+accepted.
        guard let gather = cb.makeBlitCommandEncoder() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 MTP accept gather encoder")
        }
        for (row, pair) in (firstMissingPair...lastPair).enumerated() {
            if pair == P - 1 {
                gather.copy(from: lastHiddenBuf, sourceOffset: 0,
                            to: aHiddenSrc, destinationOffset: row * D * h,
                            size: D * h)
            } else {
                gather.copy(from: normedHidden, sourceOffset: (pair - P) * D * h,
                            to: aHiddenSrc, destinationOffset: row * D * h,
                            size: D * h)
            }
        }
        // Persist the accepted position's hidden for the next round before
        // the next verify pass overwrites `normedHidden`.
        gather.copy(from: normedHidden, sourceOffset: accepted * D * h,
                    to: lastHiddenBuf, destinationOffset: 0, size: D * h)
        gather.endEncoding()

        let emb = model.embedding
        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer, tableOffset: Int(emb.offset),
                            scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                            tokens: tokenIdsBuf,
                            out: aEmb,
                            t: UInt32(R), d: UInt32(D),
                            outScale: 1.0)
        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: aEmb,
                               weight: weights.preFcNormEmbedding.buffer,
                               weightOffset: Int(weights.preFcNormEmbedding.offset),
                               out: aEmbNorm,
                               t: UInt32(R), d: UInt32(D), eps: Self.epsilon)
        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: aHiddenSrc,
                               weight: weights.preFcNormHidden.buffer,
                               weightOffset: Int(weights.preFcNormHidden.offset),
                               out: aHidNorm,
                               t: UInt32(R), d: UInt32(D), eps: Self.epsilon)
        guard let interleave = cb.makeBlitCommandEncoder() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 MTP fc-interleave encoder")
        }
        for row in 0..<R {
            interleave.copy(from: aEmbNorm, sourceOffset: row * D * h,
                            to: aFcIn, destinationOffset: row * 2 * D * h,
                            size: D * h)
            interleave.copy(from: aHidNorm, sourceOffset: row * D * h,
                            to: aFcIn, destinationOffset: (row * 2 + 1) * D * h,
                            size: D * h)
        }
        interleave.endEncoding()

        multix.encode(commandBuffer: cb,
                      weights: weights.fc.buffer, weightsOffset: Int(weights.fc.offset),
                      scales: weights.fc.buffer, scalesOffset: Int(weights.fc.scaleOffset),
                      biases: weights.fc.buffer, biasesOffset: Int(weights.fc.biasOffset),
                      x: aFcIn, y: aH, m: D, n: 2 * D, tokens: R)
        try encodeMTPLayerRows(cb, rows: R, startSlot: mtpLen)

        // Seed for the next round: argmax + post-norm hidden of the last row.
        rms.encodeBF16W(commandBuffer: cb,
                        x: aH, xOffset: (R - 1) * D * h,
                        weight: weights.finalNorm.buffer,
                        weightOffset: Int(weights.finalNorm.offset),
                        out: seedHiddenBuf,
                        d: UInt32(D), eps: Self.epsilon)
        let lm = model.lmHead
        draftHead.encodeGreedyDecode(commandBuffer: cb,
                                     hidden: aH, hiddenOffset: (R - 1) * D * h,
                                     normWeight: weights.finalNorm.buffer,
                                     normOffset: Int(weights.finalNorm.offset),
                                     weights: lm.buffer, weightsOffset: Int(lm.offset),
                                     scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                     biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                     outToken: dTokenBuf,
                                     d: UInt32(D), vocab: UInt32(cfg.vocabSize),
                                     rmsEps: Self.epsilon)
        try finish(cb)
        seedToken = Int32(bitPattern: dTokenBuf.contents().load(as: UInt32.self))
        haveSeed = true
        mtpLen += R
    }

    /// The MTP decoder layer over `rows` accept rows in `aH`. Rows are
    /// independent up to attention (their fc inputs use exact target
    /// hiddens), and attend causally over the drafter cache.
    private func encodeMTPLayerRows(_ cb: MTLCommandBuffer, rows R: Int,
                                    startSlot: Int) throws {
        precondition(startSlot + R <= maxContext, "MTP cache overflow")
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = cfg.numHeads * headDim
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
        let h = MemoryLayout<Float16>.stride

        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: aH,
                               weight: weights.inputNorm.buffer,
                               weightOffset: Int(weights.inputNorm.offset),
                               out: aNormed,
                               t: UInt32(R), d: UInt32(D), eps: Self.epsilon)
        func mx(_ w: TensorView, x: MTLBuffer, y: MTLBuffer,
                yOffset: Int = 0, m: Int, n: Int) {
            multix.encode(commandBuffer: cb,
                          weights: w.buffer, weightsOffset: Int(w.offset),
                          scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                          biases: w.buffer, biasesOffset: Int(w.biasOffset),
                          x: x, y: y, yOffset: yOffset, m: m, n: n, tokens: R)
        }
        mx(weights.q, x: aNormed, y: aQPacked, m: 2 * qDim, n: D)
        // K/V rows land directly in the drafter cache: slots are contiguous
        // and the multi-x output stride equals the slot stride.
        mx(weights.k, x: aNormed, y: mtpK, yOffset: startSlot * mtpKVDim * h,
           m: mtpKVDim, n: D)
        mx(weights.v, x: aNormed, y: mtpV, yOffset: startSlot * mtpKVDim * h,
           m: mtpKVDim, n: D)
        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: aQPacked, q: aQ, gate: aGate,
                                     heads: cfg.numHeads, dim: headDim, rows: R)
        for i in 0..<R {
            let slot = startSlot + i
            let slotOffset = slot * mtpKVDim * h
            rms.encodeBF16WPerHead(commandBuffer: cb,
                                   x: aQ, xOffset: i * qDim * h,
                                   weight: weights.qNorm.buffer,
                                   weightOffset: Int(weights.qNorm.offset),
                                   out: aQ, outOffset: i * qDim * h,
                                   headDim: UInt32(headDim), numHeads: cfg.numHeads,
                                   eps: Self.epsilon)
            rms.encodeBF16WPerHead(commandBuffer: cb,
                                   x: mtpK, xOffset: slotOffset,
                                   weight: weights.kNorm.buffer,
                                   weightOffset: Int(weights.kNorm.offset),
                                   out: mtpK, outOffset: slotOffset,
                                   headDim: UInt32(headDim), numHeads: numKV,
                                   eps: Self.epsilon)
            rope.encodeNeoxSubdim(commandBuffer: cb,
                                  data: aQ, dataOffset: i * qDim * h,
                                  position: UInt32(basePair + slot),
                                  headDim: UInt32(headDim),
                                  numHeads: UInt32(cfg.numHeads),
                                  rotaryDim: rotaryDim,
                                  theta: Float(cfg.fullRopeTheta))
            rope.encodeNeoxSubdim(commandBuffer: cb,
                                  data: mtpK, dataOffset: slotOffset,
                                  position: UInt32(basePair + slot),
                                  headDim: UInt32(headDim),
                                  numHeads: UInt32(numKV),
                                  rotaryDim: rotaryDim,
                                  theta: Float(cfg.fullRopeTheta))
        }
        for i in 0..<R {
            mtpAttention.encodeFull(commandBuffer: cb,
                                    q: aQ, qOffset: i * qDim * h,
                                    k: mtpK, v: mtpV,
                                    out: aAttnOut, outOffset: i * qDim * h,
                                    headDim: UInt32(headDim),
                                    numQHeads: UInt32(cfg.numHeads),
                                    numKVHeads: UInt32(numKV),
                                    seqLen: UInt32(startSlot + i + 1),
                                    scale: Float(cfg.attentionScale))
        }
        elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: aAttnOut, gate: aGate,
                                         count: R * qDim)
        mx(weights.o, x: aAttnOut, y: aOut1, m: D, n: qDim)
        elementwise.encodeResidualAdd(commandBuffer: cb,
                                      hidden: aH, delta: aOut1, count: R * D)
        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: aH,
                               weight: weights.postAttnNorm.buffer,
                               weightOffset: Int(weights.postAttnNorm.offset),
                               out: aMlpX,
                               t: UInt32(R), d: UInt32(D), eps: Self.epsilon)
        mx(SharedExpertProjectionView(weights.mlpGate), x: aMlpX, y: aMlpGate, m: F, n: D)
        mx(SharedExpertProjectionView(weights.mlpUp), x: aMlpX, y: aMlpUp, m: F, n: D)
        try encodeSiluMul(cb, gate: aMlpGate, up: aMlpUp, out: aMlpAct, count: R * F)
        mx(SharedExpertProjectionView(weights.mlpDown), x: aMlpAct, y: aMlpOut, m: D, n: F)
        elementwise.encodeResidualAdd(commandBuffer: cb,
                                      hidden: aH, delta: aMlpOut, count: R * D)
    }

    // MARK: - Utilities

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 MTP command buffer")
        }
        return cb
    }

    private func finish(_ cb: MTLCommandBuffer) throws {
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw Qwen38ForwardRunnerError.commandFailed(
                cb.error?.localizedDescription
                    ?? "Qwen 3.8 MTP command buffer did not complete")
        }
    }
}

/// Adapter: a `SharedExpertProjection` viewed as a plain tensor for the
/// multi-x GEMV (weights + companions share the projection's buffer).
private func SharedExpertProjectionView(_ p: SharedExpertProjection) -> TensorView {
    TensorView(buffer: p.weights,
               offset: UInt64(p.weightsOffset),
               length: 0,
               scaleOffset: UInt64(p.scalesOffset),
               scaleLength: 0,
               biasOffset: UInt64(p.biasesOffset),
               biasLength: 0,
               shape: (p.rows, p.cols, 0, 0),
               dtype: 0)
}
