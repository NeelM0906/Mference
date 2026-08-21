import Foundation
import Metal

/// Faults detected at the Inkling output head. Both cases mean an
/// intermediate went non-finite somewhere in the 42-layer stack; the head is
/// simply the first place that can see it cheaply.
public enum InklingHeadError: Error, CustomStringConvertible, Equatable {
    case nonFiniteLogits(position: Int, count: Int)
    case noFiniteLogit(position: Int)

    public var description: String {
        switch self {
        case .nonFiniteLogits(let position, let count):
            return "Inkling head produced \(count) non-finite logits at "
                + "position \(position); an intermediate overflowed upstream"
        case .noFiniteLogit(let position):
            return "Inkling head produced no finite logit at position "
                + "\(position); greedy argmax has no candidate"
        }
    }
}

public enum RDAdvicePolicyMode: String, Codable, Sendable, Equatable {
    case `default`
    case off
    case bounded
    case adaptive

    public static func parse(_ raw: String?) -> RDAdvicePolicyMode {
        switch raw?.lowercased() {
        case "off", "none", "disabled":
            return .off
        case "bounded":
            return .bounded
        case "adaptive":
            return .adaptive
        default:
            return .default
        }
    }
}

public struct RDAdviceAdaptivePolicyConfig: Sendable, Equatable {
    public var missCap: Int
    public var byteCap: UInt64
    public var slowCallNanos: UInt64

    public init(missCap: Int,
                byteCap: UInt64,
                slowCallNanos: UInt64) {
        self.missCap = missCap
        self.byteCap = byteCap
        self.slowCallNanos = slowCallNanos
    }

    public static let conservative = RDAdviceAdaptivePolicyConfig(
        missCap: 12,
        byteCap: 384 * 1_048_576,
        slowCallNanos: 1_000_000)
}

struct RDAdviceAdaptivePolicyState: Sendable, Equatable {
    var config: RDAdviceAdaptivePolicyConfig
    private var skipUntilPosition: Int = -1
    private(set) var recentSlowCallNanos: UInt64 = 0

    init(config: RDAdviceAdaptivePolicyConfig = .conservative) {
        self.config = config
    }

    mutating func reset() {
        skipUntilPosition = -1
        recentSlowCallNanos = 0
    }

    func shouldSkip(position: Int,
                    requestedMisses: Int,
                    estimatedBytes: UInt64,
                    canOverlapUsefulGPUWork: Bool) -> Bool {
        position <= skipUntilPosition ||
        !canOverlapUsefulGPUWork ||
        requestedMisses > config.missCap ||
        estimatedBytes > config.byteCap
    }

    mutating func update(after result: ExpertIOAdviceResult,
                                position: Int) {
        recentSlowCallNanos = max(recentSlowCallNanos, result.maxCallNanos)
        if result.maxCallNanos >= config.slowCallNanos {
            skipUntilPosition = max(skipUntilPosition, position)
        }
    }
}

/// Gemma 4 real-forward decode pass.
///
/// Composes the production kernels against the `.gturbo` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<30:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = (SWA) v_proj(a) | (full) k_proj(a)
///     per-head q/k_norm (bf16w), per-head v_norm (no_scale)
///     NeoX RoPE on Q + K (default for SWA, proportional for full)
///     write K and V into separate cache slots
///     attn = attention(scale=1.0, SWA window or full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)
///     h1 = rmsnorm_bf16w(h1, post_feedforward_layernorm_1)
///     // router + routed branch
///     xr   = rmsnorm_no_scale(h)
///     idx, w = router_topk_gemma4(xr, effective_scale[L], per_expert_scale[L])
///     h2 = rmsnorm_bf16w(h, pre_feedforward_layernorm_2)
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=0, routedBlobs=fetch(idx), w)
///     h2 = rmsnorm_bf16w(h2, post_feedforward_layernorm_2)
///     h = h + rmsnorm_bf16w(h1 + h2, post_feedforward_layernorm)
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), embed_table^T)
///   // final softcap and softmax happen in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
    case shared
    case routed
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    static func selectedDispatch(for family: PrefillProjectionFamily,
                                 chunkTokens: Int) -> PrefillProjectionDispatch {
        guard chunkTokens >= 32 else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return .repeatedGEMV
        case .kv, .o, .shared, .routed:
            return .qmm
        }
    }
}

public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, FusedHeadLogitProducer, @unchecked Sendable {
    /// Per-layer fp32 short-convolution states, one buffer per conv site.
    /// k/v carry the last K-1 KV-stream inputs; attn/mlp the last K-1
    /// sublayer outputs.
    struct InklingLayerConvState {
        let k: MTLBuffer
        let v: MTLBuffer
        let attn: MTLBuffer
        let mlp: MTLBuffer
    }

    struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Gemma-only post_feedforward_layernorm_1; nil when the arch has no
        /// FFN sandwich norms.
        let postF1: TensorView?
        /// Qwen-only [1, hidden] scalar gate on the shared expert branch.
        let scalarGate: TensorView?
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue
    private let fusedPostAttentionSetup: FusedPostAttentionSetup
    private let fusedTail: FusedLayerTail

    // Qwen 3.6 kernels. Nil on architectures that never dispatch them.
    private let elementwise: Elementwise?
    private let gdn: GDN?
    private let gdnState: GDNStateManager?
    private let rope: RoPE?
    private let int8ScalarGate: DequantInt8GEMV?

    // DeepSeek-V4 kernels and state. Nil on architectures without CSA/HCA
    // layers.
    private let dsv4: DSV4Kernels?
    private let moeDSV4: MoEDeepseekV4?
    private let dsv4State: DSV4StateManager?

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillPostAttention: PrefillPostAttentionSetup
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillLayerTail: PrefillLayerTail
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    let hidden: MTLBuffer        // [D] FP16
    let normed: MTLBuffer        // [D] FP16
    let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    let kStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    let vStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    let oOut: MTLBuffer          // [D] FP16
    let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: MTLBuffer // [F=2112] FP16
    private let denseScratchUp: MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: MTLBuffer  // [F=2112] FP16
    private let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    let outIndices: MTLBuffer    // [topK] UInt32
    let outWeights: MTLBuffer    // [topK] FP16
    /// Lookahead router output: layer L+1's routing computed against layer
    /// L's pre-FFN normed state, read at the same router wake as
    /// `outIndices`. Feeds the speculative prefetcher; never bound by the
    /// real routed path.
    let pilotIndices: MTLBuffer  // [topK] UInt32
    /// S3 slot-map scratch: slab byte offsets for the routed experts and the
    /// GPU-written all-hit flag, both produced by `router_slot_lookup_k8`
    /// before the router signal so the CPU can trust them at wake.
    let slotMapOffsets: MTLBuffer // [topK] UInt32
    let slotMapAllHit: MTLBuffer  // [1] UInt32
    let pilotWeights: MTLBuffer  // [topK] FP16
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    let moeActs: MTLBuffer       // [topK * FmoE] FP16
    private let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    private let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    private let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    // Qwen 3.6 decode scratch (nil on architectures that never use it).
    private let qPackedScratch: MTLBuffer?   // [2 * N_HEADS * head_dim] packed [q ; gate]
    private let attnGateScratch: MTLBuffer?  // [N_HEADS * head_dim]
    private let gdnQKVRaw: MTLBuffer?        // [qkvDim] raw in_proj_qkv output
    private let gdnConvOut: MTLBuffer?       // [qkvDim] conv + SiLU output
    private let gdnZ: MTLBuffer?             // [valueDim]
    private let gdnA: MTLBuffer?             // [numVHeads]
    private let gdnB: MTLBuffer?             // [numVHeads]
    private let gdnY: MTLBuffer?             // [valueDim] delta-rule output
    private let gdnOut: MTLBuffer?           // [valueDim] gated-norm output
    private let sharedScalarGateBuf: MTLBuffer? // [1] shared-expert gate logit
    // DeepSeek-V4 decode scratch (nil on other architectures). The residual
    // streams ping-pong: `dsv4Streams` at layer entry/exit, `dsv4StreamsAlt`
    // between the attention and FFN sites.
    private let dsv4Streams: MTLBuffer?          // [mult * D] FP16
    private let dsv4StreamsAlt: MTLBuffer?       // [mult * D] FP16
    private let dsv4QA: MTLBuffer?               // [qLoraRank] FP16
    private let dsv4OGrouped: MTLBuffer?         // [oGroups * oLoraRank] FP16
    private let dsv4HCPreA: MTLBuffer?           // [mult] FP32, attention site
    private let dsv4HCPostA: MTLBuffer?          // [mult] FP32
    private let dsv4HCCombA: MTLBuffer?          // [mult * mult] FP32
    private let dsv4HCPreF: MTLBuffer?           // [mult] FP32, FFN site
    private let dsv4HCPostF: MTLBuffer?          // [mult] FP32
    private let dsv4HCCombF: MTLBuffer?          // [mult * mult] FP32
    private let dsv4IndexerQ: MTLBuffer?         // [indexNHeads * indexHeadDim]
    private let dsv4IndexerW: MTLBuffer?         // [indexNHeads] FP16
    private let dsv4IndexerScores: MTLBuffer?    // [maxContext / csaRate] FP32
    private let dsv4Selected: MTLBuffer?         // [indexTopK] UInt32

    // Inkling-Small kernels + scratch, keyed off the family so other
    // architectures pay no PSO compile or allocation cost. Conv states are
    // fp32 [C, K-1] per site (k/v on the KV streams, attn/mlp on the sublayer
    // outputs); `inklingSharedProjections[L]` holds one projection set per
    // shared expert (empty for the leading dense layers, which instead use
    // `inklingDenseProjections[L]`).
    let inkling: InklingKernels?
    let inklingPrefillGLU: InklingPrefillExpertGLU?
    let inklingConvStates: [InklingLayerConvState]
    let inklingRelScratch: MTLBuffer?        // [numHeads * dRel] FP16
    /// FP32 residual stream: Inkling's hidden state overflows FP16 by layer
    /// 23 (max channel 55k at L22, cap 65 504). Every consumer is an RMS
    /// norm, so the stream converts to FP16 only at norm outputs.
    let inklingHiddenF32: MTLBuffer?         // [D] FP32
    let inklingDeltaF32: MTLBuffer?          // [D] FP32 sublayer delta
    // Layer-major prefill chunk state (see prefillInklingChunk): per-token
    // FP32 hidden rows, per-token router outputs, and the expert-major FFN
    // accumulator. Sized to maxPrefillChunkTokens.
    let inklingHiddenChunk: MTLBuffer?       // [chunk, D] FP32
    let inklingRoutedXChunk: MTLBuffer?      // [chunk, D] FP16
    let inklingQChunk: MTLBuffer?            // [chunk, numHeads * headDim] FP16
    let inklingKChunk: MTLBuffer?            // [chunk, numKVHeads * headDim] FP16
    let inklingVChunk: MTLBuffer?            // [chunk, numKVHeads * headDim] FP16
    let inklingRelChunk: MTLBuffer?          // [chunk, numHeads * dRel] FP16
    let inklingAttentionChunk: MTLBuffer?    // [chunk, numHeads * headDim] FP16
    let inklingRouterLogitsChunk: MTLBuffer? // [chunk, routed + shared] FP32
    let inklingIdxChunk: MTLBuffer?          // [chunk, 8] u32
    let inklingWChunk: MTLBuffer?            // [chunk, 8] FP16
    let inklingGammaChunk: MTLBuffer?        // [chunk, 2] FP32
    let inklingAccChunk: MTLBuffer?          // [chunk, D] FP32
    let inklingChunkCapacity: Int
    /// Raw shared-expert down-projection rows, FP32. These are pre-gamma, so
    /// the 1/32 FFN prescale does not protect them and single channels run to
    /// ~6e4 on the released checkpoint — an FP16 store clips to infinity.
    let inklingSharedY0: MTLBuffer?          // [D] FP32
    let inklingSharedY1: MTLBuffer?          // [D] FP32
    /// Batched prefill expert application: the chunk's (token, expert) pairs
    /// grouped by expert, and one expert's activation tile.
    let inklingPairTokens: MTLBuffer?        // [chunk * 8] UInt32
    let inklingPairWeights: MTLBuffer?       // [chunk * 8] FP32
    let inklingExpertActScratch: MTLBuffer?  // [chunk, F] FP16
    let inklingSharedProjections: [[LayerSharedExpertProjections]]
    let inklingDenseProjections: [LayerSharedExpertProjections?]
    /// BF16 ones over [numExperts]; neutral per_expert_scale when the router
    /// has no auxiliary scale tensors.
    private let onesPerExpertScale: MTLBuffer?
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?
    /// Lazily built layer-major DeepSeek-V4 prefill; see
    /// `RealForwardRunner+DSV4Prefill.swift`.
    private var dsv4Prefill: DSV4ChunkedPrefill?

    private static let rdadviseBoundedMissCap = 12
    private static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    private static let rdadviseAdaptiveMissCap = 12
    private static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    private static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    private let useFusedGreedyHead: Bool
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    private var rdadviseSkipUntilPosition: Int = -1
    private var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    private var rdadviseAdaptivePosition: Int = -1
    private var rdadviseAdaptivePositionBytes: UInt64 = 0

    /// Router-readback synchronization. The layer's first command buffer
    /// signals this event immediately after the router top-k has written
    /// `outIndices`, and encodes the shared-expert FFN *after* the signal. The
    /// CPU wakes on the signal, plans slots and starts the routed-expert pread
    /// while that shared-expert work is still executing, so expert I/O overlaps
    /// GPU work instead of following it. Nil when the device cannot vend a
    /// shared event; the code then falls back to waiting on the whole buffer.
    private let routerEvent: MTLSharedEvent?
    /// Eager routed decode: the routed command buffer is committed before
    /// its expert fills land, gated on this event. Fill completions advance
    /// it in contiguous issue order, so an out-of-order completion can never
    /// unblock an earlier layer's command buffer.
    private let eagerFetchEvent: MTLSharedEvent?
    private var eagerFetchIssued: UInt64 = 0
    private var eagerFetchDone = Set<UInt64>()
    private var eagerFetchSignaled: UInt64 = 0
    private let eagerFetchLock = NSLock()
    /// First eager fill failure (layer index). The event still advances on
    /// failure — the gated command buffer is already committed and must not
    /// deadlock — so the decode loop checks this and aborts the step before
    /// its output is used.
    private var eagerFetchFailedLayer: Int?
    /// Accepted default (A/B 2026-08-08: +1.5–3.3% median on every case,
    /// byte-identical): the routed CB commits before its fills land, gated
    /// on the fill event. `MFERENCE_EAGER_ROUTED=0` disables.
    static let eagerRoutedDefault =
        ProcessInfo.processInfo.environment["MFERENCE_EAGER_ROUTED"] != "0"
    var eagerRoutedEnabled = RealForwardRunner.eagerRoutedDefault
    private var routerEventValue: UInt64 = 0
    private static let routerEventTimeoutMS: UInt64 = 60_000
    /// Phase probes cost a completion handler per command buffer, so they are
    /// only wired up when the phase report is going to be printed.
    private static let phaseInstrumentationEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_PHASES"] == "1"
    private static let routerEventWaitDefault =
        ProcessInfo.processInfo.environment["MFERENCE_ROUTER_EVENT"] != "0"
    /// Escape hatch and test seam for the early router readback. When false the
    /// CPU waits for the whole attention buffer instead of the mid-buffer
    /// signal, which serializes the shared expert ahead of the pread again.
    /// Both settings encode the same kernels in the same order, so decode
    /// output is identical either way. `MFERENCE_ROUTER_EVENT=0` flips the
    /// default for A/B benchmarking.
    var routerEventWaitEnabled = RealForwardRunner.routerEventWaitDefault

    /// Speculative cross-layer expert prefetch. The predictor is the *previous
    /// token's* routing for the same layer — free (the indices are already on
    /// the CPU), and unlike an LFU-top-k guess it predicts the experts that
    /// actually miss: LFU is already the eviction policy, so the resident slots
    /// are by construction the LFU favourites and a miss is by definition a
    /// non-favourite.
    enum SpeculativePrefetchMode: String {
        /// No speculation at all (default).
        case off
        /// Real preads into the next layer's slots, driven by the previous
        /// token's routing. Measured to be a no-op — see
        /// `previousTokenExperts`.
        case prefetch
        /// F_RDADVISE only: warms the page cache, writes no slots, needs no
        /// join. The low-risk fallback if the slot-write mode misbehaves.
        case advise
        /// PILOT: layer L+1's router run against layer L's post-attention
        /// state inside layer L's command buffer, feeding real preads. The
        /// only mode whose predictor knows anything about the current token.
        case pilot
        /// SHADOW: pilot's predictor with a non-blocking transport. Real
        /// plans never wait for an in-flight speculative read (the slot
        /// reservation system already keeps them safe to plan around), and
        /// each layer issues at most `shadowIssueBudget` of its top-ranked
        /// missing predictions, so speculation cannot flood the SSD or evict
        /// more than it earns. Response to the measured pilot failure: 73%
        /// recall but slower end-to-end from joins and unthrottled reads.
        case shadow

        static func parse(_ raw: String?) -> SpeculativePrefetchMode {
            switch raw?.lowercased() {
            case "1", "on", "prefetch": return .prefetch
            case "advise": return .advise
            case "pilot": return .pilot
            case "shadow": return .shadow
            default: return .off
            }
        }
    }

    /// Shadow mode: speculative reads issued per layer, taken from the front
    /// of the pilot's weight-ranked prediction list. Env-tunable while the
    /// knob is being characterized; the measured best becomes the constant.
    private static let shadowIssueBudget: Int = {
        if let raw = ProcessInfo.processInfo.environment["MFERENCE_SHADOW_BUDGET"],
           let parsed = Int(raw), parsed > 0 {
            return parsed
        }
        return 2
    }()
    var speculativePrefetchMode = SpeculativePrefetchMode.parse(
        ProcessInfo.processInfo.environment["MFERENCE_SPEC_PREFETCH"])
    /// S3 experiment gate: run all-hit layers' routed FFN entirely on-GPU
    /// via the slot map, skipping the per-layer CPU round-trip. Instance
    /// state so tests can flip it; the env variable sets the process
    /// default.
    /// Accepted Qwen production default (community A/B 2026-08-08: short
    /// +3.5%, medium +3.3%, long +1.6% median over four alternating blocks,
    /// byte-identical). `MFERENCE_SLOT_MAP=0` is the kill-switch.
    static let slotMapEnabledDefault =
        ProcessInfo.processInfo.environment["MFERENCE_SLOT_MAP"] != "0"
    /// Inkling prefill expert streaming: depth-1 pipeline — pread expert e+1
    /// while expert e's GLU runs, misses placed only in slots the in-flight
    /// command buffer does not touch. `MFERENCE_INKLING_PREFILL_PIPELINE=0`
    /// is the kill-switch back to the serialized fetch->encode->drain loop.
    static let inklingPrefillPipelineEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_INKLING_PREFILL_PIPELINE"] != "0"
    var slotMapEnabled = RealForwardRunner.slotMapEnabledDefault
    /// Debug discriminator: encode the whole slot-map chain but feed the
    /// lookup an all-empty table, so the guarded kernels always no-op and
    /// the CPU always falls back. Byte-identity under this mode proves the
    /// encoding itself is inert.
    static let slotMapForceMiss =
        ProcessInfo.processInfo.environment["MFERENCE_SLOT_MAP_FORCE_MISS"] == "1"
    /// Debug bisector: arm the slot map on exactly one layer index.
    static let slotMapOnlyLayer: Int? =
        ProcessInfo.processInfo.environment["MFERENCE_SLOT_MAP_LAYER"].flatMap(Int.init)
    /// Debug: run the guarded chain into shadow buffers, never skip, and
    /// compare against the fallback's activations after each layer.
    static let slotMapDebugCompare =
        ProcessInfo.processInfo.environment["MFERENCE_SLOT_MAP_DEBUG"] == "1"
    lazy var slotMapDebugActs: MTLBuffer? = ctx.device.makeBuffer(
        length: 8 * 4096 * MemoryLayout<Float16>.stride, options: .storageModeShared)
    lazy var slotMapDebugOut: MTLBuffer? = ctx.device.makeBuffer(
        length: 4096 * MemoryLayout<Float16>.stride, options: .storageModeShared)
    lazy var slotMapDebugHidden: MTLBuffer? = ctx.device.makeBuffer(
        length: 4096 * MemoryLayout<Float16>.stride, options: .storageModeShared)
    lazy var slotMapEmptyTable: MTLBuffer? = {
        guard let buf = ctx.device.makeBuffer(
            length: 256 * MemoryLayout<Int16>.stride,
            options: .storageModeShared) else { return nil }
        let ptr = buf.contents().bindMemory(to: Int16.self, capacity: 256)
        for i in 0..<256 { ptr[i] = -1 }
        return buf
    }()
    /// Last token's routed expert ids per layer; empty until a layer has been
    /// routed at least once. This is the default predictor.
    ///
    /// NOTE (measured): on its own this predictor can never issue a read. Each
    /// layer owns a private slot cache that only that layer's plans touch, so
    /// between token t and token t+1 nothing evicts layer L's slots — last
    /// token's experts are *always* still resident when we would predict them.
    /// "Same experts as last token" is a statement about the cache hit rate the
    /// LRU/LFU cache already delivers for free; a prefetch has to predict the
    /// complement (the experts that changed), about which last token's routing
    /// says nothing. Substituting a predictor with real information about the
    /// *current* token (PILOT router-lookahead) is the only way this pays, and
    /// `speculativeExpertPredictor` is where it plugs in.
    private var previousTokenExperts: [[Int]] = []
    private var pendingSpeculation: SpeculativeExpertPrefetch?
    /// Overrides the predictor for a layer. The seam a real predictor plugs
    /// into, and what lets tests drive the reserve/read/join/confirm machinery
    /// without depending on a fixture whose routing happens to churn the cache.
    var speculativeExpertPredictor: (@Sendable (Int) -> [Int])?
    /// PILOT lookahead kernels, built on first use so `off` pays nothing and
    /// the (concurrently edited) DSV4 init block stays untouched.
    private var pilotRouter: SpeculativeRouterDSV4?
    /// Inkling twin of `pilotRouter`: same lazy construction, same lifecycle,
    /// family-matched scoring kernels.
    private var inklingPilotRouter: SpeculativeRouterInkling?
    /// The prediction read out of `pilotRouter`/`inklingPilotRouter` (or the
    /// hash table) at the current layer's router wake, consumed by
    /// `issueSpeculativePrefetch`.
    private var pilotPrediction: (layer: Int, experts: [Int])?

    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.routerEvent = context.device.makeSharedEvent()
        self.eagerFetchEvent = context.device.makeSharedEvent()
        self.maxContext = maxContext
        // Shadow prefetch is the accepted DSV4 production default
        // (community A/B 2026-08-07: short +18%, long +13%, byte-identical;
        // docs/experiments/summaries/14-dsv4-shadow-prefetch.md). The env
        // variable still overrides in either direction; other families keep
        // `off` until they have their own accepted A/B.
        if ProcessInfo.processInfo.environment["MFERENCE_SPEC_PREFETCH"] == nil,
           model.config.family == .deepseekV4Flash {
            self.speculativePrefetchMode = .shadow
        }
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        self.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        self.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: useFP16Ring,
                                     slidingWindow: cfg.slidingWindow,
                                     // Sized from the CONFIGURED chunk so
                                     // sliding-window ring memory grows only
                                     // when a larger prefill chunk is opted
                                     // into, never from the static cap.
                                     maxPrefillChunkTokens: runtimeConfiguration.prefillConfig.chunkTokens)

        let silu = cfg.hiddenActivation == "silu"
        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(
            context: context,
            additionalShapes: cfg.decodeInt4GEMVShapes)
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits,
                                                  siluActivation: silu,
                                                  specializedD: cfg.hiddenSize,
                                                  specializedF: cfg.intermediateSize)
        self.moe       = try MoE(context: context,
                                 siluActivation: silu,
                                 specializedD: UInt32(cfg.hiddenSize),
                                 specializedF: UInt32(cfg.moeIntermediateSize),
                                 specializedNumExperts: UInt32(cfg.numExperts),
                                 specializedTopK: UInt32(cfg.topKExperts))
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.fusedPostAttentionSetup = try FusedPostAttentionSetup(context: context)
        self.fusedTail = try FusedLayerTail(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(context: context)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillPostAttention = try PrefillPostAttentionSetup(context: context)
        self.prefillRouter = try PrefillRouter(context: context)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.sharedExpertWeightBits,
            siluActivation: silu)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context,
                                                             siluActivation: silu)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context,
                                                               maxD: cfg.hiddenSize)

        // Qwen 3.6 kernels, keyed off the data flags so architectures that
        // never dispatch them pay no PSO compile cost.
        let needsElementwise = cfg.attnOutputGate
            || cfg.sharedExpertGated
            || !cfg.ffnSandwichNorms
            || cfg.hasLinearAttentionLayers
        self.elementwise = needsElementwise ? try Elementwise(context: context) : nil
        if cfg.hasLinearAttentionLayers {
            self.gdn = try GDN(context: context, config: cfg.linearAttention,
                               specializedHiddenSize: cfg.hiddenSize)
            self.gdnState = try GDNStateManager(device: context.device, config: cfg)
        } else {
            self.gdn = nil
            self.gdnState = nil
        }
        self.rope = cfg.ropeNeoxSubdim ? try RoPE(context: context) : nil
        self.int8ScalarGate = cfg.sharedExpertGated
            ? try DequantInt8GEMV(context: context,
                                  additionalShapes: cfg.decodeInt8GEMVShapes)
            : nil

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return b
        }
        self.hidden        = try buf(D)
        self.normed        = try buf(D)
        self.attnOut       = try buf(maxQ)
        self.qScratch      = try buf(maxQ)
        self.kStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.vStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.oOut          = try buf(D)
        self.h1Buf         = try buf(D)
        self.h2Buf         = try buf(D)
        self.routedX       = try buf(D)
        self.denseX        = try buf(D)
        // Inkling's two leading dense layers run through the shared-expert
        // kernel at denseIntermediateSize (16 384), so the scratch must cover
        // the wider of the two FFN widths.
        let scratchF = max(F, cfg.denseIntermediateSize)
        self.denseScratchGate = try buf(scratchF)
        self.denseScratchUp   = try buf(scratchF)
        self.denseScratchAct  = try buf(scratchF)
        self.routerInput   = try buf(D)
        self.zeroResidual  = try buf(D)
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        let paddedTopK = max(cfg.topKExperts, MoE.maxStreamedExperts)
        self.outIndices    = try buf(paddedTopK, MemoryLayout<UInt32>.size)
        self.outWeights    = try buf(paddedTopK)
        self.pilotIndices  = try buf(paddedTopK, MemoryLayout<UInt32>.size)
        self.slotMapOffsets = try buf(paddedTopK, MemoryLayout<UInt32>.size)
        self.slotMapAllHit = try buf(1, MemoryLayout<UInt32>.size)
        memset(self.slotMapAllHit.contents(), 0, self.slotMapAllHit.length)
        self.pilotWeights  = try buf(paddedTopK)
        self.moeActs       = try buf(paddedTopK * cfg.moeIntermediateSize)
        // Families that route fewer than 8 experts (Inkling: 6) pad the
        // streamed-expert list with duplicates carrying weight 0; zeroing the
        // weight buffer once makes those pad slots permanent no-ops.
        memset(self.outWeights.contents(), 0, self.outWeights.length)
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.greedyTokenBuf = tok

        // Qwen 3.6 decode scratch — allocated once here, never in the hot path.
        if cfg.attnOutputGate {
            self.qPackedScratch = try buf(2 * maxQ)
            self.attnGateScratch = try buf(maxQ)
        } else {
            self.qPackedScratch = nil
            self.attnGateScratch = nil
        }
        if cfg.hasLinearAttentionLayers {
            let la = cfg.linearAttention
            self.gdnQKVRaw = try buf(la.qkvDim)
            self.gdnConvOut = try buf(la.qkvDim)
            self.gdnZ = try buf(la.valueDim)
            self.gdnA = try buf(la.numVHeads)
            self.gdnB = try buf(la.numVHeads)
            self.gdnY = try buf(la.valueDim)
            self.gdnOut = try buf(la.valueDim)
        } else {
            self.gdnQKVRaw = nil
            self.gdnConvOut = nil
            self.gdnZ = nil
            self.gdnA = nil
            self.gdnB = nil
            self.gdnY = nil
            self.gdnOut = nil
        }
        self.sharedScalarGateBuf = cfg.sharedExpertGated ? try buf(1) : nil

        // DeepSeek-V4 kernels + scratch, keyed off the compressed-attention
        // flag so other architectures pay no PSO compile or allocation cost.
        if cfg.hasCompressedAttentionLayers {
            let ca = cfg.compressedAttention
            let mult = cfg.hyperConnections.mult
            self.dsv4 = try DSV4Kernels(context: context, config: cfg)
            self.moeDSV4 = try MoEDeepseekV4(
                context: context,
                specializedD: UInt32(cfg.hiddenSize),
                specializedF: UInt32(cfg.moeIntermediateSize),
                specializedNumExperts: UInt32(cfg.numExperts),
                routeScale: Float(cfg.routedScalingFactor),
                swigluLimit: Float(cfg.swigluLimit))
            self.dsv4State = try DSV4StateManager(device: device,
                                                  config: cfg,
                                                  maxContext: maxContext)
            self.dsv4Streams = try buf(mult * D)
            self.dsv4StreamsAlt = try buf(mult * D)
            self.dsv4QA = try buf(ca.qLoraRank)
            self.dsv4OGrouped = try buf(ca.oGroups * ca.oLoraRank)
            self.dsv4HCPreA = try buf(mult, MemoryLayout<Float>.size)
            self.dsv4HCPostA = try buf(mult, MemoryLayout<Float>.size)
            self.dsv4HCCombA = try buf(mult * mult, MemoryLayout<Float>.size)
            self.dsv4HCPreF = try buf(mult, MemoryLayout<Float>.size)
            self.dsv4HCPostF = try buf(mult, MemoryLayout<Float>.size)
            self.dsv4HCCombF = try buf(mult * mult, MemoryLayout<Float>.size)
            self.dsv4IndexerQ = try buf(ca.indexNHeads * ca.indexHeadDim)
            self.dsv4IndexerW = try buf(ca.indexNHeads)
            self.dsv4IndexerScores = try buf(
                max(1, maxContext / max(ca.csaCompressRate, 1)) + 1,
                MemoryLayout<Float>.size)
            self.dsv4Selected = try buf(ca.indexTopK, MemoryLayout<UInt32>.size)
        } else {
            self.dsv4 = nil
            self.moeDSV4 = nil
            self.dsv4State = nil
            self.dsv4Streams = nil
            self.dsv4StreamsAlt = nil
            self.dsv4QA = nil
            self.dsv4OGrouped = nil
            self.dsv4HCPreA = nil
            self.dsv4HCPostA = nil
            self.dsv4HCCombA = nil
            self.dsv4HCPreF = nil
            self.dsv4HCPostF = nil
            self.dsv4HCCombF = nil
            self.dsv4IndexerQ = nil
            self.dsv4IndexerW = nil
            self.dsv4IndexerScores = nil
            self.dsv4Selected = nil
        }

        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: rows,
                                 cols: cols)
        }
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<(cfg.family == .inklingSmall ? 0 : cfg.numLayers) {
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate, rows: UInt32(F), cols: UInt32(D)),
                up: sharedProj(up, rows: UInt32(F), cols: UInt32(D)),
                down: sharedProj(down, rows: UInt32(D), cols: UInt32(F)),
                postF1: cfg.ffnSandwichNorms ? try model.postFFN1(layer: L) : nil,
                scalarGate: cfg.sharedExpertGated
                    ? try model.sharedExpertScalarGate(layer: L) : nil))
        }
        self.sharedExpertProjections = sharedViews

        if cfg.family == .inklingSmall {
            self.inklingPrefillGLU = try InklingPrefillExpertGLU(context: context)
            self.inkling = try InklingKernels(context: context,
                                              numRouted: cfg.numExperts,
                                              numShared: cfg.numSharedExperts)
            let dRelWidth = cfg.relativePosition.projDim
            self.inklingRelScratch = try buf(dRelWidth)
            self.inklingHiddenF32 = try buf(D, MemoryLayout<Float>.size)
            self.inklingDeltaF32 = try buf(D, MemoryLayout<Float>.size)
            // Sized from the configured prefill chunk, not pinned at 128.
            // Routed-expert prefill I/O scales with the CHUNK COUNT: a chunk
            // re-reads every expert its tokens route to, and by ~128 tokens a
            // chunk already touches ~95% of all 256 experts per layer
            // (256 * (1 - (1 - 6/256)^128)), so the bytes read are essentially
            // `ceil(prompt / chunk) * 40 layers * 3.6 GB`. Doubling the chunk
            // halves the reads; the scratch below is linear in the cap and
            // costs ~1.3 MB per token of capacity (~168 MB at 4096).
            let chunkCap = max(1, min(runtimeConfiguration.prefillChunkTokens,
                                      RuntimeConfiguration.allowedPrefillChunkTokens.last!))
            self.inklingChunkCapacity = chunkCap
            self.inklingHiddenChunk = try buf(chunkCap * D, MemoryLayout<Float>.size)
            self.inklingRoutedXChunk = try buf(chunkCap * D)
            self.inklingQChunk = try buf(chunkCap * cfg.numHeads * cfg.headDim)
            self.inklingKChunk = try buf(chunkCap * cfg.numKVHeads * cfg.headDim)
            self.inklingVChunk = try buf(chunkCap * cfg.numKVHeads * cfg.headDim)
            self.inklingRelChunk = try buf(chunkCap * dRelWidth)
            self.inklingAttentionChunk = try buf(chunkCap * cfg.numHeads * cfg.headDim)
            self.inklingRouterLogitsChunk = try buf(
                chunkCap * (cfg.numExperts + cfg.numSharedExperts),
                MemoryLayout<Float>.size)
            self.inklingIdxChunk = try buf(chunkCap * 8, MemoryLayout<UInt32>.size)
            self.inklingWChunk = try buf(chunkCap * 8)
            self.inklingGammaChunk = try buf(chunkCap * 2, MemoryLayout<Float>.size)
            self.inklingAccChunk = try buf(chunkCap * D, MemoryLayout<Float>.size)
            self.inklingSharedY0 = try buf(D, MemoryLayout<Float>.size)
            self.inklingSharedY1 = try buf(D, MemoryLayout<Float>.size)
            // Pair lists cover top-6 routed + 2 shared per token.
            self.inklingPairTokens = try buf(chunkCap * 8, MemoryLayout<UInt32>.size)
            self.inklingPairWeights = try buf(chunkCap * 8, MemoryLayout<Float>.size)
            self.inklingExpertActScratch = try buf(chunkCap * F)
            var convStates: [InklingLayerConvState] = []
            convStates.reserveCapacity(cfg.numLayers)
            let kvDim = cfg.numKVHeads * cfg.headDim
            let km1 = cfg.sconvKernelSize - 1
            for _ in 0..<cfg.numLayers {
                convStates.append(InklingLayerConvState(
                    k: try buf(kvDim * km1, MemoryLayout<Float>.size),
                    v: try buf(kvDim * km1, MemoryLayout<Float>.size),
                    attn: try buf(D * km1, MemoryLayout<Float>.size),
                    mlp: try buf(D * km1, MemoryLayout<Float>.size)))
            }
            self.inklingConvStates = convStates
            var sharedPerLayer: [[LayerSharedExpertProjections]] = []
            var densePerLayer: [LayerSharedExpertProjections?] = []
            for L in 0..<cfg.numLayers {
                if L < cfg.numDenseLayers {
                    sharedPerLayer.append([])
                    let fd = cfg.denseIntermediateSize
                    densePerLayer.append(LayerSharedExpertProjections(
                        gate: sharedProj(try model.inklingDenseGate(layer: L),
                                         rows: UInt32(fd), cols: UInt32(D)),
                        up: sharedProj(try model.inklingDenseUp(layer: L),
                                       rows: UInt32(fd), cols: UInt32(D)),
                        down: sharedProj(try model.inklingDenseDown(layer: L),
                                         rows: UInt32(D), cols: UInt32(fd)),
                        postF1: nil, scalarGate: nil))
                    continue
                }
                densePerLayer.append(nil)
                var experts: [LayerSharedExpertProjections] = []
                for e in 0..<cfg.numSharedExperts {
                    experts.append(LayerSharedExpertProjections(
                        gate: sharedProj(try model.inklingSharedExpert(
                            "gate_proj", layer: L, expert: e,
                            of: cfg.numSharedExperts),
                            rows: UInt32(F), cols: UInt32(D)),
                        up: sharedProj(try model.inklingSharedExpert(
                            "up_proj", layer: L, expert: e,
                            of: cfg.numSharedExperts),
                            rows: UInt32(F), cols: UInt32(D)),
                        down: sharedProj(try model.inklingSharedExpert(
                            "down_proj", layer: L, expert: e,
                            of: cfg.numSharedExperts),
                            rows: UInt32(D), cols: UInt32(F)),
                        postF1: nil, scalarGate: nil))
                }
                sharedPerLayer.append(experts)
            }
            self.inklingSharedProjections = sharedPerLayer
            self.inklingDenseProjections = densePerLayer
        } else {
            self.inkling = nil
            self.inklingPrefillGLU = nil
            self.inklingConvStates = []
            self.inklingRelScratch = nil
            self.inklingHiddenF32 = nil
            self.inklingDeltaF32 = nil
            self.inklingHiddenChunk = nil
            self.inklingRoutedXChunk = nil
            self.inklingQChunk = nil
            self.inklingKChunk = nil
            self.inklingVChunk = nil
            self.inklingRelChunk = nil
            self.inklingAttentionChunk = nil
            self.inklingRouterLogitsChunk = nil
            self.inklingIdxChunk = nil
            self.inklingWChunk = nil
            self.inklingGammaChunk = nil
            self.inklingAccChunk = nil
            self.inklingChunkCapacity = 0
            self.inklingSharedY0 = nil
            self.inklingSharedY1 = nil
            self.inklingPairTokens = nil
            self.inklingPairWeights = nil
            self.inklingExpertActScratch = nil
            self.inklingSharedProjections = []
            self.inklingDenseProjections = []
        }

        func bf16OnesBuffer(count: Int, label: String) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: count * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = 0x3F80 }  // BF16 1.0
            buf.label = label
            return buf
        }

        if cfg.routerScaled {
            // Pre-fold 1/sqrt(D) into router.scale per layer. Each layer gets
            // its own BF16 [D] buffer — the kernel reads `effective_scale[i]`
            // and we pay for the multiply once per generation, not per token.
            var perLayer: [MTLBuffer] = []
            perLayer.reserveCapacity(cfg.numLayers)
            let invSqrtD = Float(1.0) / Float(D).squareRoot()
            let dInts = D
            for L in 0..<cfg.numLayers {
                let scaleView = try model.routerScale(layer: L)
                guard let buf = device.makeBuffer(length: dInts * MemoryLayout<UInt16>.size,
                                                  options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                let src = scaleView.buffer.contents()
                    .advanced(by: Int(scaleView.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<dInts {
                    let v = Quantization.bf16ToFloat(src[i]) * invSqrtD
                    dst[i] = Quantization.bf16Bits(v)
                }
                buf.label = "effective_scale.L\(L)"
                perLayer.append(buf)
            }
            self.effectiveScaleBuffers = perLayer
            self.onesPerExpertScale = nil
        } else {
            // Plain linear router (Qwen): one shared BF16 ones buffer keeps
            // the router kernel's effective_scale multiply neutral, and a ones
            // per_expert_scale keeps the top-k weights untouched. (Softmax
            // over top-k then renormalize equals Qwen's softmax over all
            // experts then renormalize the selected top-k.)
            let ones = try bf16OnesBuffer(count: D, label: "effective_scale.ones")
            self.effectiveScaleBuffers = [MTLBuffer](repeating: ones,
                                                     count: cfg.numLayers)
            self.onesPerExpertScale = try bf16OnesBuffer(count: cfg.numExperts,
                                                         label: "per_expert_scale.ones")
        }
    }

    public func reset() {
        kv?.reset()
        gdnState?.reset()
        dsv4State?.reset()
        // Short-conv history dies with the KV cache, and ONLY with it:
        // `prepareForContinuation` retains both, so a resumed conversation
        // keeps the last K-1 conv inputs that match the cached prefix.
        for state in inklingConvStates {
            memset(state.k.contents(), 0, state.k.length)
            memset(state.v.contents(), 0, state.v.length)
            memset(state.attn.contents(), 0, state.attn.length)
            memset(state.mlp.contents(), 0, state.mlp.length)
        }
        resetTransientState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        resetTransientState()
    }

    private func resetTransientState() {
        // Reset must drain fully even in shadow mode: a background read may
        // not write into slots after this point.
        if let pending = pendingSpeculation {
            pendingSpeculation = nil
            totalSpecPrefetchBytes &+= pending.join()
        }
        joinPendingSpeculation()
        previousTokenExperts = []
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
    }

    public private(set) var totalIoNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    /// Split of `totalIoNanos`: the part of each layer's expert pread that ran
    /// while GPU work was still in flight versus the part that ran with an idle
    /// GPU. Only populated when `MFERENCE_PHASES=1`.
    public private(set) var totalIoOverlappedNanos: UInt64 = 0
    /// S2 decision metric for the GPU-resident slot map: how many routed
    /// layer-steps needed no miss at all — those are the steps S3 can run
    /// entirely GPU-side.
    public private(set) var totalRoutedLayerSteps: UInt64 = 0
    public private(set) var totalAllHitLayerSteps: UInt64 = 0
    let gpuTimeLock = NSLock()
    public private(set) var totalGpuBusyNanos: UInt64 = 0
    var gpuSpanFirstStart: Double = .infinity
    var gpuSpanLastEnd: Double = 0
    public private(set) var totalIoExposedNanos: UInt64 = 0
    /// Speculative prefetch accounting: experts speculatively read (or advised)
    /// and how many of those the following real plan actually asked for. The
    /// ratio is the predictor's recall and is what makes an A/B interpretable.
    /// Experts named by the predictor. `confirmed / predicted` is realized
    /// recall — the number that decides whether PILOT is worth its GEMV.
    public private(set) var totalSpecPrefetchPredicted: UInt64 = 0
    public private(set) var totalSpecPrefetchIssued: UInt64 = 0
    public private(set) var totalSpecPrefetchConfirmed: UInt64 = 0
    public private(set) var totalSpecPrefetchBytes: UInt64 = 0

    /// Zeroes the per-phase counters. Prompt prefill runs through the same
    /// per-token code paths (DeepSeek-V4 prefill *is* a decode loop), so
    /// without a reset at the prefill/decode boundary the phase report prints
    /// prompt-time nanoseconds against a decode-only wall clock and the
    /// unaccounted remainder goes negative.
    public func beginDecodePhaseWindow() {
        totalIoNanos = 0
        totalCb1Nanos = 0
        totalCb2Nanos = 0
        totalHeadNanos = 0
        totalHeadFusedNanos = 0
        totalIoOverlappedNanos = 0
        totalRoutedLayerSteps = 0
        totalAllHitLayerSteps = 0
        gpuTimeLock.lock()
        totalGpuBusyNanos = 0
        gpuSpanFirstStart = .infinity
        gpuSpanLastEnd = 0
        gpuTimeLock.unlock()
        totalIoExposedNanos = 0
        totalSpecPrefetchPredicted = 0
        totalSpecPrefetchIssued = 0
        totalSpecPrefetchConfirmed = 0
        totalSpecPrefetchBytes = 0
    }

    /// One outstanding speculative read set. The runner keeps at most one alive
    /// (issued for layer L+1 while layer L finishes), and every real plan joins
    /// whatever is pending before it plans, so a speculative write and a real
    /// plan never touch the same streamer concurrently.
    private final class SpeculativeExpertPrefetch: @unchecked Sendable {
        let layer: Int
        /// Everything the predictor named — the denominator of recall. A subset
        /// of these is actually read (the non-resident ones).
        let predicted: Set<Int>
        private let finished = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var bytes: UInt64 = 0
        private var joined = false

        init(layer: Int, predicted: Set<Int>) {
            self.layer = layer
            self.predicted = predicted
        }

        func complete(bytes value: UInt64) {
            lock.lock()
            bytes = value
            lock.unlock()
            finished.signal()
        }

        /// Blocks until the background reads land. Idempotent.
        @discardableResult
        func join() -> UInt64 {
            lock.lock()
            let alreadyJoined = joined
            joined = true
            lock.unlock()
            if !alreadyJoined { finished.wait() }
            lock.lock()
            defer { lock.unlock() }
            return bytes
        }

        /// Non-blocking join: nil while the reads are still in flight.
        func tryJoin() -> UInt64? {
            lock.lock()
            if joined {
                defer { lock.unlock() }
                return bytes
            }
            lock.unlock()
            guard finished.wait(timeout: .now()) == .success else { return nil }
            lock.lock()
            defer { lock.unlock() }
            joined = true
            return bytes
        }
    }

    /// Drains any outstanding speculation. Called before every real plan (so no
    /// slot is replanned while a background read is filling it) and on reset.
    /// Returns the joined record so the caller can score its prediction.
    ///
    /// Shadow mode never blocks here: an unfinished read stays pending — the
    /// streamer's `speculativeInFlight` reservations already exclude its slots
    /// from hits and eviction, so planning around it is safe — and it is
    /// reaped by a later call once it has landed.
    @discardableResult
    private func joinPendingSpeculation() -> (layer: Int, predicted: Set<Int>)? {
        guard let pending = pendingSpeculation else { return nil }
        if speculativePrefetchMode == .shadow {
            guard let bytes = pending.tryJoin() else { return nil }
            pendingSpeculation = nil
            totalSpecPrefetchBytes &+= bytes
            return (pending.layer, pending.predicted)
        }
        pendingSpeculation = nil
        totalSpecPrefetchBytes &+= pending.join()
        return (pending.layer, pending.predicted)
    }

    /// Joins outstanding speculation and credits it against the experts the
    /// real plan turned out to need.
    ///
    /// Shadow mode joins **only when it pays**: if a predicted expert for
    /// this layer is actually needed, the in-flight read is nearly done and
    /// joining is cheaper than re-reading it on the demand path; otherwise
    /// the record is reaped lazily and the plan never waits.
    private func settleSpeculation(layer: Int, actualExperts: [Int]) {
        if speculativePrefetchMode == .shadow,
           let pending = pendingSpeculation,
           pending.layer == layer,
           actualExperts.contains(where: pending.predicted.contains) {
            pendingSpeculation = nil
            totalSpecPrefetchBytes &+= pending.join()
            var counted = Set<Int>()
            for expert in actualExperts
                where pending.predicted.contains(expert)
                    && counted.insert(expert).inserted {
                totalSpecPrefetchConfirmed &+= 1
            }
            return
        }
        guard let joined = joinPendingSpeculation(), joined.layer == layer else { return }
        var counted = Set<Int>()
        for expert in actualExperts
            where joined.predicted.contains(expert) && counted.insert(expert).inserted {
            totalSpecPrefetchConfirmed &+= 1
        }
    }

    /// Speculatively fetches the experts `layer` used on the previous token.
    /// Issued after the current layer's routed command buffer is committed —
    /// the window where the GPU is busy with routed work plus the next layer's
    /// attention and the CPU has nothing to do — so it never competes with the
    /// critical-path pread for SSD bandwidth.
    private func issueSpeculativePrefetch(layer: Int) {
        guard speculativePrefetchMode != .off,
              pendingSpeculation == nil,
              layer >= 0, layer < cfg.numLayers else { return }
        let predicted = predictedExperts(for: layer)
        guard !predicted.isEmpty else { return }

        // The record is created even when nothing needs reading, so recall is
        // still scored at full residency.
        let record = SpeculativeExpertPrefetch(layer: layer, predicted: Set(predicted))
        pendingSpeculation = record
        totalSpecPrefetchPredicted &+= UInt64(record.predicted.count)

        guard let streamer = try? model.routedExpertStreamer(layer: layer) else {
            record.complete(bytes: 0)
            return
        }
        var missing = streamer.nonResidentExperts(predicted)
        if speculativePrefetchMode == .shadow {
            // The prediction list is weight-ranked; a strict per-layer issue
            // budget keeps speculation from flooding the SSD or evicting
            // more slots than a right guess earns back.
            missing = Array(missing.prefix(Self.shadowIssueBudget))
        }
        guard !missing.isEmpty else {
            record.complete(bytes: 0)
            return
        }

        if speculativePrefetchMode == .advise {
            // Page-cache warming only: no slot writes, nothing to join.
            totalSpecPrefetchIssued &+= UInt64(missing.count)
            DispatchQueue.global(qos: .utility).async {
                record.complete(bytes: streamer.adviseExperts(experts: missing).bytes)
            }
            return
        }

        // Leave the next real plan room for a full top-k of its own misses, so
        // a wrong guess can never make it unplaceable.
        let reservation = streamer.reserveSpeculativeSlots(experts: missing,
                                                           keepEvictable: cfg.topKExperts)
        guard !reservation.isEmpty else {
            record.complete(bytes: 0)
            return
        }
        totalSpecPrefetchIssued &+= UInt64(reservation.count)
        // Shadow joins these reads on the critical path the moment a
        // prediction is confirmed, so they must not run at utility I/O
        // priority; utility stays correct for the fire-and-forget modes.
        let readQoS: DispatchQoS.QoSClass =
            speculativePrefetchMode == .shadow ? .userInitiated : .utility
        DispatchQueue.global(qos: readQoS).async {
            record.complete(bytes: streamer.executeSpeculativeReservation(reservation))
        }
    }

    private func predictedExperts(for layer: Int) -> [Int] {
        if let speculativeExpertPredictor { return speculativeExpertPredictor(layer) }
        switch speculativePrefetchMode {
        case .pilot, .shadow:
            guard let pilotPrediction, pilotPrediction.layer == layer else { return [] }
            return pilotPrediction.experts
        case .prefetch, .advise:
            guard layer < previousTokenExperts.count else { return [] }
            return previousTokenExperts[layer]
        case .off:
            return []
        }
    }

    /// Hash-routed layers select experts as a pure function of the token id, so
    /// the "prediction" for such a layer is exact and needs no GEMV at all.
    private func hashRoutedExperts(layer: Int, token: Int32) -> [Int]? {
        guard cfg.layerIsHashRouted(layer),
              let table = try? model.dsv4HashTable(layer: layer) else { return nil }
        let base = table.buffer.contents().advanced(by: Int(table.offset))
        let row = min(max(Int(token), 0), cfg.vocabSize - 1) * cfg.topKExperts
        let cap = cfg.numExperts - 1
        if table.dtype == 4 {
            let ptr = base.assumingMemoryBound(to: Int64.self)
            return (0..<cfg.topKExperts).map { min(max(0, Int(ptr[row + $0])), cap) }
        }
        let ptr = base.assumingMemoryBound(to: UInt32.self)
        return (0..<cfg.topKExperts).map { min(Int(ptr[row + $0]), cap) }
    }

    /// True when a lookahead GEMV should be encoded for `layer` — i.e. pilot is
    /// on, the layer exists, and its expert set is not already exactly known
    /// from the hash table.
    private func shouldEncodePilotGemv(nextLayer: Int) -> Bool {
        (speculativePrefetchMode == .pilot || speculativePrefetchMode == .shadow)
            && speculativeExpertPredictor == nil
            && nextLayer < cfg.numLayers
            && !cfg.layerIsHashRouted(nextLayer)
    }

    /// Reads the lookahead result at the router wake. Hash-routed next layers
    /// bypass the GEMV entirely (exact from the token id).
    private func capturePilotPrediction(nextLayer: Int, token: Int32, gemvEncoded: Bool) {
        pilotPrediction = nil
        guard speculativePrefetchMode == .pilot || speculativePrefetchMode == .shadow, nextLayer < cfg.numLayers else { return }
        if let hashed = hashRoutedExperts(layer: nextLayer, token: token) {
            pilotPrediction = (nextLayer, hashed)
            return
        }
        guard gemvEncoded, let buffer = pilotRouter?.predictedIndices else { return }
        let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
        let cap = cfg.numExperts - 1
        pilotPrediction = (nextLayer, (0..<cfg.topKExperts).map { min(Int(ptr[$0]), cap) })
    }

    private func ensurePilotRouter() -> SpeculativeRouterDSV4? {
        if let pilotRouter { return pilotRouter }
        pilotRouter = try? SpeculativeRouterDSV4(context: ctx,
                                                 numExperts: UInt32(cfg.numExperts),
                                                 d: UInt32(cfg.hiddenSize),
                                                 topK: UInt32(cfg.topKExperts))
        return pilotRouter
    }

    private func ensureInklingPilotRouter() -> SpeculativeRouterInkling? {
        if let inklingPilotRouter { return inklingPilotRouter }
        inklingPilotRouter = try? SpeculativeRouterInkling(
            context: ctx,
            numRouted: cfg.numExperts,
            numShared: cfg.numSharedExperts,
            topK: cfg.topKExperts)
        return inklingPilotRouter
    }

    /// Inkling wake-side twin of `capturePilotPrediction`: no hash-routed
    /// layers exist in this family, so the prediction comes only from the
    /// pilot GEMV.
    private func captureInklingPilotPrediction(nextLayer: Int, gemvEncoded: Bool) {
        pilotPrediction = nil
        guard speculativePrefetchMode == .pilot || speculativePrefetchMode == .shadow,
              nextLayer < cfg.numLayers, gemvEncoded,
              let buffer = inklingPilotRouter?.predictedIndices else { return }
        let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
        let cap = cfg.numExperts - 1
        pilotPrediction = (nextLayer, (0..<cfg.topKExperts).map { min(Int(ptr[$0]), cap) })
    }

    /// Records this layer's routing for the next token's predictor.
    private func recordRoutedExperts(_ experts: [Int], layer: Int) {
        if previousTokenExperts.count != cfg.numLayers {
            previousTokenExperts = [[Int]](repeating: [], count: cfg.numLayers)
        }
        previousTokenExperts[layer] = experts
    }

    /// Attributes one layer's expert-I/O window against the GPU work that was
    /// meant to hide it. `probe` tracks the command buffers committed before
    /// the pread started; a nil completion time means they were still running
    /// when the pread returned, i.e. the I/O was fully hidden.
    private func recordExpertIOOverlap(probe: GPUOverlapProbe?,
                                       startNanos: UInt64,
                                       endNanos: UInt64) {
        guard let probe, endNanos > startNanos else { return }
        let span = endNanos - startNanos
        guard let finished = probe.finishedNanos else {
            totalIoOverlappedNanos &+= span
            return
        }
        if finished <= startNanos {
            totalIoExposedNanos &+= span
        } else {
            totalIoOverlappedNanos &+= min(finished, endNanos) - startNanos
            totalIoExposedNanos &+= finished < endNanos ? endNanos - finished : 0
        }
    }
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    public private(set) var totalRDAdviseNanos: UInt64 = 0
    public private(set) var totalRDAdviseCalls: UInt64 = 0
    public private(set) var totalRDAdviseBytes: UInt64 = 0
    public private(set) var totalRDAdviseFailures: UInt64 = 0
    public private(set) var totalRDAdviseSkipped: UInt64 = 0

    private func recordRDAdvice(_ result: ExpertIOAdviceResult, wallNanos: UInt64) {
        totalRDAdviseNanos &+= wallNanos
        totalRDAdviseCalls &+= UInt64(result.calls)
        totalRDAdviseBytes &+= result.bytes
        totalRDAdviseFailures &+= UInt64(result.failed)
        totalRDAdviseSkipped &+= UInt64(result.skipped)
    }

    private func shouldSkipRDAdvice(position: Int,
                                    requestedMisses: Int,
                                    estimatedBytes: UInt64,
                                    canOverlapUsefulGPUWork: Bool) -> ExpertIOAdviceResult? {
        switch rdadvisePolicyMode {
        case .bounded:
            if position <= rdadviseSkipUntilPosition {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            if requestedMisses > Self.rdadviseBoundedMissCap {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            return nil
        case .adaptive:
            if position != rdadviseAdaptivePosition {
                rdadviseAdaptivePosition = position
                rdadviseAdaptivePositionBytes = 0
            }
            let cumulativeEstimatedBytes = rdadviseAdaptivePositionBytes &+ estimatedBytes
            let shouldSkip = rdadviseAdaptiveState.shouldSkip(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: cumulativeEstimatedBytes,
                canOverlapUsefulGPUWork: canOverlapUsefulGPUWork)
            rdadviseAdaptivePositionBytes = cumulativeEstimatedBytes
            guard shouldSkip else { return nil }
            return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                bytes: estimatedBytes)
        case .default, .off:
            return nil
        }
    }

    private func updateRDAdvicePolicy(after result: ExpertIOAdviceResult,
                                      position: Int) {
        switch rdadvisePolicyMode {
        case .bounded:
            if result.maxCallNanos > Self.rdadviseBoundedMaxCallNanos {
                rdadviseSkipUntilPosition = max(rdadviseSkipUntilPosition, position + 1)
            }
        case .adaptive:
            rdadviseAdaptiveState.update(after: result, position: position)
        case .default, .off:
            break
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        // Prompt prefill and decode share these counters; drop the prompt's
        // contribution on the way out so the phase report describes decode.
        defer { beginDecodePhaseWindow() }
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        if cfg.family == .inklingSmall {
            // Layer-major chunked prefill with expert-major streaming (each
            // unique routed expert read once per chunk-layer). The sequential
            // decode replay remains as the correctness reference behind
            // MFERENCE_INKLING_PREFILL=seq.
            let sequential = ProcessInfo.processInfo
                .environment["MFERENCE_INKLING_PREFILL"] == "seq"
            var pos = startPosition
            if sequential {
                var index = 0
                for t in tokens {
                    let isLast = index == tokens.count - 1
                    try await produceTokenInkling(token: t, position: pos,
                                                  into: logits,
                                                  emitHead: isLast,
                                                  outputMode: outputMode)
                    pos += 1
                    index += 1
                    if index % 16 == 0 { onProgress(index) }
                }
            } else {
                // Simple contiguous spans capped by the chunk buffers (the
                // KV ring also assumes writes at most maxPrefillChunkTokens
                // ahead of the cursor).
                let step = max(1, min(config.chunkTokens, inklingChunkCapacity))
                var offset = 0
                while offset < tokens.count {
                    let count = min(step, tokens.count - offset)
                    let lower = tokens.index(tokens.startIndex, offsetBy: offset)
                    let upper = tokens.index(lower, offsetBy: count)
                    prefillChunkState.markDirty(startPosition: pos,
                                                tokenCount: count)
                    try await prefillInklingChunk(
                        tokens: tokens[lower..<upper],
                        startPosition: pos,
                        emitHead: offset + count == tokens.count,
                        outputMode: outputMode,
                        into: logits)
                    prefillChunkState.markCommitted()
                    pos += count
                    offset += count
                    onProgress(offset)
                }
            }
            onProgress(tokens.count)
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                return PrefillResult(newPosition: pos,
                                     seed: .greedyToken(lastGreedyToken))
            }
            return PrefillResult(newPosition: pos, seed: .logitsWritten)
        }

        if cfg.hasCompressedAttentionLayers {
            // DeepSeek-V4 prefill runs layer-major over each chunk (see
            // `RealForwardRunner+DSV4Prefill.swift`), which amortizes the
            // routed-expert reads over the chunk instead of paying them per
            // token. Spans that the batched path cannot serve — currently only
            // those long enough to trigger lightning-indexer selection — fall
            // back to the token-by-token decode replay, which is the
            // correctness reference for both.
            let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                                  startPosition: startPosition,
                                                  config: config)
            let expertSlots = model.routedExpertCacheSlotCount(layer: 0)
            var pos = startPosition
            var done = 0
            for (spanIndex, span) in spans.enumerated() {
                let isLastSpan = spanIndex == spans.count - 1
                // A span crossing the lightning-selection cutover keeps its
                // eligible prefix batched; only the remainder replays
                // token-by-token.
                var batchedCount = DSV4ChunkedPrefill.batchedTokenPrefix(
                    config: cfg,
                    startPosition: span.startPosition,
                    tokenCount: span.tokenCount,
                    expertCacheSlots: expertSlots)
                if dsv4 == nil || moeDSV4 == nil || dsv4State == nil {
                    batchedCount = 0
                }
                if batchedCount > 0, let dsv4, let moeDSV4, let dsv4State {
                    let bindings = DSV4PrefillBindings(
                        model: model, ctx: ctx, cfg: cfg,
                        dsv4: dsv4, moeDSV4: moeDSV4, state: dsv4State,
                        int4: int4, rms: rms, embed: embedInt4,
                        fusionHead: fusionHead,
                        sharedExperts: sharedExpertProjections.map {
                            DSV4PrefillSharedExpert(gate: $0.gate, up: $0.up, down: $0.down)
                        },
                        effectiveScales: effectiveScaleBuffers,
                        useFusedGreedyHead: useFusedGreedyHead)
                    let runner: DSV4ChunkedPrefill
                    if let cached = dsv4Prefill, cached.chunkCapacity >= batchedCount {
                        runner = cached
                    } else {
                        runner = try DSV4ChunkedPrefill(
                            bindings: bindings,
                            chunkTokens: max(config.chunkTokens, batchedCount))
                        dsv4Prefill = runner
                    }
                    let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
                    let upper = tokens.index(lower, offsetBy: batchedCount)
                    // The batched runner advances DSV4 layer state as it
                    // encodes; if it throws partway the layer state is ahead
                    // of the KV cursor, so the runner must reject further use
                    // until reset() — same discipline as executePrefillChunk.
                    prefillChunkState.markDirty(startPosition: span.startPosition,
                                                tokenCount: batchedCount)
                    let greedy = try await runner.run(tokens: tokens[lower..<upper],
                                                      startPosition: span.startPosition,
                                                      emitHead: isLastSpan && batchedCount == span.tokenCount,
                                                      outputMode: outputMode,
                                                      logits: logits)
                    if let greedy { lastGreedyToken = greedy }
                    for _ in 0..<batchedCount { kv?.advance() }
                    pos += batchedCount
                    prefillChunkState.markCommitted()
                }
                if batchedCount < span.tokenCount {
                    let remainder = span.tokenCount - batchedCount
                    let lower = tokens.index(tokens.startIndex,
                                             offsetBy: span.tokenOffset + batchedCount)
                    let upper = tokens.index(lower, offsetBy: remainder)
                    var index = 0
                    for t in tokens[lower..<upper] {
                        let isLast = isLastSpan && index == remainder - 1
                        try await produceTokenDSV4(token: t, position: pos,
                                                   into: logits,
                                                   emitHead: isLast,
                                                   outputMode: outputMode)
                        pos += 1
                        index += 1
                    }
                }
                done += span.tokenCount
                onProgress(done)
            }
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                return PrefillResult(newPosition: pos,
                                     seed: .greedyToken(lastGreedyToken))
            }
            return PrefillResult(newPosition: pos, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        for (spanIndex, span) in spans.enumerated() {
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: config,
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

    @discardableResult
    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg, runtime: config)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires FP16 KV")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }

        struct LayerPrefillQKVViews {
            let inputNorm: TensorView
            let postAttention: TensorView
            let router: TensorView
            // Softmax-attention layers only (nil on linear-attention layers).
            let q: TensorView?
            let k: TensorView?
            let v: TensorView?
            let o: TensorView?
            let qNorm: TensorView?
            let kNorm: TensorView?
            // Gemma FFN sandwich only.
            let preFFN: TensorView?
            let preFFN2: TensorView?
            let postFFN2: TensorView?
            let postFFN: TensorView?
            let layerScalar: TensorView?
            let routerPerExpertScale: TensorView?
            // Gated-DeltaNet linear-attention layers only.
            let linQKV: TensorView?
            let linZ: TensorView?
            let linA: TensorView?
            let linB: TensorView?
            let linOut: TensorView?
            let linConv: TensorView?
            let linALog: TensorView?
            let linDtBias: TensorView?
            let linNorm: TensorView?
        }

        let layerViews = try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isLinear = cfg.layerIsLinear(L)
            let sandwich = cfg.ffnSandwichNorms
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                router: try model.router(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: isLinear ? nil : try model.kProj(layer: L),
                v: isLinear ? nil
                    : ((isFull && cfg.attentionKEqV)
                        ? (try model.kProj(layer: L))
                        : (try model.vProj(layer: L))),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: isLinear ? nil : try model.qNorm(layer: L),
                kNorm: isLinear ? nil : try model.kNorm(layer: L),
                preFFN: sandwich ? try model.preFFN(layer: L) : nil,
                preFFN2: sandwich ? try model.preFFN2(layer: L) : nil,
                postFFN2: sandwich ? try model.postFFN2(layer: L) : nil,
                postFFN: sandwich ? try model.postFFN(layer: L) : nil,
                layerScalar: sandwich ? try model.layerScalar(layer: L) : nil,
                routerPerExpertScale: cfg.routerScaled
                    ? try model.routerPerExpertScale(layer: L) : nil,
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: isLinear ? try model.linearInProjZ(layer: L) : nil,
                linA: isLinear ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear ? try model.linearOutProj(layer: L) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear ? try model.linearALog(layer: L) : nil,
                linDtBias: isLinear ? try model.linearDtBias(layer: L) : nil,
                linNorm: isLinear ? try model.linearNorm(layer: L) : nil)
        }

        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = ctx.device.makeBuffer(bytes: tokenIDs,
                                                      length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(D).squareRoot()
            : 1.0
        let t = tokens.count
        let emb = model.embedding

        func encodeInt4Projection(commandBuffer: MTLCommandBuffer,
                                  family: PrefillProjectionFamily,
                                  weights: TensorView,
                                  x: MTLBuffer,
                                  y: MTLBuffer,
                                  rows: Int,
                                  columns: Int,
                                  tokenCount: Int,
                                  xStrideElements: Int,
                                  yStrideElements: Int) {
            if tokenCount >= 32,
               family == .q || family == .kv || family == .o,
               let candidate = prefillMPPAffineInt4 {
                let path = candidate.encode(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: tokenCount,
                    n: rows,
                    k: columns)
                if path == .affineThreadgroupF16 {
                    return
                }
            }
            if PrefillProjectionDispatchPolicy.selectedDispatch(for: family,
                                                                chunkTokens: tokenCount) == .qmm {
                prefillQMM.encode(commandBuffer: commandBuffer,
                                  weights: weights.buffer,
                                  weightsOffset: Int(weights.offset),
                                  scales: weights.buffer,
                                  scalesOffset: Int(weights.scaleOffset),
                                  biases: weights.buffer,
                                  biasesOffset: Int(weights.biasOffset),
                                  x: x,
                                  y: y,
                                  t: tokenCount,
                                  n: rows,
                                  k: columns)
                return
            }
            for row in 0..<tokenCount {
                int4.encode(commandBuffer: commandBuffer,
                            weights: weights.buffer,
                            weightsOffset: Int(weights.offset),
                            scales: weights.buffer,
                            scalesOffset: Int(weights.scaleOffset),
                            biases: weights.buffer,
                            biasesOffset: Int(weights.biasOffset),
                            x: x,
                            xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                            y: y,
                            yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                            m: UInt32(rows),
                            n: UInt32(columns))
            }
        }

        func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                           source: MTLBuffer,
                           destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                           sourceTokenOffset: Int,
                           tokenCount: Int,
                           bytesPerToken: Int) throws {
            guard tokenCount > 0 else { return }
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(from: source,
                      sourceOffset: sourceTokenOffset * bytesPerToken,
                      to: destination.buffer,
                      destinationOffset: destination.offset,
                      size: tokenCount * bytesPerToken)
            blit.endEncoding()
        }

        func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                                  kv: KVCacheManager,
                                  layer: Int,
                                  startPosition: Int,
                                  tokenCount: Int,
                                  keySource: MTLBuffer,
                                  valueSource: MTLBuffer,
                                  bytesPerToken: Int) throws {
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
            let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keyFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            guard firstSpan < tokenCount else { return }

            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
            let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keySecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueSecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
        }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer,
                            tableOffset: Int(emb.offset),
                            scales: emb.buffer,
                            scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer,
                            biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: scratch.hidden,
                            t: UInt32(t),
                            d: UInt32(D),
                            outScale: embedOutScale)

        for L in 0..<cfg.numLayers {
            model.beginOpeningRoutedExpertStreamer(layer: L)
            let views = layerViews[L]
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.inputNorm.buffer,
                                   weightOffset: Int(views.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            if isLinear {
                // Gated-DeltaNet linear attention over the chunk: batched
                // projections, causal conv (+ tail carry), delta-rule
                // recurrence, gated norm, out_proj. No KV writes, no
                // attention, no blit.
                guard let gdn, let gdnState else {
                    preconditionFailure("linear-attention layer without GDN kernels")
                }
                let la = cfg.linearAttention
                encodeInt4Projection(commandBuffer: cb,
                                     family: .q,
                                     weights: views.linQKV!,
                                     x: scratch.normed,
                                     y: scratch.q,
                                     rows: la.qkvDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.qkvDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linZ!,
                                     x: scratch.normed,
                                     y: scratch.gdnZ,
                                     rows: la.valueDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.valueDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linA!,
                                     x: scratch.normed,
                                     y: scratch.gdnA,
                                     rows: la.numVHeads,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.numVHeads)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linB!,
                                     x: scratch.normed,
                                     y: scratch.gdnB,
                                     rows: la.numVHeads,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.numVHeads)
                let convW = views.linConv!
                let tail = gdnState.convTailBuffer(layer: L)
                gdn.encodeConvPrefill(commandBuffer: cb,
                                      tail: tail,
                                      qkvRows: scratch.q,
                                      convWeight: convW.buffer,
                                      convWeightOffset: Int(convW.offset),
                                      out: scratch.gdnConvOut,
                                      rows: t)
                gdn.encodeConvTailUpdate(commandBuffer: cb,
                                         tail: tail,
                                         qkvRows: scratch.q,
                                         rows: t)
                gdn.encodeQKNorm(commandBuffer: cb,
                                 convOut: scratch.gdnConvOut,
                                 rows: t)
                let aLog = views.linALog!
                let dtBias = views.linDtBias!
                gdn.encodeDeltaStepPrefill(commandBuffer: cb,
                                           convOut: scratch.gdnConvOut,
                                           aProj: scratch.gdnA,
                                           bProj: scratch.gdnB,
                                           aLog: aLog.buffer,
                                           aLogOffset: Int(aLog.offset),
                                           dtBias: dtBias.buffer,
                                           dtBiasOffset: Int(dtBias.offset),
                                           state: gdnState.stateBuffer(layer: L),
                                           y: scratch.gdnY,
                                           rows: t)
                let gatedNormW = views.linNorm!
                gdn.encodeGatedNorm(commandBuffer: cb,
                                    y: scratch.gdnY,
                                    z: scratch.gdnZ,
                                    weight: gatedNormW.buffer,
                                    weightOffset: Int(gatedNormW.offset),
                                    out: scratch.attentionOutput,
                                    rows: t)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .o,
                                     weights: views.linOut!,
                                     x: scratch.attentionOutput,
                                     y: scratch.h1,
                                     rows: D,
                                     columns: la.valueDim,
                                     tokenCount: t,
                                     xStrideElements: la.valueDim,
                                     yStrideElements: D)
            } else {
                let qProjRows = cfg.attnOutputGate ? 2 * qDim : qDim
                encodeInt4Projection(commandBuffer: cb,
                                     family: .q,
                                     weights: views.q!,
                                     x: scratch.normed,
                                     y: scratch.q,
                                     rows: qProjRows,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: qProjRows)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.k!,
                                     x: scratch.normed,
                                     y: scratch.kStage,
                                     rows: kvDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: kvDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.v!,
                                     x: scratch.normed,
                                     y: scratch.vStage,
                                     rows: kvDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: kvDim)

                // The attention input Q: the packed q_proj output is split
                // into per-head query/gate halves for gated architectures.
                let attnQ: MTLBuffer
                if cfg.attnOutputGate {
                    elementwise!.encodeSplitQGate(commandBuffer: cb,
                                                  packed: scratch.q,
                                                  q: scratch.attnQ,
                                                  gate: scratch.attnGate,
                                                  heads: cfg.numHeads,
                                                  dim: headDim,
                                                  rows: t)
                    attnQ = scratch.attnQ
                } else {
                    attnQ = scratch.q
                }

                if cfg.ropeNeoxSubdim {
                    let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
                    prefillQKVEpilogue.encodeNeoxSubdimNoVNorm(
                        commandBuffer: cb,
                        q: attnQ,
                        k: scratch.kStage,
                        qWeight: views.qNorm!.buffer,
                        qWeightOffset: Int(views.qNorm!.offset),
                        kWeight: views.kNorm!.buffer,
                        kWeightOffset: Int(views.kNorm!.offset),
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKVHeads),
                        qTokenStrideElements: UInt32(qDim),
                        kvTokenStrideElements: UInt32(kvDim),
                        theta: Float(cfg.fullRopeTheta),
                        rotaryDim: rotaryDim,
                        eps: eps)
                } else {
                    let rotatedPairs = isFull
                        ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                        : UInt32(headDim / 2)
                    prefillQKVEpilogue.encode(commandBuffer: cb,
                                               q: attnQ,
                                               k: scratch.kStage,
                                               v: scratch.vStage,
                                               qWeight: views.qNorm!.buffer,
                                               qWeightOffset: Int(views.qNorm!.offset),
                                               kWeight: views.kNorm!.buffer,
                                               kWeightOffset: Int(views.kNorm!.offset),
                                               startPosition: UInt32(startPosition),
                                               queryCount: UInt32(t),
                                               headDim: UInt32(headDim),
                                               numQHeads: UInt32(cfg.numHeads),
                                               numKVHeads: UInt32(numKVHeads),
                                               qTokenStrideElements: UInt32(qDim),
                                               kvTokenStrideElements: UInt32(kvDim),
                                               theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                               rotatedPairs: rotatedPairs,
                                               eps: eps)
                }

                if let kv {
                    let bytes = t * kvDim * MemoryLayout<Float16>.stride
                    try copyPrefillKVToCache(commandBuffer: cb,
                                             kv: kv,
                                             layer: L,
                                             startPosition: startPosition,
                                             tokenCount: t,
                                             keySource: scratch.kStage,
                                             valueSource: scratch.vStage,
                                             bytesPerToken: bytes / t)
                }
                let params = PrefillAttentionParams(
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKVHeads),
                        kvValidCount: UInt32(startPosition + t),
                        slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                        kvTokenStrideElements: UInt32(kvDim),
                        qTokenStrideElements: UInt32(qDim),
                        oTokenStrideElements: UInt32(qDim),
                        scale: Float(cfg.attentionScale))
                if let kv {
                        let keyBuffer = kv.keyBuffer(layer: L, validTokenCount: startPosition + t)
                        let valueBuffer = kv.valueBuffer(layer: L, validTokenCount: startPosition + t)
                        let ringCapacity = kv.ringCapacity(layer: L)
                        let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                            ? UInt32(ringCapacity)
                            : 0
                        prefillAttention.encodeCausal(commandBuffer: cb,
                                                      q: attnQ,
                                                      k: keyBuffer,
                                                      v: valueBuffer,
                                                      out: scratch.attentionOutput,
                                                      params: params,
                                                      kvRingCapacity: activeRingCapacity,
                                                      path: prefillAttentionPath)
                } else {
                    throw PrefillError.chunkedUnsupported(
                        "chunked prefill attention requires FP16 KV")
                }
                if cfg.attnOutputGate {
                    elementwise!.encodeSigmoidGateMul(commandBuffer: cb,
                                                      out: scratch.attentionOutput,
                                                      gate: scratch.attnGate,
                                                      count: t * qDim)
                }
                encodeInt4Projection(commandBuffer: cb,
                                         family: .o,
                                         weights: views.o!,
                                         x: scratch.attentionOutput,
                                         y: scratch.h1,
                                         rows: D,
                                         columns: qDim,
                                         tokenCount: t,
                                         xStrideElements: qDim,
                                         yStrideElements: D)
            }
            if cfg.ffnSandwichNorms {
                prefillPostAttention.encode(commandBuffer: cb,
                                                hidden: scratch.hidden,
                                                attn: scratch.h1,
                                                denseX: scratch.denseX,
                                                routedX: scratch.routedX,
                                                routerX: scratch.routerX,
                                                postAttentionWeight: views.postAttention.buffer,
                                                postAttentionWeightOffset: Int(views.postAttention.offset),
                                                preFFNWeight: views.preFFN!.buffer,
                                                preFFNWeightOffset: Int(views.preFFN!.offset),
                                                preFFN2Weight: views.preFFN2!.buffer,
                                                preFFN2WeightOffset: Int(views.preFFN2!.offset),
                                                queryCount: UInt32(t),
                                                d: UInt32(D),
                                                hiddenStrideElements: UInt32(D),
                                                attnStrideElements: UInt32(D),
                                                denseStrideElements: UInt32(D),
                                                routedStrideElements: UInt32(D),
                                                routerStrideElements: UInt32(D),
                                                eps: eps)
            } else {
                // Plain pre-norm residual block: hidden += attention branch,
                // then one post-attention norm feeds router, shared expert,
                // and routed phase 1 (routedX doubles as moeX).
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: scratch.hidden,
                                               delta: scratch.h1,
                                               count: t * D)
                prefillRMS.encodeBF16W(commandBuffer: cb,
                                       x: scratch.hidden,
                                       weight: views.postAttention.buffer,
                                       weightOffset: Int(views.postAttention.offset),
                                       out: scratch.routedX,
                                       t: UInt32(t),
                                       d: UInt32(D),
                                       eps: eps)
            }
            let perExpertScale: (buffer: MTLBuffer, offset: Int)
            if cfg.routerScaled {
                let view = views.routerPerExpertScale!
                perExpertScale = (view.buffer, Int(view.offset))
            } else {
                perExpertScale = (onesPerExpertScale!, 0)
            }
            prefillRouter.encodeGemma4Block(
                        commandBuffer: cb,
                        weights: views.router.buffer,
                        weightsOffset: Int(views.router.offset),
                        scales: views.router.buffer,
                        scalesOffset: Int(views.router.scaleOffset),
                        biases: views.router.buffer,
                        biasesOffset: Int(views.router.biasOffset),
                        hidden: cfg.ffnSandwichNorms ? scratch.routerX : scratch.routedX,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: perExpertScale.offset,
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))

                    cb.commit()
                    waitForCompletion(cb)
                    if let error = cb.error {
                        throw error
                    }

                    // The shared branch depends only on routedX, which the
                    // completed attention/router command has already
                    // produced. Submit it before CPU route grouping and SSD
                    // binding so those tasks overlap its batched projections.
                    guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    let sharedProj = sharedExpertProjections[L]
                    try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                        x: cfg.ffnSandwichNorms
                                                            ? scratch.denseX
                                                            : scratch.routedX,
                                                        y: scratch.h1,
                                                        gate: sharedProj.gate,
                                                        up: sharedProj.up,
                                                        down: sharedProj.down,
                                                        scratchGate: scratch.sharedGateScratch,
                                                        scratchUp: scratch.sharedUpScratch,
                                                        scratchAct: scratch.sharedActScratch,
                                                        queryCount: t,
                                                        d: D,
                                                        intermediate: cfg.intermediateSize,
                                                        xStrideElements: D,
                                                        yStrideElements: D)
                    if cfg.ffnSandwichNorms {
                        let postF1 = sharedProj.postF1!
                        prefillRMS.encodeBF16W(commandBuffer: sharedCB,
                                               x: scratch.h1,
                                               weight: postF1.buffer,
                                               weightOffset: Int(postF1.offset),
                                               out: scratch.h1,
                                               t: UInt32(t),
                                               d: UInt32(D),
                                               eps: eps)
                    } else if cfg.sharedExpertGated {
                        let gateView = sharedProj.scalarGate!
                        let scalarGate = SharedExpertInt8Proj(
                            weights: gateView.buffer,
                            scales: gateView.buffer,
                            biases: gateView.buffer,
                            weightsOffset: Int(gateView.offset),
                            scalesOffset: Int(gateView.scaleOffset),
                            biasesOffset: Int(gateView.biasOffset),
                            rows: 1,
                            cols: UInt32(D))
                        try prefillSharedExpert.encodeQwenScalarGate(
                            commandBuffer: sharedCB,
                            x: scratch.routedX,
                            gate: scalarGate,
                            y: scratch.h1,
                            queryCount: t,
                            d: D,
                            xStrideElements: D,
                            yStrideElements: D)
                    }
                    sharedCB.commit()

                    let routeCount = t * cfg.topKExperts
                    let idPtr = scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    var routeIDs = [UInt32]()
                    routeIDs.reserveCapacity(routeCount)
                    var routeWeights = [Float16]()
                    routeWeights.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        routeWeights.append(weightPtr[i])
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                                   weights: routeWeights,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                    let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                    let routeTileExpertCount: Int
                    if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
                        guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                            throw PrefillError.chunkedUnsupported(
                                "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
                        }
                        routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
                    } else {
                        routeTileExpertCount = schedulerConfig.tileExperts
                    }
                    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: routeTileExpertCount,
                        expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

                    let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    let routedOffsets = model.routedExpertOffsets(layer: L)
                    struct PendingPrefillTile {
                        let tileIndex: Int
                        let commandBuffer: MTLCommandBuffer
                        let fetch: PrefillStreamedTileFetchResult
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                    }
                    var pendingTiles: [PendingPrefillTile] = []
                    var tileLifetime = PrefillStreamedTileSlotLifetime()
                    func drainOldestPendingTile() throws {
                        guard !pendingTiles.isEmpty else { return }
                        let pending = pendingTiles.removeFirst()
                        withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                            waitForCompletion(pending.commandBuffer)
                        }
                        if let error = pending.commandBuffer.error {
                            throw error
                        }
                        if !pending.fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                    for (tileIndex, tile) in routes.tiles.enumerated() {
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                        var plannedFetch: RoutedExpertFetchPlan?
                        if !pendingTiles.isEmpty {
                            let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                            if !pendingAssignedSlots.isEmpty {
                                let pendingSlots = Set(pendingAssignedSlots)
                                let plan = try model.planRoutedExpertsIfPossible(
                                    layer: L,
                                    experts: expertIDs,
                                    avoidingSlots: pendingSlots)
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: pendingAssignedSlots,
                                        avoidingSlotPlanAvailable: plan != nil))
                                switch decision {
                                case .prefetchNext:
                                    guard let plan else {
                                        throw ModelError.indexCorrupt(
                                            detail: "routed tile scheduler requested missing plan")
                                    }
                                    plannedFetch = plan
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler ignored pending tile")
                                }
                            } else {
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: [],
                                        avoidingSlotPlanAvailable: false))
                                switch decision {
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending, .prefetchNext:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler failed to drain empty-slot pending tile")
                                }
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: false,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .issueWithoutPending:
                                break
                            case .prefetchNext, .drainBeforeIssue:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler requested pending action without pending tile")
                            }
                        }
                        let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                            model: model,
                            layer: L,
                            tileIndex: tileIndex,
                            routes: routes,
                            plannedFetch: plannedFetch,
                            avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        if !fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedMissSlots)
                        }
                        let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                            device: ctx.device,
                            binding: fetch.binding)
                        let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: fetch.binding,
                            offsets: routedOffsets)
                        guard let tileCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        _ = prefillGroupedMoE.encodeStreamedBatched(
                            commandBuffer: tileCB,
                            hidden: scratch.routedX,
                            sortedPairs: metadata.sortedPairs,
                            routePartials: scratch.routePartials,
                            gateUpActScratch: scratch.routedGateUpActScratch,
                            downScratch: scratch.routedDownScratch,
                            argumentBuffer: argumentBuffer,
                            binding: fetch.binding,
                            params: streamedParams,
                            pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    guard let tailCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                      routePartials: scratch.routePartials,
                                                      routeWeights: scratch.routeWeights,
                                                      h2: scratch.h2,
                                                      queryCount: UInt32(t),
                                                      topK: UInt32(cfg.topKExperts),
                                                      d: UInt32(D))
                    if cfg.ffnSandwichNorms {
                        let layerScalarView = views.layerScalar!
                        let scalarBits = layerScalarView.buffer.contents()
                            .advanced(by: Int(layerScalarView.offset))
                            .assumingMemoryBound(to: UInt16.self)[0]
                        prefillLayerTail.encode(commandBuffer: tailCB,
                                                h2: scratch.h2,
                                                h1: scratch.h1,
                                                hidden: scratch.hidden,
                                                postFFN2Weight: views.postFFN2!.buffer,
                                                postFFN2WeightOffset: Int(views.postFFN2!.offset),
                                                postFFNWeight: views.postFFN!.buffer,
                                                postFFNWeightOffset: Int(views.postFFN!.offset),
                                                queryCount: UInt32(t),
                                                d: UInt32(D),
                                                h2StrideElements: UInt32(D),
                                                h1StrideElements: UInt32(D),
                                                hiddenStrideElements: UInt32(D),
                                                eps: eps,
                                                layerScalar: Quantization.bf16ToFloat(scalarBits))
                    } else {
                        // Plain pre-norm tail: hidden += gated shared branch
                        // + routed branch.
                        elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                       hidden: scratch.hidden,
                                                       delta: scratch.h1,
                                                       count: t * D)
                        elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                       hidden: scratch.hidden,
                                                       delta: scratch.h2,
                                                       count: t * D)
                    }
                    tailCB.commit()
                    withExtendedLifetime(metadata) {
                        waitForCompletion(tailCB)
                    }
                    if let error = tailCB.error {
                        throw error
                    }
                    if let error = sharedCB.error {
                        throw error
                    }
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        cb = nextCB
                    }
                    continue
        }

        if writeFinalHead {
            let finalNorm = model.finalNorm
            let lm = model.lmHead
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                fusionHead.encodeGreedyDecode(
                    commandBuffer: finalCB,
                    hidden: scratch.hidden,
                    hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lm.buffer,
                    weightsOffset: Int(lm.offset),
                    scales: lm.buffer,
                    scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer,
                    biasesOffset: Int(lm.biasOffset),
                    outToken: greedyTokenBuf,
                    d: UInt32(D),
                    vocab: UInt32(cfg.vocabSize),
                    rmsEps: eps)
            } else {
                prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                                 hiddenBlock: scratch.hidden,
                                                 row: t - 1,
                                                 rowStrideElements: D,
                                                 normWeight: finalNorm.buffer,
                                                 normWeightOffset: Int(finalNorm.offset),
                                                 weights: lm.buffer,
                                                 weightsOffset: Int(lm.offset),
                                                 scales: lm.buffer,
                                                 scalesOffset: Int(lm.scaleOffset),
                                                 biases: lm.buffer,
                                                 biasesOffset: Int(lm.biasOffset),
                                                 logits: logits,
                                                 d: UInt32(D),
                                                 vocab: UInt32(cfg.vocabSize),
                                                 rmsEps: eps)
            }
            finalCB.commit()
            waitForCompletion(finalCB)
            if let error = finalCB.error {
                throw error
            }
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }

        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        if cfg.hasCompressedAttentionLayers {
            try await produceTokenDSV4(token: token, position: position,
                                       into: logits, emitHead: emitHead,
                                       outputMode: outputMode)
            return
        }
        if cfg.family == .inklingSmall {
            try await produceTokenInkling(token: token, position: position,
                                          into: logits, emitHead: emitHead,
                                          outputMode: outputMode)
            return
        }
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(cfg.hiddenSize).squareRoot()
            : 1.0
        struct PendingRoutedCommand {
            let cb: MTLCommandBuffer
            /// The layer's attention+router+shared-expert buffer. Always
            /// completed before `cb` (same queue, committed first); retained
            /// only so its error surfaces.
            let attentionCB: MTLCommandBuffer?
            let phase1HitCB: MTLCommandBuffer?
            let encodeAndCommitNanos: UInt64
        }
        var pendingRoutedCommand: PendingRoutedCommand?

        func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                        waitIfNeeded: Bool) {
            if waitIfNeeded {
                func wait(_ cb: MTLCommandBuffer) {
                    waitForCompletion(cb)
                }
                if let attentionCB = pending.attentionCB {
                    wait(attentionCB)
                }
                if let phase1HitCB = pending.phase1HitCB {
                    wait(phase1HitCB)
                }
                wait(pending.cb)
            } else if let err = pending.cb.error {
                print("CB error: \(err)")
            }
            if let attentionCB = pending.attentionCB {
                if let err = attentionCB.error {
                    print("CB error: \(err)")
                }
            }
            if let phase1HitCB = pending.phase1HitCB,
               let err = phase1HitCB.error {
                print("CB error: \(err)")
            }
            totalCb2Nanos &+= pending.encodeAndCommitNanos
        }

        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }

        // Embed lookup + sqrt(H) fused.
        let emb = model.embedding
        do {
            runSync { cb in
                embedInt4.encode(commandBuffer: cb,
                                 table:  emb.buffer, tableOffset:  Int(emb.offset),
                                 scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                 biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                 out: hidden,
                                 tokenId: UInt32(bitPattern: token),
                                 d: D,
                                 outScale: embedOutScale)
            }
        }

        for L in 0..<cfg.numLayers {
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim     = UInt32(cfg.numHeads * headDimL)
            let kvDim    = UInt32(numKVL * headDimL)
            let seqLen   = UInt32(position + 1)

            let inNorm   = try model.inputNorm(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let routerW  = try model.router(layer: L)
            let perExpertScale: (buffer: MTLBuffer, offset: Int)
            if cfg.routerScaled {
                let view = try model.routerPerExpertScale(layer: L)
                perExpertScale = (view.buffer, Int(view.offset))
            } else {
                perExpertScale = (onesPerExpertScale!, 0)
            }

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Everything up to and including the router runs in a single CB:
            // the only reason to break is the CPU readback of router indices
            // needed to issue I/O for the routed-expert blobs.
            let cb = ctx.queue.makeCommandBuffer()!
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed,
                            d: D, eps: eps)

            if isLinear {
                // Gated-DeltaNet linear attention: no KV slots, no RoPE — a
                // fixed-size recurrent state updated in place.
                try encodeLinearAttentionDecode(cb, layer: L)
            } else if cfg.attnOutputGate {
                // Qwen full attention: packed [query ; gate] q_proj, real
                // v_proj, no V norm, NeoX sub-dim RoPE, sigmoid output gate.
                try encodeGatedFullAttentionDecode(cb, layer: L,
                                                   position: position,
                                                   seqLen: seqLen)
            } else {
                let kSlot = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
                let vSlot = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
                let q     = try model.qProj(layer: L)
                let k     = try model.kProj(layer: L)
                // Under the K=V quirk full layers reuse k_proj; otherwise
                // v_proj is a real tensor.
                let vProj = (isFull && cfg.attentionKEqV) ? k : (try model.vProj(layer: L))
                let o     = try model.oProj(layer: L)
                let qNorm = try model.qNorm(layer: L)
                let kNorm = try model.kNorm(layer: L)

                fusedQKVGEMV.encode(commandBuffer: cb,
                                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                    x: normed,
                                    qOut: qScratch,
                                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                    qRows: qDim,
                                    kvRows: kvDim,
                                    n: D)

                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                fusedQKVEpilogue.encode(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer,
                                        kOffset: kSlot.offset,
                                        v: vSlot.buffer,
                                        vOffset: vSlot.offset,
                                        qWeight: qNorm.buffer,
                                        qWeightOffset: Int(qNorm.offset),
                                        kWeight: kNorm.buffer,
                                        kWeightOffset: Int(kNorm.offset),
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        position: UInt32(position),
                                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                        rotatedPairs: rotated,
                                        eps: eps)

                guard kv != nil else {
                    preconditionFailure("FP16 attention requires an FP16 KV cache")
                }
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: Float(cfg.attentionScale))
                } else {
                    let ringCapacity = kv?.ringCapacity(layer: L) ?? 0
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer, kOffset: 0,
                                        v: vSlot.buffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: Float(cfg.attentionScale),
                                        ringCapacity: activeRingCapacity)
                }
                int4.encode(commandBuffer: cb,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                            biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim)
            }

            if cfg.ffnSandwichNorms {
                let preFFN   = try model.preFFN(layer: L)
                let preFFN2  = try model.preFFN2(layer: L)
                fusedPostAttentionSetup.encode(commandBuffer: cb,
                                               hidden: hidden,
                                               attn: oOut,
                                               denseX: denseX,
                                               routedX: routedX,
                                               routerX: routerInput,
                                               postAttentionWeight: postAttn.buffer,
                                               postAttentionWeightOffset: Int(postAttn.offset),
                                               preFFNWeight: preFFN.buffer,
                                               preFFNWeightOffset: Int(preFFN.offset),
                                               preFFN2Weight: preFFN2.buffer,
                                               preFFN2WeightOffset: Int(preFFN2.offset),
                                               d: D,
                                               eps: eps)
            } else {
                // Plain pre-norm residual block: hidden += attention branch,
                // then one post-attention norm feeds router, shared expert,
                // and routed phase 1 (routedX doubles as moeX).
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: hidden,
                                               delta: oOut,
                                               count: cfg.hiddenSize)
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: postAttn.buffer,
                                weightOffset: Int(postAttn.offset),
                                out: routedX,
                                d: D, eps: eps)
            }

            moe.encodeRouterGemma4(commandBuffer: cb,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                hidden: cfg.ffnSandwichNorms ? routerInput : routedX,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale.buffer,
                perExpertScaleOffset: perExpertScale.offset,
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))

            // Lookahead: layer L+1's router against layer L's pre-FFN normed
            // state, in the same CB so the result is ready at the router
            // wake below — no extra synchronization. Approximate by
            // construction; scored as recall by the phase diagnostics and
            // consumed only by the speculative prefetcher.
            if speculativePrefetchMode == .pilot || speculativePrefetchMode == .shadow,
               speculativeExpertPredictor == nil,
               L + 1 < cfg.numLayers,
               let nextRouterW = try? model.router(layer: L + 1) {
                let nextPerExpertScale: (buffer: MTLBuffer, offset: Int)
                if cfg.routerScaled,
                   let view = try? model.routerPerExpertScale(layer: L + 1) {
                    nextPerExpertScale = (view.buffer, Int(view.offset))
                } else {
                    nextPerExpertScale = (onesPerExpertScale!, 0)
                }
                moe.encodeRouterGemma4(commandBuffer: cb,
                    weights: nextRouterW.buffer,
                    weightsOffset: Int(nextRouterW.offset),
                    scales:  nextRouterW.buffer,
                    scalesOffset:  Int(nextRouterW.scaleOffset),
                    biases:  nextRouterW.buffer,
                    biasesOffset:  Int(nextRouterW.biasOffset),
                    hidden: cfg.ffnSandwichNorms ? routerInput : routedX,
                    effectiveScale: effectiveScaleBuffers[L + 1],
                    perExpertScale: nextPerExpertScale.buffer,
                    perExpertScaleOffset: nextPerExpertScale.offset,
                    outIndices: pilotIndices, outWeights: pilotWeights,
                    numExperts: UInt32(cfg.numExperts), d: D,
                    topK: UInt32(cfg.topKExperts))
            }

            // S3: resolve routed experts to slab offsets and write the
            // all-hit flag BEFORE the signal, so both are trustworthy the
            // moment the CPU wakes. Qwen-only (plain pre-norm tail) while
            // the guarded chain has an accepted A/B for exactly one family.
            var slotMapArmed = false
            var slotMapBinding: (slab: MTLBuffer, table: MTLBuffer, slotStride: Int)?
            if slotMapEnabled,
               Self.slotMapOnlyLayer == nil || Self.slotMapOnlyLayer == L,
               !cfg.ffnSandwichNorms,
               !isLinear,
               model.config.family == .qwen36,
               cfg.topKExperts == MoE.maxStreamedExperts,
               let binding = try? model.routedSlotMapBinding(layer: L) {
                slotMapBinding = binding
                slotMapArmed = true
                moe.encodeSlotLookup(commandBuffer: cb,
                                     indices: outIndices,
                                     table: Self.slotMapForceMiss
                                         ? (slotMapEmptyTable ?? binding.table)
                                         : binding.table,
                                     slotStride: binding.slotStride,
                                     slotOffsets: slotMapOffsets,
                                     allHit: slotMapAllHit,
                                     numExperts: UInt32(cfg.numExperts),
                                     topK: UInt32(cfg.topKExperts))
            }

            // S3 guarded routed FFN in its own command buffer, committed
            // right behind cb1 (tracked hazards order it after the lookup,
            // shared expert, and gate writes). Completes all-hit layers
            // entirely on-GPU; no-ops otherwise.
            var slotMapGuardCB: MTLCommandBuffer?
            if slotMapArmed, let binding = slotMapBinding {
                let debug = Self.slotMapDebugCompare
                let guardCB = ctx.queue.makeCommandBuffer()!
                moe.encodeSlotMapGuardedFFN(commandBuffer: guardCB,
                    slab: binding.slab,
                    slotOffsets: slotMapOffsets,
                    allHit: slotMapAllHit,
                    routedOffsets: model.routedExpertOffsets(layer: L),
                    x: routedX,
                    acts: debug ? slotMapDebugActs! : moeActs,
                    routingWeights: outWeights,
                    residual: h1Buf,
                    y: debug ? slotMapDebugOut! : h2Buf,
                    hidden: debug ? slotMapDebugHidden! : hidden,
                    d: D, f: FmoE, topK: UInt32(cfg.topKExperts))
                slotMapGuardCB = guardCB
            }

            // The router indices are written at this point in the buffer, so
            // signal the CPU here and keep encoding into the SAME buffer. The
            // shared dense MLP depends only on the post-attention norm, never
            // on the routed experts, so the GPU runs it while the CPU wakes,
            // plans slots and preads the routed-expert blobs.
            let routerSignal = encodeRouterSignal(cb)
            try! shared.encode(commandBuffer: cb,
                               x: cfg.ffnSandwichNorms ? denseX : routedX,
                               gate: sharedProj.gate,
                               up: sharedProj.up,
                               down: sharedProj.down,
                               y: h1Buf,
                               scratchGate: denseScratchGate,
                               scratchUp: denseScratchUp,
                               scratchAct: denseScratchAct)
            if cfg.ffnSandwichNorms {
                let postF1 = sharedProj.postF1!
                rms.encodeBF16W(commandBuffer: cb, x: h1Buf,
                                weight: postF1.buffer,
                                weightOffset: Int(postF1.offset),
                                out: h1Buf, d: D, eps: eps)
            } else if cfg.sharedExpertGated {
                // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
                let gateView = sharedProj.scalarGate!
                int8ScalarGate!.encode(commandBuffer: cb,
                                       weights: gateView.buffer,
                                       weightsOffset: Int(gateView.offset),
                                       scales: gateView.buffer,
                                       scalesOffset: Int(gateView.scaleOffset),
                                       biases: gateView.buffer,
                                       biasesOffset: Int(gateView.biasOffset),
                                       x: routedX,
                                       y: sharedScalarGateBuf!,
                                       m: 1, n: D)
                elementwise!.encodeSigmoidScalarMul(commandBuffer: cb,
                                                    y: h1Buf,
                                                    gate: sharedScalarGateBuf!,
                                                    count: cfg.hiddenSize)
            }
            let overlapProbe = Self.phaseInstrumentationEnabled ? GPUOverlapProbe() : nil
            overlapProbe?.track(cb)
            trackGpuInterval(cb)
            cb.commit()
            if let guardCB = slotMapGuardCB {
                trackGpuInterval(guardCB)
                guardCB.commit()
            }
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            waitForRouterSignal(routerSignal, fallback: cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            if let pending = pendingRoutedCommand {
                finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            // CPU readback to fetch routed-expert blobs from disk.
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }

            // The lookahead router result is complete at this wake; it feeds
            // the L+1 prefetch issued after this layer's routed CB commits.
            if speculativePrefetchMode == .pilot || speculativePrefetchMode == .shadow,
               speculativeExpertPredictor == nil,
               L + 1 < cfg.numLayers {
                let pilotPtr = pilotIndices.contents().bindMemory(
                    to: UInt32.self, capacity: cfg.topKExperts)
                pilotPrediction = (L + 1, (0..<cfg.topKExperts).map {
                    min(Int(pilotPtr[$0]), cfg.numExperts - 1)
                })
            }

            // Join any speculative read aimed at this layer before planning —
            // the plan must not evict a slot a background pread is filling —
            // and score the prediction. Then this layer's routing becomes the
            // next token's prediction for the same layer.
            settleSpeculation(layer: L, actualExperts: experts)
            recordRoutedExperts(experts, layer: L)

            if slotMapArmed, !Self.slotMapDebugCompare,
               slotMapAllHit.contents().load(as: UInt32.self) == 1 {
                // The GPU already finished this layer's routed FFN inside
                // cb1: no plan, no fetch, no routed command buffer. Keep the
                // LFU counters exactly as a plan would have left them.
                if let streamer = try? model.routedExpertStreamer(layer: L) {
                    streamer.noteAllHitUse(experts: experts)
                }
                if Self.phaseInstrumentationEnabled {
                    totalRoutedLayerSteps &+= 1
                    totalAllHitLayerSteps &+= 1
                }
                issueSpeculativePrefetch(layer: L + 1)
                continue
            }

            let routedOffsets = model.routedExpertOffsets(layer: L)
            let topK = UInt32(cfg.topKExperts)
            let canPlanPhase1HitSplit =
                cfg.topKExperts <= MoE.maxStreamedExperts
            let plannedFetch = canPlanPhase1HitSplit
                ? try model.planRoutedExperts(layer: L, experts: experts)
                : nil
            if Self.phaseInstrumentationEnabled {
                totalRoutedLayerSteps &+= 1
                if let plannedFetch, plannedFetch.misses.isEmpty {
                    totalAllHitLayerSteps &+= 1
                }
            }
            var phase1HitCB: MTLCommandBuffer?
            var phase1HitSplitArgBuf: MTLBuffer?
            var phase1HitSplitRoutedBufs: [(buffer: MTLBuffer, offset: Int)] = []
            var phase1HitSlots: [UInt32] = []
            var phase1MissSlots: [UInt32] = []
            if let plan = plannedFetch {
                let missSet = Set(plan.misses)
                phase1HitSlots = (0..<cfg.topKExperts)
                    .filter { !missSet.contains($0) }
                    .map { UInt32($0) }
                phase1MissSlots = plan.misses.map { UInt32($0) }
            }
            func encodeRoutedPhase1Full(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [(buffer: MTLBuffer, offset: Int)]
            ) {
                moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                        routedArgBuffer: argBuf,
                                                        routedBlobs: routedBufs,
                                                        routedOffsets: routedOffsets,
                                                        x: routedX,
                                                        acts: moeActs,
                                                        d: D,
                                                        f: FmoE,
                                                        topK: topK)
            }

            func encodeRoutedPhase1Subset(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [(buffer: MTLBuffer, offset: Int)],
                activeSlots: MTLBuffer,
                activeSlotIndices: [UInt32],
                activeCount: UInt32
            ) {
                moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: cb,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX,
                    acts: moeActs,
                    activeSlots: activeSlots,
                    activeSlotIndices: activeSlotIndices,
                    activeCount: activeCount,
                    d: D,
                    f: FmoE,
                    topK: topK)
            }

            if let plan = plannedFetch,
               plan.hits > 0,
               !plan.misses.isEmpty {
                let plannedBlobs = try model.routedExpertBuffers(for: plan)
                phase1HitSplitRoutedBufs = plannedBlobs.map { (buffer: $0.buffer, offset: Int($0.offset)) }
                phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                    routedBlobs: phase1HitSplitRoutedBufs,
                    topK: topK)
                if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                    writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                    let cb = ctx.queue.makeCommandBuffer()!
                    encodeRoutedPhase1Subset(
                        cb,
                        argBuf: argBuf,
                        routedBufs: phase1HitSplitRoutedBufs,
                        activeSlots: moeHitActiveSlots,
                        activeSlotIndices: phase1HitSlots,
                        activeCount: UInt32(phase1HitSlots.count))
                    phase1HitCB = cb
                }
            }

            // Phase-1 for the resident (hit) experts needs no I/O, so it goes
            // to the GPU immediately, extending the window the pread hides
            // behind. It follows the shared MLP on the same queue.
            if let hitCB = phase1HitCB {
                overlapProbe?.track(hitCB)
                trackGpuInterval(hitCB)
                hitCB.commit()
            }
            if rdadviseEnabled && rdadvisePolicyMode != .off {
                let requestedMisses = plannedFetch?.misses.count ?? experts.count
                let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                    layer: L,
                    missCount: requestedMisses)
                if let skipped = shouldSkipRDAdvice(position: position,
                                                    requestedMisses: requestedMisses,
                                                    estimatedBytes: estimatedAdviceBytes,
                                                    canOverlapUsefulGPUWork: true) {
                    recordRDAdvice(skipped, wallNanos: 0)
                } else {
                    let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let result: ExpertIOAdviceResult
                    if let plannedFetch {
                        result = try model.adviseRoutedExperts(plan: plannedFetch)
                    } else {
                        result = try model.adviseRoutedExperts(layer: L, experts: experts)
                    }
                    let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                    recordRDAdvice(result, wallNanos: wallNanos)
                    updateRDAdvicePolicy(after: result, position: position)
                }
            }

            // Routed-expert fills. Eager mode encodes the routed command
            // buffer against the plan's slot views (slab offsets need no
            // I/O), gates it on the fill event, and runs the preads in the
            // background — the fetch leaves the CPU critical path entirely.
            // The awaited path remains the fallback.
            let routedBufs: [(buffer: MTLBuffer, offset: Int)]
            var eagerSeq: UInt64?
            if eagerRoutedEnabled, let plannedFetch, eagerFetchEvent != nil {
                routedBufs = try model.routedExpertBuffers(for: plannedFetch)
                    .map { (buffer: $0.buffer, offset: Int($0.offset)) }
                eagerFetchIssued &+= 1
                eagerSeq = eagerFetchIssued
            } else {
                let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let blobs: [TensorView]
                if let plannedFetch {
                    blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
                } else {
                    blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
                }
                let tIoEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                totalIoNanos &+= tIoEnd - tIoStart
                recordExpertIOOverlap(probe: overlapProbe,
                                      startNanos: tIoStart,
                                      endNanos: tIoEnd)
                routedBufs = blobs.map { (buffer: $0.buffer, offset: Int($0.offset)) }
            }
            let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let gTail: (MTLCommandBuffer) -> Void
            if cfg.ffnSandwichNorms {
                let postF2 = try model.postFFN2(layer: L)
                let postF = try model.postFFN(layer: L)
                let layerScalarView = try model.layerScalar(layer: L)
                let scalarPtr = layerScalarView.buffer.contents()
                    .advanced(by: Int(layerScalarView.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let layerScalar = Quantization.bf16ToFloat(scalarPtr[0])
                gTail = { [self] cb in
                    fusedTail.encode(commandBuffer: cb,
                                     h2: h2Buf,
                                     h1: h1Buf,
                                     hidden: hidden,
                                     postFFN2Weight: postF2.buffer,
                                     postFFN2WeightOffset: Int(postF2.offset),
                                     postFFNWeight: postF.buffer,
                                     postFFNWeightOffset: Int(postF.offset),
                                     d: D,
                                     eps: eps,
                                     layerScalar: layerScalar)
                }
            } else {
                // The phase-2 reduce already folded the shared branch (h1Buf
                // as its residual); the tail is a plain residual add.
                gTail = { [self] cb in
                    elementwise!.encodeResidualAdd(commandBuffer: cb,
                                                   hidden: hidden,
                                                   delta: h2Buf,
                                                   count: cfg.hiddenSize)
                }
            }
            let routedCB = ctx.queue.makeCommandBuffer()!
            if let seq = eagerSeq, let event = eagerFetchEvent {
                routedCB.encodeWaitForEvent(event, value: seq)
            }
            let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
                ? phase1HitSplitArgBuf
                : nil
            let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs,
                topK: topK)
            if splitArgBuf != nil {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                encodeRoutedPhase1Subset(
                    routedCB,
                    argBuf: argBuf,
                    routedBufs: routedBufs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count))
            } else {
                encodeRoutedPhase1Full(routedCB,
                                       argBuf: argBuf,
                                       routedBufs: routedBufs)
            }
            moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: routedBufs,
                                                   routedOffsets: routedOffsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: cfg.ffnSandwichNorms ? zeroResidual : h1Buf,
                                                   y: h2Buf,
                                                   d: D,
                                                   f: FmoE,
                                                   topK: topK)
            gTail(routedCB)
            trackGpuInterval(routedCB)
            routedCB.commit()
            if let seq = eagerSeq, let plannedFetch {
                try model.fillRoutedExpertsAsync(plan: plannedFetch) {
                    [weak self, layer = plannedFetch.layer] success in
                    self?.eagerFetchCompleted(seq, success: success, layer: layer)
                }
            }
            if Self.slotMapDebugCompare, slotMapArmed,
               slotMapAllHit.contents().load(as: UInt32.self) == 1 {
                while routedCB.status != .completed { usleep(200) }
                let n = Int(topK) * Int(FmoE)
                let a = moeActs.contents().bindMemory(to: UInt16.self, capacity: n)
                let b = slotMapDebugActs!.contents().bindMemory(to: UInt16.self, capacity: n)
                var firstActs = -1
                for i in 0..<n where a[i] != b[i] { firstActs = i; break }
                let m = Int(D)
                let ya = h2Buf.contents().bindMemory(to: UInt16.self, capacity: m)
                let yb = slotMapDebugOut!.contents().bindMemory(to: UInt16.self, capacity: m)
                var firstY = -1
                for i in 0..<m where ya[i] != yb[i] { firstY = i; break }
                if firstActs >= 0 || firstY >= 0 {
                    FileHandle.standardError.write(Data(
                        "[slotmap-debug] L=\(L) actsDiff@\(firstActs) yDiff@\(firstY) a=\(a[max(0,firstActs)]) b=\(b[max(0,firstActs)])\n".utf8))
                }
            }
            // GPU is busy with routed work and the next layer's attention, CPU
            // is idle: the natural window for the speculative read.
            issueSpeculativePrefetch(layer: L + 1)
            precondition(pendingRoutedCommand == nil,
                         "routed command-buffer pipeline drained before queuing the next layer")
            pendingRoutedCommand = PendingRoutedCommand(
                cb: routedCB,
                attentionCB: cb,
                phase1HitCB: phase1HitCB,
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
            continue
        }
        if let pending = pendingRoutedCommand {
            finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }
        try throwIfEagerFillFailed()

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = model.finalNorm
        let lm    = model.lmHead
        let gFinalNorm: (MTLCommandBuffer) -> Void = { cb in
            self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                 weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                 out: self.normed, d: D, eps: eps)
        }
        let gLmHead: (MTLCommandBuffer) -> Void = { cb in
            self.int4.encode(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) -> Void = { cb in
            self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if useFusedHeadForThisToken {
                runSync(gFusionHead)
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                runSync { cb in
                    gFinalNorm(cb)
                    gLmHead(cb)
                }
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv?.advance()
    }

    /// Env-gated numeric probe for the Inkling bring-up: prints value stats
    /// of a buffer after forcing the queue to drain. Debug only — the extra
    /// waits serialize the pipeline.
    private static let inklingDebugEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_INKLING_DEBUG"] == "1"
    private func inklingDebugStats(_ label: String, _ buf: MTLBuffer,
                                   count: Int, fp32: Bool = false) {
        guard Self.inklingDebugEnabled else { return }
        let cb = ctx.queue.makeCommandBuffer()!
        trackGpuInterval(cb)
        cb.commit()
        cb.waitUntilCompleted()
        var mn = Float.greatestFiniteMagnitude
        var mx = -Float.greatestFiniteMagnitude
        var sum: Double = 0
        var bad = 0
        if fp32 {
            let p = buf.contents().bindMemory(to: Float.self, capacity: count)
            for i in 0..<count {
                let v = p[i]
                if !v.isFinite { bad += 1; continue }
                mn = min(mn, v); mx = max(mx, v); sum += Double(v)
            }
        } else {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: count)
            for i in 0..<count {
                let v = Float(p[i])
                if !v.isFinite { bad += 1; continue }
                mn = min(mn, v); mx = max(mx, v); sum += Double(v)
            }
        }
        let mean = sum / Double(max(count - bad, 1))
        print("[inkling] \(label) n=\(count) "
              + String(format: "min=%.4f max=%.4f mean=%.5f", mn, mx, mean)
              + " nonfinite=\(bad)")
    }

    /// Throws if the head epilogue counted any non-finite real-vocabulary
    /// logit. A NaN or infinity in the logit row is always an engine fault
    /// upstream (an overflowed intermediate, a poisoned cache entry), never a
    /// model output — and left unchecked it does not look like a fault: the
    /// argmax and the sampler both fall through to a low token id, which in
    /// this vocabulary is `!`. Failing here turns silent mid-word corruption
    /// into a report that names the position.
    private func checkFiniteLogits(_ inkling: InklingKernels,
                                   position: Int) throws {
        let bad = inkling.nonFiniteLogits
        guard bad == 0 else {
            throw InklingHeadError.nonFiniteLogits(position: position, count: bad)
        }
    }

    /// FFN output pre-scale for the Inkling MoE path: router weights and
    /// shared-expert gammas are divided by this before the FP16 expert
    /// pipeline, and the residual add multiplies it back (linear, exact).
    /// Keeps BF16-magnitude outlier channels (observed > 65k at layer 41)
    /// inside half range through h1/h2 and the phase-2 reduce.
    static let inklingFFNPrescale: Float = 32.0

    /// Splits Inkling prefill's routed-expert loop into fetch / encode / drain,
    /// the prefill counterpart to `MFERENCE_PHASES` for decode. Off by default:
    /// when disabled the loop skips the clock reads entirely.
    ///
    /// Worth keeping because the shape it revealed is counter-intuitive. On a
    /// 2 785-token prompt the whole expert path — streaming *and* compute — is
    /// ~15 % of prefill; the other ~85 % is the per-token attention / norm /
    /// short-conv replay outside this loop, and it grows superlinearly with
    /// context (13.3x for 6.6x the tokens) because 7 of 42 layers are global.
    /// Optimize against a measurement from this, not against dispatch counts.
    static let prefillBreakdownEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_PREFILL_BREAKDOWN"] == "1"
    nonisolated(unsafe) static var prefillFetchNanos: UInt64 = 0
    nonisolated(unsafe) static var prefillEncodeNanos: UInt64 = 0
    nonisolated(unsafe) static var prefillDrainNanos: UInt64 = 0
    nonisolated(unsafe) static var prefillExpertCount: UInt64 = 0
    public static func dumpPrefillBreakdown() {
        let f = Double(prefillFetchNanos) / 1e9
        let e = Double(prefillEncodeNanos) / 1e9
        let d = Double(prefillDrainNanos) / 1e9
        FileHandle.standardError.write(Data(
            ("[prefill] experts=\(prefillExpertCount) fetch=\(String(format: "%.1f", f))s " +
             "encode=\(String(format: "%.1f", e))s drain=\(String(format: "%.1f", d))s " +
             "perExpert: fetch=\(String(format: "%.2f", f*1000/Double(max(1,prefillExpertCount))))ms " +
             "drain=\(String(format: "%.2f", d*1000/Double(max(1,prefillExpertCount))))ms\n").utf8))
    }

    /// Non-finite real logits counted by the last head dispatch. Zero on every
    /// healthy step; a regression test asserts that directly.
    var inklingNonFiniteLogitCount: Int { inkling?.nonFiniteLogits ?? 0 }

    /// Parity-test helper: advance the KV cursor after a stop-layer run so a
    /// following produce() at the next position lands in the right slot.
    func inklingParityAdvanceKV() { kv?.advance() }
    func inklingParityKSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        kv!.kSlot(layer: layer, position: position)
    }
    func inklingParityVSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        kv!.vSlot(layer: layer, position: position)
    }

    /// Debug-only early exit: stop after this layer completes, leaving the
    /// scratch buffers inspectable (CPU parity tests). -1 = disabled.
    static let inklingStopLayer: Int =
        ProcessInfo.processInfo.environment["MFERENCE_INKLING_STOP_LAYER"]
            .flatMap(Int.init) ?? -1
    /// Restricts the stop layer to one position so multi-token composition
    /// tests can run earlier tokens through all 42 layers. -1 = any position.
    static let inklingStopPosition: Int =
        ProcessInfo.processInfo.environment["MFERENCE_INKLING_STOP_POSITION"]
            .flatMap(Int.init) ?? -1
    /// Forces a full GPU drain after every layer (disables the routed-CB
    /// pipelining). Debug aid to discriminate ordering hazards.
    static let inklingSyncMode: Bool =
        ProcessInfo.processInfo.environment["MFERENCE_INKLING_SYNC"] == "1"
    /// When set (with MFERENCE_INKLING_DEBUG=1), per-layer FP32 hidden rows
    /// are dumped as raw f32 files for cross-checking against the reference.
    static let inklingDumpDir: String? =
        ProcessInfo.processInfo.environment["MFERENCE_INKLING_DUMP_DIR"]

    /// Inkling-Small decode step. Semantics per docs/INKLING_SMALL.md
    /// "Forward-pass contract": no RoPE (learned relative-position bias),
    /// depthwise short convs on K/V and on both sublayer outputs (inside the
    /// residual), sigmoid router with 2 shared-expert sinks, 2 leading dense
    /// layers, muP logit scaling, and vocab truncation to the unpadded size.
    private func produceTokenInkling(token: Int32,
                                     position: Int,
                                     into logits: MTLBuffer,
                                     emitHead: Bool,
                                     outputMode: PrefillOutputMode) async throws {
        guard let inkling, let relScratch = inklingRelScratch,
              let hiddenF32 = inklingHiddenF32,
              let deltaF32 = inklingDeltaF32,
              let sharedY0 = inklingSharedY0, let sharedY1 = inklingSharedY1,
              let kv else {
            preconditionFailure("Inkling runtime state missing")
        }
        let D = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = Float(1e-6)
        let qDim = UInt32(cfg.numHeads * cfg.headDim)
        let kvDim = UInt32(cfg.numKVHeads * cfg.headDim)
        let seqLen = UInt32(position + 1)
        let sconvK = UInt32(cfg.sconvKernelSize)
        let rel = cfg.relativePosition

        struct PendingInkling {
            let cb: MTLCommandBuffer
            let attentionCB: MTLCommandBuffer
            let phase1HitCB: MTLCommandBuffer?
        }
        var pending: PendingInkling?
        func finishPending(_ p: PendingInkling, waitIfNeeded: Bool) {
            if waitIfNeeded {
                waitForCompletion(p.attentionCB)
                if let hitCB = p.phase1HitCB { waitForCompletion(hitCB) }
                waitForCompletion(p.cb)
            } else {
                if let err = p.cb.error { print("CB error: \(err)") }
                if let err = p.attentionCB.error { print("CB error: \(err)") }
                if let err = p.phase1HitCB?.error { print("CB error: \(err)") }
            }
        }
        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }

        // Embed lookup, then the family's embed norm (in place).
        let emb = model.embedding
        let embedNorm = model.embedNorm
        runSync { cb in
            embedInt4.encode(commandBuffer: cb,
                             table:  emb.buffer, tableOffset:  Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden,
                             tokenId: UInt32(bitPattern: token),
                             d: D,
                             outScale: 1.0)
            // The embed norm is part of the stream seed itself
            // (`embed_tokens = embed_norm(embed(ids))`), then the stream
            // widens to FP32 for the residual accumulation.
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: embedNorm.buffer,
                            weightOffset: Int(embedNorm.offset),
                            out: hidden, d: D, eps: eps)
            inkling.encodeF16ToF32(commandBuffer: cb, src: hidden,
                                   dst: hiddenF32, count: D)
        }
        inklingDebugStats("embed f32", hiddenF32, count: cfg.hiddenSize, fp32: true)

        for L in 0..<cfg.numLayers {
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isDense = L < cfg.numDenseLayers
            let conv = inklingConvStates[L]

            let inNorm = try model.inputNorm(layer: L)
            let mlpNorm = try model.postAttnNorm(layer: L)
            let q = try model.inklingWqDu(layer: L)
            let k = try model.inklingWkDv(layer: L)
            let v = try model.inklingWvDv(layer: L)
            let o = try model.inklingWoUd(layer: L)
            let wr = try model.inklingWrDu(layer: L)
            let relProj = try model.inklingRelProj(layer: L)
            let qNorm = try model.inklingQNorm(layer: L)
            let kNorm = try model.inklingKNorm(layer: L)
            let kSconvW = try model.inklingKSconv(layer: L)
            let vSconvW = try model.inklingVSconv(layer: L)
            let attnSconvW = try model.inklingAttnSconv(layer: L)
            let mlpSconvW = try model.inklingMlpSconv(layer: L)

            let kSlot = kv.kSlot(layer: L, position: position)
            let vSlot = kv.vSlot(layer: L, position: position)

            var cb = ctx.queue.makeCommandBuffer()!
            // Debug-only pipeline cut: drain, probe, continue in a fresh CB.
            func dbgCut(_ probes: [(String, MTLBuffer, Int)]) {
                guard Self.inklingDebugEnabled, L == 2 else { return }
                cb.commit()
                waitForCompletion(cb)
                for (label, buffer, count) in probes {
                    inklingDebugStats("L\(L)* \(label)", buffer, count: count)
                }
                cb = ctx.queue.makeCommandBuffer()!
            }
            inkling.encodeRMSF32(commandBuffer: cb, x: hiddenF32,
                                 weight: inNorm.buffer,
                                 weightOffset: Int(inNorm.offset),
                                 out: normed, d: D, eps: eps)

            // Q/K/V projections into scratch; K and V then pass through their
            // short convolutions on the way into the KV slot (the cache stores
            // the convolved, K-normed values, exactly like the reference's
            // post-norm cache). The fused kernel was briefly suspected during
            // the corrupt-shard incident and later exonerated (the NaNs were
            // bad wq_du bytes); its generic path is A/B-verified against the
            // three plain GEMVs at this family's (4096, 1024, 4096) shape.
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
                                qOut: qScratch,
                                kOut: kStage, kOutOffset: 0,
                                vOut: vStage, vOutOffset: 0,
                                qRows: qDim,
                                kvRows: kvDim,
                                n: D)
            dbgCut([("post-qkv qScratch", qScratch, Int(qDim)),
                    ("post-qkv kStage", kStage, Int(kvDim)),
                    ("post-qkv vStage", vStage, Int(kvDim))])
            int4.encode(commandBuffer: cb,
                        weights: wr.buffer, weightsOffset: Int(wr.offset),
                        scales:  wr.buffer, scalesOffset:  Int(wr.scaleOffset),
                        biases:  wr.buffer, biasesOffset:  Int(wr.biasOffset),
                        x: normed, y: relScratch,
                        m: UInt32(rel.projDim), n: D)
            inkling.encodeSconvStep(commandBuffer: cb,
                                    x: kStage, state: conv.k,
                                    weight: kSconvW.buffer,
                                    weightOffset: Int(kSconvW.offset),
                                    out: kSlot.buffer, outOffset: kSlot.offset,
                                    channels: kvDim, kernelSize: sconvK)
            inkling.encodeSconvStep(commandBuffer: cb,
                                    x: vStage, state: conv.v,
                                    weight: vSconvW.buffer,
                                    weightOffset: Int(vSconvW.offset),
                                    out: vSlot.buffer, outOffset: vSlot.offset,
                                    channels: kvDim, kernelSize: sconvK)
            dbgCut([("post-sconv kSlot", kSlot.buffer, Int(kvDim)),
                    ("post-sconv rel", relScratch, rel.projDim)])
            inkling.encodeQKNorm(commandBuffer: cb,
                                 q: qScratch,
                                 k: kSlot.buffer, kOffset: kSlot.offset,
                                 qWeight: qNorm.buffer, qWeightOffset: Int(qNorm.offset),
                                 kWeight: kNorm.buffer, kWeightOffset: Int(kNorm.offset),
                                 headDim: UInt32(cfg.headDim),
                                 numQHeads: UInt32(cfg.numHeads),
                                 numKVHeads: UInt32(cfg.numKVHeads),
                                 eps: eps)
            dbgCut([("post-norm qScratch", qScratch, Int(qDim)),
                    ("post-norm kSlot", kSlot.buffer, Int(kvDim))])

            let window = UInt32(cfg.slidingWindow)
            let kvStart: UInt32 = isFull ? 0 : (seqLen > window ? seqLen - window : 0)
            let ringCapacity = kv.ringCapacity(layer: L)
            let activeRing: UInt32 = (!isFull && ringCapacity > 0 && Int(seqLen) > ringCapacity)
                ? UInt32(ringCapacity) : 0
            let relExtent = UInt32(isFull ? rel.extent : cfg.slidingWindow)
            var tau: Float = 1.0
            if isFull && rel.logScalingFloor > 0 {
                let n = Float(position + 1)
                let floorN = Float(rel.logScalingFloor)
                if n > floorN {
                    tau = 1.0 + Float(rel.logScalingAlpha) * log(n / floorN)
                }
            }
            inkling.encodeAttentionDecode(commandBuffer: cb,
                                          q: qScratch,
                                          k: kSlot.buffer,
                                          v: vSlot.buffer,
                                          rel: relScratch,
                                          proj: relProj.buffer,
                                          projOffset: Int(relProj.offset),
                                          out: attnOut,
                                          headDim: UInt32(cfg.headDim),
                                          numQHeads: UInt32(cfg.numHeads),
                                          numKVHeads: UInt32(cfg.numKVHeads),
                                          seqLen: seqLen, kvStart: kvStart,
                                          relExtent: relExtent,
                                          dRel: UInt32(rel.dRel),
                                          ringCapacity: activeRing,
                                          scale: Float(cfg.attentionScale),
                                          tau: tau)
            int4.encode(commandBuffer: cb,
                        weights: o.buffer, weightsOffset: Int(o.offset),
                        scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                        biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                        x: attnOut, y: oOut, m: D, n: qDim)
            // attn_sconv sits on the attention OUTPUT, inside the residual.
            // FP32 out: deltas exceed FP16 range deep in the stack.
            inkling.encodeSconvStepF32Out(commandBuffer: cb,
                                          x: oOut, state: conv.attn,
                                          weight: attnSconvW.buffer,
                                          weightOffset: Int(attnSconvW.offset),
                                          out: deltaF32,
                                          channels: D, kernelSize: sconvK)
            inkling.encodeResidualAddF32Delta(commandBuffer: cb,
                                              hidden: hiddenF32, delta: deltaF32,
                                              count: D)
            inkling.encodeRMSF32(commandBuffer: cb, x: hiddenF32,
                                 weight: mlpNorm.buffer,
                                 weightOffset: Int(mlpNorm.offset),
                                 out: routedX, d: D, eps: eps)

            if isDense {
                // Dense SwiGLU with a learned scalar output gain, then the
                // mlp short conv, then the residual.
                let proj = inklingDenseProjections[L]!
                let gain = model.inklingScalar(
                    try model.inklingDenseGlobalScale(layer: L))
                try! shared.encode(commandBuffer: cb,
                                   x: routedX,
                                   gate: proj.gate, up: proj.up, down: proj.down,
                                   y: h2Buf,
                                   scratchGate: denseScratchGate,
                                   scratchUp: denseScratchUp,
                                   scratchAct: denseScratchAct)
                inkling.encodeScale(commandBuffer: cb, x: h2Buf,
                                    by: gain, count: D)
                inkling.encodeSconvStepF32Out(commandBuffer: cb,
                                              x: h2Buf, state: conv.mlp,
                                              weight: mlpSconvW.buffer,
                                              weightOffset: Int(mlpSconvW.offset),
                                              out: deltaF32,
                                              channels: D, kernelSize: sconvK)
                inkling.encodeResidualAddF32Delta(commandBuffer: cb,
                                                  hidden: hiddenF32, delta: deltaF32,
                                                  count: D)
                cb.commit()
                if let p = pending { finishPending(p, waitIfNeeded: false); pending = nil }
                waitForCompletion(cb)
                inklingDebugStats("L\(L) dense out", hiddenF32, count: cfg.hiddenSize, fp32: true)
                if let dir = Self.inklingDumpDir, Self.inklingDebugEnabled {
                    let data = Data(bytes: hiddenF32.contents(),
                                    count: cfg.hiddenSize * 4)
                    try? data.write(to: URL(fileURLWithPath:
                        "\(dir)/pos\(position)_L\(L).f32"))
                }
                if L == Self.inklingStopLayer,
                   Self.inklingStopPosition < 0 || Self.inklingStopPosition == position {
                    return
                }
                continue
            }

            // MoE layer: sigmoid router (+ signal for the CPU expert fetch),
            // then both shared experts weighted by their sink gammas while the
            // routed blobs stream in.
            let routerW = try model.router(layer: L)
            let gateBias = try model.inklingGateBias(layer: L)
            let gateScale = try model.inklingGateGlobalScale(layer: L)
            inkling.encodeRouter(commandBuffer: cb,
                                 weights: routerW.buffer,
                                 weightsOffset: Int(routerW.offset),
                                 hidden: routedX,
                                 onesScale: effectiveScaleBuffers[L],
                                 gateBias: gateBias.buffer,
                                 gateBiasOffset: Int(gateBias.offset),
                                 globalScale: gateScale.buffer,
                                 globalScaleOffset: Int(gateScale.offset),
                                 outIndices: outIndices,
                                 outWeights: outWeights,
                                 numRouted: UInt32(cfg.numExperts),
                                 numShared: UInt32(cfg.numSharedExperts),
                                 topK: UInt32(cfg.topKExperts),
                                 routeScale: Float(cfg.routedScalingFactor)
                                     / Self.inklingFFNPrescale,
                                 d: D)
            // PILOT lookahead: layer L+1's router against *this* layer's
            // post-attention state, into private buffers, before the signal —
            // the CPU reads real L indices and speculative L+1 indices at one
            // wake. Dense layers all precede the MoE layers in this family,
            // so L+1 here is always routed (or past the last layer, which
            // `shouldEncodePilotGemv` rejects).
            var pilotGemvEncoded = false
            if shouldEncodePilotGemv(nextLayer: L + 1),
               let pilot = ensureInklingPilotRouter(),
               let nextRouter = try? model.router(layer: L + 1),
               let nextGateBias = try? model.inklingGateBias(layer: L + 1),
               let nextGateScale = try? model.inklingGateGlobalScale(layer: L + 1) {
                pilot.encodePrediction(
                    commandBuffer: cb,
                    weights: nextRouter.buffer,
                    weightsOffset: Int(nextRouter.offset),
                    hidden: routedX,
                    onesScale: effectiveScaleBuffers[L + 1],
                    gateBias: nextGateBias.buffer,
                    gateBiasOffset: Int(nextGateBias.offset),
                    globalScale: nextGateScale.buffer,
                    globalScaleOffset: Int(nextGateScale.offset),
                    numRouted: UInt32(cfg.numExperts),
                    numShared: UInt32(cfg.numSharedExperts),
                    topK: UInt32(cfg.topKExperts),
                    routeScale: Float(cfg.routedScalingFactor)
                        / Self.inklingFFNPrescale,
                    d: D)
                pilotGemvEncoded = true
            }
            let routerSignal = encodeRouterSignal(cb)
            let sharedProjs = inklingSharedProjections[L]
            try! shared.encode(commandBuffer: cb,
                               x: routedX,
                               gate: sharedProjs[0].gate,
                               up: sharedProjs[0].up,
                               down: sharedProjs[0].down,
                               y: sharedY0,
                               scratchGate: denseScratchGate,
                               scratchUp: denseScratchUp,
                               scratchAct: denseScratchAct,
                               outputFloat32: true)
            try! shared.encode(commandBuffer: cb,
                               x: routedX,
                               gate: sharedProjs[1].gate,
                               up: sharedProjs[1].up,
                               down: sharedProjs[1].down,
                               y: sharedY1,
                               scratchGate: denseScratchGate,
                               scratchUp: denseScratchUp,
                               scratchAct: denseScratchAct,
                               outputFloat32: true)
            // Gammas carry the 1/32 prescale, so h1Buf is back in FP16 range.
            inkling.encodeGammaCombine(commandBuffer: cb,
                                       a: sharedY0, b: sharedY1,
                                       y: h1Buf, count: D,
                                       inputsAreFloat32: true)
            cb.commit()
            waitForRouterSignal(routerSignal, fallback: cb)
            if let p = pending { finishPending(p, waitIfNeeded: false); pending = nil }

            // CPU readback of the six routed experts. A mixed cache plan sends
            // the resident subset to Metal immediately, then preads only the
            // misses while the shared branch and resident phase 1 execute.
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }
            // Join any speculative read aimed at this layer before planning —
            // the plan must not evict a slot a background pread is filling —
            // and score the prediction. Then this layer's routing becomes the
            // next token's prediction for the same layer.
            settleSpeculation(layer: L, actualExperts: experts)
            recordRoutedExperts(experts, layer: L)
            captureInklingPilotPrediction(nextLayer: L + 1,
                                          gemvEncoded: pilotGemvEncoded)
            let topK = UInt32(cfg.topKExperts)
            let routedOffsets = model.routedExpertOffsets(layer: L)
            let plannedFetch = try model.planRoutedExperts(layer: L, experts: experts)
            let missSet = Set(plannedFetch?.misses ?? [])
            let phase1HitSlots = (0..<cfg.topKExperts)
                .filter { !missSet.contains($0) }
                .map { UInt32($0) }
            let phase1MissSlots = (plannedFetch?.misses ?? []).map { UInt32($0) }
            var phase1HitCB: MTLCommandBuffer?
            var splitArgBuf: MTLBuffer?
            var splitRoutedBufs: [(buffer: MTLBuffer, offset: Int)] = []

            if let plan = plannedFetch,
               plan.hits > 0,
               !plan.misses.isEmpty {
                splitRoutedBufs = try model.routedExpertBuffers(for: plan)
                    .map { (buffer: $0.buffer, offset: Int($0.offset)) }
                // Router readback is a same-queue fence for the prior layer,
                // so MoE's preallocated argument storage is no longer in use.
                splitArgBuf = moe.makeRoutedArgumentBuffer(
                    routedBlobs: splitRoutedBufs, topK: topK)
                if let argBuf = splitArgBuf {
                    writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                    let hitCB = ctx.queue.makeCommandBuffer()!
                    moe.encodeRoutedPersistentPhase1SubsetU16Load(
                        commandBuffer: hitCB,
                        routedArgBuffer: argBuf,
                        routedBlobs: splitRoutedBufs,
                        routedOffsets: routedOffsets,
                        x: routedX,
                        acts: moeActs,
                        activeSlots: moeHitActiveSlots,
                        activeSlotIndices: phase1HitSlots,
                        activeCount: UInt32(phase1HitSlots.count),
                        d: D,
                        f: FmoE,
                        topK: topK)
                    hitCB.commit()
                    phase1HitCB = hitCB
                }
            }

            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs: [TensorView]
            if let plannedFetch {
                blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
            } else {
                blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
            }
            totalIoNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
            let routedBufs = blobs.map { (buffer: $0.buffer, offset: Int($0.offset)) }

            let routedCB = ctx.queue.makeCommandBuffer()!
            let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs, topK: topK)
            if splitArgBuf != nil {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: routedCB,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX,
                    acts: moeActs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count),
                    d: D,
                    f: FmoE,
                    topK: topK)
            } else {
                moe.encodeRoutedPersistentPhase1U16Load(
                    commandBuffer: routedCB,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX,
                    acts: moeActs,
                    d: D,
                    f: FmoE,
                    topK: topK)
            }
            // Phase 2 seeds y with the shared-expert branch, so the sum is
            // routed + shared exactly as the reference computes it.
            moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: routedBufs,
                                                   routedOffsets: routedOffsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: h1Buf,
                                                   y: h2Buf,
                                                   d: D, f: FmoE,
                                                   topK: topK)
            inkling.encodeSconvStepF32Out(commandBuffer: routedCB,
                                          x: h2Buf, state: conv.mlp,
                                          weight: mlpSconvW.buffer,
                                          weightOffset: Int(mlpSconvW.offset),
                                          out: deltaF32,
                                          channels: D, kernelSize: sconvK)
            inkling.encodeResidualAddF32Delta(commandBuffer: routedCB,
                                              hidden: hiddenF32, delta: deltaF32,
                                              count: D,
                                              scale: Self.inklingFFNPrescale)
            routedCB.commit()
            // The GPU is now busy with routed work; the CPU is free until the
            // next layer's router wake — the window where a speculative read
            // never competes with a critical-path pread.
            issueSpeculativePrefetch(layer: L + 1)
            if Self.inklingDebugEnabled {
                waitForCompletion(routedCB)
                if !(L <= 6 || L == 41) {
                    inklingDebugStats("L\(L) hidden", hiddenF32, count: cfg.hiddenSize, fp32: true)
                }
                if let dir = Self.inklingDumpDir {
                    let data = Data(bytes: hiddenF32.contents(),
                                    count: cfg.hiddenSize * 4)
                    try? data.write(to: URL(fileURLWithPath:
                        "\(dir)/pos\(position)_L\(L).f32"))
                }
            }
            if Self.inklingDebugEnabled, (18...21).contains(L) {
                inklingDebugStats("L\(L) normed", normed, count: cfg.hiddenSize)
                inklingDebugStats("L\(L) oOut", oOut, count: cfg.hiddenSize)
                inklingDebugStats("L\(L) attn delta", deltaF32, count: cfg.hiddenSize, fp32: true)
                inklingDebugStats("L\(L) routedX", routedX, count: cfg.hiddenSize)
            }
            if Self.inklingDebugEnabled, L <= 6 || L == 41 {
                inklingDebugStats("L\(L) normed", normed, count: cfg.hiddenSize)
                inklingDebugStats("L\(L) qScratch", qScratch, count: Int(qDim))
                inklingDebugStats("L\(L) kStage", kStage, count: Int(kvDim))
                inklingDebugStats("L\(L) kSlot", kSlot.buffer, count: Int(kvDim))
                inklingDebugStats("L\(L) relScratch", relScratch, count: rel.projDim)
                inklingDebugStats("L\(L) moeActs", moeActs, count: 8 * Int(FmoE))
                inklingDebugStats("L\(L) attnOut", attnOut, count: Int(qDim))
                inklingDebugStats("L\(L) oOut", oOut, count: cfg.hiddenSize)
                inklingDebugStats("L\(L) sharedY", h1Buf, count: cfg.hiddenSize)
                inklingDebugStats("L\(L) moe h2", h2Buf, count: cfg.hiddenSize)
                inklingDebugStats("L\(L) hidden", hiddenF32, count: cfg.hiddenSize, fp32: true)
                let g = inkling.sharedGammas.contents()
                    .bindMemory(to: Float.self, capacity: 2)
                let w = outWeights.contents()
                    .bindMemory(to: Float16.self, capacity: 8)
                let idx = outIndices.contents()
                    .bindMemory(to: UInt32.self, capacity: 6)
                print("[inkling] L\(L) gammas=(\(g[0]), \(g[1])) " +
                      "w=\((0..<8).map { Float(w[$0]) }) " +
                      "idx=\((0..<6).map { idx[$0] })")
            }
            pending = PendingInkling(cb: routedCB,
                                     attentionCB: cb,
                                     phase1HitCB: phase1HitCB)
            if Self.inklingSyncMode {
                finishPending(pending!, waitIfNeeded: true)
                pending = nil
            }
            if L == Self.inklingStopLayer,
               Self.inklingStopPosition < 0 || Self.inklingStopPosition == position {
                finishPending(pending!, waitIfNeeded: true)
                return
            }
        }
        if let p = pending { finishPending(p, waitIfNeeded: true); pending = nil }

        if emitHead {
            let fNorm = model.finalNorm
            let lm = model.lmHead
            let validVocab = UInt32(cfg.unpaddedVocabSize > 0
                ? cfg.unpaddedVocabSize : cfg.vocabSize)
            let muP = Float(1.0 / cfg.logitsWidthMultiplier)
            if useFusedGreedyHead && outputMode == .greedyIfAvailable {
                // Greedy argmax is invariant to the positive muP scale, so
                // only the vocab truncation matters. The final norm runs from
                // the FP32 stream first; the fused chain then consumes the
                // pre-normed row via unit gains (fNorm already applied).
                inkling.resetNonFiniteLogitCount()
                runSync { cb in
                    inkling.encodeRMSF32(commandBuffer: cb, x: hiddenF32,
                                         weight: fNorm.buffer,
                                         weightOffset: Int(fNorm.offset),
                                         out: self.normed, d: D, eps: eps)
                    self.int4.encode(commandBuffer: cb,
                                     weights: lm.buffer, weightsOffset: Int(lm.offset),
                                     scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                                     biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                                     x: self.normed, y: logits,
                                     m: validVocab, n: D)
                    inkling.encodeHeadEpilogue(commandBuffer: cb,
                                               logits: logits,
                                               scale: muP,
                                               validVocab: validVocab,
                                               totalVocab: UInt32(self.cfg.vocabSize))
                }
                try checkFiniteLogits(inkling, position: position)
                // Argmax on CPU over the truncated logits (bring-up path;
                // the fused-head chain is FP16-hidden and not yet FP32-aware).
                //
                // `best` starts at index -1, not 0: a NaN fails every `>`
                // comparison, so seeding at index 0 made an all-NaN row return
                // token 0 ("!") as if it had won. The guard above already
                // throws on that row; this keeps the invariant local.
                let lp = logits.contents()
                    .bindMemory(to: Float16.self, capacity: Int(validVocab))
                var best: (Int, Float) = (-1, -.infinity)
                for i in 0..<Int(validVocab) {
                    let v = Float(lp[i])
                    if v > best.1 { best = (i, v) }
                }
                guard best.0 >= 0 else {
                    throw InklingHeadError.noFiniteLogit(position: position)
                }
                lastGreedyToken = UInt32(best.0)
            } else {
                inkling.resetNonFiniteLogitCount()
                runSync { cb in
                    inkling.encodeRMSF32(commandBuffer: cb, x: hiddenF32,
                                         weight: fNorm.buffer,
                                         weightOffset: Int(fNorm.offset),
                                         out: self.normed, d: D, eps: eps)
                    self.int4.encode(commandBuffer: cb,
                                     weights: lm.buffer, weightsOffset: Int(lm.offset),
                                     scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                                     biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                                     x: self.normed, y: logits,
                                     m: validVocab, n: D)
                    inkling.encodeHeadEpilogue(commandBuffer: cb,
                                               logits: logits,
                                               scale: muP,
                                               validVocab: validVocab,
                                               totalVocab: UInt32(self.cfg.vocabSize))
                }
                try checkFiniteLogits(inkling, position: position)
                if Self.inklingDebugEnabled {
                    inklingDebugStats("head normed", normed, count: cfg.hiddenSize)
                    let lp = logits.contents()
                        .bindMemory(to: Float16.self, capacity: Int(validVocab))
                    var top: [(Int, Float)] = []
                    for i in 0..<Int(validVocab) {
                        let v = Float(lp[i])
                        if top.count < 8 {
                            top.append((i, v)); top.sort { $0.1 > $1.1 }
                        } else if v > top[7].1 {
                            top[7] = (i, v); top.sort { $0.1 > $1.1 }
                        }
                    }
                    print("[inkling] head top8 = \(top)")
                }
            }
        }

        kv.advance()
    }

    /// Layer-major batched prefill for Inkling. The chunk's tokens advance
    /// through one layer at a time; within a MoE layer the routed experts are
    /// processed EXPERT-major — each unique expert in the chunk's routing is
    /// streamed once and applied to every token routed to it — so cold-cache
    /// expert I/O amortizes over the chunk instead of being paid per token
    /// (~3.4 GB/token when sequential). Attention, projections, norms, routing,
    /// and fixed-width short convolutions operate on whole chunks; Apple10
    /// attention switches to cooperative QK/PV tiles after the 512-token
    /// crossover while preserving both linear and wrapped-ring KV layouts.
    private func prefillInklingChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     emitHead: Bool,
                                     outputMode: PrefillOutputMode,
                                     into logits: MTLBuffer) async throws {
        guard let inkling,
              let hiddenChunk = inklingHiddenChunk,
              let routedXChunk = inklingRoutedXChunk,
              let qChunk = inklingQChunk,
              let kChunk = inklingKChunk,
              let vChunk = inklingVChunk,
              let relChunk = inklingRelChunk,
              let attentionChunk = inklingAttentionChunk,
              let routerLogitsChunk = inklingRouterLogitsChunk,
              let idxChunk = inklingIdxChunk,
              let wChunk = inklingWChunk,
              let gammaChunk = inklingGammaChunk,
              let accChunk = inklingAccChunk,
              let pairTokens = inklingPairTokens,
              let pairWeights = inklingPairWeights,
              let actScratch = inklingExpertActScratch,
              let prefillGLU = inklingPrefillGLU,
              let kv else {
            preconditionFailure("Inkling prefill state missing")
        }
        let N = tokens.count
        precondition(N <= inklingChunkCapacity,
                     "chunk \(N) exceeds capacity \(inklingChunkCapacity)")
        let D = UInt32(cfg.hiddenSize)
        let Dh = cfg.hiddenSize
        let eps: Float = 1e-6
        let qDim = UInt32(cfg.numHeads * cfg.headDim)
        let kvDim = UInt32(cfg.numKVHeads * cfg.headDim)
        let sconvK = UInt32(cfg.sconvKernelSize)
        let rel = cfg.relativePosition
        let toks = Array(tokens)

        func hOff(_ i: Int) -> Int { i * Dh * 4 }
        func xOff(_ i: Int) -> Int { i * Dh * 2 }

        // Inkling's prefill projections are contiguous `[token, output]`
        // matrices. Their shapes favor the packed block QMM in the measured
        // Inkling path. This replaces N decode GEMV dispatches with one matrix
        // dispatch.
        func encodeProjection(_ cb: MTLCommandBuffer,
                              _ view: TensorView,
                              x: MTLBuffer,
                              y: MTLBuffer,
                              rows: Int,
                              columns: Int) {
            prefillQMM.encode(commandBuffer: cb,
                              weights: view.buffer,
                              weightsOffset: Int(view.offset),
                              scales: view.buffer,
                              scalesOffset: Int(view.scaleOffset),
                              biases: view.buffer,
                              biasesOffset: Int(view.biasOffset),
                              x: x,
                              y: y,
                              t: N,
                              n: rows,
                              k: columns)
        }

        func encodeKVWrite(_ cb: MTLCommandBuffer,
                           layer: Int,
                           key: MTLBuffer,
                           value: MTLBuffer) {
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstCount = min(N, capacity - physicalStart)
            let bytesPerRow = Int(kvDim) * MemoryLayout<Float16>.stride
            guard let blit = cb.makeBlitCommandEncoder() else {
                preconditionFailure("Inkling prefill KV blit encoder unavailable")
            }
            let keyFirst = kv.kRange(layer: layer,
                                     start: startPosition,
                                     count: firstCount)
            let valueFirst = kv.vRange(layer: layer,
                                       start: startPosition,
                                       count: firstCount)
            blit.copy(from: key, sourceOffset: 0,
                      to: keyFirst.buffer, destinationOffset: keyFirst.offset,
                      size: firstCount * bytesPerRow)
            blit.copy(from: value, sourceOffset: 0,
                      to: valueFirst.buffer, destinationOffset: valueFirst.offset,
                      size: firstCount * bytesPerRow)
            if firstCount < N {
                let secondCount = N - firstCount
                let secondStart = startPosition + firstCount
                let keySecond = kv.kRange(layer: layer,
                                          start: secondStart,
                                          count: secondCount)
                let valueSecond = kv.vRange(layer: layer,
                                            start: secondStart,
                                            count: secondCount)
                blit.copy(from: key, sourceOffset: firstCount * bytesPerRow,
                          to: keySecond.buffer, destinationOffset: keySecond.offset,
                          size: secondCount * bytesPerRow)
                blit.copy(from: value, sourceOffset: firstCount * bytesPerRow,
                          to: valueSecond.buffer, destinationOffset: valueSecond.offset,
                          size: secondCount * bytesPerRow)
            }
            blit.endEncoding()
        }

        // Seed the chunk's FP32 hidden rows: embed + embed_norm per token.
        let emb = model.embedding
        let embedNorm = model.embedNorm
        runSync { cb in
            for (i, t) in toks.enumerated() {
                embedInt4.encode(commandBuffer: cb,
                                 table:  emb.buffer, tableOffset:  Int(emb.offset),
                                 scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                 biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                 out: hidden,
                                 tokenId: UInt32(bitPattern: t),
                                 d: D, outScale: 1.0)
                rms.encodeBF16W(commandBuffer: cb, x: hidden,
                                weight: embedNorm.buffer,
                                weightOffset: Int(embedNorm.offset),
                                out: hidden, d: D, eps: eps)
                inkling.encodeF16ToF32(commandBuffer: cb, src: hidden,
                                       dst: hiddenChunk, dstOffset: hOff(i),
                                       count: D)
            }
        }

        for L in 0..<cfg.numLayers {
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isDense = L < cfg.numDenseLayers
            let conv = inklingConvStates[L]
            let inNorm = try model.inputNorm(layer: L)
            let mlpNorm = try model.postAttnNorm(layer: L)
            let q = try model.inklingWqDu(layer: L)
            let k = try model.inklingWkDv(layer: L)
            let v = try model.inklingWvDv(layer: L)
            let o = try model.inklingWoUd(layer: L)
            let wr = try model.inklingWrDu(layer: L)
            let relProj = try model.inklingRelProj(layer: L)
            let qNorm = try model.inklingQNorm(layer: L)
            let kNorm = try model.inklingKNorm(layer: L)
            let kSconvW = try model.inklingKSconv(layer: L)
            let vSconvW = try model.inklingVSconv(layer: L)
            let attnSconvW = try model.inklingAttnSconv(layer: L)
            let mlpSconvW = try model.inklingMlpSconv(layer: L)

            let cbA = ctx.queue.makeCommandBuffer()!
            if !isDense, let blit = cbA.makeBlitCommandEncoder() {
                blit.fill(buffer: accChunk, range: 0..<(N * Dh * 4), value: 0)
                blit.endEncoding()
            }
            inkling.encodeRMSF32Prefill(commandBuffer: cbA,
                                        x: hiddenChunk,
                                        weight: inNorm.buffer,
                                        weightOffset: Int(inNorm.offset),
                                        out: routedXChunk,
                                        d: D, rows: UInt32(N), eps: eps)
            encodeProjection(cbA, q, x: routedXChunk, y: qChunk,
                             rows: Int(qDim), columns: Dh)
            encodeProjection(cbA, k, x: routedXChunk, y: kChunk,
                             rows: Int(kvDim), columns: Dh)
            encodeProjection(cbA, v, x: routedXChunk, y: vChunk,
                             rows: Int(kvDim), columns: Dh)
            encodeProjection(cbA, wr, x: routedXChunk, y: relChunk,
                             rows: rel.projDim, columns: Dh)
            inkling.encodeSconvPrefill(commandBuffer: cbA,
                                       x: kChunk, state: conv.k,
                                       weight: kSconvW.buffer,
                                       weightOffset: Int(kSconvW.offset),
                                       out: kChunk,
                                       channels: kvDim, rows: UInt32(N))
            inkling.encodeSconvPrefill(commandBuffer: cbA,
                                       x: vChunk, state: conv.v,
                                       weight: vSconvW.buffer,
                                       weightOffset: Int(vSconvW.offset),
                                       out: vChunk,
                                       channels: kvDim, rows: UInt32(N))
            inkling.encodeQKNormPrefill(
                commandBuffer: cbA,
                q: qChunk, k: kChunk,
                qWeight: qNorm.buffer, qWeightOffset: Int(qNorm.offset),
                kWeight: kNorm.buffer, kWeightOffset: Int(kNorm.offset),
                headDim: UInt32(cfg.headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(cfg.numKVHeads),
                rows: UInt32(N), eps: eps)
            encodeKVWrite(cbA, layer: L, key: kChunk, value: vChunk)
            let ringCapacity = kv.ringCapacity(layer: L)
            let validCount = startPosition + N
            let activeRing = !isFull && ringCapacity > 0 && validCount > ringCapacity
                ? UInt32(ringCapacity) : 0
            inkling.encodeAttentionPrefill(
                commandBuffer: cbA,
                q: qChunk,
                k: kv.keyBuffer(layer: L, validTokenCount: validCount),
                v: kv.valueBuffer(layer: L, validTokenCount: validCount),
                rel: relChunk,
                proj: relProj.buffer, projOffset: Int(relProj.offset),
                out: attentionChunk,
                headDim: UInt32(cfg.headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(cfg.numKVHeads),
                startPosition: UInt32(startPosition),
                rows: UInt32(N),
                slidingWindow: isFull ? 0 : UInt32(cfg.slidingWindow),
                relExtent: UInt32(isFull ? rel.extent : cfg.slidingWindow),
                dRel: UInt32(rel.dRel),
                ringCapacity: activeRing,
                logScalingFloor: UInt32(rel.logScalingFloor),
                scale: Float(cfg.attentionScale),
                logScalingAlpha: Float(rel.logScalingAlpha))
            // Q is dead after attention, so reuse its chunk allocation for O.
            encodeProjection(cbA, o, x: attentionChunk, y: qChunk,
                             rows: Dh, columns: Int(qDim))
            inkling.encodeSconvPrefillResidual(
                commandBuffer: cbA,
                x: qChunk, state: conv.attn,
                weight: attnSconvW.buffer,
                weightOffset: Int(attnSconvW.offset),
                hidden: hiddenChunk,
                channels: D, rows: UInt32(N))
            inkling.encodeRMSF32Prefill(commandBuffer: cbA,
                                        x: hiddenChunk,
                                        weight: mlpNorm.buffer,
                                        weightOffset: Int(mlpNorm.offset),
                                        out: routedXChunk,
                                        d: D, rows: UInt32(N), eps: eps)
            if isDense {
                for i in 0..<N {
                    let proj = inklingDenseProjections[L]!
                    let gain = model.inklingScalar(
                        try model.inklingDenseGlobalScale(layer: L))
                    try! shared.encode(commandBuffer: cbA,
                                       x: routedXChunk, xOffset: xOff(i),
                                       gate: proj.gate, up: proj.up, down: proj.down,
                                       y: h2Buf,
                                       scratchGate: denseScratchGate,
                                       scratchUp: denseScratchUp,
                                       scratchAct: denseScratchAct)
                    inkling.encodeScale(commandBuffer: cbA, x: h2Buf,
                                        by: gain, count: D)
                    inkling.encodeSconvStepF32Out(commandBuffer: cbA,
                                                  x: h2Buf, state: conv.mlp,
                                                  weight: mlpSconvW.buffer,
                                                  weightOffset: Int(mlpSconvW.offset),
                                                  out: inklingDeltaF32!,
                                                  channels: D, kernelSize: sconvK)
                    inkling.encodeResidualAddF32Delta(commandBuffer: cbA,
                                                      hidden: hiddenChunk,
                                                      hiddenOffset: hOff(i),
                                                      delta: inklingDeltaF32!,
                                                      count: D)
                }
            } else {
                let routerW = try model.router(layer: L)
                let gateBias = try model.inklingGateBias(layer: L)
                let gateScale = try model.inklingGateGlobalScale(layer: L)
                inkling.encodeRouterPrefill(
                    commandBuffer: cbA,
                    weights: routerW.buffer,
                    weightsOffset: Int(routerW.offset),
                    hidden: routedXChunk,
                    effectiveScale: effectiveScaleBuffers[L],
                    logits: routerLogitsChunk,
                    gateBias: gateBias.buffer,
                    gateBiasOffset: Int(gateBias.offset),
                    globalScale: gateScale.buffer,
                    globalScaleOffset: Int(gateScale.offset),
                    outIndices: idxChunk,
                    outWeights: wChunk,
                    gammasOut: gammaChunk,
                    numRouted: UInt32(cfg.numExperts),
                    numShared: UInt32(cfg.numSharedExperts),
                    topK: UInt32(cfg.topKExperts),
                    routeScale: Float(cfg.routedScalingFactor)
                        / Self.inklingFFNPrescale,
                    d: D,
                    rows: UInt32(N))
            }
            cbA.commit()
            waitForCompletion(cbA)
            if isDense { continue }

            // CPU: routing table for the whole chunk, then expert-major GLUs.
            let idxPtr = idxChunk.contents().bindMemory(to: UInt32.self,
                                                        capacity: N * 8)
            let wPtr = wChunk.contents().bindMemory(to: Float16.self,
                                                    capacity: N * 8)
            let gPtr = gammaChunk.contents().bindMemory(to: Float.self,
                                                        capacity: N * 2)
            var expertTokens: [Int: [(Int, Float)]] = [:]
            for i in 0..<N {
                for s in 0..<cfg.topKExperts {
                    let e = min(Int(idxPtr[i * 8 + s]), cfg.numExperts - 1)
                    expertTokens[e, default: []].append((i, Float(wPtr[i * 8 + s])))
                }
            }
            let routedOffsets = model.routedExpertOffsets(layer: L)
            let F = UInt32(cfg.moeIntermediateSize)
            // Flatten every (token, expert) pair into one expert-grouped list,
            // uploaded once per chunk-layer. Each expert then costs two
            // dispatches over its whole token set instead of three GEMVs per
            // token: the v1 per-pair encoding spent ~250 us of dispatch
            // overhead against ~17 us of bandwidth, and it dominated prefill.
            //
            // Shared experts come first, then routed experts in ascending id —
            // the same accumulation order the per-pair path used, so the FP32
            // sum per token is unchanged.
            let tokPtr = pairTokens.contents().bindMemory(to: UInt32.self,
                                                          capacity: N * 8)
            let wgtPtr = pairWeights.contents().bindMemory(to: Float.self,
                                                           capacity: N * 8)
            struct ExpertPairRange { let expert: Int; let start: Int; let count: Int }
            var sharedRanges: [ExpertPairRange] = []
            var routedRanges: [ExpertPairRange] = []
            var cursor = 0
            for s2 in 0..<cfg.numSharedExperts {
                let start = cursor
                for i in 0..<N {
                    tokPtr[cursor] = UInt32(i)
                    wgtPtr[cursor] = gPtr[i * 2 + s2]
                    cursor += 1
                }
                sharedRanges.append(ExpertPairRange(expert: s2, start: start,
                                                    count: N))
            }
            for (e, pairs) in expertTokens.sorted(by: { $0.key < $1.key }) {
                let start = cursor
                for (i, w) in pairs {
                    tokPtr[cursor] = UInt32(i)
                    wgtPtr[cursor] = w
                    cursor += 1
                }
                routedRanges.append(ExpertPairRange(expert: e, start: start,
                                                    count: pairs.count))
            }

            func expertParams(_ range: ExpertPairRange,
                              base: Int) -> InklingPrefillExpertGLU.Params {
                InklingPrefillExpertGLU.Params(
                    d: D, f: F,
                    pairStart: UInt32(range.start),
                    pairCount: UInt32(range.count),
                    hiddenStride: D,
                    gateWOff: UInt32(base) + routedOffsets.gateWOff,
                    gateSOff: UInt32(base) + routedOffsets.gateSOff,
                    gateBOff: UInt32(base) + routedOffsets.gateBOff,
                    upWOff: UInt32(base) + routedOffsets.upWOff,
                    upSOff: UInt32(base) + routedOffsets.upSOff,
                    upBOff: UInt32(base) + routedOffsets.upBOff,
                    downWOff: UInt32(base) + routedOffsets.downWOff,
                    downSOff: UInt32(base) + routedOffsets.downSOff,
                    downBOff: UInt32(base) + routedOffsets.downBOff)
            }

            // Shared experts (resident; per-token gammas as the pair weights).
            // Their projections are separate tensors rather than one packed
            // blob, so the byte offsets come from the views directly.
            let sharedCB = ctx.queue.makeCommandBuffer()!
            for range in sharedRanges {
                let projs = inklingSharedProjections[L][range.expert]
                var p = InklingPrefillExpertGLU.Params(
                    d: D, f: F,
                    pairStart: UInt32(range.start),
                    pairCount: UInt32(range.count),
                    hiddenStride: D,
                    gateWOff: UInt32(projs.gate.weightsOffset),
                    gateSOff: UInt32(projs.gate.scalesOffset),
                    gateBOff: UInt32(projs.gate.biasesOffset),
                    upWOff: UInt32(projs.up.weightsOffset),
                    upSOff: UInt32(projs.up.scalesOffset),
                    upBOff: UInt32(projs.up.biasesOffset),
                    downWOff: UInt32(projs.down.weightsOffset),
                    downSOff: UInt32(projs.down.scalesOffset),
                    downBOff: UInt32(projs.down.biasesOffset))
                p.pairStart = UInt32(range.start)
                prefillGLU.encode(commandBuffer: sharedCB,
                                  hidden: routedXChunk,
                                  pairTokens: pairTokens,
                                  pairWeights: pairWeights,
                                  expertBuffer: projs.gate.weights,
                                  expertOffset: 0,
                                  act: actScratch,
                                  acc: accChunk,
                                  params: p)
            }
            sharedCB.commit()
            waitForCompletion(sharedCB)

            // Routed experts, each streamed once for the whole chunk.
            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let breakdown = Self.prefillBreakdownEnabled
            func now() -> UInt64 {
                breakdown ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0
            }
            // Depth-1 pipeline: while the GPU runs expert e's GLU, the CPU
            // plans and preads expert e+1 into slots the in-flight buffer
            // does not touch (`avoidingSlots`), so the fetch overlaps GPU
            // work. `acc`/`act` hazards need no explicit sync: consecutive
            // buffers on the serial queue are hazard-tracked on those
            // resources, so expert accumulation order — and therefore the
            // output — is byte-identical to the serialized loop. When slot
            // pressure leaves no avoiding plan (or the pipeline is switched
            // off), the next iteration drains first and demand-fetches,
            // which *is* the old serialized behavior.
            let tileScheduler = PrefillRoutedTileScheduler(
                config: Self.prefillRoutedTileSchedulerConfig)
            var pendingExpertCB: (cb: MTLCommandBuffer, slots: [Int])?
            var prefetched: (expert: Int, plan: RoutedExpertFetchPlan,
                             blob: TensorView)?
            // Error-path backstop: any throw below (a failed prefetch pread,
            // a drain surfacing a GPU error) must not leave the in-flight
            // buffer running against the persistent act/acc scratch — an
            // immediate reset or retry would race it. The success path
            // drains and clears `pendingExpertCB`, so this no-ops there.
            defer {
                if let pending = pendingExpertCB {
                    waitForCompletion(pending.cb)
                    pendingExpertCB = nil
                }
            }
            func drainPendingExpertCB() throws {
                guard let pending = pendingExpertCB else { return }
                pendingExpertCB = nil
                let tDrain = now()
                waitForCompletion(pending.cb)
                if breakdown { Self.prefillDrainNanos &+= now() - tDrain }
                if let error = pending.cb.error { throw error }
            }
            for (index, range) in routedRanges.enumerated() {
                let t0 = now()
                let blob: TensorView
                let blobSlots: [Int]
                if let ready = prefetched, ready.expert == range.expert {
                    blob = ready.blob
                    blobSlots = ready.plan.assignedSlots
                    prefetched = nil
                } else {
                    prefetched = nil
                    try drainPendingExpertCB()
                    guard let plan = try model.planRoutedExperts(
                        layer: L, experts: [range.expert]) else {
                        throw ModelError.routedExpertPlanUnavailable(layer: L)
                    }
                    blob = try await model.fetchRoutedExperts(plan: plan)[0]
                    blobSlots = plan.assignedSlots
                }
                let t1 = now()
                let cb = ctx.queue.makeCommandBuffer()!
                prefillGLU.encode(commandBuffer: cb,
                                  hidden: routedXChunk,
                                  pairTokens: pairTokens,
                                  pairWeights: pairWeights,
                                  expertBuffer: blob.buffer,
                                  expertOffset: 0,
                                  act: actScratch,
                                  acc: accChunk,
                                  params: expertParams(range, base: Int(blob.offset)))
                cb.commit()
                let t2 = now()
                if breakdown {
                    Self.prefillFetchNanos &+= t1 - t0
                    Self.prefillEncodeNanos &+= t2 - t1
                    Self.prefillExpertCount &+= 1
                }
                try drainPendingExpertCB()
                pendingExpertCB = (cb, blobSlots)

                if Self.inklingPrefillPipelineEnabled,
                   index + 1 < routedRanges.count {
                    let nextExpert = routedRanges[index + 1].expert
                    let nextPlan = try model.planRoutedExpertsIfPossible(
                        layer: L,
                        experts: [nextExpert],
                        avoidingSlots: Set(blobSlots))
                    let decision = tileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingAssignedSlots: blobSlots,
                            avoidingSlotPlanAvailable: nextPlan != nil))
                    if case .prefetchNext = decision, let nextPlan {
                        let tFetch = now()
                        let nextBlob =
                            try await model.fetchRoutedExperts(plan: nextPlan)[0]
                        if breakdown { Self.prefillFetchNanos &+= now() - tFetch }
                        prefetched = (nextExpert, nextPlan, nextBlob)
                    }
                }
            }
            try drainPendingExpertCB()
            totalIoNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart

            // Tail: one causal channel-wise dispatch replaces N narrowing,
            // short-conv, and residual-add dispatch triplets.
            let cbC = ctx.queue.makeCommandBuffer()!
            inkling.encodeF32ToF16(commandBuffer: cbC,
                                   src: accChunk, srcOffset: 0,
                                   dst: qChunk, count: UInt32(N * Dh))
            inkling.encodeSconvPrefillResidual(
                commandBuffer: cbC,
                x: qChunk, state: conv.mlp,
                weight: mlpSconvW.buffer,
                weightOffset: Int(mlpSconvW.offset),
                hidden: hiddenChunk,
                channels: D, rows: UInt32(N),
                scale: Self.inklingFFNPrescale)
            cbC.commit()
            waitForCompletion(cbC)
        }

        if emitHead {
            let fNorm = model.finalNorm
            let lm = model.lmHead
            let validVocab = UInt32(cfg.unpaddedVocabSize > 0
                ? cfg.unpaddedVocabSize : cfg.vocabSize)
            let muP = Float(1.0 / cfg.logitsWidthMultiplier)
            inkling.resetNonFiniteLogitCount()
            runSync { cb in
                inkling.encodeRMSF32(commandBuffer: cb,
                                     x: hiddenChunk, xOffset: hOff(N - 1),
                                     weight: fNorm.buffer,
                                     weightOffset: Int(fNorm.offset),
                                     out: self.normed, d: D, eps: eps)
                self.int4.encode(commandBuffer: cb,
                                 weights: lm.buffer, weightsOffset: Int(lm.offset),
                                 scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                                 biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                                 x: self.normed, y: logits,
                                 m: validVocab, n: D)
                inkling.encodeHeadEpilogue(commandBuffer: cb,
                                           logits: logits,
                                           scale: muP,
                                           validVocab: validVocab,
                                           totalVocab: UInt32(self.cfg.vocabSize))
            }
            try checkFiniteLogits(inkling, position: startPosition + N - 1)
            if useFusedGreedyHead && outputMode == .greedyIfAvailable {
                let lp = logits.contents()
                    .bindMemory(to: Float16.self, capacity: Int(validVocab))
                var best: (Int, Float) = (-1, -.infinity)
                for i in 0..<Int(validVocab) {
                    let vl = Float(lp[i])
                    if vl > best.1 { best = (i, vl) }
                }
                guard best.0 >= 0 else {
                    throw InklingHeadError.noFiniteLogit(
                        position: startPosition + N - 1)
                }
                lastGreedyToken = UInt32(best.0)
            }
        }
        kv.advance(by: N)
    }

    /// Gated-DeltaNet linear attention (layer mask 2), one decode step.
    /// Reads `normed`, updates the layer's recurrent state + conv tail in
    /// place, and leaves the attention-branch output in `oOut`.
    private func encodeLinearAttentionDecode(_ cb: MTLCommandBuffer, layer L: Int) throws {
        guard let gdn, let gdnState, let gdnQKVRaw, let gdnConvOut,
              let gdnZ, let gdnA, let gdnB, let gdnY, let gdnOut else {
            preconditionFailure("linear-attention layer without GDN kernels")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)
        let qkvW = try model.linearInProjQKV(layer: L)
        let zW = try model.linearInProjZ(layer: L)
        let aW = try model.linearInProjA(layer: L)
        let bW = try model.linearInProjB(layer: L)
        let outW = try model.linearOutProj(layer: L)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.linearALog(layer: L)
        let dtBias = try model.linearDtBias(layer: L)
        let gatedNormW = try model.linearNorm(layer: L)

        // One dispatch over the concatenated qkv/z/a/b row space instead of four
        // separate GEMVs (a and b were 4 threadgroups each).
        gdn.encodeInputProjections(commandBuffer: cb,
                                   x: normed,
                                   qkv: qkvW, qkvOut: gdnQKVRaw,
                                   z: zW, zOut: gdnZ,
                                   a: aW, aOut: gdnA,
                                   b: bW, bOut: gdnB,
                                   hiddenSize: cfg.hiddenSize)

        gdn.encodeConvDecode(commandBuffer: cb,
                             tail: gdnState.convTailBuffer(layer: L),
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
            state: gdnState.stateBuffer(layer: L),
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
                                      state: gdnState.stateBuffer(layer: L),
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
    /// (no V norm), NeoX sub-dim RoPE, full attention with the configured
    /// scale, sigmoid output gate, then o_proj into `oOut`.
    private func encodeGatedFullAttentionDecode(_ cb: MTLCommandBuffer,
                                                layer L: Int,
                                                position: Int,
                                                seqLen: UInt32) throws {
        guard let elementwise, let rope, let qPackedScratch, let attnGateScratch else {
            preconditionFailure("attn_output_gate layer without gate kernels")
        }
        guard let kv else {
            preconditionFailure("FP16 attention requires an FP16 KV cache")
        }
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot = kv.kSlot(layer: L, position: position)
        let vSlot = kv.vSlot(layer: L, position: position)
        let q = try model.qProj(layer: L)
        let k = try model.kProj(layer: L)
        let v = try model.vProj(layer: L)
        let o = try model.oProj(layer: L)
        let qNormW = try model.qNorm(layer: L)
        let kNormW = try model.kNorm(layer: L)
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
                               eps: eps)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: kSlot.buffer, xOffset: kSlot.offset,
                               weight: kNormW.buffer,
                               weightOffset: Int(kNormW.offset),
                               out: kSlot.buffer, outOffset: kSlot.offset,
                               headDim: UInt32(headDim),
                               numHeads: numKV,
                               eps: eps)
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

    /// DeepSeek-V4 decode: 4 mHC residual streams, shared-KV MQA attention
    /// over [window ring ‖ compressed entries], softmax-gated window
    /// compressors, lightning-indexer selection past `index_topk` entries,
    /// sqrtsoftplus/hash top-6 routing, and INT2 streamed experts. Mirrors
    /// `produceToken`'s cb1 → readback → I/O-overlap → routed-cb pipeline.
    private func produceTokenDSV4(token: Int32,
                                  position: Int,
                                  into logits: MTLBuffer,
                                  emitHead: Bool,
                                  outputMode: PrefillOutputMode) async throws {
        guard let dsv4, let moeDSV4, let dsv4State,
              let streams = dsv4Streams, let streamsAlt = dsv4StreamsAlt,
              let qaBuf = dsv4QA, let oGrouped = dsv4OGrouped,
              let hcPreA = dsv4HCPreA, let hcPostA = dsv4HCPostA,
              let hcCombA = dsv4HCCombA,
              let hcPreF = dsv4HCPreF, let hcPostF = dsv4HCPostF,
              let hcCombF = dsv4HCCombF,
              let indexerQ = dsv4IndexerQ, let indexerW = dsv4IndexerW,
              let indexerScores = dsv4IndexerScores,
              let selected = dsv4Selected else {
            preconditionFailure("DeepSeek-V4 decode without DSV4 kernels")
        }
        guard model.routedExpertWeightBits == 2 else {
            throw PrefillError.prefillCursorMismatch(
                "DeepSeek-V4 runtime supports 2-bit routed experts; manifest says \(model.routedExpertWeightBits)")
        }
        let D = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let FShared = UInt32(cfg.intermediateSize)
        let eps: Float = 1e-6
        let ca = cfg.compressedAttention
        let hc = cfg.hyperConnections
        let headDim = cfg.fullHeadDim
        let numHeads = cfg.numHeads
        let fp16 = MemoryLayout<Float16>.stride

        struct PendingRoutedCommand {
            let cb: MTLCommandBuffer
            /// The layer's attention+router+shared-expert buffer. Always
            /// completed before `cb` (same queue, committed first); retained
            /// only so its error surfaces.
            let attentionCB: MTLCommandBuffer?
            let phase1HitCB: MTLCommandBuffer?
            let encodeAndCommitNanos: UInt64
        }
        var pendingRouted: PendingRoutedCommand?
        func finishPending(_ pending: PendingRoutedCommand, waitIfNeeded: Bool) {
            if waitIfNeeded {
                if let attentionCB = pending.attentionCB { waitForCompletion(attentionCB) }
                if let hitCB = pending.phase1HitCB { waitForCompletion(hitCB) }
                waitForCompletion(pending.cb)
            } else if let err = pending.cb.error {
                print("CB error: \(err)")
            }
            totalCb2Nanos &+= pending.encodeAndCommitNanos
        }
        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }
        func gemv(_ cb: MTLCommandBuffer, _ view: TensorView,
                  x: MTLBuffer, xOffset: Int = 0,
                  y: MTLBuffer, yOffset: Int = 0,
                  m: Int, n: Int) {
            int4.encode(commandBuffer: cb,
                        weights: view.buffer, weightsOffset: Int(view.offset),
                        scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                        biases: view.buffer, biasesOffset: Int(view.biasOffset),
                        x: x, xOffset: xOffset,
                        y: y, yOffset: yOffset,
                        m: UInt32(m), n: UInt32(n))
        }

        // Embed, then broadcast into the residual streams.
        let emb = model.embedding
        runSync { cb in
            embedInt4.encode(commandBuffer: cb,
                             table: emb.buffer, tableOffset: Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden,
                             tokenId: UInt32(bitPattern: token),
                             d: D,
                             outScale: 1.0)
            dsv4.encodeBroadcastStreams(commandBuffer: cb, x: hidden,
                                        streams: streams,
                                        hcMult: hc.mult, hidden: cfg.hiddenSize)
        }

        for L in 0..<cfg.numLayers {
            let isCSA = cfg.layerIsCSA(L)
            let isHCA = cfg.layerIsHCA(L)
            let isCompressed = isCSA || isHCA
            let ropeKind: DSV4Kernels.RopeKind = isCompressed ? .compress : .main
            var counters = dsv4State.counters[L]

            let inNorm = try model.inputNorm(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let qaView = try model.dsv4QAProj(layer: L)
            let qaNormView = try model.dsv4QANorm(layer: L)
            let qbView = try model.dsv4QBProj(layer: L)
            let kvView = try model.dsv4KVProj(layer: L)
            let kvNormView = try model.dsv4KVNorm(layer: L)
            let oaView = try model.dsv4OAProj(layer: L)
            let obView = try model.dsv4OBProj(layer: L)
            let sinksView = try model.dsv4Sinks(layer: L)
            let attnFn = try model.dsv4AttnHCFn(layer: L)
            let attnBase = try model.dsv4AttnHCBase(layer: L)
            let attnScale3 = try model.dsv4AttnHCScale(layer: L)
            let ffnFn = try model.dsv4FFNHCFn(layer: L)
            let ffnBase = try model.dsv4FFNHCBase(layer: L)
            let ffnScale3 = try model.dsv4FFNHCScale(layer: L)
            let routerW = try model.router(layer: L)
            let sharedProj = sharedExpertProjections[L]

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            var cb = ctx.queue.makeCommandBuffer()!

            // Attention-site mHC: weights + collapse, then the input norm.
            dsv4.encodeHCWeights(commandBuffer: cb, streams: streams,
                                 fn: attnFn.buffer, fnOffset: Int(attnFn.offset),
                                 base: attnBase.buffer, baseOffset: Int(attnBase.offset),
                                 scale: attnScale3.buffer, scaleOffset: Int(attnScale3.offset),
                                 outPre: hcPreA, outPost: hcPostA, outComb: hcCombA,
                                 hcMult: hc.mult, hidden: cfg.hiddenSize,
                                 sinkhornIters: hc.sinkhornIters,
                                 hcEps: Float(hc.eps), rmsEps: eps)
            dsv4.encodeHCCollapse(commandBuffer: cb, streams: streams,
                                  pre: hcPreA, x: hidden,
                                  hcMult: hc.mult, hidden: cfg.hiddenSize)
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed, d: D, eps: eps)

            // Q path: low-rank down + norm + up, per-head unweighted norm,
            // trailing interleaved RoPE.
            gemv(cb, qaView, x: normed, y: qaBuf, m: ca.qLoraRank, n: cfg.hiddenSize)
            rms.encodeBF16W(commandBuffer: cb, x: qaBuf,
                            weight: qaNormView.buffer,
                            weightOffset: Int(qaNormView.offset),
                            out: qaBuf, d: UInt32(ca.qLoraRank), eps: eps)
            gemv(cb, qbView, x: qaBuf, y: qScratch,
                 m: numHeads * headDim, n: ca.qLoraRank)
            rms.encodeNoScalePerHead(commandBuffer: cb, x: qScratch,
                                     out: qScratch,
                                     headDim: UInt32(headDim),
                                     numHeads: numHeads, eps: eps)
            dsv4.encodeRope(commandBuffer: cb, x: qScratch,
                            numHeads: numHeads, headDim: headDim,
                            ropeDim: ca.ropeHeadDim,
                            position: position, rope: ropeKind, direction: 1)

            // Shared K=V row straight into the window ring slot, then norm +
            // RoPE in place.
            let slot = dsv4State.windowSlot(position: position)
            let ring = dsv4State.windowKV[L]
            let slotOffset = slot * headDim * fp16
            gemv(cb, kvView, x: normed, y: ring, yOffset: slotOffset,
                 m: headDim, n: cfg.hiddenSize)
            rms.encodeBF16W(commandBuffer: cb, x: ring, xOffset: slotOffset,
                            weight: kvNormView.buffer,
                            weightOffset: Int(kvNormView.offset),
                            out: ring, outOffset: slotOffset,
                            d: UInt32(headDim), eps: eps)
            dsv4.encodeRope(commandBuffer: cb, x: ring, xOffset: slotOffset,
                            numHeads: 1, headDim: headDim,
                            ropeDim: ca.ropeHeadDim,
                            position: position, rope: ropeKind, direction: 1)

            // Compressor: project this token's kv/gate rows into the pending
            // window; emit one compressed entry when the window fills. The
            // emitted entry is visible to this token's attention (the
            // reference runs the compressor before core attention).
            var willEmit = false
            var willEmitIndexer = false
            if isCompressed {
                let rate = isCSA ? ca.csaCompressRate : ca.hcaCompressRate
                let rowWidth = isCSA ? 2 * headDim : headDim
                let compKV = try model.dsv4CompressorKVProj(layer: L)
                let compGate = try model.dsv4CompressorGateProj(layer: L)
                let compNorm = try model.dsv4CompressorKVNorm(layer: L)
                let compBias = try model.dsv4CompressorPositionBias(layer: L)
                // The GEMVs land directly in the pending window at this
                // token's row offset — no staging row, no blit.
                let rowOffset = counters.pendingRows * rowWidth * fp16
                gemv(cb, compKV, x: normed,
                     y: dsv4State.pendingKV[L]!, yOffset: rowOffset,
                     m: rowWidth, n: cfg.hiddenSize)
                gemv(cb, compGate, x: normed,
                     y: dsv4State.pendingGate[L]!, yOffset: rowOffset,
                     m: rowWidth, n: cfg.hiddenSize)
                willEmit = counters.pendingRows + 1 == rate
                if willEmit {
                    let entry = counters.compressedEntries
                    dsv4.encodeCompressEmit(
                        commandBuffer: cb,
                        pendingKV: dsv4State.pendingKV[L]!,
                        pendingGate: dsv4State.pendingGate[L]!,
                        priorCaKV: dsv4State.priorCaKV[L] ?? dsv4State.pendingKV[L]!,
                        priorCaGate: dsv4State.priorCaGate[L] ?? dsv4State.pendingGate[L]!,
                        positionBias: compBias.buffer,
                        positionBiasOffset: Int(compBias.offset),
                        normWeight: compNorm.buffer,
                        normWeightOffset: Int(compNorm.offset),
                        outEntry: dsv4State.compressedKV[L]!,
                        outEntryOffset: entry * headDim * fp16,
                        nextPriorCaKV: dsv4State.priorCaKV[L] ?? dsv4State.pendingKV[L]!,
                        nextPriorCaGate: dsv4State.priorCaGate[L] ?? dsv4State.pendingGate[L]!,
                        rate: rate, dim: headDim, dual: isCSA,
                        hasPrior: counters.hasPrior, eps: eps)
                    dsv4.encodeRope(commandBuffer: cb,
                                    x: dsv4State.compressedKV[L]!,
                                    xOffset: entry * headDim * fp16,
                                    numHeads: 1, headDim: headDim,
                                    ropeDim: ca.ropeHeadDim,
                                    position: entry * rate,
                                    rope: .compress,
                                    direction: 1)
                }
                if isCSA {
                    let idxRate = ca.csaCompressRate
                    let idxDim = ca.indexHeadDim
                    let idxKV = try model.dsv4IndexerKVProj(layer: L)
                    let idxGate = try model.dsv4IndexerGateProj(layer: L)
                    let idxNorm = try model.dsv4IndexerKVNorm(layer: L)
                    let idxBias = try model.dsv4IndexerPositionBias(layer: L)
                    let idxRowOffset = counters.indexerPendingRows * 2 * idxDim * fp16
                    gemv(cb, idxKV, x: normed,
                         y: dsv4State.indexerPendingKV[L]!, yOffset: idxRowOffset,
                         m: 2 * idxDim, n: cfg.hiddenSize)
                    gemv(cb, idxGate, x: normed,
                         y: dsv4State.indexerPendingGate[L]!, yOffset: idxRowOffset,
                         m: 2 * idxDim, n: cfg.hiddenSize)
                    willEmitIndexer = counters.indexerPendingRows + 1 == idxRate
                    if willEmitIndexer {
                        let entry = counters.indexerEntries
                        dsv4.encodeCompressEmit(
                            commandBuffer: cb,
                            pendingKV: dsv4State.indexerPendingKV[L]!,
                            pendingGate: dsv4State.indexerPendingGate[L]!,
                            priorCaKV: dsv4State.indexerPriorCaKV[L]!,
                            priorCaGate: dsv4State.indexerPriorCaGate[L]!,
                            positionBias: idxBias.buffer,
                            positionBiasOffset: Int(idxBias.offset),
                            normWeight: idxNorm.buffer,
                            normWeightOffset: Int(idxNorm.offset),
                            outEntry: dsv4State.indexerKeys[L]!,
                            outEntryOffset: entry * idxDim * fp16,
                            nextPriorCaKV: dsv4State.indexerPriorCaKV[L]!,
                            nextPriorCaGate: dsv4State.indexerPriorCaGate[L]!,
                            rate: idxRate, dim: idxDim, dual: true,
                            hasPrior: counters.indexerHasPrior, eps: eps)
                        dsv4.encodeRope(commandBuffer: cb,
                                        x: dsv4State.indexerKeys[L]!,
                                        xOffset: entry * idxDim * fp16,
                                        numHeads: 1, headDim: idxDim,
                                        ropeDim: ca.ropeHeadDim,
                                        position: entry * idxRate,
                                        rope: .compress,
                                        direction: 1)
                    }
                }
            }

            let compressedCount = isCompressed
                ? counters.compressedEntries + (willEmit ? 1 : 0)
                : 0

            // Lightning-indexer selection is only needed once the compressed
            // entries exceed index_topk (context > topk * rate); the extra
            // CPU sync point exists only on those layers/positions.
            let needsSelection = isCSA && compressedCount > ca.indexTopK
            var selectedCount = DSV4Kernels.selectAll
            if needsSelection {
                let idxQB = try model.dsv4IndexerQBProj(layer: L)
                let idxWProj = try model.dsv4IndexerWeightsProj(layer: L)
                let idxEntries = counters.indexerEntries + (willEmitIndexer ? 1 : 0)
                gemv(cb, idxQB, x: qaBuf, y: indexerQ,
                     m: ca.indexNHeads * ca.indexHeadDim, n: ca.qLoraRank)
                dsv4.encodeRope(commandBuffer: cb, x: indexerQ,
                                numHeads: ca.indexNHeads,
                                headDim: ca.indexHeadDim,
                                ropeDim: ca.ropeHeadDim,
                                position: position,
                                rope: .compress,
                                direction: 1)
                gemv(cb, idxWProj, x: normed, y: indexerW,
                     m: ca.indexNHeads, n: cfg.hiddenSize)
                dsv4.encodeIndexerScore(
                    commandBuffer: cb, q: indexerQ,
                    keys: dsv4State.indexerKeys[L]!,
                    weights: indexerW, scores: indexerScores,
                    numHeads: ca.indexNHeads, indexDim: ca.indexHeadDim,
                    entryCount: idxEntries,
                    headScale: 1.0 / Float(ca.indexHeadDim).squareRoot(),
                    weightScale: 1.0 / Float(ca.indexNHeads).squareRoot())
                cb.commit()
                waitForCompletion(cb)
                // CPU top-k over the scores; attention gathers the same
                // entry indices from the compressor cache (both caches share
                // the emission schedule).
                let scoresPtr = indexerScores.contents()
                    .assumingMemoryBound(to: Float.self)
                let k = min(ca.indexTopK, compressedCount)
                var order = Array(0..<compressedCount)
                order.sort { scoresPtr[$0] > scoresPtr[$1] }
                let picks = order.prefix(k).sorted()
                let selPtr = selected.contents().assumingMemoryBound(to: UInt32.self)
                for (i, e) in picks.enumerated() { selPtr[i] = UInt32(e) }
                selectedCount = UInt32(k)
                cb = ctx.queue.makeCommandBuffer()!
            }

            // Core attention + conjugate output rotation.
            dsv4.encodeAttention(
                commandBuffer: cb, q: qScratch,
                windowKV: ring,
                compressedKV: dsv4State.compressedKV[L] ?? ring,
                selected: selected,
                sinks: sinksView.buffer, sinksOffset: Int(sinksView.offset),
                out: attnOut,
                headDim: headDim, numHeads: numHeads,
                windowCount: dsv4State.windowCount(position: position),
                windowStartPos: dsv4State.windowStartPosition(position: position),
                ringCapacity: dsv4State.ringCapacity,
                compressedCount: compressedCount,
                selectedCount: selectedCount,
                scale: Float(cfg.attentionScale))
            dsv4.encodeRope(commandBuffer: cb, x: attnOut,
                            numHeads: numHeads, headDim: headDim,
                            ropeDim: ca.ropeHeadDim,
                            position: position, rope: ropeKind, direction: -1)

            // Grouped low-rank output projection: one dispatch covering every
            // head group (the output row index selects both the weight block
            // and the activation slice), then the mixing projection.
            dsv4.encodeOGroupProjection(
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

            // Attention-site placement + residual mix: streams -> streamsAlt.
            dsv4.encodeHCPlaceMix(commandBuffer: cb, streams: streams,
                                  sub: oOut, post: hcPostA, comb: hcCombA,
                                  outStreams: streamsAlt,
                                  hcMult: hc.mult, hidden: cfg.hiddenSize)

            // FFN-site mHC + norm, producing the MoE input.
            dsv4.encodeHCWeights(commandBuffer: cb, streams: streamsAlt,
                                 fn: ffnFn.buffer, fnOffset: Int(ffnFn.offset),
                                 base: ffnBase.buffer, baseOffset: Int(ffnBase.offset),
                                 scale: ffnScale3.buffer, scaleOffset: Int(ffnScale3.offset),
                                 outPre: hcPreF, outPost: hcPostF, outComb: hcCombF,
                                 hcMult: hc.mult, hidden: cfg.hiddenSize,
                                 sinkhornIters: hc.sinkhornIters,
                                 hcEps: Float(hc.eps), rmsEps: eps)
            dsv4.encodeHCCollapse(commandBuffer: cb, streams: streamsAlt,
                                  pre: hcPreF, x: hidden,
                                  hcMult: hc.mult, hidden: cfg.hiddenSize)
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: postAttn.buffer,
                            weightOffset: Int(postAttn.offset),
                            out: routedX, d: D, eps: eps)

            // Router. Hash layers take their expert set from tid2eid[token]
            // (written CPU-side before commit); the gate only weights them.
            let isHash = cfg.layerIsHashRouted(L)
            if isHash {
                let table = try model.dsv4HashTable(layer: L)
                // The table rides the resident file in its source dtype:
                // the real checkpoint ships I64 [vocab, topK]; synthetic
                // fixtures may use U32. Read CPU-side, dtype-aware.
                let base = table.buffer.contents().advanced(by: Int(table.offset))
                let row = min(max(Int(token), 0), cfg.vocabSize - 1) * cfg.topKExperts
                let idxPtr = outIndices.contents().assumingMemoryBound(to: UInt32.self)
                let expertCap = UInt32(cfg.numExperts - 1)
                if table.dtype == 4 {
                    let tPtr = base.assumingMemoryBound(to: Int64.self)
                    for i in 0..<cfg.topKExperts {
                        idxPtr[i] = min(UInt32(clamping: max(0, tPtr[row + i])), expertCap)
                    }
                } else {
                    let tPtr = base.assumingMemoryBound(to: UInt32.self)
                    for i in 0..<cfg.topKExperts {
                        idxPtr[i] = min(tPtr[row + i], expertCap)
                    }
                }
                moeDSV4.encodeRouterHashWeights(
                    commandBuffer: cb,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    hidden: routedX,
                    onesScale: effectiveScaleBuffers[L],
                    indices: outIndices,
                    outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D)
            } else {
                let bias = try model.dsv4RouterCorrectionBias(layer: L)
                moeDSV4.encodeRouterTopK(
                    commandBuffer: cb,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    hidden: routedX,
                    onesScale: effectiveScaleBuffers[L],
                    correctionBias: bias.buffer, correctionBiasOffset: Int(bias.offset),
                    outIndices: outIndices,
                    outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D)
            }

            // PILOT lookahead: run layer L+1's router against *this* layer's
            // post-attention state, into private buffers, before the signal.
            // The CPU then reads the real L indices and the speculative L+1
            // indices at one wake — no extra command buffer, no extra sync.
            // The mHC streams mean L+1's true router input also carries this
            // layer's FFN contribution, so this is an approximation whose
            // recall the predicted/confirmed counters measure.
            var pilotGemvEncoded = false
            if shouldEncodePilotGemv(nextLayer: L + 1),
               let pilot = ensurePilotRouter(),
               let nextRouter = try? model.router(layer: L + 1),
               let nextBias = try? model.dsv4RouterCorrectionBias(layer: L + 1) {
                pilot.encodePrediction(
                    commandBuffer: cb,
                    weights: nextRouter.buffer, weightsOffset: Int(nextRouter.offset),
                    hidden: routedX,
                    onesScale: effectiveScaleBuffers[L + 1],
                    correctionBias: nextBias.buffer,
                    correctionBiasOffset: Int(nextBias.offset),
                    numExperts: UInt32(cfg.numExperts), d: D,
                    routeScale: Float(cfg.routedScalingFactor))
                pilotGemvEncoded = true
            }

            // The router indices are written at this point in the buffer, so
            // signal the CPU here and keep encoding into the SAME buffer: the
            // shared expert (clamped SwiGLU) depends only on `routedX`, never
            // on the routed experts, so the GPU runs it while the CPU wakes,
            // plans slots and preads the routed-expert blobs.
            let routerSignal = encodeRouterSignal(cb)
            int4.encode(commandBuffer: cb,
                        weights: sharedProj.gate.weights,
                        weightsOffset: sharedProj.gate.weightsOffset,
                        scales: sharedProj.gate.scales,
                        scalesOffset: sharedProj.gate.scalesOffset,
                        biases: sharedProj.gate.biases,
                        biasesOffset: sharedProj.gate.biasesOffset,
                        x: routedX, y: denseScratchGate,
                        m: FShared, n: D)
            int4.encode(commandBuffer: cb,
                        weights: sharedProj.up.weights,
                        weightsOffset: sharedProj.up.weightsOffset,
                        scales: sharedProj.up.scales,
                        scalesOffset: sharedProj.up.scalesOffset,
                        biases: sharedProj.up.biases,
                        biasesOffset: sharedProj.up.biasesOffset,
                        x: routedX, y: denseScratchUp,
                        m: FShared, n: D)
            dsv4.encodeSwigluClampMul(commandBuffer: cb,
                                      gate: denseScratchGate,
                                      up: denseScratchUp,
                                      out: denseScratchAct,
                                      n: cfg.intermediateSize,
                                      limit: Float(cfg.swigluLimit))
            int4.encode(commandBuffer: cb,
                        weights: sharedProj.down.weights,
                        weightsOffset: sharedProj.down.weightsOffset,
                        scales: sharedProj.down.scales,
                        scalesOffset: sharedProj.down.scalesOffset,
                        biases: sharedProj.down.biases,
                        biasesOffset: sharedProj.down.biasesOffset,
                        x: denseScratchAct, y: h1Buf,
                        m: D, n: FShared)
            let overlapProbe = Self.phaseInstrumentationEnabled ? GPUOverlapProbe() : nil
            overlapProbe?.track(cb)
            cb.commit()
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            waitForRouterSignal(routerSignal, fallback: cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            if let pending = pendingRouted {
                finishPending(pending, waitIfNeeded: false)
                pendingRouted = nil
            }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            // Commit the compressor bookkeeping for this token.
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
            dsv4State.counters[L] = counters

            // Expert readback -> plan -> advise -> pread -> routed CB, the
            // same overlap structure as `produceToken`.
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }
            // Join any speculative read aimed at this layer before planning —
            // the plan must not evict a slot a background pread is filling —
            // and score the prediction. Then this layer's routing becomes the
            // next token's prediction for the same layer.
            settleSpeculation(layer: L, actualExperts: experts)
            recordRoutedExperts(experts, layer: L)
            capturePilotPrediction(nextLayer: L + 1,
                                   token: token,
                                   gemvEncoded: pilotGemvEncoded)

            let routedOffsets = model.routedExpertOffsets(layer: L)
            let gateGroupSize = UInt32(model.routedGateGroupSize(layer: L))
            guard let plannedFetch = try model.planRoutedExperts(layer: L, experts: experts)
            else {
                throw ModelError.routedExpertPlanUnavailable(layer: L)
            }
            var phase1HitCB: MTLCommandBuffer?
            var phase1HitSlots: [UInt32] = []
            var phase1MissSlots: [UInt32] = []
            var phase1HitSplitArgBuf: MTLBuffer?
            let missSet = Set(plannedFetch.misses)
            phase1HitSlots = (0..<cfg.topKExperts)
                .filter { !missSet.contains($0) }
                .map { UInt32($0) }
            phase1MissSlots = plannedFetch.misses.map { UInt32($0) }

            if plannedFetch.hits > 0, !plannedFetch.misses.isEmpty {
                let plannedBlobs = try model.routedExpertBuffers(for: plannedFetch)
                let bufs = plannedBlobs.map { (buffer: $0.buffer, offset: Int($0.offset)) }
                writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                let hitCB = ctx.queue.makeCommandBuffer()!
                let argBuf = moeDSV4.makeReusedRoutedArgumentBuffer(routedBlobs: bufs)
                phase1HitSplitArgBuf = argBuf
                moeDSV4.encodeRoutedPhase1Subset(
                    commandBuffer: hitCB,
                    routedArgBuffer: argBuf,
                    routedBlobs: bufs,
                    routedOffsets: routedOffsets,
                    x: routedX, acts: moeActs,
                    activeSlots: moeHitActiveSlots,
                    activeSlotIndices: phase1HitSlots,
                    activeCount: UInt32(phase1HitSlots.count),
                    d: D, f: FmoE,
                    gateGroupSize: gateGroupSize)
                phase1HitCB = hitCB
            }

            // Phase-1 for the resident (hit) experts needs no I/O, so it goes
            // to the GPU immediately, extending the window the pread hides
            // behind. It follows the shared expert on the same queue.
            if let hitCB = phase1HitCB {
                overlapProbe?.track(hitCB)
                hitCB.commit()
            }

            if rdadviseEnabled && rdadvisePolicyMode != .off {
                let requestedMisses = plannedFetch.misses.count
                let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                    layer: L, missCount: requestedMisses)
                if let skipped = shouldSkipRDAdvice(position: position,
                                                    requestedMisses: requestedMisses,
                                                    estimatedBytes: estimatedAdviceBytes,
                                                    canOverlapUsefulGPUWork: true) {
                    recordRDAdvice(skipped, wallNanos: 0)
                } else {
                    let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let result = try model.adviseRoutedExperts(plan: plannedFetch)
                    let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                    recordRDAdvice(result, wallNanos: wallNanos)
                    updateRDAdvicePolicy(after: result, position: position)
                }
            }

            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
            let tIoEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            totalIoNanos &+= tIoEnd - tIoStart
            recordExpertIOOverlap(probe: overlapProbe,
                                  startNanos: tIoStart,
                                  endNanos: tIoEnd)
            let routedBufs = blobs.map { (buffer: $0.buffer, offset: Int($0.offset)) }

            let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let routedCB = ctx.queue.makeCommandBuffer()!
            let argBuf = phase1HitSplitArgBuf
                ?? moeDSV4.makeReusedRoutedArgumentBuffer(routedBlobs: routedBufs)
            if phase1HitCB != nil {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                moeDSV4.encodeRoutedPhase1Subset(
                    commandBuffer: routedCB,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX, acts: moeActs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count),
                    d: D, f: FmoE,
                    gateGroupSize: gateGroupSize)
            } else {
                moeDSV4.encodeRoutedPhase1(
                    commandBuffer: routedCB,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX, acts: moeActs,
                    d: D, f: FmoE,
                    gateGroupSize: gateGroupSize)
            }
            // Phase-2 seeds with the shared-expert output, so h2Buf holds
            // routed + shared — exactly the reference's `routed +
            // shared_experts(residual)`.
            moeDSV4.encodeRoutedPhase2Reduce(
                commandBuffer: routedCB,
                routedArgBuffer: argBuf,
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                acts: moeActs,
                routingWeights: outWeights,
                residual: h1Buf,
                y: h2Buf,
                d: D, f: FmoE)
            // FFN-site placement + residual mix: streamsAlt -> streams.
            dsv4.encodeHCPlaceMix(commandBuffer: routedCB, streams: streamsAlt,
                                  sub: h2Buf, post: hcPostF, comb: hcCombF,
                                  outStreams: streams,
                                  hcMult: hc.mult, hidden: cfg.hiddenSize)
            routedCB.commit()
            // GPU is busy with routed work and the next layer's attention, CPU
            // is idle: the natural window for the speculative read.
            issueSpeculativePrefetch(layer: L + 1)
            precondition(pendingRouted == nil,
                         "routed command-buffer pipeline drained before queuing the next layer")
            pendingRouted = PendingRoutedCommand(
                cb: routedCB,
                attentionCB: cb,
                phase1HitCB: phase1HitCB,
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
        }
        if let pending = pendingRouted {
            finishPending(pending, waitIfNeeded: true)
            pendingRouted = nil
        }

        if emitHead {
            // Collapse the residual streams, then the standard head.
            let hhFn = try model.dsv4HyperHeadFn
            let hhBase = try model.dsv4HyperHeadBase
            let hhScale = try model.dsv4HyperHeadScale
            runSync { cb in
                dsv4.encodeHyperHead(commandBuffer: cb, streams: streams,
                                     fn: hhFn.buffer, fnOffset: Int(hhFn.offset),
                                     base: hhBase.buffer, baseOffset: Int(hhBase.offset),
                                     scale: hhScale.buffer, scaleOffset: Int(hhScale.offset),
                                     x: hidden,
                                     hcMult: hc.mult, hidden: cfg.hiddenSize,
                                     hcEps: Float(hc.eps), rmsEps: eps)
            }
            let fNorm = model.finalNorm
            let lm = model.lmHead
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if useFusedHeadForThisToken {
                runSync { cb in
                    self.fusionHead.encodeGreedyDecode(
                        commandBuffer: cb,
                        hidden: self.hidden,
                        normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                        weights: lm.buffer, weightsOffset: Int(lm.offset),
                        scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                        biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                        outToken: self.greedyTokenBuf,
                        d: D, vocab: UInt32(self.cfg.vocabSize),
                        rmsEps: eps)
                }
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                runSync { cb in
                    self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                         weight: fNorm.buffer,
                                         weightOffset: Int(fNorm.offset),
                                         out: self.normed, d: D, eps: eps)
                    self.int4.encode(commandBuffer: cb,
                                     weights: lm.buffer, weightsOffset: Int(lm.offset),
                                     scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                     biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                     x: self.normed, y: logits,
                                     m: UInt32(self.cfg.vocabSize), n: D)
                }
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv?.advance()
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) {
        let cb = ctx.queue.makeCommandBuffer()!
        body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("CB error: \(err)")
        }
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) {
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("CB error: \(err)")
        }
    }

    /// Encodes the router-readback signal at the current point in `cb`. Must be
    /// called with no encoder open and immediately after the kernel that writes
    /// `outIndices`, so that the GPU write is complete when the CPU wakes.
    /// Returns nil when no shared event exists (fall back to a full wait).
    private func encodeRouterSignal(_ cb: MTLCommandBuffer) -> UInt64? {
        guard let routerEvent else { return nil }
        routerEventValue &+= 1
        cb.encodeSignalEvent(routerEvent, value: routerEventValue)
        return routerEventValue
    }

    /// Passive wait for the router signal, never a spin loop: a busy CPU steals
    /// the shared SoC power budget and throttles the GPU on Apple silicon.
    /// Because command buffers on one queue execute in commit order, reaching
    /// this signal also proves every previously committed buffer has finished —
    /// which is what makes it safe for the caller to pread into expert slots.
    private func waitForRouterSignal(_ value: UInt64?, fallback cb: MTLCommandBuffer) {
        guard routerEventWaitEnabled else {
            waitForCompletion(cb)
            return
        }
        guard let routerEvent, let value else {
            waitForCompletion(cb)
            return
        }
        if routerEvent.signaledValue >= value { return }
        if !routerEvent.wait(untilSignaledValue: value,
                             timeoutMS: Self.routerEventTimeoutMS) {
            waitForCompletion(cb)
        }
    }

    /// Tracks when the command buffers that are supposed to hide a pread
    /// actually finished. Completion handlers run on a Metal thread, so the
    /// bookkeeping is lock-guarded.
    private func eagerFetchCompleted(_ seq: UInt64, success: Bool, layer: Int) {
        eagerFetchLock.lock()
        if !success, eagerFetchFailedLayer == nil {
            eagerFetchFailedLayer = layer
        }
        eagerFetchDone.insert(seq)
        while eagerFetchDone.contains(eagerFetchSignaled + 1) {
            eagerFetchDone.remove(eagerFetchSignaled + 1)
            eagerFetchSignaled += 1
        }
        // Signal inside the lock: two completions racing outside it could
        // write the event backwards.
        if let event = eagerFetchEvent, event.signaledValue < eagerFetchSignaled {
            event.signaledValue = eagerFetchSignaled
        }
        eagerFetchLock.unlock()
    }

    /// A failed eager fill leaves its slot unpublished while the gated GPU
    /// work runs against stale slab bytes; the step's output is garbage and
    /// must not reach the caller.
    private func throwIfEagerFillFailed() throws {
        eagerFetchLock.lock()
        let failedLayer = eagerFetchFailedLayer
        eagerFetchFailedLayer = nil
        eagerFetchLock.unlock()
        if let failedLayer {
            throw ModelError.eagerExpertFillFailed(layer: failedLayer)
        }
    }

    /// GPU-timeline attribution (`MFERENCE_PHASES=1`): busy time summed over
    /// tracked decode command buffers, plus the wall span they cover. The
    /// difference is scheduling gap — the target of command-buffer
    /// consolidation work.
    func trackGpuInterval(_ cb: MTLCommandBuffer) {
        guard Self.phaseInstrumentationEnabled else { return }
        cb.addCompletedHandler { [self] done in
            let start = done.gpuStartTime
            let end = done.gpuEndTime
            guard end > start else { return }
            gpuTimeLock.lock()
            totalGpuBusyNanos &+= UInt64((end - start) * 1e9)
            gpuSpanFirstStart = min(gpuSpanFirstStart, start)
            gpuSpanLastEnd = max(gpuSpanLastEnd, end)
            gpuTimeLock.unlock()
        }
    }

    public var totalGpuSpanNanos: UInt64 {
        gpuTimeLock.lock()
        defer { gpuTimeLock.unlock() }
        guard gpuSpanLastEnd > gpuSpanFirstStart else { return 0 }
        return UInt64((gpuSpanLastEnd - gpuSpanFirstStart) * 1e9)
    }

    final class GPUOverlapProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var remaining = 0
        private var lastNanos: UInt64 = 0

        /// Register a command buffer; call before committing it.
        func track(_ cb: MTLCommandBuffer) {
            lock.lock()
            remaining += 1
            lock.unlock()
            cb.addCompletedHandler { [self] _ in
                let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                lock.lock()
                remaining -= 1
                lastNanos = max(lastNanos, now)
                lock.unlock()
            }
        }

        /// When every tracked buffer has finished, the time the last one
        /// finished; nil while any is still running.
        var finishedNanos: UInt64? {
            lock.lock()
            defer { lock.unlock() }
            return remaining == 0 ? lastNanos : nil
        }
    }

}
