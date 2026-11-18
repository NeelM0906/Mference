import Foundation
import Metal

extension RealForwardRunner: RuntimeMemoryReporting {
    var diagnosticKVStateBytes: UInt64? {
        (kv?.diagnosticBufferBytes ?? 0) + (gdnState?.diagnosticBufferBytes ?? 0)
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

/// The real-forward runner for Qwen 3.6 (`.qwen36`): chunked layer-major
/// prefill and one-token decode with streamed routed experts, gated-DeltaNet
/// linear-attention layers and gated full-attention layers. Generic config
/// axes (FFN sandwich norms, sliding window, router scale, ...) keep their
/// paths. Direct against `Model`.
public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting, FusedHeadLogitProducer, @unchecked Sendable {
    struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// post_feedforward_layernorm_1; nil when the arch has no FFN
        /// sandwich norms.
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
    /// BF16 ones over [numExperts]; neutral per_expert_scale when the router
    /// has no auxiliary scale tensors.
    private let onesPerExpertScale: MTLBuffer?
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?
    /// Grouped-GEMM routed experts for well-filled Qwen 3.6 prefill tiles
    /// (`usesGroupedRoutedExpertGEMM`).
    private let prefillGroupedGEMM: MPPGroupedRoutedMoE?
    /// Test-only: false sends every routed tile to the row kernel, the control
    /// arm of the Qwen 3.6 prefill tolerance gate.
    private var prefillGroupedGEMMEnabled = true
    /// Routed prefill tiles that ran as grouped GEMM rather than per-row GEMV.
    private(set) var prefillGroupedExpertTiles = 0
    /// Simdgroup-matrix grouped routed experts for Qwen 3.6 prefill tiles on
    /// GPUs without the Apple10 TensorOps path
    /// (`usesSimdgroupGroupedRoutedExperts`). Ahead of the MPP kernel, which
    /// stays the fallback where this one is absent.
    private let prefillSimdGroupedGEMM: SimdgroupGroupedRoutedMoE?
    /// Test-only: false leaves routed tiles to MPP and the row kernel, as
    /// before this kernel (the control arm of its tolerance gate).
    private var prefillSimdGroupedGEMMEnabled = true
    /// Routed prefill tiles that ran on the simdgroup grouped kernel.
    private(set) var prefillSimdGroupedExpertTiles = 0
    /// Prefill attention layer chunks that ran the tensor-ops kernel. It falls
    /// back to the tiled or GQA kernels silently when the pipeline is missing.
    private(set) var prefillTensorOpsAttentionLayers = 0
    /// How the last prefill layer ran its shared expert. A batched layout that
    /// silently falls back to per-row dispatch costs a third of a long prefill.
    private(set) var lastPrefillSharedExpertPath: PrefillSharedExpert.BlockPath?
    /// Test-only failure seam. Prior layer work is drained at this boundary;
    /// production leaves it nil and adds no GPU synchronization.
    var prefillWillEncodeLayer: ((Int) throws -> Void)?

    /// Shared chunked-path stage timing, on only under the existing
    /// `MFERENCE_PREFILL_BREAKDOWN=1` diagnostic toggle (see `PrefillStageBreakdown`).
    /// Tests inject an instance to prove output is unchanged with it on.
    var prefillStageBreakdown: PrefillStageBreakdown? =
        RealForwardRunner.prefillBreakdownEnabled ? .shared : nil

    private static let rdadviseBoundedMissCap = 12
    private static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    private static let rdadviseAdaptiveMissCap = 12
    private static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    private static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    /// Two tiles of 16 experts in flight: the next tile's reads start while
    /// the GPU still holds two, so tile drains stop idling the SSD and each
    /// fetch has twice the experts in flight (Qwen 3.6, 506 tokens, 24 GiB
    /// M4 Pro: prefill 3.47-3.79 -> 3.18-3.42 s against depth 1 x 8).
    private static let prefillRoutedTileSchedulerConfig =
        PrefillRoutedTileSchedulerConfig(maxPendingDepth: 2, tileExperts: 16)

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
    struct SpeculativePrefetchPlan: Equatable {
        let mode: SpeculativePrefetchMode
        /// Shadow mode: speculative reads issued per layer.
        let shadowBudget: Int
    }

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

        /// The prefetch a runner starts with. An explicit `--shadow-budget`
        /// decides everything (0 turns speculation off, a positive value selects
        /// shadow prefetch with that budget); otherwise `MFERENCE_SPEC_PREFETCH`
        /// and `MFERENCE_SHADOW_BUDGET` apply; otherwise the family default.
        /// Every mode is byte-identical to no speculation.
        ///
        /// Qwen 3.6 on hosts from 16 GiB to under 24 GiB: shadow, budget 4; at
        /// 24 GiB, budget 8 (below). There the GPU idles about half of decode waiting for expert
        /// reads; on an M2 with 16 GiB (diagnostic runs, 2026-09-21) Qwen 3.6
        /// decoded 7.35 / 8.91 / 9.62 / 10.55 tok/s at off / 1 / 2 / 4, at no
        /// memory cost. On a 24 GiB M4 Pro (2026-10-27)
        /// Qwen 3.6 went 17.8-18.3 -> 19.6-20.6 tok/s at budget 4 and
        /// 14.8-20.3 -> 19.7-21.8 at budget 8, so 24 GiB hosts run budget 8.
        ///
        /// Qwen 3.6 above 24 GiB up to 32 GiB: shadow, budget 8. Only 32 GiB is
        /// measured; no current Mac sits between. On a 32 GiB M1 Max (diagnostic
        /// runs, 2026-11-17) long decode went 28.45-28.55 -> 30.10-30.19 tok/s,
        /// with budget 4 at 29.92-30.02. At 32K context decode stayed flat at
        /// budget 8 (23.88-24.41 -> 23.96): fewer expert misses, but GPU busy
        /// per token rose 25-29%, at budget 4 as at 8.
        ///
        /// Qwen 3.6 below 16 GiB or above 32 GiB is unmeasured and stays off,
        /// as does every other family.
        static func plan(requestedShadowBudget: Int?,
                         environment: [String: String],
                         family: ModelFamily,
                         physicalMemoryBytes: UInt64) -> SpeculativePrefetchPlan {
            let gib = UInt64(1) << 30
            let familyMode: SpeculativePrefetchMode
            let familyBudget: Int
            switch family {
            case .qwen36:
                let measuredCeiling = 32 * gib
                let measuredTier = physicalMemoryBytes >= 16 * gib && physicalMemoryBytes <= measuredCeiling
                familyMode = measuredTier ? .shadow : .off
                familyBudget = physicalMemoryBytes >= 24 * gib ? 8 : 4
            default:
                familyMode = .off; familyBudget = 2
            }
            if let requestedShadowBudget {
                return requestedShadowBudget > 0
                    ? SpeculativePrefetchPlan(mode: .shadow, shadowBudget: requestedShadowBudget)
                    : SpeculativePrefetchPlan(mode: .off, shadowBudget: familyBudget)
            }
            let environmentBudget = environment["MFERENCE_SHADOW_BUDGET"].flatMap(Int.init).flatMap { $0 > 0 ? $0 : nil }
            return SpeculativePrefetchPlan(
                mode: environment["MFERENCE_SPEC_PREFETCH"].map(parse) ?? familyMode,
                shadowBudget: environmentBudget ?? familyBudget)
        }

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
    /// of the pilot's weight-ranked prediction list. Resolved with the mode.
    var shadowIssueBudget = 2
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
    /// Read-only numerical qualification evidence from the existing decode
    /// readback. Does not add GPU work or alter routing/prefetch decisions.
    var lastDecodeRoutedExperts: [[Int]] { previousTokenExperts }
    private var pendingSpeculation: SpeculativeExpertPrefetch?
    /// Overrides the predictor for a layer. The seam a real predictor plugs
    /// into, and what lets tests drive the reserve/read/join/confirm machinery
    /// without depending on a fixture whose routing happens to churn the cache.
    var speculativeExpertPredictor: (@Sendable (Int) -> [Int])?
    /// The lookahead router's prediction (`pilotIndices`), read at the current
    /// layer's router wake and consumed by `issueSpeculativePrefetch`.
    private var pilotPrediction: (layer: Int, experts: [Int])?

    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.routerEvent = context.device.makeSharedEvent()
        self.eagerFetchEvent = context.device.makeSharedEvent()
        self.maxContext = maxContext
        let prefetchPlan = SpeculativePrefetchMode.plan(
            requestedShadowBudget: runtimeConfiguration.shadowPrefetchBudget,
            environment: ProcessInfo.processInfo.environment,
            family: model.config.family,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
        self.speculativePrefetchMode = prefetchPlan.mode
        self.shadowIssueBudget = prefetchPlan.shadowBudget
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
                                     maxPrefillChunkTokens: runtimeConfiguration.prefillConfig.chunkTokens,
                                     // Qwen 3.6 grows full-attention KV with the
                                     // conversation.
                                     fullAttentionGrowthStep: cfg.family == .qwen36
                                         ? runtimeConfiguration.kvGrowthTokens : nil)

        let silu = cfg.hiddenActivation == "silu"
        let int4GroupSize = model.affineInt4GroupSize
        self.embedInt4 = try EmbedLookupInt4(context: context, groupSize: int4GroupSize)
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(
            context: context,
            additionalShapes: cfg.decodeInt4GEMVShapes,
            groupSize: int4GroupSize)
        self.attention = try Attention(context: context)
        // Qwen 3.6 decode runs the batched grouped full-attention kernel (class
        // T, gated by Qwen36PrefillEquivalenceGateTests' decode-attention gate).
        self.attention.groupedFull256Batched = cfg.family == .qwen36
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits,
                                                  siluActivation: silu,
                                                  specializedD: cfg.hiddenSize,
                                                  specializedF: cfg.intermediateSize,
                                                  groupSize: int4GroupSize)
        self.moe       = try MoE(context: context,
                                 siluActivation: silu,
                                 specializedD: UInt32(cfg.hiddenSize),
                                 specializedF: UInt32(cfg.moeIntermediateSize),
                                 specializedNumExperts: UInt32(cfg.numExperts),
                                 specializedTopK: UInt32(cfg.topKExperts),
                                 groupSize: int4GroupSize,
                                 routerBF16: false)
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize,
                                              groupSize: int4GroupSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context, groupSize: int4GroupSize)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.fusedPostAttentionSetup = try FusedPostAttentionSetup(context: context)
        self.fusedTail = try FusedLayerTail(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context, groupSize: int4GroupSize)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context, groupSize: int4GroupSize)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(context: context, groupSize: int4GroupSize)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        // Qwen 3.6 full attention (headDim 256, 16/2 heads) never fits the
        // TensorOps kernel's shape gate, so off Apple10 it takes the
        // GQA-shared simdgroup kernel instead of the per-head tiled one.
        self.prefillAttention = try PrefillAttention(context: context,
                                                     gqaSimdgroup: Self.usesQwenGQASimdAttention(
                                                        family: cfg.family,
                                                        supportsApple10TensorOps: context.device.supportsApple10TensorOps))
        self.prefillPostAttention = try PrefillPostAttentionSetup(context: context)
        self.prefillRouter = try PrefillRouter(context: context, routerBF16: false)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.sharedExpertWeightBits,
            siluActivation: silu,
            groupSize: int4GroupSize)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context,
                                                             siluActivation: silu,
                                                             groupSize: int4GroupSize)
        self.prefillGroupedGEMM = Self.usesGroupedRoutedExpertGEMM(
            family: cfg.family,
            supportsApple10TensorOps: context.device.supportsApple10TensorOps)
            ? MPPGroupedRoutedMoE(context: context, groupSize: int4GroupSize, gelu: !silu)
            : nil
        self.prefillSimdGroupedGEMM = Self.usesSimdgroupGroupedRoutedExperts(
            family: cfg.family,
            supportsApple10TensorOps: context.device.supportsApple10TensorOps)
            && silu
            && SimdgroupGroupedRoutedMoE.supports(d: cfg.hiddenSize,
                                                  routedIntermediate: cfg.moeIntermediateSize,
                                                  groupSize: int4GroupSize)
            // A pipeline that fails to build leaves MPP and the row kernel.
            ? try? SimdgroupGroupedRoutedMoE(context: context, groupSize: int4GroupSize)
            : nil
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context,
                                                               maxD: cfg.hiddenSize,
                                                               groupSize: int4GroupSize)

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
        // Dense FFN layers run through the shared-expert kernel at
        // denseIntermediateSize, so the scratch covers the wider of the two
        // FFN widths.
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
        // Families that route fewer than 8 experts pad the
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
        for L in 0..<cfg.numLayers {
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
            // Fold `router.scale * D^-0.5` into BF16 once. Stored router
            // weights and scale assets remain untouched.
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
                    let scale = Quantization.bf16ToFloat(src[i])
                    dst[i] = Quantization.bf16Bits(scale * invSqrtD)
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
        resetTransientState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        try prefillChunkState.requireClean(operation: "prepareForContinuation")
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
    /// Phase-report expert I/O state. Held behind one optional reference, and
    /// touched only from non-inlined methods called under
    /// `phaseInstrumentationEnabled`, so the default decode path carries a
    /// single nil pointer and none of the recording code.
    private final class PhaseIOState {
        var routing: ExpertRoutingCounters
        var decodeWindowOpen = false
        var decodeDiskStart: ProcessDiskIO.Counters?
        var lastPrefillReads: ExpertReadStats.Snapshot?
        var lastPrefillDiskBytes: UInt64?
        var prefillDiskStart: ProcessDiskIO.Counters?
        var prefillWindowOpen = false

        init(numLayers: Int, numExperts: Int) {
            routing = ExpertRoutingCounters(numLayers: numLayers, numExperts: numExperts)
        }
    }
    private var phaseIO: PhaseIOState?

    /// Per-(layer, expert) routed hits and misses over the decode window.
    /// Nil unless `MFERENCE_PHASES=1` and a decode window has opened.
    public var expertRoutingCounters: ExpertRoutingCounters? {
        guard let phaseIO, phaseIO.decodeWindowOpen else { return nil }
        return phaseIO.routing
    }
    /// Process storage counters at the start of the decode window, so the
    /// phase report can print the bytes decode physically read.
    public var decodeWindowDiskIOStart: ProcessDiskIO.Counters? { phaseIO?.decodeDiskStart }
    /// Routed-expert reads and physical bytes read during the most recent
    /// chunked prefill, captured when the decode window opens.
    public var lastPrefillExpertReads: ExpertReadStats.Snapshot? { phaseIO?.lastPrefillReads }
    public var lastPrefillDiskBytesRead: UInt64? { phaseIO?.lastPrefillDiskBytes }

    private func phaseIOState() -> PhaseIOState {
        if let phaseIO { return phaseIO }
        let state = PhaseIOState(numLayers: cfg.numLayers, numExperts: cfg.numExperts)
        phaseIO = state
        return state
    }

    /// Opens the prefill side of the phase report's I/O counters.
    @inline(never)
    private func beginPrefillIOWindow() {
        let state = phaseIOState()
        ExpertReadStats.shared?.reset()
        state.prefillDiskStart = ProcessDiskIO.current()
        state.prefillWindowOpen = true
        state.lastPrefillReads = nil
        state.lastPrefillDiskBytes = nil
    }

    /// Closes the prefill I/O window (if open) and opens the decode one.
    @inline(never)
    private func openDecodeIOWindow() {
        let state = phaseIOState()
        let disk = ProcessDiskIO.current()
        if state.prefillWindowOpen {
            // Read timing does not depend on rusage; keep it even when the
            // storage counters were unavailable.
            state.lastPrefillReads = ExpertReadStats.shared?.snapshot()
            if let start = state.prefillDiskStart, let disk, disk.bytesRead >= start.bytesRead {
                state.lastPrefillDiskBytes = disk.bytesRead - start.bytesRead
            }
            state.prefillDiskStart = nil
            state.prefillWindowOpen = false
        }
        ExpertReadStats.shared?.reset()
        state.decodeDiskStart = disk
        state.routing.reset()
        state.decodeWindowOpen = true
    }

    /// Records one routed layer step for the phase report. `misses` nil means
    /// the experts were loaded without a CPU plan (top-k beyond the planner):
    /// the cached load decides hits internally, so only the step is counted.
    @inline(never)
    private func notePhaseLayerStep(_ layer: Int, _ experts: [Int], misses: [Int]?) {
        guard let phaseIO, phaseIO.decodeWindowOpen else { return }
        if let misses {
            phaseIO.routing.recordLayerStep(layer: layer, experts: experts, missIndices: misses)
        } else {
            phaseIO.routing.recordUnplannedLayerStep()
        }
    }

    /// Zeroes the per-phase counters. Prompt prefill and decode share these
    /// counters, so without a reset at the prefill/decode boundary the phase
    /// report prints
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
        if Self.phaseInstrumentationEnabled { openDecodeIOWindow() }
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
            missing = Array(missing.prefix(shadowIssueBudget))
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
        var execution = PrefillExecutionReport()
        try prefillChunkState.requireClean(operation: "prefillChunked")
        if Self.phaseInstrumentationEnabled { beginPrefillIOWindow() }
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
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten, execution: execution)
        }
        // A prompt gets one full growth step past its end to answer into.
        try ensureKVCapacity(for: startPosition + tokens.count)

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        for (spanIndex, span) in spans.enumerated() {
            try Task.checkCancellation()
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
            execution.recordBatch(span.tokenCount)
            onProgress(span.completedCount)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken), execution: execution)
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten, execution: execution)
    }

    @discardableResult
    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(
            config: cfg, runtime: config,
            groupedExperts: prefillGroupedGEMM?.isAvailable == true || prefillSimdGroupedGEMM != nil)
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
            // FFN sandwich norms only.
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
                if path == .affineThreadgroupF16 || path == .sourceAffineF16 {
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
        // Diagnostic (`MFERENCE_PREFILL_BREAKDOWN=1`): read once per chunk; with
        // it off every site below is a branch on `breakdownOn`.
        let stageBreakdown = prefillStageBreakdown
        let breakdownOn = stageBreakdown != nil
        let chunkStartNanos = breakdownOn ? PrefillStageBreakdown.now() : 0
        // Diagnostic only: ends the current command buffer at a stage boundary so
        // the stage gets its own GPU span. Same kernels, inputs and queue order.
        func endStage(_ stage: PrefillStage) throws {
            if breakdownOn, let stageBreakdown {
                try splitPrefillStage(&cb, stage: stage, into: stageBreakdown)
            }
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

        var prefillLookahead: SpeculativeExpertPrefetch?
        for L in 0..<cfg.numLayers {
            let layerStartNanos = breakdownOn ? PrefillStageBreakdown.now() : 0
            try prefillWillEncodeLayer?(L)
            try Task.checkCancellation()
            cb.label = "prefill start=\(startPosition) count=\(t) layer=\(L) phase="
                + (L == 0 ? "embed_attention_router" : "attention_router")
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
                try endStage(.gdnInputProjections)
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
                try endStage(.gdnCore)
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

                try endStage(.attentionInputProjections)
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
                        if prefillAttention.encodeCausal(commandBuffer: cb,
                                                         q: attnQ,
                                                         k: keyBuffer,
                                                         v: valueBuffer,
                                                         out: scratch.attentionOutput,
                                                         params: params,
                                                         kvRingCapacity: activeRingCapacity,
                                                         path: prefillAttentionPath) {
                            prefillTensorOpsAttentionLayers += 1
                        }
                } else {
                    throw PrefillError.chunkedUnsupported(
                        "chunked prefill attention requires FP16 KV")
                }
                try endStage(.attentionCore)
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
            try endStage(isLinear ? .gdnOutputProjection : .attentionOutputProjection)
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
                    try checkCommandBufferError(cb)
                    stageBreakdown?.addGPU(.postAttentionAndRouter, cb)
                    let routePlanStartNanos = breakdownOn ? PrefillStageBreakdown.now() : 0

                    // The shared branch depends only on routedX, which the
                    // completed attention/router command has already
                    // produced. Submit it before CPU route grouping and SSD
                    // binding so those tasks overlap its batched projections.
                    guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    sharedCB.label = "prefill start=\(startPosition) count=\(t) layer=\(L) phase=shared_expert"
                    let sharedProj = sharedExpertProjections[L]
                    lastPrefillSharedExpertPath = try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
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
                    let requestedScheduler = Self.prefillRoutedTileSchedulerConfig
                    let schedulerConfig: PrefillRoutedTileSchedulerConfig
                    let routeTileExpertCount: Int
                    if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
                        guard let bounded = requestedScheduler.fittingSlotBudget(slotCount: slotCount) else {
                            throw PrefillError.chunkedUnsupported(
                                "prefill routed tile depth \(requestedScheduler.maxPendingDepth) needs at least \(requestedScheduler.maxPendingDepth + 1) slots, has \(slotCount)")
                        }
                        schedulerConfig = bounded
                        routeTileExpertCount = bounded.tileExperts
                    } else {
                        schedulerConfig = requestedScheduler
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
                        try checkCommandBufferError(pending.commandBuffer)
                        stageBreakdown?.addGPU(.routedTiles, pending.commandBuffer)
                        if !pending.fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    if let streamer = try? model.routedExpertStreamer(layer: L) {
                        var routeCounts = [Int](repeating: 0, count: cfg.numExperts)
                        for id in routeIDs { routeCounts[Int(id)] += 1 }
                        streamer.noteRouteCounts(routeCounts)
                    }
                    if let stageBreakdown {
                        stageBreakdown.addRoutedPlan(nanos: PrefillStageBreakdown.now() - routePlanStartNanos)
                    }
                    // Join the previous layer's lookahead before planning, so
                    // the tile plans see its slots as hits.
                    if let lookahead = prefillLookahead, lookahead.layer == L {
                        _ = lookahead.join()
                        prefillLookahead = nil
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
                        let fetchStartNanos = breakdownOn ? PrefillStageBreakdown.now() : 0
                        let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                            model: model,
                            layer: L,
                            tileIndex: tileIndex,
                            routes: routes,
                            plannedFetch: plannedFetch,
                            avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                        if let stageBreakdown {
                            stageBreakdown.addRoutedFetchWait(nanos: PrefillStageBreakdown.now() - fetchStartNanos)
                        }
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        if !fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedMissSlots)
                        }
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
                        tileCB.label = "prefill start=\(startPosition) count=\(t) layer=\(L) "
                            + "phase=routed_tile tile=\(tileIndex)"
                        let tilePairCounts = routes.groups[
                            Int(tile.groupStart)..<Int(tile.groupStart + tile.groupCount)]
                            .map { Int($0.pairCount) }
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                        if prefillSimdGroupedGEMMEnabled, prefillGroupedGEMMEnabled,
                           let simd = prefillSimdGroupedGEMM,
                           let activation = scratch.groupedExpertActivation,
                           SimdgroupGroupedRoutedMoE.supportsOffsets(streamedParams),
                           PrefillGroupedExpertGate.usesSimdgroupGroupedGEMM(pairCounts: tilePairCounts) {
                            // Same group-range contract as the MPP branch below.
                            argumentBuffer = try simd.makeArgumentBuffer(device: ctx.device,
                                                                         binding: fetch.binding)
                            var groupedParams = streamedParams
                            groupedParams.pairStart = tile.groupStart
                            groupedParams.pairCount = tile.groupCount
                            let encoded = simd.encode(
                                commandBuffer: tileCB,
                                hidden: scratch.routedX,
                                sortedPairs: metadata.sortedPairs,
                                groups: metadata.groups,
                                activation: activation,
                                routePartials: scratch.routePartials,
                                argumentBuffer: argumentBuffer,
                                binding: fetch.binding,
                                params: groupedParams,
                                maxPairsPerGroup: tilePairCounts.max() ?? 0)
                            guard encoded else { throw ModelError.residentBufferWrapFailed }
                            prefillSimdGroupedExpertTiles += 1
                        } else if prefillGroupedGEMMEnabled,
                           let grouped = prefillGroupedGEMM, grouped.isAvailable,
                           let activation = scratch.groupedExpertActivation,
                           PrefillGroupedExpertGate.usesGroupedGEMM(pairCounts: tilePairCounts) {
                            // Whole 64-row matrix tiles per expert; its group
                            // range replaces the row kernel's pair range.
                            let groupedArguments = try grouped.makeArgumentBuffer(
                                device: ctx.device, binding: fetch.binding)
                            var groupedParams = streamedParams
                            groupedParams.pairStart = tile.groupStart
                            groupedParams.pairCount = tile.groupCount
                            let encoded = grouped.encode(
                                commandBuffer: tileCB,
                                hidden: scratch.routedX,
                                sortedPairs: metadata.sortedPairs,
                                groups: metadata.groups,
                                activation: activation,
                                routePartials: scratch.routePartials,
                                argumentBuffer: groupedArguments,
                                binding: fetch.binding,
                                params: groupedParams,
                                maxPairsPerGroup: tilePairCounts.max() ?? 0)
                            guard encoded else { throw ModelError.residentBufferWrapFailed }
                            argumentBuffer = PrefillStreamedTileArgumentBuffer(buffer: groupedArguments.buffer)
                            prefillGroupedExpertTiles += 1
                        } else {
                            argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                                device: ctx.device,
                                binding: fetch.binding)
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
                        }
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    // The SSD idles from here through the next layer's
                    // attention and router. Tiles sweep experts in file order
                    // and a prompt routes to most of a layer's experts, so read
                    // the next layer's first experts by offset into its free
                    // slots now; its tile plans then find them as hits.
                    if L + 1 < cfg.numLayers,
                       let slotCount = model.routedExpertCacheSlotCount(layer: L + 1),
                       let streamer = try? model.routedExpertStreamer(layer: L + 1) {
                        let tileSlots = (schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts
                        let offsets = model.routedExpertPhysicalOffsets(layer: L + 1)
                        let firstByOffset = offsets.indices.sorted { offsets[$0] < offsets[$1] }
                            .prefix(max(0, slotCount - tileSlots))
                        let ahead = streamer.nonResidentExperts(Array(firstByOffset))
                        let reservation = streamer.reserveSpeculativeSlots(experts: ahead,
                                                                           keepEvictable: tileSlots)
                        if !reservation.isEmpty {
                            let record = SpeculativeExpertPrefetch(layer: L + 1, predicted: Set(ahead))
                            prefillLookahead = record
                            // Part of the sweep: unbuffered wherever the
                            // model's regime rule made the sweep unbuffered.
                            DispatchQueue.global(qos: .userInitiated).async {
                                record.complete(bytes: streamer.executeSpeculativeReservation(
                                    reservation, sweep: true))
                            }
                        }
                    }
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    guard let tailCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    tailCB.label = "prefill start=\(startPosition) count=\(t) layer=\(L) phase=moe_tail"
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
                    try checkCommandBufferError(tailCB)
                    stageBreakdown?.addGPU(.moeTail, tailCB)
                    // Complete already: the tail read the shared branch's
                    // output. The wait only makes the status check exact.
                    waitForCompletion(sharedCB)
                    try checkCommandBufferError(sharedCB)
                    stageBreakdown?.addGPU(.sharedExpert, sharedCB)
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        cb = nextCB
                    }
                    if let stageBreakdown {
                        stageBreakdown.addLayerWall(nanos: PrefillStageBreakdown.now() - layerStartNanos)
                    }
                    continue
        }

        if writeFinalHead {
            let finalNorm = model.finalNorm
            let lm = model.lmHead
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            finalCB.label = "prefill start=\(startPosition) count=\(t) phase=final_head"
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
            try checkCommandBufferError(finalCB)
            stageBreakdown?.addGPU(.head, finalCB)
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }

        if let stageBreakdown {
            stageBreakdown.addChunk(wallNanos: PrefillStageBreakdown.now() - chunkStartNanos,
                                    tokens: tokens.count)
        }
        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        ctx.keepWarm.touch()
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        // An answer that outgrows that room grows by half a step at a time.
        try ensureKVCapacity(for: position + 1, headroom: kv?.fullAttentionGrowthStep.map { $0 / 2 })
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
                    notePhaseLayerStep(L, experts, misses: [])
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
                notePhaseLayerStep(L, experts, misses: plannedFetch?.misses)
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

    /// The chunked-prefill attention kernels, for the tolerance gate's control
    /// arm (`PrefillAttention.useTiledKernelForGQAShapes()`) and its counters.
    var prefillAttentionKernels: PrefillAttention { prefillAttention }

    /// Whether chunked prefill's full-attention layers use the GQA-shared
    /// simdgroup kernel (`attention_prefill_gqa8_d256_simd`). Only Qwen 3.6,
    /// and only where the Apple10 TensorOps path is absent; it changes output
    /// within tolerance, so every other family and M5-class host keeps its bytes.
    static func usesQwenGQASimdAttention(family: ModelFamily,
                                         supportsApple10TensorOps: Bool) -> Bool {
        family == .qwen36 && !supportsApple10TensorOps
    }

    /// Whether chunked prefill's routed experts may take the grouped-GEMM
    /// kernel (`MPPGroupedRoutedMoE`) for tiles that pass
    /// `PrefillGroupedExpertGate`. Only Qwen 3.6, and only where the Apple10
    /// TensorOps path is absent, the hardware it was measured on; it changes
    /// output within tolerance, so every other family and M5-class host keeps
    /// its bytes. Where the `tensorops` module does not build the kernel is
    /// unavailable and the row kernel runs.
    static func usesGroupedRoutedExpertGEMM(family: ModelFamily,
                                            supportsApple10TensorOps: Bool) -> Bool {
        family == .qwen36 && !supportsApple10TensorOps
    }

    /// Whether chunked prefill's routed experts may take the simdgroup-matrix
    /// grouped kernel (`SimdgroupGroupedRoutedMoE`) for tiles that pass
    /// `PrefillGroupedExpertGate.usesSimdgroupGroupedGEMM`. Only Qwen 3.6, and
    /// only where the Apple10 TensorOps path is absent, the hardware it was
    /// measured on; it changes output within tolerance, so every other family
    /// and M5-class host keeps its bytes. Ahead of the MPP kernel, which stays
    /// the fallback for tiles this one skips.
    static func usesSimdgroupGroupedRoutedExperts(family: ModelFamily,
                                                  supportsApple10TensorOps: Bool) -> Bool {
        family == .qwen36 && !supportsApple10TensorOps
    }

    /// Test-only: routes every prefill tile to the row kernel.
    func useRowKernelForRoutedExperts() {
        prefillGroupedGEMMEnabled = false
        prefillSimdGroupedGEMMEnabled = false
    }

    /// Test-only: leaves routed tiles to the MPP grouped kernel and the row
    /// kernel, the selection before the simdgroup kernel.
    func useMPPOrRowKernelForRoutedExperts() { prefillSimdGroupedGEMMEnabled = false }

    /// Test-only: decode full attention goes back to the per-component
    /// grouped kernel at a fixed 16 chunks.
    func usePerComponentDecodeAttention() { attention.groupedFull256Batched = false }

    /// Whether decode full attention runs the batched grouped kernel (Qwen 3.6 default).
    var decodeAttentionIsBatched: Bool { attention.groupedFull256Batched }

    /// Chunked-prefill stage timing (`PrefillStageBreakdown`), the prefill
    /// counterpart to `MFERENCE_PHASES` for decode. Off by default.
    static let prefillBreakdownEnabled =
        ProcessInfo.processInfo.environment["MFERENCE_PREFILL_BREAKDOWN"] == "1"

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

    /// Grows full-attention KV before anything is encoded at these positions.
    /// Growing re-wraps the same pages, so nothing in flight is disturbed.
    private func ensureKVCapacity(for tokens: Int, headroom: Int? = nil) throws {
        try kv?.ensureCapacity(for: tokens, headroom: headroom)
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) {
        let cb = ctx.queue.makeCommandBuffer()!
        body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        reportCommandBufferFailure(cb)
    }

    /// `MFERENCE_PREFILL_BREAKDOWN=1` only: commits and waits the command
    /// buffer at a stage boundary, records its GPU span, and continues on a
    /// fresh buffer with the same label. Out of line so none of it is inlined
    /// into `executePrefillChunk` when the diagnostic is off.
    @inline(never)
    private func splitPrefillStage(_ cb: inout MTLCommandBuffer,
                                   stage: PrefillStage,
                                   into breakdown: PrefillStageBreakdown) throws {
        cb.commit()
        waitForCompletion(cb)
        try checkCommandBufferError(cb)
        breakdown.addGPU(stage, cb)
        guard let next = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        next.label = cb.label
        cb = next
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) {
        cb.waitUntilCompleted()
        reportCommandBufferFailure(cb)
    }

    /// Prints a completed wait's failure; callers that stop on it also call
    /// `checkCommandBufferError`.
    private nonisolated func reportCommandBufferFailure(_ cb: MTLCommandBuffer) {
        if let detail = metalCommandBufferFailureDetail(label: cb.label, status: cb.status,
                                                        error: cb.error) {
            print("CB error: \(detail)")
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

// MARK: - Prefix state snapshots (server multi-prefix cache)

/// Qwen 3.6 only: its numerical state at a position is full-attention KV rows
/// [0, position) plus the gated-DeltaNet state and conv tail. A layout these
/// copies do not cover (sliding rings, no recurrent state) reports no support
/// and the server keeps its single-prefix behaviour for it.
extension RealForwardRunner: PrefixStateSnapshotting {
    public var supportsPrefixStateSnapshots: Bool {
        model.config.family == .qwen36
            && gdnState != nil && kv?.supportsPrefixSnapshots == true
    }

    public func prefixKVRowBytes(tokens: Int) -> Int {
        kv?.prefixSnapshotBytes(tokens: tokens) ?? 0
    }

    public var prefixRecurrentStateBytes: Int { gdnState?.recurrentCheckpointBytes ?? 0 }

    private func requireSnapshotReady(_ operation: String) throws -> (KVCacheManager, GDNStateManager) {
        try Task.checkCancellation()
        try prefillChunkState.requireClean(operation: operation)
        guard supportsPrefixStateSnapshots, let kv, let gdnState else {
            throw PrefixSnapshotError.unsupported("\(model.config.family) runner")
        }
        return (kv, gdnState)
    }

    public func capturePrefixKVRows(count: Int, into region: PrefixSnapshotRegion) throws {
        let (kv, _) = try requireSnapshotReady("prefix KV capture")
        try kv.copyPrefixRows(count: count, into: region)
    }

    public func capturePrefixRecurrentState(into region: PrefixSnapshotRegion) throws {
        let (_, gdnState) = try requireSnapshotReady("prefix recurrent capture")
        try gdnState.copyRecurrentState(into: region)
    }

    public func restorePrefixState(rows: (region: PrefixSnapshotRegion, count: Int)?,
                                   position: Int,
                                   recurrentState: PrefixSnapshotRegion) throws {
        let (kv, gdnState) = try requireSnapshotReady("prefix restore")
        if let rows {
            try kv.restorePrefixRows(count: rows.count, from: rows.region, position: position)
        } else {
            guard position > 0, position <= kv.position else {
                throw PrefixSnapshotError.outOfRange(
                    "rewind to \(position) with cursor at \(kv.position)")
            }
            kv.rewind(to: position)
        }
        try gdnState.restoreRecurrentState(from: recurrentState)
        resetTransientState()
    }
}
