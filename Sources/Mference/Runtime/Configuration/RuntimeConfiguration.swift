import Foundation

public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
}

/// Paged KV cache for long contexts (Qwen 3.8 full-attention layers): fixed
/// 64-token pages with an SSD spill tier and Quest-style query-aware sparse
/// decode. `.off` keeps the linear FP16 cache and the dense decode path.
public enum RuntimeKVPagedPolicy: String, Codable, Sendable {
    case off
    case on
}

public struct RuntimeConfiguration: Sendable, Equatable {
    /// 96 and 128 are the near-resident rungs: large wired LFU sets for hosts
    /// with RAM to spare but not enough to cache the whole expert pool.
    public static let allowedExpertCacheSlots = [8, 16, 24, 32, 64, 96, 128]
    public static let allowedPrefillChunkTokens = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath
    /// Opt-in approximate singleton-decode head for Maple checkpoints that
    /// retain FlashHead tensors. Prefill continues to use the exact head.
    public let useMapleFlashHead: Bool
    public let kvPagedPolicy: RuntimeKVPagedPolicy
    /// Sparse decode selection budget, in 64-token pages.
    public let kvTopKPages: Int
    public let kvSinkPages: Int
    public let kvRecentPages: Int
    /// Pool residency per full-attention layer, in pages. `nil` sizes the
    /// pool to the full context (everything resident; SSD tier idle).
    public let kvPoolPagesPerLayer: Int?

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                forceLogitsHead: Bool = false,
                useMapleFlashHead: Bool = false,
                kvPagedPolicy: RuntimeKVPagedPolicy = .off,
                kvTopKPages: Int = 60,
                kvSinkPages: Int = 2,
                kvRecentPages: Int = 4,
                kvPoolPagesPerLayer: Int? = nil) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.useMapleFlashHead = useMapleFlashHead
        precondition(kvTopKPages >= 0 && kvSinkPages >= 0 && kvRecentPages >= 1,
                     "invalid paged-KV selection parameters")
        self.kvPagedPolicy = kvPagedPolicy
        self.kvTopKPages = kvTopKPages
        self.kvSinkPages = kvSinkPages
        self.kvRecentPages = kvRecentPages
        self.kvPoolPagesPerLayer = kvPoolPagesPerLayer
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    /// The server prefills long prompts in chunks; every chunk re-reads most
    /// routed experts, so 128-token chunks made a 3,015-token Gemma prompt
    /// spend 134 s of 205 s on expert I/O. Gemma takes 1,024 tokens on hosts
    /// with at least 16 GiB: about +309 MB (scratch +125 MB, sliding KV ring
    /// +184 MB), the accepted budget. `MFERENCE_SERVER_PREFILL_CHUNK` overrides.
    public static func defaultServerPrefillChunkTokens(
        for family: ModelFamily,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        if let requested = environment["MFERENCE_SERVER_PREFILL_CHUNK"].flatMap(Int.init),
           allowedPrefillChunkTokens.contains(requested) {
            return requested
        }
        let gib = UInt64(1) << 30
        return family == .gemma4 && physicalMemoryBytes >= 16 * gib ? 1024 : 128
    }

    /// Qwen 3.6's large expert table needs more cache coverage to avoid repeated
    /// SSD reads. Keep the larger footprint family- and RAM-specific.
    /// 64 slots beat 32 by 4.6–5.9% (2026-08-08 A/B); with the GPU slot map
    /// the economics improved further and 96 slots (~6.8 GB wired, 50.7%
    /// all-hit layer rate) beat 64 on every case — so ≥24 GiB hosts run 96.
    /// Pre-slot-map, 96 lost to working-set pressure; both verdicts are
    /// recorded in experiments/summaries 10 and 15.
    public static func defaultExpertCacheSlots(
        for family: ModelFamily,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        let gib = UInt64(1) << 30
        switch family {
        case .qwen36:
            if physicalMemoryBytes >= 24 * gib { return 96 }
            if physicalMemoryBytes >= 16 * gib { return 32 }
            return 16
        default:
            return 16
        }
    }

    /// Fixed reserve the resident rung leaves for the KV cache, scratch, the
    /// process, and the OS. Clean file-backed expert pages degrade toward
    /// page-cache streaming under pressure, so the rung only needs the nominal
    /// working set to fit.
    static let residentHeadroomBytes = UInt64(32) * 1024 * 1024 * 1024

    /// The auto profile's streaming mode. For Qwen 3.6, measured on the 24 GB M5
    /// (2026-08-07): `.resident` lost the community A/B on every case
    /// (short −2%, long −56% from page-cache thrash), and 128 near-resident
    /// slots beat nothing, because at 32 slots the page cache already holds
    /// the whole expert pool. Qwen 3.6 therefore stays on the slot rule.
    /// Flash-Next separately selects resident mapping at ≥192 GiB when the
    /// routed pool and core still leave the fixed headroom above.
    ///
    /// GLM-5.3-Flash is the other measured exception: its whole INT4 expert
    /// set (~171 GB) fits a 256 GB host beside the ~10 GB core, and with the
    /// experts resident its runner routes on the GPU with no per-layer CPU
    /// round trip, so `auto` picks `.resident` whenever pool + core + the
    /// resident reserve + 16 GiB for KV, scratch and the process fit physical
    /// memory. Smaller hosts fall back to the slot rule.
    public static func defaultExpertStreamingMode(
        for family: ModelFamily,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        expertPoolBytes: UInt64,
        coreWeightsBytes: UInt64
    ) -> ExpertStreamingMode {
        let gib = UInt64(1) << 30
        let reserve = residentHeadroomBytes
        if family == .qwen38flashnext,
           physicalMemoryBytes >= 192 * gib,
           coreWeightsBytes <= physicalMemoryBytes - reserve,
           expertPoolBytes <= physicalMemoryBytes - reserve - coreWeightsBytes {
            // The routed pool is ~68 GiB. Mapping it on the 256 GiB M3 Ultra
            // improved the full protocol to 15.12/15.39/14.91 tok/s and avoids
            // the 17 GiB zero-filled allocation of the runner-up 128-slot rung.
            return .resident
        }
        if family == .glm53Flash {
            let runtimeReserve = UInt64(16) * gib
            let (sum1, o1) = expertPoolBytes.addingReportingOverflow(coreWeightsBytes)
            let (sum2, o2) = sum1.addingReportingOverflow(reserve + runtimeReserve)
            if !o1, !o2, physicalMemoryBytes >= sum2 { return .resident }
        }
        return .pread(slotCount: defaultExpertCacheSlots(
            for: family,
            physicalMemoryBytes: physicalMemoryBytes))
    }

    /// Auto pool sizing for the paged KV cache: everything resident when it
    /// fits, otherwise whatever RAM remains after weights and headroom
    /// (~physical − 22 GiB on the 24 GiB M5 → ~2 GiB of pool ≈ 32k resident
    /// tokens per full-attention layer), never below 1 GiB. Measured: a
    /// 4 GiB pool beside 14 GiB of weights pushed the host into compression
    /// and cost ~2× decode; 2 GiB keeps full speed and the SSD tier absorbs
    /// the rest.
    public static func defaultKVPoolPagesPerLayer(
        config: ArchConfig,
        maxContext: Int,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        let pageTokens = 64
        let pagesPerLayer = (maxContext + pageTokens - 1) / pageTokens
        let numFull = config.fullAttentionLayerMask.lazy.filter { $0 == 1 }.count
        guard numFull > 0 else { return pagesPerLayer }
        let pagePairBytes = 2 * pageTokens * config.numFullKVHeads * config.fullHeadDim * 2
        let gib = UInt64(1) << 30
        let headroom = UInt64(22) * gib
        let budget = max(gib, physicalMemoryBytes > headroom
                         ? physicalMemoryBytes - headroom : gib)
        let budgetPages = Int(budget) / (numFull * pagePairBytes)
        return max(1, min(pagesPerLayer, budgetPages))
    }

    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        expertCachePolicy == .lru ? .lru : .lfu
    }
}
