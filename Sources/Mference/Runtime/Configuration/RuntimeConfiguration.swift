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
    /// `--shadow-budget`: 0 turns speculative expert prefetch off, 1...8 selects
    /// shadow prefetch with that many speculative reads per layer.
    public static let allowedShadowPrefetchBudgets = 0...8
    /// Full-attention KV (Qwen 3.6) starts at this many tokens. A
    /// prompt that does not fit grows it to the prompt plus this many more,
    /// room for the answer; an answer that outgrows that room grows it by half
    /// as much at a time, up to `--max-context`. Metal charges a KV buffer in full once
    /// bound, so reserving the whole `--max-context` up front costs memory a
    /// short chat never uses: on a 16 GiB M2 a 262,144-token reservation took
    /// 2.6 GiB of expert page cache from 128,000 and cut QAT decode from 6.3 to
    /// 4.6 tok/s. `--kv-reserve` reserves the whole context instead.
    public static let defaultKVGrowthTokens = 16_384

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
    /// `--shadow-budget`; nil keeps the family default.
    public let shadowPrefetchBudget: Int?
    /// Growth step of full-attention KV; nil (`--kv-reserve`) reserves the
    /// whole context up front.
    public let kvGrowthTokens: Int?

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
                kvPoolPagesPerLayer: Int? = nil,
                shadowPrefetchBudget: Int? = nil,
                kvGrowthTokens: Int? = RuntimeConfiguration.defaultKVGrowthTokens) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        precondition(shadowPrefetchBudget.map(Self.allowedShadowPrefetchBudgets.contains) ?? true,
                     "unsupported shadow prefetch budget")
        precondition(kvGrowthTokens.map { $0 > 0 } ?? true, "KV growth step must be positive")
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
        self.shadowPrefetchBudget = shadowPrefetchBudget
        self.kvGrowthTokens = kvGrowthTokens
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    /// The server prefills long prompts in chunks; every chunk re-reads most
    /// routed experts, so 128-token chunks made a 3,015-token Gemma prompt
    /// spend 134 s of 205 s on expert I/O. On hosts with at least 16 GiB Gemma
    /// took 1,024 tokens (about +309 MB over 128) and since 2026-09-26 takes
    /// 2,048: a 19,098-token QAT prompt fell from 386.3 s to 350.2 s (mean of
    /// an A B B A run) for +351 MB of Metal allocation, 210 MB of it sliding
    /// KV ring. Qwen 3.6 has no sliding ring to grow, so 2,048 costs it +270 MB
    /// of scratch: a 2,940-token prompt went from 148.7 s to 42.7 s with decode
    /// unchanged (2026-09-21). The Unsloth UD-IQ3_XXS Flash-Next install
    /// re-reads its 47 GB routed-expert pool once per chunk: through the
    /// server a 2,940-token prompt's first token took 321 s at 128 before
    /// prefill batched its per-token GEMV loops; with them it takes 108-110 s
    /// at 1,024 and 87-88 s at 4,096 (one chunk, one sweep), with peak
    /// footprints of 2.3 and 4.4 GB (2026-11-17, 32 GB M1 Max). It takes
    /// 4,096 on hosts with at least 32 GiB and 1,024 on hosts from 16 GiB up
    /// to but not including 32 GiB. Vendor Flash-Next keeps 128 (unmeasured
    /// here). A family with its own chunk (`familyPrefillChunkTokens(for:)`,
    /// Inkling 2's 2,048) takes it on every host. The server's
    /// `--prefill-chunk` and then `MFERENCE_SERVER_PREFILL_CHUNK` override.
    public static func defaultServerPrefillChunkTokens(
        for family: ModelFamily,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        if let requested = environment["MFERENCE_SERVER_PREFILL_CHUNK"].flatMap(Int.init),
           allowedPrefillChunkTokens.contains(requested) {
            return requested
        }
        if let chunk = familyPrefillChunkTokens(for: family) { return chunk }
        let gib = UInt64(1) << 30
        guard physicalMemoryBytes >= 16 * gib else { return 128 }
        switch family {
        case .qwen36: return 2048
        default: return 128
        }
    }

    /// A family's own prefill chunk, the single source for it on every path:
    /// the server's default (`--library` and `--model`), the app's through
    /// the engine, the CLI's `--chat` (all through
    /// `defaultServerPrefillChunkTokens`), and the most the CLI's one-shot
    /// `auto` picks (`autoPrefillChunkTokens`). Nil for the families whose
    /// chunk follows the rules that predate it, which stay exactly as they
    /// were. Inkling 2 Mini: 2,048 on every host. On a 4K-token prompt,
    /// 2,048-token chunks prefilled at 398 tok/s and one 4,096-token chunk at
    /// 348 tok/s (K2 and K0, 2026-11-23); the runner sizes its scratch from
    /// the chunk, and the product's memory estimate already assumes
    /// chunk-2,048 scratch (`Inkling2MemoryFootprint.stateBytes`). A shorter
    /// prompt still prefills as one chunk of its own length.
    public static func familyPrefillChunkTokens(for family: ModelFamily) -> Int? {
        switch family {
        case .inkling2Loop: 2048
        case .qwen36, .qwen38, .maple, .minicpm5: nil
        }
    }

    /// `auto` for a prompt of `promptTokens` known before prefill (the CLI's
    /// one-shot run): the smallest allowed chunk that covers the prompt, so
    /// the prompt prefills in one chunk when it fits and reads each layer's
    /// routed experts once, up to the family's own chunk
    /// (`familyPrefillChunkTokens(for:)`). A nil `family` (unknown before
    /// load) is bounded by the allowed sizes only.
    public static func autoPrefillChunkTokens(promptTokens: Int, family: ModelFamily?) -> Int {
        let largest = family.flatMap(familyPrefillChunkTokens(for:)) ?? allowedPrefillChunkTokens.last!
        let sizes = allowedPrefillChunkTokens.filter { $0 <= largest }
        return sizes.first(where: { $0 >= promptTokens }) ?? sizes.last!
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
    public static func defaultExpertStreamingMode(
        for family: ModelFamily,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        expertPoolBytes: UInt64,
        coreWeightsBytes: UInt64
    ) -> ExpertStreamingMode {
        return .pread(slotCount: defaultExpertCacheSlots(
            for: family,
            physicalMemoryBytes: physicalMemoryBytes))
    }

    /// Memory the prefill-sweep rule treats as unavailable to the page cache
    /// beside the wired expert slots: the OS, other processes, and the
    /// process's own KV, scratch and Metal heaps. On the M1 Max 32 GiB
    /// (2026-11-15 A/B), file-backed pages never exceeded 14.36 GB while
    /// Qwen 3.6 ran 96 slots (6.79 GB), so at least 34.36 − 6.79 − 14.36 =
    /// 13.2 GB (12.3 GiB) was unavailable; 12 GiB rounds that down.
    static let prefillSweepReserveBytes = UInt64(12) << 30

    /// Whether prefill expert sweeps read through an `F_NOCACHE` descriptor.
    /// A buffered sweep pulls most of each layer's experts through the page
    /// cache; when the pool cannot stay cached anyway, that pass only evicts
    /// pages decode reuses, the mapped core weights among them. Measured on
    /// Qwen 3.6 at 32 GiB (2026-11-15, A B B A): the unbuffered sweep cut
    /// medium-review prefill 9–18% and the server's 2,940-token TTFT 12–15%,
    /// with identical output. Every other family keeps buffered sweeps: none
    /// was measured, and `F_NOCACHE` would change its I/O.
    static func prefillSweepBypassesPageCache(
        for family: ModelFamily,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        expertPoolBytes: UInt64,
        wiredSlotBytes: UInt64,
        coreWeightsBytes: UInt64
    ) -> Bool {
        guard family == .qwen36 else { return false }
        var required = prefillSweepReserveBytes
        for term in [expertPoolBytes, wiredSlotBytes, coreWeightsBytes] {
            let (sum, overflow) = required.addingReportingOverflow(term)
            if overflow { return true }
            required = sum
        }
        return required > physicalMemoryBytes
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
