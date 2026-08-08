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

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                forceLogitsHead: Bool = false) {
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
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    /// Qwen's 256 experts per layer need twice Gemma's cache coverage to avoid
    /// repeated SSD reads. Keep the larger footprint family- and RAM-specific.
    public static func defaultExpertCacheSlots(
        for family: ModelFamily,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        let sixteenGiB = UInt64(16) * 1024 * 1024 * 1024
        return family == .qwen36 && physicalMemoryBytes >= sixteenGiB ? 32 : 16
    }

    /// Fixed reserve the resident rung leaves for the KV cache, scratch, the
    /// process, and the OS. Clean file-backed expert pages degrade toward
    /// page-cache streaming under pressure, so the rung only needs the nominal
    /// working set to fit.
    static let residentHeadroomBytes = UInt64(4) * 1024 * 1024 * 1024

    /// The auto profile's streaming mode. Measured on the 24 GB M5
    /// (2026-08-07): `.resident` lost the community A/B on every case
    /// (short −2%, long −56% from page-cache thrash), and 128 near-resident
    /// slots beat nothing, because at 32 slots the page cache already holds
    /// the whole Qwen expert pool. Auto therefore stays on the slot rule;
    /// `resident`, 96, and 128 remain explicit flags for hosts where the
    /// arithmetic differs. `expertPoolBytes`/`coreWeightsBytes` stay in the
    /// signature so a future measured rung can use them without replumbing
    /// callers.
    public static func defaultExpertStreamingMode(
        for family: ModelFamily,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        expertPoolBytes _: UInt64,
        coreWeightsBytes _: UInt64
    ) -> ExpertStreamingMode {
        return .pread(slotCount: defaultExpertCacheSlots(
            for: family,
            physicalMemoryBytes: physicalMemoryBytes))
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
