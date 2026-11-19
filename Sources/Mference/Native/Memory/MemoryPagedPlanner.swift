import Foundation

extension Native {
    /// The `MemoryPlanner` for `PagedKVStore`: exact arithmetic over the
    /// contract's geometry, resolved before any allocation.
    ///
    /// The store allocates exactly what the plan says:
    /// - `kvArena`: `ceil(contextLength / tokensPerPage)` pages of
    ///   `KVPageGeometry.pageBytes` (FP16, or INT8 with its scale plane);
    /// - `prefixCacheRAM`: `floor(prefixCacheRAMBudget / pageBytes)` more
    ///   pages of the same arena. Cached KV pages and recurrent snapshots
    ///   both live in arena pages, so this is the cache's whole RAM;
    /// - `recurrentState`: two banks of every GDN layer's `RecurrentLayout`;
    /// - `verifyArena`: `maxVerifyRows` slots per GDN layer;
    /// - `other`: the page-table ring (two slots of `u32` per context page);
    /// - `weightsMapped` and `activations` as given.
    ///
    /// It refuses with `memoryBudgetExceeded` when the total is over `budget`.
    /// One model is resident at a time, so `budget` is the machine's share
    /// for that one model.
    public final class PagedMemoryPlanner: MemoryPlanner {
        public init() {}

        /// Tokens per KV page.
        public static let tokensPerPage = 64
        /// Page-table ring depth: two steps may be in flight.
        public static let pageTableRingDepth = 2

        public static func geometry(config: Qwen35Config, precision: KVPrecision) -> KVPageGeometry {
            KVPageGeometry(tokensPerPage: tokensPerPage, attentionLayers: config.fullAttentionLayers.count,
                           kvHeads: config.kvHeadCount, headDim: config.keyLength, precision: precision)
        }

        public static func recurrentLayout(config: Qwen35Config) -> RecurrentLayout {
            RecurrentLayout(convKernel: config.ssmConvKernel, convChannels: config.ssmConvChannels,
                            valueHeads: config.ssmTimeStepRank, keyDim: config.ssmStateSize,
                            valueDim: config.ssmStateSize)
        }

        /// Pages that hold `tokens` tokens.
        public static func pages(tokens: Int, geometry: KVPageGeometry) -> Int {
            (tokens + geometry.tokensPerPage - 1) / geometry.tokensPerPage
        }

        /// Bytes of one recurrent snapshot (bank L of every GDN layer).
        public static func snapshotBytes(config: Qwen35Config) -> Int {
            config.gatedDeltaNetLayers.count * recurrentLayout(config: config).layerBytes
        }

        /// Arena pages one recurrent snapshot occupies in the prefix cache.
        public static func snapshotPages(config: Qwen35Config, geometry: KVPageGeometry) -> Int {
            (snapshotBytes(config: config) + geometry.pageBytes - 1) / geometry.pageBytes
        }

        /// The planned footprint (no budget check).
        public static func footprint(config: Qwen35Config, weightsMapped: UInt64, contextLength: Int,
                                     precision: KVPrecision, maxVerifyRows: Int, activationBytes: UInt64,
                                     prefixCacheRAMBudget: UInt64) -> MemoryPlan.Footprint {
            let g = geometry(config: config, precision: precision)
            let layout = recurrentLayout(config: config)
            let pageBytes = UInt64(g.pageBytes)
            let contextPages = UInt64(pages(tokens: contextLength, geometry: g))
            let gdn = UInt64(config.gatedDeltaNetLayers.count)
            return MemoryPlan.Footprint(
                weightsMapped: weightsMapped,
                kvArena: contextPages * pageBytes,
                recurrentState: 2 * gdn * UInt64(layout.layerBytes),
                verifyArena: gdn * UInt64(maxVerifyRows) * UInt64(layout.layerBytes),
                activations: activationBytes,
                prefixCacheRAM: prefixCacheRAMBudget / pageBytes * pageBytes,
                other: UInt64(pageTableRingDepth) * contextPages * 4)
        }

        public func resolve(config: Qwen35Config, weightsMapped: UInt64, contextLength: Int, precision: KVPrecision,
                            maxVerifyRows: Int, maxPrefillChunk: Int, activationBytes: UInt64,
                            prefixCacheRAMBudget: UInt64, budget: UInt64) throws -> MemoryPlan {
            guard contextLength > 0, maxPrefillChunk > 0, (1...16).contains(maxVerifyRows) else {
                throw NativeError.invalidShape(
                    "plan limits: context \(contextLength), prefill chunk \(maxPrefillChunk), verify rows \(maxVerifyRows)")
            }
            guard contextLength <= config.contextLength else {
                throw NativeError.invalidShape("context \(contextLength) exceeds the model's \(config.contextLength)")
            }
            let planned = Self.footprint(config: config, weightsMapped: weightsMapped, contextLength: contextLength,
                                         precision: precision, maxVerifyRows: maxVerifyRows,
                                         activationBytes: activationBytes, prefixCacheRAMBudget: prefixCacheRAMBudget)
            guard planned.total <= budget else {
                throw NativeError.memoryBudgetExceeded(required: planned.total, budget: budget)
            }
            return MemoryPlan(contextLength: contextLength, precision: precision, maxVerifyRows: maxVerifyRows,
                              maxPrefillChunk: maxPrefillChunk, planned: planned, budget: budget)
        }

        /// The largest context, in whole pages, whose plan fits `budget`
        /// (nil when not even one page fits). `activationBytes` does not
        /// depend on the context.
        public func largestContext(config: Qwen35Config, weightsMapped: UInt64, precision: KVPrecision,
                                   maxVerifyRows: Int, activationBytes: UInt64, prefixCacheRAMBudget: UInt64,
                                   budget: UInt64) -> Int? {
            func fits(_ tokens: Int) -> Bool {
                Self.footprint(config: config, weightsMapped: weightsMapped, contextLength: tokens,
                               precision: precision, maxVerifyRows: maxVerifyRows, activationBytes: activationBytes,
                               prefixCacheRAMBudget: prefixCacheRAMBudget).total <= budget
            }
            let step = Self.tokensPerPage
            var lo = 0, hi = config.contextLength / step
            guard hi > 0, fits(step) else { return nil }
            while lo < hi {
                let mid = (lo + hi + 1) / 2
                if fits(mid * step) { lo = mid } else { hi = mid - 1 }
            }
            return lo * step
        }
    }
}
