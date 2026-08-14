import Foundation
import Metal

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }

    public init(layer: Int, cachePlan: ExpertCachePlan) {
        self.layer = layer
        self.cachePlan = cachePlan
    }
}

extension Model {
    public func routedExpertOffsets(layer: Int) -> MoEExpertOffsets {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            UInt32(expert.subTensors[role]?.offset ?? 0)
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"))
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        packedExpertsLayout.layers[layer].experts.map(\.offset)
    }

    /// The opened backend for `layer`. Callers must `ensureLayerOpened` first.
    private func expertBackend(_ layer: Int) -> ExpertBackend {
        streamersQueue.sync { streamersBox.streamers[layer]! }
    }

    /// Resident mode never misses: every expert is a hit against the mapped
    /// layer file and no slot is assigned.
    private static func residentCachePlan(experts: [Int]) -> ExpertCachePlan {
        ExpertCachePlan(experts: experts,
                        assignedSlots: [],
                        misses: [],
                        hits: experts.count)
    }

    public func adviseRoutedExperts(layer: Int,
                                    experts: [Int]) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(layer)
        switch expertBackend(layer) {
        case .pread(let streamer):
            return streamer.adviseExpertMisses(experts: experts)
        case .resident:
            return .skipped(requested: experts.count)
        }
    }

    public func routedExpertAdviceByteEstimate(layer: Int,
                                               missCount: Int) throws -> UInt64 {
        guard missCount > 0 else { return 0 }
        try ensureLayerOpened(layer)
        switch expertBackend(layer) {
        case .pread(let streamer):
            return UInt64(missCount) * streamer.layout.expertStride
        case .resident:
            return 0
        }
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        switch expertBackend(layer) {
        case .pread(let streamer):
            let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
            return RoutedExpertFetchPlan(
                layer: layer,
                cachePlan: streamer.planExpertsCached(experts: experts,
                                                      avoidingSlots: validSlots))
        case .resident:
            return RoutedExpertFetchPlan(
                layer: layer,
                cachePlan: Self.residentCachePlan(experts: experts))
        }
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        switch expertBackend(layer) {
        case .pread(let streamer):
            let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
            guard let cachePlan = streamer.planExpertsCachedIfPossible(
                experts: experts,
                avoidingSlots: validSlots)
            else {
                return nil
            }
            return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
        case .resident:
            return RoutedExpertFetchPlan(
                layer: layer,
                cachePlan: Self.residentCachePlan(experts: experts))
        }
    }

    /// Per-layer gate_proj quant group size, derived from the manifest's
    /// scale-slice byte size. DeepSeek V4's conversion ships gate_proj at
    /// group 32 on most layers and 64 on the last; up/down stay at the
    /// base group size.
    public func routedGateGroupSize(layer: Int) -> Int {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        guard let scales = expert.subTensors["gate_scales"] else {
            return Quantization.groupSize
        }
        let rows = UInt64(config.moeIntermediateSize)
        let cols = UInt64(config.hiddenSize)
        guard rows > 0 else { return Quantization.groupSize }
        let groups = scales.size / (2 * rows)
        guard groups > 0, cols.isMultiple(of: groups) else {
            return Quantization.groupSize
        }
        return Int(cols / groups)
    }

    public func routedExpertCacheSlotCount(layer _: Int) -> Int? {
        guard case .pread(let slotCount) = streamingMode else { return nil }
        return slotCount
    }

    /// Direct handle on a layer's expert streamer. Used by the speculative
    /// prefetch path, which reserves slots on the caller's thread and then
    /// executes the reads on a background queue without re-resolving the layer.
    /// Eager decode path: fill the plan's miss slots asynchronously.
    /// `completion(true)` fires when the slots hold their experts; resident
    /// mode completes immediately (nothing to fill). `completion(false)`
    /// means a read failed and the results must not be used.
    public func fillRoutedExpertsAsync(plan: RoutedExpertFetchPlan,
                                       completion: @escaping @Sendable (Bool) -> Void) throws {
        try ensureLayerOpened(plan.layer)
        switch expertBackend(plan.layer) {
        case .pread(let streamer):
            streamer.beginAsyncFill(plan.cachePlan, completion: completion)
        case .resident:
            completion(true)
        }
    }

    /// GPU bindings for the slot-map decode path; nil for backends without a
    /// slot cache (resident mode).
    public func routedSlotMapBinding(layer: Int) throws
        -> (slab: MTLBuffer, table: MTLBuffer, slotStride: Int)? {
        try ensureLayerOpened(layer)
        switch expertBackend(layer) {
        case .pread(let streamer):
            return streamer.slotMapBinding
        case .resident:
            return nil
        }
    }

    public func routedExpertStreamer(layer: Int) throws -> PreadExpertStreamer {
        try ensureLayerOpened(layer)
        switch expertBackend(layer) {
        case .pread(let streamer):
            return streamer
        case .resident:
            throw StreamerError.noSlotCache
        }
    }

    /// Resident-only: views for `experts` served directly from the mapped
    /// layer file. Per-layer files address experts under layer index 0.
    private func residentExpertViews(_ streamer: ResidentExpertStreamer,
                                     layer: Int,
                                     experts: [Int]) throws -> [TensorView] {
        let buffers = try experts.map {
            try streamer.expertBuffer(layer: 0, expert: $0)
        }
        return Self.makeExpertViews(buffers, layer: layer, experts: experts)
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        switch expertBackend(plan.layer) {
        case .pread(let streamer):
            return Self.makeExpertViews(
                streamer.expertCachePlanBuffers(plan.cachePlan),
                layer: plan.layer,
                experts: plan.experts)
        case .resident(let streamer):
            return try residentExpertViews(streamer,
                                           layer: plan.layer,
                                           experts: plan.experts)
        }
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        switch expertBackend(plan.layer) {
        case .pread(let streamer):
            return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
        case .resident:
            return .skipped(requested: plan.experts.count)
        }
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        switch expertBackend(plan.layer) {
        case .pread(let streamer):
            return try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        let buffers = try streamer.executeExpertCachePlan(plan.cachePlan)
                        continuation.resume(returning: Self.makeExpertViews(
                            buffers,
                            layer: plan.layer,
                            experts: plan.experts))
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        case .resident(let streamer):
            return try residentExpertViews(streamer,
                                           layer: plan.layer,
                                           experts: plan.experts)
        }
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try ensureLayerOpened(layer)
        switch expertBackend(layer) {
        case .pread(let streamer):
            return try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        let buffers = try streamer.loadExpertsCached(experts: experts)
                        continuation.resume(returning: Self.makeExpertViews(
                            buffers,
                            layer: layer,
                            experts: experts))
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        case .resident(let streamer):
            return try residentExpertViews(streamer,
                                           layer: layer,
                                           experts: experts)
        }
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: 0)
        }
    }
}
