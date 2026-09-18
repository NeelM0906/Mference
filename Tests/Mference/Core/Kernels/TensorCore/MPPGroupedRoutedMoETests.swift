import Metal
import Testing
import MferenceValidationSupport

@testable import Mference

private let mppGroupedRoutedMoEAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPGroupedRoutedMoE(context: context).isAvailable
}()

@Suite struct MPPGroupedRoutedMoETests {
    /// Skipped where the runtime has no MPP tensor ops (GitHub's macOS
    /// runners), like the other MPP suites; `isAvailable` is asserted where
    /// the test does run.
    @Test(.enabled(if: mppGroupedRoutedMoEAvailable,
                   "Requires runtime MPP TensorOps support"))
    func groupedTensorOpsMatchesSiluGemvReference() throws {
        let d = 64, f = 64, rows = 13, topK = 2
        let pairs = (0..<rows).flatMap { token in
            (0..<topK).map { rank in
                PrefillGroupedRoutedMoETests.pair(
                    token: UInt32(token),
                    expert: UInt32((token + rank) % 3),
                    rank: UInt32(rank))
            }
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: rows, topK: topK, numExperts: 3,
            tileExpertCount: 3)
        let pool = PrefillGroupedRoutedMoETests.makeSyntheticExpertPool(
            numExperts: 3, d: d, f: f)
        let hidden = (0..<(rows * d)).map {
            Float16(Float(($0 % 23) - 11) * 0.01)
        }

        let context = try MetalContext()
        let grouped = try PrefillGroupedRoutedMoE(
            context: context, siluActivation: true)
        let mpp = MPPGroupedRoutedMoE(context: context)
        #expect(mpp.isAvailable)
        let binding = try PrefillStreamedTileBinding(
            expertIDs: [0, 1, 2],
            views: PrefillGroupedRoutedMoETests.streamedViewsWithNonzeroOffsets(
                device: context.device, pool: pool, expertIDs: [0, 1, 2]))
        let metadata = try grouped.makeStreamedMetadataBuffers(
            device: context.device, routes: routes)
        let oldArgument = try grouped.makeStreamedArgumentBuffer(
            device: context.device, binding: binding)
        let mppArgument = try mpp.makeArgumentBuffer(
            device: context.device, binding: binding)
        let params = PrefillGroupedRoutedMoEStreamedParams(
            pairStart: 0, pairCount: UInt32(pairs.count),
            d: UInt32(d), routedIntermediate: UInt32(f),
            topK: UInt32(topK), hiddenStrideElements: UInt32(d),
            binding: binding, offsets: pool.offsets)
        var mppParams = params
        mppParams.pairStart = 0
        mppParams.pairCount = UInt32(routes.groups.count)

        let half = MemoryLayout<Float16>.stride
        let hiddenBuffer = try #require(Fp16Buffer.make(context.device,
                                                        halves: hidden))
        func buffer(_ elements: Int) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(
                length: elements * half, options: .storageModeShared))
        }
        let oldOutput = try buffer(rows * topK * d)
        let newOutput = try buffer(rows * topK * d)
        let residentOutput = try buffer(rows * topK * d)
        let oldAct = try buffer(3 * pairs.count * f)
        let oldDown = try buffer(pairs.count * d)
        let newAct = try buffer(pairs.count * f)
        let residentAct = try buffer(pairs.count * f)
        let residentSlab = try #require(context.device.makeBuffer(
            bytes: pool.bytes, length: pool.bytes.count,
            options: .storageModeShared))

        let oldCB = try #require(context.queue.makeCommandBuffer())
        _ = grouped.encodeStreamedBatched(
            commandBuffer: oldCB, hidden: hiddenBuffer,
            sortedPairs: metadata.sortedPairs, routePartials: oldOutput,
            gateUpActScratch: oldAct, downScratch: oldDown,
            argumentBuffer: oldArgument, binding: binding,
            params: params, pairMicrobatchRows: pairs.count)
        oldCB.commit()
        oldCB.waitUntilCompleted()
        try #require(oldCB.error == nil)

        let newCB = try #require(context.queue.makeCommandBuffer())
        #expect(mpp.encode(
            commandBuffer: newCB, hidden: hiddenBuffer,
            sortedPairs: metadata.sortedPairs, groups: metadata.groups,
            activation: newAct, routePartials: newOutput,
            argumentBuffer: mppArgument, binding: binding,
            params: mppParams,
            maxPairsPerGroup: routes.maxPairsPerExpert))
        newCB.commit()
        newCB.waitUntilCompleted()
        try #require(newCB.error == nil)

        let residentParams = PrefillGroupedRoutedMoEStreamedParams(
            groupStart: 0, groupCount: UInt32(routes.groups.count),
            d: UInt32(d), routedIntermediate: UInt32(f),
            topK: UInt32(topK), hiddenStrideElements: UInt32(d),
            offsets: pool.offsets)
        let residentCB = try #require(context.queue.makeCommandBuffer())
        #expect(mpp.encodeResident(
            commandBuffer: residentCB, hidden: hiddenBuffer,
            sortedPairs: metadata.sortedPairs, groups: metadata.groups,
            activation: residentAct, routePartials: residentOutput,
            slab: residentSlab, params: residentParams,
            residentExpertStride: UInt32(pool.stride),
            maxPairsPerGroup: routes.maxPairsPerExpert))
        residentCB.commit()
        residentCB.waitUntilCompleted()
        try #require(residentCB.error == nil)

        let old = Fp16Buffer.readHalf(oldOutput, count: rows * topK * d)
        let new = Fp16Buffer.readHalf(newOutput, count: rows * topK * d)
        let resident = Fp16Buffer.readHalf(residentOutput,
                                           count: rows * topK * d)
        let maxError = zip(old, new).reduce(Float(0)) {
            max($0, abs(Float($1.0) - Float($1.1)))
        }
        #expect(maxError <= 0.003, "maxError=\(maxError)")
        #expect(resident == new)

        // Production resident grouping and indirect dispatch share one command
        // buffer. Include empty groups and a partial matrix tile.
        let grouping = try PrefillDeviceMoEGrouping(context: context)
        let routeScratch = try PrefillDeviceMoEGrouping.Scratch(
            device: context.device, maxPairs: pairs.count, experts: 5)
        let ids = pairs.map { $0.expert }
        let weights = pairs.map { $0.weight }
        let idBuffer = try #require(context.device.makeBuffer(bytes: ids,
            length: ids.count * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let weightBuffer = try #require(Fp16Buffer.make(context.device, halves: weights))
        let indirectCB = try #require(context.queue.makeCommandBuffer())
        try grouping.encode(commandBuffer: indirectCB, ids: idBuffer, weights: weightBuffer,
            scratch: routeScratch, rows: rows, topK: topK, hidden: d, intermediate: f)
        let indirectParams = PrefillGroupedRoutedMoEStreamedParams(
            groupStart: 0, groupCount: 5, d: UInt32(d), routedIntermediate: UInt32(f),
            topK: UInt32(topK), hiddenStrideElements: UInt32(d), offsets: pool.offsets)
        #expect(mpp.encodeResident(commandBuffer: indirectCB, hidden: hiddenBuffer,
            sortedPairs: routeScratch.pairs, groups: routeScratch.groups,
            activation: residentAct, routePartials: residentOutput,
            slab: residentSlab, params: indirectParams,
            residentExpertStride: UInt32(pool.stride), maxPairsPerGroup: rows,
            indirectDispatch: routeScratch.dispatch))
        indirectCB.commit()
        indirectCB.waitUntilCompleted()
        try #require(indirectCB.error == nil)
        #expect(Fp16Buffer.readHalf(residentOutput, count: rows * topK * d) == new)
    }
}
