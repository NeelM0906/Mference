import Metal
import Testing
@testable import Mference

@Suite struct PrefillDeviceMoEGroupingTests {
    @Test func deviceGroupingMatchesStableCPUOrderAndReusesScratch() throws {
        let ctx = try MetalContext()
        let kernel = try PrefillDeviceMoEGrouping(context: ctx)
        for experts in [7, 512] {
            let topK = min(experts, 10)
            let scratch = try PrefillDeviceMoEGrouping.Scratch(device: ctx.device,
                maxPairs: 1025 * topK, experts: experts)
            for rows in [1, 37, 128, 1025, 3] {
                var ids: [UInt32] = []
                var weights: [Float16] = []
                for t in 0..<rows {
                    for rank in 0..<topK {
                        ids.append(UInt32((t * 17 + rank) % experts))
                        weights.append(rank == 0 ? Float16(-0.0) : Float16(Float(rank) / 17))
                    }
                }
                let input = try #require(ctx.device.makeBuffer(bytes: ids, length: ids.count * 4,
                                                               options: .storageModeShared))
                let scores = try #require(ctx.device.makeBuffer(bytes: weights, length: weights.count * 2,
                                                                options: .storageModeShared))
                let cb = try #require(ctx.queue.makeCommandBuffer())
                try kernel.encode(commandBuffer: cb, ids: input, weights: scores, scratch: scratch,
                    rows: rows, topK: topK, hidden: 4096, intermediate: 1536)
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.error == nil)
                let pairs = PrefillRouter.makeTokenExpertPairs(indices: ids, weights: weights,
                    queryCount: rows, topK: topK)
                let reference = try PrefillMoEGrouping.groupTokenExpertPairs(pairs,
                    queryCount: rows, topK: topK, numExperts: experts)
                let actualPairs = Array(UnsafeBufferPointer(
                    start: scratch.pairs.contents().assumingMemoryBound(to: PrefillTokenExpertPair.self),
                    count: pairs.count))
                #expect(actualPairs == reference.sortedPairs)
                let groups = scratch.groups.contents().assumingMemoryBound(to: PrefillMoEGroup.self)
                let counts = scratch.counts.contents().assumingMemoryBound(to: UInt32.self)
                var offset: UInt32 = 0
                for expert in 0..<experts {
                    #expect(counts[expert] == reference.perExpertCounts[expert])
                    #expect(groups[expert] == PrefillMoEGroup(expert: UInt32(expert), pairStart: offset,
                                                             pairCount: counts[expert]))
                    offset += counts[expert]
                }
                #expect(offset == UInt32(pairs.count))
                let dispatch = Array(UnsafeBufferPointer(
                    start: scratch.dispatch.contents().assumingMemoryBound(to: UInt32.self), count: 6))
                let height = UInt32((reference.maxPairsPerExpert + 63) / 64)
                #expect(dispatch == [48, height, UInt32(experts), 128, height, UInt32(experts)])
            }
        }
    }
}
