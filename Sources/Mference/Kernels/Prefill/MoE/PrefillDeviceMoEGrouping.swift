import Metal

/// Stable expert-major grouping, without a route readback. Empty experts have
/// zero-sized groups; within each group token/rank order matches the CPU path.
final class PrefillDeviceMoEGrouping {
    struct Scratch {
        let counts: MTLBuffer
        let groups: MTLBuffer
        let pairs: MTLBuffer
        let dispatch: MTLBuffer
        let maxPairs: Int
        let experts: Int

        init(device: MTLDevice, maxPairs: Int, experts: Int) throws {
            self.maxPairs = maxPairs
            self.experts = experts
            func buffer(_ bytes: Int) throws -> MTLBuffer {
                guard let result = device.makeBuffer(length: max(4, bytes), options: .storageModeShared) else {
                    throw PrefillGroupedRoutedMoEError.allocationFailed("device route grouping")
                }
                return result
            }
            counts = try buffer(experts * 4)
            groups = try buffer(experts * MemoryLayout<PrefillMoEGroup>.stride)
            pairs = try buffer(maxPairs * MemoryLayout<PrefillTokenExpertPair>.stride)
            dispatch = try buffer(6 * 4)
        }
    }

    private let count: MTLComputePipelineState
    private let prefix: MTLComputePipelineState
    private let scatter: MTLComputePipelineState

    init(context: MetalContext) throws {
        count = try context.pipeline("prefill_route_count_by_expert")
        prefix = try context.pipeline("prefill_route_group_prefix")
        scatter = try context.pipeline("prefill_route_scatter_by_expert")
    }

    func encode(commandBuffer cb: MTLCommandBuffer, ids: MTLBuffer,
                weights: MTLBuffer, scratch: Scratch, rows: Int, topK: Int,
                hidden: Int, intermediate: Int) throws {
        precondition(rows > 0 && topK > 0 && rows * topK <= scratch.maxPairs)
        precondition(scratch.experts > 0)
        var pairCount = UInt32(rows * topK)
        var experts = UInt32(scratch.experts)
        var k = UInt32(topK)
        var d = UInt32(hidden)
        var f = UInt32(intermediate)
        guard let a = cb.makeComputeCommandEncoder() else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("route count encoder")
        }
        a.setComputePipelineState(count)
        a.setBuffer(ids, offset: 0, index: 0)
        a.setBuffer(scratch.counts, offset: 0, index: 1)
        a.setBytes(&pairCount, length: 4, index: 2)
        a.setBytes(&experts, length: 4, index: 3)
        a.dispatchThreads(MTLSize(width: scratch.experts, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        a.endEncoding()
        guard let b = cb.makeComputeCommandEncoder() else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("route prefix encoder")
        }
        b.setComputePipelineState(prefix)
        b.setBuffer(scratch.counts, offset: 0, index: 0)
        b.setBuffer(scratch.groups, offset: 0, index: 1)
        b.setBuffer(scratch.dispatch, offset: 0, index: 2)
        b.setBytes(&experts, length: 4, index: 3)
        b.setBytes(&d, length: 4, index: 4)
        b.setBytes(&f, length: 4, index: 5)
        b.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        b.endEncoding()
        guard let c = cb.makeComputeCommandEncoder() else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("route scatter encoder")
        }
        c.setComputePipelineState(scatter)
        c.setBuffer(ids, offset: 0, index: 0)
        c.setBuffer(weights, offset: 0, index: 1)
        c.setBuffer(scratch.groups, offset: 0, index: 2)
        c.setBuffer(scratch.pairs, offset: 0, index: 3)
        c.setBytes(&pairCount, length: 4, index: 4)
        c.setBytes(&experts, length: 4, index: 5)
        c.setBytes(&k, length: 4, index: 6)
        c.dispatchThreads(MTLSize(width: scratch.experts, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        c.endEncoding()
    }
}
