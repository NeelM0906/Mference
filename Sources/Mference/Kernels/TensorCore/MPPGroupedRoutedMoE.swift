import Foundation
import Metal

struct MPPGroupedRoutedMoEArgumentBuffer {
    let buffer: MTLBuffer
}

/// MPP's Metal ABI extends the established 128-byte streamed parameter prefix
/// without changing that prefix for the older prefill kernels.
private struct MPPGroupedRoutedMoEParams {
    var streamed: PrefillGroupedRoutedMoEStreamedParams
    var residentExpertStride: UInt32
}

/// TensorOps grouped-GEMM for routed prefill. One dispatch covers up to 16
/// expert groups; each cooperative tile is 64 routed rows by 32 output rows,
/// matching the dense affine kernel's proven matrix shape. The runtime selects
/// this path only for long chunks whose per-expert groups amortize that tile.
final class MPPGroupedRoutedMoE {
    private var phase1: MTLComputePipelineState?
    private var down: MTLComputePipelineState?
    private var argumentEncoder: MTLArgumentEncoder?

    init(context: MetalContext) {
        do {
            let library = try MetalContext.moduleLibrary(
                device: context.device, module: "tensorops")
            guard let phase1Function = library.makeFunction(
                    name: "mpp_grouped_routed_moe_phase1"),
                  let downFunction = library.makeFunction(
                    name: "mpp_grouped_routed_moe_down") else {
                throw MetalError.missingFunction("mpp grouped routed MoE")
            }
            self.phase1 = try context.device.makeComputePipelineState(
                function: phase1Function)
            self.down = try context.device.makeComputePipelineState(
                function: downFunction)
            self.argumentEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 5)
        } catch {
            self.phase1 = nil
            self.down = nil
            self.argumentEncoder = nil
        }
    }

    var isAvailable: Bool { phase1 != nil && down != nil && argumentEncoder != nil }

    func makeArgumentBuffer(device: MTLDevice,
                            binding: PrefillStreamedTileBinding) throws
        -> MPPGroupedRoutedMoEArgumentBuffer {
        guard let argumentEncoder,
              let buffer = device.makeBuffer(length: argumentEncoder.encodedLength,
                                             options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed(
                "MPP grouped routed expert argument buffer")
        }
        argumentEncoder.setArgumentBuffer(buffer, offset: 0)
        for index in binding.views.indices {
            let view = binding.views[index]
            argumentEncoder.setBuffer(view.buffer, offset: Int(view.offset), index: index)
        }
        return MPPGroupedRoutedMoEArgumentBuffer(buffer: buffer)
    }

    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                hidden: MTLBuffer,
                sortedPairs: MTLBuffer,
                groups: MTLBuffer,
                activation: MTLBuffer,
                routePartials: MTLBuffer,
                argumentBuffer: MPPGroupedRoutedMoEArgumentBuffer,
                binding: PrefillStreamedTileBinding,
                params: PrefillGroupedRoutedMoEStreamedParams,
                maxPairsPerGroup: Int) -> Bool {
        guard params.pairCount > 0, maxPairsPerGroup > 0,
              let phase1, let down else { return false }

        func bind(_ enc: MTLComputeCommandEncoder) {
            enc.setBuffer(hidden, offset: 0, index: 0)
            enc.setBuffer(sortedPairs, offset: 0, index: 1)
            enc.setBuffer(groups, offset: 0, index: 2)
            enc.setBuffer(activation, offset: 0, index: 3)
            enc.setBuffer(routePartials, offset: 0, index: 4)
            enc.setBuffer(argumentBuffer.buffer, offset: 0, index: 5)
            var p = MPPGroupedRoutedMoEParams(
                streamed: params, residentExpertStride: 0)
            enc.setBytes(&p,
                         length: MemoryLayout<MPPGroupedRoutedMoEParams>.stride,
                         index: 6)
            // Streamed mode never dereferences buffer 7, but Metal requires
            // every statically referenced argument to be bound.
            enc.setBuffer(binding.views[0].buffer, offset: 0, index: 7)
            for view in binding.views { enc.useResource(view.buffer, usage: .read) }
        }

        guard let first = commandBuffer.makeComputeCommandEncoder() else { return false }
        first.setComputePipelineState(phase1)
        bind(first)
        first.dispatchThreadgroups(
            MTLSize(width: (Int(params.routedIntermediate) + 31) / 32,
                    height: (maxPairsPerGroup + 63) / 64,
                    depth: Int(params.pairCount)),
            threadsPerThreadgroup: MTLSize(width: phase1.threadExecutionWidth * 4,
                                           height: 1, depth: 1))
        first.endEncoding()

        guard let second = commandBuffer.makeComputeCommandEncoder() else { return false }
        second.setComputePipelineState(down)
        bind(second)
        second.dispatchThreadgroups(
            MTLSize(width: (Int(params.d) + 31) / 32,
                    height: (maxPairsPerGroup + 63) / 64,
                    depth: Int(params.pairCount)),
            threadsPerThreadgroup: MTLSize(width: down.threadExecutionWidth * 4,
                                           height: 1, depth: 1))
        second.endEncoding()
        return true
    }

    /// One pair of TensorOps dispatches covers every live group in a resident
    /// layer slab. `group.expert * expertStride` replaces 16-entry streamed
    /// argument-buffer tiles on high-memory Macs.
    @discardableResult
    func encodeResident(commandBuffer: MTLCommandBuffer,
                        hidden: MTLBuffer,
                        sortedPairs: MTLBuffer,
                        groups: MTLBuffer,
                        activation: MTLBuffer,
                        routePartials: MTLBuffer,
                        slab: MTLBuffer,
                        params: PrefillGroupedRoutedMoEStreamedParams,
                        residentExpertStride: UInt32,
                        maxPairsPerGroup: Int) -> Bool {
        guard params.pairCount > 0, maxPairsPerGroup > 0,
              params.liveExpertCount == 0,
              residentExpertStride > 0,
              let phase1, let down else { return false }

        func bind(_ enc: MTLComputeCommandEncoder) {
            enc.setBuffer(hidden, offset: 0, index: 0)
            enc.setBuffer(sortedPairs, offset: 0, index: 1)
            enc.setBuffer(groups, offset: 0, index: 2)
            enc.setBuffer(activation, offset: 0, index: 3)
            enc.setBuffer(routePartials, offset: 0, index: 4)
            // Resident mode never dereferences the streamed argument buffer.
            enc.setBuffer(slab, offset: 0, index: 5)
            var p = MPPGroupedRoutedMoEParams(
                streamed: params,
                residentExpertStride: residentExpertStride)
            enc.setBytes(&p,
                         length: MemoryLayout<MPPGroupedRoutedMoEParams>.stride,
                         index: 6)
            enc.setBuffer(slab, offset: 0, index: 7)
            enc.useResource(slab, usage: .read)
        }

        guard let first = commandBuffer.makeComputeCommandEncoder() else { return false }
        first.setComputePipelineState(phase1)
        bind(first)
        first.dispatchThreadgroups(
            MTLSize(width: (Int(params.routedIntermediate) + 31) / 32,
                    height: (maxPairsPerGroup + 63) / 64,
                    depth: Int(params.pairCount)),
            threadsPerThreadgroup: MTLSize(width: phase1.threadExecutionWidth * 4,
                                           height: 1, depth: 1))
        first.endEncoding()

        guard let second = commandBuffer.makeComputeCommandEncoder() else { return false }
        second.setComputePipelineState(down)
        bind(second)
        second.dispatchThreadgroups(
            MTLSize(width: (Int(params.d) + 31) / 32,
                    height: (maxPairsPerGroup + 63) / 64,
                    depth: Int(params.pairCount)),
            threadsPerThreadgroup: MTLSize(width: down.threadExecutionWidth * 4,
                                           height: 1, depth: 1))
        second.endEncoding()
        return true
    }
}
