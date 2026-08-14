import Foundation
import Metal

/// DeepSeek-V4 routed-MoE runtime: sqrtsoftplus top-6 routing (learned or
/// hash-selected), INT2 affine expert weights, and the swiglu_limit clamp.
///
/// Kept separate from `MoE` because that class's kernels and argument-buffer
/// contract are validated against exactly 8 streamed experts and INT4 rows;
/// this one is validated against exactly 6 and INT2. The router gate is
/// unquantized BF16 in the mlx-community checkpoint, so the GEMV is the
/// dedicated `router_gemv_bf16_r4` (with a ones effective-scale buffer,
/// the Qwen precedent for unscaled routers).
final class MoEDeepseekV4 {
    static let topK = 6

    private let realDecodeD: UInt32
    private let realDecodeF: UInt32
    private let realDecodeNumExperts: UInt32
    private let routeScale: Float

    private let routerGemvPSO: MTLComputePipelineState
    private let routerSelectK6PSO: MTLComputePipelineState
    private let hashWeightsK6PSO: MTLComputePipelineState
    private let phase1Int2PSO: MTLComputePipelineState
    private let phase1SubsetInt2PSO: MTLComputePipelineState
    private let phase2ReduceInt2K6PSO: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer

    init(context: MetalContext,
         specializedD: UInt32,
         specializedF: UInt32,
         specializedNumExperts: UInt32,
         routeScale: Float,
         swigluLimit: Float) throws {
        self.realDecodeD = specializedD
        self.realDecodeF = specializedF
        self.realDecodeNumExperts = specializedNumExperts
        self.routeScale = routeScale
        let moeConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 0, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 1, value: .uint32(specializedF)),
            MetalFunctionConstant(index: 2, value: .uint32(UInt32(Self.topK))),
            MetalFunctionConstant(index: 3, value: .bool(true)),
            MetalFunctionConstant(index: 4, value: .bool(true)),   // silu
            MetalFunctionConstant(index: 5, value: .float(swigluLimit)),
        ]
        let routerConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 40, value: .uint32(specializedNumExperts)),
            MetalFunctionConstant(index: 41, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 42, value: .uint32(UInt32(Self.topK))),
            MetalFunctionConstant(index: 43, value: .bool(true)),
        ]
        self.routerGemvPSO = try context.pipeline(
            "router_gemv_bf16_r4",
            constants: routerConstants,
            maxTotalThreadsPerThreadgroup: 512)
        // One-SIMD parallel selection and weighting; bit-identical to the serial
        // reference kernels (see RouterTopKParityTests).
        self.routerSelectK6PSO = try context.pipeline(
            "router_topk_select_sqrtsoftplus_k6_par",
            constants: routerConstants)
        self.hashWeightsK6PSO = try context.pipeline("router_hash_weights_k6_par")
        self.phase1Int2PSO = try context.pipeline(
            "moe_phase1_gate_up_act_int2", constants: moeConstants)
        self.phase1SubsetInt2PSO = try context.pipeline(
            "moe_phase1_gate_up_act_subset_int2", constants: moeConstants)
        self.phase2ReduceInt2K6PSO = try context.pipeline(
            "moe_phase2_down_reduce_int2_k6", constants: moeConstants)

        guard let logits = context.device.makeBuffer(
            length: Int(specializedNumExperts) * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let phase1Function = context.library.makeFunction(
                name: "moe_phase1_gate_up_act_int2") else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    /// Router logits + learned top-6 selection. `correctionBias` is the FP32
    /// `e_score_correction_bias` tensor (selection only; weights use raw
    /// sqrtsoftplus scores).
    func encodeRouterTopK(commandBuffer: MTLCommandBuffer,
                          weights: MTLBuffer, weightsOffset: Int,
                          hidden: MTLBuffer,
                          onesScale: MTLBuffer,
                          correctionBias: MTLBuffer, correctionBiasOffset: Int,
                          outIndices: MTLBuffer,
                          outWeights: MTLBuffer,
                          numExperts: UInt32,
                          d: UInt32) {
        precondition(numExperts <= 256)
        encodeRouterGemv(commandBuffer: commandBuffer,
                         weights: weights, weightsOffset: weightsOffset,
                         hidden: hidden, onesScale: onesScale,
                         numExperts: numExperts, d: d)

        var expertCount = numExperts
        var scale = routeScale
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(routerSelectK6PSO)
            encoder.setBuffer(routerLogits, offset: 0, index: 0)
            encoder.setBuffer(correctionBias, offset: correctionBiasOffset, index: 1)
            encoder.setBuffer(outIndices, offset: 0, index: 2)
            encoder.setBuffer(outWeights, offset: 0, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    /// Hash-routed layers: `indices` already holds the 6 `tid2eid[token]`
    /// experts (written by the CPU before commit); the gate only weights them.
    func encodeRouterHashWeights(commandBuffer: MTLCommandBuffer,
                                 weights: MTLBuffer, weightsOffset: Int,
                                 hidden: MTLBuffer,
                                 onesScale: MTLBuffer,
                                 indices: MTLBuffer,
                                 outWeights: MTLBuffer,
                                 numExperts: UInt32,
                                 d: UInt32) {
        encodeRouterGemv(commandBuffer: commandBuffer,
                         weights: weights, weightsOffset: weightsOffset,
                         hidden: hidden, onesScale: onesScale,
                         numExperts: numExperts, d: d)

        var scale = routeScale
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(hashWeightsK6PSO)
            encoder.setBuffer(routerLogits, offset: 0, index: 0)
            encoder.setBuffer(indices, offset: 0, index: 1)
            encoder.setBuffer(outWeights, offset: 0, index: 2)
            encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    private func encodeRouterGemv(commandBuffer: MTLCommandBuffer,
                                  weights: MTLBuffer, weightsOffset: Int,
                                  hidden: MTLBuffer,
                                  onesScale: MTLBuffer,
                                  numExperts: UInt32,
                                  d: UInt32) {
        var expertCount = numExperts
        var dimension = d
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(routerGemvPSO)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(hidden, offset: 0, index: 1)
        encoder.setBuffer(onesScale, offset: 0, index: 2)
        encoder.setBuffer(routerLogits, offset: 0, index: 3)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [(buffer: MTLBuffer, offset: Int)]) -> MTLBuffer {
        validate(routedBlobs: routedBlobs)
        routedArgEncoder.setArgumentBuffer(reusableRoutedArgBuffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(blob.buffer, offset: blob.offset, index: index)
        }
        // The Metal-side RoutedBlobs struct carries 8 slots; point the two
        // unused ones at a valid buffer so the argument buffer never holds a
        // dangling reference.
        if let first = routedBlobs.first {
            routedArgEncoder.setBuffer(first.buffer, offset: first.offset, index: 6)
            routedArgEncoder.setBuffer(first.buffer, offset: first.offset, index: 7)
        }
        return reusableRoutedArgBuffer
    }

    func encodeRoutedPhase1(commandBuffer: MTLCommandBuffer,
                            routedArgBuffer: MTLBuffer,
                            routedBlobs: [(buffer: MTLBuffer, offset: Int)],
                            routedOffsets: MoEExpertOffsets,
                            x: MTLBuffer,
                            acts: MTLBuffer,
                            d: UInt32,
                            f: UInt32,
                            gateGroupSize: UInt32) {
        validate(routedBlobs: routedBlobs)
        var dimension = d
        var intermediate = f
        var expertCount = UInt32(Self.topK)
        var gateGroup = gateGroupSize
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(phase1Int2PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for blob in routedBlobs { encoder.useResource(blob.buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&gateGroup, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Self.topK * Int(f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPhase1Subset(commandBuffer: MTLCommandBuffer,
                                  routedArgBuffer: MTLBuffer,
                                  routedBlobs: [(buffer: MTLBuffer, offset: Int)],
                                  routedOffsets: MoEExpertOffsets,
                                  x: MTLBuffer,
                                  acts: MTLBuffer,
                                  activeSlots: MTLBuffer,
                                  activeSlotIndices: [UInt32],
                                  activeCount: UInt32,
                                  d: UInt32,
                                  f: UInt32,
                                  gateGroupSize: UInt32) {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs)
        precondition(activeSlotIndices.count == Int(activeCount))
        var dimension = d
        var intermediate = f
        var expertCount = UInt32(Self.topK)
        var active = activeCount
        var gateGroup = gateGroupSize
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(phase1SubsetInt2PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for slot in activeSlotIndices {
            encoder.useResource(routedBlobs[Int(slot)].buffer, usage: .read)
        }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(activeSlots, offset: 0, index: 7)
        encoder.setBytes(&active, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&gateGroup, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount) * Int(f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPhase2Reduce(commandBuffer: MTLCommandBuffer,
                                  routedArgBuffer: MTLBuffer,
                                  routedBlobs: [(buffer: MTLBuffer, offset: Int)],
                                  routedOffsets: MoEExpertOffsets,
                                  acts: MTLBuffer,
                                  routingWeights: MTLBuffer,
                                  residual: MTLBuffer,
                                  y: MTLBuffer,
                                  d: UInt32,
                                  f: UInt32) {
        validate(routedBlobs: routedBlobs)
        var dimension = d
        var intermediate = f
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(phase2ReduceInt2K6PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for blob in routedBlobs { encoder.useResource(blob.buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        // Six simdgroups: one per streamed expert.
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func validate(routedBlobs: [(buffer: MTLBuffer, offset: Int)]) {
        precondition(routedBlobs.count == Self.topK)
    }
}
