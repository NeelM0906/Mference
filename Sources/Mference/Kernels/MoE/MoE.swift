import Foundation
import Metal

@frozen
public struct MoEExpertOffsets {
    public var gateWOff: UInt32
    public var gateSOff: UInt32
    public var gateBOff: UInt32
    public var upWOff: UInt32
    public var upSOff: UInt32
    public var upBOff: UInt32
    public var downWOff: UInt32
    public var downSOff: UInt32
    public var downBOff: UInt32

    public init(gateWOff: UInt32, gateSOff: UInt32, gateBOff: UInt32,
                upWOff: UInt32, upSOff: UInt32, upBOff: UInt32,
                downWOff: UInt32, downSOff: UInt32, downBOff: UInt32) {
        self.gateWOff = gateWOff
        self.gateSOff = gateSOff
        self.gateBOff = gateBOff
        self.upWOff = upWOff
        self.upSOff = upSOff
        self.upBOff = upBOff
        self.downWOff = downWOff
        self.downSOff = downSOff
        self.downBOff = downBOff
    }
}

final class MoE {
    static let maxStreamedExperts = 8

    private let realDecodeD: UInt32
    private let realDecodeF: UInt32
    private let realDecodeTopK: UInt32
    private let realDecodeNumExperts: UInt32

    private let routerGemvPSO: MTLComputePipelineState
    private let routerGemvSpecializedPSO: MTLComputePipelineState
    private let routerSelectK8PSO: MTLComputePipelineState
    private let routerSelectK8SpecializedPSO: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let phase1U16PSO: MTLComputePipelineState
    private let phase1U16SpecializedPSO: MTLComputePipelineState
    private let slotLookupPSO: MTLComputePipelineState
    private let phase1SlotmapPSO: MTLComputePipelineState
    private let phase1SlotmapSpecializedPSO: MTLComputePipelineState
    private let phase2SlotmapPSO: MTLComputePipelineState
    private let phase2SlotmapSpecializedPSO: MTLComputePipelineState
    private let residualAddGuardedPSO: MTLComputePipelineState
    private let phase1SubsetU16PSO: MTLComputePipelineState
    private let phase1SubsetU16SpecializedPSO: MTLComputePipelineState
    private let phase2ReduceK8PSO: MTLComputePipelineState
    private let phase2ReduceK8SpecializedPSO: MTLComputePipelineState
    private let phase2ReduceK6PSO: MTLComputePipelineState
    private let phase2ReduceK6SpecializedPSO: MTLComputePipelineState
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer

    /// `specializedD`/`specializedF`/`specializedNumExperts` describe the
    /// production shape this instance specializes for (Gemma 4 by default;
    /// Qwen 3.6 passes 2048/512/256). `siluActivation` selects the expert
    /// FFN activation (false = gelu_pytorch_tanh, true = silu).
    init(context: MetalContext,
         siluActivation: Bool = false,
         specializedD: UInt32 = 2816,
         specializedF: UInt32 = 704,
         specializedNumExperts: UInt32 = 128,
         specializedTopK: UInt32 = 8) throws {
        precondition(specializedTopK == 6 || specializedTopK == 8,
                     "routed INT4 decode supports top-k 6 or 8")
        self.realDecodeD = specializedD
        self.realDecodeF = specializedF
        self.realDecodeTopK = specializedTopK
        self.realDecodeNumExperts = specializedNumExperts
        let activationConstants: [MetalFunctionConstant] = siluActivation
            ? [MetalFunctionConstant(index: 4, value: .bool(true))]
            : []
        let moeConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 0, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 1, value: .uint32(specializedF)),
            MetalFunctionConstant(index: 2, value: .uint32(specializedTopK)),
            MetalFunctionConstant(index: 3, value: .bool(true)),
        ] + activationConstants
        let routerConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 40, value: .uint32(specializedNumExperts)),
            MetalFunctionConstant(index: 41, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 42, value: .uint32(specializedTopK)),
            MetalFunctionConstant(index: 43, value: .bool(true)),
        ]
        let routerName = "router_gemv_gemma4_r4"
        self.routerGemvPSO = try context.pipeline(
            routerName,
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.routerGemvSpecializedPSO = try context.pipeline(
            routerName,
            constants: routerConstants,
            maxTotalThreadsPerThreadgroup: 512)
        // One-SIMD parallel selection; bit-identical to the serial
        // `router_topk_select_k8` reference kernel (see RouterTopKParityTests).
        self.routerSelectK8PSO = try context.pipeline("router_topk_select_k8_par")
        self.routerSelectK8SpecializedPSO = try context.pipeline(
            "router_topk_select_k8_par",
            constants: routerConstants)
        self.phase1U16PSO = try context.pipeline(
            "moe_phase1_gate_up_act_u16load", constants: activationConstants)
        self.phase1U16SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_u16load",
            constants: moeConstants)
        self.slotLookupPSO = try context.pipeline("router_slot_lookup_k8")
        self.phase1SlotmapPSO = try context.pipeline(
            "moe_phase1_gate_up_act_slotmap", constants: activationConstants)
        self.phase1SlotmapSpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_slotmap", constants: moeConstants)
        self.phase2SlotmapPSO = try context.pipeline(
            "moe_phase2_down_reduce_k8_slotmap")
        self.phase2SlotmapSpecializedPSO = try context.pipeline(
            "moe_phase2_down_reduce_k8_slotmap", constants: moeConstants)
        self.residualAddGuardedPSO = try context.pipeline("residual_add_fp16_guarded")
        self.phase1SubsetU16PSO = try context.pipeline(
            "moe_phase1_gate_up_act_subset_u16load", constants: activationConstants)
        self.phase1SubsetU16SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_subset_u16load",
            constants: moeConstants)
        self.phase2ReduceK8PSO = try context.pipeline("moe_phase2_down_reduce_k8")
        self.phase2ReduceK8SpecializedPSO = try context.pipeline(
            "moe_phase2_down_reduce_k8",
            constants: moeConstants)
        self.phase2ReduceK6PSO = try context.pipeline("moe_phase2_down_reduce_k6")
        self.phase2ReduceK6SpecializedPSO = try context.pipeline(
            "moe_phase2_down_reduce_k6",
            constants: moeConstants)

        guard let logits = context.device.makeBuffer(
            length: 256 * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let phase1Function = context.library.makeFunction(
                name: "moe_phase1_gate_up_act_u16load") else {
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

    func encodeRouterGemma4(commandBuffer: MTLCommandBuffer,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   scales: MTLBuffer, scalesOffset: Int = 0,
                                   biases: MTLBuffer, biasesOffset: Int = 0,
                                   hidden: MTLBuffer,
                                   effectiveScale: MTLBuffer, effectiveScaleOffset: Int = 0,
                                   perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                                   outIndices: MTLBuffer,
                                   outWeights: MTLBuffer,
                                   numExperts: UInt32,
                                   d: UInt32,
                                   topK: UInt32) {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(numExperts <= 256)
        precondition(topK == UInt32(Self.maxStreamedExperts))

        var expertCount = numExperts
        var dimension = d
        let useSpecialized = numExperts == realDecodeNumExperts
            && d == realDecodeD
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            encoder.setBuffer(biases, offset: biasesOffset, index: 2)
            encoder.setBuffer(hidden, offset: 0, index: 3)
            encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
            encoder.setBuffer(routerLogits, offset: 0, index: 5)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerSelectK8SpecializedPSO : routerSelectK8PSO)
            encoder.setBuffer(routerLogits, offset: 0, index: 0)
            encoder.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
            encoder.setBuffer(outIndices, offset: 0, index: 2)
            encoder.setBuffer(outWeights, offset: 0, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    func makeRoutedArgumentBuffer(routedBlobs: [(buffer: MTLBuffer, offset: Int)],
                                         topK: UInt32) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer,
                                   routedBlobs: routedBlobs)
        return reusableRoutedArgBuffer
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [(buffer: MTLBuffer, offset: Int)],
                                               topK: UInt32) -> MTLBuffer {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer, routedBlobs: routedBlobs)
        return reusableRoutedArgBuffer
    }

    func encodeRoutedPersistentPhase1U16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [(buffer: MTLBuffer, offset: Int)],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f, topK: topK)
                ? phase1U16SpecializedPSO
                : phase1U16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for blob in routedBlobs { encoder.useResource(blob.buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase1SubsetU16Load(
        commandBuffer: MTLCommandBuffer,
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
        topK: UInt32
    ) {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs, topK: topK)
        precondition(activeSlotIndices.count == Int(activeCount))
        var dimension = d
        var intermediate = f
        var expertCount = topK
        var active = activeCount
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f, topK: topK)
                ? phase1SubsetU16SpecializedPSO
                : phase1SubsetU16PSO)
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
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// S3 lookup, encoded BEFORE the router signal so the CPU can trust the
    /// all-hit flag and slab offsets the moment it wakes.
    func encodeSlotLookup(commandBuffer: MTLCommandBuffer,
                          indices: MTLBuffer,
                          table: MTLBuffer,
                          slotStride: Int,
                          slotOffsets: MTLBuffer,
                          allHit: MTLBuffer,
                          numExperts: UInt32,
                          topK: UInt32) {
        precondition(topK == UInt32(Self.maxStreamedExperts))
        var expertCount = numExperts
        var k = topK
        var stride = UInt32(slotStride)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(slotLookupPSO)
        encoder.setBuffer(indices, offset: 0, index: 0)
        encoder.setBuffer(table, offset: 0, index: 1)
        encoder.setBuffer(slotOffsets, offset: 0, index: 2)
        encoder.setBuffer(allHit, offset: 0, index: 3)
        encoder.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&stride, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// S3 guarded routed FFN, encoded after the shared expert in cb1. The
    /// three kernels no-op unless the lookup found every routed expert in a
    /// slot; the math bodies are the production ones, so an all-hit layer is
    /// byte-identical to the argument-buffer fallback.
    func encodeSlotMapGuardedFFN(commandBuffer: MTLCommandBuffer,
                                 slab: MTLBuffer,
                                 slotOffsets: MTLBuffer,
                                 allHit: MTLBuffer,
                                 routedOffsets: MoEExpertOffsets,
                                 x: MTLBuffer,
                                 acts: MTLBuffer,
                                 routingWeights: MTLBuffer,
                                 residual: MTLBuffer,
                                 y: MTLBuffer,
                                 hidden: MTLBuffer,
                                 d: UInt32,
                                 f: UInt32,
                                 topK: UInt32) {
        precondition(topK == UInt32(Self.maxStreamedExperts))
        var offsets = routedOffsets
        var dimension = d
        var intermediate = f
        var k = topK
        let specialized = useRealDecodeConstants(d: d, f: f, topK: topK)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            preconditionFailure("slot-map guarded FFN: no compute encoder")
        }
        encoder.useResource(slab, usage: .read)
        encoder.setComputePipelineState(
            specialized ? phase1SlotmapSpecializedPSO : phase1SlotmapPSO)
        encoder.setBuffer(slab, offset: 0, index: 0)
        encoder.setBuffer(slotOffsets, offset: 0, index: 1)
        encoder.setBuffer(allHit, offset: 0, index: 2)
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 3)
        encoder.setBuffer(x, offset: 0, index: 4)
        encoder.setBuffer(acts, offset: 0, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        encoder.setComputePipelineState(
            specialized ? phase2SlotmapSpecializedPSO : phase2SlotmapPSO)
        encoder.setBuffer(slab, offset: 0, index: 0)
        encoder.setBuffer(slotOffsets, offset: 0, index: 1)
        encoder.setBuffer(allHit, offset: 0, index: 2)
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 3)
        encoder.setBuffer(acts, offset: 0, index: 4)
        encoder.setBuffer(routingWeights, offset: 0, index: 5)
        encoder.setBuffer(residual, offset: 0, index: 6)
        encoder.setBuffer(y, offset: 0, index: 7)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        encoder.setComputePipelineState(residualAddGuardedPSO)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(y, offset: 0, index: 1)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBuffer(allHit, offset: 0, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(d) + 255) / 256, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase2Reduce(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [(buffer: MTLBuffer, offset: Int)],
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        routingWeights: MTLBuffer,
        residual: MTLBuffer,
        y: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let specialized = useRealDecodeConstants(d: d, f: f, topK: topK)
        if topK == 6 {
            encoder.setComputePipelineState(
                specialized ? phase2ReduceK6SpecializedPSO : phase2ReduceK6PSO)
        } else {
            encoder.setComputePipelineState(
                specialized ? phase2ReduceK8SpecializedPSO : phase2ReduceK8PSO)
        }
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
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: Int(topK) * 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func validate(routedBlobs: [(buffer: MTLBuffer, offset: Int)], topK: UInt32) {
        precondition(topK == 6 || topK == 8,
                     "routed INT4 decode supports top-k 6 or 8")
        precondition(routedBlobs.count == Int(topK))
    }

    private func encodeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                            routedBlobs: [(buffer: MTLBuffer, offset: Int)]) {
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(blob.buffer, offset: blob.offset, index: index)
        }
    }

    private func useRealDecodeConstants(d: UInt32,
                                        f: UInt32,
                                        topK: UInt32) -> Bool {
        d == realDecodeD && f == realDecodeF && topK == realDecodeTopK
    }
}
