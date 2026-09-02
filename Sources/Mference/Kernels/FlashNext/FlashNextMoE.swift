import Foundation
import Metal

/// Routed-expert compute for a Flash-Next install whose experts are dense BF16.
///
/// The production `qwen38flashnext` install quantizes every routed expert to INT4
/// affine group-64 and goes through the shipped `MoE` kernels
/// (`moe_phase1_gate_up_act_u16load` + `moe_phase2_down_reduce_k10`). The parity
/// install carries them at their source BF16 — its `moe_intermediate_size` is 32,
/// which group-64 cannot quantize at all — so the runner needs both, the same
/// dual-dtype split `FlashNextWeightMatrix` makes for the resident projections.
///
/// Both paths bind the same 10-slot `RoutedBlobs` argument buffer and the same
/// `MoEExpertOffsets`, so a runner picks a pipeline per install and leaves its
/// expert-streaming plumbing alone. Which one applies is read from the layout: a
/// sub-tensor with no `_scales` companion is dense BF16.
final class FlashNextMoE {

    /// Slots in the argument buffer. Mirrors `kFlashNextRoutedBlobSlots` in
    /// `flashnext_moe.metal` and `MoE.routedBlobSlots`.
    static let routedBlobSlots = MoE.routedBlobSlots

    private let phase1PSO: MTLComputePipelineState
    private let phase1SubsetPSO: MTLComputePipelineState
    private let phase2PSO: MTLComputePipelineState
    private let sharedGatePSO: MTLComputePipelineState
    private let routerSelectPSO: MTLComputePipelineState
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer

    init(context: MetalContext) throws {
        self.phase1PSO = try context.pipeline("flashnext_moe_phase1_gate_up_bf16")
        self.phase1SubsetPSO =
            try context.pipeline("flashnext_moe_phase1_gate_up_bf16_subset")
        self.phase2PSO = try context.pipeline("flashnext_moe_phase2_down_reduce_bf16")
        self.sharedGatePSO = try context.pipeline("flashnext_moe_shared_gate_scale")
        // The shipped wide selection kernel. `RouterWideTopK10Tests` gates it
        // against `FlashNextRouterReference` at 512/top-10, so this runner takes
        // it as given rather than re-deriving a selection.
        self.routerSelectPSO = try context.pipeline("router_topk_select_k10_par")
        guard let function = context.library.makeFunction(
                name: "flashnext_moe_phase1_gate_up_bf16") else {
            throw MetalError.noDevice
        }
        let encoder = function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
                length: encoder.encodedLength, options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.routedArgEncoder = encoder
        self.reusableRoutedArgBuffer = reusable
    }

    /// Bind the routed blobs for one layer. Reuses one preallocated argument
    /// buffer, exactly as `MoE.makeReusedRoutedArgumentBuffer` does.
    func makeRoutedArgumentBuffer(
        routedBlobs: [(buffer: MTLBuffer, offset: Int)]
    ) -> MTLBuffer {
        precondition(!routedBlobs.isEmpty)
        precondition(routedBlobs.count <= Self.routedBlobSlots,
                     "the argument buffer holds \(Self.routedBlobSlots) slots")
        routedArgEncoder.setArgumentBuffer(reusableRoutedArgBuffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(blob.buffer, offset: blob.offset, index: index)
        }
        // Unused slots point at slot 0 rather than staying null: the kernels
        // index only `slot < top_k`, so this is belt-and-braces against a
        // dangling reference in the argument buffer, never arithmetic.
        if let first = routedBlobs.first {
            for index in routedBlobs.count..<Self.routedBlobSlots {
                routedArgEncoder.setBuffer(first.buffer, offset: first.offset,
                                           index: index)
            }
        }
        return reusableRoutedArgBuffer
    }

    /// `acts[slot * F + f] = silu(gate_f . x) * (up_f . x)` for every slot.
    func encodePhase1(commandBuffer: MTLCommandBuffer,
                      routedArgBuffer: MTLBuffer,
                      routedBlobs: [(buffer: MTLBuffer, offset: Int)],
                      routedOffsets: MoEExpertOffsets,
                      x: MTLBuffer, xOffset: Int = 0,
                      acts: MTLBuffer,
                      d: UInt32, f: UInt32, topK: UInt32) {
        precondition(routedBlobs.count == Int(topK))
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(phase1PSO)
        enc.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for blob in routedBlobs { enc.useResource(blob.buffer, usage: .read) }
        bindPhase1(enc, routedOffsets: routedOffsets, x: x, xOffset: xOffset,
                   acts: acts, d: d, f: f, topK: topK)
        enc.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// The hit/miss split form: compute only the named slots. The union of two
    /// subset dispatches is bit-identical to one full `encodePhase1`.
    func encodePhase1Subset(commandBuffer: MTLCommandBuffer,
                            routedArgBuffer: MTLBuffer,
                            routedBlobs: [(buffer: MTLBuffer, offset: Int)],
                            routedOffsets: MoEExpertOffsets,
                            x: MTLBuffer, xOffset: Int = 0,
                            acts: MTLBuffer,
                            activeSlots: MTLBuffer,
                            activeSlotIndices: [UInt32],
                            d: UInt32, f: UInt32, topK: UInt32) {
        guard !activeSlotIndices.isEmpty else { return }
        precondition(routedBlobs.count == Int(topK))
        var active = UInt32(activeSlotIndices.count)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(phase1SubsetPSO)
        enc.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for slot in activeSlotIndices {
            enc.useResource(routedBlobs[Int(slot)].buffer, usage: .read)
        }
        bindPhase1(enc, routedOffsets: routedOffsets, x: x, xOffset: xOffset,
                   acts: acts, d: d, f: f, topK: topK)
        enc.setBuffer(activeSlots, offset: 0, index: 7)
        enc.setBytes(&active, length: MemoryLayout<UInt32>.stride, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(width: (Int(active * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `y[d] = residual[d] + sum_rank routing_w[rank] * (down_d . acts[rank])`,
    /// summed in rank order with the residual first.
    func encodePhase2(commandBuffer: MTLCommandBuffer,
                      routedArgBuffer: MTLBuffer,
                      routedBlobs: [(buffer: MTLBuffer, offset: Int)],
                      routedOffsets: MoEExpertOffsets,
                      acts: MTLBuffer,
                      routingWeights: MTLBuffer, routingWeightsOffset: Int = 0,
                      residual: MTLBuffer, residualOffset: Int = 0,
                      y: MTLBuffer, yOffset: Int = 0,
                      d: UInt32, f: UInt32, topK: UInt32) {
        precondition(routedBlobs.count == Int(topK))
        precondition(topK >= 1 && Int(topK) <= Self.routedBlobSlots)
        var offsets = routedOffsets
        var dimension = d
        var intermediate = f
        var k = topK
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(phase2PSO)
        enc.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for blob in routedBlobs { enc.useResource(blob.buffer, usage: .read) }
        enc.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        enc.setBuffer(acts, offset: 0, index: 2)
        enc.setBuffer(routingWeights, offset: routingWeightsOffset, index: 3)
        enc.setBuffer(residual, offset: residualOffset, index: 4)
        enc.setBuffer(y, offset: yOffset, index: 5)
        enc.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        enc.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Int(topK) * 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Top-`topK` of `softmax(logits)`, renormalized — the shipped wide
    /// selection kernel, driven from FP32 logits this family computes with its
    /// own dtype-dispatching mat-vec rather than the INT4-only router GEMV.
    ///
    /// The reference softmaxes over all 512 experts, takes the top-k of the
    /// probs and renormalizes; the kernel softmaxes over the k selected logits.
    /// Those are algebraically identical because softmax is strictly monotone in
    /// the logit, which is why no 512-wide prob vector is ever materialized.
    func encodeRouterSelect(commandBuffer: MTLCommandBuffer,
                            logits: MTLBuffer,
                            perExpertScale: MTLBuffer,
                            outIndices: MTLBuffer,
                            outWeights: MTLBuffer,
                            numExperts: UInt32) {
        var expertCount = numExperts
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(routerSelectPSO)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(perExpertScale, offset: 0, index: 1)
        enc.setBuffer(outIndices, offset: 0, index: 2)
        enc.setBuffer(outWeights, offset: 0, index: 3)
        enc.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1,
                                                                depth: 1))
        enc.endEncoding()
    }

    /// `out *= sigmoid(scalar[0])` — the shared expert's `[1, hidden]` gate,
    /// applied to its whole output. `scalar` is the FP32 pre-sigmoid value.
    func encodeSharedGateScale(commandBuffer: MTLCommandBuffer,
                               out: MTLBuffer, outOffset: Int = 0,
                               scalar: MTLBuffer, scalarOffset: Int = 0,
                               count: Int) {
        guard count > 0, let enc = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        var countVar = UInt32(count)
        enc.setComputePipelineState(sharedGatePSO)
        enc.setBuffer(out, offset: outOffset, index: 0)
        enc.setBuffer(scalar, offset: scalarOffset, index: 1)
        enc.setBytes(&countVar, length: MemoryLayout<UInt32>.stride, index: 2)
        let width = min(Int(sharedGatePSO.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(width, count),
                                                           height: 1, depth: 1))
        enc.endEncoding()
    }

    private func bindPhase1(_ enc: MTLComputeCommandEncoder,
                            routedOffsets: MoEExpertOffsets,
                            x: MTLBuffer, xOffset: Int,
                            acts: MTLBuffer,
                            d: UInt32, f: UInt32, topK: UInt32) {
        var offsets = routedOffsets
        var dimension = d
        var intermediate = f
        var k = topK
        enc.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        enc.setBuffer(x, offset: xOffset, index: 2)
        enc.setBuffer(acts, offset: 0, index: 3)
        enc.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 6)
    }
}
