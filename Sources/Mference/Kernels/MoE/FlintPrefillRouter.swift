import Metal

/// Batched BF16 router for Flint prompt prefill. The decode router remains a
/// separate object so its private logits buffer and validated path cannot be
/// perturbed by the multi-token schedule.
final class FlintPrefillRouter {
    static let expertCount = 256
    static let topK = 6
    static let hiddenSize = 2_048

    private let gemv: MTLComputePipelineState
    private let select: MTLComputePipelineState
    private let logits: MTLBuffer

    init(context: MetalContext, maxTokens: Int) throws {
        precondition(maxTokens > 0)
        self.gemv = try context.pipeline(
            "flint_router_gemv_bf16_batched",
            constants: [],
            maxTotalThreadsPerThreadgroup: 128)
        self.select = try context.pipeline(
            "flint_router_sigmoid_top6_norm_batched")
        guard let logits = context.device.makeBuffer(
            length: maxTokens * Self.expertCount * MemoryLayout<Float>.stride,
            options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.logits = logits
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                hidden: MTLBuffer, hiddenOffset: Int = 0,
                hiddenStrideElements: Int,
                indices: MTLBuffer, indicesOffset: Int = 0,
                routingWeights: MTLBuffer, routingWeightsOffset: Int = 0,
                tokens: Int) {
        precondition(tokens > 0)
        precondition(tokens <= logits.length
            / MemoryLayout<Float>.stride / Self.expertCount,
            "Flint prefill router token count exceeds its logits buffer")
        var experts = UInt32(Self.expertCount)
        var dimension = UInt32(Self.hiddenSize)
        var stride = UInt32(hiddenStrideElements)
        var topK = UInt32(Self.topK)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(gemv)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(hidden, offset: hiddenOffset, index: 1)
            encoder.setBuffer(logits, offset: 0, index: 2)
            encoder.setBytes(&experts, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: Self.expertCount / 4, height: tokens, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(select)
            encoder.setBuffer(logits, offset: 0, index: 0)
            encoder.setBuffer(indices, offset: indicesOffset, index: 1)
            encoder.setBuffer(routingWeights, offset: routingWeightsOffset, index: 2)
            encoder.setBytes(&experts, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&topK, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: tokens, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }
}
