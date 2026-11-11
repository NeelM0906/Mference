import Metal

/// BF16 router GEMV plus Flint's sigmoid top-6 selection and renormalization.
final class FlintRouter {
    static let expertCount = 256
    static let topK = 6
    static let hiddenSize = 2_048

    private let gemv: MTLComputePipelineState
    private let select: MTLComputePipelineState
    private let logits: MTLBuffer

    init(context: MetalContext) throws {
        self.gemv = try context.pipeline(
            "flint_router_gemv_bf16",
            constants: [],
            maxTotalThreadsPerThreadgroup: 128)
        self.select = try context.pipeline("flint_router_sigmoid_top6_norm")
        guard let logits = context.device.makeBuffer(
            length: Self.expertCount * MemoryLayout<Float>.stride,
            options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.logits = logits
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                hidden: MTLBuffer, hiddenOffset: Int = 0,
                indices: MTLBuffer, indicesOffset: Int = 0,
                routingWeights: MTLBuffer, routingWeightsOffset: Int = 0) {
        var experts = UInt32(Self.expertCount)
        var dimension = UInt32(Self.hiddenSize)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(gemv)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(hidden, offset: hiddenOffset, index: 1)
            encoder.setBuffer(logits, offset: 0, index: 2)
            encoder.setBytes(&experts, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: Self.expertCount / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(select)
            encoder.setBuffer(logits, offset: 0, index: 0)
            encoder.setBuffer(indices, offset: indicesOffset, index: 1)
            encoder.setBuffer(routingWeights, offset: routingWeightsOffset, index: 2)
            encoder.setBytes(&experts, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }
}
