import Metal

/// 8-bit affine embedding lookup for MLX INT8 checkpoints.
final class EmbedLookupInt8 {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("embed_lookup_int8")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                table: MTLBuffer, tableOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                out: MTLBuffer, outOffset: Int = 0,
                tokenId: UInt32,
                d: UInt32) {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table, offset: tableOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var token = tokenId
        var width = d
        enc.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&width, length: MemoryLayout<UInt32>.size, index: 5)
        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: Int(d), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
