import Metal

/// Exact FP16 GQA decode attention for Flint's 32 Q / 4 KV / 128-dim shape,
/// including the per-head learned sink logit and optional sliding ring.
final class FlintAttention {
    static let numQHeads = 32
    static let numKVHeads = 4
    static let headDim = 128
    static let slidingWindow = 513

    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = try MetalContext.moduleLibrary(
            device: context.device, module: "flint", safeMath: true)
        guard let function = library.makeFunction(
            name: "flint_attention_sink_fp16_d128") else {
            throw MetalError.missingFunction("flint_attention_sink_fp16_d128")
        }
        self.pipeline = try context.device.makeComputePipelineState(function: function)
        precondition(pipeline.maxTotalThreadsPerThreadgroup >= 1024,
                     "Flint attention requires 1024-thread threadgroups")
    }

    /// `start` is inclusive and `end` is exclusive in logical positions.
    /// `ringCapacity == 0` means the cache is linear.
    func encode(commandBuffer: MTLCommandBuffer,
                q: MTLBuffer, qOffset: Int = 0,
                k: MTLBuffer, kOffset: Int = 0,
                v: MTLBuffer, vOffset: Int = 0,
                sinks: MTLBuffer, sinksOffset: Int = 0,
                out: MTLBuffer, outOffset: Int = 0,
                start: UInt32,
                end: UInt32,
                ringCapacity: UInt32,
                scale: Float) {
        precondition(start < end)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        enc.setBuffer(sinks, offset: sinksOffset, index: 4)
        var first = start
        var last = end
        var cap = ringCapacity
        var attentionScale = scale
        enc.setBytes(&first, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&last, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&cap, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&attentionScale, length: MemoryLayout<Float>.size, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(width: Self.numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        enc.endEncoding()
    }
}
