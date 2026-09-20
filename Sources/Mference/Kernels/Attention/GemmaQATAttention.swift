import Metal

/// The pinned QAT checkpoint's FP16 attention arithmetic. Its full-attention
/// scratch is bounded by the runner's actual context capacity, shared across
/// layers, and hazard-tracked within the serial command stream.
final class GemmaQATAttention {
    private let swa: MTLComputePipelineState
    private let score: MTLComputePipelineState
    private let probability: MTLComputePipelineState
    private let value: MTLComputePipelineState
    private let scores: MTLBuffer
    private let probabilities: MTLBuffer
    private let maxContext: Int

    init(context: MetalContext, maxContext: Int) throws {
        precondition(maxContext > 0 && maxContext <= Int(UInt32.max) / 16)
        self.maxContext = maxContext
        func pipeline(_ name: String, threads: Int) throws -> MTLComputePipelineState {
            let pso = try context.pipeline(name, constants: [], maxTotalThreadsPerThreadgroup: threads,
                safeMathModule: "gemma_qat_attention")
            guard pso.maxTotalThreadsPerThreadgroup >= threads else {
                throw MetalError.libraryCompileFailed("QAT attention requires \(threads) threads for \(name)")
            }
            return pso
        }
        swa = try pipeline("gemma_qat_attention_swa", threads: 1024)
        score = try pipeline("gemma_qat_attention_scores", threads: 256)
        probability = try pipeline("gemma_qat_attention_probabilities", threads: 1024)
        value = try pipeline("gemma_qat_attention_values", threads: 128)
        let bytes = maxContext * 16 * MemoryLayout<Float16>.size
        guard let scores = context.device.makeBuffer(length: bytes, options: .storageModePrivate),
              let probabilities = context.device.makeBuffer(length: bytes, options: .storageModePrivate) else {
            throw MetalError.missingFunction("QAT attention FP16 scratch")
        }
        self.scores = scores
        self.probabilities = probabilities
    }

    func encode(commandBuffer: MTLCommandBuffer,
                q: MTLBuffer, qOffset: Int, k: MTLBuffer, kOffset: Int,
                v: MTLBuffer, vOffset: Int, out: MTLBuffer, outOffset: Int,
                headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
                seqLen: UInt32, kvStart: UInt32, scale: Float, ringCapacity: UInt32) {
        precondition(seqLen > kvStart && Int(seqLen) <= maxContext)
        precondition(numQHeads == 16 && scale == 1)
        var length = seqLen
        if headDim == 256 {
            precondition(numKVHeads == 8)
            precondition(ringCapacity == 0 || seqLen - kvStart <= ringCapacity)
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(swa)
            for (index, pair) in [(q, qOffset), (k, kOffset), (v, vOffset), (out, outOffset)].enumerated() {
                enc.setBuffer(pair.0, offset: pair.1, index: index)
            }
            var start = kvStart, capacity = ringCapacity
            enc.setBytes(&length, length: 4, index: 4)
            enc.setBytes(&start, length: 4, index: 5)
            enc.setBytes(&capacity, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            enc.endEncoding()
            return
        }
        precondition(headDim == 512 && numKVHeads == 2 && kvStart == 0 && ringCapacity == 0)
        guard let first = commandBuffer.makeComputeCommandEncoder() else { return }
        first.setComputePipelineState(score)
        first.setBuffer(q, offset: qOffset, index: 0)
        first.setBuffer(k, offset: kOffset, index: 1)
        first.setBuffer(scores, offset: 0, index: 2)
        first.setBytes(&length, length: 4, index: 3)
        first.dispatchThreadgroups(MTLSize(width: 16, height: Int(seqLen), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        first.endEncoding()

        guard let second = commandBuffer.makeComputeCommandEncoder() else { return }
        second.setComputePipelineState(probability)
        second.setBuffer(scores, offset: 0, index: 0)
        second.setBuffer(probabilities, offset: 0, index: 1)
        second.setBytes(&length, length: 4, index: 2)
        let softmaxThreads = min(1024, ((Int(seqLen) + 127) / 128) * 32)
        second.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: softmaxThreads, height: 1, depth: 1))
        second.endEncoding()

        guard let third = commandBuffer.makeComputeCommandEncoder() else { return }
        third.setComputePipelineState(value)
        third.setBuffer(probabilities, offset: 0, index: 0)
        third.setBuffer(v, offset: vOffset, index: 1)
        third.setBuffer(out, offset: outOffset, index: 2)
        third.setBytes(&length, length: 4, index: 3)
        third.dispatchThreadgroups(MTLSize(width: 8, height: 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        third.endEncoding()
    }
}
