import Metal

/// Batched causal attention with the pinned QAT FP16 reduction boundaries.
/// Full-attention query tiles share at most 8 MiB of scratch, or one row's
/// context-sized scratch when that alone exceeds 8 MiB. No model replay.
final class GemmaQATPrefillAttention {
    private struct Parameters {
        var attention: PrefillAttentionParams
        var queryOffset: UInt32 = 0
        var scratchStride: UInt32
    }

    private let swa: MTLComputePipelineState
    private let score: MTLComputePipelineState
    private let probability: MTLComputePipelineState
    private let value: MTLComputePipelineState
    private let scores: MTLBuffer
    private let probabilities: MTLBuffer
    private let scratchElements: Int
    private let maxContext: Int

    init(context: MetalContext, maxContext: Int) throws {
        precondition(maxContext > 0 && maxContext <= Int(UInt32.max) / 16)
        self.maxContext = maxContext
        func pipeline(_ name: String, threads: Int) throws -> MTLComputePipelineState {
            let pso = try context.pipeline(name, constants: [], maxTotalThreadsPerThreadgroup: threads,
                safeMathModule: "gemma_qat_prefill_attention")
            guard pso.maxTotalThreadsPerThreadgroup >= threads else {
                throw MetalError.libraryCompileFailed("QAT prefill attention requires \(threads) threads for \(name)")
            }
            return pso
        }
        swa = try pipeline("gemma_qat_prefill_attention_swa", threads: 1024)
        score = try pipeline("gemma_qat_prefill_attention_scores", threads: 256)
        probability = try pipeline("gemma_qat_prefill_attention_probabilities", threads: 1024)
        value = try pipeline("gemma_qat_prefill_attention_values", threads: 128)
        let rowBytes = maxContext * 16 * MemoryLayout<Float16>.size * 2
        let rows = max(1, min(32, (8 * 1024 * 1024) / rowBytes))
        self.scratchElements = rows * maxContext * 16
        let bytes = scratchElements * MemoryLayout<Float16>.size
        guard let scores = context.device.makeBuffer(length: bytes, options: .storageModePrivate),
              let probabilities = context.device.makeBuffer(length: bytes, options: .storageModePrivate) else {
            throw MetalError.libraryCompileFailed("QAT prefill attention scratch allocation failed")
        }
        self.scores = scores
        self.probabilities = probabilities
    }

    func encode(commandBuffer: MTLCommandBuffer,
                q: MTLBuffer, qOffset: Int, k: MTLBuffer, kOffset: Int,
                v: MTLBuffer, vOffset: Int, out: MTLBuffer, outOffset: Int,
                params: PrefillAttentionParams, ringCapacity: UInt32) {
        precondition(params.numQHeads == 16 && params.scale == 1)
        precondition(Int(params.kvValidCount) <= maxContext)
        var p = Parameters(attention: params, scratchStride: params.kvValidCount)
        if params.headDim == 256 {
            precondition(params.numKVHeads == 8 && params.slidingWindow > 0)
            // All queries must retain their complete window after the whole
            // chunk has been copied into the ring.
            precondition(ringCapacity == 0 || ringCapacity >=
                min(params.slidingWindow, params.startPosition + 1) + params.queryCount - 1)
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(swa)
            for (index, pair) in [(q, qOffset), (k, kOffset), (v, vOffset), (out, outOffset)].enumerated() {
                enc.setBuffer(pair.0, offset: pair.1, index: index)
            }
            var capacity = ringCapacity
            enc.setBytes(&p, length: MemoryLayout<Parameters>.stride, index: 4)
            enc.setBytes(&capacity, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: 16, height: Int(params.queryCount), depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            enc.endEncoding()
            return
        }
        precondition(params.headDim == 512 && params.numKVHeads == 2 && ringCapacity == 0)
        precondition(params.slidingWindow >= params.kvValidCount)
        let tileCapacity = min(32, scratchElements / (16 * Int(params.kvValidCount)))
        precondition(tileCapacity > 0)
        var firstQuery = 0
        while firstQuery < Int(params.queryCount) {
            let rows = min(tileCapacity, Int(params.queryCount) - firstQuery)
            let end = Int(params.startPosition) + firstQuery + rows
            p.queryOffset = UInt32(firstQuery)
            guard let first = commandBuffer.makeComputeCommandEncoder() else { return }
            first.setComputePipelineState(score)
            first.setBuffer(q, offset: qOffset, index: 0)
            first.setBuffer(k, offset: kOffset, index: 1)
            first.setBuffer(scores, offset: 0, index: 2)
            first.setBytes(&p, length: MemoryLayout<Parameters>.stride, index: 3)
            first.dispatchThreadgroups(MTLSize(width: 16, height: end, depth: rows),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            first.endEncoding()

            guard let second = commandBuffer.makeComputeCommandEncoder() else { return }
            second.setComputePipelineState(probability)
            second.setBuffer(scores, offset: 0, index: 0)
            second.setBuffer(probabilities, offset: 0, index: 1)
            second.setBytes(&p, length: MemoryLayout<Parameters>.stride, index: 2)
            // Extra SIMD groups for shorter rows contribute only zero sums;
            // each row retains the source's four-contiguous-values partition.
            let threads = min(1024, ((end + 127) / 128) * 32)
            second.dispatchThreadgroups(MTLSize(width: 16, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            second.endEncoding()

            guard let third = commandBuffer.makeComputeCommandEncoder() else { return }
            third.setComputePipelineState(value)
            third.setBuffer(probabilities, offset: 0, index: 0)
            third.setBuffer(v, offset: vOffset, index: 1)
            third.setBuffer(out, offset: outOffset, index: 2)
            third.setBytes(&p, length: MemoryLayout<Parameters>.stride, index: 3)
            third.dispatchThreadgroups(MTLSize(width: 8, height: 16, depth: rows),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            third.endEncoding()
            firstQuery += rows
        }
    }
}
