import Foundation
import Metal

/// Wrappers for the paged-KV maintenance kernels: page seal summaries
/// (`kv_page_minmax`) and Quest page criticality scores
/// (`attention_page_scores`). Both ride the token command buffer.
final class KVPageKernels {
    private let ctx: MetalContext
    private let psoMinMax: MTLComputePipelineState
    private let psoScores: MTLComputePipelineState

    private static let threadsPerGroup = 256

    init(context: MetalContext) throws {
        self.ctx = context
        self.psoMinMax = try context.pipeline("kv_page_minmax")
        self.psoScores = try context.pipeline("attention_page_scores")
    }

    /// Reduce a page's K rows to element-wise min/max vectors, written to
    /// the page's slot in the metadata buffer.
    func encodePageMinMax(commandBuffer: MTLCommandBuffer,
                          kPool: MTLBuffer,
                          slot: UInt32,
                          validTokens: UInt32,
                          metadata: MTLBuffer,
                          metadataOffset: Int,
                          numKVHeads: UInt32,
                          headDim: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoMinMax)
        enc.setBuffer(kPool, offset: 0, index: 0)
        enc.setBuffer(metadata, offset: metadataOffset, index: 1)
        var s = slot, vt = validTokens, nkv = numKVHeads, hd = headDim
        enc.setBytes(&s,   length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&vt,  length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&hd,  length: MemoryLayout<UInt32>.size, index: 5)
        let width = min(Self.threadsPerGroup, psoMinMax.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Score every sealed page of one layer against the current query.
    /// `metadataOffset` addresses the layer's metadata base; `scores` receives
    /// one float per page (read back by the CPU after the token completes —
    /// the lag-one selection input for the next token).
    func encodePageScores(commandBuffer: MTLCommandBuffer,
                          q: MTLBuffer, qOffset: Int = 0,
                          metadata: MTLBuffer,
                          metadataOffset: Int,
                          scores: MTLBuffer,
                          scoresOffset: Int,
                          numPages: UInt32,
                          headDim: UInt32,
                          numQHeads: UInt32,
                          numKVHeads: UInt32) {
        guard numPages > 0, let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoScores)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(metadata, offset: metadataOffset, index: 1)
        enc.setBuffer(scores, offset: scoresOffset, index: 2)
        var np = numPages, hd = headDim, nq = numQHeads, nkv = numKVHeads
        enc.setBytes(&np,  length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hd,  length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&nq,  length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 6)
        let width = min(Self.threadsPerGroup, psoScores.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(MTLSize(width: Int(numPages), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }
}
