import Foundation
import Metal

/// Wrappers for the paged-KV maintenance kernels: page seal summaries
/// (`kv_page_minmax`) and Quest page criticality scores
/// (`attention_page_scores`). Both ride the token command buffer.
final class KVPageKernels {
    private let ctx: MetalContext
    private let psoMinMax: MTLComputePipelineState
    private let psoScores: MTLComputePipelineState
    private let psoFlashInit: MTLComputePipelineState
    private let psoFlashUpdate: MTLComputePipelineState
    private let psoFlashFinalize: MTLComputePipelineState

    private static let threadsPerGroup = 256
    private static let flashSimdgroupsPerTG = 8

    init(context: MetalContext) throws {
        self.ctx = context
        self.psoMinMax = try context.pipeline("kv_page_minmax")
        self.psoScores = try context.pipeline("attention_page_scores")
        self.psoFlashInit = try context.pipeline("attention_prefill_flash_init")
        self.psoFlashUpdate = try context.pipeline("attention_prefill_flash_update")
        self.psoFlashFinalize = try context.pipeline("attention_prefill_flash_finalize")
    }

    // MARK: - Blocked prefill attention

    /// Reset the running online-softmax state for a chunk's queries.
    func encodeFlashInit(commandBuffer: MTLCommandBuffer,
                         mState: MTLBuffer, dState: MTLBuffer, oState: MTLBuffer,
                         rows: UInt32, headDim: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoFlashInit)
        enc.setBuffer(mState, offset: 0, index: 0)
        enc.setBuffer(dState, offset: 0, index: 1)
        enc.setBuffer(oState, offset: 0, index: 2)
        var r = rows, hd = headDim
        enc.setBytes(&r,  length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 4)
        let width = min(Self.threadsPerGroup, psoFlashInit.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: Int(rows), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Fold one KV window into the running state. `pageTable` maps the
    /// window's 64-token pages to slots in `kPool`/`vPool` (staging passes a
    /// stride-2 identity for the interleaved [K|V] spill layout). `causal`
    /// applies `p <= q_pos` for the chunk's own pages.
    func encodeFlashUpdate(commandBuffer: MTLCommandBuffer,
                           q: MTLBuffer,
                           kPool: MTLBuffer, kPoolOffset: Int = 0,
                           vPool: MTLBuffer, vPoolOffset: Int = 0,
                           pageTable: MTLBuffer, pageTableOffset: Int = 0,
                           mState: MTLBuffer, dState: MTLBuffer, oState: MTLBuffer,
                           queryCount: UInt32,
                           qStartPosition: UInt32,
                           headDim: UInt32,
                           numQHeads: UInt32,
                           numKVHeads: UInt32,
                           windowStartPosition: UInt32,
                           windowTokens: UInt32,
                           qStrideElements: UInt32,
                           scale: Float,
                           causal: Bool) {
        guard windowTokens > 0, let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoFlashUpdate)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(kPool, offset: kPoolOffset, index: 1)
        enc.setBuffer(vPool, offset: vPoolOffset, index: 2)
        enc.setBuffer(pageTable, offset: pageTableOffset, index: 3)
        enc.setBuffer(mState, offset: 0, index: 4)
        enc.setBuffer(dState, offset: 0, index: 5)
        enc.setBuffer(oState, offset: 0, index: 6)
        var qc = queryCount, qs = qStartPosition, hd = headDim
        var nq = numQHeads, nkv = numKVHeads
        var ws = windowStartPosition, wt = windowTokens, qst = qStrideElements
        var sc = scale
        var cz: UInt32 = causal ? 1 : 0
        enc.setBytes(&qc,  length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&qs,  length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&hd,  length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&nq,  length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&ws,  length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBytes(&wt,  length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&qst, length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&sc,  length: MemoryLayout<Float>.size,  index: 15)
        enc.setBytes(&cz,  length: MemoryLayout<UInt32>.size, index: 16)
        let rows = Int(queryCount * numQHeads)
        let groups = (rows + Self.flashSimdgroupsPerTG - 1) / Self.flashSimdgroupsPerTG
        enc.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.flashSimdgroupsPerTG * 32,
                                           height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Normalize the running state into the chunk's FP16 attention output.
    func encodeFlashFinalize(commandBuffer: MTLCommandBuffer,
                             mState: MTLBuffer, dState: MTLBuffer, oState: MTLBuffer,
                             out: MTLBuffer,
                             queryCount: UInt32,
                             headDim: UInt32,
                             numQHeads: UInt32,
                             oStrideElements: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoFlashFinalize)
        enc.setBuffer(mState, offset: 0, index: 0)
        enc.setBuffer(dState, offset: 0, index: 1)
        enc.setBuffer(oState, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        var qc = queryCount, hd = headDim, nq = numQHeads, os = oStrideElements
        enc.setBytes(&qc, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nq, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&os, length: MemoryLayout<UInt32>.size, index: 7)
        let total = Int(queryCount * numQHeads * headDim)
        let width = min(Self.threadsPerGroup, psoFlashFinalize.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
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
