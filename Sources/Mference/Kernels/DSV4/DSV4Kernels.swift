import Foundation
import Metal

/// Swift wrappers for the DeepSeek-V4 kernel family (`Metal/DSV4/dsv4.metal`):
/// interleaved trailing partial RoPE, shared-KV MQA decode attention with
/// per-head sinks over [window ring ‖ compressed entries], the CSA/HCA window
/// compressor, the lightning-indexer scorer, mHC stream mixing, and the
/// clamped-SwiGLU elementwise.
final class DSV4Kernels {
    /// Compressed-entry ceiling baked into the attention kernel's threadgroup
    /// logits array (`kDSV4MaxEntries` = 128 window + 2048 compressed).
    static let maxCompressedEntries = 2048
    /// Sentinel `selected_count` meaning "attend to every compressed entry".
    static let selectAll: UInt32 = 0xFFFF_FFFF

    private let ropePSO: MTLComputePipelineState
    private let attentionPSO: MTLComputePipelineState
    private let compressEmitPSO: MTLComputePipelineState
    private let indexerScorePSO: MTLComputePipelineState
    private let hcWeightsPSO: MTLComputePipelineState
    private let hcCollapsePSO: MTLComputePipelineState
    private let hcPlaceMixPSO: MTLComputePipelineState
    private let hyperHeadPSO: MTLComputePipelineState
    private let swigluClampPSO: MTLComputePipelineState
    private let broadcastPSO: MTLComputePipelineState

    init(context: MetalContext, config: ArchConfig) throws {
        let constants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 100, value: .uint32(UInt32(config.fullHeadDim))),
            MetalFunctionConstant(index: 101, value: .uint32(UInt32(config.numHeads))),
            MetalFunctionConstant(index: 102,
                                  value: .uint32(UInt32(config.compressedAttention.ropeHeadDim))),
            MetalFunctionConstant(index: 103,
                                  value: .uint32(UInt32(config.hyperConnections.mult))),
            MetalFunctionConstant(index: 104, value: .uint32(UInt32(config.hiddenSize))),
            MetalFunctionConstant(index: 105, value: .bool(true)),
        ]
        self.ropePSO = try context.pipeline(
            "dsv4_rope_interleaved_trailing", constants: constants)
        self.attentionPSO = try context.pipeline(
            "dsv4_attention_decode", constants: constants,
            maxTotalThreadsPerThreadgroup: 256)
        self.compressEmitPSO = try context.pipeline("dsv4_compress_emit")
        self.indexerScorePSO = try context.pipeline("dsv4_indexer_score")
        self.hcWeightsPSO = try context.pipeline(
            "dsv4_hc_weights", constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        self.hcCollapsePSO = try context.pipeline("dsv4_hc_collapse")
        self.hcPlaceMixPSO = try context.pipeline("dsv4_hc_place_mix")
        self.hyperHeadPSO = try context.pipeline(
            "dsv4_hyper_head", constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        self.swigluClampPSO = try context.pipeline("dsv4_swiglu_clamp_mul")
        self.broadcastPSO = try context.pipeline("dsv4_broadcast_streams")
    }

    /// Interleaved RoPE on the trailing `ropeDim` channels of `numHeads`
    /// contiguous heads at `position`. `direction` -1 applies the conjugate
    /// rotation (attention-output un-rotation).
    func encodeRope(commandBuffer: MTLCommandBuffer,
                    x: MTLBuffer, xOffset: Int = 0,
                    numHeads: Int, headDim: Int, ropeDim: Int,
                    position: Int, theta: Float, direction: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(ropePSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        var hd = UInt32(headDim)
        var rd = UInt32(ropeDim)
        var pos = UInt32(position)
        var th = theta
        var dir = direction
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&rd, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&pos, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&th, length: MemoryLayout<Float>.size, index: 4)
        enc.setBytes(&dir, length: MemoryLayout<Float>.size, index: 5)
        let threads = max(32, ropeDim / 2)
        enc.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Decode attention for one token. `selectedCount == DSV4Kernels.selectAll`
    /// attends to every compressed entry (windowed layers pass
    /// `compressedCount == 0`).
    func encodeAttention(commandBuffer: MTLCommandBuffer,
                         q: MTLBuffer,
                         windowKV: MTLBuffer,
                         compressedKV: MTLBuffer,
                         selected: MTLBuffer,
                         sinks: MTLBuffer, sinksOffset: Int,
                         out: MTLBuffer,
                         headDim: Int, numHeads: Int,
                         windowCount: Int, windowStartPos: Int, ringCapacity: Int,
                         compressedCount: Int, selectedCount: UInt32,
                         scale: Float) {
        precondition(compressedCount <= Self.maxCompressedEntries)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(attentionPSO)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(windowKV, offset: 0, index: 1)
        enc.setBuffer(compressedKV, offset: 0, index: 2)
        enc.setBuffer(selected, offset: 0, index: 3)
        enc.setBuffer(sinks, offset: sinksOffset, index: 4)
        enc.setBuffer(out, offset: 0, index: 5)
        var hd = UInt32(headDim)
        var wc = UInt32(windowCount)
        var ws = UInt32(windowStartPos)
        var rc = UInt32(ringCapacity)
        var cc = UInt32(compressedCount)
        var sc = selectedCount
        var sl = scale
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&wc, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&ws, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&rc, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&cc, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&sc, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&sl, length: MemoryLayout<Float>.size, index: 12)
        enc.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Emit one compressed entry from a full pending window into
    /// `outEntry` (an offset slot inside the compressed cache). RoPE at the
    /// window position is a separate `encodeRope` on the emitted slot.
    func encodeCompressEmit(commandBuffer: MTLCommandBuffer,
                            pendingKV: MTLBuffer, pendingGate: MTLBuffer,
                            priorCaKV: MTLBuffer, priorCaGate: MTLBuffer,
                            positionBias: MTLBuffer, positionBiasOffset: Int,
                            normWeight: MTLBuffer, normWeightOffset: Int,
                            outEntry: MTLBuffer, outEntryOffset: Int,
                            nextPriorCaKV: MTLBuffer, nextPriorCaGate: MTLBuffer,
                            rate: Int, dim: Int, dual: Bool, hasPrior: Bool,
                            eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(compressEmitPSO)
        enc.setBuffer(pendingKV, offset: 0, index: 0)
        enc.setBuffer(pendingGate, offset: 0, index: 1)
        enc.setBuffer(priorCaKV, offset: 0, index: 2)
        enc.setBuffer(priorCaGate, offset: 0, index: 3)
        enc.setBuffer(positionBias, offset: positionBiasOffset, index: 4)
        enc.setBuffer(normWeight, offset: normWeightOffset, index: 5)
        enc.setBuffer(outEntry, offset: outEntryOffset, index: 6)
        enc.setBuffer(nextPriorCaKV, offset: 0, index: 7)
        enc.setBuffer(nextPriorCaGate, offset: 0, index: 8)
        var r = UInt32(rate)
        var d = UInt32(dim)
        var du: UInt32 = dual ? 1 : 0
        var hp: UInt32 = hasPrior ? 1 : 0
        var e = eps
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&du, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&hp, length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBytes(&e, length: MemoryLayout<Float>.size, index: 13)
        let threads = min(max(dim, 32), 512)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Lightning-indexer scores over `entryCount` compressed index keys.
    /// `weights` is the raw FP16 weights_proj output; `weightScale` carries
    /// the heads^-0.5 factor.
    func encodeIndexerScore(commandBuffer: MTLCommandBuffer,
                            q: MTLBuffer,
                            keys: MTLBuffer,
                            weights: MTLBuffer,
                            scores: MTLBuffer,
                            numHeads: Int, indexDim: Int, entryCount: Int,
                            headScale: Float, weightScale: Float) {
        guard entryCount > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(indexerScorePSO)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(keys, offset: 0, index: 1)
        enc.setBuffer(weights, offset: 0, index: 2)
        enc.setBuffer(scores, offset: 0, index: 3)
        var nh = UInt32(numHeads)
        var id = UInt32(indexDim)
        var hs = headScale
        var ws = weightScale
        enc.setBytes(&nh, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&id, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hs, length: MemoryLayout<Float>.size, index: 6)
        enc.setBytes(&ws, length: MemoryLayout<Float>.size, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: entryCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// mHC mixing weights for one sublayer site.
    func encodeHCWeights(commandBuffer: MTLCommandBuffer,
                         streams: MTLBuffer,
                         fn: MTLBuffer, fnOffset: Int,
                         base: MTLBuffer, baseOffset: Int,
                         scale: MTLBuffer, scaleOffset: Int,
                         outPre: MTLBuffer, outPost: MTLBuffer, outComb: MTLBuffer,
                         hcMult: Int, hidden: Int, sinkhornIters: Int,
                         hcEps: Float, rmsEps: Float) {
        precondition(hcMult <= 4, "dsv4_hc_weights sizes its mix scratch for hc_mult <= 4")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcWeightsPSO)
        enc.setBuffer(streams, offset: 0, index: 0)
        enc.setBuffer(fn, offset: fnOffset, index: 1)
        enc.setBuffer(base, offset: baseOffset, index: 2)
        enc.setBuffer(scale, offset: scaleOffset, index: 3)
        enc.setBuffer(outPre, offset: 0, index: 4)
        enc.setBuffer(outPost, offset: 0, index: 5)
        enc.setBuffer(outComb, offset: 0, index: 6)
        var mult = UInt32(hcMult)
        var hid = UInt32(hidden)
        var iters = UInt32(sinkhornIters)
        var he = hcEps
        var re = rmsEps
        enc.setBytes(&mult, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&hid, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&iters, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&he, length: MemoryLayout<Float>.size, index: 10)
        enc.setBytes(&re, length: MemoryLayout<Float>.size, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeHCCollapse(commandBuffer: MTLCommandBuffer,
                          streams: MTLBuffer, pre: MTLBuffer, x: MTLBuffer,
                          hcMult: Int, hidden: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcCollapsePSO)
        enc.setBuffer(streams, offset: 0, index: 0)
        enc.setBuffer(pre, offset: 0, index: 1)
        enc.setBuffer(x, offset: 0, index: 2)
        var mult = UInt32(hcMult)
        var hid = UInt32(hidden)
        enc.setBytes(&mult, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hid, length: MemoryLayout<UInt32>.size, index: 4)
        enc.dispatchThreads(MTLSize(width: hidden, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `outStreams[k] = post[k] * sub + combᵀ @ streams` — reads `streams`,
    /// writes `outStreams` (caller ping-pongs).
    func encodeHCPlaceMix(commandBuffer: MTLCommandBuffer,
                          streams: MTLBuffer, sub: MTLBuffer,
                          post: MTLBuffer, comb: MTLBuffer,
                          outStreams: MTLBuffer,
                          hcMult: Int, hidden: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcPlaceMixPSO)
        enc.setBuffer(streams, offset: 0, index: 0)
        enc.setBuffer(sub, offset: 0, index: 1)
        enc.setBuffer(post, offset: 0, index: 2)
        enc.setBuffer(comb, offset: 0, index: 3)
        enc.setBuffer(outStreams, offset: 0, index: 4)
        var mult = UInt32(hcMult)
        var hid = UInt32(hidden)
        enc.setBytes(&mult, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hid, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreads(MTLSize(width: hidden, height: hcMult, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeHyperHead(commandBuffer: MTLCommandBuffer,
                         streams: MTLBuffer,
                         fn: MTLBuffer, fnOffset: Int,
                         base: MTLBuffer, baseOffset: Int,
                         scale: MTLBuffer, scaleOffset: Int,
                         x: MTLBuffer,
                         hcMult: Int, hidden: Int, hcEps: Float, rmsEps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hyperHeadPSO)
        enc.setBuffer(streams, offset: 0, index: 0)
        enc.setBuffer(fn, offset: fnOffset, index: 1)
        enc.setBuffer(base, offset: baseOffset, index: 2)
        enc.setBuffer(scale, offset: scaleOffset, index: 3)
        enc.setBuffer(x, offset: 0, index: 4)
        var mult = UInt32(hcMult)
        var hid = UInt32(hidden)
        var he = hcEps
        var re = rmsEps
        enc.setBytes(&mult, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hid, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&he, length: MemoryLayout<Float>.size, index: 7)
        enc.setBytes(&re, length: MemoryLayout<Float>.size, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeSwigluClampMul(commandBuffer: MTLCommandBuffer,
                              gate: MTLBuffer, up: MTLBuffer, out: MTLBuffer,
                              n: Int, limit: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(swigluClampPSO)
        enc.setBuffer(gate, offset: 0, index: 0)
        enc.setBuffer(up, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var count = UInt32(n)
        var lim = limit
        enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&lim, length: MemoryLayout<Float>.size, index: 4)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeBroadcastStreams(commandBuffer: MTLCommandBuffer,
                                x: MTLBuffer, streams: MTLBuffer,
                                hcMult: Int, hidden: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(broadcastPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(streams, offset: 0, index: 1)
        var mult = UInt32(hcMult)
        var hid = UInt32(hidden)
        enc.setBytes(&mult, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&hid, length: MemoryLayout<UInt32>.size, index: 3)
        enc.dispatchThreads(MTLSize(width: hidden, height: hcMult, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}
