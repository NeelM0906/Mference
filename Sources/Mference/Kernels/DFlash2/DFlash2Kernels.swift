import Foundation
import Metal

/// Encoders for the DFlash2 drafter kernels (`Metal/DFlash2/dflash2.metal`).
/// All BF16 weight consumers accumulate in fp32; the drafter's numerics only
/// influence acceptance length, never emitted bytes.
final class DFlash2Kernels {

    /// Multi-x row cap of `dflash2_bf16_gemv_multix`; callers chunk larger
    /// row counts (the first post-prefill context projection).
    static let maxTokens = 8

    private let gemvPSO: MTLComputePipelineState
    private let gemvF32PSO: MTLComputePipelineState
    private let attentionPSO: MTLComputePipelineState
    private let dynconvPSO: MTLComputePipelineState
    private let dynconvF32PSO: MTLComputePipelineState
    private let topkPSO: MTLComputePipelineState
    private let tapGatherPSO: MTLComputePipelineState
    private let residualAddF32PSO: MTLComputePipelineState
    private let rmsF32RowsPSO: MTLComputePipelineState
    private let f16ToF32PSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.gemvPSO = try context.pipeline("dflash2_bf16_gemv_multix")
        self.gemvF32PSO = try context.pipeline("dflash2_bf16_gemv_multix_f32out")
        self.attentionPSO = try context.pipeline("dflash2_block_attention")
        self.dynconvPSO = try context.pipeline("dflash2_dynconv")
        self.dynconvF32PSO = try context.pipeline("dflash2_dynconv_f32io")
        self.topkPSO = try context.pipeline("dflash2_topk16")
        self.tapGatherPSO = try context.pipeline("dflash2_tap_gather")
        self.residualAddF32PSO = try context.pipeline("dflash2_residual_add_f32f32")
        self.rmsF32RowsPSO = try context.pipeline("inkling_rms_f32in_prefill")
        self.f16ToF32PSO = try context.pipeline("inkling_f16_to_f32")
    }

    /// Y[t, m] = sum_n W[m, n] * X[t, n]; weights read once for all rows.
    /// Chunks `tokens > maxTokens` internally by advancing X/Y.
    /// `outputFloat32` writes an FP32 Y (the residual-path projections).
    func encodeGEMV(commandBuffer cb: MTLCommandBuffer,
                    weights: MTLBuffer, weightsOffset: Int,
                    x: MTLBuffer, xOffset: Int = 0,
                    y: MTLBuffer, yOffset: Int = 0,
                    m: Int, n: Int, tokens: Int,
                    outputFloat32: Bool = false) {
        precondition(weightsOffset % 2 == 0, "bf16 weights need a 2-aligned offset")
        var remaining = tokens
        var xByte = xOffset
        var yByte = yOffset
        let h = MemoryLayout<Float16>.stride
        let yStride = outputFloat32 ? MemoryLayout<Float>.stride : h
        while remaining > 0 {
            let t = min(remaining, Self.maxTokens)
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(outputFloat32 ? gemvF32PSO : gemvPSO)
            enc.setBuffer(weights, offset: weightsOffset, index: 0)
            enc.setBuffer(x, offset: xByte, index: 1)
            enc.setBuffer(y, offset: yByte, index: 2)
            var mv = UInt32(m), nv = UInt32(n), tv = UInt32(t)
            enc.setBytes(&mv, length: 4, index: 3)
            enc.setBytes(&nv, length: 4, index: 4)
            enc.setBytes(&tv, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (m + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
            remaining -= t
            xByte += t * n * h
            yByte += t * m * yStride
        }
    }

    /// hidden (FP32) += delta (FP32).
    func encodeResidualAddF32(commandBuffer cb: MTLCommandBuffer,
                              hidden: MTLBuffer, delta: MTLBuffer,
                              count: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(residualAddF32PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(delta, offset: 0, index: 1)
        var cv = UInt32(count)
        enc.setBytes(&cv, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Row-batched RMS norm over an FP32 input with BF16 gains, FP16 out
    /// (`inkling_rms_f32in_prefill`).
    func encodeRMSF32Rows(commandBuffer cb: MTLCommandBuffer,
                          x: MTLBuffer, xOffset: Int = 0,
                          weight: MTLBuffer, weightOffset: Int,
                          out: MTLBuffer, outOffset: Int = 0,
                          rows: Int, d: Int, eps: Float) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(rmsF32RowsPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var dv = UInt32(d), rv = UInt32(rows), ev = eps
        enc.setBytes(&dv, length: 4, index: 3)
        enc.setBytes(&rv, length: 4, index: 4)
        enc.setBytes(&ev, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// FP16 -> FP32 element copy (`inkling_f16_to_f32`).
    func encodeF16ToF32(commandBuffer cb: MTLCommandBuffer,
                        src: MTLBuffer, dst: MTLBuffer, count: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(f16ToF32PSO)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        var cv = UInt32(count)
        enc.setBytes(&cv, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeBlockAttention(commandBuffer cb: MTLCommandBuffer,
                              q: MTLBuffer,
                              ctxK: MTLBuffer, ctxV: MTLBuffer,
                              ctxByteOffset: Int = 0,
                              blkK: MTLBuffer, blkV: MTLBuffer,
                              out: MTLBuffer,
                              tokens: Int, ctxLen: Int, window: Int,
                              numQHeads: Int, numKVHeads: Int,
                              scale: Float) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(attentionPSO)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(ctxK, offset: ctxByteOffset, index: 1)
        enc.setBuffer(ctxV, offset: ctxByteOffset, index: 2)
        enc.setBuffer(blkK, offset: 0, index: 3)
        enc.setBuffer(blkV, offset: 0, index: 4)
        enc.setBuffer(out, offset: 0, index: 5)
        var tv = UInt32(tokens), cv = UInt32(ctxLen), wv = UInt32(window)
        var qh = UInt32(numQHeads), kh = UInt32(numKVHeads), sv = scale
        enc.setBytes(&tv, length: 4, index: 6)
        enc.setBytes(&cv, length: 4, index: 7)
        enc.setBytes(&wv, length: 4, index: 8)
        enc.setBytes(&qh, length: 4, index: 9)
        enc.setBytes(&kh, length: 4, index: 10)
        enc.setBytes(&sv, length: 4, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `plane` 0 = prepare (base kernel row 0), 1 = finish (row 1).
    /// `base` must point at the [K, H] plane for `plane` inside the
    /// checkpoint's [2, K, H] tensor. `float32IO` runs the FP32-in/out twin
    /// used at the residual-path finish sites.
    func encodeDynConv(commandBuffer cb: MTLCommandBuffer,
                       x: MTLBuffer,
                       dynamic dyn: MTLBuffer,
                       base: MTLBuffer, baseOffset: Int,
                       out: MTLBuffer,
                       tokens: Int, hidden: Int,
                       kernelSize: Int, groupSize: Int, plane: Int,
                       float32IO: Bool = false) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(float32IO ? dynconvF32PSO : dynconvPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(dyn, offset: 0, index: 1)
        enc.setBuffer(base, offset: baseOffset, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        var hv = UInt32(hidden), kv = UInt32(kernelSize)
        var gv = UInt32(groupSize), pv = UInt32(plane)
        enc.setBytes(&hv, length: 4, index: 4)
        enc.setBytes(&kv, length: 4, index: 5)
        enc.setBytes(&gv, length: 4, index: 6)
        enc.setBytes(&pv, length: 4, index: 7)
        enc.dispatchThreads(
            MTLSize(width: hidden, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeTopK16(commandBuffer cb: MTLCommandBuffer,
                      logits: MTLBuffer,
                      outIndices: MTLBuffer, outValues: MTLBuffer,
                      rows: Int, vocab: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(topkPSO)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(outIndices, offset: 0, index: 1)
        enc.setBuffer(outValues, offset: 0, index: 2)
        var vv = UInt32(vocab)
        enc.setBytes(&vv, length: 4, index: 3)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Copies `tokens` rows of `src [tokens, d]` into the concatenated tap
    /// staging matrix at column `tapIndex * d`, starting at `dstRow`.
    func encodeTapGather(commandBuffer cb: MTLCommandBuffer,
                         src: MTLBuffer, srcOffset: Int = 0,
                         dst: MTLBuffer,
                         d: Int, tapCount: Int, tapIndex: Int,
                         dstRow: Int, tokens: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(tapGatherPSO)
        enc.setBuffer(src, offset: srcOffset, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        var dv = UInt32(d), tc = UInt32(tapCount)
        var ti = UInt32(tapIndex), dr = UInt32(dstRow)
        enc.setBytes(&dv, length: 4, index: 2)
        enc.setBytes(&tc, length: 4, index: 3)
        enc.setBytes(&ti, length: 4, index: 4)
        enc.setBytes(&dr, length: 4, index: 5)
        enc.dispatchThreads(
            MTLSize(width: d, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}
