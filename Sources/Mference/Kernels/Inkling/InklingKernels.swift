import Foundation
import Metal

/// Inkling-Small family kernels: depthwise short convolutions, per-head Q/K
/// RMS norm (no V norm), single-token attention with the learned
/// relative-position bias, the sigmoid router with shared-expert sinks, and
/// two small elementwise helpers.
///
/// The router GEMV reuses `router_gemv_bf16_r4` (the DSV4/Qwen unquantized
/// BF16 router precedent) over 258 rows — 256 routed logits plus the 2
/// shared-expert sink logits — into a dedicated fp32 logits buffer;
/// `inkling_router_select` then applies the family's selection and weighting
/// semantics (see docs/INKLING_SMALL.md "Forward-pass contract").
final class InklingKernels {
    private let sconvPSO: MTLComputePipelineState
    private let qkNormPSO: MTLComputePipelineState
    private let attentionPSO: MTLComputePipelineState
    private let routerGemvPSO: MTLComputePipelineState
    private let routerSelectPSO: MTLComputePipelineState
    private let gammaCombinePSO: MTLComputePipelineState
    private let gammaCombineF32InPSO: MTLComputePipelineState
    private let scalePSO: MTLComputePipelineState
    private let headEpiloguePSO: MTLComputePipelineState
    private let scaleAccumPSO: MTLComputePipelineState
    private let scaleAccumF32InPSO: MTLComputePipelineState
    private let f32ToF16PSO: MTLComputePipelineState
    private let f16ToF32PSO: MTLComputePipelineState
    private let sconvF32OutPSO: MTLComputePipelineState
    private let residualAddF32DPSO: MTLComputePipelineState
    private let residualAddF32PSO: MTLComputePipelineState
    private let rmsF32PSO: MTLComputePipelineState
    /// fp32 logits for the 256 routed + 2 shared sink scores.
    let routerLogits: MTLBuffer
    /// fp32 gammas for the 2 shared experts, written by the select kernel.
    let sharedGammas: MTLBuffer
    /// One `uint`: how many real-vocabulary logits the head epilogue found
    /// non-finite. Reset before each head dispatch, read after it completes.
    let nonFiniteLogitCount: MTLBuffer

    init(context: MetalContext,
         numRouted: Int,
         numShared: Int) throws {
        self.sconvPSO = try context.pipeline("inkling_sconv_step")
        self.qkNormPSO = try context.pipeline("inkling_qk_norm")
        self.attentionPSO = try context.pipeline("inkling_attention_decode")
        self.routerGemvPSO = try context.pipeline("router_gemv_bf16_r4")
        self.routerSelectPSO = try context.pipeline("inkling_router_select")
        self.gammaCombinePSO = try context.pipeline("inkling_gamma_combine")
        self.gammaCombineF32InPSO =
            try context.pipeline("inkling_gamma_combine_f32in")
        self.scalePSO = try context.pipeline("inkling_scale_f16")
        self.headEpiloguePSO = try context.pipeline("inkling_head_epilogue")
        self.f16ToF32PSO = try context.pipeline("inkling_f16_to_f32")
        self.scaleAccumPSO = try context.pipeline("inkling_scale_accum_f32")
        self.scaleAccumF32InPSO =
            try context.pipeline("inkling_scale_accum_f32_from_f32")
        self.f32ToF16PSO = try context.pipeline("inkling_f32_to_f16")
        self.sconvF32OutPSO = try context.pipeline("inkling_sconv_step_f32out")
        self.residualAddF32DPSO = try context.pipeline("inkling_residual_add_f32d")
        self.residualAddF32PSO = try context.pipeline("inkling_residual_add_f32")
        self.rmsF32PSO = try context.pipeline("inkling_rms_f32in")
        guard let logits = context.device.makeBuffer(
            length: (numRouted + numShared) * MemoryLayout<Float>.size,
            options: .storageModeShared) else {
            throw MetalError.libraryCompileFailed("inkling router logits buffer allocation failed")
        }
        logits.label = "inkling.routerLogits"
        self.routerLogits = logits
        guard let gammas = context.device.makeBuffer(
            length: max(numShared, 1) * MemoryLayout<Float>.size,
            options: .storageModeShared) else {
            throw MetalError.libraryCompileFailed("inkling shared gammas buffer allocation failed")
        }
        gammas.label = "inkling.sharedGammas"
        self.sharedGammas = gammas
        guard let guardBuf = context.device.makeBuffer(
            length: MemoryLayout<UInt32>.size,
            options: .storageModeShared) else {
            throw MetalError.libraryCompileFailed("inkling logit guard buffer allocation failed")
        }
        guardBuf.label = "inkling.nonFiniteLogitCount"
        guardBuf.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        self.nonFiniteLogitCount = guardBuf
    }

    func resetNonFiniteLogitCount() {
        nonFiniteLogitCount.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    }

    var nonFiniteLogits: Int {
        Int(nonFiniteLogitCount.contents().load(as: UInt32.self))
    }

    /// One decode step of the depthwise causal conv: `out = conv(x) + x`,
    /// fp32 state (last K-1 inputs per channel) updated in place.
    func encodeSconvStep(commandBuffer cb: MTLCommandBuffer,
                         x: MTLBuffer, xOffset: Int = 0,
                         state: MTLBuffer,
                         weight: MTLBuffer, weightOffset: Int,
                         out: MTLBuffer, outOffset: Int = 0,
                         channels: UInt32,
                         kernelSize: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var c = channels
        var k = kernelSize
        enc.setComputePipelineState(sconvPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(state, offset: 0, index: 1)
        enc.setBuffer(weight, offset: weightOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        enc.setBytes(&c, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: Int(channels), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Per-head weighted RMS norm on Q [nq, hd] and K [nkv, hd] in place.
    func encodeQKNorm(commandBuffer cb: MTLCommandBuffer,
                      q: MTLBuffer,
                      k: MTLBuffer, kOffset: Int,
                      qWeight: MTLBuffer, qWeightOffset: Int,
                      kWeight: MTLBuffer, kWeightOffset: Int,
                      headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
                      eps: Float) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var hd = headDim, nq = numQHeads, nkv = numKVHeads, e = eps
        enc.setComputePipelineState(qkNormPSO)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 2)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 3)
        enc.setBytes(&hd, length: 4, index: 4)
        enc.setBytes(&nq, length: 4, index: 5)
        enc.setBytes(&nkv, length: 4, index: 6)
        enc.setBytes(&e, length: 4, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(numQHeads + numKVHeads), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Single-token GQA attention with the inline relative-position bias.
    /// `relExtent` is the bias width for THIS layer (window for sliding
    /// layers, `rel_extent` for global ones); `tau` is the log-scaling factor
    /// (1.0 below the floor and on sliding layers); `ringCapacity` is 0 for
    /// non-ring caches.
    func encodeAttentionDecode(commandBuffer cb: MTLCommandBuffer,
                               q: MTLBuffer,
                               k: MTLBuffer,
                               v: MTLBuffer,
                               rel: MTLBuffer,
                               proj: MTLBuffer, projOffset: Int,
                               out: MTLBuffer,
                               headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
                               seqLen: UInt32, kvStart: UInt32,
                               relExtent: UInt32, dRel: UInt32,
                               ringCapacity: UInt32,
                               scale: Float, tau: Float) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var hd = headDim, nq = numQHeads, nkv = numKVHeads
        var sl = seqLen, ks = kvStart, re = relExtent, dr = dRel, rc = ringCapacity
        var sc = scale, tu = tau
        enc.setComputePipelineState(attentionPSO)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(rel, offset: 0, index: 3)
        enc.setBuffer(proj, offset: projOffset, index: 4)
        enc.setBuffer(out, offset: 0, index: 5)
        enc.setBytes(&hd, length: 4, index: 6)
        enc.setBytes(&nq, length: 4, index: 7)
        enc.setBytes(&nkv, length: 4, index: 8)
        enc.setBytes(&sl, length: 4, index: 9)
        enc.setBytes(&ks, length: 4, index: 10)
        enc.setBytes(&re, length: 4, index: 11)
        enc.setBytes(&dr, length: 4, index: 12)
        enc.setBytes(&rc, length: 4, index: 13)
        enc.setBytes(&sc, length: 4, index: 14)
        enc.setBytes(&tu, length: 4, index: 15)
        enc.dispatchThreadgroups(MTLSize(width: Int(numQHeads), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// BF16 router GEMV over `numRouted + numShared` rows into `routerLogits`,
    /// then selection + weighting. `onesScale` is a bf16 all-ones [d] buffer.
    func encodeRouter(commandBuffer cb: MTLCommandBuffer,
                      weights: MTLBuffer, weightsOffset: Int,
                      hidden: MTLBuffer, hiddenOffset: Int = 0,
                      onesScale: MTLBuffer,
                      gateBias: MTLBuffer, gateBiasOffset: Int,
                      globalScale: MTLBuffer, globalScaleOffset: Int,
                      outIndices: MTLBuffer, outIndicesOffset: Int = 0,
                      outWeights: MTLBuffer, outWeightsOffset: Int = 0,
                      gammasOut: MTLBuffer? = nil, gammasOffset: Int = 0,
                      numRouted: UInt32, numShared: UInt32,
                      topK: UInt32, routeScale: Float,
                      d: UInt32) {
        var total = numRouted + numShared
        var dim = d
        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(routerGemvPSO)
            enc.setBuffer(weights, offset: weightsOffset, index: 0)
            enc.setBuffer(hidden, offset: hiddenOffset, index: 1)
            enc.setBuffer(onesScale, offset: 0, index: 2)
            enc.setBuffer(routerLogits, offset: 0, index: 3)
            enc.setBytes(&total, length: 4, index: 4)
            enc.setBytes(&dim, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (Int(total) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
        }
        encodeRouterSelectOnly(commandBuffer: cb,
                               gateBias: gateBias, gateBiasOffset: gateBiasOffset,
                               globalScale: globalScale,
                               globalScaleOffset: globalScaleOffset,
                               outIndices: outIndices,
                               outIndicesOffset: outIndicesOffset,
                               outWeights: outWeights,
                               outWeightsOffset: outWeightsOffset,
                               gammasOut: gammasOut, gammasOffset: gammasOffset,
                               numRouted: numRouted, numShared: numShared,
                               topK: topK, routeScale: routeScale)
    }

    /// Selection + weighting over an already-populated `routerLogits`.
    /// Split out so tests can feed synthetic logits directly.
    func encodeRouterSelectOnly(commandBuffer cb: MTLCommandBuffer,
                                gateBias: MTLBuffer, gateBiasOffset: Int,
                                globalScale: MTLBuffer, globalScaleOffset: Int,
                                outIndices: MTLBuffer, outIndicesOffset: Int = 0,
                                outWeights: MTLBuffer, outWeightsOffset: Int = 0,
                                gammasOut: MTLBuffer? = nil, gammasOffset: Int = 0,
                                numRouted: UInt32, numShared: UInt32,
                                topK: UInt32, routeScale: Float) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var nr = numRouted, ns = numShared, tk = topK, rs = routeScale
        enc.setComputePipelineState(routerSelectPSO)
        enc.setBuffer(routerLogits, offset: 0, index: 0)
        enc.setBuffer(gateBias, offset: gateBiasOffset, index: 1)
        enc.setBuffer(globalScale, offset: globalScaleOffset, index: 2)
        enc.setBuffer(outIndices, offset: outIndicesOffset, index: 3)
        enc.setBuffer(outWeights, offset: outWeightsOffset, index: 4)
        enc.setBuffer(gammasOut ?? sharedGammas, offset: gammasOffset, index: 5)
        enc.setBytes(&nr, length: 4, index: 6)
        enc.setBytes(&ns, length: 4, index: 7)
        enc.setBytes(&tk, length: 4, index: 8)
        enc.setBytes(&rs, length: 4, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// y = gammas[0]*a + gammas[1]*b.
    func encodeGammaCombine(commandBuffer cb: MTLCommandBuffer,
                            a: MTLBuffer, b: MTLBuffer,
                            y: MTLBuffer,
                            count: UInt32,
                            inputsAreFloat32: Bool = false) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var n = count
        enc.setComputePipelineState(
            inputsAreFloat32 ? gammaCombineF32InPSO : gammaCombinePSO)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(sharedGammas, offset: 0, index: 2)
        enc.setBuffer(y, offset: 0, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// acc(f32, at offset) += w * x, with `x` FP16 or (for expert outputs whose
    /// raw rows leave FP16 range) FP32.
    func encodeScaleAccum(commandBuffer cb: MTLCommandBuffer,
                          acc: MTLBuffer, accOffset: Int,
                          x: MTLBuffer,
                          weight: Float, count: UInt32,
                          xIsFloat32: Bool = false) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var w = weight
        var n = count
        enc.setComputePipelineState(
            xIsFloat32 ? scaleAccumF32InPSO : scaleAccumPSO)
        enc.setBuffer(acc, offset: accOffset, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBytes(&w, length: 4, index: 2)
        enc.setBytes(&n, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// dst(f16) = src(f32 at offset), narrowing.
    func encodeF32ToF16(commandBuffer cb: MTLCommandBuffer,
                        src: MTLBuffer, srcOffset: Int,
                        dst: MTLBuffer, count: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var n = count
        enc.setComputePipelineState(f32ToF16PSO)
        enc.setBuffer(src, offset: srcOffset, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Sconv step with FP32 output (attn/mlp sublayer-output sites).
    func encodeSconvStepF32Out(commandBuffer cb: MTLCommandBuffer,
                               x: MTLBuffer,
                               state: MTLBuffer,
                               weight: MTLBuffer, weightOffset: Int,
                               out: MTLBuffer,
                               channels: UInt32,
                               kernelSize: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var c = channels
        var k = kernelSize
        enc.setComputePipelineState(sconvF32OutPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(state, offset: 0, index: 1)
        enc.setBuffer(weight, offset: weightOffset, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.setBytes(&c, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: Int(channels), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// hidden(f32) += scale * delta(f32).
    func encodeResidualAddF32Delta(commandBuffer cb: MTLCommandBuffer,
                                   hidden: MTLBuffer, hiddenOffset: Int = 0,
                                   delta: MTLBuffer,
                                   count: UInt32, scale: Float = 1.0) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var n = count
        var sc = scale
        enc.setComputePipelineState(residualAddF32DPSO)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
        enc.setBuffer(delta, offset: 0, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.setBytes(&sc, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// hidden(f32) = f16 src widened; the embed seeds the FP32 stream once.
    func encodeF16ToF32(commandBuffer cb: MTLCommandBuffer,
                        src: MTLBuffer, dst: MTLBuffer, dstOffset: Int = 0,
                        count: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var n = count
        enc.setComputePipelineState(f16ToF32PSO)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: dstOffset, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// hidden(f32) += delta(f16).
    func encodeResidualAddF32(commandBuffer cb: MTLCommandBuffer,
                              hidden: MTLBuffer, delta: MTLBuffer, count: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var n = count
        enc.setComputePipelineState(residualAddF32PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(delta, offset: 0, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// out(f16) = rmsnorm(x_f32) * weight(bf16).
    func encodeRMSF32(commandBuffer cb: MTLCommandBuffer,
                      x: MTLBuffer, xOffset: Int = 0,
                      weight: MTLBuffer, weightOffset: Int,
                      out: MTLBuffer,
                      d: UInt32, eps: Float) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var dd = d, e = eps
        enc.setComputePipelineState(rmsF32PSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBytes(&dd, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Logits epilogue: scale the real vocabulary by `c`, pin the padding tail
    /// to -inf so it can never be sampled, and count any non-finite real logit
    /// into `nonFiniteLogitCount`. Callers must reset that counter (see
    /// `resetNonFiniteLogitCount()`) before encoding and read it after the
    /// command buffer completes.
    func encodeHeadEpilogue(commandBuffer cb: MTLCommandBuffer,
                            logits: MTLBuffer,
                            scale c: Float,
                            validVocab: UInt32,
                            totalVocab: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var scale = c
        var valid = validVocab
        var total = totalVocab
        enc.setComputePipelineState(headEpiloguePSO)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBytes(&scale, length: 4, index: 1)
        enc.setBytes(&valid, length: 4, index: 2)
        enc.setBytes(&total, length: 4, index: 3)
        enc.setBuffer(nonFiniteLogitCount, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: Int(totalVocab), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// In-place x *= c over `count` half elements.
    func encodeScale(commandBuffer cb: MTLCommandBuffer,
                     x: MTLBuffer,
                     by c: Float,
                     count: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var scale = c
        var n = count
        enc.setComputePipelineState(scalePSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBytes(&scale, length: 4, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}
