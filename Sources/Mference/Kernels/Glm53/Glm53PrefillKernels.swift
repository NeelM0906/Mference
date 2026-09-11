import Foundation
import Metal

/// Encoders for `glm53_prefill.metal` and the grouped-expert prefill kernels
/// in `moe.metal`: the chunked (batched) GLM-5.3 prefill.
final class Glm53PrefillKernels {
    private let int8GemmPSO: MTLComputePipelineState
    private let bf16GemmPSO: MTLComputePipelineState
    private let headedGemvPSO: MTLComputePipelineState
    private let embedPSO: MTLComputePipelineState
    private let broadcastPSO: MTLComputePipelineState
    private let hcDotsPSO: MTLComputePipelineState
    private let hcFinalizePSO: MTLComputePipelineState
    private let hcCollapsePSO: MTLComputePipelineState
    private let hcPlaceMixPSO: MTLComputePipelineState
    private let kdaChunkPSO: MTLComputePipelineState
    private let latentCausalPSO: MTLComputePipelineState
    private let layerNormPSO: MTLComputePipelineState
    private let poolKeysPSO: MTLComputePipelineState
    private let routerSelectPSO: MTLComputePipelineState
    private let moePhase1PSO: MTLComputePipelineState
    private let moeDownPSO: MTLComputePipelineState
    private let moeReducePSO: MTLComputePipelineState

    /// Tokens per GEMM threadgroup pass (`acc[16]` in the BF16 kernel).
    static let gemmTokenTile = 16
    /// Tokens per INT8 GEMM threadgroup (`kG53PTokenTile`, staged tile of 32 x 256).
    static let int8TokenTile = 32

    init(context: MetalContext, swigluLimit: Float) throws {
        int8GemmPSO = try context.pipeline("glm53p_int8_gemm", constants: [],
                                           maxTotalThreadsPerThreadgroup: 256)
        bf16GemmPSO = try context.pipeline("glm53p_bf16_gemm", constants: [],
                                           maxTotalThreadsPerThreadgroup: 256)
        headedGemvPSO = try context.pipeline("glm53p_headed_int8_gemv_batched", constants: [],
                                             maxTotalThreadsPerThreadgroup: 256)
        embedPSO = try context.pipeline("glm53p_embed_lookup_int8_batched")
        broadcastPSO = try context.pipeline("glm53p_broadcast_streams_batched")
        hcDotsPSO = try context.pipeline("glm53p_hc_dots_batched", constants: [],
                                         maxTotalThreadsPerThreadgroup: 256)
        hcFinalizePSO = try context.pipeline("glm53p_hc_finalize_batched")
        hcCollapsePSO = try context.pipeline("glm53p_hc_collapse_batched")
        hcPlaceMixPSO = try context.pipeline("glm53p_hc_place_mix_batched")
        kdaChunkPSO = try context.pipeline("glm53p_kda_chunk", constants: [],
                                           maxTotalThreadsPerThreadgroup: 256)
        latentCausalPSO = try context.pipeline("glm53p_latent_attention_causal", constants: [],
                                               maxTotalThreadsPerThreadgroup: 256)
        layerNormPSO = try context.pipeline("glm53p_layernorm_bias_batched", constants: [],
                                            maxTotalThreadsPerThreadgroup: 256)
        poolKeysPSO = try context.pipeline("glm53p_pool_keys_batched")
        routerSelectPSO = try context.pipeline("glm53p_router_select_k8_batched")
        // The grouped expert kernels share moe.metal's INT4 bodies; the swiglu
        // clamp and SiLU come through the same function constants as `MoE`.
        var moeConstants: [MetalFunctionConstant] = [MetalFunctionConstant(index: 4, value: .bool(true))]
        if swigluLimit > 0 {
            moeConstants.append(MetalFunctionConstant(index: 5, value: .float(swigluLimit)))
        }
        moePhase1PSO = try context.pipeline("glm53p_moe_grouped_phase1", constants: moeConstants)
        moeDownPSO = try context.pipeline("glm53p_moe_grouped_down", constants: moeConstants)
        moeReducePSO = try context.pipeline("glm53p_moe_route_reduce")
    }

    private static func tg(_ w: Int, _ h: Int = 1, _ d: Int = 1) -> MTLSize { MTLSize(width: w, height: h, depth: d) }

    /// `Y[t][m] = W x[t]` for `t < tokens`; INT8 g64 affine `[m, n]` weights.
    func encodeInt8GEMM(commandBuffer cb: MTLCommandBuffer, weights: TensorView,
                        x: MTLBuffer, xOffset: Int = 0, xStride: Int,
                        y: MTLBuffer, yOffset: Int = 0, yStride: Int,
                        m: Int, n: Int, tokens: Int) {
        precondition(n % Quantization.groupSize == 0)
        guard tokens > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(int8GemmPSO)
        enc.setBuffer(weights.buffer, offset: Int(weights.offset), index: 0)
        enc.setBuffer(weights.buffer, offset: Int(weights.scaleOffset), index: 1)
        enc.setBuffer(weights.buffer, offset: Int(weights.biasOffset), index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var mm = UInt32(m), nn = UInt32(n), tt = UInt32(tokens), xs = UInt32(xStride), ys = UInt32(yStride)
        enc.setBytes(&mm, length: 4, index: 5)
        enc.setBytes(&nn, length: 4, index: 6)
        enc.setBytes(&tt, length: 4, index: 7)
        enc.setBytes(&xs, length: 4, index: 8)
        enc.setBytes(&ys, length: 4, index: 9)
        enc.dispatchThreadgroups(Self.tg((m + 7) / 8, (tokens + Self.int8TokenTile - 1) / Self.int8TokenTile),
                                 threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }

    /// BF16 `[m, n]` weights; fp16 or fp32 output.
    func encodeBF16GEMM(commandBuffer cb: MTLCommandBuffer, weights: TensorView,
                        x: MTLBuffer, xStride: Int,
                        y: MTLBuffer, yOffset: Int = 0, yStride: Int, outputFloat32: Bool,
                        m: Int, n: Int, tokens: Int) {
        precondition(n % 64 == 0)
        guard tokens > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(bf16GemmPSO)
        enc.setBuffer(weights.buffer, offset: Int(weights.offset), index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: yOffset, index: 2)
        enc.setBuffer(y, offset: yOffset, index: 3)
        var mm = UInt32(m), nn = UInt32(n), tt = UInt32(tokens), xs = UInt32(xStride), ys = UInt32(yStride)
        var f32: UInt32 = outputFloat32 ? 1 : 0
        enc.setBytes(&mm, length: 4, index: 4)
        enc.setBytes(&nn, length: 4, index: 5)
        enc.setBytes(&tt, length: 4, index: 6)
        enc.setBytes(&xs, length: 4, index: 7)
        enc.setBytes(&ys, length: 4, index: 8)
        enc.setBytes(&f32, length: 4, index: 9)
        enc.dispatchThreadgroups(Self.tg((m + 7) / 8, (tokens + Self.gemmTokenTile - 1) / Self.gemmTokenTile),
                                 threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }

    func encodeHeadedGEMV(commandBuffer cb: MTLCommandBuffer, weights: TensorView,
                          x: MTLBuffer, y: MTLBuffer, heads: Int, m: Int, n: Int, tokens: Int) {
        guard tokens > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(headedGemvPSO)
        enc.setBuffer(weights.buffer, offset: Int(weights.offset), index: 0)
        enc.setBuffer(weights.buffer, offset: Int(weights.scaleOffset), index: 1)
        enc.setBuffer(weights.buffer, offset: Int(weights.biasOffset), index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        var mm = UInt32(m), nn = UInt32(n), hh = UInt32(heads)
        enc.setBytes(&mm, length: 4, index: 5)
        enc.setBytes(&nn, length: 4, index: 6)
        enc.setBytes(&hh, length: 4, index: 7)
        enc.dispatchThreadgroups(Self.tg((m + 7) / 8, heads, tokens), threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }

    func encodeEmbed(commandBuffer cb: MTLCommandBuffer, table: TensorView, tokens: MTLBuffer,
                     out: MTLBuffer, d: Int, count: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(embedPSO)
        enc.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        enc.setBuffer(table.buffer, offset: Int(table.scaleOffset), index: 1)
        enc.setBuffer(table.buffer, offset: Int(table.biasOffset), index: 2)
        enc.setBuffer(tokens, offset: 0, index: 3)
        enc.setBuffer(out, offset: 0, index: 4)
        var dd = UInt32(d)
        enc.setBytes(&dd, length: 4, index: 5)
        enc.dispatchThreads(Self.tg(d, count), threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }

    func encodeBroadcast(commandBuffer cb: MTLCommandBuffer, x: MTLBuffer, streams: MTLBuffer,
                         hcMult: Int, hidden: Int, tokens: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(broadcastPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(streams, offset: 0, index: 1)
        var h = UInt32(hcMult), d = UInt32(hidden)
        enc.setBytes(&h, length: 4, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.dispatchThreads(Self.tg(hidden, hcMult * tokens), threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }

    /// mHC mixes for every token of the chunk: `pre/post/comb` per token.
    func encodeHCWeights(commandBuffer cb: MTLCommandBuffer, streams: MTLBuffer,
                         fn: TensorView, base: TensorView, scale: TensorView, partials: MTLBuffer,
                         outPre: MTLBuffer, outPost: MTLBuffer, outComb: MTLBuffer,
                         hcMult: Int, hidden: Int, sinkhornIters: Int, hcEps: Float, rmsEps: Float, tokens: Int) {
        let rows = (2 + hcMult) * hcMult
        precondition(rows <= 24 && (hcMult * hidden) % 4 == 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var flat = UInt32(hcMult * hidden), rowCount = UInt32(rows)
        enc.setComputePipelineState(hcDotsPSO)
        enc.setBuffer(streams, offset: 0, index: 0)
        enc.setBuffer(fn.buffer, offset: Int(fn.offset), index: 1)
        enc.setBuffer(partials, offset: 0, index: 2)
        enc.setBytes(&flat, length: 4, index: 3)
        enc.setBytes(&rowCount, length: 4, index: 4)
        enc.dispatchThreadgroups(Self.tg(rows + 1, tokens), threadsPerThreadgroup: Self.tg(256))

        var mult = UInt32(hcMult), iters = UInt32(sinkhornIters), he = hcEps, re = rmsEps, tt = UInt32(tokens)
        enc.setComputePipelineState(hcFinalizePSO)
        enc.setBuffer(partials, offset: 0, index: 0)
        enc.setBuffer(base.buffer, offset: Int(base.offset), index: 1)
        enc.setBuffer(scale.buffer, offset: Int(scale.offset), index: 2)
        enc.setBuffer(outPre, offset: 0, index: 3)
        enc.setBuffer(outPost, offset: 0, index: 4)
        enc.setBuffer(outComb, offset: 0, index: 5)
        enc.setBytes(&mult, length: 4, index: 6)
        enc.setBytes(&flat, length: 4, index: 7)
        enc.setBytes(&iters, length: 4, index: 8)
        enc.setBytes(&he, length: 4, index: 9)
        enc.setBytes(&re, length: 4, index: 10)
        enc.setBytes(&tt, length: 4, index: 11)
        enc.dispatchThreads(Self.tg(tokens), threadsPerThreadgroup: Self.tg(min(64, tokens)))
        enc.endEncoding()
    }

    func encodeHCCollapse(commandBuffer cb: MTLCommandBuffer, streams: MTLBuffer, streamsOffset: Int = 0,
                          pre: MTLBuffer, x: MTLBuffer, hcMult: Int, hidden: Int, tokens: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcCollapsePSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        enc.setBuffer(pre, offset: 0, index: 1)
        enc.setBuffer(x, offset: 0, index: 2)
        var h = UInt32(hcMult), d = UInt32(hidden)
        enc.setBytes(&h, length: 4, index: 3)
        enc.setBytes(&d, length: 4, index: 4)
        enc.dispatchThreads(Self.tg(hidden, tokens), threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }

    func encodeHCPlaceMix(commandBuffer cb: MTLCommandBuffer, streams: MTLBuffer, sub: MTLBuffer,
                          post: MTLBuffer, comb: MTLBuffer, outStreams: MTLBuffer,
                          hcMult: Int, hidden: Int, tokens: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcPlaceMixPSO)
        enc.setBuffer(streams, offset: 0, index: 0)
        enc.setBuffer(sub, offset: 0, index: 1)
        enc.setBuffer(post, offset: 0, index: 2)
        enc.setBuffer(comb, offset: 0, index: 3)
        enc.setBuffer(outStreams, offset: 0, index: 4)
        var h = UInt32(hcMult), d = UInt32(hidden)
        enc.setBytes(&h, length: 4, index: 5)
        enc.setBytes(&d, length: 4, index: 6)
        enc.dispatchThreads(Self.tg(hidden, hcMult * tokens), threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }

    func encodeKDAChunk(commandBuffer cb: MTLCommandBuffer, convOut: MTLBuffer, a: MTLBuffer, b: MTLBuffer,
                        gate: MTLBuffer, aLog: TensorView, dtBias: TensorView, oNorm: TensorView,
                        state: MTLBuffer, out: MTLBuffer, heads: Int, headDim: Int, tokens: Int,
                        lowerBound: Float, eps: Float) {
        precondition(headDim % 32 == 0 && headDim <= 128)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(kdaChunkPSO)
        enc.setBuffer(convOut, offset: 0, index: 0)
        enc.setBuffer(a, offset: 0, index: 1)
        enc.setBuffer(b, offset: 0, index: 2)
        enc.setBuffer(gate, offset: 0, index: 3)
        enc.setBuffer(aLog.buffer, offset: Int(aLog.offset), index: 4)
        enc.setBuffer(dtBias.buffer, offset: Int(dtBias.offset), index: 5)
        enc.setBuffer(oNorm.buffer, offset: Int(oNorm.offset), index: 6)
        enc.setBuffer(state, offset: 0, index: 7)
        enc.setBuffer(out, offset: 0, index: 8)
        var h = UInt32(heads), d = UInt32(headDim), t = UInt32(tokens), lb = lowerBound, e = eps
        enc.setBytes(&h, length: 4, index: 9)
        enc.setBytes(&d, length: 4, index: 10)
        enc.setBytes(&t, length: 4, index: 11)
        enc.setBytes(&lb, length: 4, index: 12)
        enc.setBytes(&e, length: 4, index: 13)
        enc.dispatchThreadgroups(Self.tg(heads), threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }

    func encodeLatentAttentionCausal(commandBuffer cb: MTLCommandBuffer, qLatent: MTLBuffer, latents: MTLBuffer,
                                     out: MTLBuffer, heads: Int, latentDim: Int, base: Int, tokens: Int,
                                     scale: Float) {
        precondition(latentDim % 32 == 0 && latentDim <= 512)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(latentCausalPSO)
        enc.setBuffer(qLatent, offset: 0, index: 0)
        enc.setBuffer(latents, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var kv = UInt32(latentDim), b = UInt32(base), h = UInt32(heads), s = scale
        enc.setBytes(&kv, length: 4, index: 3)
        enc.setBytes(&b, length: 4, index: 4)
        enc.setBytes(&h, length: 4, index: 5)
        enc.setBytes(&s, length: 4, index: 6)
        enc.dispatchThreadgroups(Self.tg(heads, tokens), threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }

    func encodeLayerNormBias(commandBuffer cb: MTLCommandBuffer, x: MTLBuffer, weight: TensorView, bias: TensorView,
                             out: MTLBuffer, outOffset: Int, d: Int, eps: Float, tokens: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(layerNormPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: Int(weight.offset), index: 1)
        enc.setBuffer(bias.buffer, offset: Int(bias.offset), index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var dd = UInt32(d), e = eps
        enc.setBytes(&dd, length: 4, index: 4)
        enc.setBytes(&e, length: 4, index: 5)
        enc.dispatchThreadgroups(Self.tg(tokens), threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }

    func encodePoolKeys(commandBuffer cb: MTLCommandBuffer, keys: MTLBuffer, gates: MTLBuffer, ape: TensorView,
                        pooled: MTLBuffer, firstPool: Int, count: Int, kPool: Int, dim: Int) {
        guard count > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(poolKeysPSO)
        enc.setBuffer(keys, offset: 0, index: 0)
        enc.setBuffer(gates, offset: 0, index: 1)
        enc.setBuffer(ape.buffer, offset: Int(ape.offset), index: 2)
        enc.setBuffer(pooled, offset: 0, index: 3)
        var fp = UInt32(firstPool), kp = UInt32(kPool), d = UInt32(dim)
        enc.setBytes(&fp, length: 4, index: 4)
        enc.setBytes(&kp, length: 4, index: 5)
        enc.setBytes(&d, length: 4, index: 6)
        enc.dispatchThreads(Self.tg(dim, count), threadsPerThreadgroup: Self.tg(min(256, dim)))
        enc.endEncoding()
    }

    func encodeRouterSelect(commandBuffer cb: MTLCommandBuffer, logits: MTLBuffer, bias: TensorView,
                            outIndices: MTLBuffer, outWeights: MTLBuffer, numExperts: Int, routeScale: Float,
                            tokens: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(routerSelectPSO)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(bias.buffer, offset: Int(bias.offset), index: 1)
        enc.setBuffer(outIndices, offset: 0, index: 2)
        enc.setBuffer(outWeights, offset: 0, index: 3)
        var e = UInt32(numExperts), s = routeScale
        enc.setBytes(&e, length: 4, index: 4)
        enc.setBytes(&s, length: 4, index: 5)
        enc.dispatchThreadgroups(Self.tg(tokens), threadsPerThreadgroup: Self.tg(32))
        enc.endEncoding()
    }

    /// The grouped routed FFN over a chunk: phase 1 (gate/up/act per route),
    /// down projection per route, and the per-token reduce onto `residual`.
    func encodeGroupedMoE(commandBuffer cb: MTLCommandBuffer, slab: ResidentExpertSlab, offsets: MoEExpertOffsets,
                          x: MTLBuffer, acts: MTLBuffer, partial: MTLBuffer,
                          pairToken: MTLBuffer, segStart: MTLBuffer, activeExperts: MTLBuffer, activeCount: Int,
                          routePair: MTLBuffer, weights: MTLBuffer, residual: MTLBuffer, y: MTLBuffer,
                          d: Int, f: Int, topK: Int, tokens: Int) {
        guard activeCount > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        var off = offsets
        var dd = UInt32(d), ff = UInt32(f), stride = UInt32(slab.expertStride), kk = UInt32(topK)
        enc.useResource(slab.buffer, usage: .read)
        enc.setComputePipelineState(moePhase1PSO)
        enc.setBuffer(slab.buffer, offset: slab.baseOffset, index: 0)
        enc.setBytes(&off, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        enc.setBuffer(x, offset: 0, index: 2)
        enc.setBuffer(acts, offset: 0, index: 3)
        enc.setBuffer(pairToken, offset: 0, index: 4)
        enc.setBuffer(segStart, offset: 0, index: 5)
        enc.setBuffer(activeExperts, offset: 0, index: 6)
        enc.setBytes(&dd, length: 4, index: 7)
        enc.setBytes(&ff, length: 4, index: 8)
        enc.setBytes(&stride, length: 4, index: 9)
        enc.dispatchThreadgroups(Self.tg((f + 7) / 8, activeCount), threadsPerThreadgroup: Self.tg(256))

        enc.setComputePipelineState(moeDownPSO)
        enc.setBuffer(slab.buffer, offset: slab.baseOffset, index: 0)
        enc.setBytes(&off, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        enc.setBuffer(acts, offset: 0, index: 2)
        enc.setBuffer(partial, offset: 0, index: 3)
        enc.setBuffer(segStart, offset: 0, index: 4)
        enc.setBuffer(activeExperts, offset: 0, index: 5)
        enc.setBytes(&dd, length: 4, index: 6)
        enc.setBytes(&ff, length: 4, index: 7)
        enc.setBytes(&stride, length: 4, index: 8)
        enc.dispatchThreadgroups(Self.tg((d + 7) / 8, activeCount), threadsPerThreadgroup: Self.tg(256))

        enc.setComputePipelineState(moeReducePSO)
        enc.setBuffer(partial, offset: 0, index: 0)
        enc.setBuffer(routePair, offset: 0, index: 1)
        enc.setBuffer(weights, offset: 0, index: 2)
        enc.setBuffer(residual, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        enc.setBytes(&dd, length: 4, index: 5)
        enc.setBytes(&kk, length: 4, index: 6)
        enc.dispatchThreads(Self.tg(d, tokens), threadsPerThreadgroup: Self.tg(256))
        enc.endEncoding()
    }
}
