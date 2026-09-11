import Foundation
import Metal

/// Swift wrappers for the GLM-5.3-Flash kernels (`Metal/Glm53/glm53.metal`)
/// and thin encoders for the shared-library kernels the family reuses
/// unchanged: DeepSeek V4's mHC mixes / collapse / place-mix, its swiglu
/// clamp and stream broadcast, its lightning-indexer scorer, and Qwen's
/// depthwise conv decode.
final class Glm53Kernels {
    private let embedInt8PSO: MTLComputePipelineState
    private let convDecodePSO: MTLComputePipelineState
    private let kdaDecodePSO: MTLComputePipelineState
    private let layerNormPSO: MTLComputePipelineState
    private let poolKeysPSO: MTLComputePipelineState
    private let latentAttentionPSO: MTLComputePipelineState
    private let headedInt8PSO: MTLComputePipelineState
    private let indexerScorePSO: MTLComputePipelineState
    private let hcWeightsPSO: MTLComputePipelineState
    private let hcCollapsePSO: MTLComputePipelineState
    private let hcPlaceMixPSO: MTLComputePipelineState
    private let swigluClampPSO: MTLComputePipelineState
    private let broadcastPSO: MTLComputePipelineState

    /// Widest KDA head the decode kernel's threadgroup scratch holds.
    static let maxKDAHeadDim = 128
    /// Widest latent the attention kernel's per-lane registers hold.
    static let maxLatentDim = 512
    /// Sentinel `selectedCount` that attends every cached row.
    static let attendAll: UInt32 = 0xFFFF_FFFF

    init(context: MetalContext) throws {
        embedInt8PSO = try context.pipeline("glm53_embed_lookup_int8")
        convDecodePSO = try context.pipeline("gdn_conv_mix_decode")
        kdaDecodePSO = try context.pipeline("glm53_kda_decode", constants: [],
                                            maxTotalThreadsPerThreadgroup: 256)
        layerNormPSO = try context.pipeline("glm53_layernorm_bias", constants: [],
                                            maxTotalThreadsPerThreadgroup: 256)
        poolKeysPSO = try context.pipeline("glm53_pool_keys")
        latentAttentionPSO = try context.pipeline("glm53_latent_attention", constants: [],
                                                  maxTotalThreadsPerThreadgroup: 256)
        headedInt8PSO = try context.pipeline("glm53_headed_int8_gemv", constants: [],
                                             maxTotalThreadsPerThreadgroup: 256)
        indexerScorePSO = try context.pipeline("dsv4_indexer_score")
        hcWeightsPSO = try context.pipeline("dsv4_hc_weights", constants: [],
                                            maxTotalThreadsPerThreadgroup: 256)
        hcCollapsePSO = try context.pipeline("dsv4_hc_collapse")
        hcPlaceMixPSO = try context.pipeline("dsv4_hc_place_mix")
        swigluClampPSO = try context.pipeline("dsv4_swiglu_clamp_mul")
        broadcastPSO = try context.pipeline("dsv4_broadcast_streams")
    }

    // MARK: - Embedding

    func encodeEmbedLookupInt8(commandBuffer: MTLCommandBuffer,
                               table: TensorView, out: MTLBuffer, tokenId: UInt32, d: Int) {
        precondition(d % Quantization.groupSize == 0)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(embedInt8PSO)
        enc.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        enc.setBuffer(table.buffer, offset: Int(table.scaleOffset), index: 1)
        enc.setBuffer(table.buffer, offset: Int(table.biasOffset), index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        var token = tokenId
        var dim = UInt32(d)
        enc.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreads(MTLSize(width: d, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, d), height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - Kimi Delta Attention

    /// Depthwise causal conv over `[tail | mixed]` with SiLU, shifting the tail
    /// in place (`gdn_conv_mix_decode`). `convWeight` is BF16 `[channels, taps]`.
    func encodeConvDecode(commandBuffer: MTLCommandBuffer,
                          tail: MTLBuffer, mixed: MTLBuffer,
                          convWeight: MTLBuffer, convWeightOffset: Int,
                          out: MTLBuffer, channels: Int, taps: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(convDecodePSO)
        enc.setBuffer(tail, offset: 0, index: 0)
        enc.setBuffer(mixed, offset: 0, index: 1)
        enc.setBuffer(convWeight, offset: convWeightOffset, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        var c = UInt32(channels)
        var k = UInt32(taps)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreads(MTLSize(width: channels, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// The KDA recurrence for one token, all heads; see `glm53_kda_decode`.
    func encodeKDADecode(commandBuffer: MTLCommandBuffer,
                         convOut: MTLBuffer, a: MTLBuffer, b: MTLBuffer, gate: MTLBuffer,
                         aLog: TensorView, dtBias: TensorView, oNorm: TensorView,
                         state: MTLBuffer, out: MTLBuffer, yOut: MTLBuffer,
                         heads: Int, headDim: Int, lowerBound: Float, eps: Float) {
        precondition(headDim % 32 == 0 && headDim <= Self.maxKDAHeadDim)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(kdaDecodePSO)
        enc.setBuffer(convOut, offset: 0, index: 0)
        enc.setBuffer(a, offset: 0, index: 1)
        enc.setBuffer(b, offset: 0, index: 2)
        enc.setBuffer(gate, offset: 0, index: 3)
        enc.setBuffer(aLog.buffer, offset: Int(aLog.offset), index: 4)
        enc.setBuffer(dtBias.buffer, offset: Int(dtBias.offset), index: 5)
        enc.setBuffer(oNorm.buffer, offset: Int(oNorm.offset), index: 6)
        enc.setBuffer(state, offset: 0, index: 7)
        enc.setBuffer(out, offset: 0, index: 8)
        enc.setBuffer(yOut, offset: 0, index: 9)
        var h = UInt32(heads)
        var d = UInt32(headDim)
        var lb = lowerBound
        var e = eps
        enc.setBytes(&h, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&lb, length: MemoryLayout<Float>.size, index: 12)
        enc.setBytes(&e, length: MemoryLayout<Float>.size, index: 13)
        enc.dispatchThreadgroups(MTLSize(width: heads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - Sparse attention and indexer

    func encodeLayerNormBias(commandBuffer: MTLCommandBuffer,
                             x: MTLBuffer, xOffset: Int = 0,
                             weight: TensorView, bias: TensorView,
                             out: MTLBuffer, outOffset: Int = 0, d: Int, eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(layerNormPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight.buffer, offset: Int(weight.offset), index: 1)
        enc.setBuffer(bias.buffer, offset: Int(bias.offset), index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var dim = UInt32(d)
        var e = eps
        enc.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&e, length: MemoryLayout<Float>.size, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Pooled key for complete pool `pool` (tokens `pool*kPool ..< (pool+1)*kPool`).
    func encodePoolKeys(commandBuffer: MTLCommandBuffer,
                        keys: MTLBuffer, gates: MTLBuffer, ape: TensorView,
                        pooled: MTLBuffer, pool: Int, kPool: Int, dim: Int) {
        precondition(kPool <= 8)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(poolKeysPSO)
        enc.setBuffer(keys, offset: 0, index: 0)
        enc.setBuffer(gates, offset: 0, index: 1)
        enc.setBuffer(ape.buffer, offset: Int(ape.offset), index: 2)
        enc.setBuffer(pooled, offset: 0, index: 3)
        var p = UInt32(pool)
        var kp = UInt32(kPool)
        var d = UInt32(dim)
        enc.setBytes(&p, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&kp, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreads(MTLSize(width: dim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, dim), height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `score[j] = sum_h w[h] * wScale * relu(q_h . key_j * headScale)` over
    /// `entryCount` pooled keys (`dsv4_indexer_score`).
    func encodeIndexerScore(commandBuffer: MTLCommandBuffer,
                            q: MTLBuffer, keys: MTLBuffer, weights: MTLBuffer, scores: MTLBuffer,
                            numHeads: Int, indexDim: Int, entryCount: Int,
                            headScale: Float, weightScale: Float) {
        guard entryCount > 0, let enc = commandBuffer.makeComputeCommandEncoder() else { return }
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
        enc.dispatchThreadgroups(MTLSize(width: entryCount, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeLatentAttention(commandBuffer: MTLCommandBuffer,
                               qLatent: MTLBuffer, latents: MTLBuffer, selected: MTLBuffer,
                               out: MTLBuffer, heads: Int, latentDim: Int,
                               cachedRows: Int, selectedCount: UInt32, scale: Float) {
        precondition(latentDim % 32 == 0 && latentDim <= Self.maxLatentDim)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(latentAttentionPSO)
        enc.setBuffer(qLatent, offset: 0, index: 0)
        enc.setBuffer(latents, offset: 0, index: 1)
        enc.setBuffer(selected, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        var kv = UInt32(latentDim)
        var total = UInt32(cachedRows)
        var sc = selectedCount
        var s = scale
        enc.setBytes(&kv, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&total, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&sc, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&s, length: MemoryLayout<Float>.size, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: heads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `y[h] = W[h] x[h]` for `heads` INT8 `[m, n]` slabs stored `[heads, m, n]`.
    func encodeHeadedInt8GEMV(commandBuffer: MTLCommandBuffer,
                              weights: TensorView, x: MTLBuffer, y: MTLBuffer,
                              heads: Int, m: Int, n: Int) {
        precondition(n % Quantization.groupSize == 0)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(headedInt8PSO)
        enc.setBuffer(weights.buffer, offset: Int(weights.offset), index: 0)
        enc.setBuffer(weights.buffer, offset: Int(weights.scaleOffset), index: 1)
        enc.setBuffer(weights.buffer, offset: Int(weights.biasOffset), index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        var mm = UInt32(m)
        var nn = UInt32(n)
        enc.setBytes(&mm, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nn, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: (m + 7) / 8, height: heads, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - Hyper-connections (DeepSeek V4 kernels, reused)

    /// `fn` is fp32 `[(2 + mult) * mult, mult * hidden]`, `base` fp32, `scale` fp32 `[3]`.
    func encodeHCWeights(commandBuffer: MTLCommandBuffer, streams: MTLBuffer,
                         fn: TensorView, base: TensorView, scale: TensorView,
                         outPre: MTLBuffer, outPost: MTLBuffer, outComb: MTLBuffer,
                         hcMult: Int, hidden: Int, sinkhornIters: Int, hcEps: Float, rmsEps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcWeightsPSO)
        enc.setBuffer(streams, offset: 0, index: 0)
        enc.setBuffer(fn.buffer, offset: Int(fn.offset), index: 1)
        enc.setBuffer(base.buffer, offset: Int(base.offset), index: 2)
        enc.setBuffer(scale.buffer, offset: Int(scale.offset), index: 3)
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
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `x[d] = sum_j pre[j] * streams[j][d]`; with `pre` = 1/mult this is the
    /// family's final stream mean.
    func encodeHCCollapse(commandBuffer: MTLCommandBuffer, streams: MTLBuffer, pre: MTLBuffer,
                          x: MTLBuffer, hcMult: Int, hidden: Int) {
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

    /// `out[k][d] = post[k] * sub[d] + sum_j comb[j][k] * streams[j][d]`.
    func encodeHCPlaceMix(commandBuffer: MTLCommandBuffer, streams: MTLBuffer, sub: MTLBuffer,
                          post: MTLBuffer, comb: MTLBuffer, outStreams: MTLBuffer,
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

    // MARK: - FFN and embedding broadcast

    /// `y = silu(min(g, limit)) * clamp(u, -limit, limit)`.
    func encodeSwigluClampMul(commandBuffer: MTLCommandBuffer, gate: MTLBuffer, up: MTLBuffer,
                              out: MTLBuffer, n: Int, limit: Float) {
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

    func encodeBroadcastStreams(commandBuffer: MTLCommandBuffer, x: MTLBuffer, streams: MTLBuffer,
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
