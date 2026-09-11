import Foundation
import Metal

/// The batched (chunked) prefill for `Glm53ForwardRunner`: a chunk of up to
/// `capacity` prompt tokens walks the layers with every weight read once per
/// chunk (`glm53_prefill.metal`, the grouped expert kernels in `moe.metal`).
///
/// Valid while the chunk ends at or below `index_topk`: there the pooled
/// indexer's selection is exhaustive, so the sparse layers attend densely and
/// causally with no CPU decision in the loop. The one readback per MoE layer
/// is the router's chunk of indices, which the CPU groups by expert.
///
/// State handoff is the decode path's: conv tails (GDN tail update), the KDA
/// state (written back by `glm53p_kda_chunk`), latent and indexer caches at
/// their absolute rows, pooled keys for every pool the chunk completes. The
/// GEMMs accumulate in a different order than the decode GEMVs, so the
/// chunk's logits are FP16-close to sequential decode rather than identical
/// (`Glm53ForwardRunnerTests` measures the gap; the per-token path stays the
/// exactness reference and serves everything past `index_topk`).
final class Glm53PrefillEngine {
    static let capacity = 128

    private unowned let r: Glm53ForwardRunner
    private let k: Glm53PrefillKernels
    private let gdn: GDN
    private let prefillNorm: PrefillRMSNorm
    private let poolGates: [TensorView?]
    private let routers: [TensorView?]

    // Scratch, sized to `capacity` tokens (fp16 unless noted).
    private let tokenIDs: MTLBuffer
    private let xC: MTLBuffer
    private let streamsA: MTLBuffer
    private let streamsB: MTLBuffer
    private let normedC: MTLBuffer
    private let hcPartials: MTLBuffer         // fp32 [C][25]
    private let preA: MTLBuffer, postA: MTLBuffer, combA: MTLBuffer   // fp32
    private let preF: MTLBuffer, postF: MTLBuffer, combF: MTLBuffer
    private let mixedC: MTLBuffer
    private let convOutC: MTLBuffer
    private let aC: MTLBuffer, bC: MTLBuffer, gateC: MTLBuffer, lowC: MTLBuffer
    private let attnHeadsC: MTLBuffer
    private let attnOutC: MTLBuffer
    private let qrC: MTLBuffer, qC: MTLBuffer, qLatC: MTLBuffer, oLatC: MTLBuffer
    private let idxKRawC: MTLBuffer
    private let routerLogitsC: MTLBuffer      // fp32 [C][E]
    private let routerIdxC: MTLBuffer         // u32 [C][K]
    private let routerWC: MTLBuffer           // fp16 [C][K]
    private let pairToken: MTLBuffer          // u32 [C*K]
    private let routePair: MTLBuffer          // u32 [C*K]
    private let segStart: MTLBuffer           // u32 [E+1]
    private let activeExperts: MTLBuffer      // u32 [E]
    private let actsC: MTLBuffer              // [C*K][moeF]
    private let partialC: MTLBuffer           // fp32 [C*K][hidden]
    private let ffnGateC: MTLBuffer, ffnUpC: MTLBuffer, ffnActC: MTLBuffer
    private let sharedOutC: MTLBuffer
    private let mlpOutC: MTLBuffer

    init(runner: Glm53ForwardRunner) throws {
        r = runner
        k = try Glm53PrefillKernels(context: runner.ctx, swigluLimit: Float(runner.cfg.swigluLimit))
        gdn = try GDN(context: runner.ctx, config: runner.cfg.linearAttention)
        prefillNorm = try PrefillRMSNorm(context: runner.ctx)
        var gates: [TensorView?] = [], routers: [TensorView?] = []
        for L in 0..<runner.cfg.numLayers {
            gates.append(runner.cfg.layerIsKDA(L) ? nil : try runner.model.glm53IndexerPoolGate(layer: L))
            routers.append(runner.cfg.layerIsDenseFFN(L) ? nil : try runner.model.router(layer: L))
        }
        poolGates = gates
        self.routers = routers

        let C = Self.capacity
        let device = runner.ctx.device
        func buf(_ bytes: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(16, bytes), options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            return b
        }
        let h = runner.hidden, hc = runner.hc, qkv = runner.kdaHeads * runner.kdaDim, K = runner.topK
        tokenIDs = try buf(C * 4)
        xC = try buf(C * h * 2)
        streamsA = try buf(C * hc * h * 2)
        streamsB = try buf(C * hc * h * 2)
        normedC = try buf(C * h * 2)
        hcPartials = try buf(C * 25 * 4)
        preA = try buf(C * hc * 4); postA = try buf(C * hc * 4); combA = try buf(C * hc * hc * 4)
        preF = try buf(C * hc * 4); postF = try buf(C * hc * 4); combF = try buf(C * hc * hc * 4)
        mixedC = try buf(C * 3 * qkv * 2)
        convOutC = try buf(C * 3 * qkv * 2)
        aC = try buf(C * qkv * 2); bC = try buf(C * runner.kdaHeads * 2)
        gateC = try buf(C * qkv * 2); lowC = try buf(C * runner.kdaDim * 2)
        attnHeadsC = try buf(C * max(qkv, runner.numHeads * runner.vDim) * 2)
        attnOutC = try buf(C * h * 2)
        qrC = try buf(C * runner.qRank * 2)
        qC = try buf(C * runner.numHeads * runner.qkDim * 2)
        qLatC = try buf(C * runner.numHeads * runner.kvRank * 2)
        oLatC = try buf(C * runner.numHeads * runner.kvRank * 2)
        idxKRawC = try buf(C * runner.idxDim * 2)
        routerLogitsC = try buf(C * runner.numExperts * 4)
        routerIdxC = try buf(C * K * 4)
        routerWC = try buf(C * K * 2)
        pairToken = try buf(C * K * 4)
        routePair = try buf(C * K * 4)
        segStart = try buf((runner.numExperts + 1) * 4)
        activeExperts = try buf(runner.numExperts * 4)
        actsC = try buf(C * K * runner.moeF * 2)
        partialC = try buf(C * K * h * 4)
        let ffnW = max(runner.sharedF, runner.denseF)
        ffnGateC = try buf(C * ffnW * 2); ffnUpC = try buf(C * ffnW * 2); ffnActC = try buf(C * ffnW * 2)
        sharedOutC = try buf(C * h * 2)
        mlpOutC = try buf(C * h * 2)
    }

    /// Runs `tokens` (at most `capacity`) at positions `p0...`; the chunk must
    /// end at or below `index_topk`. Writes the last token's logits when asked.
    func run(tokens: ArraySlice<Int32>, startPosition p0: Int, into logits: MTLBuffer?) throws {
        let T = tokens.count
        precondition(T > 0 && T <= Self.capacity)
        precondition(p0 + T <= r.idxTopK, "batched prefill is only the model below index_topk")
        let ids = tokenIDs.contents().bindMemory(to: UInt32.self, capacity: T)
        for (i, t) in tokens.enumerated() { ids[i] = UInt32(t) }
        let h = r.hidden, hc = r.hc, eps = r.eps

        var cb = try r.open()
        k.encodeEmbed(commandBuffer: cb, table: r.embedding, tokens: tokenIDs, out: xC, d: h, count: T)
        k.encodeBroadcast(commandBuffer: cb, x: xC, streams: streamsA, hcMult: hc, hidden: h, tokens: T)

        for L in 0..<r.cfg.numLayers {
            try Task.checkCancellation()
            let layer = r.layers[L]
            cb = try r.open()
            // Attention site.
            k.encodeHCWeights(commandBuffer: cb, streams: streamsA, fn: layer.hcAttnFn, base: layer.hcAttnBase,
                              scale: layer.hcAttnScale, partials: hcPartials, outPre: preA, outPost: postA,
                              outComb: combA, hcMult: hc, hidden: h, sinkhornIters: r.cfg.hyperConnections.sinkhornIters,
                              hcEps: r.hcEps, rmsEps: eps, tokens: T)
            k.encodeHCCollapse(commandBuffer: cb, streams: streamsA, pre: preA, x: xC, hcMult: hc, hidden: h, tokens: T)
            prefillNorm.encodeBF16W(commandBuffer: cb, x: xC, weight: layer.attnNorm.buffer,
                                    weightOffset: Int(layer.attnNorm.offset), out: normedC,
                                    t: UInt32(T), d: UInt32(h), eps: eps)
            if r.cfg.layerIsKDA(L) {
                try encodeKDA(cb, layer: layer, index: L, tokens: T)
            } else {
                try encodeSparse(cb, layer: layer, index: L, p0: p0, tokens: T)
            }
            k.encodeHCPlaceMix(commandBuffer: cb, streams: streamsA, sub: attnOutC, post: postA, comb: combA,
                               outStreams: streamsB, hcMult: hc, hidden: h, tokens: T)
            // FFN site.
            k.encodeHCWeights(commandBuffer: cb, streams: streamsB, fn: layer.hcFFNFn, base: layer.hcFFNBase,
                              scale: layer.hcFFNScale, partials: hcPartials, outPre: preF, outPost: postF,
                              outComb: combF, hcMult: hc, hidden: h, sinkhornIters: r.cfg.hyperConnections.sinkhornIters,
                              hcEps: r.hcEps, rmsEps: eps, tokens: T)
            k.encodeHCCollapse(commandBuffer: cb, streams: streamsB, pre: preF, x: xC, hcMult: hc, hidden: h, tokens: T)
            prefillNorm.encodeBF16W(commandBuffer: cb, x: xC, weight: layer.ffnNorm.buffer,
                                    weightOffset: Int(layer.ffnNorm.offset), out: normedC,
                                    t: UInt32(T), d: UInt32(h), eps: eps)
            if r.cfg.layerIsDenseFFN(L) {
                try encodeDense(cb, layer: layer, index: L, tokens: T)
            } else {
                cb = try encodeMoE(cb, layer: layer, index: L, tokens: T)
            }
            k.encodeHCPlaceMix(commandBuffer: cb, streams: streamsB, sub: mlpOutC, post: postF, comb: combF,
                               outStreams: streamsA, hcMult: hc, hidden: h, tokens: T)
        }

        // The last token's logits through the decode head kernels.
        if let logits {
            cb = try r.open()
            let lastOffset = (T - 1) * hc * h * MemoryLayout<Float16>.stride
            k.encodeHCCollapse(commandBuffer: cb, streams: streamsA, streamsOffset: lastOffset, pre: r.meanPre,
                               x: r.hiddenBuf, hcMult: hc, hidden: h, tokens: 1)
            r.rms.encodeBF16W(commandBuffer: cb, x: r.hiddenBuf, weight: r.finalNorm.buffer,
                              weightOffset: Int(r.finalNorm.offset), out: r.normed, d: UInt32(h), eps: eps)
            r.gemvInt8(cb, r.lmHead, x: r.normed, y: logits, m: r.cfg.vocabSize, n: h)
            try r.sync()
        } else {
            try r.flush()
        }
    }

    private func encodeKDA(_ cb: MTLCommandBuffer, layer: Glm53ForwardRunner.LayerTensors, index L: Int,
                           tokens T: Int) throws {
        guard let qP = layer.qProj, let kP = layer.kProj, let vP = layer.vProj, let conv = layer.conv,
              let fA = layer.fA, let fB = layer.fB, let gA = layer.gA, let gB = layer.gB, let bP = layer.bProj,
              let aLog = layer.aLog, let dtBias = layer.dtBias, let oNorm = layer.oNorm,
              let tail = r.state.convTail[L], let st = r.state.kdaState[L] else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) is not a KDA layer")
        }
        let h = r.hidden, qkv = r.kdaHeads * r.kdaDim, rowBytes = qkv * 2
        k.encodeInt8GEMM(commandBuffer: cb, weights: qP, x: normedC, xStride: h, y: mixedC, yStride: 3 * qkv,
                         m: qkv, n: h, tokens: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: kP, x: normedC, xStride: h, y: mixedC, yOffset: rowBytes,
                         yStride: 3 * qkv, m: qkv, n: h, tokens: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: vP, x: normedC, xStride: h, y: mixedC, yOffset: 2 * rowBytes,
                         yStride: 3 * qkv, m: qkv, n: h, tokens: T)
        gdn.encodeConvPrefill(commandBuffer: cb, tail: tail, qkvRows: mixedC, convWeight: conv.buffer,
                              convWeightOffset: Int(conv.offset), out: convOutC, rows: T)
        gdn.encodeConvTailUpdate(commandBuffer: cb, tail: tail, qkvRows: mixedC, rows: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: fA, x: normedC, xStride: h, y: lowC, yStride: r.kdaDim,
                         m: r.kdaDim, n: h, tokens: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: fB, x: lowC, xStride: r.kdaDim, y: aC, yStride: qkv,
                         m: qkv, n: r.kdaDim, tokens: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: gA, x: normedC, xStride: h, y: lowC, yStride: r.kdaDim,
                         m: r.kdaDim, n: h, tokens: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: gB, x: lowC, xStride: r.kdaDim, y: gateC, yStride: qkv,
                         m: qkv, n: r.kdaDim, tokens: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: bP, x: normedC, xStride: h, y: bC, yStride: r.kdaHeads,
                         m: r.kdaHeads, n: h, tokens: T)
        k.encodeKDAChunk(commandBuffer: cb, convOut: convOutC, a: aC, b: bC, gate: gateC, aLog: aLog, dtBias: dtBias,
                         oNorm: oNorm, state: st, out: attnHeadsC, heads: r.kdaHeads, headDim: r.kdaDim, tokens: T,
                         lowerBound: Float(r.g53.kdaGateLowerBound), eps: r.eps)
        k.encodeInt8GEMM(commandBuffer: cb, weights: layer.oProj, x: attnHeadsC, xStride: qkv, y: attnOutC, yStride: h,
                         m: h, n: qkv, tokens: T)
    }

    private func encodeSparse(_ cb: MTLCommandBuffer, layer: Glm53ForwardRunner.LayerTensors, index L: Int,
                              p0: Int, tokens T: Int) throws {
        guard let qA = layer.qA, let qANorm = layer.qANorm, let qB = layer.qB, let kvA = layer.kvA,
              let kvANorm = layer.kvANorm, let embedQ = layer.embedQ, let unembed = layer.unembedOut,
              let wk = layer.idxK, let kw = layer.idxKNormWeight, let kb = layer.idxKNormBias, let ape = layer.idxApe,
              let poolGate = poolGates[L], let latents = r.state.latents[L], let keys = r.state.indexKeys[L],
              let gates = r.state.indexGates[L], let pooled = r.state.pooledKeys[L] else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) is not a sparse layer")
        }
        let h = r.hidden, heads = r.numHeads, kvRank = r.kvRank, qRank = r.qRank
        k.encodeInt8GEMM(commandBuffer: cb, weights: qA, x: normedC, xStride: h, y: qrC, yStride: qRank,
                         m: qRank, n: h, tokens: T)
        prefillNorm.encodeBF16W(commandBuffer: cb, x: qrC, weight: qANorm.buffer, weightOffset: Int(qANorm.offset),
                                out: qrC, t: UInt32(T), d: UInt32(qRank), eps: r.eps)
        k.encodeInt8GEMM(commandBuffer: cb, weights: qB, x: qrC, xStride: qRank, y: qC, yStride: heads * r.qkDim,
                         m: heads * r.qkDim, n: qRank, tokens: T)
        let latentOffset = p0 * kvRank * 2
        k.encodeInt8GEMM(commandBuffer: cb, weights: kvA, x: normedC, xStride: h, y: latents, yOffset: latentOffset,
                         yStride: kvRank, m: kvRank, n: h, tokens: T)
        prefillNorm.encodeBF16W(commandBuffer: cb, x: latents, xOffset: latentOffset, weight: kvANorm.buffer,
                                weightOffset: Int(kvANorm.offset), out: latents, outOffset: latentOffset,
                                t: UInt32(T), d: UInt32(kvRank), eps: r.eps)
        // Indexer bookkeeping (no scoring: the chunk stays below index_topk).
        let rowOffset = p0 * r.idxDim * 2
        k.encodeInt8GEMM(commandBuffer: cb, weights: wk, x: normedC, xStride: h, y: idxKRawC, yStride: r.idxDim,
                         m: r.idxDim, n: h, tokens: T)
        k.encodeLayerNormBias(commandBuffer: cb, x: idxKRawC, weight: kw, bias: kb, out: keys, outOffset: rowOffset,
                              d: r.idxDim, eps: Float(r.g53.indexerKNormEps), tokens: T)
        k.encodeBF16GEMM(commandBuffer: cb, weights: poolGate, x: normedC, xStride: h, y: gates, yOffset: rowOffset,
                         yStride: r.idxDim, outputFloat32: false, m: r.idxDim, n: h, tokens: T)
        let firstPool = p0 / r.kPool
        let lastPool = (p0 + T) / r.kPool - 1
        if lastPool >= firstPool {
            k.encodePoolKeys(commandBuffer: cb, keys: keys, gates: gates, ape: ape, pooled: pooled,
                             firstPool: firstPool, count: lastPool - firstPool + 1, kPool: r.kPool, dim: r.idxDim)
        }
        k.encodeHeadedGEMV(commandBuffer: cb, weights: embedQ, x: qC, y: qLatC, heads: heads, m: kvRank, n: r.qkDim,
                           tokens: T)
        k.encodeLatentAttentionCausal(commandBuffer: cb, qLatent: qLatC, latents: latents, out: oLatC, heads: heads,
                                      latentDim: kvRank, base: p0, tokens: T, scale: Float(r.cfg.attentionScale))
        k.encodeHeadedGEMV(commandBuffer: cb, weights: unembed, x: oLatC, y: attnHeadsC, heads: heads, m: r.vDim,
                           n: kvRank, tokens: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: layer.oProj, x: attnHeadsC, xStride: heads * r.vDim, y: attnOutC,
                         yStride: h, m: h, n: heads * r.vDim, tokens: T)
    }

    private func encodeDense(_ cb: MTLCommandBuffer, layer: Glm53ForwardRunner.LayerTensors, index L: Int,
                             tokens T: Int) throws {
        guard let g = layer.denseGate, let u = layer.denseUp, let d = layer.denseDown else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) has no dense FFN")
        }
        let h = r.hidden, f = r.denseF
        k.encodeInt8GEMM(commandBuffer: cb, weights: g, x: normedC, xStride: h, y: ffnGateC, yStride: f, m: f, n: h, tokens: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: u, x: normedC, xStride: h, y: ffnUpC, yStride: f, m: f, n: h, tokens: T)
        r.kernels.encodeSwigluClampMul(commandBuffer: cb, gate: ffnGateC, up: ffnUpC, out: ffnActC, n: T * f,
                                       limit: Float(r.cfg.swigluLimit))
        k.encodeInt8GEMM(commandBuffer: cb, weights: d, x: ffnActC, xStride: f, y: mlpOutC, yStride: h, m: h, n: f, tokens: T)
    }

    /// Router and shared expert on the GPU, one readback of the chunk's
    /// routes, the CPU grouping by expert, then the grouped kernels. Returns
    /// the stream that continues after the sync.
    private func encodeMoE(_ cbIn: MTLCommandBuffer, layer: Glm53ForwardRunner.LayerTensors, index L: Int,
                           tokens T: Int) throws -> MTLCommandBuffer {
        guard let router = routers[L], let bias = layer.routerBias, let sg = layer.sharedGate, let su = layer.sharedUp,
              let sd = layer.sharedDown, let offsets = layer.expertOffsets, let slab = layer.slab else {
            throw Glm53ForwardRunnerError.invalidConfiguration("layer \(L) has no resident MoE block")
        }
        let h = r.hidden, f = r.sharedF, E = r.numExperts, K = r.topK
        var cb = cbIn
        k.encodeBF16GEMM(commandBuffer: cb, weights: router, x: normedC, xStride: h, y: routerLogitsC, yStride: E,
                         outputFloat32: true, m: E, n: h, tokens: T)
        k.encodeRouterSelect(commandBuffer: cb, logits: routerLogitsC, bias: bias, outIndices: routerIdxC,
                             outWeights: routerWC, numExperts: E, routeScale: Float(r.cfg.routedScalingFactor), tokens: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: sg, x: normedC, xStride: h, y: ffnGateC, yStride: f, m: f, n: h, tokens: T)
        k.encodeInt8GEMM(commandBuffer: cb, weights: su, x: normedC, xStride: h, y: ffnUpC, yStride: f, m: f, n: h, tokens: T)
        r.kernels.encodeSwigluClampMul(commandBuffer: cb, gate: ffnGateC, up: ffnUpC, out: ffnActC, n: T * f,
                                       limit: Float(r.cfg.swigluLimit))
        k.encodeInt8GEMM(commandBuffer: cb, weights: sd, x: ffnActC, xStride: f, y: sharedOutC, yStride: h, m: h, n: f, tokens: T)
        try r.sync()

        // Group the chunk's routes by expert.
        let idx = routerIdxC.contents().bindMemory(to: UInt32.self, capacity: T * K)
        var perExpert = [[Int]](repeating: [], count: E)
        for t in 0..<T {
            for kk in 0..<K {
                let e = min(Int(idx[t * K + kk]), E - 1)
                perExpert[e].append(t * K + kk)
            }
        }
        let pairTokenPtr = pairToken.contents().bindMemory(to: UInt32.self, capacity: T * K)
        let routePairPtr = routePair.contents().bindMemory(to: UInt32.self, capacity: T * K)
        let segPtr = segStart.contents().bindMemory(to: UInt32.self, capacity: E + 1)
        let activePtr = activeExperts.contents().bindMemory(to: UInt32.self, capacity: E)
        var pair = 0, active = 0
        segPtr[0] = 0
        for e in 0..<E where !perExpert[e].isEmpty {
            activePtr[active] = UInt32(e)
            for route in perExpert[e] {
                pairTokenPtr[pair] = UInt32(route / K)
                routePairPtr[route] = UInt32(pair)
                pair += 1
            }
            active += 1
            segPtr[active] = UInt32(pair)
        }

        cb = try r.open()
        k.encodeGroupedMoE(commandBuffer: cb, slab: slab, offsets: offsets, x: normedC, acts: actsC, partial: partialC,
                           pairToken: pairToken, segStart: segStart, activeExperts: activeExperts, activeCount: active,
                           routePair: routePair, weights: routerWC, residual: sharedOutC, y: mlpOutC,
                           d: h, f: r.moeF, topK: K, tokens: T)
        return cb
    }
}
