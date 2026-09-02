import Foundation
@testable import Mference

/// Float32 CPU reference forward for `qwen38flashnext` (upstream `qwen4_exp`).
///
/// Correctness only: straight-line scalar arithmetic, no Metal, no kernels, no
/// production runner. Its job is to prove that the *semantics* in
/// `docs/superpowers/specs/2026-09-01-qwen38flashnext-runtime-design.md` are
/// reproducible from a real install, against the committed reference goldens —
/// the port's stop/go parity milestone.
///
/// Every weight comes out of a loaded `Model` (see `FlashNextWeights`); nothing
/// is read from the source checkpoint. Both a full-prompt prefill and cached
/// single-token decode run through the same `step`, so `stepCount` sequential
/// one-token calls are byte-comparable against one `T`-token call.
///
/// Where this file's arithmetic is unobvious it names the reference expression
/// it transcribes; the transcription was validated module by module against
/// `Tests/Mference/Fixtures/qwen4exp-bf16/`.
final class FlashNextReferenceRunner {

    // MARK: - Capture

    /// Per-forward record of everything the goldens pin.
    struct Capture {
        var floats: [String: [Float]] = [:]
        var integers: [String: [[Int]]] = [:]
        /// `(layer, absolute query position)` where the indexer's block score at
        /// rank `k` equals the score at rank `k+1` bit-for-bit, so the selected
        /// set is decided by the sorting algorithm rather than by the model.
        var indexerBoundaryTies: [(layer: Int, query: Int, score: Float)] = []
    }

    // MARK: - Caches

    /// Decode state for one layer. Which fields are live depends on the layer
    /// type; the PLE fields exist only on the PLE layer.
    private final class LayerCache {
        // Gated DeltaNet.
        var convTail: [Float] = []       // [(K-1) * qkvDim]
        var recurrent: [Float] = []      // [numVHeads * keyHeadDim * valueHeadDim]
        // Full attention.
        var keys: [Float] = []           // [L * numKVHeads * headDim]
        var values: [Float] = []         // [L * numKVHeads * headDim]
        var indexerKeys: [Float] = []    // [L * indexerHeadDim], RAW (un-normed, un-roped)
        // PLE.
        var pleConv: [Float] = []        // [pleStateLength * bundle]
        var pleHistory: [Int] = []       // last (ngramSize - 1) token ids
    }

    // MARK: - Geometry

    private let weights: FlashNextWeights
    private let config: ArchConfig

    private let hidden: Int
    private let hcCount: Int
    private let bundle: Int
    private let lowRank: Int
    private let eps: Float

    private let numHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let rotaryDim: Int
    private let ropeTheta: Float

    private let numKHeads: Int
    private let numVHeads: Int
    private let keyHeadDim: Int
    private let valueHeadDim: Int
    private let convKernel: Int
    private let keyDim: Int
    private let valueDim: Int
    private let qkvDim: Int

    private let numExperts: Int
    private let topK: Int
    private let moeIntermediate: Int
    private let sharedIntermediate: Int

    private let indexerHeads: Int
    private let indexerKVHeads: Int
    private let indexerHeadDim: Int
    private let compressRatio: Int
    private let blockTopK: Int

    private let pleLayer: Int
    private let ngramSize: Int
    private let ngramHeads: Int
    private let pleConvKernel: Int
    private let pleDilation: Int
    private let pleStateLength: Int
    private let eosTokenID: Int

    private var caches: [LayerCache] = []
    /// Tokens already consumed; the absolute position of the next one.
    private(set) var seen: Int = 0

    // Cached weight bundles.
    private var attnHC: [Int: FlashNextWeights.GatedResidualWeights] = [:]
    private var mlpHC: [Int: FlashNextWeights.GatedResidualWeights] = [:]
    private var attn: [Int: FlashNextWeights.AttentionWeights] = [:]
    private var gdnW: [Int: FlashNextWeights.GDNWeights] = [:]
    private var moeW: [Int: FlashNextWeights.MoEWeights] = [:]
    private var pleW: FlashNextWeights.PLEWeights?
    private var mixer: FlashNextWeights.GatedResidualWeights!
    private var embedding: [Float] = []
    private var lmHead: [Float] = []
    /// `[row]` of the n-gram table, cached after the first read through the pool.
    private var ngramRows: [Int: [Float]] = [:]

    // MARK: - Init

    init(weights: FlashNextWeights) throws {
        self.weights = weights
        let c = weights.config
        config = c
        hidden = c.hiddenSize
        hcCount = c.flashNext.hcCount
        bundle = hcCount * hidden
        lowRank = c.flashNext.hcLowRank
        // `rms_norm_eps` is not carried on ArchConfig; every Qwen4-Exp norm uses
        // the config default, and the goldens' toy config states it explicitly.
        eps = 1e-6

        numHeads = c.numHeads
        numKVHeads = c.numFullKVHeads
        headDim = c.fullHeadDim
        rotaryDim = Int(Double(c.fullHeadDim) * c.partialRotaryFactor)
        ropeTheta = Float(c.fullRopeTheta)

        let la = c.linearAttention
        numKHeads = la.numKHeads
        numVHeads = la.numVHeads
        keyHeadDim = la.keyHeadDim
        valueHeadDim = la.valueHeadDim
        convKernel = la.convKernelSize
        keyDim = la.numKHeads * la.keyHeadDim
        valueDim = la.numVHeads * la.valueHeadDim
        qkvDim = 2 * keyDim + valueDim

        numExperts = c.numExperts
        topK = c.topKExperts
        moeIntermediate = c.moeIntermediateSize
        sharedIntermediate = c.intermediateSize

        indexerHeads = c.flashNext.indexerNumHeads
        indexerKVHeads = c.flashNext.indexerNumKVHeads
        indexerHeadDim = c.flashNext.indexerHeadDim
        compressRatio = c.flashNext.indexerCompressRatio
        blockTopK = c.flashNext.indexerBlockBudget

        pleLayer = c.flashNext.pleLayerIndices.first ?? -1
        pleConvKernel = c.flashNext.pleConvKernelSize
        eosTokenID = c.flashNext.pleEosTokenID

        embedding = try weights.embedding()
        lmHead = try weights.lmHead()
        mixer = try weights.globalMixer()
        for L in 0..<c.numLayers {
            attnHC[L] = try weights.hyperConnection(site: .attention, layer: L)
            mlpHC[L] = try weights.hyperConnection(site: .mlp, layer: L)
            moeW[L] = try weights.moe(layer: L)
            if c.layerIsLinear(L) {
                gdnW[L] = try weights.gdn(layer: L)
            } else {
                attn[L] = try weights.attention(layer: L)
            }
        }
        if pleLayer >= 0 {
            let p = try weights.ple(layer: pleLayer)
            pleW = p
            ngramSize = p.multipliers.count
            ngramHeads = p.headOffsets.count
            pleDilation = ngramSize
            pleStateLength = (pleConvKernel - 1) * pleDilation
        } else {
            ngramSize = 0; ngramHeads = 0; pleDilation = 0; pleStateLength = 0
        }
        reset()
    }

    /// Drop every cache and rewind to position zero.
    func reset() {
        seen = 0
        ngramRows.removeAll()
        caches = (0..<config.numLayers).map { L in
            let cache = LayerCache()
            if config.layerIsLinear(L) {
                cache.convTail = [Float](repeating: 0, count: (convKernel - 1) * qkvDim)
                cache.recurrent = [Float](
                    repeating: 0, count: numVHeads * keyHeadDim * valueHeadDim)
            }
            if L == pleLayer {
                cache.pleConv = [Float](repeating: 0, count: pleStateLength * bundle)
                cache.pleHistory = [Int](repeating: eosTokenID, count: max(0, ngramSize - 1))
            }
            return cache
        }
    }

    // MARK: - Forward

    /// Run `tokens` through the model, advancing every cache. Returns the logits
    /// for each position, `[tokens.count * vocabSize]`.
    @discardableResult
    func step(tokens: [Int], capture: inout Capture?) throws -> [Float] {
        let S = tokens.count
        precondition(S > 0)
        let total = seen + S
        let (cos, sin) = ropeTable(upTo: total)

        // embed -> tile across the hc streams. `repeat(1, 1, hc_count)` is a
        // TILE: stream j is an exact copy of the embedding, not an interleave.
        var embed = [Float](repeating: 0, count: S * hidden)
        for t in 0..<S {
            let row = tokens[t] * hidden
            for d in 0..<hidden { embed[t * hidden + d] = embedding[row + d] }
        }
        capture?.floats["embed_out"] = embed
        var hyper = [Float](repeating: 0, count: S * bundle)
        for t in 0..<S {
            for j in 0..<hcCount {
                for d in 0..<hidden {
                    hyper[t * bundle + j * hidden + d] = embed[t * hidden + d]
                }
            }
        }

        for L in 0..<config.numLayers {
            let key = String(format: "layer%02d.", L)
            let cache = caches[L]

            if L == pleLayer {
                let out = try pleBlock(hyper: hyper, tokens: tokens, cache: cache,
                                       capture: &capture, key: key)
                for i in 0..<hyper.count { hyper[i] += out[i] }
            }
            capture?.floats[key + "attn_hc_stream_in"] = hyper

            let attnMix = gatedResidual(hyper, attnHC[L]!, rows: S)
            capture?.floats[key + "attn_hc_mixed"] = attnMix.mixed
            capture?.floats[key + "attn_hc_inject"] = attnMix.inject!

            let blockOut: [Float]
            if config.layerIsLinear(L) {
                blockOut = gatedDeltaNet(attnMix.mixed, gdnW[L]!, rows: S, cache: cache)
            } else {
                let w = attn[L]!
                let selected = indexerSelect(attnMix.mixed, w, rows: S, cache: cache,
                                             cos: cos, sin: sin, layer: L,
                                             capture: &capture)
                capture?.integers[key + "indexer_selected"] = selected
                capture?.integers[key + "indexer_visible"] =
                    (0..<S).map { Array(0...(seen + $0)) }
                blockOut = attention(attnMix.mixed, w, rows: S, cache: cache,
                                     cos: cos, sin: sin, selected: selected)
            }
            capture?.floats[key + "block_out"] = blockOut
            hyper = injectBlock(attnMix.raw, blockOut, attnMix.inject!, rows: S)
            capture?.floats[key + "mlp_hc_stream_in"] = hyper

            let mlpMix = gatedResidual(hyper, mlpHC[L]!, rows: S)
            capture?.floats[key + "mlp_hc_mixed"] = mlpMix.mixed
            capture?.floats[key + "mlp_hc_inject"] = mlpMix.inject!

            let moe = try sparseMoE(mlpMix.mixed, moeW[L]!, layer: L, rows: S)
            capture?.floats[key + "moe_out"] = moe.out
            capture?.floats[key + "router_weights"] = moe.weights
            capture?.integers[key + "router_indices"] = moe.indices
            hyper = injectBlock(mlpMix.raw, moe.out, mlpMix.inject!, rows: S)
            capture?.floats[key + "stream_out"] = hyper
        }

        // No final norm: the global mixer stands in for it.
        let last = gatedResidual(hyper, mixer, rows: S).mixed
        capture?.floats["last_hidden_state"] = last
        var logits = [Float](repeating: 0, count: S * config.vocabSize)
        for t in 0..<S {
            matVec(lmHead, rows: config.vocabSize, cols: hidden,
                   x: last, xOffset: t * hidden,
                   into: &logits, outOffset: t * config.vocabSize)
        }
        capture?.floats["logits"] = logits
        seen += S
        return logits
    }

    /// Greedy argmax over the final position's logits.
    func greedyToken(_ logits: [Float]) -> Int {
        let v = config.vocabSize
        let base = logits.count - v
        var best = 0
        for i in 1..<v where logits[base + i] > logits[base + best] { best = i }
        return best
    }

    // MARK: - Hyper connections

    private struct Mix {
        let mixed: [Float]      // [rows * hidden]
        let raw: [Float]        // [rows * bundle] — the UN-normed stream
        let inject: [Float]?    // [rows * hcCount]
    }

    /// `Qwen4ExpTextGatedResidual`. Three separate `/hc_count` divisions: before
    /// the silu, in the mean over streams, and before the injection sigmoid.
    /// The residual re-add uses the raw stream, not the normed one.
    private func gatedResidual(_ hyper: [Float],
                               _ w: FlashNextWeights.GatedResidualWeights,
                               rows S: Int) -> Mix {
        var mixed = [Float](repeating: 0, count: S * hidden)
        var inject = w.inject == nil
            ? nil : [Float](repeating: 0, count: S * hcCount)
        var down = [Float](repeating: 0, count: lowRank)
        var up = [Float](repeating: 0, count: bundle)
        var injRow = [Float](repeating: 0, count: hcCount)
        for t in 0..<S {
            var normed = [Float](repeating: 0, count: bundle)
            groupRMSNorm(hyper, offset: t * bundle, weight: w.norm, into: &normed)
            matVec(w.mixDown, rows: lowRank, cols: bundle, x: normed, xOffset: 0,
                   into: &down, outOffset: 0)
            for i in 0..<lowRank { down[i] = silu(down[i] / Float(hcCount)) }
            matVec(w.mixUp, rows: bundle, cols: lowRank, x: down, xOffset: 0,
                   into: &up, outOffset: 0)
            for d in 0..<hidden {
                var acc: Float = 0
                for j in 0..<hcCount {
                    let i = j * hidden + d
                    acc += sigmoid(up[i]) * normed[i]
                }
                mixed[t * hidden + d] = acc / Float(hcCount)
            }
            if let injectWeight = w.inject {
                matVec(injectWeight, rows: hcCount, cols: bundle, x: normed, xOffset: 0,
                       into: &injRow, outOffset: 0)
                for j in 0..<hcCount {
                    inject![t * hcCount + j] = 2 * sigmoid(injRow[j] / Float(hcCount))
                }
            }
        }
        return Mix(mixed: mixed, raw: hyper, inject: inject)
    }

    /// `hyper + flatten(block_out[None, H] * inject[hc, None])`.
    private func injectBlock(_ hyper: [Float], _ block: [Float],
                             _ inject: [Float], rows S: Int) -> [Float] {
        var out = hyper
        for t in 0..<S {
            for j in 0..<hcCount {
                let g = inject[t * hcCount + j]
                let base = t * bundle + j * hidden
                for d in 0..<hidden { out[base + d] += g * block[t * hidden + d] }
            }
        }
        return out
    }

    // MARK: - MoE

    private struct MoEResult {
        let out: [Float]
        let weights: [Float]
        let indices: [[Int]]
    }

    /// `Qwen3NextSparseMoeBlock`: fp32 softmax over ALL experts, then top-k of
    /// the probs, then renormalize. The fused `gate_up` split is gate-then-up.
    private func sparseMoE(_ x: [Float], _ w: FlashNextWeights.MoEWeights,
                           layer L: Int, rows S: Int) throws -> MoEResult {
        var out = [Float](repeating: 0, count: S * hidden)
        var topWeights = [Float](repeating: 0, count: S * topK)
        var topIndices: [[Int]] = []
        var logits = [Float](repeating: 0, count: numExperts)
        var gate = [Float](repeating: 0, count: sharedIntermediate)
        var up = [Float](repeating: 0, count: sharedIntermediate)
        var shared = [Float](repeating: 0, count: hidden)
        var scalar = [Float](repeating: 0, count: 1)

        for t in 0..<S {
            let xo = t * hidden
            matVec(w.router, rows: numExperts, cols: hidden, x: x, xOffset: xo,
                   into: &logits, outOffset: 0)
            var maximum = logits[0]
            for i in 1..<numExperts where logits[i] > maximum { maximum = logits[i] }
            var probs = [Float](repeating: 0, count: numExperts)
            var sum: Float = 0
            for i in 0..<numExperts {
                probs[i] = expf(logits[i] - maximum)
                sum += probs[i]
            }
            for i in 0..<numExperts { probs[i] /= sum }
            let chosen = descendingTopK(probs, k: topK)
            var total: Float = 0
            for j in 0..<topK { total += probs[chosen[j]] }
            for j in 0..<topK { topWeights[t * topK + j] = probs[chosen[j]] / total }
            topIndices.append(chosen)

            for j in 0..<topK {
                let expert = try weights.expert(layer: L, expert: chosen[j])
                var g = [Float](repeating: 0, count: moeIntermediate)
                var u = [Float](repeating: 0, count: moeIntermediate)
                matVec(expert.gate, rows: moeIntermediate, cols: hidden,
                       x: x, xOffset: xo, into: &g, outOffset: 0)
                matVec(expert.up, rows: moeIntermediate, cols: hidden,
                       x: x, xOffset: xo, into: &u, outOffset: 0)
                for i in 0..<moeIntermediate { g[i] = silu(g[i]) * u[i] }
                var y = [Float](repeating: 0, count: hidden)
                matVec(expert.down, rows: hidden, cols: moeIntermediate,
                       x: g, xOffset: 0, into: &y, outOffset: 0)
                let scale = topWeights[t * topK + j]
                for d in 0..<hidden { out[xo + d] += scale * y[d] }
            }

            matVec(w.sharedGateProj, rows: sharedIntermediate, cols: hidden,
                   x: x, xOffset: xo, into: &gate, outOffset: 0)
            matVec(w.sharedUpProj, rows: sharedIntermediate, cols: hidden,
                   x: x, xOffset: xo, into: &up, outOffset: 0)
            for i in 0..<sharedIntermediate { gate[i] = silu(gate[i]) * up[i] }
            matVec(w.sharedDownProj, rows: hidden, cols: sharedIntermediate,
                   x: gate, xOffset: 0, into: &shared, outOffset: 0)
            matVec(w.sharedGate, rows: 1, cols: hidden, x: x, xOffset: xo,
                   into: &scalar, outOffset: 0)
            let g = sigmoid(scalar[0])
            for d in 0..<hidden { out[xo + d] += g * shared[d] }
        }
        return MoEResult(out: out, weights: topWeights, indices: topIndices)
    }

    // MARK: - Gated DeltaNet

    /// `Qwen4ExpTextGatedDeltaNet` — the `qwen3_5` class with four separate
    /// projections and a **sigmoid** gated output norm.
    ///
    /// Implements `torch_recurrent_gated_delta_rule`, which is the ground-truth
    /// semantics; the reference's chunked prefill path is a performance rewrite
    /// that agrees to float32 rounding.
    private func gatedDeltaNet(_ x: [Float], _ w: FlashNextWeights.GDNWeights,
                               rows S: Int, cache: LayerCache) -> [Float] {
        var qkv = [Float](repeating: 0, count: S * qkvDim)
        var z = [Float](repeating: 0, count: S * valueDim)
        var a = [Float](repeating: 0, count: S * numVHeads)
        var b = [Float](repeating: 0, count: S * numVHeads)
        for t in 0..<S {
            matVec(w.qkv, rows: qkvDim, cols: hidden, x: x, xOffset: t * hidden,
                   into: &qkv, outOffset: t * qkvDim)
            matVec(w.z, rows: valueDim, cols: hidden, x: x, xOffset: t * hidden,
                   into: &z, outOffset: t * valueDim)
            matVec(w.a, rows: numVHeads, cols: hidden, x: x, xOffset: t * hidden,
                   into: &a, outOffset: t * numVHeads)
            matVec(w.b, rows: numVHeads, cols: hidden, x: x, xOffset: t * hidden,
                   into: &b, outOffset: t * numVHeads)
        }

        // Causal depthwise conv over [cached tail | current], then SiLU.
        let tailRows = convKernel - 1
        var padded = cache.convTail
        padded.append(contentsOf: qkv)
        var conv = [Float](repeating: 0, count: S * qkvDim)
        for t in 0..<S {
            for c in 0..<qkvDim {
                var acc: Float = 0
                for j in 0..<convKernel {
                    acc += w.conv[c * convKernel + j] * padded[(t + j) * qkvDim + c]
                }
                conv[t * qkvDim + c] = silu(acc)
            }
        }
        // Keep the last (K - 1) rows of [tail | current] as the next tail.
        cache.convTail = Array(padded[(padded.count - tailRows * qkvDim)...])

        let rep = numVHeads / numKHeads
        let scale = 1 / sqrtf(Float(keyHeadDim))
        var y = [Float](repeating: 0, count: S * valueDim)
        for t in 0..<S {
            for h in 0..<numVHeads {
                let hk = h / rep
                // l2norm uses SUM (not mean) with eps inside the rsqrt; the
                // query is additionally divided by sqrt(head_k_dim).
                var q = [Float](repeating: 0, count: keyHeadDim)
                var k = [Float](repeating: 0, count: keyHeadDim)
                var qn: Float = 0, kn: Float = 0
                for d in 0..<keyHeadDim {
                    q[d] = conv[t * qkvDim + hk * keyHeadDim + d]
                    k[d] = conv[t * qkvDim + keyDim + hk * keyHeadDim + d]
                    qn += q[d] * q[d]
                    kn += k[d] * k[d]
                }
                let qi = 1 / sqrtf(qn + 1e-6) * scale
                let ki = 1 / sqrtf(kn + 1e-6)
                for d in 0..<keyHeadDim { q[d] *= qi; k[d] *= ki }

                let beta = sigmoid(b[t * numVHeads + h])
                let decay = expf(-expf(w.aLog[h])
                                 * softplus(a[t * numVHeads + h] + w.dtBias[h]))
                let stateBase = h * keyHeadDim * valueHeadDim
                let vBase = t * qkvDim + 2 * keyDim + h * valueHeadDim
                var memory = [Float](repeating: 0, count: valueHeadDim)
                for dk in 0..<keyHeadDim {
                    let row = stateBase + dk * valueHeadDim
                    let kk = k[dk]
                    for dv in 0..<valueHeadDim {
                        cache.recurrent[row + dv] *= decay
                        memory[dv] += cache.recurrent[row + dv] * kk
                    }
                }
                var delta = [Float](repeating: 0, count: valueHeadDim)
                for dv in 0..<valueHeadDim {
                    delta[dv] = (conv[vBase + dv] - memory[dv]) * beta
                }
                for dk in 0..<keyHeadDim {
                    let row = stateBase + dk * valueHeadDim
                    let kk = k[dk]
                    let qq = q[dk]
                    for dv in 0..<valueHeadDim {
                        cache.recurrent[row + dv] += kk * delta[dv]
                        y[t * valueDim + h * valueHeadDim + dv] +=
                            cache.recurrent[row + dv] * qq
                    }
                }
            }
        }

        // Qwen3_5RMSNormGated: ones-centered weight (no +1 bake), sigmoid gate.
        for t in 0..<S {
            for h in 0..<numVHeads {
                let base = t * valueDim + h * valueHeadDim
                var ms: Float = 0
                for d in 0..<valueHeadDim { ms += y[base + d] * y[base + d] }
                let inv = 1 / sqrtf(ms / Float(valueHeadDim) + eps)
                for d in 0..<valueHeadDim {
                    y[base + d] = y[base + d] * inv * w.norm[d] * sigmoid(z[base + d])
                }
            }
        }
        var out = [Float](repeating: 0, count: S * hidden)
        for t in 0..<S {
            matVec(w.out, rows: hidden, cols: valueDim, x: y, xOffset: t * valueDim,
                   into: &out, outOffset: t * hidden)
        }
        return out
    }

    // MARK: - QSA indexer

    /// Per query, the sorted absolute KV indices the attention layer may see.
    /// Appends this call's raw indexer keys to the cache as a side effect.
    private func indexerSelect(_ x: [Float],
                               _ w: FlashNextWeights.AttentionWeights,
                               rows S: Int, cache: LayerCache,
                               cos: [Float], sin: [Float], layer L: Int,
                               capture: inout Capture?) -> [[Int]] {
        let qRows = indexerHeads * indexerHeadDim
        let projRows = qRows + indexerKVHeads * indexerHeadDim
        var proj = [Float](repeating: 0, count: projRows)
        var queries = [Float](repeating: 0, count: S * qRows)
        for t in 0..<S {
            matVec(w.indexerQK, rows: projRows, cols: hidden, x: x, xOffset: t * hidden,
                   into: &proj, outOffset: 0)
            // Query heads first, then the single raw key head.
            for h in 0..<indexerHeads {
                var head = [Float](repeating: 0, count: indexerHeadDim)
                rmsNorm(proj, offset: h * indexerHeadDim, count: indexerHeadDim,
                        weight: w.indexerQNorm, into: &head, at: 0)
                applyRoPE(&head, at: 0, position: seen + t, cos: cos, sin: sin)
                for d in 0..<indexerHeadDim {
                    queries[t * qRows + h * indexerHeadDim + d] = head[d]
                }
            }
            // The RAW key is cached: un-normed, un-roped.
            for d in 0..<indexerHeadDim { cache.indexerKeys.append(proj[qRows + d]) }
        }

        var selected: [[Int]] = []
        for t in 0..<S {
            let query = seen + t
            let visible = query + 1
            let completeBlocks = visible / compressRatio
            var chosen: [Int] = []
            if completeBlocks > 0 {
                var scores = [Float](repeating: 0, count: completeBlocks)
                var pooled = [Float](repeating: 0, count: indexerHeadDim)
                var normed = [Float](repeating: 0, count: indexerHeadDim)
                for blockIndex in 0..<completeBlocks {
                    let start = blockIndex * compressRatio
                    // fp32 mean over the block's raw keys, THEN k_layernorm,
                    // THEN RoPE at the block's FIRST position.
                    for d in 0..<indexerHeadDim {
                        var acc: Float = 0
                        for j in 0..<compressRatio {
                            acc += cache.indexerKeys[(start + j) * indexerHeadDim + d]
                        }
                        pooled[d] = acc / Float(compressRatio)
                    }
                    rmsNorm(pooled, offset: 0, count: indexerHeadDim,
                            weight: w.indexerKNorm, into: &normed, at: 0)
                    applyRoPE(&normed, at: 0, position: start, cos: cos, sin: sin)
                    var score: Float = 0
                    for h in 0..<indexerHeads {
                        var dot: Float = 0
                        for d in 0..<indexerHeadDim {
                            dot += queries[t * qRows + h * indexerHeadDim + d] * normed[d]
                        }
                        if dot > 0 { score += dot }          // relu, then sum over heads
                    }
                    scores[blockIndex] = score / sqrtf(Float(indexerHeadDim))
                }
                let k = min(blockTopK, completeBlocks)
                if k < completeBlocks {
                    let ranked = scores.sorted(by: >)
                    if ranked[k - 1] == ranked[k] {
                        capture?.indexerBoundaryTies.append(
                            (layer: L, query: query, score: ranked[k]))
                    }
                }
                for blockIndex in descendingTopK(scores, k: k) {
                    for j in 0..<compressRatio {
                        chosen.append(blockIndex * compressRatio + j)
                    }
                }
            }
            // The incomplete tail is always kept. There is no "always keep
            // self" rule: a query whose own block loses the top-k does not
            // attend to itself.
            for i in (completeBlocks * compressRatio)..<visible { chosen.append(i) }
            selected.append(chosen.sorted())
        }
        return selected
    }

    // MARK: - Full attention

    private func attention(_ x: [Float], _ w: FlashNextWeights.AttentionWeights,
                           rows S: Int, cache: LayerCache,
                           cos: [Float], sin: [Float],
                           selected: [[Int]]) -> [Float] {
        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim
        var packed = [Float](repeating: 0, count: 2 * qDim)
        var queries = [Float](repeating: 0, count: S * qDim)
        var gates = [Float](repeating: 0, count: S * qDim)
        var kRow = [Float](repeating: 0, count: kvDim)
        var vRow = [Float](repeating: 0, count: kvDim)

        for t in 0..<S {
            matVec(w.q, rows: 2 * qDim, cols: hidden, x: x, xOffset: t * hidden,
                   into: &packed, outOffset: 0)
            // q_proj packs PER HEAD as [q | gate]; it is not a global split.
            for h in 0..<numHeads {
                let src = h * 2 * headDim
                var head = [Float](repeating: 0, count: headDim)
                rmsNorm(packed, offset: src, count: headDim,
                        weight: w.qNorm, into: &head, at: 0)
                applyRoPE(&head, at: 0, position: seen + t, cos: cos, sin: sin)
                for d in 0..<headDim {
                    queries[t * qDim + h * headDim + d] = head[d]
                    gates[t * qDim + h * headDim + d] = packed[src + headDim + d]
                }
            }
            matVec(w.k, rows: kvDim, cols: hidden, x: x, xOffset: t * hidden,
                   into: &kRow, outOffset: 0)
            matVec(w.v, rows: kvDim, cols: hidden, x: x, xOffset: t * hidden,
                   into: &vRow, outOffset: 0)
            for h in 0..<numKVHeads {
                var head = [Float](repeating: 0, count: headDim)
                rmsNorm(kRow, offset: h * headDim, count: headDim,
                        weight: w.kNorm, into: &head, at: 0)
                applyRoPE(&head, at: 0, position: seen + t, cos: cos, sin: sin)
                cache.keys.append(contentsOf: head)
            }
            cache.values.append(contentsOf: vRow)   // v is neither normed nor roped
        }

        let groups = numHeads / numKVHeads
        let scale = 1 / sqrtf(Float(headDim))
        var out = [Float](repeating: 0, count: S * qDim)
        for t in 0..<S {
            let keep = selected[t]
            for h in 0..<numHeads {
                let kv = h / groups
                var scores = [Float](repeating: 0, count: keep.count)
                var maximum = -Float.greatestFiniteMagnitude
                for (i, position) in keep.enumerated() {
                    var dot: Float = 0
                    let kBase = (position * numKVHeads + kv) * headDim
                    for d in 0..<headDim {
                        dot += queries[t * qDim + h * headDim + d] * cache.keys[kBase + d]
                    }
                    scores[i] = dot * scale
                    if scores[i] > maximum { maximum = scores[i] }
                }
                var sum: Float = 0
                for i in 0..<scores.count {
                    scores[i] = expf(scores[i] - maximum)
                    sum += scores[i]
                }
                for (i, position) in keep.enumerated() {
                    let p = scores[i] / sum
                    let vBase = (position * numKVHeads + kv) * headDim
                    for d in 0..<headDim {
                        out[t * qDim + h * headDim + d] += p * cache.values[vBase + d]
                    }
                }
            }
        }
        // The sigmoid output gate applies BEFORE o_proj, on the flattened
        // per-head attention output. The gate is never normed or rotated.
        for i in 0..<out.count { out[i] *= sigmoid(gates[i]) }
        var projected = [Float](repeating: 0, count: S * hidden)
        for t in 0..<S {
            matVec(w.o, rows: hidden, cols: qDim, x: out, xOffset: t * qDim,
                   into: &projected, outOffset: t * hidden)
        }
        return projected
    }

    // MARK: - PLE

    private func pleBlock(hyper: [Float], tokens: [Int], cache: LayerCache,
                          capture: inout Capture?, key: String) throws -> [Float] {
        guard let w = pleW else { return [Float](repeating: 0, count: hyper.count) }
        let S = tokens.count
        // Decode reads the cached history BEFORE overwriting it, and the shift
        // runs over this window only — a real quirk of the reference's windowed
        // decode path versus a full prefill.
        let window = cache.pleHistory + tokens
        let rows = ngramRowIDs(window, w: w)
        let kept = Array(rows[(rows.count - S)...])
        capture?.integers[key + "ple_ngram_row_ids"] = kept
        cache.pleHistory = Array(window.suffix(max(0, ngramSize - 1)))

        var embeds = [Float](repeating: 0, count: S * hidden)
        for t in 0..<S {
            var offset = 0
            for row in kept[t] {
                let values = try ngramRow(row, pool: w.pool)
                for (i, v) in values.enumerated() { embeds[t * hidden + offset + i] = v }
                offset += values.count
            }
            precondition(offset == hidden)
        }
        capture?.floats[key + "ple_ngram_embeds"] = embeds

        var gatedValue = [Float](repeating: 0, count: S * bundle)
        var normedGated = [Float](repeating: 0, count: S * bundle)
        var keyProjected = [Float](repeating: 0, count: bundle)
        var keyNormed = [Float](repeating: 0, count: bundle)
        var queryNormed = [Float](repeating: 0, count: bundle)
        var value = [Float](repeating: 0, count: hidden)
        for t in 0..<S {
            matVec(w.keyProj, rows: bundle, cols: hidden, x: embeds,
                   xOffset: t * hidden, into: &keyProjected, outOffset: 0)
            groupRMSNorm(keyProjected, offset: 0, weight: w.normKey, into: &keyNormed)
            matVec(w.valueProj, rows: hidden, cols: hidden, x: embeds,
                   xOffset: t * hidden, into: &value, outOffset: 0)
            groupRMSNorm(hyper, offset: t * bundle, weight: w.normQuery,
                         into: &queryNormed)
            for j in 0..<hcCount {
                var dot: Float = 0
                for d in 0..<hidden {
                    dot += keyNormed[j * hidden + d] * queryNormed[j * hidden + d]
                }
                var g = dot / sqrtf(Float(hidden))
                // Signed sqrt with the 1e-6 floor applied to the MAGNITUDE
                // before the sqrt, so |g'| >= 1e-3 unless g is exactly zero.
                let sign: Float = g > 0 ? 1 : (g < 0 ? -1 : 0)
                g = sqrtf(max(abs(g), 1e-6)) * sign
                let gate = sigmoid(g)
                for d in 0..<hidden {
                    gatedValue[t * bundle + j * hidden + d] = gate * value[d]
                }
            }
            groupRMSNorm(gatedValue, offset: t * bundle, weight: w.normConv,
                         into: &queryNormed)
            for i in 0..<bundle { normedGated[t * bundle + i] = queryNormed[i] }
        }

        // Depthwise causal conv, kernel 4, dilation = ngramSize, over
        // [cached state | current]. Taps step back by the dilation.
        var padded = cache.pleConv
        padded.append(contentsOf: normedGated)
        var out = gatedValue
        for t in 0..<S {
            for c in 0..<bundle {
                var acc: Float = 0
                for j in 0..<pleConvKernel {
                    let row = t + pleStateLength - (pleConvKernel - 1 - j) * pleDilation
                    acc += w.conv[c * pleConvKernel + j] * padded[row * bundle + c]
                }
                out[t * bundle + c] += silu(acc)
            }
        }
        cache.pleConv = Array(padded[(padded.count - pleStateLength * bundle)...])
        capture?.floats[key + "ple_out"] = out
        return out
    }

    /// Absolute row indices into the padded n-gram table, one list per position
    /// of `history`. All arithmetic is exact 64-bit: the mix consumes the full
    /// positive `Int64` range by construction, so `Double` cannot hold it.
    private func ngramRowIDs(_ history: [Int],
                             w: FlashNextWeights.PLEWeights) -> [[Int]] {
        let n = history.count
        var shifted: [[Int]] = []
        for s in 0..<ngramSize { shifted.append(shiftRightIgnoringEOS(history, by: s)) }
        var rows = [[Int]](repeating: [], count: n)
        let perOrder = ngramHeads / max(1, ngramSize - 1)
        for order in 2...max(2, ngramSize) {
            let start = (order - 2) * perOrder
            for i in 0..<n {
                var mix: UInt64 = 0
                for p in 0..<order {
                    mix ^= UInt64(bitPattern: Int64(shifted[p][i]))
                        &* UInt64(bitPattern: w.multipliers[p])
                }
                // `torch.remainder` with a positive divisor is non-negative;
                // the mix stays below 2^63 by construction, so an unsigned
                // modulo is the same value and cannot go negative.
                for h in 0..<perOrder {
                    let head = start + h
                    let residue = mix % UInt64(w.headVocabSizes[head])
                    rows[i].append(Int(Int64(residue) + w.headOffsets[head]))
                }
            }
        }
        return rows
    }

    /// `_shift_right_ignore_eos`: look back `shift` tokens, but never across an
    /// EOS and never before position 0 — either violation reads `eos_token_id`.
    /// The boundary is computed with a strict `<`, so the EOS token itself is
    /// the last token of its own segment.
    private func shiftRightIgnoringEOS(_ ids: [Int], by shift: Int) -> [Int] {
        if shift == 0 { return ids }
        var out = [Int](repeating: eosTokenID, count: ids.count)
        var lastEOS = -1
        for i in 0..<ids.count {
            let previousEOS = lastEOS            // strictly before i
            let segmentStart = previousEOS + 1
            let source = i - shift
            if i - segmentStart >= shift, source >= 0 { out[i] = ids[source] }
            if ids[i] == eosTokenID { lastEOS = i }
        }
        return out
    }

    private func ngramRow(_ row: Int, pool: PleRowPool) throws -> [Float] {
        if let hit = ngramRows[row] { return hit }
        let values = try pool.readRow(row)
        ngramRows[row] = values
        return values
    }

    // MARK: - Norms

    /// `Qwen4ExpTextRMSNorm` with `group_size = hidden`: each of the `hcCount`
    /// streams is normalized over its own `hidden` channels, but the weight
    /// spans the full bundle — one scale per (stream, channel).
    private func groupRMSNorm(_ x: [Float], offset: Int,
                              weight: [Float], into out: inout [Float]) {
        for j in 0..<hcCount {
            let base = offset + j * hidden
            var ms: Float = 0
            for d in 0..<hidden { ms += x[base + d] * x[base + d] }
            let inv = 1 / sqrtf(ms / Float(hidden) + eps)
            for d in 0..<hidden {
                out[j * hidden + d] = x[base + d] * inv * weight[j * hidden + d]
            }
        }
    }

    /// Ungrouped `Qwen4ExpTextRMSNorm` over `count` channels. The `(1 + w)`
    /// bake is already in `weight` — `Model.normWeight` applied it at load.
    private func rmsNorm(_ x: [Float], offset: Int, count: Int,
                         weight: [Float], into out: inout [Float], at outOffset: Int) {
        var ms: Float = 0
        for d in 0..<count { ms += x[offset + d] * x[offset + d] }
        let inv = 1 / sqrtf(ms / Float(count) + eps)
        for d in 0..<count { out[outOffset + d] = x[offset + d] * inv * weight[d] }
    }

    // MARK: - RoPE

    /// `[position * rotaryDim/2]` interleaved cos and sin tables.
    private func ropeTable(upTo length: Int) -> (cos: [Float], sin: [Float]) {
        let half = rotaryDim / 2
        var cos = [Float](repeating: 0, count: length * half)
        var sin = [Float](repeating: 0, count: length * half)
        for p in 0..<length {
            for i in 0..<half {
                let inv = 1 / powf(ropeTheta, Float(2 * i) / Float(rotaryDim))
                let angle = Float(p) * inv
                cos[p * half + i] = cosf(angle)
                sin[p * half + i] = sinf(angle)
            }
        }
        return (cos, sin)
    }

    /// Partial NeoX RoPE: only the leading `rotaryDim` channels rotate, paired
    /// `(i, i + rotaryDim/2)`; the tail is spliced back unchanged.
    private func applyRoPE(_ head: inout [Float], at offset: Int, position: Int,
                           cos: [Float], sin: [Float]) {
        let half = rotaryDim / 2
        let base = position * half
        for i in 0..<half {
            let c = cos[base + i], s = sin[base + i]
            let a = head[offset + i], b = head[offset + i + half]
            head[offset + i] = a * c - b * s
            head[offset + i + half] = b * c + a * s
        }
    }

    // MARK: - Numeric helpers

    private func matVec(_ w: [Float], rows: Int, cols: Int,
                        x: [Float], xOffset: Int,
                        into out: inout [Float], outOffset: Int) {
        for r in 0..<rows {
            var acc: Float = 0
            let base = r * cols
            for c in 0..<cols { acc += w[base + c] * x[xOffset + c] }
            out[outOffset + r] = acc
        }
    }

    private func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
    private func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }
    private func softplus(_ x: Float) -> Float { x > 20 ? x : logf(1 + expf(x)) }

    /// The indices `torch.topk(v, k, largest=True)` puts in the first `k` slots
    /// on CPU, including its tie behaviour.
    ///
    /// PyTorch's CPU topk takes the `std::nth_element` branch whenever
    /// `k * 64 > n` — always, at these widths — calling
    /// `nth_element(begin, begin + k - 1, end, greater)` and reading the first
    /// `k` entries. libc++'s `__nth_element` short-circuits by length: `2` swaps
    /// if out of order, `3` runs `__sort3`, `<= 7` runs `__selection_sort`
    /// (first maximum, swapped into place — the swap displaces whatever was
    /// there, which is why ties do NOT resolve to lowest-index-first), and
    /// larger ranges run a median-of-3 quickselect.
    ///
    /// Only the first three branches are reproduced. Across all four golden
    /// runs the only boundary ties occur at 3 and 4 candidate blocks (all with
    /// the tied score exactly 0 after the ReLU); above 7 candidates the
    /// documented lowest-index-first fallback applies, and it is unobservable
    /// unless a tie occurs there. `Capture.indexerBoundaryTies` records any tie
    /// so a parity failure can be attributed rather than guessed at.
    private func descendingTopK(_ values: [Float], k: Int) -> [Int] {
        var a = values.enumerated().map { (index: $0.offset, value: $0.element) }
        let n = a.count
        if k >= n { return a.map(\.index) }
        func greater(_ x: Int, _ y: Int) -> Bool { a[x].value > a[y].value }
        switch n {
        case 0, 1:
            break
        case 2:
            if greater(1, 0) { a.swapAt(0, 1) }
        case 3:
            if !greater(1, 0) {
                if greater(2, 1) {
                    a.swapAt(1, 2)
                    if greater(1, 0) { a.swapAt(0, 1) }
                }
            } else if greater(2, 1) {
                a.swapAt(0, 2)
            } else {
                a.swapAt(0, 1)
                if greater(2, 1) { a.swapAt(1, 2) }
            }
        case 4...7:
            for i in 0..<(n - 1) {
                var best = i
                for m in (i + 1)..<n where greater(m, best) { best = m }
                if best != i { a.swapAt(i, best) }
            }
        default:
            a.sort { $0.value != $1.value ? $0.value > $1.value : $0.index < $1.index }
        }
        return (0..<k).map { a[$0].index }
    }
}
