import Foundation
import MferenceValidationSupport

/// A second Inkling 2 forward pass, written from the transformers reference
/// (`modeling_inkling2_loop.py` at `be48e7a`: `Inkling2LoopTextModel.forward`
/// and the four decoder-layer classes) with its structure kept: no caches,
/// every layer sees the whole sequence through causal masks, convolutions run
/// over the whole sequence with zero padding, block states are appended by
/// the model loop after a boundary layer (and inside the layer for its MoE
/// input), and later passes read `block_states[num_pre_loop_blocks + 1:]`
/// and the previous output. It shares only the small, separately tested
/// operations (norms, clips, depth mix, gate, router, expert) with
/// `Inkling2ReferenceModel`; the wiring, attention, bias gather and
/// convolutions are its own.
struct Inkling2TransformersMirror {
    typealias Op = Inkling2ReferenceOps
    /// `[token][values]`.
    typealias Sequence = [[Float]]

    let c: Inkling2ReferenceConfig
    let w: Inkling2ReferenceWeights

    /// Hidden states after each visit in execution order (`[visit][token][hidden]`).
    private(set) var visitOutputs: [Sequence] = []

    // MARK: Model

    mutating func forward(tokens: [Int32]) -> [[Float]] {
        visitOutputs = []
        var hidden: Sequence = tokens.map {
            Op.rmsNorm(w.embedding.row(Int($0)), weight: w.embeddingNorm, eps: c.rmsEps)
        }
        var blockStates: [Sequence] = []

        func isBoundary(_ layer: Int) -> Bool { layer % c.attnResBlockSize == 0 }

        // Pre loop layers.
        for layer in 0..<c.coreStart {
            let new = decoderLayer(layer, hidden: hidden, blockStates: blockStates, messages: nil)
            if isBoundary(layer) { blockStates.append(hidden) }
            hidden = new
            visitOutputs.append(hidden)
        }
        let preLoopHidden = hidden
        let preLoopBlocks = blockStates
        let numPreLoopBlocks = preLoopBlocks.count

        // First pass over the loop layers.
        for layer in c.coreStart..<c.coreEnd {
            let new = decoderLayer(layer, hidden: hidden, blockStates: blockStates, messages: nil)
            if isBoundary(layer) { blockStates.append(hidden) }
            hidden = new
            visitOutputs.append(hidden)
        }
        // Remaining passes restart from the pre-loop states.
        for _ in 1..<max(c.numPasses, 1) {
            let previousBlocks = Array(blockStates[(numPreLoopBlocks + 1)...])
            let previousHidden = hidden
            hidden = preLoopHidden
            blockStates = preLoopBlocks
            for (loopLayerIndex, layer) in (c.coreStart..<c.coreEnd).enumerated() {
                let messages = messageInput(previousBlocks, previousHidden, loopLayerIndex)
                let new = decoderLayer(layer, hidden: hidden, blockStates: blockStates, messages: messages)
                if isBoundary(layer) { blockStates.append(hidden) }
                hidden = new
                visitOutputs.append(hidden)
            }
        }
        // Post loop layers.
        for layer in c.coreEnd..<c.numLayers {
            let new = decoderLayer(layer, hidden: hidden, blockStates: blockStates, messages: nil)
            if isBoundary(layer) { blockStates.append(hidden) }
            hidden = new
            visitOutputs.append(hidden)
        }
        let final = blockAttention(blockStates, hidden, queries: [w.outputResQuery])[0]
        return final.map { h in
            let normed = Op.rmsNorm(h, weight: w.outputNorm, eps: c.rmsEps)
            return (0..<c.vocabUnpadded).map {
                Op.softcap(Op.dot(w.output.row($0), normed), c.finalLogitSoftcap)
            }
        }
    }

    /// `Inkling2LoopMessageAttentionResidual`: per token, the four messages
    /// (query, key, value, mlp) of loop layer `index`.
    private func messageInput(_ blocks: [Sequence], _ partial: Sequence, _ index: Int) -> [Sequence] {
        let perToken = blockAttention(blocks, partial, queries: w.loopLayers[index].messageQueries)
        return perToken   // [message][token][hidden]
    }

    /// `Inkling2LoopBlockAttentionResidual`: one output sequence per query.
    private func blockAttention(_ blocks: [Sequence], _ partial: Sequence, queries: [[Float]]) -> [Sequence] {
        let perToken: [[[Float]]] = (0..<partial.count).map { t in
            Op.mix(states: blocks.map { $0[t] } + [partial[t]], queries: queries)
        }
        return (0..<queries.count).map { q in perToken.map { $0[q] } }
    }

    // MARK: Decoder layers

    /// The four layer classes in one function: sliding or full, loop or not.
    private func decoderLayer(_ layer: Int, hidden: Sequence, blockStates: [Sequence],
                              messages: [Sequence]?) -> Sequence {
        let lw = w.layers[layer]
        let sliding = c.slidingLayers[layer]
        let blockResidual = hidden
        var q: Sequence, k: Sequence, v: Sequence
        if sliding {
            var h = hidden
            if !lw.attnResQueries.isEmpty {
                h = blockAttention(blockStates, hidden, queries: lw.attnResQueries)[0]
            }
            if let m = messages {
                q = norm(gate(h, m[0], layer, 0), lw.attnNorm!)
                k = norm(gate(h, m[1], layer, 1), lw.attnNorm!)
                v = norm(gate(h, m[2], layer, 2), lw.attnNorm!)
            } else {
                q = norm(h, lw.attnNorm!)
                k = q
                v = q
            }
        } else {
            let three = blockAttention(blockStates, hidden, queries: lw.attnResQueries)
            var (hq, hk, hv) = (three[0], three[1], three[2])
            if let m = messages {
                hq = gate(hq, m[0], layer, 0)
                hk = gate(hk, m[1], layer, 1)
                hv = gate(hv, m[2], layer, 2)
            }
            q = norm(hq, lw.qInputNorm!)
            k = norm(hk, lw.kInputNorm!)
            v = norm(hv, lw.vInputNorm!)
        }
        var attn = selfAttention(layer, query: q, key: k, value: v)
        attn = shortConvolution(attn, staticTaps: lw.attnPostConv)

        var residual = attn
        var localBlocks = blockStates
        if layer % c.attnResBlockSize == 0 {
            localBlocks.append(blockResidual)
        } else {
            residual = zip(blockResidual, attn).map { Op.add($0, $1) }
        }
        var h = blockAttention(localBlocks, residual, queries: [lw.ffnResQuery])[0]
        if let m = messages { h = gate(h, m[3], layer, 3) }
        h = norm(h, lw.ffnNorm).map { x in
            let route = Op.route(x, router: lw.router, bias: lw.routerBias, topK: c.topK, routeScale: c.routeScale)
            return Op.moe(x, route: route, gateUp: lw.expertGateUp, down: lw.expertDown,
                          gateSoftcap: c.gateSoftcap, upSoftcap: c.upSoftcap)
        }
        h = shortConvolution(h, staticTaps: lw.ffnPostConv)
        return zip(residual, h).map { Op.add($0, $1) }
    }

    private func norm(_ x: Sequence, _ weight: [Float]) -> Sequence {
        x.map { Op.rmsNorm($0, weight: weight, eps: c.rmsEps) }
    }

    /// `Inkling2LoopMessageGate` with controller `index` (q, k, v, mlp).
    private func gate(_ x: Sequence, _ message: Sequence, _ layer: Int, _ index: Int) -> Sequence {
        let lp = w.loopLayers[layer - c.coreStart]
        return zip(x, message).map {
            Op.control($0, message: $1, down: lp.controlDown[index],
                       gain: lp.controlGain[index], forget: lp.controlForget[index])
        }
    }

    // MARK: Attention

    /// `Inkling2LoopSlidingAttention` / `Inkling2LoopFullAttention` over the
    /// whole sequence with `eager_attention_forward`.
    private func selfAttention(_ layer: Int, query: Sequence, key: Sequence, value: Sequence) -> Sequence {
        let lw = w.layers[layer]
        let sliding = c.slidingLayers[layer]
        let n = query.count
        let hd = c.headDim
        let extent = sliding ? c.slidingWindow : c.relExtent
        func logScale(_ t: Int) -> Float {
            sliding ? 1 : Op.logScale(position: t, floor: c.logScaleFloor, alpha: c.logScaleAlpha)
        }
        func heads(_ x: [Float], _ count: Int) -> [[Float]] {
            let width = x.count / count
            return (0..<count).map { Array(x[($0 * width)..<(($0 + 1) * width)]) }
        }
        // Query: linear, per-head scale and clip, log scale.
        let qStates: [[[Float]]] = (0..<n).map { t in
            let qs = heads(lw.qClipScale, c.numHeads)
            return heads(lw.wq.apply(query[t]), c.numHeads).enumerated().map { h, x in
                Op.l2Clip(x, scale: qs[h], threshold: c.qkClip).map { $0 * logScale(t) }
            }
        }
        // Key and value: linear, short convolution (dynamic on full layers), clip or norm.
        let kLinear = key.map { lw.wk.apply($0) }
        let vLinear = value.map { lw.wv.apply($0) }
        let kConv = sliding
            ? shortConvolution(kLinear, staticTaps: lw.kConv)
            : dynamicConvolution(kLinear, dynamicInput: key, staticTaps: lw.kConv, projection: lw.kConvProj!)
        let vConv = sliding
            ? shortConvolution(vLinear, staticTaps: lw.vConv)
            : dynamicConvolution(vLinear, dynamicInput: value, staticTaps: lw.vConv, projection: lw.vConvProj!)
        let kStates = kConv.map { heads($0, c.numKVHeads).map { Op.l2Clip($0, threshold: c.qkClip) } }
        let vStates = vConv.map { heads($0, c.numKVHeads).map { Op.rmsNorm($0, eps: c.rmsEps) } }
        // Relative bias: relative states @ proj, gathered by distance, zero outside the extent.
        let relScale = heads(lw.relClipScale, c.numHeads)
        let relLogits: [[[Double]]] = (0..<n).map { t in   // [t][head][distance]
            heads(lw.rel.apply(query[t]), c.numHeads).enumerated().map { h, x in
                let r = Op.l2Clip(x, scale: relScale[h], threshold: c.relClip).map { $0 * logScale(t) }
                return (0..<extent).map { d in
                    (0..<c.relDim).reduce(0.0) { $0 + Double(r[$1]) * Double(lw.relBias[d * c.relDim + $1]) }
                }
            }
        }
        let sinks = query.map { lw.sinkProj.apply($0) }
        let scaling = 1.0 / Double(hd).squareRoot()
        let groups = c.numHeads / c.numKVHeads

        var out: Sequence = []
        for t in 0..<n {
            var row: [Float] = []
            for h in 0..<c.numHeads {
                let kvHead = h / groups   // repeat_kv: each KV head serves `groups` query heads
                var logits: [Double] = []
                var keys: [Int] = []
                for j in 0...t {
                    let distance = t - j
                    if sliding && distance >= c.slidingWindow { continue }   // sliding mask
                    var s = 0.0
                    for i in 0..<hd { s += Double(qStates[t][h][i]) * Double(kStates[j][kvHead][i]) }
                    let bias = distance < extent ? relLogits[t][h][distance] : 0
                    logits.append(s * scaling + bias)
                    keys.append(j)
                }
                // The sink joins the softmax and is dropped.
                let combined = logits + [Double(sinks[t][h])]
                let m = combined.max()!
                let e = combined.map { exp($0 - m) }
                let z = e.reduce(0, +)
                var acc = [Double](repeating: 0, count: hd)
                for (p, j) in zip(e.dropLast(), keys) {
                    for i in 0..<hd { acc[i] += p / z * Double(vStates[j][kvHead][i]) }
                }
                row += acc.map { Float($0) }
            }
            // Gated output projection.
            let g = lw.outputGate.apply(query[t]).map(Op.sigmoid)
            out.append(lw.wo.apply(zip(row, g).map { $0 * $1 }))
        }
        return out
    }

    // MARK: Convolutions

    /// `Inkling2LoopShortConvolution`: depthwise `Conv1d` with `K - 1` zeros
    /// of left padding, then `(conv + residual) / sqrt(K + 1)`. The GGUF taps
    /// are in conv1d order (tap `K - 1` weights the current token).
    private func shortConvolution(_ x: Sequence, staticTaps: [Float]) -> Sequence {
        let k = c.convKernel
        let channels = x[0].count
        return (0..<x.count).map { t in
            (0..<channels).map { ch in
                var s = 0.0
                for tap in 0..<k {
                    let source = t - (k - 1) + tap
                    if source >= 0 { s += Double(x[source][ch]) * Double(staticTaps[ch * k + tap]) }
                }
                return Float((s + Double(x[t][ch])) / Double(k + 1).squareRoot())
            }
        }
    }

    /// `Inkling2LoopDynamicShortConvolution`: taps indexed by backward
    /// distance (lag 0 = current token), `static_weight[lag]` plus the
    /// per-token projection shared by a head's channels, then
    /// `(conv + residual) / sqrt(2K + 1)`. The GGUF stores the static taps
    /// flipped into conv1d order, so `static_weight[lag][ch]` is tap `K - 1 - lag`.
    private func dynamicConvolution(_ x: Sequence, dynamicInput: Sequence, staticTaps: [Float],
                                    projection: any Inkling2ReferenceMatrix) -> Sequence {
        let k = c.convKernel
        let channels = x[0].count
        let perHead = channels / c.numKVHeads
        return (0..<x.count).map { t in
            let dynamic = projection.apply(dynamicInput[t])   // view (heads, 1, K)
            return (0..<channels).map { ch in
                let head = ch / perHead
                var s = 0.0
                for lag in 0..<k where t - lag >= 0 {
                    let weight = Double(staticTaps[ch * k + (k - 1 - lag)]) + Double(dynamic[head * k + lag])
                    s += Double(x[t - lag][ch]) * weight
                }
                return Float((s + Double(x[t][ch])) / Double(2 * k + 1).squareRoot())
            }
        }
    }
}
