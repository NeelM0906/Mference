import Foundation

/// A weight matrix of the CPU reference, `rows x cols`, applied as `y = W x`.
///
/// The reference only reads rows, so a matrix can be dense floats (the toy
/// checkpoint) or decode a quantized row on demand (NVFP4 and Q8_0 decoders,
/// which keep a real checkpoint mapped instead of in Swift heap memory).
public protocol Inkling2ReferenceMatrix: Sendable {
    var rows: Int { get }
    var cols: Int { get }
    /// Row `index` as `cols` floats.
    func row(_ index: Int) -> [Float]
    /// `y = W x`, each dot product accumulated in double precision. The
    /// default reads `row(_:)`; a conforming type may compute it directly.
    func apply(_ x: [Float]) -> [Float]
}

extension Inkling2ReferenceMatrix {
    public func apply(_ x: [Float]) -> [Float] {
        precondition(x.count == cols, "matrix \(rows)x\(cols) applied to \(x.count) values")
        return (0..<rows).map { Inkling2ReferenceOps.dot(row($0), x) }
    }
}

/// Dense row-major float matrix.
public struct Inkling2DenseMatrix: Inkling2ReferenceMatrix {
    public let rows: Int
    public let cols: Int
    /// Row-major, `rows * cols` values.
    public let values: [Float]

    public init(rows: Int, cols: Int, values: [Float]) {
        precondition(values.count == rows * cols, "dense matrix \(rows)x\(cols) given \(values.count) values")
        self.rows = rows
        self.cols = cols
        self.values = values
    }

    public func row(_ index: Int) -> [Float] {
        precondition(index >= 0 && index < rows, "row \(index) of \(rows)")
        return Array(values[(index * cols)..<((index + 1) * cols)])
    }

    /// The same product as the default, without copying rows.
    public func apply(_ x: [Float]) -> [Float] {
        precondition(x.count == cols, "matrix \(rows)x\(cols) applied to \(x.count) values")
        return values.withUnsafeBufferPointer { w in
            x.withUnsafeBufferPointer { v in
                (0..<rows).map { r in
                    var s = 0.0
                    let base = r * cols
                    for i in 0..<cols { s += Double(w[base + i]) * Double(v[i]) }
                    return Float(s)
                }
            }
        }
    }

    /// The same matrix with every value transformed.
    public func map(_ transform: (Float) -> Float) -> Inkling2DenseMatrix {
        Inkling2DenseMatrix(rows: rows, cols: cols, values: values.map(transform))
    }
}

/// One stored decoder layer, in the tensor semantics of the upstream
/// llama.cpp `inkling2` model (the GGUF the program ships). Every matrix is
/// `[out x in]` applied as `y = W x`; the GGUF dimension order `{in, out}`
/// is the same storage.
public struct Inkling2ReferenceLayerWeights: Sendable {
    // Attention projections.
    /// `attn_q` `[qDim x hidden]`.
    public var wq: any Inkling2ReferenceMatrix
    /// `attn_k` `[kvDim x hidden]`.
    public var wk: any Inkling2ReferenceMatrix
    /// `attn_v` `[kvDim x hidden]`.
    public var wv: any Inkling2ReferenceMatrix
    /// `attn_output` `[hidden x qDim]`.
    public var wo: any Inkling2ReferenceMatrix
    /// `attn_gate` `[qDim x hidden]`: the attention output is multiplied by
    /// `sigmoid(outputGate x)` before `wo`.
    public var outputGate: any Inkling2ReferenceMatrix
    /// `attn_rel` `[numHeads*relDim x hidden]`: the per-head relative state.
    public var rel: any Inkling2ReferenceMatrix
    /// `attn_sink_proj` `[numHeads x hidden]`: one sink logit per head.
    public var sinkProj: any Inkling2ReferenceMatrix

    /// `attn_q_clip` `[numHeads*headDim]`, multiplied into the query before its L2 clip.
    public var qClipScale: [Float]
    /// `attn_rel_clip` `[numHeads*relDim]`, multiplied into the relative state before its L2 clip.
    public var relClipScale: [Float]
    /// `attn_rel_bias` `[extent x relDim]`, row = backward distance. The bias
    /// of head `h` at distance `d` is `relBias[d] . relState[h]`.
    public var relBias: [Float]

    // Short convolutions, `[channels x convKernel]`, tap `convKernel - 1`
    // weighting the current token (the ssm_conv order).
    /// `attn_k_conv` `[kvDim x K]` (static taps on global layers).
    public var kConv: [Float]
    /// `attn_v_conv` `[kvDim x K]`.
    public var vConv: [Float]
    /// `attn_post_conv` `[hidden x K]`, after the output projection.
    public var attnPostConv: [Float]
    /// `ffn_post_conv` `[hidden x K]`, after the experts.
    public var ffnPostConv: [Float]

    // Input norms.
    /// `attn_norm` `[hidden]`, sliding layers only.
    public var attnNorm: [Float]?
    /// `attn_q_inp_norm`, `attn_k_inp_norm`, `attn_v_inp_norm` `[hidden]`, global layers only.
    public var qInputNorm: [Float]?
    public var kInputNorm: [Float]?
    public var vInputNorm: [Float]?
    /// `attn_k_conv_proj`, `attn_v_conv_proj` `[numKVHeads*K x hidden]`,
    /// global layers only: input-dependent taps added to the static ones,
    /// shared by the channels of a KV head. Row `head * K + lag`, where lag 0
    /// weights the current token (the opposite order of the static taps).
    public var kConvProj: (any Inkling2ReferenceMatrix)?
    public var vConvProj: (any Inkling2ReferenceMatrix)?

    // Block attention residuals.
    /// `attn_res_query`: none for layer 0, one `[hidden]` query for a
    /// sliding layer, three (query, key, value inputs) for a global layer.
    public var attnResQueries: [[Float]]
    /// `ffn_res_query` `[hidden]`.
    public var ffnResQuery: [Float]

    // Mixture of experts.
    /// `ffn_norm` `[hidden]`.
    public var ffnNorm: [Float]
    /// `ffn_gate_inp` `[numExperts x hidden]`, F32 in the file.
    public var router: any Inkling2ReferenceMatrix
    /// `exp_probs_b` `[numExperts]`: added to the sigmoid scores for the
    /// top-k choice only.
    public var routerBias: [Float]
    /// `ffn_gate_up_exps`, one `[2*expertFF x hidden]` per expert: rows
    /// `0 ..< expertFF` are the gate, the rest are `up`.
    public var expertGateUp: [any Inkling2ReferenceMatrix]
    /// `ffn_down_exps`, one `[hidden x expertFF]` per expert.
    public var expertDown: [any Inkling2ReferenceMatrix]

    public init(wq: any Inkling2ReferenceMatrix, wk: any Inkling2ReferenceMatrix,
                wv: any Inkling2ReferenceMatrix, wo: any Inkling2ReferenceMatrix,
                outputGate: any Inkling2ReferenceMatrix, rel: any Inkling2ReferenceMatrix,
                sinkProj: any Inkling2ReferenceMatrix, qClipScale: [Float],
                relClipScale: [Float], relBias: [Float], kConv: [Float],
                vConv: [Float], attnPostConv: [Float], ffnPostConv: [Float],
                attnNorm: [Float]?, qInputNorm: [Float]?, kInputNorm: [Float]?,
                vInputNorm: [Float]?, kConvProj: (any Inkling2ReferenceMatrix)?,
                vConvProj: (any Inkling2ReferenceMatrix)?, attnResQueries: [[Float]],
                ffnResQuery: [Float], ffnNorm: [Float],
                router: any Inkling2ReferenceMatrix, routerBias: [Float],
                expertGateUp: [any Inkling2ReferenceMatrix],
                expertDown: [any Inkling2ReferenceMatrix]) {
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.outputGate = outputGate
        self.rel = rel
        self.sinkProj = sinkProj
        self.qClipScale = qClipScale
        self.relClipScale = relClipScale
        self.relBias = relBias
        self.kConv = kConv
        self.vConv = vConv
        self.attnPostConv = attnPostConv
        self.ffnPostConv = ffnPostConv
        self.attnNorm = attnNorm
        self.qInputNorm = qInputNorm
        self.kInputNorm = kInputNorm
        self.vInputNorm = vInputNorm
        self.kConvProj = kConvProj
        self.vConvProj = vConvProj
        self.attnResQueries = attnResQueries
        self.ffnResQuery = ffnResQuery
        self.ffnNorm = ffnNorm
        self.router = router
        self.routerBias = routerBias
        self.expertGateUp = expertGateUp
        self.expertDown = expertDown
    }
}

/// The cross-pass message path of one looped (core) layer, unused in pass 1.
/// Controller order is query, key, value, MoE input (`LOOP_CTRL_Q`, `_K`,
/// `_V`, `_FFN` upstream).
public struct Inkling2ReferenceLoopWeights: Sendable {
    /// `loop_msg_query` 4 x `[hidden]`: the message read for each input.
    public var messageQueries: [[Float]]
    /// `*_ctrl_down` 4 x `[gateRank x 2*hidden]`, applied to
    /// `[rmsnorm(x); rmsnorm(message)]`.
    public var controlDown: [any Inkling2ReferenceMatrix]
    /// `*_ctrl_gain` 4 x `[hidden x gateRank]`.
    public var controlGain: [any Inkling2ReferenceMatrix]
    /// `*_ctrl_forget` 4 x `[hidden x gateRank]`.
    public var controlForget: [any Inkling2ReferenceMatrix]

    public init(messageQueries: [[Float]], controlDown: [any Inkling2ReferenceMatrix],
                controlGain: [any Inkling2ReferenceMatrix],
                controlForget: [any Inkling2ReferenceMatrix]) {
        self.messageQueries = messageQueries
        self.controlDown = controlDown
        self.controlGain = controlGain
        self.controlForget = controlForget
    }
}

/// Every weight of the text decoder.
public struct Inkling2ReferenceWeights: Sendable {
    /// `token_embd` `[vocabPadded x hidden]`, read by row.
    public var embedding: any Inkling2ReferenceMatrix
    /// `token_embd_norm` `[hidden]`, applied to every looked-up row.
    public var embeddingNorm: [Float]
    /// `output_res_query` `[hidden]`: mixes the final block states.
    public var outputResQuery: [Float]
    /// `output_norm` `[hidden]`.
    public var outputNorm: [Float]
    /// `output` `[vocabPadded x hidden]`; only the first `vocabUnpadded` rows produce logits.
    public var output: any Inkling2ReferenceMatrix
    /// The 24 stored layers.
    public var layers: [Inkling2ReferenceLayerWeights]
    /// One entry per core layer (`coreStart ..< coreEnd`), in order.
    public var loopLayers: [Inkling2ReferenceLoopWeights]

    public init(embedding: any Inkling2ReferenceMatrix, embeddingNorm: [Float],
                outputResQuery: [Float], outputNorm: [Float],
                output: any Inkling2ReferenceMatrix,
                layers: [Inkling2ReferenceLayerWeights],
                loopLayers: [Inkling2ReferenceLoopWeights]) {
        self.embedding = embedding
        self.embeddingNorm = embeddingNorm
        self.outputResQuery = outputResQuery
        self.outputNorm = outputNorm
        self.output = output
        self.layers = layers
        self.loopLayers = loopLayers
    }

    /// Checks every tensor against `config`; throws naming the first mismatch.
    public func validate(against config: Inkling2ReferenceConfig) throws {
        try config.validate()
        func fail(_ message: String) -> Error {
            NSError(domain: "Inkling2ReferenceWeights", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: message])
        }
        func check(_ m: any Inkling2ReferenceMatrix, _ rows: Int, _ cols: Int, _ name: String) throws {
            guard m.rows == rows, m.cols == cols else {
                throw fail("\(name) is \(m.rows)x\(m.cols), expected \(rows)x\(cols)")
            }
        }
        func count(_ v: [Float], _ n: Int, _ name: String) throws {
            guard v.count == n else { throw fail("\(name) has \(v.count) values, expected \(n)") }
        }
        let d = config.hiddenSize
        let k = config.convKernel
        try check(embedding, config.vocabPadded, d, "embedding")
        try check(output, config.vocabPadded, d, "output")
        try count(embeddingNorm, d, "embeddingNorm")
        try count(outputResQuery, d, "outputResQuery")
        try count(outputNorm, d, "outputNorm")
        guard layers.count == config.numLayers else { throw fail("\(layers.count) layers, expected \(config.numLayers)") }
        guard loopLayers.count == config.numCoreLayers else { throw fail("\(loopLayers.count) loop layers, expected \(config.numCoreLayers)") }
        for (i, l) in layers.enumerated() {
            let p = "layer \(i)"
            let sliding = config.slidingLayers[i]
            try check(l.wq, config.qDim, d, "\(p) wq")
            try check(l.wk, config.kvDim, d, "\(p) wk")
            try check(l.wv, config.kvDim, d, "\(p) wv")
            try check(l.wo, d, config.qDim, "\(p) wo")
            try check(l.outputGate, config.qDim, d, "\(p) outputGate")
            try check(l.rel, config.numHeads * config.relDim, d, "\(p) rel")
            try check(l.sinkProj, config.numHeads, d, "\(p) sinkProj")
            try count(l.qClipScale, config.qDim, "\(p) qClipScale")
            try count(l.relClipScale, config.numHeads * config.relDim, "\(p) relClipScale")
            try count(l.relBias, config.relativeExtent(layer: i) * config.relDim, "\(p) relBias")
            try count(l.kConv, config.kvDim * k, "\(p) kConv")
            try count(l.vConv, config.kvDim * k, "\(p) vConv")
            try count(l.attnPostConv, d * k, "\(p) attnPostConv")
            try count(l.ffnPostConv, d * k, "\(p) ffnPostConv")
            if sliding {
                guard let n = l.attnNorm else { throw fail("\(p) is sliding and has no attnNorm") }
                try count(n, d, "\(p) attnNorm")
                guard l.kConvProj == nil, l.vConvProj == nil else { throw fail("\(p) is sliding and has dynamic conv taps") }
            } else {
                guard let q = l.qInputNorm, let kn = l.kInputNorm, let v = l.vInputNorm else {
                    throw fail("\(p) is global and lacks q/k/v input norms")
                }
                try count(q, d, "\(p) qInputNorm")
                try count(kn, d, "\(p) kInputNorm")
                try count(v, d, "\(p) vInputNorm")
                guard let kp = l.kConvProj, let vp = l.vConvProj else { throw fail("\(p) is global and lacks dynamic conv taps") }
                try check(kp, config.numKVHeads * k, d, "\(p) kConvProj")
                try check(vp, config.numKVHeads * k, d, "\(p) vConvProj")
            }
            let queries = i == 0 ? 0 : (sliding ? 1 : 3)
            guard l.attnResQueries.count == queries else {
                throw fail("\(p) has \(l.attnResQueries.count) attention-residual queries, expected \(queries)")
            }
            for q in l.attnResQueries { try count(q, d, "\(p) attnResQuery") }
            try count(l.ffnResQuery, d, "\(p) ffnResQuery")
            try count(l.ffnNorm, d, "\(p) ffnNorm")
            try check(l.router, config.numExperts, d, "\(p) router")
            try count(l.routerBias, config.numExperts, "\(p) routerBias")
            guard l.expertGateUp.count == config.numExperts, l.expertDown.count == config.numExperts else {
                throw fail("\(p) expert count mismatch")
            }
            for e in 0..<config.numExperts {
                try check(l.expertGateUp[e], 2 * config.expertFF, d, "\(p) expert \(e) gateUp")
                try check(l.expertDown[e], d, config.expertFF, "\(p) expert \(e) down")
            }
        }
        for (i, l) in loopLayers.enumerated() {
            let p = "loop layer \(config.coreStart + i)"
            guard l.messageQueries.count == 4, l.controlDown.count == 4,
                  l.controlGain.count == 4, l.controlForget.count == 4 else {
                throw fail("\(p) needs four message queries and controllers")
            }
            for c in 0..<4 {
                try count(l.messageQueries[c], d, "\(p) messageQuery \(c)")
                try check(l.controlDown[c], config.gateRank, 2 * d, "\(p) controlDown \(c)")
                try check(l.controlGain[c], d, config.gateRank, "\(p) controlGain \(c)")
                try check(l.controlForget[c], d, config.gateRank, "\(p) controlForget \(c)")
            }
        }
    }
}
