import Foundation

extension Inkling2ReferenceConfig {
    /// Tiny Inkling 2 with Mini's full schedule: 24 stored layers, prelude
    /// 0 to 5, core 6 to 17 three times, coda 18 to 23, 48 visits, global
    /// layers 5, 11, 17 and 23. Width and vocabulary are small; the window
    /// (8), relative extent (12, different from the window so a mix-up
    /// shows) and log-scale floor (4) are small enough that a short sequence
    /// crosses all three. Clip thresholds and softcaps are set where the toy
    /// weights reach them, so every nonlinearity is exercised.
    public static let toy = Inkling2ReferenceConfig(
        hiddenSize: 64, numLayers: 24, coreStart: 6, coreEnd: 18,
        numPasses: 3, gateRank: 8, attnResBlockSize: 6,
        slidingLayers: (0..<24).map { ($0 + 1) % 6 != 0 },
        numHeads: 4, numKVHeads: 2, headDim: 32,
        slidingWindow: 8, relDim: 16, relExtent: 12,
        qkClip: 2.5, relClip: 1.5,
        logScaleFloor: 4, logScaleAlpha: 0.1, convKernel: 4,
        numExperts: 16, topK: 4, expertFF: 32, routeScale: 8.0,
        gateSoftcap: 2.0, upSoftcap: 3.0, finalLogitSoftcap: 4.0,
        vocabPadded: 320, vocabUnpadded: 300, rmsEps: 1e-6)
}

/// Deterministic synthetic weights for an `Inkling2ReferenceConfig`
/// (normally `.toy`), with every tensor of the real file present: dynamic
/// convolution taps and three residual queries on global layers, message
/// queries and four rank-`gateRank` controllers on core layers, a router
/// bias, a padded vocabulary.
///
/// Values are seeded per tensor name (`SeedTree`), so one tensor's values do
/// not depend on the shape of another. Weights are dense floats; the
/// quantized toy install (NVFP4 and Q8_0 bytes) is built from the same
/// streams once the decoders land.
public enum Inkling2ToyCheckpoint {

    public static let defaultSeed: UInt64 = 0x1A_2B_3C_4D

    public static func weights(config c: Inkling2ReferenceConfig = .toy,
                               seed: UInt64 = defaultSeed) -> Inkling2ReferenceWeights {
        let tree = SeedTree(seed)
        func vector(_ name: String, _ count: Int, _ lo: Float, _ hi: Float) -> [Float] {
            var rng = tree.key(name)
            return (0..<count).map { _ in rng.uniform(lo, hi) }
        }
        func matrix(_ name: String, _ rows: Int, _ cols: Int, _ amplitude: Float) -> Inkling2DenseMatrix {
            Inkling2DenseMatrix(rows: rows, cols: cols, values: vector(name, rows * cols, -amplitude, amplitude))
        }
        func norm(_ name: String) -> [Float] { vector(name, c.hiddenSize, 0.8, 1.2) }

        let d = c.hiddenSize
        let k = c.convKernel
        var layers: [Inkling2ReferenceLayerWeights] = []
        for i in 0..<c.numLayers {
            let p = "blk.\(i)."
            let sliding = c.slidingLayers[i]
            let queries = i == 0 ? 0 : (sliding ? 1 : 3)
            layers.append(Inkling2ReferenceLayerWeights(
                wq: matrix(p + "attn_q", c.qDim, d, 0.12),
                wk: matrix(p + "attn_k", c.kvDim, d, 0.17),
                wv: matrix(p + "attn_v", c.kvDim, d, 0.12),
                wo: matrix(p + "attn_output", d, c.qDim, 0.12),
                outputGate: matrix(p + "attn_gate", c.qDim, d, 0.15),
                rel: matrix(p + "attn_rel", c.numHeads * c.relDim, d, 0.12),
                sinkProj: matrix(p + "attn_sink_proj", c.numHeads, d, 0.15),
                qClipScale: vector(p + "attn_q_clip", c.qDim, 0.6, 1.4),
                relClipScale: vector(p + "attn_rel_clip", c.numHeads * c.relDim, 0.6, 1.4),
                relBias: vector(p + "attn_rel_bias", c.relativeExtent(layer: i) * c.relDim, -1, 1),
                kConv: vector(p + "attn_k_conv", c.kvDim * k, -0.5, 0.5),
                vConv: vector(p + "attn_v_conv", c.kvDim * k, -0.5, 0.5),
                attnPostConv: vector(p + "attn_post_conv", d * k, -0.5, 0.5),
                ffnPostConv: vector(p + "ffn_post_conv", d * k, -0.5, 0.5),
                attnNorm: sliding ? norm(p + "attn_norm") : nil,
                qInputNorm: sliding ? nil : norm(p + "attn_q_inp_norm"),
                kInputNorm: sliding ? nil : norm(p + "attn_k_inp_norm"),
                vInputNorm: sliding ? nil : norm(p + "attn_v_inp_norm"),
                kConvProj: sliding ? nil : matrix(p + "attn_k_conv_proj", c.numKVHeads * k, d, 0.04),
                vConvProj: sliding ? nil : matrix(p + "attn_v_conv_proj", c.numKVHeads * k, d, 0.04),
                attnResQueries: (0..<queries).map { vector(p + "attn_res_query.\($0)", d, -1, 1) },
                ffnResQuery: vector(p + "ffn_res_query", d, -1, 1),
                ffnNorm: norm(p + "ffn_norm"),
                router: matrix(p + "ffn_gate_inp", c.numExperts, d, 0.2),
                routerBias: vector(p + "exp_probs_b", c.numExperts, -0.15, 0.15),
                expertGateUp: (0..<c.numExperts).map {
                    matrix(p + "ffn_gate_up_exps.\($0)", 2 * c.expertFF, d, 0.4)
                },
                expertDown: (0..<c.numExperts).map {
                    matrix(p + "ffn_down_exps.\($0)", d, c.expertFF, 0.15)
                }))
        }
        let loopLayers = (c.coreStart..<c.coreEnd).map { i -> Inkling2ReferenceLoopWeights in
            let p = "blk.\(i).loop_"
            return Inkling2ReferenceLoopWeights(
                messageQueries: (0..<4).map { vector(p + "msg_query.\($0)", d, -1, 1) },
                controlDown: (0..<4).map { matrix(p + "ctrl_down.\($0)", c.gateRank, 2 * d, 0.15) },
                controlGain: (0..<4).map { matrix(p + "ctrl_gain.\($0)", d, c.gateRank, 0.3) },
                controlForget: (0..<4).map { matrix(p + "ctrl_forget.\($0)", d, c.gateRank, 0.3) })
        }
        return Inkling2ReferenceWeights(
            embedding: matrix("token_embd", c.vocabPadded, d, 1.0),
            embeddingNorm: norm("token_embd_norm"),
            outputResQuery: vector("output_res_query", d, -1, 1),
            outputNorm: norm("output_norm"),
            output: matrix("output", c.vocabPadded, d, 0.45),
            layers: layers,
            loopLayers: loopLayers)
    }

    /// `weights` with every message gate silenced (gain and forget
    /// projections zero, so each controller returns its input unchanged).
    /// Every pass then computes the same function of the same input, which
    /// tests the restart rule and the per-pass state separately from the
    /// message path.
    public static func silencingMessageGates(_ weights: Inkling2ReferenceWeights) -> Inkling2ReferenceWeights {
        var w = weights
        w.loopLayers = w.loopLayers.map { l in
            func zero(_ m: any Inkling2ReferenceMatrix) -> any Inkling2ReferenceMatrix {
                Inkling2DenseMatrix(rows: m.rows, cols: m.cols, values: [Float](repeating: 0, count: m.rows * m.cols))
            }
            return Inkling2ReferenceLoopWeights(messageQueries: l.messageQueries,
                                                controlDown: l.controlDown,
                                                controlGain: l.controlGain.map(zero),
                                                controlForget: l.controlForget.map(zero))
        }
        return w
    }

    /// A deterministic token sequence inside the toy's unpadded vocabulary.
    public static func tokens(count: Int, config c: Inkling2ReferenceConfig = .toy,
                              seed: UInt64 = 7) -> [Int32] {
        var rng = SplitMix64(seed: seed)
        return (0..<count).map { _ in Int32(rng.next() % UInt64(c.vocabUnpadded)) }
    }
}
