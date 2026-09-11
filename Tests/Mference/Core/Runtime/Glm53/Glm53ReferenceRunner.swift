import Foundation
@testable import Mference

/// Float32 CPU transcription of PipeNetwork's GLM-5.3-Flash forward
/// (`glm53_flash_mlx/glm5_next/language.py`, the reference the goldens were
/// captured from), reading the toy checkpoint directly. One token per `step`,
/// the shape the Metal runner takes, with every intermediate the goldens
/// record captured under the goldens' own keys.
///
/// This is the fp32 tier of the family's parity gates: it establishes that
/// the *semantics* (per-channel KDA decay, the depthwise conv and its carried
/// tail, the absorbed NoPE latent attention, the pooled indexer with its tail
/// rule and zero-score ordering, the clamped sigmoid-routed MoE, the mHC
/// arithmetic and the stream-mean collapse) are transcribed correctly before
/// any FP16 kernel is judged against them.
final class Glm53ReferenceRunner {
    struct Capture {
        var floats: [String: [Float]] = [:]
        var ints: [String: [Int]] = [:]
        var selections: [String: [Int]?] = [:]
        var poolVisible: [String: [Bool]] = [:]
        var margins: [String: Float] = [:]
    }

    let checkpoint: Glm53ToyCheckpoint
    let config: ArchConfig
    private(set) var position = 0
    var capture = Capture()
    var recording = true
    /// Optional teacher forcing: when set, the reference's own values replace
    /// the oracle's at the layer boundaries (`layerNN.stream_in`) and at the
    /// cache appends (`dsa_latent_new`, `idx_k_new`, `idx_gate_new`), so each
    /// layer is judged on the reference's inputs rather than on the oracle's
    /// own accumulated fp32 drift. The keys are the goldens' local keys.
    var anchor: ((String) -> [Float]?)?

    private let hidden: Int
    private let hc: Int
    private let eps: Float
    private let hcEps: Float
    private let sinkhorn: Int
    private let kdaHeads: Int
    private let kdaDim: Int
    private let convK: Int
    private let numHeads: Int
    private let qkDim: Int
    private let vDim: Int
    private let kvRank: Int
    private let qRank: Int
    private let idxHeads: Int
    private let idxDim: Int
    private let idxTopK: Int
    private let kPool: Int
    private let lowerBound: Float
    private let limit: Float

    private final class LayerState {
        var convTail: [[Float]] = []        // [K-1][3 * qkv]
        var kdaState: [[[Float]]] = []      // [H][Dv][Dk]
        var latents: [[Float]] = []         // [T][kvRank]
        var idxKeys: [[Float]] = []         // [T][idxDim]
        var idxGates: [[Float]] = []        // [T][idxDim]
    }
    private var layers: [LayerState] = []

    init(checkpoint: Glm53ToyCheckpoint, config: ArchConfig = .glm53Toy()) throws {
        self.checkpoint = checkpoint
        self.config = config
        hidden = config.hiddenSize
        hc = config.hyperConnections.mult
        eps = Float(config.glm53.rmsNormEps)
        hcEps = Float(config.hyperConnections.eps)
        sinkhorn = config.hyperConnections.sinkhornIters
        kdaHeads = config.linearAttention.numVHeads
        kdaDim = config.linearAttention.keyHeadDim
        convK = config.linearAttention.convKernelSize
        numHeads = config.numHeads
        qkDim = config.glm53.qkNopeHeadDim
        vDim = config.glm53.vHeadDim
        kvRank = config.glm53.kvLoraRank
        qRank = config.compressedAttention.qLoraRank
        idxHeads = config.compressedAttention.indexNHeads
        idxDim = config.compressedAttention.indexHeadDim
        idxTopK = config.compressedAttention.indexTopK
        kPool = config.glm53.indexKPool
        lowerBound = Float(config.glm53.kdaGateLowerBound)
        limit = Float(config.swigluLimit)
        reset()
    }

    func reset() {
        position = 0
        layers = (0..<config.numLayers).map { _ in LayerState() }
        for L in 0..<config.numLayers where config.layerIsKDA(L) {
            let qkv = 3 * kdaHeads * kdaDim
            layers[L].convTail = Array(repeating: [Float](repeating: 0, count: qkv), count: convK - 1)
            layers[L].kdaState = Array(repeating: Array(repeating: [Float](repeating: 0, count: kdaDim),
                                                        count: kdaDim), count: kdaHeads)
        }
    }

    // MARK: - Helpers

    private func name(_ L: Int, _ suffix: String) -> String {
        "language_model.model.layers.\(L).\(suffix)"
    }

    private func matvec(_ m: Glm53ToyCheckpoint.Matrix, _ x: [Float]) -> [Float] {
        precondition(m.cols == x.count, "matvec \(m.rows)x\(m.cols) with \(x.count)")
        var y = [Float](repeating: 0, count: m.rows)
        for r in 0..<m.rows {
            var acc: Float = 0
            let row = m.row(r)
            var i = row.startIndex
            for c in 0..<m.cols { acc += row[i] * x[c]; i += 1 }
            y[r] = acc
        }
        return y
    }

    private func rmsNorm(_ x: [Float], weight: [Float], eps e: Float) -> [Float] {
        var ss: Float = 0
        for v in x { ss += v * v }
        let inv = 1 / (ss / Float(x.count) + e).squareRoot()
        return (0..<x.count).map { x[$0] * inv * weight[$0] }
    }

    private func layerNorm(_ x: [Float], weight: [Float], bias: [Float], eps e: Float) -> [Float] {
        var mean: Float = 0
        for v in x { mean += v }
        mean /= Float(x.count)
        var variance: Float = 0
        for v in x { variance += (v - mean) * (v - mean) }
        variance /= Float(x.count)
        let inv = 1 / (variance + e).squareRoot()
        return (0..<x.count).map { (x[$0] - mean) * inv * weight[$0] + bias[$0] }
    }

    private func l2norm(_ x: [Float]) -> [Float] {
        var ss: Float = 0
        for v in x { ss += v * v }
        let inv = 1 / (ss + 1e-6).squareRoot()
        return x.map { $0 * inv }
    }

    private func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }
    private func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }

    private func record(_ key: String, _ values: [Float]) {
        if recording { capture.floats[key] = values }
    }

    private func layerKey(_ L: Int) -> String { String(format: "layer%02d.", L) }

    // MARK: - Hyper-connections

    private struct Mixes { var pre: [Float]; var post: [Float]; var comb: [Float] }

    private func hcMixes(streams: [[Float]], fn: Glm53ToyCheckpoint.Matrix,
                         base: [Float], scale: [Float]) -> Mixes {
        // Unweighted RMSNorm over the flattened streams, then the mix.
        let flat = streams.flatMap { $0 }
        var ss: Float = 0
        for v in flat { ss += v * v }
        let inv = 1 / (ss / Float(flat.count) + eps).squareRoot()
        let z = flat.map { $0 * inv }
        let mix = matvec(fn, z)
        var pre = [Float](repeating: 0, count: hc)
        var post = [Float](repeating: 0, count: hc)
        for i in 0..<hc {
            pre[i] = sigmoid(mix[i] * scale[0] + base[i]) + hcEps
            post[i] = 2 * sigmoid(mix[hc + i] * scale[1] + base[hc + i])
        }
        var comb = [Float](repeating: 0, count: hc * hc)
        for r in 0..<hc {
            var mx: Float = -.infinity
            for c in 0..<hc {
                comb[r * hc + c] = mix[2 * hc + r * hc + c] * scale[2] + base[2 * hc + r * hc + c]
                mx = max(mx, comb[r * hc + c])
            }
            var sum: Float = 0
            for c in 0..<hc { comb[r * hc + c] = expf(comb[r * hc + c] - mx); sum += comb[r * hc + c] }
            for c in 0..<hc { comb[r * hc + c] = comb[r * hc + c] / sum + hcEps }
        }
        func normalizeColumns() {
            for c in 0..<hc {
                var col: Float = 0
                for r in 0..<hc { col += comb[r * hc + c] }
                for r in 0..<hc { comb[r * hc + c] /= (col + hcEps) }
            }
        }
        func normalizeRows() {
            for r in 0..<hc {
                var row: Float = 0
                for c in 0..<hc { row += comb[r * hc + c] }
                for c in 0..<hc { comb[r * hc + c] /= (row + hcEps) }
            }
        }
        normalizeColumns()
        for _ in 1..<max(sinkhorn, 1) { normalizeRows(); normalizeColumns() }
        return Mixes(pre: pre, post: post, comb: comb)
    }

    private func collapse(_ streams: [[Float]], _ pre: [Float]) -> [Float] {
        var x = [Float](repeating: 0, count: hidden)
        for j in 0..<hc { for d in 0..<hidden { x[d] += pre[j] * streams[j][d] } }
        return x
    }

    private func expand(_ y: [Float], residual: [[Float]], post: [Float], comb: [Float]) -> [[Float]] {
        var out = Array(repeating: [Float](repeating: 0, count: hidden), count: hc)
        for k in 0..<hc {
            for d in 0..<hidden {
                var acc = post[k] * y[d]
                for j in 0..<hc { acc += comb[j * hc + k] * residual[j][d] }
                out[k][d] = acc
            }
        }
        return out
    }

    // MARK: - Kimi Delta Attention

    private func kdaAttention(layer L: Int, x: [Float]) throws -> [Float] {
        let key = layerKey(L)
        let st = layers[L]
        let q = matvec(try checkpoint.matrix(name(L, "self_attn.q_proj")), x)
        let k = matvec(try checkpoint.matrix(name(L, "self_attn.k_proj")), x)
        let v = matvec(try checkpoint.matrix(name(L, "self_attn.v_proj")), x)
        let mixed = q + k + v
        record(key + "kda_mixed", mixed)
        // Depthwise causal conv over [tail rows | mixed], then SiLU.
        let conv = try checkpoint.floats(name(L, "self_attn.conv1d.weight"))   // [C, K, 1]
        let C = mixed.count
        var convOut = [Float](repeating: 0, count: C)
        for c in 0..<C {
            var acc: Float = 0
            for t in 0..<convK {
                let input = t < convK - 1 ? st.convTail[t][c] : mixed[c]
                acc += conv[c * convK + t] * input
            }
            convOut[c] = silu(acc)
        }
        st.convTail.removeFirst()
        st.convTail.append(mixed)
        record(key + "kda_conv_out", convOut)

        let D = kdaDim
        let qkv = kdaHeads * D
        let a = matvec(try checkpoint.matrix(name(L, "self_attn.forget_gate.f_b_proj")),
                       matvec(try checkpoint.matrix(name(L, "self_attn.forget_gate.f_a_proj")), x))
        let aLog = try checkpoint.floats(name(L, "self_attn.forget_gate.A_log"))
        let dtBias = try checkpoint.floats(name(L, "self_attn.forget_gate.dt_bias"))
        let bRaw = matvec(try checkpoint.matrix(name(L, "self_attn.b_proj")), x)
        let gate = matvec(try checkpoint.matrix(name(L, "self_attn.g_b_proj")),
                          matvec(try checkpoint.matrix(name(L, "self_attn.g_a_proj")), x))
        record(key + "kda_gate", gate)

        var qN: [Float] = [], kN: [Float] = [], vAll: [Float] = []
        var decay = [Float](repeating: 0, count: qkv)
        var beta = [Float](repeating: 0, count: kdaHeads)
        let scale = 1 / Float(D).squareRoot()
        for h in 0..<kdaHeads {
            let qh = l2norm(Array(convOut[(h * D)..<((h + 1) * D)])).map { $0 * scale }
            let kh = l2norm(Array(convOut[(qkv + h * D)..<(qkv + (h + 1) * D)]))
            qN += qh; kN += kh
            let vStart = 2 * qkv + h * D
            let vEnd = vStart + D
            vAll.append(contentsOf: convOut[vStart..<vEnd])
            let expA = expf(aLog[h])
            for d in 0..<D {
                decay[h * D + d] = expf(lowerBound * sigmoid(expA * (a[h * D + d] + dtBias[h * D + d])))
            }
            beta[h] = sigmoid(bRaw[h])
        }
        record(key + "kda_q", qN); record(key + "kda_k", kN); record(key + "kda_v", vAll)
        record(key + "kda_decay", decay); record(key + "kda_beta", beta)

        // Recurrence, state [H][Dv][Dk]: decay along dk, delta rule, read-out.
        var y = [Float](repeating: 0, count: qkv)
        for h in 0..<kdaHeads {
            var s = st.kdaState[h]
            let qh = qN[(h * D)..<((h + 1) * D)], kh = kN[(h * D)..<((h + 1) * D)]
            for dv in 0..<D {
                var kv: Float = 0
                for dk in 0..<D {
                    s[dv][dk] *= decay[h * D + dk]
                    kv += s[dv][dk] * kh[kh.startIndex + dk]
                }
                let delta = (vAll[h * D + dv] - kv) * beta[h]
                var out: Float = 0
                for dk in 0..<D {
                    s[dv][dk] += kh[kh.startIndex + dk] * delta
                    out += s[dv][dk] * qh[qh.startIndex + dk]
                }
                y[h * D + dv] = out
            }
            st.kdaState[h] = s
        }
        record(key + "kda_y", y)

        // Sigmoid-gated per-head RMSNorm, then o_proj.
        let oNorm = try checkpoint.floats(name(L, "self_attn.o_norm.weight"))
        var gated = [Float](repeating: 0, count: qkv)
        for h in 0..<kdaHeads {
            let normed = rmsNorm(Array(y[(h * D)..<((h + 1) * D)]), weight: oNorm, eps: eps)
            for d in 0..<D { gated[h * D + d] = normed[d] * sigmoid(gate[h * D + d]) }
        }
        return matvec(try checkpoint.matrix(name(L, "self_attn.o_proj")), gated)
    }

    // MARK: - NoPE latent sparse attention with the pooled indexer

    private func sparseAttention(layer L: Int, x: [Float]) throws -> [Float] {
        let key = layerKey(L)
        let st = layers[L]
        let qr = rmsNorm(matvec(try checkpoint.matrix(name(L, "self_attn.q_a_proj")), x),
                         weight: try checkpoint.floats(name(L, "self_attn.q_a_layernorm.weight")), eps: eps)
        record(key + "dsa_qr", qr)
        record(key + "self_attn_q_a_layernorm_out", qr)
        let q = matvec(try checkpoint.matrix(name(L, "self_attn.q_b_proj")), qr)   // [H * qkDim]
        record(key + "dsa_q", q)
        let latent = rmsNorm(matvec(try checkpoint.matrix(name(L, "self_attn.kv_a_proj_with_mqa")), x),
                             weight: try checkpoint.floats(name(L, "self_attn.kv_a_layernorm.weight")), eps: eps)
        record(key + "dsa_latent_new", latent)
        record(key + "self_attn_kv_a_layernorm_out", latent)
        st.latents.append(anchor?(key + "dsa_latent_new") ?? latent)

        let selected = try indexer(layer: L, x: x, qr: qr)
        let T = st.latents.count
        let attended: [Int] = selected ?? Array(0..<T)

        var oAll: [Float] = []
        var qLatAll: [Float] = []
        let scale = 1 / Float(qkDim).squareRoot()
        for h in 0..<numHeads {
            let qh = Array(q[(h * qkDim)..<((h + 1) * qkDim)])
            let qLat = matvec(try checkpoint.matrix(name(L, "self_attn.embed_q"), slab: h), qh)   // [kvRank]
            qLatAll += qLat
            var scores = attended.map { t -> Float in
                var dot: Float = 0
                for i in 0..<kvRank { dot += qLat[i] * st.latents[t][i] }
                return dot * scale
            }
            let mx = scores.max() ?? 0
            var sum: Float = 0
            for i in scores.indices { scores[i] = expf(scores[i] - mx); sum += scores[i] }
            var oLat = [Float](repeating: 0, count: kvRank)
            for (i, t) in attended.enumerated() {
                let p = scores[i] / sum
                for j in 0..<kvRank { oLat[j] += p * st.latents[t][j] }
            }
            oAll += matvec(try checkpoint.matrix(name(L, "self_attn.unembed_out"), slab: h), oLat)  // [vDim]
        }
        record(key + "dsa_q_latent", qLatAll)
        return matvec(try checkpoint.matrix(name(L, "self_attn.o_proj")), oAll)
    }

    /// The pooled lightning indexer for the newest query. Returns nil while the
    /// cache holds at most `indexTopK` tokens (the dense bypass), else the
    /// sorted token set: the top `indexTopK / kPool` visible complete pools
    /// (stable descending order, so equal scores keep the lower pool index)
    /// expanded to their tokens, plus the incomplete tail.
    private func indexer(layer L: Int, x: [Float], qr: [Float]) throws -> [Int]? {
        let key = layerKey(L)
        let st = layers[L]
        let kRaw = matvec(try checkpoint.matrix(name(L, "self_attn.indexer.wk")), x)
        let k = layerNorm(kRaw, weight: try checkpoint.floats(name(L, "self_attn.indexer.k_norm.weight")),
                          bias: try checkpoint.floats(name(L, "self_attn.indexer.k_norm.bias")),
                          eps: Float(config.glm53.indexerKNormEps))
        let gate = matvec(try checkpoint.matrix(name(L, "self_attn.indexer.index_kpool_compress_gate")), x)
        record(key + "idx_k_new", k)
        record(key + "self_attn_indexer_k_norm_out", k)
        record(key + "idx_gate_new", gate)
        st.idxKeys.append(anchor?(key + "idx_k_new") ?? k)
        st.idxGates.append(anchor?(key + "idx_gate_new") ?? gate)
        let T = st.idxKeys.count
        if recording { capture.ints[key + "idx_visible"] = [T] }
        guard T > idxTopK else {
            if recording { capture.selections[key + "idx_selected"] = .some(nil) }
            return nil
        }
        // Pooled keys: per-channel softmax over `gate + ape` within each
        // complete group of kPool consecutive tokens.
        let ape = try checkpoint.floats(name(L, "self_attn.indexer.index_kpool_compress_ape"))  // [kPool, idxDim]
        let complete = T / kPool
        let P = (T + kPool - 1) / kPool
        var poolKeys: [[Float]] = []
        var visible: [Bool] = []
        for j in 0..<P {
            let count = min(kPool, T - j * kPool)
            var pooled = [Float](repeating: 0, count: idxDim)
            for d in 0..<idxDim {
                var logits = [Float](repeating: -1e30, count: kPool)
                for c in 0..<count { logits[c] = st.idxGates[j * kPool + c][d] + ape[c * idxDim + d] }
                let mx = logits.max()!
                var sum: Float = 0
                var w = [Float](repeating: 0, count: kPool)
                for c in 0..<kPool { w[c] = expf(logits[c] - mx); sum += w[c] }
                for c in 0..<count { pooled[d] += (w[c] / sum) * st.idxKeys[j * kPool + c][d] }
            }
            poolKeys.append(pooled)
            visible.append(j < complete)
        }
        record(key + "idx_pool_keys", poolKeys.flatMap { $0 })
        let qIdx = matvec(try checkpoint.matrix(name(L, "self_attn.indexer.wq_b")), qr)     // [idxHeads * idxDim]
        let weights = matvec(try checkpoint.matrix(name(L, "self_attn.indexer.weights_proj")), x)
            .map { $0 / Float(idxHeads).squareRoot() }
        record(key + "idx_q", qIdx)
        record(key + "idx_weights", weights)
        let headScale = 1 / Float(idxDim).squareRoot()
        var scores = [Float](repeating: 0, count: P)
        for j in 0..<P {
            var s: Float = 0
            for h in 0..<idxHeads {
                var dot: Float = 0
                for d in 0..<idxDim { dot += qIdx[h * idxDim + d] * poolKeys[j][d] }
                s += weights[h] * max(dot * headScale, 0)
            }
            scores[j] = s
        }
        record(key + "idx_scores", scores)
        if recording { capture.poolVisible[key + "idx_pool_visible"] = visible }
        let selectK = min(idxTopK / kPool, P)
        // Stable descending: ties keep the lower pool index (mx.argsort).
        let order = (0..<P).filter { visible[$0] }.sorted { a, b in
            scores[a] != scores[b] ? scores[a] > scores[b] : a < b
        }
        let chosen = Array(order.prefix(selectK))
        if order.count > selectK, recording {
            capture.margins[key + "indexer"] = scores[order[selectK - 1]] - scores[order[selectK]]
        }
        var tokens = Set<Int>()
        for j in chosen { for c in 0..<kPool { tokens.insert(j * kPool + c) } }
        let tail = T - complete * kPool
        if config.glm53.indexKPoolAlwaysSelectTail { for t in (T - tail)..<T { tokens.insert(t) } }
        let sorted = tokens.sorted()
        if recording { capture.selections[key + "idx_selected"] = .some(sorted) }
        return sorted
    }

    // MARK: - FFN

    private func clampedSwiGLU(gate: [Float], up: [Float]) -> [Float] {
        (0..<gate.count).map { i in
            let g = min(gate[i], limit)
            let u = min(max(up[i], -limit), limit)
            return silu(g) * u
        }
    }

    private func denseFFN(layer L: Int, x: [Float]) throws -> [Float] {
        let g = matvec(try checkpoint.matrix(name(L, "mlp.gate_proj")), x)
        let u = matvec(try checkpoint.matrix(name(L, "mlp.up_proj")), x)
        return matvec(try checkpoint.matrix(name(L, "mlp.down_proj")), clampedSwiGLU(gate: g, up: u))
    }

    private func moe(layer L: Int, x: [Float]) throws -> [Float] {
        let key = layerKey(L)
        let logits = matvec(try checkpoint.matrix(name(L, "mlp.gate")), x)
        let bias = try checkpoint.floats(name(L, "mlp.gate.e_score_correction_bias"))
        let scores = logits.map { sigmoid($0) }
        record(key + "router_logits", logits)
        record(key + "router_scores", scores)
        let biased = (0..<scores.count).map { scores[$0] + bias[$0] }
        let order = (0..<scores.count).sorted { biased[$0] != biased[$1] ? biased[$0] > biased[$1] : $0 < $1 }
        let k = config.topKExperts
        let chosen = Array(order.prefix(k))
        if recording {
            capture.ints[key + "router_indices"] = chosen
            capture.margins[key + "router"] = biased[order[k - 1]] - biased[order[k]]
        }
        var sum: Float = 0
        for e in chosen { sum += scores[e] }
        let weights = chosen.map { scores[$0] / sum * Float(config.routedScalingFactor) }
        record(key + "router_weights", weights)

        var y = [Float](repeating: 0, count: hidden)
        for (i, e) in chosen.enumerated() {
            let g = matvec(try checkpoint.matrix(name(L, "mlp.switch_mlp.gate_proj"), slab: e), x)
            let u = matvec(try checkpoint.matrix(name(L, "mlp.switch_mlp.up_proj"), slab: e), x)
            let out = matvec(try checkpoint.matrix(name(L, "mlp.switch_mlp.down_proj"), slab: e),
                             clampedSwiGLU(gate: g, up: u))
            for d in 0..<hidden { y[d] += weights[i] * out[d] }
        }
        let sg = matvec(try checkpoint.matrix(name(L, "mlp.shared_experts.gate_proj")), x)
        let su = matvec(try checkpoint.matrix(name(L, "mlp.shared_experts.up_proj")), x)
        let shared = matvec(try checkpoint.matrix(name(L, "mlp.shared_experts.down_proj")),
                            clampedSwiGLU(gate: sg, up: su))
        record(key + "shared_out", shared)
        for d in 0..<hidden { y[d] += shared[d] }
        return y
    }

    // MARK: - One token

    /// Runs one token at the current position and returns the logits.
    func step(token: Int) throws -> [Float] {
        capture = Capture()
        let embed = try checkpoint.matrix("language_model.model.embed_tokens")
        let row = Array(embed.row(token))
        record("embed_out", row)
        var streams = Array(repeating: row, count: hc)

        for L in 0..<config.numLayers {
            let key = layerKey(L)
            if let forced = anchor?(key + "stream_in"), forced.count == hc * hidden {
                streams = (0..<hc).map { Array(forced[($0 * hidden)..<(($0 + 1) * hidden)]) }
            }
            record(key + "stream_in", streams.flatMap { $0 })
            let attnMix = hcMixes(streams: streams,
                                  fn: try checkpoint.matrix(name(L, "attn_hc.fn")),
                                  base: try checkpoint.floats(name(L, "attn_hc.base")),
                                  scale: try checkpoint.floats(name(L, "attn_hc.scale")))
            record(key + "attn_hc_pre", attnMix.pre); record(key + "attn_hc_post", attnMix.post)
            record(key + "attn_hc_comb", attnMix.comb)
            let xc = collapse(streams, attnMix.pre)
            record(key + "attn_collapsed", xc)
            let normed = rmsNorm(xc, weight: try checkpoint.floats(name(L, "input_layernorm.weight")), eps: eps)
            record(key + "input_layernorm_out", normed)
            let r = config.layerIsKDA(L)
                ? try kdaAttention(layer: L, x: normed)
                : try sparseAttention(layer: L, x: normed)
            record(key + "attn_out", r)
            streams = expand(r, residual: streams, post: attnMix.post, comb: attnMix.comb)

            let ffnMix = hcMixes(streams: streams,
                                 fn: try checkpoint.matrix(name(L, "ffn_hc.fn")),
                                 base: try checkpoint.floats(name(L, "ffn_hc.base")),
                                 scale: try checkpoint.floats(name(L, "ffn_hc.scale")))
            record(key + "ffn_hc_pre", ffnMix.pre); record(key + "ffn_hc_post", ffnMix.post)
            record(key + "ffn_hc_comb", ffnMix.comb)
            let fc = collapse(streams, ffnMix.pre)
            record(key + "ffn_collapsed", fc)
            let fn = rmsNorm(fc, weight: try checkpoint.floats(name(L, "post_attention_layernorm.weight")), eps: eps)
            record(key + "post_attention_layernorm_out", fn)
            let m = config.layerIsDenseFFN(L) ? try denseFFN(layer: L, x: fn) : try moe(layer: L, x: fn)
            record(key + "mlp_out", m)
            streams = expand(m, residual: streams, post: ffnMix.post, comb: ffnMix.comb)
            record(key + "stream_out", streams.flatMap { $0 })
        }

        var mean = [Float](repeating: 0, count: hidden)
        for j in 0..<hc { for d in 0..<hidden { mean[d] += streams[j][d] / Float(hc) } }
        let normed = rmsNorm(mean, weight: try checkpoint.floats("language_model.model.norm.weight"), eps: eps)
        record("final_norm_out", normed)
        record("model_norm_out", normed)
        let logits = matvec(try checkpoint.matrix("language_model.lm_head"), normed)
        record("logits", logits)
        position += 1
        return logits
    }

    // MARK: - State readbacks (the goldens' `*.final.*` snapshots)

    func kdaState(layer L: Int) -> [Float] { layers[L].kdaState.flatMap { $0.flatMap { $0 } } }
    func kdaConvTail(layer L: Int) -> [Float] { layers[L].convTail.flatMap { $0 } }
    func latentCache(layer L: Int) -> [Float] { layers[L].latents.flatMap { $0 } }
    /// The reference packs `[k | gate | valid]` per token.
    func indexerPackedCache(layer L: Int) -> [Float] {
        let st = layers[L]
        var out: [Float] = []
        for t in 0..<st.idxKeys.count { out += st.idxKeys[t]; out += st.idxGates[t]; out.append(1) }
        return out
    }
}
