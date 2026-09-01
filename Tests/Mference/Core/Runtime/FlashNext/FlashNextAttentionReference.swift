import Foundation
@testable import Mference

/// Flash-Next gated full attention (`Qwen4ExpTextAttention`) in float32, over an
/// explicit selected KV set.
///
/// A transcription of `FlashNextReferenceRunner.attention`, discharged by
/// `FlashNextIndexerReferenceTieBackTests` against the runner's captured
/// `block_out`. Four details the shape alone will not tell you:
///
/// * `q_proj` packs **per head** as `[query | gate]`, 2 x headDim rows per head.
///   It is not a global `[all queries | all gates]` split.
/// * q and k are per-head RMSNormed with `(1 + w)`-baked weights and then
///   partially roped; **v is neither normed nor roped**.
/// * The sigmoid output gate multiplies the flattened per-head attention output
///   **before** `o_proj`, and the gate half is never normed or rotated.
/// * Softmax runs over the selected positions only, in their sorted order, with
///   a max subtraction — the mask is expressed by which positions are present,
///   never by dropping KV from the cache.
enum FlashNextAttentionReference {

    struct Geometry {
        let numHeads: Int
        let numKVHeads: Int
        let headDim: Int
        let rotaryDim: Int
        let theta: Float
        let eps: Float
        var qDim: Int { numHeads * headDim }
        var kvDim: Int { numKVHeads * headDim }
    }

    struct Weights {
        let q: [Float]       // [2 * qDim, hidden]
        let k: [Float]       // [kvDim, hidden]
        let v: [Float]       // [kvDim, hidden]
        let o: [Float]       // [hidden, qDim]
        let qNorm: [Float]   // [headDim]
        let kNorm: [Float]   // [headDim]
    }

    struct Result {
        /// `[rows * hidden]` — the block output, post `o_proj`.
        let out: [Float]
        /// `[rows * qDim]` — pre-gate, pre-projection attention output.
        let attention: [Float]
        /// The KV cache after this call, `[.., kvDim]` each.
        let keys: [Float]
        let values: [Float]
    }

    /// One attention layer over `rows` tokens starting at `startPosition`, with
    /// the KV cache carried in and out so cached decode and full prefill share
    /// the path.
    static func run(x: [Float], hidden: Int, rows: Int, startPosition: Int,
                    w: Weights, selected: [[Int]],
                    keys carriedKeys: [Float], values carriedValues: [Float],
                    scale: Float, g: Geometry) -> Result {
        var keys = carriedKeys
        var values = carriedValues
        var queries = [Float](repeating: 0, count: rows * g.qDim)
        var gates = [Float](repeating: 0, count: rows * g.qDim)

        for t in 0..<rows {
            let packed = matVec(w.q, rows: 2 * g.qDim, cols: hidden,
                                x: x, xOffset: t * hidden)
            for h in 0..<g.numHeads {
                let src = h * 2 * g.headDim
                var head = rmsNorm(packed, offset: src, count: g.headDim,
                                   weight: w.qNorm, eps: g.eps)
                applyRoPE(&head, position: startPosition + t,
                          rotaryDim: g.rotaryDim, theta: g.theta)
                for d in 0..<g.headDim {
                    queries[t * g.qDim + h * g.headDim + d] = head[d]
                    gates[t * g.qDim + h * g.headDim + d] = packed[src + g.headDim + d]
                }
            }
            let kRow = matVec(w.k, rows: g.kvDim, cols: hidden,
                              x: x, xOffset: t * hidden)
            let vRow = matVec(w.v, rows: g.kvDim, cols: hidden,
                              x: x, xOffset: t * hidden)
            for h in 0..<g.numKVHeads {
                var head = rmsNorm(kRow, offset: h * g.headDim, count: g.headDim,
                                   weight: w.kNorm, eps: g.eps)
                applyRoPE(&head, position: startPosition + t,
                          rotaryDim: g.rotaryDim, theta: g.theta)
                keys.append(contentsOf: head)
            }
            values.append(contentsOf: vRow)
        }

        let groups = g.numHeads / g.numKVHeads
        var attention = [Float](repeating: 0, count: rows * g.qDim)
        for t in 0..<rows {
            let keep = selected[t]
            for h in 0..<g.numHeads {
                let kv = h / groups
                var scores = [Float](repeating: 0, count: keep.count)
                var maximum = -Float.greatestFiniteMagnitude
                for (i, position) in keep.enumerated() {
                    var dot: Float = 0
                    let base = (position * g.numKVHeads + kv) * g.headDim
                    for d in 0..<g.headDim {
                        dot += queries[t * g.qDim + h * g.headDim + d] * keys[base + d]
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
                    let base = (position * g.numKVHeads + kv) * g.headDim
                    for d in 0..<g.headDim {
                        attention[t * g.qDim + h * g.headDim + d] += p * values[base + d]
                    }
                }
            }
        }
        var gated = attention
        for i in 0..<gated.count { gated[i] *= sigmoid(gates[i]) }
        var out = [Float](repeating: 0, count: rows * hidden)
        for t in 0..<rows {
            let projected = matVec(w.o, rows: hidden, cols: g.qDim,
                                   x: gated, xOffset: t * g.qDim)
            for d in 0..<hidden { out[t * hidden + d] = projected[d] }
        }
        return Result(out: out, attention: gated, keys: keys, values: values)
    }

    // MARK: - Numeric helpers (the runner's exact forms)

    static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }

    static func matVec(_ w: [Float], rows: Int, cols: Int,
                       x: [Float], xOffset: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            var acc: Float = 0
            let base = r * cols
            for c in 0..<cols { acc += w[base + c] * x[xOffset + c] }
            out[r] = acc
        }
        return out
    }

    static func rmsNorm(_ x: [Float], offset: Int, count: Int,
                        weight: [Float], eps: Float) -> [Float] {
        FlashNextIndexerReference.rmsNorm(x, offset: offset, count: count,
                                          weight: weight, eps: eps)
    }

    static func applyRoPE(_ head: inout [Float], position: Int,
                          rotaryDim: Int, theta: Float) {
        FlashNextIndexerReference.applyRoPE(&head, position: position,
                                            rotaryDim: rotaryDim, theta: theta)
    }
}
