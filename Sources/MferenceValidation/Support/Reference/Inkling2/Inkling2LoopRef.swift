import Foundation

/// The looped core's cross-pass operations, FP64 inside.
///
/// Message retrieval (`Inkling2LoopMessageAttentionResidual`) and the block
/// attention residuals (`Inkling2LoopBlockAttentionResidual`, llama.cpp
/// `inkling2::graph::mix`) are the same depth mix: per token, stack the given
/// states, normalise each with a weightless RMS norm, and let each learned
/// query attend over them with scale `1 / sqrt(hidden)`; keys and values are
/// the normalised states. The message gate
/// (`Inkling2LoopMessageGate`, llama.cpp `control`) then mixes a message into
/// a layer input: `(1 - tanh(F z)) * x + tanh(G z) * m` with
/// `z = W_down [rms(x); rms(m)]`.
public enum Inkling2LoopRef {
    /// `states`: each `[rows, hidden]`; `queries`: `[queryCount, hidden]`.
    /// Result: `[rows, queryCount, hidden]`. The order of `states` does not
    /// change the result.
    public static func depthMix(states: [[Float]], queries: [Float], queryCount: Int,
                                rows: Int, hidden: Int,
                                eps: Float = Inkling2RefShape.rmsEps) -> [Float] {
        precondition(!states.isEmpty && queries.count == queryCount * hidden)
        for state in states { precondition(state.count == rows * hidden) }
        let scale = 1.0 / Double(hidden).squareRoot()
        var out = [Float](repeating: 0, count: rows * queryCount * hidden)
        var normalised = [[Double]](repeating: [Double](repeating: 0, count: hidden), count: states.count)
        for t in 0..<rows {
            for (n, state) in states.enumerated() {
                var sumSquares = 0.0
                for d in 0..<hidden { sumSquares += Double(state[t * hidden + d]) * Double(state[t * hidden + d]) }
                let inv = 1.0 / (sumSquares / Double(hidden) + Double(eps)).squareRoot()
                for d in 0..<hidden { normalised[n][d] = Double(state[t * hidden + d]) * inv }
            }
            for qi in 0..<queryCount {
                var scores = [Double](repeating: 0, count: states.count)
                for n in 0..<states.count {
                    var dot = 0.0
                    for d in 0..<hidden { dot += Double(queries[qi * hidden + d]) * normalised[n][d] }
                    scores[n] = dot * scale
                }
                let maximum = scores.max()!
                let weights = scores.map { exp($0 - maximum) }
                let total = weights.reduce(0, +)
                for d in 0..<hidden {
                    var acc = 0.0
                    for n in 0..<states.count { acc += weights[n] * normalised[n][d] }
                    out[(t * queryCount + qi) * hidden + d] = Float(acc / total)
                }
            }
        }
        return out
    }

    /// `x`, `message`: `[rows, hidden]`. Weights in GGUF row-major
    /// `[out][in]`: `down` `[rank][2 * hidden]`, `forget` and `gain`
    /// `[hidden][rank]` (the vendor's `forget_weight` and `gain_weight`).
    public static func messageGate(x: [Float], message: [Float],
                                   down: [Float], forget: [Float], gain: [Float],
                                   rows: Int, hidden: Int, rank: Int,
                                   eps: Float = Inkling2RefShape.rmsEps) -> [Float] {
        precondition(x.count == rows * hidden && message.count == rows * hidden)
        precondition(down.count == rank * 2 * hidden)
        precondition(forget.count == hidden * rank && gain.count == hidden * rank)
        var out = [Float](repeating: 0, count: x.count)
        var joined = [Double](repeating: 0, count: 2 * hidden)
        var z = [Double](repeating: 0, count: rank)
        for t in 0..<rows {
            for (half, source) in [x, message].enumerated() {
                var sumSquares = 0.0
                for d in 0..<hidden { sumSquares += Double(source[t * hidden + d]) * Double(source[t * hidden + d]) }
                let inv = 1.0 / (sumSquares / Double(hidden) + Double(eps)).squareRoot()
                for d in 0..<hidden { joined[half * hidden + d] = Double(source[t * hidden + d]) * inv }
            }
            for r in 0..<rank {
                var acc = 0.0
                for i in 0..<(2 * hidden) { acc += Double(down[r * 2 * hidden + i]) * joined[i] }
                z[r] = acc
            }
            for d in 0..<hidden {
                var f = 0.0, g = 0.0
                for r in 0..<rank {
                    f += Double(forget[d * rank + r]) * z[r]
                    g += Double(gain[d * rank + r]) * z[r]
                }
                let xv = Double(x[t * hidden + d]), mv = Double(message[t * hidden + d])
                out[t * hidden + d] = Float((1 - tanh(f)) * xv + tanh(g) * mv)
            }
        }
        return out
    }
}
