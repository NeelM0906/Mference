import Foundation

/// Per-head clips and norms of the Inkling 2 attention block, FP64 inside.
///
/// - Queries: `clip_t(scale_q ⊙ (W_q x))`, per query head, t = 22.627.
/// - Relative states: `clip_8(scale_r ⊙ (W_r x))`, per query head.
/// - Keys: `clip_t(sconv(W_k x))`, per KV head, no scale.
/// - Values: `rms_noweight(sconv(W_v x))`, per KV head, eps 1e-6.
///
/// `clip_t(v) = v * min(1, t / (||v||_2 + 1e-6))`
/// (transformers `Inkling2LoopL2Clip`; llama.cpp `inkling2::graph::clip`).
public enum Inkling2NormRef {
    /// Clips every `width`-wide vector of `x` (`[rows, heads, width]`) after
    /// an optional per-(head, lane) scale (`[heads, width]`).
    public static func l2Clip(_ x: [Float], rows: Int, heads: Int, width: Int,
                              scale: [Float]?, threshold: Float,
                              eps: Float = Inkling2RefShape.clipEps) -> [Float] {
        precondition(x.count == rows * heads * width)
        if let scale { precondition(scale.count == heads * width) }
        var out = [Float](repeating: 0, count: x.count)
        var scaled = [Double](repeating: 0, count: width)
        for row in 0..<rows {
            for h in 0..<heads {
                let base = (row * heads + h) * width
                var sumSquares = 0.0
                for d in 0..<width {
                    let value = Double(x[base + d]) * Double(scale?[h * width + d] ?? 1)
                    scaled[d] = value
                    sumSquares += value * value
                }
                let factor = min(1.0, Double(threshold) / (sumSquares.squareRoot() + Double(eps)))
                for d in 0..<width {
                    out[base + d] = Float(scaled[d] * factor)
                }
            }
        }
        return out
    }

    /// Weightless RMS norm of every `width`-wide vector (`[rows, heads, width]`).
    public static func unweightedRMS(_ x: [Float], rows: Int, heads: Int, width: Int,
                                     eps: Float = Inkling2RefShape.rmsEps) -> [Float] {
        precondition(x.count == rows * heads * width)
        var out = [Float](repeating: 0, count: x.count)
        for row in 0..<rows {
            for h in 0..<heads {
                let base = (row * heads + h) * width
                var sumSquares = 0.0
                for d in 0..<width { sumSquares += Double(x[base + d]) * Double(x[base + d]) }
                let inv = 1.0 / (sumSquares / Double(width) + Double(eps)).squareRoot()
                for d in 0..<width { out[base + d] = Float(Double(x[base + d]) * inv) }
            }
        }
        return out
    }

    /// Weighted RMS norm of each `width`-wide row (`[rows, width]`); the
    /// attention input norms and the MoE input norm.
    public static func weightedRMS(_ x: [Float], rows: Int, width: Int, weight: [Float],
                                   eps: Float = Inkling2RefShape.rmsEps) -> [Float] {
        precondition(x.count == rows * width && weight.count == width)
        var out = [Float](repeating: 0, count: x.count)
        for row in 0..<rows {
            let base = row * width
            var sumSquares = 0.0
            for d in 0..<width { sumSquares += Double(x[base + d]) * Double(x[base + d]) }
            let inv = 1.0 / (sumSquares / Double(width) + Double(eps)).squareRoot()
            for d in 0..<width {
                out[base + d] = Float(Double(x[base + d]) * inv * Double(weight[d]))
            }
        }
        return out
    }
}
