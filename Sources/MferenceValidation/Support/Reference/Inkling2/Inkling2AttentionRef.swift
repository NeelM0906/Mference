import Foundation

/// FP64-accumulated reference for Inkling 2 attention with the learned
/// relative bias, the per-query sink and both mask kinds.
///
/// Semantics (transformers `eager_attention_forward` with
/// `Inkling2LoopRelativePositionBias`; llama.cpp `inkling2` builds the same
/// logits with the sink folded into the bias):
///
///     tau_i      = global ? 1 + alpha * ln(max((pos_i + 1) / floor, 1)) : 1
///     d          = pos_i - pos_j                           (backward distance)
///     bias(h,i,j) = d < relExtent ? (tau_i r[i,h,:]) . P[d,:] : 0
///     logit(h,i,j) = (tau_i q[i,h,:]) . k[j,h/G,:] * scale + bias(h,i,j)
///     visible    = 0 <= d  (global)  or  0 <= d < window  (sliding)
///     p          = softmax over the visible logits plus one sink logit
///                  s[i,h] that has no value
///     out[i,h,:] = sum_j p_j v[j,h/G,:]
///
/// `q` and `r` come in already scaled and L2-clipped; tau (global layers
/// only) is applied here, which is the same product the vendor forms by
/// scaling `q` and `r` after their clips. The sink is not scaled by tau.
public enum Inkling2AttentionRef {
    /// The global layers' log position scaling.
    public struct LogScale: Sendable, Equatable {
        public var floor: Int
        public var alpha: Float

        public init(floor: Int = Inkling2RefShape.logScaleFloor,
                    alpha: Float = Inkling2RefShape.logScaleAlpha) {
            self.floor = floor
            self.alpha = alpha
        }

        /// `1 + alpha * ln(max((position + 1) / floor, 1))`.
        public func tau(position: Int) -> Double {
            let ratio = Double(position + 1) / Double(floor)
            return 1.0 + Double(alpha) * log(max(ratio, 1.0))
        }
    }

    /// Layouts:
    /// - `q`:    `[queryCount, numQHeads, headDim]`
    /// - `k`, `v`: `[keyCount, numKVHeads, headDim]`; row `n` holds logical
    ///   position `keyBase + n`. Every key a query can see must be present.
    /// - `rel`:  `[queryCount, numQHeads, dRel]`
    /// - `table`: `[relExtent, dRel]` (the GGUF `attn_rel_bias` layout:
    ///   distance-major, `dRel` contiguous values per distance)
    /// - `sink`: `[queryCount, numQHeads]`
    /// - result: `[queryCount, numQHeads, headDim]`
    ///
    /// Query row `i` sits at logical position `startPosition + i`.
    /// `window == nil` means global attention.
    public static func attend(
        q: [Float], k: [Float], v: [Float],
        rel: [Float], table: [Float], sink: [Float],
        queryCount: Int, startPosition: Int,
        keyBase: Int = 0,
        numQHeads: Int = Inkling2RefShape.numQHeads,
        numKVHeads: Int = Inkling2RefShape.numKVHeads,
        headDim: Int = Inkling2RefShape.headDim,
        dRel: Int = Inkling2RefShape.dRel,
        relExtent: Int = Inkling2RefShape.relExtent,
        window: Int?,
        logScale: LogScale?,
        scale: Float? = nil
    ) -> [Float] {
        precondition(numQHeads % numKVHeads == 0)
        precondition(q.count == queryCount * numQHeads * headDim)
        precondition(rel.count == queryCount * numQHeads * dRel)
        precondition(sink.count == queryCount * numQHeads)
        precondition(table.count == relExtent * dRel)
        precondition(k.count == v.count && k.count % (numKVHeads * headDim) == 0)
        let keyCount = k.count / (numKVHeads * headDim)
        let group = numQHeads / numKVHeads
        let scale = Double(scale ?? 1.0 / Float(headDim).squareRoot())
        let kvStride = numKVHeads * headDim

        var out = [Float](repeating: 0, count: q.count)
        var logits = [Double]()
        for i in 0..<queryCount {
            let position = startPosition + i
            let tau = logScale?.tau(position: position) ?? 1.0
            let first = window.map { max(0, position - $0 + 1) } ?? 0
            precondition(first >= keyBase && position - keyBase < keyCount,
                         "a visible key is missing from k/v")
            for h in 0..<numQHeads {
                let kvHead = h / group
                let qBase = (i * numQHeads + h) * headDim
                let rBase = (i * numQHeads + h) * dRel
                logits.removeAll(keepingCapacity: true)
                var maximum = Double(sink[i * numQHeads + h])
                for key in first...position {
                    let kBase = (key - keyBase) * kvStride + kvHead * headDim
                    var dot = 0.0
                    for d in 0..<headDim {
                        dot += Double(q[qBase + d]) * Double(k[kBase + d])
                    }
                    let distance = position - key
                    var bias = 0.0
                    if distance < relExtent {
                        let pBase = distance * dRel
                        for d in 0..<dRel {
                            bias += Double(rel[rBase + d]) * Double(table[pBase + d])
                        }
                    }
                    let logit = tau * dot * scale + tau * bias
                    logits.append(logit)
                    maximum = max(maximum, logit)
                }
                var denominator = exp(Double(sink[i * numQHeads + h]) - maximum)
                var acc = [Double](repeating: 0, count: headDim)
                for (offset, logit) in logits.enumerated() {
                    let weight = exp(logit - maximum)
                    denominator += weight
                    let vBase = (first + offset - keyBase) * kvStride + kvHead * headDim
                    for d in 0..<headDim {
                        acc[d] += weight * Double(v[vBase + d])
                    }
                }
                for d in 0..<headDim {
                    out[qBase + d] = Float(acc[d] / denominator)
                }
            }
        }
        return out
    }

    /// The attention output gate: `out * sigmoid(gateLogits)`, elementwise.
    public static func sigmoidGate(_ out: [Float], gateLogits: [Float]) -> [Float] {
        precondition(out.count == gateLogits.count)
        return zip(out, gateLogits).map { value, logit in
            Float(Double(value) / (1.0 + exp(-Double(logit))))
        }
    }
}
