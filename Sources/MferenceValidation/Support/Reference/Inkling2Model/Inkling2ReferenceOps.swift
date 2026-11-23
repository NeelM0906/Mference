import Foundation

/// The operations of an Inkling 2 layer visit, one function each, in plain
/// loops with double-precision accumulation. `Inkling2ReferenceModel`
/// composes them; GPU kernels and other references are checked against them
/// one by one.
///
/// Semantics are those of upstream llama.cpp `src/models/inkling2.cpp` and
/// `inkling2-loop.cpp` at release b12635, cross-checked line by line with the
/// transformers reference `modeling_inkling2_loop.py` (`be48e7a`). Where a
/// function encodes a choice the two express differently, its comment says so.
public enum Inkling2ReferenceOps {

    /// Epsilon of the weightless norms inside the attention residuals and the
    /// message gates (hard-coded `1e-6` upstream, the default in transformers).
    public static let residualNormEps: Float = 1e-6

    /// Epsilon added to the L2 norm before the clip factor (`1e-6` in both).
    public static let clipNormEps: Double = 1e-6

    // MARK: Elementwise

    public static func dot(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "dot of \(a.count) and \(b.count) values")
        var s = 0.0
        for i in 0..<a.count { s += Double(a[i]) * Double(b[i]) }
        return Float(s)
    }

    public static func sigmoid(_ x: Float) -> Float {
        Float(1.0 / (1.0 + Foundation.exp(-Double(x))))
    }

    /// `cap * tanh(x / cap)`; a non-positive cap leaves `x` unchanged.
    public static func softcap(_ x: Float, _ cap: Float) -> Float {
        guard cap > 0 else { return x }
        return Float(Double(cap) * Foundation.tanh(Double(x) / Double(cap)))
    }

    public static func add(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count)
        return (0..<a.count).map { a[$0] + b[$0] }
    }

    // MARK: Norms and clips

    /// `x / sqrt(mean(x^2) + eps)`, times `weight` when given.
    public static func rmsNorm(_ x: [Float], weight: [Float]? = nil, eps: Float) -> [Float] {
        var ss = 0.0
        for v in x { ss += Double(v) * Double(v) }
        let inv = 1.0 / (ss / Double(x.count) + Double(eps)).squareRoot()
        if let w = weight {
            precondition(w.count == x.count, "norm weight \(w.count) for \(x.count) values")
            return (0..<x.count).map { Float(Double(x[$0]) * inv * Double(w[$0])) }
        }
        return x.map { Float(Double($0) * inv) }
    }

    /// Scales `x` (after multiplying by `scale`) so its L2 norm is at most
    /// `threshold`: `v * min(1, threshold / (|v| + 1e-6))`. Shorter vectors
    /// pass unchanged. Applied per head vector.
    public static func l2Clip(_ x: [Float], scale: [Float]? = nil, threshold: Float) -> [Float] {
        var v = x
        if let s = scale {
            precondition(s.count == x.count)
            for i in 0..<v.count { v[i] *= s[i] }
        }
        var ss = 0.0
        for e in v { ss += Double(e) * Double(e) }
        let factor = min(1.0, Double(threshold) / (ss.squareRoot() + clipNormEps))
        return v.map { Float(Double($0) * factor) }
    }

    /// True when `l2Clip` would scale `x` (after `scale`) down.
    public static func l2ClipActive(_ x: [Float], scale: [Float]? = nil, threshold: Float) -> Bool {
        var ss = 0.0
        for i in 0..<x.count {
            let e = Double(x[i]) * Double(scale?[i] ?? 1)
            ss += e * e
        }
        return Double(threshold) / (ss.squareRoot() + clipNormEps) < 1.0
    }

    /// `1 + alpha * log(max(1, (position + 1) / floor))`, the factor global
    /// layers apply to the clipped query and relative state.
    public static func logScale(position: Int, floor: Int, alpha: Float) -> Float {
        let ratio = Double(position + 1) / Double(floor)
        return Float(1.0 + Double(alpha) * Foundation.log(max(1.0, ratio)))
    }

    // MARK: Depth attention (block attention residuals and messages)

    /// Attention over a stack of states with learned queries: every state is
    /// RMS-normalized without a weight, each query scores the normalized
    /// states with scale `1/sqrt(hidden)`, and the result is the softmax-
    /// weighted sum of the normalized states. One output per query. The
    /// order of `states` does not matter.
    public static func mix(states: [[Float]], queries: [[Float]]) -> [[Float]] {
        precondition(!states.isEmpty && !queries.isEmpty)
        let d = states[0].count
        let normed = states.map { rmsNorm($0, eps: residualNormEps) }
        let scale = 1.0 / Double(d).squareRoot()
        return queries.map { q in
            precondition(q.count == d)
            let scores = normed.map { Double(dot($0, q)) * scale }
            let m = scores.max()!
            let e = scores.map { Foundation.exp($0 - m) }
            let z = e.reduce(0, +)
            var out = [Double](repeating: 0, count: d)
            for (s, w) in zip(normed, e) {
                let p = w / z
                for i in 0..<d { out[i] += p * Double(s[i]) }
            }
            return out.map { Float($0) }
        }
    }

    /// The loop message gate: `z = down [rms(x); rms(m)]`,
    /// `(1 - tanh(forget z)) * x + tanh(gain z) * m`. Pass 1 never calls it.
    public static func control(_ x: [Float], message m: [Float],
                               down: any Inkling2ReferenceMatrix,
                               gain: any Inkling2ReferenceMatrix,
                               forget: any Inkling2ReferenceMatrix) -> [Float] {
        // Concatenation, 2 * hidden values.
        let g = rmsNorm(x, eps: residualNormEps) + rmsNorm(m, eps: residualNormEps)
        let z = down.apply(g)
        let gainGate = gain.apply(z).map { Float(Foundation.tanh(Double($0))) }
        let forgetGate = forget.apply(z).map { Float(Foundation.tanh(Double($0))) }
        return (0..<x.count).map {
            Float((1.0 - Double(forgetGate[$0])) * Double(x[$0]) + Double(gainGate[$0]) * Double(m[$0]))
        }
    }

    // MARK: Short convolution

    /// One step of the causal depthwise short convolution with its residual,
    /// for the current input `x` of `channels` values.
    ///
    /// `tail` holds the previous `K - 1` inputs, oldest first (zeros before
    /// the first token). Static `weights` are `[channels x K]` with tap
    /// `K - 1` weighting `x` (ssm_conv order). `dynamicTaps`, when given, are
    /// `[heads x K]` with lag 0 weighting `x`, shared by the channels of a
    /// head. Output `(x + conv) / sqrt(n * K + 1)`, `n = 2` with dynamic taps
    /// and 1 without.
    public static func shortConv(_ x: [Float], tail: [[Float]], weights: [Float],
                                 dynamicTaps: [Float]? = nil, heads: Int = 1) -> [Float] {
        let c = x.count
        let k = tail.count + 1
        precondition(weights.count == c * k, "conv weights \(weights.count) for \(c) channels x \(k) taps")
        // window[j] = input j steps back from the oldest: window[k-1] == x.
        let window = tail + [x]
        let channelsPerHead = c / heads
        let divisor = (Double((dynamicTaps == nil ? 1 : 2) * k + 1)).squareRoot()
        var out = [Float](repeating: 0, count: c)
        for ch in 0..<c {
            var s = 0.0
            for tap in 0..<k {
                s += Double(window[tap][ch]) * Double(weights[ch * k + tap])
            }
            if let dyn = dynamicTaps {
                let head = ch / channelsPerHead
                for lag in 0..<k {
                    s += Double(window[k - 1 - lag][ch]) * Double(dyn[head * k + lag])
                }
            }
            out[ch] = Float((Double(x[ch]) + s) / divisor)
        }
        return out
    }

    /// `tail` advanced by one input: drop the oldest, append `x`.
    public static func advance(tail: [[Float]], with x: [Float]) -> [[Float]] {
        guard !tail.isEmpty else { return [] }
        return Array(tail.dropFirst()) + [x]
    }

    // MARK: Attention

    /// Relative bias of one head at backward `distance`: `table[distance] .
    /// relState` inside the extent (`table` has `extent` rows of `relDim`),
    /// zero outside it.
    public static func relativeBias(table: [Float], relState: [Float], distance: Int) -> Float {
        let r = relState.count
        let extent = table.count / r
        guard distance >= 0, distance < extent else { return 0 }
        return dot(Array(table[(distance * r)..<((distance + 1) * r)]), relState)
    }

    /// One query head against its keys: logits `scale * q.k + bias`, a sink
    /// logit that joins the softmax with a zero value, output the weighted sum
    /// of `values`. `keys`, `values` and `biases` are aligned; an empty key set
    /// returns zeros.
    public static func attendHead(query: [Float], keys: [[Float]], values: [[Float]],
                                  biases: [Float], sink: Float, scale: Float) -> [Float] {
        let hd = query.count
        guard !keys.isEmpty else { return [Float](repeating: 0, count: hd) }
        let logits = (0..<keys.count).map {
            Double(scale) * Double(dot(query, keys[$0])) + Double(biases[$0])
        }
        let m = max(logits.max()!, Double(sink))
        let e = logits.map { Foundation.exp($0 - m) }
        let z = e.reduce(0, +) + Foundation.exp(Double(sink) - m)
        var out = [Double](repeating: 0, count: hd)
        for (v, w) in zip(values, e) {
            let p = w / z
            for i in 0..<hd { out[i] += p * Double(v[i]) }
        }
        return out.map { Float($0) }
    }

    // MARK: Mixture of experts

    /// Router: `scores = sigmoid(router x)` in FP32 semantics, top-k on
    /// `scores + bias` (ties to the lower index), weights = the chosen
    /// unbiased scores normalized to sum 1 (`/(sum + 1e-20)`), times
    /// `routeScale`. Experts in descending selection order.
    public static func route(_ x: [Float], router: any Inkling2ReferenceMatrix,
                             bias: [Float], topK: Int, routeScale: Float) -> Inkling2Route {
        let scores = router.apply(x).map(sigmoid)
        let selection = (0..<scores.count).map { Double(scores[$0]) + Double(bias[$0]) }
        let chosen = (0..<scores.count).sorted {
            selection[$0] != selection[$1] ? selection[$0] > selection[$1] : $0 < $1
        }.prefix(topK)
        let sum = chosen.reduce(0.0) { $0 + Double(scores[$1]) }
        let weights = chosen.map { Float(Double(scores[$0]) / (sum + 1e-20) * Double(routeScale)) }
        return Inkling2Route(experts: Array(chosen), weights: weights)
    }

    /// One expert: `[gate; up] = gateUp x`,
    /// `act = softcap(gate, gateCap) * sigmoid(gate) * softcap(up, upCap)`
    /// (the sigmoid sees the raw gate), output `down act`.
    /// `observe`, when given, sees the `[gate; up]` pre-activations.
    public static func expert(_ x: [Float], gateUp: any Inkling2ReferenceMatrix,
                              down: any Inkling2ReferenceMatrix,
                              gateSoftcap: Float, upSoftcap: Float,
                              observe: (([Float]) -> Void)? = nil) -> [Float] {
        let gu = gateUp.apply(x)
        observe?(gu)
        let ff = gu.count / 2
        let act = (0..<ff).map { i -> Float in
            let g = gu[i]
            return Float(Double(softcap(g, gateSoftcap)) * Double(sigmoid(g))
                         * Double(softcap(gu[ff + i], upSoftcap)))
        }
        return down.apply(act)
    }

    /// The routed MoE output, accumulated in double precision over the
    /// chosen experts.
    public static func moe(_ x: [Float], route: Inkling2Route,
                           gateUp: [any Inkling2ReferenceMatrix],
                           down: [any Inkling2ReferenceMatrix],
                           gateSoftcap: Float, upSoftcap: Float,
                           observe: (([Float]) -> Void)? = nil) -> [Float] {
        var acc = [Double](repeating: 0, count: x.count)
        for (e, w) in zip(route.experts, route.weights) {
            let y = expert(x, gateUp: gateUp[e], down: down[e],
                           gateSoftcap: gateSoftcap, upSoftcap: upSoftcap, observe: observe)
            for i in 0..<acc.count { acc[i] += Double(w) * Double(y[i]) }
        }
        return acc.map { Float($0) }
    }
}

/// The experts one token chose at one visit, with their routing weights.
public struct Inkling2Route: Sendable, Equatable {
    public let experts: [Int]
    public let weights: [Float]

    public init(experts: [Int], weights: [Float]) {
        self.experts = experts
        self.weights = weights
    }
}
