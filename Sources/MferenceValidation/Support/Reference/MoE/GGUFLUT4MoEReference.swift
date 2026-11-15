import Foundation

/// CPU reference for the LUT4 routed-expert decode pair (WS4 U3). It works
/// from the experts' exact fp32 weights (`GGUFLUT4Reference.dequantize`, equal
/// to llama.cpp's dequantization), not from LUT4 parts, so it is independent of
/// the kernel's (D, M)-grouped arithmetic. Dot products accumulate in Double.
public enum GGUFLUT4MoEReference {
    /// One expert's weights, row-major: gate and up `[F][D]`, down `[D][F]`.
    public struct Expert {
        public let gate: [Float], up: [Float], down: [Float]
        public init(gate: [Float], up: [Float], down: [Float]) {
            self.gate = gate; self.up = up; self.down = down
        }
    }

    static func dot(_ w: [Float], row: Int, n: Int, _ x: [Float]) -> Double {
        var acc = 0.0
        for i in 0..<n { acc += Double(w[row * n + i]) * Double(x[i]) }
        return acc
    }

    /// `acts[slot][f] = half(silu(gate·x) · (up·x))`, x given as fp16 values.
    public static func phase1(experts: [Expert], x: [Float16], d: Int, f: Int) -> [Float16] {
        let xf = x.map(Float.init)
        var acts = [Float16]()
        for e in experts {
            for row in 0..<f {
                let g = dot(e.gate, row: row, n: d, xf), u = dot(e.up, row: row, n: d, xf)
                acts.append(Float16(Float(g / (1 + exp(-g)) * u)))
            }
        }
        return acts
    }

    /// `y[d] = half(residual[d] + Σ_r w[r] · (down_r · acts_r))`.
    public static func phase2(experts: [Expert], acts: [Float16], routingWeights: [Float16],
                              residual: [Float16], d: Int, f: Int) -> [Float16] {
        let af = acts.map(Float.init)
        return (0..<d).map { row in
            var acc = Double(Float(residual[row]))
            for (r, e) in experts.enumerated() {
                let slice = Array(af[(r * f)..<((r + 1) * f)])
                acc += Double(Float(routingWeights[r])) * dot(e.down, row: row, n: f, slice)
            }
            return Float16(Float(acc))
        }
    }
}
