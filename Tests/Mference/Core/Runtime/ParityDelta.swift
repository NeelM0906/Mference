import Foundation

/// Elementwise comparison against a reference-parity golden's fp32 gate,
/// reporting the worst offender rather than just failing. Shared by the
/// reference-parity suites (MiniCPM5 today).
struct ParityDelta {
    var maxAbs: Float = 0
    var maxRel: Float = 0
    var worstIndex: Int = -1
    var count: Int = 0
    /// `max over i of |a-b| - (atol + rtol*|b|)`. Non-positive means every
    /// element satisfies `numpy.allclose`.
    var worstSlack: Float = -.infinity
    var mismatched: Int = 0

    /// Element-wise `numpy.allclose`: `|a - b| <= atol + rtol * |b|`.
    static func compare(_ actual: [Float], _ expected: [Float],
                        atol: Float, rtol: Float) -> ParityDelta {
        var delta = ParityDelta()
        precondition(actual.count == expected.count,
                     "shape mismatch: \(actual.count) vs \(expected.count)")
        delta.count = expected.count
        for i in 0..<delta.count {
            let absolute = abs(actual[i] - expected[i])
            let relative = absolute / max(abs(expected[i]), 1e-30)
            let slack = absolute - (atol + rtol * abs(expected[i]))
            if absolute > delta.maxAbs { delta.maxAbs = absolute }
            delta.maxRel = max(delta.maxRel, relative)
            if slack > delta.worstSlack {
                delta.worstSlack = slack
                delta.worstIndex = i
            }
            if slack > 0 { delta.mismatched += 1 }
        }
        return delta
    }

    var passes: Bool { mismatched == 0 }

    var description: String {
        String(format: "maxAbs=%.3e maxRel=%.3e (%d/%d outside the gate, worst at %d)",
               maxAbs, maxRel, mismatched, count, worstIndex)
    }
}
