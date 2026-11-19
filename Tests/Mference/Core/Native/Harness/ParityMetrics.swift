import Accelerate
import Foundation

/// The D1 parity metrics (`docs/NATIVE_PARITY.md`): greedy agreement, per-token
/// KL divergence of the logits, and per-op error statistics. Pure functions over
/// F32 rows; F64 accumulation; Accelerate for the vocabulary-wide passes, so a
/// debug test build stays fast. No model, no GPU.
enum ParityMetrics {
    /// `log_softmax(x)` in F64.
    static func logSoftmax(_ x: UnsafeBufferPointer<Float>) -> [Double] {
        let n = x.count
        precondition(n > 0)
        var d = [Double](repeating: 0, count: n)
        vDSP_vspdp(x.baseAddress!, 1, &d, 1, vDSP_Length(n))
        var mx = 0.0
        vDSP_maxvD(d, 1, &mx, vDSP_Length(n))
        var shift = -mx
        vDSP_vsaddD(d, 1, &shift, &d, 1, vDSP_Length(n))
        var e = [Double](repeating: 0, count: n)
        var count = Int32(n)
        vvexp(&e, d, &count)
        var sum = 0.0
        vDSP_sveD(e, 1, &sum, vDSP_Length(n))
        var lse = -log(sum)
        vDSP_vsaddD(d, 1, &lse, &d, 1, vDSP_Length(n))
        return d
    }

    static func logSoftmax(_ x: [Float]) -> [Double] { x.withUnsafeBufferPointer { logSoftmax($0) } }

    /// `KL(P || Q) = sum p (log p - log q)` with P the golden (reference)
    /// distribution and Q the candidate, both given as logits. A small
    /// negative value from rounding is clamped to 0. A NaN or infinite logit
    /// on either side gives `.infinity`, never a score, so it always fails the
    /// bar.
    static func klDivergence(golden: UnsafeBufferPointer<Float>, candidate: UnsafeBufferPointer<Float>) -> Double {
        precondition(golden.count == candidate.count, "vocabulary sizes differ")
        if !allFinite(golden) || !allFinite(candidate) { return .infinity }
        let lp = logSoftmax(golden)
        let lq = logSoftmax(candidate)
        let n = lp.count
        var p = [Double](repeating: 0, count: n)
        var count = Int32(n)
        vvexp(&p, lp, &count)
        var diff = [Double](repeating: 0, count: n)
        vDSP_vsubD(lq, 1, lp, 1, &diff, 1, vDSP_Length(n))  // lp - lq
        var kl = 0.0
        vDSP_dotprD(p, 1, diff, 1, &kl, vDSP_Length(n))
        return kl.isFinite ? max(0, kl) : .infinity
    }

    /// True when every element is finite (no NaN, no ±inf).
    static func allFinite(_ x: UnsafeBufferPointer<Float>) -> Bool {
        var sum: Float = 0
        vDSP_sve(x.baseAddress!, 1, &sum, vDSP_Length(x.count))
        if sum.isFinite { return true }
        // a finite sum can overflow only from huge finite values: check each
        return x.allSatisfy(\.isFinite)
    }

    static func klDivergence(golden: [Float], candidate: [Float]) -> Double {
        golden.withUnsafeBufferPointer { g in candidate.withUnsafeBufferPointer { klDivergence(golden: g, candidate: $0) } }
    }

    /// Index of the largest element, ties to the lowest index (llama.cpp's
    /// greedy choice and `mfe-golden`'s).
    static func argmax(_ x: UnsafeBufferPointer<Float>) -> Int {
        var best = 0
        var v = x[0]
        for i in 1..<x.count where x[i] > v {
            v = x[i]
            best = i
        }
        return best
    }

    static func argmax(_ x: [Float]) -> Int { x.withUnsafeBufferPointer { argmax($0) } }

    /// Element-wise error of a candidate row against the golden row.
    struct ErrorStats: Equatable {
        var count: Int = 0
        var maxAbs: Double = 0
        var sumAbs: Double = 0
        /// Largest |golden| seen, for relative statements.
        var maxGolden: Double = 0
        var sumSquaredError: Double = 0
        var sumSquaredGolden: Double = 0

        var meanAbs: Double { count == 0 ? 0 : sumAbs / Double(count) }
        /// `||c - g|| / ||g||` over everything accumulated.
        var relativeL2: Double { sumSquaredGolden == 0 ? (sumSquaredError == 0 ? 0 : .infinity) : (sumSquaredError / sumSquaredGolden).squareRoot() }

        mutating func add(golden: UnsafeBufferPointer<Float>, candidate: UnsafeBufferPointer<Float>) {
            precondition(golden.count == candidate.count, "row sizes differ")
            for i in 0..<golden.count {
                let g = Double(golden[i])
                let e = Double(candidate[i]) - g
                let a = abs(e)
                if a > maxAbs || a.isNaN { maxAbs = a.isNaN ? .nan : a }
                sumAbs += a
                maxGolden = max(maxGolden, abs(g))
                sumSquaredError += e * e
                sumSquaredGolden += g * g
            }
            count += golden.count
        }

        mutating func add(golden: [Float], candidate: [Float]) {
            golden.withUnsafeBufferPointer { g in candidate.withUnsafeBufferPointer { add(golden: g, candidate: $0) } }
        }

        mutating func merge(_ o: ErrorStats) {
            count += o.count
            maxAbs = max(maxAbs, o.maxAbs)
            sumAbs += o.sumAbs
            maxGolden = max(maxGolden, o.maxGolden)
            sumSquaredError += o.sumSquaredError
            sumSquaredGolden += o.sumSquaredGolden
        }
    }

    /// Summary of a set of per-token values (KL divergences).
    struct Distribution: Equatable {
        var count: Int
        var mean: Double
        var median: Double
        var p99: Double
        var max: Double

        init(_ values: [Double]) {
            count = values.count
            guard !values.isEmpty else {
                mean = 0; median = 0; p99 = 0; max = 0
                return
            }
            let s = values.sorted()
            mean = s.reduce(0, +) / Double(s.count)
            median = Self.quantile(s, 0.5)
            p99 = Self.quantile(s, 0.99)
            max = s.last!
        }

        /// Nearest-rank quantile of a sorted array.
        static func quantile(_ sorted: [Double], _ q: Double) -> Double {
            let rank = Int((q * Double(sorted.count)).rounded(.up))
            return sorted[Swift.max(0, Swift.min(sorted.count - 1, rank - 1))]
        }
    }
}
