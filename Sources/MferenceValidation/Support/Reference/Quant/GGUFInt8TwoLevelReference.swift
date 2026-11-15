import Foundation
import Mference

/// CPU reference for the INT8-2L resident GEMV (WS4 U3): `y = W · x` in
/// Double over the exact fp32 weights, which come from `GGUFBlockDecode`
/// (the runtime twin of the installer's INT8-2L expander).
public enum GGUFInt8TwoLevelReference {
    /// `weights` is row-major `[rows][n]`; x is given as fp16 values.
    public static func gemv(weights: [Float], rows: Int, n: Int, x: [Float16]) -> [Float] {
        precondition(weights.count == rows * n && x.count == n)
        let xf = x.map { Double(Float($0)) }
        return (0..<rows).map { r in
            var acc = 0.0
            for i in 0..<n { acc += Double(weights[r * n + i]) * xf[i] }
            return Float(acc)
        }
    }
}
