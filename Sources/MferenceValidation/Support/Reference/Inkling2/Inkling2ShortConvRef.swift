import Foundation

/// Kernel-4 causal depthwise short convolutions of Inkling 2, FP64 inside.
///
/// Layouts follow the GGUF file (llama.cpp `ggml_ssm_conv` order):
/// - `weight`: `[channels, kernel]`; tap `j` multiplies `x[t - (kernel-1) + j]`,
///   so tap `kernel-1` weights the current row. (The converter flips the
///   vendor's lag-ordered `static_w` into this order.)
/// - `dynamic` taps: `[rows, heads, kernel]` in lag order: tap `lag`
///   multiplies `x[t - lag]`, shared by the `channels / heads` channels of
///   one KV head (the vendor's `weight_proj` output and llama.cpp's view).
/// - `history`: `[kernel-1, channels]`, time-major, row 0 the oldest; the
///   last `kernel-1` inputs before this chunk (zeros at sequence start).
///
/// Static:  `y_t = (sum_j w[c,j] x[t-(K-1)+j] + x_t) / sqrt(K + 1)`
/// Dynamic: `y_t = (sum_j w[c,j] x[t-(K-1)+j] + sum_lag a[t,h,lag] x[t-lag] + x_t) / sqrt(2K + 1)`
///
/// Both return the new history (the last `kernel-1` inputs of
/// `history ++ x`), which is what the next chunk or decode step reads.
public enum Inkling2ShortConvRef {
    public struct Result: Sendable {
        public var output: [Float]
        public var history: [Float]
    }

    public static func staticConv(_ x: [Float], weight: [Float], history: [Float],
                                  rows: Int, channels: Int,
                                  kernel: Int = Inkling2RefShape.convKernel) -> Result {
        convolve(x, weight: weight, dynamic: nil, heads: 1, history: history,
                 rows: rows, channels: channels, kernel: kernel)
    }

    public static func dynamicConv(_ x: [Float], weight: [Float], dynamic: [Float],
                                   heads: Int, history: [Float],
                                   rows: Int, channels: Int,
                                   kernel: Int = Inkling2RefShape.convKernel) -> Result {
        convolve(x, weight: weight, dynamic: dynamic, heads: heads, history: history,
                 rows: rows, channels: channels, kernel: kernel)
    }

    public static func zeroHistory(channels: Int,
                                   kernel: Int = Inkling2RefShape.convKernel) -> [Float] {
        [Float](repeating: 0, count: (kernel - 1) * channels)
    }

    private static func convolve(_ x: [Float], weight: [Float], dynamic: [Float]?,
                                 heads: Int, history: [Float],
                                 rows: Int, channels: Int, kernel: Int) -> Result {
        precondition(kernel >= 2)
        precondition(x.count == rows * channels)
        precondition(weight.count == channels * kernel)
        precondition(history.count == (kernel - 1) * channels)
        precondition(channels % heads == 0)
        if let dynamic { precondition(dynamic.count == rows * heads * kernel) }
        let past = kernel - 1
        let headWidth = channels / heads
        // context row n is history row n for n < past, else x row n - past.
        func context(_ n: Int, _ c: Int) -> Double {
            n < past ? Double(history[n * channels + c]) : Double(x[(n - past) * channels + c])
        }
        let divisor = (Double((dynamic == nil ? 1 : 2) * kernel + 1)).squareRoot()
        var out = [Float](repeating: 0, count: x.count)
        for t in 0..<rows {
            for c in 0..<channels {
                var acc = 0.0
                for j in 0..<kernel {
                    acc += Double(weight[c * kernel + j]) * context(t + j, c)
                }
                if let dynamic {
                    let h = c / headWidth
                    for lag in 0..<kernel {
                        acc += Double(dynamic[(t * heads + h) * kernel + lag]) * context(t + past - lag, c)
                    }
                }
                out[t * channels + c] = Float((acc + Double(x[t * channels + c])) / divisor)
            }
        }
        var newHistory = [Float](repeating: 0, count: history.count)
        for n in 0..<past {
            for c in 0..<channels {
                newHistory[n * channels + c] = Float(context(rows + n, c))
            }
        }
        return Result(output: out, history: newHistory)
    }
}
