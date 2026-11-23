import Foundation

/// GGUF Q8_0 blocks and the stacked projection that reads them.
///
/// A Q8_0 block is 34 bytes: an IEEE half scale `d` and 32 signed bytes `q`;
/// the value is `float(d) * q`. A weight with GGUF shape `{in, out}` is `out`
/// rows of `in / 32` consecutive blocks (row-major `[out][in]`).
public enum Inkling2Q8Ref {
    public static let blockValues = 32
    public static let blockBytes = 34

    public static func rowBytes(inputs: Int) -> Int {
        precondition(inputs % blockValues == 0)
        return inputs / blockValues * blockBytes
    }

    /// ggml `quantize_row_q8_0_ref`: `d = amax / 127`, `q = round(x / d)`,
    /// with the reciprocal taken from the FP32 `d` before it is stored as half.
    public static func quantize(rows: Int, inputs: Int, values: [Float]) -> [UInt8] {
        precondition(values.count == rows * inputs && inputs % blockValues == 0)
        var bytes = [UInt8]()
        bytes.reserveCapacity(rows * rowBytes(inputs: inputs))
        for row in 0..<rows {
            for block in 0..<(inputs / blockValues) {
                let base = row * inputs + block * blockValues
                var amax: Float = 0
                for j in 0..<blockValues { amax = max(amax, abs(values[base + j])) }
                let d = amax / 127
                let id: Float = d != 0 ? 1 / d : 0
                let bits = Float16(d).bitPattern
                bytes.append(UInt8(truncatingIfNeeded: bits))
                bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
                for j in 0..<blockValues {
                    let q = Int8((values[base + j] * id).rounded(.toNearestOrAwayFromZero))
                    bytes.append(UInt8(bitPattern: q))
                }
            }
        }
        return bytes
    }

    /// Decodes `rows` rows of `inputs` values starting at byte `offset`.
    public static func dequantize(_ bytes: [UInt8], offset: Int = 0,
                                  rows: Int, inputs: Int) -> [Float] {
        let stride = rowBytes(inputs: inputs)
        precondition(offset + rows * stride <= bytes.count)
        var out = [Float](repeating: 0, count: rows * inputs)
        for row in 0..<rows {
            for block in 0..<(inputs / blockValues) {
                let base = offset + row * stride + block * blockBytes
                let bits = UInt16(bytes[base]) | (UInt16(bytes[base + 1]) << 8)
                let d = Float(Float16(bitPattern: bits))
                for j in 0..<blockValues {
                    out[row * inputs + block * blockValues + j] =
                        d * Float(Int8(bitPattern: bytes[base + 2 + j]))
                }
            }
        }
        return out
    }

    /// One segment of a stacked projection: `rows` output columns read
    /// `input` (`[tokens, inputs]`) through a Q8_0 weight.
    public struct Segment {
        public var input: [Float]
        public var weight: [UInt8]
        public var weightOffset: Int
        public var rows: Int

        public init(input: [Float], weight: [UInt8], weightOffset: Int = 0, rows: Int) {
            self.input = input
            self.weight = weight
            self.weightOffset = weightOffset
            self.rows = rows
        }
    }

    /// `y = x W^T` for every segment, columns concatenated in segment order:
    /// result `[tokens, sum(rows)]`, FP64 accumulation.
    public static func stackedProjection(_ segments: [Segment], tokens: Int, inputs: Int) -> [Float] {
        let total = segments.reduce(0) { $0 + $1.rows }
        var out = [Float](repeating: 0, count: tokens * total)
        var column = 0
        for segment in segments {
            precondition(segment.input.count == tokens * inputs)
            let w = dequantize(segment.weight, offset: segment.weightOffset,
                               rows: segment.rows, inputs: inputs)
            for t in 0..<tokens {
                for r in 0..<segment.rows {
                    var acc = 0.0
                    for i in 0..<inputs {
                        acc += Double(w[r * inputs + i]) * Double(segment.input[t * inputs + i])
                    }
                    out[t * total + column + r] = Float(acc)
                }
            }
            column += segment.rows
        }
        return out
    }
}
