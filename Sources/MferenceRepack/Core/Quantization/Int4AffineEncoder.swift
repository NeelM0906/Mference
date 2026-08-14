import Foundation

/// Production INT4 group-64 affine quantizer for the streaming repacker.
///
/// The algorithm is deliberately a bit-exact twin of the runtime module's
/// `Quantization.quantizeInt4Affine` reference (min/max per group, scale
/// and bias rounded through BF16 *before* index quantization, low nibble =
/// even index): `Int4AffineEncoderParityTests` locks the two together, so
/// weights quantized here are indistinguishable from the fixture semantics
/// every runtime decode kernel was verified against. The duplication is the
/// price of `MferenceRepackCore` staying free of the runtime module.
public enum Int4AffineEncoder {
    public static let groupSize = 64

    public struct EncodedRows: Equatable, Sendable {
        public let packed: [UInt8]   // N/2 bytes per row; low nibble = even index
        public let scales: [UInt16]  // N/64 BF16 bit patterns per row
        public let biases: [UInt16]  // N/64 BF16 bit patterns per row
    }

    @inline(__always)
    static func bf16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let lsb = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF &+ lsb
        return UInt16(truncatingIfNeeded: (bits &+ roundingBias) >> 16)
    }

    @inline(__always)
    static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    public static func encodeRow(_ row: [Float]) -> EncodedRows {
        row.withUnsafeBufferPointer { encodeTensor($0, rowLength: row.count) }
    }

    /// Quantize `values` as consecutive rows of `rowLength` floats. Layout
    /// matches the gturbo companion arrangement: packed nibbles for all
    /// rows, then per-group scales, then per-group biases, row-major.
    public static func encodeTensor(_ values: UnsafeBufferPointer<Float>,
                                    rowLength: Int) -> EncodedRows {
        precondition(rowLength % groupSize == 0,
                     "row length \(rowLength) is not a multiple of \(groupSize)")
        precondition(values.count % rowLength == 0,
                     "value count is not a multiple of the row length")
        let rows = values.count / rowLength
        let groupsPerRow = rowLength / groupSize
        var packed = [UInt8](repeating: 0, count: values.count / 2)
        var scales = [UInt16](repeating: 0, count: rows * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: rows * groupsPerRow)

        for r in 0..<rows {
            let rowBase = r * rowLength
            for g in 0..<groupsPerRow {
                var wmin: Float = .infinity
                var wmax: Float = -.infinity
                for k in 0..<groupSize {
                    let w = values[rowBase + g * groupSize + k]
                    if w < wmin { wmin = w }
                    if w > wmax { wmax = w }
                }
                let scaleF: Float
                let biasF: Float
                if wmax == wmin {
                    // Constant group: scale=1, bias=value reconstructs exactly.
                    scaleF = 1
                    biasF = wmin
                } else {
                    scaleF = (wmax - wmin) / 15.0
                    biasF = wmin
                }
                let sBits = bf16Bits(scaleF)
                let bBits = bf16Bits(biasF)
                scales[r * groupsPerRow + g] = sBits
                biases[r * groupsPerRow + g] = bBits
                // Quantize against the BF16-rounded values so the runtime
                // decode (which reads BF16) reproduces the stored q.
                let scale = bf16ToFloat(sBits)
                let bias = bf16ToFloat(bBits)
                let invScale = scale == 0 ? Float(0) : 1.0 / scale
                for k in 0..<groupSize {
                    let w = values[rowBase + g * groupSize + k]
                    var q = Int(((w - bias) * invScale).rounded())
                    q = max(0, min(15, q))
                    let nibble = UInt8(q) & 0x0F
                    let byteIdx = (rowBase + g * groupSize) / 2 + k / 2
                    if (k & 1) == 0 {
                        packed[byteIdx] = (packed[byteIdx] & 0xF0) | nibble
                    } else {
                        packed[byteIdx] = (packed[byteIdx] & 0x0F) | (nibble << 4)
                    }
                }
            }
        }
        return EncodedRows(packed: packed, scales: scales, biases: biases)
    }
}
