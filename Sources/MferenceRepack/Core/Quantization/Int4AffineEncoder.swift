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

    /// The quantization nucleus: one group of `groupSize` floats in, that
    /// group's `groupSize / 2` packed nibble bytes plus one BF16 scale and one
    /// BF16 bias out.
    ///
    /// Every quantizing path in the repacker funnels through this function —
    /// `encodeTensor` loops it row-major, and `StreamingInt4Quantizer` calls it
    /// per group on bounded scratch — so bit-exactness with the runtime's
    /// `Quantization.quantizeInt4Affine` reference is structural rather than
    /// merely tested. `packed` is fully overwritten; it need not be zeroed.
    @inline(__always)
    public static func encodeGroup(_ values: UnsafePointer<Float>,
                                   packed: UnsafeMutablePointer<UInt8>,
                                   scale: UnsafeMutablePointer<UInt16>,
                                   bias: UnsafeMutablePointer<UInt16>) {
        var wmin: Float = .infinity
        var wmax: Float = -.infinity
        for k in 0..<groupSize {
            let w = values[k]
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
        scale.pointee = sBits
        bias.pointee = bBits
        // Quantize against the BF16-rounded values so the runtime decode
        // (which reads BF16) reproduces the stored q.
        let s = bf16ToFloat(sBits)
        let b = bf16ToFloat(bBits)
        let invScale = s == 0 ? Float(0) : 1.0 / s
        for k in stride(from: 0, to: groupSize, by: 2) {
            var lo = Int(((values[k] - b) * invScale).rounded())
            lo = max(0, min(15, lo))
            var hi = Int(((values[k + 1] - b) * invScale).rounded())
            hi = max(0, min(15, hi))
            packed[k / 2] = UInt8(lo) & 0x0F | (UInt8(hi) & 0x0F) << 4
        }
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

        packed.withUnsafeMutableBufferPointer { packedBuffer in
            scales.withUnsafeMutableBufferPointer { scaleBuffer in
                biases.withUnsafeMutableBufferPointer { biasBuffer in
                    for r in 0..<rows {
                        let rowBase = r * rowLength
                        for g in 0..<groupsPerRow {
                            let element = rowBase + g * groupSize
                            let companion = r * groupsPerRow + g
                            encodeGroup(values.baseAddress! + element,
                                        packed: packedBuffer.baseAddress! + element / 2,
                                        scale: scaleBuffer.baseAddress! + companion,
                                        bias: biasBuffer.baseAddress! + companion)
                        }
                    }
                }
            }
        }
        return EncodedRows(packed: packed, scales: scales, biases: biases)
    }
}
