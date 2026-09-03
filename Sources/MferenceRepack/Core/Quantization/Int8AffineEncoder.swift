import Foundation

/// Production INT8 group-64 affine quantizer for the streaming repacker.
///
/// The INT8 twin of `Int4AffineEncoder`, and deliberately the *same* affine
/// convention: plain min/max grid, `scale = (max - min) / (levels - 1)` always
/// positive, `bias = min`, both rounded through BF16 *before* the indices are
/// computed, no zero-point snapping. Only two things differ from the INT4
/// encoder — the level count (256 rather than 16) and the packing (one whole
/// byte per weight rather than two nibbles per byte).
///
/// Keeping the convention identical is a decision, not an accident.
/// `Int4AffineEncoderConventionTests` locks the INT4 grid against MLX's
/// endpoint-anchoring/zero-snap variant; `Int8AffineEncoderConventionTests`
/// locks this one the same way. A repacker whose two widths disagreed about
/// what an affine grid means would make `docs/QUANTIZER_QUALITY.md`'s
/// weight-level comparison uninterpretable — the measured error would mix the
/// width change with a convention change.
///
/// The algorithm is a bit-exact twin of the runtime module's
/// `Quantization.quantizeInt8Affine` reference;
/// `Int8AffineEncoderParityTests` locks the two together, so weights quantized
/// here are indistinguishable from the fixture semantics the runtime's INT8
/// decode paths (the Qwen 3.6 router GEMV, `moe.metal`) were verified against.
/// The duplication is the price of `MferenceRepackCore` staying free of the
/// runtime module.
public enum Int8AffineEncoder {
    public static let groupSize = 64
    /// Representable levels: `q` spans `0...255`.
    public static let levels = 256

    public struct EncodedRows: Equatable, Sendable {
        public let packed: [UInt8]   // N bytes per row, one per weight
        public let scales: [UInt16]  // N/64 BF16 bit patterns per row
        public let biases: [UInt16]  // N/64 BF16 bit patterns per row
    }

    public static func encodeRow(_ row: [Float]) -> EncodedRows {
        row.withUnsafeBufferPointer { encodeTensor($0, rowLength: row.count) }
    }

    /// The quantization nucleus: one group of `groupSize` floats in, that
    /// group's `groupSize` packed bytes plus one BF16 scale and one BF16 bias
    /// out.
    ///
    /// Every INT8-quantizing path in the repacker funnels through this
    /// function — `encodeTensor` loops it row-major, and
    /// `StreamingInt8Quantizer` calls it per group on bounded scratch — so
    /// bit-exactness with the runtime's `Quantization.quantizeInt8Affine`
    /// reference is structural rather than merely tested. `packed` is fully
    /// overwritten; it need not be zeroed.
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
            scaleF = (wmax - wmin) / Float(levels - 1)
            biasF = wmin
        }
        let sBits = Int4AffineEncoder.bf16Bits(scaleF)
        let bBits = Int4AffineEncoder.bf16Bits(biasF)
        scale.pointee = sBits
        bias.pointee = bBits
        // Quantize against the BF16-rounded values so the runtime decode
        // (which reads BF16) reproduces the stored q.
        let s = Int4AffineEncoder.bf16ToFloat(sBits)
        let b = Int4AffineEncoder.bf16ToFloat(bBits)
        let invScale = s == 0 ? Float(0) : 1.0 / s
        for k in 0..<groupSize {
            var q = Int(((values[k] - b) * invScale).rounded())
            q = max(0, min(levels - 1, q))
            packed[k] = UInt8(q)
        }
    }

    /// Quantize `values` as consecutive rows of `rowLength` floats. Layout
    /// matches the gturbo companion arrangement: packed bytes for all rows,
    /// then per-group scales, then per-group biases, row-major.
    public static func encodeTensor(_ values: UnsafeBufferPointer<Float>,
                                    rowLength: Int) -> EncodedRows {
        precondition(rowLength % groupSize == 0,
                     "row length \(rowLength) is not a multiple of \(groupSize)")
        precondition(values.count % rowLength == 0,
                     "value count is not a multiple of the row length")
        let rows = values.count / rowLength
        let groupsPerRow = rowLength / groupSize
        var packed = [UInt8](repeating: 0, count: values.count)
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
                                        packed: packedBuffer.baseAddress! + element,
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
