import Foundation
import Testing
@testable import MferenceRepackCore

/// Locks the **affine-grid convention** `Int8AffineEncoder` implements, and
/// locks it to be the *same* convention `Int4AffineEncoder` uses.
///
/// `Int4AffineEncoderConventionTests` explains why the grid is a decision
/// rather than an accident: MLX's `affine_quantize` — what mlx-community's
/// conversions are built with — anchors on the larger-magnitude endpoint and
/// snaps the grid so 0.0 is exactly representable, while ours is a plain
/// min/max grid. The INT8 encoder deliberately does not deviate.
///
/// The reason to pin that here as well is specific to `docs/QUANTIZER_QUALITY.md`:
/// the gate compares our mixed-width install against a mixed-width control. If
/// our two widths disagreed about what an affine grid means, the measured
/// difference on the 80 INT8 router tensors would confound a width effect with
/// a convention effect and the comparison would stop being interpretable. Any
/// future adoption of zero-point snapping has to change both encoders together,
/// and these assertions are what force that.
@Suite struct Int8AffineEncoderConventionTests {

    private static func groups(seed: UInt64, count: Int) -> [[Float]] {
        var state = seed
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: Int(state >> 33))) / Float(Int32.max)
        }
        return (0..<count).map { _ in
            (0..<Int8AffineEncoder.groupSize).map { _ in next() * 0.05 }
        }
    }

    @Test("scale is the BF16 min/max range over 255, bias is the BF16 minimum")
    func gridIsPlainMinMax() {
        for group in Self.groups(seed: 0x5EED_1234, count: 64) {
            let encoded = Int8AffineEncoder.encodeRow(group)
            let scale = Int4AffineEncoder.bf16ToFloat(encoded.scales[0])
            let bias = Int4AffineEncoder.bf16ToFloat(encoded.biases[0])
            let low = group.min()!
            let high = group.max()!
            // No zero-point snapping: the bias is the group minimum verbatim
            // (through BF16), and the scale spans the whole range.
            #expect(encoded.biases[0] == Int4AffineEncoder.bf16Bits(low))
            #expect(encoded.scales[0]
                == Int4AffineEncoder.bf16Bits((high - low) / 255.0))
            // A negative scale is MLX's signal that it anchored on the maximum.
            // Ours never does that.
            #expect(scale > 0)
            #expect(bias <= high)
        }
    }

    /// The one structural difference from INT4 that is allowed: 256 levels
    /// rather than 16, one byte per weight rather than two per byte.
    @Test("the grid has 256 levels and packs one byte per weight")
    func gridWidthIsTheOnlyDifference() {
        #expect(Int8AffineEncoder.levels == 256)
        #expect(Int8AffineEncoder.groupSize == Int4AffineEncoder.groupSize)
        for group in Self.groups(seed: 0x1E7E_1500, count: 8) {
            let int8 = Int8AffineEncoder.encodeRow(group)
            let int4 = Int4AffineEncoder.encodeRow(group)
            #expect(int8.packed.count == group.count)
            #expect(int4.packed.count == group.count / 2)
            // Same anchor, different step: the INT8 step is exactly the INT4
            // step scaled by 15/255, up to BF16 rounding of each.
            #expect(int8.biases == int4.biases)
        }
    }

    @Test("reconstruction stays inside the half-step bound the grid promises")
    func reconstructionWithinHalfStep() {
        for group in Self.groups(seed: 0xC0FF_EE01, count: 64) {
            let encoded = Int8AffineEncoder.encodeRow(group)
            let scale = Int4AffineEncoder.bf16ToFloat(encoded.scales[0])
            let bias = Int4AffineEncoder.bf16ToFloat(encoded.biases[0])
            // BF16 carries 8 mantissa bits, so rounding the bias can move the
            // grid by that much of the group's magnitude; everything beyond
            // that must be the honest half-step.
            let magnitude = max(abs(group.min()!), abs(group.max()!))
            let bound = scale / 2 + magnitude / 256 + .ulpOfOne
            for (index, value) in group.enumerated() {
                let reconstructed = scale * Float(encoded.packed[index]) + bias
                #expect(abs(reconstructed - value) <= bound,
                        "index \(index): |\(reconstructed) - \(value)| > \(bound)")
            }
        }
    }

    /// The whole point of spending the extra four bits: at the same anchor and
    /// the same convention, INT8 must reconstruct strictly better than INT4 on
    /// a non-degenerate group.
    @Test("INT8 reconstructs a non-degenerate group better than INT4 does")
    func int8BeatsInt4OnTheSameGroup() {
        for group in Self.groups(seed: 0xBE77_E12, count: 64) {
            func error(_ reconstruct: (Int) -> Float) -> Float {
                var sum: Float = 0
                for (index, value) in group.enumerated() {
                    let d = reconstruct(index) - value
                    sum += d * d
                }
                return sum
            }
            let e8 = Int8AffineEncoder.encodeRow(group)
            let s8 = Int4AffineEncoder.bf16ToFloat(e8.scales[0])
            let b8 = Int4AffineEncoder.bf16ToFloat(e8.biases[0])
            let e4 = Int4AffineEncoder.encodeRow(group)
            let s4 = Int4AffineEncoder.bf16ToFloat(e4.scales[0])
            let b4 = Int4AffineEncoder.bf16ToFloat(e4.biases[0])
            let int8Error = error { s8 * Float(e8.packed[$0]) + b8 }
            let int4Error = error { index in
                let byte = e4.packed[index / 2]
                let nibble = index % 2 == 0 ? byte & 0x0F : byte >> 4
                return s4 * Float(nibble) + b4
            }
            #expect(int8Error < int4Error)
        }
    }

    @Test("a constant group reconstructs exactly")
    func constantGroupIsExact() {
        for value: Float in [0, 0.25, -3.5, 1.0 / 64.0] {
            let group = [Float](repeating: value,
                                count: Int8AffineEncoder.groupSize)
            let encoded = Int8AffineEncoder.encodeRow(group)
            let scale = Int4AffineEncoder.bf16ToFloat(encoded.scales[0])
            let bias = Int4AffineEncoder.bf16ToFloat(encoded.biases[0])
            #expect(scale == 1)
            #expect(bias == value)
            // Every index must decode back to the constant, so every byte is
            // zero and the reconstruction is exact rather than merely close.
            #expect(encoded.packed.allSatisfy { $0 == 0 })
        }
    }
}
