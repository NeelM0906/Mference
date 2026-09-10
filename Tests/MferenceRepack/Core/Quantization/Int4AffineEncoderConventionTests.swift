import Foundation
import Testing
@testable import MferenceRepackCore

/// Locks the **affine-grid convention** `Int4AffineEncoder` implements.
///
/// W2.1a already proves the encoder is bit-identical to the runtime's own
/// reference quantizer. That says the repacker and the runtime agree; it says
/// nothing about how either compares to the outside world. W2.1b measured that
/// (docs/QUANTIZER_QUALITY.md) and found a real, systematic difference from
/// MLX's `affine_quantize`, which is what mlx-community's conversions are built
/// with:
///
///   * **Ours** — `scale = (max - min) / 15`, `bias = min`, both rounded to
///     BF16 before the indices are computed. The scale is always positive and
///     the bias is always the group minimum.
///   * **MLX** — anchors the grid on whichever endpoint has the larger
///     magnitude (so its stored scale is often *negative*) and then rescales so
///     that some integer bin lands exactly on 0.0, making 0.0 exactly
///     representable.
///
/// Neither is wrong. Ours has lower reconstruction error on 118 of 124 sampled
/// tensors of the Qwen 3.6 checkpoint; MLX's wins on tensors carrying a large
/// mass of (near-)exact zeros, where exact-zero representability pays off.
///
/// These tests exist so that convention is a **decision**, not an accident: if
/// someone later adopts zero-point snapping (a reasonable thing to want — see
/// the layer-0 `down_proj` result in the doc), these assertions fail and force
/// the W2.1b numbers to be re-measured rather than silently invalidated.
@Suite struct Int4AffineEncoderConventionTests {

    private static func groups(seed: UInt64, count: Int) -> [[Float]] {
        var state = seed
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: Int(state >> 33))) / Float(Int32.max)
        }
        return (0..<count).map { _ in
            (0..<Int4AffineEncoder.groupSize).map { _ in next() * 0.05 }
        }
    }

    @Test("scale is the BF16 min/max range over 15, bias is the BF16 minimum")
    func gridIsPlainMinMax() {
        for group in Self.groups(seed: 0x5EED_1234, count: 64) {
            let encoded = Int4AffineEncoder.encodeRow(group)
            let scale = Int4AffineEncoder.bf16ToFloat(encoded.scales[0])
            let bias = Int4AffineEncoder.bf16ToFloat(encoded.biases[0])
            let low = group.min()!
            let high = group.max()!
            // No zero-point snapping: the bias is the group minimum verbatim
            // (through BF16), and the scale spans the whole range.
            #expect(encoded.biases[0] == Int4AffineEncoder.bf16Bits(low))
            #expect(encoded.scales[0]
                == Int4AffineEncoder.bf16Bits((high - low) / 15.0))
            // A negative scale is MLX's signal that it anchored on the maximum.
            // Ours never does that.
            #expect(scale > 0)
            #expect(bias <= high)
        }
    }

    @Test("reconstruction stays inside the half-step bound the grid promises")
    func reconstructionWithinHalfStep() {
        for group in Self.groups(seed: 0xC0FF_EE01, count: 64) {
            let encoded = Int4AffineEncoder.encodeRow(group)
            let scale = Int4AffineEncoder.bf16ToFloat(encoded.scales[0])
            let bias = Int4AffineEncoder.bf16ToFloat(encoded.biases[0])
            // BF16 carries 8 mantissa bits, so rounding the bias can move the
            // grid by that much of the group's magnitude; everything beyond
            // that must be the honest half-step.
            let magnitude = max(abs(group.min()!), abs(group.max()!))
            let bound = scale / 2 + magnitude / 256 + .ulpOfOne
            for (index, value) in group.enumerated() {
                let byte = encoded.packed[index / 2]
                let nibble = index % 2 == 0 ? byte & 0x0F : byte >> 4
                let reconstructed = scale * Float(nibble) + bias
                #expect(abs(reconstructed - value) <= bound,
                        "index \(index): |\(reconstructed) - \(value)| > \(bound)")
            }
        }
    }

    @Test("a constant group reconstructs exactly")
    func constantGroupIsExact() {
        for value: Float in [0, 0.25, -3.5, 1.0 / 64.0] {
            let group = [Float](repeating: value,
                                count: Int4AffineEncoder.groupSize)
            let encoded = Int4AffineEncoder.encodeRow(group)
            let scale = Int4AffineEncoder.bf16ToFloat(encoded.scales[0])
            let bias = Int4AffineEncoder.bf16ToFloat(encoded.biases[0])
            #expect(scale == 1)
            #expect(bias == value)
            // Every index must decode back to the constant, so every nibble is
            // zero and the reconstruction is exact rather than merely close.
            #expect(encoded.packed.allSatisfy { $0 == 0 })
        }
    }
}
