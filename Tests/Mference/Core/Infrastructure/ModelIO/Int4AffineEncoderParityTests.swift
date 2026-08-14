import Foundation
import Testing
import MferenceRepackCore
@testable import Mference

/// Phase A W2a gate: the repacker's production quantizer must match the
/// runtime module's reference (`Quantization.quantizeInt4Affine`) bit for
/// bit — packed nibbles, BF16 scale bits, BF16 bias bits — so weights we
/// quantize ourselves are indistinguishable from the fixture semantics the
/// runtime decode was verified against.
@Suite struct Int4AffineEncoderParityTests {

    private static func assertParity(_ row: [Float],
                                     _ note: Comment,
                                     sourceLocation: SourceLocation = #_sourceLocation) {
        let reference = Quantization.quantizeInt4Affine(row)
        let encoded = Int4AffineEncoder.encodeRow(row)
        #expect(encoded.packed == reference.packed, note,
                sourceLocation: sourceLocation)
        #expect(encoded.scales == reference.scales, note,
                sourceLocation: sourceLocation)
        #expect(encoded.biases == reference.biases, note,
                sourceLocation: sourceLocation)
    }

    @Test("Random rows match the reference bit for bit",
          arguments: [64, 128, 2048, 4096])
    func randomRowsMatch(length: Int) {
        var state: UInt64 = 0x9E3779B97F4A7C15 &* UInt64(length)
        func nextFloat() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u = Float(state >> 40) / Float(1 << 24)
            return (u - 0.5) * 4.0
        }
        for trial in 0..<8 {
            let row = (0..<length).map { _ in nextFloat() }
            Self.assertParity(row, "length \(length) trial \(trial)")
        }
    }

    @Test("Edge cases match: constant, zero, extreme, sign-skewed groups")
    func edgeCasesMatch() {
        Self.assertParity([Float](repeating: 0, count: 64), "all zero")
        Self.assertParity([Float](repeating: 3.25, count: 128), "constant nonzero")
        Self.assertParity((0..<64).map { Float($0) * 1e-30 }, "denormal-scale group")
        Self.assertParity((0..<64).map { $0 == 0 ? -1e4 : 1e4 }, "extreme spread")
        Self.assertParity((0..<64).map { -Float($0) - 1 }, "all negative")
        // One constant group followed by a spread group in the same row.
        Self.assertParity([Float](repeating: -2, count: 64)
                     + (0..<64).map { Float($0) / 7 }, "mixed groups")
    }

    @Test("Tensor encode equals row-by-row encode")
    func tensorMatchesRows() {
        var state: UInt64 = 0xC0FFEE
        func nextFloat() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return (Float(state >> 40) / Float(1 << 24) - 0.5) * 2
        }
        let rows = 6
        let cols = 192
        let values = (0..<rows * cols).map { _ in nextFloat() }
        let tensor = values.withUnsafeBufferPointer {
            Int4AffineEncoder.encodeTensor($0, rowLength: cols)
        }
        var packed = [UInt8]()
        var scales = [UInt16]()
        var biases = [UInt16]()
        for r in 0..<rows {
            let row = Array(values[r * cols ..< (r + 1) * cols])
            let e = Int4AffineEncoder.encodeRow(row)
            packed += e.packed
            scales += e.scales
            biases += e.biases
        }
        #expect(tensor.packed == packed)
        #expect(tensor.scales == scales)
        #expect(tensor.biases == biases)
    }
}
