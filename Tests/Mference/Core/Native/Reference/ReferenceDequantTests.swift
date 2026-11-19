import Foundation
import Testing
@testable import Mference

/// The reference CPU dequantizers against ggml's own `dequantize_row_<type>`
/// (no model, no GPU).
@Suite("ReferenceDequantTests")
struct ReferenceDequantTests {
    /// Digests printed by `Scripts/native/reference/run_ggml_dequant_digests.sh`
    /// (ggml's `dequantize_row_<type>` at the pin, `-ffp-contract=off`); the
    /// run is recorded in `$PROGRAM/results/r/ggml-dequant-digests.log`.
    static let ggmlDigests: [(Native.GGMLType, UInt64, UInt64)] = [
        (.q8_0, 1, 0x5e1a_2faf_da0f_5878),
        (.q4_K, 2, 0x8a8a_08c2_7aed_cc28),
        (.q5_K, 3, 0xc1e4_2f70_fceb_e8e7),
        (.q6_K, 4, 0xb904_0e71_8d60_96e3),
        (.iq4_xs, 5, 0x4108_1ba9_3f23_9fd4),
        (.iq3_xxs, 6, 0x0687_a55e_e4d8_8d68),
        (.iq3_s, 7, 0xd6e1_c44a_10d7_38a7),
    ]

    @Test func cpuDequantMatchesGGMLBitForBit() throws {
        for (type, seed, digest) in Self.ggmlDigests {
            let blocks = RefTest.randomBlocks(type, blocks: 64, seed: seed)
            let values = blocks.withUnsafeBytes { Native.ReferenceDequant.dequantizeRow(type, $0.baseAddress!, blocks: 64) }
            #expect(RefTest.fnv(values) == digest, "\(type)")
        }
    }

    /// The same generator's unquantized values, against the conversions
    /// ggml's `to_float` uses (`GGML_FP16_TO_FP32`, `GGML_BF16_TO_FP32`, and
    /// the F32 bit pattern itself).
    static let unquantizedDigests: [(Native.GGMLType, UInt64, UInt64)] = [
        (.f16, 8, 0x1229_e905_25e4_1162),
        (.bf16, 9, 0x2092_e35c_c9e2_bdc9),
        (.f32, 10, 0x6d23_bdce_e8cf_af27),
    ]

    @Test func cpuUnquantizedConversionsMatchGGML() throws {
        for (type, seed, digest) in Self.unquantizedDigests {
            var rng = RefTest.SplitMix64(seed)
            var bytes: [UInt8] = []
            for _ in 0..<1024 {
                if type == .f32 {
                    let bits = UInt32(truncatingIfNeeded: rng.next() & 0x807F_FFFF) | 0x3F00_0000
                    bytes += withUnsafeBytes(of: bits.littleEndian) { Array($0) }
                } else {
                    let h = rng.scale16()
                    bytes += [UInt8(h & 0xff), UInt8(h >> 8)]
                }
            }
            let values = bytes.withUnsafeBytes { Native.ReferenceDequant.dequantizeRow(type, $0.baseAddress!, blocks: 1024) }
            #expect(RefTest.fnv(values) == digest, "\(type)")
        }
    }
}
