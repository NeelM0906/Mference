import Foundation
import Testing
@testable import Mference

/// The reference CPU dequantizers against ggml's own `dequantize_row_<type>`
/// (no model, no GPU).
@Suite("ReferenceDequantTests")
struct ReferenceDequantTests {
    /// Digests printed by `Scripts/native-reference/run_ggml_dequant_digests.sh`
    /// (ggml's `dequantize_row_<type>` at the pin, `-ffp-contract=off`).
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
}
