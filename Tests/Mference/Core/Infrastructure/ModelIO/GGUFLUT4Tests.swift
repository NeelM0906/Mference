import CryptoKit
import Foundation
import Testing
@testable import Mference
import MferenceValidationSupport

/// Random source blocks of one LUT4 type. Every byte pattern is a valid block
/// except a non-finite fp16 `d`, which is redrawn: exactness is a statement
/// about finite scales.
func randomGGUFBlocks(_ type: GGUFLUT4.SourceType, count: Int, rng: inout SplitMix64) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: count * type.blockBytes)
    for i in 0..<bytes.count { bytes[i] = UInt8(truncatingIfNeeded: rng.next()) }
    for block in 0..<count {
        let at = block * type.blockBytes
        while bytes[at + 1] & 0x7C == 0x7C {  // fp16 exponent all ones: inf or NaN
            bytes[at] = UInt8(truncatingIfNeeded: rng.next())
            bytes[at + 1] = UInt8(truncatingIfNeeded: rng.next())
        }
    }
    return bytes
}

@Suite struct GGUFQuantTablesTests {
    @Test func tablesMatchTheGeneratedDigest() {
        var bytes = Data()
        GGUFQuantTables.ksignsIQ2XS.withUnsafeBytes { bytes.append(contentsOf: $0) }
        GGUFQuantTables.kvaluesIQ4NL.withUnsafeBytes { bytes.append(contentsOf: $0) }
        for v in GGUFQuantTables.iq3xxsGrid { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
        for v in GGUFQuantTables.iq2sGrid { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        #expect(digest == GGUFQuantTables.tableDigest)
        #expect(GGUFQuantTables.ksignsIQ2XS.count == 128)
        #expect(GGUFQuantTables.iq3xxsGrid.count == 256)
        #expect(GGUFQuantTables.iq2sGrid.count == 1024)
    }

    /// U1 §1's magnitude sets, now derived from the tables themselves.
    @Test func gridMagnitudesAreTheRecordedSets() {
        #expect(GGUFLUT4.iq3xxsMagnitudes == [4, 12, 20, 28, 36, 44, 52, 62])
        #expect(GGUFLUT4.iq2sMagnitudes == [8, 25, 43])
    }

    @Test func indexGridsRoundTripToTheGrids() {
        for (g, entry) in GGUFQuantTables.iq3xxsGrid.enumerated() {
            for j in 0..<4 {
                let mag = GGUFLUT4.iq3xxsMagnitudes[Int((GGUFLUT4.iq3xxsIndexGrid[g] >> UInt16(3 * j)) & 7)]
                #expect(mag == UInt8(truncatingIfNeeded: entry >> UInt32(8 * j)))
            }
        }
        for (g, entry) in GGUFQuantTables.iq2sGrid.enumerated() {
            for j in 0..<8 {
                let mag = GGUFLUT4.iq2sMagnitudes[Int((GGUFLUT4.iq2sIndexGrid[g] >> UInt16(2 * j)) & 3)]
                #expect(mag == UInt8(truncatingIfNeeded: entry >> UInt64(8 * j)))
            }
        }
    }
}

/// LUT4 (`D · M · 2^K · T[idx]`) reproduces llama.cpp's dequantized fp32 bit
/// for bit for every source type (U1 §1). The reference follows llama.cpp's
/// loops and evaluation order; the decomposition is written per element.
@Suite struct GGUFLUT4ExactnessTests {
    static let blocksPerType = 1_500

    @Test(arguments: GGUFLUT4.SourceType.allCases)
    func lut4ReproducesReferenceBits(type: GGUFLUT4.SourceType) {
        var rng = SeedTree(0x4C55_5434).key("lut4-exact-\(type.rawValue)")
        let bytes = randomGGUFBlocks(type, count: Self.blocksPerType, rng: &rng)
        let (reference, lut4): ([Float], [Float]) = bytes.withUnsafeBytes { raw in
            let ref: [Float]
            switch type {
            case .iq3_xxs: ref = GGUFReferenceDequant.iq3XXS(raw)
            case .iq2_s: ref = GGUFReferenceDequant.iq2S(raw)
            case .iq4_xs: ref = GGUFReferenceDequant.iq4XS(raw)
            case .iq4_nl: ref = GGUFReferenceDequant.iq4NL(raw)
            case .q2_0: ref = GGUFReferenceDequant.q2_0(raw)
            }
            return (ref, GGUFLUT4Reference.dequantize(type, GGUFLUT4Reference.decompose(type, raw)))
        }
        #expect(reference.count == Self.blocksPerType * type.blockWeights)
        #expect(lut4.count == reference.count)
        let mismatches = zip(reference, lut4).filter { $0.bitPattern != $1.bitPattern }.count
        #expect(mismatches == 0)
    }

    @Test func parametersMatchTheFormatRecord() {
        #expect(GGUFLUT4.parameters(.iq3_xxs).bd == 256 && GGUFLUT4.parameters(.iq3_xxs).bm == 32)
        #expect(GGUFLUT4.parameters(.iq2_s).bm == 16 && GGUFLUT4.parameters(.iq2_s).k == -3)
        #expect(GGUFLUT4.parameters(.iq4_nl).bd == 32 && GGUFLUT4.parameters(.iq4_nl).bm == 0)
        #expect(GGUFLUT4.parameters(.q2_0).bd == 64)
        for type in GGUFLUT4.SourceType.allCases {
            #expect(GGUFLUT4.parameters(type).table.count == 16)
        }
    }
}
