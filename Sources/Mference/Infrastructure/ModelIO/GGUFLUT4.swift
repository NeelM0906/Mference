import Foundation

/// The LUT4 routed-expert execution format (WS4 U1 §3a).
///
/// A weight is `float(D[b]) * float(M[g]) * 2^K * T[idx]`:
/// - `idx`: a 4-bit index into the per-tensor 16-entry int8 table `T`;
/// - `D`: the source block's fp16 scale, one per `bd` weights;
/// - `M`: a small int8 multiplier, one per `bm` weights (absent when `bm == 0`);
/// - `K`: a per-type power of two.
///
/// Every product is exact in fp32 (U1 §1: at most 23 significant bits, no
/// subnormal), so any grouping reproduces llama.cpp's dequantized fp32 bit for
/// bit. For the grid types the index is `magnitudeIndex | sign << 3`, where
/// `magnitudeIndex` is the position of the grid byte in the type's sorted
/// magnitude set; `T[i]` is that magnitude and `T[i + 8]` its negation.
public enum GGUFLUT4 {
    /// GGUF source types that execute as LUT4. Raw values are ggml type ids
    /// (`ggml/include/ggml.h` at llama.cpp b1b770d9).
    public enum SourceType: UInt32, Sendable, CaseIterable {
        case iq3_xxs = 18
        case iq4_nl = 20
        case iq2_s = 22
        case iq4_xs = 23
        case q2_0 = 42

        public var blockWeights: Int {
            switch self {
            case .iq3_xxs, .iq2_s, .iq4_xs: 256
            case .iq4_nl: 32
            case .q2_0: 64
            }
        }

        public var blockBytes: Int {
            switch self {
            case .iq3_xxs: 98
            case .iq2_s: 82
            case .iq4_xs: 136
            case .iq4_nl, .q2_0: 18
            }
        }
    }

    public struct Parameters: Equatable, Sendable {
        /// Weights per fp16 `D`.
        public let bd: Int
        /// Weights per int8 `M`; 0 when there is no `M` (M ≡ 1).
        public let bm: Int
        /// Power-of-two exponent applied to every weight.
        public let k: Int
        /// The 16-entry signed table (unused entries are 0).
        public let table: [Int8]
    }

    /// Sorted distinct magnitudes of a grid's bytes.
    static func magnitudes<T: FixedWidthInteger>(_ grid: [T], bytesPerEntry: Int) -> [UInt8] {
        var set = Set<UInt8>()
        for entry in grid {
            for j in 0..<bytesPerEntry { set.insert(UInt8(truncatingIfNeeded: entry >> (8 * j))) }
        }
        return set.sorted()
    }

    public static let iq3xxsMagnitudes = magnitudes(GGUFQuantTables.iq3xxsGrid, bytesPerEntry: 4)
    public static let iq2sMagnitudes = magnitudes(GGUFQuantTables.iq2sGrid, bytesPerEntry: 8)

    static func signedTable(_ mags: [UInt8]) -> [Int8] {
        precondition(mags.count <= 8 && mags.allSatisfy { $0 <= 127 })
        var t = [Int8](repeating: 0, count: 16)
        for (i, m) in mags.enumerated() {
            t[i] = Int8(m)
            t[i + 8] = -Int8(m)
        }
        return t
    }

    public static func parameters(_ type: SourceType) -> Parameters {
        switch type {
        case .iq3_xxs: Parameters(bd: 256, bm: 32, k: -2, table: signedTable(iq3xxsMagnitudes))
        case .iq2_s: Parameters(bd: 256, bm: 16, k: -3, table: signedTable(iq2sMagnitudes))
        case .iq4_xs: Parameters(bd: 256, bm: 32, k: 0, table: GGUFQuantTables.kvaluesIQ4NL)
        case .iq4_nl: Parameters(bd: 32, bm: 0, k: 0, table: GGUFQuantTables.kvaluesIQ4NL)
        case .q2_0: Parameters(bd: 64, bm: 0, k: 0, table: [-1, 0, 1, 2] + [Int8](repeating: 0, count: 12))
        }
    }

    /// `iq3xxs_grid` as magnitude indices: entry `g`, byte `j` → bits
    /// `3j..<3j+3` of `iq3xxsIndexGrid[g]`.
    public static let iq3xxsIndexGrid: [UInt16] = GGUFQuantTables.iq3xxsGrid.map { entry in
        var packed: UInt16 = 0
        for j in 0..<4 {
            let m = UInt8(truncatingIfNeeded: entry >> (8 * UInt32(j)))
            packed |= UInt16(iq3xxsMagnitudes.firstIndex(of: m)!) << (3 * j)
        }
        return packed
    }

    /// `iq2s_grid` as magnitude indices: entry `g`, byte `j` → bits
    /// `2j..<2j+2` of `iq2sIndexGrid[g]`.
    public static let iq2sIndexGrid: [UInt16] = GGUFQuantTables.iq2sGrid.map { entry in
        var packed: UInt16 = 0
        for j in 0..<8 {
            let m = UInt8(truncatingIfNeeded: entry >> (8 * UInt64(j)))
            packed |= UInt16(iq2sMagnitudes.firstIndex(of: m)!) << (2 * j)
        }
        return packed
    }

    /// The exact fp32 value of one LUT4 weight, `((D · M) · T) · 2^K`. Every
    /// step is exact (at most 23 significant bits), and every product stays
    /// in fp32 so a zero carries llama.cpp's sign: IQ4_XS has M = 0 when
    /// ls = 32, and llama.cpp's `(d · 0) · kvalue` is −0 for a negative table
    /// entry. Multiplying M · T as integers first would give +0 there.
    @inline(__always)
    public static func weight(d: UInt16, m: Int8, k: Int, t: Int8) -> Float {
        Float(Float16(bitPattern: d)) * Float(m) * Float(t) * Float(sign: .plus, exponent: k, significand: 1)
    }
}
