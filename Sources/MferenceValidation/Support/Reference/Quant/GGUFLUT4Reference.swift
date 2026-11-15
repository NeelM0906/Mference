import Foundation
import Mference

/// CPU reference for expanding native GGUF blocks into LUT4 parts (WS4 U1
/// §3a), in natural element order. It is written per element from the block
/// layouts, not with llama.cpp's loop structure, so checking
/// `GGUFLUT4.weight` over these parts against `GGUFReferenceDequant` is a
/// cross-check rather than a copy compared with itself. The GPU expand kernel
/// is checked against this bit for bit.
public enum GGUFLUT4Reference {
    public struct Parts: Equatable, Sendable {
        /// One 4-bit index per weight, natural order.
        public var indices: [UInt8]
        /// fp16 bits, one per `bd` weights.
        public var d: [UInt16]
        /// One per `bm` weights; empty when the type has no `M`.
        public var m: [Int8]
    }

    @inline(__always)
    static func u16(_ b: UnsafeRawBufferPointer, _ at: Int) -> UInt16 { UInt16(b[at]) | UInt16(b[at + 1]) << 8 }

    @inline(__always)
    static func u32(_ b: UnsafeRawBufferPointer, _ at: Int) -> UInt32 {
        UInt32(u16(b, at)) | UInt32(u16(b, at + 2)) << 16
    }

    public static func decompose(_ type: GGUFLUT4.SourceType, _ blocks: UnsafeRawBufferPointer) -> Parts {
        let bb = type.blockBytes, bw = type.blockWeights
        precondition(blocks.count % bb == 0)
        let nBlocks = blocks.count / bb
        var parts = Parts(indices: [], d: [], m: [])
        parts.indices.reserveCapacity(nBlocks * bw)
        for i in 0..<nBlocks {
            let b = UnsafeRawBufferPointer(rebasing: blocks[(i * bb)..<((i + 1) * bb)])
            parts.d.append(u16(b, 0))
            for e in 0..<bw {
                parts.indices.append(index(type, b, e))
            }
            parts.m += multipliers(type, b)
        }
        return parts
    }

    /// The LUT4 index of element `e` of one block.
    static func index(_ type: GGUFLUT4.SourceType, _ b: UnsafeRawBufferPointer, _ e: Int) -> UInt8 {
        switch type {
        case .iq4_nl:
            // qs at 2; element j < 16 is the low nibble of qs[j], j ≥ 16 the high nibble of qs[j−16].
            let byte = b[2 + e % 16]
            return e < 16 ? byte & 0xF : byte >> 4
        case .iq4_xs:
            // qs at 8, 16 bytes per 32 elements, same nibble order as IQ4_NL.
            let g = e / 32, r = e % 32
            let byte = b[8 + 16 * g + r % 16]
            return r < 16 ? byte & 0xF : byte >> 4
        case .q2_0:
            // qs at 2; element e at byte e/4, bits 2·(e%4). Index = q (T = {−1, 0, 1, 2}).
            return (b[2 + e / 4] >> UInt8(2 * (e % 4))) & 0x3
        case .iq3_xxs:
            // qs at 2: 64 grid-index bytes (8 per 32 elements), then 8 × u32 sign/scale words.
            // Element e: group g = e/32; within it, pair l = (e%32)/8, half h = (e%8)/4, byte j = e%4.
            let g = e / 32, r = e % 32, l = r / 8, h = (r % 8) / 4, j = r % 4
            let gridIndex = Int(b[2 + 8 * g + 2 * l + h])
            let word = u32(b, 2 + 64 + 4 * g)
            let signs = GGUFQuantTables.ksignsIQ2XS[Int((word >> UInt32(7 * l)) & 127)]
            let negative = signs & UInt8(1 << (4 * h + j)) != 0
            let mag = UInt8((GGUFLUT4.iq3xxsIndexGrid[gridIndex] >> UInt16(3 * j)) & 0x7)
            return mag | (negative ? 8 : 0)
        case .iq2_s:
            // qs at 2: 32 low index bytes then 32 sign bytes; qh at 66; scales at 74.
            // Element e: group g = e/32, sub-block l = (e%32)/8, byte j = e%8.
            let g = e / 32, l = (e % 32) / 8, j = e % 8
            let low = Int(b[2 + 4 * g + l])
            let high = (Int(b[66 + g]) >> (2 * l)) & 0x3
            let gridIndex = low | high << 8
            let negative = b[2 + 32 + 4 * g + l] & UInt8(1 << j) != 0
            let mag = UInt8((GGUFLUT4.iq2sIndexGrid[gridIndex] >> UInt16(2 * j)) & 0x3)
            return mag | (negative ? 8 : 0)
        }
    }

    static func multipliers(_ type: GGUFLUT4.SourceType, _ b: UnsafeRawBufferPointer) -> [Int8] {
        switch type {
        case .iq4_nl, .q2_0:
            return []
        case .iq3_xxs:
            // M = 2·ls + 1 per 32, ls = top 4 bits of each sign/scale word.
            return (0..<8).map { g in Int8(2 * Int(u32(b, 2 + 64 + 4 * g) >> 28) + 1) }
        case .iq2_s:
            // M = 2·ls + 1 per 16, two 4-bit ls per scales byte (low nibble first).
            return (0..<16).map { s in
                let byte = b[74 + s / 2]
                return Int8(2 * Int(s % 2 == 0 ? byte & 0xF : byte >> 4) + 1)
            }
        case .iq4_xs:
            // M = ls − 32 per 32; ls = 4 low bits from scales_l, 2 high bits from scales_h.
            let scalesH = u16(b, 2)
            return (0..<8).map { g in
                let low = Int((b[4 + g / 2] >> UInt8(4 * (g % 2))) & 0xF)
                let high = Int((scalesH >> UInt16(2 * g)) & 0x3)
                return Int8((low | high << 4) - 32)
            }
        }
    }

    /// Dequantize LUT4 parts to fp32 with `GGUFLUT4.weight`.
    public static func dequantize(_ type: GGUFLUT4.SourceType, _ parts: Parts) -> [Float] {
        let p = GGUFLUT4.parameters(type)
        return parts.indices.enumerated().map { e, idx in
            GGUFLUT4.weight(
                d: parts.d[e / p.bd],
                m: p.bm == 0 ? 1 : parts.m[e / p.bm],
                k: p.k,
                t: p.table[Int(idx)])
        }
    }
}
