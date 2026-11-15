import Foundation
import Mference

/// CPU reference dequantization of the GGUF block types a WS4 install routes
/// through LUT4 (U1 §1, §3a). Each function follows the loop structure and
/// the fp32 evaluation order of `dequantize_row_*` in llama.cpp b1b770d9
/// `ggml/src/ggml-quants.c`, so its output is llama.cpp's fp32 bit for bit.
/// The LUT4 path is checked against these, not against itself.
///
/// Swift does not contract `a * b * c` into a fused multiply-add, so the
/// left-to-right products below round exactly as the C source does.
public enum GGUFReferenceDequant {
    public static let qkK = 256

    @inline(__always)
    static func fp16(_ b: UnsafeRawBufferPointer, _ at: Int) -> Float {
        Float(Float16(bitPattern: UInt16(b[at]) | UInt16(b[at + 1]) << 8))
    }

    @inline(__always)
    static func u32(_ b: UnsafeRawBufferPointer, _ at: Int) -> UInt32 {
        UInt32(b[at]) | UInt32(b[at + 1]) << 8 | UInt32(b[at + 2]) << 16 | UInt32(b[at + 3]) << 24
    }

    @inline(__always)
    static func gridByte(_ entry: UInt64, _ j: Int) -> Float { Float(UInt8(truncatingIfNeeded: entry >> (8 * UInt64(j)))) }

    /// `block_iq3_xxs`: `d`, then `qs[96]` (64 grid-index bytes, then 8 × u32
    /// of four 7-bit sign indices and a 4-bit scale). 98 bytes per 256.
    public static func iq3XXS(_ blocks: UnsafeRawBufferPointer) -> [Float] {
        let bytes = 98
        precondition(blocks.count % bytes == 0)
        var y = [Float]()
        y.reserveCapacity(blocks.count / bytes * qkK)
        for i in 0..<(blocks.count / bytes) {
            let base = i * bytes
            let d = fp16(blocks, base)
            var qs = base + 2
            let scalesAndSigns = base + 2 + qkK / 4
            for ib32 in 0..<(qkK / 32) {
                let aux32 = u32(blocks, scalesAndSigns + 4 * ib32)
                let db = d * (0.5 + Float(aux32 >> 28)) * 0.5
                for l in 0..<4 {
                    let signs = GGUFQuantTables.ksignsIQ2XS[Int((aux32 >> UInt32(7 * l)) & 127)]
                    let grid1 = UInt64(GGUFQuantTables.iq3xxsGrid[Int(blocks[qs + 2 * l])])
                    let grid2 = UInt64(GGUFQuantTables.iq3xxsGrid[Int(blocks[qs + 2 * l + 1])])
                    var lo = [Float](repeating: 0, count: 4), hi = lo
                    for j in 0..<4 {
                        lo[j] = db * gridByte(grid1, j) * (signs & UInt8(1 << j) != 0 ? -1 : 1)
                        hi[j] = db * gridByte(grid2, j) * (signs & UInt8(1 << (j + 4)) != 0 ? -1 : 1)
                    }
                    y += lo
                    y += hi
                }
                qs += 8
            }
        }
        return y
    }

    /// `block_iq2_s`: `d`, `qs[64]` (32 low index bytes, then 32 sign bytes),
    /// `qh[8]`, `scales[8]`. 82 bytes per 256.
    public static func iq2S(_ blocks: UnsafeRawBufferPointer) -> [Float] {
        let bytes = 82
        precondition(blocks.count % bytes == 0)
        var y = [Float]()
        y.reserveCapacity(blocks.count / bytes * qkK)
        for i in 0..<(blocks.count / bytes) {
            let base = i * bytes
            let d = fp16(blocks, base)
            var qs = base + 2
            let qh = base + 2 + qkK / 4
            let scales = qh + qkK / 32
            var signs = qs + qkK / 8
            for ib32 in 0..<(qkK / 32) {
                let sc = blocks[scales + ib32]
                let db = [d * (0.5 + Float(sc & 0xF)) * 0.25, d * (0.5 + Float(sc >> 4)) * 0.25]
                for l in 0..<4 {
                    let dl = db[l / 2]
                    let index = Int(blocks[qs + l]) | (Int(blocks[qh + ib32]) << (8 - 2 * l) & 0x300)
                    let grid = GGUFQuantTables.iq2sGrid[index]
                    for j in 0..<8 {
                        y.append(dl * gridByte(grid, j) * (blocks[signs + l] & UInt8(1 << j) != 0 ? -1 : 1))
                    }
                }
                qs += 4
                signs += 4
            }
        }
        return y
    }

    /// `block_iq4_xs`: `d`, `scales_h` (u16), `scales_l[4]`, `qs[128]`.
    /// 136 bytes per 256.
    public static func iq4XS(_ blocks: UnsafeRawBufferPointer) -> [Float] {
        let bytes = 136
        precondition(blocks.count % bytes == 0)
        var y = [Float](repeating: 0, count: blocks.count / bytes * qkK)
        var out = 0
        for i in 0..<(blocks.count / bytes) {
            let base = i * bytes
            let d = fp16(blocks, base)
            let scalesH = UInt16(blocks[base + 2]) | UInt16(blocks[base + 3]) << 8
            var qs = base + 8
            for ib in 0..<(qkK / 32) {
                let ls = Int((blocks[base + 4 + ib / 2] >> UInt8(4 * (ib % 2))) & 0xF)
                    | Int((scalesH >> UInt16(2 * ib)) & 3) << 4
                let dl = d * Float(ls - 32)
                for j in 0..<16 {
                    y[out + j] = dl * Float(GGUFQuantTables.kvaluesIQ4NL[Int(blocks[qs + j] & 0xF)])
                    y[out + j + 16] = dl * Float(GGUFQuantTables.kvaluesIQ4NL[Int(blocks[qs + j] >> 4)])
                }
                out += 32
                qs += 16
            }
        }
        return y
    }

    /// `block_iq4_nl`: `d`, `qs[16]`. 18 bytes per 32.
    public static func iq4NL(_ blocks: UnsafeRawBufferPointer) -> [Float] {
        let bytes = 18
        precondition(blocks.count % bytes == 0)
        var y = [Float](repeating: 0, count: blocks.count / bytes * 32)
        for i in 0..<(blocks.count / bytes) {
            let base = i * bytes
            let d = fp16(blocks, base)
            for j in 0..<16 {
                y[i * 32 + j] = d * Float(GGUFQuantTables.kvaluesIQ4NL[Int(blocks[base + 2 + j] & 0xF)])
                y[i * 32 + j + 16] = d * Float(GGUFQuantTables.kvaluesIQ4NL[Int(blocks[base + 2 + j] >> 4)])
            }
        }
        return y
    }

    /// `block_q2_0`: `d`, `qs[16]` (2 bits per element). 18 bytes per 64.
    public static func q2_0(_ blocks: UnsafeRawBufferPointer) -> [Float] {
        let bytes = 18
        precondition(blocks.count % bytes == 0)
        var y = [Float](repeating: 0, count: blocks.count / bytes * 64)
        for i in 0..<(blocks.count / bytes) {
            let base = i * bytes
            let d = fp16(blocks, base)
            for j in 0..<64 {
                let q = (blocks[base + 2 + j / 4] >> UInt8((j % 4) * 2)) & 0x03
                y[i * 64 + j] = Float(Int(q) - 1) * d
            }
        }
        return y
    }
}
