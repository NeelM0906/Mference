import Foundation

extension Native {
    /// CPU twins of the reference dequantizers: direct ports of ggml's
    /// `dequantize_row_<type>` at the pin (`ggml-quants.c`), one block at a
    /// time, with the same expression order as `native_reference.metal`.
    /// Swift does not contract `a * b - c` into an FMA, so these match ggml's
    /// C code compiled with `-ffp-contract=off` bit for bit
    /// (`ReferenceDequantTests` checks digests produced that way).
    public enum ReferenceDequant {
        /// Types the reference dequantizes.
        public static let supportedTypes: Set<GGMLType> =
            [.f32, .f16, .bf16, .q8_0, .q4_K, .q5_K, .q6_K, .iq4_xs, .iq3_xxs, .iq3_s]

        @inline(__always) static func f16(_ b: UnsafeRawPointer, _ offset: Int) -> Float {
            Float(Float16(bitPattern: UInt16(b.load(fromByteOffset: offset, as: UInt8.self))
                | (UInt16(b.load(fromByteOffset: offset + 1, as: UInt8.self)) << 8)))
        }

        @inline(__always) static func u8(_ b: UnsafeRawPointer, _ offset: Int) -> Int {
            Int(b.load(fromByteOffset: offset, as: UInt8.self))
        }

        @inline(__always) static func i8(_ b: UnsafeRawPointer, _ offset: Int) -> Int {
            Int(b.load(fromByteOffset: offset, as: Int8.self))
        }

        @inline(__always) static func u32(_ b: UnsafeRawPointer, _ offset: Int) -> UInt32 {
            UInt32(u8(b, offset)) | (UInt32(u8(b, offset + 1)) << 8)
                | (UInt32(u8(b, offset + 2)) << 16) | (UInt32(u8(b, offset + 3)) << 24)
        }

        static func scaleMinK4(_ j: Int, _ q: UnsafeRawPointer, _ base: Int) -> (Int, Int) {
            if j < 4 {
                return (u8(q, base + j) & 63, u8(q, base + j + 4) & 63)
            }
            let d = (u8(q, base + j + 4) & 0xF) | ((u8(q, base + j - 4) >> 6) << 4)
            let m = (u8(q, base + j + 4) >> 4) | ((u8(q, base + j) >> 6) << 4)
            return (d, m)
        }

        /// Dequantize one block of `type` at `block` into `out[0 ..< elements]`.
        public static func dequantizeBlock(_ type: GGMLType, _ block: UnsafeRawPointer,
                                           into out: UnsafeMutablePointer<Float>) {
            let b = block
            switch type {
            case .f32:
                out[0] = Float(bitPattern: u32(b, 0))
            case .f16:
                out[0] = f16(b, 0)
            case .bf16:
                out[0] = Float(bitPattern: (UInt32(u8(b, 0)) | (UInt32(u8(b, 1)) << 8)) << 16)
            case .q8_0:
                let d = f16(b, 0)
                for j in 0..<32 { out[j] = Float(i8(b, 2 + j)) * d }
            case .q4_K:
                let d = f16(b, 0), dmin = f16(b, 2)
                var q = 16, isc = 0, o = 0
                for _ in stride(from: 0, to: 256, by: 64) {
                    var (sc, m) = scaleMinK4(isc + 0, b, 4)
                    let d1 = d * Float(sc), m1 = dmin * Float(m)
                    (sc, m) = scaleMinK4(isc + 1, b, 4)
                    let d2 = d * Float(sc), m2 = dmin * Float(m)
                    for l in 0..<32 { out[o] = d1 * Float(u8(b, q + l) & 0xF) - m1; o += 1 }
                    for l in 0..<32 { out[o] = d2 * Float(u8(b, q + l) >> 4) - m2; o += 1 }
                    q += 32; isc += 2
                }
            case .q5_K:
                let d = f16(b, 0), dmin = f16(b, 2)
                let qh = 16
                var ql = 48, isc = 0, o = 0
                var u1 = 1, u2 = 2
                for _ in stride(from: 0, to: 256, by: 64) {
                    var (sc, m) = scaleMinK4(isc + 0, b, 4)
                    let d1 = d * Float(sc), m1 = dmin * Float(m)
                    (sc, m) = scaleMinK4(isc + 1, b, 4)
                    let d2 = d * Float(sc), m2 = dmin * Float(m)
                    for l in 0..<32 {
                        out[o] = d1 * Float((u8(b, ql + l) & 0xF) + ((u8(b, qh + l) & u1) != 0 ? 16 : 0)) - m1
                        o += 1
                    }
                    for l in 0..<32 {
                        out[o] = d2 * Float((u8(b, ql + l) >> 4) + ((u8(b, qh + l) & u2) != 0 ? 16 : 0)) - m2
                        o += 1
                    }
                    ql += 32; isc += 2
                    u1 <<= 2; u2 <<= 2
                }
            case .q6_K:
                let d = f16(b, 208)
                var ql = 0, qh = 128, sc = 192, y = 0
                for _ in stride(from: 0, to: 256, by: 128) {
                    for l in 0..<32 {
                        let isc = l / 16
                        let q1 = Int(Int8(truncatingIfNeeded: (u8(b, ql + l) & 0xF) | (((u8(b, qh + l) >> 0) & 3) << 4))) - 32
                        let q2 = Int(Int8(truncatingIfNeeded: (u8(b, ql + l + 32) & 0xF) | (((u8(b, qh + l) >> 2) & 3) << 4))) - 32
                        let q3 = Int(Int8(truncatingIfNeeded: (u8(b, ql + l) >> 4) | (((u8(b, qh + l) >> 4) & 3) << 4))) - 32
                        let q4 = Int(Int8(truncatingIfNeeded: (u8(b, ql + l + 32) >> 4) | (((u8(b, qh + l) >> 6) & 3) << 4))) - 32
                        out[y + l + 0] = d * Float(i8(b, sc + isc + 0)) * Float(q1)
                        out[y + l + 32] = d * Float(i8(b, sc + isc + 2)) * Float(q2)
                        out[y + l + 64] = d * Float(i8(b, sc + isc + 4)) * Float(q3)
                        out[y + l + 96] = d * Float(i8(b, sc + isc + 6)) * Float(q4)
                    }
                    y += 128; ql += 64; qh += 32; sc += 8
                }
            case .iq4_xs:
                let d = f16(b, 0)
                let scalesH = u8(b, 2) | (u8(b, 3) << 8)
                var qs = 8, y = 0
                let k = ReferenceQuantTables.kvalues_iq4nl
                for ib in 0..<8 {
                    let ls = ((u8(b, 4 + ib / 2) >> (4 * (ib % 2))) & 0xf) | (((scalesH >> (2 * ib)) & 3) << 4)
                    let dl = d * Float(ls - 32)
                    for j in 0..<16 {
                        out[y + j + 0] = dl * Float(k[u8(b, qs + j) & 0xf])
                        out[y + j + 16] = dl * Float(k[u8(b, qs + j) >> 4])
                    }
                    y += 32; qs += 16
                }
            case .iq3_xxs:
                let d = f16(b, 0)
                var qs = 2, y = 0
                let sas = 2 + 64
                let grid = ReferenceQuantTables.iq3xxs_grid
                let signsTable = ReferenceQuantTables.ksigns_iq2xs
                let mask = ReferenceQuantTables.kmask_iq2xs
                for ib32 in 0..<8 {
                    let aux32 = u32(b, sas + 4 * ib32)
                    let db = d * (0.5 + Float(aux32 >> 28)) * 0.5
                    for l in 0..<4 {
                        let signs = signsTable[Int((aux32 >> (7 * UInt32(l))) & 127)]
                        let g1 = grid[u8(b, qs + 2 * l + 0)]
                        let g2 = grid[u8(b, qs + 2 * l + 1)]
                        for j in 0..<4 {
                            let b1 = Float((g1 >> (8 * UInt32(j))) & 0xff)
                            let b2 = Float((g2 >> (8 * UInt32(j))) & 0xff)
                            out[y + j + 0] = db * b1 * ((signs & mask[j + 0]) != 0 ? -1 : 1)
                            out[y + j + 4] = db * b2 * ((signs & mask[j + 4]) != 0 ? -1 : 1)
                        }
                        y += 8
                    }
                    qs += 8
                }
            case .iq3_s:
                let d = f16(b, 0)
                var qs = 2, qh = 66, signs = 74, y = 0
                let scales = 106
                let grid = ReferenceQuantTables.iq3s_grid
                let mask = ReferenceQuantTables.kmask_iq2xs
                for ib32 in stride(from: 0, to: 8, by: 2) {
                    let db1 = d * Float(1 + 2 * (u8(b, scales + ib32 / 2) & 0xf))
                    let db2 = d * Float(1 + 2 * (u8(b, scales + ib32 / 2) >> 4))
                    for (half, db) in [(0, db1), (1, db2)] {
                        let h = u8(b, qh + half)
                        for l in 0..<4 {
                            let g1 = grid[u8(b, qs + 2 * l + 0) | ((h << (8 - 2 * l)) & 256)]
                            let g2 = grid[u8(b, qs + 2 * l + 1) | ((h << (7 - 2 * l)) & 256)]
                            let s = UInt8(u8(b, signs + l))
                            for j in 0..<4 {
                                let b1 = Float((g1 >> (8 * UInt32(j))) & 0xff)
                                let b2 = Float((g2 >> (8 * UInt32(j))) & 0xff)
                                out[y + j + 0] = db * b1 * ((s & mask[j + 0]) != 0 ? -1 : 1)
                                out[y + j + 4] = db * b2 * ((s & mask[j + 4]) != 0 ? -1 : 1)
                            }
                            y += 8
                        }
                        qs += 8; signs += 4
                    }
                    qh += 2
                }
            default:
                preconditionFailure("ReferenceDequant: unsupported type \(type)")
            }
        }

        /// Dequantize `count` blocks laid out back to back (GGUF order).
        public static func dequantizeRow(_ type: GGMLType, _ bytes: UnsafeRawPointer, blocks: Int) -> [Float] {
            guard let format = QuantBlockFormat.of(type) else {
                preconditionFailure("ReferenceDequant: no block format for \(type)")
            }
            var out = [Float](repeating: 0, count: blocks * format.elements)
            out.withUnsafeMutableBufferPointer { o in
                for blk in 0..<blocks {
                    dequantizeBlock(type, bytes + blk * format.bytes, into: o.baseAddress! + blk * format.elements)
                }
            }
            return out
        }

        /// Gather the GGUF-order bytes of block `block` of row `row` of a
        /// tensor stored in `layout` at `base` (the CPU twin of
        /// `ref_gather_block`).
        public static func gatherBlock(base: UnsafeRawPointer, layout: TileLayout, format: QuantBlockFormat,
                                       row: Int, block: Int, blocksPerRow: Int,
                                       into out: UnsafeMutablePointer<UInt8>) {
            if layout == .gguf {
                let src = base + TileLayout.ggufOffset(row: row, block: block, byteInBlock: 0,
                                                       blocksPerRow: blocksPerRow, format: format)
                out.update(from: src.assumingMemoryBound(to: UInt8.self), count: format.bytes)
                return
            }
            for i in 0..<format.bytes {
                let off = layout.packedOffset(row: row, block: block, byteInBlock: i,
                                              blocksPerRow: blocksPerRow, format: format)
                out[i] = base.load(fromByteOffset: off, as: UInt8.self)
            }
        }

        /// Row `row` of `tensor`, dequantized (reads `tensor.buffer`, which
        /// must be CPU-visible).
        public static func row(of tensor: QuantTensor, _ row: Int) -> [Float] {
            guard let format = QuantBlockFormat.of(tensor.type) else {
                preconditionFailure("ReferenceDequant: no block format for \(tensor.type)")
            }
            let blocksPerRow = tensor.cols / format.elements
            let base = UnsafeRawPointer(tensor.buffer.contents() + tensor.offset)
            var out = [Float](repeating: 0, count: tensor.cols)
            var block = [UInt8](repeating: 0, count: format.bytes)
            out.withUnsafeMutableBufferPointer { o in
                block.withUnsafeMutableBufferPointer { blk in
                    for b in 0..<blocksPerRow {
                        gatherBlock(base: base, layout: tensor.layout, format: format, row: row, block: b,
                                    blocksPerRow: blocksPerRow, into: blk.baseAddress!)
                        dequantizeBlock(tensor.type, UnsafeRawPointer(blk.baseAddress!),
                                        into: o.baseAddress! + b * format.elements)
                    }
                }
            }
            return out
        }

        /// Every value of a 1-D or 2-D tensor (small tensors only: tests and
        /// norm weights).
        public static func values(of tensor: QuantTensor) -> [Float] {
            (0..<tensor.rows).flatMap { row(of: tensor, $0) }
        }
    }
}
