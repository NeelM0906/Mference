import Accelerate
import Foundation
import Metal

/// A reference matrix whose rows live in a Metal buffer in one of the
/// shipped file's formats, decoded on demand (nothing is copied into Swift
/// heap memory beyond a tile of rows). The CPU reference reads exactly the bytes
/// the GPU reads: an install's mapped weights or a synthetic model.
public struct Inkling2BufferMatrix: Inkling2ReferenceMatrix, @unchecked Sendable {
    public enum Format: Sendable, Equatable {
        case f32
        case bf16
        case q8_0
        /// NVFP4 blocks; `scale` is the expert's F32 factor the projection
        /// output is multiplied by (folded into each row here).
        case nvfp4(scale: Float)
    }

    public let rows: Int
    public let cols: Int
    public let format: Format
    private let buffer: MTLBuffer
    private let offset: Int
    private let rowBytes: Int

    public init(buffer: MTLBuffer, offset: Int, rows: Int, cols: Int, format: Format) {
        switch format {
        case .f32: rowBytes = cols * 4
        case .bf16: rowBytes = cols * 2
        case .q8_0: rowBytes = Inkling2Q8_0.rowBytes(k: cols)
        case .nvfp4: rowBytes = Inkling2NVFP4.rowBytes(k: cols)
        }
        precondition(buffer.storageMode != .private, "the CPU reference reads the buffer's contents")
        precondition(offset + rows * rowBytes <= buffer.length, "matrix past the end of its buffer")
        self.buffer = buffer
        self.offset = offset
        self.rows = rows
        self.cols = cols
        self.format = format
    }

    public func row(_ index: Int) -> [Float] {
        precondition(index >= 0 && index < rows)
        let p = UnsafeRawBufferPointer(start: buffer.contents() + offset + index * rowBytes, count: rowBytes)
        switch format {
        case .f32:
            return Array(p.bindMemory(to: Float.self))
        case .bf16:
            return (0..<cols).map { i in
                Float(bitPattern: UInt32(p.load(fromByteOffset: 2 * i, as: UInt16.self)) << 16)
            }
        case .q8_0:
            return Inkling2Q8_0.dequantize(p, k: cols)
        case .nvfp4(let scale):
            return Inkling2NVFP4.dequantize(p, k: cols).map { $0 * scale }
        }
    }

    /// `y = W x`. Every format but BF16 (the small sink and tap
    /// projections) runs through vDSP, a tile of rows at a time, with FP32
    /// accumulation.
    public func apply(_ x: [Float]) -> [Float] {
        precondition(x.count == cols)
        switch format {
        case .f32:
            var out = [Float](repeating: 0, count: rows)
            let w = (buffer.contents() + offset).bindMemory(to: Float.self, capacity: rows * cols)
            vDSP_mmul(w, 1, x, 1, &out, 1, vDSP_Length(rows), 1, vDSP_Length(cols))
            return out
        case .q8_0:
            return applyQ8_0(x)
        case .nvfp4(let scale):
            return applyNVFP4(x, scale: scale)
        case .bf16:
            return (0..<rows).map { Inkling2ReferenceOps.dot(row($0), x) }
        }
    }

    /// Rows per tile (a tile is a few `tileRows * rowBytes` floats of
    /// scratch, 627 KB each for a Q8_0 row of 2,304).
    static let tileRows = 64

    /// `values` repeated for every row of a tile.
    private static func tiled(_ values: [Float]) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(tileRows * values.count)
        for _ in 0..<tileRows { out += values }
        return out
    }

    /// Q8_0 `W x`: the tile's bytes converted to floats as they lie (each
    /// block's two scale bytes included), times the input spread to the
    /// same layout (zero under the scale bytes), summed per block, times
    /// the block scales, summed per row.
    private func applyQ8_0(_ x: [Float]) -> [Float] {
        let width = Inkling2Q8_0.blockWeights, blockBytes = Inkling2Q8_0.blockBytes
        let blocks = cols / width
        precondition(offset % 2 == 0, "Q8_0 scales must be 2-byte aligned")
        var spread = [Float](repeating: 0, count: rowBytes)
        for b in 0..<blocks {
            for j in 0..<width { spread[b * blockBytes + 2 + j] = x[b * width + j] }
        }
        let spreads = Self.tiled(spread)
        let ones = [Float](repeating: 1, count: max(blockBytes, blocks))
        let tile = Self.tileRows
        var floats = [Float](repeating: 0, count: tile * rowBytes)
        var sums = [Float](repeating: 0, count: tile * blocks)
        var scales = [Float](repeating: 0, count: tile * blocks)
        var out = [Float](repeating: 0, count: rows)
        let base = buffer.contents() + offset
        floats.withUnsafeMutableBufferPointer { f in
            sums.withUnsafeMutableBufferPointer { s in
                scales.withUnsafeMutableBufferPointer { sc in
                    out.withUnsafeMutableBufferPointer { o in
                        for first in stride(from: 0, to: rows, by: tile) {
                            let n = min(tile, rows - first)
                            let bytes = base + first * rowBytes
                            vDSP_vflt8(bytes.assumingMemoryBound(to: Int8.self), 1, f.baseAddress!, 1,
                                       vDSP_Length(n * rowBytes))
                            vDSP_vmul(f.baseAddress!, 1, spreads, 1, f.baseAddress!, 1, vDSP_Length(n * rowBytes))
                            // [n * blocks][34] x ones: the per-block dot products.
                            vDSP_mmul(f.baseAddress!, 1, ones, 1, s.baseAddress!, 1,
                                      vDSP_Length(n * blocks), 1, vDSP_Length(blockBytes))
                            var source = vImage_Buffer(data: bytes, height: vImagePixelCount(n * blocks),
                                                       width: 1, rowBytes: blockBytes)
                            var target = vImage_Buffer(data: sc.baseAddress!, height: vImagePixelCount(n * blocks),
                                                       width: 1, rowBytes: 4)
                            vImageConvert_Planar16FtoPlanarF(&source, &target, vImage_Flags(kvImageNoFlags))
                            vDSP_vmul(s.baseAddress!, 1, sc.baseAddress!, 1, s.baseAddress!, 1,
                                      vDSP_Length(n * blocks))
                            vDSP_mmul(s.baseAddress!, 1, ones, 1, o.baseAddress! + first, 1,
                                      vDSP_Length(n), 1, vDSP_Length(blocks))
                        }
                    }
                }
            }
        }
        return out
    }

    /// E2M1 values (doubled, as ggml stores them) of a code byte's low and
    /// high nibble, and `ue4m3Half` of every scale byte.
    private static let nvfp4Low: [Float] = (0..<256).map { Float(Inkling2NVFP4.doubledE2M1[$0 & 0xF]) }
    private static let nvfp4High: [Float] = (0..<256).map { Float(Inkling2NVFP4.doubledE2M1[$0 >> 4]) }
    private static let nvfp4Scale: [Float] = (0..<256).map { Inkling2NVFP4.ue4m3Half(UInt8($0)) }
    /// `[36][4]`: sums each sub-block's eight code bytes of a block.
    private static let nvfp4SubBlockSum: [Float] = {
        let bytes = Inkling2NVFP4.blockBytes, subs = Inkling2NVFP4.scaleBytesPerBlock
        var m = [Float](repeating: 0, count: bytes * subs)
        for s in 0..<subs {
            for j in 0..<8 { m[(subs + s * 8 + j) * subs + s] = 1 }
        }
        return m
    }()

    /// NVFP4 `W x` times the expert scale: each byte's two code values
    /// looked up, times the input spread to the byte layout (the low nibble
    /// of byte `j` of sub-block `s` is weight `16 s + j`, the high nibble
    /// `16 s + 8 + j`; zero under the scale bytes), summed per sub-block,
    /// times the sub-block scales, summed per row.
    private func applyNVFP4(_ x: [Float], scale: Float) -> [Float] {
        let width = Inkling2NVFP4.blockWeights, blockBytes = Inkling2NVFP4.blockBytes
        let subs = Inkling2NVFP4.scaleBytesPerBlock, sub = Inkling2NVFP4.subBlockWeights
        let blocks = cols / width
        var low = [Float](repeating: 0, count: rowBytes), high = [Float](repeating: 0, count: rowBytes)
        for b in 0..<blocks {
            for s in 0..<subs {
                for j in 0..<(sub / 2) {
                    low[b * blockBytes + subs + s * (sub / 2) + j] = x[b * width + s * sub + j]
                    high[b * blockBytes + subs + s * (sub / 2) + j] = x[b * width + s * sub + sub / 2 + j]
                }
            }
        }
        let lows = Self.tiled(low), highs = Self.tiled(high)
        let ones = [Float](repeating: 1, count: blocks * subs)
        let tile = Self.tileRows
        var index = [Float](repeating: 0, count: tile * rowBytes)
        var lowValues = [Float](repeating: 0, count: tile * rowBytes)
        var highValues = [Float](repeating: 0, count: tile * rowBytes)
        var sums = [Float](repeating: 0, count: tile * blocks * subs)
        var scales = [Float](repeating: 0, count: tile * blocks * subs)
        var out = [Float](repeating: 0, count: rows)
        let base = buffer.contents() + offset
        index.withUnsafeMutableBufferPointer { ix in
            lowValues.withUnsafeMutableBufferPointer { lv in
                highValues.withUnsafeMutableBufferPointer { hv in
                    sums.withUnsafeMutableBufferPointer { s in
                        scales.withUnsafeMutableBufferPointer { sc in
                            out.withUnsafeMutableBufferPointer { o in
                                for first in stride(from: 0, to: rows, by: tile) {
                                    let n = min(tile, rows - first)
                                    let count = vDSP_Length(n * rowBytes)
                                    let bytes = (base + first * rowBytes).assumingMemoryBound(to: UInt8.self)
                                    vDSP_vfltu8(bytes, 1, ix.baseAddress!, 1, count)
                                    vDSP_vindex(Self.nvfp4Low, ix.baseAddress!, 1, lv.baseAddress!, 1, count)
                                    vDSP_vindex(Self.nvfp4High, ix.baseAddress!, 1, hv.baseAddress!, 1, count)
                                    vDSP_vmma(lv.baseAddress!, 1, lows, 1, hv.baseAddress!, 1, highs, 1,
                                              lv.baseAddress!, 1, count)
                                    // [n * blocks][36] x [36][4]: the per-sub-block dot products.
                                    vDSP_mmul(lv.baseAddress!, 1, Self.nvfp4SubBlockSum, 1, s.baseAddress!, 1,
                                              vDSP_Length(n * blocks), vDSP_Length(subs), vDSP_Length(blockBytes))
                                    for k in 0..<subs {
                                        vDSP_vindex(Self.nvfp4Scale, ix.baseAddress! + k, vDSP_Stride(blockBytes),
                                                    sc.baseAddress! + k, vDSP_Stride(subs), vDSP_Length(n * blocks))
                                    }
                                    vDSP_vmul(s.baseAddress!, 1, sc.baseAddress!, 1, s.baseAddress!, 1,
                                              vDSP_Length(n * blocks * subs))
                                    vDSP_mmul(s.baseAddress!, 1, ones, 1, o.baseAddress! + first, 1,
                                              vDSP_Length(n), 1, vDSP_Length(blocks * subs))
                                }
                            }
                        }
                    }
                }
            }
        }
        var factor = scale
        out.withUnsafeMutableBufferPointer { o in
            vDSP_vsmul(o.baseAddress!, 1, &factor, o.baseAddress!, 1, vDSP_Length(rows))
        }
        return out
    }

    /// F32 vector of `count` values at `offset`.
    public static func floats(_ buffer: MTLBuffer, offset: Int, count: Int) -> [Float] {
        let p = (buffer.contents() + offset).bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }
}
