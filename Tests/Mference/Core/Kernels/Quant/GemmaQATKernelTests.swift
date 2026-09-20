import Foundation
import Metal
import Testing
@testable import Mference

/// Independent nibble/BF16 fixtures, without the group-64 fixture quantizer.
/// Dyadic inputs make these small reductions exactly representable in FP32;
/// the expected FP16 rounding therefore requires exact equality, not a fitted
/// tolerance. Adjacent groups have different scales AND biases.
@Suite struct GemmaQATKernelTests {
    struct Matrix {
        let rows: Int
        let columns: Int
        let groupSize: Int
        let packed: [UInt8]
        let scales: [UInt16]
        let biases: [UInt16]

        init(rows: Int, columns: Int, groupSize: Int) {
            self.rows = rows
            self.columns = columns
            self.groupSize = groupSize
            packed = (0..<(rows * columns / 2)).map { i in
                UInt8((i * 7 + 3) % 16 | (((i * 11 + 5) % 16) << 4))
            }
            scales = (0..<(rows * columns / groupSize)).map { i in
                UInt16(Float((i % 7) + 1).bitPattern >> 16) - 6 * 128
            }
            biases = (0..<(rows * columns / groupSize)).map { i in
                UInt16(truncatingIfNeeded: (Float((i % 11) - 5) / 128).bitPattern >> 16)
            }
        }

        func value(row: Int, column: Int) -> Double {
            let byte = packed[(row * columns + column) / 2]
            let q = (column % 2 == 0) ? byte & 15 : byte >> 4
            let g = row * (columns / groupSize) + column / groupSize
            let s = Float(bitPattern: UInt32(scales[g]) << 16)
            let b = Float(bitPattern: UInt32(biases[g]) << 16)
            return Double(q) * Double(s) + Double(b)
        }
    }

    static func buffer<T>(_ values: [T], context: MetalContext, offset: Int = 0) throws -> MTLBuffer {
        let bytes = values.count * MemoryLayout<T>.stride
        let result = try #require(context.device.makeBuffer(length: bytes + offset,
                                                            options: .storageModeShared))
        memset(result.contents(), 0xA5, bytes + offset)
        values.withUnsafeBytes { raw in
            result.contents().advanced(by: offset).copyMemory(from: raw.baseAddress!, byteCount: bytes)
        }
        return result
    }

    @Test func nativeAffineEmbeddingAndProjection() throws {
        let context = try MetalContext()
        for groupSize in [32, 64] {
            let embed = try EmbedLookupInt4(context: context, groupSize: groupSize)
            let gemv = try DequantInt4GEMV(context: context, groupSize: groupSize)
            for n in [32, 96, 704, 2112, 2816] where n % groupSize == 0 {
                let matrix = Matrix(rows: 11, columns: n, groupSize: groupSize)
                let weights = try Self.buffer(matrix.packed, context: context, offset: 2)
                let scales = try Self.buffer(matrix.scales, context: context, offset: 6)
                let biases = try Self.buffer(matrix.biases, context: context, offset: 10)
                let input = (0..<n).map { Float16(Float(($0 * 3) % 17 - 8) / 32) }
                let x = try Self.buffer(input, context: context, offset: 8)
                let row = try Self.buffer([Float16](repeating: .nan, count: n), context: context, offset: 4)
                let y = try Self.buffer([Float16](repeating: .nan, count: 11), context: context, offset: 4)
                let yf = try Self.buffer([Float](repeating: .nan, count: 11), context: context, offset: 4)
                let cb = try #require(context.queue.makeCommandBuffer())
                embed.encode(commandBuffer: cb,
                    table: weights, tableOffset: 2, scales: scales, scalesOffset: 6,
                    biases: biases, biasesOffset: 10, out: row, outOffset: 4,
                    tokenId: 9, d: UInt32(n), outScale: 2)
                for (out, fp32) in [(y, false), (yf, true)] {
                    gemv.encode(commandBuffer: cb,
                        weights: weights, weightsOffset: 2, scales: scales, scalesOffset: 6,
                        biases: biases, biasesOffset: 10, x: x, xOffset: 8,
                        y: out, yOffset: 4, m: 11, n: UInt32(n), outputFloat32: fp32)
                }
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed, "\(String(describing: cb.error))")
                let actualRow = row.contents().advanced(by: 4).assumingMemoryBound(to: Float16.self)
                let expectedRow = (0..<n).map { Float16(matrix.value(row: 9, column: $0) * 2) }
                #expect(Array(UnsafeBufferPointer(start: actualRow, count: n)) == expectedRow,
                        "embedding group=\(groupSize), N=\(n)")
                let expected = (0..<11).map { r in
                    (0..<n).reduce(0.0) { $0 + matrix.value(row: r, column: $1) * Double(input[$1]) }
                }
                let actual = y.contents().advanced(by: 4).assumingMemoryBound(to: Float16.self)
                let actualF = yf.contents().advanced(by: 4).assumingMemoryBound(to: Float.self)
                #expect(Array(UnsafeBufferPointer(start: actual, count: 11)) == expected.map(Float16.init),
                        "FP16 projection group=\(groupSize), N=\(n)")
                #expect(Array(UnsafeBufferPointer(start: actualF, count: 11)) == expected.map(Float.init),
                        "FP32 projection group=\(groupSize), N=\(n)")
            }
        }
    }
}
