import Foundation
import Metal
import Testing
@testable import Mference

extension GemmaQATKernelTests {
    @Test func nativeAffineTensorTilesKeepSeparateScaleGroups() throws {
        let context = try MetalContext()
        let baseline = MPPPrefillInt4QMM(context: context)
        let kernel = MPPPrefillInt4QMM(context: context, groupSize: 32)
        // Group-32 specialization must not silently remove an otherwise
        // available production path. Unsupported systems retain the same
        // explicit capability result, which is not numerical MPP proof.
        #expect(kernel.isAvailable == baseline.isAvailable)
        #expect(kernel.isFloat32Available == baseline.isFloat32Available)
        guard baseline.isAvailable, baseline.isFloat32Available else {
            print("QAT MPP: hardware/compiler unavailable; numerical coverage unverified")
            return
        }
        print("QAT MPP: executing FP16 and FP32 numerical comparison")
        for n in [704, 2112, 2816] {
            let matrix = Matrix(rows: 35, columns: n, groupSize: 32)
            let weights = try Self.buffer(matrix.packed, context: context, offset: 2)
            let scales = try Self.buffer(matrix.scales, context: context, offset: 6)
            let biases = try Self.buffer(matrix.biases, context: context, offset: 10)
            let inputs = (0..<65 * n).map { Float16(Float(($0 * 3) % 17 - 8) / 32) }
            let x = try Self.buffer(inputs, context: context, offset: 8)
            let y = try Self.buffer([Float16](repeating: .nan, count: 65 * 35), context: context, offset: 4)
            let yf = try Self.buffer([Float](repeating: .nan, count: 65 * 35), context: context, offset: 4)
            let cb = try #require(context.queue.makeCommandBuffer())
            #expect(kernel.encode(commandBuffer: cb, weights: weights, weightsOffset: 2,
                scales: scales, scalesOffset: 6, biases: biases, biasesOffset: 10,
                x: x, xOffset: 8, y: y, yOffset: 4, m: 65, n: 35, k: n) == .affineThreadgroupF16)
            #expect(kernel.encodeFloat32(commandBuffer: cb, weights: weights, weightsOffset: 2,
                scales: scales, scalesOffset: 6, biases: biases, biasesOffset: 10,
                x: x, xOffset: 8, y: yf, yOffset: 4, m: 65, n: 35, k: n) == .affineThreadgroupF32)
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed, "\(String(describing: cb.error))")
            let expected = (0..<65).flatMap { t in
                (0..<35).map { r in
                    (0..<n).reduce(0.0) {
                        $0 + matrix.value(row: r, column: $1) * Double(inputs[t * n + $1])
                    }
                }
            }
            let actual = y.contents().advanced(by: 4).assumingMemoryBound(to: Float16.self)
            let actualF = yf.contents().advanced(by: 4).assumingMemoryBound(to: Float.self)
            #expect(Array(UnsafeBufferPointer(start: actual, count: 65 * 35)) == expected.map(Float16.init),
                    "MPP FP16 group-32 N=\(n)")
            #expect(Array(UnsafeBufferPointer(start: actualF, count: 65 * 35)) == expected.map(Float.init),
                    "MPP FP32 group-32 N=\(n)")
        }
    }
}
