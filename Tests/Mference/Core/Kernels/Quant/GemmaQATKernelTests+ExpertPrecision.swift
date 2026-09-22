import Foundation
import Metal
import Testing
@testable import Mference

extension GemmaQATKernelTests {
    static func sourceProjection(_ matrix: Matrix, row: Int, input: [Float16]) -> Double {
        var result = (0..<matrix.columns).reduce(0.0) {
            $0 + matrix.value(row: row, column: $1) * Double(input[$1])
        }
        for k in stride(from: 0, to: matrix.columns, by: 4) {
            let partial = ((input[k] + input[k + 1]) + input[k + 2]) + input[k + 3]
            let exact = (0..<4).reduce(0.0) { $0 + Double(input[k + $1]) }
            let group = row * matrix.columns / matrix.groupSize + k / matrix.groupSize
            let bias = Float(bitPattern: UInt32(matrix.biases[group]) << 16)
            result += Double(bias) * (Double(partial) - exact)
        }
        return result
    }

    // Same FP16 operation boundaries as the pinned MLX source graph. Verified
    // independently on 11,264 installed-expert activations before these tests.
    static func sourceGeGLU(_ gate: Float16, _ up: Float16) -> Float16 {
        let cube = Float16(pow(Double(gate), 3))
        let inner = Float16(sqrt(2 / Double.pi)) * (gate + Float16(0.044715) * cube)
        let activation = (Float16(0.5) * gate) * (Float16(1) + Float16(tanh(Double(inner))))
        return activation * up
    }

    @Test func sourceFP16GELUBoundaries() throws {
        let context = try MetalContext()
        let gate: [Float16] = [-6, -3.7, -2.5, -1.75, -1.2, -0.5, 0, 0.125, 0.8, 1.7, 3.7, 5]
        let up = gate.indices.map { Float16(1 + Double($0) / 32) }
        let expected = zip(gate, up).map(Self.sourceGeGLU)
        let g = try Self.buffer(gate, context: context)
        let u = try Self.buffer(up, context: context)
        let out = try Self.buffer([Float16](repeating: .nan, count: gate.count), context: context)
        let pipeline = try context.pipeline("gelu_mul_fp16",
            constants: Quantization.gemmaSourceConstants(enabled: true))
        let cb = try #require(context.queue.makeCommandBuffer())
        let enc = try #require(cb.makeComputeCommandEncoder())
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(g, offset: 0, index: 0)
        enc.setBuffer(u, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var count = UInt32(gate.count)
        enc.setBytes(&count, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: gate.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.status == .completed)
        let actual = Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float16.self), count: gate.count))
        #expect(actual == expected)
    }

    @Test func sourceFP16AffineBiasPartialSums() throws {
        let context = try MetalContext()
        for k in [32, 96, 704, 2112, 2816] {
            let n = 11, rows = 3
            // With zero nibbles, only affine bias contributes. This isolates
            // MLX's FP16 sum-of-four boundary from projection reduction order.
            let packed = try Self.buffer([UInt8](repeating: 0, count: n * k / 2), context: context, offset: 2)
            let scales = try Self.buffer([UInt16](repeating: 0x3C80, count: n * k / 32), context: context)
            let biasValues = (0..<n * k / 32).map { Float(($0 % 7) + 1) / 128 }
            let biases = try Self.buffer(biasValues.map(Quantization.bf16Bits), context: context)
            let quad: [Float16] = [1, 1 / 2048, -1, 1 / 4096]
            let input = (0..<k).map { quad[$0 % 4] }
            let x = try Self.buffer(Array(repeating: input, count: rows).flatMap { $0 }, context: context)
            let quadSum = ((quad[0] + quad[1]) + quad[2]) + quad[3]
            #expect(Double(quadSum) != quad.reduce(0.0) { $0 + Double($1) })
            let groups = k / 32
            func expectedAffine(row: Int) -> Float16 {
                var total = 0.0
                for group in 0..<groups {
                    total += Double(biasValues[row * groups + group]) * Double(quadSum) * 8
                }
                return Float16(total)
            }
            let expected: [Float16] = (0..<n).map { expectedAffine(row: $0) }
            let out = try Self.buffer([Float16](repeating: .nan, count: n * rows), context: context)
            let gemv = try DequantInt4GEMV(context: context, additionalShapes: [(n, k)],
                groupSize: 32, sourceFP16: true)
            let qmm = try PrefillInt4QMM(context: context, groupSize: 32, sourceFP16: true)
            let tensor = MPPPrefillInt4QMM(context: context, groupSize: 32, sourceFP16: true)
            for path in 0..<3 where path != 2 || (k % 64 == 0 && tensor.isAvailable) {
                let cb = try #require(context.queue.makeCommandBuffer())
                if path == 0 {
                    gemv.encode(commandBuffer: cb, weights: packed, weightsOffset: 2,
                        scales: scales, biases: biases, x: x, y: out, m: UInt32(n), n: UInt32(k))
                } else if path == 1 {
                    qmm.encode(commandBuffer: cb, weights: packed, weightsOffset: 2,
                        scales: scales, biases: biases, x: x, y: out, t: rows, n: n, k: k)
                } else {
                    #expect(tensor.encode(commandBuffer: cb, weights: packed, weightsOffset: 2,
                        scales: scales, biases: biases, x: x, y: out, m: rows, n: n, k: k) == .sourceAffineF16)
                }
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed)
                let count = path == 0 ? n : n * rows
                let actual = Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float16.self), count: count))
                let expectedRows = Array(repeating: expected, count: path == 0 ? 1 : rows).flatMap { $0 }
                #expect(actual == expectedRows, "source affine K=\(k), path=\(path)")
            }
        }
    }

    @Test func sourceFP16GELUSaturationStaysFinite() throws {
        // BOS, layer 0, routed expert 79, activation 557: the unchanged MLX
        // source returns 125.625 from these exact traced gate/up values.
        let gates: [Float16] = [10.09375, -10.09375, 40, -40, 41, -41, 100, -100]
        let ups: [Float16] = [12.4453125, 12.4453125, 1, 1, 1, 1, 1, 1]
        let expected = zip(gates, ups).map(Self.sourceGeGLU)
        #expect(expected[0] == 125.625)
        let expectedFinite = expected.allSatisfy { $0.isFinite }
        try #require(expectedFinite)
        let context = try MetalContext()
        let g = try Self.buffer(gates, context: context)
        let u = try Self.buffer(ups, context: context)
        let out = try Self.buffer([Float16](repeating: .nan, count: gates.count), context: context)
        let pipeline = try context.pipeline("gelu_mul_fp16",
            constants: Quantization.gemmaSourceConstants(enabled: true))
        let cb = try #require(context.queue.makeCommandBuffer())
        let enc = try #require(cb.makeComputeCommandEncoder())
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(g, offset: 0, index: 0)
        enc.setBuffer(u, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var count = UInt32(gates.count)
        enc.setBytes(&count, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: gates.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.status == .completed)
        let actual = Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float16.self), count: gates.count))
        #expect(actual == expected)
    }
}
