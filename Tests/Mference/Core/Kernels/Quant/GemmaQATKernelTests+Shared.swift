import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

extension GemmaQATKernelTests {
    static func projection(_ matrix: Matrix, context: MetalContext,
                           multiplier: Float = 1) throws -> SharedExpertProjection {
        func scaled(_ bits: [UInt16]) -> [UInt16] {
            bits.map {
                UInt16(truncatingIfNeeded:
                    (Float(bitPattern: UInt32($0) << 16) * multiplier).bitPattern >> 16)
            }
        }
        return SharedExpertProjection(
            weights: try buffer(matrix.packed, context: context, offset: 2),
            scales: try buffer(scaled(matrix.scales), context: context, offset: 6),
            biases: try buffer(scaled(matrix.biases), context: context, offset: 10),
            weightsOffset: 2, scalesOffset: 6, biasesOffset: 10,
            rows: UInt32(matrix.rows), cols: UInt32(matrix.columns))
    }

    @Test(arguments: [false, true]) func nativeAffineSharedFFNUsesEveryProjection(sourceFP16: Bool) throws {
        let context = try MetalContext()
        let d = 2816, f = 2112
        let gate = Matrix(rows: f, columns: d, groupSize: 32)
        let down = Matrix(rows: d, columns: f, groupSize: 32)
        let input = (0..<d).map { Float16(Float(($0 * 3) % 17 - 8) / 32) }
        let activations: [Double] = (0..<f).map { r in
            let dot = (0..<d).reduce(0.0) { $0 + gate.value(row: r, column: $1) * Double(input[$1]) }
            if sourceFP16 {
                let sourceDot = Self.sourceProjection(gate, row: r, input: input)
                return Double(Self.sourceGeGLU(Float16(sourceDot), Float16(sourceDot * 0.5)))
            }
            let g = Double(Float16(dot)), u = Double(Float16(dot * 0.5))
            let gelu = 0.5 * g * (1 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))
            return Double(Float16(gelu * u))
        }
        let expected: [Float] = (0..<d).map { r in
            let value = sourceFP16
                ? Self.sourceProjection(down, row: r, input: activations.map(Float16.init))
                : (0..<f).reduce(0.0) { $0 + down.value(row: r, column: $1) * activations[$1] }
            return Float(Float16(value))
        }
        let gateProjection = try Self.projection(gate, context: context)
        let upProjection = try Self.projection(gate, context: context, multiplier: 0.5)
        let downProjection = try Self.projection(down, context: context)
        let x = try Self.buffer(Array(repeating: input, count: 33).flatMap { $0 }, context: context, offset: 8)
        let output = try Self.buffer([Float16](repeating: .nan, count: 33 * d), context: context, offset: 4)
        let scratch = try (0..<3).map { _ in
            try Self.buffer([Float16](repeating: .nan, count: 33 * f), context: context, offset: 8)
        }
        func check(_ count: Int, mode: String, activationRows: Int? = nil) {
            if sourceFP16 {
                let act = scratch[2].contents().advanced(by: 8).assumingMemoryBound(to: Float16.self)
                let storedRows = activationRows ?? count
                let actualActs = Array(UnsafeBufferPointer(start: act, count: storedRows * f))
                let expectedActs = Array(repeating: activations.map(Float16.init), count: storedRows).flatMap { $0 }
                #expect(actualActs == expectedActs, "source shared activation \(mode)")
            }
            let actual = output.contents().advanced(by: 4).assumingMemoryBound(to: Float16.self)
            for row in 0..<count {
                let values = (0..<d).map { Float(actual[row * d + $0]) }
                // GELU and FP16 boundaries can differ by rounding across CPU
                // and Metal; 0.2% L2 allows that without hiding group errors.
                #expect(RelError.compute(actual: values, reference: expected) < 0.002,
                        "shared FFN \(mode), row=\(row)")
            }
        }
        for fused in [false, true] {
            let kernel = try SharedExpertInt4(context: context, useFusedGateUp: fused,
                specializedD: d, specializedF: f, groupSize: 32, sourceFP16: sourceFP16)
            let cb = try #require(context.queue.makeCommandBuffer())
            try kernel.encode(commandBuffer: cb, x: x, xOffset: 8,
                gate: gateProjection, up: upProjection, down: downProjection,
                y: output, yOffset: 4,
                scratchGate: scratch[0], scratchGateOffset: 8,
                scratchUp: scratch[1], scratchUpOffset: 8,
                scratchAct: scratch[2], scratchActOffset: 8)
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed, "\(String(describing: cb.error))")
            check(1, mode: "scalar fused=\(fused)")
        }
        let kernel = try PrefillSharedExpert(context: context, weightBits: 4, groupSize: 32, sourceFP16: sourceFP16)
        let mppAvailable = MPPPrefillInt4QMM(context: context, groupSize: 32, sourceFP16: sourceFP16).isAvailable
        for count in [3, 33] {
            let cb = try #require(context.queue.makeCommandBuffer())
            let path = try kernel.encodeBlock(commandBuffer: cb, x: x, xOffset: 8,
                y: output, yOffset: 4, gate: gateProjection, up: upProjection, down: downProjection,
                scratchGate: scratch[0], scratchGateOffset: 8,
                scratchUp: scratch[1], scratchUpOffset: 8,
                scratchAct: scratch[2], scratchActOffset: 8,
                queryCount: count, d: d, intermediate: f, xStrideElements: d, yStrideElements: d)
            #expect(path == (count >= 32 && mppAvailable
                ? (sourceFP16 ? .sourceAffineInt4 : .tensorOpsInt4) : .repeatedRows))
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed, "\(String(describing: cb.error))")
            // Repeated rows intentionally reuse one scratch row; all output
            // rows are still checked against the independent reference.
            check(count, mode: "prefill \(path)", activationRows: path == .repeatedRows ? 1 : count)
        }
    }
}
