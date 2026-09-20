import Foundation
import Metal
import Testing
@testable import Mference

extension GemmaQATKernelTests {
    @Test(arguments: [false, true]) func nativeBF16RouterLogitsTopEightAndGains(sourceFP16: Bool) throws {
        let context = try MetalContext()
        let d = 2816, experts = 128, rows = 3
        func bf16(_ values: [Float]) -> [UInt16] {
            values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        }
        let weights = (0..<experts * d).map { i -> Float in
            Float(((i / d) * 7 + (i % d) * 3) % 31 - 15) / 128
        }
        let effective = (0..<d).map { col in
            sourceFP16 ? Float(Float16((1 + Float(col % 3) / 3) / Float(d).squareRoot()))
                : 1 + Float(col % 3) / 4
        }
        let gains = (0..<experts).map { 0.5 + Float($0 % 4) / 2 }
        let inputs = (0..<rows).map { row in
            (0..<d).map { col in Float16(Float((col * 3 + row * 5) % 17 - 8) / 32) }
        }
        let expected: [[Float]] = inputs.map { input in
            (0..<experts).map { expert in
                let sum = (0..<d).reduce(0.0) { sum, col in
                    let x = Double(input[col]) * Double(effective[col])
                    return sum + Double(weights[expert * d + col]) * (sourceFP16 ? Double(Float16(x)) : x)
                }
                return sourceFP16 ? Float(Float16(sum)) : Float(sum)
            }
        }
        let expectedIDs: [[UInt32]] = expected.map { logits in
            Array((0..<experts).sorted {
                logits[$0] == logits[$1]
                    ? (sourceFP16 ? $0 > $1 : $0 < $1) : logits[$0] > logits[$1]
            }.prefix(8)).map(UInt32.init)
        }
        let expectedWeights: [[Float]] = zip(expected, expectedIDs).map { logits, ids in
            let exps = ids.map { exp(Double(logits[Int($0)] - logits[Int(ids[0])])) }
            let total = exps.reduce(0, +)
            return zip(ids, exps).map {
                let probability = sourceFP16 ? Double(Float16($1 / total)) : $1 / total
                return Float(Float16(probability * Double(gains[Int($0)])))
            }
        }
        let w = try Self.buffer(bf16(weights), context: context, offset: 2)
        let scaleBits = sourceFP16 ? effective.map { Float16($0).bitPattern } : bf16(effective)
        let scale = try Self.buffer(scaleBits, context: context, offset: 2)
        let gain = try Self.buffer(bf16(gains), context: context, offset: 6)
        // Poisoned quantized companions are never read by a BF16 router.
        // Sized for the old INT8 path so its red run is a numerical failure,
        // not an out-of-bounds memory access.
        let unused = try Self.buffer([UInt16](repeating: 0x7FC0, count: experts * d / 64), context: context)
        let indices = try Self.buffer([UInt32](repeating: .max, count: 8), context: context)
        let routeWeights = try Self.buffer([Float16](repeating: .nan, count: 8), context: context)
        let logits = try Self.buffer([Float](repeating: .nan, count: experts), context: context)
        for specialized in [false, true] {
            let router = try MoE(context: context, specializedD: UInt32(specialized ? d : 128),
                groupSize: 32, routerBF16: true, sourceFP16: sourceFP16)
            var constants = [MetalFunctionConstant(index: 109, value: .bool(true))]
                + Quantization.gemmaSourceConstants(enabled: sourceFP16)
            if specialized {
                constants += [MetalFunctionConstant(index: 40, value: .uint32(UInt32(experts))),
                    MetalFunctionConstant(index: 41, value: .uint32(UInt32(d))),
                    MetalFunctionConstant(index: 43, value: .bool(true))]
            }
            let projection = try context.pipeline("router_gemv_gemma4_r4", constants: constants,
                maxTotalThreadsPerThreadgroup: nil,
                safeMathModule: sourceFP16 ? "moe" : nil)
            for row in 0..<rows {
                let hidden = try Self.buffer(inputs[row], context: context)
                let cb = try #require(context.queue.makeCommandBuffer())
                router.encodeRouterGemma4(commandBuffer: cb, weights: w, weightsOffset: 2,
                    scales: unused, biases: unused, hidden: hidden,
                    effectiveScale: scale, effectiveScaleOffset: 2,
                    perExpertScale: gain, perExpertScaleOffset: 6,
                    outIndices: indices, outWeights: routeWeights,
                    numExperts: UInt32(experts), d: UInt32(d), topK: 8)
                let encoder = try #require(cb.makeComputeCommandEncoder())
                encoder.setComputePipelineState(projection)
                for (index, buffer, offset) in [(0, w, 2), (1, unused, 0), (2, unused, 0),
                                               (3, hidden, 0), (4, scale, 2), (5, logits, 0)] {
                    encoder.setBuffer(buffer, offset: offset, index: index)
                }
                var count = UInt32(experts), width = UInt32(d)
                encoder.setBytes(&count, length: 4, index: 6)
                encoder.setBytes(&width, length: 4, index: 7)
                encoder.dispatchThreadgroups(MTLSize(width: experts / 4, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                encoder.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed, "\(String(describing: cb.error))")
                let actualLogits = logits.contents().assumingMemoryBound(to: Float.self)
                #expect(Array(UnsafeBufferPointer(start: actualLogits, count: experts)) == expected[row],
                        "BF16 router logits row=\(row), specialized=\(specialized)")
                let actualIDs = indices.contents().assumingMemoryBound(to: UInt32.self)
                #expect(Array(UnsafeBufferPointer(start: actualIDs, count: 8)) == expectedIDs[row])
                let actualWeights = routeWeights.contents().assumingMemoryBound(to: Float16.self)
                for rank in 0..<8 {
                    #expect(abs(Float(actualWeights[rank]) - expectedWeights[row][rank]) < 0.0005)
                }
            }
        }
        let batched = try PrefillRouter(context: context, routerBF16: true, sourceFP16: sourceFP16)
        let hidden = try Self.buffer(inputs.flatMap { $0 + [Float16](repeating: .nan, count: 8) },
                                     context: context, offset: 8)
        let batchIDs = try Self.buffer([UInt32](repeating: .max, count: rows * 8), context: context, offset: 4)
        let batchWeights = try Self.buffer([Float16](repeating: .nan, count: rows * 8), context: context, offset: 6)
        let cb = try #require(context.queue.makeCommandBuffer())
        batched.encodeGemma4Block(commandBuffer: cb, weights: w, weightsOffset: 2,
            scales: unused, biases: unused, hidden: hidden, hiddenOffset: 8,
            effectiveScale: scale, effectiveScaleOffset: 2, perExpertScale: gain, perExpertScaleOffset: 6,
            outIndices: batchIDs, outIndicesOffset: 4, outWeights: batchWeights, outWeightsOffset: 6,
            queryCount: UInt32(rows), numExperts: UInt32(experts), d: UInt32(d), topK: 8,
            hiddenStrideElements: UInt32(d + 8))
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.status == .completed, "\(String(describing: cb.error))")
        let actualIDs = batchIDs.contents().advanced(by: 4).assumingMemoryBound(to: UInt32.self)
        #expect(Array(UnsafeBufferPointer(start: actualIDs, count: rows * 8)) == expectedIDs.flatMap { $0 })
        let actualWeights = batchWeights.contents().advanced(by: 6).assumingMemoryBound(to: Float16.self)
        for row in 0..<rows {
            for rank in 0..<8 {
                // Fixed absolute allowance for FP16 softmax/gain rounding;
                // logits and selected IDs above require exact equality.
                #expect(abs(Float(actualWeights[row * 8 + rank]) - expectedWeights[row][rank]) < 0.0005)
            }
        }
    }
}
