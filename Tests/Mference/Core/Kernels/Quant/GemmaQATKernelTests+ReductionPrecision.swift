import Foundation
import Metal
import Testing
@testable import Mference

extension GemmaQATKernelTests {
    @Test(arguments: [false, true]) func sourceFP16RouterTieSelection(sourceFP16: Bool) throws {
        // MLX 0.32.2's stable ascending argpartition/sort keeps the last eight
        // IDs at an equal-score cutoff. Original Mference keeps the first eight.
        let context = try MetalContext()
        let logits = try Self.buffer([Float](repeating: 0, count: 128), context: context)
        let gains = try Self.buffer([UInt16](repeating: Quantization.bf16Bits(1), count: 128), context: context)
        let indices = try Self.buffer([UInt32](repeating: .max, count: 8), context: context)
        let weights = try Self.buffer([Float16](repeating: .nan, count: 8), context: context)
        let expected = sourceFP16 ? (0..<8).map { UInt32(127 - $0) } : (0..<8).map(UInt32.init)
        func check() {
            #expect(Array(UnsafeBufferPointer(start: indices.contents().assumingMemoryBound(to: UInt32.self), count: 8)) == expected)
            #expect(Array(UnsafeBufferPointer(start: weights.contents().assumingMemoryBound(to: Float16.self), count: 8)) == [Float16](repeating: 0.125, count: 8))
        }
        for name in ["router_topk_select_k8", "router_topk_select_k8_par"] {
            let pso = try context.pipeline(name, constants: Quantization.gemmaSourceConstants(enabled: sourceFP16))
            let cb = try #require(context.queue.makeCommandBuffer())
            let enc = try #require(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(pso)
            for (i, buffer) in [logits, gains, indices, weights].enumerated() {
                enc.setBuffer(buffer, offset: 0, index: i)
            }
            var count = UInt32(128)
            enc.setBytes(&count, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed)
            check()
        }
        let matrix = try Self.buffer([UInt16](repeating: 0, count: 128 * 64), context: context)
        let x = try Self.buffer([Float16](repeating: 1, count: 64), context: context)
        let scale = try Self.buffer([UInt16](repeating: sourceFP16 ? Float16(1).bitPattern : Quantization.bf16Bits(1), count: 64), context: context)
        let router = try PrefillRouter(context: context, routerBF16: true, sourceFP16: sourceFP16)
        let cb = try #require(context.queue.makeCommandBuffer())
        router.encodeGemma4Block(commandBuffer: cb, weights: matrix, scales: matrix, biases: matrix,
            hidden: x, effectiveScale: scale, perExpertScale: gains, outIndices: indices,
            outWeights: weights, queryCount: 1, numExperts: 128, d: 64, topK: 8,
            hiddenStrideElements: 64)
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.status == .completed)
        check()
    }

    @Test func sourceFP16RoutedReductionBoundaries() throws {
        let context = try MetalContext()
        let d = 32, f = 704, groups = f / 32
        var bytes = [UInt8](repeating: 0, count: d * f / 2)
        func append(_ values: [UInt16]) -> UInt32 {
            let offset = UInt32(bytes.count)
            values.withUnsafeBytes { bytes.append(contentsOf: $0) }
            return offset
        }
        let scales = append([UInt16](repeating: 0, count: d * groups))
        let factors = (0..<d).map { Float(1) + Float($0) / 32 }
        let biases = append((0..<d * groups).map {
            Quantization.bf16Bits($0 % groups < 2 ? factors[$0 / groups] : 0)
        })
        let offsets = MoEExpertOffsets(gateWOff: 0, gateSOff: 0, gateBOff: 0,
            upWOff: 0, upSOff: 0, upBOff: 0,
            downWOff: 0, downSOff: scales, downBOff: biases)
        let blob = try Self.buffer(bytes, context: context)
        let blobs = Array(repeating: (buffer: blob, offset: 0), count: 8)
        var values = [Float16](repeating: 0, count: 8 * f)
        for rank in 0..<8 {
            values[rank * f] = 1
            values[rank * f + 32] = Float16(3 * (rank + 1)) / 4096
        }
        let activations = try Self.buffer(values, context: context)
        let partials = (0..<8).flatMap { rank in
            factors.map { Float16(Double($0) * (1 + Double(3 * (rank + 1)) / 4096)) }
        }
        let partialBuffer = try Self.buffer(partials, context: context)
        let residual = try Self.buffer([Float16](repeating: 0, count: d), context: context)
        let kernel = try MoE(context: context, specializedD: UInt32(d), specializedF: UInt32(f),
                             groupSize: 32, sourceFP16: true)
        let arguments = try #require(kernel.makeRoutedArgumentBuffer(routedBlobs: blobs, topK: 8))
        let prefill = try PrefillMoE(context: context, sourceFP16: true)
        let cases: [[Float16]] = [
            [0.462890625, 0, 0, 0, 0, 0, 0, 0],
            [0.462890625, 0.257080078125, 0.0535888671875, 0.05419921875,
             0.046661376953125, 0.04547119140625, 0.0396728515625, 0.03839111328125],
        ]
        for weights in cases {
            let expected = (0..<d).map { col in
                // Independently rechecked with pinned MLX's column reduction:
                // source rank order is ascending and every addition is FP16.
                var sum = Float16(0)
                for rank in (0..<8).reversed() { sum += partials[rank * d + col] * weights[rank] }
                return sum
            }
            let routes = try Self.buffer(weights, context: context)
            for batched in [false, true] {
                let output = try Self.buffer([Float16](repeating: .nan, count: d), context: context)
                let cb = try #require(context.queue.makeCommandBuffer())
                if batched {
                    prefill.encodeReduceTokenMajor(commandBuffer: cb, routePartials: partialBuffer,
                        routeWeights: routes, h2: output, queryCount: 1, topK: 8, d: UInt32(d))
                } else {
                    kernel.encodeRoutedPersistentPhase2Reduce(commandBuffer: cb,
                        routedArgBuffer: arguments, routedBlobs: blobs, routedOffsets: offsets,
                        acts: activations, routingWeights: routes, residual: residual,
                        y: output, d: UInt32(d), f: UInt32(f), topK: 8)
                }
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed)
                let actual = Array(UnsafeBufferPointer(
                    start: output.contents().assumingMemoryBound(to: Float16.self), count: d))
                #expect(actual == expected, "source half projection/product boundaries, prefill=\(batched)")
            }
        }
    }

    @Test func sourceFP16RouterPreservesSourceHalfSoftmax() throws {
        // BOS layer-0 native scores. Pinned MLX 0.32.2 argpartition returns
        // ascending scores; its default softmax uses FP16 exp/sum/reciprocal.
        // Frozen from that independent operation, mapped to descending ranks.
        let scores: [Float] = [4.3359375, 3.720703125, 2.17578125, 2.1640625,
                              2.021484375, 2.015625, 1.87109375, 1.8505859375]
        let gains: [Float] = [0.98828125, 1.015625, 0.9921875, 1.015625,
                             1.0078125, 0.98828125, 0.99609375, 0.984375]
        let expected: [Float16] = [0.463134765625, 0.257080078125, 0.0535888671875,
            0.05419921875, 0.046661376953125, 0.045501708984375,
            0.0396728515625, 0.0384521484375]
        let context = try MetalContext()
        let allScores = scores + [Float](repeating: -10, count: 120)
        let logits = try Self.buffer(allScores, context: context)
        let gain = try Self.buffer((gains + [Float](repeating: 1, count: 120)).map(Quantization.bf16Bits), context: context)
        let ids = try Self.buffer([UInt32](repeating: .max, count: 8), context: context)
        let weights = try Self.buffer([Float16](repeating: .nan, count: 8), context: context)
        for name in ["router_topk_select_k8", "router_topk_select_k8_par"] {
            let pipeline = try context.pipeline(name, constants: Quantization.gemmaSourceConstants(enabled: true))
            let cb = try #require(context.queue.makeCommandBuffer())
            let enc = try #require(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(pipeline)
            for (i, buffer) in [logits, gain, ids, weights].enumerated() {
                enc.setBuffer(buffer, offset: 0, index: i)
            }
            var count = UInt32(128)
            enc.setBytes(&count, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed)
            let actual = Array(UnsafeBufferPointer(start: weights.contents().assumingMemoryBound(to: Float16.self), count: 8))
            #expect(actual == expected, "source softmax \(name)")
            #expect(Array(UnsafeBufferPointer(start: ids.contents().assumingMemoryBound(to: UInt32.self), count: 8)) == (0..<8).map(UInt32.init))
        }
        // Exercise the separate batched router through its projection entry.
        let d = 64
        var matrix = [UInt16](repeating: 0, count: 128 * d)
        for e in 0..<128 {
            let high = Float(bitPattern: allScores[e].bitPattern & 0xFFFF0000)
            matrix[e * d] = Quantization.bf16Bits(high)
            matrix[e * d + 1] = Quantization.bf16Bits(allScores[e] - high)
        }
        let w = try Self.buffer(matrix, context: context)
        let x = try Self.buffer([Float16(1), 1] + [Float16](repeating: 0, count: d - 2), context: context)
        let scale = try Self.buffer([Float16](repeating: 1, count: d), context: context)
        let router = try PrefillRouter(context: context, routerBF16: true, sourceFP16: true)
        let cb = try #require(context.queue.makeCommandBuffer())
        router.encodeGemma4Block(commandBuffer: cb, weights: w, scales: w, biases: w,
            hidden: x, effectiveScale: scale, perExpertScale: gain, outIndices: ids,
            outWeights: weights, queryCount: 1, numExperts: 128, d: UInt32(d), topK: 8,
            hiddenStrideElements: UInt32(d))
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.status == .completed)
        #expect(Array(UnsafeBufferPointer(start: weights.contents().assumingMemoryBound(to: Float16.self), count: 8)) == expected)
        #expect(Array(UnsafeBufferPointer(start: ids.contents().assumingMemoryBound(to: UInt32.self), count: 8)) == (0..<8).map(UInt32.init))
    }
}
