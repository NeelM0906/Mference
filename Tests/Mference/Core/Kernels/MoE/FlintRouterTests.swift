import Foundation
import Metal
import Testing
@testable import Mference

@Suite
struct FlintRouterTests {
    @Test func sigmoidTop6RenormalizesSelectedScores() throws {
        let context = try MetalContext()
        let router = try FlintRouter(context: context)
        let experts = FlintRouter.expertCount
        let width = FlintRouter.hiddenSize
        let weightBits: [UInt16] = (0..<(experts * width)).map { index in
            Quantization.bf16Bits(Float((index * 13) % 41 - 20) / 128)
        }
        let hiddenValues: [Float16] = (0..<width).map { index in
            Float16(Float((index * 17) % 37 - 18) / 64)
        }
        let weights = try buffer(context, weightBits, scalar: UInt16.self)
        let hidden = try buffer(context, hiddenValues, scalar: Float16.self)
        let indicesValues = [UInt32](repeating: 0xFFFF_FFFF, count: FlintRouter.topK)
        let indices = try buffer(context, indicesValues, scalar: UInt32.self)
        let routingValues = [Float16](repeating: 0, count: FlintRouter.topK)
        let routing = try buffer(context, routingValues, scalar: Float16.self)

        let cb = try #require(context.queue.makeCommandBuffer())
        router.encode(
            commandBuffer: cb,
            weights: weights,
            hidden: hidden,
            indices: indices,
            routingWeights: routing)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)

        let actualIndices = read(indices, count: FlintRouter.topK, scalar: UInt32.self)
        let actualWeights = read(routing, count: FlintRouter.topK, scalar: Float16.self)
        let logits: [(Int, Float)] = (0..<experts).map { expert in
            var value: Float = 0
            for i in 0..<width {
                value += Quantization.bf16ToFloat(weightBits[expert * width + i])
                    * Float(hiddenValues[i])
            }
            return (expert, value)
        }
        let expected = logits.sorted {
            if $0.1 != $1.1 { return $0.1 > $1.1 }
            return $0.0 < $1.0
        }.prefix(FlintRouter.topK)
        #expect(actualIndices.map(Int.init) == expected.map(\.0))

        let selected = expected.map { score -> Float in
            score.1 >= 0
                ? 1 / (1 + exp(-score.1))
                : exp(score.1) / (1 + exp(score.1))
        }
        let denom = selected.reduce(0, +)
        for i in 0..<FlintRouter.topK {
            #expect(abs(Float(actualWeights[i]) - selected[i] / denom) <= 0.001)
        }
        #expect(abs(actualWeights.reduce(0) { $0 + Float($1) } - 1) <= 0.003)
    }

    @Test func batchedRouterMatchesPerTokenSelection() throws {
        let context = try MetalContext()
        let tokens = 3
        let router = try FlintPrefillRouter(context: context, maxTokens: tokens)
        let experts = FlintRouter.expertCount
        let width = FlintRouter.hiddenSize
        let weightBits: [UInt16] = (0..<(experts * width)).map { index in
            Quantization.bf16Bits(Float((index * 7) % 43 - 21) / 128)
        }
        let hiddenValues: [Float16] = (0..<(tokens * width)).map { index in
            Float16(Float((index * 19) % 47 - 23) / 64)
        }
        let weights = try buffer(context, weightBits, scalar: UInt16.self)
        let hidden = try buffer(context, hiddenValues, scalar: Float16.self)
        let indices = try buffer(
            context,
            [UInt32](repeating: 0xFFFF_FFFF, count: tokens * FlintRouter.topK),
            scalar: UInt32.self)
        let routing = try buffer(
            context,
            [Float16](repeating: 0, count: tokens * FlintRouter.topK),
            scalar: Float16.self)

        let cb = try #require(context.queue.makeCommandBuffer())
        router.encode(
            commandBuffer: cb,
            weights: weights,
            hidden: hidden,
            hiddenStrideElements: width,
            indices: indices,
            routingWeights: routing,
            tokens: tokens)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)

        let actualIndices = read(
            indices, count: tokens * FlintRouter.topK, scalar: UInt32.self)
        let actualWeights = read(
            routing, count: tokens * FlintRouter.topK, scalar: Float16.self)
        for token in 0..<tokens {
            let logits: [(Int, Float)] = (0..<experts).map { expert in
                var value: Float = 0
                for i in 0..<width {
                    value += Quantization.bf16ToFloat(weightBits[expert * width + i])
                        * Float(hiddenValues[token * width + i])
                }
                return (expert, value)
            }
            let expected = logits.sorted {
                if $0.1 != $1.1 { return $0.1 > $1.1 }
                return $0.0 < $1.0
            }.prefix(FlintRouter.topK)
            let base = token * FlintRouter.topK
            #expect(actualIndices[base..<(base + FlintRouter.topK)].map(Int.init)
                    == expected.map(\.0))
            let selected = expected.map { score -> Float in
                score.1 >= 0
                    ? 1 / (1 + exp(-score.1))
                    : exp(score.1) / (1 + exp(score.1))
            }
            let denom = selected.reduce(0, +)
            for rank in 0..<FlintRouter.topK {
                #expect(abs(Float(actualWeights[base + rank])
                            - selected[rank] / denom) <= 0.001)
            }
        }
    }

    private func buffer<T>(_ context: MetalContext,
                           _ values: [T],
                           scalar: T.Type) throws -> MTLBuffer {
        let bytes = values.count * MemoryLayout<T>.stride
        let buffer = try #require(
            context.device.makeBuffer(length: bytes, options: .storageModeShared))
        _ = values.withUnsafeBytes {
            memcpy(buffer.contents(), $0.baseAddress!, bytes)
        }
        return buffer
    }

    private func read<T>(_ buffer: MTLBuffer,
                         count: Int,
                         scalar: T.Type) -> [T] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: T.self, capacity: count),
            count: count))
    }
}
