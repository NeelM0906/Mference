import Metal
import Testing
@testable import Mference

@Suite struct FlashNextRouterWidthTests {
    @Test(arguments: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    func selectsAndWritesOnlyTheConfiguredWidth(topK: Int) throws {
        let context = try MetalContext()
        let kernel = try FlashNextMoE(context: context, routerTopK: topK)
        let experts = topK <= 6 ? 8 : topK <= 8 ? 32 : 512
        // Unequal top-k logits expose normalization over the wrong width.
        let logits: [Float] = (0..<experts).map { Float(($0 * 17 + 11) % experts) / Float(experts) }
        let expected = FlashNextRouterReference.select(logits: logits, k: topK)
        let logitBuffer = try #require(context.device.makeBuffer(bytes: logits, length: experts * 4,
                                                                 options: .storageModeShared))
        let scale = [UInt16](repeating: Quantization.bf16Bits(1), count: experts)
        let scaleBuffer = try #require(context.device.makeBuffer(bytes: scale, length: experts * 2,
                                                                 options: .storageModeShared))
        let prefix = 3, rows = 3, count = prefix + rows * topK + 12
        let ids = try #require(context.device.makeBuffer(length: count * 4, options: .storageModeShared))
        let weights = try #require(context.device.makeBuffer(length: count * 2, options: .storageModeShared))
        let ip = ids.contents().assumingMemoryBound(to: UInt32.self)
        let wp = weights.contents().assumingMemoryBound(to: Float16.self)
        ip.update(repeating: .max, count: count)
        wp.update(repeating: -123, count: count)
        let cb = try #require(context.queue.makeCommandBuffer())
        for row in 0..<rows {
            kernel.encodeRouterSelect(commandBuffer: cb, logits: logitBuffer, perExpertScale: scaleBuffer,
                outIndices: ids, outIndicesOffset: (prefix + row * topK) * 4,
                outWeights: weights, outWeightsOffset: (prefix + row * topK) * 2,
                numExperts: UInt32(experts))
        }
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.error == nil)
        for row in 0..<rows {
            var sum: Float = 0
            for rank in 0..<topK {
                let i = prefix + row * topK + rank
                #expect(ip[i] == UInt32(expected.indices[rank]))
                #expect(wp[i].isFinite)
                #expect(abs(Float(wp[i]) - expected.weights[rank]) < 0.001)
                sum += Float(wp[i])
            }
            #expect(abs(sum - 1) < 0.001)
        }
        for i in 0..<count where i < prefix || i >= prefix + rows * topK {
            #expect(ip[i] == .max, "router overwrote another row or canary")
            #expect(wp[i] == -123, "router overwrote another row or canary")
        }
    }
}
