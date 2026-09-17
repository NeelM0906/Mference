import Metal
import Testing
@testable import Mference

@Suite struct DSV4IndexerSelectionTests {
    @Test(arguments: [(1, 1), (13, 12), (513, 512), (1027, 512), (8193, 512), (37, 0)])
    func selectionMatchesStableCPUOrder(count: Int, topK: Int) throws {
        let ctx = try MetalContext()
        let kernels = try DSV4Kernels(context: ctx, config: .deepseekV4Flash_284B_A13B)
        let scores = try #require(ctx.device.makeBuffer(length: count * 4, options: .storageModeShared))
        let selected = try #require(ctx.device.makeBuffer(length: max(topK, 1) * 4, options: .storageModeShared))
        let input = scores.contents().assumingMemoryBound(to: Float.self)
        let output = selected.contents().assumingMemoryBound(to: UInt32.self)
        // Ties at and across the cutoff, signed zero, monotonic order, and a
        // deterministic mixed distribution exercise heap construction/replacement.
        for mode in 0..<5 {
            for i in 0..<count {
                switch mode {
                case 0: input[i] = Float((i * 8191 + 37) % 101) - 50
                case 1: input[i] = Float(i)
                case 2: input[i] = -Float(i)
                case 3: input[i] = i.isMultiple(of: 2) ? 0.0 : -0.0
                default: input[i] = i.isMultiple(of: 3) ? .infinity : -.infinity
                }
            }
            output[0] = .max
            let expected = (0..<count).sorted {
                input[$0] == input[$1] ? $0 < $1 : input[$0] > input[$1]
            }.prefix(topK).sorted().map(UInt32.init)
            let cb = try #require(ctx.queue.makeCommandBuffer())
            kernels.encodeIndexerSelect(commandBuffer: cb, scores: scores,
                selected: selected, entryCount: count, topK: topK)
            cb.commit(); cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            #expect(Array(UnsafeBufferPointer(start: output, count: topK)) == expected)
            if topK == 0 { #expect(output[0] == .max) }
        }
    }
}
