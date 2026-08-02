import Foundation
import Metal
import Testing

@testable import Mference

/// Regression coverage for the compressed-entry ceiling: a CSA layer past
/// 8K tokens has more than 2048 *emitted* entries, but the indexer narrows
/// attention to its top-512 picks, so encoding must bound what the kernel
/// actually loads — not the total cache count. The old precondition
/// terminated the process on `compressedCount > 2048` even with selection
/// active, which crashed every context option above 8K.
@Suite struct DSV4AttentionPreconditionTests {

    private func makeKernels(_ ctx: MetalContext) throws -> DSV4Kernels {
        try DSV4Kernels(context: ctx, config: .deepseekV4Flash_284B_A13B)
    }

    @Test func selectionBoundedEntryCountsEncodePastEmittedCeiling() throws {
        let ctx = try MetalContext()
        let kernels = try makeKernels(ctx)
        let device = ctx.device
        func buffer(_ length: Int) throws -> MTLBuffer {
            try #require(device.makeBuffer(length: length,
                                           options: .storageModeShared))
        }
        let scratch = try buffer(1 << 20)
        let cb = try #require(ctx.queue.makeCommandBuffer())

        // CSA shape at ~16K tokens: 4096 emitted entries, indexer selection
        // active at 512. Encoding must not trap; the kernel only gathers
        // the selected 512.
        kernels.encodeAttention(
            commandBuffer: cb,
            q: scratch,
            windowKV: scratch,
            compressedKV: scratch,
            selected: scratch,
            sinks: scratch, sinksOffset: 0,
            out: scratch,
            headDim: 512, numHeads: 64,
            windowCount: 128, windowStartPos: 16_256, ringCapacity: 128,
            compressedCount: 4096, selectedCount: 512,
            scale: 0.044)

        // HCA at the 64K context ceiling: 512 dense entries, no selection.
        kernels.encodeAttention(
            commandBuffer: cb,
            q: scratch,
            windowKV: scratch,
            compressedKV: scratch,
            selected: scratch,
            sinks: scratch, sinksOffset: 0,
            out: scratch,
            headDim: 512, numHeads: 64,
            windowCount: 128, windowStartPos: 65_408, ringCapacity: 128,
            compressedCount: 512, selectedCount: DSV4Kernels.selectAll,
            scale: 0.044)
        // Encode-only: nothing is committed, the buffers carry no real
        // model state. Reaching this line is the regression assertion.
        #expect(Bool(true))
    }
}
