import Foundation
import Metal
import Testing

@testable import Mference

/// Regression coverage for the compressed-entry ceiling. A precondition on
/// the total cache count once terminated the process at `compressedCount >
/// 2048`, crashing every context option above 8K even though the indexer had
/// already narrowed attention to its top-512 picks. The online-softmax kernel
/// streams entries instead of materializing logits, so no ceiling remains at
/// all — encoding must stay trap-free at any entry count.
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

        // Well past the retired 2048-entry ceiling with no selection active:
        // the kernel no longer caps the enumerated region.
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
            compressedCount: 4096, selectedCount: DSV4Kernels.selectAll,
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
