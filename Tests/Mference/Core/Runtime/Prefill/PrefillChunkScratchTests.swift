import Testing
import Metal
@testable import Mference

@Suite struct PrefillChunkScratchTests {
    @Test func gemma4T32LayoutMatchesTask7ScratchContract() {
        let layout = PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 32)

        #expect(layout.chunkTokens == 32)
        #expect(layout.hiddenElements == 32 * 2816)
        #expect(layout.normedElements == 32 * 2816)
        #expect(layout.qElements == 32 * 8192)
        #expect(layout.kStageElements == 32 * 2048)
        #expect(layout.vStageElements == 32 * 2048)
        #expect(layout.attentionOutputElements == 32 * 8192)
        #expect(layout.denseXElements == 32 * 2816)
        #expect(layout.routedXElements == 32 * 2816)
        #expect(layout.routerXElements == 32 * 2816)
        #expect(layout.h1Elements == 32 * 2816)
        #expect(layout.h2Elements == 32 * 2816)
        #expect(layout.routePartialElements == 32 * 8 * 2816)
        #expect(layout.routeIDElements == 32 * 8)
        #expect(layout.routeWeightElements == 32 * 8)
        #expect(layout.sharedExpertScratchElements == 2112)
        #expect(layout.routedPairMicrobatchRows == 32)
        #expect(layout.routedGateUpActElements == 3 * 32 * 704)
        #expect(layout.routedDownOutputElements == 32 * 2816)

        let worksheetT32UpperBound = Int(4.5 * 1_048_576.0)
        #expect(layout.totalPersistentBytes <= worksheetT32UpperBound)
    }

    @Test func layoutClampsChunkSizeToRuntimeBounds() {
        #expect(PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 0).chunkTokens == 1)
        #expect(PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 512).chunkTokens == 512)
        #expect(PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 8192).chunkTokens
                == PrefillRuntimeConfig.maxChunkTokens)
    }

    @Test(arguments: [256, 512, 1024, 2048, 4096])
    func layoutScalesWithLargerChunks(chunk: Int) {
        let layout = PrefillChunkScratchLayout(config: .gemma4_26B_A4B,
                                               chunkTokens: chunk)
        #expect(layout.chunkTokens == chunk)
        #expect(layout.hiddenElements == chunk * 2816)
        #expect(layout.routeIDElements == chunk * 8)
        // The arena stays bounded: linear in the chunk, no hidden square
        // term. 128 tokens measured ~15.6 MiB; admit modest slack.
        let perTokenBytes = 16.0 * 1_048_576.0 / 128.0
        #expect(layout.totalPersistentBytes <= Int(Double(chunk) * perTokenBytes * 1.3))
    }

    @Test func batchedSharedExpertKeepsOneScratchRowPerToken() {
        let batched = PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 1024,
                                                batchedSharedExpert: true)
        #expect(batched.sharedExpertScratchElements == 1024 * 2112)
        // Families that dispatch the shared expert row by row keep one row.
        #expect(PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 1024)
            .sharedExpertScratchElements == 2112)
    }

    @Test func groupedExpertActivationGetsItsOwnBufferWhenAttentionOutputIsTooSmall() throws {
        // Toy shape: 64 x 8 x 128 activation rows against 64 x 4 x 32 attention output.
        let layout = PrefillChunkScratchLayout(config: .gemma4Toy(topKExperts: 8), chunkTokens: 64,
                                               groupedExperts: true)
        #expect(layout.groupedExpertActivationElements > layout.attentionOutputElements)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let buffers = try PrefillChunkScratchBuffers.allocate(device: device, layout: layout)
        let activation = try #require(buffers.groupedExpertActivation)
        #expect(activation !== buffers.attentionOutput)
        #expect(activation.length >= layout.groupedExpertActivationElements * MemoryLayout<Float16>.stride)
        // Row-kernel-only layouts allocate nothing for it.
        let rowOnly = try PrefillChunkScratchBuffers.allocate(device: device,
            layout: PrefillChunkScratchLayout(config: .gemma4Toy(topKExperts: 8), chunkTokens: 64))
        #expect(rowOnly.groupedExpertActivation == nil)
    }

    @Test func groupedExpertActivationReusesTheIdleAttentionOutput() throws {
        let layout = PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 64,
                                               batchedSharedExpert: true, groupedExperts: true)
        #expect(layout.groupedExpertActivationElements == 64 * 8 * 704)
        #expect(layout.groupedExpertActivationElements <= layout.attentionOutputElements)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let buffers = try PrefillChunkScratchBuffers.allocate(device: device, layout: layout)
        // Attention output is consumed by the O projection before any routed
        // tile runs, so grouped experts borrow it instead of growing the arena.
        #expect(buffers.groupedExpertActivation === buffers.attentionOutput)
    }

    /// Owner budget (2026-09-21): raising the server's Gemma prefill chunk from
    /// 128 to 1024 tokens may cost about 310 MB and no more.
    @Test func serverChunkGrowthStaysWithinTheOwnerMemoryBudget() {
        let config = ArchConfig.gemma4_26B_A4B
        let today = PrefillChunkScratchLayout(config: config, chunkTokens: 128).totalPersistentBytes
        let planned = PrefillChunkScratchLayout(config: config, chunkTokens: 1024,
                                                batchedSharedExpert: true,
                                                groupedExperts: true).totalPersistentBytes
        let slidingLayers = config.fullAttentionLayerMask.filter { $0 == 0 }.count
        let ringBytesPerToken = slidingLayers * 2 * config.numKVHeads * config.headDim
            * MemoryLayout<Float16>.stride
        let growth = (planned - today) + (1024 - 128) * ringBytesPerToken
        #expect(growth <= 310_000_000, "growth=\(growth)")
    }

    @Test func allocationUsesPrivateScratchAndSharedRouteMetadata() throws {
        let ctx = try MetalContext()
        let toy = ArchConfig(hiddenSize: 64,
                             intermediateSize: 48,
                             moeIntermediateSize: 16,
                             numHeads: 4,
                             numKVHeads: 2,
                             numFullKVHeads: 1,
                             headDim: 16,
                             fullHeadDim: 32,
                             vocabSize: 128,
                             slidingWindow: 16,
                             finalLogitSoftcap: 30.0,
                             ropeTheta: 10_000,
                             fullRopeTheta: 1_000_000,
                             partialRotaryFactor: 0.25,
                             numLayers: 2,
                             numExperts: 8,
                             topKExperts: 2,
                             tieWordEmbeddings: true,
                             attentionKEqV: true,
                             fullAttentionLayerMask: [0, 1],
                             hiddenActivation: "gelu_pytorch_tanh")
        let layout = PrefillChunkScratchLayout(config: toy, chunkTokens: 4)

        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)

        #expect(scratch.layout == layout)
        #expect(scratch.hidden.length == layout.hiddenElements * MemoryLayout<Float16>.stride)
        #expect(scratch.denseX.length == layout.denseXElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routedX.length == layout.routedXElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routerX.length == layout.routerXElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routePartials.length == layout.routePartialElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routeIDs.length == layout.routeIDElements * MemoryLayout<UInt32>.stride)
        #expect(scratch.routeWeights.length == layout.routeWeightElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routedGateUpActScratch.length == layout.routedGateUpActElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routedDownScratch.length == layout.routedDownOutputElements * MemoryLayout<Float16>.stride)
        #expect(scratch.hidden.storageMode == MTLStorageMode.private)
        #expect(scratch.denseX.storageMode == MTLStorageMode.private)
        #expect(scratch.routedX.storageMode == MTLStorageMode.private)
        #expect(scratch.routerX.storageMode == MTLStorageMode.private)
        #expect(scratch.routedGateUpActScratch.storageMode == MTLStorageMode.private)
        #expect(scratch.routedDownScratch.storageMode == MTLStorageMode.private)
        #expect(scratch.routeIDs.storageMode == MTLStorageMode.shared)
        #expect(scratch.routeWeights.storageMode == MTLStorageMode.shared)
    }
}
