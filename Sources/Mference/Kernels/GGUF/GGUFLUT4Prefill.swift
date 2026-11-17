import Metal

/// LUT4 routed-expert prefill for one streamed tile (WS4 U3):
/// `gguf_lut4_prefill_phase1` / `gguf_lut4_prefill_down` in
/// `Metal/GGUF/gguf_lut4.metal`. The interface is that of
/// `prefill_grouped_routed_moe_batched_phase1` / `_down`: sorted
/// `PrefillTokenExpertPair`s, a ≤16-slot tile blob table, the streamed params
/// layout with the nine offsets read as LUT4 (index, D, M) per projection, the
/// three-plane gate/up/act scratch, and unweighted `routePartials` for the
/// existing token-major reduce. `Params` is layout-identical to
/// `PrefillGroupedRoutedMoEStreamedParams`.
final class GGUFLUT4Prefill {
    static let maxTileExperts = 16

    struct Params {
        var pairStart: UInt32, pairCount: UInt32, d: UInt32, f: UInt32, topK: UInt32
        var hiddenStrideElements: UInt32, liveExpertCount: UInt32
        var localExperts: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                           UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
        var offsets: GGUFLUT4MoE.ExpertOffsets

        init(pairStart: UInt32, pairCount: UInt32, d: UInt32, f: UInt32, topK: UInt32,
             hiddenStrideElements: UInt32, expertIDs: [UInt32], offsets: GGUFLUT4MoE.ExpertOffsets) {
            precondition(expertIDs.count <= GGUFLUT4Prefill.maxTileExperts)
            let ids = expertIDs + [UInt32](repeating: .max, count: GGUFLUT4Prefill.maxTileExperts - expertIDs.count)
            self.pairStart = pairStart; self.pairCount = pairCount; self.d = d; self.f = f; self.topK = topK
            self.hiddenStrideElements = hiddenStrideElements
            self.liveExpertCount = UInt32(expertIDs.count)
            localExperts = (ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
                            ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15])
            self.offsets = offsets
        }
    }

    private let phase1: MTLComputePipelineState
    private let down: MTLComputePipelineState
    private let argumentEncoder: MTLArgumentEncoder
    private let argumentBuffer: MTLBuffer

    init(context: MetalContext) throws {
        let library = try MetalContext.moduleLibrary(device: context.device, module: "gguf_lut4")
        guard let f1 = library.makeFunction(name: "gguf_lut4_prefill_phase1"),
              let f2 = library.makeFunction(name: "gguf_lut4_prefill_down") else {
            throw MetalError.missingFunction("gguf_lut4_prefill_*")
        }
        phase1 = try context.device.makeComputePipelineState(function: f1)
        down = try context.device.makeComputePipelineState(function: f2)
        argumentEncoder = f1.makeArgumentEncoder(bufferIndex: 9)
        guard let buffer = context.device.makeBuffer(length: argumentEncoder.encodedLength,
                                                     options: .storageModeShared) else {
            throw MetalError.commandBufferFailed("LUT4 prefill argument buffer allocation failed")
        }
        argumentBuffer = buffer
    }

    /// Encodes phase 1 then down for one tile. The tile blob table is reused:
    /// it must not be re-encoded while a command reading it is in flight.
    func encode(commandBuffer: MTLCommandBuffer,
                hidden: MTLBuffer, sortedPairs: MTLBuffer,
                gateUpActScratch: MTLBuffer, downScratch: MTLBuffer, routePartials: MTLBuffer,
                blobs: [(buffer: MTLBuffer, offset: Int)], params: Params,
                gateUp: GGUFLUT4.SourceType, downType: GGUFLUT4.SourceType) {
        precondition(blobs.count == Int(params.liveExpertCount))
        argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
        for (i, blob) in blobs.enumerated() { argumentEncoder.setBuffer(blob.buffer, offset: blob.offset, index: i) }
        var p = params
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        for (pipeline, rows, type) in [(phase1, params.f, gateUp), (down, params.d, downType)] {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
            var proj = GGUFLUT4MoE.ProjParams(type)
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(hidden, offset: 0, index: 0)
            encoder.setBuffer(sortedPairs, offset: 0, index: 1)
            encoder.setBuffer(routePartials, offset: 0, index: 5)
            encoder.setBuffer(gateUpActScratch, offset: 0, index: 7)
            encoder.setBuffer(downScratch, offset: 0, index: 8)
            encoder.setBuffer(argumentBuffer, offset: 0, index: 9)
            for blob in blobs { encoder.useResource(blob.buffer, usage: .read) }
            encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 10)
            encoder.setBytes(&proj, length: MemoryLayout<GGUFLUT4MoE.ProjParams>.stride, index: 11)
            encoder.dispatchThreads(MTLSize(width: Int(rows) * 32, height: Int(params.pairCount), depth: 1),
                                    threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}
