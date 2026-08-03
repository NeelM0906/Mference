import Foundation
import Metal
import Testing

@testable import Mference
import MferenceValidationSupport

/// `dsv4_attention_decode` was restructured from a two-pass kernel — one
/// threadgroup per query head, every logit materialized in a fixed-size
/// threadgroup array, every thread re-walking all rows for the value sum —
/// into an online-softmax kernel where one threadgroup services eight heads
/// and stages each shared K=V row into threadgroup memory once for all of
/// them. The pre-restructuring kernel survives in the shader source as
/// `dsv4_attention_decode_reference`; these tests pin the new kernel to it
/// over randomized inputs across the window-only, dense-compressed,
/// indexer-selected, and ragged-head-count shapes.
@Suite struct DSV4AttentionParityTests {

    /// The two kernels differ only in reduction order (sequential dot and a
    /// materialized two-pass softmax versus a simdgroup dot and an online
    /// one), so the FP16 outputs land within a couple of ulps. Measured max
    /// over these shapes: 1.2e-4.
    private static let tolerance: Float = 5e-4

    struct Shape: Sendable {
        let name: String
        let numHeads: Int
        let windowCount: Int
        let windowStartPos: Int
        let compressedCount: Int
        /// Selected-entry count, or `nil` for "attend to all compressed".
        let selectedCount: Int?
    }

    private static let shapes: [Shape] = [
        // Sliding-window layer, ring not yet full, row count not a multiple
        // of the staging block.
        Shape(name: "window-only-partial-stage", numHeads: 64,
              windowCount: 37, windowStartPos: 0,
              compressedCount: 0, selectedCount: nil),
        // HCA layer: full window plus a dense compressed region.
        Shape(name: "window-plus-dense-compressed", numHeads: 64,
              windowCount: 128, windowStartPos: 501,
              compressedCount: 200, selectedCount: nil),
        // CSA layer past index_topk: the indexer narrows the compressed
        // region to a scattered subset.
        Shape(name: "indexer-selected-subset", numHeads: 64,
              windowCount: 128, windowStartPos: 4123,
              compressedCount: 300, selectedCount: 97),
        // Compressed rows only.
        Shape(name: "compressed-only", numHeads: 64,
              windowCount: 0, windowStartPos: 0,
              compressedCount: 64, selectedCount: nil),
        // Head count that does not fill the last threadgroup: the tail
        // simdgroup must still reach every barrier and write nothing.
        Shape(name: "ragged-head-count", numHeads: 60,
              windowCount: 128, windowStartPos: 900,
              compressedCount: 300, selectedCount: 137),
    ]

    private static let headDim = 512
    private static let ringCapacity = 128
    private static let maxHeads = 64
    private static let maxCompressed = 512

    @Test(arguments: shapes)
    func matchesTheTwoPassReference(shape: Shape) throws {
        let ctx = try MetalContext()
        let kernels = try DSV4Kernels(context: ctx,
                                      config: .deepseekV4Flash_284B_A13B)
        let referencePSO = try ctx.pipeline("dsv4_attention_decode_reference",
                                            constants: [],
                                            maxTotalThreadsPerThreadgroup: 256)
        let device = ctx.device

        func halfBuffer(_ count: Int) throws -> MTLBuffer {
            try #require(device.makeBuffer(
                length: count * MemoryLayout<Float16>.stride,
                options: .storageModeShared))
        }

        var rng = SplitMix64(seed: 0xD5_4000_0000_0001)
        let q = try halfBuffer(Self.maxHeads * Self.headDim)
        let windowKV = try halfBuffer(Self.ringCapacity * Self.headDim)
        let compressedKV = try halfBuffer(Self.maxCompressed * Self.headDim)
        fillUnitHalves(q, count: Self.maxHeads * Self.headDim, rng: &rng)
        fillUnitHalves(windowKV, count: Self.ringCapacity * Self.headDim,
                       rng: &rng)
        fillUnitHalves(compressedKV, count: Self.maxCompressed * Self.headDim,
                       rng: &rng)

        let sinks = try #require(device.makeBuffer(
            length: Self.maxHeads * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let sinkPtr = sinks.contents().assumingMemoryBound(to: Float.self)
        for h in 0..<Self.maxHeads {
            sinkPtr[h] = rng.uniform(-2.0, 2.0)
        }

        let selected = try #require(device.makeBuffer(
            length: max(1, Self.maxCompressed) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let selectedCount: UInt32
        if let k = shape.selectedCount {
            // Production hands the kernel a sorted, distinct subset.
            var picks = Set<UInt32>()
            while picks.count < k {
                picks.insert(UInt32(rng.next() % UInt64(shape.compressedCount)))
            }
            let sorted = picks.sorted()
            let ptr = selected.contents().assumingMemoryBound(to: UInt32.self)
            for (i, e) in sorted.enumerated() { ptr[i] = e }
            selectedCount = UInt32(k)
        } else {
            selectedCount = DSV4Kernels.selectAll
        }

        let outNew = try halfBuffer(Self.maxHeads * Self.headDim)
        let outRef = try halfBuffer(Self.maxHeads * Self.headDim)
        memset(outNew.contents(), 0, outNew.length)
        memset(outRef.contents(), 0, outRef.length)

        let scale = Float(ArchConfig.deepseekV4Flash_284B_A13B.attentionScale)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        kernels.encodeAttention(
            commandBuffer: cb,
            q: q, windowKV: windowKV, compressedKV: compressedKV,
            selected: selected,
            sinks: sinks, sinksOffset: 0,
            out: outNew,
            headDim: Self.headDim, numHeads: shape.numHeads,
            windowCount: shape.windowCount,
            windowStartPos: shape.windowStartPos,
            ringCapacity: Self.ringCapacity,
            compressedCount: shape.compressedCount,
            selectedCount: selectedCount,
            scale: scale)
        try encodeReference(
            pso: referencePSO, commandBuffer: cb,
            q: q, windowKV: windowKV, compressedKV: compressedKV,
            selected: selected, sinks: sinks, out: outRef,
            numHeads: shape.numHeads,
            windowCount: shape.windowCount,
            windowStartPos: shape.windowStartPos,
            compressedCount: shape.compressedCount,
            selectedCount: selectedCount,
            scale: scale)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let newPtr = outNew.contents().assumingMemoryBound(to: Float16.self)
        let refPtr = outRef.contents().assumingMemoryBound(to: Float16.self)
        var maxDiff: Float = 0
        var worst = 0
        for i in 0..<(shape.numHeads * Self.headDim) {
            let d = abs(Float(newPtr[i]) - Float(refPtr[i]))
            if d > maxDiff { maxDiff = d; worst = i }
        }
        #expect(maxDiff <= Self.tolerance,
                "\(shape.name): max |new - reference| = \(maxDiff) at head \(worst / Self.headDim), channel \(worst % Self.headDim)")

        // A convex combination of unit-magnitude rows is never all zero, so a
        // silently skipped kernel would pass the diff check above.
        var nonZero = 0
        for i in 0..<(shape.numHeads * Self.headDim) where newPtr[i] != 0 {
            nonZero += 1
        }
        #expect(nonZero > shape.numHeads * Self.headDim / 2,
                "\(shape.name): output is mostly zero (\(nonZero) non-zero)")

        // Heads past `numHeads` belong to no simdgroup with a store.
        for i in (shape.numHeads * Self.headDim)..<(Self.maxHeads * Self.headDim) {
            #expect(newPtr[i] == 0, "\(shape.name): wrote past numHeads")
        }
    }

    /// Hand-encode the retired two-pass kernel: one threadgroup per head, no
    /// `num_heads` argument.
    private func encodeReference(pso: MTLComputePipelineState,
                                 commandBuffer: MTLCommandBuffer,
                                 q: MTLBuffer, windowKV: MTLBuffer,
                                 compressedKV: MTLBuffer, selected: MTLBuffer,
                                 sinks: MTLBuffer, out: MTLBuffer,
                                 numHeads: Int,
                                 windowCount: Int, windowStartPos: Int,
                                 compressedCount: Int, selectedCount: UInt32,
                                 scale: Float) throws {
        let enc = try #require(commandBuffer.makeComputeCommandEncoder())
        enc.setComputePipelineState(pso)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(windowKV, offset: 0, index: 1)
        enc.setBuffer(compressedKV, offset: 0, index: 2)
        enc.setBuffer(selected, offset: 0, index: 3)
        enc.setBuffer(sinks, offset: 0, index: 4)
        enc.setBuffer(out, offset: 0, index: 5)
        var hd = UInt32(Self.headDim)
        var wc = UInt32(windowCount)
        var ws = UInt32(windowStartPos)
        var rc = UInt32(Self.ringCapacity)
        var cc = UInt32(compressedCount)
        var sc = selectedCount
        var sl = scale
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&wc, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&ws, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&rc, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&cc, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&sc, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&sl, length: MemoryLayout<Float>.size, index: 12)
        enc.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func fillUnitHalves(_ buffer: MTLBuffer, count: Int,
                                rng: inout SplitMix64) {
        let ptr = buffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<count { ptr[i] = Float16(rng.uniform(-1.0, 1.0)) }
    }
}
