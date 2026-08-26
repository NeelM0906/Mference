import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// Parity for `attention_decode_paged_partial`: with the full selection the
/// paged kernel must reproduce the contiguous `encodeFull` result exactly —
/// identical split geometry and accumulation order, so bit-identical FP16
/// output — regardless of how pages are scattered across pool slots.
@Suite struct PagedAttentionParityTests {

    private static let pageTokens = 64

    private struct Harness {
        let ctx: MetalContext
        let kernel: Attention
        let q: MTLBuffer
        let kLinear: MTLBuffer     // [seqLen, numKVHeads, headDim] logical order
        let vLinear: MTLBuffer
        let kPool: MTLBuffer       // page-scattered copy
        let vPool: MTLBuffer
        let pageTable: MTLBuffer
        let outLinear: MTLBuffer
        let outPaged: MTLBuffer
        let headDim: Int
        let numQHeads: Int
        let numKVHeads: Int
        let seqLen: Int

        /// `slotOf[pageIndex]` scatters logical pages across pool slots.
        init(seqLen: Int, headDim: Int, numQHeads: Int, numKVHeads: Int,
             slotOf: [Int], seed: UInt64) throws {
            self.headDim = headDim
            self.numQHeads = numQHeads
            self.numKVHeads = numKVHeads
            self.seqLen = seqLen
            let pages = (seqLen + pageTokens - 1) / pageTokens
            precondition(slotOf.count == pages)

            var rng = SeedTree(seed).key("paged-attn")
            let qVals = (0..<numQHeads * headDim).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
            let rowElems = numKVHeads * headDim
            let kVals = (0..<seqLen * rowElems).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
            let vVals = (0..<seqLen * rowElems).map { _ in Float16(rng.uniform(-0.5, 0.5)) }

            self.ctx = try MetalContext()
            self.kernel = try Attention(context: ctx)
            let device = ctx.device
            guard let qB = Fp16Buffer.make(device, halves: qVals),
                  let kL = Fp16Buffer.make(device, halves: kVals),
                  let vL = Fp16Buffer.make(device, halves: vVals),
                  let oL = Fp16Buffer.make(device, count: numQHeads * headDim),
                  let oP = Fp16Buffer.make(device, count: numQHeads * headDim) else {
                throw KVPageStoreError.allocationFailed("test buffers")
            }
            self.q = qB; self.kLinear = kL; self.vLinear = vL
            self.outLinear = oL; self.outPaged = oP

            // Scatter logical pages into pool slots. Pool sized to the max
            // slot referenced (+1); rows keep the [token, kvHead, headDim]
            // stride within each page.
            let poolSlots = (slotOf.max() ?? 0) + 1
            let pageElems = Self.pageTokens_ * rowElems
            guard let kP = Fp16Buffer.make(device, count: poolSlots * pageElems),
                  let vP = Fp16Buffer.make(device, count: poolSlots * pageElems) else {
                throw KVPageStoreError.allocationFailed("pool buffers")
            }
            let kSrc = kL.contents().assumingMemoryBound(to: Float16.self)
            let vSrc = vL.contents().assumingMemoryBound(to: Float16.self)
            let kDst = kP.contents().assumingMemoryBound(to: Float16.self)
            let vDst = vP.contents().assumingMemoryBound(to: Float16.self)
            for page in 0..<pages {
                let tokens = min(Self.pageTokens_, seqLen - page * Self.pageTokens_)
                let src = page * Self.pageTokens_ * rowElems
                let dst = slotOf[page] * pageElems
                for e in 0..<(tokens * rowElems) {
                    kDst[dst + e] = kSrc[src + e]
                    vDst[dst + e] = vSrc[src + e]
                }
            }
            self.kPool = kP; self.vPool = vP

            var table = slotOf.map(UInt32.init)
            guard let tB = device.makeBuffer(bytes: &table,
                                             length: table.count * 4,
                                             options: .storageModeShared) else {
                throw KVPageStoreError.allocationFailed("page table")
            }
            self.pageTable = tB
        }

        private static let pageTokens_ = 64

        func run() throws -> (linear: [Float16], paged: [Float16]) {
            let cb = ctx.queue.makeCommandBuffer()!
            kernel.encodeFull(commandBuffer: cb,
                              q: q, k: kLinear, v: vLinear, out: outLinear,
                              headDim: UInt32(headDim),
                              numQHeads: UInt32(numQHeads),
                              numKVHeads: UInt32(numKVHeads),
                              seqLen: UInt32(seqLen))
            cb.commit(); cb.waitUntilCompleted()

            let cb2 = ctx.queue.makeCommandBuffer()!
            kernel.encodeFullPaged(commandBuffer: cb2,
                                   q: q, kPool: kPool, vPool: vPool,
                                   pageTable: pageTable,
                                   out: outPaged,
                                   headDim: UInt32(headDim),
                                   numQHeads: UInt32(numQHeads),
                                   numKVHeads: UInt32(numKVHeads),
                                   selTokens: UInt32(seqLen))
            cb2.commit(); cb2.waitUntilCompleted()

            let n = numQHeads * headDim
            let lp = outLinear.contents().assumingMemoryBound(to: Float16.self)
            let pp = outPaged.contents().assumingMemoryBound(to: Float16.self)
            return ((0..<n).map { lp[$0] }, (0..<n).map { pp[$0] })
        }
    }

    /// Identity table: paged pool is laid out exactly like the linear cache.
    @Test func identitySelection_bitIdenticalToContiguous_qwen38Shape() throws {
        let seqLen = 512   // 8 full pages
        let h = try Harness(seqLen: seqLen, headDim: 256, numQHeads: 24, numKVHeads: 4,
                            slotOf: Array(0..<8), seed: 0xA11CE)
        let (linear, paged) = try h.run()
        #expect(linear == paged)
    }

    /// Scattered slots: logical page i lives at an arbitrary pool slot. Same
    /// bits must come out — slot placement cannot leak into the math.
    @Test func scatteredSlots_bitIdenticalToContiguous() throws {
        let seqLen = 512
        let h = try Harness(seqLen: seqLen, headDim: 256, numQHeads: 24, numKVHeads: 4,
                            slotOf: [5, 2, 7, 0, 3, 6, 1, 4], seed: 0xB0B)
        let (linear, paged) = try h.run()
        #expect(linear == paged)
    }

    /// Partial tail page: seqLen not a page multiple.
    @Test func partialTailPage_bitIdenticalToContiguous() throws {
        let seqLen = 7 * 64 + 17
        let h = try Harness(seqLen: seqLen, headDim: 256, numQHeads: 24, numKVHeads: 4,
                            slotOf: [3, 1, 4, 0, 7, 2, 6, 5], seed: 0xCAFE)
        let (linear, paged) = try h.run()
        #expect(linear == paged)
    }

    /// Generic (non-specialized PSO) shape exercises the fallback pipeline.
    @Test func genericShape_bitIdenticalToContiguous() throws {
        let seqLen = 3 * 64 + 5
        let h = try Harness(seqLen: seqLen, headDim: 64, numQHeads: 8, numKVHeads: 2,
                            slotOf: [2, 0, 3, 1], seed: 0xD00D)
        let (linear, paged) = try h.run()
        #expect(linear == paged)
    }

    /// A true sparse subset must equal the contiguous kernel run over a
    /// compacted copy of just the selected pages — softmax over the subset.
    @Test func sparseSubset_matchesCompactedContiguous() throws {
        let headDim = 256, numQHeads = 24, numKVHeads = 4
        let rowElems = numKVHeads * headDim
        let fullSeq = 8 * 64
        var rng = SeedTree(0xFEED).key("sparse-subset")
        let qVals = (0..<numQHeads * headDim).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let kVals = (0..<fullSeq * rowElems).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let vVals = (0..<fullSeq * rowElems).map { _ in Float16(rng.uniform(-0.5, 0.5)) }

        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        let device = ctx.device
        let selectedPages = [0, 3, 6]     // sinks + spread, 192 tokens
        let selTokens = selectedPages.count * 64

        // Compacted linear copy of the selected pages.
        var compactK = [Float16](); compactK.reserveCapacity(selTokens * rowElems)
        var compactV = [Float16]()
        for page in selectedPages {
            let base = page * 64 * rowElems
            compactK.append(contentsOf: kVals[base..<(base + 64 * rowElems)])
            compactV.append(contentsOf: vVals[base..<(base + 64 * rowElems)])
        }

        // Paged pool: identity layout of the full cache; table selects pages.
        var table = selectedPages.map(UInt32.init)
        guard let qB = Fp16Buffer.make(device, halves: qVals),
              let kPool = Fp16Buffer.make(device, halves: kVals),
              let vPool = Fp16Buffer.make(device, halves: vVals),
              let kC = Fp16Buffer.make(device, halves: compactK),
              let vC = Fp16Buffer.make(device, halves: compactV),
              let oC = Fp16Buffer.make(device, count: numQHeads * headDim),
              let oP = Fp16Buffer.make(device, count: numQHeads * headDim),
              let tB = device.makeBuffer(bytes: &table, length: table.count * 4,
                                         options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeFull(commandBuffer: cb,
                          q: qB, k: kC, v: vC, out: oC,
                          headDim: UInt32(headDim), numQHeads: UInt32(numQHeads),
                          numKVHeads: UInt32(numKVHeads), seqLen: UInt32(selTokens))
        cb.commit(); cb.waitUntilCompleted()

        let cb2 = ctx.queue.makeCommandBuffer()!
        kernel.encodeFullPaged(commandBuffer: cb2,
                               q: qB, kPool: kPool, vPool: vPool, pageTable: tB,
                               out: oP,
                               headDim: UInt32(headDim), numQHeads: UInt32(numQHeads),
                               numKVHeads: UInt32(numKVHeads),
                               selTokens: UInt32(selTokens))
        cb2.commit(); cb2.waitUntilCompleted()

        let n = numQHeads * headDim
        let c = oC.contents().assumingMemoryBound(to: Float16.self)
        let p = oP.contents().assumingMemoryBound(to: Float16.self)
        #expect((0..<n).allSatisfy { c[$0] == p[$0] })
    }
}
