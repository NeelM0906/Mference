import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// `kv_page_minmax` and `attention_page_scores` against CPU references, plus
/// `KVPageSelector` policy behavior.
@Suite struct KVPageKernelsTests {

    private static let pageTokens = 64

    // MARK: kv_page_minmax

    @Test func pageMinMax_matchesCPUReference() throws {
        let numKVHeads = 4, headDim = 256
        let elems = numKVHeads * headDim
        let poolSlots = 3, slot = 2       // non-zero slot exercises addressing
        var rng = SeedTree(0x3117).key("minmax")
        let pool = (0..<poolSlots * Self.pageTokens * elems)
            .map { _ in Float16(rng.uniform(-2.0, 2.0)) }

        let ctx = try MetalContext()
        let kernels = try KVPageKernels(context: ctx)
        guard let poolBuf = Fp16Buffer.make(ctx.device, halves: pool),
              let metaBuf = Fp16Buffer.make(ctx.device, count: 2 * elems) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernels.encodePageMinMax(commandBuffer: cb,
                                 kPool: poolBuf,
                                 slot: UInt32(slot),
                                 validTokens: UInt32(Self.pageTokens),
                                 metadata: metaBuf, metadataOffset: 0,
                                 numKVHeads: UInt32(numKVHeads),
                                 headDim: UInt32(headDim))
        cb.commit(); cb.waitUntilCompleted()

        let meta = metaBuf.contents().assumingMemoryBound(to: Float16.self)
        let base = slot * Self.pageTokens * elems
        for e in 0..<elems {
            var mn = Float.infinity, mx = -Float.infinity
            for t in 0..<Self.pageTokens {
                let v = Float(pool[base + t * elems + e])
                mn = min(mn, v); mx = max(mx, v)
            }
            #expect(meta[e] == Float16(mn))
            #expect(meta[elems + e] == Float16(mx))
        }
    }

    @Test func pageMinMax_partialPage_ignoresInvalidRows() throws {
        let numKVHeads = 2, headDim = 64
        let elems = numKVHeads * headDim
        var rng = SeedTree(0x3118).key("minmax-partial")
        var pool = (0..<Self.pageTokens * elems).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        // Poison the invalid tail rows; they must not leak into the summary.
        for t in 17..<Self.pageTokens {
            for e in 0..<elems { pool[t * elems + e] = Float16(1000.0) }
        }

        let ctx = try MetalContext()
        let kernels = try KVPageKernels(context: ctx)
        guard let poolBuf = Fp16Buffer.make(ctx.device, halves: pool),
              let metaBuf = Fp16Buffer.make(ctx.device, count: 2 * elems) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernels.encodePageMinMax(commandBuffer: cb,
                                 kPool: poolBuf, slot: 0, validTokens: 17,
                                 metadata: metaBuf, metadataOffset: 0,
                                 numKVHeads: UInt32(numKVHeads),
                                 headDim: UInt32(headDim))
        cb.commit(); cb.waitUntilCompleted()

        let meta = metaBuf.contents().assumingMemoryBound(to: Float16.self)
        for e in 0..<elems {
            #expect(Float(meta[elems + e]) < 999.0)
        }
    }

    // MARK: attention_page_scores

    @Test func pageScores_matchCPUReference() throws {
        let numQHeads = 24, numKVHeads = 4, headDim = 256
        let elems = numKVHeads * headDim
        let numPages = 12
        var rng = SeedTree(0x5C0).key("scores")
        let q = (0..<numQHeads * headDim).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let metadata = (0..<numPages * 2 * elems).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

        let ctx = try MetalContext()
        let kernels = try KVPageKernels(context: ctx)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q),
              let metaBuf = Fp16Buffer.make(ctx.device, halves: metadata),
              let scoreBuf = ctx.device.makeBuffer(length: numPages * 4,
                                                   options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernels.encodePageScores(commandBuffer: cb,
                                 q: qBuf,
                                 metadata: metaBuf, metadataOffset: 0,
                                 scores: scoreBuf, scoresOffset: 0,
                                 numPages: UInt32(numPages),
                                 headDim: UInt32(headDim),
                                 numQHeads: UInt32(numQHeads),
                                 numKVHeads: UInt32(numKVHeads))
        cb.commit(); cb.waitUntilCompleted()

        let gpu = scoreBuf.contents().assumingMemoryBound(to: Float.self)
        let qPerKV = numQHeads / numKVHeads
        for page in 0..<numPages {
            let mnBase = page * 2 * elems
            let mxBase = mnBase + elems
            var best = -Float.infinity
            for qh in 0..<numQHeads {
                let kvh = qh / qPerKV
                var s: Float = 0
                for i in 0..<headDim {
                    let qv = Float(q[qh * headDim + i])
                    let lo = Float(metadata[mnBase + kvh * headDim + i])
                    let hi = Float(metadata[mxBase + kvh * headDim + i])
                    s += max(qv * lo, qv * hi)
                }
                best = max(best, s)
            }
            let rel = abs(gpu[page] - best) / max(1.0, abs(best))
            #expect(rel < 1e-3, "page \(page): gpu \(gpu[page]) vs cpu \(best)")
        }
    }

    // MARK: selector policy

    @Test func selector_unionsSinksRecentsAndTopK_ascending() {
        let selector = KVPageSelector(sinkPages: 2, recentPages: 2, topKPages: 3)
        // 10 sealed pages + a 20-token tail. Scores favor pages 5, 3, 8.
        var scores = [Float](repeating: 0, count: 10)
        scores[5] = 9; scores[3] = 8; scores[8] = 7
        let sel = selector.select(scores: scores, sealedPages: 10, tailValidTokens: 20)
        // sinks {0,1}; recent {9,10}; topk {5,3,8}.
        #expect(sel.pages == [0, 1, 3, 5, 8, 9, 10])
        #expect(sel.selTokens == 6 * 64 + 20)
    }

    @Test func selector_tieBreaksByLowerIndex_deterministically() {
        let selector = KVPageSelector(sinkPages: 0, recentPages: 1, topKPages: 2)
        let scores: [Float] = [5, 5, 5, 5]
        let sel = selector.select(scores: scores, sealedPages: 4, tailValidTokens: 0)
        // Recent pins page 3; top-2 of the tied rest is {0, 1}.
        #expect(sel.pages == [0, 1, 3])
        #expect(sel.selTokens == 3 * 64)
    }

    @Test func selector_warmup_noScores_fillsWithTrailingPages() {
        let selector = KVPageSelector(sinkPages: 2, recentPages: 2, topKPages: 3)
        let sel = selector.select(scores: [], sealedPages: 10, tailValidTokens: 5)
        // sinks {0,1}; recent {9,10}; trailing fill {8,7,6}.
        #expect(sel.pages == [0, 1, 6, 7, 8, 9, 10])
        #expect(sel.selTokens == 6 * 64 + 5)
    }

    @Test func selector_smallContexts_selectEverything() {
        let selector = KVPageSelector(sinkPages: 2, recentPages: 4, topKPages: 60)
        let sel = selector.select(scores: [1, 2], sealedPages: 2, tailValidTokens: 30)
        #expect(sel.pages == [0, 1, 2])
        #expect(sel.selTokens == 2 * 64 + 30)

        let empty = selector.select(scores: [], sealedPages: 0, tailValidTokens: 0)
        #expect(empty.pages.isEmpty && empty.selTokens == 0)
    }

    @Test func selector_coversEntireContext_flipsAtTheBudgetBoundary() {
        let selector = KVPageSelector(sinkPages: 2, recentPages: 4, topKPages: 60)
        // 66 pages = sinks(2) + recent(4) + topk(60) exactly.
        #expect(selector.coversEntireContext(totalPages: 66, maxUnscoredSealedPages: 1))
        #expect(!selector.coversEntireContext(totalPages: 67, maxUnscoredSealedPages: 1))
        #expect(selector.coversEntireContext(totalPages: 1, maxUnscoredSealedPages: 1))
    }

    @Test func selector_coversEntireContext_agreesWithSelect() {
        let selector = KVPageSelector(sinkPages: 2, recentPages: 4, topKPages: 60)
        for totalPages in [1, 6, 40, 65, 66, 67, 80] {
            let sealed = totalPages - 1
            let sel = selector.select(scores: (0..<sealed).map { Float($0) },
                                      sealedPages: sealed, tailValidTokens: 10)
            let covered = sel.pages.count == totalPages
            #expect(selector.coversEntireContext(totalPages: totalPages,
                                                 maxUnscoredSealedPages: 1) == covered,
                    "totalPages \(totalPages)")
        }
    }

    @Test func selector_coversEntireContext_requiresRecentOverUnscoredPages() {
        // recent(1) covers only the tail, so an unscored just-sealed page
        // would fall in the gap — coverage cannot be guaranteed.
        let narrow = KVPageSelector(sinkPages: 0, recentPages: 1, topKPages: 10)
        #expect(!narrow.coversEntireContext(totalPages: 5, maxUnscoredSealedPages: 1))
        let wide = KVPageSelector(sinkPages: 0, recentPages: 2, topKPages: 10)
        #expect(wide.coversEntireContext(totalPages: 5, maxUnscoredSealedPages: 1))
        #expect(!wide.coversEntireContext(totalPages: 5, maxUnscoredSealedPages: 2))
    }

    @Test func selector_boundaryPosition_hasNoTailPage() {
        let selector = KVPageSelector(sinkPages: 1, recentPages: 2, topKPages: 0)
        let sel = selector.select(scores: [1, 2, 3, 4], sealedPages: 4, tailValidTokens: 0)
        #expect(sel.pages == [0, 2, 3])
        #expect(sel.selTokens == 3 * 64)
    }
}
