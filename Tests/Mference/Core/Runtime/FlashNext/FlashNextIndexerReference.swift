import Foundation
@testable import Mference

/// The Flash-Next QSA indexer (`Qwen4ExpTextQSAIndexer`) in float32, staged so
/// each Metal kernel has an oracle for exactly what it computes.
///
/// This is a transcription of `FlashNextReferenceRunner.indexerSelect`, not a
/// re-derivation. The runner is oracle code and is not modified to share it;
/// `FlashNextIndexerReferenceTieBackTests` discharges the transcription by
/// replaying the runner's captured `attn_hc_mixed` through this type and
/// checking the result against its captured `indexer_selected`.
///
/// Ordering notes that are load-bearing and easy to lose:
///
/// * The query head is normed **then** roped, at the query's own absolute
///   position. The pooled block key is meaned **then** normed **then** roped, at
///   the block's **first** position.
/// * The raw key that goes into the cache is neither normed nor roped.
/// * The relu is per head, before the sum over heads; the `/sqrt(headDim)`
///   comes after the sum.
/// * The incomplete tail is always selected. There is no self-guarantee.
enum FlashNextIndexerReference {

    struct Geometry {
        let numHeads: Int
        let numKVHeads: Int
        let headDim: Int
        let compressRatio: Int
        let blockBudget: Int
        let rotaryDim: Int
        let theta: Float
        let eps: Float
        var projRows: Int { (numHeads + numKVHeads) * headDim }
        var keyOffset: Int { numHeads * headDim }
    }

    // MARK: - Stages

    /// `proj = index_qk_proj . x` for one token.
    static func projection(_ weight: [Float], x: [Float], xOffset: Int,
                           hidden: Int, g: Geometry) -> [Float] {
        var out = [Float](repeating: 0, count: g.projRows)
        for r in 0..<g.projRows {
            var acc: Float = 0
            let base = r * hidden
            for c in 0..<hidden { acc += weight[base + c] * x[xOffset + c] }
            out[r] = acc
        }
        return out
    }

    /// The `numHeads` normed-and-roped query heads of one projection row,
    /// concatenated — what `flashnext_indexer_prepare_queries` writes.
    static func preparedQueries(projection proj: [Float], qNorm: [Float],
                                position: Int, g: Geometry) -> [Float] {
        var out = [Float](repeating: 0, count: g.numHeads * g.headDim)
        for h in 0..<g.numHeads {
            var head = rmsNorm(proj, offset: h * g.headDim,
                               count: g.headDim, weight: qNorm, eps: g.eps)
            applyRoPE(&head, position: position,
                      rotaryDim: g.rotaryDim, theta: g.theta)
            for d in 0..<g.headDim { out[h * g.headDim + d] = head[d] }
        }
        return out
    }

    /// The raw key of one projection row — un-normed, un-roped.
    static func rawKey(projection proj: [Float], g: Geometry) -> [Float] {
        Array(proj[g.keyOffset..<(g.keyOffset + g.headDim)])
    }

    /// `rope_at(block * ratio, rmsnorm(mean_fp32(raw keys of the block)))` —
    /// what `flashnext_indexer_pool_block_keys` writes, and what the cache holds
    /// unchanged for the rest of the sequence.
    static func blockKey(rawKeys: [Float], block: Int, kNorm: [Float],
                         g: Geometry) -> [Float] {
        let start = block * g.compressRatio
        var pooled = [Float](repeating: 0, count: g.headDim)
        for d in 0..<g.headDim {
            var acc: Float = 0
            for j in 0..<g.compressRatio {
                acc += rawKeys[(start + j) * g.headDim + d]
            }
            pooled[d] = acc / Float(g.compressRatio)
        }
        var normed = rmsNorm(pooled, offset: 0, count: g.headDim,
                             weight: kNorm, eps: g.eps)
        applyRoPE(&normed, position: start, rotaryDim: g.rotaryDim, theta: g.theta)
        return normed
    }

    /// `sum_h relu(q_h . k_blk) / sqrt(headDim)` for one query over
    /// `completeBlocks` blocks.
    static func scores(queries: [Float], blockKeys: [Float],
                       completeBlocks: Int, g: Geometry) -> [Float] {
        var out = [Float](repeating: 0, count: completeBlocks)
        for b in 0..<completeBlocks {
            var score: Float = 0
            for h in 0..<g.numHeads {
                var dot: Float = 0
                for d in 0..<g.headDim {
                    dot += queries[h * g.headDim + d] * blockKeys[b * g.headDim + d]
                }
                if dot > 0 { score += dot }
            }
            out[b] = score / sqrtf(Float(g.headDim))
        }
        return out
    }

    /// The sorted absolute KV positions a query at `query` may read.
    static func select(scores: [Float], query: Int, g: Geometry) -> [Int] {
        let visible = query + 1
        let completeBlocks = visible / g.compressRatio
        var chosen: [Int] = []
        if completeBlocks > 0 {
            let k = min(g.blockBudget, completeBlocks)
            for block in FlashNextDescendingTopK.indices(scores, k: k) {
                for j in 0..<g.compressRatio {
                    chosen.append(block * g.compressRatio + j)
                }
            }
        }
        for i in (completeBlocks * g.compressRatio)..<visible { chosen.append(i) }
        return chosen.sorted()
    }

    // MARK: - Whole-layer driver

    /// Everything one attention layer's indexer does for a run of `rows` tokens
    /// starting at `startPosition`, with `rawKeys` carried in and out so cached
    /// decode and full prefill go through the same code.
    struct Result {
        let selected: [[Int]]
        /// The FP32 raw-key cache after the append, `[.., headDim]`.
        let rawKeys: [Float]
        /// The pooled block keys for every complete block, `[.., headDim]`.
        let blockKeys: [Float]
        /// Per row, that row's block scores over its own visible blocks.
        let scores: [[Float]]
        /// Per row, the normed-and-roped query heads.
        let queries: [[Float]]
    }

    static func run(x: [Float], hidden: Int, rows: Int, startPosition: Int,
                    indexerQK: [Float], qNorm: [Float], kNorm: [Float],
                    rawKeys carried: [Float], g: Geometry) -> Result {
        var rawKeys = carried
        var queries: [[Float]] = []
        for t in 0..<rows {
            let proj = projection(indexerQK, x: x, xOffset: t * hidden,
                                  hidden: hidden, g: g)
            queries.append(preparedQueries(projection: proj,
                                           qNorm: qNorm,
                                           position: startPosition + t, g: g))
            rawKeys.append(contentsOf: rawKey(projection: proj, g: g))
        }
        let totalBlocks = (startPosition + rows) / g.compressRatio
        var blockKeys = [Float]()
        blockKeys.reserveCapacity(totalBlocks * g.headDim)
        for b in 0..<totalBlocks {
            blockKeys.append(contentsOf: blockKey(rawKeys: rawKeys, block: b,
                                                  kNorm: kNorm, g: g))
        }
        var selected: [[Int]] = []
        var allScores: [[Float]] = []
        for t in 0..<rows {
            let query = startPosition + t
            let complete = (query + 1) / g.compressRatio
            let s = scores(queries: queries[t], blockKeys: blockKeys,
                           completeBlocks: complete, g: g)
            allScores.append(s)
            selected.append(select(scores: s, query: query, g: g))
        }
        return Result(selected: selected, rawKeys: rawKeys, blockKeys: blockKeys,
                      scores: allScores, queries: queries)
    }

    // MARK: - Numeric helpers (the runner's exact forms)

    static func rmsNorm(_ x: [Float], offset: Int, count: Int,
                        weight: [Float], eps: Float) -> [Float] {
        var ms: Float = 0
        for d in 0..<count { ms += x[offset + d] * x[offset + d] }
        let inv = 1 / sqrtf(ms / Float(count) + eps)
        return (0..<count).map { x[offset + $0] * inv * weight[$0] }
    }

    /// Partial NeoX RoPE, pairing `(i, i + rotaryDim/2)`. The inverse frequency
    /// is written as a reciprocal of a positive power, matching both the runner
    /// and `flashnext_indexer_inv_freq`.
    static func applyRoPE(_ head: inout [Float], position: Int,
                          rotaryDim: Int, theta: Float) {
        let half = rotaryDim / 2
        for i in 0..<half {
            let inv = 1 / powf(theta, Float(2 * i) / Float(rotaryDim))
            let angle = Float(position) * inv
            let c = cosf(angle), s = sinf(angle)
            let a = head[i], b = head[i + half]
            head[i] = a * c - b * s
            head[i + half] = b * c + a * s
        }
    }
}
