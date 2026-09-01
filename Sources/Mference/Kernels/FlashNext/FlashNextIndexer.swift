import Foundation
import Metal

/// The Flash-Next QSA indexer (`Qwen4ExpTextQSAIndexer`): per-query block
/// selection over the visible prefix, and the gather that hands gated full
/// attention the KV subset it selected.
///
/// # What is on the GPU and what is not
///
/// The projection, the query heads, the pooled block keys and the scores are GPU
/// work. The **ranking is not**: scores come back as at most a few thousand FP32
/// values per query and the top-k runs through `FlashNextDescendingTopK`, the
/// exact `torch.topk` CPU ordering the reference uses. Selection is a support
/// set, not a tensor — a flipped boundary changes which KV a layer may read —
/// and relu-zero ties at the boundary are common enough that no GPU sort with a
/// different tie rule would be safe. The readback is small next to the attention
/// it gates.
///
/// # Caches
///
/// Two per attention layer, both FP32 (see `flashnext_indexer.metal` for why the
/// rest of the runtime's FP16 convention is deliberately broken here):
///
/// * `rawKeys` — append-only, one `headDim` row per token. Un-normed, un-roped.
/// * `blockKeys` — one row per **complete** block of `compressRatio` tokens,
///   written once when the block's last token lands and immutable thereafter.
///   That immutability is the whole point: a block completed at token 40 scores
///   identically for every later query, so decode pools one new row every four
///   tokens instead of re-pooling the prefix.
///
/// # Prefill
///
/// Selection is computed **per position**, never lagged. `encodeScores` writes a
/// `[rows, scoreStride]` grid in one dispatch, with each row masked to its own
/// visible block count, so a chunked prefill reproduces the reference's
/// per-position selection exactly. The paged-KV Quest path's lag-one policy is
/// explicitly not reused here — the design doc rules it out.
final class FlashNextIndexer {

    struct Geometry {
        let numHeads: Int
        let numKVHeads: Int
        let headDim: Int
        let compressRatio: Int
        /// `indexer_budget / compress_ratio` — how many complete blocks survive.
        let blockBudget: Int
        /// Shared with full attention: `fullHeadDim * partialRotaryFactor`.
        let rotaryDim: Int
        let theta: Float
        let eps: Float

        /// Rows of `index_qk_proj`: the query heads then the single key head.
        var projRows: Int { (numHeads + numKVHeads) * headDim }
        /// Channel offset of the raw key head inside a projection row.
        var keyOffset: Int { numHeads * headDim }
    }

    /// Per-forward scratch, sized for the widest chunk and context the runner
    /// will drive. Owned by the caller so it is allocated once.
    struct Scratch {
        /// `[rows, projRows]` FP32 — the whole `index_qk_proj` output.
        let projection: MTLBuffer
        /// `[rows, numHeads, headDim]` FP32 — normed and roped query heads.
        let queries: MTLBuffer
        /// `[rows, scoreStride]` FP32, **shared** so the CPU can rank it.
        let scores: MTLBuffer
        /// `[rows, selectionStride]` UInt32, shared. Row-indexed rather than a
        /// single list because the gather reads it at GPU execution time: a
        /// prefill wave that overwrote one slot per row before committing would
        /// gather the last row's selection for every row.
        let selection: MTLBuffer
        let maxRows: Int
        let scoreStride: Int
        /// `blockBudget * compressRatio + compressRatio` — the widest selection
        /// any query can produce. Above the budget a query takes exactly
        /// `blockBudget` blocks plus at most `compressRatio - 1` tail tokens;
        /// below it, everything visible, which is smaller still.
        let selectionStride: Int
    }

    /// One attention layer's indexer state.
    struct LayerCache {
        let rawKeys: MTLBuffer      // [maxTokens, headDim] FP32
        let blockKeys: MTLBuffer    // [maxTokens / ratio, headDim] FP32
        let maxTokens: Int
    }

    /// Largest indexer head dim the kernels' thread-local scratch supports.
    /// Mirrors `kFlashNextIndexerMaxHeadDim` in `flashnext_indexer.metal`.
    static let maxHeadDim = 128

    let geometry: Geometry
    private let matVec: FlashNextMatVec
    private let preparePSO: MTLComputePipelineState
    private let appendPSO: MTLComputePipelineState
    private let poolPSO: MTLComputePipelineState
    private let scoresPSO: MTLComputePipelineState
    private let gatherPSO: MTLComputePipelineState

    init(context: MetalContext, matVec: FlashNextMatVec,
         geometry: Geometry) throws {
        precondition(geometry.headDim <= Self.maxHeadDim,
                     "indexer head dim \(geometry.headDim) exceeds the kernels' "
                     + "\(Self.maxHeadDim)-wide thread scratch")
        precondition(geometry.numKVHeads == 1,
                     "the indexer has exactly one key head")
        precondition(geometry.compressRatio > 0 && geometry.blockBudget > 0)
        precondition(geometry.rotaryDim.isMultiple(of: 2))
        precondition(geometry.rotaryDim <= geometry.headDim)
        self.geometry = geometry
        self.matVec = matVec
        self.preparePSO = try context.pipeline("flashnext_indexer_prepare_queries")
        self.appendPSO = try context.pipeline("flashnext_indexer_append_raw_keys")
        self.poolPSO = try context.pipeline("flashnext_indexer_pool_block_keys")
        self.scoresPSO = try context.pipeline("flashnext_indexer_scores")
        self.gatherPSO = try context.pipeline("flashnext_indexer_gather_kv")
    }

    // MARK: - Allocation

    func makeScratch(device: MTLDevice, rows: Int, maxTokens: Int) throws -> Scratch {
        let stride = maxTokens / geometry.compressRatio + 1
        func buffer(_ elements: Int, _ byteStride: Int,
                    _ mode: MTLResourceOptions) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(1, elements) * byteStride,
                                            options: mode) else {
                throw MetalError.noDevice
            }
            return b
        }
        let float = MemoryLayout<Float>.stride
        let selectionStride = maxSelected
        return Scratch(
            projection: try buffer(rows * geometry.projRows, float, .storageModePrivate),
            queries: try buffer(rows * geometry.numHeads * geometry.headDim,
                                float, .storageModePrivate),
            scores: try buffer(rows * stride, float, .storageModeShared),
            selection: try buffer(rows * selectionStride,
                                  MemoryLayout<UInt32>.stride, .storageModeShared),
            maxRows: rows,
            scoreStride: stride,
            selectionStride: selectionStride)
    }

    /// The widest selection any query can produce.
    var maxSelected: Int {
        geometry.blockBudget * geometry.compressRatio + geometry.compressRatio
    }

    func makeLayerCache(device: MTLDevice, maxTokens: Int) throws -> LayerCache {
        let float = MemoryLayout<Float>.stride
        let blocks = maxTokens / geometry.compressRatio + 1
        guard let raw = device.makeBuffer(
                  length: max(1, maxTokens * geometry.headDim) * float,
                  options: .storageModePrivate),
              let pooled = device.makeBuffer(
                  length: max(1, blocks * geometry.headDim) * float,
                  options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        raw.label = "flashnext.indexer.rawKeys"
        pooled.label = "flashnext.indexer.blockKeys"
        return LayerCache(rawKeys: raw, blockKeys: pooled, maxTokens: maxTokens)
    }

    // MARK: - Encode

    /// `proj = index_qk_proj . x`, one row at a time, straight into FP32.
    ///
    /// PERF, not correctness: one compute encoder per row. Decode is one row;
    /// a wide prefill chunk pays a dispatch per token per attention layer. The
    /// fix is a batched mat-vec (rows on the grid's second axis) and it belongs
    /// with the perf pass — the same note `FlashNextHyperConnections.encodeMix`
    /// carries for the identical reason.
    func encodeProjection(commandBuffer: MTLCommandBuffer,
                          weight: FlashNextWeightMatrix,
                          x: MTLBuffer, xOffset: Int,
                          hidden: Int,
                          scratch: Scratch,
                          rows: Int) {
        precondition(rows <= scratch.maxRows)
        for row in 0..<rows {
            matVec.encode(commandBuffer: commandBuffer,
                          matrix: weight,
                          x: x,
                          xOffset: xOffset + row * hidden * MemoryLayout<Float16>.stride,
                          y: scratch.projection,
                          yOffset: row * geometry.projRows * MemoryLayout<Float>.stride,
                          rows: geometry.projRows, cols: hidden,
                          outputFloat32: true)
        }
    }

    /// Query heads (norm + RoPE at each row's own position), the raw-key append,
    /// and the pooled block keys for whichever blocks this call completed.
    func encodePrepare(commandBuffer: MTLCommandBuffer,
                       qNorm: MTLBuffer, qNormOffset: Int,
                       kNorm: MTLBuffer, kNormOffset: Int,
                       scratch: Scratch,
                       cache: LayerCache,
                       rows: Int,
                       startPosition: Int) {
        precondition(rows > 0 && rows <= scratch.maxRows)
        precondition(startPosition + rows <= cache.maxTokens,
                     "indexer raw-key cache holds \(cache.maxTokens) tokens")
        var heads = UInt32(geometry.numHeads)
        var headDim = UInt32(geometry.headDim)
        var projRows = UInt32(geometry.projRows)
        var start = UInt32(startPosition)
        var rotaryDim = UInt32(geometry.rotaryDim)
        var theta = geometry.theta
        var eps = geometry.eps
        var rowCount = UInt32(rows)

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(preparePSO)
            enc.setBuffer(scratch.projection, offset: 0, index: 0)
            enc.setBuffer(qNorm, offset: qNormOffset, index: 1)
            enc.setBuffer(scratch.queries, offset: 0, index: 2)
            enc.setBytes(&heads, length: 4, index: 3)
            enc.setBytes(&headDim, length: 4, index: 4)
            enc.setBytes(&projRows, length: 4, index: 5)
            enc.setBytes(&start, length: 4, index: 6)
            enc.setBytes(&rotaryDim, length: 4, index: 7)
            enc.setBytes(&theta, length: 4, index: 8)
            enc.setBytes(&eps, length: 4, index: 9)
            enc.setBytes(&rowCount, length: 4, index: 10)
            dispatch1D(enc, pso: preparePSO, count: rows * geometry.numHeads)
            enc.endEncoding()
        }

        var keyOffset = UInt32(geometry.keyOffset)
        var seen = UInt32(startPosition)
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(appendPSO)
            enc.setBuffer(scratch.projection, offset: 0, index: 0)
            enc.setBuffer(cache.rawKeys, offset: 0, index: 1)
            enc.setBytes(&headDim, length: 4, index: 2)
            enc.setBytes(&projRows, length: 4, index: 3)
            enc.setBytes(&keyOffset, length: 4, index: 4)
            enc.setBytes(&seen, length: 4, index: 5)
            enc.setBytes(&rowCount, length: 4, index: 6)
            dispatch2D(enc, pso: appendPSO, width: geometry.headDim, height: rows)
            enc.endEncoding()
        }

        // Blocks this call completed: everything between the block counts before
        // and after the append. Immutable once written, so never recomputed.
        let before = startPosition / geometry.compressRatio
        let after = (startPosition + rows) / geometry.compressRatio
        guard after > before else { return }
        var ratio = UInt32(geometry.compressRatio)
        var firstBlock = UInt32(before)
        var blockCount = UInt32(after - before)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(poolPSO)
        enc.setBuffer(cache.rawKeys, offset: 0, index: 0)
        enc.setBuffer(kNorm, offset: kNormOffset, index: 1)
        enc.setBuffer(cache.blockKeys, offset: 0, index: 2)
        enc.setBytes(&headDim, length: 4, index: 3)
        enc.setBytes(&ratio, length: 4, index: 4)
        enc.setBytes(&firstBlock, length: 4, index: 5)
        enc.setBytes(&blockCount, length: 4, index: 6)
        enc.setBytes(&rotaryDim, length: 4, index: 7)
        enc.setBytes(&theta, length: 4, index: 8)
        enc.setBytes(&eps, length: 4, index: 9)
        dispatch1D(enc, pso: poolPSO, count: after - before)
        enc.endEncoding()
    }

    /// `[rows, scoreStride]` FP32 block scores, each row masked to its own
    /// visible block count. No lag: row `r` scores exactly the blocks complete at
    /// absolute position `startPosition + r`.
    func encodeScores(commandBuffer: MTLCommandBuffer,
                      scratch: Scratch,
                      cache: LayerCache,
                      rows: Int,
                      startPosition: Int) {
        let visibleBlocks = (startPosition + rows) / geometry.compressRatio
        guard visibleBlocks > 0 else { return }
        var heads = UInt32(geometry.numHeads)
        var headDim = UInt32(geometry.headDim)
        var stride = UInt32(scratch.scoreStride)
        var start = UInt32(startPosition)
        var ratio = UInt32(geometry.compressRatio)
        var rowCount = UInt32(rows)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(scoresPSO)
        enc.setBuffer(scratch.queries, offset: 0, index: 0)
        enc.setBuffer(cache.blockKeys, offset: 0, index: 1)
        enc.setBuffer(scratch.scores, offset: 0, index: 2)
        enc.setBytes(&heads, length: 4, index: 3)
        enc.setBytes(&headDim, length: 4, index: 4)
        enc.setBytes(&stride, length: 4, index: 5)
        enc.setBytes(&start, length: 4, index: 6)
        enc.setBytes(&ratio, length: 4, index: 7)
        enc.setBytes(&rowCount, length: 4, index: 8)
        dispatch2D(enc, pso: scoresPSO, width: visibleBlocks, height: rows)
        enc.endEncoding()
    }

    /// Rank the scores this call wrote and return, per row, the sorted absolute
    /// KV positions attention may read.
    ///
    /// The incomplete tail — including the query's own token — is **always**
    /// selected. There is no "keep self" rule beyond that: a query whose own
    /// block is complete and loses the top-k does not attend to itself.
    ///
    /// The command buffer that wrote `scratch.scores` must have completed.
    func selections(scratch: Scratch, rows: Int, startPosition: Int) -> [[Int]] {
        let ratio = geometry.compressRatio
        let base = scratch.scores.contents()
            .bindMemory(to: Float.self, capacity: rows * scratch.scoreStride)
        var out: [[Int]] = []
        out.reserveCapacity(rows)
        for row in 0..<rows {
            let visible = startPosition + row + 1
            let complete = visible / ratio
            var chosen: [Int] = []
            if complete > 0 {
                let offset = row * scratch.scoreStride
                let scores = (0..<complete).map { base[offset + $0] }
                let k = min(geometry.blockBudget, complete)
                for block in FlashNextDescendingTopK.indices(scores, k: k) {
                    for j in 0..<ratio { chosen.append(block * ratio + j) }
                }
            }
            for i in (complete * ratio)..<visible { chosen.append(i) }
            out.append(chosen.sorted())
        }
        return out
    }

    /// Whether a row's selection boundary is a bit-exact tie, i.e. decided by the
    /// ordering policy rather than by the model. Used to attribute a parity
    /// failure rather than guess at one.
    func boundaryIsTied(scratch: Scratch, row: Int, startPosition: Int) -> Bool {
        let visible = startPosition + row + 1
        let complete = visible / geometry.compressRatio
        guard complete > geometry.blockBudget else { return false }
        let base = scratch.scores.contents().bindMemory(
            to: Float.self, capacity: (row + 1) * scratch.scoreStride)
        let offset = row * scratch.scoreStride
        let scores = (0..<complete).map { base[offset + $0] }
        return FlashNextDescendingTopK.boundaryIsTied(scores, k: geometry.blockBudget)
    }

    /// Write one row's selection into that row's slot of the shared index
    /// buffer, ready for the gather. Returns the count the gather and the
    /// attention dispatch need.
    @discardableResult
    func writeSelection(_ positions: [Int], row: Int, into scratch: Scratch) -> Int {
        precondition(row < scratch.maxRows)
        precondition(positions.count <= scratch.selectionStride,
                     "selection of \(positions.count) exceeds the "
                     + "\(scratch.selectionStride)-slot row")
        let capacity = scratch.maxRows * scratch.selectionStride
        let base = scratch.selection.contents()
            .bindMemory(to: UInt32.self, capacity: capacity)
        let offset = row * scratch.selectionStride
        for (i, p) in positions.enumerated() { base[offset + i] = UInt32(p) }
        return positions.count
    }

    /// Copy the selected K and V rows into contiguous scratch, so the shipped
    /// dense decode attention kernel can run over the subset unchanged. Every
    /// gathered position is at or before the query, so no causal mask is needed
    /// inside the gathered run.
    func encodeGatherKV(commandBuffer: MTLCommandBuffer,
                        kCache: MTLBuffer, kCacheOffset: Int,
                        vCache: MTLBuffer, vCacheOffset: Int,
                        scratch: Scratch,
                        selectionRow: Int,
                        kOut: MTLBuffer, kOutOffset: Int,
                        vOut: MTLBuffer, vOutOffset: Int,
                        kvDim: Int, count: Int) {
        guard count > 0, let enc = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        var kvDimVar = UInt32(kvDim)
        var countVar = UInt32(count)
        enc.setComputePipelineState(gatherPSO)
        enc.setBuffer(kCache, offset: kCacheOffset, index: 0)
        enc.setBuffer(vCache, offset: vCacheOffset, index: 1)
        enc.setBuffer(scratch.selection,
                      offset: selectionRow * scratch.selectionStride
                          * MemoryLayout<UInt32>.stride,
                      index: 2)
        enc.setBuffer(kOut, offset: kOutOffset, index: 3)
        enc.setBuffer(vOut, offset: vOutOffset, index: 4)
        enc.setBytes(&kvDimVar, length: 4, index: 5)
        enc.setBytes(&countVar, length: 4, index: 6)
        dispatch2D(enc, pso: gatherPSO, width: kvDim, height: count)
        enc.endEncoding()
    }

    // MARK: - Dispatch helpers

    private func dispatch1D(_ enc: MTLComputeCommandEncoder,
                            pso: MTLComputePipelineState, count: Int) {
        guard count > 0 else { return }
        let width = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(width, count),
                                                           height: 1, depth: 1))
    }

    private func dispatch2D(_ enc: MTLComputeCommandEncoder,
                            pso: MTLComputePipelineState,
                            width: Int, height: Int) {
        guard width > 0, height > 0 else { return }
        let w = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(w, width),
                                                           height: 1, depth: 1))
    }
}
