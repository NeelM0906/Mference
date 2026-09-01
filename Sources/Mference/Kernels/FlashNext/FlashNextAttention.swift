import Foundation
import Metal

/// Flash-Next gated full attention (`Qwen4ExpTextAttention`) over the KV subset
/// the QSA indexer selected.
///
/// The math is `Qwen3_5Attention` — gated `q_proj`, per-head q/k RMSNorm, partial
/// NeoX RoPE, GQA, sigmoid output gate — so the shipped Qwen 3.8 kernels do the
/// work: `split_q_gate_fp16`, `PrefillQKVEpilogue`'s fused per-head norm + RoPE
/// (which, unlike the plain `rope_neox_subdim`, advances the position per row and
/// so serves prefill and decode alike), the dense `attention_full` decode kernel,
/// and `sigmoid_gate_mul_fp16`.
///
/// # How the sparsity is applied
///
/// Not by masking and not by dropping KV. The full KV cache is written for every
/// token; the selected positions are **gathered into contiguous scratch** and the
/// dense kernel runs over that short run. The design doc is explicit that
/// selection sparsity never removes KV, and gathering keeps the attention kernel
/// itself unmodified: every gathered position is at or before the query, so the
/// gathered run needs no causal mask.
///
/// The cost is bounded by construction — a selection is at most
/// `indexer_budget + compress_ratio` positions (2051 in production) whatever the
/// context length, which is the whole point of the indexer. Prefill gathers in
/// waves of `gatherSlots` rows so one command buffer can cover several queries
/// without each overwriting the previous one's scratch.
final class FlashNextAttention {

    struct Geometry {
        let hidden: Int
        let numHeads: Int
        let numKVHeads: Int
        let headDim: Int
        let rotaryDim: Int
        let theta: Float
        let eps: Float
        let scale: Float
        var qDim: Int { numHeads * headDim }
        var kvDim: Int { numKVHeads * headDim }
    }

    struct Weights {
        /// `[2 * qDim, hidden]` — packed **per head** as `[query | gate]`.
        let q: FlashNextWeightMatrix
        let k: FlashNextWeightMatrix     // [kvDim, hidden]
        let v: FlashNextWeightMatrix     // [kvDim, hidden]
        let o: FlashNextWeightMatrix     // [hidden, qDim]
        let qNorm: MTLBuffer
        let qNormOffset: Int
        let kNorm: MTLBuffer
        let kNormOffset: Int
    }

    struct Scratch {
        let packed: MTLBuffer       // [rows, 2 * qDim] FP16
        let queries: MTLBuffer      // [rows, qDim] FP16
        let gates: MTLBuffer        // [rows, qDim] FP16
        let attnOut: MTLBuffer      // [rows, qDim] FP16
        let gatheredK: MTLBuffer    // [gatherSlots, maxSelected, kvDim] FP16
        let gatheredV: MTLBuffer
        let maxRows: Int
        let gatherSlots: Int
        let maxSelected: Int
    }

    /// The dense KV this family always keeps: 12 layers x 2 heads x 256, the
    /// 24 KiB/token the dossier prices. Selection never touches it.
    struct KVCache {
        let keys: MTLBuffer         // [maxTokens, kvDim] FP16
        let values: MTLBuffer       // [maxTokens, kvDim] FP16
        let maxTokens: Int
    }

    let geometry: Geometry
    private let matVec: FlashNextMatVec
    private let elementwise: Elementwise
    private let epilogue: PrefillQKVEpilogue
    private let attention: Attention

    init(context: MetalContext, matVec: FlashNextMatVec,
         elementwise: Elementwise, epilogue: PrefillQKVEpilogue,
         attention: Attention, geometry: Geometry) {
        precondition(geometry.numHeads % geometry.numKVHeads == 0)
        precondition(geometry.rotaryDim.isMultiple(of: 2))
        precondition(geometry.rotaryDim <= geometry.headDim)
        self.geometry = geometry
        self.matVec = matVec
        self.elementwise = elementwise
        self.epilogue = epilogue
        self.attention = attention
    }

    // MARK: - Allocation

    func makeScratch(device: MTLDevice, rows: Int, maxSelected: Int,
                     gatherSlots: Int) throws -> Scratch {
        precondition(rows > 0 && gatherSlots > 0 && maxSelected > 0)
        let half = MemoryLayout<Float16>.stride
        func buffer(_ elements: Int, _ mode: MTLResourceOptions) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(1, elements) * half,
                                            options: mode) else {
                throw MetalError.noDevice
            }
            return b
        }
        return Scratch(
            packed: try buffer(rows * 2 * geometry.qDim, .storageModePrivate),
            queries: try buffer(rows * geometry.qDim, .storageModePrivate),
            gates: try buffer(rows * geometry.qDim, .storageModePrivate),
            attnOut: try buffer(rows * geometry.qDim, .storageModePrivate),
            gatheredK: try buffer(gatherSlots * maxSelected * geometry.kvDim,
                                  .storageModePrivate),
            gatheredV: try buffer(gatherSlots * maxSelected * geometry.kvDim,
                                  .storageModePrivate),
            maxRows: rows,
            gatherSlots: gatherSlots,
            maxSelected: maxSelected)
    }

    func makeKVCache(device: MTLDevice, maxTokens: Int) throws -> KVCache {
        let half = MemoryLayout<Float16>.stride
        guard let k = device.makeBuffer(
                  length: max(1, maxTokens * geometry.kvDim) * half,
                  options: .storageModePrivate),
              let v = device.makeBuffer(
                  length: max(1, maxTokens * geometry.kvDim) * half,
                  options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        k.label = "flashnext.attn.k"
        v.label = "flashnext.attn.v"
        return KVCache(keys: k, values: v, maxTokens: maxTokens)
    }

    // MARK: - Encode

    /// Projections, KV append, per-head q/k norm and partial RoPE.
    ///
    /// K and V are written **straight into the cache** at the rows this call
    /// owns, then normed and roped in place there — the same in-cache norm the
    /// Qwen 3.8 decode path uses, which is why the cache holds post-norm,
    /// post-RoPE keys and raw values.
    ///
    /// PERF, not correctness: the three projections are per-row mat-vecs, one
    /// compute encoder each. Decode is one row; a wide prefill chunk wants the
    /// batched form, which belongs with the perf pass.
    func encodeProjectAndCache(commandBuffer: MTLCommandBuffer,
                               weights w: Weights,
                               scratch: Scratch,
                               cache: KVCache,
                               x: MTLBuffer, xOffset: Int,
                               rows: Int,
                               startPosition: Int) {
        precondition(rows > 0 && rows <= scratch.maxRows)
        precondition(startPosition + rows <= cache.maxTokens,
                     "KV cache holds \(cache.maxTokens) tokens")
        let half = MemoryLayout<Float16>.stride
        let qDim = geometry.qDim
        let kvDim = geometry.kvDim
        let kvBase = startPosition * kvDim * half

        for row in 0..<rows {
            let rowX = xOffset + row * geometry.hidden * half
            matVec.encode(commandBuffer: commandBuffer, matrix: w.q,
                          x: x, xOffset: rowX,
                          y: scratch.packed, yOffset: row * 2 * qDim * half,
                          rows: 2 * qDim, cols: geometry.hidden)
            matVec.encode(commandBuffer: commandBuffer, matrix: w.k,
                          x: x, xOffset: rowX,
                          y: cache.keys, yOffset: kvBase + row * kvDim * half,
                          rows: kvDim, cols: geometry.hidden)
            matVec.encode(commandBuffer: commandBuffer, matrix: w.v,
                          x: x, xOffset: rowX,
                          y: cache.values, yOffset: kvBase + row * kvDim * half,
                          rows: kvDim, cols: geometry.hidden)
        }

        elementwise.encodeSplitQGate(commandBuffer: commandBuffer,
                                     packed: scratch.packed,
                                     q: scratch.queries,
                                     gate: scratch.gates,
                                     heads: geometry.numHeads,
                                     dim: geometry.headDim,
                                     rows: rows)

        // Per-head RMSNorm on q and k, then partial NeoX RoPE at each row's own
        // absolute position. V is neither normed nor roped, which is what the
        // `NoVNorm` epilogue means.
        epilogue.encodeNeoxSubdimNoVNorm(
            commandBuffer: commandBuffer,
            q: scratch.queries, qOffset: 0,
            k: cache.keys, kOffset: kvBase,
            qWeight: w.qNorm, qWeightOffset: w.qNormOffset,
            kWeight: w.kNorm, kWeightOffset: w.kNormOffset,
            startPosition: UInt32(startPosition),
            queryCount: UInt32(rows),
            headDim: UInt32(geometry.headDim),
            numQHeads: UInt32(geometry.numHeads),
            numKVHeads: UInt32(geometry.numKVHeads),
            qTokenStrideElements: UInt32(qDim),
            kvTokenStrideElements: UInt32(kvDim),
            theta: geometry.theta,
            rotaryDim: UInt32(geometry.rotaryDim),
            eps: geometry.eps)
    }

    /// Attention for one query row over its own gathered KV run.
    ///
    /// `slot` selects which gather scratch slot this row uses, so a wave of rows
    /// can share one command buffer. The caller must have written the row's
    /// selection with `FlashNextIndexer.writeSelection` and gathered into the
    /// same slot.
    func encodeAttendRow(commandBuffer: MTLCommandBuffer,
                         scratch: Scratch,
                         row: Int, slot: Int, selectedCount: Int) {
        precondition(row < scratch.maxRows)
        precondition(slot < scratch.gatherSlots)
        precondition(selectedCount > 0 && selectedCount <= scratch.maxSelected)
        let half = MemoryLayout<Float16>.stride
        let runBytes = scratch.maxSelected * geometry.kvDim * half
        attention.encodeFull(
            commandBuffer: commandBuffer,
            q: scratch.queries, qOffset: row * geometry.qDim * half,
            k: scratch.gatheredK, kOffset: slot * runBytes,
            v: scratch.gatheredV, vOffset: slot * runBytes,
            out: scratch.attnOut, outOffset: row * geometry.qDim * half,
            headDim: UInt32(geometry.headDim),
            numQHeads: UInt32(geometry.numHeads),
            numKVHeads: UInt32(geometry.numKVHeads),
            seqLen: UInt32(selectedCount),
            scale: geometry.scale)
    }

    /// Byte offsets of one gather slot, for `FlashNextIndexer.encodeGatherKV`.
    func gatherSlotOffset(_ slot: Int, scratch: Scratch) -> Int {
        slot * scratch.maxSelected * geometry.kvDim * MemoryLayout<Float16>.stride
    }

    /// The sigmoid output gate and `o_proj`. The gate multiplies the flattened
    /// per-head attention output **before** the projection, and the gate half is
    /// never normed or rotated — it comes straight out of `q_proj`.
    func encodeGateAndProject(commandBuffer: MTLCommandBuffer,
                              weights w: Weights,
                              scratch: Scratch,
                              out: MTLBuffer, outOffset: Int,
                              rows: Int) {
        let half = MemoryLayout<Float16>.stride
        elementwise.encodeSigmoidGateMul(commandBuffer: commandBuffer,
                                         out: scratch.attnOut,
                                         gate: scratch.gates,
                                         count: rows * geometry.qDim)
        for row in 0..<rows {
            matVec.encode(commandBuffer: commandBuffer, matrix: w.o,
                          x: scratch.attnOut,
                          xOffset: row * geometry.qDim * half,
                          y: out, yOffset: outOffset + row * geometry.hidden * half,
                          rows: geometry.hidden, cols: geometry.qDim)
        }
    }
}
