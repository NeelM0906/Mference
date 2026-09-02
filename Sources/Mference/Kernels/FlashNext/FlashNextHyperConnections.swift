import Foundation
import Metal

/// `Qwen4ExpTextGatedResidual` — the low-rank hyper-connection that mixes the
/// `hc_count`-stream residual bundle down to one block input and injects the
/// block's output back into all the streams.
///
/// Semantics (design doc, "Hyper-connections"), for `hyper` in R^(hc*H):
///
/// ```
/// h_n   = group_rmsnorm_H(hyper) * hc_norm_w            // hc*H
/// m     = silu( W_down . h_n / hc )                     // lowRank
/// g     = sigmoid( W_up . m )                           // hc*H
/// mixed = mean over streams of (g * h_n)                // H
/// inj   = 2 * sigmoid( W_inject . h_n / hc )            // hc
/// hyper = hyper + flatten( block_out * inj )            // raw stream, not h_n
/// ```
///
/// The global `hyper_connection_mixer` is the same minus the inject path, which
/// is why `encodeMix` and `encodeInject` are separate entry points: the mixer
/// calls only the first.
///
/// # Precision
///
/// Activations are FP16 in memory, per the runtime's convention. Two values are
/// deliberately kept FP32 through the chain because they feed a sigmoid, where
/// FP16 rounding is amplified rather than absorbed: the 10240-wide pre-sigmoid
/// mix gate `W_up . m`, and the four injection scalars. The low-rank vector
/// rounds to FP16 once, on the way into the second GEMV, because the shipped
/// INT4 mat-vec takes a half activation.
final class FlashNextHyperConnections {

    /// Buffers this encoder needs per (rows) batch. Owned by the caller so a
    /// runner can size them once for its prefill chunk.
    struct Scratch {
        /// `[rows * bundle]` FP16 — the group-normed stream.
        let normed: MTLBuffer
        /// `[rows * lowRank]` FP32 — `W_down . h_n`, before the SiLU.
        let lowRankRaw: MTLBuffer
        /// `[rows * lowRank]` FP16 — `silu(raw / hc)`.
        let lowRank: MTLBuffer
        /// `[rows * bundle]` FP32 — `W_up . m`, before the sigmoid.
        let mixGate: MTLBuffer
        /// `[rows * hcCount]` FP32 — `W_inject . h_n`, before the gate.
        let injectRaw: MTLBuffer
        /// `[rows * hcCount]` FP32 — `2 * sigmoid(raw / hc)`.
        let injectGate: MTLBuffer
    }

    struct Weights {
        let norm: MTLBuffer            // [bundle] BF16, (1 + w) already baked
        let normOffset: Int
        let mixDown: FlashNextWeightMatrix   // [lowRank, bundle]
        let mixUp: FlashNextWeightMatrix     // [bundle, lowRank]
        /// Absent on the global mixer.
        let inject: FlashNextWeightMatrix?   // [hcCount, bundle]
    }

    private let hidden: Int
    private let hcCount: Int
    private let lowRank: Int
    private let eps: Float

    private let rms: RMSNorm
    private let matVec: FlashNextMatVec
    private let lowRankActivationPSO: MTLComputePipelineState
    private let mixPSO: MTLComputePipelineState
    private let injectGatePSO: MTLComputePipelineState
    private let injectAccumulatePSO: MTLComputePipelineState
    private let tileEmbeddingPSO: MTLComputePipelineState

    var bundle: Int { hidden * hcCount }

    init(context: MetalContext,
         rms: RMSNorm,
         matVec: FlashNextMatVec,
         hidden: Int, hcCount: Int, lowRank: Int, eps: Float) throws {
        precondition(hidden > 0 && hcCount > 0 && lowRank > 0)
        self.hidden = hidden
        self.hcCount = hcCount
        self.lowRank = lowRank
        self.eps = eps
        self.rms = rms
        self.matVec = matVec
        self.lowRankActivationPSO =
            try context.pipeline("flashnext_hc_lowrank_activation")
        self.mixPSO = try context.pipeline("flashnext_hc_mix")
        self.injectGatePSO = try context.pipeline("flashnext_hc_inject_gate")
        self.injectAccumulatePSO =
            try context.pipeline("flashnext_hc_inject_accumulate")
        self.tileEmbeddingPSO = try context.pipeline("flashnext_hc_tile_embedding")
    }

    func makeScratch(device: MTLDevice, rows: Int) throws -> Scratch {
        func buffer(_ elements: Int, _ stride: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(1, elements) * stride,
                                            options: .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return b
        }
        let half = MemoryLayout<Float16>.stride
        let float = MemoryLayout<Float>.stride
        return Scratch(
            normed: try buffer(rows * bundle, half),
            lowRankRaw: try buffer(rows * lowRank, float),
            lowRank: try buffer(rows * lowRank, half),
            mixGate: try buffer(rows * bundle, float),
            injectRaw: try buffer(rows * hcCount, float),
            injectGate: try buffer(rows * hcCount, float))
    }

    /// `hidden_states = embed(ids).repeat(1, 1, hc_count)` — a tile, so stream
    /// `j` is an exact copy of the embedding row.
    func encodeTileEmbedding(commandBuffer: MTLCommandBuffer,
                             embedding: MTLBuffer, embeddingOffset: Int = 0,
                             hyper: MTLBuffer, hyperOffset: Int = 0,
                             rows: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(tileEmbeddingPSO)
        enc.setBuffer(embedding, offset: embeddingOffset, index: 0)
        enc.setBuffer(hyper, offset: hyperOffset, index: 1)
        var hiddenVar = UInt32(hidden)
        var hcVar = UInt32(hcCount)
        enc.setBytes(&hiddenVar, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&hcVar, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch2D(enc, pso: tileEmbeddingPSO, width: bundle, height: rows)
        enc.endEncoding()
    }

    /// The mix half: group norm, the low-rank gate, and the mean over streams.
    /// Leaves the normed stream in `scratch.normed` because `encodeInject` reads
    /// it — the inject path scores the NORMED stream even though the residual
    /// add lands on the raw one.
    func encodeMix(commandBuffer: MTLCommandBuffer,
                   weights: Weights,
                   scratch: Scratch,
                   hyper: MTLBuffer, hyperOffset: Int = 0,
                   mixed: MTLBuffer, mixedOffset: Int = 0,
                   rows: Int) {
        rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                               x: hyper, xOffset: hyperOffset,
                               weight: weights.norm, weightOffset: weights.normOffset,
                               out: scratch.normed,
                               groupSize: UInt32(hidden), groups: UInt32(hcCount),
                               rows: rows, eps: eps)
        // The two GEMVs are per-token, so a chunk walks them one row at a time.
        //
        // PERF, not correctness: that is one compute encoder per row per GEMV.
        // Decode (rows == 1) is fine; a wide prefill chunk is not — at 48 layers
        // and two hyper-connections each, a 256-token chunk would encode ~49k
        // dispatches for these two matrices alone. Batching this into a 2-D
        // mat-vec (rows on the grid's second axis, which the BF16 kernel here
        // could take directly and `dequant_int4_gemv_simd` would need a variant
        // for) is the fix, and it belongs with the production runner rather than
        // ahead of it — the shapes are small enough that the dispatch count,
        // not the bandwidth, is what it buys back.
        for row in 0..<rows {
            matVec.encode(commandBuffer: commandBuffer,
                          matrix: weights.mixDown,
                          x: scratch.normed,
                          xOffset: row * bundle * MemoryLayout<Float16>.stride,
                          y: scratch.lowRankRaw,
                          yOffset: row * lowRank * MemoryLayout<Float>.stride,
                          rows: lowRank, cols: bundle, outputFloat32: true)
        }
        encodeElementwise(commandBuffer: commandBuffer,
                          pso: lowRankActivationPSO,
                          input: scratch.lowRankRaw, output: scratch.lowRank,
                          count: rows * lowRank, divisor: Float(hcCount))
        for row in 0..<rows {
            matVec.encode(commandBuffer: commandBuffer,
                          matrix: weights.mixUp,
                          x: scratch.lowRank,
                          xOffset: row * lowRank * MemoryLayout<Float16>.stride,
                          y: scratch.mixGate,
                          yOffset: row * bundle * MemoryLayout<Float>.stride,
                          rows: bundle, cols: lowRank, outputFloat32: true)
        }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(mixPSO)
        enc.setBuffer(scratch.mixGate, offset: 0, index: 0)
        enc.setBuffer(scratch.normed, offset: 0, index: 1)
        enc.setBuffer(mixed, offset: mixedOffset, index: 2)
        var hiddenVar = UInt32(hidden)
        var hcVar = UInt32(hcCount)
        enc.setBytes(&hiddenVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hcVar, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch2D(enc, pso: mixPSO, width: hidden, height: rows)
        enc.endEncoding()
    }

    /// The inject gate, from the normed stream `encodeMix` left behind.
    func encodeInjectGate(commandBuffer: MTLCommandBuffer,
                          weights: Weights,
                          scratch: Scratch,
                          rows: Int) {
        guard let inject = weights.inject else {
            preconditionFailure("the global mixer has no inject path")
        }
        for row in 0..<rows {
            matVec.encode(commandBuffer: commandBuffer,
                          matrix: inject,
                          x: scratch.normed,
                          xOffset: row * bundle * MemoryLayout<Float16>.stride,
                          y: scratch.injectRaw,
                          yOffset: row * hcCount * MemoryLayout<Float>.stride,
                          rows: hcCount, cols: bundle, outputFloat32: true)
        }
        encodeElementwise(commandBuffer: commandBuffer,
                          pso: injectGatePSO,
                          input: scratch.injectRaw, output: scratch.injectGate,
                          count: rows * hcCount, divisor: Float(hcCount))
    }

    /// `hyper += flatten(block_out * inject)`, in place on the raw stream.
    func encodeInjectAccumulate(commandBuffer: MTLCommandBuffer,
                                scratch: Scratch,
                                hyper: MTLBuffer, hyperOffset: Int = 0,
                                block: MTLBuffer, blockOffset: Int = 0,
                                rows: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(injectAccumulatePSO)
        enc.setBuffer(hyper, offset: hyperOffset, index: 0)
        enc.setBuffer(block, offset: blockOffset, index: 1)
        enc.setBuffer(scratch.injectGate, offset: 0, index: 2)
        var hiddenVar = UInt32(hidden)
        var hcVar = UInt32(hcCount)
        enc.setBytes(&hiddenVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hcVar, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch2D(enc, pso: injectAccumulatePSO, width: bundle, height: rows)
        enc.endEncoding()
    }

    // MARK: - Dispatch helpers

    private func encodeElementwise(commandBuffer: MTLCommandBuffer,
                                   pso: MTLComputePipelineState,
                                   input: MTLBuffer, output: MTLBuffer,
                                   count: Int, divisor: Float) {
        guard count > 0,
              let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        var countVar = UInt32(count)
        var divisorVar = divisor
        enc.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&divisorVar, length: MemoryLayout<Float>.size, index: 3)
        let width = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(width, count),
                                                           height: 1, depth: 1))
        enc.endEncoding()
    }

    private func dispatch2D(_ enc: MTLComputeCommandEncoder,
                            pso: MTLComputePipelineState,
                            width: Int, height: Int) {
        let w = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(w, width),
                                                           height: 1, depth: 1))
    }
}
