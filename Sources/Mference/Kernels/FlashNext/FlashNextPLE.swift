import Foundation
import Metal

/// `Qwen4ExpTextPLELayer` — the per-layer n-gram embedding, applied to the raw
/// hyper stream before attention at one layer.
///
/// The hash and the row gather live outside this type (`FlashNextPleHash` and
/// `PleRowPool`): 16 int64 hashes and 16 row reads per token, both CPU work by
/// design, since the table is 102 GB on disk and the reads go through an LFU
/// cache. What this type encodes is everything after the gather:
///
/// ```
/// k      = group_rmsnorm(W_key . e)        viewed [hc, H]
/// v      = W_value . e                                       // H
/// qn     = group_rmsnorm(hyper)            viewed [hc, H]
/// gate_s = sigmoid( signed_sqrt( (k_s . qn_s) / sqrt(H) ) )  // per stream
/// gv     = flatten(gate_s * v)                               // hc*H
/// out    = gv + silu(depthwise_conv1d(group_rmsnorm(gv)))
/// hyper += out
/// ```
///
/// # Decode state
///
/// Two pieces, both per sequence: the previous `ngram_size - 1` token ids
/// (`FlashNextPleHash`, pre-filled with `eos_token_id`) and
/// `(kernel - 1) * dilation = 9` rows of the *normed* gated value for the conv.
/// The conv scratch is laid out `[state | rows]` so the norm writes straight
/// past the state and the taps index backwards into it — no separate
/// concatenation pass.
final class FlashNextPLE {

    struct Weights {
        let keyProj: FlashNextWeightMatrix     // [bundle, hidden]
        let valueProj: FlashNextWeightMatrix   // [hidden, hidden]
        let conv: MTLBuffer                    // [bundle, kernel] BF16
        let convOffset: Int
        let normKey: MTLBuffer                 // [bundle] BF16
        let normKeyOffset: Int
        let normQuery: MTLBuffer
        let normQueryOffset: Int
        let normConv: MTLBuffer
        let normConvOffset: Int
    }

    struct Scratch {
        /// `[rows * hidden]` FP16 — the gathered n-gram embedding, staged from
        /// the row pool.
        let embeds: MTLBuffer
        /// `[rows * bundle]` FP16 — `W_key . e`, then its group norm.
        let keyProjected: MTLBuffer
        let keyNormed: MTLBuffer
        /// `[rows * bundle]` FP16 — the group-normed raw hyper stream.
        let queryNormed: MTLBuffer
        /// `[rows * hidden]` FP16 — `W_value . e`.
        let value: MTLBuffer
        /// `[rows * hcCount]` FP32 — the per-stream gate.
        let gate: MTLBuffer
        /// `[rows * bundle]` FP16 — `gate_s * v`, and the kernel's output
        /// accumulator (it is seeded with exactly this).
        let gatedValue: MTLBuffer
        /// `[(stateLength + rows) * bundle]` FP16 — `[conv state | normed rows]`.
        let convPadded: MTLBuffer
        /// `[stateLength * bundle]` FP16 — the carried conv state, held apart
        /// from `convPadded` so rolling it forward is always a copy between two
        /// distinct buffers. Blits within one encoder are not ordered against
        /// each other, so an in-place shift would race whenever
        /// `rows < stateLength` — which is every decode step.
        let convState: MTLBuffer
        let rows: Int
    }

    private let hidden: Int
    private let hcCount: Int
    private let convKernel: Int
    private let dilation: Int
    private let eps: Float

    private let rms: RMSNorm
    private let matVec: FlashNextMatVec
    private let elementwise: Elementwise
    private let streamGatePSO: MTLComputePipelineState
    private let applyGatePSO: MTLComputePipelineState
    private let convPSO: MTLComputePipelineState

    var bundle: Int { hidden * hcCount }
    /// `(kernel - 1) * dilation` rows of normed gated value carried between
    /// steps — the `L` of the reference cache's `update_conv_state`.
    var stateLength: Int { (convKernel - 1) * dilation }

    /// `dilation` is `ngram_size`, per the reference's `Qwen4ExpTextPLELayer`.
    init(context: MetalContext,
         rms: RMSNorm,
         matVec: FlashNextMatVec,
         elementwise: Elementwise,
         hidden: Int, hcCount: Int,
         convKernel: Int, dilation: Int,
         eps: Float) throws {
        precondition(hidden > 0 && hcCount > 0)
        precondition(convKernel > 0 && dilation > 0)
        self.hidden = hidden
        self.hcCount = hcCount
        self.convKernel = convKernel
        self.dilation = dilation
        self.eps = eps
        self.rms = rms
        self.matVec = matVec
        self.elementwise = elementwise
        self.streamGatePSO = try context.pipeline("flashnext_ple_stream_gate")
        self.applyGatePSO = try context.pipeline("flashnext_ple_apply_gate")
        self.convPSO = try context.pipeline("flashnext_ple_conv")
    }

    func makeScratch(device: MTLDevice, rows: Int,
                     storageMode: MTLResourceOptions = .storageModePrivate) throws
        -> Scratch {
        func buffer(_ elements: Int, _ stride: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(1, elements) * stride,
                                            options: storageMode) else {
                throw MetalError.noDevice
            }
            return b
        }
        let half = MemoryLayout<Float16>.stride
        let float = MemoryLayout<Float>.stride
        return Scratch(
            embeds: try buffer(rows * hidden, half),
            keyProjected: try buffer(rows * bundle, half),
            keyNormed: try buffer(rows * bundle, half),
            queryNormed: try buffer(rows * bundle, half),
            value: try buffer(rows * hidden, half),
            gate: try buffer(rows * hcCount, float),
            gatedValue: try buffer(rows * bundle, half),
            convPadded: try buffer((stateLength + rows) * bundle, half),
            convState: try buffer(stateLength * bundle, half),
            rows: rows)
    }

    /// Zero the conv state. A fresh sequence carries nine rows of zeros, which
    /// is what the reference's cache produces before `has_previous_state`.
    func encodeResetState(commandBuffer: MTLCommandBuffer, scratch: Scratch) {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.fill(buffer: scratch.convState,
                  range: 0..<(stateLength * bundle * MemoryLayout<Float16>.stride),
                  value: 0)
        blit.endEncoding()
    }

    /// The whole mixing block. `scratch.embeds` must already hold the gathered
    /// n-gram embedding for these rows; the carried conv state lives in
    /// `scratch.convState` and is staged into the scratch here.
    ///
    /// On return `hyper` has the PLE output added, `scratch.gatedValue` holds
    /// that output, and the conv state has been rolled forward.
    func encode(commandBuffer: MTLCommandBuffer,
                weights: Weights,
                scratch: Scratch,
                hyper: MTLBuffer, hyperOffset: Int = 0,
                rows: Int) {
        precondition(rows > 0 && rows <= scratch.rows)
        let half = MemoryLayout<Float16>.stride
        encodeLoadConvState(commandBuffer: commandBuffer, scratch: scratch)

        // k = group_rmsnorm(W_key . e)
        for row in 0..<rows {
            matVec.encode(commandBuffer: commandBuffer, matrix: weights.keyProj,
                          x: scratch.embeds, xOffset: row * hidden * half,
                          y: scratch.keyProjected, yOffset: row * bundle * half,
                          rows: bundle, cols: hidden)
        }
        rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                               x: scratch.keyProjected,
                               weight: weights.normKey,
                               weightOffset: weights.normKeyOffset,
                               out: scratch.keyNormed,
                               groupSize: UInt32(hidden), groups: UInt32(hcCount),
                               rows: rows, eps: eps)

        // v = W_value . e
        for row in 0..<rows {
            matVec.encode(commandBuffer: commandBuffer, matrix: weights.valueProj,
                          x: scratch.embeds, xOffset: row * hidden * half,
                          y: scratch.value, yOffset: row * hidden * half,
                          rows: hidden, cols: hidden)
        }

        // qn = group_rmsnorm(hyper) — the RAW stream, before this block's add.
        rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                               x: hyper, xOffset: hyperOffset,
                               weight: weights.normQuery,
                               weightOffset: weights.normQueryOffset,
                               out: scratch.queryNormed,
                               groupSize: UInt32(hidden), groups: UInt32(hcCount),
                               rows: rows, eps: eps)

        encodeStreamGate(commandBuffer: commandBuffer, scratch: scratch, rows: rows)
        encodeApplyGate(commandBuffer: commandBuffer, scratch: scratch, rows: rows)

        // The conv reads the NORMED gated value; the output accumulates onto the
        // UN-normed one, which `gatedValue` already holds.
        rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                               x: scratch.gatedValue,
                               weight: weights.normConv,
                               weightOffset: weights.normConvOffset,
                               out: scratch.convPadded,
                               outOffset: stateLength * bundle * half,
                               groupSize: UInt32(hidden), groups: UInt32(hcCount),
                               rows: rows, eps: eps)
        encodeConv(commandBuffer: commandBuffer, weights: weights,
                   scratch: scratch, rows: rows)

        elementwise.encodeResidualAdd(commandBuffer: commandBuffer,
                                      hidden: hyper, hiddenOffset: hyperOffset,
                                      delta: scratch.gatedValue,
                                      count: rows * bundle)
        encodeRollConvState(commandBuffer: commandBuffer, scratch: scratch, rows: rows)
    }

    // MARK: - Steps

    private func encodeStreamGate(commandBuffer: MTLCommandBuffer,
                                  scratch: Scratch, rows: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(streamGatePSO)
        enc.setBuffer(scratch.keyNormed, offset: 0, index: 0)
        enc.setBuffer(scratch.queryNormed, offset: 0, index: 1)
        enc.setBuffer(scratch.gate, offset: 0, index: 2)
        var hiddenVar = UInt32(hidden)
        var hcVar = UInt32(hcCount)
        enc.setBytes(&hiddenVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hcVar, length: MemoryLayout<UInt32>.size, index: 4)
        let w = min(Int(streamGatePSO.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(
            MTLSize(width: rows * hcCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeApplyGate(commandBuffer: MTLCommandBuffer,
                                 scratch: Scratch, rows: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(applyGatePSO)
        enc.setBuffer(scratch.value, offset: 0, index: 0)
        enc.setBuffer(scratch.gate, offset: 0, index: 1)
        enc.setBuffer(scratch.gatedValue, offset: 0, index: 2)
        var hiddenVar = UInt32(hidden)
        var hcVar = UInt32(hcCount)
        enc.setBytes(&hiddenVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hcVar, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch2D(enc, pso: applyGatePSO, width: bundle, height: rows)
        enc.endEncoding()
    }

    private func encodeConv(commandBuffer: MTLCommandBuffer,
                            weights: Weights, scratch: Scratch, rows: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(convPSO)
        enc.setBuffer(scratch.convPadded, offset: 0, index: 0)
        enc.setBuffer(scratch.gatedValue, offset: 0, index: 1)
        enc.setBuffer(weights.conv, offset: weights.convOffset, index: 2)
        var bundleVar = UInt32(bundle)
        var kernelVar = UInt32(convKernel)
        var dilationVar = UInt32(dilation)
        var stateVar = UInt32(stateLength)
        enc.setBytes(&bundleVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&kernelVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dilationVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&stateVar, length: MemoryLayout<UInt32>.size, index: 6)
        dispatch2D(enc, pso: convPSO, width: bundle, height: rows)
        enc.endEncoding()
    }

    /// Stage the carried state into the head of the conv scratch, so the taps
    /// can index backwards into it without a special case at the boundary.
    private func encodeLoadConvState(commandBuffer: MTLCommandBuffer,
                                     scratch: Scratch) {
        let bytes = stateLength * bundle * MemoryLayout<Float16>.stride
        guard bytes > 0, let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: scratch.convState, sourceOffset: 0,
                  to: scratch.convPadded, destinationOffset: 0, size: bytes)
        blit.endEncoding()
    }

    /// Keep the last `stateLength` rows of `[state | rows]` as the next state —
    /// `update_conv_state`'s "stores the last L columns".
    private func encodeRollConvState(commandBuffer: MTLCommandBuffer,
                                     scratch: Scratch, rows: Int) {
        let rowBytes = bundle * MemoryLayout<Float16>.stride
        let bytes = stateLength * rowBytes
        guard bytes > 0, let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: scratch.convPadded,
                  sourceOffset: (stateLength + rows - stateLength) * rowBytes,
                  to: scratch.convState, destinationOffset: 0, size: bytes)
        blit.endEncoding()
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
