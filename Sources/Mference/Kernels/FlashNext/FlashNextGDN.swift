import Foundation
import Metal

/// The dimension-generic gated DeltaNet decode path.
///
/// Flash-Next's GDN block is Qwen 3.8's geometry (Hk 16, Hv 48, Dk 128, Dv 128,
/// conv 4), so the production runner takes the shipped `GDN` kernels — the fused
/// Hv=48 decode included — with the gated norm switched to sigmoid. This type is
/// the fallback for a geometry those kernels refuse: they are 32-lane tiled and
/// require `key_head_dim % 32 == 0`, which the parity toy's Dk of 8 is not.
///
/// See `flashnext_gdn.metal` for the recurrence and the two easy-to-lose details
/// (the l2norm's sum-with-eps-inside, and the ones-centered gated-norm weight).
///
/// The state layout is `[head][dk][dv]` — the reference's own indexing, and a
/// different order from the shipped kernels' `[Hv, Dv, Dk]`. The two paths are
/// never mixed on one install, but a runner must not switch between them
/// mid-sequence.
final class FlashNextGDN {

    struct Geometry {
        let numKHeads: Int
        let numVHeads: Int
        let keyHeadDim: Int
        let valueHeadDim: Int
        let convKernel: Int
        let eps: Float

        var keyDim: Int { numKHeads * keyHeadDim }
        var valueDim: Int { numVHeads * valueHeadDim }
        var qkvDim: Int { 2 * keyDim + valueDim }
        /// Rows of conv input carried between steps.
        var convStateRows: Int { convKernel - 1 }
    }

    /// Per-layer decode state. The recurrent state is FP32 and the conv tail
    /// FP16, matching what the shipped `GDNStateManager` holds — but allocated
    /// here because the element ORDER differs.
    struct LayerState {
        let recurrent: MTLBuffer    // [Hv, Dk, Dv] FP32
        let convTail: MTLBuffer     // [K-1, qkvDim] FP16
    }

    struct Scratch {
        /// `[(K - 1) + rows, qkvDim]` FP16 — `[carried tail | new rows]`.
        let convPadded: MTLBuffer
        /// `[rows, qkvDim]` FP16 — post-conv, post-SiLU.
        let convOut: MTLBuffer
        let rows: Int
    }

    /// Largest key head dim the kernel's threadgroup scratch supports.
    static let maxKeyHeadDim = 128

    let geometry: Geometry
    private let convPSO: MTLComputePipelineState
    private let recurrencePSO: MTLComputePipelineState

    init(context: MetalContext, geometry: Geometry) throws {
        precondition(geometry.keyHeadDim <= Self.maxKeyHeadDim,
                     "generic GDN supports key head dims up to \(Self.maxKeyHeadDim)")
        precondition(geometry.numVHeads % geometry.numKHeads == 0,
                     "numVHeads must be a multiple of numKHeads")
        precondition(geometry.convKernel >= 1)
        self.geometry = geometry
        self.convPSO = try context.pipeline("flashnext_gdn_conv")
        self.recurrencePSO = try context.pipeline("flashnext_gdn_recurrence")
    }

    // MARK: - Allocation

    func makeState(device: MTLDevice) throws -> LayerState {
        let recurrentCount = geometry.numVHeads * geometry.keyHeadDim
            * geometry.valueHeadDim
        guard let recurrent = device.makeBuffer(
                  length: max(1, recurrentCount) * MemoryLayout<Float>.stride,
                  options: .storageModePrivate),
              let tail = device.makeBuffer(
                  length: max(1, geometry.convStateRows * geometry.qkvDim)
                      * MemoryLayout<Float16>.stride,
                  options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        recurrent.label = "flashnext.gdn.state"
        tail.label = "flashnext.gdn.convTail"
        return LayerState(recurrent: recurrent, convTail: tail)
    }

    func makeScratch(device: MTLDevice, rows: Int) throws -> Scratch {
        let padded = (geometry.convStateRows + rows) * geometry.qkvDim
        guard let p = device.makeBuffer(
                  length: max(1, padded) * MemoryLayout<Float16>.stride,
                  options: .storageModePrivate),
              let o = device.makeBuffer(
                  length: max(1, rows * geometry.qkvDim)
                      * MemoryLayout<Float16>.stride,
                  options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        return Scratch(convPadded: p, convOut: o, rows: rows)
    }

    /// Zero the recurrent state and the conv tail — a fresh sequence.
    func encodeReset(commandBuffer: MTLCommandBuffer, state: LayerState) {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.fill(buffer: state.recurrent, range: 0..<state.recurrent.length, value: 0)
        blit.fill(buffer: state.convTail, range: 0..<state.convTail.length, value: 0)
        blit.endEncoding()
    }

    // MARK: - Encode

    /// Conv over `[carried tail | qkv]`, SiLU, then roll the tail forward.
    func encodeConv(commandBuffer cb: MTLCommandBuffer,
                    qkv: MTLBuffer, qkvOffset: Int = 0,
                    convWeight: MTLBuffer, convWeightOffset: Int,
                    scratch: Scratch, state: LayerState, rows: Int) {
        precondition(rows > 0 && rows <= scratch.rows)
        let half = MemoryLayout<Float16>.stride
        let rowBytes = geometry.qkvDim * half
        let stateRows = geometry.convStateRows
        if let blit = cb.makeBlitCommandEncoder() {
            if stateRows > 0 {
                blit.copy(from: state.convTail, sourceOffset: 0,
                          to: scratch.convPadded, destinationOffset: 0,
                          size: stateRows * rowBytes)
            }
            blit.copy(from: qkv, sourceOffset: qkvOffset,
                      to: scratch.convPadded,
                      destinationOffset: stateRows * rowBytes,
                      size: rows * rowBytes)
            blit.endEncoding()
        }
        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(convPSO)
            enc.setBuffer(scratch.convPadded, offset: 0, index: 0)
            enc.setBuffer(scratch.convOut, offset: 0, index: 1)
            enc.setBuffer(convWeight, offset: convWeightOffset, index: 2)
            var qkvDim = UInt32(geometry.qkvDim)
            var kernelWidth = UInt32(geometry.convKernel)
            enc.setBytes(&qkvDim, length: 4, index: 3)
            enc.setBytes(&kernelWidth, length: 4, index: 4)
            let w = min(Int(convPSO.maxTotalThreadsPerThreadgroup), 256)
            enc.dispatchThreads(
                MTLSize(width: geometry.qkvDim, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(w, geometry.qkvDim),
                                               height: 1, depth: 1))
            enc.endEncoding()
        }
        // Keep the last (K - 1) rows of [tail | rows] as the next tail. A blit
        // between two distinct buffers, never in place: blits inside one encoder
        // are unordered, so a self-overlapping shift would race whenever
        // `rows < stateRows` — which is every decode step.
        guard stateRows > 0, let blit = cb.makeBlitCommandEncoder() else { return }
        blit.copy(from: scratch.convPadded,
                  sourceOffset: rows * rowBytes,
                  to: state.convTail, destinationOffset: 0,
                  size: stateRows * rowBytes)
        blit.endEncoding()
    }

    /// One decode step of the recurrence plus the sigmoid gated norm.
    func encodeRecurrence(commandBuffer cb: MTLCommandBuffer,
                          scratch: Scratch, state: LayerState,
                          z: MTLBuffer, a: MTLBuffer, b: MTLBuffer,
                          aLog: MTLBuffer, aLogOffset: Int,
                          dtBias: MTLBuffer, dtBiasOffset: Int,
                          normWeight: MTLBuffer, normWeightOffset: Int,
                          out: MTLBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(recurrencePSO)
        enc.setBuffer(scratch.convOut, offset: 0, index: 0)
        enc.setBuffer(z, offset: 0, index: 1)
        enc.setBuffer(a, offset: 0, index: 2)
        enc.setBuffer(b, offset: 0, index: 3)
        enc.setBuffer(aLog, offset: aLogOffset, index: 4)
        enc.setBuffer(dtBias, offset: dtBiasOffset, index: 5)
        enc.setBuffer(normWeight, offset: normWeightOffset, index: 6)
        enc.setBuffer(state.recurrent, offset: 0, index: 7)
        enc.setBuffer(out, offset: 0, index: 8)
        var hk = UInt32(geometry.numKHeads)
        var hv = UInt32(geometry.numVHeads)
        var dk = UInt32(geometry.keyHeadDim)
        var dv = UInt32(geometry.valueHeadDim)
        var eps = geometry.eps
        enc.setBytes(&hk, length: 4, index: 9)
        enc.setBytes(&hv, length: 4, index: 10)
        enc.setBytes(&dk, length: 4, index: 11)
        enc.setBytes(&dv, length: 4, index: 12)
        enc.setBytes(&eps, length: 4, index: 13)
        // One threadgroup per value head, one thread per value channel, rounded
        // up to a full SIMD group so the reductions are well formed.
        let threads = max(32, ((geometry.valueHeadDim + 31) / 32) * 32)
        enc.dispatchThreadgroups(
            MTLSize(width: geometry.numVHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(threads, Int(recurrencePSO.maxTotalThreadsPerThreadgroup)),
                height: 1, depth: 1))
        enc.endEncoding()
    }
}
