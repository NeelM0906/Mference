import Foundation
import Metal

/// Swift wrappers for the DeepSeek-V4 kernel family (`Metal/DSV4/dsv4.metal`):
/// interleaved trailing partial RoPE, shared-KV MQA decode attention with
/// per-head sinks over [window ring ‖ compressed entries], the grouped
/// low-rank output projection, the CSA/HCA window compressor, the
/// lightning-indexer scorer, mHC stream mixing, and the clamped-SwiGLU
/// elementwise.
final class DSV4Kernels {
    /// Query heads serviced by one attention threadgroup, one simdgroup each.
    /// Must equal `kDSV4HeadsPerTG` in `dsv4.metal`.
    static let attentionHeadsPerThreadgroup = 8
    /// KV width the attention kernel is written for (`kDSV4AttnHeadDim`).
    static let attentionHeadDim = 512
    /// Output rows one `dsv4_o_group_gemv_int4` threadgroup covers, one
    /// simdgroup each. Must equal its `rows_per_tg`.
    private static let oGroupRowsPerThreadgroup = 8
    /// Sentinel `selected_count` meaning "attend to every compressed entry".
    static let selectAll: UInt32 = 0xFFFF_FFFF

    /// Which rope-parameter set a rotation uses: sliding-window layers rope
    /// at the plain `main` frequencies; CSA/HCA layers, their compressors,
    /// and the lightning indexer at the (YaRN-corrected) `compress` ones.
    enum RopeKind {
        case main
        case compress
    }

    private let ropePSO: MTLComputePipelineState
    /// Unspecialized rope PSO for non-attention head shapes (the 128-dim
    /// lightning-indexer keys/queries). The specialized `ropePSO` bakes
    /// `fullHeadDim` in via function constant and would address 512-wide
    /// heads regardless of the dispatch's `head_dim_in`.
    private let ropeGenericPSO: MTLComputePipelineState
    private let specializedHeadDim: Int
    private let mainInvFreq: MTLBuffer
    private let compressInvFreq: MTLBuffer
    private let attentionPSO: MTLComputePipelineState
    private let oGroupPSO: MTLComputePipelineState
    private let compressEmitPSO: MTLComputePipelineState
    private let indexerScorePSO: MTLComputePipelineState
    private let hcWeightsPSO: MTLComputePipelineState
    private let hcCollapsePSO: MTLComputePipelineState
    private let hcPlaceMixPSO: MTLComputePipelineState
    private let hyperHeadPSO: MTLComputePipelineState
    private let swigluClampPSO: MTLComputePipelineState
    private let broadcastPSO: MTLComputePipelineState

    init(context: MetalContext, config: ArchConfig) throws {
        let constants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 100, value: .uint32(UInt32(config.fullHeadDim))),
            MetalFunctionConstant(index: 101, value: .uint32(UInt32(config.numHeads))),
            MetalFunctionConstant(index: 102,
                                  value: .uint32(UInt32(config.compressedAttention.ropeHeadDim))),
            MetalFunctionConstant(index: 103,
                                  value: .uint32(UInt32(config.hyperConnections.mult))),
            MetalFunctionConstant(index: 104, value: .uint32(UInt32(config.hiddenSize))),
            MetalFunctionConstant(index: 105, value: .bool(true)),
            MetalFunctionConstant(
                index: 106,
                value: .uint32(UInt32(config.compressedAttention.oLoraRank))),
            MetalFunctionConstant(
                index: 107,
                value: .uint32(UInt32(config.numHeads * config.fullHeadDim
                                      / max(config.compressedAttention.oGroups, 1)))),
        ]
        self.ropePSO = try context.pipeline(
            "dsv4_rope_interleaved_trailing", constants: constants)
        self.ropeGenericPSO = try context.pipeline(
            "dsv4_rope_interleaved_trailing", constants: [])
        self.specializedHeadDim = config.fullHeadDim

        // Per-pair inverse-frequency tables. YaRN corrects the compress
        // frequencies only; cos/sin are never rescaled (attention_factor 1).
        let ca = config.compressedAttention
        let ropeDim = max(ca.ropeHeadDim, 2)
        let mainTable = DSV4RopeTables.plainInvFreq(
            theta: config.ropeTheta, ropeDim: ropeDim)
        let compressTable = ca.ropeScalingFactor > 0
            ? DSV4RopeTables.yarnInvFreq(
                theta: ca.compressRopeTheta, ropeDim: ropeDim,
                factor: ca.ropeScalingFactor,
                originalMaxPositions: ca.ropeScalingOriginalMax,
                betaFast: ca.ropeScalingBetaFast,
                betaSlow: ca.ropeScalingBetaSlow)
            : DSV4RopeTables.plainInvFreq(
                theta: max(ca.compressRopeTheta, 1), ropeDim: ropeDim)
        guard let mainBuf = context.device.makeBuffer(
                bytes: mainTable,
                length: mainTable.count * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let compressBuf = context.device.makeBuffer(
                bytes: compressTable,
                length: compressTable.count * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.mainInvFreq = mainBuf
        self.compressInvFreq = compressBuf
        self.attentionPSO = try context.pipeline(
            "dsv4_attention_decode", constants: constants,
            maxTotalThreadsPerThreadgroup: 256)
        self.oGroupPSO = try context.pipeline(
            "dsv4_o_group_gemv_int4", constants: constants,
            maxTotalThreadsPerThreadgroup: 256)
        self.compressEmitPSO = try context.pipeline("dsv4_compress_emit")
        self.indexerScorePSO = try context.pipeline("dsv4_indexer_score")
        self.hcWeightsPSO = try context.pipeline(
            "dsv4_hc_weights", constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        self.hcCollapsePSO = try context.pipeline("dsv4_hc_collapse")
        self.hcPlaceMixPSO = try context.pipeline("dsv4_hc_place_mix")
        self.hyperHeadPSO = try context.pipeline(
            "dsv4_hyper_head", constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        self.swigluClampPSO = try context.pipeline("dsv4_swiglu_clamp_mul")
        self.broadcastPSO = try context.pipeline("dsv4_broadcast_streams")
    }

    /// Interleaved RoPE on the trailing `ropeDim` channels of `numHeads`
    /// contiguous heads at `position`. `direction` -1 applies the conjugate
    /// rotation (attention-output un-rotation).
    func encodeRope(commandBuffer: MTLCommandBuffer,
                    x: MTLBuffer, xOffset: Int = 0,
                    numHeads: Int, headDim: Int, ropeDim: Int,
                    position: Int, rope: RopeKind, direction: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(
            headDim == specializedHeadDim ? ropePSO : ropeGenericPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        var hd = UInt32(headDim)
        var rd = UInt32(ropeDim)
        var pos = UInt32(position)
        var dir = direction
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&rd, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&pos, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBuffer(rope == .compress ? compressInvFreq : mainInvFreq,
                      offset: 0, index: 4)
        enc.setBytes(&dir, length: MemoryLayout<Float>.size, index: 5)
        let threads = max(32, ropeDim / 2)
        enc.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Decode attention for one token. `selectedCount == DSV4Kernels.selectAll`
    /// attends to every compressed entry (windowed layers pass
    /// `compressedCount == 0`).
    func encodeAttention(commandBuffer: MTLCommandBuffer,
                         q: MTLBuffer, qOffset: Int = 0,
                         windowKV: MTLBuffer,
                         compressedKV: MTLBuffer,
                         selected: MTLBuffer,
                         sinks: MTLBuffer, sinksOffset: Int,
                         out: MTLBuffer, outOffset: Int = 0,
                         headDim: Int, numHeads: Int,
                         windowCount: Int, windowStartPos: Int, ringCapacity: Int,
                         compressedCount: Int, selectedCount: UInt32,
                         scale: Float) {
        // The online-softmax kernel streams entries, so there is no ceiling on
        // window + compressed rows. It does assume the absorbed-MLA KV width:
        // one simdgroup lane owns four half4 of a row.
        precondition(headDim == Self.attentionHeadDim,
                     "dsv4_attention_decode is written for headDim \(Self.attentionHeadDim), got \(headDim)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(attentionPSO)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(windowKV, offset: 0, index: 1)
        enc.setBuffer(compressedKV, offset: 0, index: 2)
        enc.setBuffer(selected, offset: 0, index: 3)
        enc.setBuffer(sinks, offset: sinksOffset, index: 4)
        enc.setBuffer(out, offset: outOffset, index: 5)
        var hd = UInt32(headDim)
        var wc = UInt32(windowCount)
        var ws = UInt32(windowStartPos)
        var rc = UInt32(ringCapacity)
        var cc = UInt32(compressedCount)
        var sc = selectedCount
        var sl = scale
        var nh = UInt32(numHeads)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&wc, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&ws, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&rc, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&cc, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&sc, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&sl, length: MemoryLayout<Float>.size, index: 12)
        enc.setBytes(&nh, length: MemoryLayout<UInt32>.size, index: 13)
        let headsPerGroup = Self.attentionHeadsPerThreadgroup
        let groups = (numHeads + headsPerGroup - 1) / headsPerGroup
        enc.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: headsPerGroup * 32,
                                           height: 1, depth: 1))
        enc.endEncoding()
    }

    /// The grouped low-rank output projection in one dispatch: `groups`
    /// contiguous INT4 weight blocks of `rank` rows each, group `g` reading
    /// `x` at `g * groupIn`. Replaces one GEMV per group.
    func encodeOGroupProjection(commandBuffer: MTLCommandBuffer,
                                weights: MTLBuffer, weightsOffset: Int,
                                scales: MTLBuffer, scalesOffset: Int,
                                biases: MTLBuffer, biasesOffset: Int,
                                x: MTLBuffer, xOffset: Int = 0,
                                y: MTLBuffer, yOffset: Int = 0,
                                rank: Int, groupIn: Int, groups: Int) {
        precondition(groupIn % Quantization.groupSize == 0,
                     "groupIn must be a multiple of \(Quantization.groupSize)")
        precondition(weightsOffset % 2 == 0,
                     "dsv4_o_group_gemv_int4 needs a 2-aligned weightsOffset, got \(weightsOffset)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(oGroupPSO)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var r = UInt32(rank)
        var gi = UInt32(groupIn)
        var g = UInt32(groups)
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&gi, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&g, length: MemoryLayout<UInt32>.size, index: 7)
        let rowsPerGroup = Self.oGroupRowsPerThreadgroup
        let rows = rank * groups
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + rowsPerGroup - 1) / rowsPerGroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: rowsPerGroup * 32,
                                           height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Emit one compressed entry from a full pending window into
    /// `outEntry` (an offset slot inside the compressed cache). RoPE at the
    /// window position is a separate `encodeRope` on the emitted slot.
    func encodeCompressEmit(commandBuffer: MTLCommandBuffer,
                            pendingKV: MTLBuffer, pendingGate: MTLBuffer,
                            priorCaKV: MTLBuffer, priorCaGate: MTLBuffer,
                            positionBias: MTLBuffer, positionBiasOffset: Int,
                            normWeight: MTLBuffer, normWeightOffset: Int,
                            outEntry: MTLBuffer, outEntryOffset: Int,
                            nextPriorCaKV: MTLBuffer, nextPriorCaGate: MTLBuffer,
                            rate: Int, dim: Int, dual: Bool, hasPrior: Bool,
                            eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(compressEmitPSO)
        enc.setBuffer(pendingKV, offset: 0, index: 0)
        enc.setBuffer(pendingGate, offset: 0, index: 1)
        enc.setBuffer(priorCaKV, offset: 0, index: 2)
        enc.setBuffer(priorCaGate, offset: 0, index: 3)
        enc.setBuffer(positionBias, offset: positionBiasOffset, index: 4)
        enc.setBuffer(normWeight, offset: normWeightOffset, index: 5)
        enc.setBuffer(outEntry, offset: outEntryOffset, index: 6)
        enc.setBuffer(nextPriorCaKV, offset: 0, index: 7)
        enc.setBuffer(nextPriorCaGate, offset: 0, index: 8)
        var r = UInt32(rate)
        var d = UInt32(dim)
        var du: UInt32 = dual ? 1 : 0
        var hp: UInt32 = hasPrior ? 1 : 0
        var e = eps
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&du, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&hp, length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBytes(&e, length: MemoryLayout<Float>.size, index: 13)
        let threads = min(max(dim, 32), 512)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Lightning-indexer scores over `entryCount` compressed index keys.
    /// `weights` is the raw FP16 weights_proj output; `weightScale` carries
    /// the heads^-0.5 factor.
    func encodeIndexerScore(commandBuffer: MTLCommandBuffer,
                            q: MTLBuffer,
                            keys: MTLBuffer,
                            weights: MTLBuffer,
                            scores: MTLBuffer,
                            numHeads: Int, indexDim: Int, entryCount: Int,
                            headScale: Float, weightScale: Float) {
        guard entryCount > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(indexerScorePSO)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(keys, offset: 0, index: 1)
        enc.setBuffer(weights, offset: 0, index: 2)
        enc.setBuffer(scores, offset: 0, index: 3)
        var nh = UInt32(numHeads)
        var id = UInt32(indexDim)
        var hs = headScale
        var ws = weightScale
        enc.setBytes(&nh, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&id, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hs, length: MemoryLayout<Float>.size, index: 6)
        enc.setBytes(&ws, length: MemoryLayout<Float>.size, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: entryCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// mHC mixing weights for one sublayer site.
    ///
    /// `streamsOffset` selects one token's stream block inside a chunk-wide
    /// `[tokens, hcMult * hidden]` buffer; chunked prefill uses it to replay
    /// the decode dispatch per token without copying. Zero (the default) is
    /// the decode path.
    func encodeHCWeights(commandBuffer: MTLCommandBuffer,
                         streams: MTLBuffer, streamsOffset: Int = 0,
                         fn: MTLBuffer, fnOffset: Int,
                         base: MTLBuffer, baseOffset: Int,
                         scale: MTLBuffer, scaleOffset: Int,
                         outPre: MTLBuffer, outPreOffset: Int = 0,
                         outPost: MTLBuffer, outPostOffset: Int = 0,
                         outComb: MTLBuffer, outCombOffset: Int = 0,
                         hcMult: Int, hidden: Int, sinkhornIters: Int,
                         hcEps: Float, rmsEps: Float) {
        precondition(hcMult <= 4, "dsv4_hc_weights sizes its mix scratch for hc_mult <= 4")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcWeightsPSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        enc.setBuffer(fn, offset: fnOffset, index: 1)
        enc.setBuffer(base, offset: baseOffset, index: 2)
        enc.setBuffer(scale, offset: scaleOffset, index: 3)
        enc.setBuffer(outPre, offset: outPreOffset, index: 4)
        enc.setBuffer(outPost, offset: outPostOffset, index: 5)
        enc.setBuffer(outComb, offset: outCombOffset, index: 6)
        var mult = UInt32(hcMult)
        var hid = UInt32(hidden)
        var iters = UInt32(sinkhornIters)
        var he = hcEps
        var re = rmsEps
        enc.setBytes(&mult, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&hid, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&iters, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&he, length: MemoryLayout<Float>.size, index: 10)
        enc.setBytes(&re, length: MemoryLayout<Float>.size, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeHCCollapse(commandBuffer: MTLCommandBuffer,
                          streams: MTLBuffer, streamsOffset: Int = 0,
                          pre: MTLBuffer, preOffset: Int = 0,
                          x: MTLBuffer, xOffset: Int = 0,
                          hcMult: Int, hidden: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcCollapsePSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        enc.setBuffer(pre, offset: preOffset, index: 1)
        enc.setBuffer(x, offset: xOffset, index: 2)
        var mult = UInt32(hcMult)
        var hid = UInt32(hidden)
        enc.setBytes(&mult, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hid, length: MemoryLayout<UInt32>.size, index: 4)
        enc.dispatchThreads(MTLSize(width: hidden, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `outStreams[k] = post[k] * sub + combᵀ @ streams` — reads `streams`,
    /// writes `outStreams` (caller ping-pongs).
    func encodeHCPlaceMix(commandBuffer: MTLCommandBuffer,
                          streams: MTLBuffer, streamsOffset: Int = 0,
                          sub: MTLBuffer, subOffset: Int = 0,
                          post: MTLBuffer, postOffset: Int = 0,
                          comb: MTLBuffer, combOffset: Int = 0,
                          outStreams: MTLBuffer, outStreamsOffset: Int = 0,
                          hcMult: Int, hidden: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcPlaceMixPSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        enc.setBuffer(sub, offset: subOffset, index: 1)
        enc.setBuffer(post, offset: postOffset, index: 2)
        enc.setBuffer(comb, offset: combOffset, index: 3)
        enc.setBuffer(outStreams, offset: outStreamsOffset, index: 4)
        var mult = UInt32(hcMult)
        var hid = UInt32(hidden)
        enc.setBytes(&mult, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hid, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreads(MTLSize(width: hidden, height: hcMult, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeHyperHead(commandBuffer: MTLCommandBuffer,
                         streams: MTLBuffer, streamsOffset: Int = 0,
                         fn: MTLBuffer, fnOffset: Int,
                         base: MTLBuffer, baseOffset: Int,
                         scale: MTLBuffer, scaleOffset: Int,
                         x: MTLBuffer,
                         hcMult: Int, hidden: Int, hcEps: Float, rmsEps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hyperHeadPSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        enc.setBuffer(fn, offset: fnOffset, index: 1)
        enc.setBuffer(base, offset: baseOffset, index: 2)
        enc.setBuffer(scale, offset: scaleOffset, index: 3)
        enc.setBuffer(x, offset: 0, index: 4)
        var mult = UInt32(hcMult)
        var hid = UInt32(hidden)
        var he = hcEps
        var re = rmsEps
        enc.setBytes(&mult, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hid, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&he, length: MemoryLayout<Float>.size, index: 7)
        enc.setBytes(&re, length: MemoryLayout<Float>.size, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeSwigluClampMul(commandBuffer: MTLCommandBuffer,
                              gate: MTLBuffer, gateOffset: Int = 0,
                              up: MTLBuffer, upOffset: Int = 0,
                              out: MTLBuffer, outOffset: Int = 0,
                              n: Int, limit: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(swigluClampPSO)
        enc.setBuffer(gate, offset: gateOffset, index: 0)
        enc.setBuffer(up, offset: upOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var count = UInt32(n)
        var lim = limit
        enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&lim, length: MemoryLayout<Float>.size, index: 4)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeBroadcastStreams(commandBuffer: MTLCommandBuffer,
                                x: MTLBuffer, xOffset: Int = 0,
                                streams: MTLBuffer, streamsOffset: Int = 0,
                                hcMult: Int, hidden: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(broadcastPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(streams, offset: streamsOffset, index: 1)
        var mult = UInt32(hcMult)
        var hid = UInt32(hidden)
        enc.setBytes(&mult, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&hid, length: MemoryLayout<UInt32>.size, index: 3)
        enc.dispatchThreads(MTLSize(width: hidden, height: hcMult, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}
