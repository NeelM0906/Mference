import Foundation
import Metal

/// The Inkling 2 runner's own kernels (`Metal/Inkling2/inkling2_runner.metal`):
/// the block-attention-residual depth mix, the cross-pass message gate, the
/// Q8_0 embedding with its norm, a weighted RMSNorm and the Q8_0 output head
/// with the logit softcap. They sit between the attention block (K3's
/// kernels) and the routed experts (K2's) in each layer visit. Every encode
/// adds dispatches to the caller's compute encoder (serial dispatch type, so
/// each dispatch sees the previous one's writes). FP32 in and out, FP16
/// logits. Checked against `Inkling2ReferenceOps`.
final class Inkling2RunnerKernels {
    static let module = "inkling2_runner"
    static let threads = 256
    static let maxStates = 8
    static let maxQueries = 4
    static let maxGateRank = 64
    static let gateTile = 16
    static let maxHidden = 4096
    static let headRowsPerThreadgroup = 8 * 4   // simdgroups x rows per simdgroup

    private let mix: MTLComputePipelineState
    private let gate: MTLComputePipelineState
    private let embed: MTLComputePipelineState
    private let norm: MTLComputePipelineState
    private let normDual: MTLComputePipelineState
    private let head: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try Inkling2ShaderLibrary.compile(device: device, module: Self.module)
        func make(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw MetalError.missingFunction(name)
            }
            return try device.makeComputePipelineState(function: function)
        }
        mix = try make("inkling2_depth_mix")
        gate = try make("inkling2_message_gate")
        embed = try make("inkling2_embed_q8_0")
        norm = try make("inkling2_rms_norm")
        normDual = try make("inkling2_rms_norm_dual")
        head = try make("inkling2_head_q8_0")
    }

    // MARK: Parameter blocks (layout-identical to the shader structs)

    struct MixParams {
        var d: UInt32
        var tokens: UInt32
        var states: UInt32
        var queries: UInt32
        var stateStride: UInt32
        var outStride: UInt32
        var outQueryStride: UInt32
        var eps: Float
        var stateOffsets: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }

    struct GateParams {
        var d: UInt32
        var tokens: UInt32
        var rank: UInt32
        var xStride: UInt32
        var mStride: UInt32
        var outStride: UInt32
        var eps: Float
    }

    struct NormParams {
        var d: UInt32
        var tokens: UInt32
        var inStride: UInt32
        var outStride: UInt32
        var rowBytes: UInt32
        var eps: Float
    }

    struct HeadParams {
        var d: UInt32
        var rows: UInt32
        var rowBytes: UInt32
        var softcap: Float
    }

    /// Epsilon of the weightless norms in the depth mix and the gate.
    static let residualNormEps: Float = 1e-6

    // MARK: Encoders

    /// Depth mix over `stateOffsets.count` states (float offsets of each
    /// state's token row 0 in `arena`, rows `stateStride` floats apart) with
    /// `queries` (`[queryCount][d]` F32 at `queriesOffset` bytes). Output `q`
    /// of token `t` goes to `out` at float `q * outQueryStride + t * outStride`.
    func encodeDepthMix(_ encoder: MTLComputeCommandEncoder,
                        arena: MTLBuffer, stateOffsets: [Int], stateStride: Int,
                        queries: MTLBuffer, queriesOffset: Int = 0, queryCount: Int,
                        out: MTLBuffer, outOffset: Int = 0, outStride: Int, outQueryStride: Int,
                        d: Int, tokens: Int) {
        precondition(!stateOffsets.isEmpty && stateOffsets.count <= Self.maxStates)
        precondition(queryCount >= 1 && queryCount <= Self.maxQueries)
        var o = [UInt32](repeating: 0, count: Self.maxStates)
        for (i, offset) in stateOffsets.enumerated() { o[i] = UInt32(offset) }
        var p = MixParams(d: UInt32(d), tokens: UInt32(tokens), states: UInt32(stateOffsets.count),
                          queries: UInt32(queryCount), stateStride: UInt32(stateStride),
                          outStride: UInt32(outStride), outQueryStride: UInt32(outQueryStride),
                          eps: Self.residualNormEps,
                          stateOffsets: (o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]))
        encoder.setComputePipelineState(mix)
        encoder.setBuffer(arena, offset: 0, index: 0)
        encoder.setBuffer(queries, offset: queriesOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBytes(&p, length: MemoryLayout<MixParams>.stride, index: 3)
        dispatchPerToken(encoder, tokens: tokens)
    }

    /// Message gate of one controller: `down` `[rank][2d]`, `gain` and
    /// `forget` `[d][rank]`, all F32.
    func encodeMessageGate(_ encoder: MTLComputeCommandEncoder,
                           x: MTLBuffer, xOffset: Int = 0, xStride: Int,
                           message: MTLBuffer, messageOffset: Int = 0, messageStride: Int,
                           down: MTLBuffer, downOffset: Int,
                           gain: MTLBuffer, gainOffset: Int,
                           forget: MTLBuffer, forgetOffset: Int,
                           rank: Int,
                           out: MTLBuffer, outOffset: Int = 0, outStride: Int,
                           d: Int, tokens: Int) {
        precondition(rank >= 1 && rank <= Self.maxGateRank && (2 * d) % 64 == 0)
        var p = GateParams(d: UInt32(d), tokens: UInt32(tokens), rank: UInt32(rank),
                           xStride: UInt32(xStride), mStride: UInt32(messageStride),
                           outStride: UInt32(outStride), eps: Self.residualNormEps)
        encoder.setComputePipelineState(gate)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(message, offset: messageOffset, index: 1)
        encoder.setBuffer(down, offset: downOffset, index: 2)
        encoder.setBuffer(gain, offset: gainOffset, index: 3)
        encoder.setBuffer(forget, offset: forgetOffset, index: 4)
        encoder.setBuffer(out, offset: outOffset, index: 5)
        encoder.setBytes(&p, length: MemoryLayout<GateParams>.stride, index: 6)
        let tiles = (tokens + Self.gateTile - 1) / Self.gateTile
        encoder.dispatchThreadgroups(MTLSize(width: tiles, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Self.threads, height: 1, depth: 1))
    }

    /// Embedding rows of `ids` (`tokens` UInt32) from a Q8_0 table of
    /// `rowBytes`-byte rows, then the weighted RMSNorm (`weight` F32 `[d]`).
    func encodeEmbedding(_ encoder: MTLComputeCommandEncoder,
                         ids: MTLBuffer, idsOffset: Int = 0,
                         table: MTLBuffer, tableOffset: Int, rowBytes: Int,
                         weight: MTLBuffer, weightOffset: Int, eps: Float,
                         out: MTLBuffer, outOffset: Int = 0, outStride: Int,
                         d: Int, tokens: Int) {
        precondition(d % 32 == 0 && rowBytes == d / 32 * 34 && tableOffset % 2 == 0)
        var p = NormParams(d: UInt32(d), tokens: UInt32(tokens), inStride: 0,
                           outStride: UInt32(outStride), rowBytes: UInt32(rowBytes), eps: eps)
        encoder.setComputePipelineState(embed)
        encoder.setBuffer(ids, offset: idsOffset, index: 0)
        encoder.setBuffer(table, offset: tableOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        encoder.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: 4)
        dispatchPerToken(encoder, tokens: tokens)
    }

    /// Weighted RMSNorm of `tokens` FP32 rows.
    func encodeRMSNorm(_ encoder: MTLComputeCommandEncoder,
                       x: MTLBuffer, xOffset: Int = 0, xStride: Int,
                       weight: MTLBuffer, weightOffset: Int, eps: Float,
                       out: MTLBuffer, outOffset: Int = 0, outStride: Int,
                       d: Int, tokens: Int) {
        var p = NormParams(d: UInt32(d), tokens: UInt32(tokens), inStride: UInt32(xStride),
                           outStride: UInt32(outStride), rowBytes: 0, eps: eps)
        encoder.setComputePipelineState(norm)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: 3)
        dispatchPerToken(encoder, tokens: tokens)
    }

    /// Weighted RMSNorm writing FP32 (`out`) and FP16 (`outHalf`, rows
    /// `halfStride` halves apart) copies, the MoE input of the router and of
    /// the grouped experts.
    func encodeRMSNormDual(_ encoder: MTLComputeCommandEncoder,
                           x: MTLBuffer, xOffset: Int = 0, xStride: Int,
                           weight: MTLBuffer, weightOffset: Int, eps: Float,
                           out: MTLBuffer, outOffset: Int = 0, outStride: Int,
                           outHalf: MTLBuffer, outHalfOffset: Int = 0, halfStride: Int,
                           d: Int, tokens: Int) {
        var p = NormParams(d: UInt32(d), tokens: UInt32(tokens), inStride: UInt32(xStride),
                           outStride: UInt32(outStride), rowBytes: 0, eps: eps)
        var hs = UInt32(halfStride)
        encoder.setComputePipelineState(normDual)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBuffer(outHalf, offset: outHalfOffset, index: 3)
        encoder.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: 4)
        encoder.setBytes(&hs, length: 4, index: 5)
        dispatchPerToken(encoder, tokens: tokens)
    }

    /// FP16 logits of one normalized FP32 row `x` (`d` values) against the
    /// first `rows` rows of a Q8_0 head, with the logit softcap (0 disables).
    func encodeHead(_ encoder: MTLComputeCommandEncoder,
                    x: MTLBuffer, xOffset: Int = 0,
                    table: MTLBuffer, tableOffset: Int, rowBytes: Int, rows: Int,
                    softcap: Float, logits: MTLBuffer, logitsOffset: Int = 0, d: Int) {
        precondition(d % 32 == 0 && d <= Self.maxHidden && rowBytes == d / 32 * 34 && tableOffset % 2 == 0)
        var p = HeadParams(d: UInt32(d), rows: UInt32(rows), rowBytes: UInt32(rowBytes), softcap: softcap)
        encoder.setComputePipelineState(head)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(table, offset: tableOffset, index: 1)
        encoder.setBuffer(logits, offset: logitsOffset, index: 2)
        encoder.setBytes(&p, length: MemoryLayout<HeadParams>.stride, index: 3)
        let groups = (rows + Self.headRowsPerThreadgroup - 1) / Self.headRowsPerThreadgroup
        encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Self.threads, height: 1, depth: 1))
    }

    private func dispatchPerToken(_ encoder: MTLComputeCommandEncoder, tokens: Int) {
        encoder.dispatchThreadgroups(MTLSize(width: tokens, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Self.threads, height: 1, depth: 1))
    }
}
