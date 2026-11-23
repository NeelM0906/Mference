import Foundation
import Metal

/// The MoE router for one token (decode), in two small dispatches: the logits
/// `W x` (F32 `[experts][width]`) by four simdgroups per expert row, then the
/// selection of K2's `inkling2_moe_router` for one token (sigmoid of the
/// clamped logit, top-k of sigmoid + bias with ties to the lower expert,
/// weights sigmoid / (sum + 1e-20) times `routeScale`). The prefill router tiles
/// 32 tokens on one threadgroup, which at one token streams the 1.2 MB matrix
/// through a single threadgroup; this form reads it with all of them.
///
/// The ids and weights it writes have the prefill router's layout for one
/// token (`[topK]` each). The logits are summed in another order than the
/// prefill router's, so a near tie between two experts can resolve the other way.
public final class Inkling2DecodeRouter {
    private let logitsPipeline: MTLComputePipelineState
    private let selectPipeline: MTLComputePipelineState

    public static let maxExperts = 128

    public init(library: Inkling2KernelLibrary) throws {
        logitsPipeline = try library.pipeline("inkling2_router_logits_decode")
        selectPipeline = try library.pipeline("inkling2_router_select_decode")
    }

    /// - `x`: the normed MoE input, F32 `[width]`.
    /// - `weight`: `ffn_gate_inp`, F32 `[experts][width]`; `bias`: `exp_probs_b`, F32 `[experts]`.
    /// - `logits`: F32 scratch of `experts` floats.
    /// - `ids` (UInt32) and `routeWeights` (F32): `[topK]` each.
    public func encode(commandBuffer: MTLCommandBuffer, x: Inkling2BufferRef, weight: Inkling2BufferRef,
                       bias: Inkling2BufferRef, logits: Inkling2BufferRef, ids: Inkling2BufferRef,
                       routeWeights: Inkling2BufferRef, experts: Int = 128, width: Int = Inkling2AttentionShape.hidden,
                       topK: Int = 8, routeScale: Float = 8) {
        commandBuffer.inkling2Compute {
            encode(encoder: $0, x: x, weight: weight, bias: bias, logits: logits, ids: ids, routeWeights: routeWeights,
                   experts: experts, width: width, topK: topK, routeScale: routeScale)
        }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encode(encoder: MTLComputeCommandEncoder, x: Inkling2BufferRef, weight: Inkling2BufferRef,
                       bias: Inkling2BufferRef, logits: Inkling2BufferRef, ids: Inkling2BufferRef,
                       routeWeights: Inkling2BufferRef, experts: Int = 128, width: Int = Inkling2AttentionShape.hidden,
                       topK: Int = 8, routeScale: Float = 8) {
        precondition((1...Self.maxExperts).contains(experts) && (1...32).contains(topK) && topK <= experts)
        precondition(width > 0 && width % 16 == 0 && x.offset % 16 == 0 && weight.offset % 16 == 0)
        precondition(logits.buffer.length - logits.offset >= experts * 4)
        var params = Inkling2RouterDecodeParams(width: UInt32(width), experts: UInt32(experts),
                                                topK: UInt32(topK), routeScale: routeScale)
        encoder.setComputePipelineState(logitsPipeline)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2RouterDecodeParams>.stride, index: 0)
        encoder.setBuffer(x.buffer, offset: x.offset, index: 1)
        encoder.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        encoder.setBuffer(logits.buffer, offset: logits.offset, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: (experts + 1) / 2, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.setComputePipelineState(selectPipeline)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2RouterDecodeParams>.stride, index: 0)
        encoder.setBuffer(logits.buffer, offset: logits.offset, index: 1)
        encoder.setBuffer(bias.buffer, offset: bias.offset, index: 2)
        encoder.setBuffer(ids.buffer, offset: ids.offset, index: 3)
        encoder.setBuffer(routeWeights.buffer, offset: routeWeights.offset, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}

/// Mirrors `Inkling2RouterDecodeParams` in `inkling2_attention.metal`.
struct Inkling2RouterDecodeParams {
    var width: UInt32
    var experts: UInt32
    var topK: UInt32
    var routeScale: Float
}
