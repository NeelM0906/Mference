import Foundation
import Metal

/// Per-expert batched GLU for Inkling prefill.
///
/// Replaces the v1 pattern of one 3-GEMV dispatch per (token, expert) pair with
/// one dispatch pair per expert covering every token routed to it. Worth 1.11x
/// on a 2 785-token prompt; the kernel comment in
/// `Metal/Prefill/prefill.metal` records why the ceiling is low (the expert
/// loop is ~15 % of Inkling prefill), and `docs/INKLING_SMALL.md` explains why
/// the accumulator has to be FP32.
///
/// Both routed and shared experts go through this: a shared expert is just an
/// "expert" whose pair list is every chunk token, weighted by its per-token
/// gamma.
final class InklingPrefillExpertGLU {

    /// Mirrors `InklingPrefillExpertParamsMSL`.
    struct Params {
        var d: UInt32
        var f: UInt32
        var pairStart: UInt32
        var pairCount: UInt32
        var hiddenStride: UInt32
        var gateWOff: UInt32, gateSOff: UInt32, gateBOff: UInt32
        var upWOff: UInt32,   upSOff: UInt32,   upBOff: UInt32
        var downWOff: UInt32, downSOff: UInt32, downBOff: UInt32
    }

    private let actPSO: MTLComputePipelineState
    private let downPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        // Inkling's hidden activation is SiLU; the shared prefill helper
        // defaults to GELU unless function constant 77 says otherwise.
        self.actPSO = try context.pipeline(
            "inkling_prefill_expert_act",
            constants: [MetalFunctionConstant(index: 77, value: .bool(true))])
        self.downPSO = try context.pipeline("inkling_prefill_expert_down_accum")
    }

    /// Encodes gate/up/activation then down + weighted accumulate for one
    /// expert. `expertBuffer` may be a streamed routed blob or the resident
    /// tensor holding a shared expert; `params` carries the byte offsets either
    /// way. `acc` is FP32 [tokens, d] and is accumulated into, not overwritten.
    func encode(commandBuffer cb: MTLCommandBuffer,
                hidden: MTLBuffer,
                hiddenOffset: Int = 0,
                pairTokens: MTLBuffer,
                pairWeights: MTLBuffer,
                expertBuffer: MTLBuffer,
                expertOffset: Int,
                act: MTLBuffer,
                acc: MTLBuffer,
                params: Params) {
        guard params.pairCount > 0 else { return }
        var p = params

        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(actPSO)
            enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
            enc.setBuffer(pairTokens, offset: 0, index: 1)
            enc.setBuffer(expertBuffer, offset: expertOffset, index: 2)
            enc.setBuffer(act, offset: 0, index: 3)
            enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 4)
            enc.dispatchThreads(
                MTLSize(width: Int(p.f), height: Int(p.pairCount), depth: 1),
                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc.endEncoding()
        }

        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(downPSO)
            enc.setBuffer(act, offset: 0, index: 0)
            enc.setBuffer(pairTokens, offset: 0, index: 1)
            enc.setBuffer(pairWeights, offset: 0, index: 2)
            enc.setBuffer(expertBuffer, offset: expertOffset, index: 3)
            enc.setBuffer(acc, offset: 0, index: 4)
            enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 5)
            enc.dispatchThreads(
                MTLSize(width: Int(p.d), height: Int(p.pairCount), depth: 1),
                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc.endEncoding()
        }
    }
}
