import Foundation
import Metal

/// PILOT router-lookahead for Inkling-Small: layer L+1's sigmoid router
/// applied to layer L's post-attention state, encoded into layer L's command
/// buffer so the CPU learns a *guess* at the next layer's expert set at the
/// same wake as the real router readback — no extra command buffer, no extra
/// sync. The Inkling twin of `SpeculativeRouterDSV4`.
///
/// Deliberately a separate object from `InklingKernels` rather than an extra
/// method on it: it must not share that class's `routerLogits` singleton (the
/// real router's logits are live in the same command buffer), and keeping it
/// in its own file means the speculative path cannot perturb the validated
/// decode router. Same kernels (`router_gemv_bf16_r4` +
/// `inkling_router_select`, both built without function constants exactly as
/// `InklingKernels` builds them), so the ranking — sigmoid scoring with the
/// per-expert gate bias — is bit-identical to what layer L+1 will compute for
/// real. Recall depends entirely on that fidelity; only the *input
/// activation* is an approximation.
final class SpeculativeRouterInkling {

    private let gemvPSO: MTLComputePipelineState
    private let selectPSO: MTLComputePipelineState
    private let logits: MTLBuffer
    /// Predicted expert ids, `topK` x UInt32. Read by the CPU on the router
    /// signal.
    let predictedIndices: MTLBuffer
    /// The select kernel writes routing weights and shared-expert gammas
    /// alongside the indices; the prediction never uses them, but the kernel
    /// contract needs the buffers.
    private let discardedWeights: MTLBuffer
    private let discardedGammas: MTLBuffer

    init(context: MetalContext,
         numRouted: Int,
         numShared: Int,
         topK: Int) throws {
        self.gemvPSO = try context.pipeline("router_gemv_bf16_r4")
        self.selectPSO = try context.pipeline("inkling_router_select")

        guard let logits = context.device.makeBuffer(
            length: (numRouted + numShared) * MemoryLayout<Float>.stride,
            options: .storageModeShared),
            let indices = context.device.makeBuffer(
                length: topK * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
            let weights = context.device.makeBuffer(
                length: topK * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
            let gammas = context.device.makeBuffer(
                length: max(numShared, 1) * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        logits.label = "inkling.pilot_router_logits"
        indices.label = "inkling.pilot_router_indices"
        self.logits = logits
        self.predictedIndices = indices
        self.discardedWeights = weights
        self.discardedGammas = gammas
    }

    /// Encodes the lookahead GEMV + sigmoid top-k. Must be encoded *before*
    /// the router signal so the CPU sees the prediction at the same wake as
    /// the real indices. Dispatch geometry mirrors
    /// `InklingKernels.encodeRouter` exactly.
    func encodePrediction(commandBuffer cb: MTLCommandBuffer,
                          weights: MTLBuffer, weightsOffset: Int,
                          hidden: MTLBuffer,
                          onesScale: MTLBuffer,
                          gateBias: MTLBuffer, gateBiasOffset: Int,
                          globalScale: MTLBuffer, globalScaleOffset: Int,
                          numRouted: UInt32, numShared: UInt32,
                          topK: UInt32, routeScale: Float,
                          d: UInt32) {
        var total = numRouted + numShared
        var dim = d
        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(gemvPSO)
            enc.setBuffer(weights, offset: weightsOffset, index: 0)
            enc.setBuffer(hidden, offset: 0, index: 1)
            enc.setBuffer(onesScale, offset: 0, index: 2)
            enc.setBuffer(logits, offset: 0, index: 3)
            enc.setBytes(&total, length: 4, index: 4)
            enc.setBytes(&dim, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (Int(total) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
        }

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var nr = numRouted, ns = numShared, tk = topK, rs = routeScale
        enc.setComputePipelineState(selectPSO)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(gateBias, offset: gateBiasOffset, index: 1)
        enc.setBuffer(globalScale, offset: globalScaleOffset, index: 2)
        enc.setBuffer(predictedIndices, offset: 0, index: 3)
        enc.setBuffer(discardedWeights, offset: 0, index: 4)
        enc.setBuffer(discardedGammas, offset: 0, index: 5)
        enc.setBytes(&nr, length: 4, index: 6)
        enc.setBytes(&ns, length: 4, index: 7)
        enc.setBytes(&tk, length: 4, index: 8)
        enc.setBytes(&rs, length: 4, index: 9)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }
}
