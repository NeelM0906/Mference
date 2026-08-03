import Foundation
import Metal

/// PILOT router-lookahead for DeepSeek-V4: layer L+1's router applied to layer
/// L's post-attention state, encoded into layer L's command buffer so the CPU
/// learns a *guess* at the next layer's expert set at the same wake as the real
/// router readback — no extra command buffer, no extra sync.
///
/// Deliberately a separate object from `MoEDeepseekV4` rather than an extra
/// method on it: it must not share that class's `routerLogits` singleton (the
/// real router's logits are live in the same command buffer), and keeping it in
/// its own file means the speculative path cannot perturb the validated decode
/// router. Same kernels and same function constants as the real router, so the
/// ranking — sqrtsoftplus scoring, correction-bias-adjusted selection — is
/// bit-identical to what layer L+1 will compute for real. Recall depends
/// entirely on that fidelity; only the *input activation* is an approximation.
final class SpeculativeRouterDSV4 {

    private let gemvPSO: MTLComputePipelineState
    private let selectPSO: MTLComputePipelineState
    private let logits: MTLBuffer
    /// Predicted expert ids, `topK` x UInt32. Read by the CPU on the router
    /// signal.
    let predictedIndices: MTLBuffer
    /// The select kernel writes gate weights alongside the indices; the
    /// prediction never uses them, but the kernel contract needs the buffer.
    private let discardedWeights: MTLBuffer

    init(context: MetalContext,
         numExperts: UInt32,
         d: UInt32,
         topK: UInt32) throws {
        let routerConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 40, value: .uint32(numExperts)),
            MetalFunctionConstant(index: 41, value: .uint32(d)),
            MetalFunctionConstant(index: 42, value: .uint32(topK)),
            MetalFunctionConstant(index: 43, value: .bool(true)),
        ]
        self.gemvPSO = try context.pipeline(
            "router_gemv_bf16_r4",
            constants: routerConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.selectPSO = try context.pipeline(
            "router_topk_select_sqrtsoftplus_k6_par",
            constants: routerConstants)

        guard let logits = context.device.makeBuffer(
            length: Int(numExperts) * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let indices = context.device.makeBuffer(
                length: Int(topK) * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let weights = context.device.makeBuffer(
                length: Int(topK) * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        logits.label = "pilot_router_logits"
        indices.label = "pilot_router_indices"
        self.logits = logits
        self.predictedIndices = indices
        self.discardedWeights = weights
    }

    /// Encodes the lookahead GEMV + top-k. Must be encoded *before* the router
    /// signal so the CPU sees the prediction at the same wake as the real
    /// indices.
    func encodePrediction(commandBuffer: MTLCommandBuffer,
                          weights: MTLBuffer, weightsOffset: Int,
                          hidden: MTLBuffer,
                          onesScale: MTLBuffer,
                          correctionBias: MTLBuffer, correctionBiasOffset: Int,
                          numExperts: UInt32,
                          d: UInt32,
                          routeScale: Float) {
        var expertCount = numExperts
        var dimension = d
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(gemvPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(hidden, offset: 0, index: 1)
            encoder.setBuffer(onesScale, offset: 0, index: 2)
            encoder.setBuffer(logits, offset: 0, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }

        var scale = routeScale
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(selectPSO)
            encoder.setBuffer(logits, offset: 0, index: 0)
            encoder.setBuffer(correctionBias, offset: correctionBiasOffset, index: 1)
            encoder.setBuffer(predictedIndices, offset: 0, index: 2)
            encoder.setBuffer(discardedWeights, offset: 0, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }
}
