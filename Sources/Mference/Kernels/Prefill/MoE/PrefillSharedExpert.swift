import Foundation
import Metal

final class PrefillSharedExpert {
    enum BlockPath: Sendable, Equatable {
        case repeatedRows
        case tensorOpsInt4
    }

    private let shared: SharedExpertRuntime
    private let weightBits: Int
    private let tensorOpsInt4: MPPPrefillInt4QMM
    private let blockActivationPSO: MTLComputePipelineState
    private let qwenScalarGatePSO: MTLComputePipelineState

    init(context: MetalContext, weightBits: Int = 8, siluActivation: Bool = false) throws {
        self.weightBits = weightBits
        self.shared = try SharedExpertRuntime(context: context,
                                              weightBits: weightBits,
                                              siluActivation: siluActivation)
        self.tensorOpsInt4 = MPPPrefillInt4QMM(context: context)
        self.blockActivationPSO = try context.pipeline(
            siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16")
        self.qwenScalarGatePSO = try context.pipeline("prefill_qwen_shared_scalar_gate")
    }

    func encodeQwenScalarGate(commandBuffer cb: MTLCommandBuffer,
                              x: MTLBuffer,
                              xOffset: Int = 0,
                              gate: SharedExpertInt8Proj,
                              y: MTLBuffer,
                              yOffset: Int = 0,
                              queryCount: Int,
                              d: Int,
                              xStrideElements: Int,
                              yStrideElements: Int) throws {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(d > 0 && d.isMultiple(of: Quantization.groupSize),
                     "d must be a positive INT8 group multiple")
        precondition(xStrideElements >= d, "x stride is too small")
        precondition(yStrideElements >= d, "y stride is too small")
        guard gate.rows == 1, gate.cols == UInt32(d) else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "expected scalar gate=(1,\(d)), got (\(gate.rows),\(gate.cols))")
        }
        guard queryCount > 0 else { return }

        let halfBytes = MemoryLayout<Float16>.stride
        let xEnd = xOffset + ((queryCount - 1) * xStrideElements + d) * halfBytes
        let yEnd = yOffset + ((queryCount - 1) * yStrideElements + d) * halfBytes
        guard xOffset >= 0, xEnd <= x.length else {
            throw SharedExpertInt8Error.scratchTooSmall(
                "scalar gate x range ends at \(xEnd), buffer length \(x.length)")
        }
        guard yOffset >= 0, yEnd <= y.length else {
            throw SharedExpertInt8Error.scratchTooSmall(
                "scalar gate y range ends at \(yEnd), buffer length \(y.length)")
        }
        guard let encoder = cb.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(qwenScalarGatePSO)
        encoder.setBuffer(gate.weights, offset: gate.weightsOffset, index: 0)
        encoder.setBuffer(gate.scales, offset: gate.scalesOffset, index: 1)
        encoder.setBuffer(gate.biases, offset: gate.biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var rows = UInt32(queryCount)
        var dimension = UInt32(d)
        var xStride = UInt32(xStrideElements)
        var yStride = UInt32(yStrideElements)
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&xStride, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&yStride, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (queryCount + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    @discardableResult
    func encodeBlock(commandBuffer cb: MTLCommandBuffer,
                            x: MTLBuffer,
                            xOffset: Int = 0,
                            y: MTLBuffer,
                            yOffset: Int = 0,
                            gate: SharedExpertInt8Proj,
                            up: SharedExpertInt8Proj,
                            down: SharedExpertInt8Proj,
                            scratchGate: MTLBuffer,
                            scratchGateOffset: Int = 0,
                            scratchUp: MTLBuffer,
                            scratchUpOffset: Int = 0,
                            scratchAct: MTLBuffer,
                            scratchActOffset: Int = 0,
                            queryCount: Int,
                            d: Int,
                            intermediate: Int,
                            xStrideElements: Int,
                            yStrideElements: Int) throws -> BlockPath {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(d > 0, "d must be positive")
        precondition(intermediate > 0, "intermediate must be positive")
        precondition(xStrideElements >= d, "x stride is too small")
        precondition(yStrideElements >= d, "y stride is too small")
        guard gate.rows == UInt32(intermediate), gate.cols == UInt32(d),
              up.rows == UInt32(intermediate), up.cols == UInt32(d),
              down.rows == UInt32(d), down.cols == UInt32(intermediate) else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "expected gate/up=(\(intermediate),\(d)) down=(\(d),\(intermediate))")
        }

        let halfBytes = MemoryLayout<Float16>.stride
        let batchIntermediateBytes = queryCount * intermediate * halfBytes
        let canBatchInt4 = weightBits == 4
            && queryCount >= MPPPrefillInt4QMM.tileN
            && tensorOpsInt4.isAvailable
            && xStrideElements == d
            && yStrideElements == d
            && scratchGateOffset >= 0
            && scratchGateOffset + batchIntermediateBytes <= scratchGate.length
            && scratchUpOffset >= 0
            && scratchUpOffset + batchIntermediateBytes <= scratchUp.length
            && scratchActOffset >= 0
            && scratchActOffset + batchIntermediateBytes <= scratchAct.length
        if canBatchInt4 {
            let gatePath = tensorOpsInt4.encode(
                commandBuffer: cb,
                weights: gate.weights, weightsOffset: gate.weightsOffset,
                scales: gate.scales, scalesOffset: gate.scalesOffset,
                biases: gate.biases, biasesOffset: gate.biasesOffset,
                x: x, xOffset: xOffset,
                y: scratchGate, yOffset: scratchGateOffset,
                m: queryCount, n: intermediate, k: d)
            let upPath = tensorOpsInt4.encode(
                commandBuffer: cb,
                weights: up.weights, weightsOffset: up.weightsOffset,
                scales: up.scales, scalesOffset: up.scalesOffset,
                biases: up.biases, biasesOffset: up.biasesOffset,
                x: x, xOffset: xOffset,
                y: scratchUp, yOffset: scratchUpOffset,
                m: queryCount, n: intermediate, k: d)
            precondition(gatePath == .affineThreadgroupF16
                            && upPath == .affineThreadgroupF16,
                         "validated TensorOps shared-expert projection became unavailable")

            guard let activation = cb.makeComputeCommandEncoder() else {
                throw SharedExpertInt8Error.dimensionMismatch("encoder allocation failed")
            }
            activation.setComputePipelineState(blockActivationPSO)
            activation.setBuffer(scratchGate, offset: scratchGateOffset, index: 0)
            activation.setBuffer(scratchUp, offset: scratchUpOffset, index: 1)
            activation.setBuffer(scratchAct, offset: scratchActOffset, index: 2)
            var count = UInt32(queryCount * intermediate)
            activation.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            let width = min(blockActivationPSO.maxTotalThreadsPerThreadgroup, 256)
            activation.dispatchThreads(
                MTLSize(width: Int(count), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            activation.endEncoding()

            let downPath = tensorOpsInt4.encode(
                commandBuffer: cb,
                weights: down.weights, weightsOffset: down.weightsOffset,
                scales: down.scales, scalesOffset: down.scalesOffset,
                biases: down.biases, biasesOffset: down.biasesOffset,
                x: scratchAct, xOffset: scratchActOffset,
                y: y, yOffset: yOffset,
                m: queryCount, n: d, k: intermediate)
            precondition(downPath == .affineThreadgroupF16,
                         "validated TensorOps shared-expert down projection became unavailable")
            return .tensorOpsInt4
        }

        for row in 0..<queryCount {
            try shared.encode(commandBuffer: cb,
                              x: x,
                              xOffset: xOffset + row * xStrideElements * halfBytes,
                              gate: gate,
                              up: up,
                              down: down,
                              y: y,
                              yOffset: yOffset + row * yStrideElements * halfBytes,
                              scratchGate: scratchGate,
                              scratchGateOffset: scratchGateOffset,
                              scratchUp: scratchUp,
                              scratchUpOffset: scratchUpOffset,
                              scratchAct: scratchAct,
                              scratchActOffset: scratchActOffset)
        }
        return .repeatedRows
    }
}
