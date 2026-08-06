import Foundation
import Metal

public enum SharedExpertError: Error, CustomStringConvertible {
    case unsupportedWeightBits(Int)
    case dimensionMismatch(String)
    case scratchTooSmall(String)

    public var description: String {
        switch self {
        case .unsupportedWeightBits(let bits):
            return "SharedExpert unsupported weight bits: \(bits)"
        case .dimensionMismatch(let detail):
            return "SharedExpert dimension mismatch: \(detail)"
        case .scratchTooSmall(let detail):
            return "SharedExpert scratch too small: \(detail)"
        }
    }
}

public final class SharedExpertInt4 {
    private let int4: DequantInt4GEMV
    private let geluMulPSO: MTLComputePipelineState
    private let fusedGateUpActPSO: MTLComputePipelineState?
    private let specializedFusedGateUpActPSO: MTLComputePipelineState?
    private let specializedD: UInt32?
    private let specializedF: UInt32?

    public init(context: MetalContext,
                siluActivation: Bool = false,
                useFusedGateUp: Bool = true,
                specializedD: Int? = nil,
                specializedF: Int? = nil) throws {
        let specializedShapes: [(m: Int, n: Int)]
        if let d = specializedD, let f = specializedF {
            specializedShapes = [(m: f, n: d), (m: d, n: f)]
        } else {
            specializedShapes = []
        }
        self.int4 = try DequantInt4GEMV(context: context,
                                        additionalShapes: specializedShapes)
        self.geluMulPSO = try context.pipeline(
            siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16")
        let activationConstants = [
            MetalFunctionConstant(index: 27, value: .bool(siluActivation)),
        ]
        self.fusedGateUpActPSO = useFusedGateUp
            ? try context.pipeline("shared_int4_gate_up_act_simd",
                                   constants: activationConstants,
                                   maxTotalThreadsPerThreadgroup: 256)
            : nil
        if useFusedGateUp, let d = specializedD, let f = specializedF {
            self.specializedFusedGateUpActPSO = try context.pipeline(
                "shared_int4_gate_up_act_simd",
                constants: activationConstants + [
                    MetalFunctionConstant(index: 20, value: .uint32(UInt32(f))),
                    MetalFunctionConstant(index: 21, value: .uint32(UInt32(d))),
                    MetalFunctionConstant(index: 22, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 256)
            self.specializedD = UInt32(d)
            self.specializedF = UInt32(f)
        } else {
            self.specializedFusedGateUpActPSO = nil
            self.specializedD = nil
            self.specializedF = nil
        }
    }

    public func encode(commandBuffer cb: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertProjection,
                       up: SharedExpertProjection,
                       down: SharedExpertProjection,
                       y: MTLBuffer, yOffset: Int = 0,
                       scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
                       scratchUp: MTLBuffer, scratchUpOffset: Int = 0,
                       scratchAct: MTLBuffer, scratchActOffset: Int = 0,
                       outputFloat32: Bool = false) throws {
        guard gate.rows == up.rows, gate.cols == up.cols,
              down.rows == gate.cols, down.cols == gate.rows else {
            throw SharedExpertError.dimensionMismatch(
                "gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols)) down=(\(down.rows),\(down.cols))")
        }
        let intermediate = Int(gate.rows)
        let required = intermediate * MemoryLayout<Float16>.stride
        guard scratchGateOffset >= 0, scratchGateOffset + required <= scratchGate.length,
              scratchUpOffset >= 0, scratchUpOffset + required <= scratchUp.length,
              scratchActOffset >= 0, scratchActOffset + required <= scratchAct.length else {
            throw SharedExpertError.scratchTooSmall("need \(required) bytes per intermediate buffer")
        }
        let inputBytes = Int(gate.cols) * MemoryLayout<Float16>.stride
        let outputBytes = Int(down.rows) * (outputFloat32
            ? MemoryLayout<Float>.stride : MemoryLayout<Float16>.stride)
        guard xOffset >= 0, xOffset + inputBytes <= x.length else {
            throw SharedExpertError.scratchTooSmall("input range exceeds x buffer")
        }
        guard yOffset >= 0, yOffset + outputBytes <= y.length else {
            throw SharedExpertError.scratchTooSmall("output range exceeds y buffer")
        }

        let selectedFusedGateUpActPSO: MTLComputePipelineState?
        if gate.rows == specializedF, gate.cols == specializedD {
            selectedFusedGateUpActPSO = specializedFusedGateUpActPSO
                ?? fusedGateUpActPSO
        } else {
            selectedFusedGateUpActPSO = fusedGateUpActPSO
        }
        if let selectedFusedGateUpActPSO {
            guard let encoder = cb.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(selectedFusedGateUpActPSO)
            encoder.setBuffer(gate.weights, offset: gate.weightsOffset, index: 0)
            encoder.setBuffer(gate.scales, offset: gate.scalesOffset, index: 1)
            encoder.setBuffer(gate.biases, offset: gate.biasesOffset, index: 2)
            encoder.setBuffer(up.weights, offset: up.weightsOffset, index: 3)
            encoder.setBuffer(up.scales, offset: up.scalesOffset, index: 4)
            encoder.setBuffer(up.biases, offset: up.biasesOffset, index: 5)
            encoder.setBuffer(x, offset: xOffset, index: 6)
            encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 7)
            var rows = gate.rows
            var columns = gate.cols
            encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 8)
            encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 9)
            encoder.dispatchThreadgroups(
                MTLSize(width: (intermediate + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        } else {
            int4.encode(commandBuffer: cb,
                        weights: gate.weights, weightsOffset: gate.weightsOffset,
                        scales: gate.scales, scalesOffset: gate.scalesOffset,
                        biases: gate.biases, biasesOffset: gate.biasesOffset,
                        x: x, xOffset: xOffset,
                        y: scratchGate, yOffset: scratchGateOffset,
                        m: gate.rows, n: gate.cols)
            int4.encode(commandBuffer: cb,
                        weights: up.weights, weightsOffset: up.weightsOffset,
                        scales: up.scales, scalesOffset: up.scalesOffset,
                        biases: up.biases, biasesOffset: up.biasesOffset,
                        x: x, xOffset: xOffset,
                        y: scratchUp, yOffset: scratchUpOffset,
                        m: up.rows, n: up.cols)

            guard let encoder = cb.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(geluMulPSO)
            encoder.setBuffer(scratchGate, offset: scratchGateOffset, index: 0)
            encoder.setBuffer(scratchUp, offset: scratchUpOffset, index: 1)
            encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 2)
            var count = UInt32(intermediate)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            let width = min(geluMulPSO.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(MTLSize(width: intermediate, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            encoder.endEncoding()
        }

        int4.encode(commandBuffer: cb,
                    weights: down.weights, weightsOffset: down.weightsOffset,
                    scales: down.scales, scalesOffset: down.scalesOffset,
                    biases: down.biases, biasesOffset: down.biasesOffset,
                    x: scratchAct, xOffset: scratchActOffset,
                    y: y, yOffset: yOffset,
                    m: down.rows, n: down.cols,
                    outputFloat32: outputFloat32)
    }

}

public final class SharedExpertRuntime {
    private enum Implementation {
        case int4(SharedExpertInt4)
        case int8(SharedExpertInt8)
    }

    private let implementation: Implementation
    public let weightBits: Int

    public init(context: MetalContext, weightBits: Int,
                siluActivation: Bool = false,
                specializedD: Int? = nil,
                specializedF: Int? = nil) throws {
        self.weightBits = weightBits
        switch weightBits {
        case 4: self.implementation = .int4(try SharedExpertInt4(
            context: context, siluActivation: siluActivation,
            specializedD: specializedD, specializedF: specializedF))
        case 8: self.implementation = .int8(try SharedExpertInt8(
            context: context, siluActivation: siluActivation))
        default: throw SharedExpertError.unsupportedWeightBits(weightBits)
        }
    }

    public func encode(commandBuffer: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertProjection,
                       up: SharedExpertProjection,
                       down: SharedExpertProjection,
                       y: MTLBuffer, yOffset: Int = 0,
                       scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
                       scratchUp: MTLBuffer, scratchUpOffset: Int = 0,
                       scratchAct: MTLBuffer, scratchActOffset: Int = 0,
                       outputFloat32: Bool = false) throws {
        switch implementation {
        case .int4(let runtime):
            try runtime.encode(commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                               scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset,
                               outputFloat32: outputFloat32)
        case .int8(let runtime):
            guard !outputFloat32 else {
                throw SharedExpertError.dimensionMismatch(
                    "FP32 output is only implemented for the INT4 shared expert")
            }
            try runtime.encode(commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        }
    }

}
