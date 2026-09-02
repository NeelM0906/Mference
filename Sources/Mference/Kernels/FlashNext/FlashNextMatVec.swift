import Foundation
import Metal

/// A resident Flash-Next projection matrix, in whichever of the two dtypes the
/// install carries it.
///
/// The production `qwen38flashnext` install quantizes every rank-2 BF16
/// `.weight` whose row length the group size divides to INT4 affine group-64;
/// norms, conv kernels and any row width 64 does not divide ride through as
/// BF16. The parity install is BF16 for *everything*, so the same runner has to
/// drive both — hence one type that carries the dtype with the buffer rather
/// than a runner-wide assumption.
enum FlashNextWeightMatrix {
    case int4(weights: MTLBuffer, weightsOffset: Int,
              scales: MTLBuffer, scalesOffset: Int,
              biases: MTLBuffer, biasesOffset: Int)
    case bf16(buffer: MTLBuffer, offset: Int)

    /// Build from a loaded tensor view. Dtype 0 is INT4 affine with companion
    /// scale/bias slices; dtype 1 is dense BF16.
    static func from(_ view: TensorView) -> FlashNextWeightMatrix {
        switch view.dtype {
        case 0:
            return .int4(weights: view.buffer, weightsOffset: Int(view.offset),
                         scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                         biases: view.buffer, biasesOffset: Int(view.biasOffset))
        case 1:
            return .bf16(buffer: view.buffer, offset: Int(view.offset))
        default:
            preconditionFailure(
                "Flash-Next projections are INT4 affine or BF16, got dtype \(view.dtype)")
        }
    }
}

/// `y = W . x` for a Flash-Next projection, dispatching on the stored dtype.
///
/// INT4 goes through the shipped `DequantInt4GEMV`; BF16 through this family's
/// own `flashnext_gemv_bf16`. Both have an FP32-output form, which the
/// hyper-connection path uses wherever a value is about to be pushed through a
/// sigmoid — rounding a pre-activation to FP16 costs more than the buffer saves.
final class FlashNextMatVec {

    private let int4: DequantInt4GEMV
    private let bf16PSO: MTLComputePipelineState
    private let bf16F32PSO: MTLComputePipelineState

    /// Rows per threadgroup; mirrors `kFlashNextGemvRowsPerThreadgroup`.
    private static let rowsPerThreadgroup = 8

    init(context: MetalContext, int4: DequantInt4GEMV) throws {
        self.int4 = int4
        self.bf16PSO = try context.pipeline("flashnext_gemv_bf16",
                                            constants: [],
                                            maxTotalThreadsPerThreadgroup: 256)
        self.bf16F32PSO = try context.pipeline("flashnext_gemv_bf16_f32out",
                                               constants: [],
                                               maxTotalThreadsPerThreadgroup: 256)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                matrix: FlashNextWeightMatrix,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                rows: Int, cols: Int,
                outputFloat32: Bool = false) {
        precondition(rows > 0 && cols > 0)
        switch matrix {
        case let .int4(weights, weightsOffset, scales, scalesOffset,
                       biases, biasesOffset):
            int4.encode(commandBuffer: commandBuffer,
                        weights: weights, weightsOffset: weightsOffset,
                        scales: scales, scalesOffset: scalesOffset,
                        biases: biases, biasesOffset: biasesOffset,
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                        m: UInt32(rows), n: UInt32(cols),
                        outputFloat32: outputFloat32)
        case let .bf16(buffer, offset):
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
            let pso = outputFloat32 ? bf16F32PSO : bf16PSO
            enc.setComputePipelineState(pso)
            enc.setBuffer(buffer, offset: offset, index: 0)
            enc.setBuffer(x, offset: xOffset, index: 1)
            enc.setBuffer(y, offset: yOffset, index: 2)
            var rowsVar = UInt32(rows)
            var colsVar = UInt32(cols)
            enc.setBytes(&rowsVar, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&colsVar, length: MemoryLayout<UInt32>.size, index: 4)
            let groups = (rows + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
            enc.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                               height: 1, depth: 1))
            enc.endEncoding()
        }
    }
}
