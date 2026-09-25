import Foundation
import Metal

final class MPPPrefillInt4QMM {
    enum Path: String, Sendable {
        case affineThreadgroupF16 = "affine-threadgroup-f16"
        case affineThreadgroupF32 = "affine-threadgroup-f32"
        case sourceAffineF16 = "source-affine-f16"
        case sourceAffineF32 = "source-affine-f32"
        case unavailable
    }

    static let tileM = 64
    static let tileN = 32
    static let tileK = 64

    private var pipeline: MTLComputePipelineState?
    private var pipelineF32: MTLComputePipelineState?
    private let sourceFP16: Bool

    init(context: MetalContext, groupSize: Int = Quantization.groupSize, sourceFP16: Bool = false) {
        precondition(groupSize == 32 || groupSize == 64)
        self.sourceFP16 = sourceFP16
        do {
            let library = try Self.compileTensorOpsLibrary(device: context.device, safeMath: sourceFP16)
            let constants = MTLFunctionConstantValues()
            var group = UInt32(groupSize)
            var sourcePrecision = sourceFP16
            constants.setConstantValue(&sourcePrecision, type: .bool, index: 110)
            constants.setConstantValue(&group, type: .uint, index: 108)
            let function = try library.makeFunction(
                name: "mpp_prefill_affine_threadgroup_f16", constantValues: constants)
            self.pipeline = try context.device.makeComputePipelineState(function: function)
            let functionF32 = try library.makeFunction(
                name: "mpp_prefill_affine_threadgroup_f32", constantValues: constants)
            self.pipelineF32 = try context.device.makeComputePipelineState(
                function: functionF32)
        } catch {
            self.pipeline = nil
            self.pipelineF32 = nil
        }
    }

    var isAvailable: Bool {
        pipeline != nil
    }

    var isFloat32Available: Bool { pipelineF32 != nil }

    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       m: Int,
                       n: Int,
                       k: Int) -> Path {
        guard m > 0,
              n > 0,
              k > 0,
              k.isMultiple(of: Self.tileK),
              weightsOffset >= 0,
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              xOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              yOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              let pipeline,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return .unavailable
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
        return sourceFP16 ? .sourceAffineF16 : .affineThreadgroupF16
    }

    @discardableResult
    func encodeFloat32(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       m: Int, n: Int, k: Int) -> Path {
        guard m > 0, n > 0, k > 0,
              k.isMultiple(of: Self.tileK),
              weightsOffset >= 0,
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              xOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              yOffset.isMultiple(of: MemoryLayout<Float>.stride),
              let pipelineF32,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return .unavailable
        }
        encoder.setComputePipelineState(pipelineF32)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pipelineF32.threadExecutionWidth * 4,
                                           height: 1, depth: 1))
        encoder.endEncoding()
        return sourceFP16 ? .sourceAffineF32 : .affineThreadgroupF32
    }

    private static func compileTensorOpsLibrary(device: MTLDevice, safeMath: Bool) throws -> MTLLibrary {
        try MetalContext.moduleLibrary(device: device, module: "tensorops", safeMath: safeMath)
    }
}
