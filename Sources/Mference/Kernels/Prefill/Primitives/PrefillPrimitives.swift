import Foundation
import Metal

final class PrefillEmbedLookupInt4 {
    private let pso: MTLComputePipelineState
    private let groupSize: Int

    init(context: MetalContext, groupSize: Int = Quantization.groupSize,
         sourceFP16: Bool = false) throws {
        self.groupSize = groupSize
        self.pso = try context.pipeline("prefill_embed_lookup_int4_block",
            constants: Quantization.int4Constants(groupSize: groupSize)
                + Quantization.gemmaSourceConstants(enabled: sourceFP16))
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       table: MTLBuffer, tableOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       tokens: MTLBuffer, tokensOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       t: UInt32,
                       d: UInt32,
                       outScale: Float) {
        precondition(d % UInt32(groupSize) == 0,
                     "D must be a multiple of \(groupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table, offset: tableOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(tokens, offset: tokensOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        var tVar = t
        var dVar = d
        var scaleVar = outScale
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 7)
        enc.dispatchThreads(MTLSize(width: Int(d), height: Int(t), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}

final class PrefillRMSNorm {
    private let psoBF16W: MTLComputePipelineState

    init(context: MetalContext, sourceFP16: Bool = false) throws {
        self.psoBF16W = try context.pipeline("prefill_rmsnorm_bf16w_block",
            constants: Quantization.gemmaSourceConstants(enabled: sourceFP16))
    }

    func encodeBF16W(commandBuffer: MTLCommandBuffer,
                            x: MTLBuffer, xOffset: Int = 0,
                            weight: MTLBuffer, weightOffset: Int = 0,
                            out: MTLBuffer, outOffset: Int = 0,
                            t: UInt32,
                            d: UInt32,
                            eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoBF16W)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var tVar = t
        var dVar = d
        var epsVar = eps
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 5)
        let threads = min(Int(psoBF16W.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(t), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

final class PrefillInt4QMM {
    private let pso: MTLComputePipelineState
    private let groupSize: Int
    private let decodeOrder: Bool
    private let qmmThreads: MTLSize
    private let decodeOrderPipelines: [MTLComputePipelineState]
    private static let decodeTileTokens = 4

    init(context: MetalContext, decodeOrder: Bool = false,
         groupSize: Int = Quantization.groupSize, sourceFP16: Bool = false) throws {
        self.groupSize = groupSize
        self.qmmThreads = sourceFP16
            ? MTLSize(width: 32, height: 8, depth: 1)
            : MTLSize(width: 8, height: 8, depth: 1)
        let quantizationConstants = Quantization.int4Constants(groupSize: groupSize)
            + Quantization.gemmaSourceConstants(enabled: sourceFP16)
        self.decodeOrder = decodeOrder
        if decodeOrder {
            // Safe math preserves each token's decode reduction while a small
            // register tile shares packed-weight reads, as in MTP verification.
            let library = try MetalContext.moduleLibrary(device: context.device,
                module: "dequant_int4", safeMath: true)
            let pipelines = try (1...Self.decodeTileTokens).map { tokens in
                let values = MTLFunctionConstantValues()
                var count = UInt32(tokens)
                var enabled = true
                var group = UInt32(groupSize)
                var sourcePrecision = sourceFP16
                values.setConstantValue(&sourcePrecision, type: .bool, index: 110)
                values.setConstantValue(&group, type: .uint, index: 108)
                values.setConstantValue(&count, type: .uint, index: 45)
                values.setConstantValue(&enabled, type: .bool, index: 46)
                let function = try library.makeFunction(
                    name: "prefill_dequant_int4_multix_block", constantValues: values)
                return try context.device.makeComputePipelineState(function: function)
            }
            self.decodeOrderPipelines = pipelines
            self.pso = pipelines[Self.decodeTileTokens - 1]
        } else {
            self.decodeOrderPipelines = []
            self.pso = try context.pipeline("prefill_dequant_int4_qmm_f16_block",
                constants: quantizationConstants, maxTotalThreadsPerThreadgroup: nil,
                safeMathModule: sourceFP16 ? "prefill" : nil)
        }
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       t: Int,
                       n: Int,
                       k: Int) {
        precondition(k % groupSize == 0,
                     "K must be a multiple of \(groupSize)")
        precondition(!decodeOrder || weightsOffset.isMultiple(of: 2),
                     "decode-order INT4 projection needs two-byte aligned weights")
        guard t > 0, n > 0, k > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var tVar = UInt32(t)
        var nVar = UInt32(n)
        var kVar = UInt32(k)
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 7)
        if decodeOrder {
            let tiles = t / Self.decodeTileTokens
            if tiles > 0 {
                enc.dispatchThreadgroups(MTLSize(width: (n + 7) / 8, height: tiles, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            let tail = t % Self.decodeTileTokens
            if tail > 0 {
                let row = tiles * Self.decodeTileTokens
                enc.setComputePipelineState(decodeOrderPipelines[tail - 1])
                enc.setBuffer(x, offset: xOffset + row * k * MemoryLayout<Float16>.stride, index: 3)
                enc.setBuffer(y, offset: yOffset + row * n * MemoryLayout<Float16>.stride, index: 4)
                tVar = UInt32(tail)
                enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: (n + 7) / 8, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        } else {
            enc.dispatchThreadgroups(MTLSize(width: (n + 7) / 8, height: (t + 7) / 8, depth: 1),
                threadsPerThreadgroup: qmmThreads)
        }
        enc.endEncoding()
    }
}
