import Foundation
import Metal

/// Grouped-GEMM routed experts for GPUs without the TensorOps prefill path
/// (`prefill_simd_grouped_moe.metal`): 8x8 simdgroup matrices with float
/// accumulators, weights dequantized into threadgroup tiles 32 deep. The
/// dequant family is INT4 affine or the GGUF LUT4 execution format
/// (`GGUFLUT4`), whose per-projection parameters are bound per dispatch. Same ABI and call shape as `MPPGroupedRoutedMoE.encode` (buffers
/// 0-6, a 16-expert argument buffer, group-range params), SwiGLU only.
///
/// It needs no TensorOps, so it builds on the macOS 15 floor too. Shapes it
/// does not cover (`supports`) are left to the caller's other kernels.
final class SimdgroupGroupedRoutedMoE {
    static let rowsPerBlock = 64
    static let phase1Columns = 64
    static let downColumns = 128
    static let kStep = 32
    static let threadsPerThreadgroup = 128

    /// How expert weights are stored.
    enum Dequant: Equatable {
        /// MLX-style INT4 with bf16 scale and bias per `groupSize` weights.
        case int4Affine(groupSize: Int)
        /// GGUF LUT4: the nine offsets are (indices, D, M) per projection.
        case lut4
    }

    let dequant: Dequant
    private let phase1: MTLComputePipelineState
    private let down: MTLComputePipelineState
    private let argumentEncoder: MTLArgumentEncoder

    /// Layout-identical to `PSGMParams`: the streamed params' 128-byte
    /// prefix plus the resident stride word, as the MPP kernel's.
    private struct Params {
        var streamed: PrefillGroupedRoutedMoEStreamedParams
        var residentExpertStride: UInt32
    }

    convenience init(context: MetalContext, groupSize: Int = Quantization.groupSize) throws {
        try self.init(context: context, dequant: .int4Affine(groupSize: groupSize))
    }

    init(context: MetalContext, dequant: Dequant) throws {
        self.dequant = dequant
        let constants: [MetalFunctionConstant]
        switch dequant {
        case .int4Affine(let groupSize):
            constants = Quantization.int4Constants(groupSize: groupSize)
        case .lut4:
            constants = [MetalFunctionConstant(index: 113, value: .uint32(1))]
        }
        self.phase1 = try context.pipeline("prefill_simd_grouped_moe_phase1", constants: constants)
        self.down = try context.pipeline("prefill_simd_grouped_moe_down", constants: constants)
        guard let function = context.library.makeFunction(name: "prefill_simd_grouped_moe_phase1") else {
            throw MetalError.missingFunction("prefill_simd_grouped_moe_phase1")
        }
        self.argumentEncoder = function.makeArgumentEncoder(bufferIndex: 5)
        guard phase1.maxTotalThreadsPerThreadgroup >= Self.threadsPerThreadgroup,
              down.maxTotalThreadsPerThreadgroup >= Self.threadsPerThreadgroup else {
            throw MetalError.missingFunction("prefill_simd_grouped_moe at \(Self.threadsPerThreadgroup) threads")
        }
    }

    /// Static threadgroup memory of each pipeline (tests: the INT4 pipeline
    /// must not pay for the LUT4 tables).
    var staticThreadgroupMemoryLengths: (phase1: Int, down: Int) {
        (phase1.staticThreadgroupMemoryLength, down.staticThreadgroupMemoryLength)
    }

    /// Hidden size and routed intermediate size the tiles cover exactly.
    static func supports(d: Int, routedIntermediate f: Int, groupSize: Int) -> Bool {
        d > 0 && f > 0
            && d % downColumns == 0 && d % kStep == 0
            && f % phase1Columns == 0 && f % kStep == 0
            && (groupSize == 32 || groupSize == 64)
            && d % groupSize == 0 && f % groupSize == 0
    }

    /// The kernel reads packed weights (or LUT4 indices) as `uint` and
    /// scales and biases (or LUT4 `D`) as 16-bit values, so those section
    /// offsets must be 4- and 2-byte aligned; LUT4 `M` is int8.
    static func supportsOffsets(_ params: PrefillGroupedRoutedMoEStreamedParams, lut4: Bool = false) -> Bool {
        [params.gateWOff, params.upWOff, params.downWOff].allSatisfy { $0 % 4 == 0 }
            && [params.gateSOff, params.upSOff, params.downSOff].allSatisfy { $0 % 2 == 0 }
            && (lut4 || [params.gateBOff, params.upBOff, params.downBOff].allSatisfy { $0 % 2 == 0 })
    }

    /// LUT4 shapes: whole tiles, and every type's 32-weight index groups and
    /// D blocks (BD >= 32) align to the 32-deep K step.
    static func supportsLUT4(d: Int, routedIntermediate f: Int) -> Bool {
        d > 0 && f > 0 && d % downColumns == 0 && f % phase1Columns == 0
            && d % kStep == 0 && f % kStep == 0
    }

    func makeArgumentBuffer(device: MTLDevice,
                            binding: PrefillStreamedTileBinding) throws -> PrefillStreamedTileArgumentBuffer {
        guard let buffer = device.makeBuffer(length: argumentEncoder.encodedLength,
                                             options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("simdgroup grouped routed expert argument buffer")
        }
        argumentEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, view) in binding.views.enumerated() {
            argumentEncoder.setBuffer(view.buffer, offset: Int(view.offset), index: index)
        }
        return PrefillStreamedTileArgumentBuffer(buffer: buffer)
    }

    /// Encodes one routed tile. `params.pairStart`/`pairCount` are the tile's
    /// group range (as for the MPP kernel); `activation` holds
    /// (tile pairs) x f halves.
    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                hidden: MTLBuffer,
                sortedPairs: MTLBuffer,
                groups: MTLBuffer,
                activation: MTLBuffer,
                routePartials: MTLBuffer,
                argumentBuffer: PrefillStreamedTileArgumentBuffer,
                binding: PrefillStreamedTileBinding,
                params: PrefillGroupedRoutedMoEStreamedParams,
                maxPairsPerGroup: Int,
                lut4: [GGUFLUT4MoE.ProjParams]? = nil) -> Bool {
        // Streamed tiles only: unlike MPP's ABI, a zero live-expert count is
        // not a resident mode here.
        let isLUT4 = dequant == .lut4
        // LUT4 needs (gate, up, down) parameters; INT4 binds zeros there.
        guard params.pairCount > 0, maxPairsPerGroup > 0, params.liveExpertCount > 0,
              Self.supportsOffsets(params, lut4: isLUT4),
              isLUT4 == (lut4?.count == 3) else { return false }
        var projections = lut4 ?? []
        let projectionBytes = 3 * MemoryLayout<GGUFLUT4MoE.ProjParams>.stride
        var zeros = [UInt8](repeating: 0, count: projectionBytes)
        func bind(_ encoder: MTLComputeCommandEncoder) {
            encoder.setBuffer(hidden, offset: 0, index: 0)
            encoder.setBuffer(sortedPairs, offset: 0, index: 1)
            encoder.setBuffer(groups, offset: 0, index: 2)
            encoder.setBuffer(activation, offset: 0, index: 3)
            encoder.setBuffer(routePartials, offset: 0, index: 4)
            encoder.setBuffer(argumentBuffer.buffer, offset: 0, index: 5)
            var p = Params(streamed: params, residentExpertStride: 0)
            encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 6)
            if isLUT4 {
                encoder.setBytes(&projections, length: projectionBytes, index: 7)
            } else {
                encoder.setBytes(&zeros, length: projectionBytes, index: 7)
            }
            for view in binding.views { encoder.useResource(view.buffer, usage: .read) }
        }
        let rowBlocks = (maxPairsPerGroup + Self.rowsPerBlock - 1) / Self.rowsPerBlock
        let threads = MTLSize(width: Self.threadsPerThreadgroup, height: 1, depth: 1)
        guard let first = commandBuffer.makeComputeCommandEncoder() else { return false }
        first.setComputePipelineState(phase1)
        bind(first)
        first.dispatchThreadgroups(
            MTLSize(width: Int(params.routedIntermediate) / Self.phase1Columns,
                    height: rowBlocks, depth: Int(params.pairCount)),
            threadsPerThreadgroup: threads)
        first.endEncoding()
        guard let second = commandBuffer.makeComputeCommandEncoder() else { return false }
        second.setComputePipelineState(down)
        bind(second)
        second.dispatchThreadgroups(
            MTLSize(width: Int(params.d) / Self.downColumns,
                    height: rowBlocks, depth: Int(params.pairCount)),
            threadsPerThreadgroup: threads)
        second.endEncoding()
        return true
    }
}
