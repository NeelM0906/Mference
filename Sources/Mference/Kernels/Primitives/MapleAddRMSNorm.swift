import Metal

/// Native-BF16 residual carry and RMSNorm for Maple's 2048-wide decode path.
final class MapleAddRMSNorm {
    static let hiddenSize = 2048

    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = try MetalContext.moduleLibrary(device: context.device,
                                                     module: "maple_add_rmsnorm",
                                                     safeMath: true)
        guard let function = library.makeFunction(name: "maple_add_rmsnorm_bf16_d2048") else {
            throw MetalError.missingFunction("maple_add_rmsnorm_bf16_d2048")
        }
        self.pipeline = try context.device.makeComputePipelineState(function: function)
    }

    /// Rounds `hidden + delta` to BF16 in `hidden`, then writes BF16 RMSNorm to `normed`.
    func encode(commandBuffer: MTLCommandBuffer,
                hidden: MTLBuffer, hiddenOffset: Int = 0,
                delta: MTLBuffer, deltaOffset: Int = 0,
                weight: MTLBuffer, weightOffset: Int = 0,
                normed: MTLBuffer, normedOffset: Int = 0,
                eps: Float) {
        let bytes = Self.hiddenSize * MemoryLayout<UInt16>.stride
        precondition(eps.isFinite && eps > 0,
                     "Maple add+RMSNorm epsilon must be positive and finite")
        Self.requireRange(hidden, offset: hiddenOffset, bytes: bytes, named: "hidden")
        Self.requireRange(delta, offset: deltaOffset, bytes: bytes, named: "delta")
        Self.requireRange(weight, offset: weightOffset, bytes: bytes, named: "weight")
        Self.requireRange(normed, offset: normedOffset, bytes: bytes, named: "normed")
        Self.requireDisjoint(hidden, offset: hiddenOffset,
                             normed, offset: normedOffset,
                             bytes: bytes, named: "hidden and normed")
        Self.requireDisjoint(hidden, offset: hiddenOffset,
                             delta, offset: deltaOffset,
                             bytes: bytes, named: "hidden and delta")
        Self.requireDisjoint(hidden, offset: hiddenOffset,
                             weight, offset: weightOffset,
                             bytes: bytes, named: "hidden and weight")
        Self.requireDisjoint(normed, offset: normedOffset,
                             delta, offset: deltaOffset,
                             bytes: bytes, named: "normed and delta")
        Self.requireDisjoint(normed, offset: normedOffset,
                             weight, offset: weightOffset,
                             bytes: bytes, named: "normed and weight")

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 0)
        encoder.setBuffer(delta, offset: deltaOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(normed, offset: normedOffset, index: 3)
        var epsilon = eps
        encoder.setBytes(&epsilon, length: MemoryLayout<Float>.size, index: 4)
        let threads = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }

    private static func requireRange(_ buffer: MTLBuffer,
                                     offset: Int,
                                     bytes: Int,
                                     named: String) {
        precondition(offset >= 0 && bytes >= 0 && offset <= buffer.length &&
                     bytes <= buffer.length - offset,
                     "Maple add+RMSNorm \(named) buffer is too small")
        precondition(offset.isMultiple(of: MemoryLayout<UInt16>.stride),
                     "Maple add+RMSNorm \(named) offset must be BF16-aligned")
    }

    private static func requireDisjoint(_ first: MTLBuffer,
                                        offset firstOffset: Int,
                                        _ second: MTLBuffer,
                                        offset secondOffset: Int,
                                        bytes: Int,
                                        named: String) {
        guard first === second else { return }
        let firstEnd = firstOffset + bytes
        let secondEnd = secondOffset + bytes
        precondition(firstEnd <= secondOffset || secondEnd <= firstOffset,
                     "Maple add+RMSNorm \(named) ranges must not overlap")
    }
}
