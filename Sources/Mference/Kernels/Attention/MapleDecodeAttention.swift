import Metal

/// Exact native-BF16 Maple GQA vector-SDPA over physical cache order.
/// A runner owns one instance and encodes serially; two-pass calls reuse its
/// private scratch buffers.
final class MapleDecodeAttention {
    struct DispatchPlan: Equatable, Sendable {
        let twoPassThreshold: UInt32
        let blocks: Int
    }

    static let numQHeads = 16
    static let numKVHeads = 4
    static let headDim = 128
    static let slidingCapacity = 512
    static let maximumSequenceLength: UInt32 = 128_000
    static let attentionScale = Float(bitPattern: 0x3db5_04f3)

    private static let threads = 1024
    private static let maximumTwoPassBlocks = 128

    private let twoPassThreshold: UInt32
    private let twoPassBlocks: Int
    private let singlePass: MTLComputePipelineState
    private let twoPassFirst: MTLComputePipelineState
    private let twoPassSecond: MTLComputePipelineState
    private let partials: MTLBuffer
    private let sums: MTLBuffer
    private let maxs: MTLBuffer

    init(context: MetalContext) throws {
        let library = try MetalContext.moduleLibrary(
            device: context.device, module: "maple_attention", safeMath: true)
        let plan = Self.dispatchPlan(architectureName: context.device.architecture.name)
        self.twoPassThreshold = plan.twoPassThreshold
        self.twoPassBlocks = plan.blocks
        guard let single = library.makeFunction(name: "maple_sdpa_vector_bf16_d128"),
              let second = library.makeFunction(name: "maple_sdpa_vector_2pass_2_bf16_d128")
        else {
            throw MetalError.missingFunction("Maple vector SDPA kernels")
        }
        let constants = MTLFunctionConstantValues()
        var blocks = Int32(plan.blocks)
        constants.setConstantValue(&blocks, type: .int, index: 1)
        let first = try library.makeFunction(
            name: "maple_sdpa_vector_2pass_1_bf16_d128", constantValues: constants)
        self.singlePass = try Self.pipeline(single, device: context.device, threads: Self.threads)
        self.twoPassFirst = try Self.pipeline(first, device: context.device, threads: 128)
        self.twoPassSecond = try Self.pipeline(second, device: context.device, threads: Self.threads)
        precondition(singlePass.maxTotalThreadsPerThreadgroup >= Self.threads &&
                     twoPassSecond.maxTotalThreadsPerThreadgroup >= Self.threads,
                     "Maple vector SDPA requires 1024-thread threadgroups")

        let partialCount = Self.numQHeads * Self.maximumTwoPassBlocks * Self.headDim
        let statisticCount = Self.numQHeads * Self.maximumTwoPassBlocks
        guard let partials = context.device.makeBuffer(
            length: partialCount * MemoryLayout<UInt16>.stride, options: .storageModePrivate),
              let sums = context.device.makeBuffer(
                length: statisticCount * MemoryLayout<Float>.stride, options: .storageModePrivate),
              let maxs = context.device.makeBuffer(
                length: statisticCount * MemoryLayout<Float>.stride, options: .storageModePrivate)
        else {
            throw MetalError.noDevice
        }
        self.partials = partials
        self.sums = sums
        self.maxs = maxs
    }

    /// Attends a sliding layer over the cache's physical 0..<count row order.
    /// After a rotating-cache wrap, that order intentionally starts at slot 0.
    func encodeSliding(commandBuffer: MTLCommandBuffer,
                       q: MTLBuffer, qOffset: Int = 0,
                       k: MTLBuffer, kOffset: Int = 0,
                       v: MTLBuffer, vOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       physicalCount: UInt32) {
        precondition(physicalCount > 0 && physicalCount <= Self.slidingCapacity,
                     "Maple sliding attention requires 1...512 physical KV rows")
        encode(commandBuffer: commandBuffer, q: q, qOffset: qOffset,
               k: k, kOffset: kOffset, v: v, vOffset: vOffset,
               out: out, outOffset: outOffset, sequenceLength: physicalCount,
               allowTwoPass: false)
    }

    /// Attends a global NoPE layer over its linear 0..<sequenceLength prefix.
    func encodeFull(commandBuffer: MTLCommandBuffer,
                    q: MTLBuffer, qOffset: Int = 0,
                    k: MTLBuffer, kOffset: Int = 0,
                    v: MTLBuffer, vOffset: Int = 0,
                    out: MTLBuffer, outOffset: Int = 0,
                    sequenceLength: UInt32) {
        encode(commandBuffer: commandBuffer, q: q, qOffset: qOffset,
               k: k, kOffset: kOffset, v: v, vOffset: vOffset,
               out: out, outOffset: outOffset, sequenceLength: sequenceLength,
               allowTwoPass: true)
    }

    private func encode(commandBuffer: MTLCommandBuffer,
                        q: MTLBuffer, qOffset: Int,
                        k: MTLBuffer, kOffset: Int,
                        v: MTLBuffer, vOffset: Int,
                        out: MTLBuffer, outOffset: Int,
                        sequenceLength: UInt32,
                        allowTwoPass: Bool) {
        precondition(sequenceLength > 0 && sequenceLength <= Self.maximumSequenceLength,
                     "Maple vector SDPA requires 1...128,000 KV rows")
        let qBytes = Self.numQHeads * Self.headDim * MemoryLayout<UInt16>.stride
        let kvBytes = Int(sequenceLength) * Self.numKVHeads * Self.headDim
            * MemoryLayout<UInt16>.stride
        Self.requireRange(q, offset: qOffset, bytes: qBytes, named: "Q")
        Self.requireRange(k, offset: kOffset, bytes: kvBytes, named: "K")
        Self.requireRange(v, offset: vOffset, bytes: kvBytes, named: "V")
        Self.requireRange(out, offset: outOffset, bytes: qBytes, named: "output")
        Self.requireDisjoint(out, outOffset, qBytes, q, qOffset, qBytes,
                             left: "output", right: "Q")
        Self.requireDisjoint(out, outOffset, qBytes, k, kOffset, kvBytes,
                             left: "output", right: "K")
        Self.requireDisjoint(out, outOffset, qBytes, v, vOffset, kvBytes,
                             left: "output", right: "V")

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let useTwoPass = allowTwoPass && sequenceLength >= twoPassThreshold
        encoder.setComputePipelineState(useTwoPass ? twoPassFirst : singlePass)
        encoder.setBuffer(q, offset: qOffset, index: 0)
        encoder.setBuffer(k, offset: kOffset, index: 1)
        encoder.setBuffer(v, offset: vOffset, index: 2)
        var length = Int32(sequenceLength)
        var scale = Self.attentionScale
        if !useTwoPass {
            encoder.setBuffer(out, offset: outOffset, index: 3)
            encoder.setBytes(&length, length: MemoryLayout<Int32>.stride, index: 4)
            encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: Self.numQHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: Self.threads, height: 1, depth: 1))
            encoder.endEncoding()
            return
        }

        encoder.setBuffer(partials, offset: 0, index: 3)
        encoder.setBuffer(sums, offset: 0, index: 4)
        encoder.setBuffer(maxs, offset: 0, index: 5)
        encoder.setBytes(&length, length: MemoryLayout<Int32>.stride, index: 6)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: Self.numKVHeads, height: 1, depth: twoPassBlocks),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))

        encoder.setComputePipelineState(twoPassSecond)
        encoder.setBuffer(partials, offset: 0, index: 0)
        encoder.setBuffer(sums, offset: 0, index: 1)
        encoder.setBuffer(maxs, offset: 0, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var blocks = Int32(twoPassBlocks)
        encoder.setBytes(&blocks, length: MemoryLayout<Int32>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: Self.numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Mirrors MLX 0.32.0's vector-SDPA architecture split.
    static func dispatchPlan(architectureName: String) -> DispatchPlan {
        switch architectureName.lowercased().last {
        case "d": return DispatchPlan(twoPassThreshold: 1024, blocks: 128)
        case "s": return DispatchPlan(twoPassThreshold: 1024, blocks: 64)
        default: return DispatchPlan(twoPassThreshold: 4096, blocks: 64)
        }
    }

    private static func pipeline(_ function: MTLFunction,
                                 device: MTLDevice,
                                 threads: Int) throws -> MTLComputePipelineState {
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.maxTotalThreadsPerThreadgroup = threads
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        return try device.makeComputePipelineState(descriptor: descriptor,
                                                   options: [], reflection: nil)
    }

    private static func requireRange(_ buffer: MTLBuffer,
                                     offset: Int,
                                     bytes: Int,
                                     named: String) {
        precondition(offset >= 0 && offset <= buffer.length &&
                     bytes <= buffer.length - offset,
                     "Maple \(named) buffer is too small")
        precondition(offset.isMultiple(of: MemoryLayout<UInt16>.stride),
                     "Maple \(named) offset must be BF16-aligned")
    }

    private static func requireDisjoint(_ leftBuffer: MTLBuffer,
                                        _ leftOffset: Int,
                                        _ leftBytes: Int,
                                        _ rightBuffer: MTLBuffer,
                                        _ rightOffset: Int,
                                        _ rightBytes: Int,
                                        left: String,
                                        right: String) {
        guard leftBuffer === rightBuffer else { return }
        precondition(leftOffset + leftBytes <= rightOffset ||
                     rightOffset + rightBytes <= leftOffset,
                     "Maple \(left) and \(right) ranges must not overlap")
    }
}
