import Metal

/// LUT4 routed-expert decode (WS4 U3): `gguf_lut4_moe_phase1_gate_up_act`
/// and `gguf_lut4_moe_phase2_down_reduce` in `Metal/GGUF/gguf_lut4.metal`.
/// Same contract as the INT4 `MoE` phase 1 / phase 2 pair on the slot path:
/// x and acts in half, FP32 accumulation, SiLU · up, and a residual-first FP32
/// reduce in rank order written as half.
final class GGUFLUT4MoE {
    static let routedSlots = 10

    /// Byte offsets of each projection's LUT4 sections inside an expert slot.
    struct ExpertOffsets: Equatable {
        var gateIndex: UInt32, gateD: UInt32, gateM: UInt32
        var upIndex: UInt32, upD: UInt32, upM: UInt32
        var downIndex: UInt32, downD: UInt32, downM: UInt32
    }

    /// Mirror of the Metal `LUT4ProjParams`.
    struct ProjParams {
        var bdShift: UInt32
        var bmShift: UInt32
        var hasM: UInt32
        var scaleK: Float
        var table: (Float, Float, Float, Float, Float, Float, Float, Float,
                    Float, Float, Float, Float, Float, Float, Float, Float)

        init(_ type: GGUFLUT4.SourceType) {
            let p = GGUFLUT4.parameters(type)
            precondition(p.bd.nonzeroBitCount == 1 && (p.bm == 0 || p.bm.nonzeroBitCount == 1))
            bdShift = UInt32(p.bd.trailingZeroBitCount)
            bmShift = p.bm == 0 ? 0 : UInt32(p.bm.trailingZeroBitCount)
            hasM = p.bm == 0 ? 0 : 1
            scaleK = Float(sign: .plus, exponent: p.k, significand: 1)
            let t = p.table.map(Float.init)
            table = (t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7],
                     t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15])
        }
    }

    private let phase1: MTLComputePipelineState
    private let phase2: MTLComputePipelineState
    private let argumentEncoder: MTLArgumentEncoder
    private let argumentBuffer: MTLBuffer

    init(context: MetalContext) throws {
        let library = try MetalContext.moduleLibrary(device: context.device, module: "gguf_lut4")
        guard let f1 = library.makeFunction(name: "gguf_lut4_moe_phase1_gate_up_act"),
              let f2 = library.makeFunction(name: "gguf_lut4_moe_phase2_down_reduce") else {
            throw MetalError.missingFunction("gguf_lut4_moe_phase*")
        }
        phase1 = try context.device.makeComputePipelineState(function: f1)
        phase2 = try context.device.makeComputePipelineState(function: f2)
        argumentEncoder = f1.makeArgumentEncoder(bufferIndex: 0)
        guard let buffer = context.device.makeBuffer(length: argumentEncoder.encodedLength,
                                                     options: .storageModeShared) else {
            throw MetalError.commandBufferFailed("LUT4 routed argument buffer allocation failed")
        }
        argumentBuffer = buffer
    }

    /// Points the routed argument buffer at `blobs` (one expert slot each).
    /// The buffer is reused: the caller must not re-encode it while a command
    /// that reads it is in flight (the same rule as `MoE`'s reused buffer).
    func encodeBlobs(_ blobs: [(buffer: MTLBuffer, offset: Int)]) -> MTLBuffer {
        precondition(!blobs.isEmpty && blobs.count <= Self.routedSlots)
        argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
        for (i, blob) in blobs.enumerated() { argumentEncoder.setBuffer(blob.buffer, offset: blob.offset, index: i) }
        return argumentBuffer
    }

    func encodePhase1(commandBuffer: MTLCommandBuffer,
                      blobs: [(buffer: MTLBuffer, offset: Int)],
                      offsets: ExpertOffsets,
                      gateUp: GGUFLUT4.SourceType,
                      x: MTLBuffer, acts: MTLBuffer,
                      d: UInt32, f: UInt32) {
        precondition(d % 32 == 0 && f % 32 == 0)
        let args = encodeBlobs(blobs)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var dd = d, ff = f, k = UInt32(blobs.count)
        var off = offsets
        var params = ProjParams(gateUp)
        encoder.setComputePipelineState(phase1)
        encoder.setBuffer(args, offset: 0, index: 0)
        for blob in blobs { encoder.useResource(blob.buffer, usage: .read) }
        encoder.setBytes(&off, length: MemoryLayout<ExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dd, length: 4, index: 4)
        encoder.setBytes(&ff, length: 4, index: 5)
        encoder.setBytes(&k, length: 4, index: 6)
        encoder.setBytes(&params, length: MemoryLayout<ProjParams>.stride, index: 7)
        encoder.dispatchThreadgroups(MTLSize(width: (Int(k * f) + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodePhase2(commandBuffer: MTLCommandBuffer,
                      blobs: [(buffer: MTLBuffer, offset: Int)],
                      offsets: ExpertOffsets,
                      down: GGUFLUT4.SourceType,
                      acts: MTLBuffer, routingWeights: MTLBuffer,
                      residual: MTLBuffer, y: MTLBuffer,
                      d: UInt32, f: UInt32) {
        precondition(d % 32 == 0 && f % 32 == 0)
        let args = encodeBlobs(blobs)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var dd = d, ff = f, k = UInt32(blobs.count)
        var off = offsets
        var params = ProjParams(down)
        encoder.setComputePipelineState(phase2)
        encoder.setBuffer(args, offset: 0, index: 0)
        for blob in blobs { encoder.useResource(blob.buffer, usage: .read) }
        encoder.setBytes(&off, length: MemoryLayout<ExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setBytes(&dd, length: 4, index: 6)
        encoder.setBytes(&ff, length: 4, index: 7)
        encoder.setBytes(&k, length: 4, index: 8)
        encoder.setBytes(&params, length: MemoryLayout<ProjParams>.stride, index: 9)
        encoder.dispatchThreadgroups(MTLSize(width: Int(d), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Int(k) * 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
