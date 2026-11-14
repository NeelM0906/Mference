import Metal

/// Higher-precision native draft attention, isolated from the target decoder.
/// Cache rows are append-only; the owning draft runner controls rollback by
/// restoring its position and overwriting rejected rows on the next branch.
final class FlashNextMTPAttention {
    private let geometry: FlashNextAttention.Geometry
    private let capacity: Int
    private let projection: FlashNextMTPFloat32Projection
    private let widen: MTLComputePipelineState
    private let narrow: MTLComputePipelineState
    private let prepareQ: MTLComputePipelineState
    private let prepareK: MTLComputePipelineState
    private let attend: MTLComputePipelineState
    private let input: MTLBuffer
    private let packed: MTLBuffer
    private let queries: MTLBuffer
    private let attended: MTLBuffer
    private let projected: MTLBuffer
    private let keys: MTLBuffer
    private let values: MTLBuffer
    private let float32IO: Bool

    init(context: MetalContext, geometry: FlashNextAttention.Geometry, maxContext: Int, float32IO: Bool = false) throws {
        precondition(maxContext > 0 && geometry.headDim > 0 && geometry.headDim <= 256)
        precondition(geometry.numHeads > 0 && geometry.numKVHeads > 0 && geometry.hidden > 0)
        precondition(geometry.numHeads.isMultiple(of: geometry.numKVHeads))
        precondition(geometry.rotaryDim >= 0 && geometry.rotaryDim <= geometry.headDim && geometry.rotaryDim.isMultiple(of: 2))
        self.geometry = geometry
        self.float32IO = float32IO
        capacity = maxContext
        projection = try FlashNextMTPFloat32Projection(context: context)
        widen = try context.pipeline("flashnext_mtp_half_to_float")
        narrow = try context.pipeline("flashnext_mtp_float_to_half")
        prepareQ = try context.pipeline("flashnext_mtp_prepare_q_f32")
        prepareK = try context.pipeline("flashnext_mtp_prepare_k_f32")
        attend = try context.pipeline("flashnext_mtp_attend_f32")
        func buffer(_ count: Int) throws -> MTLBuffer {
            guard let result = context.device.makeBuffer(length: count * 4, options: .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return result
        }
        input = try buffer(geometry.hidden)
        packed = try buffer(geometry.qDim * 2)
        queries = try buffer(geometry.qDim)
        attended = try buffer(geometry.qDim)
        projected = try buffer(geometry.hidden)
        keys = try buffer(maxContext * geometry.kvDim)
        values = try buffer(maxContext * geometry.kvDim)
    }

    func encode(commandBuffer: MTLCommandBuffer, weights: FlashNextAttention.Weights,
                x: MTLBuffer, output: MTLBuffer, position: Int,
                selected: MTLBuffer, selectedCount: Int, contiguous: Bool) {
        let g = geometry
        precondition(position >= 0 && position < capacity && selectedCount > 0 && selectedCount <= position + 1)
        precondition(contiguous || selected.length >= selectedCount * 4)
        precondition(x.length >= g.hidden * (float32IO ? 4 : 2) && output.length >= g.hidden * (float32IO ? 4 : 2))
        precondition(weights.qNormOffset >= 0 && weights.kNormOffset >= 0)
        precondition(weights.qNormOffset + g.headDim * 4 <= weights.qNorm.length)
        precondition(weights.kNormOffset + g.headDim * 4 <= weights.kNorm.length)
        func dispatch(_ pipeline: MTLComputePipelineState, _ count: Int,
                      _ configure: (MTLComputeCommandEncoder) -> Void) {
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            configure(encoder)
            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(32, count), height: 1, depth: 1))
            encoder.endEncoding()
        }
        if !float32IO { dispatch(widen, g.hidden) {
            $0.setBuffer(x, offset: 0, index: 0)
            $0.setBuffer(input, offset: 0, index: 1)
        } }
        let wideInput = float32IO ? x : input
        projection.encode(commandBuffer: commandBuffer, matrix: weights.q, x: wideInput,
            out: packed, rows: 2 * g.qDim, columns: g.hidden)
        let offset = position * g.kvDim * 4
        projection.encode(commandBuffer: commandBuffer, matrix: weights.k, x: wideInput,
            out: keys, outOffset: offset, rows: g.kvDim, columns: g.hidden)
        projection.encode(commandBuffer: commandBuffer, matrix: weights.v, x: wideInput,
            out: values, outOffset: offset, rows: g.kvDim, columns: g.hidden)
        var d = UInt32(g.headDim), rotary = UInt32(g.rotaryDim), pos = UInt32(position)
        var theta = g.theta, eps = g.eps
        dispatch(prepareQ, g.numHeads) {
            $0.setBuffer(packed, offset: 0, index: 0)
            $0.setBuffer(weights.qNorm, offset: weights.qNormOffset, index: 1)
            $0.setBuffer(queries, offset: 0, index: 2)
            $0.setBytes(&d, length: 4, index: 3)
            $0.setBytes(&rotary, length: 4, index: 4)
            $0.setBytes(&pos, length: 4, index: 5)
            $0.setBytes(&theta, length: 4, index: 6)
            $0.setBytes(&eps, length: 4, index: 7)
        }
        dispatch(prepareK, g.numKVHeads) {
            $0.setBuffer(keys, offset: offset, index: 0)
            $0.setBuffer(weights.kNorm, offset: weights.kNormOffset, index: 1)
            $0.setBytes(&d, length: 4, index: 2)
            $0.setBytes(&rotary, length: 4, index: 3)
            $0.setBytes(&pos, length: 4, index: 4)
            $0.setBytes(&theta, length: 4, index: 5)
            $0.setBytes(&eps, length: 4, index: 6)
        }
        var heads = UInt32(g.numHeads), kvHeads = UInt32(g.numKVHeads)
        var count = UInt32(selectedCount), dense: UInt32 = contiguous ? 1 : 0
        var scale = g.scale
        dispatch(attend, g.numHeads) {
            $0.setBuffer(queries, offset: 0, index: 0)
            $0.setBuffer(packed, offset: 0, index: 1)
            $0.setBuffer(keys, offset: 0, index: 2)
            $0.setBuffer(values, offset: 0, index: 3)
            $0.setBuffer(selected, offset: 0, index: 4)
            $0.setBuffer(attended, offset: 0, index: 5)
            $0.setBytes(&d, length: 4, index: 6)
            $0.setBytes(&heads, length: 4, index: 7)
            $0.setBytes(&kvHeads, length: 4, index: 8)
            $0.setBytes(&count, length: 4, index: 9)
            $0.setBytes(&dense, length: 4, index: 10)
            $0.setBytes(&scale, length: 4, index: 11)
        }
        projection.encode(commandBuffer: commandBuffer, matrix: weights.o, x: attended,
            out: float32IO ? output : projected, rows: g.hidden, columns: g.qDim)
        if !float32IO { dispatch(narrow, g.hidden) {
            $0.setBuffer(projected, offset: 0, index: 0)
            $0.setBuffer(output, offset: 0, index: 1)
        } }
    }
}
