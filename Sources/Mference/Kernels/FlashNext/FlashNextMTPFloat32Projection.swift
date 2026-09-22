import Metal

/// Native-draft-only matrix/vector projection with FP32 activations and output.
/// Matrices retain their installed BF16/INT4/INT8 storage and offsets.
final class FlashNextMTPFloat32Projection {
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        pipeline = try context.pipeline("flashnext_mtp_project_f32")
    }

    func encode(commandBuffer: MTLCommandBuffer, matrix: FlashNextWeightMatrix,
                x: MTLBuffer, xOffset: Int = 0, out: MTLBuffer, outOffset: Int = 0,
                rows: Int, columns: Int) {
        precondition(rows > 0 && columns > 0)
        precondition(xOffset >= 0 && xOffset + columns * 4 <= x.length)
        precondition(outOffset >= 0 && outOffset + rows * 4 <= out.length)
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        var bits: UInt32
        switch matrix {
        case let .bf16(buffer, offset):
            bits = 16
            for index in 0...2 { encoder.setBuffer(buffer, offset: offset, index: index) }
        case let .int4(w, wo, s, so, b, bo):
            bits = 4
            encoder.setBuffer(w, offset: wo, index: 0)
            encoder.setBuffer(s, offset: so, index: 1)
            encoder.setBuffer(b, offset: bo, index: 2)
        case let .int8(w, wo, s, so, b, bo):
            bits = 8
            encoder.setBuffer(w, offset: wo, index: 0)
            encoder.setBuffer(s, offset: so, index: 1)
            encoder.setBuffer(b, offset: bo, index: 2)
        }
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(out, offset: outOffset, index: 4)
        var n = UInt32(columns)
        encoder.setBytes(&n, length: 4, index: 5)
        encoder.setBytes(&bits, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
