import Metal

/// INT8-2L resident GEMV (WS4 U1 §3b, U3): `gguf_int8_2l_gemv_f16out` /
/// `_f32out` in `Metal/GGUF/gguf_int8_2l.metal`. `y = W · x` for one token,
/// with x in half, FP32 accumulation and a half or FP32 result. The weights
/// are the three regions the installer writes (`GGUFInt8TwoLevelExpander`):
/// int8 values, fp16 `d` per BD weights, and for Q6_K an int8 `m` per 16.
final class GGUFInt8TwoLevelGEMV {
    enum Source: Sendable {
        case q6_K, q8_0
        var blockWeights: Int { self == .q6_K ? 256 : 32 }
    }

    /// One INT8-2L tensor of `rows × n` weights.
    struct Weights {
        let source: Source
        let rows: Int, n: Int
        let values: (buffer: MTLBuffer, offset: Int)
        let d: (buffer: MTLBuffer, offset: Int)
        /// Q6_K only.
        let m: (buffer: MTLBuffer, offset: Int)?
    }

    /// Mirror of the Metal `Int8TwoLevelParams`.
    struct Params { var rows: UInt32; var n: UInt32; var bdShift: UInt32; var hasM: UInt32 }

    private let f16Out: MTLComputePipelineState
    private let f32Out: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = try MetalContext.moduleLibrary(device: context.device, module: "gguf_int8_2l")
        guard let a = library.makeFunction(name: "gguf_int8_2l_gemv_f16out"),
              let b = library.makeFunction(name: "gguf_int8_2l_gemv_f32out") else {
            throw MetalError.missingFunction("gguf_int8_2l_gemv_*")
        }
        f16Out = try context.device.makeComputePipelineState(function: a)
        f32Out = try context.device.makeComputePipelineState(function: b)
    }

    func encode(commandBuffer: MTLCommandBuffer, weights w: Weights,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0, floatOutput: Bool) {
        precondition(w.n % w.source.blockWeights == 0 && w.n % 8 == 0)
        precondition((w.source == .q6_K) == (w.m != nil), "Q6_K needs m; Q8_0 has none")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var params = Params(rows: UInt32(w.rows), n: UInt32(w.n),
                            bdShift: UInt32(w.source.blockWeights.trailingZeroBitCount),
                            hasM: w.m == nil ? 0 : 1)
        encoder.setComputePipelineState(floatOutput ? f32Out : f16Out)
        encoder.setBuffer(w.values.buffer, offset: w.values.offset, index: 0)
        encoder.setBuffer(w.d.buffer, offset: w.d.offset, index: 1)
        // Q8_0 binds the values again as a placeholder; the kernel never reads m without has_m.
        encoder.setBuffer(w.m?.buffer ?? w.values.buffer, offset: w.m?.offset ?? w.values.offset, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 3)
        encoder.setBuffer(x, offset: xOffset, index: 4)
        encoder.setBuffer(y, offset: yOffset, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: (w.rows + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
