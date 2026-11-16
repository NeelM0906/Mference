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

    /// Mirror of the Metal `Int8TwoLevelBatchParams`.
    struct BatchParams { var tokens: UInt32; var xStride: UInt32; var yStride: UInt32; var pad: UInt32 = 0 }

    private let f16Out: MTLComputePipelineState
    private let f32Out: MTLComputePipelineState
    private let batchedF16Out: MTLComputePipelineState
    private let q6KEmbedRows: MTLComputePipelineState

    init(context: MetalContext) throws {
        // Safe math: the batched form is claimed bit-identical to the GEMV and the
        // Q6_K gather bit-exact, so the compiler must not reassociate either.
        let library = try MetalContext.moduleLibrary(device: context.device, module: "gguf_int8_2l",
                                                     safeMath: true)
        guard let a = library.makeFunction(name: "gguf_int8_2l_gemv_f16out"),
              let b = library.makeFunction(name: "gguf_int8_2l_gemv_f32out"),
              let c = library.makeFunction(name: "gguf_int8_2l_gemm_t8_f16out"),
              let e = library.makeFunction(name: "gguf_q6k_embed_rows") else {
            throw MetalError.missingFunction("gguf_int8_2l_*")
        }
        f16Out = try context.device.makeComputePipelineState(function: a)
        f32Out = try context.device.makeComputePipelineState(function: b)
        batchedF16Out = try context.device.makeComputePipelineState(function: c)
        q6KEmbedRows = try context.device.makeComputePipelineState(function: e)
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

    /// Prefill form: `Y[t] = W · X[t]` for `tokens` rows, half output. Each
    /// token's row is bit-identical to `encode(…, floatOutput: false)` on that
    /// token alone; the weights are read once per group of 8 tokens.
    func encodeBatched(commandBuffer: MTLCommandBuffer, weights w: Weights,
                       x: MTLBuffer, xOffset: Int = 0, xStride: Int,
                       y: MTLBuffer, yOffset: Int = 0, yStride: Int, tokens: Int) {
        precondition(w.n % w.source.blockWeights == 0 && w.n % 8 == 0)
        precondition((w.source == .q6_K) == (w.m != nil), "Q6_K needs m; Q8_0 has none")
        precondition(xStride >= w.n && xStride % 4 == 0 && yStride >= w.rows && tokens > 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var params = Params(rows: UInt32(w.rows), n: UInt32(w.n),
                            bdShift: UInt32(w.source.blockWeights.trailingZeroBitCount),
                            hasM: w.m == nil ? 0 : 1)
        var batch = BatchParams(tokens: UInt32(tokens), xStride: UInt32(xStride), yStride: UInt32(yStride))
        encoder.setComputePipelineState(batchedF16Out)
        encoder.setBuffer(w.values.buffer, offset: w.values.offset, index: 0)
        encoder.setBuffer(w.d.buffer, offset: w.d.offset, index: 1)
        encoder.setBuffer(w.m?.buffer ?? w.values.buffer, offset: w.m?.offset ?? w.values.offset, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 3)
        encoder.setBuffer(x, offset: xOffset, index: 4)
        encoder.setBuffer(y, offset: yOffset, index: 5)
        encoder.setBytes(&batch, length: MemoryLayout<BatchParams>.stride, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: (w.rows + 7) / 8, height: (tokens + 7) / 8, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// `token_embd` gather from a native Q6_K table (`[vocab][hidden/256]`
    /// blocks of 210 B): `out[t] = half(row ids[t])`, decoded as ggml does.
    /// One call covers a decode token or a prefill chunk.
    func encodeQ6KEmbedRows(commandBuffer: MTLCommandBuffer,
                            table: MTLBuffer, tableOffset: Int = 0,
                            ids: MTLBuffer, idsOffset: Int = 0,
                            out: MTLBuffer, outOffset: Int = 0,
                            hidden: Int, tokens: Int) {
        precondition(hidden % 256 == 0 && tokens > 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var h = UInt32(hidden), t = UInt32(tokens)
        encoder.setComputePipelineState(q6KEmbedRows)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(ids, offset: idsOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBytes(&h, length: 4, index: 3)
        encoder.setBytes(&t, length: 4, index: 4)
        encoder.dispatchThreads(MTLSize(width: hidden, height: tokens, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
