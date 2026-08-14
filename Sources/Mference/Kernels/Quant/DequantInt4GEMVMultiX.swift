import Metal

/// Multi-token MLX-affine INT4 GEMV for the MTP speculative-verify pass:
/// one weight read applied to up to `maxTokens` token rows, with per-token
/// arithmetic bit-identical to `DequantInt4GEMV` (see the kernel comment in
/// `dequant_int4.metal`). `x` is `[T, n]` row-major and `y` is `[T, m]`
/// row-major.
final class DequantInt4GEMVMultiX {
    static let maxTokens = 8
    private static let rowsPerThreadgroup = 8

    private let pipelines: [MTLComputePipelineState]        // index = T - 1
    private let f32OutPipelines: [MTLComputePipelineState]  // index = T - 1

    init(context: MetalContext) throws {
        // One pipeline per token count: the specialized T lets the compiler
        // unroll the per-token loops and keep the accumulators in registers.
        // Compiled from a safe-math library: fast math is free to reassociate
        // the unrolled per-token chains, which would break the bit-identity
        // with the decode GEMV that the speculative verify depends on
        // (`MultiXKernelParityTests` locks this).
        let library = try MetalContext.moduleLibrary(device: context.device,
                                                     module: "dequant_int4",
                                                     safeMath: true)
        func specialized(_ name: String, tokens: Int) throws -> MTLComputePipelineState {
            let values = MTLFunctionConstantValues()
            var t = UInt32(tokens)
            var use = true
            values.setConstantValue(&t, type: .uint, index: 45)
            values.setConstantValue(&use, type: .bool, index: 46)
            let function = try library.makeFunction(name: name, constantValues: values)
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function
            descriptor.maxTotalThreadsPerThreadgroup = 512
            var reflection: MTLAutoreleasedComputePipelineReflection?
            return try context.device.makeComputePipelineState(descriptor: descriptor,
                                                               options: [],
                                                               reflection: &reflection)
        }
        self.pipelines = try (1...Self.maxTokens).map {
            try specialized("dequant_int4_gemv_simd_multix", tokens: $0)
        }
        self.f32OutPipelines = try (1...Self.maxTokens).map {
            try specialized("dequant_int4_gemv_simd_multix_f32out", tokens: $0)
        }
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                m: Int, n: Int, tokens: Int,
                outputFloat32: Bool = false) {
        precondition(n % Quantization.groupSize == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        precondition(weightsOffset % 2 == 0,
                     "dequant_int4_gemv_simd_multix needs a 2-aligned weightsOffset")
        precondition(tokens > 0 && tokens <= Self.maxTokens,
                     "token count \(tokens) outside 1...\(Self.maxTokens)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            outputFloat32 ? f32OutPipelines[tokens - 1] : pipelines[tokens - 1])
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var tValue = UInt32(tokens)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&tValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (m + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }
}
