import Metal

/// Multi-token fused greedy lm_head for the MTP speculative-verify pass.
/// Consumes pre-normalized hidden rows `[T, d]` and emits one greedy token
/// per row into `outTokens` (`[T]` UInt32). Row logits and argmax semantics
/// are bit-identical to `LMHeadChainInt4`'s single-token pair (see
/// `logit.metal`), while the packed lm_head is read once for all rows.
final class LMHeadGreedyMultiX {
    static let maxTokens = 8
    private static let rowsPerThreadgroup = 8
    private static let rowSummaryStride = 2

    private let rowGreedyByTokens: [MTLComputePipelineState] // index = T - 1
    private let rowReducer: MTLComputePipelineState
    private let summariesBuffer: MTLBuffer
    private let maxVocab: Int

    init(context: MetalContext, maxVocab: Int) throws {
        // Safe-math module compile: see `DequantInt4GEMVMultiX` — the
        // unrolled per-token chains must not be reassociated.
        let library = try MetalContext.moduleLibrary(device: context.device,
                                                     module: "logit",
                                                     safeMath: true)
        self.rowGreedyByTokens = try (1...Self.maxTokens).map { tokens in
            let values = MTLFunctionConstantValues()
            var t = UInt32(tokens)
            var use = true
            values.setConstantValue(&t, type: .uint, index: 47)
            values.setConstantValue(&use, type: .bool, index: 48)
            let function = try library.makeFunction(
                name: "lm_head_greedy_int4_rows_chunk_raw_multix",
                constantValues: values)
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function
            descriptor.maxTotalThreadsPerThreadgroup = 256
            var reflection: MTLAutoreleasedComputePipelineReflection?
            return try context.device.makeComputePipelineState(descriptor: descriptor,
                                                               options: [],
                                                               reflection: &reflection)
        }
        self.rowReducer = try context.pipeline(
            "lm_head_greedy_int4_rows_reduce_multix",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        self.maxVocab = maxVocab
        let rowGroups = (maxVocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        let length = rowGroups * Self.maxTokens * Self.rowSummaryStride
            * MemoryLayout<Float>.size
        guard let summaries = context.device.makeBuffer(
            length: length, options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        summaries.label = "mtp.head.summaries"
        self.summariesBuffer = summaries
    }

    /// `xNormed` holds `tokens` rows of final-norm output, `[T, d]` FP16.
    func encode(commandBuffer: MTLCommandBuffer,
                xNormed: MTLBuffer, xNormedOffset: Int = 0,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                outTokens: MTLBuffer,
                d: Int, vocab: Int, tokens: Int) {
        precondition(vocab <= maxVocab, "vocab \(vocab) exceeds wrapper maxVocab \(maxVocab)")
        precondition(d % Quantization.groupSize == 0,
                     "d must be a multiple of \(Quantization.groupSize)")
        precondition(tokens > 0 && tokens <= Self.maxTokens,
                     "token count \(tokens) outside 1...\(Self.maxTokens)")
        precondition(weightsOffset % 2 == 0,
                     "lm_head multix needs a 2-aligned weightsOffset")
        let rowGroups = (vocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(rowGreedyByTokens[tokens - 1])
            encoder.setBuffer(xNormed, offset: xNormedOffset, index: 0)
            encoder.setBuffer(weights, offset: weightsOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            encoder.setBuffer(biases, offset: biasesOffset, index: 3)
            encoder.setBuffer(summariesBuffer, offset: 0, index: 4)
            var dValue = UInt32(d)
            var vocabValue = UInt32(vocab)
            var tValue = UInt32(tokens)
            encoder.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.setBytes(&tValue, length: MemoryLayout<UInt32>.size, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                               height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(rowReducer)
            encoder.setBuffer(summariesBuffer, offset: 0, index: 0)
            encoder.setBuffer(outTokens, offset: 0, index: 1)
            var rowGroupCount = UInt32(rowGroups)
            var tValue = UInt32(tokens)
            encoder.setBytes(&rowGroupCount, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&tValue, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: tokens, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }
}
