import Metal

/// Native Flash-Next MTP input fusion, not a complete/enabled draft decoder.
/// Unlike dense Qwen MTP, normalize the ENTIRE target HC bundle, project each
/// stream with the same hidden matrix, and add the embedding projection to
/// every stream. Norm weights must already contain the zero-centered +1 fold.
/// Contract: pinned SGLang qwen4_exp_mtp._fuse_residual_linear_shared at
/// 745de73ba3c136b6f99b7a3e2177ed1a8eef4a56. Keep this out of ordinary decode
/// until sidecar loading, target verification and performance gates pass.
final class FlashNextMTPInputFusion {
    private let hidden: Int
    private let streams: Int
    private let normPipeline: MTLComputePipelineState
    private let projector: FlashNextMTPFloat32Projection
    private let addPipeline: MTLComputePipelineState
    private let normEmbedding: MTLBuffer
    private let normHidden: MTLBuffer
    private let projectedEmbedding: MTLBuffer
    private let projectedHidden: MTLBuffer
    private let outputFloat32: Bool

    init(context: MetalContext, hidden: Int, streams: Int, normWeightsFloat32: Bool = false, outputFloat32: Bool = false) throws {
        precondition(hidden > 0 && streams > 1)
        self.hidden = hidden
        self.streams = streams
        self.outputFloat32 = outputFloat32
        self.normPipeline = try context.pipeline("flashnext_mtp_norm_f32",
            constants: [.init(index: 401, value: .bool(normWeightsFloat32))])
        self.projector = try FlashNextMTPFloat32Projection(context: context)
        self.addPipeline = try context.pipeline(outputFloat32 ? "flashnext_mtp_add_wide" : "flashnext_mtp_add_f32", constants: [])
        func buffer(_ count: Int) throws -> MTLBuffer {
            guard let result = context.device.makeBuffer(length: count * 4,
                                                         options: .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return result
        }
        self.normEmbedding = try buffer(hidden)
        self.normHidden = try buffer(hidden * streams)
        self.projectedEmbedding = try buffer(hidden)
        self.projectedHidden = try buffer(hidden * streams)
    }

    /// One FP16 embedding row [D] and one target HC row [H*D]. Output [H*D],
    /// FP16 by default or FP32 when selected at construction.
    /// The two FC matrices are [D,D]; norms are [D] and [H*D], respectively.
    /// Norm storage is BF16 by default, or FP32 when selected at construction.
    /// Inputs are never modified; the output must not alias either input.
    func encode(commandBuffer: MTLCommandBuffer,
                embedding: MTLBuffer, targetHidden: MTLBuffer,
                embeddingNorm: MTLBuffer, embeddingNormOffset: Int = 0,
                hiddenNorm: MTLBuffer, hiddenNormOffset: Int = 0,
                embeddingProjection: FlashNextWeightMatrix,
                hiddenProjection: FlashNextWeightMatrix,
                output: MTLBuffer, epsilon: Float = 1e-6) {
        precondition(embedding.length >= hidden * 2 && targetHidden.length >= hidden * streams * 2)
        precondition(output.length >= hidden * streams * (outputFloat32 ? 4 : 2))
        precondition(output !== embedding && output !== targetHidden)
        func normalize(_ x: MTLBuffer, _ weight: MTLBuffer, _ offset: Int,
                       _ out: MTLBuffer, _ count: Int) {
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(normPipeline)
            encoder.setBuffer(x, offset: 0, index: 0)
            encoder.setBuffer(weight, offset: offset, index: 1)
            encoder.setBuffer(out, offset: 0, index: 2)
            var n = UInt32(count), eps = epsilon
            encoder.setBytes(&n, length: 4, index: 3)
            encoder.setBytes(&eps, length: 4, index: 4)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
        func project(_ matrix: FlashNextWeightMatrix, _ x: MTLBuffer, _ xOffset: Int,
                     _ out: MTLBuffer, _ outOffset: Int) {
            projector.encode(commandBuffer: commandBuffer, matrix: matrix, x: x, xOffset: xOffset,
                out: out, outOffset: outOffset, rows: hidden, columns: hidden)
        }
        normalize(embedding, embeddingNorm, embeddingNormOffset, normEmbedding, hidden)
        // Normalize the complete bundle, not each HC stream independently.
        normalize(targetHidden, hiddenNorm, hiddenNormOffset, normHidden, hidden * streams)
        project(embeddingProjection, normEmbedding, 0, projectedEmbedding, 0)
        for stream in 0..<streams {
            let offset = stream * hidden * 4
            project(hiddenProjection, normHidden, offset, projectedHidden, offset)
        }
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(addPipeline)
        encoder.setBuffer(projectedEmbedding, offset: 0, index: 0)
        encoder.setBuffer(projectedHidden, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var d = UInt32(hidden)
        encoder.setBytes(&d, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: hidden * streams, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
