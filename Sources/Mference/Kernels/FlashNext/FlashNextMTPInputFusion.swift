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
    private let rms: RMSNorm
    private let matVec: FlashNextMatVec
    private let elementwise: Elementwise
    private let normEmbedding: MTLBuffer
    private let normHidden: MTLBuffer
    private let projectedEmbedding: MTLBuffer

    init(context: MetalContext, hidden: Int, streams: Int) throws {
        precondition(hidden > 0 && streams > 1)
        self.hidden = hidden
        self.streams = streams
        self.rms = try RMSNorm(context: context)
        self.matVec = try FlashNextMatVec(context: context,
            int4: DequantInt4GEMV(context: context), int8Columns: hidden)
        self.elementwise = try Elementwise(context: context)
        func buffer(_ count: Int) throws -> MTLBuffer {
            guard let result = context.device.makeBuffer(length: count * 2,
                                                         options: .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return result
        }
        self.normEmbedding = try buffer(hidden)
        self.normHidden = try buffer(hidden * streams)
        self.projectedEmbedding = try buffer(hidden)
    }

    /// One FP16 embedding row [D] and one target HC row [H*D]. Output [H*D].
    /// The two FC matrices are [D,D]; BF16 norms are [D] and [H*D], respectively.
    /// Inputs are never modified; the output must not alias either input.
    func encode(commandBuffer: MTLCommandBuffer,
                embedding: MTLBuffer, targetHidden: MTLBuffer,
                embeddingNorm: MTLBuffer, embeddingNormOffset: Int = 0,
                hiddenNorm: MTLBuffer, hiddenNormOffset: Int = 0,
                embeddingProjection: FlashNextWeightMatrix,
                hiddenProjection: FlashNextWeightMatrix,
                output: MTLBuffer, epsilon: Float = 1e-6) {
        precondition(embedding.length >= hidden * 2 && targetHidden.length >= hidden * streams * 2)
        precondition(output.length >= hidden * streams * 2)
        precondition(output !== embedding && output !== targetHidden)
        rms.encodeBF16W(commandBuffer: commandBuffer, x: embedding,
            weight: embeddingNorm, weightOffset: embeddingNormOffset, out: normEmbedding,
            d: UInt32(hidden), eps: epsilon)
        // Not the trunk's grouped HC RMSNorm: the draft's pre-FC normalization
        // reduces across all streams, including streams with different scales.
        rms.encodeBF16W(commandBuffer: commandBuffer, x: targetHidden,
            weight: hiddenNorm, weightOffset: hiddenNormOffset, out: normHidden,
            d: UInt32(hidden * streams), eps: epsilon)
        matVec.encode(commandBuffer: commandBuffer, matrix: embeddingProjection,
            x: normEmbedding, y: projectedEmbedding, rows: hidden, cols: hidden)
        for stream in 0..<streams {
            let offset = stream * hidden * 2
            matVec.encode(commandBuffer: commandBuffer, matrix: hiddenProjection,
                x: normHidden, xOffset: offset, y: output, yOffset: offset,
                rows: hidden, cols: hidden)
            elementwise.encodeResidualAdd(commandBuffer: commandBuffer,
                hidden: output, hiddenOffset: offset, delta: projectedEmbedding, count: hidden)
        }
    }
}
