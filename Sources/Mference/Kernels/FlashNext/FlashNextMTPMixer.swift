import Metal

/// Native HC mix/injection and final head, retaining FP32 intermediates.
/// Mix and inject must be paired before starting the next HC site. The public
/// draft logits remain FP16, matching the sampler ABI.
final class FlashNextMTPMixer {
    private let hidden: Int
    private let streams: Int
    private let rank: Int
    private let projection: FlashNextMTPFloat32Projection
    private let normalize: MTLComputePipelineState
    private let activation: MTLComputePipelineState
    private let mix: MTLComputePipelineState
    private let narrow: MTLComputePipelineState
    private let inject: MTLComputePipelineState
    private let injection: MTLBuffer
    private let normed: MTLBuffer
    private let lowRank: MTLBuffer
    private let gates: MTLBuffer
    private let mixed: MTLBuffer
    private let logits: MTLBuffer
    private let vocab: Int

    init(context: MetalContext, hidden: Int, streams: Int, rank: Int, vocab: Int) throws {
        precondition(hidden > 0 && streams > 0 && rank > 0 && vocab > 0)
        self.hidden = hidden; self.streams = streams; self.rank = rank; self.vocab = vocab
        projection = try FlashNextMTPFloat32Projection(context: context)
        normalize = try context.pipeline("flashnext_mtp_group_norm_f32")
        activation = try context.pipeline("flashnext_mtp_lowrank_f32")
        mix = try context.pipeline("flashnext_mtp_mix_f32")
        narrow = try context.pipeline("flashnext_mtp_float_to_half")
        inject = try context.pipeline("flashnext_mtp_inject_f32")
        func buffer(_ count: Int) throws -> MTLBuffer {
            guard let b = context.device.makeBuffer(length: count * 4, options: .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return b
        }
        normed = try buffer(hidden * streams); lowRank = try buffer(rank)
        gates = try buffer(hidden * streams); mixed = try buffer(hidden); logits = try buffer(vocab)
        injection = try buffer(streams)
    }

    func encode(commandBuffer cb: MTLCommandBuffer, weights: FlashNextHyperConnections.Weights,
                head: FlashNextWeightMatrix, hyper: MTLBuffer, output: MTLBuffer) {
        precondition(output.length >= vocab * 2)
        encodeMix(commandBuffer: cb, weights: weights, hyper: hyper, output: mixed)
        projection.encode(commandBuffer: cb, matrix: head, x: mixed,
            out: logits, rows: vocab, columns: hidden)
        dispatch(cb, narrow, count: vocab) {
            $0.setBuffer(logits, offset: 0, index: 0)
            $0.setBuffer(output, offset: 0, index: 1)
        }
    }

    private func dispatch(_ cb: MTLCommandBuffer, _ pipeline: MTLComputePipelineState, count: Int,
                          _ configure: (MTLComputeCommandEncoder) -> Void) {
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pipeline)
        configure(e)
        e.dispatchThreads(.init(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: .init(width: min(32, count), height: 1, depth: 1))
        e.endEncoding()
    }

    func encodeMix(commandBuffer cb: MTLCommandBuffer, weights: FlashNextHyperConnections.Weights,
                   hyper: MTLBuffer, output: MTLBuffer) {
        precondition(hyper.length >= hidden * streams * 4 && output.length >= hidden * 4)
        precondition(weights.normOffset >= 0 && weights.normOffset + hidden * streams * 4 <= weights.norm.length)
        precondition(output !== hyper)
        var d = UInt32(hidden), h = UInt32(streams)
        dispatch(cb, normalize, count: streams) {
            $0.setBuffer(hyper, offset: 0, index: 0)
            $0.setBuffer(weights.norm, offset: weights.normOffset, index: 1)
            $0.setBuffer(normed, offset: 0, index: 2)
            $0.setBytes(&d, length: 4, index: 3)
        }
        projection.encode(commandBuffer: cb, matrix: weights.mixDown, x: normed,
            out: lowRank, rows: rank, columns: hidden * streams)
        dispatch(cb, activation, count: rank) {
            $0.setBuffer(lowRank, offset: 0, index: 0)
            $0.setBytes(&h, length: 4, index: 1)
        }
        projection.encode(commandBuffer: cb, matrix: weights.mixUp, x: lowRank,
            out: gates, rows: hidden * streams, columns: rank)
        dispatch(cb, mix, count: hidden) {
            $0.setBuffer(normed, offset: 0, index: 0)
            $0.setBuffer(gates, offset: 0, index: 1)
            $0.setBuffer(output, offset: 0, index: 2)
            $0.setBytes(&d, length: 4, index: 3)
            $0.setBytes(&h, length: 4, index: 4)
        }
        if let matrix = weights.inject {
            projection.encode(commandBuffer: cb, matrix: matrix, x: normed,
                out: injection, rows: streams, columns: hidden * streams)
        }
    }

    /// Consumes the injection projection left by the immediately preceding mix.
    func encodeInject(commandBuffer cb: MTLCommandBuffer, hyper: MTLBuffer, block: MTLBuffer) {
        var d = UInt32(hidden), h = UInt32(streams)
        dispatch(cb, inject, count: hidden * streams) {
            $0.setBuffer(hyper, offset: 0, index: 0)
            $0.setBuffer(block, offset: 0, index: 1)
            $0.setBuffer(injection, offset: 0, index: 2)
            $0.setBytes(&d, length: 4, index: 3)
            $0.setBytes(&h, length: 4, index: 4)
        }
    }
}
