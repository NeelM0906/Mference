import Foundation
import Metal

/// Depth mix: the block attention residuals (layer inputs and MoE inputs mix
/// the closed block sums and the current partial sum) and the loop's
/// cross-pass message retrieval (four queries over the previous pass's two
/// states). Per token every state is normalised with a weightless RMS norm,
/// each query scores the states with scale `1 / sqrt(width)`, and output `j`
/// is the softmax-weighted sum of the normalised states. FP32 in and out.
public final class Inkling2DepthMix {
    public static let maxStates = 6
    public static let maxQueries = 4

    private let pipeline: MTLComputePipelineState

    public init(library: Inkling2KernelLibrary) throws {
        pipeline = try library.pipeline("inkling2_depth_mix")
    }

    /// `states`: up to 6 FP32 `[rows][stride]` inputs; `queries`: FP32
    /// `[queryCount][width]` (GGUF `attn_res_query`, `ffn_res_query`,
    /// `output_res_query`, `loop_msg_query`). Output row `t` holds query `j`
    /// at columns `j * width ..< (j + 1) * width`.
    public func encode(commandBuffer: MTLCommandBuffer,
                       states: [(ref: Inkling2BufferRef, stride: Int)],
                       queries: Inkling2BufferRef, queryCount: Int,
                       output: Inkling2BufferRef, outputStride: Int,
                       rows: Int, width: Int = Inkling2AttentionShape.hidden,
                       eps: Float = Inkling2AttentionShape.rmsEps) {
        precondition(rows > 0 && (1...Self.maxStates).contains(states.count))
        precondition((1...Self.maxQueries).contains(queryCount) && outputStride >= queryCount * width)
        var strides = [UInt32](repeating: 0, count: 6)
        for (i, state) in states.enumerated() { strides[i] = UInt32(state.stride) }
        var params = Inkling2MixParams(rows: UInt32(rows), width: UInt32(width),
                                       stateCount: UInt32(states.count), queryCount: UInt32(queryCount),
                                       stateStride: (strides[0], strides[1], strides[2], strides[3], strides[4], strides[5]),
                                       outStride: UInt32(outputStride), eps: eps)
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2MixParams>.stride, index: 0)
        for slot in 0..<Self.maxStates {
            let ref = states[min(slot, states.count - 1)].ref
            encoder.setBuffer(ref.buffer, offset: ref.offset, index: 1 + slot)
        }
        encoder.setBuffer(queries.buffer, offset: queries.offset, index: 7)
        encoder.setBuffer(output.buffer, offset: output.offset, index: 8)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

/// The loop's message gate, `(1 - tanh(F z)) x + tanh(G z) m` with
/// `z = W_down [rms(x); rms(m)]`, in three steps per controller (query, key,
/// value or MoE input):
/// 1. `encodeInputs`: `h = half([rms(x); rms(m)])`, FP16 `[rows][2 * width]`;
/// 2. `Inkling2Projection` with the controller's `loop_*_ctrl_down` as an
///    `.f32` segment (`inputs` 4608, 64 rows) into `z`; the three attention
///    controllers of a visit fit one projection dispatch;
/// 3. `encodeCombine`: forget and gain products in FP32 from `z` and the mix.
public final class Inkling2MessageGate {
    private let inputs: MTLComputePipelineState
    private let combine: MTLComputePipelineState

    public init(library: Inkling2KernelLibrary) throws {
        inputs = try library.pipeline("inkling2_gate_inputs")
        combine = try library.pipeline("inkling2_gate_combine")
    }

    public func encodeInputs(commandBuffer: MTLCommandBuffer,
                             x: Inkling2BufferRef, xStride: Int,
                             message: Inkling2BufferRef, messageStride: Int,
                             output: Inkling2BufferRef, outputStride: Int,
                             rows: Int, width: Int = Inkling2AttentionShape.hidden,
                             eps: Float = Inkling2AttentionShape.rmsEps) {
        precondition(rows > 0 && outputStride >= 2 * width)
        var params = Inkling2GateInputParams(rows: UInt32(rows), width: UInt32(width), xStride: UInt32(xStride),
                                             mStride: UInt32(messageStride), outStride: UInt32(outputStride), eps: eps)
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(inputs)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2GateInputParams>.stride, index: 0)
        encoder.setBuffer(x.buffer, offset: x.offset, index: 1)
        encoder.setBuffer(message.buffer, offset: message.offset, index: 2)
        encoder.setBuffer(output.buffer, offset: output.offset, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// `forget`, `gain`: FP32 `[width][rank]` (GGUF `loop_*_ctrl_forget`,
    /// `loop_*_ctrl_gain`). `output` may alias `x`.
    public func encodeCombine(commandBuffer: MTLCommandBuffer,
                              x: Inkling2BufferRef, xStride: Int,
                              message: Inkling2BufferRef, messageStride: Int,
                              z: Inkling2BufferRef, zStride: Int, zColumn: Int,
                              forget: Inkling2BufferRef, gain: Inkling2BufferRef,
                              output: Inkling2BufferRef, outputStride: Int,
                              rows: Int, width: Int = Inkling2AttentionShape.hidden, rank: Int = 64) {
        precondition(rows > 0 && rank % 4 == 0 && rank <= 128)
        precondition(forget.offset % 16 == 0 && gain.offset % 16 == 0)
        var params = Inkling2GateCombineParams(rows: UInt32(rows), width: UInt32(width), rank: UInt32(rank),
                                               xStride: UInt32(xStride), mStride: UInt32(messageStride),
                                               zStride: UInt32(zStride), zColumn: UInt32(zColumn),
                                               outStride: UInt32(outputStride))
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(combine)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2GateCombineParams>.stride, index: 0)
        for (index, ref) in [x, message, z, forget, gain, output].enumerated() {
            encoder.setBuffer(ref.buffer, offset: ref.offset, index: index + 1)
        }
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

// Mirrors of the parameter structs in `inkling2_attention.metal`.

struct Inkling2MixParams {
    var rows: UInt32
    var width: UInt32
    var stateCount: UInt32
    var queryCount: UInt32
    var stateStride: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    var outStride: UInt32
    var eps: Float
}

struct Inkling2GateInputParams {
    var rows: UInt32
    var width: UInt32
    var xStride: UInt32
    var mStride: UInt32
    var outStride: UInt32
    var eps: Float
}

struct Inkling2GateCombineParams {
    var rows: UInt32
    var width: UInt32
    var rank: UInt32
    var xStride: UInt32
    var mStride: UInt32
    var zStride: UInt32
    var zColumn: UInt32
    var outStride: UInt32
}
