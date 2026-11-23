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

    private let library: Inkling2KernelLibrary

    /// Builds the specialisations the model uses: layer and MoE inputs (1 to 6
    /// states, 1 query; global layers' attention inputs 3 queries) and message
    /// retrieval (2 states, 4 queries). Others build on first use.
    public init(library: Inkling2KernelLibrary) throws {
        self.library = library
        for states in 1...Self.maxStates {
            _ = try pipeline(states: states, queries: 1)
            _ = try pipeline(states: states, queries: 3)
        }
        _ = try pipeline(states: 2, queries: 4)
    }

    /// The kernel specialised for a state and query count (cached).
    func pipeline(states: Int, queries: Int) throws -> MTLComputePipelineState {
        try library.pipeline("inkling2_depth_mix", constants: [
            .init(index: 200, value: .uint32(UInt32(states))),
            .init(index: 201, value: .uint32(UInt32(queries))),
        ])
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
                       eps: Float = Inkling2AttentionShape.rmsEps) throws {
        precondition(rows > 0 && (1...Self.maxStates).contains(states.count))
        precondition((1...Self.maxQueries).contains(queryCount) && outputStride >= queryCount * width)
        var strides = [UInt32](repeating: 0, count: 6)
        for (i, state) in states.enumerated() { strides[i] = UInt32(state.stride) }
        var params = Inkling2MixParams(rows: UInt32(rows), width: UInt32(width),
                                       stateCount: UInt32(states.count), queryCount: UInt32(queryCount),
                                       stateStride: (strides[0], strides[1], strides[2], strides[3], strides[4], strides[5]),
                                       outStride: UInt32(outputStride), eps: eps)
        let pipeline = try self.pipeline(states: states.count, queries: queryCount)
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
/// `z = W_down [rms(x); rms(m)]`, all in FP32 as the vendor computes it, in
/// four dispatches per visit's controllers (query, key, value; the MoE input
/// controller the same way):
/// 1. `encodeInputs`: `h = [rms(x); rms(m)]`, FP32 `[rows][2 * width]`, per controller;
/// 2. `Inkling2Projection.encodeFloat` with each controller's `loop_*_ctrl_down`
///    (`.f32`, 4608 inputs, 64 rows) into `z`; the three attention controllers
///    share one dispatch;
/// 3. `Inkling2Projection.encodeFloat` with `loop_*_ctrl_forget` and
///    `loop_*_ctrl_gain` (`.f32`, 64 inputs, 2304 rows each) on `z` into `fg`;
/// 4. `encodeCombine`: the elementwise mix.
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

    /// `gates`: FP32 `[rows][gatesStride]` holding `F z` at `forgetColumn` and
    /// `G z` at `gainColumn`. `output` may alias `x`.
    public func encodeCombine(commandBuffer: MTLCommandBuffer,
                              x: Inkling2BufferRef, xStride: Int,
                              message: Inkling2BufferRef, messageStride: Int,
                              gates: Inkling2BufferRef, gatesStride: Int, forgetColumn: Int, gainColumn: Int,
                              output: Inkling2BufferRef, outputStride: Int,
                              rows: Int, width: Int = Inkling2AttentionShape.hidden) {
        precondition(rows > 0)
        var params = Inkling2GateCombineParams(rows: UInt32(rows), width: UInt32(width),
                                               xStride: UInt32(xStride), mStride: UInt32(messageStride),
                                               fgStride: UInt32(gatesStride), forgetColumn: UInt32(forgetColumn),
                                               gainColumn: UInt32(gainColumn), outStride: UInt32(outputStride))
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(combine)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2GateCombineParams>.stride, index: 0)
        for (index, ref) in [x, message, gates, output].enumerated() {
            encoder.setBuffer(ref.buffer, offset: ref.offset, index: index + 1)
        }
        encoder.dispatchThreads(MTLSize(width: width, height: rows, depth: 1),
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
    var xStride: UInt32
    var mStride: UInt32
    var fgStride: UInt32
    var forgetColumn: UInt32
    var gainColumn: UInt32
    var outStride: UInt32
}
