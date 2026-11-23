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
        try commandBuffer.inkling2Compute {
            try encode(encoder: $0, states: states, queries: queries, queryCount: queryCount, output: output,
                       outputStride: outputStride, rows: rows, width: width, eps: eps)
        }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encode(encoder: MTLComputeCommandEncoder,
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
    }

    /// One query's norm in `encodeNormed`: its norm weight (F32 `[width]`), its FP16 output and,
    /// optionally, the same normed rows in FP32 (the MoE router's input).
    public struct Norm {
        public var weight: Inkling2BufferRef
        public var output: Inkling2BufferRef
        public var output32: Inkling2BufferRef?

        public init(weight: Inkling2BufferRef, output: Inkling2BufferRef, output32: Inkling2BufferRef? = nil) {
            self.weight = weight
            self.output = output
            self.output32 = output32
        }
    }

    /// The mix, then each query's output normed with its own weight (`Inkling2InputNorm`'s
    /// weighted RMS norm), in one dispatch: for the visits where nothing sits between the two
    /// (first-pass attention inputs; the MoE input outside the loop passes). `output` takes the
    /// mix as in `encode` (scratch); each `norms[j].output` gets the same bytes as `encode`
    /// followed by `Inkling2InputNorm.encode` on query `j`. One to three queries; every norm
    /// either has `output32` or none does.
    public func encodeNormed(commandBuffer: MTLCommandBuffer,
                             states: [(ref: Inkling2BufferRef, stride: Int)],
                             queries: Inkling2BufferRef, queryCount: Int,
                             output: Inkling2BufferRef, outputStride: Int,
                             norms: [Norm], normedStride: Int, normed32Stride: Int = 0,
                             rows: Int, width: Int = Inkling2AttentionShape.hidden,
                             eps: Float = Inkling2AttentionShape.rmsEps,
                             normEps: Float = Inkling2AttentionShape.rmsEps) throws {
        try commandBuffer.inkling2Compute {
            try encodeNormed(encoder: $0, states: states, queries: queries, queryCount: queryCount, output: output,
                             outputStride: outputStride, norms: norms, normedStride: normedStride,
                             normed32Stride: normed32Stride, rows: rows, width: width, eps: eps, normEps: normEps)
        }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encodeNormed(encoder: MTLComputeCommandEncoder,
                             states: [(ref: Inkling2BufferRef, stride: Int)],
                             queries: Inkling2BufferRef, queryCount: Int,
                             output: Inkling2BufferRef, outputStride: Int,
                             norms: [Norm], normedStride: Int, normed32Stride: Int = 0,
                             rows: Int, width: Int = Inkling2AttentionShape.hidden,
                             eps: Float = Inkling2AttentionShape.rmsEps,
                             normEps: Float = Inkling2AttentionShape.rmsEps) throws {
        precondition(rows > 0 && (1...Self.maxStates).contains(states.count))
        precondition((1...3).contains(queryCount) && norms.count == queryCount && outputStride >= queryCount * width)
        let dual = norms[0].output32 != nil
        precondition(norms.allSatisfy { ($0.output32 != nil) == dual } && (dual == (normed32Stride > 0)))
        var strides = [UInt32](repeating: 0, count: 6)
        for (i, state) in states.enumerated() { strides[i] = UInt32(state.stride) }
        var params = Inkling2MixParams(rows: UInt32(rows), width: UInt32(width),
                                       stateCount: UInt32(states.count), queryCount: UInt32(queryCount),
                                       stateStride: (strides[0], strides[1], strides[2], strides[3], strides[4], strides[5]),
                                       outStride: UInt32(outputStride), eps: eps)
        var normParams = Inkling2MixNormParams(normedStride: UInt32(normedStride),
                                               normed32Stride: UInt32(dual ? normed32Stride : 0),
                                               normEps: normEps, reserved: 0)
        let pipeline = try library.pipeline("inkling2_depth_mix_norm", constants: [
            .init(index: 200, value: .uint32(UInt32(states.count))),
            .init(index: 201, value: .uint32(UInt32(queryCount))),
        ])
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2MixParams>.stride, index: 0)
        for slot in 0..<Self.maxStates {
            let ref = states[min(slot, states.count - 1)].ref
            encoder.setBuffer(ref.buffer, offset: ref.offset, index: 1 + slot)
        }
        encoder.setBuffer(queries.buffer, offset: queries.offset, index: 7)
        encoder.setBuffer(output.buffer, offset: output.offset, index: 8)
        encoder.setBytes(&normParams, length: MemoryLayout<Inkling2MixNormParams>.stride, index: 9)
        for j in 0..<3 {
            let norm = norms[min(j, norms.count - 1)]          // unused slots repeat the last query's
            encoder.setBuffer(norm.weight.buffer, offset: norm.weight.offset, index: 10 + j)
            encoder.setBuffer(norm.output.buffer, offset: norm.output.offset, index: 13 + j)
            let wide = norm.output32 ?? output                  // bound but unread without FP32 outputs
            encoder.setBuffer(wide.buffer, offset: wide.offset, index: 16 + j)
        }
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

/// Mirrors `Inkling2MixNormParams` in `inkling2_attention.metal`.
struct Inkling2MixNormParams {
    var normedStride: UInt32
    var normed32Stride: UInt32
    var normEps: Float
    var reserved: UInt32
}

/// The loop's message gate, `(1 - tanh(F z)) x + tanh(G z) m` with
/// `z = W_down [rms(x); rms(m)]`, all in FP32 as llama.cpp's `control` and the
/// CPU references compute it (the vendor keeps the forget and gain products
/// in FP32 and runs the down product in the model dtype), in
/// four dispatches per visit's controllers (query, key, value; the MoE input
/// controller the same way):
/// 1. `encodeInputs`: `h = [rms(x); rms(m)]`, FP32 `[rows][2 * width]`, per controller;
/// 2. `Inkling2Projection.encodeFloat` with each controller's `loop_*_ctrl_down`
///    (`.f32`, 4608 inputs, 64 rows) into `z`; the three attention controllers
///    share one dispatch;
/// 3. `Inkling2Projection.encodeFloat` with `loop_*_ctrl_forget` and
///    `loop_*_ctrl_gain` (`.f32`, 64 inputs, 2304 rows each) on `z` into `fg`;
/// 4. `encodeCombine`: the elementwise mix.
public final class Inkling2MessageGateKernels {
    private let inputs: MTLComputePipelineState
    private let combine: MTLComputePipelineState
    private let downDecode: MTLComputePipelineState
    private let combineDecode: MTLComputePipelineState

    public init(library: Inkling2KernelLibrary) throws {
        inputs = try library.pipeline("inkling2_gate_inputs")
        combine = try library.pipeline("inkling2_gate_combine")
        downDecode = try library.pipeline("inkling2_gate_down_decode")
        combineDecode = try library.pipeline("inkling2_gate_combine_decode")
    }

    /// One controller of a single-row (decode) gate: byte offsets of its F32
    /// weights in the weight buffer, float offsets of its rows in the
    /// activation buffer.
    public struct DecodeController: Sendable, Equatable {
        public var down: Int
        public var forget: Int
        public var gain: Int
        public var x: Int
        public var message: Int
        public var output: Int

        public init(down: Int, forget: Int, gain: Int, x: Int, message: Int, output: Int) {
            self.down = down
            self.forget = forget
            self.gain = gain
            self.x = x
            self.message = message
            self.output = output
        }
    }

    /// The gate for one row and up to four controllers in two dispatches (the
    /// many-row path takes about ten for three controllers). `z` is FP32
    /// scratch of `controllers.count * rank` floats, 16-byte aligned. An output
    /// may alias its own `x` only; it may not overlap any other row of the call.
    /// Not byte-identical to the many-row path (the down product sums
    /// `inv_x (W_x . x) + inv_m (W_m . m)` per simdgroup); agreement is within
    /// FP32 rounding.
    public func encodeDecode(commandBuffer: MTLCommandBuffer, weights: MTLBuffer, activations: MTLBuffer,
                             z: Inkling2BufferRef, controllers: [DecodeController],
                             width: Int = Inkling2AttentionShape.hidden, rank: Int = 64,
                             eps: Float = Inkling2AttentionShape.rmsEps) {
        commandBuffer.inkling2Compute {
            encodeDecode(encoder: $0, weights: weights, activations: activations, z: z,
                         controllers: controllers, width: width, rank: rank, eps: eps)
        }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encodeDecode(encoder: MTLComputeCommandEncoder, weights: MTLBuffer, activations: MTLBuffer,
                             z: Inkling2BufferRef, controllers: [DecodeController],
                             width: Int = Inkling2AttentionShape.hidden, rank: Int = 64,
                             eps: Float = Inkling2AttentionShape.rmsEps) {
        precondition((1...4).contains(controllers.count) && width % 128 == 0 && rank % 8 == 0 && rank <= 128)
        precondition(z.offset % 16 == 0 && z.buffer.length - z.offset >= controllers.count * rank * 4)
        func overlaps(_ a: Int, _ b: Int) -> Bool { a < b + width && b < a + width }
        for (i, c) in controllers.enumerated() {
            precondition([c.down, c.forget, c.gain].allSatisfy { $0 % 16 == 0 })
            precondition([c.x, c.message, c.output].allSatisfy { $0 % 4 == 0 })
            precondition(!overlaps(c.output, c.message), "controller \(i) output overwrites its message")
            precondition(c.output == c.x || !overlaps(c.output, c.x), "controller \(i) output partly overlaps its x")
            for (j, other) in controllers.enumerated() where j != i {
                precondition(!overlaps(c.output, other.x) && !overlaps(c.output, other.message)
                             && !overlaps(c.output, other.output),
                             "controller \(i) output overlaps controller \(j) rows")
            }
        }
        var bytes = [UInt8](repeating: 0, count: 16 + 4 * 40)
        func put32(_ v: Int, _ o: Int) { for k in 0..<4 { bytes[o + k] = UInt8(truncatingIfNeeded: UInt32(v) >> (8 * UInt32(k))) } }
        func put64(_ v: Int, _ o: Int) { for k in 0..<8 { bytes[o + k] = UInt8(truncatingIfNeeded: UInt64(v) >> (8 * UInt64(k))) } }
        put32(width, 0)
        put32(rank, 4)
        put32(controllers.count, 8)
        withUnsafeBytes(of: eps.bitPattern) { for k in 0..<4 { bytes[12 + k] = $0[k] } }
        for (i, c) in controllers.enumerated() {
            let base = 16 + i * 40
            put64(c.down, base)
            put64(c.forget, base + 8)
            put64(c.gain, base + 16)
            put32(c.x, base + 24)
            put32(c.message, base + 28)
            put32(c.output, base + 32)
        }
        encoder.setComputePipelineState(downDecode)
        bytes.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(activations, offset: 0, index: 2)
        encoder.setBuffer(z.buffer, offset: z.offset, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: controllers.count, height: rank / 2, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.setComputePipelineState(combineDecode)
        bytes.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(activations, offset: 0, index: 2)
        encoder.setBuffer(z.buffer, offset: z.offset, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: width / 16, height: controllers.count, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    public func encodeInputs(commandBuffer: MTLCommandBuffer,
                             x: Inkling2BufferRef, xStride: Int,
                             message: Inkling2BufferRef, messageStride: Int,
                             output: Inkling2BufferRef, outputStride: Int,
                             rows: Int, width: Int = Inkling2AttentionShape.hidden,
                             eps: Float = Inkling2AttentionShape.rmsEps) {
        commandBuffer.inkling2Compute {
            encodeInputs(encoder: $0, x: x, xStride: xStride, message: message, messageStride: messageStride,
                         output: output, outputStride: outputStride, rows: rows, width: width, eps: eps)
        }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encodeInputs(encoder: MTLComputeCommandEncoder,
                             x: Inkling2BufferRef, xStride: Int,
                             message: Inkling2BufferRef, messageStride: Int,
                             output: Inkling2BufferRef, outputStride: Int,
                             rows: Int, width: Int = Inkling2AttentionShape.hidden,
                             eps: Float = Inkling2AttentionShape.rmsEps) {
        precondition(rows > 0 && outputStride >= 2 * width)
        var params = Inkling2GateInputParams(rows: UInt32(rows), width: UInt32(width), xStride: UInt32(xStride),
                                             mStride: UInt32(messageStride), outStride: UInt32(outputStride), eps: eps)
        encoder.setComputePipelineState(inputs)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2GateInputParams>.stride, index: 0)
        encoder.setBuffer(x.buffer, offset: x.offset, index: 1)
        encoder.setBuffer(message.buffer, offset: message.offset, index: 2)
        encoder.setBuffer(output.buffer, offset: output.offset, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// `gates`: FP32 `[rows][gatesStride]` holding `F z` at `forgetColumn` and
    /// `G z` at `gainColumn`. `output` may alias `x`.
    public func encodeCombine(commandBuffer: MTLCommandBuffer,
                              x: Inkling2BufferRef, xStride: Int,
                              message: Inkling2BufferRef, messageStride: Int,
                              gates: Inkling2BufferRef, gatesStride: Int, forgetColumn: Int, gainColumn: Int,
                              output: Inkling2BufferRef, outputStride: Int,
                              rows: Int, width: Int = Inkling2AttentionShape.hidden) {
        commandBuffer.inkling2Compute {
            encodeCombine(encoder: $0, x: x, xStride: xStride, message: message, messageStride: messageStride,
                          gates: gates, gatesStride: gatesStride, forgetColumn: forgetColumn,
                          gainColumn: gainColumn, output: output, outputStride: outputStride, rows: rows,
                          width: width)
        }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encodeCombine(encoder: MTLComputeCommandEncoder,
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
        encoder.setComputePipelineState(combine)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2GateCombineParams>.stride, index: 0)
        for (index, ref) in [x, message, gates, output].enumerated() {
            encoder.setBuffer(ref.buffer, offset: ref.offset, index: index + 1)
        }
        encoder.dispatchThreads(MTLSize(width: width, height: rows, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
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
