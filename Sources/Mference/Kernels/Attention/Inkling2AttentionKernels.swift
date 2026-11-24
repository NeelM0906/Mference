import Foundation
import Metal

/// Everything between a visit's stacked projection and its attention, in one
/// dispatch: query and relative-state clips, the key and value short
/// convolutions (static, or static plus per-token taps on global layers), the
/// key clip and value norm, and the FP16 stores into the visit's KV cache.
/// It reads the convolution tails but does not advance them: encode
/// `Inkling2ConvTail` on the same key and value columns afterwards, or, for a
/// single row (decode), set `advanceTails` and skip that dispatch.
public final class Inkling2QKVEpilogue {
    /// Column offsets (floats) inside one row of the FP32 projection output.
    public struct Columns: Sendable, Equatable {
        public var queries: Int
        public var relative: Int
        public var keys: Int
        public var values: Int
        /// The global layers' dynamic taps (8 x 4 each); nil on window layers.
        public var keyTaps: Int?
        public var valueTaps: Int?

        public init(queries: Int, relative: Int, keys: Int, values: Int,
                    keyTaps: Int? = nil, valueTaps: Int? = nil) {
            self.queries = queries
            self.relative = relative
            self.keys = keys
            self.values = values
            self.keyTaps = keyTaps
            self.valueTaps = valueTaps
        }
    }

    public struct Arguments {
        /// FP32 `[rows][projectionStride]`.
        public var projections: Inkling2BufferRef
        public var projectionStride: Int
        public var columns: Columns
        /// FP32 `[24][128]` and `[24][16]` (GGUF `attn_q_clip`, `attn_rel_clip`).
        public var queryScale: Inkling2BufferRef
        public var relativeScale: Inkling2BufferRef
        /// FP32 `[1024][4]`, taps oldest first (GGUF `attn_k_conv`, `attn_v_conv`).
        public var keyConv: Inkling2BufferRef
        public var valueConv: Inkling2BufferRef
        /// FP32 `[3][1024]` tails of the visit (pre-convolution inputs).
        public var keyTail: Inkling2BufferRef
        public var valueTail: Inkling2BufferRef
        /// FP16 `[rows][queryStride]` and FP32 `[rows][relativeStride]` outputs.
        public var queries: Inkling2BufferRef
        public var queryStride: Int
        public var relative: Inkling2BufferRef
        public var relativeStride: Int
        public var cache: Inkling2KVView
        public var rows: Int
        public var startPosition: Int
        /// One row only: advance the key and value tails in place in the same dispatch.
        public var advanceTails: Bool

        public init(projections: Inkling2BufferRef, projectionStride: Int, columns: Columns,
                    queryScale: Inkling2BufferRef, relativeScale: Inkling2BufferRef,
                    keyConv: Inkling2BufferRef, valueConv: Inkling2BufferRef,
                    keyTail: Inkling2BufferRef, valueTail: Inkling2BufferRef,
                    queries: Inkling2BufferRef, queryStride: Int = Inkling2AttentionShape.qWidth,
                    relative: Inkling2BufferRef, relativeStride: Int = Inkling2AttentionShape.relWidth,
                    cache: Inkling2KVView, rows: Int, startPosition: Int, advanceTails: Bool = false) {
            self.projections = projections
            self.projectionStride = projectionStride
            self.columns = columns
            self.queryScale = queryScale
            self.relativeScale = relativeScale
            self.keyConv = keyConv
            self.valueConv = valueConv
            self.keyTail = keyTail
            self.valueTail = valueTail
            self.queries = queries
            self.queryStride = queryStride
            self.relative = relative
            self.relativeStride = relativeStride
            self.cache = cache
            self.rows = rows
            self.startPosition = startPosition
            self.advanceTails = advanceTails
        }
    }

    private let pipeline: MTLComputePipelineState

    public init(library: Inkling2KernelLibrary) throws {
        pipeline = try library.pipeline("inkling2_qkv_epilogue")
    }

    public func encode(commandBuffer: MTLCommandBuffer, _ a: Arguments) {
        commandBuffer.inkling2Compute { encode(encoder: $0, a) }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encode(encoder: MTLComputeCommandEncoder, _ a: Arguments) {
        precondition(a.rows > 0)
        precondition((a.columns.keyTaps == nil) == (a.columns.valueTaps == nil))
        precondition(!a.advanceTails || a.rows == 1, "in-place tail advance needs one row")
        // float4 reads of the projection rows and half4 stores.
        precondition(a.projectionStride % 4 == 0 && a.projections.offset % 16 == 0)
        precondition([a.columns.queries, a.columns.keys, a.columns.values].allSatisfy { $0 % 4 == 0 })
        precondition(a.queryStride % 4 == 0 && a.cache.slotStride % 4 == 0)
        let dynamic = a.columns.keyTaps != nil
        var params = Inkling2QKVParams(
            tokens: UInt32(a.rows), startPosition: UInt32(a.startPosition),
            yStride: UInt32(a.projectionStride),
            qColumn: UInt32(a.columns.queries), relColumn: UInt32(a.columns.relative),
            kColumn: UInt32(a.columns.keys), vColumn: UInt32(a.columns.values),
            kTapColumn: UInt32(a.columns.keyTaps ?? 0), vTapColumn: UInt32(a.columns.valueTaps ?? 0),
            dynamicConv: dynamic ? 1 : 0,
            qOutStride: UInt32(a.queryStride), relOutStride: UInt32(a.relativeStride),
            kvStride: UInt32(a.cache.slotStride), ringCapacity: UInt32(a.cache.ringCapacity),
            advanceTails: a.advanceTails ? 1 : 0,
            qkClip: Inkling2AttentionShape.qkClip, relClip: Inkling2AttentionShape.relClip,
            clipEps: 1e-6, rmsEps: Inkling2AttentionShape.rmsEps)
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2QKVParams>.stride, index: 0)
        let refs = [a.projections, a.queryScale, a.relativeScale, a.keyConv, a.valueConv, a.keyTail,
                    a.valueTail, a.queries, a.relative, a.cache.keys, a.cache.values]
        for (index, ref) in refs.enumerated() {
            encoder.setBuffer(ref.buffer, offset: ref.offset, index: index + 1)
        }
        encoder.dispatchThreadgroups(MTLSize(width: a.rows, height: 16, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}

/// Advances short-convolution tails over a chunk: afterwards each tail holds
/// the last three pre-convolution inputs of `tail ++ x` (FP32 `[3][channels]`,
/// row 2 the newest). Up to two runs per dispatch; in-place safe for any
/// chunk length. Encode it after every kernel that reads the old tails.
public final class Inkling2ConvTail {
    public struct Run: Sendable, Equatable {
        /// First column of the run inside a row of `x`.
        public var xColumn: Int
        /// Floats from the bound tails offset to this run's `[3][channels]` block.
        public var tailOffset: Int
        public var channels: Int

        public init(xColumn: Int, tailOffset: Int, channels: Int) {
            self.xColumn = xColumn
            self.tailOffset = tailOffset
            self.channels = channels
        }
    }

    private let pipeline: MTLComputePipelineState

    public init(library: Inkling2KernelLibrary) throws {
        pipeline = try library.pipeline("inkling2_conv_tail_update")
    }

    public func encode(commandBuffer: MTLCommandBuffer, x: Inkling2BufferRef, xStride: Int, rows: Int,
                       tails: Inkling2BufferRef, runs: [Run]) {
        commandBuffer.inkling2Compute {
            encode(encoder: $0, x: x, xStride: xStride, rows: rows, tails: tails, runs: runs)
        }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encode(encoder: MTLComputeCommandEncoder, x: Inkling2BufferRef, xStride: Int, rows: Int,
                       tails: Inkling2BufferRef, runs: [Run]) {
        precondition(rows > 0 && (1...2).contains(runs.count))
        var params = Inkling2TailParams(tokens: UInt32(rows), xStride: UInt32(xStride),
                                        runCount: UInt32(runs.count), reserved: 0,
                                        run0: Self.packed(runs[0]),
                                        run1: runs.count > 1 ? Self.packed(runs[1]) : Inkling2TailRun())
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2TailParams>.stride, index: 0)
        encoder.setBuffer(x.buffer, offset: x.offset, index: 1)
        encoder.setBuffer(tails.buffer, offset: tails.offset, index: 2)
        let total = runs.reduce(0) { $0 + $1.channels }
        encoder.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, total), height: 1, depth: 1))
    }

    private static func packed(_ run: Run) -> Inkling2TailRun {
        Inkling2TailRun(xColumn: UInt32(run.xColumn), tailOffset: UInt32(run.tailOffset),
                        channels: UInt32(run.channels), reserved: 0)
    }
}

/// The static kernel-4 short convolution after the attention output
/// projection and after the MoE, FP32 in and out, optionally adding a
/// residual: `out = (conv(x) + x) / sqrt(5) (+ residual)`. `out` may alias
/// `residual`, never `x`. Advance the tail with `Inkling2ConvTail` afterwards,
/// or, for a single row (decode), with `advanceTail` in the same dispatch.
public final class Inkling2ShortConv {
    private let pipeline: MTLComputePipelineState

    public init(library: Inkling2KernelLibrary) throws {
        pipeline = try library.pipeline("inkling2_sconv_f32")
    }

    public func encode(commandBuffer: MTLCommandBuffer,
                       x: Inkling2BufferRef, xStride: Int,
                       weight: Inkling2BufferRef, tail: Inkling2BufferRef,
                       residual: Inkling2BufferRef?, residualStride: Int = 0,
                       output: Inkling2BufferRef, outputStride: Int,
                       rows: Int, channels: Int = Inkling2AttentionShape.hidden,
                       advanceTail: Bool = false) {
        commandBuffer.inkling2Compute {
            encode(encoder: $0, x: x, xStride: xStride, weight: weight, tail: tail, residual: residual,
                   residualStride: residualStride, output: output, outputStride: outputStride, rows: rows,
                   channels: channels, advanceTail: advanceTail)
        }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encode(encoder: MTLComputeCommandEncoder,
                       x: Inkling2BufferRef, xStride: Int,
                       weight: Inkling2BufferRef, tail: Inkling2BufferRef,
                       residual: Inkling2BufferRef?, residualStride: Int = 0,
                       output: Inkling2BufferRef, outputStride: Int,
                       rows: Int, channels: Int = Inkling2AttentionShape.hidden,
                       advanceTail: Bool = false) {
        precondition(rows > 0 && channels % 4 == 0 && xStride % 4 == 0 && outputStride % 4 == 0)
        precondition(!advanceTail || rows == 1, "in-place tail advance needs one row")
        precondition(residual == nil || (residualStride > 0 && residualStride % 4 == 0))
        var params = Inkling2ConvParams(tokens: UInt32(rows), channels: UInt32(channels),
                                        xStride: UInt32(xStride), outStride: UInt32(outputStride),
                                        residualStride: UInt32(residual == nil ? 0 : residualStride),
                                        advanceTail: advanceTail ? 1 : 0, reserved1: 0, reserved2: 0)
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2ConvParams>.stride, index: 0)
        encoder.setBuffer(x.buffer, offset: x.offset, index: 1)
        encoder.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        encoder.setBuffer(tail.buffer, offset: tail.offset, index: 3)
        let r = residual ?? x
        encoder.setBuffer(r.buffer, offset: r.offset, index: 4)
        encoder.setBuffer(output.buffer, offset: output.offset, index: 5)
        encoder.dispatchThreads(MTLSize(width: channels / 4, height: rows, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(64, channels / 4), height: 4, depth: 1))
    }
}

/// Weighted RMS norm from the FP32 stream to FP16 projection inputs, with FP32
/// weights (the GGUF `attn_norm`, `attn_*_inp_norm`, `ffn_norm` tensors).
public final class Inkling2InputNorm {
    private let pipeline: MTLComputePipelineState
    private let setPipeline: MTLComputePipelineState

    /// One input of `encodeSet`: its rows, its norm weight and its FP16 output.
    public struct Input {
        public var x: Inkling2BufferRef
        public var weight: Inkling2BufferRef
        public var output: Inkling2BufferRef

        public init(x: Inkling2BufferRef, weight: Inkling2BufferRef, output: Inkling2BufferRef) {
            self.x = x
            self.weight = weight
            self.output = output
        }
    }

    public init(library: Inkling2KernelLibrary) throws {
        pipeline = try library.pipeline("inkling2_rms_f32_to_f16")
        setPipeline = try library.pipeline("inkling2_rms_f32_to_f16_set")
    }

    public func encode(commandBuffer: MTLCommandBuffer, x: Inkling2BufferRef, xStride: Int,
                       weight: Inkling2BufferRef, output: Inkling2BufferRef, outputStride: Int,
                       rows: Int, width: Int = Inkling2AttentionShape.hidden,
                       eps: Float = Inkling2AttentionShape.rmsEps) {
        commandBuffer.inkling2Compute {
            encode(encoder: $0, x: x, xStride: xStride, weight: weight, output: output,
                   outputStride: outputStride, rows: rows, width: width, eps: eps)
        }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encode(encoder: MTLComputeCommandEncoder, x: Inkling2BufferRef, xStride: Int,
                       weight: Inkling2BufferRef, output: Inkling2BufferRef, outputStride: Int,
                       rows: Int, width: Int = Inkling2AttentionShape.hidden,
                       eps: Float = Inkling2AttentionShape.rmsEps) {
        precondition(rows > 0)
        var params = Inkling2NormParams(width: UInt32(width), xStride: UInt32(xStride),
                                        outStride: UInt32(outputStride), eps: eps)
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2NormParams>.stride, index: 0)
        encoder.setBuffer(x.buffer, offset: x.offset, index: 1)
        encoder.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        encoder.setBuffer(output.buffer, offset: output.offset, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// One to three inputs (an attention block's q, k and v inputs), each with its own weight
    /// and output and the same strides, in one dispatch. It writes the same bytes as one
    /// `encode` per input; at one row (decode) the norm's latency is paid once, not per input.
    public func encodeSet(commandBuffer: MTLCommandBuffer, _ inputs: [Input], xStride: Int, outputStride: Int,
                          rows: Int, width: Int = Inkling2AttentionShape.hidden,
                          eps: Float = Inkling2AttentionShape.rmsEps) {
        commandBuffer.inkling2Compute {
            encodeSet(encoder: $0, inputs, xStride: xStride, outputStride: outputStride, rows: rows,
                      width: width, eps: eps)
        }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encodeSet(encoder: MTLComputeCommandEncoder, _ inputs: [Input], xStride: Int, outputStride: Int,
                          rows: Int, width: Int = Inkling2AttentionShape.hidden,
                          eps: Float = Inkling2AttentionShape.rmsEps) {
        precondition(rows > 0 && (1...3).contains(inputs.count))
        var params = Inkling2NormParams(width: UInt32(width), xStride: UInt32(xStride),
                                        outStride: UInt32(outputStride), eps: eps)
        encoder.setComputePipelineState(setPipeline)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2NormParams>.stride, index: 0)
        for k in 0..<3 {
            let input = inputs[min(k, inputs.count - 1)]          // unused slots repeat the last input
            encoder.setBuffer(input.x.buffer, offset: input.x.offset, index: 1 + k)
            encoder.setBuffer(input.weight.buffer, offset: input.weight.offset, index: 4 + k)
            encoder.setBuffer(input.output.buffer, offset: input.output.offset, index: 7 + k)
        }
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: inputs.count, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

/// One decode step's attention for all 24 query heads, split over blocks of
/// keys (pass 1) and merged with the sink and the gate (pass 2). Same inputs
/// as `Inkling2PrefillAttention` with one row; `partials` is scratch of at
/// least `partialsLength(keys:)` bytes.
public final class Inkling2DecodeAttention {
    public static let blockKeys = 128

    public struct Arguments {
        public var queries: Inkling2BufferRef
        public var cache: Inkling2KVView
        public var relative: Inkling2BufferRef
        public var table: Inkling2BufferRef
        public var sink: Inkling2BufferRef
        public var gate: Inkling2BufferRef?
        public var output: Inkling2BufferRef
        public var partials: Inkling2BufferRef
        public var position: Int
        public var mask: Inkling2AttentionMask
        public var relExtent: Int

        public init(queries: Inkling2BufferRef, cache: Inkling2KVView, relative: Inkling2BufferRef,
                    table: Inkling2BufferRef, sink: Inkling2BufferRef, gate: Inkling2BufferRef?,
                    output: Inkling2BufferRef, partials: Inkling2BufferRef, position: Int,
                    mask: Inkling2AttentionMask, relExtent: Int = Inkling2AttentionShape.relExtent) {
            self.queries = queries
            self.cache = cache
            self.relative = relative
            self.table = table
            self.sink = sink
            self.gate = gate
            self.output = output
            self.partials = partials
            self.position = position
            self.mask = mask
            self.relExtent = relExtent
        }
    }

    private let partial: MTLComputePipelineState
    private let merge: MTLComputePipelineState

    public init(library: Inkling2KernelLibrary) throws {
        partial = try library.pipeline("inkling2_attention_decode_partial")
        merge = try library.pipeline("inkling2_attention_decode_merge")
    }

    public static func blocks(keys: Int) -> Int { (keys + blockKeys - 1) / blockKeys }

    /// Bytes of pass-1 scratch for `keys` visible keys.
    public static func partialsLength(keys: Int) -> Int {
        Inkling2AttentionShape.numQHeads * blocks(keys: keys) * (2 + Inkling2AttentionShape.headDim)
            * MemoryLayout<Float>.stride
    }

    /// First visible key and the visible key count of a query at `position`.
    public static func keyRange(position: Int, mask: Inkling2AttentionMask) -> (begin: Int, count: Int) {
        var begin = 0
        if case .window(let window) = mask { begin = max(0, position - window + 1) }
        return (begin, position - begin + 1)
    }

    public static func tau(position: Int, mask: Inkling2AttentionMask) -> Float {
        guard case .global(let floor, let alpha) = mask, floor > 0 else { return 1 }
        let ratio = Float(position + 1) / Float(floor)
        return ratio > 1 ? 1 + alpha * log(ratio) : 1
    }

    public func encode(commandBuffer: MTLCommandBuffer, _ a: Arguments) {
        commandBuffer.inkling2Compute { encode(encoder: $0, a) }
    }

    /// The same, into `encoder`: a serial compute encoder the caller owns and ends. A serial
    /// encoder orders its dispatches, so no barrier is needed before or after this one.
    public func encode(encoder: MTLComputeCommandEncoder, _ a: Arguments) {
        let (begin, count) = Self.keyRange(position: a.position, mask: a.mask)
        if case .window(let window) = a.mask {
            precondition(a.cache.ringCapacity == 0 || a.cache.ringCapacity >= window)
        }
        precondition(a.partials.buffer.length - a.partials.offset >= Self.partialsLength(keys: count))
        precondition(a.queries.offset % 8 == 0 && a.cache.slotStride % 4 == 0)   // half4 loads
        let blocks = Self.blocks(keys: count)
        var params = Inkling2DecodeParams(
            position: UInt32(a.position), keyBegin: UInt32(begin), keyCount: UInt32(count),
            blockKeys: UInt32(Self.blockKeys), blocks: UInt32(blocks),
            kvStride: UInt32(a.cache.slotStride), ringCapacity: UInt32(a.cache.ringCapacity),
            relExtent: UInt32(a.relExtent), tau: Self.tau(position: a.position, mask: a.mask),
            scale: 1 / Float(Inkling2AttentionShape.headDim).squareRoot(),
            numQHeads: UInt32(Inkling2AttentionShape.numQHeads),
            numKVHeads: UInt32(Inkling2AttentionShape.numKVHeads),
            hasGate: a.gate == nil ? 0 : 1, reserved0: 0, reserved1: 0, reserved2: 0)
        encoder.setComputePipelineState(partial)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2DecodeParams>.stride, index: 0)
        for (index, ref) in [a.queries, a.cache.keys, a.cache.values, a.relative, a.table, a.partials].enumerated() {
            encoder.setBuffer(ref.buffer, offset: ref.offset, index: index + 1)
        }
        encoder.dispatchThreadgroups(MTLSize(width: blocks, height: Inkling2AttentionShape.numKVHeads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.setComputePipelineState(merge)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2DecodeParams>.stride, index: 0)
        let gate = a.gate ?? a.sink
        for (index, ref) in [a.partials, a.sink, gate, a.output].enumerated() {
            encoder.setBuffer(ref.buffer, offset: ref.offset, index: index + 1)
        }
        encoder.dispatchThreadgroups(MTLSize(width: Inkling2AttentionShape.numQHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}

// Mirrors of the parameter structs in `inkling2_attention.metal`.

struct Inkling2QKVParams {
    var tokens: UInt32
    var startPosition: UInt32
    var yStride: UInt32
    var qColumn: UInt32
    var relColumn: UInt32
    var kColumn: UInt32
    var vColumn: UInt32
    var kTapColumn: UInt32
    var vTapColumn: UInt32
    var dynamicConv: UInt32
    var qOutStride: UInt32
    var relOutStride: UInt32
    var kvStride: UInt32
    var ringCapacity: UInt32
    var advanceTails: UInt32
    var qkClip: Float
    var relClip: Float
    var clipEps: Float
    var rmsEps: Float
}

struct Inkling2TailRun {
    var xColumn: UInt32 = 0
    var tailOffset: UInt32 = 0
    var channels: UInt32 = 0
    var reserved: UInt32 = 0
}

struct Inkling2TailParams {
    var tokens: UInt32
    var xStride: UInt32
    var runCount: UInt32
    var reserved: UInt32
    var run0: Inkling2TailRun
    var run1: Inkling2TailRun
}

struct Inkling2ConvParams {
    var tokens: UInt32
    var channels: UInt32
    var xStride: UInt32
    var outStride: UInt32
    var residualStride: UInt32
    var advanceTail: UInt32
    var reserved1: UInt32
    var reserved2: UInt32
}

struct Inkling2NormParams {
    var width: UInt32
    var xStride: UInt32
    var outStride: UInt32
    var eps: Float
}

struct Inkling2DecodeParams {
    var position: UInt32
    var keyBegin: UInt32
    var keyCount: UInt32
    var blockKeys: UInt32
    var blocks: UInt32
    var kvStride: UInt32
    var ringCapacity: UInt32
    var relExtent: UInt32
    var tau: Float
    var scale: Float
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var hasGate: UInt32
    var reserved0: UInt32
    var reserved1: UInt32
    var reserved2: UInt32
}
