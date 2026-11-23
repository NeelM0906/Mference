import Foundation
import Metal

/// A buffer and a byte offset into it.
public struct Inkling2BufferRef {
    public var buffer: MTLBuffer
    public var offset: Int

    public init(_ buffer: MTLBuffer, offset: Int = 0) {
        self.buffer = buffer
        self.offset = offset
    }
}

/// Where one layer visit's keys and values live: FP16 `[slot][8 KV heads][128]`.
/// A window layer's ring has `ringCapacity` slots and holds position `p` at
/// slot `p % ringCapacity`; a global layer's linear storage holds position
/// `p` at slot `p` (`ringCapacity == 0`).
public struct Inkling2KVView {
    public var keys: Inkling2BufferRef
    public var values: Inkling2BufferRef
    /// Halves between consecutive slots (`8 * 128` for a dense cache).
    public var slotStride: Int
    public var ringCapacity: Int

    public init(keys: Inkling2BufferRef, values: Inkling2BufferRef,
                slotStride: Int = Inkling2AttentionShape.kvWidth, ringCapacity: Int) {
        self.keys = keys
        self.values = values
        self.slotStride = slotStride
        self.ringCapacity = ringCapacity
    }
}

/// Mask kind of one attention visit.
public enum Inkling2AttentionMask: Sendable, Equatable {
    /// Keys `pos - window < key <= pos`, no log scaling.
    case window(Int)
    /// Every key `<= pos`; queries and relative states scaled by
    /// `tau = 1 + alpha * ln(max((pos + 1) / floor, 1))`.
    case global(logFloor: Int, logAlpha: Float)

    public static let slidingDefault = Inkling2AttentionMask.window(Inkling2AttentionShape.slidingWindow)
    public static let globalDefault = Inkling2AttentionMask.global(
        logFloor: Inkling2AttentionShape.logScaleFloor, logAlpha: Inkling2AttentionShape.logScaleAlpha)
}

/// Relative-bias attention with a per-query sink over one chunk of query rows
/// (prefill; also correct for one row). Inputs, all row-major with strides in
/// elements:
/// - `queries`: FP16 `[rows][24 x 128]` (scaled and L2-clipped, no tau)
/// - `relative`: FP32 `[rows][24 x 16]` (scaled and clipped, no tau)
/// - `table`: FP32 `[relExtent][16]`, the GGUF `attn_rel_bias` layout
/// - `sink`: FP32 `[rows][24]`; `gate` (optional): FP32 gate logits `[rows][24 x 128]`
/// - output: FP16 `[rows][24 x 128]`, multiplied by `sigmoid(gate)` when given
///
/// The chunk's own keys and values must already be in the cache.
public final class Inkling2PrefillAttention {
    /// Kernel form: simdgroup fragments with 16- or 32-key tiles (every
    /// GPU), or Metal 4 `matmul2d` with 8 rows x 64 keys (tensor support).
    public enum Variant: String, Sendable, CaseIterable {
        case keys16
        case keys32
        case tensorOps

        var rowsPerThreadgroup: Int { self == .tensorOps ? 8 : 32 }
    }

    public struct Arguments {
        public var queries: Inkling2BufferRef
        public var queryStride: Int
        public var cache: Inkling2KVView
        public var relative: Inkling2BufferRef
        public var relativeStride: Int
        public var table: Inkling2BufferRef
        public var sink: Inkling2BufferRef
        public var sinkStride: Int
        public var gate: Inkling2BufferRef?
        public var gateStride: Int
        public var output: Inkling2BufferRef
        public var outputStride: Int
        public var rows: Int
        public var startPosition: Int
        public var mask: Inkling2AttentionMask
        public var relExtent: Int

        public init(queries: Inkling2BufferRef, queryStride: Int = Inkling2AttentionShape.qWidth,
                    cache: Inkling2KVView,
                    relative: Inkling2BufferRef, relativeStride: Int = Inkling2AttentionShape.relWidth,
                    table: Inkling2BufferRef,
                    sink: Inkling2BufferRef, sinkStride: Int = Inkling2AttentionShape.numQHeads,
                    gate: Inkling2BufferRef?, gateStride: Int = Inkling2AttentionShape.qWidth,
                    output: Inkling2BufferRef, outputStride: Int = Inkling2AttentionShape.qWidth,
                    rows: Int, startPosition: Int, mask: Inkling2AttentionMask,
                    relExtent: Int = Inkling2AttentionShape.relExtent) {
            self.queries = queries
            self.queryStride = queryStride
            self.cache = cache
            self.relative = relative
            self.relativeStride = relativeStride
            self.table = table
            self.sink = sink
            self.sinkStride = sinkStride
            self.gate = gate
            self.gateStride = gateStride
            self.output = output
            self.outputStride = outputStride
            self.rows = rows
            self.startPosition = startPosition
            self.mask = mask
            self.relExtent = relExtent
        }
    }

    private let pipelines: [Variant: MTLComputePipelineState]
    /// The variant `encode` uses when none is forced.
    public let variant: Variant

    /// `variant` defaults to tensor-ops on GPUs with Apple10 tensor support
    /// (when it builds), else 16-key simdgroup tiles.
    public init(library: Inkling2KernelLibrary, variant: Variant? = nil) throws {
        var states: [Variant: MTLComputePipelineState] = [:]
        states[.keys16] = try library.pipeline("inkling2_attention_prefill_bk16")
        states[.keys32] = try library.pipeline("inkling2_attention_prefill_bk32")
        states[.tensorOps] = try? library.pipeline("inkling2_attention_prefill_tensorops")
        pipelines = states
        let tensorDefault = states[.tensorOps] != nil && library.context.device.supportsApple10TensorOps
        let chosen = variant ?? (tensorDefault ? .tensorOps : .keys16)
        precondition(states[chosen] != nil, "variant \(chosen) did not build on this GPU")
        self.variant = chosen
    }

    /// Whether `variant` built on this GPU.
    public func supports(_ variant: Variant) -> Bool { pipelines[variant] != nil }

    public func encode(commandBuffer: MTLCommandBuffer, _ a: Arguments, variant forced: Variant? = nil) {
        precondition(a.rows > 0)
        if case .window(let window) = a.mask {
            precondition(window > 0)
            // The chunk's keys and the window behind its first row must coexist in the ring.
            precondition(a.cache.ringCapacity == 0 || a.cache.ringCapacity >= window - 1 + a.rows)
        }
        let variant = forced ?? self.variant
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipelines[variant]!)
        var params = Self.params(a)
        encoder.setBytes(&params, length: MemoryLayout<Inkling2AttnParams>.stride, index: 0)
        encoder.setBuffer(a.queries.buffer, offset: a.queries.offset, index: 1)
        encoder.setBuffer(a.cache.keys.buffer, offset: a.cache.keys.offset, index: 2)
        encoder.setBuffer(a.cache.values.buffer, offset: a.cache.values.offset, index: 3)
        encoder.setBuffer(a.relative.buffer, offset: a.relative.offset, index: 4)
        encoder.setBuffer(a.table.buffer, offset: a.table.offset, index: 5)
        encoder.setBuffer(a.sink.buffer, offset: a.sink.offset, index: 6)
        let gate = a.gate ?? a.sink
        encoder.setBuffer(gate.buffer, offset: gate.offset, index: 7)
        encoder.setBuffer(a.output.buffer, offset: a.output.offset, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (a.rows + variant.rowsPerThreadgroup - 1) / variant.rowsPerThreadgroup,
                    height: Inkling2AttentionShape.numQHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    static func params(_ a: Arguments) -> Inkling2AttnParams {
        var window = 0, logFloor = 0
        var logAlpha: Float = 0
        switch a.mask {
        case .window(let w): window = w
        case .global(let floor, let alpha): logFloor = floor; logAlpha = alpha
        }
        return Inkling2AttnParams(
            queryCount: UInt32(a.rows), startPosition: UInt32(a.startPosition),
            qStride: UInt32(a.queryStride), relStride: UInt32(a.relativeStride),
            sinkStride: UInt32(a.sinkStride), gateStride: UInt32(a.gate == nil ? 0 : a.gateStride),
            outStride: UInt32(a.outputStride), kvStride: UInt32(a.cache.slotStride),
            ringCapacity: UInt32(a.cache.ringCapacity), window: UInt32(window),
            relExtent: UInt32(a.relExtent), logFloor: UInt32(logFloor), logAlpha: logAlpha,
            scale: 1 / Float(Inkling2AttentionShape.headDim).squareRoot(),
            numQHeads: UInt32(Inkling2AttentionShape.numQHeads),
            numKVHeads: UInt32(Inkling2AttentionShape.numKVHeads))
    }
}

/// Mirrors `Inkling2AttnParams` in `inkling2_attention.metal`.
struct Inkling2AttnParams {
    var queryCount: UInt32
    var startPosition: UInt32
    var qStride: UInt32
    var relStride: UInt32
    var sinkStride: UInt32
    var gateStride: UInt32
    var outStride: UInt32
    var kvStride: UInt32
    var ringCapacity: UInt32
    var window: UInt32
    var relExtent: UInt32
    var logFloor: UInt32
    var logAlpha: Float
    var scale: Float
    var numQHeads: UInt32
    var numKVHeads: UInt32
}
