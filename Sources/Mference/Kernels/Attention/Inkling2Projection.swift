import Foundation
import Metal

/// Storage format of a projection weight in the GGUF file.
public enum Inkling2WeightFormat: UInt32, Sendable {
    /// 34-byte blocks `{half d; int8 q[32]}` (every dense projection).
    case q8_0 = 0
    /// Plain BF16 rows (`attn_sink_proj`, `attn_k_conv_proj`, `attn_v_conv_proj`).
    case bf16 = 1
    /// Plain F32 rows, decoded to half per tile on the half-input paths. (The
    /// loop gates' F32 weights go through `encodeFloat`, which keeps FP32.)
    case f32 = 2

    /// Byte alignment the kernels' vector loads need for a segment's weight offset.
    var offsetAlignment: Int {
        switch self {
        case .q8_0: return 2
        case .bf16: return 8
        case .f32: return 16
        }
    }
}

/// One weight of a stacked projection: `rows` rows starting at byte
/// `weightOffset` of the weight buffer, applied to the input rows at
/// `inputOffset` (halves, token 0) with `inputStride` halves between tokens,
/// written to output columns `outColumn ..< outColumn + rows`.
public struct Inkling2ProjectionSegment: Sendable, Equatable {
    public var weightOffset: Int
    public var inputOffset: Int
    public var inputStride: Int
    public var rows: Int
    public var outColumn: Int
    public var format: Inkling2WeightFormat

    public init(weightOffset: Int, inputOffset: Int = 0, inputStride: Int, rows: Int, outColumn: Int,
                format: Inkling2WeightFormat = .q8_0) {
        self.weightOffset = weightOffset
        self.inputOffset = inputOffset
        self.inputStride = inputStride
        self.rows = rows
        self.outColumn = outColumn
        self.format = format
    }
}

/// `out[t, outColumn + r] = sum_k x[t, k] W[r, k]` for up to eight Q8_0,
/// BF16 or F32 weights in one dispatch: half inputs, FP32 accumulation and output. Every
/// projection of a layer visit that shares an input can be one segment list
/// (query side: Q, relative states, sink, output gate; keys and their
/// dynamic taps; values and theirs; or the output projection alone).
public final class Inkling2Projection {
    public enum Path: String, Sendable {
        /// 64 x 32 simdgroup tiles; any token count.
        case tiled
        /// 64 x 64 simdgroup tiles (32 x 32 per simdgroup); any token count.
        case tiled64
        /// Two weight rows per simdgroup; at most 8 tokens (decode).
        case rows
        /// Metal 4 `matmul2d`, 64 tokens x 32 rows; inputs a multiple of 64.
        /// For GPUs with tensor support (Apple10 and later).
        case tensorOps
    }

    public static let maxSegments = 8
    public static let maxRowsPathTokens = 8
    static let tileRows = 64
    static let tileTokens = 32

    static let tensorTileRows = 32
    static let tensorTileTokens = 64

    private let tiled: MTLComputePipelineState
    private let tiled64: MTLComputePipelineState
    private let float64: MTLComputePipelineState
    /// Small-batch kernels for at most 1, 2, 4 and 8 tokens.
    private let rows: [(tokens: Int, state: MTLComputePipelineState)]
    private let tensor: MTLComputePipelineState?
    /// Whether the many-token default is `.tensorOps` rather than a simdgroup kernel.
    public let prefersTensorOps: Bool

    /// `preferTensorOps` defaults to the device's Apple10 tensor support
    /// (the simdgroup kernel is faster on earlier GPUs).
    public init(library: Inkling2KernelLibrary, preferTensorOps: Bool? = nil) throws {
        tiled = try library.pipeline("inkling2_q8_stacked_mm")
        tiled64 = try library.pipeline("inkling2_q8_stacked_mm64")
        float64 = try library.pipeline("inkling2_f32_mm64")
        rows = try [1, 2, 4, 8].map { ($0, try library.pipeline("inkling2_q8_stacked_mv_t\($0)")) }
        tensor = try? library.pipeline("inkling2_q8_stacked_mm_tensorops")
        prefersTensorOps = tensor != nil && (preferTensorOps ?? library.context.device.supportsApple10TensorOps)
    }

    /// Whether the tensor-ops kernel built on this GPU.
    public var supportsTensorOps: Bool { tensor != nil }

    /// The path `encode` takes for `tokens` rows of `inputs` when none is forced.
    public func defaultPath(tokens: Int, inputs: Int) -> Path {
        if tokens <= Self.maxRowsPathTokens { return .rows }
        if prefersTensorOps && inputs % 64 == 0 { return .tensorOps }
        // 64 x 64 tiles measured 6-8% faster than 64 x 32 at 512, 1,024 and 2,048
        // rows (M1 Max); below 512 the 64 x 32 tile stays the default.
        return tokens >= 512 ? .tiled64 : .tiled
    }

    /// Requires `inputs % 32 == 0`, inputs aligned to 4 halves (offsets and
    /// strides), and disjoint output column ranges.
    @discardableResult
    public func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer,
                       input: MTLBuffer,
                       output: MTLBuffer, outputOffset: Int = 0,
                       tokens: Int, inputs: Int, outStride: Int,
                       segments: [Inkling2ProjectionSegment],
                       path forced: Path? = nil) -> Path {
        precondition(tokens > 0 && inputs > 0 && inputs % 32 == 0)
        precondition(!segments.isEmpty && segments.count <= Self.maxSegments)
        precondition(outputOffset % MemoryLayout<Float>.stride == 0)
        let path = forced ?? defaultPath(tokens: tokens, inputs: inputs)
        precondition(path != .rows || tokens <= Self.maxRowsPathTokens)
        precondition(path != .tensorOps || (tensor != nil && inputs % 64 == 0))

        var groups = 0
        var table: [(Inkling2ProjectionSegment, Int)] = []
        for segment in segments {
            precondition(segment.rows > 0 && segment.weightOffset >= 0)
            precondition(segment.inputOffset % 4 == 0 && segment.inputStride % 4 == 0)
            precondition(segment.weightOffset % segment.format.offsetAlignment == 0)
            table.append((segment, groups))
            switch path {
            case .tiled, .tiled64: groups += (segment.rows + Self.tileRows - 1) / Self.tileRows
            case .rows: groups += (segment.rows + 1) / 2
            case .tensorOps: groups += (segment.rows + Self.tensorTileRows - 1) / Self.tensorTileRows
            }
        }
        var params = Self.packParams(tokens: tokens, inputs: inputs, outStride: outStride, table: table)

        let encoder = commandBuffer.makeComputeCommandEncoder()!
        switch path {
        case .tiled: encoder.setComputePipelineState(tiled)
        case .tiled64: encoder.setComputePipelineState(tiled64)
        case .rows: encoder.setComputePipelineState(rows.first { tokens <= $0.tokens }!.state)
        case .tensorOps: encoder.setComputePipelineState(tensor!)
        }
        params.withUnsafeMutableBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(input, offset: 0, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        switch path {
        case .tiled:
            encoder.dispatchThreadgroups(
                MTLSize(width: (tokens + Self.tileTokens - 1) / Self.tileTokens, height: groups, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        case .tiled64:
            encoder.dispatchThreadgroups(
                MTLSize(width: (tokens + 63) / 64, height: groups, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        case .rows:
            encoder.dispatchThreadgroups(
                MTLSize(width: (groups + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        case .tensorOps:
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: (tokens + Self.tensorTileTokens - 1) / Self.tensorTileTokens, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        encoder.endEncoding()
        return path
    }

    /// The FP32 product `out[t, outColumn + r] = sum_k x[t, k] W[r, k]` with
    /// FP32 fragments: every segment `.f32`, `input` FP32 (segment offsets and
    /// strides in floats, multiples of 4). For the loop gates' products, which
    /// the vendor keeps in FP32.
    public func encodeFloat(commandBuffer: MTLCommandBuffer,
                            weights: MTLBuffer, input: MTLBuffer,
                            output: MTLBuffer, outputOffset: Int = 0,
                            tokens: Int, inputs: Int, outStride: Int,
                            segments: [Inkling2ProjectionSegment]) {
        precondition(tokens > 0 && inputs > 0 && inputs % 32 == 0)
        precondition(!segments.isEmpty && segments.count <= Self.maxSegments)
        precondition(outputOffset % MemoryLayout<Float>.stride == 0)
        var groups = 0
        var table: [(Inkling2ProjectionSegment, Int)] = []
        for segment in segments {
            precondition(segment.format == .f32 && segment.rows > 0 && segment.weightOffset % 16 == 0)
            precondition(segment.inputOffset % 4 == 0 && segment.inputStride % 4 == 0)
            table.append((segment, groups))
            groups += (segment.rows + Self.tileRows - 1) / Self.tileRows
        }
        var params = Self.packParams(tokens: tokens, inputs: inputs, outStride: outStride, table: table)
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(float64)
        params.withUnsafeMutableBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(input, offset: 0, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: (tokens + 63) / 64, height: groups, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// `Inkling2ProjParams`: four uints, then eight 32-byte segments
    /// {u64 weightOffset; u32 inputOffset, inputStride, rows, outColumn, tileStart, format}.
    static func packParams(tokens: Int, inputs: Int, outStride: Int,
                           table: [(Inkling2ProjectionSegment, Int)]) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: 16 + maxSegments * 32)
        func put32(_ value: Int, at offset: Int) {
            let v = UInt32(value)
            for i in 0..<4 { bytes[offset + i] = UInt8(truncatingIfNeeded: v >> (8 * UInt32(i))) }
        }
        func put64(_ value: Int, at offset: Int) {
            let v = UInt64(value)
            for i in 0..<8 { bytes[offset + i] = UInt8(truncatingIfNeeded: v >> (8 * UInt64(i))) }
        }
        put32(tokens, at: 0)
        put32(inputs, at: 4)
        put32(outStride, at: 8)
        put32(table.count, at: 12)
        for (index, (segment, start)) in table.enumerated() {
            let base = 16 + index * 32
            put64(segment.weightOffset, at: base)
            put32(segment.inputOffset, at: base + 8)
            put32(segment.inputStride, at: base + 12)
            put32(segment.rows, at: base + 16)
            put32(segment.outColumn, at: base + 20)
            put32(start, at: base + 24)
            put32(Int(segment.format.rawValue), at: base + 28)
        }
        return bytes
    }
}
