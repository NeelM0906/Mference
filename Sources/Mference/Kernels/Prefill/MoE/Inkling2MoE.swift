import Foundation
import Metal

/// Encoding of one Inkling 2 Mini layer's routed experts, as stored in the
/// Unsloth UD-Q4_K_XL file and copied unchanged into the install.
enum Inkling2MoEEncoding: UInt32, CaseIterable, Sendable {
    /// ggml `block_nvfp4`: 4 UE4M3 scales + 32 code bytes per 64 weights (layers 0-19).
    case nvfp4 = 0
    /// ggml `block_q8_0`: FP16 scale + 32 int8 per 32 weights (layers 20-23).
    case q8_0 = 1

    var blockWeights: Int { self == .nvfp4 ? 64 : 32 }
    var blockBytes: Int { self == .nvfp4 ? 36 : 34 }
    /// Alignment the kernels need for every matrix start (they read NVFP4
    /// scales and codes as `uint`, Q8_0 scales and codes as `half`/`ushort`).
    var alignment: Int { self == .nvfp4 ? 4 : 2 }

    func rowBytes(k: Int) -> Int {
        precondition(k % blockWeights == 0)
        return k / blockWeights * blockBytes
    }
}

/// What the experts' products see of their inputs (the GPU side of
/// `Inkling2ExpertInputQuantization` in the CPU reference). Off by default;
/// K0's measurement against the base reference sets the shipped value.
enum Inkling2MoEInputQuantization: UInt32, CaseIterable, Sendable {
    /// The FP16 inputs as given.
    case none = 0
    /// ggml's Q8_0 round trip per 32 values (what the reference's CPU product
    /// does for its NVFP4 layers).
    case q8_0Blocks = 1
    /// The vendor's required activation format: NVFP4 four-over-six (NVFP4
    /// layers) or MXFP8 (8-bit layers), route weight before the down input.
    case vendorRequired = 2
}

/// Which grouped-GEMM kernels prefill runs: 8x8 simdgroup matrices (every
/// GPU), or Metal 4 `matmul2d` tensor operations (GPUs that build them).
/// `Inkling2MoECalibration` picks per GPU by timing both.
enum Inkling2MoEGEMMVariant: String, CaseIterable, Sendable, Codable {
    case simdgroup
    case tensorOps
}

/// Shape of one routed-expert layer.
struct Inkling2MoEShape: Hashable, Sendable {
    var d: Int
    var f: Int
    var experts: Int
    var topK: Int

    static let inkling2Mini = Inkling2MoEShape(d: 2304, f: 1152, experts: 128, topK: 8)

    /// The grouped kernels tile 64 gate/up and 128 down outputs and step K by
    /// 32; NVFP4 rows are whole 64-weight blocks; route planning runs one
    /// thread per expert. The router takes up to 128 experts.
    var isSupported: Bool {
        d > 0 && f > 0 && d % 128 == 0 && f % 64 == 0 && topK > 0 && topK <= 32
            && experts > 0 && experts <= 1024
    }

    func gateUpBytes(_ encoding: Inkling2MoEEncoding) -> Int { 2 * f * encoding.rowBytes(k: d) }
    func downBytes(_ encoding: Inkling2MoEEncoding) -> Int { d * encoding.rowBytes(k: f) }
    /// Gate-up then down, unpadded: 4,478,976 bytes (NVFP4) or 8,460,288
    /// (Q8_0) at Inkling 2 Mini's shape.
    func expertBytes(_ encoding: Inkling2MoEEncoding) -> Int { gateUpBytes(encoding) + downBytes(encoding) }
    /// Multiply-adds of one routed (token, expert) pair.
    var macsPerPair: Int { 2 * f * d + d * f }
}

/// One layer's routed experts as the kernels read them: blob `e` starts at
/// `offset + e * expertStride` in `buffer`, its gate-up matrix at
/// `gateUpOffset` inside the blob and its down matrix at `downOffset`.
/// `gateUpScales` and `downScales` hold `experts` floats each: the NVFP4
/// layer's `ffn_gate_up_exps.scale` and `ffn_down_exps.scale` tensors as
/// stored, or ones for a Q8_0 layer.
struct Inkling2ExpertPool {
    let buffer: MTLBuffer
    let offset: Int
    let encoding: Inkling2MoEEncoding
    /// The layer shape the pool was validated against; every encode checks it.
    let shape: Inkling2MoEShape
    var experts: Int { shape.experts }
    let expertStride: Int
    let gateUpOffset: Int
    let downOffset: Int
    let gateUpScales: MTLBuffer
    let gateUpScalesOffset: Int
    let downScales: MTLBuffer
    let downScalesOffset: Int
    /// A decode cache's slots (`Inkling2ExpertStore`): `shape.experts` is the
    /// slot count and ids name slots, so it may be smaller than the model's.
    let isSlotPool: Bool

    enum LayoutError: Error, Equatable {
        case misaligned(String)
        case outOfBounds(String)
        case overlapping
    }

    init(buffer: MTLBuffer, offset: Int = 0, encoding: Inkling2MoEEncoding, shape: Inkling2MoEShape,
         expertStride: Int, gateUpOffset: Int = 0, downOffset: Int? = nil,
         gateUpScales: MTLBuffer, gateUpScalesOffset: Int = 0,
         downScales: MTLBuffer, downScalesOffset: Int = 0, isSlotPool: Bool = false) throws {
        precondition(shape.isSupported, "unsupported Inkling 2 expert shape \(shape)")
        let down = downOffset ?? gateUpOffset + shape.gateUpBytes(encoding)
        for (name, value) in [("offset", offset), ("expert stride", expertStride),
                              ("gate-up offset", gateUpOffset), ("down offset", down)]
        where value % encoding.alignment != 0 {
            throw LayoutError.misaligned(name)
        }
        guard gateUpScalesOffset % 4 == 0, downScalesOffset % 4 == 0 else {
            throw LayoutError.misaligned("scales offset")
        }
        let gateUpEnd = gateUpOffset + shape.gateUpBytes(encoding)
        let downEnd = down + shape.downBytes(encoding)
        guard gateUpEnd <= down || downEnd <= gateUpOffset else { throw LayoutError.overlapping }
        guard max(gateUpEnd, downEnd) <= expertStride || shape.experts == 1 else {
            throw LayoutError.outOfBounds("expert blob exceeds its stride")
        }
        let last = offset + (shape.experts - 1) * expertStride + max(gateUpEnd, downEnd)
        guard last <= buffer.length, expertStride <= Int(UInt32.max) else {
            throw LayoutError.outOfBounds("pool")
        }
        guard gateUpScalesOffset + shape.experts * 4 <= gateUpScales.length,
              downScalesOffset + shape.experts * 4 <= downScales.length else {
            throw LayoutError.outOfBounds("scales")
        }
        self.buffer = buffer
        self.offset = offset
        self.encoding = encoding
        self.shape = shape
        self.expertStride = expertStride
        self.gateUpOffset = gateUpOffset
        self.downOffset = down
        self.gateUpScales = gateUpScales
        self.gateUpScalesOffset = gateUpScalesOffset
        self.downScales = downScales
        self.downScalesOffset = downScalesOffset
        self.isSlotPool = isSlotPool
    }
}

extension Inkling2ExpertPool {
    /// The same layout over another buffer: a staging slab that holds this
    /// layer's experts at `offset`.
    func rebound(to other: MTLBuffer, offset newOffset: Int = 0) throws -> Inkling2ExpertPool {
        try Inkling2ExpertPool(buffer: other, offset: newOffset, encoding: encoding, shape: shape,
                               expertStride: expertStride, gateUpOffset: gateUpOffset, downOffset: downOffset,
                               gateUpScales: gateUpScales, gateUpScalesOffset: gateUpScalesOffset,
                               downScales: downScales, downScalesOffset: downScalesOffset,
                               isSlotPool: isSlotPool)
    }

    /// Whether a decode step with the model's `shape` may bind this pool:
    /// the same shape, or a decode cache's slots of the same D, F and top-k.
    func servesDecode(of shape: Inkling2MoEShape) -> Bool {
        self.shape == shape
            || (isSlotPool && self.shape.d == shape.d && self.shape.f == shape.f && self.shape.topK == shape.topK)
    }
}

/// Layout-identical to `I2Params` in `inkling2_moe.metal`.
struct Inkling2MoEParams {
    var d: UInt32
    var f: UInt32
    var topK: UInt32
    var hiddenStride: UInt32
    var expertStride: UInt32
    var gateUpOffset: UInt32
    var downOffset: UInt32
    var experts: UInt32
    var workTailBase: UInt32
    var flags: UInt32
}

/// Layout-identical to `I2QuantParams`.
private struct Inkling2InputQuantParams {
    var width: UInt32
    var srcStride: UInt32
    var dstStride: UInt32
    var mode: UInt32
    var weightStride: UInt32
}

/// Layout-identical to `I2PlanParams`.
private struct Inkling2RoutePlanParams {
    var experts: UInt32
    var rowsPerBlock: UInt32
    var gateUpColumns: UInt32
    var downColumns: UInt32
    var workCapacity: UInt32
}

/// `I2Pair`: one routed (token, rank) row in its expert's group.
struct Inkling2RoutedPair: Equatable {
    var token: UInt32
    var rank: UInt32
    var weight: Float
    var reserved: UInt32
}

/// `I2Work`: one block of up to 64 rows of one expert.
struct Inkling2WorkBlock: Equatable {
    var expert: UInt32
    var pairStart: UInt32
    var rowCount: UInt32
    var reserved: UInt32
}

/// Device scratch for one prefill chunk of up to `maxTokens` rows. The work
/// list holds `fullBlockCapacity` 64-row blocks, then up to `experts` tail
/// blocks of at most 32 rows; `workCount` holds both counts and `dispatch`
/// four indirect argument records (gate-up full, gate-up tail, down full,
/// down tail). Every
/// buffer is written and read on the GPU only; the shared storage mode is
/// for tests and inspection.
final class Inkling2MoEPrefillScratch {
    let shape: Inkling2MoEShape
    let maxTokens: Int
    let ids: MTLBuffer
    let routeWeights: MTLBuffer
    let counts: MTLBuffer
    let cursors: MTLBuffer
    let pairs: MTLBuffer
    let work: MTLBuffer
    let workCount: MTLBuffer
    let dispatch: MTLBuffer
    let activation: MTLBuffer
    let partials: MTLBuffer
    /// The chunk's expert inputs after an input-quantization pass.
    let quantizedHidden: MTLBuffer

    /// Upper bound of full (up to 64-row) blocks: every expert's last block
    /// may be partial. Tail blocks start at this index.
    var fullBlockCapacity: Int {
        (maxTokens * shape.topK + Inkling2MoEKernels.rowsPerBlock - 1) / Inkling2MoEKernels.rowsPerBlock + shape.experts
    }

    /// Bytes this scratch allocates; the activation and FP32 partials dominate
    /// (2 * T * topK * (F + 2D) bytes).
    static func bytes(shape: Inkling2MoEShape, maxTokens: Int) -> Int {
        let pairs = maxTokens * shape.topK
        let work = (pairs + Inkling2MoEKernels.rowsPerBlock - 1) / Inkling2MoEKernels.rowsPerBlock + 2 * shape.experts
        return pairs * (4 + 4 + 16 + 2 * shape.f + 4 * shape.d) + maxTokens * shape.d * 2
            + shape.experts * 8 + work * 16 + 8 + 48
    }

    init(device: MTLDevice, shape: Inkling2MoEShape, maxTokens: Int) throws {
        precondition(maxTokens > 0 && shape.isSupported)
        self.shape = shape
        self.maxTokens = maxTokens
        let pairs = maxTokens * shape.topK
        func buffer(_ bytes: Int, _ name: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(16, bytes), options: .storageModeShared) else {
                throw PrefillGroupedRoutedMoEError.allocationFailed("Inkling 2 \(name)")
            }
            b.label = "inkling2.moe.\(name)"
            return b
        }
        ids = try buffer(pairs * 4, "ids")
        routeWeights = try buffer(pairs * 4, "route-weights")
        counts = try buffer(shape.experts * 4, "counts")
        cursors = try buffer(shape.experts * 4, "cursors")
        self.pairs = try buffer(pairs * MemoryLayout<Inkling2RoutedPair>.stride, "pairs")
        let capacity = (pairs + Inkling2MoEKernels.rowsPerBlock - 1) / Inkling2MoEKernels.rowsPerBlock + shape.experts
        work = try buffer((capacity + shape.experts) * MemoryLayout<Inkling2WorkBlock>.stride, "work")
        workCount = try buffer(2 * 4, "work-count")
        dispatch = try buffer(12 * 4, "dispatch")
        activation = try buffer(pairs * shape.f * 2, "activation")
        partials = try buffer(pairs * shape.d * 4, "partials")
        quantizedHidden = try buffer(maxTokens * shape.d * 2, "quantized-hidden")
    }
}

/// Inkling 2 Mini routed-expert kernels (`Metal/Inkling2/inkling2_moe.metal`):
/// GPU router top-8, route planning with no readback, grouped gate-up (with
/// the per-expert scale and SITU fused) and down GEMMs for prefill, the
/// rank-ordered reduce, and the one-token decode step. Every `encode` call
/// adds dispatches to the caller's compute encoder, which must use the
/// default serial dispatch type (each dispatch sees the previous one's writes).
final class Inkling2MoEKernels {
    static let module = "inkling2_moe"
    static let rowsPerBlock = 64
    static let gateUpColumns = 64
    static let downColumns = 128
    static let threads = 128
    static let routerTokens = 32
    static let routerExperts = 128
    /// Decode gate-up: 16 (gate, up) row pairs per threadgroup; down: 8 rows.
    static let decodeRowsPerThreadgroup = 16
    static let decodeDownRowsPerThreadgroup = 8

    private struct Grouped {
        let gateUp: MTLComputePipelineState
        let gateUpTail: MTLComputePipelineState
        let down: MTLComputePipelineState
        let downTail: MTLComputePipelineState
    }

    private struct Encoded {
        let simdgroup: Grouped
        let tensorOps: Grouped?
        let decodeGateUp: MTLComputePipelineState
        let decodeDown: MTLComputePipelineState
        let inputQuant: MTLComputePipelineState
    }

    private let router: MTLComputePipelineState
    private let clear: MTLComputePipelineState
    private let count: MTLComputePipelineState
    private let plan: MTLComputePipelineState
    private let scatter: MTLComputePipelineState
    private let reduce: MTLComputePipelineState
    private let encoded: [Inkling2MoEEncoding: Encoded]

    /// How the one-token NVFP4 decode turns codes and scale bytes into values
    /// (`FC_I2_PAIR_DECODE`): as FP16 bit patterns two codes at a time, exact
    /// only where the GPU converts FP16 subnormals exactly (E2M1 code 1 and
    /// UE4M3 bytes with exponent 0 are subnormal there), or one code at a time
    /// through a table. Where both are exact they write the same bits.
    enum NVFP4Decode: Sendable, Equatable {
        case pairs
        case table
    }

    /// The NVFP4 decode these pipelines run: `.pairs` when the probe found
    /// this GPU exact (or a caller asked for it), otherwise `.table`.
    let nvfp4Decode: NVFP4Decode

    /// `pipelined`: the simdgroup kernels load the next K step before the
    /// current multiply (the default; false keeps the plain loop, for A/B
    /// measurements). `nvfp4Decode`: nil (the default) probes the GPU once
    /// (`probeNVFP4Decode`); tests pass a path to run it whatever the probe says.
    init(context: MetalContext, pipelined: Bool = true, nvfp4Decode requested: NVFP4Decode? = nil) throws {
        let device = context.device
        let library = try Inkling2ShaderLibrary.compile(device: device, module: Self.module)
        func make(_ name: String, encoding: Inkling2MoEEncoding? = nil,
                  pairDecode: Bool? = nil) throws -> MTLComputePipelineState {
            let values = MTLFunctionConstantValues()
            var raw = encoding?.rawValue ?? 0
            var prefetch = pipelined
            values.setConstantValue(&raw, type: .uint, index: 200)
            values.setConstantValue(&prefetch, type: .bool, index: 201)
            if var pairs = pairDecode { values.setConstantValue(&pairs, type: .bool, index: 202) }
            let function = try library.makeFunction(name: name, constantValues: values)
            let pipeline = try device.makeComputePipelineState(function: function)
            guard pipeline.maxTotalThreadsPerThreadgroup >= Self.threads else {
                throw MetalError.missingFunction("\(name) at \(Self.threads) threads")
            }
            return pipeline
        }
        router = try make("inkling2_moe_router")
        clear = try make("inkling2_route_clear")
        count = try make("inkling2_route_count")
        plan = try make("inkling2_route_plan")
        scatter = try make("inkling2_route_scatter")
        reduce = try make("inkling2_moe_reduce")
        let decode = try requested ?? Self.probeNVFP4Decode(context: context,
                                                            probe: make("inkling2_nvfp4_decode_probe"))
        nvfp4Decode = decode
        // The tensor-ops kernels exist only where the GPU builds them
        // (`__HAVE_TENSOR__`); a pipeline that fails leaves the simdgroup path.
        let hasTensorOps = library.functionNames.contains("inkling2_moe_gate_up_tensor")
        var encoded: [Inkling2MoEEncoding: Encoded] = [:]
        for encoding in Inkling2MoEEncoding.allCases {
            let tensorOps: Grouped? = hasTensorOps ? try? Grouped(
                gateUp: make("inkling2_moe_gate_up_tensor", encoding: encoding),
                gateUpTail: make("inkling2_moe_gate_up_tail_tensor", encoding: encoding),
                down: make("inkling2_moe_down_tensor", encoding: encoding),
                downTail: make("inkling2_moe_down_tail_tensor", encoding: encoding)) : nil
            encoded[encoding] = Encoded(
                simdgroup: Grouped(
                    gateUp: try make("inkling2_moe_gate_up", encoding: encoding),
                    gateUpTail: try make("inkling2_moe_gate_up_tail", encoding: encoding),
                    down: try make("inkling2_moe_down", encoding: encoding),
                    downTail: try make("inkling2_moe_down_tail", encoding: encoding)),
                tensorOps: tensorOps,
                decodeGateUp: try make("inkling2_moe_decode_gate_up", encoding: encoding,
                                       pairDecode: decode == .pairs),
                decodeDown: try make("inkling2_moe_decode_down", encoding: encoding,
                                     pairDecode: decode == .pairs),
                inputQuant: try make("inkling2_moe_input_quant", encoding: encoding))
        }
        self.encoded = encoded
    }

    /// Runs `inkling2_nvfp4_decode_probe` once and compares its outputs with
    /// their exact values: every E2M1 code through the pair decode (sixteen of
    /// one code against ones: 16 x E2M1(c) x 2^-14) and every UE4M3 byte
    /// through the scale decode (UE4M3(b) x 2^14, 0x7F zero, bit 7 ignored).
    /// Any difference, or a failed dispatch, selects the table decode, with
    /// one line on stderr.
    static func probeNVFP4Decode(context: MetalContext, probe: MTLComputePipelineState) throws -> NVFP4Decode {
        let device = context.device
        let count = 16 + 256
        guard let ones = device.makeBuffer(length: 16 * 2, options: .storageModeShared),
              let out = device.makeBuffer(length: count * 4, options: .storageModeShared),
              let cb = context.queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("NVFP4 decode probe")
        }
        let h = ones.contents().bindMemory(to: Float16.self, capacity: 16)
        for i in 0..<16 { h[i] = 1 }
        memset(out.contents(), 0xFF, count * 4)
        encoder.setComputePipelineState(probe)
        encoder.setBuffer(ones, offset: 0, index: 0)
        encoder.setBuffer(out, offset: 0, index: 1)
        encoder.dispatchThreadgroups(MTLSize(width: (count + threads - 1) / threads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let got = out.contents().bindMemory(to: Float.self, capacity: count)
        let exact = cb.status == .completed && nvfp4ProbeIsExact(Array(UnsafeBufferPointer(start: got, count: count)))
        if !exact {
            FileHandle.standardError.write(Data(
                "inkling2: this GPU does not convert FP16 subnormals exactly; the NVFP4 expert decode uses its table path\n".utf8))
        }
        return exact ? .pairs : .table
    }

    /// The probe's exact outputs: 16 x E2M1(c) x 2^-14 for codes c = 0..15,
    /// then UE4M3(b) x 2^14 for bytes b = 0..255 (0x7F zero, bit 7 ignored).
    static let nvfp4ProbeExpected: [Float] = {
        let e2m1: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        func ue4m3(_ byte: Int) -> Float {
            if byte == 0 || byte == 0x7F { return 0 }
            let e = (byte >> 3) & 15, m = Float(byte & 7)
            return e == 0 ? m * 0.001953125 : Float(sign: .plus, exponent: e - 7, significand: 1 + m / 8)
        }
        var values: [Float] = []
        for c in 0..<16 {
            let sixteen: Float = c >= 8 ? -16 : 16
            values.append(sixteen * e2m1[c & 7] / 16384)
        }
        for b in 0..<256 { values.append(ue4m3(b) * 16384) }
        return values
    }()

    /// True when the probe's outputs equal `nvfp4ProbeExpected` exactly.
    static func nvfp4ProbeIsExact(_ got: [Float]) -> Bool {
        got.count == nvfp4ProbeExpected.count && zip(got, nvfp4ProbeExpected).allSatisfy { $0 == $1 }
    }

    /// The grouped-GEMM kernels `encodeGroupedExperts` uses, per weight
    /// encoding; `Inkling2MoECalibration` sets it. An unavailable choice falls
    /// back to `.simdgroup`. The two variants sum in different orders, so
    /// outputs differ between them at the FP32 rounding level.
    private var variants: [Inkling2MoEEncoding: Inkling2MoEGEMMVariant] = [:]
    private let variantLock = NSLock()

    func variant(for encoding: Inkling2MoEEncoding) -> Inkling2MoEGEMMVariant {
        variantLock.lock(); defer { variantLock.unlock() }
        return variants[encoding] ?? .simdgroup
    }

    func setVariant(_ variant: Inkling2MoEGEMMVariant, for encoding: Inkling2MoEEncoding) {
        variantLock.lock(); defer { variantLock.unlock() }
        variants[encoding] = variant
    }

    /// Every encoding at once (tests and benchmarks).
    var variant: Inkling2MoEGEMMVariant {
        get { variant(for: .nvfp4) }
        set { for encoding in Inkling2MoEEncoding.allCases { setVariant(newValue, for: encoding) } }
    }

    /// Variants this GPU built for every encoding.
    var availableVariants: [Inkling2MoEGEMMVariant] {
        encoded.values.allSatisfy { $0.tensorOps != nil } ? [.simdgroup, .tensorOps] : [.simdgroup]
    }

    private func grouped(_ kernels: Encoded, encoding: Inkling2MoEEncoding) -> Grouped {
        variant(for: encoding) == .tensorOps ? (kernels.tensorOps ?? kernels.simdgroup) : kernels.simdgroup
    }

    private static func params(_ shape: Inkling2MoEShape, pool: Inkling2ExpertPool, hiddenStride: Int,
                               workTailBase: Int = 0, routeWeightInDownInput: Bool = false) -> Inkling2MoEParams {
        Inkling2MoEParams(d: UInt32(shape.d), f: UInt32(shape.f), topK: UInt32(shape.topK),
                          hiddenStride: UInt32(hiddenStride), expertStride: UInt32(pool.expertStride),
                          gateUpOffset: UInt32(pool.gateUpOffset), downOffset: UInt32(pool.downOffset),
                          experts: UInt32(pool.experts), workTailBase: UInt32(workTailBase),
                          flags: routeWeightInDownInput ? 1 : 0)
    }

    /// Rounds `rows` rows of FP16 expert inputs (`width` values each) as a
    /// quantized product would see them; `rowWeights` (buffer, offset,
    /// stride in floats) multiplies each row first (the vendor's down input).
    func encodeInputQuantization(_ encoder: MTLComputeCommandEncoder, encoding: Inkling2MoEEncoding,
                                 mode: Inkling2MoEInputQuantization,
                                 source: MTLBuffer, sourceOffset: Int = 0, sourceStride: Int,
                                 destination: MTLBuffer, destinationOffset: Int = 0, destinationStride: Int,
                                 rows: Int, width: Int, rowWeights: (MTLBuffer, Int, Int)? = nil) {
        precondition(mode != .none && width % 32 == 0 && width <= 4096 && rows > 0)
        guard let kernels = encoded[encoding] else { preconditionFailure("encoding") }
        var q = Inkling2InputQuantParams(width: UInt32(width), srcStride: UInt32(sourceStride),
                                         dstStride: UInt32(destinationStride), mode: mode.rawValue,
                                         weightStride: UInt32(rowWeights?.2 ?? 0))
        encoder.setComputePipelineState(kernels.inputQuant)
        encoder.setBuffer(source, offset: sourceOffset, index: 0)
        encoder.setBuffer(destination, offset: destinationOffset, index: 1)
        encoder.setBuffer(rowWeights?.0 ?? source, offset: rowWeights?.1 ?? 0, index: 2)
        encoder.setBytes(&q, length: MemoryLayout<Inkling2InputQuantParams>.stride, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Self.threads, height: 1, depth: 1))
    }

    /// Router logits (FP32, `weight` is `[experts][d]` F32, up to 128 experts), sigmoid, top-k of
    /// sigmoid + `bias`, normalized weights times `routeScale`, for `tokens`
    /// rows of `x` (F32, `xStride` floats apart). Writes `ids` and
    /// `routeWeights` (`[tokens][topK]`, uint32 and float).
    func encodeRouter(_ encoder: MTLComputeCommandEncoder, shape: Inkling2MoEShape,
                      x: MTLBuffer, xOffset: Int = 0, xStride: Int,
                      weight: MTLBuffer, weightOffset: Int = 0, bias: MTLBuffer, biasOffset: Int = 0,
                      ids: MTLBuffer, idsOffset: Int = 0, routeWeights: MTLBuffer, routeWeightsOffset: Int = 0,
                      tokens: Int, routeScale: Float = 8) {
        precondition(shape.experts <= Self.routerExperts && shape.topK <= shape.experts,
                     "the router kernel handles up to 128 experts")
        precondition(shape.d % 32 == 0 && xStride % 4 == 0 && xStride >= shape.d && tokens > 0)
        precondition(shape.topK <= 32, "one lane per rank")
        precondition(xOffset % 16 == 0 && weightOffset % 16 == 0, "the router reads float4")
        precondition(biasOffset % 4 == 0 && idsOffset % 4 == 0 && routeWeightsOffset % 4 == 0)
        var t = UInt32(tokens), d = UInt32(shape.d), stride = UInt32(xStride), k = UInt32(shape.topK)
        var scale = routeScale, experts = UInt32(shape.experts)
        encoder.setComputePipelineState(router)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(bias, offset: biasOffset, index: 2)
        encoder.setBuffer(ids, offset: idsOffset, index: 3)
        encoder.setBuffer(routeWeights, offset: routeWeightsOffset, index: 4)
        encoder.setBytes(&t, length: 4, index: 5)
        encoder.setBytes(&d, length: 4, index: 6)
        encoder.setBytes(&stride, length: 4, index: 7)
        encoder.setBytes(&k, length: 4, index: 8)
        encoder.setBytes(&scale, length: 4, index: 9)
        encoder.setBytes(&experts, length: 4, index: 10)
        encoder.dispatchThreadgroups(
            MTLSize(width: (tokens + Self.routerTokens - 1) / Self.routerTokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threads, height: 1, depth: 1))
    }

    /// Groups the chunk's (token, rank) pairs by expert from `scratch.ids`
    /// and `scratch.routeWeights`, and writes the 64-row work list and the
    /// indirect dispatch arguments the grouped GEMMs read. Nothing returns
    /// to the CPU.
    func encodeRoutePlan(_ encoder: MTLComputeCommandEncoder, scratch: Inkling2MoEPrefillScratch, tokens: Int) {
        let shape = scratch.shape
        precondition(tokens > 0 && tokens <= scratch.maxTokens)
        var pairs = UInt32(tokens * shape.topK), experts = UInt32(shape.experts), k = UInt32(shape.topK)
        let expertThreads = MTLSize(width: shape.experts, height: 1, depth: 1)
        let pairGroups = MTLSize(width: (tokens * shape.topK + 255) / 256, height: 1, depth: 1)
        let wide = MTLSize(width: 256, height: 1, depth: 1)

        encoder.setComputePipelineState(clear)
        encoder.setBuffer(scratch.counts, offset: 0, index: 0)
        encoder.setBytes(&experts, length: 4, index: 1)
        encoder.dispatchThreads(expertThreads, threadsPerThreadgroup: MTLSize(width: min(shape.experts, 256), height: 1, depth: 1))

        encoder.setComputePipelineState(count)
        encoder.setBuffer(scratch.ids, offset: 0, index: 0)
        encoder.setBuffer(scratch.counts, offset: 0, index: 1)
        encoder.setBytes(&pairs, length: 4, index: 2)
        encoder.setBytes(&experts, length: 4, index: 3)
        encoder.dispatchThreadgroups(pairGroups, threadsPerThreadgroup: wide)

        var planParams = Inkling2RoutePlanParams(
            experts: experts, rowsPerBlock: UInt32(Self.rowsPerBlock),
            gateUpColumns: UInt32(shape.f / Self.gateUpColumns), downColumns: UInt32(shape.d / Self.downColumns),
            workCapacity: UInt32(scratch.fullBlockCapacity))
        encoder.setComputePipelineState(plan)
        encoder.setBuffer(scratch.counts, offset: 0, index: 0)
        encoder.setBuffer(scratch.cursors, offset: 0, index: 1)
        encoder.setBuffer(scratch.work, offset: 0, index: 2)
        encoder.setBuffer(scratch.workCount, offset: 0, index: 3)
        encoder.setBuffer(scratch.dispatch, offset: 0, index: 4)
        encoder.setBytes(&planParams, length: MemoryLayout<Inkling2RoutePlanParams>.stride, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: expertThreads)

        encoder.setComputePipelineState(scatter)
        encoder.setBuffer(scratch.ids, offset: 0, index: 0)
        encoder.setBuffer(scratch.routeWeights, offset: 0, index: 1)
        encoder.setBuffer(scratch.cursors, offset: 0, index: 2)
        encoder.setBuffer(scratch.pairs, offset: 0, index: 3)
        encoder.setBytes(&pairs, length: 4, index: 4)
        encoder.setBytes(&k, length: 4, index: 5)
        encoder.setBytes(&experts, length: 4, index: 6)
        encoder.dispatchThreadgroups(pairGroups, threadsPerThreadgroup: wide)
    }

    /// Grouped gate-up (SITU fused) and down GEMMs over the planned work
    /// list: `hidden` holds the chunk's `tokens` FP16 expert inputs,
    /// `hiddenStride` halves apart; the result is `scratch.partials`, one
    /// FP32 row per (token, rank), already scaled by the expert scale and
    /// route weight. `inputQuantization` adds a pass over the inputs before
    /// each product.
    func encodeGroupedExperts(_ encoder: MTLComputeCommandEncoder, hidden: MTLBuffer, hiddenOffset: Int = 0,
                              hiddenStride: Int, pool: Inkling2ExpertPool, scratch: Inkling2MoEPrefillScratch,
                              tokens: Int, inputQuantization: Inkling2MoEInputQuantization = .none) {
        let shape = scratch.shape
        precondition(pool.shape == shape, "pool and scratch shapes differ")
        precondition(hiddenStride % 4 == 0 && hiddenStride >= shape.d && hiddenOffset % 8 == 0)
        precondition(tokens > 0 && tokens <= scratch.maxTokens)
        guard let kernels = encoded[pool.encoding] else { preconditionFailure("encoding") }
        let vendor = inputQuantization == .vendorRequired
        var input = (buffer: hidden, offset: hiddenOffset, stride: hiddenStride)
        if inputQuantization != .none {
            encodeInputQuantization(encoder, encoding: pool.encoding, mode: inputQuantization,
                                    source: hidden, sourceOffset: hiddenOffset, sourceStride: hiddenStride,
                                    destination: scratch.quantizedHidden, destinationStride: shape.d,
                                    rows: tokens, width: shape.d)
            input = (scratch.quantizedHidden, 0, shape.d)
        }
        var p = Self.params(shape, pool: pool, hiddenStride: input.stride, workTailBase: scratch.fullBlockCapacity,
                            routeWeightInDownInput: vendor)
        let threads = MTLSize(width: Self.threads, height: 1, depth: 1)
        let gemm = grouped(kernels, encoding: pool.encoding)

        encoder.setComputePipelineState(gemm.gateUp)
        encoder.setBuffer(input.buffer, offset: input.offset, index: 0)
        encoder.setBuffer(scratch.pairs, offset: 0, index: 1)
        encoder.setBuffer(scratch.work, offset: 0, index: 2)
        encoder.setBuffer(scratch.workCount, offset: 0, index: 3)
        encoder.setBuffer(scratch.activation, offset: 0, index: 4)
        encoder.setBuffer(pool.buffer, offset: pool.offset, index: 5)
        encoder.setBuffer(pool.gateUpScales, offset: pool.gateUpScalesOffset, index: 6)
        encoder.setBytes(&p, length: MemoryLayout<Inkling2MoEParams>.stride, index: 7)
        encoder.dispatchThreadgroups(indirectBuffer: scratch.dispatch, indirectBufferOffset: 0,
                                     threadsPerThreadgroup: threads)
        encoder.setComputePipelineState(gemm.gateUpTail)
        encoder.dispatchThreadgroups(indirectBuffer: scratch.dispatch, indirectBufferOffset: 12,
                                     threadsPerThreadgroup: threads)

        if inputQuantization != .none {
            // Activation rows are the sorted pairs; the vendor's rule takes
            // each pair's route weight first (`I2Pair.weight` at byte 8).
            encodeInputQuantization(encoder, encoding: pool.encoding, mode: inputQuantization,
                                    source: scratch.activation, sourceStride: shape.f,
                                    destination: scratch.activation, destinationStride: shape.f,
                                    rows: tokens * shape.topK, width: shape.f,
                                    rowWeights: vendor ? (scratch.pairs, 8, MemoryLayout<Inkling2RoutedPair>.stride / 4) : nil)
            encoder.setBuffer(scratch.pairs, offset: 0, index: 1)
            encoder.setBuffer(scratch.work, offset: 0, index: 2)
            encoder.setBuffer(scratch.workCount, offset: 0, index: 3)
            encoder.setBuffer(pool.buffer, offset: pool.offset, index: 5)
            encoder.setBytes(&p, length: MemoryLayout<Inkling2MoEParams>.stride, index: 7)
        }

        encoder.setComputePipelineState(gemm.down)
        encoder.setBuffer(scratch.activation, offset: 0, index: 0)
        encoder.setBuffer(scratch.partials, offset: 0, index: 4)
        encoder.setBuffer(pool.downScales, offset: pool.downScalesOffset, index: 6)
        encoder.dispatchThreadgroups(indirectBuffer: scratch.dispatch, indirectBufferOffset: 24,
                                     threadsPerThreadgroup: threads)
        encoder.setComputePipelineState(gemm.downTail)
        encoder.dispatchThreadgroups(indirectBuffer: scratch.dispatch, indirectBufferOffset: 36,
                                     threadsPerThreadgroup: threads)
    }

    /// `out[t] = sum_r partials[t][r]` in rank order, FP32, `outStride` floats apart.
    func encodeReduce(_ encoder: MTLComputeCommandEncoder, scratch: Inkling2MoEPrefillScratch, tokens: Int,
                      out: MTLBuffer, outOffset: Int = 0, outStride: Int) {
        let shape = scratch.shape
        precondition(outStride % 4 == 0 && outStride >= shape.d && outOffset % 16 == 0)
        var d = UInt32(shape.d), k = UInt32(shape.topK), stride = UInt32(outStride)
        encoder.setComputePipelineState(reduce)
        encoder.setBuffer(scratch.partials, offset: 0, index: 0)
        encoder.setBuffer(out, offset: outOffset, index: 1)
        encoder.setBytes(&d, length: 4, index: 2)
        encoder.setBytes(&k, length: 4, index: 3)
        encoder.setBytes(&stride, length: 4, index: 4)
        encoder.dispatchThreads(MTLSize(width: shape.d / 4, height: tokens, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(shape.d / 4, 256), height: 1, depth: 1))
    }

    /// The whole routed branch of one prefill layer visit: plan from the
    /// router's ids and weights already in `scratch`, grouped experts, reduce.
    func encodePrefill(_ encoder: MTLComputeCommandEncoder, hidden: MTLBuffer, hiddenOffset: Int = 0, hiddenStride: Int,
                       pool: Inkling2ExpertPool, scratch: Inkling2MoEPrefillScratch, tokens: Int,
                       out: MTLBuffer, outOffset: Int = 0, outStride: Int,
                       inputQuantization: Inkling2MoEInputQuantization = .none) {
        encodeRoutePlan(encoder, scratch: scratch, tokens: tokens)
        encodeGroupedExperts(encoder, hidden: hidden, hiddenOffset: hiddenOffset, hiddenStride: hiddenStride,
                             pool: pool, scratch: scratch, tokens: tokens, inputQuantization: inputQuantization)
        encodeReduce(encoder, scratch: scratch, tokens: tokens, out: out, outOffset: outOffset, outStride: outStride)
    }

    /// A staged layer's decode visit through its cache binding: the binding's
    /// slot ids and slot pool, everything else as `encodeDecode`.
    func encodeDecode(_ encoder: MTLComputeCommandEncoder, shape: Inkling2MoEShape, x: MTLBuffer, xOffset: Int = 0,
                      binding: Inkling2DecodeBinding, routeWeights: MTLBuffer, routeWeightsOffset: Int = 0,
                      activation: MTLBuffer, activationOffset: Int = 0, out: MTLBuffer, outOffset: Int = 0,
                      inputQuantization: Inkling2MoEInputQuantization = .none,
                      inputScratch: (buffer: MTLBuffer, offset: Int)? = nil) {
        encodeDecode(encoder, shape: shape, x: x, xOffset: xOffset, ids: binding.ids, routeWeights: routeWeights,
                     routeWeightsOffset: routeWeightsOffset, pool: binding.pool, activation: activation,
                     activationOffset: activationOffset, out: out, outOffset: outOffset,
                     inputQuantization: inputQuantization, inputScratch: inputScratch)
    }

    /// The first half of `encodeDecode`: gate-up and SITU into `activation`.
    func encodeDecodeGateUp(_ encoder: MTLComputeCommandEncoder, shape: Inkling2MoEShape,
                            x: MTLBuffer, xOffset: Int = 0, ids: MTLBuffer, idsOffset: Int = 0,
                            pool: Inkling2ExpertPool, activation: MTLBuffer, activationOffset: Int = 0) {
        precondition(pool.servesDecode(of: shape), "pool and decode shapes differ")
        precondition(shape.isSupported && shape.f % Self.decodeRowsPerThreadgroup == 0)
        precondition(xOffset % 8 == 0 && activationOffset % 8 == 0 && idsOffset % 4 == 0,
                     "the decode kernels read half4 and uint")
        guard let kernels = encoded[pool.encoding] else { preconditionFailure("encoding") }
        var p = Self.params(shape, pool: pool, hiddenStride: shape.d)
        encoder.setComputePipelineState(kernels.decodeGateUp)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(ids, offset: idsOffset, index: 1)
        encoder.setBuffer(activation, offset: activationOffset, index: 2)
        encoder.setBuffer(pool.buffer, offset: pool.offset, index: 3)
        encoder.setBuffer(pool.gateUpScales, offset: pool.gateUpScalesOffset, index: 4)
        encoder.setBytes(&p, length: MemoryLayout<Inkling2MoEParams>.stride, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: shape.f / Self.decodeRowsPerThreadgroup, height: shape.topK, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threads, height: 1, depth: 1))
    }

    /// One token's routed branch: `x` (FP16, `d` values), `ids` and
    /// `routeWeights` (`topK` each, as the router writes them), `activation`
    /// scratch of `topK * f` halves; `out` gets `d` FP32 values, the experts
    /// summed in rank order.
    /// `inputQuantization` other than `.none` needs `inputScratch`, room for
    /// `d` halves (8-byte aligned), for the rounded input.
    func encodeDecode(_ encoder: MTLComputeCommandEncoder, shape: Inkling2MoEShape,
                      x: MTLBuffer, xOffset: Int = 0, ids: MTLBuffer, idsOffset: Int = 0,
                      routeWeights: MTLBuffer, routeWeightsOffset: Int = 0, pool: Inkling2ExpertPool,
                      activation: MTLBuffer, activationOffset: Int = 0, out: MTLBuffer, outOffset: Int = 0,
                      inputQuantization: Inkling2MoEInputQuantization = .none,
                      inputScratch: (buffer: MTLBuffer, offset: Int)? = nil) {
        precondition(pool.servesDecode(of: shape), "pool and decode shapes differ")
        precondition(shape.isSupported && shape.f % Self.decodeRowsPerThreadgroup == 0
                     && shape.d % Self.decodeRowsPerThreadgroup == 0)
        precondition(xOffset % 8 == 0 && activationOffset % 8 == 0, "the decode kernels read half4")
        precondition(idsOffset % 4 == 0 && routeWeightsOffset % 4 == 0 && outOffset % 4 == 0)
        guard let kernels = encoded[pool.encoding] else { preconditionFailure("encoding") }
        let vendor = inputQuantization == .vendorRequired
        var p = Self.params(shape, pool: pool, hiddenStride: shape.d, routeWeightInDownInput: vendor)
        let threads = MTLSize(width: Self.threads, height: 1, depth: 1)

        var input = (buffer: x, offset: xOffset)
        if inputQuantization != .none {
            guard let inputScratch else { preconditionFailure("input quantization needs inputScratch") }
            precondition(inputScratch.offset % 8 == 0 && inputScratch.buffer.length >= inputScratch.offset + shape.d * 2)
            encodeInputQuantization(encoder, encoding: pool.encoding, mode: inputQuantization,
                                    source: x, sourceOffset: xOffset, sourceStride: shape.d,
                                    destination: inputScratch.buffer, destinationOffset: inputScratch.offset,
                                    destinationStride: shape.d, rows: 1, width: shape.d)
            input = inputScratch
        }
        encodeDecodeGateUp(encoder, shape: shape, x: input.buffer, xOffset: input.offset, ids: ids, idsOffset: idsOffset,
                           pool: pool, activation: activation, activationOffset: activationOffset)
        if inputQuantization != .none {
            encodeInputQuantization(encoder, encoding: pool.encoding, mode: inputQuantization,
                                    source: activation, sourceOffset: activationOffset, sourceStride: shape.f,
                                    destination: activation, destinationOffset: activationOffset, destinationStride: shape.f,
                                    rows: shape.topK, width: shape.f,
                                    rowWeights: vendor ? (routeWeights, routeWeightsOffset, 1) : nil)
        }

        encoder.setComputePipelineState(kernels.decodeDown)
        encoder.setBuffer(activation, offset: activationOffset, index: 0)
        encoder.setBuffer(ids, offset: idsOffset, index: 1)
        encoder.setBuffer(routeWeights, offset: routeWeightsOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        encoder.setBuffer(pool.buffer, offset: pool.offset, index: 4)
        encoder.setBuffer(pool.downScales, offset: pool.downScalesOffset, index: 5)
        encoder.setBytes(&p, length: MemoryLayout<Inkling2MoEParams>.stride, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: shape.d / Self.decodeDownRowsPerThreadgroup, height: 1, depth: 1),
                                     threadsPerThreadgroup: threads)
    }
}

/// Compiles an Inkling 2 shader module (`Metal/Inkling2/<module>.metal`) on
/// its own, at the runtime's MSL version, as `MetalContext.moduleLibrary`
/// does for its registered modules. It resolves the resource bundle the
/// same way (the packaged app's `Contents/Resources` first), so this family
/// adds no entry to the shared module tables.
enum Inkling2ShaderLibrary {
    static let subdirectory = "Metal/Inkling2"

    private static let bundle: Bundle = {
        if let resources = Bundle.main.resourceURL {
            let packaged = resources.appendingPathComponent("Mference_Mference.bundle")
            if FileManager.default.fileExists(atPath: packaged.path), let bundle = Bundle(url: packaged) {
                return bundle
            }
        }
        return .module
    }()

    /// FNV-1a of the module's source text (0 when it is missing), so a
    /// calibration recorded for one build is not reused by another.
    static func sourceHash(module: String) -> UInt64 {
        guard let url = bundle.url(forResource: module, withExtension: "metal", subdirectory: subdirectory),
              let data = try? Data(contentsOf: url) else { return 0 }
        var hash: UInt64 = 0xCBF29CE484222325
        for byte in data {
            hash ^= UInt64(byte)
            hash &*= 0x100000001B3
        }
        return hash
    }

    static func compile(device: MTLDevice, module: String) throws -> MTLLibrary {
        guard let url = bundle.url(forResource: module, withExtension: "metal", subdirectory: subdirectory) else {
            throw MetalError.missingShaderResource(module)
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = MetalContext.shaderLanguageVersion
        do {
            return try device.makeLibrary(source: source, options: options)
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
    }
}

/// Picks the grouped-GEMM variant per GPU and weight encoding by timing a
/// prefill visit of the model's own layer, instead of a hard-coded family or
/// chip gate. The tensor-ops path must beat the simdgroup path by
/// `requiredMargin` to be chosen, so GPUs where the two time within noise
/// keep the same (simdgroup) path from run to run; a variant that fails to
/// run is skipped. The choice is kept for the process, keyed by GPU, shader
/// source, encoding, shape and chunk size, so later loads skip the timing.
enum Inkling2MoECalibration {
    struct Key: Hashable {
        let device: String
        let shaderHash: UInt64
        let encoding: Inkling2MoEEncoding
        let shape: Inkling2MoEShape
        let tokens: Int
    }

    struct Result: Sendable {
        let variant: Inkling2MoEGEMMVariant
        /// Best GPU seconds per visit, per variant that ran.
        let seconds: [Inkling2MoEGEMMVariant: Double]
    }

    /// Tensor-ops must be this much faster (fraction of the simdgroup time).
    static let requiredMargin = 0.05

    private static let lock = NSLock()
    nonisolated(unsafe) private static var results: [Key: Result] = [:]

    /// Times `repeats` prefill visits of `tokens` rows (synthetic inputs,
    /// uniform routes over the pool's experts) with each variant the kernels
    /// built, sets the kernels' variant for the pool's encoding and returns
    /// the timings. Overwrites `scratch`'s route buffers.
    @discardableResult
    static func calibrate(context: MetalContext, kernels: Inkling2MoEKernels, pool: Inkling2ExpertPool,
                          scratch: Inkling2MoEPrefillScratch, tokens requested: Int? = nil,
                          repeats: Int = 3) throws -> Result {
        let shape = scratch.shape
        let tokens = min(requested ?? scratch.maxTokens, scratch.maxTokens)
        let key = Key(device: context.device.name,
                      shaderHash: Inkling2ShaderLibrary.sourceHash(module: Inkling2MoEKernels.module),
                      encoding: pool.encoding, shape: shape, tokens: tokens)
        lock.lock()
        let cached = results[key]
        lock.unlock()
        if let cached {
            kernels.setVariant(cached.variant, for: pool.encoding)
            return cached
        }
        func keep(_ result: Result) -> Result {
            lock.lock(); results[key] = result; lock.unlock()
            kernels.setVariant(result.variant, for: pool.encoding)
            return result
        }
        let variants = kernels.availableVariants
        if variants.count == 1 { return keep(Result(variant: .simdgroup, seconds: [:])) }

        guard let hidden = context.device.makeBuffer(length: tokens * shape.d * 2, options: .storageModeShared),
              let out = context.device.makeBuffer(length: tokens * shape.d * 4, options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("Inkling 2 calibration")
        }
        var state: UInt64 = 0x9E3779B97F4A7C15
        func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        let h = hidden.contents().bindMemory(to: Float16.self, capacity: tokens * shape.d)
        for i in 0..<(tokens * shape.d) { h[i] = Float16(Float(Int(next() % 2001) - 1000) / 1000) }
        let ids = scratch.ids.contents().bindMemory(to: UInt32.self, capacity: tokens * shape.topK)
        let weights = scratch.routeWeights.contents().bindMemory(to: Float.self, capacity: tokens * shape.topK)
        for t in 0..<tokens {
            var chosen: [UInt32] = []
            while chosen.count < shape.topK {
                let e = UInt32(next() % UInt64(shape.experts))
                if !chosen.contains(e) { chosen.append(e) }
            }
            for r in 0..<shape.topK {
                ids[t * shape.topK + r] = chosen[r]
                weights[t * shape.topK + r] = 1
            }
        }
        let previous = kernels.variant(for: pool.encoding)
        var seconds: [Inkling2MoEGEMMVariant: Double] = [:]
        func time(_ variants: [Inkling2MoEGEMMVariant]) throws {
        for variant in variants {
            kernels.setVariant(variant, for: pool.encoding)
            var best = Double.infinity
            var failed = false
            for run in 0..<(repeats + 1) where !failed {
                guard let cb = context.queue.makeCommandBuffer(), let encoder = cb.makeComputeCommandEncoder() else {
                    throw PrefillGroupedRoutedMoEError.allocationFailed("Inkling 2 calibration encoder")
                }
                kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: pool, scratch: scratch,
                                      tokens: tokens, out: out, outStride: shape.d)
                encoder.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                if cb.status != .completed {
                    // The simdgroup path is the fallback; a failure there is an error.
                    if variant == .simdgroup {
                        throw MetalError.commandBufferFailed("Inkling 2 calibration: \(String(describing: cb.error))")
                    }
                    failed = true
                } else if run > 0 {
                    best = min(best, cb.gpuEndTime - cb.gpuStartTime)   // run 0 warms up
                }
            }
            if !failed { seconds[variant] = best }
        }
        }
        do {
            try time(variants)
        } catch {
            kernels.setVariant(previous, for: pool.encoding)
            throw error
        }
        var winner = Inkling2MoEGEMMVariant.simdgroup
        if let tensor = seconds[.tensorOps], let simd = seconds[.simdgroup], tensor < simd * (1 - requiredMargin) {
            winner = .tensorOps
        }
        return keep(Result(variant: winner, seconds: seconds))
    }
}
