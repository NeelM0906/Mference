import Foundation
import Metal
import MetalPerformanceShaders
import Testing
@testable import Mference

/// **Measurement**, not a gate (WS1, Unsloth prefill): GPU and CPU-encode cost
/// of the Unsloth UD-IQ3_XXS Flash-Next install's resident (non-routed)
/// projections at one 1,024-token prefill chunk, through the production
/// `FlashNextMatVec.encodeBatched` entry, beside an FP16 roofline
/// (`MPSMatrixMultiplication` at the same shapes) and beside batched
/// prototypes of the paths that fell back to one GEMV per token before the
/// batched BF16 / FP32-out INT8-2L kernels (that loop is rebuilt here from
/// `FlashNextMatVec.encode`).
/// Synthetic weights at the install's shapes and formats (INT8-2L Q6_K/Q8_0,
/// BF16 for the narrowed F32 matrices). Env-gated:
/// `MFERENCE_FLASHNEXT_PREFILL_KBENCH=1`. Prints one line per case; asserts
/// only that each prototype matches the per-token GEMV output bit for bit.
@Suite(.serialized) struct FlashNextPrefillProjectionBenchmarks {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_FLASHNEXT_PREFILL_KBENCH"] == "1"
    }
    static let tokens = 1024
    static let hidden = 2560

    enum Format: String { case q6K = "Q6_K", q8_0 = "Q8_0", bf16 = "BF16" }

    /// One resident projection: `rows × cols` weights, how many of the 48
    /// layers run it per prefill token, and whether its consumer wants FP32.
    struct Shape: CustomTestStringConvertible {
        let name: String, rows: Int, cols: Int, format: Format, perStack: Int, f32: Bool
        var testDescription: String { name }
        var macs: Double { Double(rows) * Double(cols) }
    }

    /// Every resident matmul a Flash-Next GGUF prefill layer runs (tensor table
    /// `blk.N.*`, 36 GDN + 12 full-attention layers). Mixed Q6_K/Q8_0 tensors
    /// are benched as Q6_K, the dearer of the two.
    static let fp16OutShapes: [Shape] = [
        Shape(name: "attn_qkv 2560->10240", rows: 10240, cols: 2560, format: .q6K, perStack: 36, f32: false),
        Shape(name: "attn_gate 2560->6144", rows: 6144, cols: 2560, format: .q6K, perStack: 36, f32: false),
        Shape(name: "ssm_out 6144->2560", rows: 2560, cols: 6144, format: .q6K, perStack: 36, f32: false),
        Shape(name: "attn_q 2560->12288", rows: 12288, cols: 2560, format: .q6K, perStack: 12, f32: false),
        Shape(name: "attn_output 6144->2560", rows: 2560, cols: 6144, format: .q6K, perStack: 12, f32: false),
        Shape(name: "attn_k|v 2560->512", rows: 512, cols: 2560, format: .q6K, perStack: 24, f32: false),
        Shape(name: "shexp gate|up 2560->640", rows: 640, cols: 2560, format: .q6K, perStack: 96, f32: false),
        Shape(name: "shexp down 640->2560", rows: 2560, cols: 640, format: .q8_0, perStack: 48, f32: false),
    ]

    /// The projections whose batched call took the per-token GEMV loop:
    /// INT8-2L with an FP32 output (hyper-connection mixDown/mixUp) and every
    /// BF16 matrix (F32 tensors narrowed at load).
    static let perTokenShapes: [Shape] = [
        Shape(name: "hc mixDown 10240->320 (INT8-2L, f32)", rows: 320, cols: 10240, format: .q8_0, perStack: 96, f32: true),
        Shape(name: "hc mixUp 320->10240 (INT8-2L, f32)", rows: 10240, cols: 320, format: .q8_0, perStack: 96, f32: true),
        Shape(name: "hc inject 10240->4 (BF16, f32)", rows: 4, cols: 10240, format: .bf16, perStack: 96, f32: true),
        Shape(name: "router 2560->512 (BF16, f32)", rows: 512, cols: 2560, format: .bf16, perStack: 48, f32: true),
        Shape(name: "shexp gate 2560->1 (BF16, f32)", rows: 1, cols: 2560, format: .bf16, perStack: 48, f32: true),
        Shape(name: "ssm alpha|beta 2560->48 (BF16, f16)", rows: 48, cols: 2560, format: .bf16, perStack: 72, f32: false),
        Shape(name: "indexer q 2560->512 (BF16, f32)", rows: 512, cols: 2560, format: .bf16, perStack: 12, f32: true),
        Shape(name: "indexer k 2560->128 (BF16, f32)", rows: 128, cols: 2560, format: .bf16, perStack: 12, f32: true),
    ]

    // MARK: - Synthetic tensors

    struct LCG {
        var s: UInt64
        mutating func next() -> UInt32 { s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407; return UInt32(s >> 33) }
        mutating func unit() -> Float { Float(next() & 0xFFFF) / 65535 * 2 - 1 }
    }

    static func buffer(_ device: MTLDevice, _ bytes: Int) throws -> MTLBuffer {
        try #require(device.makeBuffer(length: max(16, bytes), options: .storageModeShared))
    }

    /// A projection as the runner sees it, with its weights in one buffer.
    static func matrix(_ device: MTLDevice, _ s: Shape, rng: inout LCG) throws -> FlashNextWeightMatrix {
        let n = s.rows * s.cols
        switch s.format {
        case .bf16:
            let b = try buffer(device, n * 2)
            let p = b.contents().bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n { p[i] = UInt16((rng.unit() * 0.05).bitPattern >> 16) }
            return .bf16(buffer: b, offset: 0)
        case .q6K, .q8_0:
            let source: GGUFInt8TwoLevelGEMV.Source = s.format == .q6K ? .q6_K : .q8_0
            let dCount = n / source.blockWeights, mCount = s.format == .q6K ? n / 16 : 0
            let dOff = (n + 255) / 256 * 256, mOff = dOff + (dCount * 2 + 255) / 256 * 256
            let b = try buffer(device, mOff + mCount)
            let v = b.contents().bindMemory(to: Int8.self, capacity: n)
            for i in 0..<n { v[i] = Int8(truncatingIfNeeded: Int(rng.next() % 255) - 127) }
            let d = b.contents().advanced(by: dOff).bindMemory(to: Float16.self, capacity: dCount)
            for i in 0..<dCount { d[i] = Float16(0.0005 + 0.0005 * abs(rng.unit())) }
            if mCount > 0 {
                let m = b.contents().advanced(by: mOff).bindMemory(to: Int8.self, capacity: mCount)
                for i in 0..<mCount { m[i] = Int8(truncatingIfNeeded: Int(rng.next() % 63) - 31) }
            }
            return .ggufInt8TwoLevel(.init(source: source, rows: s.rows, n: s.cols,
                                           values: (b, 0), d: (b, dOff),
                                           m: mCount > 0 ? (b, mOff) : nil))
        }
    }

    static func activations(_ device: MTLDevice, count: Int, rng: inout LCG) throws -> MTLBuffer {
        let b = try buffer(device, count * 2)
        let p = b.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { p[i] = Float16(rng.unit()) }
        return b
    }

    // MARK: - Timing

    struct Timing { let gpu: Double; let encode: Double; let wall: Double }

    /// Best of `iterations` command buffers after one warm-up: GPU span, CPU
    /// encode time and commit-to-complete wall time, in milliseconds.
    static func time(_ ctx: MetalContext, iterations: Int = 3,
                     _ body: (MTLCommandBuffer) -> Void) throws -> Timing {
        var best = Timing(gpu: .infinity, encode: .infinity, wall: .infinity)
        for i in 0...iterations {
            let cb = try #require(ctx.queue.makeCommandBuffer())
            let start = Date()
            body(cb)
            let encoded = Date()
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.error == nil, "\(String(describing: cb.error))")
            let done = Date()
            guard i > 0 else { continue }
            let t = Timing(gpu: (cb.gpuEndTime - cb.gpuStartTime) * 1e3,
                           encode: encoded.timeIntervalSince(start) * 1e3,
                           wall: done.timeIntervalSince(start) * 1e3)
            if t.wall < best.wall { best = t }
        }
        return best
    }

    /// FP16 roofline: `Y[T, rows] = X[T, cols] · W[rows, cols]ᵀ` through MPS.
    static func mpsFP16(_ ctx: MetalContext, rows: Int, cols: Int, tokens: Int) throws -> Timing {
        let device = ctx.device
        var rng = LCG(s: 7)
        let x = try activations(device, count: tokens * cols, rng: &rng)
        let w = try activations(device, count: rows * cols, rng: &rng)
        let y = try buffer(device, tokens * rows * 2)
        let left = MPSMatrix(buffer: x, descriptor: MPSMatrixDescriptor(
            rows: tokens, columns: cols, rowBytes: cols * 2, dataType: .float16))
        let right = MPSMatrix(buffer: w, descriptor: MPSMatrixDescriptor(
            rows: rows, columns: cols, rowBytes: cols * 2, dataType: .float16))
        let out = MPSMatrix(buffer: y, descriptor: MPSMatrixDescriptor(
            rows: tokens, columns: rows, rowBytes: rows * 2, dataType: .float16))
        let mm = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                         resultRows: tokens, resultColumns: rows, interiorColumns: cols,
                                         alpha: 1, beta: 0)
        return try time(ctx) { mm.encode(commandBuffer: $0, leftMatrix: left, rightMatrix: right, resultMatrix: out) }
    }

    static func line(_ s: String) { print("[fn-prefill-kbench] " + s) }

    // MARK: - Dense FP16-output projections

    @Test(.enabled(if: enabled)) func denseInt8TwoLevelVsFP16Roofline() throws {
        let ctx = try MetalContext()
        let matVec = try FlashNextMatVec(context: ctx, int4: DequantInt4GEMV(context: ctx),
                                         int8Columns: Self.hidden, ggufInt8TwoLevel: true)
        var rng = LCG(s: 0x5EED)
        let t = Self.tokens
        Self.line("dense resident projections, \(t) tokens; per-stack = calls per token over 48 layers")
        var prodPerToken = 0.0, mpsPerToken = 0.0
        for s in Self.fp16OutShapes {
            let w = try Self.matrix(ctx.device, s, rng: &rng)
            let x = try Self.activations(ctx.device, count: t * s.cols, rng: &rng)
            let y = try Self.buffer(ctx.device, t * s.rows * 2)
            let prod = try Self.time(ctx) {
                matVec.encodeBatched(commandBuffer: $0, matrix: w, x: x, y: y,
                                     matrixRows: s.rows, matrixColumns: s.cols, tokens: t)
            }
            let mps = try Self.mpsFP16(ctx, rows: s.rows, cols: s.cols, tokens: t)
            let flops = 2 * s.macs * Double(t)
            prodPerToken += prod.gpu / Double(t) * Double(s.perStack)
            mpsPerToken += mps.gpu / Double(t) * Double(s.perStack)
            Self.line(String(format: "%@ %@ x%d: int8-2l t8 %.2f ms (%.2f TFLOP/s) | mps fp16 %.2f ms (%.2f TFLOP/s) | %.1fx",
                             s.name as NSString, s.format.rawValue as NSString, s.perStack,
                             prod.gpu, flops / prod.gpu / 1e9, mps.gpu, flops / mps.gpu / 1e9, prod.gpu / mps.gpu))
        }
        Self.line(String(format: "dense total per prompt token: int8-2l t8 %.3f ms, mps fp16 %.3f ms; per 2,940 tokens %.1f s vs %.1f s",
                         prodPerToken, mpsPerToken, prodPerToken * 2.94, mpsPerToken * 2.94))
    }

    // MARK: - Per-token fallback loops vs batched prototypes

    static let prototypeSource = """
    #include <metal_stdlib>
    using namespace metal;
    struct P { uint rows; uint n; uint bd_shift; uint has_m; };
    struct B { uint tokens; uint x_stride; uint y_stride; uint pad; };
    // gguf_int8_2l_gemm_t8_f16out with an FP32 store: same chunking, same
    // per-chunk FMA order, same simd_sum, so each token's row equals the
    // single-token gguf_int8_2l_gemv_f32out bit for bit.
    kernel void proto_int8_2l_gemm_t8_f32out(
        device const char* values [[buffer(0)]], device const ushort* dbits [[buffer(1)]],
        device const char* m [[buffer(2)]], constant P& p [[buffer(3)]],
        device const half* x [[buffer(4)]], device float* y [[buffer(5)]],
        constant B& bp [[buffer(6)]],
        uint2 tg [[threadgroup_position_in_grid]], uint sg_idx [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]])
    {
        const uint row = tg.x * 8 + sg_idx;
        const uint t0 = tg.y * 8;
        if (row >= p.rows || t0 >= bp.tokens) { return; }
        const uint nt = min(8u, bp.tokens - t0);
        const uint n = p.n;
        device const char* row_v = values + ulong(row) * n;
        const ulong row_w = ulong(row) * n;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint c = lane; c < n / 8; c += 32) {
            const uint w = 8 * c;
            const char4 va = *((device const char4*)(row_v + w));
            const char4 vb = *((device const char4*)(row_v + w + 4));
            const ulong gw = row_w + w;
            float scale = float(as_type<half>(dbits[gw >> p.bd_shift]));
            if (p.has_m != 0) { scale *= float(m[gw >> 4]); }
            for (uint t = 0; t < nt; ++t) {
                device const half* xt = x + ulong(t0 + t) * bp.x_stride;
                const half4 xa = *((device const half4*)(xt + w));
                const half4 xb = *((device const half4*)(xt + w + 4));
                float s = 0.0f;
                s = fma(float(va.x), float(xa.x), s); s = fma(float(va.y), float(xa.y), s);
                s = fma(float(va.z), float(xa.z), s); s = fma(float(va.w), float(xa.w), s);
                s = fma(float(vb.x), float(xb.x), s); s = fma(float(vb.y), float(xb.y), s);
                s = fma(float(vb.z), float(xb.z), s); s = fma(float(vb.w), float(xb.w), s);
                acc[t] = fma(scale, s, acc[t]);
            }
        }
        for (uint t = 0; t < nt; ++t) {
            const float v = simd_sum(acc[t]);
            if (lane == 0) { y[ulong(t0 + t) * bp.y_stride + row] = v; }
        }
    }
    // flashnext_gemv_bf16{,_f32out} over 8 tokens per simdgroup: each lane's
    // FMA chain per token is the GEMV's, so each token's row is bit-identical.
    template <typename T>
    static void bf16_t8(device const bfloat* W, device const half* x, device T* y,
                        uint M, uint N, uint tokens, uint2 tg, uint sg_idx, uint lane) {
        const uint row = tg.x * 8 + sg_idx;
        const uint t0 = tg.y * 8;
        if (row >= M || t0 >= tokens) { return; }
        const uint nt = min(8u, tokens - t0);
        device const bfloat* W_row = W + ulong(row) * ulong(N);
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint i = lane; i < N; i += 32u) {
            const float w = float(W_row[i]);
            for (uint t = 0; t < nt; ++t) {
                acc[t] = fma(w, float(x[ulong(t0 + t) * N + i]), acc[t]);
            }
        }
        for (uint t = 0; t < nt; ++t) {
            const float v = simd_sum(acc[t]);
            if (lane == 0) { y[ulong(t0 + t) * M + row] = T(v); }
        }
    }
    kernel void proto_bf16_t8_f32out(device const bfloat* W [[buffer(0)]], device const half* x [[buffer(1)]],
        device float* y [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
        constant uint& T [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
    { bf16_t8<float>(W, x, y, M, N, T, tg, sg, lane); }
    kernel void proto_bf16_t8_f16out(device const bfloat* W [[buffer(0)]], device const half* x [[buffer(1)]],
        device half* y [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
        constant uint& T [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
    { bf16_t8<half>(W, x, y, M, N, T, tg, sg, lane); }
    """

    @Test(.enabled(if: enabled)) func perTokenFallbackVsBatchedPrototype() throws {
        let ctx = try MetalContext()
        let device = ctx.device
        let matVec = try FlashNextMatVec(context: ctx, int4: DequantInt4GEMV(context: ctx),
                                         int8Columns: Self.hidden, ggufInt8TwoLevel: true)
        // The INT8-2L prototype mirrors the production module's safe math; the
        // BF16 one mirrors the main library's default options.
        let safe = MTLCompileOptions(); safe.languageVersion = MetalContext.shaderLanguageVersion; safe.mathMode = .safe
        let plain = MTLCompileOptions(); plain.languageVersion = MetalContext.shaderLanguageVersion
        let safeLib = try device.makeLibrary(source: Self.prototypeSource, options: safe)
        let plainLib = try device.makeLibrary(source: Self.prototypeSource, options: plain)
        let int8F32 = try device.makeComputePipelineState(function: try #require(safeLib.makeFunction(name: "proto_int8_2l_gemm_t8_f32out")))
        let bf16F32 = try device.makeComputePipelineState(function: try #require(plainLib.makeFunction(name: "proto_bf16_t8_f32out")))
        let bf16F16 = try device.makeComputePipelineState(function: try #require(plainLib.makeFunction(name: "proto_bf16_t8_f16out")))

        var rng = LCG(s: 0xFA11)
        let t = Self.tokens
        Self.line("per-token GEMV loop (FlashNextMatVec.encode per token) vs batched t8 prototype, \(t) tokens")
        var loopPerToken = 0.0, loopWallPerToken = 0.0, protoPerToken = 0.0, dispatchesPerToken = 0
        for s in Self.perTokenShapes {
            let w = try Self.matrix(device, s, rng: &rng)
            let x = try Self.activations(device, count: t * s.cols, rng: &rng)
            let outBytes = t * s.rows * (s.f32 ? 4 : 2)
            let yLoop = try Self.buffer(device, outBytes)
            let yProto = try Self.buffer(device, outBytes)
            // The fallback `encodeBatched` used before it batched these formats:
            // one single-token GEMV per prompt token at that token's offsets.
            let outStride = s.f32 ? 4 : 2
            let loop = try Self.time(ctx) { cb in
                for row in 0..<t {
                    matVec.encode(commandBuffer: cb, matrix: w,
                                  x: x, xOffset: row * s.cols * 2,
                                  y: yLoop, yOffset: row * s.rows * outStride,
                                  rows: s.rows, cols: s.cols, outputFloat32: s.f32)
                }
            }
            let proto = try Self.time(ctx) { cb in
                guard let enc = cb.makeComputeCommandEncoder() else { return }
                let grid = MTLSize(width: (s.rows + 7) / 8, height: (t + 7) / 8, depth: 1)
                switch w {
                case let .ggufInt8TwoLevel(q):
                    var p = GGUFInt8TwoLevelGEMV.Params(rows: UInt32(s.rows), n: UInt32(s.cols),
                                                        bdShift: UInt32(q.source.blockWeights.trailingZeroBitCount),
                                                        hasM: q.m == nil ? 0 : 1)
                    var b = GGUFInt8TwoLevelGEMV.BatchParams(tokens: UInt32(t), xStride: UInt32(s.cols), yStride: UInt32(s.rows))
                    enc.setComputePipelineState(int8F32)
                    enc.setBuffer(q.values.buffer, offset: q.values.offset, index: 0)
                    enc.setBuffer(q.d.buffer, offset: q.d.offset, index: 1)
                    enc.setBuffer(q.m?.buffer ?? q.values.buffer, offset: q.m?.offset ?? q.values.offset, index: 2)
                    enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 3)
                    enc.setBuffer(x, offset: 0, index: 4)
                    enc.setBuffer(yProto, offset: 0, index: 5)
                    enc.setBytes(&b, length: MemoryLayout.size(ofValue: b), index: 6)
                case let .bf16(buffer, offset):
                    var m = UInt32(s.rows), n = UInt32(s.cols), tt = UInt32(t)
                    enc.setComputePipelineState(s.f32 ? bf16F32 : bf16F16)
                    enc.setBuffer(buffer, offset: offset, index: 0)
                    enc.setBuffer(x, offset: 0, index: 1)
                    enc.setBuffer(yProto, offset: 0, index: 2)
                    enc.setBytes(&m, length: 4, index: 3)
                    enc.setBytes(&n, length: 4, index: 4)
                    enc.setBytes(&tt, length: 4, index: 5)
                default:
                    preconditionFailure("unexpected format")
                }
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }
            let identical = memcmp(yLoop.contents(), yProto.contents(), outBytes) == 0
            loopPerToken += loop.gpu / Double(t) * Double(s.perStack)
            loopWallPerToken += loop.wall / Double(t) * Double(s.perStack)
            protoPerToken += proto.gpu / Double(t) * Double(s.perStack)
            dispatchesPerToken += s.perStack
            Self.line(String(format: "%@ x%d: loop gpu %.2f ms, encode %.2f ms, wall %.2f ms | batched gpu %.3f ms | %.0fx | bit-identical %@",
                             s.name as NSString, s.perStack, loop.gpu, loop.encode, loop.wall, proto.gpu,
                             loop.wall / proto.gpu, identical ? "yes" : "NO"))
            #expect(identical, "\(s.name): the batched prototype must equal the per-token loop bit for bit")
        }
        Self.line(String(format: "per prompt token: %d GEMV dispatches; loop gpu %.3f ms, loop wall %.3f ms, batched gpu %.3f ms; per 2,940 tokens %.1f s gpu / %.1f s wall vs %.2f s batched",
                         dispatchesPerToken, loopPerToken, loopWallPerToken, protoPerToken,
                         loopPerToken * 2.94, loopWallPerToken * 2.94, protoPerToken * 2.94))
    }
}
