import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// `FlashNextMatVec.encodeBatched` for the formats that used to fall back to
/// one GEMV dispatch per token: dense BF16 (FP16 and FP32 out) and INT8-2L
/// with an FP32 output. Each batched token row must equal the single-token
/// `encode` of that token bit for bit — the prefill path stays decode-exact —
/// at the Flash-Next GGUF install's shapes (HC inject 4 × 10240, router
/// 512 × 2560, shared-expert gate 1 × 2560, GDN alpha/beta 48 × 2560, HC
/// mixDown 320 × 10240 and mixUp 10240 × 320), with a partial last group of 8.
@Suite struct FlashNextMatVecBatchedTests {
    struct Case: CustomTestStringConvertible, Sendable {
        let name: String, rows: Int, cols: Int, int8: Bool, f32: Bool
        var testDescription: String { name }
    }
    static let cases = [
        Case(name: "bf16 4x10240 f32", rows: 4, cols: 10240, int8: false, f32: true),
        Case(name: "bf16 512x2560 f32", rows: 512, cols: 2560, int8: false, f32: true),
        Case(name: "bf16 1x2560 f32", rows: 1, cols: 2560, int8: false, f32: true),
        Case(name: "bf16 48x2560 f16", rows: 48, cols: 2560, int8: false, f32: false),
        Case(name: "int8-2l q8_0 320x10240 f32", rows: 320, cols: 10240, int8: true, f32: true),
        Case(name: "int8-2l q8_0 10240x320 f32", rows: 10240, cols: 320, int8: true, f32: true),
    ]

    static func matrix(_ c: Case, ctx: MetalContext, rng: inout SplitMix64) throws -> FlashNextWeightMatrix {
        let n = c.rows * c.cols
        if !c.int8 {
            let bits = (0..<n).map { _ in Quantization.bf16Bits(rng.uniform(-0.05, 0.05)) }
            let buf = try #require(bits.withUnsafeBytes {
                ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            })
            return .bf16(buffer: buf, offset: 0)
        }
        // Q8_0 INT8-2L regions: int8 values, then one fp16 d per 32 weights.
        let dCount = n / 32
        var bytes = [UInt8](repeating: 0, count: n + dCount * 2)
        for i in 0..<n { bytes[i] = UInt8(truncatingIfNeeded: rng.next()) }
        for i in 0..<dCount {
            let b = Float16(rng.uniform(1e-4, 2e-3)).bitPattern
            bytes[n + 2 * i] = UInt8(b & 0xFF); bytes[n + 2 * i + 1] = UInt8(b >> 8)
        }
        let buf = try #require(bytes.withUnsafeBytes {
            ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
        return .ggufInt8TwoLevel(.init(source: .q8_0, rows: c.rows, n: c.cols,
                                       values: (buf, 0), d: (buf, n), m: nil))
    }

    @Test(arguments: cases)
    func batchedEqualsSingleTokenBitForBit(c: Case) throws {
        let ctx = try MetalContext()
        let matVec = try FlashNextMatVec(context: ctx, int4: DequantInt4GEMV(context: ctx),
                                         int8Columns: 2560, ggufInt8TwoLevel: true)
        var rng = SplitMix64(seed: 0xB47C_0000 &+ UInt64(c.rows &* 31 &+ c.cols))
        let w = try Self.matrix(c, ctx: ctx, rng: &rng)
        let width = c.f32 ? 4 : 2
        for tokens in [1, 7, 8, 9, 13] {
            let x = (0..<(tokens * c.cols)).map { _ in Float16(rng.uniform(-1, 1)) }
            let xBuf = try #require(x.withUnsafeBytes {
                ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            })
            let bytes = tokens * c.rows * width
            let batched = try #require(ctx.device.makeBuffer(length: bytes + 64, options: .storageModeShared))
            let single = try #require(ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
            memset(batched.contents(), 0xEE, bytes + 64)
            let cb = try #require(ctx.queue.makeCommandBuffer())
            matVec.encodeBatched(commandBuffer: cb, matrix: w, x: xBuf, y: batched,
                                 matrixRows: c.rows, matrixColumns: c.cols, tokens: tokens,
                                 outputFloat32: c.f32)
            for t in 0..<tokens {
                matVec.encode(commandBuffer: cb, matrix: w, x: xBuf, xOffset: t * c.cols * 2,
                              y: single, yOffset: t * c.rows * width,
                              rows: c.rows, cols: c.cols, outputFloat32: c.f32)
            }
            cb.commit(); cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            #expect(memcmp(batched.contents(), single.contents(), bytes) == 0,
                    "\(tokens) tokens: the batched rows differ from the single-token GEMV")
            let tail = UnsafeRawBufferPointer(start: batched.contents().advanced(by: bytes), count: 64)
            #expect(tail.allSatisfy { $0 == 0xEE }, "\(tokens) tokens: wrote past the last row")
        }
    }

    /// Which kernel `encodeBatched` dispatches. One token of BF16 or of
    /// FP32-out INT8-2L (decode's hyper-connection and indexer projections)
    /// takes the single-token GEMV, so decode runs the kernels it always has;
    /// two or more tokens take the 8-token batched kernel. FP16-out INT8-2L
    /// keeps the batched kernel at one token, as before.
    @Test func oneTokenRoutesToTheGEMVAndMoreToTheBatchedKernel() throws {
        let ctx = try MetalContext()
        let matVec = try FlashNextMatVec(context: ctx, int4: DequantInt4GEMV(context: ctx),
                                         int8Columns: 2560, ggufInt8TwoLevel: true)
        var rng = SplitMix64(seed: 0x7E57_0001)
        let routes: [(Case, expectedAtOneToken: Int)] = [
            (Case(name: "bf16 f16", rows: 16, cols: 256, int8: false, f32: false), 0),
            (Case(name: "bf16 f32", rows: 16, cols: 256, int8: false, f32: true), 0),
            (Case(name: "int8-2l f32", rows: 16, cols: 256, int8: true, f32: true), 0),
            (Case(name: "int8-2l f16", rows: 16, cols: 256, int8: true, f32: false), 1),
        ]
        for (c, expectedAtOneToken) in routes {
            let w = try Self.matrix(c, ctx: ctx, rng: &rng)
            for tokens in [1, 2, 9] {
                let x = try #require(ctx.device.makeBuffer(length: tokens * c.cols * 2, options: .storageModeShared))
                let y = try #require(ctx.device.makeBuffer(length: tokens * c.rows * 4, options: .storageModeShared))
                let cb = try #require(ctx.queue.makeCommandBuffer())
                let before = matVec.batchedT8Dispatches
                matVec.encodeBatched(commandBuffer: cb, matrix: w, x: x, y: y,
                                     matrixRows: c.rows, matrixColumns: c.cols, tokens: tokens,
                                     outputFloat32: c.f32)
                cb.commit(); cb.waitUntilCompleted()
                #expect(cb.status == .completed)
                #expect(matVec.batchedT8Dispatches - before == (tokens == 1 ? expectedAtOneToken : 1),
                        "\(c.name), \(tokens) tokens: wrong kernel")
            }
        }
    }
}
