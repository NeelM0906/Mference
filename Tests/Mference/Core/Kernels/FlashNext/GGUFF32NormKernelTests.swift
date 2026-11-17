import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// U5 step 1 (WS4): the three norm kernels' F32-weight variants, used by the
/// GGUF install, whose `1 + w` norm vectors are F32 and not BF16-exact:
/// grouped RMSNorm (`FC_RMS_GROUPED_W_F32`), the per-head q/k norm
/// (`FC_PREFILL_PERHEAD_W_F32`) and the indexer's q/k norm
/// (`FC_FN_INDEXER_NORM_W_F32`). For each:
/// - with BF16-exact weights the F32 variant is bit-identical to the shipped
///   BF16 variant, which the vendor install still runs: the read is the only
///   change;
/// - with weights that are not BF16-exact it matches a CPU reference (or the
///   BF16 path within rounding), and it differs from the BF16-rounded weights,
///   which proves the constant reached the compiler.
@Suite struct GGUFF32NormKernelTests {
    static func buffer<T>(_ ctx: MetalContext, _ values: [T]) throws -> MTLBuffer {
        try #require(values.withUnsafeBytes {
            ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
    }

    static func read<T>(_ ctx: MetalContext, _ buffer: MTLBuffer, count: Int, as: T.Type) throws -> [T] {
        let bytes = count * MemoryLayout<T>.stride
        let shared = try #require(ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        let blit = try #require(cb.makeBlitCommandEncoder())
        blit.copy(from: buffer, sourceOffset: 0, to: shared, destinationOffset: 0, size: bytes)
        blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return Array(UnsafeBufferPointer(start: shared.contents().bindMemory(to: T.self, capacity: count), count: count))
    }

    /// `1 + w` weights: exact ones are BF16 values, inexact ones are not.
    static func weights(_ n: Int, exact: Bool, rng: inout SplitMix64) -> [Float] {
        (0..<n).map { _ in
            let w: Float = 1 + rng.uniform(-0.2, 0.2)
            return exact ? Quantization.bf16ToFloat(Quantization.bf16Bits(w)) : w
        }
    }

    static func bf16(_ w: [Float]) -> [UInt16] { w.map(Quantization.bf16Bits) }

    /// Number of elements that differ, plus the worst relative error, of two FP16 outputs.
    static func compare(_ a: [UInt16], _ b: [Float]) -> (differ: Int, rel: Float) {
        let av = a.map { Float(Float16(bitPattern: $0)) }
        return (zip(av, b).filter { $0 != $1 }.count, RelError.compute(actual: av, reference: b))
    }

    @Test func groupedNorm() throws {
        let ctx = try MetalContext()
        var rng = SplitMix64(seed: 0x5531_0001)
        let G = 2560, groups = 4, rows = 2, eps: Float = 1e-6
        let x = (0..<(rows * groups * G)).map { _ in Float16(rng.uniform(-3, 3)) }
        let exact = Self.weights(groups * G, exact: true, rng: &rng)
        let inexact = Self.weights(groups * G, exact: false, rng: &rng)
        let bf16Kernel = try RMSNorm(context: ctx)
        let f32Kernel = try RMSNorm(context: ctx, groupedWeightF32: true)
        func run(_ k: RMSNorm, _ w: MTLBuffer) throws -> [UInt16] {
            let out = try #require(ctx.device.makeBuffer(length: x.count * 2, options: .storageModeShared))
            let cb = try #require(ctx.queue.makeCommandBuffer())
            k.encodeBF16WGrouped(commandBuffer: cb, x: try Self.buffer(ctx, x), weight: w, out: out,
                                 groupSize: UInt32(G), groups: UInt32(groups), rows: rows, eps: eps)
            cb.commit(); cb.waitUntilCompleted()
            #expect(cb.error == nil)
            return try Self.read(ctx, out, count: x.count, as: UInt16.self)
        }
        #expect(try run(f32Kernel, Self.buffer(ctx, exact)) == run(bf16Kernel, Self.buffer(ctx, Self.bf16(exact))),
                "F32 variant must be bit-identical on BF16-exact weights")

        // CPU reference: per (row, group) RMS in Double, x * inv * w.
        var ref = [Float](repeating: 0, count: x.count)
        for slot in 0..<(rows * groups) {
            let base = slot * G, wbase = (slot % groups) * G
            let ms = (0..<G).reduce(0.0) { $0 + Double(Float(x[base + $1])) * Double(Float(x[base + $1])) } / Double(G)
            let inv = 1 / (ms + Double(eps)).squareRoot()
            for i in 0..<G { ref[base + i] = Float(Double(Float(x[base + i])) * inv * Double(inexact[wbase + i])) }
        }
        let got = try run(f32Kernel, Self.buffer(ctx, inexact))
        let (_, rel) = Self.compare(got, ref)
        #expect(rel < Tolerance.fp16Reduction, "F32-weight grouped norm vs CPU reference: \(rel)")
        let rounded = try run(bf16Kernel, Self.buffer(ctx, Self.bf16(inexact)))
        #expect(got != rounded, "F32 weights gave the BF16-rounded result: the constant had no effect")
    }

    @Test func perHeadNorm() throws {
        let ctx = try MetalContext()
        var rng = SplitMix64(seed: 0x5531_0002)
        let T = 3, heads = 4, D = 256, eps: Float = 1e-6, stride = heads * D
        let x = (0..<(T * stride)).map { _ in Float16(rng.uniform(-3, 3)) }
        let exact = Self.weights(D, exact: true, rng: &rng)
        let inexact = Self.weights(D, exact: false, rng: &rng)
        let bf16Kernel = try PrefillPerHeadNorm(context: ctx)
        let f32Kernel = try PrefillPerHeadNorm(context: ctx, weightF32: true)
        func run(_ k: PrefillPerHeadNorm, _ w: MTLBuffer) throws -> [UInt16] {
            let out = try #require(ctx.device.makeBuffer(length: x.count * 2, options: .storageModeShared))
            let cb = try #require(ctx.queue.makeCommandBuffer())
            k.encodeBF16W(commandBuffer: cb, x: try Self.buffer(ctx, x), weight: w, out: out,
                          queryCount: UInt32(T), headDim: UInt32(D), numHeads: UInt32(heads),
                          tokenStrideElements: UInt32(stride), eps: eps)
            cb.commit(); cb.waitUntilCompleted()
            #expect(cb.error == nil)
            return try Self.read(ctx, out, count: x.count, as: UInt16.self)
        }
        #expect(try run(f32Kernel, Self.buffer(ctx, exact)) == run(bf16Kernel, Self.buffer(ctx, Self.bf16(exact))),
                "F32 variant must be bit-identical on BF16-exact weights")
        var ref = [Float](repeating: 0, count: x.count)
        for h in 0..<(T * heads) {
            let base = h * D
            let ms = (0..<D).reduce(0.0) { $0 + Double(Float(x[base + $1])) * Double(Float(x[base + $1])) } / Double(D)
            let inv = 1 / (ms + Double(eps)).squareRoot()
            for i in 0..<D { ref[base + i] = Float(Double(Float(x[base + i])) * inv * Double(inexact[i])) }
        }
        let got = try run(f32Kernel, Self.buffer(ctx, inexact))
        let (_, rel) = Self.compare(got, ref)
        #expect(rel < Tolerance.fp16Reduction, "F32-weight per-head norm vs CPU reference: \(rel)")
        #expect(got != (try run(bf16Kernel, Self.buffer(ctx, Self.bf16(inexact)))),
                "F32 weights gave the BF16-rounded result: the constant had no effect")
    }

    /// The indexer prepare (query norm + RoPE) and block pooling (key norm), at
    /// the production indexer geometry, over one completed block.
    @Test func indexerNorms() throws {
        let ctx = try MetalContext()
        var rng = SplitMix64(seed: 0x5531_0003)
        let geometry = FlashNextIndexer.Geometry(numHeads: 4, numKVHeads: 1, headDim: 128, compressRatio: 4,
                                                 blockBudget: 8, rotaryDim: 64, theta: 1e7, eps: 1e-6)
        let rows = geometry.compressRatio
        let projection = (0..<(rows * geometry.projRows)).map { _ in rng.uniform(-2, 2) }
        let qExact = Self.weights(geometry.headDim, exact: true, rng: &rng)
        let kExact = Self.weights(geometry.headDim, exact: true, rng: &rng)
        let qInexact = Self.weights(geometry.headDim, exact: false, rng: &rng)
        let kInexact = Self.weights(geometry.headDim, exact: false, rng: &rng)
        let matVec = try FlashNextMatVec(context: ctx, int4: try DequantInt4GEMV(context: ctx))
        func run(f32: Bool, q: MTLBuffer, k: MTLBuffer) throws -> (queries: [Float], blocks: [Float]) {
            let indexer = try FlashNextIndexer(context: ctx, matVec: matVec, geometry: geometry, normWeightsF32: f32)
            let scratch = try indexer.makeScratch(device: ctx.device, rows: rows, maxTokens: 16)
            let cache = try indexer.makeLayerCache(device: ctx.device, maxTokens: 16)
            let source = try Self.buffer(ctx, projection)
            let cb = try #require(ctx.queue.makeCommandBuffer())
            let blit = try #require(cb.makeBlitCommandEncoder())
            blit.copy(from: source, sourceOffset: 0, to: scratch.projection, destinationOffset: 0,
                      size: projection.count * 4)
            blit.endEncoding()
            indexer.encodePrepare(commandBuffer: cb, qNorm: q, qNormOffset: 0, kNorm: k, kNormOffset: 0,
                                  scratch: scratch, cache: cache, rows: rows, startPosition: 0)
            cb.commit(); cb.waitUntilCompleted()
            #expect(cb.error == nil)
            return (try Self.read(ctx, scratch.queries, count: rows * geometry.numHeads * geometry.headDim, as: Float.self),
                    try Self.read(ctx, cache.blockKeys, count: geometry.headDim, as: Float.self))
        }
        let a = try run(f32: true, q: Self.buffer(ctx, qExact), k: Self.buffer(ctx, kExact))
        let b = try run(f32: false, q: Self.buffer(ctx, Self.bf16(qExact)), k: Self.buffer(ctx, Self.bf16(kExact)))
        #expect(a.queries.map(\.bitPattern) == b.queries.map(\.bitPattern), "queries: F32 variant not bit-identical")
        #expect(a.blocks.map(\.bitPattern) == b.blocks.map(\.bitPattern), "block keys: F32 variant not bit-identical")
        let c = try run(f32: true, q: Self.buffer(ctx, qInexact), k: Self.buffer(ctx, kInexact))
        let d = try run(f32: false, q: Self.buffer(ctx, Self.bf16(qInexact)), k: Self.buffer(ctx, Self.bf16(kInexact)))
        #expect(c.queries != d.queries && c.blocks != d.blocks, "the F32 constant had no effect")
        // The only difference is the weights' BF16 rounding (≤ 2^-9 relative).
        #expect(RelError.compute(actual: c.queries, reference: d.queries) < 4e-3)
        #expect(RelError.compute(actual: c.blocks, reference: d.blocks) < 4e-3)
    }
}
