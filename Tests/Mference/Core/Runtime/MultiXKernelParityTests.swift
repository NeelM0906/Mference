import Foundation
import Metal
import Testing
@testable import Mference

/// Byte-identity locks for the MTP speculative-verify kernels: the
/// multi-token INT4 GEMV and the multi-token fused greedy head must be
/// bit-identical, per token, to the single-token decode kernels they stand
/// in for. This is the property that makes speculative decode's emitted
/// tokens equal plain decode's for any draft quality.
@Suite struct MultiXKernelParityTests {

    private struct Fixture {
        let ctx: MetalContext
        let w: MTLBuffer
        let s: MTLBuffer
        let b: MTLBuffer
        let x: MTLBuffer
        let k: Int
        let n: Int
        let t: Int
    }

    private func makeFixture(k: Int, n: Int, t: Int, seed: UInt64) throws -> Fixture {
        let ctx = try MetalContext()
        let device = ctx.device
        var rng = seed
        func nextByte() -> UInt8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return UInt8(truncatingIfNeeded: rng >> 33)
        }
        let groups = k / 64
        let w = device.makeBuffer(length: n * k / 2, options: .storageModeShared)!
        let wp = w.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<(n * k / 2) { wp[i] = nextByte() }
        let s = device.makeBuffer(length: n * groups * 2, options: .storageModeShared)!
        let b = device.makeBuffer(length: n * groups * 2, options: .storageModeShared)!
        let sp = s.contents().assumingMemoryBound(to: UInt16.self)
        let bp = b.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<(n * groups) {
            sp[i] = Quantization.bf16Bits(0.001 + 0.0007 * Float(i % 13))
            bp[i] = Quantization.bf16Bits(-0.03 + 0.006 * Float(i % 9))
        }
        let x = device.makeBuffer(length: t * k * 2, options: .storageModeShared)!
        let xp = x.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<(t * k) { xp[i] = Float16(Float(nextByte()) / 64.0 - 2.0) }
        return Fixture(ctx: ctx, w: w, s: s, b: b, x: x, k: k, n: n, t: t)
    }

    @Test("multi-x GEMV rows equal per-token decode GEMV bit for bit",
          arguments: [(64, 64, 2), (5120, 1024, 4), (1024, 5120, 7)])
    func multixMatchesGEMV(k: Int, n: Int, t: Int) throws {
        let f = try makeFixture(k: k, n: n, t: t, seed: 0xFEED_0000 &+ UInt64(k &* n))
        let ctx = f.ctx
        let yMulti = ctx.device.makeBuffer(length: t * n * 2, options: .storageModeShared)!
        let ySingle = ctx.device.makeBuffer(length: t * n * 2, options: .storageModeShared)!
        let gemv = try DequantInt4GEMV(context: ctx)
        let multix = try DequantInt4GEMVMultiX(context: ctx)

        let cb = ctx.queue.makeCommandBuffer()!
        multix.encode(commandBuffer: cb, weights: f.w, scales: f.s, biases: f.b,
                      x: f.x, y: yMulti, m: n, n: k, tokens: t)
        for row in 0..<t {
            gemv.encode(commandBuffer: cb,
                        weights: f.w, scales: f.s, biases: f.b,
                        x: f.x, xOffset: row * k * 2,
                        y: ySingle, yOffset: row * n * 2,
                        m: UInt32(n), n: UInt32(k))
        }
        cb.commit(); cb.waitUntilCompleted()

        let a = yMulti.contents().bindMemory(to: UInt16.self, capacity: t * n)
        let c = ySingle.contents().bindMemory(to: UInt16.self, capacity: t * n)
        var mismatches = 0
        for i in 0..<(t * n) where a[i] != c[i] { mismatches += 1 }
        #expect(mismatches == 0, "k=\(k) n=\(n) t=\(t): \(mismatches) mismatched outputs")
    }

    @Test("multi-x greedy head equals per-token fused head",
          arguments: [(64, 1024, 3), (256, 1000, 5)])
    func multixHeadMatchesFusedHead(d: Int, vocab: Int, t: Int) throws {
        let f = try makeFixture(k: d, n: vocab, t: t, seed: 0xABCD_1234 &+ UInt64(d))
        let ctx = f.ctx
        // The fused head normalizes internally; feed it a unit norm weight so
        // both paths consume the same normalized rows.
        let unitNorm = ctx.device.makeBuffer(length: d * 2, options: .storageModeShared)!
        let np = unitNorm.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<d { np[i] = Quantization.bf16Bits(1.0 + 0.05 * Float(i % 3)) }

        let rms = try RMSNorm(context: ctx)
        let normedRows = ctx.device.makeBuffer(length: t * d * 2, options: .storageModeShared)!
        let outMulti = ctx.device.makeBuffer(length: t * 4, options: .storageModeShared)!
        let outSingle = ctx.device.makeBuffer(length: 4, options: .storageModeShared)!
        let fused = try LMHeadChainInt4(context: ctx, maxD: d, maxVocab: vocab)
        let multi = try LMHeadGreedyMultiX(context: ctx, maxVocab: vocab)

        // Multi-x path: explicit per-row rms (the decode kernel) + one pass.
        let cb = ctx.queue.makeCommandBuffer()!
        for row in 0..<t {
            rms.encodeBF16W(commandBuffer: cb,
                            x: f.x, xOffset: row * d * 2,
                            weight: unitNorm,
                            out: normedRows, outOffset: row * d * 2,
                            d: UInt32(d), eps: 1e-6)
        }
        multi.encode(commandBuffer: cb,
                     xNormed: normedRows,
                     weights: f.w, scales: f.s, biases: f.b,
                     outTokens: outMulti,
                     d: d, vocab: vocab, tokens: t)
        cb.commit(); cb.waitUntilCompleted()

        // Reference: the decode fused head once per row.
        var reference: [UInt32] = []
        for row in 0..<t {
            let cb2 = ctx.queue.makeCommandBuffer()!
            fused.encodeGreedyDecode(commandBuffer: cb2,
                                     hidden: f.x, hiddenOffset: row * d * 2,
                                     normWeight: unitNorm,
                                     weights: f.w, scales: f.s, biases: f.b,
                                     outToken: outSingle,
                                     d: UInt32(d), vocab: UInt32(vocab))
            cb2.commit(); cb2.waitUntilCompleted()
            reference.append(outSingle.contents().load(as: UInt32.self))
        }
        let got = Array(UnsafeBufferPointer(
            start: outMulti.contents().bindMemory(to: UInt32.self, capacity: t),
            count: t))
        #expect(got == reference, "d=\(d) vocab=\(vocab) t=\(t)")
    }
}
