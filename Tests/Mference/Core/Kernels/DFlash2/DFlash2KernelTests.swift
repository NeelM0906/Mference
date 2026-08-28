import Foundation
import Metal
import Testing

@testable import Mference
import MferenceValidationSupport

/// CPU-reference parity for every DFlash2 kernel on random data — the
/// repo's standard bring-up gate before any kernel touches real weights.
@Suite struct DFlash2KernelTests {

    private static func bf16(_ v: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: v.bitPattern >> 16)
    }
    private static func bf16ToFloat(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }
    private static func buffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
        values.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                              options: .storageModeShared)!
        }
    }
    private static func run(_ context: MetalContext, _ body: (MTLCommandBuffer) -> Void) {
        let cb = context.queue.makeCommandBuffer()!
        body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
    }
    private static func halves(_ buf: MTLBuffer, _ count: Int) -> [Float] {
        let p = buf.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(p[$0]) }
    }

    @Test func bf16GemvMultixMatchesCPU() throws {
        let context = try MetalContext()
        let kernels = try DFlash2Kernels(context: context)
        var rng = SplitMix64(seed: 0xDF001)
        let M = 96, N = 160, T = 5
        let wF = (0..<M * N).map { _ in rng.uniform(0, 1) * 2 - 1 }
        let xF = (0..<T * N).map { _ in rng.uniform(0, 1) * 2 - 1 }
        let w = Self.buffer(context.device, wF.map(Self.bf16))
        let x = Self.buffer(context.device, xF.map { Float16($0) })
        let y = context.device.makeBuffer(length: T * M * 2, options: .storageModeShared)!
        Self.run(context) { cb in
            kernels.encodeGEMV(commandBuffer: cb, weights: w, weightsOffset: 0,
                               x: x, y: y, m: M, n: N, tokens: T)
        }
        let got = Self.halves(y, T * M)
        for t in 0..<T {
            for m in 0..<M {
                var acc: Float = 0
                for i in 0..<N {
                    acc += Self.bf16ToFloat(Self.bf16(wF[m * N + i]))
                        * Float(Float16(xF[t * N + i]))
                }
                #expect(abs(got[t * M + m] - acc) < 2e-2,
                        "t=\(t) m=\(m) got \(got[t * M + m]) want \(acc)")
            }
        }
    }

    @Test func dynConvMatchesCPU() throws {
        let context = try MetalContext()
        let kernels = try DFlash2Kernels(context: context)
        var rng = SplitMix64(seed: 0xDF002)
        let T = 6, H = 64, K = 2, groupSize = 16
        let G = H / groupSize
        let xF = (0..<T * H).map { _ in rng.uniform(0, 1) * 2 - 1 }
        let dynF = (0..<T * 2 * K * G).map { _ in rng.uniform(0, 1) * 2 - 1 }
        let baseF = (0..<2 * K * H).map { _ in rng.uniform(0, 1) * 2 - 1 }
        let x = Self.buffer(context.device, xF.map { Float16($0) })
        let dyn = Self.buffer(context.device, dynF.map { Float16($0) })
        let base = Self.buffer(context.device, baseF.map(Self.bf16))
        let out = context.device.makeBuffer(length: T * H * 2, options: .storageModeShared)!

        for plane in 0..<2 {
            Self.run(context) { cb in
                kernels.encodeDynConv(commandBuffer: cb, x: x, dynamic: dyn,
                                      base: base, baseOffset: plane * K * H * 2,
                                      out: out, tokens: T, hidden: H,
                                      kernelSize: K, groupSize: groupSize,
                                      plane: plane)
            }
            let got = Self.halves(out, T * H)
            for i in 0..<T {
                for c in 0..<H {
                    let g = c / groupSize
                    var acc: Float = 0
                    for t in 0..<K where t <= i {
                        let baseV = Self.bf16ToFloat(
                            Self.bf16(baseF[plane * K * H + t * H + c]))
                        let dynV = Float(Float16(
                            dynF[i * 2 * K * G + plane * K * G + t * G + g]))
                        let xv = Float(Float16(xF[(i - t) * H + c]))
                        acc += (baseV + dynV) * xv
                    }
                    #expect(abs(got[i * H + c] - acc) < 2e-2,
                            "plane=\(plane) i=\(i) c=\(c)")
                }
            }
        }
    }

    @Test func blockAttentionMatchesCPU() throws {
        let context = try MetalContext()
        let kernels = try DFlash2Kernels(context: context)
        var rng = SplitMix64(seed: 0xDF003)
        let T = 4, ctxLen = 9, window = 8
        let HQ = 4, HKV = 2, HD = 128
        let scale: Float = 1.0 / Float(HD).squareRoot()
        let qF = (0..<T * HQ * HD).map { _ in rng.uniform(0, 1) * 2 - 1 }
        let ckF = (0..<ctxLen * HKV * HD).map { _ in rng.uniform(0, 1) * 2 - 1 }
        let cvF = (0..<ctxLen * HKV * HD).map { _ in rng.uniform(0, 1) * 2 - 1 }
        let bkF = (0..<T * HKV * HD).map { _ in rng.uniform(0, 1) * 2 - 1 }
        let bvF = (0..<T * HKV * HD).map { _ in rng.uniform(0, 1) * 2 - 1 }
        let q = Self.buffer(context.device, qF.map { Float16($0) })
        let ck = Self.buffer(context.device, ckF.map { Float16($0) })
        let cv = Self.buffer(context.device, cvF.map { Float16($0) })
        let bk = Self.buffer(context.device, bkF.map { Float16($0) })
        let bv = Self.buffer(context.device, bvF.map { Float16($0) })
        let out = context.device.makeBuffer(length: T * HQ * HD * 2,
                                            options: .storageModeShared)!
        Self.run(context) { cb in
            kernels.encodeBlockAttention(commandBuffer: cb, q: q,
                                         ctxK: ck, ctxV: cv,
                                         blkK: bk, blkV: bv, out: out,
                                         tokens: T, ctxLen: ctxLen,
                                         window: window,
                                         numQHeads: HQ, numKVHeads: HKV,
                                         scale: scale)
        }
        let got = Self.halves(out, T * HQ * HD)
        for t in 0..<T {
            for h in 0..<HQ {
                let kvh = h / (HQ / HKV)
                var scores: [Float] = []
                var rows: [[Float]] = []
                for j in 0..<(ctxLen + T) {
                    let kRow: [Float]
                    let vRow: [Float]
                    if j < ctxLen {
                        if (ctxLen + t) - j >= window { continue }
                        kRow = (0..<HD).map { Float(Float16(ckF[(j * HKV + kvh) * HD + $0])) }
                        vRow = (0..<HD).map { Float(Float16(cvF[(j * HKV + kvh) * HD + $0])) }
                    } else {
                        kRow = (0..<HD).map { Float(Float16(bkF[((j - ctxLen) * HKV + kvh) * HD + $0])) }
                        vRow = (0..<HD).map { Float(Float16(bvF[((j - ctxLen) * HKV + kvh) * HD + $0])) }
                    }
                    var dot: Float = 0
                    for e in 0..<HD {
                        dot += Float(Float16(qF[(t * HQ + h) * HD + e])) * kRow[e]
                    }
                    scores.append(dot * scale)
                    rows.append(vRow)
                }
                let mx = scores.max()!
                let weights = scores.map { expf($0 - mx) }
                let denom = weights.reduce(0, +)
                for e in stride(from: 0, to: HD, by: 17) {
                    var expect: Float = 0
                    for (wi, w) in weights.enumerated() { expect += w * rows[wi][e] }
                    expect /= denom
                    let g = got[(t * HQ + h) * HD + e]
                    #expect(abs(g - expect) < 3e-2,
                            "t=\(t) h=\(h) e=\(e) got \(g) want \(expect)")
                }
            }
        }
    }

    @Test func topK16MatchesCPU() throws {
        let context = try MetalContext()
        let kernels = try DFlash2Kernels(context: context)
        var rng = SplitMix64(seed: 0xDF004)
        let T = 3, V = 5000
        let logitsF = (0..<T * V).map { _ in rng.uniform(0, 1) * 8 - 4 }
        let logits = Self.buffer(context.device, logitsF.map { Float16($0) })
        let outIdx = context.device.makeBuffer(length: T * 16 * 4,
                                               options: .storageModeShared)!
        let outVal = context.device.makeBuffer(length: T * 16 * 4,
                                               options: .storageModeShared)!
        Self.run(context) { cb in
            kernels.encodeTopK16(commandBuffer: cb, logits: logits,
                                 outIndices: outIdx, outValues: outVal,
                                 rows: T, vocab: V)
        }
        let idx = outIdx.contents().bindMemory(to: UInt32.self, capacity: T * 16)
        for t in 0..<T {
            let row = (0..<V).map { Float(Float16(logitsF[t * V + $0])) }
            let expected = Set(row.indices.sorted { row[$0] > row[$1] }.prefix(16))
            let gotSet = Set((0..<16).map { Int(idx[t * 16 + $0]) })
            // fp16 ties at the boundary make the 16th slot ambiguous; require
            // the sets to agree except where values are exactly tied.
            let missing = expected.subtracting(gotSet)
            for m in missing {
                let boundary = row[Array(expected).min(by: { row[$0] < row[$1] })!]
                #expect(row[m] == boundary,
                        "row \(t): missing non-tied candidate \(m)")
            }
        }
    }
}
