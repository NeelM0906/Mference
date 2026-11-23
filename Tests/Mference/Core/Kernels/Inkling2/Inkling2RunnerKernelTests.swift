import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The runner's own kernels against `Inkling2ReferenceOps` at Mini's width
/// (2304) and rank (64): depth mix over five states with three queries and
/// over the two-state message bank with four, the message gate, the Q8_0
/// embedding with its norm, the weighted norm, and the Q8_0 head with the
/// logit softcap.
@Suite("Inkling2RunnerKernels")
struct Inkling2RunnerKernelTests {
    typealias Op = Inkling2ReferenceOps
    let context: MetalContext
    let kernels: Inkling2RunnerKernels
    static let d = 2304

    init() throws {
        context = try MetalContext()
        kernels = try Inkling2RunnerKernels(device: context.device)
    }

    private func buffer(_ values: [Float]) -> MTLBuffer {
        values.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)!
        }
    }

    private func buffer(bytes: [UInt8]) -> MTLBuffer {
        bytes.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)!
        }
    }

    private func floats(_ b: MTLBuffer, offset: Int = 0, count: Int) -> [Float] {
        let p = (b.contents() + offset * 4).bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    private func run(_ body: (MTLComputeCommandEncoder) -> Void) throws {
        let cb = try #require(context.queue.makeCommandBuffer())
        let enc = try #require(cb.makeComputeCommandEncoder())
        body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)
    }

    private func random(_ count: Int, _ seed: String, _ lo: Float = -1, _ hi: Float = 1) -> [Float] {
        var rng = SeedTree(0x5EED).key(seed)
        return (0..<count).map { _ in rng.uniform(lo, hi) }
    }

    private func maxRelative(_ got: [Float], _ want: [Float]) -> Float {
        let scale = max(want.map(abs).max() ?? 0, 1e-6)
        return zip(got, want).map { abs($0 - $1) }.max()! / scale
    }

    @Test func depthMixMatchesTheReference() throws {
        let d = Self.d, tokens = 5, states = 5, queries = 3
        // Five states in one arena with a gap between them, rows d apart.
        let stride = d
        let gap = 17
        var arena: [Float] = []
        var offsets: [Int] = []
        for s in 0..<states {
            offsets.append(arena.count)
            arena += random(tokens * stride, "state\(s)", -3, 3)
            arena += [Float](repeating: 99, count: gap)
        }
        let q = random(queries * d, "queries", -2, 2)
        let out = try #require(context.device.makeBuffer(length: queries * tokens * d * 4, options: .storageModeShared))
        try run { enc in
            kernels.encodeDepthMix(enc, arena: buffer(arena), stateOffsets: offsets, stateStride: stride,
                                   queries: buffer(q), queryCount: queries, out: out,
                                   outStride: d, outQueryStride: tokens * d, d: d, tokens: tokens)
        }
        for t in 0..<tokens {
            let rows = offsets.map { Array(arena[($0 + t * stride)..<($0 + t * stride + d)]) }
            let want = Op.mix(states: rows, queries: (0..<queries).map { Array(q[($0 * d)..<(($0 + 1) * d)]) })
            for k in 0..<queries {
                let got = floats(out, offset: k * tokens * d + t * d, count: d)
                #expect(maxRelative(got, want[k]) < 1e-5, "token \(t) query \(k)")
            }
        }
    }

    @Test func messageBankMixWithFourQueries() throws {
        let d = Self.d, tokens = 3
        let bank0 = random(tokens * d, "bank0"), bank1 = random(tokens * d, "bank1", -4, 4)
        let arena = bank0 + bank1
        let q = random(4 * d, "msgq")
        let out = try #require(context.device.makeBuffer(length: 4 * tokens * d * 4, options: .storageModeShared))
        try run { enc in
            kernels.encodeDepthMix(enc, arena: buffer(arena), stateOffsets: [0, tokens * d], stateStride: d,
                                   queries: buffer(q), queryCount: 4, out: out,
                                   outStride: 4 * d, outQueryStride: d, d: d, tokens: tokens)
        }
        for t in 0..<tokens {
            let want = Op.mix(states: [Array(bank0[(t * d)..<((t + 1) * d)]), Array(bank1[(t * d)..<((t + 1) * d)])],
                              queries: (0..<4).map { Array(q[($0 * d)..<(($0 + 1) * d)]) })
            for k in 0..<4 {
                #expect(maxRelative(floats(out, offset: t * 4 * d + k * d, count: d), want[k]) < 1e-5)
            }
        }
    }

    @Test(arguments: [(64, 37), (8, 5)])
    func messageGateMatchesTheReference(rank: Int, tokens: Int) throws {
        // 37 tokens: two full tiles of 16 and a partial one.
        let d = Self.d
        let x = random(tokens * d, "x\(rank)", -2, 2), m = random(tokens * d, "m\(rank)", -2, 2)
        let down = random(rank * 2 * d, "down\(rank)", -0.05, 0.05)
        let gain = random(d * rank, "gain\(rank)", -0.3, 0.3), forget = random(d * rank, "forget\(rank)", -0.3, 0.3)
        let out = try #require(context.device.makeBuffer(length: tokens * d * 4, options: .storageModeShared))
        let weights = buffer(down + gain + forget)
        try run { enc in
            kernels.encodeMessageGate(enc, x: buffer(x), xStride: d, message: buffer(m), messageStride: d,
                                      down: weights, downOffset: 0,
                                      gain: weights, gainOffset: rank * 2 * d * 4,
                                      forget: weights, forgetOffset: (rank * 2 * d + d * rank) * 4,
                                      rank: rank, out: out, outStride: d, d: d, tokens: tokens)
        }
        let downM = Inkling2DenseMatrix(rows: rank, cols: 2 * d, values: down)
        let gainM = Inkling2DenseMatrix(rows: d, cols: rank, values: gain)
        let forgetM = Inkling2DenseMatrix(rows: d, cols: rank, values: forget)
        for t in Set([0, 1, 15, 16, tokens - 1]) where t < tokens {
            let want = Op.control(Array(x[(t * d)..<((t + 1) * d)]), message: Array(m[(t * d)..<((t + 1) * d)]),
                                  down: downM, gain: gainM, forget: forgetM)
            #expect(maxRelative(floats(out, offset: t * d, count: d), want) < 1e-5, "token \(t)")
        }
    }

    /// Q8_0 rows: per 32 values an FP16 scale then 32 int8; returns the bytes
    /// and the values they decode to.
    private func q8Rows(_ rows: Int, _ d: Int, _ seed: String) -> ([UInt8], [[Float]]) {
        var rng = SeedTree(0x0808).key(seed)
        var bytes: [UInt8] = []
        var values: [[Float]] = []
        for _ in 0..<rows {
            var row: [Float] = []
            for _ in 0..<(d / 32) {
                let scale = Float16(rng.uniform(0.001, 0.02))
                withUnsafeBytes(of: scale.bitPattern.littleEndian) { bytes += $0 }
                for _ in 0..<32 {
                    let q = Int8(truncatingIfNeeded: Int(rng.next() % 255) - 127)
                    bytes.append(UInt8(bitPattern: q))
                    row.append(Float(scale) * Float(q))
                }
            }
            values.append(row)
        }
        return (bytes, values)
    }

    @Test func embeddingDecodesQ8RowsAndNormalizes() throws {
        let d = Self.d, vocab = 50
        let (bytes, rows) = q8Rows(vocab, d, "embed")
        let weight = random(d, "embedNorm", 0.5, 1.5)
        let ids: [UInt32] = [7, 0, 49, 7]
        let idsBuffer = try #require(ids.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        let out = try #require(context.device.makeBuffer(length: ids.count * d * 4, options: .storageModeShared))
        // A two-byte offset into the table buffer, as a GGUF tensor may sit.
        try run { enc in
            kernels.encodeEmbedding(enc, ids: idsBuffer, table: buffer(bytes: [0, 0] + bytes), tableOffset: 2,
                                    rowBytes: d / 32 * 34, weight: buffer(weight), weightOffset: 0, eps: 1e-6,
                                    out: out, outStride: d, d: d, tokens: ids.count)
        }
        for (t, id) in ids.enumerated() {
            let want = Op.rmsNorm(rows[Int(id)], weight: weight, eps: 1e-6)
            #expect(maxRelative(floats(out, offset: t * d, count: d), want) < 1e-5)
        }
    }

    @Test func rmsNormMatchesTheReference() throws {
        let d = Self.d, tokens = 3
        let x = random(tokens * (d + 8), "normx", -5, 5)
        let w = random(d, "normw", 0.5, 1.5)
        let out = try #require(context.device.makeBuffer(length: tokens * d * 4, options: .storageModeShared))
        try run { enc in
            kernels.encodeRMSNorm(enc, x: buffer(x), xStride: d + 8, weight: buffer(w), weightOffset: 0,
                                  eps: 1e-6, out: out, outStride: d, d: d, tokens: tokens)
        }
        for t in 0..<tokens {
            let want = Op.rmsNorm(Array(x[(t * (d + 8))..<(t * (d + 8) + d)]), weight: w, eps: 1e-6)
            #expect(maxRelative(floats(out, offset: t * d, count: d), want) < 1e-5)
        }
    }

    @Test func dualNormWritesMatchingFP32AndFP16Rows() throws {
        let d = Self.d, tokens = 3
        let x = random(tokens * d, "dualx", -5, 5)
        let w = random(d, "dualw", 0.5, 1.5)
        let out = try #require(context.device.makeBuffer(length: tokens * d * 4, options: .storageModeShared))
        let half = try #require(context.device.makeBuffer(length: tokens * (d + 4) * 2, options: .storageModeShared))
        try run { enc in
            kernels.encodeRMSNormDual(enc, x: buffer(x), xStride: d, weight: buffer(w), weightOffset: 0, eps: 1e-6,
                                      out: out, outStride: d, outHalf: half, halfStride: d + 4, d: d, tokens: tokens)
        }
        let h = half.contents().bindMemory(to: Float16.self, capacity: tokens * (d + 4))
        for t in 0..<tokens {
            let want = Op.rmsNorm(Array(x[(t * d)..<((t + 1) * d)]), weight: w, eps: 1e-6)
            let got = floats(out, offset: t * d, count: d)
            #expect(maxRelative(got, want) < 1e-5)
            #expect((0..<d).allSatisfy { Float(h[t * (d + 4) + $0]) == Float(Float16(got[$0])) })
        }
    }

    @Test func headMatchesTheReferenceWithTheSoftcap() throws {
        let d = Self.d, rows = 1_000   // not a multiple of the 32 rows per threadgroup
        let (bytes, table) = q8Rows(rows + 3, d, "head")   // padded rows past `rows`
        let x = random(d, "headx", -1, 1)
        let logits = try #require(context.device.makeBuffer(length: (rows + 3) * 2, options: .storageModeShared))
        memset(logits.contents(), 0x7B, (rows + 3) * 2)
        try run { enc in
            kernels.encodeHead(enc, x: buffer(x), table: buffer(bytes: bytes), tableOffset: 0,
                               rowBytes: d / 32 * 34, rows: rows, softcap: 30, logits: logits, d: d)
        }
        let got = logits.contents().bindMemory(to: Float16.self, capacity: rows + 3)
        var worst: Float = 0
        for r in 0..<rows {
            let want = Op.softcap(Op.dot(table[r], x), 30)
            worst = max(worst, abs(Float(got[r]) - want) / max(abs(want), 1))
        }
        #expect(worst < 2e-3)   // FP16 output rounding
        // Padded rows are not written.
        #expect((rows..<(rows + 3)).allSatisfy { got[$0].bitPattern == 0x7B7B })
    }
}
