import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// INT8-2L GEMV against a Double dot product of the exact weights from
/// `GGUFBlockDecode` (the runtime twin the installer parity test locks), at
/// the install's resident row lengths: Q6_K with rows of 2560 and 6144, and
/// Q8_0 with rows of 2560 and 640 (`ffn_down_shexp`). Regions are built the
/// way the installer writes them (values, d, m), at region offsets inside
/// one buffer that are not 16-aligned (the m region starts at an odd byte).
@Suite struct GGUFInt8TwoLevelGEMVTests {
    struct Case: CustomTestStringConvertible, Sendable {
        let q6K: Bool, rows: Int, n: Int
        var testDescription: String { "\(q6K ? "Q6_K" : "Q8_0") \(rows)x\(n)" }
    }
    static let cases = [Case(q6K: true, rows: 512, n: 2560), Case(q6K: true, rows: 256, n: 6144),
                        Case(q6K: false, rows: 512, n: 2560), Case(q6K: false, rows: 2560, n: 640)]

    /// Random Q6_K / Q8_0 blocks with a realistic positive scale, expanded to
    /// INT8-2L regions laid out as the installer writes them, plus the exact
    /// fp32 weights from `GGUFBlockDecode`.
    static func makeWeights(_ c: Case, ctx: MetalContext, rng: inout SplitMix64) throws
        -> (GGUFInt8TwoLevelGEMV.Weights, [Float]) {
        let blockBytes = c.q6K ? GGUFBlockDecode.q6KBlockBytes : GGUFBlockDecode.q8_0BlockBytes
        let blockWeights = c.q6K ? 256 : 32
        let blocks = c.rows * c.n / blockWeights

        var raw = [UInt8](repeating: 0, count: blocks * blockBytes)
        for i in 0..<raw.count { raw[i] = UInt8(truncatingIfNeeded: rng.next()) }
        let dAt = c.q6K ? 208 : 0
        for b in 0..<blocks {  // a realistic positive scale
            let bits = Float16(rng.uniform(1e-4, 2e-3)).bitPattern
            raw[b * blockBytes + dAt] = UInt8(bits & 0xFF)
            raw[b * blockBytes + dAt + 1] = UInt8(bits >> 8)
        }

        var values = [UInt8](), d = [UInt8](), m = [UInt8]()
        let weights: [Float] = raw.withUnsafeBytes { all in
            for b in 0..<blocks {
                let block = UnsafeRawBufferPointer(rebasing: all[(b * blockBytes)..<((b + 1) * blockBytes)])
                let t = c.q6K ? GGUFBlockDecode.q6KToInt8TwoLevel(block) : GGUFBlockDecode.q8_0ToInt8TwoLevel(block)
                values += t.values.map { UInt8(bitPattern: $0) }
                d += [UInt8(t.d & 0xFF), UInt8(t.d >> 8)]
                m += t.m.map { UInt8(bitPattern: $0) }
            }
            return c.q6K ? GGUFBlockDecode.dequantizeQ6K(all) : GGUFBlockDecode.dequantizeQ8_0(all)
        }

        // values | pad 4 | d (2-aligned) | pad 1 | m (odd offset)
        let vAt = 0, dOff = values.count + 4, mOff = dOff + d.count + 1
        var region = values + [UInt8](repeating: 0xEE, count: 4) + d + [0xEE] + m
        region += [UInt8](repeating: 0xEE, count: 16)
        let buf = try #require(region.withUnsafeBytes {
            ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
        let w = GGUFInt8TwoLevelGEMV.Weights(
            source: c.q6K ? .q6_K : .q8_0, rows: c.rows, n: c.n,
            values: (buf, vAt), d: (buf, dOff), m: c.q6K ? (buf, mOff) : nil)
        return (w, weights)
    }

    @Test(arguments: cases)
    func gemvMatchesReference(c: Case) throws {
        let ctx = try MetalContext()
        let kernel = try GGUFInt8TwoLevelGEMV(context: ctx)
        var rng = SeedTree(0x4932_4C47).key("int8-2l-\(c.testDescription)")
        let (w, weights) = try Self.makeWeights(c, ctx: ctx, rng: &rng)

        let x = (0..<c.n).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBuf = try #require(x.withUnsafeBytes {
            ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
        let reference = GGUFInt8TwoLevelReference.gemv(weights: weights, rows: c.rows, n: c.n, x: x)

        for floatOutput in [false, true] {
            let yBuf = try #require(ctx.device.makeBuffer(length: c.rows * 4, options: .storageModeShared))
            memset(yBuf.contents(), 0xFF, c.rows * 4)  // NaN poison
            let cb = try #require(ctx.queue.makeCommandBuffer())
            kernel.encode(commandBuffer: cb, weights: w, x: xBuf, y: yBuf, floatOutput: floatOutput)
            cb.commit(); cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            let got: [Float] = floatOutput
                ? Array(UnsafeBufferPointer(start: yBuf.contents().bindMemory(to: Float.self, capacity: c.rows), count: c.rows))
                : Array(UnsafeBufferPointer(start: yBuf.contents().bindMemory(to: Float16.self, capacity: c.rows), count: c.rows)).map(Float.init)
            let err = RelError.compute(actual: got, reference: reference)
            #expect(err < Tolerance.fp16Reduction, "\(floatOutput ? "f32" : "f16") out rel error \(err)")
        }
    }

    /// The batched prefill form equals the single-token GEMV bit for bit for
    /// every token, including a partial last group of 8, with padded x and y
    /// row strides.
    @Test(arguments: [cases[0], cases[3]])
    func batchedMatchesSingleTokenBitForBit(c: Case) throws {
        let ctx = try MetalContext()
        let kernel = try GGUFInt8TwoLevelGEMV(context: ctx)
        var rng = SeedTree(0x4932_4C42).key("int8-2l-batched-\(c.testDescription)")
        let (w, _) = try Self.makeWeights(c, ctx: ctx, rng: &rng)
        for tokens in [1, 7, 8, 13] {
            let xStride = c.n + 8, yStride = c.rows + 4
            let x = (0..<(tokens * xStride)).map { _ in Float16(rng.uniform(-1, 1)) }
            let xBuf = try #require(x.withUnsafeBytes {
                ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            })
            let yBuf = try #require(ctx.device.makeBuffer(length: tokens * yStride * 2, options: .storageModeShared))
            let single = try #require(ctx.device.makeBuffer(length: tokens * c.rows * 2, options: .storageModeShared))
            memset(yBuf.contents(), 0xFF, tokens * yStride * 2)
            let cb = try #require(ctx.queue.makeCommandBuffer())
            kernel.encodeBatched(commandBuffer: cb, weights: w, x: xBuf, xStride: xStride,
                                 y: yBuf, yStride: yStride, tokens: tokens)
            for t in 0..<tokens {
                kernel.encode(commandBuffer: cb, weights: w, x: xBuf, xOffset: t * xStride * 2,
                              y: single, yOffset: t * c.rows * 2, floatOutput: false)
            }
            cb.commit(); cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            let yb = yBuf.contents().bindMemory(to: UInt16.self, capacity: tokens * yStride)
            let ys = single.contents().bindMemory(to: UInt16.self, capacity: tokens * c.rows)
            var mismatches = 0, padTouched = 0
            for t in 0..<tokens {
                for r in 0..<c.rows where yb[t * yStride + r] != ys[t * c.rows + r] { mismatches += 1 }
                for r in c.rows..<yStride where yb[t * yStride + r] != 0xFFFF { padTouched += 1 }
            }
            #expect(mismatches == 0, "\(tokens) tokens: \(mismatches) rows differ from the single-token GEMV")
            #expect(padTouched == 0, "\(tokens) tokens: wrote into the y row padding")
        }
    }

    /// The FP32-output batched form (the hyper-connection gates' prefill path)
    /// equals `gemv_f32out` per token bit for bit, at the HC shapes too
    /// (Q8_0 10240 → 320 and 320 → 10240).
    @Test(arguments: [cases[0], cases[3], Case(q6K: false, rows: 320, n: 10240), Case(q6K: false, rows: 10240, n: 320)])
    func batchedFloatOutputMatchesSingleTokenBitForBit(c: Case) throws {
        let ctx = try MetalContext()
        let kernel = try GGUFInt8TwoLevelGEMV(context: ctx)
        var rng = SeedTree(0x4932_4C46).key("int8-2l-batched-f32-\(c.testDescription)")
        let (w, _) = try Self.makeWeights(c, ctx: ctx, rng: &rng)
        for tokens in [1, 7, 8, 13] {
            let xStride = c.n + 8, yStride = c.rows + 4
            let x = (0..<(tokens * xStride)).map { _ in Float16(rng.uniform(-1, 1)) }
            let xBuf = try #require(x.withUnsafeBytes {
                ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            })
            let yBuf = try #require(ctx.device.makeBuffer(length: tokens * yStride * 4, options: .storageModeShared))
            let single = try #require(ctx.device.makeBuffer(length: tokens * c.rows * 4, options: .storageModeShared))
            memset(yBuf.contents(), 0xFF, tokens * yStride * 4)
            let cb = try #require(ctx.queue.makeCommandBuffer())
            kernel.encodeBatched(commandBuffer: cb, weights: w, x: xBuf, xStride: xStride,
                                 y: yBuf, yStride: yStride, tokens: tokens, floatOutput: true)
            for t in 0..<tokens {
                kernel.encode(commandBuffer: cb, weights: w, x: xBuf, xOffset: t * xStride * 2,
                              y: single, yOffset: t * c.rows * 4, floatOutput: true)
            }
            cb.commit(); cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            let yb = yBuf.contents().bindMemory(to: UInt32.self, capacity: tokens * yStride)
            let ys = single.contents().bindMemory(to: UInt32.self, capacity: tokens * c.rows)
            var mismatches = 0, padTouched = 0
            for t in 0..<tokens {
                for r in 0..<c.rows where yb[t * yStride + r] != ys[t * c.rows + r] { mismatches += 1 }
                for r in c.rows..<yStride where yb[t * yStride + r] != 0xFFFF_FFFF { padTouched += 1 }
            }
            #expect(mismatches == 0, "\(tokens) tokens: \(mismatches) rows differ from the single-token f32 GEMV")
            #expect(padTouched == 0, "\(tokens) tokens: wrote into the y row padding")
        }
    }
}

/// `token_embd` gather from a native Q6_K table against
/// `GGUFBlockDecode.dequantizeQ6K` rounded to half, bit for bit (both sides
/// round the same exact fp32 value to nearest-even), with repeated, first and
/// last row ids and an untouched tail.
@Suite struct GGUFQ6KEmbedRowsTests {
    @Test func gatherMatchesCPUDecodeBits() throws {
        let ctx = try MetalContext()
        let kernel = try GGUFInt8TwoLevelGEMV(context: ctx)
        var rng = SeedTree(0x5136_4B45).key("q6k-embed")
        let vocab = 37, hidden = 2560, blocksPerRow = hidden / 256
        var table = [UInt8](repeating: 0, count: vocab * blocksPerRow * 210)
        for i in 0..<table.count { table[i] = UInt8(truncatingIfNeeded: rng.next()) }
        for b in 0..<(vocab * blocksPerRow) {
            let bits = Float16(rng.uniform(1e-4, 2e-3)).bitPattern
            table[b * 210 + 208] = UInt8(bits & 0xFF)
            table[b * 210 + 209] = UInt8(bits >> 8)
        }
        let ids: [UInt32] = [0, 36, 5, 5, 17, 0, 23]
        func buffer<T>(_ v: [T]) throws -> MTLBuffer {
            try #require(v.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        let tableBuf = try buffer(table), idsBuf = try buffer(ids)
        let outBuf = try buffer([Float16](repeating: .nan, count: (ids.count + 1) * hidden))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        kernel.encodeQ6KEmbedRows(commandBuffer: cb, table: tableBuf, ids: idsBuf, out: outBuf,
                                  hidden: hidden, tokens: ids.count)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.status == .completed)
        let out = outBuf.contents().bindMemory(to: UInt16.self, capacity: (ids.count + 1) * hidden)
        var mismatches = 0
        for (t, id) in ids.enumerated() {
            let rowBytes = blocksPerRow * 210, at = Int(id) * rowBytes
            let want = table[at..<(at + rowBytes)].withUnsafeBytes { GGUFBlockDecode.dequantizeQ6K($0) }
            for e in 0..<hidden where out[t * hidden + e] != Float16(want[e]).bitPattern { mismatches += 1 }
        }
        #expect(mismatches == 0)
        let tail = (0..<hidden).filter { !Float16(bitPattern: out[ids.count * hidden + $0]).isNaN }.count
        #expect(tail == 0, "wrote past the last token")
    }
}
