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

    @Test(arguments: cases)
    func gemvMatchesReference(c: Case) throws {
        let ctx = try MetalContext()
        let kernel = try GGUFInt8TwoLevelGEMV(context: ctx)
        var rng = SeedTree(0x4932_4C47).key("int8-2l-\(c.testDescription)")
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
}
