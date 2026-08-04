import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// `dequant_int4_gemv_simd_f32out` exists because Inkling's shared-expert down
/// projection produces rows that leave FP16 range *before* the routing weight
/// or shared gamma that brings them back. On the released 4-bit checkpoint,
/// layer 41 channel 3895 sits at 1.5e4-6e4 on every token, and the token that
/// pushed it to 69 307 clipped to `inf`, poisoned the FP32 residual, and made
/// the whole logit row NaN — which the greedy argmax reported as token 0, `!`.
///
/// Two properties are locked here: the FP32 store is faithful to the FP16 one
/// wherever FP16 can represent the row, and it does *not* clip where FP16
/// would.
@Suite struct DequantInt4GEMVF32OutTests {

    private struct Fixture {
        var rows: [Quantization.Int4AffineRow]
        var weights: MTLBuffer
        var scales: MTLBuffer
        var biases: MTLBuffer
        var x: MTLBuffer
        var xRef: [Float]
        var m: Int
        var n: Int
    }

    /// `weightRange`/`xRange` are drawn uniformly. Signed ranges cancel and
    /// keep rows small; all-positive ranges make every row a large sum, which
    /// is how the overflow case is built.
    private static func makeFixture(m: Int, n: Int, key: String,
                                    weightRange: (Float, Float) = (-0.5, 0.5),
                                    xRange: (Float, Float) = (-1, 1)) throws
        -> Fixture
    {
        precondition(n % Quantization.groupSize == 0)
        var rng = SeedTree(0x1AB_4F32).key(key)
        var rows: [Quantization.Int4AffineRow] = []
        rows.reserveCapacity(m)
        for _ in 0..<m {
            let raw = (0..<n).map { _ in
                rng.uniform(weightRange.0, weightRange.1)
            }
            rows.append(Quantization.quantizeInt4Affine(raw))
        }
        let n2 = rows[0].packed.count
        let s = rows[0].scales.count
        var packed = [UInt8](repeating: 0, count: m * n2)
        var scales = [UInt16](repeating: 0, count: m * s)
        var biases = [UInt16](repeating: 0, count: m * s)
        for row in 0..<m {
            for i in 0..<n2 { packed[row * n2 + i] = rows[row].packed[i] }
            for i in 0..<s {
                scales[row * s + i] = rows[row].scales[i]
                biases[row * s + i] = rows[row].biases[i]
            }
        }
        let ctx = try MetalContext()
        let xFp16 = (0..<n).map { _ in Float16(rng.uniform(xRange.0, xRange.1)) }
        return Fixture(
            rows: rows,
            weights: ctx.device.makeBuffer(bytes: packed, length: packed.count,
                                           options: .storageModeShared)!,
            scales: ctx.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared)!,
            biases: ctx.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared)!,
            x: Fp16Buffer.make(ctx.device, halves: xFp16)!,
            xRef: xFp16.map { Float($0) },
            m: m, n: n)
    }

    private static func runBoth(_ f: Fixture)
        throws -> (half: [Float], float: [Float])
    {
        let ctx = try MetalContext()
        let kernel = try DequantInt4GEMV(context: ctx)
        let yHalf = Fp16Buffer.make(ctx.device, count: f.m)!
        let yFloat = ctx.device.makeBuffer(
            length: f.m * MemoryLayout<Float>.size,
            options: .storageModeShared)!
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb,
                      weights: f.weights, scales: f.scales, biases: f.biases,
                      x: f.x, y: yHalf,
                      m: UInt32(f.m), n: UInt32(f.n))
        kernel.encode(commandBuffer: cb,
                      weights: f.weights, scales: f.scales, biases: f.biases,
                      x: f.x, y: yFloat,
                      m: UInt32(f.m), n: UInt32(f.n),
                      outputFloat32: true)
        cb.commit()
        cb.waitUntilCompleted()
        let fp = yFloat.contents().bindMemory(to: Float.self, capacity: f.m)
        return (Fp16Buffer.read(yHalf, count: f.m), (0..<f.m).map { fp[$0] })
    }

    /// In-range rows: narrowing the FP32 store reproduces the FP16 store
    /// exactly, i.e. only the store width differs — the accumulation is
    /// untouched.
    @Test func f32OutRoundsToTheF16KernelInRange() throws {
        let f = try Self.makeFixture(m: 256, n: 2816, key: "in-range")
        let (half, float) = try Self.runBoth(f)
        for i in 0..<f.m {
            // One interpolated literal, not a `+` concatenation: `#expect`'s
            // comment argument is a `Comment`, which is expressible by string
            // literal and interpolation but not by a `String` expression.
            #expect(Float(Float16(float[i])) == half[i],
                    "row \(i): f32=\(float[i]) narrows to \(Float16(float[i])) but the f16 kernel wrote \(half[i])")
        }
        let ref = DequantInt4GemvRef.apply(weightRows: f.rows, x: f.xRef, n: f.n)
        let rel = RelError.compute(actual: float, reference: ref)
        #expect(rel < Tolerance.fp16Reduction, "rel=\(rel)")
    }

    /// The regression proper: all-positive weights and inputs make every row a
    /// sum well past FP16's 65 504. The FP16 kernel saturates to infinity —
    /// the Inkling layer-41 failure in miniature — and the FP32 kernel must
    /// stay finite and track the reference.
    @Test func f32OutDoesNotClipWhereF16Saturates() throws {
        let f = try Self.makeFixture(m: 128, n: 2816, key: "overflow",
                                     weightRange: (4, 8), xRange: (20, 40))
        let (half, float) = try Self.runBoth(f)
        #expect(half.allSatisfy { !$0.isFinite },
                "fixture must overflow FP16 to be a regression test; f16 rows were \(half.prefix(4)), f32 rows \(float.prefix(4))")
        #expect(float.allSatisfy { $0.isFinite },
                "FP32 output must not produce non-finite rows: \(float.prefix(4))")
        #expect(float.allSatisfy { $0 > 65_504 },
                "rows should land above the FP16 ceiling: \(float.prefix(4))")
        let ref = DequantInt4GemvRef.apply(weightRows: f.rows, x: f.xRef, n: f.n)
        let rel = RelError.compute(actual: float, reference: ref)
        #expect(rel < Tolerance.fp16Reduction, "rel=\(rel)")
    }
}
