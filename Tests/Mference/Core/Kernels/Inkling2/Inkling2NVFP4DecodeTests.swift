import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The NVFP4 expert decode (`inkling2_moe_decode_gate_up` / `_down`) against
/// the CPU reference (`Inkling2ExpertRef.routedSum`), in both of its paths:
/// the table decode always, and the pair decode (codes as FP16 bit patterns
/// two at a time, lanes walking their 32-weight groups two per step) where
/// this GPU's probe passes, with the two paths' outputs equal to the bit.
/// Cases:
/// - shapes whose group counts leave lanes with odd and unequal numbers of
///   groups, so the paired loop and its tail both run (random blocks, scale
///   bytes drawn from all 256 values, where the largest scales carry the
///   sums);
/// - each class of scale byte alone in a pool (subnormal, low and high
///   exponents, the zero bytes), so a decode that flushed FP16 subnormals or
///   misread an exponent range shows in the output.
@Suite struct Inkling2NVFP4DecodeTests {
    private static func check(shape: Inkling2MoEShape, seed: UInt64) throws {
        let context = try MetalContext()
        // Scale bytes up to 480 times codes up to 6: per-expert scales near
        // 1e-4, as the file's, keep the products O(1).
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: .nvfp4, shape: shape,
                                                       seed: seed, quantized: false, fullScaleRange: true,
                                                       expertScales: 5e-5...2e-4)
        try check(context, synthetic, seed: seed, label: "all scale bytes")
    }

    private static func check(_ context: MetalContext, _ synthetic: Inkling2SyntheticPool, seed: UInt64,
                              label: String) throws {
        let shape = synthetic.shape
        let probed = try Inkling2MoEKernels(context: context)
        var paths: [(Inkling2MoEKernels.NVFP4Decode, Inkling2MoEKernels)] =
            [(.table, try Inkling2MoEKernels(context: context, nvfp4Decode: .table))]
        if probed.nvfp4Decode == .pairs { paths.append((.pairs, probed)) }
        let (x, rows) = Inkling2MoETestSupport.halfBuffer(context.device, rows: 1, stride: shape.d, d: shape.d,
                                                          seed: seed &+ 1)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: 1, experts: shape.experts, topK: shape.topK,
                                                         seed: seed &+ 2)
        let ids = context.device.makeBuffer(length: shape.topK * 4, options: .storageModeShared)!
        let weights = context.device.makeBuffer(length: shape.topK * 4, options: .storageModeShared)!
        Inkling2MoETestSupport.write(routes.ids, to: ids)
        Inkling2MoETestSupport.write(routes.weights, to: weights)
        let experts = (0..<shape.experts).map { synthetic.expert($0) }
        let want = Inkling2ExpertRef.routedSum(experts: { experts[$0] }, ids: routes.ids.map { Int($0) },
                                               weights: routes.weights, x: rows[0])
        var bits: [(activation: [UInt16], out: [UInt32])] = []
        for (path, kernels) in paths {
            let activation = context.device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
            let out = context.device.makeBuffer(length: shape.d * 4, options: .storageModeShared)!
            let cb = Inkling2MoETestSupport.run(context) {
                kernels.encodeDecode($0, shape: shape, x: x, ids: ids, routeWeights: weights, pool: synthetic.pool,
                                     activation: activation, out: out)
            }
            #expect(cb.status == .completed)
            let got = Inkling2MoETestSupport.read(out, as: Float.self, count: shape.d)
            let finite = got.allSatisfy { $0.isFinite }
            #expect(finite, "\(path)")
            let err = RelError.computeStrict(actual: got, reference: want)
            #expect(err < Tolerance.fp16Reduction, "\(path) rel err \(err)")
            let l1 = zip(got, want).reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / want.reduce(Float(0)) { $0 + abs($1) }
            #expect(l1 < Tolerance.fp16Reduction, "\(path) L1 rel err \(l1)")
            print(String(format: "[ink2-nvfp4-decode] %@, %@ decode, d %d f %d experts %d top-%d: rel err %.2e, L1 %.2e",
                         label, "\(path)", shape.d, shape.f, shape.experts, shape.topK, err, l1))
            bits.append((Inkling2MoETestSupport.read(activation, as: UInt16.self, count: shape.topK * shape.f),
                         got.map(\.bitPattern)))
        }
        if bits.count == 2 {
            #expect(bits[0].activation == bits[1].activation, "the pair and table decodes' activations differ")
            #expect(bits[0].out == bits[1].out, "the pair and table decodes' outputs differ")
        }
    }

    /// UE4M3 as the kernels read it (`i2_ue4m3`): bit 7 ignored, the byte 0x7F zero.
    private static func ue4m3(_ byte: UInt8) -> Float {
        if byte == 0 || byte == 0x7F { return 0 }
        let e = Int(byte >> 3) & 15, m = Float(byte & 7)
        return e == 0 ? m * 0.001953125 : Float(sign: .plus, exponent: e - 7, significand: 1 + m / 8)
    }

    /// The device's cached answer is a fresh probe's; a probe that cannot run
    /// (no pipeline) selects the table decode and says why.
    @Test func probeIsCachedAndFallsBackWhenItCannotRun() throws {
        let context = try MetalContext()
        let library = try Inkling2ShaderLibrary.compile(device: context.device, module: Inkling2MoEKernels.module)
        let function = try #require(library.makeFunction(name: "inkling2_nvfp4_decode_probe"))
        let pipeline = try context.device.makeComputePipelineState(function: function)
        let fresh = Inkling2MoEKernels.runNVFP4Probe(device: context.device, pipeline: pipeline)
        let kernels = try Inkling2MoEKernels(context: context)
        #expect(kernels.nvfp4Decode == fresh.0)
        #expect(Inkling2MoEKernels.probedNVFP4Decode(device: context.device) { nil } == fresh.0)
        #expect((fresh.0 == .pairs) == (fresh.failure == nil))
        let unrunnable = Inkling2MoEKernels.runNVFP4Probe(device: context.device, pipeline: nil)
        #expect(unrunnable.0 == .table && unrunnable.failure != nil)
        print("[ink2-nvfp4-decode] probe on \(context.device.name): \(fresh.0) decode\(fresh.failure.map { " (\($0))" } ?? "")")
    }

    /// The probe's check: the exact values pass; a flushed code 1 or 9 (0.5
    /// as the FP16 subnormal 2^-15), a flushed subnormal scale byte (0x01,
    /// 0x87), a NaN or a short read select the table decode.
    @Test func probeRejectsFlushedSubnormals() {
        let exact = Inkling2MoEKernels.nvfp4ProbeExpected
        #expect(exact.count == 16 + 256)
        #expect(exact[1] == 16 * 0.5 / 16384 && exact[16 + 0x01] == 0.001953125 * 16384)
        #expect(Inkling2MoEKernels.nvfp4ProbeIsExact(exact))
        for index in [1, 9, 16 + 0x01, 16 + 0x87] {
            var flushed = exact
            flushed[index] = 0
            #expect(!Inkling2MoEKernels.nvfp4ProbeIsExact(flushed), "flushed output \(index) passed")
        }
        var nan = exact
        nan[100] = .nan
        #expect(!Inkling2MoEKernels.nvfp4ProbeIsExact(nan))
        #expect(!Inkling2MoEKernels.nvfp4ProbeIsExact(Array(exact.dropLast())))
    }

    /// d 384: 12 groups, three per lane (one pair and the tail). f 320: 10
    /// groups, three for lanes 0-1 and two for lanes 2-3.
    @Test func oddGroupCountsMatchReference() throws {
        try Self.check(shape: Inkling2MoEShape(d: 384, f: 320, experts: 12, topK: 8), seed: 0x3B01)
    }

    /// Inkling 2 Mini's d 2304 (72 groups) and f 1152 (36 groups, nine per
    /// lane), eight experts.
    @Test(.fullTier) func productionShapeMatchesReference() throws {
        try Self.check(shape: Inkling2MoEShape(d: 2304, f: 1152, experts: 8, topK: 8), seed: 0x3B02)
    }

    /// Every block's four scale bytes from one class, either sign bit, with
    /// per-expert scales that keep the gate and up inputs near 1 (out of SITU
    /// saturation), so that class alone carries the output.
    static let scaleClasses: [(name: String, bytes: [Int])] = {
        let subnormal: [Int] = Array(0x01...0x07) + Array(0x81...0x87)
        let low: [Int] = Array(0x08...0x37) + Array(0x88...0xB7)
        let high: [Int] = Array(0x38...0x7E) + Array(0xB8...0xFF)
        let zeros: [Int] = [0x00, 0x7F, 0x80, 0x38, 0xB8]
        return [("subnormal", subnormal), ("low exponents", low), ("high exponents", high),
                ("zero bytes among ones", zeros)]
    }()

    @Test(arguments: Self.scaleClasses)
    func scaleByteClassMatchesReference(_ c: (name: String, bytes: [Int])) throws {
        let context = try MetalContext()
        let shape = Inkling2MoEShape(d: 384, f: 320, experts: 12, topK: 8)
        var rng = SeedTree(0x3B03).key("scale-class-\(c.name)")
        let blobBytes = shape.expertBytes(.nvfp4)
        let expertStride = (blobBytes + 15) / 16 * 16
        let buffer = try #require(context.device.makeBuffer(length: expertStride * shape.experts,
                                                            options: .storageModeShared))
        var blobs: [[UInt8]] = []
        for e in 0..<shape.experts {
            var blob = [UInt8](repeating: 0, count: blobBytes)
            for block in stride(from: 0, to: blobBytes, by: 36) {
                for j in 0..<4 { blob[block + j] = UInt8(c.bytes[Int(rng.next() % UInt64(c.bytes.count))]) }
                for j in 4..<36 { blob[block + j] = UInt8(truncatingIfNeeded: rng.next()) }
            }
            blob.withUnsafeBytes {
                (buffer.contents() + e * expertStride).copyMemory(from: $0.baseAddress!, byteCount: blobBytes)
            }
            blobs.append(blob)
        }
        // |gate| ~ sqrt(k) x rms code (2.93) x rms x (1.15) x rms scale x the expert scale.
        let values = c.bytes.map { Self.ue4m3(UInt8($0)) }
        let rms = (values.map { $0 * $0 }.reduce(0, +) / Float(values.count)).squareRoot()
        let scales = [Float](repeating: 1 / (Float(shape.d).squareRoot() * 3.4 * rms), count: shape.experts)
            + [Float](repeating: 1 / (Float(shape.f).squareRoot() * 3.4 * rms), count: shape.experts)
        let scaleBuffer = try #require(scales.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
        let pool = try Inkling2ExpertPool(buffer: buffer, encoding: .nvfp4, shape: shape, expertStride: expertStride,
                                          gateUpScales: scaleBuffer, gateUpScalesOffset: 0,
                                          downScales: scaleBuffer, downScalesOffset: shape.experts * 4)
        let synthetic = Inkling2SyntheticPool(encoding: .nvfp4, shape: shape, blobs: blobs, scales: scales, pool: pool)
        try Self.check(context, synthetic, seed: 0x3B04, label: c.name)
    }
}
