import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The NVFP4 expert decode (`inkling2_moe_decode_gate_up` / `_down`), which
/// turns codes into FP16 bit patterns two at a time and walks each lane's
/// 32-weight groups two per step, against the CPU reference
/// (`Inkling2ExpertRef.routedSum`). Shapes whose group counts leave lanes
/// with odd and unequal numbers of groups, so the paired loop and its tail
/// both run; random blocks with scale bytes over all 256 values.
@Suite struct Inkling2NVFP4DecodeTests {
    private static func check(shape: Inkling2MoEShape, seed: UInt64) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        // Scale bytes up to 480 times codes up to 6: per-expert scales near
        // 1e-4, as the file's, keep the products O(1).
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: .nvfp4, shape: shape,
                                                       seed: seed, quantized: false, fullScaleRange: true,
                                                       expertScales: 5e-5...2e-4)
        let (x, rows) = Inkling2MoETestSupport.halfBuffer(context.device, rows: 1, stride: shape.d, d: shape.d,
                                                          seed: seed &+ 1)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: 1, experts: shape.experts, topK: shape.topK,
                                                         seed: seed &+ 2)
        let ids = context.device.makeBuffer(length: shape.topK * 4, options: .storageModeShared)!
        let weights = context.device.makeBuffer(length: shape.topK * 4, options: .storageModeShared)!
        Inkling2MoETestSupport.write(routes.ids, to: ids)
        Inkling2MoETestSupport.write(routes.weights, to: weights)
        let activation = context.device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
        let out = context.device.makeBuffer(length: shape.d * 4, options: .storageModeShared)!
        let cb = Inkling2MoETestSupport.run(context) {
            kernels.encodeDecode($0, shape: shape, x: x, ids: ids, routeWeights: weights, pool: synthetic.pool,
                                 activation: activation, out: out)
        }
        #expect(cb.status == .completed)
        let got = Inkling2MoETestSupport.read(out, as: Float.self, count: shape.d)
        let experts = (0..<shape.experts).map { synthetic.expert($0) }
        let want = Inkling2ExpertRef.routedSum(experts: { experts[$0] }, ids: routes.ids.map { Int($0) },
                                               weights: routes.weights, x: rows[0])
        let finite = got.allSatisfy { $0.isFinite }
        #expect(finite)
        let err = RelError.computeStrict(actual: got, reference: want)
        #expect(err < Tolerance.fp16Reduction, "rel err \(err)")
        let l1 = zip(got, want).reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / want.reduce(Float(0)) { $0 + abs($1) }
        #expect(l1 < Tolerance.fp16Reduction, "L1 rel err \(l1)")
        print(String(format: "[ink2-nvfp4-decode] d %d f %d experts %d top-%d: rel err %.2e, L1 %.2e",
                     shape.d, shape.f, shape.experts, shape.topK, err, l1))
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
}
