import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// GPU time of one 16-expert routed tile at the Unsloth Flash-Next shape
/// (D 2560, F 640) through the LUT4 row-kernel prefill (`GGUFLUT4Prefill`) and
/// the simdgroup grouped GEMM with the LUT4 dequant family, at fixed rows per
/// expert. Four distinct experts fill the 16 tile slots. Env-gated:
/// `MFERENCE_PREFILL_MOE_LUT4_KBENCH=1`. Prints one line per shape and checks
/// that the two kernels' route partials agree within the fp16 reduction
/// tolerance; asserts nothing about speed.
@Suite(.serialized) struct SimdgroupGroupedRoutedMoELUT4Benchmarks {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_PREFILL_MOE_LUT4_KBENCH"] == "1"
    }
    static let d = 2560, f = 640, topK = 8, experts = 16

    @Test(.enabled(if: enabled), arguments: [GGUFLUT4MoETests.mixes[1], GGUFLUT4MoETests.mixes[0]])
    func lut4TileRowVsGrouped(mix: GGUFLUT4MoETests.Mix) throws {
        let ctx = try MetalContext()
        let device = ctx.device
        let grouped = try SimdgroupGroupedRoutedMoE(context: ctx, dequant: .lut4)
        let rowKernel = try GGUFLUT4Prefill(context: ctx)
        var rng = SeedTree(0x4C55_4B42).key("psgm-lut4-kbench-\(mix.gateUp.rawValue)")
        let n = Self.d * Self.f
        var slots = [[UInt8]]()
        var offsets: GGUFLUT4MoE.ExpertOffsets?
        for _ in 0..<4 {
            var slot = [UInt8]()
            func projection(_ t: GGUFLUT4.SourceType) -> (UInt32, UInt32, UInt32) {
                let raw = GGUFLUT4MoETests.blocks(t, weights: n, rng: &rng)
                let parts = raw.withUnsafeBytes { GGUFLUT4Reference.decompose(t, $0) }
                return GGUFLUT4MoETests.append(parts, to: &slot)
            }
            let g = projection(mix.gateUp), u = projection(mix.gateUp), dn = projection(mix.down)
            offsets = GGUFLUT4MoE.ExpertOffsets(gateIndex: g.0, gateD: g.1, gateM: g.2,
                                                upIndex: u.0, upD: u.1, upM: u.2,
                                                downIndex: dn.0, downD: dn.1, downM: dn.2)
            slots.append(slot)
        }
        let o = try #require(offsets)
        let stride = (slots[0].count + 16_383) / 16_384 * 16_384
        let slab = try #require(device.makeBuffer(length: stride * 4, options: .storageModeShared))
        for (i, s) in slots.enumerated() {
            s.withUnsafeBytes { slab.contents().advanced(by: i * stride).copyMemory(from: $0.baseAddress!, byteCount: s.count) }
        }
        let lut4 = [GGUFLUT4MoE.ProjParams(mix.gateUp), GGUFLUT4MoE.ProjParams(mix.gateUp),
                    GGUFLUT4MoE.ProjParams(mix.down)]
        let views = (0..<Self.experts).map { i in
            TensorView(buffer: slab, offset: UInt64((i % 4) * stride), length: UInt64(stride),
                       scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                       shape: (0, UInt32(i), 0, 0), dtype: 0)
        }
        let binding = try PrefillStreamedTileBinding(expertIDs: Array(0..<Self.experts), views: views)
        let metadataKernel = try PrefillGroupedRoutedMoE(context: ctx, siluActivation: true)

        for rowsPerExpert in [4, 8, 16, 32, 64, 128] {
            let tokens = rowsPerExpert * Self.experts / Self.topK
            var pairs: [PrefillTokenExpertPair] = []
            for token in 0..<tokens {
                for rank in 0..<Self.topK {
                    let expert: Int = (token * Self.topK + rank) % Self.experts
                    pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: UInt32(expert),
                                                                   rank: UInt32(rank)))
                }
            }
            let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                pairs, queryCount: tokens, topK: Self.topK, numExperts: Self.experts, tileExpertCount: Self.experts)
            let metadata = try metadataKernel.makeStreamedMetadataBuffers(device: device, routes: routes)
            let hidden = (0..<(tokens * Self.d)).map { _ in Float16(rng.uniform(-1, 1)) }
            let hiddenBuf = try #require(Fp16Buffer.make(device, halves: hidden))
            func buffer(_ halves: Int) throws -> MTLBuffer {
                try #require(device.makeBuffer(length: max(1, halves) * 2, options: .storageModeShared))
            }
            let rowPartials = try buffer(tokens * Self.topK * Self.d)
            let groupedPartials = try buffer(tokens * Self.topK * Self.d)
            let scratch = try buffer(3 * pairs.count * Self.f)
            let downScratch = try buffer(pairs.count * Self.d)
            let activation = try buffer(pairs.count * Self.f)

            var params = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: 0, pairCount: UInt32(pairs.count), d: UInt32(Self.d), routedIntermediate: UInt32(Self.f),
                topK: UInt32(Self.topK), hiddenStrideElements: UInt32(Self.d), binding: binding,
                offsets: MoEExpertOffsets(gateWOff: o.gateIndex, gateSOff: o.gateD, gateBOff: o.gateM,
                                          upWOff: o.upIndex, upSOff: o.upD, upBOff: o.upM,
                                          downWOff: o.downIndex, downSOff: o.downD, downBOff: o.downM))
            params.pairCount = UInt32(routes.groups.count)
            let argument = try grouped.makeArgumentBuffer(device: device, binding: binding)
            let rowParams = GGUFLUT4Prefill.Params(
                pairStart: 0, pairCount: UInt32(pairs.count), d: UInt32(Self.d), f: UInt32(Self.f),
                topK: UInt32(Self.topK), hiddenStrideElements: UInt32(Self.d),
                expertIDs: (0..<Self.experts).map(UInt32.init), offsets: o)

            func best(_ encode: (MTLCommandBuffer) -> Void) throws -> Double {
                var best = Double.infinity
                for _ in 0..<3 {
                    let cb = try #require(ctx.queue.makeCommandBuffer())
                    encode(cb)
                    cb.commit()
                    cb.waitUntilCompleted()
                    try #require(cb.error == nil, "\(String(describing: cb.error))")
                    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
                }
                return best
            }
            let row = try best { cb in
                rowKernel.encode(commandBuffer: cb, hidden: hiddenBuf, sortedPairs: metadata.sortedPairs,
                                 gateUpActScratch: scratch, downScratch: downScratch, routePartials: rowPartials,
                                 blobs: (0..<Self.experts).map { (buffer: slab, offset: ($0 % 4) * stride) },
                                 params: rowParams, gateUp: mix.gateUp, downType: mix.down)
            }
            let simd = try best { cb in
                _ = grouped.encode(commandBuffer: cb, hidden: hiddenBuf, sortedPairs: metadata.sortedPairs,
                                   groups: metadata.groups, activation: activation, routePartials: groupedPartials,
                                   argumentBuffer: argument, binding: binding, params: params,
                                   maxPairsPerGroup: routes.maxPairsPerExpert, lut4: lut4)
            }
            let count = tokens * Self.topK * Self.d
            let a = Fp16Buffer.readHalf(rowPartials, count: count).map(Float.init)
            let b = Fp16Buffer.readHalf(groupedPartials, count: count).map(Float.init)
            let rel = RelError.compute(actual: b, reference: a)
            print(String(format: "[lut4-kbench] %@ rows/expert=%3d row %.2f ms simd %.2f ms (%.2fx) rel %.1e",
                         mix.testDescription, rowsPerExpert, row * 1000, simd * 1000, row / simd, rel))
            #expect(rel < Tolerance.fp16Reduction, "rows/expert \(rowsPerExpert): rel \(rel)")
        }
    }
}
