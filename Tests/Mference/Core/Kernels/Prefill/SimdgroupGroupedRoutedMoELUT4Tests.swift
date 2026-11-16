import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// `SimdgroupGroupedRoutedMoE` with the LUT4 dequant family against
/// `GGUFLUT4MoEReference` at the Unsloth Flash-Next shape (D 2560, F 640), for
/// each gate/up + down type mix the LUT4 kernels are tested with. Three
/// experts in one tile: a 65-row group (a full 64-row block and one row), and
/// two one-row groups. The down check consumes the GPU's own activations, as
/// `GGUFLUT4PrefillTests` does.
@Suite struct SimdgroupGroupedRoutedMoELUT4Tests {
    static let d = 2560, f = 640, topK = 1

    @Test(arguments: GGUFLUT4MoETests.mixes)
    func lut4TileMatchesReference(mix: GGUFLUT4MoETests.Mix) throws {
        let ctx = try MetalContext()
        let device = ctx.device
        #expect(SimdgroupGroupedRoutedMoE.supportsLUT4(d: Self.d, routedIntermediate: Self.f))
        let kernel = try SimdgroupGroupedRoutedMoE(context: ctx, dequant: .lut4)
        var rng = SeedTree(0x5053_474D).key("psgm-lut4-\(mix.gateUp.rawValue)-\(mix.down.rawValue)")
        let n = Self.d * Self.f

        var slots = [[UInt8]](), experts = [GGUFLUT4MoEReference.Expert]()
        var offsets: GGUFLUT4MoE.ExpertOffsets?
        for _ in 0..<3 {
            var slot = [UInt8]()
            func projection(_ t: GGUFLUT4.SourceType) -> ((UInt32, UInt32, UInt32), [Float]) {
                let raw = GGUFLUT4MoETests.blocks(t, weights: n, rng: &rng)
                let parts = raw.withUnsafeBytes { GGUFLUT4Reference.decompose(t, $0) }
                return (GGUFLUT4MoETests.append(parts, to: &slot), GGUFLUT4Reference.dequantize(t, parts))
            }
            let (g, gw) = projection(mix.gateUp), (u, uw) = projection(mix.gateUp), (dn, dw) = projection(mix.down)
            offsets = GGUFLUT4MoE.ExpertOffsets(gateIndex: g.0, gateD: g.1, gateM: g.2,
                                                upIndex: u.0, upD: u.1, upM: u.2,
                                                downIndex: dn.0, downD: dn.1, downM: dn.2)
            slots.append(slot)
            experts.append(GGUFLUT4MoEReference.Expert(gate: gw, up: uw, down: dw))
        }
        let o = try #require(offsets)
        let stride = (slots[0].count + 16_383) / 16_384 * 16_384
        let slab = try #require(device.makeBuffer(length: stride * 3, options: .storageModeShared))
        for (i, s) in slots.enumerated() {
            s.withUnsafeBytes { slab.contents().advanced(by: i * stride).copyMemory(from: $0.baseAddress!, byteCount: s.count) }
        }

        // 67 tokens, top-1: tokens 0..<65 to expert 2, token 65 to expert 0, token 66 to expert 1.
        let tokens = 67
        var pairs: [PrefillTokenExpertPair] = []
        for token in 0..<tokens {
            let expert: UInt32 = token < 65 ? 2 : (token == 65 ? 0 : 1)
            pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: expert, rank: 0))
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: tokens, topK: Self.topK, numExperts: 3, tileExpertCount: 3)
        #expect(routes.groups.map(\.pairCount) == [1, 1, 65])
        let rowKernel = try PrefillGroupedRoutedMoE(context: ctx, siluActivation: true)
        let metadata = try rowKernel.makeStreamedMetadataBuffers(device: device, routes: routes)
        let views = (0..<3).map { i in
            TensorView(buffer: slab, offset: UInt64(i * stride), length: UInt64(stride),
                       scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                       shape: (0, UInt32(i), 0, 0), dtype: 0)
        }
        let binding = try PrefillStreamedTileBinding(expertIDs: [0, 1, 2], views: views)
        var params = PrefillGroupedRoutedMoEStreamedParams(
            pairStart: 0, pairCount: UInt32(pairs.count), d: UInt32(Self.d), routedIntermediate: UInt32(Self.f),
            topK: UInt32(Self.topK), hiddenStrideElements: UInt32(Self.d), binding: binding,
            offsets: MoEExpertOffsets(gateWOff: o.gateIndex, gateSOff: o.gateD, gateBOff: o.gateM,
                                      upWOff: o.upIndex, upSOff: o.upD, upBOff: o.upM,
                                      downWOff: o.downIndex, downSOff: o.downD, downBOff: o.downM))
        #expect(SimdgroupGroupedRoutedMoE.supportsOffsets(params, lut4: true))
        params.pairStart = 0
        params.pairCount = UInt32(routes.groups.count)

        let hidden = (0..<(tokens * Self.d)).map { _ in Float16(rng.uniform(-1, 1)) }
        let hiddenBuf = try #require(Fp16Buffer.make(device, halves: hidden))
        let nanBits = Float16.nan.bitPattern
        func poisoned(_ count: Int) throws -> MTLBuffer {
            let b = try #require(device.makeBuffer(length: count * 2, options: .storageModeShared))
            let p = b.contents().bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count { p[i] = nanBits }
            return b
        }
        let activation = try poisoned(pairs.count * Self.f)
        let partials = try poisoned(tokens * Self.topK * Self.d)
        let argument = try kernel.makeArgumentBuffer(device: device, binding: binding)
        let lut4 = [GGUFLUT4MoE.ProjParams(mix.gateUp), GGUFLUT4MoE.ProjParams(mix.gateUp),
                    GGUFLUT4MoE.ProjParams(mix.down)]
        let cb = try #require(ctx.queue.makeCommandBuffer())
        #expect(kernel.encode(commandBuffer: cb, hidden: hiddenBuf, sortedPairs: metadata.sortedPairs,
                              groups: metadata.groups, activation: activation, routePartials: partials,
                              argumentBuffer: argument, binding: binding, params: params,
                              maxPairsPerGroup: routes.maxPairsPerExpert, lut4: lut4))
        // INT4 parameters are refused for a LUT4 kernel.
        #expect(!kernel.encode(commandBuffer: cb, hidden: hiddenBuf, sortedPairs: metadata.sortedPairs,
                               groups: metadata.groups, activation: activation, routePartials: partials,
                               argumentBuffer: argument, binding: binding, params: params,
                               maxPairsPerGroup: routes.maxPairsPerExpert))
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.error == nil)

        let acts = Fp16Buffer.readHalf(activation, count: pairs.count * Self.f)
        let rp = Fp16Buffer.readHalf(partials, count: tokens * Self.topK * Self.d)
        var gotActs = [Float](), refActs = [Float](), gotDown = [Float](), refDown = [Float]()
        for (i, pair) in routes.sortedPairs.enumerated() {
            let e = experts[Int(pair.expert)]
            let x = Array(hidden[(Int(pair.token) * Self.d)..<((Int(pair.token) + 1) * Self.d)])
            let pActs = Array(acts[(i * Self.f)..<((i + 1) * Self.f)])
            gotActs += pActs.map(Float.init)
            refActs += GGUFLUT4MoEReference.phase1(experts: [e], x: x, d: Self.d, f: Self.f).map(Float.init)
            let at = (Int(pair.token) * Self.topK + Int(pair.rank)) * Self.d
            gotDown += rp[at..<(at + Self.d)].map(Float.init)
            refDown += GGUFLUT4MoEReference.phase2(experts: [e], acts: pActs, routingWeights: [1],
                                                   residual: [Float16](repeating: 0, count: Self.d),
                                                   d: Self.d, f: Self.f).map(Float.init)
        }
        #expect(gotActs.allSatisfy(\.isFinite) && gotDown.allSatisfy(\.isFinite), "a pair's outputs were not written")
        let scale = refDown.reduce(Float(0)) { max($0, abs($1)) }
        #expect(scale > 0.01, "outputs too small to compare: \(scale)")
        let e1 = RelError.compute(actual: gotActs, reference: refActs)
        let e2 = RelError.compute(actual: gotDown, reference: refDown)
        print(String(format: "[psgm-lut4] %@ phase1 rel %.2e down rel %.2e max|ref down| %.3f",
                     mix.testDescription, e1, e2, scale))
        #expect(e1 < Tolerance.fp16Reduction, "phase 1 rel error \(e1)")
        #expect(e2 < Tolerance.fp16Reduction, "down rel error \(e2)")
    }
}
