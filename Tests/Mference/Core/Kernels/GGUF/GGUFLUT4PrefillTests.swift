import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// LUT4 routed-expert prefill for one streamed tile against
/// `GGUFLUT4MoEReference`, at the install's shape (D 2560, F 640, top-10).
/// The tile holds three experts, given in a different order from the pairs'
/// expert ids, and the pair list also carries one pair whose expert is not in
/// the tile: the kernels must leave its outputs untouched (NaN poison). The
/// down check consumes the GPU's own activations, as the decode test does.
@Suite struct GGUFLUT4PrefillTests {
    static let d = 2560, f = 640, topK = 10, tokens = 12

    @Test(arguments: GGUFLUT4MoETests.mixes)
    func tileMatchesReference(mix: GGUFLUT4MoETests.Mix) throws {
        let ctx = try MetalContext()
        let kernel = try GGUFLUT4Prefill(context: ctx)
        var rng = SeedTree(0x4C55_5450).key("lut4-prefill-\(mix.gateUp.rawValue)-\(mix.down.rawValue)")
        let n = Self.d * Self.f

        // Three experts, built on the CPU as in GGUFLUT4MoETests.
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
        let stride = (slots[0].count + 16_383) / 16_384 * 16_384
        let slab = try #require(ctx.device.makeBuffer(length: stride * 3, options: .storageModeShared))
        for (i, s) in slots.enumerated() {
            s.withUnsafeBytes { slab.contents().advanced(by: i * stride).copyMemory(from: $0.baseAddress!, byteCount: s.count) }
        }
        // Tile slot i holds model expert tileIDs[i]; slots[i] is that expert's data.
        let tileIDs: [UInt32] = [42, 5, 17]

        // Pairs sorted by expert id (5, 17, 42), then one pair for expert 99 (not in the tile).
        var pairs = [(token: UInt32, expert: UInt32, rank: UInt32)]()
        // (token, rank) is unique per pair, as real routing guarantees.
        var k = 0
        for (e, count) in [(UInt32(5), 9), (UInt32(17), 6), (UInt32(42), 7), (UInt32(99), 1)] {
            for _ in 0..<count {
                pairs.append((UInt32(k % Self.tokens), e, UInt32(k / Self.tokens)))
                k += 1
            }
        }
        let pairWords = pairs.flatMap { [$0.token, $0.expert, $0.rank, UInt32(Float16(0.1).bitPattern)] }

        let hidden = (0..<(Self.tokens * Self.d)).map { _ in Float16(rng.uniform(-1, 1)) }
        func buffer<T>(_ v: [T]) throws -> MTLBuffer {
            try #require(v.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        let hiddenBuf = try buffer(hidden), pairBuf = try buffer(pairWords)
        let scratch = try buffer([Float16](repeating: .nan, count: 3 * pairs.count * Self.f))
        let downScratch = try buffer([Float16](repeating: .nan, count: pairs.count * Self.d))
        let partials = try buffer([Float16](repeating: .nan, count: Self.tokens * Self.topK * Self.d))

        let params = GGUFLUT4Prefill.Params(
            pairStart: 0, pairCount: UInt32(pairs.count), d: UInt32(Self.d), f: UInt32(Self.f),
            topK: UInt32(Self.topK), hiddenStrideElements: UInt32(Self.d), expertIDs: tileIDs, offsets: offsets!)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: cb, hidden: hiddenBuf, sortedPairs: pairBuf,
                      gateUpActScratch: scratch, downScratch: downScratch, routePartials: partials,
                      blobs: (0..<3).map { (buffer: slab, offset: $0 * stride) }, params: params,
                      gateUp: mix.gateUp, downType: mix.down)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.status == .completed)

        let plane = pairs.count * Self.f
        let s = Array(UnsafeBufferPointer(start: scratch.contents().bindMemory(to: Float16.self, capacity: 3 * plane), count: 3 * plane))
        let ds = Array(UnsafeBufferPointer(start: downScratch.contents().bindMemory(to: Float16.self, capacity: pairs.count * Self.d), count: pairs.count * Self.d))
        let rp = Array(UnsafeBufferPointer(start: partials.contents().bindMemory(to: Float16.self, capacity: Self.tokens * Self.topK * Self.d), count: Self.tokens * Self.topK * Self.d))

        var gotActs = [Float](), refActs = [Float](), gotDown = [Float](), refDown = [Float](), partialMismatch = 0
        for (i, pair) in pairs.enumerated() {
            let pActs = Array(s[(2 * plane + i * Self.f)..<(2 * plane + (i + 1) * Self.f)])
            let pDown = Array(ds[(i * Self.d)..<((i + 1) * Self.d)])
            guard let slot = tileIDs.firstIndex(of: pair.expert) else {
                #expect(pActs.allSatisfy(\.isNaN) && pDown.allSatisfy(\.isNaN), "pair for an expert outside the tile was written")
                continue
            }
            let e = experts[slot]
            let x = Array(hidden[(Int(pair.token) * Self.d)..<((Int(pair.token) + 1) * Self.d)])
            gotActs += pActs.map(Float.init)
            refActs += GGUFLUT4MoEReference.phase1(experts: [e], x: x, d: Self.d, f: Self.f).map(Float.init)
            gotDown += pDown.map(Float.init)
            refDown += GGUFLUT4MoEReference.phase2(experts: [e], acts: pActs, routingWeights: [1],
                                                   residual: [Float16](repeating: 0, count: Self.d),
                                                   d: Self.d, f: Self.f).map(Float.init)
            let at = (Int(pair.token) * Self.topK + Int(pair.rank)) * Self.d
            for k in 0..<Self.d where rp[at + k].bitPattern != pDown[k].bitPattern { partialMismatch += 1 }
        }
        let e1 = RelError.compute(actual: gotActs, reference: refActs)
        let e2 = RelError.compute(actual: gotDown, reference: refDown)
        #expect(e1 < Tolerance.fp16Reduction, "phase 1 rel error \(e1)")
        #expect(e2 < Tolerance.fp16Reduction, "down rel error \(e2)")
        #expect(partialMismatch == 0, "routePartials differs from downScratch")
    }
}
