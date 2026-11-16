import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The grouped LUT4 GEMM against `GGUFLUT4MoEReference` at the tile shapes a
/// Flash-Next chunk-2048 prefill produces (D 2560, F 640, top-10, 512 experts,
/// 2,048 tokens), driven the way `FlashNextForwardRunner.encodeGroupedLUT4Tile`
/// drives it: one routing and metadata for the whole chunk, tiles of 16 experts
/// taken by group range, and each tile's activation in the runner's borrowed
/// 3 x 256 pair-row scratch, so later tiles run with nonzero group and pair
/// offsets and token indices up to 2,047.
///
/// Designed tiles:
/// - 8 rows x 16 experts: the smallest tile the 8x fill rule still sends to
///   the grouped kernel (7 rows x 16 does not);
/// - one 753-row expert plus 15 one-row experts: the largest group that fits
///   the borrowed activation (768 pair rows); an 800-pair tile does not fit;
/// - partial tails: groups of 1, 2, 7, 8, 9, 15, 17, 31, 63, 64, 65, 100, 127,
///   128 and 129 rows;
/// - the other 26 tiles at 41-42 rows per expert, the chunk's mean fill.
///
/// Every pair of every grouped tile must be written and every pair of a
/// row-kernel tile left untouched; a canary past the 768-row activation must
/// survive. Sampled pairs (each group's first and last row and the rows around
/// each 64-row block edge) are checked against the CPU reference, one pair at a
/// time, within `Tolerance.fp16Reduction`.
@Suite struct SimdgroupGroupedRoutedMoELUT4TileShapeTests {
    static let d = 2560, f = 640, topK = 10, experts = 512, tokens = 2048
    static let borrowedPairRows = 3 * 256

    /// Rows per expert, in expert-ID order (16 experts per tile).
    static func designedCounts() -> [Int] {
        var counts = [Int]()
        counts += Array(repeating: 8, count: 16)                                    // tile 0: smallest grouped
        counts += Array(repeating: 7, count: 16)                                    // tile 1: fails the fill rule
        counts += [753] + Array(repeating: 1, count: 15)                            // tile 2: largest group
        counts += [1, 7, 8, 9, 63, 64, 65, 100, 127, 128, 129, 31, 17, 15, 2, 2]   // tile 3: partial tails
        counts += Array(repeating: 48, count: 16)                                   // tile 4: 768 pairs
        counts += Array(repeating: 50, count: 16)                                   // tile 5: 800 pairs, too big
        let rest = experts - counts.count
        let remaining = tokens * topK - counts.reduce(0, +)
        counts += (0..<rest).map { remaining / rest + ($0 < remaining % rest ? 1 : 0) }
        return counts
    }

    /// The runner's selection: the fill rule and the borrowed activation.
    static func runsGrouped(_ pairCounts: [Int]) -> Bool {
        PrefillGroupedExpertGate.usesSimdgroupGroupedGEMM(pairCounts: pairCounts)
            && pairCounts.reduce(0, +) <= borrowedPairRows
    }

    static func sampledRows(_ count: Int) -> [Int] {
        var rows: Set<Int> = [0, count - 1]
        var edge = 64
        while edge < count {
            for r in [edge - 1, edge] where r < count { rows.insert(r) }
            edge += 64
        }
        return rows.sorted()
    }

    @Test(arguments: [GGUFLUT4MoETests.mixes[1], GGUFLUT4MoETests.mixes[2]])
    func chunk2048TileShapesMatchReference(mix: GGUFLUT4MoETests.Mix) throws {
        let ctx = try MetalContext()
        let device = ctx.device
        let kernel = try SimdgroupGroupedRoutedMoE(context: ctx, dequant: .lut4)
        var rng = SeedTree(0x5453_4850).key("psgm-lut4-shapes-\(mix.gateUp.rawValue)-\(mix.down.rawValue)")
        let d = Self.d, f = Self.f, topK = Self.topK, n = d * f

        // Four weight sets; expert e uses set e % 4.
        var slots = [[UInt8]](), sets = [GGUFLUT4MoEReference.Expert]()
        var offsets: GGUFLUT4MoE.ExpertOffsets?
        for _ in 0..<4 {
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
            sets.append(GGUFLUT4MoEReference.Expert(gate: gw, up: uw, down: dw))
        }
        let o = try #require(offsets)
        let slotStride = (slots[0].count + 16_383) / 16_384 * 16_384
        let slab = try #require(device.makeBuffer(length: slotStride * 4, options: .storageModeShared))
        for (i, s) in slots.enumerated() {
            s.withUnsafeBytes { slab.contents().advanced(by: i * slotStride).copyMemory(from: $0.baseAddress!, byteCount: s.count) }
        }

        // Routing: expert IDs laid out by count, shuffled over the (token, rank) slots.
        let counts = Self.designedCounts()
        #expect(counts.count == Self.experts && counts.reduce(0, +) == Self.tokens * topK)
        var assignment = counts.enumerated().flatMap { Array(repeating: UInt32($0.offset), count: $0.element) }
        for i in stride(from: assignment.count - 1, to: 0, by: -1) {
            assignment.swapAt(i, Int(rng.next() % UInt64(i + 1)))
        }
        let pairs = assignment.enumerated().map { i, e in
            PrefillGroupedRoutedMoETests.pair(token: UInt32(i / topK), expert: e, rank: UInt32(i % topK))
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: Self.tokens, topK: topK, numExperts: Self.experts, tileExpertCount: 16)
        #expect(routes.tiles.count == 32)
        let metadata = try PrefillGroupedRoutedMoE(context: ctx, siluActivation: true)
            .makeStreamedMetadataBuffers(device: device, routes: routes)

        let hidden = (0..<(Self.tokens * d)).map { _ in Float16(rng.uniform(-1, 1)) }
        let hiddenBuf = try #require(Fp16Buffer.make(device, halves: hidden))
        let nanBits = Float16.nan.bitPattern, canaryBits: UInt16 = 0x7BFF
        let partialCount = Self.tokens * topK * d
        let partials = try #require(device.makeBuffer(length: partialCount * 2, options: .storageModeShared))
        let pp = partials.contents().bindMemory(to: UInt16.self, capacity: partialCount)
        for i in 0..<partialCount { pp[i] = nanBits }
        let activationHalves = Self.borrowedPairRows * f, canaryHalves = 4096
        let activation = try #require(device.makeBuffer(length: (activationHalves + canaryHalves) * 2,
                                                        options: .storageModeShared))
        let ap = activation.contents().bindMemory(to: UInt16.self, capacity: activationHalves + canaryHalves)
        let lut4 = [GGUFLUT4MoE.ProjParams(mix.gateUp), GGUFLUT4MoE.ProjParams(mix.gateUp),
                    GGUFLUT4MoE.ProjParams(mix.down)]

        let expectedGrouped: [Int: Bool] = [0: true, 1: false, 2: true, 3: true, 4: true, 5: false]
        let checkedTiles: Set<Int> = [0, 2, 3, 4, 6, 31]
        var groupedTiles = 0, checkedPairs = 0
        var worst: (acts: Float, down: Float, label: String) = (0, 0, "")
        var groupedPairIndices = [Bool](repeating: false, count: pairs.count)

        for (tileIndex, tile) in routes.tiles.enumerated() {
            let groups = routes.groups[Int(tile.groupStart)..<Int(tile.groupStart + tile.groupCount)]
            let pairCounts = groups.map { Int($0.pairCount) }
            let grouped = Self.runsGrouped(pairCounts)
            if let expected = expectedGrouped[tileIndex] {
                #expect(grouped == expected, "tile \(tileIndex) \(pairCounts): grouped \(grouped)")
            }
            guard grouped else { continue }
            groupedTiles += 1
            for i in Int(tile.pairStart)..<Int(tile.pairStart + tile.pairCount) { groupedPairIndices[i] = true }

            let experts = try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes)
            let views = experts.map { e in
                TensorView(buffer: slab, offset: UInt64((e % 4) * slotStride), length: UInt64(slotStride),
                           scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                           shape: (0, UInt32(e), 0, 0), dtype: 0)
            }
            let binding = try PrefillStreamedTileBinding(expertIDs: experts, views: views)
            var params = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart, pairCount: tile.pairCount, d: UInt32(d), routedIntermediate: UInt32(f),
                topK: UInt32(topK), hiddenStrideElements: UInt32(d), binding: binding,
                offsets: MoEExpertOffsets(gateWOff: o.gateIndex, gateSOff: o.gateD, gateBOff: o.gateM,
                                          upWOff: o.upIndex, upSOff: o.upD, upBOff: o.upM,
                                          downWOff: o.downIndex, downSOff: o.downD, downBOff: o.downM))
            try #require(SimdgroupGroupedRoutedMoE.supportsOffsets(params, lut4: true))
            params.pairStart = tile.groupStart
            params.pairCount = tile.groupCount

            for i in 0..<activationHalves { ap[i] = nanBits }
            for i in activationHalves..<(activationHalves + canaryHalves) { ap[i] = canaryBits }
            let argument = try kernel.makeArgumentBuffer(device: device, binding: binding)
            let cb = try #require(ctx.queue.makeCommandBuffer())
            try #require(kernel.encode(commandBuffer: cb, hidden: hiddenBuf, sortedPairs: metadata.sortedPairs,
                                       groups: metadata.groups, activation: activation, routePartials: partials,
                                       argumentBuffer: argument, binding: binding, params: params,
                                       maxPairsPerGroup: pairCounts.max() ?? 0, lut4: lut4))
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.error == nil, "tile \(tileIndex): \(String(describing: cb.error))")

            let canaryIntact = (activationHalves..<(activationHalves + canaryHalves)).allSatisfy { ap[$0] == canaryBits }
            #expect(canaryIntact, "tile \(tileIndex) wrote past the borrowed activation")
            let liveActs = Int(tile.pairCount) * f
            let actsWritten = (0..<liveActs).allSatisfy { Float16(bitPattern: ap[$0]).isFinite }
            #expect(actsWritten, "tile \(tileIndex): an activation row was not written")
            let untouched = (liveActs..<activationHalves).allSatisfy { ap[$0] == nanBits }
            #expect(untouched, "tile \(tileIndex) wrote activation rows past its pairs")

            guard checkedTiles.contains(tileIndex) else { continue }
            var pairOffset = Int(tile.pairStart)
            for group in groups {
                for row in Self.sampledRows(Int(group.pairCount)) {
                    let sorted = pairOffset + row
                    let pair = routes.sortedPairs[sorted]
                    let set = sets[Int(pair.expert) % 4]
                    let x = Array(hidden[(Int(pair.token) * d)..<((Int(pair.token) + 1) * d)])
                    let local = sorted - Int(tile.pairStart)
                    let got = (0..<f).map { Float16(bitPattern: ap[local * f + $0]) }
                    let refActs = GGUFLUT4MoEReference.phase1(experts: [set], x: x, d: d, f: f).map(Float.init)
                    let at = (Int(pair.token) * topK + Int(pair.rank)) * d
                    let gotDown = (0..<d).map { Float(Float16(bitPattern: pp[at + $0])) }
                    let refDown = GGUFLUT4MoEReference.phase2(
                        experts: [set], acts: got, routingWeights: [1],
                        residual: [Float16](repeating: 0, count: d), d: d, f: f).map(Float.init)
                    let e1 = RelError.compute(actual: got.map(Float.init), reference: refActs)
                    let e2 = RelError.compute(actual: gotDown, reference: refDown)
                    let label = "tile \(tileIndex) expert \(group.expert) rows \(group.pairCount) row \(row) token \(pair.token)"
                    if max(e1, e2) > max(worst.acts, worst.down) { worst = (e1, e2, label) }
                    #expect(e1 < Tolerance.fp16Reduction, "\(label): phase 1 rel \(e1)")
                    #expect(e2 < Tolerance.fp16Reduction, "\(label): down rel \(e2)")
                    checkedPairs += 1
                }
                pairOffset += Int(group.pairCount)
            }
        }

        // Grouped tiles wrote every one of their pairs' partials; row-kernel tiles' pairs are untouched.
        var unwritten = 0, stray = 0
        for (i, pair) in routes.sortedPairs.enumerated() {
            let at = (Int(pair.token) * topK + Int(pair.rank)) * d
            let finite = Float16(bitPattern: pp[at]).isFinite && Float16(bitPattern: pp[at + d - 1]).isFinite
            if groupedPairIndices[i] && !finite { unwritten += 1 }
            if !groupedPairIndices[i] && (0..<d).contains(where: { pp[at + $0] != nanBits }) { stray += 1 }
        }
        print(String(format: "[psgm-lut4-shapes] %@ grouped tiles %d/%d, checked pairs %d, worst phase1 %.2e down %.2e at %@",
                     mix.testDescription, groupedTiles, routes.tiles.count, checkedPairs,
                     worst.acts, worst.down, worst.label))
        #expect(groupedTiles == 30)
        #expect(unwritten == 0, "\(unwritten) grouped pairs' partials were not written")
        #expect(stray == 0, "\(stray) row-kernel pairs' partials were written by the grouped kernel")
    }
}
