import Foundation
import Metal
import Testing
import MferenceValidationSupport

@testable import Mference

/// `SimdgroupGroupedRoutedMoE` against a float CPU reference of the routed
/// SwiGLU experts, decoded from the same INT4 affine bytes the GPU reads.
@Suite struct SimdgroupGroupedRoutedMoETests {
    /// Seeded, zero-mean, independent weights per (row, column), quantized row
    /// by row, so a K-order, nibble-order, scale-group or token-row mistake
    /// moves the output by O(1), not by the near-constant synthetic pool's
    /// fraction of a percent.
    static func randomPool(numExperts: Int, d: Int, f: Int, seed: UInt64)
        -> PrefillGroupedRoutedMoETests.SyntheticExpertPool {
        var state = seed
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24) - 0.5   // [-0.5, 0.5)
        }
        var all: [UInt8] = []
        var offsets: MoEExpertOffsets?
        var stride = 0
        for _ in 0..<numExperts {
            var bytes: [UInt8] = []
            var marks: [UInt32] = []
            for (rows, cols) in [(f, d), (f, d), (d, f)] {
                var packed: [UInt8] = [], scales: [UInt8] = [], biases: [UInt8] = []
                for _ in 0..<rows {
                    let q = Quantization.quantizeInt4Affine((0..<cols).map { _ in 0.5 * next() })
                    packed += q.packed
                    PrefillGroupedRoutedMoETests.appendU16(q.scales, to: &scales)
                    PrefillGroupedRoutedMoETests.appendU16(q.biases, to: &biases)
                }
                marks.append(UInt32(bytes.count)); bytes += packed
                marks.append(UInt32(bytes.count)); bytes += scales
                marks.append(UInt32(bytes.count)); bytes += biases
            }
            if offsets == nil {
                offsets = MoEExpertOffsets(gateWOff: marks[0], gateSOff: marks[1], gateBOff: marks[2],
                                           upWOff: marks[3], upSOff: marks[4], upBOff: marks[5],
                                           downWOff: marks[6], downSOff: marks[7], downBOff: marks[8])
                stride = bytes.count
            }
            all += bytes
        }
        return PrefillGroupedRoutedMoETests.SyntheticExpertPool(bytes: all, offsets: offsets!, stride: stride)
    }

    /// One projection of one expert, dequantized to float rows.
    static func decode(pool: PrefillGroupedRoutedMoETests.SyntheticExpertPool, expert: Int,
                       wOff: UInt32, sOff: UInt32, bOff: UInt32, rows: Int, cols: Int,
                       groupSize: Int = Quantization.groupSize) -> [[Float]] {
        let base = expert * pool.stride
        func bf16(_ offset: Int) -> Float {
            let bits = UInt32(pool.bytes[offset]) | (UInt32(pool.bytes[offset + 1]) << 8)
            return Float(bitPattern: bits << 16)
        }
        let groups = cols / groupSize
        return (0..<rows).map { n in
            (0..<cols).map { k in
                let byte = pool.bytes[base + Int(wOff) + n * (cols / 2) + k / 2]
                let q = Float(k % 2 == 0 ? byte & 0x0F : byte >> 4)
                let g = n * groups + k / groupSize
                return q * bf16(base + Int(sOff) + 2 * g) + bf16(base + Int(bOff) + 2 * g)
            }
        }
    }

    /// Five experts in two tiles of three, so the second tile's first pair is
    /// not pair 0. Expert 2 takes every token at rank 0 (70 rows: a full
    /// 64-row block and a partial one); rank 1 spreads tokens over experts
    /// 0, 1, 3 and 4 by `token % 4` (18, 18, 17, 17 rows), so one dispatch
    /// mixes groups of different block counts.
    @Test func matchesFloatReferenceAcrossTilesWithPartialBlocks() throws {
        let d = 256, f = 128, topK = 2, tokens = 70, experts = 5
        #expect(SimdgroupGroupedRoutedMoE.supports(d: d, routedIntermediate: f, groupSize: Quantization.groupSize))
        var pairs: [PrefillTokenExpertPair] = []
        for token in 0..<tokens {
            pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: 2, rank: 0))
            let second: UInt32 = [0, 1, 3, 4][token % 4]
            pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: second, rank: 1))
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: tokens, topK: topK, numExperts: experts, tileExpertCount: 3)
        #expect(routes.groups.map(\.pairCount) == [18, 18, 70, 17, 17])
        #expect(routes.tiles.count == 2)
        #expect(routes.tiles.last.map { $0.pairStart > 0 } == true)
        let pool = Self.randomPool(numExperts: experts, d: d, f: f, seed: 20261116)
        var hiddenState: UInt64 = 7
        let hidden = (0..<(tokens * d)).map { _ -> Float16 in
            hiddenState = hiddenState &* 6364136223846793005 &+ 1442695040888963407
            return Float16(2 * (Float(hiddenState >> 40) / Float(1 << 24) - 0.5))   // [-1, 1)
        }

        let context = try MetalContext()
        let device = context.device
        let kernel = try SimdgroupGroupedRoutedMoE(context: context)
        let rowKernel = try PrefillGroupedRoutedMoE(context: context, siluActivation: true)
        let metadata = try rowKernel.makeStreamedMetadataBuffers(device: device, routes: routes)
        let hiddenBuffer = try #require(Fp16Buffer.make(device, halves: hidden))
        let output = try #require(device.makeBuffer(length: tokens * topK * d * 2, options: .storageModeShared))
        let activation = try #require(device.makeBuffer(length: pairs.count * f * 2, options: .storageModeShared))
        let cb = try #require(context.queue.makeCommandBuffer())
        var keepAlive: [PrefillStreamedTileArgumentBuffer] = []
        for tile in routes.tiles {
            let ids = routes.groups[Int(tile.groupStart)..<Int(tile.groupStart + tile.groupCount)].map { Int($0.expert) }
            let binding = try PrefillStreamedTileBinding(
                expertIDs: ids,
                views: PrefillGroupedRoutedMoETests.streamedViewsWithNonzeroOffsets(
                    device: device, pool: pool, expertIDs: ids))
            let argument = try kernel.makeArgumentBuffer(device: device, binding: binding)
            keepAlive.append(argument)
            var params = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart, pairCount: tile.pairCount, d: UInt32(d),
                routedIntermediate: UInt32(f), topK: UInt32(topK), hiddenStrideElements: UInt32(d),
                binding: binding, offsets: pool.offsets)
            #expect(SimdgroupGroupedRoutedMoE.supportsOffsets(params))
            params.pairStart = tile.groupStart
            params.pairCount = tile.groupCount
            let maxPairs = routes.groups[Int(tile.groupStart)..<Int(tile.groupStart + tile.groupCount)]
                .map { Int($0.pairCount) }.max() ?? 0
            #expect(kernel.encode(commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: metadata.sortedPairs,
                                  groups: metadata.groups, activation: activation, routePartials: output,
                                  argumentBuffer: argument, binding: binding, params: params,
                                  maxPairsPerGroup: maxPairs))
        }
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.error == nil)
        let gpu = Fp16Buffer.readHalf(output, count: tokens * topK * d)

        let o = pool.offsets
        let weights = (0..<experts).map { e in
            (gate: Self.decode(pool: pool, expert: e, wOff: o.gateWOff, sOff: o.gateSOff, bOff: o.gateBOff,
                               rows: f, cols: d),
             up: Self.decode(pool: pool, expert: e, wOff: o.upWOff, sOff: o.upSOff, bOff: o.upBOff,
                             rows: f, cols: d),
             down: Self.decode(pool: pool, expert: e, wOff: o.downWOff, sOff: o.downSOff, bOff: o.downBOff,
                               rows: d, cols: f))
        }
        var maxError: Float = 0, scale: Float = 0
        for pair in pairs {
            let token = Int(pair.token), w = weights[Int(pair.expert)]
            let x = (0..<d).map { Float(hidden[token * d + $0]) }
            // The kernel stores the SwiGLU product as half, as MPP does.
            let act = (0..<f).map { n -> Float in
                let g = zip(w.gate[n], x).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                let u = zip(w.up[n], x).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                return Float(Float16(g / (1 + exp(-g)) * u))
            }
            for row in 0..<d {
                let expected = zip(w.down[row], act).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                let actual = Float(gpu[(token * topK + Int(pair.rank)) * d + row])
                scale = max(scale, abs(expected))
                maxError = max(maxError, abs(expected - actual))
            }
        }
        print(String(format: "[simd-grouped-moe] max|ref-gpu|=%.5f max|ref|=%.4f", maxError, scale))
        #expect(scale > 0.1, "outputs too small to compare: max|ref|=\(scale)")
        #expect(maxError <= 0.01 * scale + 0.002, "maxError=\(maxError) scale=\(scale)")
    }

    @Test func supportsOnlyWholeTileShapesAndAlignedOffsets() throws {
        #expect(SimdgroupGroupedRoutedMoE.supports(d: 2048, routedIntermediate: 512, groupSize: 64))
        #expect(SimdgroupGroupedRoutedMoE.supports(d: 2048, routedIntermediate: 512, groupSize: 32))
        #expect(SimdgroupGroupedRoutedMoE.supports(d: 2560, routedIntermediate: 640, groupSize: 64))
        #expect(!SimdgroupGroupedRoutedMoE.supports(d: 2048 + 64, routedIntermediate: 512, groupSize: 64))
        #expect(!SimdgroupGroupedRoutedMoE.supports(d: 2048, routedIntermediate: 512 + 32, groupSize: 64))
        #expect(!SimdgroupGroupedRoutedMoE.supports(d: 2048, routedIntermediate: 512, groupSize: 16))
        #expect(!SimdgroupGroupedRoutedMoE.supports(d: 2048, routedIntermediate: 512, groupSize: 0))

        let pool = Self.randomPool(numExperts: 1, d: 256, f: 128, seed: 1)
        let context = try MetalContext()
        let binding = try PrefillStreamedTileBinding(
            expertIDs: [0],
            views: PrefillGroupedRoutedMoETests.streamedViewsWithNonzeroOffsets(
                device: context.device, pool: pool, expertIDs: [0]))
        var params = PrefillGroupedRoutedMoEStreamedParams(
            pairStart: 0, pairCount: 1, d: 256, routedIntermediate: 128, topK: 1,
            hiddenStrideElements: 256, binding: binding, offsets: pool.offsets)
        #expect(SimdgroupGroupedRoutedMoE.supportsOffsets(params))
        params.upWOff += 2
        #expect(!SimdgroupGroupedRoutedMoE.supportsOffsets(params))
        params.upWOff -= 2
        params.downSOff += 1
        #expect(!SimdgroupGroupedRoutedMoE.supportsOffsets(params))
    }
}
