import Foundation
import Metal
import Testing
import MferenceValidationSupport

@testable import Mference

private let qwenGroupedGEMMAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPGroupedRoutedMoE(context: context).isAvailable
}()

/// Grouped-GEMM routed experts for Qwen 3.6 prefill (`MPPGroupedRoutedMoE`
/// behind `RealForwardRunner.usesGroupedRoutedExpertGEMM`).
@Suite struct QwenGroupedRoutedExpertTests {
    @Test func selectionIsQwenOffApple10AndGemmaPolicyOnly() {
        for family in ModelFamily.allCases {
            for batched in [false, true] {
                for apple10 in [false, true] {
                    let selected = RealForwardRunner.usesGroupedRoutedExpertGEMM(
                        family: family, gemmaBatchedExperts: batched,
                        supportsApple10TensorOps: apple10)
                    let expected: Bool
                    switch family {
                    case .gemma4: expected = batched
                    case .qwen36: expected = !apple10
                    default: expected = false
                    }
                    #expect(selected == expected, "\(family) batched=\(batched) apple10=\(apple10)")
                }
            }
        }
    }

    /// Qwen's grouped activation (chunk x topK x routedIntermediate) equals its
    /// attention-output scratch (chunk x 16 heads x 256), so grouped experts
    /// borrow that buffer and the prefill arena does not grow.
    @Test(arguments: [512, 2048, 4096])
    func qwenGroupedActivationBorrowsAttentionOutput(chunkTokens: Int) {
        let plain = PrefillChunkScratchLayout(config: .qwen36_35B_A3B, chunkTokens: chunkTokens)
        let grouped = PrefillChunkScratchLayout(config: .qwen36_35B_A3B, chunkTokens: chunkTokens,
                                                groupedExperts: true)
        #expect(grouped.groupedExpertActivationElements == grouped.attentionOutputElements)
        #expect(grouped.dedicatedGroupedExpertActivationElements == 0)
        #expect(grouped.devicePrivateBytes == plain.devicePrivateBytes)
    }

    /// Grouped GEMM against the row kernel at Qwen 3.6's routed shape (d 2,048,
    /// f 512, group size 64, SwiGLU), with partial 64-row tiles and a one-row group.
    @Test(.enabled(if: qwenGroupedGEMMAvailable, "Requires runtime MPP TensorOps support"))
    func groupedMatchesRowKernelAtQwenShape() throws {
        let d = 2048, f = 512, rows = 130, topK = 2
        // Expert 2 takes every token at rank 0 (130 rows); rank 1 sends tokens
        // 0..<129 to expert 0 (129 rows) and token 129 to expert 1 (one row).
        let pairs = (0..<rows).flatMap { token -> [PrefillTokenExpertPair] in
            [PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: 2, rank: 0),
             PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: token < 129 ? 0 : 1, rank: 1)]
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: rows, topK: topK, numExperts: 3, tileExpertCount: 3)
        #expect(routes.groups.map(\.pairCount).sorted() == [1, 129, 130])
        let outputs = try Self.runBothKernels(
            d: d, f: f, rows: rows, topK: topK, pairs: pairs, routes: routes,
            poolExperts: 3, viewExperts: [0, 1, 2], timedRepeats: 1)
        let scale = outputs.row.map { abs(Float($0)) }.max() ?? 0
        let maxError = zip(outputs.row, outputs.grouped).reduce(Float(0)) {
            max($0, abs(Float($1.0) - Float($1.1)))
        }
        print(String(format: "[qwen-grouped-moe] max|row-grouped|=%.5f max|row|=%.4f", maxError, scale))
        #expect(scale > 0.1, "outputs too small to compare: max|row|=\(scale)")
        #expect(maxError <= 0.01 * scale + 0.002, "maxError=\(maxError) scale=\(scale)")
    }

    struct KernelOutputs {
        let row: [Float16]
        let grouped: [Float16]
        let rowSeconds: Double
        let groupedSeconds: Double
    }

    /// Runs one routed tile through the row kernel (32-row microbatches, the
    /// production default) and the grouped kernel; returns both outputs and
    /// the best GPU time of `timedRepeats` runs each.
    static func runBothKernels(d: Int, f: Int, rows: Int, topK: Int,
                               pairs: [PrefillTokenExpertPair],
                               routes: PrefillMoEGroupedRoutes,
                               poolExperts: Int, viewExperts: [Int],
                               timedRepeats: Int) throws -> KernelOutputs {
        let context = try MetalContext()
        let pool = Self.syntheticPool(numExperts: poolExperts, d: d, f: f)
        let rowKernel = try PrefillGroupedRoutedMoE(context: context, siluActivation: true)
        let grouped = MPPGroupedRoutedMoE(context: context)
        try #require(grouped.isAvailable)
        let binding = try PrefillStreamedTileBinding(
            expertIDs: Array(0..<viewExperts.count),
            views: PrefillGroupedRoutedMoETests.streamedViewsWithNonzeroOffsets(
                device: context.device, pool: pool, expertIDs: viewExperts))
        let metadata = try rowKernel.makeStreamedMetadataBuffers(device: context.device, routes: routes)
        let rowArgument = try rowKernel.makeStreamedArgumentBuffer(device: context.device, binding: binding)
        let groupedArgument = try grouped.makeArgumentBuffer(device: context.device, binding: binding)
        let params = PrefillGroupedRoutedMoEStreamedParams(
            pairStart: 0, pairCount: UInt32(pairs.count),
            d: UInt32(d), routedIntermediate: UInt32(f),
            topK: UInt32(topK), hiddenStrideElements: UInt32(d),
            binding: binding, offsets: pool.offsets)
        var groupedParams = params
        groupedParams.pairStart = 0
        groupedParams.pairCount = UInt32(routes.groups.count)

        // A positive mean keeps the 2,048-term dot products with the synthetic
        // (positive-mean) weights from cancelling, so outputs are O(1).
        let hidden = (0..<(rows * d)).map { Float16(0.05 + Float(($0 % 23) - 11) * 0.004) }
        let hiddenBuffer = try #require(Fp16Buffer.make(context.device, halves: hidden))
        func buffer(_ elements: Int) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(length: elements * MemoryLayout<Float16>.stride,
                                                   options: .storageModeShared))
        }
        let microbatch = 32
        let rowOutput = try buffer(rows * topK * d)
        let groupedOutput = try buffer(rows * topK * d)
        let rowAct = try buffer(3 * microbatch * f)
        let rowDown = try buffer(microbatch * d)
        let groupedAct = try buffer(pairs.count * f)

        func timed(_ encode: (MTLCommandBuffer) -> Void) throws -> Double {
            var best = Double.infinity
            for _ in 0..<max(1, timedRepeats) {
                let cb = try #require(context.queue.makeCommandBuffer())
                encode(cb)
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.error == nil)
                best = min(best, cb.gpuEndTime - cb.gpuStartTime)
            }
            return best
        }
        let rowSeconds = try timed { cb in
            _ = rowKernel.encodeStreamedBatched(
                commandBuffer: cb, hidden: hiddenBuffer,
                sortedPairs: metadata.sortedPairs, routePartials: rowOutput,
                gateUpActScratch: rowAct, downScratch: rowDown,
                argumentBuffer: rowArgument, binding: binding,
                params: params, pairMicrobatchRows: microbatch)
        }
        var encoded = true
        let groupedSeconds = try timed { cb in
            encoded = encoded && grouped.encode(
                commandBuffer: cb, hidden: hiddenBuffer,
                sortedPairs: metadata.sortedPairs, groups: metadata.groups,
                activation: groupedAct, routePartials: groupedOutput,
                argumentBuffer: groupedArgument, binding: binding,
                params: groupedParams, maxPairsPerGroup: routes.maxPairsPerExpert)
        }
        #expect(encoded)
        return KernelOutputs(row: Fp16Buffer.readHalf(rowOutput, count: rows * topK * d),
                             grouped: Fp16Buffer.readHalf(groupedOutput, count: rows * topK * d),
                             rowSeconds: rowSeconds, groupedSeconds: groupedSeconds)
    }

    /// The same bytes as `PrefillGroupedRoutedMoETests.makeSyntheticExpertPool`,
    /// built fast enough for Qwen's shape: its synthetic rows depend on the row
    /// only through `row % 7`, so each projection quantizes seven distinct rows
    /// and repeats them.
    static func syntheticPool(numExperts: Int, d: Int, f: Int)
        -> PrefillGroupedRoutedMoETests.SyntheticExpertPool {
        func projection(rows: Int, cols: Int, expert: Int, role: Int)
            -> (packed: [UInt8], scales: [UInt8], biases: [UInt8]) {
            let distinct = PrefillGroupedRoutedMoETests.syntheticRows(
                rows: min(rows, 7), cols: cols, expert: expert, role: role)
                .map { Quantization.quantizeInt4Affine($0) }
            var packed: [UInt8] = [], scales: [UInt8] = [], biases: [UInt8] = []
            for row in 0..<rows {
                let q = distinct[row % 7]
                packed.append(contentsOf: q.packed)
                PrefillGroupedRoutedMoETests.appendU16(q.scales, to: &scales)
                PrefillGroupedRoutedMoETests.appendU16(q.biases, to: &biases)
            }
            return (packed, scales, biases)
        }
        var all: [UInt8] = []
        var offsets: MoEExpertOffsets?
        var stride = 0
        for expert in 0..<numExperts {
            var bytes: [UInt8] = []
            var marks: [UInt32] = []
            for (role, shape) in [(0, (f, d)), (1, (f, d)), (2, (d, f))] {
                let p = projection(rows: shape.0, cols: shape.1, expert: expert, role: role)
                marks.append(UInt32(bytes.count)); bytes += p.packed
                marks.append(UInt32(bytes.count)); bytes += p.scales
                marks.append(UInt32(bytes.count)); bytes += p.biases
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
}

/// GPU time of one routed tile through the row kernel and the grouped kernel
/// at Qwen 3.6's shape: 16 experts per tile, topK 8, 64 or 128 rows per
/// expert (a 2,048-token chunk averages 64). Env-gated:
/// `MFERENCE_PREFILL_MOE_KBENCH=1`. Prints one line per shape; asserts
/// nothing about speed.
@Suite(.serialized) struct QwenGroupedRoutedExpertKernelBenchmarks {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_PREFILL_MOE_KBENCH"] == "1"
            && qwenGroupedGEMMAvailable
    }

    @Test(.enabled(if: enabled), arguments: [64, 128])
    func qwenRoutedTile(rowsPerExpert: Int) throws {
        let experts = 16, topK = 8
        let tokens = rowsPerExpert * experts / topK
        let pairs = (0..<tokens).flatMap { (token: Int) -> [PrefillTokenExpertPair] in
            (0..<topK).map { (rank: Int) -> PrefillTokenExpertPair in
                PrefillGroupedRoutedMoETests.pair(
                    token: UInt32(token), expert: UInt32((token * topK + rank) % experts), rank: UInt32(rank))
            }
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: tokens, topK: topK, numExperts: experts, tileExpertCount: experts)
        #expect(Set(routes.groups.map(\.pairCount)) == [UInt32(rowsPerExpert)])
        // Two distinct synthetic experts bound alternately keep the pool small.
        let outputs = try QwenGroupedRoutedExpertTests.runBothKernels(
            d: 2048, f: 512, rows: tokens, topK: topK, pairs: pairs, routes: routes,
            poolExperts: 2, viewExperts: (0..<experts).map { $0 % 2 }, timedRepeats: 3)
        print(String(format: "[moe-kbench] experts=%d rows/expert=%d row %.2f ms grouped %.2f ms (%.2fx)",
                     experts, rowsPerExpert, outputs.rowSeconds * 1000, outputs.groupedSeconds * 1000,
                     outputs.rowSeconds / outputs.groupedSeconds))
    }
}

/// Padding sweep behind Qwen 3.6's grouped-expert fill rule: GPU time of every
/// routed tile of one simulated chunk through the row kernel and the grouped
/// kernel, at Qwen's shape (256 experts, topK 8, 16 experts per tile, d 2,048,
/// f 512). Routing is synthetic: each token picks 8 distinct experts, either
/// uniformly or with a fixed skew (weight ~ 1 / (rank + 1)^0.6 over a seeded
/// permutation). For each fill-rule limit it prints the chunk's total routed
/// time with tiles under the limit grouped and the rest on the row kernel.
/// Env-gated: `MFERENCE_PREFILL_MOE_SWEEP=1`. Asserts nothing about speed.
@Suite(.serialized) struct QwenGroupedRoutedExpertPaddingSweep {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_PREFILL_MOE_SWEEP"] == "1"
            && qwenGroupedGEMMAvailable
    }

    static let limits: [Double] = [1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 2.0, 99]

    /// Seeded routing: `tokens` tokens, each with `topK` distinct experts.
    static func route(tokens: Int, topK: Int, experts: Int, skewed: Bool, seed: UInt64)
        -> [PrefillTokenExpertPair] {
        var state = seed
        func next() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / Double(1 << 53)
        }
        var order = Array(0..<experts)
        for i in stride(from: experts - 1, to: 0, by: -1) {
            order.swapAt(i, Int(next() * Double(i + 1)))
        }
        var weight = [Double](repeating: 1, count: experts)
        if skewed { for (rank, expert) in order.enumerated() { weight[expert] = 1 / pow(Double(rank + 1), 0.6) } }
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(tokens * topK)
        for token in 0..<tokens {
            var chosen = Set<Int>()
            var remaining = weight
            for rank in 0..<topK {
                let total = remaining.reduce(0, +)
                var pick = next() * total, expert = 0
                while expert < experts - 1 && pick >= remaining[expert] { pick -= remaining[expert]; expert += 1 }
                while chosen.contains(expert) { expert = (expert + 1) % experts }
                chosen.insert(expert); remaining[expert] = 0
                pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: UInt32(expert),
                                                               rank: UInt32(rank)))
            }
        }
        return pairs
    }

    struct TileTime { let real: Int; let padded: Int; let row: Double; let grouped: Double }

    static func timeTiles(tokens: Int, pairs: [PrefillTokenExpertPair], experts: Int) throws -> [TileTime] {
        let d = 2048, f = 512, topK = 8
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: tokens, topK: topK, numExperts: experts, tileExpertCount: 16)
        let context = try MetalContext()
        let pool = QwenGroupedRoutedExpertTests.syntheticPool(numExperts: 2, d: d, f: f)
        let rowKernel = try PrefillGroupedRoutedMoE(context: context, siluActivation: true)
        let grouped = MPPGroupedRoutedMoE(context: context)
        try #require(grouped.isAvailable)
        let metadata = try rowKernel.makeStreamedMetadataBuffers(device: context.device, routes: routes)
        let hidden = (0..<(tokens * d)).map { Float16(0.05 + Float(($0 % 23) - 11) * 0.004) }
        let hiddenBuffer = try #require(Fp16Buffer.make(context.device, halves: hidden))
        func buffer(_ elements: Int) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(length: elements * 2, options: .storageModePrivate))
        }
        let microbatch = 32
        let partials = try buffer(tokens * topK * d)
        let rowAct = try buffer(3 * microbatch * f)
        let rowDown = try buffer(microbatch * d)
        let groupedAct = try buffer(tokens * topK * f)
        func best(_ encode: (MTLCommandBuffer) -> Void) throws -> Double {
            var best = Double.infinity
            for _ in 0..<3 {
                let cb = try #require(context.queue.makeCommandBuffer())
                encode(cb)
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.error == nil)
                best = min(best, cb.gpuEndTime - cb.gpuStartTime)
            }
            return best
        }
        var result: [TileTime] = []
        for tile in routes.tiles {
            let groups = routes.groups[Int(tile.groupStart)..<Int(tile.groupStart + tile.groupCount)]
            let ids = groups.map { Int($0.expert) }
            let binding = try PrefillStreamedTileBinding(
                expertIDs: ids,
                views: PrefillGroupedRoutedMoETests.streamedViewsWithNonzeroOffsets(
                    device: context.device, pool: pool, expertIDs: ids.indices.map { $0 % 2 }))
            let rowArgument = try rowKernel.makeStreamedArgumentBuffer(device: context.device, binding: binding)
            let groupedArgument = try grouped.makeArgumentBuffer(device: context.device, binding: binding)
            let params = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart, pairCount: tile.pairCount,
                d: UInt32(d), routedIntermediate: UInt32(f), topK: UInt32(topK),
                hiddenStrideElements: UInt32(d), binding: binding, offsets: pool.offsets)
            var groupedParams = params
            groupedParams.pairStart = tile.groupStart
            groupedParams.pairCount = tile.groupCount
            let counts = groups.map { Int($0.pairCount) }
            let row = try best { cb in
                _ = rowKernel.encodeStreamedBatched(
                    commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: metadata.sortedPairs,
                    routePartials: partials, gateUpActScratch: rowAct, downScratch: rowDown,
                    argumentBuffer: rowArgument, binding: binding, params: params,
                    pairMicrobatchRows: microbatch)
            }
            let groupedTime = try best { cb in
                _ = grouped.encode(
                    commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: metadata.sortedPairs,
                    groups: metadata.groups, activation: groupedAct, routePartials: partials,
                    argumentBuffer: groupedArgument, binding: binding, params: groupedParams,
                    maxPairsPerGroup: counts.max() ?? 0)
            }
            let tileRows = PrefillGroupedExpertGate.tileRows
            result.append(TileTime(real: counts.reduce(0, +),
                                   padded: counts.reduce(0) { $0 + ($1 + tileRows - 1) / tileRows * tileRows },
                                   row: row, grouped: groupedTime))
        }
        return result
    }

    static func report(_ label: String, _ tiles: [TileTime]) {
        for t in tiles {
            print(String(format: "[moe-sweep] %@ tile real=%d padded=%d ratio=%.3f row %.3f ms grouped %.3f ms (%.2fx)",
                         label, t.real, t.padded, Double(t.padded) / Double(t.real),
                         t.row * 1000, t.grouped * 1000, t.row / t.grouped))
        }
        let allRow = tiles.reduce(0) { $0 + $1.row }
        for limit in limits {
            let total = tiles.reduce(0.0) { $0 + (Double($1.padded) <= limit * Double($1.real) ? $1.grouped : $1.row) }
            print(String(format: "[moe-sweep] %@ limit=%.2f routed %.2f ms (row-only %.2f ms, %+.1f%%)",
                         label, limit, total * 1000, allRow * 1000, (total / allRow - 1) * 100))
        }
    }

    @Test(.enabled(if: enabled), arguments: [2048, 2940, 4096])
    func simulatedChunk(tokens: Int) throws {
        for skewed in [false, true] {
            let pairs = Self.route(tokens: tokens, topK: 8, experts: 256, skewed: skewed, seed: 20261116)
            let tiles = try Self.timeTiles(tokens: tokens, pairs: pairs, experts: 256)
            Self.report("chunk=\(tokens) \(skewed ? "skewed" : "uniform")", tiles)
        }
    }

    /// 16 experts with the same row count each: one ratio per run.
    @Test(.enabled(if: enabled),
          arguments: [40, 48, 56, 64, 72, 80, 88, 92, 96, 104, 112, 120, 128, 136, 160, 192, 256])
    func uniformRowsPerExpert(rows: Int) throws {
        let tokens = 2 * rows
        // A plain loop: the nested flatMap/map closure timed out the type
        // checker on the CI runners.
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(tokens * 8)
        for token in 0..<tokens {
            for rank in 0..<8 {
                let expert: Int = (token * 8 + rank) % 16
                pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: UInt32(expert),
                                                               rank: UInt32(rank)))
            }
        }
        Self.report("rows=\(rows)", try Self.timeTiles(tokens: tokens, pairs: pairs, experts: 16))
    }
}
