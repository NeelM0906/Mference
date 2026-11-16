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
        let pairs = (0..<tokens).flatMap { token in
            (0..<topK).map { rank in
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
