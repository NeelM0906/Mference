import Foundation
import Metal
import Testing
import MferenceValidationSupport

@testable import Mference

/// GPU time of `SimdgroupGroupedRoutedMoE` against the row kernel and
/// `MPPGroupedRoutedMoE` at Qwen 3.6's routed shape (16 experts per tile,
/// d 2,048, f 512, topK 8): single tiles at fixed fills, and every tile of a
/// simulated chunk. Env-gated: `MFERENCE_PREFILL_MOE_A7=1`. Asserts
/// agreement with the row kernel within the grouped tolerance, not speed.
@Suite(.serialized) struct SimdgroupGroupedRoutedMoEBenchmarks {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_PREFILL_MOE_A7"] == "1"
    }

    struct Outputs {
        let row: [Float16]
        let mpp: [Float16]?
        let prototype: [Float16]
        let rowSeconds: Double
        let mppSeconds: Double?
        let prototypeSeconds: Double
    }

    /// One routed tile through the row kernel (32-row microbatches, as in
    /// production), the MPP grouped kernel (when it builds) and the prototype.
    static func run(rowsPerExpert: Int, experts: Int = 16, distinctExperts: Int = 16,
                    repeats: Int) throws -> Outputs {
        let d = 2048, f = 512, topK = 8
        let tokens = rowsPerExpert * experts / topK
        var pairs: [PrefillTokenExpertPair] = []
        for token in 0..<tokens {
            for rank in 0..<topK {
                let expert: Int = (token * topK + rank) % experts
                pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: UInt32(expert),
                                                               rank: UInt32(rank)))
            }
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: tokens, topK: topK, numExperts: experts, tileExpertCount: experts)
        let context = try MetalContext()
        let device = context.device
        let pool = QwenGroupedRoutedExpertTests.syntheticPool(numExperts: distinctExperts, d: d, f: f)
        let rowKernel = try PrefillGroupedRoutedMoE(context: context, siluActivation: true)
        let mpp = MPPGroupedRoutedMoE(context: context)
        let prototype = try SimdgroupGroupedRoutedMoE(context: context)
        let binding = try PrefillStreamedTileBinding(
            expertIDs: Array(0..<experts),
            views: PrefillGroupedRoutedMoETests.streamedViewsWithNonzeroOffsets(
                device: device, pool: pool, expertIDs: (0..<experts).map { $0 % distinctExperts }))
        let metadata = try rowKernel.makeStreamedMetadataBuffers(device: device, routes: routes)
        let rowArgument = try rowKernel.makeStreamedArgumentBuffer(device: device, binding: binding)
        let params = PrefillGroupedRoutedMoEStreamedParams(
            pairStart: 0, pairCount: UInt32(pairs.count), d: UInt32(d), routedIntermediate: UInt32(f),
            topK: UInt32(topK), hiddenStrideElements: UInt32(d), binding: binding, offsets: pool.offsets)
        var groupedParams = params
        groupedParams.pairCount = UInt32(routes.groups.count)

        let hidden = (0..<(tokens * d)).map { Float16(0.05 + Float(($0 % 23) - 11) * 0.004) }
        let hiddenBuffer = try #require(Fp16Buffer.make(device, halves: hidden))
        func buffer(_ elements: Int) throws -> MTLBuffer {
            try #require(device.makeBuffer(length: elements * MemoryLayout<Float16>.stride,
                                           options: .storageModeShared))
        }
        let microbatch = 32
        let rowOutput = try buffer(tokens * topK * d)
        let mppOutput = try buffer(tokens * topK * d)
        let prototypeOutput = try buffer(tokens * topK * d)
        let rowAct = try buffer(3 * microbatch * f)
        let rowDown = try buffer(microbatch * d)
        let groupedAct = try buffer(pairs.count * f)

        func timed(_ encode: (MTLCommandBuffer) throws -> Void) throws -> Double {
            var best = Double.infinity
            for _ in 0..<max(1, repeats) {
                let cb = try #require(context.queue.makeCommandBuffer())
                try encode(cb)
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.error == nil, "\(String(describing: cb.error))")
                best = min(best, cb.gpuEndTime - cb.gpuStartTime)
            }
            return best
        }
        let rowSeconds = try timed { cb in
            _ = rowKernel.encodeStreamedBatched(
                commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: metadata.sortedPairs,
                routePartials: rowOutput, gateUpActScratch: rowAct, downScratch: rowDown,
                argumentBuffer: rowArgument, binding: binding, params: params,
                pairMicrobatchRows: microbatch)
        }
        var mppSeconds: Double?
        if mpp.isAvailable {
            let mppArgument = try mpp.makeArgumentBuffer(device: device, binding: binding)
            mppSeconds = try timed { cb in
                _ = mpp.encode(commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: metadata.sortedPairs,
                               groups: metadata.groups, activation: groupedAct, routePartials: mppOutput,
                               argumentBuffer: mppArgument, binding: binding, params: groupedParams,
                               maxPairsPerGroup: routes.maxPairsPerExpert)
            }
        }
        let prototypeArgument = try prototype.makeArgumentBuffer(device: device, binding: binding)
        let prototypeSeconds = try timed { cb in
            _ = prototype.encode(commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: metadata.sortedPairs,
                                 groups: metadata.groups, activation: groupedAct,
                                 routePartials: prototypeOutput, argumentBuffer: prototypeArgument,
                                 binding: binding, params: groupedParams,
                                 maxPairsPerGroup: routes.maxPairsPerExpert)
        }
        let count = tokens * topK * d
        return Outputs(row: Fp16Buffer.readHalf(rowOutput, count: count),
                       mpp: mppSeconds == nil ? nil : Fp16Buffer.readHalf(mppOutput, count: count),
                       prototype: Fp16Buffer.readHalf(prototypeOutput, count: count),
                       rowSeconds: rowSeconds, mppSeconds: mppSeconds, prototypeSeconds: prototypeSeconds)
    }

    static func maxError(_ a: [Float16], _ b: [Float16]) -> Float {
        zip(a, b).reduce(Float(0)) { max($0, abs(Float($1.0) - Float($1.1))) }
    }

    /// Correctness against the row kernel, including partial and one-row
    /// blocks, then speed. 16 distinct experts, so weights do not repeat.
    @Test(.enabled(if: enabled), arguments: [1, 7, 8, 10, 12, 16, 32, 64, 92, 128])
    func simdgroupMatchesRowKernelAndTimesAgainstBoth(rowsPerExpert: Int) throws {
        let outputs = try Self.run(rowsPerExpert: rowsPerExpert, repeats: 5)
        let scale = outputs.row.map { abs(Float($0)) }.max() ?? 0
        let error = Self.maxError(outputs.row, outputs.prototype)
        let mppText = outputs.mppSeconds.map { String(format: "%.2f ms (%.2fx row)", $0 * 1000,
                                                       outputs.rowSeconds / $0) } ?? "n/a"
        print(String(format: "[a7-kbench] rows/expert=%3d row %.2f ms | mpp %@ | a7 %.2f ms (%.2fx row, %@ mpp) | max|row-a7|=%.5f max|row|=%.3f",
                     rowsPerExpert, outputs.rowSeconds * 1000, mppText, outputs.prototypeSeconds * 1000,
                     outputs.rowSeconds / outputs.prototypeSeconds,
                     outputs.mppSeconds.map { String(format: "%.2fx", $0 / outputs.prototypeSeconds) } ?? "n/a",
                     error, scale))
        #expect(scale > 0.1, "outputs too small to compare: max|row|=\(scale)")
        #expect(error <= 0.01 * scale + 0.002, "maxError=\(error) scale=\(scale)")
    }

    /// Every routed tile of a simulated chunk (the padding sweep's seeded
    /// routing, 256 experts, 16 per tile) through the row kernel, MPP and the
    /// prototype. Prints the chunk total under today's selection (MPP when
    /// padded rows <= 1.5x real, else row), the prototype on every tile, and
    /// the prototype with the row kernel wherever the row kernel is faster.
    @Test(.enabled(if: enabled), arguments: [2048, 2940, 4096])
    func simulatedChunk(tokens: Int) throws {
        let d = 2048, f = 512, topK = 8, experts = 256
        let context = try MetalContext()
        let device = context.device
        let pool = QwenGroupedRoutedExpertTests.syntheticPool(numExperts: 16, d: d, f: f)
        let rowKernel = try PrefillGroupedRoutedMoE(context: context, siluActivation: true)
        let mpp = MPPGroupedRoutedMoE(context: context)
        let prototype = try SimdgroupGroupedRoutedMoE(context: context)
        let hidden = (0..<(tokens * d)).map { Float16(0.05 + Float(($0 % 23) - 11) * 0.004) }
        let hiddenBuffer = try #require(Fp16Buffer.make(device, halves: hidden))
        func buffer(_ elements: Int) throws -> MTLBuffer {
            try #require(device.makeBuffer(length: elements * 2, options: .storageModePrivate))
        }
        let partials = try buffer(tokens * topK * d)
        let rowAct = try buffer(3 * 32 * f)
        let rowDown = try buffer(32 * d)
        let groupedAct = try buffer(tokens * topK * f)
        func best(_ encode: (MTLCommandBuffer) throws -> Void) throws -> Double {
            var best = Double.infinity
            for _ in 0..<3 {
                let cb = try #require(context.queue.makeCommandBuffer())
                try encode(cb)
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.error == nil)
                best = min(best, cb.gpuEndTime - cb.gpuStartTime)
            }
            return best
        }
        for skewed in [false, true] {
            let pairs = QwenGroupedRoutedExpertPaddingSweep.route(
                tokens: tokens, topK: topK, experts: experts, skewed: skewed, seed: 20261116)
            let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                pairs, queryCount: tokens, topK: topK, numExperts: experts, tileExpertCount: 16)
            let metadata = try rowKernel.makeStreamedMetadataBuffers(device: device, routes: routes)
            var today = 0.0, rowOnly = 0.0, a7All = 0.0, a7OrRow = 0.0, production = 0.0
            for tile in routes.tiles {
                let groups = routes.groups[Int(tile.groupStart)..<Int(tile.groupStart + tile.groupCount)]
                let ids = groups.map { Int($0.expert) }
                let binding = try PrefillStreamedTileBinding(
                    expertIDs: ids,
                    views: PrefillGroupedRoutedMoETests.streamedViewsWithNonzeroOffsets(
                        device: device, pool: pool, expertIDs: ids.indices.map { $0 % 16 }))
                let params = PrefillGroupedRoutedMoEStreamedParams(
                    pairStart: tile.pairStart, pairCount: tile.pairCount, d: UInt32(d),
                    routedIntermediate: UInt32(f), topK: UInt32(topK), hiddenStrideElements: UInt32(d),
                    binding: binding, offsets: pool.offsets)
                var groupedParams = params
                groupedParams.pairStart = tile.groupStart
                groupedParams.pairCount = tile.groupCount
                let counts = groups.map { Int($0.pairCount) }
                let maxPairs = counts.max() ?? 0
                let rowArgument = try rowKernel.makeStreamedArgumentBuffer(device: device, binding: binding)
                let row = try best { cb in
                    _ = rowKernel.encodeStreamedBatched(
                        commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: metadata.sortedPairs,
                        routePartials: partials, gateUpActScratch: rowAct, downScratch: rowDown,
                        argumentBuffer: rowArgument, binding: binding, params: params, pairMicrobatchRows: 32)
                }
                var mppTime = row
                if mpp.isAvailable {
                    let mppArgument = try mpp.makeArgumentBuffer(device: device, binding: binding)
                    mppTime = try best { cb in
                        _ = mpp.encode(commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: metadata.sortedPairs,
                                       groups: metadata.groups, activation: groupedAct, routePartials: partials,
                                       argumentBuffer: mppArgument, binding: binding, params: groupedParams,
                                       maxPairsPerGroup: maxPairs)
                    }
                }
                let prototypeArgument = try prototype.makeArgumentBuffer(device: device, binding: binding)
                let a7 = try best { cb in
                    _ = prototype.encode(commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: metadata.sortedPairs,
                                         groups: metadata.groups, activation: groupedAct, routePartials: partials,
                                         argumentBuffer: prototypeArgument, binding: binding,
                                         params: groupedParams, maxPairsPerGroup: maxPairs)
                }
                let real = counts.reduce(0, +)
                let padded = counts.reduce(0) { $0 + ($1 + 63) / 64 * 64 }
                rowOnly += row
                let before = Double(padded) <= 1.5 * Double(real) ? mppTime : row
                today += before
                production += PrefillGroupedExpertGate.usesSimdgroupGroupedGEMM(pairCounts: counts) ? a7 : before
                a7All += a7
                a7OrRow += min(a7, row)
            }
            print(String(format: "[a7-chunk] tokens=%d %@ tiles=%d row-only %.1f ms | today(1.5x) %.1f ms | a7 all %.1f ms (%+.1f%% vs today) | a7-or-row %.1f ms (%+.1f%%) | production rule %.1f ms (%+.1f%%)",
                         tokens, skewed ? "skewed" : "uniform", routes.tiles.count, rowOnly * 1000, today * 1000,
                         a7All * 1000, (a7All / today - 1) * 100, a7OrRow * 1000, (a7OrRow / today - 1) * 100,
                         production * 1000, (production / today - 1) * 100))
        }
    }
}
