import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The router widening for `qwen38flashnext`: 512 experts, top-10.
///
/// Three things are gated here, in increasing scope:
///
/// 1. `router_topk_select_k10_par` is bit-identical to the serial
///    `router_topk_select_k10` (the same contract `RouterTopKParityTests` holds
///    the shipped k6/k8 kernels to), including at expert counts above 256 where
///    the wide 16-per-lane array is what makes the dispatch legal at all.
/// 2. Both k10 kernels reproduce `FlashNextRouterReference` — the CPU reference
///    forward's own router — with **exact** indices and FP16-close weights.
/// 3. The whole decode router (`MoE.encodeRouterGemma4`) and the prefill router
///    (`PrefillRouter.encodeGemma4Block`) agree with that reference, and with
///    each other, at the production 512 x 2560 shape.
///
/// The shipped k8 path is deliberately not re-derived here: `RouterTopKParityTests`
/// is its byte gate and must keep passing unchanged.
@Suite struct RouterWideTopK10Tests {

    private static let topK = 10

    private struct Selection: Equatable {
        let indices: [UInt32]
        let weightBits: [UInt16]
    }

    private enum Fixture {
        /// Spans the shipped bound (<= 256), the boundary, and the new wide
        /// range up to 512, plus counts that are not multiples of 32 so the
        /// `-INFINITY` sentinel lanes are exercised. `10` and `11` sit at and
        /// just above K, where trailing selection steps can run out of experts.
        static let expertCounts: [UInt32] = [10, 11, 24, 33, 100, 256, 257, 320, 511, 512]
        static let trials = 12
    }

    // MARK: - 1. Parallel k10 == serial k10, bit for bit

    @Test("Parallel k10 select matches the serial kernel bit for bit",
          arguments: Fixture.expertCounts)
    func k10SelectMatchesSerial(numExperts: UInt32) throws {
        let context = try MetalContext()
        let serial = try context.pipeline("router_topk_select_k10")
        let parallel = try context.pipeline("router_topk_select_k10_par")
        var rng = SplitMix64(seed: 0x5EED_0010 &+ UInt64(numExperts))

        for trial in 0..<Fixture.trials {
            let logits = Self.tieProneLogits(count: Int(numExperts),
                                             trial: trial, rng: &rng)
            let scale = (0..<Int(numExperts)).map { _ in rng.uniform(0.6, 1.4) }
            let expected = try Self.runSelect(context: context, pipeline: serial,
                                              logits: logits, expertScale: scale)
            let actual = try Self.runSelect(context: context, pipeline: parallel,
                                            logits: logits, expertScale: scale)
            #expect(actual == expected, "numExperts=\(numExperts) trial=\(trial)")
        }
    }

    /// The wide kernel must refuse rather than corrupt memory beyond its lane
    /// budget. 512 is the contract; 513 is out of contract and is a documented
    /// no-op (`sg_idx != 0 || NE > 32 * MAXPERLANE`).
    @Test func k10SelectNoOpsAboveItsExpertBound() throws {
        let context = try MetalContext()
        let parallel = try context.pipeline("router_topk_select_k10_par")
        var rng = SplitMix64(seed: 0x5EED_0011)
        let logits = (0..<600).map { _ in rng.uniform(-4.0, 4.0) }
        let scale = [Float](repeating: 1, count: 600)
        let result = try Self.runSelect(context: context, pipeline: parallel,
                                        logits: logits, expertScale: scale,
                                        sentinelIndex: 0xDEAD_BEEF)
        #expect(result.indices == [UInt32](repeating: 0xDEAD_BEEF, count: Self.topK),
                "out-of-contract dispatch must not write")
    }

    // MARK: - 2. k10 kernels == the CPU reference forward's router

    /// The reference's `descendingTopK` short-circuits at `k >= n` with
    /// `a.map(\.index)` — the identity order, *not* a descending sort. It is a
    /// divergence from `torch.topk`, and it is unobservable in the reference
    /// runner: the MoE always has `numExperts (8, and 512 in production) > topK`,
    /// and the indexer's `k = min(blockTopK, completeBlocks)` reaches `k == n`
    /// only when every block is selected, after which the chosen positions are
    /// sorted anyway. The Metal kernels always emit descending order, so the two
    /// agree as sets but not as sequences in this corner. Pinned here so the
    /// exclusion below is a known property rather than a tuned-away failure.
    @Test func referenceTopKReturnsIdentityOrderWhenKCoversEveryExpert() throws {
        let logits: [Float] = [0.1, 0.9, 0.5, 0.7]
        let all = FlashNextRouterReference.descendingTopK(logits, k: 4)
        #expect(all == [0, 1, 2, 3], "k >= n short-circuits to identity order")
        let some = FlashNextRouterReference.descendingTopK(logits, k: 3)
        #expect(some == [1, 3, 2], "k < n ranks by score")
    }

    @Test("k10 selection reproduces the Flash-Next CPU reference exactly",
          arguments: Fixture.expertCounts.filter { $0 > UInt32(topK) })
    func k10SelectionMatchesTheFlashNextReference(numExperts: UInt32) throws {
        let context = try MetalContext()
        let serial = try context.pipeline("router_topk_select_k10")
        let parallel = try context.pipeline("router_topk_select_k10_par")
        var rng = SplitMix64(seed: 0x5EED_0012 &+ UInt64(numExperts))
        let ones = [Float](repeating: 1, count: Int(numExperts))

        for trial in 0..<Fixture.trials {
            // Continuous logits only: with a tied k/k+1 boundary the two
            // implementations' *ordering policies* — not their arithmetic —
            // would decide the answer, and that is not what this gate is about.
            // The assertion below makes the assumption explicit.
            let logits = (0..<Int(numExperts)).map { _ in rng.uniform(-6.0, 6.0) }
            let expected = FlashNextRouterReference.select(logits: logits,
                                                           k: Self.topK)
            let gap = FlashNextRouterReference.boundaryGap(logits: logits,
                                                           k: Self.topK)
            #expect(gap > 0, "fixture produced a tied top-k boundary")

            for (name, pipeline) in [("serial", serial), ("parallel", parallel)] {
                let actual = try Self.runSelect(context: context, pipeline: pipeline,
                                                logits: logits, expertScale: ones)
                #expect(actual.indices.map(Int.init) == expected.indices,
                        "\(name) NE=\(numExperts) trial=\(trial) indices")
                let weights = actual.weightBits.map { Float(Float16(bitPattern: $0)) }
                let worst = zip(weights, expected.weights)
                    .map { abs($0 - $1) }.max() ?? 0
                #expect(worst < 1e-3,
                        "\(name) NE=\(numExperts) trial=\(trial) weights off by \(worst)")
                #expect(abs(weights.reduce(0, +) - 1) < 5e-3,
                        "\(name) NE=\(numExperts) renormalized weights must sum to 1")
            }
        }
    }

    // MARK: - 3. Whole decode router at the production 512 x 2560 shape

    @Test func decodeRouterAt512TopK10MatchesTheReference() throws {
        let fixture = try Self.productionFixture(seed: 0x5EED_0013)
        let actual = try Self.runDecodeRouter(fixture)
        #expect(actual.indices.map(Int.init) == fixture.expected.indices,
                "512/top-10 decode router indices")
        let worst = zip(actual.weightBits.map { Float(Float16(bitPattern: $0)) },
                        fixture.expected.weights).map { abs($0 - $1) }.max() ?? 0
        #expect(worst < 2e-3, "512/top-10 decode router weights off by \(worst)")
    }

    @Test func prefillRouterAt512TopK10MatchesTheReferenceAndDecode() throws {
        let fixture = try Self.productionFixture(seed: 0x5EED_0014)
        let decode = try Self.runDecodeRouter(fixture)
        let prefill = try Self.runPrefillRouter(fixture)
        #expect(prefill.indices.map(Int.init) == fixture.expected.indices,
                "512/top-10 prefill router indices")
        #expect(prefill.indices == decode.indices,
                "prefill and decode routers must select the same experts")
        let worst = zip(prefill.weightBits.map { Float(Float16(bitPattern: $0)) },
                        fixture.expected.weights).map { abs($0 - $1) }.max() ?? 0
        #expect(worst < 2e-3, "512/top-10 prefill router weights off by \(worst)")
    }

    // MARK: - Production-shape fixture

    private struct ProductionFixture {
        let dimension: Int
        let experts: Int
        let packed: [UInt8]
        let scales: [UInt16]
        let biases: [UInt16]
        let hidden: [Float]
        let effectiveScale: [Float]
        let expertScale: [Float]
        let expected: FlashNextRouterReference.Selection
    }

    /// Flash-Next's real router shape: `[512, 2560]`, INT8 affine group-64 — the
    /// dtype `router_gemv_gemma4_r4` and `prefill_router_gemma4_block` both
    /// consume.
    ///
    /// # Why the rows are designed rather than random
    ///
    /// The three sides being compared reduce 2560 products in three different
    /// orders: the CPU reference sequentially, the decode GEMV through a 32-lane
    /// `simd_sum` tree, and the prefill router group-by-group on one thread.
    /// They agree to float32 rounding, not bit for bit — so *index* equality is
    /// only a real property when consecutive logits are separated by more than
    /// that rounding. Random rows do not guarantee that: the first cut of this
    /// fixture drew i.i.d. rows and produced adjacent logits closer than the
    /// summation noise, which flipped ranks 2/3 between the CPU and the GPU
    /// (same set, different order) and moved the 10th pick outright.
    ///
    /// So the rows are `pattern * gain[e]`, with `pattern` sign-aligned to the
    /// scaled activation. That makes `logit(e) = gain[e] * (pattern . x)` with a
    /// large, positive common factor, and the gains an arithmetic sequence of
    /// 512 distinct values under a fixed permutation. Separation is then a
    /// property of the construction, and `separation` below reports the measured
    /// margin against the observed disagreement threshold.
    private static func productionFixture(seed: UInt64) throws -> ProductionFixture {
        let dimension = 2560
        let experts = 512
        var rng = SplitMix64(seed: seed)
        let hidden = (0..<dimension).map { _ in rng.uniform(-1.0, 1.0) }
        let invSqrtD = 1.0 / Float(dimension).squareRoot()
        let effectiveScale = (0..<dimension).map { _ in rng.uniform(0.7, 1.3) * invSqrtD }
        let scaled = zip(hidden, effectiveScale).map { $0 * $1 }
        // Sign-aligned so `pattern . scaled` is a sum of magnitudes rather than
        // a random walk: a large common factor, so a small gain gap is still a
        // wide logit gap.
        let pattern = scaled.map { Float($0 < 0 ? -0.6 : 0.6) }
        // The gain steps have to survive the quantizer, not just float32. Each
        // row's affine scale and bias are stored **BF16** (8 mantissa bits), so
        // gains closer than ~0.4% relative collapse onto the same stored pair
        // and produce byte-identical rows — exact logit ties. Spreading 512
        // gains evenly over a single binade did exactly that, and the measured
        // top-11 separation came out 0.0.
        //
        // So the twelve experts that decide the outcome get gains a full 3%
        // apart, scattered across the expert axis in a scrambled rank order, and
        // the other 500 sit below them. Ranks 10 and 11 are deliberately near
        // misses, so the k-th/k+1-th comparison is a real one.
        let winners = [7, 41, 99, 128, 173, 200, 255, 301, 366, 400, 455, 511]
        var gains = (0..<experts).map { e -> Float in
            0.1 + 0.8 * Float((e &* 331) % experts) / Float(experts)
        }
        for rank in 0..<winners.count {
            gains[winners[(rank &* 5) % winners.count]] = 1.33 - 0.03 * Float(rank)
        }
        let rows = gains.map { gain in pattern.map { $0 * gain } }
        let expertScale = [Float](repeating: 1, count: experts)

        let quantized = rows.map { Quantization.quantizeInt8Affine($0) }
        // The reference reads the SAME quantized bytes the kernels do, so the
        // only difference between the two sides is float summation order.
        let logits = DequantInt8GemvRef.apply(weightRows: quantized,
                                              x: scaled, n: dimension)
        let expected = FlashNextRouterReference.select(logits: logits, k: topK)
        let separation = minimumAdjacentGap(logits: logits, depth: topK + 1)
        print("router 512x2560 fixture: top-\(topK + 1) minimum adjacent logit gap "
                + "\(separation) (float32 summation-order noise is ~1e-4 here)")
        #expect(separation > 1e-2,
                Comment(rawValue: "fixture logits are only \(separation) apart at the "
                            + "selection boundary — index equality across three "
                            + "summation orders would not be a real property"))

        return ProductionFixture(dimension: dimension,
                                 experts: experts,
                                 packed: quantized.flatMap(\.packed),
                                 scales: quantized.flatMap(\.scales),
                                 biases: quantized.flatMap(\.biases),
                                 hidden: hidden,
                                 effectiveScale: effectiveScale,
                                 expertScale: expertScale,
                                 expected: expected)
    }

    /// The smallest gap between consecutive values among the `depth` largest
    /// logits — the margin every rank in the emitted top-k has to spare.
    private static func minimumAdjacentGap(logits: [Float], depth: Int) -> Float {
        let ranked = logits.sorted(by: >).prefix(min(depth, logits.count))
        var worst = Float.infinity
        for (a, b) in zip(ranked, ranked.dropFirst()) { worst = min(worst, a - b) }
        return worst
    }

    // MARK: - Dispatch helpers

    private static func runSelect(context: MetalContext,
                                  pipeline: MTLComputePipelineState,
                                  logits: [Float],
                                  expertScale: [Float],
                                  sentinelIndex: UInt32? = nil) throws -> Selection {
        guard let logitBuffer = context.device.makeBuffer(
                  bytes: logits,
                  length: logits.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let scaleBuffer = context.device.makeBuffer(
                  bytes: expertScale.map(Quantization.bf16Bits),
                  length: expertScale.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let indexBuffer = context.device.makeBuffer(
                  bytes: [UInt32](repeating: sentinelIndex ?? 0, count: topK),
                  length: topK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = Fp16Buffer.make(context.device, count: topK),
              let commandBuffer = context.queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw CocoaError(.fileReadUnknown)
        }
        var expertCount = UInt32(logits.count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logitBuffer, offset: 0, index: 0)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
        encoder.setBuffer(indexBuffer, offset: 0, index: 2)
        encoder.setBuffer(weightBuffer, offset: 0, index: 3)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        return Selection(indices: readIndices(indexBuffer, count: topK),
                         weightBits: readWeightBits(weightBuffer, count: topK))
    }

    private static func runDecodeRouter(_ f: ProductionFixture) throws -> Selection {
        let context = try MetalContext()
        let kernel = try MoE(context: context, siluActivation: true)
        guard let buffers = try makeWeightBuffers(context, f),
              let hiddenBuffer = Fp16Buffer.make(context.device, values: f.hidden),
              let indexBuffer = context.device.makeBuffer(
                  length: topK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = Fp16Buffer.make(context.device, count: topK),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        kernel.encodeRouterGemma4(
            commandBuffer: commandBuffer,
            weights: buffers.packed, scales: buffers.scales, biases: buffers.biases,
            hidden: hiddenBuffer,
            effectiveScale: buffers.effective, perExpertScale: buffers.expert,
            outIndices: indexBuffer, outWeights: weightBuffer,
            numExperts: UInt32(f.experts), d: UInt32(f.dimension),
            topK: UInt32(topK))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        return Selection(indices: readIndices(indexBuffer, count: topK),
                         weightBits: readWeightBits(weightBuffer, count: topK))
    }

    private static func runPrefillRouter(_ f: ProductionFixture) throws -> Selection {
        let context = try MetalContext()
        let router = try PrefillRouter(context: context)
        guard let buffers = try makeWeightBuffers(context, f),
              let hiddenBuffer = Fp16Buffer.make(context.device, values: f.hidden),
              let indexBuffer = context.device.makeBuffer(
                  length: topK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = Fp16Buffer.make(context.device, count: topK),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        router.encodeGemma4Block(
            commandBuffer: commandBuffer,
            weights: buffers.packed, scales: buffers.scales, biases: buffers.biases,
            hidden: hiddenBuffer,
            effectiveScale: buffers.effective, perExpertScale: buffers.expert,
            outIndices: indexBuffer, outWeights: weightBuffer,
            queryCount: 1, numExperts: UInt32(f.experts), d: UInt32(f.dimension),
            topK: UInt32(topK), hiddenStrideElements: UInt32(f.dimension))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        return Selection(indices: readIndices(indexBuffer, count: topK),
                         weightBits: readWeightBits(weightBuffer, count: topK))
    }

    private struct WeightBuffers {
        let packed: MTLBuffer
        let scales: MTLBuffer
        let biases: MTLBuffer
        let effective: MTLBuffer
        let expert: MTLBuffer
    }

    private static func makeWeightBuffers(_ context: MetalContext,
                                          _ f: ProductionFixture) throws
        -> WeightBuffers? {
        guard let packed = context.device.makeBuffer(
                  bytes: f.packed, length: f.packed.count,
                  options: .storageModeShared),
              let scales = context.device.makeBuffer(
                  bytes: f.scales,
                  length: f.scales.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let biases = context.device.makeBuffer(
                  bytes: f.biases,
                  length: f.biases.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let effective = context.device.makeBuffer(
                  bytes: f.effectiveScale.map(Quantization.bf16Bits),
                  length: f.effectiveScale.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let expert = context.device.makeBuffer(
                  bytes: f.expertScale.map(Quantization.bf16Bits),
                  length: f.expertScale.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared) else { return nil }
        return WeightBuffers(packed: packed, scales: scales, biases: biases,
                             effective: effective, expert: expert)
    }

    // MARK: - Fixtures

    /// Mirrors `RouterTopKParityTests.logits`: continuous, coarsely quantized,
    /// three-level and constant logits, so exact ties at the selection boundary
    /// are common and the tie rule is actually exercised.
    private static func tieProneLogits(count: Int, trial: Int,
                                       rng: inout SplitMix64) -> [Float] {
        switch trial % 4 {
        case 0:
            return (0..<count).map { _ in rng.uniform(-4.0, 4.0) }
        case 1:
            return (0..<count).map { _ in (rng.uniform(-3.0, 3.0) * 4).rounded() / 4 }
        case 2:
            let levels: [Float] = [1.5, 0.5, -0.5]
            return (0..<count).map { _ in levels[Int(rng.next() % 3)] }
        default:
            return [Float](repeating: 0.75, count: count)
        }
    }

    private static func readIndices(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
        let pointer = buffer.contents().bindMemory(to: UInt32.self, capacity: count)
        return (0..<count).map { pointer[$0] }
    }

    private static func readWeightBits(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
        let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
        return (0..<count).map { pointer[$0] }
    }
}
