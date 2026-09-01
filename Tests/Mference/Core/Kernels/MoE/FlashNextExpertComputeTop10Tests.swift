import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// Flash-Next's routed **expert compute** at 512 experts / top-10.
///
/// `RouterWideTopK10Tests` gated the selection; this suite gates what runs on the
/// ten experts the selection names, in both dtypes an install can carry:
///
/// 1. **INT4 affine group-64** — the production install, through the shipped
///    `moe_phase1_gate_up_act_u16load` + the new `moe_phase2_down_reduce_k10`.
/// 2. **Dense BF16** — the parity install (its `moe_intermediate_size` is 32,
///    which group-64 cannot quantize), through `flashnext_moe_phase1_gate_up_bf16`
///    + `flashnext_moe_phase2_down_reduce_bf16`.
///
/// Both are compared against `FlashNextExpertReference`, the CPU reference
/// forward's own expert block (tied back to `FlashNextReferenceRunner` by
/// `FlashNextExpertReferenceTieBackTests`). The reference is fed the **decoded**
/// weights in each case, so what is being measured is arithmetic and reduction
/// order, not the quantizer — `Int4AffineEncoderParityTests` owns that.
///
/// The end of the suite pins the property that makes the widening safe: with the
/// two extra ranks weighted zero, `moe_phase2_down_reduce_k10` is **bit-identical**
/// to the shipped `moe_phase2_down_reduce_k8` on the same bytes. The k6/k8 kernels
/// themselves are untouched; `MoEFusedFFNTests` remains their byte gate.
@Suite struct FlashNextExpertComputeTop10Tests {

    /// Router dimension. A multiple of the INT4 group size, as
    /// `MoE.encodeRouterGemma4` requires, and of the expert `D`.
    private static let dimension = 128
    /// Per-expert FFN width. Also group-divisible, so the down projection's
    /// INT4 rows are legal.
    private static let intermediate = 64
    private static let numExperts = 512
    private static let topK = 10

    // MARK: - Fixtures

    /// Expert `e`'s float32 weights, drawn from a per-expert seed so all 512 are
    /// well-defined while only the ten the router picks are ever materialized.
    private static func expertWeights(_ e: Int)
        -> (gate: [[Float]], up: [[Float]], down: [[Float]]) {
        var rng = SplitMix64(seed: 0xE7E7_0000 &+ UInt64(e))
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in (0..<columns).map { _ in rng.uniform(-0.4, 0.4) } }
        }
        return (matrix(rows: intermediate, columns: dimension),
                matrix(rows: intermediate, columns: dimension),
                matrix(rows: dimension, columns: intermediate))
    }

    private struct Blob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
        /// What the kernel will actually read, decoded back to float32.
        let decoded: FlashNextExpertReference.Expert
    }

    private static func int4Blob(_ e: Int) -> Blob {
        let w = expertWeights(e)
        func packed(_ rows: [[Float]])
            -> (rows: [Quantization.Int4AffineRow], decoded: [Float]) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
            let decoded = quantized.flatMap {
                Quantization.dequantizeInt4Affine($0, n: $0.packed.count * 2)
            }
            return (quantized, decoded)
        }
        let g = packed(w.gate), u = packed(w.up), d = packed(w.down)
        var bytes = [UInt8]()
        func appendBytes(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func appendHalves(_ values: [UInt16]) {
            for v in values {
                bytes.append(UInt8(truncatingIfNeeded: v))
                bytes.append(UInt8(truncatingIfNeeded: v >> 8))
            }
        }
        let gW = UInt32(bytes.count); appendBytes(g.rows.flatMap(\.packed))
        let gS = UInt32(bytes.count); appendHalves(g.rows.flatMap(\.scales))
        let gB = UInt32(bytes.count); appendHalves(g.rows.flatMap(\.biases))
        let uW = UInt32(bytes.count); appendBytes(u.rows.flatMap(\.packed))
        let uS = UInt32(bytes.count); appendHalves(u.rows.flatMap(\.scales))
        let uB = UInt32(bytes.count); appendHalves(u.rows.flatMap(\.biases))
        let dW = UInt32(bytes.count); appendBytes(d.rows.flatMap(\.packed))
        let dS = UInt32(bytes.count); appendHalves(d.rows.flatMap(\.scales))
        let dB = UInt32(bytes.count); appendHalves(d.rows.flatMap(\.biases))
        return Blob(
            bytes: bytes,
            offsets: MoEExpertOffsets(gateWOff: gW, gateSOff: gS, gateBOff: gB,
                                      upWOff: uW, upSOff: uS, upBOff: uB,
                                      downWOff: dW, downSOff: dS, downBOff: dB),
            decoded: .init(gate: g.decoded, up: u.decoded, down: d.decoded))
    }

    /// The BF16-passthrough layout: three dense sub-tensors, no companion
    /// scale/bias slices — exactly what `FlashNextParity.installToyCheckpoint()`
    /// writes and what `FlashNextWeights.expert` detects by the missing
    /// `_scales` entry.
    private static func bf16Blob(_ e: Int) -> Blob {
        let w = expertWeights(e)
        var bytes = [UInt8]()
        func append(_ rows: [[Float]]) -> [Float] {
            var decoded = [Float]()
            for row in rows {
                for v in row {
                    let bits = Quantization.bf16Bits(v)
                    bytes.append(UInt8(truncatingIfNeeded: bits))
                    bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
                    decoded.append(Quantization.bf16ToFloat(bits))
                }
            }
            return decoded
        }
        let gW = UInt32(bytes.count); let g = append(w.gate)
        let uW = UInt32(bytes.count); let u = append(w.up)
        let dW = UInt32(bytes.count); let d = append(w.down)
        return Blob(
            bytes: bytes,
            offsets: MoEExpertOffsets(gateWOff: gW, gateSOff: 0, gateBOff: 0,
                                      upWOff: uW, upSOff: 0, upBOff: 0,
                                      downWOff: dW, downSOff: 0, downBOff: 0),
            decoded: .init(gate: g, up: u, down: d))
    }

    /// The router's own selection at the real width: an INT4 `[512, 128]` weight
    /// driven through `MoE.encodeRouterGemma4` at top-10. Returns what the GPU
    /// picked, so the expert compute is gated on the selection the runtime would
    /// actually hand it rather than on a hand-written index list.
    private struct Routing {
        let indices: [Int]
        /// Read back as FP16 and widened — the exact values the reduce reads.
        let weights: [Float]
    }

    private static func routeOnGPU(context: MetalContext,
                                   hidden: [Float]) throws -> Routing {
        let kernel = try MoE(context: context, siluActivation: true,
                             specializedD: UInt32(dimension),
                             specializedF: UInt32(intermediate),
                             specializedNumExperts: UInt32(numExperts),
                             specializedTopK: UInt32(topK))
        var rows = [[Float]]()
        var rowRNG = SplitMix64(seed: 0xE7E7_1234)
        for _ in 0..<numExperts {
            rows.append((0..<dimension).map { _ in rowRNG.uniform(-0.5, 0.5) })
        }
        let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
        guard let packed = context.device.makeBuffer(
                  bytes: quantized.flatMap(\.packed),
                  length: quantized.count * dimension / 2,
                  options: .storageModeShared),
              let scales = context.device.makeBuffer(
                  bytes: quantized.flatMap(\.scales),
                  length: quantized.flatMap(\.scales).count * 2,
                  options: .storageModeShared),
              let biases = context.device.makeBuffer(
                  bytes: quantized.flatMap(\.biases),
                  length: quantized.flatMap(\.biases).count * 2,
                  options: .storageModeShared),
              let effective = context.device.makeBuffer(
                  bytes: (0..<dimension).map { _ in Quantization.bf16Bits(1) },
                  length: dimension * 2, options: .storageModeShared),
              let perExpert = context.device.makeBuffer(
                  bytes: (0..<numExperts).map { _ in Quantization.bf16Bits(1) },
                  length: numExperts * 2, options: .storageModeShared),
              let hiddenBuffer = Fp16Buffer.make(context.device, values: hidden),
              let indexBuffer = context.device.makeBuffer(
                  length: topK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = Fp16Buffer.make(context.device, count: topK),
              let cb = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        kernel.encodeRouterGemma4(
            commandBuffer: cb,
            weights: packed, scales: scales, biases: biases,
            hidden: hiddenBuffer,
            effectiveScale: effective, perExpertScale: perExpert,
            outIndices: indexBuffer, outWeights: weightBuffer,
            numExperts: UInt32(numExperts), d: UInt32(dimension),
            topK: UInt32(topK))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let indexPointer = indexBuffer.contents()
            .bindMemory(to: UInt32.self, capacity: topK)
        let indices = (0..<topK).map { Int(indexPointer[$0]) }
        #expect(Set(indices).count == topK, "the router picked a duplicate expert")
        #expect(indices.allSatisfy { $0 < numExperts })
        return Routing(indices: indices,
                       weights: Fp16Buffer.read(weightBuffer, count: topK))
    }

    // MARK: - 1. INT4 expert compute at 512 / top-10

    @Test func int4ExpertComputeAtTop10MatchesTheFlashNextReference() throws {
        let context = try MetalContext()
        var rng = SplitMix64(seed: 0xE7E7_0001)
        let x = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let routing = try Self.routeOnGPU(context: context, hidden: x)
        let blobs = routing.indices.map { Self.int4Blob($0) }

        let kernel = try MoE(context: context, siluActivation: true,
                             specializedD: UInt32(Self.dimension),
                             specializedF: UInt32(Self.intermediate),
                             specializedNumExperts: UInt32(Self.numExperts),
                             specializedTopK: UInt32(Self.topK))
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes, length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == Self.topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device,
                                                  values: routing.weights),
              let acts = Fp16Buffer.make(context.device,
                                         count: Self.topK * Self.intermediate),
              let out = Fp16Buffer.make(context.device, count: Self.dimension),
              let cb = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        let bound = routedBuffers.map { (buffer: $0, offset: 0) }
        let argBuffer = try #require(kernel.makeRoutedArgumentBuffer(
            routedBlobs: bound, topK: UInt32(Self.topK)))
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: bound,
            routedOffsets: blobs[0].offsets, x: xBuffer, acts: acts,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: bound,
            routedOffsets: blobs[0].offsets, acts: acts,
            routingWeights: routingBuffer, residual: residualBuffer, y: out,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let expected = FlashNextExpertReference.routedSum(
            experts: blobs.map(\.decoded), weights: routing.weights,
            x: x, residual: residual,
            hidden: Self.dimension, intermediate: Self.intermediate)
        let actual = Fp16Buffer.read(out, count: Self.dimension)
        let error = RelError.compute(actual: actual, reference: expected)
        print("flashnext INT4 top-10 expert compute: relative error \(error) "
                + "(max-abs \(RelError.maxAbsDiff(actual, expected)))")
        #expect(error < Tolerance.fp16ChainedReduction,
                "INT4 top-10 expert compute off by \(error)")
    }

    // MARK: - 2. BF16 expert compute at 512 / top-10

    @Test func bf16ExpertComputeAtTop10MatchesTheFlashNextReference() throws {
        let context = try MetalContext()
        var rng = SplitMix64(seed: 0xE7E7_0002)
        let x = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let routing = try Self.routeOnGPU(context: context, hidden: x)
        let blobs = routing.indices.map { Self.bf16Blob($0) }

        let kernel = try FlashNextMoE(context: context)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes, length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == Self.topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device,
                                                  values: routing.weights),
              let fullActs = Fp16Buffer.make(context.device,
                                             count: Self.topK * Self.intermediate),
              let splitActs = Fp16Buffer.make(context.device,
                                              count: Self.topK * Self.intermediate),
              let out = Fp16Buffer.make(context.device, count: Self.dimension),
              let splitOut = Fp16Buffer.make(context.device, count: Self.dimension),
              let cb = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        let bound = routedBuffers.map { (buffer: $0, offset: 0) }
        let argBuffer = kernel.makeRoutedArgumentBuffer(routedBlobs: bound)
        kernel.encodePhase1(
            commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: bound,
            routedOffsets: blobs[0].offsets, x: xBuffer, acts: fullActs,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        kernel.encodePhase2(
            commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: bound,
            routedOffsets: blobs[0].offsets, acts: fullActs,
            routingWeights: routingBuffer, residual: residualBuffer, y: out,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))

        // The hit/miss split the streaming plan produces must be a partition,
        // not a different computation: two subset dispatches covering all ten
        // ranks have to land on the same activations.
        let splitSlots: [[UInt32]] = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]
        for slots in splitSlots {
            guard let activeBuffer = context.device.makeBuffer(
                    bytes: slots, length: slots.count * MemoryLayout<UInt32>.stride,
                    options: .storageModeShared) else {
                Issue.record("active-slot buffer allocation failed")
                return
            }
            kernel.encodePhase1Subset(
                commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: bound,
                routedOffsets: blobs[0].offsets, x: xBuffer, acts: splitActs,
                activeSlots: activeBuffer, activeSlotIndices: slots,
                d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
                topK: UInt32(Self.topK))
        }
        kernel.encodePhase2(
            commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: bound,
            routedOffsets: blobs[0].offsets, acts: splitActs,
            routingWeights: routingBuffer, residual: residualBuffer, y: splitOut,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let expected = FlashNextExpertReference.routedSum(
            experts: blobs.map(\.decoded), weights: routing.weights,
            x: x, residual: residual,
            hidden: Self.dimension, intermediate: Self.intermediate)
        let actual = Fp16Buffer.read(out, count: Self.dimension)
        let split = Fp16Buffer.read(splitOut, count: Self.dimension)
        #expect(actual == split, "the hit/miss split diverged from the full pass")
        let error = RelError.compute(actual: actual, reference: expected)
        print("flashnext BF16 top-10 expert compute: relative error \(error) "
                + "(max-abs \(RelError.maxAbsDiff(actual, expected)))")
        #expect(error < Tolerance.fp16ChainedReduction,
                "BF16 top-10 expert compute off by \(error)")
    }

    // MARK: - 3. The k10 reduce is an extension of k8, not a rewrite

    /// With ranks 8 and 9 weighted zero, the k10 reduce adds two exact `+0.0`
    /// terms to the k8 sequence in the same FP32 accumulator. Bit equality on the
    /// same bytes is the property that makes widening `RoutedBlobs` safe for the
    /// shipped families — and it is checked here rather than assumed, because the
    /// two kernels are separate hand-unrolled bodies.
    @Test func k10ReduceWithTwoZeroRanksIsBitIdenticalToK8() throws {
        let context = try MetalContext()
        var rng = SplitMix64(seed: 0xE7E7_0003)
        let x = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let blobs = (0..<Self.topK).map { Self.int4Blob(100 + $0) }
        let weightsK10 = (0..<Self.topK).map { rank -> Float in
            rank < 8 ? Float(Float16(0.05 + Float(rank) * 0.02)) : 0
        }

        func run(topK: Int, weights: [Float]) throws -> [Float] {
            let kernel = try MoE(context: context, siluActivation: true,
                                 specializedD: UInt32(Self.dimension),
                                 specializedF: UInt32(Self.intermediate),
                                 specializedNumExperts: UInt32(Self.numExperts),
                                 specializedTopK: UInt32(topK))
            let buffers = blobs.prefix(topK).compactMap {
                context.device.makeBuffer(bytes: $0.bytes, length: $0.bytes.count,
                                          options: .storageModeShared)
            }
            guard buffers.count == topK,
                  let xBuffer = Fp16Buffer.make(context.device, values: x),
                  let residualBuffer = Fp16Buffer.make(context.device,
                                                       values: residual),
                  let routingBuffer = Fp16Buffer.make(context.device,
                                                      values: weights),
                  let acts = Fp16Buffer.make(context.device,
                                             count: topK * Self.intermediate),
                  let out = Fp16Buffer.make(context.device, count: Self.dimension),
                  let cb = context.queue.makeCommandBuffer() else {
                throw CocoaError(.fileReadUnknown)
            }
            let bound = buffers.map { (buffer: $0, offset: 0) }
            let argBuffer = try #require(kernel.makeRoutedArgumentBuffer(
                routedBlobs: bound, topK: UInt32(topK)))
            kernel.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: bound,
                routedOffsets: blobs[0].offsets, x: xBuffer, acts: acts,
                d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
                topK: UInt32(topK))
            kernel.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: cb, routedArgBuffer: argBuffer, routedBlobs: bound,
                routedOffsets: blobs[0].offsets, acts: acts,
                routingWeights: routingBuffer, residual: residualBuffer, y: out,
                d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
                topK: UInt32(topK))
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            return Fp16Buffer.read(out, count: Self.dimension)
        }

        let eight = try run(topK: 8, weights: Array(weightsK10.prefix(8)))
        let ten = try run(topK: 10, weights: weightsK10)
        #expect(eight == ten,
                "k10 with two zero-weight ranks must be bit-identical to k8")
    }
}
