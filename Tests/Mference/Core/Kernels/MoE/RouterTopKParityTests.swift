import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The production router selects run one SIMD wide (`*_par`); the original
/// single-thread kernels stay in `moe.metal` purely as the parity reference.
/// These tests dispatch both over the same logits and require identical
/// indices, identical order, and bit-equal FP16 weights — the parallel kernels
/// are a scheduling change only, never a numerical one.
@Suite struct RouterTopKParityTests {
    private struct Selection: Equatable {
        let indices: [UInt32]
        let weightBits: [UInt16]
    }

    private enum Fixture {
        /// Expert counts spanning the production shapes (128 Gemma/Qwen, 256
        /// DeepSeek V4) plus non-multiple-of-32 and short counts that exercise
        /// the -INFINITY sentinel lanes.
        static let expertCounts: [UInt32] = [8, 16, 24, 33, 100, 128, 168, 255, 256]
        static let trials = 12
    }

    // MARK: - k8 (Gemma 4 / Qwen 3.6)

    @Test("Parallel k8 select matches the serial kernel bit for bit",
          arguments: Fixture.expertCounts)
    func k8SelectMatchesSerial(numExperts: UInt32) throws {
        let context = try MetalContext()
        let serial = try context.pipeline("router_topk_select_k8")
        let parallel = try context.pipeline("router_topk_select_k8_par")
        var rng = SplitMix64(seed: 0x5EED_0001 &+ UInt64(numExperts))

        for trial in 0..<Fixture.trials {
            let logits = Self.logits(count: Int(numExperts),
                                     trial: trial,
                                     rng: &rng)
            let scale = (0..<Int(numExperts)).map { _ in rng.uniform(0.6, 1.4) }
            let expected = try Self.runK8(context: context, pipeline: serial,
                                          logits: logits, expertScale: scale)
            let actual = try Self.runK8(context: context, pipeline: parallel,
                                        logits: logits, expertScale: scale)
            #expect(actual == expected,
                    "numExperts=\(numExperts) trial=\(trial)")
        }
    }

    // MARK: - k6 sqrtsoftplus (DeepSeek V4 learned routing)

    @Test("Parallel sqrtsoftplus k6 select matches the serial kernel bit for bit",
          arguments: Fixture.expertCounts)
    func k6SelectMatchesSerial(numExperts: UInt32) throws {
        let context = try MetalContext()
        let serial = try context.pipeline("router_topk_select_sqrtsoftplus_k6")
        let parallel = try context.pipeline(
            "router_topk_select_sqrtsoftplus_k6_par")
        var rng = SplitMix64(seed: 0x5EED_0002 &+ UInt64(numExperts))

        for trial in 0..<Fixture.trials {
            let logits = Self.logits(count: Int(numExperts),
                                     trial: trial,
                                     rng: &rng)
            // Duplicated correction biases add a second source of exact ties in
            // the selection score.
            let bias = (0..<Int(numExperts)).map { index -> Float in
                trial % 3 == 0 ? Float((index / 7) % 4) * 0.25 : rng.uniform(-0.3, 0.3)
            }
            let expected = try Self.runK6(context: context, pipeline: serial,
                                          logits: logits, correctionBias: bias)
            let actual = try Self.runK6(context: context, pipeline: parallel,
                                        logits: logits, correctionBias: bias)
            #expect(actual == expected,
                    "numExperts=\(numExperts) trial=\(trial)")
        }
    }

    // MARK: - k6 hash weighting (DeepSeek V4 frozen routing)

    @Test("Parallel hash weights match the serial kernel bit for bit")
    func hashWeightsMatchSerial() throws {
        let context = try MetalContext()
        let serial = try context.pipeline("router_hash_weights_k6")
        let parallel = try context.pipeline("router_hash_weights_k6_par")
        var rng = SplitMix64(seed: 0x5EED_0003)
        let numExperts = 256

        for trial in 0..<Fixture.trials {
            let logits = Self.logits(count: numExperts, trial: trial, rng: &rng)
            // Repeated experts are legal in a frozen tid2eid row.
            let indices = (0..<6).map { slot -> UInt32 in
                trial % 4 == 0
                    ? UInt32((slot % 2) * 17)
                    : UInt32(rng.next() % UInt64(numExperts))
            }
            let expected = try Self.runHash(context: context, pipeline: serial,
                                            logits: logits, indices: indices)
            let actual = try Self.runHash(context: context, pipeline: parallel,
                                          logits: logits, indices: indices)
            #expect(actual == expected, "trial=\(trial)")
        }
    }

    // MARK: - Fixtures

    /// Trials deliberately mix continuous logits with heavily quantized ones so
    /// exact ties (the only case where tie-breaking is observable) are common.
    private static func logits(count: Int,
                               trial: Int,
                               rng: inout SplitMix64) -> [Float] {
        switch trial % 4 {
        case 0:
            return (0..<count).map { _ in rng.uniform(-4.0, 4.0) }
        case 1:
            // Coarse quantization: many exact duplicates.
            return (0..<count).map { _ in (rng.uniform(-3.0, 3.0) * 4).rounded() / 4 }
        case 2:
            // Only three distinct values; the top-k boundary always lands on a
            // tie run, so index order decides the outcome.
            let levels: [Float] = [1.5, 0.5, -0.5]
            return (0..<count).map { _ in levels[Int(rng.next() % 3)] }
        default:
            // Every logit identical: selection is pure index order.
            return [Float](repeating: 0.75, count: count)
        }
    }

    // MARK: - Dispatch helpers

    private static func runK8(context: MetalContext,
                              pipeline: MTLComputePipelineState,
                              logits: [Float],
                              expertScale: [Float]) throws -> Selection {
        let topK = 8
        guard let logitBuffer = context.device.makeBuffer(
                  bytes: logits,
                  length: logits.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let scaleBuffer = context.device.makeBuffer(
                  bytes: expertScale.map(Quantization.bf16Bits),
                  length: expertScale.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let indexBuffer = context.device.makeBuffer(
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

    private static func runK6(context: MetalContext,
                              pipeline: MTLComputePipelineState,
                              logits: [Float],
                              correctionBias: [Float]) throws -> Selection {
        let topK = 6
        guard let logitBuffer = context.device.makeBuffer(
                  bytes: logits,
                  length: logits.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let biasBuffer = context.device.makeBuffer(
                  bytes: correctionBias,
                  length: correctionBias.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let indexBuffer = context.device.makeBuffer(
                  length: topK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = Fp16Buffer.make(context.device, count: topK),
              let commandBuffer = context.queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw CocoaError(.fileReadUnknown)
        }
        var expertCount = UInt32(logits.count)
        var routeScale = Float(2.5)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logitBuffer, offset: 0, index: 0)
        encoder.setBuffer(biasBuffer, offset: 0, index: 1)
        encoder.setBuffer(indexBuffer, offset: 0, index: 2)
        encoder.setBuffer(weightBuffer, offset: 0, index: 3)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&routeScale, length: MemoryLayout<Float>.stride, index: 5)
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

    private static func runHash(context: MetalContext,
                                pipeline: MTLComputePipelineState,
                                logits: [Float],
                                indices: [UInt32]) throws -> Selection {
        let topK = 6
        guard let logitBuffer = context.device.makeBuffer(
                  bytes: logits,
                  length: logits.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let indexBuffer = context.device.makeBuffer(
                  bytes: indices,
                  length: indices.count * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = Fp16Buffer.make(context.device, count: topK),
              let commandBuffer = context.queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw CocoaError(.fileReadUnknown)
        }
        var routeScale = Float(2.5)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logitBuffer, offset: 0, index: 0)
        encoder.setBuffer(indexBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBytes(&routeScale, length: MemoryLayout<Float>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        return Selection(indices: indices,
                         weightBits: readWeightBits(weightBuffer, count: topK))
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
