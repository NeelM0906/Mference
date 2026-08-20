import Foundation
import Metal
import Testing

@testable import Mference
import MferenceValidationSupport

/// The Inkling pilot router's whole value is ranking fidelity: it runs the
/// same GEMV and select kernels as the real router, so on identical inputs
/// its predicted expert ids must be bit-identical to the real readback.
@Suite struct SpeculativeRouterInklingTests {

    private static let numRouted = 256
    private static let numShared = 2
    private static let topK = 6
    private static let d = 64

    private static func bf16(_ v: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: v.bitPattern >> 16)
    }

    private static func buffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
        values.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                              options: .storageModeShared)!
        }
    }

    @Test func predictionMatchesRealRouterBitForBit() throws {
        let context = try MetalContext()
        let device = context.device
        let kernels = try InklingKernels(context: context,
                                         numRouted: Self.numRouted,
                                         numShared: Self.numShared)
        let pilot = try SpeculativeRouterInkling(context: context,
                                                 numRouted: Self.numRouted,
                                                 numShared: Self.numShared,
                                                 topK: Self.topK)

        var rng = SplitMix64(seed: 0x1_2C0F)
        let total = Self.numRouted + Self.numShared
        let weights = Self.buffer(
            device,
            (0..<total * Self.d).map { _ in Self.bf16(rng.uniform(0, 1) * 2 - 1) })
        let hidden = Self.buffer(
            device,
            (0..<Self.d).map { _ in Float16(rng.uniform(0, 1) * 2 - 1) })
        let onesScale = Self.buffer(
            device,
            (0..<Self.d).map { _ in Self.bf16(rng.uniform(0, 1) + 0.5) })
        let gateBias = Self.buffer(
            device,
            (0..<total).map { _ in rng.uniform(0, 1) * 0.2 - 0.1 })
        let globalScale = Self.buffer(device, [Float(1.25)])
        let realIndices = Self.buffer(device,
                                      [UInt32](repeating: .max, count: Self.topK))
        let realWeights = Self.buffer(device,
                                      [Float16](repeating: 0, count: 8))
        let routeScale: Float = 1.7

        let cb = context.queue.makeCommandBuffer()!
        kernels.encodeRouter(commandBuffer: cb,
                             weights: weights, weightsOffset: 0,
                             hidden: hidden,
                             onesScale: onesScale,
                             gateBias: gateBias, gateBiasOffset: 0,
                             globalScale: globalScale, globalScaleOffset: 0,
                             outIndices: realIndices,
                             outWeights: realWeights,
                             numRouted: UInt32(Self.numRouted),
                             numShared: UInt32(Self.numShared),
                             topK: UInt32(Self.topK),
                             routeScale: routeScale,
                             d: UInt32(Self.d))
        pilot.encodePrediction(commandBuffer: cb,
                               weights: weights, weightsOffset: 0,
                               hidden: hidden,
                               onesScale: onesScale,
                               gateBias: gateBias, gateBiasOffset: 0,
                               globalScale: globalScale, globalScaleOffset: 0,
                               numRouted: UInt32(Self.numRouted),
                               numShared: UInt32(Self.numShared),
                               topK: UInt32(Self.topK),
                               routeScale: routeScale,
                               d: UInt32(Self.d))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let real = realIndices.contents()
            .bindMemory(to: UInt32.self, capacity: Self.topK)
        let predicted = pilot.predictedIndices.contents()
            .bindMemory(to: UInt32.self, capacity: Self.topK)
        let realIds = (0..<Self.topK).map { real[$0] }
        let predictedIds = (0..<Self.topK).map { predicted[$0] }
        #expect(realIds == predictedIds)
        #expect(Set(realIds).count == Self.topK,
                "top-k collapsed — the fixture stopped exercising the ranking")
        #expect(realIds.allSatisfy { $0 < UInt32(Self.numRouted) })
    }
}
