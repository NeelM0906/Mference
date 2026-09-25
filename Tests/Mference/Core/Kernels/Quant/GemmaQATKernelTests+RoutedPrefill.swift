import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

extension GemmaQATKernelTests {
    @Test(arguments: [false, true]) func nativeRoutedPrefillPreservesGroupGeometryAndPairMapping(sourceFP16: Bool) throws {
        let fixture = RoutedFixture(sourceFP16: sourceFP16)
        let context = try MetalContext()
        let kernel = try PrefillGroupedRoutedMoE(context: context, groupSize: 32, sourceFP16: sourceFP16)
        let d = fixture.d, f = fixture.f
        let stride = (fixture.blobs[0].count + 4095) / 4096 * 4096
        var bytes = [UInt8](repeating: 0xA5, count: stride * 8)
        for expert in 0..<8 {
            bytes.replaceSubrange(expert * stride..<expert * stride + fixture.blobs[expert].count,
                                  with: fixture.blobs[expert])
        }
        let slab = try Self.buffer(bytes, context: context)
        let order = [5, 2, 7, 0, 3, 6, 1, 4]
        let binding = try PrefillStreamedTileBinding(expertIDs: order, views: order.map { expert in
            TensorView(buffer: slab, offset: UInt64(expert * stride),
                length: UInt64(fixture.blobs[expert].count),
                scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                shape: (0, 0, 0, 0), dtype: 0)
        })
        var pairs: [PrefillTokenExpertPair] = []
        for expert in 0..<8 {
            for token in 0..<3 {
                let rank = (expert - token + 8) % 8
                pairs.append(PrefillTokenExpertPair(token: UInt32(token), expert: UInt32(expert),
                    rank: UInt32(rank), weight: Float16(0.125)))
            }
        }
        let pairBuffer = try Self.buffer(pairs, context: context)
        let hiddenRows = Array(repeating: fixture.input + [Float16](repeating: .nan, count: 8), count: 3)
        let hidden = try Self.buffer(hiddenRows.flatMap { $0 }, context: context)
        let scratch = try Self.buffer([Float16](repeating: .nan, count: 24 * 3 * f), context: context)
        let downScratch = try Self.buffer([Float16](repeating: .nan, count: 5 * d), context: context)
        let args = try kernel.makeStreamedArgumentBuffer(device: context.device, binding: binding)
        for resident in [false, true] {
            let out = try Self.buffer([Float16](repeating: .nan, count: 3 * 8 * d), context: context)
            var params = PrefillGroupedRoutedMoEStreamedParams(pairStart: 0, pairCount: 24,
                d: UInt32(d), routedIntermediate: UInt32(f), topK: 8,
                hiddenStrideElements: UInt32(d + 8), binding: binding, offsets: fixture.offsets)
            let cb = try #require(context.queue.makeCommandBuffer())
            if resident {
                params.liveExpertCount = 0
                try kernel.encodeResidentBatched(commandBuffer: cb, hidden: hidden, sortedPairs: pairBuffer,
                    activation: scratch, routePartials: out, slab: slab, stride: UInt32(stride), params: params)
            } else {
                let count = kernel.encodeStreamedBatched(commandBuffer: cb, hidden: hidden,
                    sortedPairs: pairBuffer, routePartials: out, gateUpActScratch: scratch,
                    downScratch: downScratch, argumentBuffer: args, binding: binding,
                    params: params, pairMicrobatchRows: 5)
                #expect(count == 5) // final microbatch contains four pairs
            }
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed, "\(String(describing: cb.error))")
            if sourceFP16 {
                let remaining = resident ? pairs.count : 4
                let finalPairs = resident ? pairs : Array(pairs.suffix(remaining))
                let actOffset = resident ? 0 : 2 * remaining * f
                let act = scratch.contents().assumingMemoryBound(to: Float16.self).advanced(by: actOffset)
                let actualActs = Array(UnsafeBufferPointer(start: act, count: remaining * f))
                let expectedActs = finalPairs.flatMap { fixture.expectedActivations[Int($0.expert)] }
                #expect(actualActs == expectedActs, "source routed prefill activation resident=\(resident)")
            }
            let actual = out.contents().assumingMemoryBound(to: Float16.self)
            for pair in pairs {
                let start = (Int(pair.token) * 8 + Int(pair.rank)) * d
                let values = (0..<d).map { Float(actual[start + $0]) }
                #expect(RelError.compute(actual: values,
                    reference: fixture.expectedPartials[Int(pair.expert)]) < 0.002,
                    "routed prefill resident=\(resident), token=\(pair.token), expert=\(pair.expert)")
            }
        }
    }
}
