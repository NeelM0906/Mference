import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

private let groupedGEMMAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPGroupedRoutedMoE(context: context).isAvailable
}()

extension GemmaQATKernelTests {
    /// Gemma's routed experts are group-32 (QAT) or group-64 affine INT4 with
    /// GeGLU. Adjacent fixture groups carry different scales and biases, so a
    /// kernel that indexes scales by its 64-wide column tile fails this test.
    @Test(.enabled(if: groupedGEMMAvailable, "Requires runtime MPP TensorOps support"))
    func groupedGEMMHonorsStorageGroupsAndGeGLU() throws {
        let fixture = RoutedFixture(sourceFP16: false)
        let context = try MetalContext()
        let kernel = MPPGroupedRoutedMoE(context: context, groupSize: 32, gelu: true)
        try #require(kernel.isAvailable)
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
        // 70 rows per expert: one full 64-row matrix tile plus a partial one.
        let tokens = 70
        var pairs: [PrefillTokenExpertPair] = []
        for token in 0..<tokens {
            for rank in 0..<8 {
                pairs.append(PrefillTokenExpertPair(token: UInt32(token), expert: UInt32((token + rank) % 8),
                                                    rank: UInt32(rank), weight: Float16(0.125)))
            }
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(pairs, queryCount: tokens, topK: 8,
                                                                  numExperts: 8, tileExpertCount: 8)
        try #require(routes.tiles.count == 1)
        let helper = try PrefillGroupedRoutedMoE(context: context, groupSize: 32)
        let metadata = try helper.makeStreamedMetadataBuffers(device: context.device, routes: routes)
        let hiddenRows = Array(repeating: fixture.input + [Float16](repeating: .nan, count: 8), count: tokens)
        let hidden = try Self.buffer(hiddenRows.flatMap { $0 }, context: context)
        let activation = try Self.buffer([Float16](repeating: .nan, count: pairs.count * f), context: context)
        let out = try Self.buffer([Float16](repeating: .nan, count: tokens * 8 * d), context: context)
        let arguments = try kernel.makeArgumentBuffer(device: context.device, binding: binding)
        var params = PrefillGroupedRoutedMoEStreamedParams(pairStart: 0, pairCount: 0,
            d: UInt32(d), routedIntermediate: UInt32(f), topK: 8,
            hiddenStrideElements: UInt32(d + 8), binding: binding, offsets: fixture.offsets)
        params.pairStart = routes.tiles[0].groupStart
        params.pairCount = routes.tiles[0].groupCount

        let cb = try #require(context.queue.makeCommandBuffer())
        #expect(kernel.encode(commandBuffer: cb, hidden: hidden, sortedPairs: metadata.sortedPairs,
            groups: metadata.groups, activation: activation, routePartials: out,
            argumentBuffer: arguments, binding: binding, params: params,
            maxPairsPerGroup: routes.maxPairsPerExpert))
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.status == .completed, "\(String(describing: cb.error))")

        let actual = out.contents().assumingMemoryBound(to: Float16.self)
        for pair in pairs {
            let start = (Int(pair.token) * 8 + Int(pair.rank)) * d
            let values = (0..<d).map { Float(actual[start + $0]) }
            #expect(RelError.compute(actual: values,
                reference: fixture.expectedPartials[Int(pair.expert)]) < 0.002,
                "grouped GEMM token=\(pair.token), expert=\(pair.expert)")
        }
    }
}
