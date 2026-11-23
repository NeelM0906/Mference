import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// `Inkling2InputNorm.encodeSet`: one to three inputs, each with its own weight
/// and output, in one dispatch, writes the bytes one `encode` per input writes,
/// and matches the CPU reference.
@Suite(.serialized) struct Inkling2NormSetTests {
    typealias S = Inkling2TestSupport

    @Test(arguments: [(1, 1), (2, 1), (3, 1), (3, 5)])
    func setMatchesOneDispatchPerInput(inputs count: Int, rows: Int) throws {
        let ctx = try MetalContext()
        let norm = try Inkling2InputNorm(library: Inkling2KernelLibrary(context: ctx))
        let width = 2304, xStride = 2312, outStride = 2320      // strides wider than the row
        var rng = SeedTree(0x8b00 + UInt64(count * 10 + rows)).key("ink2-norm-set")
        let xs = (0..<count).map { k in S.random(rows * xStride, &rng, k == 1 ? -1e-3 : -4, k == 1 ? 1e-3 : 4) }
        let weights = (0..<count).map { _ in S.random(width, &rng, -2, 2) }
        let xBuffers = xs.map { S.buffer(ctx.device, floats: $0) }
        let wBuffers = weights.map { S.buffer(ctx.device, floats: $0) }
        func outputs() -> [MTLBuffer] {
            (0..<count).map { _ in S.buffer(ctx.device, halves: [Float16](repeating: .nan, count: rows * outStride)) }
        }
        let separate = outputs(), together = outputs()
        S.run(ctx) { cb in
            for k in 0..<count {
                norm.encode(commandBuffer: cb, x: .init(xBuffers[k]), xStride: xStride, weight: .init(wBuffers[k]),
                            output: .init(separate[k]), outputStride: outStride, rows: rows)
            }
            norm.encodeSet(commandBuffer: cb,
                           (0..<count).map { k in .init(x: .init(xBuffers[k]), weight: .init(wBuffers[k]),
                                                        output: .init(together[k])) },
                           xStride: xStride, outputStride: outStride, rows: rows)
        }
        for k in 0..<count {
            let a = S.readHalves(separate[k], count: rows * outStride), b = S.readHalves(together[k], count: rows * outStride)
            #expect(a.map(\.bitPattern) == b.map(\.bitPattern), "input \(k): same bytes")
            let x = (0..<rows).flatMap { t in xs[k][(t * xStride)..<(t * xStride + width)] }
            let got = (0..<rows).flatMap { t in b[(t * outStride)..<(t * outStride + width)] }
            let reference = Inkling2NormRef.weightedRMS(x, rows: rows, width: width, weight: weights[k])
            #expect(RelError.computeStrict(actual: got, reference: reference) < 2e-3, "input \(k)")
        }
    }
}
