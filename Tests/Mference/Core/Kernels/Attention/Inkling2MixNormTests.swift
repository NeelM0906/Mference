import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// `Inkling2DepthMix.encodeNormed`: the depth mix and each query's weighted RMS
/// norm in one dispatch writes the bytes the mix followed by one
/// `Inkling2InputNorm.encode` per query writes (the raw mix too), and the
/// optional FP32 normed rows match the CPU norm and round to the FP16 rows.
@Suite(.serialized) struct Inkling2MixNormTests {
    typealias S = Inkling2TestSupport

    @Test(arguments: [(1, 1, false), (3, 1, false), (1, 4, true), (3, 4, true)])
    func mixNormMatchesMixThenNorm(queries q: Int, rows: Int, dual: Bool) throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let mix = try Inkling2DepthMix(library: library)
        let norm = try Inkling2InputNorm(library: library)
        let width = 2304, stateCount = 3, stride = 2320, normedStride = 2312
        var rng = SeedTree(0x9c00 + UInt64(q * 10 + rows)).key("ink2-mix-norm")
        let states = (0..<stateCount).map { _ in S.buffer(ctx.device, floats: S.random(rows * stride, &rng, -3, 3)) }
        let queries = S.buffer(ctx.device, floats: S.random(q * width, &rng, -1, 1))
        let weights = (0..<q).map { _ in S.random(width, &rng, -2, 2) }
        let weightBuffers = weights.map { S.buffer(ctx.device, floats: $0) }
        func halves() -> MTLBuffer { S.buffer(ctx.device, halves: [Float16](repeating: .nan, count: rows * normedStride)) }
        func floats(_ n: Int) -> MTLBuffer { S.buffer(ctx.device, floats: [Float](repeating: .nan, count: n)) }
        let mixA = floats(rows * q * width), mixB = floats(rows * q * width)
        let normedA = (0..<q).map { _ in halves() }, normedB = (0..<q).map { _ in halves() }
        let normed32 = (0..<q).map { _ in floats(rows * normedStride) }
        let stateRefs = states.map { (ref: Inkling2BufferRef($0), stride: stride) }
        try S.run(ctx) { cb in
            try mix.encode(commandBuffer: cb, states: stateRefs, queries: .init(queries), queryCount: q,
                           output: .init(mixA), outputStride: q * width, rows: rows)
            for j in 0..<q {
                norm.encode(commandBuffer: cb, x: .init(mixA, offset: j * width * 4), xStride: q * width,
                            weight: .init(weightBuffers[j]), output: .init(normedA[j]), outputStride: normedStride,
                            rows: rows)
            }
            try mix.encodeNormed(commandBuffer: cb, states: stateRefs, queries: .init(queries), queryCount: q,
                                 output: .init(mixB), outputStride: q * width,
                                 norms: (0..<q).map { j in
                                     .init(weight: .init(weightBuffers[j]), output: .init(normedB[j]),
                                           output32: dual ? .init(normed32[j]) : nil)
                                 },
                                 normedStride: normedStride, normed32Stride: dual ? normedStride : 0, rows: rows)
        }
        let mixGot = S.readFloats(mixB, count: rows * q * width)
        #expect(S.readFloats(mixA, count: rows * q * width).map(\.bitPattern) == mixGot.map(\.bitPattern), "raw mix")
        for j in 0..<q {
            let a = S.readHalves(normedA[j], count: rows * normedStride), b = S.readHalves(normedB[j], count: rows * normedStride)
            #expect(a.map(\.bitPattern) == b.map(\.bitPattern), "query \(j): normed FP16 bytes")
            guard dual else { continue }
            let wide = S.readFloats(normed32[j], count: rows * normedStride)
            let x = (0..<rows).flatMap { t in mixGot[(t * q * width + j * width)..<(t * q * width + (j + 1) * width)] }
            let got = (0..<rows).flatMap { t in wide[(t * normedStride)..<(t * normedStride + width)] }
            let reference = Inkling2NormRef.weightedRMS(x, rows: rows, width: width, weight: weights[j])
            #expect(RelError.computeStrict(actual: got, reference: reference) < 1e-5, "query \(j): FP32 normed rows")
            let rounded = (0..<rows).flatMap { t in b[(t * normedStride)..<(t * normedStride + width)] }
            #expect(got.map { Float(Float16($0)).bitPattern } == rounded.map(\.bitPattern), "query \(j): FP16 = FP32 rounded")
        }
    }
}
