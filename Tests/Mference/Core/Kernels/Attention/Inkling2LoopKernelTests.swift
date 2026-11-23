import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// The loop core's depth mix and message gate against `Inkling2LoopRef`.
@Suite(.serialized) struct Inkling2LoopKernelTests {
    typealias S = Inkling2TestSupport

    @Test(arguments: [(3, 3), (2, 4), (6, 1), (1, 2)])
    func depthMixMatchesReference(states count: Int, queries queryCount: Int) throws {
        let ctx = try MetalContext()
        let mix = try Inkling2DepthMix(library: Inkling2KernelLibrary(context: ctx))
        var rng = SeedTree(0x6a00 + UInt64(count * 8 + queryCount)).key("ink2-mix")
        let rows = 4, width = 2304
        var dense: [[Float]] = []
        var inputs: [(ref: Inkling2BufferRef, stride: Int)] = []
        for n in 0..<count {
            let stride = width + 4 * n        // a different row stride per state
            let values = S.random(rows * stride, &rng, -4, 4)
            dense.append((0..<rows).flatMap { t in values[(t * stride)..<(t * stride + width)] })
            inputs.append((.init(S.buffer(ctx.device, floats: values)), stride))
        }
        let queries = S.random(queryCount * width, &rng, -0.1, 0.1)
        let out = ctx.device.makeBuffer(length: rows * queryCount * width * 4, options: .storageModeShared)!
        S.run(ctx) { cb in
            mix.encode(commandBuffer: cb, states: inputs, queries: .init(S.buffer(ctx.device, floats: queries)),
                       queryCount: queryCount, output: .init(out), outputStride: queryCount * width, rows: rows)
        }
        let reference = Inkling2LoopRef.depthMix(states: dense, queries: queries, queryCount: queryCount,
                                                 rows: rows, hidden: width)
        #expect(RelError.compute(actual: S.readFloats(out, count: rows * queryCount * width), reference: reference) < 2e-5)
    }

    /// Two controllers share one down projection (F32 segments); each then
    /// combines with its own forget and gain.
    @Test(arguments: [(5, Inkling2Projection.Path.rows), (40, .tiled), (40, .tensorOps)])
    func messageGatePipelineMatchesReference(rows: Int, path: Inkling2Projection.Path) throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let gate = try Inkling2MessageGate(library: library)
        let projection = try Inkling2Projection(library: library)
        if path == .tensorOps && !projection.supportsTensorOps { return }
        var rng = SeedTree(0x6b00 + UInt64(rows)).key("ink2-gate")
        let width = 2304, rank = 64
        let controllers = 2
        var xs: [[Float]] = [], ms: [[Float]] = [], downs: [[Float]] = [], forgets: [[Float]] = [], gains: [[Float]] = []
        for _ in 0..<controllers {
            xs.append(S.random(rows * width, &rng, -3, 3))
            ms.append(S.random(rows * width, &rng, -3, 3))
            downs.append(S.random(rank * 2 * width, &rng, -0.03, 0.03))
            forgets.append(S.random(width * rank, &rng, -0.2, 0.2))
            gains.append(S.random(width * rank, &rng, -0.2, 0.2))
        }
        let h = ctx.device.makeBuffer(length: controllers * rows * 2 * width * 2, options: .storageModeShared)!
        let z = ctx.device.makeBuffer(length: rows * controllers * rank * 4, options: .storageModeShared)!
        let downBuffer = S.buffer(ctx.device, floats: downs.flatMap { $0 })
        let xBuffers = xs.map { S.buffer(ctx.device, floats: $0) }
        let mBuffers = ms.map { S.buffer(ctx.device, floats: $0) }
        let outs = (0..<controllers).map { _ in ctx.device.makeBuffer(length: rows * width * 4, options: .storageModeShared)! }
        S.run(ctx) { cb in
            for c in 0..<controllers {
                gate.encodeInputs(commandBuffer: cb, x: .init(xBuffers[c]), xStride: width,
                                  message: .init(mBuffers[c]), messageStride: width,
                                  output: .init(h, offset: c * rows * 2 * width * 2), outputStride: 2 * width, rows: rows)
            }
            let segments = (0..<controllers).map { c in
                Inkling2ProjectionSegment(weightOffset: c * rank * 2 * width * 4, inputOffset: c * rows * 2 * width,
                                          inputStride: 2 * width, rows: rank, outColumn: c * rank, format: .f32)
            }
            projection.encode(commandBuffer: cb, weights: downBuffer, input: h, output: z, tokens: rows,
                              inputs: 2 * width, outStride: controllers * rank, segments: segments, path: path)
            for c in 0..<controllers {
                gate.encodeCombine(commandBuffer: cb, x: .init(xBuffers[c]), xStride: width,
                                   message: .init(mBuffers[c]), messageStride: width,
                                   z: .init(z), zStride: controllers * rank, zColumn: c * rank,
                                   forget: .init(S.buffer(ctx.device, floats: forgets[c])),
                                   gain: .init(S.buffer(ctx.device, floats: gains[c])),
                                   output: .init(outs[c]), outputStride: width, rows: rows)
            }
        }
        for c in 0..<controllers {
            let reference = Inkling2LoopRef.messageGate(x: xs[c], message: ms[c], down: downs[c], forget: forgets[c],
                                                        gain: gains[c], rows: rows, hidden: width, rank: rank)
            // The down product reads FP16 normalised inputs and half-rounded weights.
            #expect(RelError.compute(actual: S.readFloats(outs[c], count: rows * width), reference: reference) < 3e-3)
        }
    }
}
