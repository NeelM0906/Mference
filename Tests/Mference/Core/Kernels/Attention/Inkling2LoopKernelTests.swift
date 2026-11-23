import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// The loop core's depth mix and message gate against `Inkling2LoopRef`.
@Suite(.serialized) struct Inkling2LoopKernelTests {
    typealias S = Inkling2TestSupport

    /// Query scale 0.1 gives a near-uniform softmax; 2.0 a peaked one.
    @Test(arguments: [(3, 3, Float(0.1)), (2, 4, 0.1), (6, 1, 0.1), (1, 2, 0.1), (6, 3, 2.0), (2, 4, 2.0)])
    func depthMixMatchesReference(states count: Int, queries queryCount: Int, queryScale: Float) throws {
        let ctx = try MetalContext()
        let mix = try Inkling2DepthMix(library: Inkling2KernelLibrary(context: ctx))
        var rng = SeedTree(0x6a00 + UInt64(count * 8 + queryCount)).key("ink2-mix-\(queryScale)")
        let rows = 4, width = 2304
        var dense: [[Float]] = []
        var inputs: [(ref: Inkling2BufferRef, stride: Int)] = []
        for n in 0..<count {
            let stride = width + 4 * n        // a different row stride per state
            let values = S.random(rows * stride, &rng, -4, 4)
            dense.append((0..<rows).flatMap { t in values[(t * stride)..<(t * stride + width)] })
            inputs.append((.init(S.buffer(ctx.device, floats: values)), stride))
        }
        let queries = S.random(queryCount * width, &rng, -queryScale, queryScale)
        let out = ctx.device.makeBuffer(length: rows * queryCount * width * 4, options: .storageModeShared)!
        try S.run(ctx) { cb in
            try mix.encode(commandBuffer: cb, states: inputs, queries: .init(S.buffer(ctx.device, floats: queries)),
                       queryCount: queryCount, output: .init(out), outputStride: queryCount * width, rows: rows)
        }
        let reference = Inkling2LoopRef.depthMix(states: dense, queries: queries, queryCount: queryCount,
                                                 rows: rows, hidden: width)
        #expect(RelError.computeStrict(actual: S.readFloats(out, count: rows * queryCount * width), reference: reference) < 2e-5)
    }

    /// The FP32 product with partial row and token tiles (70 rows, 37 tokens).
    @Test func floatProductMatchesReference() throws {
        let ctx = try MetalContext()
        let projection = try Inkling2Projection(library: Inkling2KernelLibrary(context: ctx))
        var rng = SeedTree(0x6c00).key("ink2-f32mm")
        let tokens = 37, inputs = 96, rows = [70, 24]
        let weights = rows.map { S.random($0 * inputs, &rng, -0.5, 0.5) }
        let x = S.random(tokens * (inputs + 4), &rng, -2, 2)          // row stride inputs + 4
        let out = S.buffer(ctx.device, floats: [Float](repeating: -1, count: tokens * 96))
        S.run(ctx) { cb in
            projection.encodeFloat(commandBuffer: cb, weights: S.buffer(ctx.device, floats: weights[0] + weights[1]),
                                   input: S.buffer(ctx.device, floats: x), output: out, tokens: tokens, inputs: inputs,
                                   outStride: 96, segments: [
                                       .init(weightOffset: 0, inputOffset: 0, inputStride: inputs + 4, rows: rows[0],
                                             outColumn: 0, format: .f32),
                                       .init(weightOffset: rows[0] * inputs * 4, inputOffset: 0, inputStride: inputs + 4,
                                             rows: rows[1], outColumn: 72, format: .f32)])
        }
        let got = S.readFloats(out, count: tokens * 96)
        for t in 0..<tokens {
            for c in 0..<96 {
                guard c < 70 || c >= 72 else { #expect(got[t * 96 + c] == -1); continue }
                let (w, r) = c < 70 ? (weights[0], c) : (weights[1], c - 72)
                var dot = 0.0
                for i in 0..<inputs { dot += Double(w[r * inputs + i]) * Double(x[t * (inputs + 4) + i]) }
                #expect(abs(got[t * 96 + c] - Float(dot)) < 1e-4 * max(1, abs(Float(dot))))
            }
        }
    }

    /// Two controllers share one down product; their forget and gain products
    /// run as one FP32 dispatch each; each combines with its own x and m.
    @Test(arguments: [5, 40, 130])
    func messageGatePipelineMatchesReference(rows: Int) throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let gate = try Inkling2MessageGateKernels(library: library)
        let projection = try Inkling2Projection(library: library)
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
        let h = ctx.device.makeBuffer(length: controllers * rows * 2 * width * 4, options: .storageModeShared)!
        let z = ctx.device.makeBuffer(length: rows * controllers * rank * 4, options: .storageModeShared)!
        let fg = ctx.device.makeBuffer(length: rows * 2 * width * 4, options: .storageModeShared)!
        let downBuffer = S.buffer(ctx.device, floats: downs.flatMap { $0 })
        let xBuffers = xs.map { S.buffer(ctx.device, floats: $0) }
        let mBuffers = ms.map { S.buffer(ctx.device, floats: $0) }
        let outs = (0..<controllers).map { _ in ctx.device.makeBuffer(length: rows * width * 4, options: .storageModeShared)! }
        S.run(ctx) { cb in
            for c in 0..<controllers {
                gate.encodeInputs(commandBuffer: cb, x: .init(xBuffers[c]), xStride: width,
                                  message: .init(mBuffers[c]), messageStride: width,
                                  output: .init(h, offset: c * rows * 2 * width * 4), outputStride: 2 * width, rows: rows)
            }
            projection.encodeFloat(commandBuffer: cb, weights: downBuffer, input: h, output: z, tokens: rows,
                                   inputs: 2 * width, outStride: controllers * rank,
                                   segments: (0..<controllers).map { c in
                                       Inkling2ProjectionSegment(weightOffset: c * rank * 2 * width * 4,
                                                                 inputOffset: c * rows * 2 * width, inputStride: 2 * width,
                                                                 rows: rank, outColumn: c * rank, format: .f32)
                                   })
        }
        for c in 0..<controllers {
            let fgWeights = S.buffer(ctx.device, floats: forgets[c] + gains[c])
            let out = outs[c]
            S.run(ctx) { cb in
                projection.encodeFloat(commandBuffer: cb, weights: fgWeights, input: z, output: fg, tokens: rows, inputs: rank,
                                       outStride: 2 * width,
                                       segments: [.init(weightOffset: 0, inputOffset: c * rank, inputStride: controllers * rank,
                                                        rows: width, outColumn: 0, format: .f32),
                                                  .init(weightOffset: width * rank * 4, inputOffset: c * rank,
                                                        inputStride: controllers * rank, rows: width, outColumn: width,
                                                        format: .f32)])
                gate.encodeCombine(commandBuffer: cb, x: .init(xBuffers[c]), xStride: width,
                                   message: .init(mBuffers[c]), messageStride: width,
                                   gates: .init(fg), gatesStride: 2 * width, forgetColumn: 0, gainColumn: width,
                                   output: .init(out), outputStride: width, rows: rows)
            }
            let reference = Inkling2LoopRef.messageGate(x: xs[c], message: ms[c], down: downs[c], forget: forgets[c],
                                                        gain: gains[c], rows: rows, hidden: width, rank: rank)
            #expect(RelError.computeStrict(actual: S.readFloats(out, count: rows * width), reference: reference) < 1e-4)
        }
    }

    /// The single-row gate: the q, k and v controllers share x (as on a window
    /// layer's loop pass), each with its own message; a fourth (the MoE input)
    /// writes over its own x.
    @Test func decodeGateMatchesReference() throws {
        let ctx = try MetalContext()
        let gate = try Inkling2MessageGateKernels(library: Inkling2KernelLibrary(context: ctx))
        var rng = SeedTree(0x6d00).key("ink2-gate-decode")
        let width = 2304, rank = 64
        let sharedX = S.random(width, &rng, -3, 3), moeX = S.random(width, &rng, -3, 3)
        // Controller 2's message is tiny (mean square ~3e-7), so eps (1e-6) decides its norm.
        let messages = (0..<4).map { c in S.random(width, &rng, c == 2 ? -1e-3 : -3, c == 2 ? 1e-3 : 3) }
        // Activations: [shared x | moe x | m0 | m1 | m2 | m3 | out0 | out1 | out2].
        let acts = S.buffer(ctx.device, floats: sharedX + moeX + messages.flatMap { $0 } + [Float](repeating: 0, count: 3 * width))
        var weights: [Float] = []
        var controllers: [Inkling2MessageGateKernels.DecodeController] = []
        var refs: [(down: [Float], forget: [Float], gain: [Float])] = []
        for c in 0..<4 {
            let down = S.random(rank * 2 * width, &rng, -0.03, 0.03)
            let forget = S.random(width * rank, &rng, -0.2, 0.2), gain = S.random(width * rank, &rng, -0.2, 0.2)
            let base = weights.count * 4
            weights += down + forget + gain
            refs.append((down, forget, gain))
            controllers.append(.init(down: base, forget: base + down.count * 4, gain: base + (down.count + forget.count) * 4,
                                     x: c < 3 ? 0 : width, message: (2 + c) * width,
                                     output: c < 3 ? (6 + c) * width : width))
        }
        let z = ctx.device.makeBuffer(length: 4 * rank * 4, options: .storageModeShared)!
        S.run(ctx) { cb in
            gate.encodeDecode(commandBuffer: cb, weights: S.buffer(ctx.device, floats: weights), activations: acts,
                              z: .init(z), controllers: controllers)
        }
        for c in 0..<4 {
            let reference = Inkling2LoopRef.messageGate(x: c < 3 ? sharedX : moeX, message: messages[c], down: refs[c].down,
                                                        forget: refs[c].forget, gain: refs[c].gain,
                                                        rows: 1, hidden: width, rank: rank)
            let got = S.readFloats(acts, count: width, offset: controllers[c].output * 4)
            #expect(RelError.computeStrict(actual: got, reference: reference) < 1e-4, "controller \(c)")
        }
    }
}
