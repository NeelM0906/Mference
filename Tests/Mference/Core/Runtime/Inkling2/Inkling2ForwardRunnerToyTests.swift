import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// `Inkling2ForwardRunner` end to end on the Mini-width synthetic model
/// (`Inkling2RunnerToy`) against `Inkling2ReferenceModel` reading the same
/// bytes: a prefill split into chunks, a second prefill, two decode steps,
/// all 48 visits, every kernel of K2, K3 and K4 in the path. Production
/// shape (about 1.7 GB of buffers, a slow CPU reference), so full tier.
///
/// Each visit is checked alone. The reference rounds to FP16 where the
/// runner stores half (`.metalRunner`) and continues after every visit from
/// the runner's output (`forcedVisitOutputs`): the random toy amplifies the
/// small differences left (fast-math `exp` and `tanh`, accumulation order)
/// about twofold every six visits, which a free-running comparison over 48
/// visits cannot tell from a bug, while each visit's own error stays small.
/// The reference's block lists, message banks, KV caches and convolution
/// tails derive from the runner's outputs, so a wrong input, bank, slot or
/// tail on the GPU shows at the visit that uses it. With 128 experts a
/// router still meets near-ties it breaks the other way, so the reference
/// also follows the runner's routes (`forcedRoutes`) and every disagreement
/// must be a near-tie: a forced expert at most `tieMargin` below the
/// reference's own top-8 cutoff.
@Suite("Inkling2ForwardRunner on the Mini-width toy", .serialized)
struct Inkling2ForwardRunnerToyTests {

    /// Prefill, a second prefill, then one decode step per token.
    static let steps: [(tokens: [Int32], decode: Bool)] = [
        ([17, 401, 3], false), ([250, 99], false), ([7], true), ([333], true),
    ]
    static let traceRows = 8
    /// Largest difference of one visit's output relative to its largest
    /// magnitude, and of the logits. Measured 2026-11-23 (M1 Max): at most
    /// 8.6e-4 per visit and 3.0e-4 for the logits over both chunk sizes.
    static let tolerance: Float = 0.005
    static let logitTolerance: Float = 0.002
    /// Selection-score shortfall of a forced expert (scores are a sigmoid
    /// plus a bias of at most 0.1). Measured: no route differed.
    static let tieMargin: Float = 0.005

    private func logits(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let p = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(p[$0]) }
    }

    private func argmax(_ v: [Float]) -> Int { v.indices.max { v[$0] < v[$1] }! }

    /// Largest difference relative to the reference's largest magnitude.
    private func relative(_ got: [Float], _ want: [Float]) -> Float {
        zip(got, want).map { abs($0 - $1) }.max()! / max(want.map(abs).max()!, 1e-6)
    }

    /// Chunks of 2 split the 3-token prefill into 2 + 1 (a one-row span
    /// through the prefill path); chunks of 4 take it whole.
    @Test(.fullTier, arguments: [2, 4])
    func runnerMatchesTheReference(maxChunkTokens: Int) async throws {
        let context = try MetalContext()
        let toy = try Inkling2RunnerToy(device: context.device)
        let c = Inkling2RunnerToy.config
        let d = c.hiddenSize, topK = c.topK, rows = Self.traceRows
        let runner = try Inkling2ForwardRunner(weights: toy.runnerWeights, context: context,
                                               maxContext: 64, maxChunkTokens: maxChunkTokens)
        let reference = try Inkling2ReferenceModel(config: c, weights: toy.referenceWeights)
        reference.captureTrace = true
        reference.precision = .metalRunner
        let device = context.device
        let out = try #require(device.makeBuffer(length: c.vocabPadded * 2, options: .storageModeShared))
        let outputs = try #require(device.makeBuffer(length: 48 * rows * d * 4, options: .storageModeShared))
        let routes = try #require(device.makeBuffer(length: 48 * rows * topK * 4, options: .storageModeShared))
        runner.visitTrace = .init(outputs: outputs, routes: routes, rows: rows)
        let ids = routes.contents().bindMemory(to: UInt32.self, capacity: 48 * rows * topK)

        var position = 0
        for (step, (chunk, decode)) in Self.steps.enumerated() {
            let n = chunk.count
            if decode {
                try await runner.produce(token: chunk[0], position: position, into: out)
            } else {
                let result = try await runner.prefillChunked(tokens: chunk[...], startPosition: position,
                                                             outputMode: .logits,
                                                             config: .production(chunkTokens: 32),
                                                             into: out, onProgress: { _ in })
                #expect(result.newPosition == position + n)
            }
            reference.forcedRoutes = (0..<48).map { visit in
                (0..<n).map { t in (0..<topK).map { Int(ids[((visit * rows) + position + t) * topK + $0]) } }
            }
            let gpuOutputs = (0..<48).map { visit in
                (0..<n).map { t in
                    Inkling2BufferMatrix.floats(outputs, offset: (visit * rows + position + t) * d * 4, count: d)
                }
            }
            reference.forcedVisitOutputs = gpuOutputs
            let want = reference.forward(tokens: chunk, logits: .last)[0]
            let trace = try #require(reference.lastTrace)

            var visitErrors: [Float] = []
            var flips = 0
            var shortfall: Float = 0
            for visit in 0..<48 {
                var err: Float = 0
                for t in 0..<n {
                    err = max(err, relative(gpuOutputs[visit][t], trace.visitOutputs[visit][t]))
                    let s = trace.routeShortfalls[visit][t]
                    if s > 0 { flips += 1 }
                    shortfall = max(shortfall, s)
                }
                visitErrors.append(err)
            }
            let all = logits(out, count: c.vocabPadded)
            let real = Array(all[..<c.vocabUnpadded])
            let logitError = relative(real, want)
            let worst = visitErrors.max()!
            print("inkling2 toy chunk \(maxChunkTokens) step \(step) (\(n) at \(position)): visit error \(worst) "
                  + "(last \(visitErrors[47])), logit error \(logitError), argmax \(argmax(real)) vs "
                  + "\(argmax(want)), \(flips) near-tie route flips, largest shortfall \(shortfall)")
            if worst >= Self.tolerance {
                print("inkling2 toy per-visit errors: \(visitErrors.map { String(format: "%.4f", $0) })")
            }
            #expect(worst < Self.tolerance, "step \(step)")
            #expect(logitError < Self.logitTolerance, "step \(step)")
            #expect(shortfall <= Self.tieMargin, "step \(step)")
            // The runner's top logit is the reference's or within the error of it.
            let maxAbs = zip(real, want).map { abs($0 - $1) }.max()!
            #expect(want[argmax(real)] >= want.max()! - 2 * maxAbs, "step \(step)")
            // Padded rows past the vocabulary are never sampled.
            #expect(all[c.vocabUnpadded...].allSatisfy { $0 < -60_000 }, "step \(step)")
            position += n
        }
        #expect(runner.continuationPosition == position)
    }
}
