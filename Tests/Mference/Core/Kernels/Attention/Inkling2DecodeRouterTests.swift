import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// `Inkling2DecodeRouter` (one token) against the CPU reference
/// (`Inkling2RouterRef`) and against K2's prefill router run at one token:
/// the same experts (unless the 8th and 9th selection scores are a near tie,
/// which the summation order may resolve either way) and the same weights
/// within FP32 rounding.
@Suite(.serialized) struct Inkling2DecodeRouterTests {
    typealias S = Inkling2TestSupport

    @Test(arguments: [(128, 8), (96, 6)])
    func decodeRouterMatchesReferenceAndPrefillRouter(experts: Int, topK: Int) throws {
        let ctx = try MetalContext()
        let router = try Inkling2DecodeRouter(library: Inkling2KernelLibrary(context: ctx))
        let moe = try Inkling2MoEKernels(context: ctx)
        let width = 2304
        var compared = 0
        var worstReference: Float = 0, worstPrefill: Float = 0, worstLogits: Float = 0
        for trial in 0..<8 {
            var rng = SeedTree(0xd700 + UInt64(trial * 10 + experts)).key("ink2-decode-router")
            let x = S.random(width, &rng, -2, 2)
            let weight = S.random(experts * width, &rng, -0.05, 0.05)
            let bias = S.random(experts, &rng, -0.1, 0.1)
            let xBuffer = S.buffer(ctx.device, floats: x), wBuffer = S.buffer(ctx.device, floats: weight)
            let bBuffer = S.buffer(ctx.device, floats: bias)
            let logits = S.buffer(ctx.device, floats: [Float](repeating: .nan, count: experts))
            let ids = ctx.device.makeBuffer(length: 32 * 4, options: .storageModeShared)!
            let weights = S.buffer(ctx.device, floats: [Float](repeating: .nan, count: 32))
            S.run(ctx) { cb in
                router.encode(commandBuffer: cb, x: .init(xBuffer), weight: .init(wBuffer), bias: .init(bBuffer),
                              logits: .init(logits), ids: .init(ids), routeWeights: .init(weights),
                              experts: experts, width: width, topK: topK)
            }
            let gotIDs = (0..<topK).map { Int(ids.contents().bindMemory(to: UInt32.self, capacity: 32)[$0]) }
            let gotWeights = S.readFloats(weights, count: topK)
            let refLogits = Inkling2RouterRef.logits(weight: weight, experts: experts, x: x)
            worstLogits = max(worstLogits, RelError.computeStrict(actual: S.readFloats(logits, count: experts), reference: refLogits))
            let reference = Inkling2RouterRef.route(logits: refLogits, selectionBias: bias, topK: topK)
            // The margin between the last chosen expert and the best one left out.
            let scores = zip(refLogits, bias).map { 1 / (1 + expf(-$0)) + $1 }.sorted(by: >)
            guard scores[topK - 1] - scores[topK] > 1e-5 else { continue }
            compared += 1
            #expect(gotIDs == reference.ids, "trial \(trial): experts")
            worstReference = max(worstReference, RelError.computeStrict(actual: gotWeights, reference: reference.weights))
            // K2's prefill router at one token (its shape has 128 experts and top-8).
            if experts == 128 && topK == 8 {
                let prefillIDs = ctx.device.makeBuffer(length: 32 * 4, options: .storageModeShared)!
                let prefillWeights = ctx.device.makeBuffer(length: 32 * 4, options: .storageModeShared)!
                S.run(ctx) { cb in
                    let e = cb.makeComputeCommandEncoder()!
                    moe.encodeRouter(e, shape: .inkling2Mini, x: xBuffer, xStride: width, weight: wBuffer, bias: bBuffer,
                                     ids: prefillIDs, routeWeights: prefillWeights, tokens: 1)
                    e.endEncoding()
                }
                let theirIDs = (0..<topK).map { Int(prefillIDs.contents().bindMemory(to: UInt32.self, capacity: 32)[$0]) }
                #expect(gotIDs == theirIDs, "trial \(trial): experts against the prefill router")
                worstPrefill = max(worstPrefill, RelError.computeStrict(actual: gotWeights,
                                                                        reference: S.readFloats(prefillWeights, count: topK)))
            }
        }
        print(String(format: "[ink2-router] experts %ld top-%ld: %ld of 8 trials compared; logits vs CPU %.2e, weights vs CPU %.2e, vs prefill router %.2e",
                     experts, topK, compared, worstLogits, worstReference, worstPrefill))
        #expect(compared >= 6)
        #expect(worstLogits < 1e-5 && worstReference < 1e-5 && worstPrefill < 1e-5)
    }
}
