import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// `FlashNextRouterReference` is the oracle the Metal router kernels are gated
/// against. It is a transcription of `FlashNextReferenceRunner`'s routing block,
/// and a transcription is a claim: this suite discharges it.
///
/// The runner is replayed over the toy install with capture on, and for every
/// (layer, token) the captured MoE input `mlp_hc_mixed` is pushed through the
/// layer's router weight and this oracle. The resulting indices must match the
/// runner's captured `router_indices` exactly and the weights bit for bit —
/// both sides are float32 doing the same operations in the same order, so
/// anything short of bit equality means the transcription drifted.
///
/// No-ops when `scratch/qwen4exp-toy-ckpt-prodlayout/` is absent, like the rest
/// of the Flash-Next parity suites.
@Suite struct FlashNextRouterReferenceTieBackTests {

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func oracleReproducesTheReferenceRunnersRouter(
        prompt: FlashNextGoldens.Prompt) throws {
        guard FlashNextParity.checkpointIsPresent else { return }
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let dir = try FlashNextParity.installToyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }

        let config = FlashNextParity.archConfig()
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: config, streamingMode: .resident)
        let weights = FlashNextWeights(model: model)
        let runner = try FlashNextReferenceRunner(weights: weights)

        let tokens = try FlashNextGoldens.promptTokens(prompt)
        var capture: FlashNextReferenceRunner.Capture? = .init()
        _ = try runner.step(tokens: tokens, capture: &capture)
        let recorded = try #require(capture)

        let hidden = config.hiddenSize
        let experts = config.numExperts
        let k = config.topKExperts
        var checked = 0

        for layer in 0..<config.numLayers {
            let key = FlashNextGoldens.layerKey(layer) + "."
            let mixed = try #require(recorded.floats[key + "mlp_hc_mixed"])
            let expectedIndices = try #require(recorded.integers[key + "router_indices"])
            let expectedWeights = try #require(recorded.floats[key + "router_weights"])
            let router = try weights.moe(layer: layer).router

            for t in 0..<tokens.count {
                let x = Array(mixed[(t * hidden)..<((t + 1) * hidden)])
                let logits = FlashNextRouterReference.matVec(
                    router, rows: experts, cols: hidden, x: x)
                let selection = FlashNextRouterReference.select(logits: logits, k: k)
                #expect(selection.indices == expectedIndices[t],
                        "\(prompt.rawValue) L\(layer) t\(t) router indices")
                #expect(selection.weights
                            == Array(expectedWeights[(t * k)..<((t + 1) * k)]),
                        "\(prompt.rawValue) L\(layer) t\(t) router weights")
                checked += 1
            }
        }
        #expect(checked == config.numLayers * tokens.count)
    }
}
