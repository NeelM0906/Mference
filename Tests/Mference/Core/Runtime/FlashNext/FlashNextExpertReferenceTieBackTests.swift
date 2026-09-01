import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// `FlashNextExpertReference` is the oracle the Metal routed-expert kernels are
/// gated against. It is a transcription of `FlashNextReferenceRunner.sparseMoE`'s
/// expert block, and a transcription is a claim: this suite discharges it.
///
/// The runner is replayed over the toy install with capture on, and for every
/// (layer, token) its captured MoE input `mlp_hc_mixed`, `router_indices` and
/// `router_weights` are pushed through this oracle. The result must equal the
/// runner's captured `moe_out` **bit for bit** — both sides are float32 doing the
/// same operations in the same order, so anything short of bit equality means the
/// transcription drifted.
///
/// Expert weights come through the same `FlashNextWeights.expert(layer:expert:)`
/// the runner uses, i.e. through the real `packed_experts/layout.json` and the
/// real streaming backend, so the dtype dispatch is exercised too.
///
/// No-ops when `scratch/qwen4exp-toy-ckpt-prodlayout/` is absent, like the rest
/// of the Flash-Next parity suites.
@Suite struct FlashNextExpertReferenceTieBackTests {

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func oracleReproducesTheReferenceRunnersExpertBlock(
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
        let k = config.topKExperts
        var checked = 0

        for layer in 0..<config.numLayers {
            let key = FlashNextGoldens.layerKey(layer) + "."
            let mixed = try #require(recorded.floats[key + "mlp_hc_mixed"])
            let expected = try #require(recorded.floats[key + "moe_out"])
            let indices = try #require(recorded.integers[key + "router_indices"])
            let routerWeights = try #require(recorded.floats[key + "router_weights"])
            let moe = try weights.moe(layer: layer)
            let shared = FlashNextExpertReference.Shared(
                gateRow: moe.sharedGate,
                gateProj: moe.sharedGateProj,
                upProj: moe.sharedUpProj,
                downProj: moe.sharedDownProj)

            for t in 0..<tokens.count {
                let x = Array(mixed[(t * hidden)..<((t + 1) * hidden)])
                let experts = try indices[t].map { e -> FlashNextExpertReference.Expert in
                    let w = try weights.expert(layer: layer, expert: e)
                    return .init(gate: w.gate, up: w.up, down: w.down)
                }
                let actual = FlashNextExpertReference.block(
                    experts: experts,
                    weights: Array(routerWeights[(t * k)..<((t + 1) * k)]),
                    shared: shared,
                    x: x,
                    hidden: hidden,
                    moeIntermediate: config.moeIntermediateSize,
                    sharedIntermediate: config.intermediateSize)
                #expect(actual == Array(expected[(t * hidden)..<((t + 1) * hidden)]),
                        "\(prompt.rawValue) L\(layer) t\(t) moe_out")
                checked += 1
            }
        }
        #expect(checked == config.numLayers * tokens.count)
    }
}
