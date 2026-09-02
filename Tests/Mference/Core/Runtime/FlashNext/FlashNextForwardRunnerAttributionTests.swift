import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// Attribution for the drift `FlashNextForwardRunnerParityTests` measures.
///
/// That suite compares the runner against the float32 reference forward, so any
/// difference is a *sum* of two things: the kernels' own arithmetic error, and
/// the runner's sensitivity to an input that has already drifted. Those need
/// separating before a number can be called a tolerance or a bug.
///
/// This suite re-runs the MoE block on the CPU from the runner's **own**
/// captured input and its **own** captured routing, and compares against the
/// runner's captured output. That isolates the kernel: whatever is left is the
/// expert-compute path's error at production scale on real toy weights, with no
/// input difference at all.
///
/// The same idea applied to the attention block is `FlashNextIndexerKernelTests`,
/// which feeds both sides identical FP16 activations.
///
/// No-ops when `scratch/qwen4exp-toy-ckpt-prodlayout/` is absent.
@Suite struct FlashNextForwardRunnerAttributionTests {

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func moeBlockMatchesTheOracleOnTheRunnersOwnInput(
        prompt: FlashNextGoldens.Prompt) async throws {
        guard FlashNextParity.checkpointIsPresent else { return }
        let context = try MetalContext()
        let dir = try FlashNextParity.installToyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = FlashNextParity.archConfig()
        let streamed = try Model.load(directoryURL: dir, device: context.device,
                                      expecting: config,
                                      streamingMode: .pread(slotCount: 16))
        let resident = try Model.load(directoryURL: dir, device: context.device,
                                      expecting: config, streamingMode: .resident)
        let weights = FlashNextWeights(model: resident)
        let runner = try FlashNextForwardRunner(model: streamed, context: context,
                                                maxContext: 64)
        guard let logits = context.device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else { return }

        let tokens = try FlashNextGoldens.promptTokens(prompt)
        let hidden = config.hiddenSize
        let k = config.topKExperts
        var worstAbs: Float = 0
        var worstRelative: Float = 0
        var worstMagnitude: Float = 0
        var worstAt = ""
        var checked = 0

        for (index, token) in tokens.enumerated() {
            runner.capture = .init()
            try await runner.produce(token: Int32(token), position: index,
                                     into: logits)
            let recorded = try #require(runner.capture)
            for layer in 0..<config.numLayers {
                let key = FlashNextGoldens.layerKey(layer) + "."
                let x = try #require(recorded.floats[key + "mlp_hc_mixed"])
                let out = try #require(recorded.floats[key + "moe_out"])
                let indices = try #require(
                    recorded.integers[key + "router_indices"])[0]
                let routing = try #require(recorded.floats[key + "router_weights"])
                #expect(indices.count == k && routing.count == k)

                let moe = try weights.moe(layer: layer)
                let experts = try indices.map { e -> FlashNextExpertReference.Expert in
                    let w = try weights.expert(layer: layer, expert: e)
                    return .init(gate: w.gate, up: w.up, down: w.down)
                }
                let expected = FlashNextExpertReference.block(
                    experts: experts, weights: routing,
                    shared: .init(gateRow: moe.sharedGate,
                                  gateProj: moe.sharedGateProj,
                                  upProj: moe.sharedUpProj,
                                  downProj: moe.sharedDownProj),
                    x: x, hidden: hidden,
                    moeIntermediate: config.moeIntermediateSize,
                    sharedIntermediate: config.intermediateSize)

                var maxAbs: Float = 0
                var magnitude: Float = 0
                for i in 0..<hidden {
                    maxAbs = max(maxAbs, abs(out[i] - expected[i]))
                    magnitude = max(magnitude, abs(expected[i]))
                }
                if maxAbs > worstAbs {
                    worstAbs = maxAbs
                    worstMagnitude = magnitude
                    worstAt = "t\(index) L\(layer)"
                }
                worstRelative = max(worstRelative, maxAbs / max(magnitude, 1e-6))
                checked += 1
            }
        }
        runner.capture = nil

        print("flashnext MoE attribution (\(prompt.rawValue)): over \(checked) "
                + "(position, layer) pairs the GPU block differs from the CPU "
                + "oracle on the SAME input by at most \(worstAbs) "
                + "(reference magnitude \(worstMagnitude) at \(worstAt)), "
                + "worst relative \(worstRelative)")

        // Fed identical input and identical routing, the two sides differ only
        // by FP16 activations inside the expert FFN. Anything beyond that is a
        // kernel bug, not sensitivity.
        #expect(worstRelative < 5e-2,
                Comment(rawValue: "\(prompt.rawValue) MoE block relative error "
                            + "\(worstRelative) on the runner's own input"))
    }
}
