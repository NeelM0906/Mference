import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// Discharges the transcription claim behind `FlashNextHyperConnectionReference`.
///
/// The reference runner captures, per layer, the raw stream going into each
/// hyper-connection (`attn_hc_stream_in` / `mlp_hc_stream_in`) alongside the
/// `mixed` and `inject` it produced. Replaying those captured inputs through
/// this oracle must reproduce those captured outputs bit for bit — both sides
/// are float32 doing the same operations in the same order.
///
/// The global mixer is covered too: `last_hidden_state` is the mixer applied to
/// the final layer's `stream_out`, with no inject path.
///
/// No-ops when `scratch/qwen4exp-toy-ckpt-prodlayout/` is absent.
@Suite struct FlashNextHyperConnectionReferenceTieBackTests {

    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func oracleReproducesTheReferenceRunnersHyperConnections(
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
        let rows = tokens.count

        let geometry = FlashNextHyperConnectionReference.Geometry(
            hidden: config.hiddenSize,
            hcCount: config.flashNext.hcCount,
            lowRank: config.flashNext.hcLowRank,
            eps: 1e-6)

        func check(_ site: Model.HyperConnectionSite, layer: Int, key: String) throws {
            let w = try weights.hyperConnection(site: site, layer: layer)
            let stream = try #require(recorded.floats[key + "_stream_in"])
            let result = FlashNextHyperConnectionReference.gatedResidual(
                stream,
                .init(norm: w.norm, mixDown: w.mixDown, mixUp: w.mixUp,
                      inject: w.inject),
                rows: rows, g: geometry)
            #expect(result.mixed == (try #require(recorded.floats[key + "_mixed"])),
                    "\(prompt.rawValue) \(key) mixed")
            #expect(result.inject == (try #require(recorded.floats[key + "_inject"])),
                    "\(prompt.rawValue) \(key) inject")
        }

        for layer in 0..<config.numLayers {
            let p = FlashNextGoldens.layerKey(layer) + "."
            try check(.attention, layer: layer, key: p + "attn_hc")
            try check(.mlp, layer: layer, key: p + "mlp_hc")
        }

        // The global mixer: same gated residual, inject path absent.
        let mixer = try weights.globalMixer()
        let finalStream = try #require(
            recorded.floats[FlashNextGoldens.layerKey(config.numLayers - 1)
                                + ".stream_out"])
        let mixed = FlashNextHyperConnectionReference.gatedResidual(
            finalStream,
            .init(norm: mixer.norm, mixDown: mixer.mixDown, mixUp: mixer.mixUp,
                  inject: nil),
            rows: rows, g: geometry)
        #expect(mixed.inject == nil)
        #expect(mixed.mixed == (try #require(recorded.floats["last_hidden_state"])),
                "\(prompt.rawValue) global mixer")
    }

    /// The injection add, replayed the same way: `stream_in` of the MLP
    /// hyper-connection is exactly the attention `stream_in` plus the attention
    /// block output scaled by the attention inject gate.
    @Test(arguments: FlashNextGoldens.Prompt.allCases)
    func oracleReproducesTheReferenceRunnersInjectionAdd(
        prompt: FlashNextGoldens.Prompt) throws {
        guard FlashNextParity.checkpointIsPresent else { return }
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let dir = try FlashNextParity.installToyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }

        let config = FlashNextParity.archConfig()
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: config, streamingMode: .resident)
        let runner = try FlashNextReferenceRunner(
            weights: FlashNextWeights(model: model))
        let tokens = try FlashNextGoldens.promptTokens(prompt)
        var capture: FlashNextReferenceRunner.Capture? = .init()
        _ = try runner.step(tokens: tokens, capture: &capture)
        let recorded = try #require(capture)

        let geometry = FlashNextHyperConnectionReference.Geometry(
            hidden: config.hiddenSize,
            hcCount: config.flashNext.hcCount,
            lowRank: config.flashNext.hcLowRank,
            eps: 1e-6)

        for layer in 0..<config.numLayers {
            let p = FlashNextGoldens.layerKey(layer) + "."
            let injected = FlashNextHyperConnectionReference.injectBlock(
                try #require(recorded.floats[p + "attn_hc_stream_in"]),
                block: try #require(recorded.floats[p + "block_out"]),
                inject: try #require(recorded.floats[p + "attn_hc_inject"]),
                rows: tokens.count, g: geometry)
            #expect(injected == (try #require(recorded.floats[p + "mlp_hc_stream_in"])),
                    "\(prompt.rawValue) L\(layer) attention injection add")

            let outgoing = FlashNextHyperConnectionReference.injectBlock(
                try #require(recorded.floats[p + "mlp_hc_stream_in"]),
                block: try #require(recorded.floats[p + "moe_out"]),
                inject: try #require(recorded.floats[p + "mlp_hc_inject"]),
                rows: tokens.count, g: geometry)
            #expect(outgoing == (try #require(recorded.floats[p + "stream_out"])),
                    "\(prompt.rawValue) L\(layer) MoE injection add")
        }
    }
}
