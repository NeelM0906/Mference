import Foundation
import Testing
@testable import MferenceRepackCore

/// Why the reference-parity install is written by `FlashNextParity` instead of
/// being produced by the production installer.
///
/// The intent was to run the real `ArchInfo.load` -> `FlashNextPlanner.plan`
/// path over the parity harness's production-layout toy checkpoint, so the
/// bytes would flow reference -> installer -> loader -> forward. It cannot: the
/// toy checkpoint is refused three times over. These tests pin each refusal, so
/// the day the toy grows or the planner gains a no-quantize mode, the ones that
/// stop applying fail here and this fixture can be deleted.
///
/// Skips when `scratch/qwen4exp-toy-ckpt-prodlayout/` has not been regenerated.
@Suite struct FlashNextToyRepackBlockerTests {

    private func configPath(_ body: [String: Any]) throws -> String {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4exp-config-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: body).write(to: url)
        return url.path
    }

    /// Blocker 1 and 2, and the fact that repairing them is enough for
    /// `ArchInfo`: the harness saves the *text-only* config
    /// (`model_type: "qwen4_exp_text"`, no `text_config` wrapper) with
    /// `layer_types` already normalized to `"qwen_sparse_attention"`.
    ///
    /// Both are artifacts of `save_pretrained` writing the post-`__post_init__`
    /// config, not production gaps: the real vendor `config.json` is a
    /// `qwen4_exp` multimodal wrapper whose `layer_types` say `"full_attention"`
    /// — exactly what `ArchInfo` accepts — and `Qwen4ExpTextConfig.__post_init__`
    /// rewrites them on load (`configuration_qwen4_exp.py:180-183`).
    @Test func toyConfigIsRefusedUntilItIsReshapedIntoProductionForm() throws {
        guard FlashNextParity.checkpointIsPresent else { return }

        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf:
            FlashNextParity.checkpointDirectory
                .appendingPathComponent("config.json"))) as? [String: Any]
        let text = try #require(raw)
        #expect(text["model_type"] as? String == "qwen4_exp_text")
        #expect(text["text_config"] == nil)
        #expect((text["layer_types"] as? [String])?.contains("qwen_sparse_attention")
                == true)

        // As shipped: no text_config, so the loader cannot even find the axes.
        #expect(throws: (any Error).self) {
            _ = try ArchInfo.load(configPath: try configPath(text))
        }

        // With only the wrapper repaired, the normalized layer type is refused
        // by name — the diagnostic a real bad config would produce.
        var wrapped = text
        wrapped["layer_types"] = text["layer_types"]
        #expect(throws: (any Error).self) {
            _ = try ArchInfo.load(configPath: try configPath(
                ["model_type": "qwen4_exp", "text_config": wrapped]))
        }

        // Both repaired: ArchInfo accepts it and derives the runtime axes.
        let arch = try ArchInfo.load(
            configPath: try configPath(FlashNextParity.productionShapedConfig()))
        #expect(arch.family == .qwen38flashnext)
        #expect(arch.numLayers == 6)
        #expect(arch.fullAttentionLayerMask == [2, 2, 2, 1, 2, 1])
        #expect(arch.moeIntermediateSize == 32)
        #expect(arch.flashNext?.hcCount == 4)
        #expect(arch.flashNext?.indexerBudget == 8)
        #expect(arch.flashNext?.pleLayerIDs == [2])
    }

    /// Blocker 3, the structural one: `moe_intermediate_size` is 32 and the
    /// planner quantizes every routed expert in flight, which needs the expert
    /// intermediate dim to be a multiple of the INT4 group size (64).
    ///
    /// This is not repairable from the test side. Removing it means either
    /// widening the toy — which regenerates every committed golden — or adding
    /// a no-quantize mode to `FlashNextPlanner`, which is a production feature.
    @Test func toyExpertWidthCannotBeGroup64Quantized() throws {
        #expect(FlashNextParity.archConfig().moeIntermediateSize
                % StreamingInt4Quantizer.groupSize != 0)
        #expect(StreamingInt4Quantizer
            .isQuantizableRowDim(FlashNextParity.archConfig().moeIntermediateSize)
                == false)
        // The hidden dim is fine; it is only the expert intermediate that fails,
        // and only for the fused `down_proj`, whose row length is that dim.
        #expect(StreamingInt4Quantizer
            .isQuantizableRowDim(FlashNextParity.archConfig().hiddenSize))
    }

    /// The resident half of the planner's policy does hold for this checkpoint:
    /// the fixture writes the same tensors the planner would keep BF16, and
    /// `FlashNextWeights(int4RoundTrip:)` quantizes exactly the rest.
    @Test func residentQuantizationPolicyAgreesWithTheFixture() throws {
        guard FlashNextParity.checkpointIsPresent else { return }
        let source = try Safetensors(url: FlashNextParity.checkpointDirectory
            .appendingPathComponent("model.safetensors"))
        var quantized = 0
        var passthrough = 0
        for name in source.names
        where FlashNextPlanner.fusedExpertRole(in: name) == nil
            && FlashNextPlanner.ngramShardIndex(in: name) == nil {
            let entry = try source.entry(name)
            let tensor = SourceTensor(
                name: name, shardPath: "/dev/null",
                dtype: entry.dtype == "I64" ? .i64 : .bf16,
                shape: entry.shape.map(UInt64.init),
                absoluteOffset: 0, sizeBytes: UInt64(entry.end - entry.begin))
            if FlashNextPlanner.quantizesResident(tensor) { quantized += 1 }
            else { passthrough += 1 }
        }
        #expect(quantized > 0)
        #expect(passthrough > 0)
        // Norms, 1-D vectors, conv kernels and the I64 tables ride through.
        for name in ["model.language_model.layers.0.linear_attn.A_log",
                     "model.language_model.layers.0.linear_attn.conv1d.weight",
                     "model.language_model.layers.1.ple.norm_key.weight",
                     "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"] {
            let entry = try source.entry(name)
            let tensor = SourceTensor(
                name: name, shardPath: "/dev/null",
                dtype: entry.dtype == "I64" ? .i64 : .bf16,
                shape: entry.shape.map(UInt64.init),
                absoluteOffset: 0, sizeBytes: UInt64(entry.end - entry.begin))
            #expect(FlashNextPlanner.quantizesResident(tensor) == false, "\(name)")
        }
    }
}
