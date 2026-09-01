import Foundation
import Metal
import Testing

@testable import Mference

/// Drafter checkpoint validation: a config whose shared dimensions disagree
/// with the target must be rejected before any buffer is built — the
/// drafter shares the target's embedding/LM head and consumes tap captures
/// of the target's hidden rows, so a mismatch would mis-stride kernels
/// instead of failing cleanly. Config-only fixtures: every guard fires
/// before `model.safetensors` is opened.
@Suite struct Qwen38DFlash2ConfigTests {

    private func writeConfig(hiddenSize: Int, vocabSize: Int,
                             targetLayers: Int) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash2-config-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: dir,
                                                withIntermediateDirectories: true)
        let config: [String: Any] = [
            "dflash_config": [
                "block_size": 8,
                "conv_group_size": 64,
                "conv_kernel_size": 4,
                "mask_token_id": 0,
                "selector_rank": 64,
                "selector_top_k": 16,
                "target_layer_ids": [0, 1],
            ],
            "head_dim": 128,
            "hidden_size": hiddenSize,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
            "num_target_layers": targetLayers,
            "rms_norm_eps": 1e-6,
            "sliding_window": 128,
            "vocab_size": vocabSize,
            "rope_parameters": ["rope_theta": 10000.0],
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    private func makeDrafter(directory: URL) throws -> Qwen38DFlash2Drafter {
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.qwen38])
        let dummy = try #require(ctx.device.makeBuffer(length: 64,
                                                       options: .storageModeShared))
        let view = TensorView(buffer: dummy, offset: 0, length: 64,
                              scaleOffset: 0, scaleLength: 0,
                              biasOffset: 0, biasLength: 0,
                              shape: (0, 0, 0, 0), dtype: 0)
        return try Qwen38DFlash2Drafter(context: ctx,
                                        directory: directory,
                                        embedding: view,
                                        lmHead: view,
                                        targetConfig: cfg,
                                        precision: .bf16)
    }

    @Test func rejectsHiddenSizeMismatch() throws {
        let cfg = try #require(ArchConfig.knownArchitectures[.qwen38])
        let dir = try writeConfig(hiddenSize: cfg.hiddenSize + 1024,
                                  vocabSize: cfg.vocabSize,
                                  targetLayers: cfg.numLayers)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: Qwen38ForwardRunnerError.self) {
            _ = try makeDrafter(directory: dir)
        }
    }

    @Test func rejectsVocabMismatch() throws {
        let cfg = try #require(ArchConfig.knownArchitectures[.qwen38])
        let dir = try writeConfig(hiddenSize: cfg.hiddenSize,
                                  vocabSize: cfg.vocabSize + 1,
                                  targetLayers: cfg.numLayers)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: Qwen38ForwardRunnerError.self) {
            _ = try makeDrafter(directory: dir)
        }
    }
}
