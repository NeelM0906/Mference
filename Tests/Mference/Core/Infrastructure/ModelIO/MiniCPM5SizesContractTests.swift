import Foundation
import Testing
@testable import MferenceRepackCore
@testable import Mference

/// The two MiniCPM5 sizes from the repacker's mirror to the runtime's
/// validation: each size's `config.json` installs as its own family, the
/// manifest the repacker writes validates against that family's baseline,
/// and neither size passes as the other.
@Suite struct MiniCPM5SizesContractTests {

    /// The vendor `config.json` keys `ArchInfo.loadMiniCPM5` reads, at either
    /// size (`openbmb/MiniCPM5-2B` @ `cd199ce3`, `openbmb/MiniCPM5-1B-MLX` @
    /// `c973e24a`).
    private static func config(hidden: Int, intermediate: Int, layers: Int) -> [String: Any] {
        [
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_act": "silu",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 131_072,
            "rms_norm_eps": 1e-06,
            "rope_scaling": NSNull(),
            "rope_theta": 5_000_000,
            "tie_word_embeddings": false,
            "vocab_size": 130_560,
        ]
    }

    private static func temporaryDirectory(_ tag: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm5-sizes-\(tag)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// The manifest the repacker writes for a pre-quantized INT4 g64 install
    /// of the given config, in a directory of its own.
    private static func installManifest(_ config: [String: Any], modelID: String) throws -> URL {
        let dir = try temporaryDirectory(modelID)
        let configURL = dir.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: config).write(to: configURL)
        let arch = try ArchInfo.load(configPath: configURL.path)
        let plan = RepackPlan(
            arch: arch, baseMode: "affine", baseGroupSize: 64, bitsOverrideCount: 0,
            resident: ResidentFilePlan(path: "model_weights.bin", entries: [], stringTable: [],
                                       stringTableOffsets: [], indexSize: 0, residentSize: 0),
            layers: [], matchedModelID: modelID, excludedMultimodalTensorNames: [], flashHead: nil)
        let files: [(relativePath: String, info: GTurboJSON.FileEntry)] = [
            ("model_weights.bin", .init(size: 16_384, sha256: String(repeating: "0", count: 64))),
            ("packed_experts/layout.json", .init(size: 16, sha256: String(repeating: "0", count: 64))),
        ]
        let data = try GTurboJSON.encodeManifest(
            plan: plan, modelID: modelID, sourceSnapshotHash: "sha256:0",
            files: files, expertsPerLayer: 0, numLayers: arch.numLayers, expertStride: 0,
            bitWidths: .init(embedding: 4, attention: 4, router: 0, sharedExpert: 0, routedExpert: 0))
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    @Test func oneBInstallValidatesAsItsOwnFamily() throws {
        let dir = try Self.installManifest(Self.config(hidden: 1536, intermediate: 4608, layers: 24),
                                           modelID: "minicpm5-1b-mlx-4bit")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try ManifestReader.peekFamily(directoryURL: dir) == .minicpm5_1b)
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: .miniCPM5_1B)
        #expect(manifest.arch.family == "minicpm5_1b")
        #expect(manifest.arch.qkNorm == false)
        #expect(throws: ModelError.self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .miniCPM5_2B)
        }
    }

    @Test func twoBInstallIsUnchanged() throws {
        let dir = try Self.installManifest(Self.config(hidden: 2048, intermediate: 6144, layers: 42),
                                           modelID: "minicpm5-2b-mlx-4bit")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try ManifestReader.peekFamily(directoryURL: dir) == .minicpm5)
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: .miniCPM5_2B)
        #expect(manifest.arch.family == "minicpm5")
        #expect(throws: ModelError.self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .miniCPM5_1B)
        }
    }

    /// Both sizes run on the shared MiniCPM5 runner and render the chat
    /// template of their own checkpoint.
    @Test func bothSizesShareTheRunnerContract() {
        for family in [ModelFamily.minicpm5, .minicpm5_1b] {
            #expect(family.isMiniCPM5)
            #expect(family.maximumContext == 131_072)
            #expect(ManifestReader.familiesWithoutRunner[family.rawValue] == nil)
        }
        #expect(MFTokenizer.MiniCPMTemplateVariant(family: .minicpm5) == .v2B)
        #expect(MFTokenizer.MiniCPMTemplateVariant(family: .minicpm5_1b) == .v1B)
        #expect(MFTokenizer.MiniCPMTemplateVariant(family: nil) == .v2B)
        #expect(!ModelFamily.qwen36.isMiniCPM5)
    }
}
