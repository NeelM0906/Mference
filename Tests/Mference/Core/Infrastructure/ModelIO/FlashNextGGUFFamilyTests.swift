import Foundation
import Testing
@testable import Mference

/// WS4 ruling R1: the Unsloth UD-IQ3_XXS GGUF install is its own family,
/// `qwen38flashnextudiq3xxs`, with its own baseline, refused by name until its
/// runner lands.
@Suite struct FlashNextGGUFFamilyTests {
    private static func writeManifest(family: String, pleNgramShardCount: Int) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-gguf-family-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let manifest: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "unsloth-qwen3.8-flash-next-ud-iq3_xxs",
            "sourceSnapshotHash": "sha256:0",
            "arch": ["family": family, "hiddenSize": 2_560, "numLayers": 48,
                     "pleNgramShardCount": pleNgramShardCount, "pleNgramVocabSizeBase": 0],
            "quant": ["routedExpert": ["scheme": "gguf"]],
            "files": [:], "expertsPerLayer": 512, "numLayers": 48, "expertStride": 2_670_592,
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    /// Refused by name through the capability gate, even with PLE fields that
    /// agree with its own baseline, so the refusal cannot be the accidental
    /// `pleNgramShardCount` mismatch against the vendor baseline.
    @Test func ggufFamilyIsRefusedByName() throws {
        let directory = try Self.writeManifest(family: "qwen38flashnextudiq3xxs", pleNgramShardCount: 1)
        defer { try? FileManager.default.removeItem(at: directory) }
        let expected = ModelError.familyRunnerNotImplemented(
            family: "qwen38flashnextudiq3xxs",
            missingAxes: ManifestReader.qwen38FlashNextGGUFRequiredAxes)
        #expect(throws: expected) { try ManifestReader.peekFamily(directoryURL: directory) }
        #expect(ManifestReader.missingRunnerAxes(forRawFamily: "qwen38flashnextudiq3xxs")
                == ManifestReader.qwen38FlashNextGGUFRequiredAxes)
        // The vendor family stays loadable.
        #expect(ManifestReader.capabilityRefusal(family: "qwen38flashnext") == nil)
    }

    /// R2: today's layout reader refuses GGUF expert storage by its format
    /// string instead of reading GGUF blocks as INT4 or BF16 experts. The keys
    /// mirror `GTurboJSON.encodeGGUFLayout` in the installer.
    @Test func layoutReaderRefusesGGUFExpertStorage() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-gguf-layout-\(UUID().uuidString)")
        let packed = directory.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: packed, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        func expert(_ e: Int) -> [String: Any] {
            ["expert": e, "offset": e * 16_384, "size": 16_384, "tensors": [
                "gate": ["offset": 0, "size": 5_248, "dtype": "GGML", "format": "IQ2_S", "shape": [64, 256]],
                "up": ["offset": 5_376, "size": 5_248, "dtype": "GGML", "format": "IQ2_S", "shape": [64, 256]],
                "down": ["offset": 10_752, "size": 4_608, "dtype": "GGML", "format": "IQ4_NL", "shape": [256, 64]],
            ]]
        }
        let layout: [String: Any] = [
            "expertStride": 16_384, "numLayers": 1, "expertsPerLayer": 2,
            "layers": [["layer": 0, "file": "layer_00.bin", "expertStride": 16_384,
                        "experts": [expert(0), expert(1)]]],
            "expertStorage": ["format": "ggufNativeBlocksV1", "storedExpertStride": 16_384,
                              "storedExpertStrides": [16_384], "slotFormat": "lut4", "slotStrides": [32_768]],
        ]
        try JSONSerialization.data(withJSONObject: layout).write(to: packed.appendingPathComponent("layout.json"))
        do {
            _ = try PackedExpertsLayoutReader.load(directoryURL: directory)
            Issue.record("GGUF expert storage was accepted")
        } catch let error as ModelError {
            #expect(error.description.contains("expertStorage has an unknown format"), "\(error)")
        }
    }

    /// R3: the GGUF family's norms arrive in full form, so the runtime must not
    /// add 1 to them: the family is outside the load-time bake set.
    @Test func ggufNormsAreNotOffsetTwice() {
        #expect(!Model.zeroCenteredNormFamilies.contains(.qwen38flashnextudiq3xxs))
        #expect(Model.zeroCenteredNormFamilies.contains(.qwen38flashnext))
    }

    /// The baseline is the vendor Flash-Next baseline field for field, except
    /// the family and the two storage-shaped PLE fields.
    @Test func baselineDiffersFromVendorOnlyInFamilyAndPleStorage() throws {
        let gguf = try #require(ArchConfig.knownArchitectures[.qwen38flashnextudiq3xxs])
        let vendor = try #require(ArchConfig.knownArchitectures[.qwen38flashnext])
        #expect(gguf.family == .qwen38flashnextudiq3xxs)
        #expect(gguf.flashNext.pleNgramShardCount == 1 && gguf.flashNext.pleNgramVocabSizeBase == 0)
        #expect(vendor.flashNext.pleNgramShardCount == 128 && vendor.flashNext.pleNgramVocabSizeBase == 20_000_000)

        func fields(_ value: Any, skipping: Set<String>) -> [String: String] {
            var out: [String: String] = [:]
            for child in Mirror(reflecting: value).children {
                guard let label = child.label, !skipping.contains(label) else { continue }
                out[label] = String(describing: child.value)
            }
            return out
        }
        #expect(fields(gguf, skipping: ["family", "flashNext"]) == fields(vendor, skipping: ["family", "flashNext"]))
        let pleStorage: Set<String> = ["pleNgramShardCount", "pleNgramVocabSizeBase"]
        #expect(fields(gguf.flashNext, skipping: pleStorage) == fields(vendor.flashNext, skipping: pleStorage))
        #expect(ModelFamily.qwen38flashnextudiq3xxs.maximumContext == ModelFamily.qwen38flashnext.maximumContext)
    }
}
