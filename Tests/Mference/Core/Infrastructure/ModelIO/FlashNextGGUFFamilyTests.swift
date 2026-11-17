import Foundation
import Testing
@testable import Mference

/// WS4 ruling R1: the Unsloth UD-IQ3_XXS GGUF install is its own family,
/// `qwen38flashnextudiq3xxs`, with its own baseline. U4 wired it onto
/// `FlashNextForwardRunner`, so it is no longer gated.
@Suite struct FlashNextGGUFFamilyTests {
    /// U4: the family runs on `FlashNextForwardRunner`, so the capability gate
    /// no longer lists it. The vendor family stays ungated too.
    @Test func ggufFamilyIsNoLongerGated() throws {
        #expect(ManifestReader.capabilityRefusal(family: "qwen38flashnextudiq3xxs") == nil)
        #expect(ManifestReader.missingRunnerAxes(forRawFamily: "qwen38flashnextudiq3xxs") == nil)
        #expect(ManifestReader.capabilityRefusal(family: "qwen38flashnext") == nil)
    }

    /// With the capability gate lifted, the paths the GGUF install does not
    /// run still refuse it by name: resident expert mode (the slots are where
    /// the raw blocks become LUT4), native MTP (the install carries no
    /// sidecar), and every plain expert read of a raw-expansion layout
    /// (`PreadExpertStreamerRawTests`). The slot cache is not refused, and the
    /// vendor family keeps both.
    @Test func unwiredPathsStillRefuseTheGGUFFamilyByName() {
        #expect(Model.ggufExpertModeRefusal(family: .qwen38flashnextudiq3xxs, mode: .resident)
                == .familyRunnerNotImplemented(family: "qwen38flashnextudiq3xxs",
                                               missingAxes: ["residentExpertMode"]))
        #expect(Model.ggufExpertModeRefusal(family: .qwen38flashnextudiq3xxs,
                                            mode: .pread(slotCount: 16)) == nil)
        #expect(!FlashNextMTPWeights.supportedFamilies.contains(.qwen38flashnextudiq3xxs))
        #expect(FlashNextMTPWeights.supportedFamilies.contains(.qwen38flashnext))
    }

    /// One layer of two experts in the installer's GGUF layout keys
    /// (`GTurboJSON.encodeGGUFLayout`): IQ2_S gate/up [64, 256], IQ4_NL down
    /// [256, 64], stored back to back at a fixed stride.
    static func ggufLayout(downFormat: String = "IQ4_NL", downSize: Int = 9_216,
                           slotStride: Int = 32_768) -> [String: Any] {
        func expert(_ e: Int) -> [String: Any] {
            ["expert": e, "offset": e * 20_480, "size": 20_480, "tensors": [
                "gate": ["offset": 0, "size": 5_248, "dtype": "GGML", "format": "IQ2_S", "shape": [64, 256]],
                "up": ["offset": 5_248, "size": 5_248, "dtype": "GGML", "format": "IQ2_S", "shape": [64, 256]],
                "down": ["offset": 10_496, "size": downSize, "dtype": "GGML", "format": downFormat,
                         "shape": [256, 64]],
            ]]
        }
        return [
            "expertStride": 20_480, "numLayers": 1, "expertsPerLayer": 2,
            "layers": [["layer": 0, "file": "layer_00.bin", "expertStride": 20_480,
                        "experts": [expert(0), expert(1)]]],
            "expertStorage": ["format": "ggufNativeBlocksV1", "storedExpertStride": 20_480,
                              "storedExpertStrides": [20_480], "slotFormat": "lut4",
                              "slotStrides": [slotStride]],
        ]
    }

    static func loadLayout(_ layout: [String: Any]) throws -> PackedExpertsLayout {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-gguf-layout-\(UUID().uuidString)")
        let packed = directory.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: packed, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: layout).write(to: packed.appendingPathComponent("layout.json"))
        return try PackedExpertsLayoutReader.load(directoryURL: directory)
    }

    /// R2 / U4: the layout reader parses GGUF expert storage into per-layer
    /// stored and slot strides and LUT4 slot sections (indices, D, M per
    /// projection, 16-aligned), and never as INT4 or BF16 experts.
    @Test func layoutReaderParsesGGUFExpertStorage() throws {
        let layout = try Self.loadLayout(Self.ggufLayout())
        #expect(layout.storage == nil)
        let layer = try #require(layout.gguf?.layers.first)
        #expect(layer.storedStride == 20_480 && layer.slotStride == 32_768)
        #expect(layer.gate.type == .iq2_s && layer.up.type == .iq2_s && layer.down.type == .iq4_nl)
        #expect(layer.gate.storedOffset == 0 && layer.up.storedOffset == 5_248 && layer.down.storedOffset == 10_496)
        // IQ2_S, 16,384 weights: 8,192 B indices, 128 B D, 1,024 B M.
        #expect([layer.gate.indexOffset, layer.gate.dOffset, layer.gate.mOffset] == [0, 8_192, 8_320])
        #expect([layer.up.indexOffset, layer.up.dOffset, layer.up.mOffset] == [9_344, 17_536, 17_664])
        // IQ4_NL: 8,192 B indices, 1,024 B D, no M.
        #expect([layer.down.indexOffset, layer.down.dOffset, layer.down.mOffset] == [18_688, 26_880, 27_904])
        #expect(layer.slotBytes == 27_904 && layer.gate.blocks == 64 && layer.down.blocks == 512)
        for p in layer.projections {
            #expect(p.indexOffset % 16 == 0 && p.dOffset % 16 == 0 && p.mOffset % 16 == 0)
        }
    }

    /// Blocks that do not tile their shape, a slot too small for its LUT4
    /// sections, and a format with no LUT4 form are all refused.
    @Test func layoutReaderRefusesInconsistentGGUFStorage() throws {
        for (label, layout) in [("block size", Self.ggufLayout(downSize: 4_608)),
                                ("slot stride", Self.ggufLayout(slotStride: 16_384)),
                                ("format", Self.ggufLayout(downFormat: "Q4_K"))] {
            #expect(throws: ModelError.self, "\(label)") { _ = try Self.loadLayout(layout) }
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
