import Foundation
import Metal
import Testing
@testable import Mference

@Suite struct FlashNextMTPWeightsTests {
    private func load(_ directory: URL) throws -> Model {
        try Model.load(directoryURL: directory, device: #require(MTLCreateSystemDefaultDevice()),
                       expecting: .qwen38FlashNextToy())
    }

    @Test func loadsOneFullAttentionLayerAndItsIndependentPool() throws {
        let dir = try FlashNextToySynthetic.write(includeMTP: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = try load(dir)
        let weights = try FlashNextMTPWeights(model: model)
        #expect(weights.expertLayout.path == dir.appendingPathComponent("packed_experts_mtp/layer_00.bin").path)
        #expect(weights.expertLayout.expertStride == 16_384)
        #expect(weights.expertLayout.expertsPerLayer == model.config.numExperts)
        #expect(weights.expertOffsets.gateWOff == 0)
        #expect(weights.expertOffsets.upWOff == 2_304)
        #expect(weights.expertOffsets.downWOff == 4_608)
        #expect(weights.mixer.inject == nil)
        #expect(weights.attentionHC.inject != nil && weights.mlpHC.inject != nil)
        for (name, cooked) in [("mtp.pre_fc_norm_embedding.weight", weights.embeddingNorm),
                               ("mtp.pre_fc_norm_hidden.weight", weights.hiddenNorm)] {
            let raw = try model.resident(name: name)
            let src = raw.buffer.contents().advanced(by: Int(raw.offset)).assumingMemoryBound(to: UInt16.self)
            let dst = cooked.buffer.contents().advanced(by: Int(cooked.offset)).assumingMemoryBound(to: UInt16.self)
            for i in 0..<(Int(raw.length) / 2) {
                #expect(dst[i] == Quantization.bf16Bits(Quantization.bf16ToFloat(src[i]) + 1))
            }
            #expect(try model.normWeight(name: name).buffer === cooked.buffer)
        }
        #expect(!Model.isZeroCenteredNorm("other.pre_fc_norm_embedding.weight"))
        #expect(!Model.isZeroCenteredNorm("mtp.fc_hidden.weight"))
    }

    @Test func absentSidecarRefusesWithoutChangingOrdinaryLoad() throws {
        let dir = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = try load(dir)
        #expect(throws: ModelError.self) { _ = try FlashNextMTPWeights(model: model) }
    }

    @Test func preFoldedSidecarNormsAreNotFoldedTwice() throws {
        let dir = try FlashNextToySynthetic.write(includeMTP: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try editManifest(dir) { $0["zeroCenteredNormsBakedAtInstall"] = true }
        let model = try load(dir)
        let weights = try FlashNextMTPWeights(model: model)
        let raw = try model.resident(name: "mtp.pre_fc_norm_hidden.weight")
        #expect(weights.hiddenNorm.buffer === raw.buffer)
        #expect(weights.hiddenNorm.offset == raw.offset)
    }

    @Test(arguments: ["stride", "count", "directory", "layer", "duplicate", "size", "hash"])
    func malformedPoolThrows(_ defect: String) throws {
        let dir = try FlashNextToySynthetic.write(includeMTP: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try editManifest(dir) { root in
            var pools = root["auxiliaryExpertPools"] as! [[String: Any]]
            switch defect {
            case "stride": pools[0]["expertStride"] = 32_768
            case "count": pools[0]["expertsPerLayer"] = 1
            case "directory": pools[0]["directory"] = "../packed_experts_mtp"
            case "layer": pools[0]["layers"] = [["layer": 1, "file": "layer_00.bin"]]
            case "duplicate": pools.append(pools[0])
            default:
                var files = root["files"] as! [String: [String: Any]]
                if defect == "size" { files["packed_experts_mtp/layer_00.bin"]!["size"] = 1 }
                else { files["packed_experts_mtp/layer_00.bin"]!["sha256"] = String(repeating: "0", count: 64) }
                root["files"] = files
            }
            root["auxiliaryExpertPools"] = pools
        }
        let model = try load(dir)
        #expect(throws: ModelError.self) { _ = try FlashNextMTPWeights(model: model) }
    }

    @Test func truncatedAuxiliaryFileIsRejected() throws {
        let dir = try FlashNextToySynthetic.write(includeMTP: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data([0]).write(to: dir.appendingPathComponent("packed_experts_mtp/layer_00.bin"))
        let model = try load(dir)
        #expect(throws: ModelError.self) { _ = try FlashNextMTPWeights(model: model) }
    }

    @Test(arguments: [0, 1, 2, 3, 4, 5, 6, 7])
    func validatesDtypesAndCompanionsBeforeMatrixDispatch(_ variant: Int) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let buffer = try #require(device.makeBuffer(length: 16_384, options: .storageModeShared))
        // INT4, INT8 and BF16 are valid; unsupported width, short scale and
        // out-of-buffer companion slices must throw before `.from` can trap.
        let length: UInt64 = variant == 1 ? 4_096 : variant == 2 ? 8_192 : variant == 3 ? 1_024 : 2_048
        let view = TensorView(buffer: buffer, offset: 0, length: length,
            scaleOffset: variant == 5 ? 16_384 : 8_192,
            scaleLength: variant == 2 ? 0 : variant == 4 ? 126 : 128,
            biasOffset: 8_320, biasLength: variant == 2 ? 0 : 128,
            shape: (64, variant == 7 ? 32 : 64, 0, 0), dtype: variant == 2 ? 1 : variant == 6 ? 3 : 0)
        if variant < 3 {
            try FlashNextMTPWeights.validate(view, name: "fixture", rows: 64, columns: 64)
            _ = FlashNextWeightMatrix.from(view)
        } else {
            #expect(throws: ModelError.self) {
                try FlashNextMTPWeights.validate(view, name: "fixture", rows: 64, columns: 64)
            }
        }
    }

    @Test func installedSidecarLoadsWithNativeDtypes() throws {
        guard let path = ProcessInfo.processInfo.environment["MFERENCE_FLASHNEXT_GTURBO"] else { return }
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
            device: #require(MTLCreateSystemDefaultDevice()), streamingMode: .pread(slotCount: 16))
        let weights = try FlashNextMTPWeights(model: model)
        #expect(weights.expertLayout.expertsPerLayer == 512)
        #expect(weights.expertLayout.expertStride == 2_768_896)
        #expect(weights.hiddenNorm.shape.0 == 10_240)
        #expect(weights.embeddingNorm.shape.0 == 2_560)
        // This gate names the INT8-router install, not a uniform-INT4 legacy
        // conversion. The loader itself accepts either stored projection width.
        if case .int8 = weights.router {} else { Issue.record("expected native INT8 MTP router") }
        if case .int8 = weights.sharedGate {} else { Issue.record("expected native INT8 shared gate") }
        if case .int4 = weights.hiddenProjection {} else { Issue.record("expected INT4 MTP projection") }
    }

    private func editManifest(_ directory: URL, _ edit: (inout [String: Any]) -> Void) throws {
        let url = directory.appendingPathComponent("manifest.json")
        var root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        edit(&root)
        try JSONSerialization.data(withJSONObject: root, options: .sortedKeys).write(to: url)
    }
}
