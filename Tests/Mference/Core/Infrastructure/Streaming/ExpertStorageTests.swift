import CryptoKit
import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// Implied expert biases: the converter drops `-8 * scale` bias arrays and the
/// streamers rebuild them, so every slot must hold the explicit install's bytes.
@Suite(.serialized) struct ExpertStorageTests {
    @Test func runtimeAndConverterAgreeOnEveryBF16Pattern() {
        let scales = (0...UInt16.max).map { $0 }
        var filled = [UInt16](repeating: 0, count: scales.count)
        scales.withUnsafeBytes { source in
            filled.withUnsafeMutableBytes { target in
                ExpertStorage.fillNeg8ScaleBiases(scales: source.baseAddress!,
                                                  biases: target.baseAddress!, count: scales.count)
            }
        }
        var mismatches = 0
        for scale in scales {
            let runtime = ExpertStorage.neg8ScaleBits(scale)
            if runtime != QATImpliedBiasConverter.neg8ScaleBits(scale) || filled[Int(scale)] != runtime {
                mismatches += 1
            }
            let value = Float(bitPattern: UInt32(scale) << 16)
            let expected = value * -8
            if value.isFinite && expected.isFinite {
                #expect(Float(bitPattern: UInt32(runtime) << 16) == expected, "scale bits \(scale)")
            }
        }
        #expect(mismatches == 0)
    }

    @Test func convertedExpertsReadBackByteIdentical() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let result = try QATImpliedBiasConverter.run(inputGTurbo: fixture.input.path,
                                                     outputGTurbo: fixture.output.path,
                                                     pageSize: Fixture.page)
        #expect(result.storedExpertStride == Fixture.storedStride)
        #expect(result.routedBytesAfter < result.routedBytesBefore)

        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: fixture.output.appendingPathComponent("manifest.json"))) as! [String: Any]
        let quant = manifest["quant"] as! [String: [String: Any]]
        #expect(quant["routedExpert"]?["biasType"] as? String == GemmaQATCheckpoint.impliedRoutedBiasType)
        let layout = try PackedExpertsLayoutReader.load(directoryURL: fixture.output)
        let storage = try #require(layout.storage)
        #expect(layout.expertStride == Fixture.expertStride)
        #expect(storage.storedExpertStride == Fixture.storedStride)

        let device = try MetalContext().device
        for layer in 0..<Fixture.layers {
            let stream = StreamLayout(
                path: fixture.output.appendingPathComponent("packed_experts/layer_0\(layer).bin").path,
                streamOffset: 0,
                streamSize: UInt64(Fixture.experts) * storage.storedExpertStride,
                expertsPerLayer: Fixture.experts,
                expertStride: layout.expertStride,
                expertOffsets: layout.layers[layer].experts.map(\.offset),
                storage: storage)

            // One coalesced run of every expert, then single loads.
            let streamer = try PreadExpertStreamer(layout: stream, device: device, slotCount: Fixture.experts)
            let loaded = try streamer.loadExpertsCached(experts: Array(0..<Fixture.experts))
            for (expert, view) in loaded.enumerated() {
                try fixture.expectExpert(layer: layer, expert: expert,
                                         bytes: view.buffer.contents().advanced(by: Int(view.offset)))
            }
            let single = try PreadExpertStreamer(layout: stream, device: device, slotCount: 1)
            for expert in 0..<Fixture.experts {
                let view = try single.loadExpert(layer: 0, expert: expert, slot: 0)
                try fixture.expectExpert(layer: layer, expert: expert,
                                         bytes: view.buffer.contents().advanced(by: Int(view.offset)))
            }
            // Speculative reads publish through the same reader.
            let speculative = try PreadExpertStreamer(layout: stream, device: device, slotCount: Fixture.experts)
            let reservation = speculative.reserveSpeculativeSlots(experts: [3, 1], keepEvictable: 0)
            #expect(speculative.executeSpeculativeReservation(reservation)
                    == UInt64(reservation.count) * storage.storedExpertStride)
            let hits = try speculative.executeExpertCachePlan(
                speculative.planExpertsCached(experts: [3, 1]))
            for (index, expert) in [3, 1].enumerated() {
                try fixture.expectExpert(layer: layer, expert: expert,
                                         bytes: hits[index].buffer.contents().advanced(by: Int(hits[index].offset)))
            }
            // Resident mode cannot map a compact file; it expands a copy.
            let resident = try ResidentExpertStreamer(layout: stream, device: device, strategy: .mapped)
            #expect(resident.strategy == .copied)
            for expert in 0..<Fixture.experts {
                let view = try resident.expertBuffer(layer: 0, expert: expert)
                try fixture.expectExpert(layer: layer, expert: expert,
                                         bytes: view.buffer.contents().advanced(by: Int(view.offset)))
            }
        }
    }

    @Test func converterRefusesABiasThatIsNotNegativeEightTimesScale() throws {
        let fixture = try Fixture(corruptBiasInLayer: 1)
        defer { fixture.remove() }
        #expect(throws: RepackError.self) {
            _ = try QATImpliedBiasConverter.run(inputGTurbo: fixture.input.path,
                                                outputGTurbo: fixture.output.path,
                                                pageSize: Fixture.page)
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
    }

    @Test func layoutReaderRejectsStorageThatDropsAKernelTensor() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        _ = try QATImpliedBiasConverter.run(inputGTurbo: fixture.input.path,
                                            outputGTurbo: fixture.output.path,
                                            pageSize: Fixture.page)
        let url = fixture.output.appendingPathComponent("packed_experts/layout.json")
        var root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        var storage = root["expertStorage"] as! [String: Any]
        var segments = storage["segments"] as! [[String: Any]]
        segments.removeLast()
        storage["segments"] = segments
        root["expertStorage"] = storage
        try JSONSerialization.data(withJSONObject: root).write(to: url)
        #expect(throws: ModelError.self) {
            _ = try PackedExpertsLayoutReader.load(directoryURL: fixture.output)
        }
    }
}

/// A two-layer, four-expert install shaped like Gemma 4 QAT's experts:
/// gate/up/down with BF16 scales and `-8 * scale` biases, three pages per
/// expanded expert and two once the biases are dropped.
private struct Fixture {
    static let page: UInt64 = 16_384
    static let layers = 2
    static let experts = 4
    static let weights = 6_144
    static let scales = 4_096
    static let expertStride: UInt64 = 3 * page
    static let storedStride: UInt64 = 2 * page
    static let roles = ["gate", "up", "down"]

    let root: URL
    var input: URL { root.appendingPathComponent("in.gturbo") }
    var output: URL { root.appendingPathComponent("out.gturbo") }
    var experts: [[Data]] = []

    static func offsets(_ role: Int) -> (weights: Int, scales: Int, biases: Int) {
        let base = role * (weights + 2 * scales)
        return (base, base + weights, base + weights + scales)
    }

    init(corruptBiasInLayer corrupt: Int? = nil) throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-storage-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: input.appendingPathComponent("packed_experts"),
                                                withIntermediateDirectories: true)
        var generator = SystemRandomNumberGenerator()
        var files: [String: [String: Any]] = [:]
        func record(_ relative: String, _ data: Data) throws {
            try data.write(to: input.appendingPathComponent(relative))
            let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            files[relative] = ["size": data.count, "sha256": sha]
        }
        var tensors: [String: [String: Any]] = [:]
        for (index, role) in Self.roles.enumerated() {
            let at = Self.offsets(index)
            tensors[role] = ["offset": at.weights, "size": Self.weights, "dtype": "U32", "shape": [8, 192], "bits": 4]
            tensors[role + "_scales"] = ["offset": at.scales, "size": Self.scales, "dtype": "BF16", "shape": [8, 256]]
            tensors[role + "_biases"] = ["offset": at.biases, "size": Self.scales, "dtype": "BF16", "shape": [8, 256]]
        }
        var layoutLayers: [[String: Any]] = []
        for layer in 0..<Self.layers {
            var blob = Data(count: Self.experts * Int(Self.expertStride))
            var layerExperts: [Data] = []
            for expert in 0..<Self.experts {
                var bytes = Data(count: Int(Self.expertStride))
                for index in 0..<Self.roles.count {
                    let at = Self.offsets(index)
                    for i in 0..<Self.weights { bytes[at.weights + i] = UInt8.random(in: 0...255, using: &generator) }
                    for g in 0..<(Self.scales / 2) {
                        // Positive normal BF16 scales, like the checkpoint's.
                        let scale = UInt16(0x3A00 + Int.random(in: 0..<0x0600, using: &generator))
                        var bias = QATImpliedBiasConverter.neg8ScaleBits(scale)
                        if layer == corrupt && expert == 2 && index == 1 && g == 7 { bias ^= 1 }
                        bytes[at.scales + 2 * g] = UInt8(scale & 0xFF)
                        bytes[at.scales + 2 * g + 1] = UInt8(scale >> 8)
                        bytes[at.biases + 2 * g] = UInt8(bias & 0xFF)
                        bytes[at.biases + 2 * g + 1] = UInt8(bias >> 8)
                    }
                }
                blob.replaceSubrange(expert * Int(Self.expertStride)..<(expert + 1) * Int(Self.expertStride),
                                     with: bytes)
                layerExperts.append(bytes)
            }
            experts.append(layerExperts)
            try record("packed_experts/layer_0\(layer).bin", blob)
            layoutLayers.append(["layer": layer, "file": "layer_0\(layer).bin",
                                 "experts": (0..<Self.experts).map { expert -> [String: Any] in
                                     ["expert": expert, "offset": expert * Int(Self.expertStride),
                                      "size": Int(Self.expertStride), "tensors": tensors]
                                 }])
        }
        try record("model_weights.bin", Data((0..<4_096).map { UInt8($0 & 0xFF) }))
        let layout: [String: Any] = ["expertStride": Int(Self.expertStride), "numLayers": Self.layers,
                                     "expertsPerLayer": Self.experts, "layers": layoutLayers]
        try record("packed_experts/layout.json", try JSONSerialization.data(withJSONObject: layout))
        let slot: [String: Any] = ["weightBits": 4, "groupSize": 32, "scheme": "affine",
                                   "scaleType": "BF16", "biasType": "BF16"]
        let manifest: [String: Any] = [
            "modelID": QATImpliedBiasConverter.modelID, "expertStride": Int(Self.expertStride),
            "expertsPerLayer": Self.experts, "numLayers": Self.layers, "files": files,
            "quant": ["embedding": slot, "attention": slot, "sharedExpert": slot, "routedExpert": slot],
        ]
        try JSONSerialization.data(withJSONObject: manifest)
            .write(to: input.appendingPathComponent("manifest.json"))
    }

    /// Compares every tensor byte of an expanded expert with the input.
    func expectExpert(layer: Int, expert: Int, bytes: UnsafeMutableRawPointer) throws {
        let original = experts[layer][expert]
        let used = Self.roles.count * (Self.weights + 2 * Self.scales)
        let actual = Data(bytes: bytes, count: used)
        #expect(actual == original.prefix(used), "layer \(layer) expert \(expert)")
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
        for directory in [input, output] {
            try? FileManager.default.removeItem(atPath: directory.path + ".install.lock")
        }
    }
}
