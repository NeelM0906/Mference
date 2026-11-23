import Foundation
import Testing
@testable import Mference

/// Per-layer expert strides in `layout.json` and the validated
/// `Inkling2ExpertLayout` view of them.
@Suite struct Inkling2ExpertLayoutTests {

    /// A production-shaped Inkling 2 layer entry: `gate_up` at 0, `down` at
    /// the next 256-byte boundary, the blob padded to 16 KiB.
    static func layerEntry(_ layer: Int, format: GGMLBlockFormat, experts: Int = 128,
                           hidden: Int = 2304, width: Int = 1152) -> [String: Any] {
        let gateUp = UInt64(2 * width * format.rowBytes(elements: hidden)!)
        let downOffset = (gateUp + 255) / 256 * 256
        let down = UInt64(hidden * format.rowBytes(elements: width)!)
        let stride = (downOffset + down + 16_383) / 16_384 * 16_384
        let tensors: [String: Any] = [
            "gate_up": ["offset": 0, "size": gateUp, "dtype": "GGML", "format": format.name,
                        "shape": [2 * width, hidden]],
            "down": ["offset": downOffset, "size": down, "dtype": "GGML", "format": format.name,
                     "shape": [hidden, width]],
        ]
        return [
            "layer": layer, "file": String(format: "layer_%02d.bin", layer),
            "expertStride": stride, "format": format.name,
            "experts": (0..<experts).map {
                ["expert": $0, "offset": UInt64($0) * stride, "size": stride, "tensors": tensors] as [String: Any]
            },
        ]
    }

    /// The layout the installer writes for the pinned file, optionally edited.
    static func productionLayout(_ edit: (inout [String: Any]) -> Void = { _ in }) -> [String: Any] {
        let formats = ArchConfig.inkling2Mini.inkling2.routedExpertFormats
        let layers = formats.enumerated().map { layerEntry($0.offset, format: GGMLBlockFormat(name: $0.element)!) }
        var root: [String: Any] = [
            "expertStride": layers.map { $0["expertStride"] as! UInt64 }.max()!,
            "numLayers": 24, "expertsPerLayer": 128, "layers": layers,
        ]
        edit(&root)
        return root
    }

    static func load(_ root: [String: Any]) throws -> PackedExpertsLayout {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-layout-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let packed = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: packed, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
            .write(to: packed.appendingPathComponent("layout.json"))
        return try PackedExpertsLayoutReader.load(directoryURL: dir)
    }

    @Test func perLayerStridesAreReadPerLayer() throws {
        let layout = try Self.load(Self.productionLayout())
        #expect(layout.hasPerLayerStrides)
        #expect(layout.expertStride == 8_470_528)
        #expect(layout.expertStride(layer: 0) == 4_489_216)
        #expect(layout.storedExpertStride(layer: 19) == 4_489_216)
        #expect(layout.expertStride(layer: 20) == 8_470_528)
        #expect(layout.storedExpertStride(of: layout.layers[23]) == 8_470_528)
        #expect(layout.layers[0].format == "NVFP4" && layout.layers[23].format == "Q8_0")
        #expect(layout.expert(layer: 3, expert: 127).offset == 127 * 4_489_216)
        // 20 x 128 x 4.49 MB + 4 x 128 x 8.47 MB, against 26.0 GB at one stride.
        let pool = layout.layers.reduce(UInt64(0)) { $0 + 128 * layout.storedExpertStride(of: $1) }
        #expect(pool == 15_829_303_296)
    }

    /// A layout without per-layer keys reads exactly as before: every layer
    /// at the single stride.
    @Test func singleStrideLayoutIsUnchanged() throws {
        let layout = try Self.load(Self.productionLayout { root in
            root["expertStride"] = UInt64(4_489_216)
            root["layers"] = (root["layers"] as! [[String: Any]]).prefix(20).map { layer -> [String: Any] in
                var layer = layer
                layer.removeValue(forKey: "expertStride")
                layer.removeValue(forKey: "format")
                return layer
            }
            root["numLayers"] = 20
        })
        #expect(!layout.hasPerLayerStrides)
        #expect(layout.layers.allSatisfy { $0.expertStride == nil && $0.format == nil })
        #expect(layout.expertStride(layer: 7) == layout.expertStride)
        #expect(layout.storedExpertStride(layer: 7) == layout.storedExpertStride)
        #expect(layout.expertStride(layer: 99) == layout.expertStride)
    }

    @Test func malformedPerLayerStridesAreRefused() throws {
        func refuses(_ label: String, _ edit: @escaping (inout [[String: Any]], inout [String: Any]) -> Void) {
            let root = Self.productionLayout { root in
                var layers = root["layers"] as! [[String: Any]]
                edit(&layers, &root)
                root["layers"] = layers
            }
            #expect(throws: ModelError.self, "\(label)") { _ = try Self.load(root) }
        }
        refuses("one layer without a stride") { layers, _ in layers[4].removeValue(forKey: "expertStride") }
        refuses("expert off its stride") { layers, _ in
            var experts = layers[2]["experts"] as! [[String: Any]]
            experts[5]["offset"] = UInt64(5 * 4_489_216 + 16_384)
            layers[2]["experts"] = experts
        }
        refuses("top-level stride not the largest") { _, root in root["expertStride"] = UInt64(4_489_216) }
        refuses("zero stride") { layers, _ in layers[0]["expertStride"] = 0 }
        refuses("non-string format") { layers, _ in layers[0]["format"] = 4 }
        refuses("stride and expertStorage together") { _, root in
            root["expertStorage"] = ["format": ExpertStorage.impliedNeg8ScaleBiasesFormat]
        }
    }

    @Test func productionLayoutValidatesAgainstTheBaseline() throws {
        let view = try Inkling2ExpertLayout.validated(layout: try Self.load(Self.productionLayout()),
                                                      config: .inkling2Mini, pageSize: 16_384)
        #expect(view.layers.count == 24)
        let nvfp4 = view.layers[0], q8 = view.layers[23]
        #expect(nvfp4.format == .nvfp4 && nvfp4.expertStride == 4_489_216)
        #expect(nvfp4.gateUp == .init(format: .nvfp4, rows: 2304, columns: 2304, offset: 0,
                                      size: 2_985_984, rowBytes: 1296))
        #expect(nvfp4.down == .init(format: .nvfp4, rows: 2304, columns: 1152, offset: 2_985_984,
                                    size: 1_492_992, rowBytes: 648))
        #expect(q8.format == .q8_0 && q8.expertStride == 8_470_528)
        #expect(q8.gateUp.size == 5_640_192 && q8.down.offset == 5_640_192 && q8.down.size == 2_820_096)
    }

    @Test func layoutDisagreeingWithTheBaselineIsRefused() throws {
        func refuses(_ label: String, _ edit: @escaping (inout [[String: Any]]) -> Void) throws {
            let layout = try Self.load(Self.productionLayout { root in
                var layers = root["layers"] as! [[String: Any]]
                edit(&layers)
                root["layers"] = layers
                root["expertStride"] = layers.map { $0["expertStride"] as! UInt64 }.max()!
            })
            #expect(throws: ModelError.self, "\(label)") {
                _ = try Inkling2ExpertLayout.validated(layout: layout, config: .inkling2Mini, pageSize: 16_384)
            }
        }
        // Layer 3 stored as Q8_0 where the baseline says NVFP4.
        try refuses("format") { layers in layers[3] = Self.layerEntry(3, format: .q8_0) }
        // The layer's own format name disagreeing with its tensors.
        try refuses("layer format") { layers in layers[21]["format"] = "NVFP4" }
        // Gate and up split instead of fused.
        try refuses("roles") { layers in
            var experts = layers[0]["experts"] as! [[String: Any]]
            for i in experts.indices {
                var tensors = experts[i]["tensors"] as! [String: Any]
                tensors["gate"] = tensors.removeValue(forKey: "gate_up")
                experts[i]["tensors"] = tensors
            }
            layers[0]["experts"] = experts
        }
        // A blob whose gate-up shape is transposed.
        try refuses("shape") { layers in
            var experts = layers[1]["experts"] as! [[String: Any]]
            for i in experts.indices {
                var tensors = experts[i]["tensors"] as! [String: [String: Any]]
                tensors["gate_up"]!["shape"] = [2304, 2304 + 64]
                experts[i]["tensors"] = tensors
            }
            layers[1]["experts"] = experts
        }
        // A non-page-aligned stride (with offsets that follow it).
        try refuses("page") { layers in
            var entry = Self.layerEntry(2, format: .nvfp4)
            let stride = UInt64(4_489_216 + 256)
            entry["expertStride"] = stride
            entry["experts"] = (entry["experts"] as! [[String: Any]]).enumerated().map { index, expert in
                var expert = expert
                expert["offset"] = UInt64(index) * stride
                expert["size"] = stride
                return expert
            }
            layers[2] = entry
        }
    }
}
