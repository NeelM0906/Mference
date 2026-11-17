import Foundation
import Metal
import Testing

@testable import Mference

/// WS4 U4: the GGUF install stores each layer's experts at that layer's own
/// stride (`expertStorage.storedExpertStrides`), so the receipt's layer-size
/// check must use it. Before this, a valid receipt failed that check, and
/// `.trustedReceiptWhenValid` silently fell back to hashing every layer file
/// on first touch. The single-stride (vendor) cases are
/// `receiptWhenValidUsesAValidReceipt` and
/// `trustedReceiptModeRejectsWrongSizedLayerFile`, which keep passing.
///
/// The toy install's layout is rewritten into GGUF storage with a different
/// page-aligned stride per layer; only `Model.load` runs (a toy Gemma config
/// never opens a GGUF layer), and the resolved policy says which verifier
/// will check the layer files.
extension ModelLoaderTests {
  static let ggufToyStrides: [UInt64] = [32_768, 49_152]

  /// Rewrites `layout.json`, the layer files and their manifest entries into
  /// GGUF native-block storage with one stride per layer.
  static func rewriteAsMixedStrideGGUF(directoryURL dir: URL,
                                       strides: [UInt64] = ggufToyStrides) throws {
    let toy = ArchConfig.gemma4Toy()
    precondition(strides.count == toy.numLayers)
    let packed = dir.appendingPathComponent("packed_experts")
    let tensors: [String: Any] = [
      "gate": ["offset": 0, "size": 5_248, "dtype": "GGML", "format": "IQ2_S", "shape": [64, 256]],
      "up": ["offset": 5_248, "size": 5_248, "dtype": "GGML", "format": "IQ2_S", "shape": [64, 256]],
      "down": ["offset": 10_496, "size": 9_216, "dtype": "GGML", "format": "IQ4_NL", "shape": [256, 64]],
    ]
    var layers: [[String: Any]] = []
    var files: [String: [String: Any]] = [:]
    for (L, stride) in strides.enumerated() {
      let basename = String(format: "layer_%02d.bin", L)
      layers.append([
        "layer": L, "file": basename, "expertStride": stride,
        "experts": (0..<toy.numExperts).map { e -> [String: Any] in
          ["expert": e, "offset": UInt64(e) * stride, "size": stride, "tensors": tensors]
        },
      ])
      let url = packed.appendingPathComponent(basename)
      try Data(count: Int(stride) * toy.numExperts).write(to: url)
      files["packed_experts/\(basename)"] = [
        "size": Int(stride) * toy.numExperts, "sha256": try Sha256Verifier.hashFile(at: url),
      ]
    }
    let layout: [String: Any] = [
      "expertStride": 16_384, "numLayers": toy.numLayers, "expertsPerLayer": toy.numExperts,
      "layers": layers,
      "expertStorage": [
        "format": "ggufNativeBlocksV1", "slotFormat": "lut4", "storedExpertStride": strides.max()!,
        "storedExpertStrides": strides, "slotStrides": strides.map { _ in 32_768 },
      ],
    ]
    let layoutURL = packed.appendingPathComponent("layout.json")
    let layoutData = try JSONSerialization.data(withJSONObject: layout, options: [.sortedKeys])
    try layoutData.write(to: layoutURL)
    files["packed_experts/layout.json"] = [
      "size": layoutData.count, "sha256": try Sha256Verifier.hashFile(at: layoutURL),
    ]

    let manifestURL = dir.appendingPathComponent("manifest.json")
    var root = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
    var manifestFiles = root["files"] as! [String: Any]
    for (rel, entry) in files { manifestFiles[rel] = entry }
    root["files"] = manifestFiles
    try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys, .withoutEscapingSlashes])
      .write(to: manifestURL)
  }

  static func writeMixedStrideGGUFToy() throws -> URL {
    let dir = try writeToySynthetic()
    try rewriteAsMixedStrideGGUF(directoryURL: dir)
    try writeVerifiedInstallReceipt(directoryURL: dir)
    return dir
  }

  @Test func mixedStrideGGUFInstallPassesTheTrustedReceiptCheck() throws {
    let dir = try Self.writeMixedStrideGGUFToy()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy(),
                               integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.packedExpertsLayout.gguf?.layers.map(\.storedStride) == Self.ggufToyStrides)
    #expect(model.integrityPolicy == .sizeCheckTrustedReceipt)
  }

  @Test func receiptWhenValidUsesTheReceiptForAMixedStrideGGUFInstall() throws {
    let dir = try Self.writeMixedStrideGGUFToy()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    // The CLI and server default: the size-check verifier, not first-touch SHA-256.
    let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy(),
                               integrityPolicy: .trustedReceiptWhenValid)
    #expect(model.integrityPolicy == .sizeCheckTrustedReceipt)
  }

  @Test func mixedStrideGGUFWrongSizedLayerFileFailsTheReceiptCheck() throws {
    let dir = try Self.writeMixedStrideGGUFToy()
    defer { try? FileManager.default.removeItem(at: dir) }
    // Layer 1 cut to layer 0's size: a valid size for some layer, wrong for this one.
    let layerURL = dir.appendingPathComponent("packed_experts").appendingPathComponent("layer_01.bin")
    let handle = try FileHandle(forWritingTo: layerURL)
    try handle.truncate(atOffset: Self.ggufToyStrides[0] * UInt64(ArchConfig.gemma4Toy().numExperts))
    try handle.close()
    let device = try #require(MTLCreateSystemDefaultDevice())

    #expect {
      _ = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy(),
                         integrityPolicy: .sizeCheckTrustedReceipt)
    } throws: { error in
      if case ModelError.trustedReceiptInvalid(let detail) = error {
        return detail.contains("packed_experts/layer_01.bin")
      }
      return false
    }
    let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy(),
                               integrityPolicy: .trustedReceiptWhenValid)
    #expect(model.integrityPolicy == .fullSha256)
  }
}
