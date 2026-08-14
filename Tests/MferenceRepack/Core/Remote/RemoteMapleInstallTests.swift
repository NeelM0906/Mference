import Foundation
import Testing

@testable import MferenceRepackCore

extension RemotePayloadCopyTests {
  @Test func mapleRemoteInstallPreservesTernaryBytesAndManifest() async throws {
    let snapshotDir = tmpDirForRemote("maple-snap")
    let output = tmpPathForRemote("maple-remote")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snapshot = try SyntheticSnapshot.buildMaple(
      at: snapshotDir,
      seed: 0x4D41_504C_4500_0001,
      includeFlashHead: true)
    let source = try mapleSourceTensors(in: snapshot.shardPath)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snapshot,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: true)
    let result = try await RemoteStreamingRepacker(
      options: remoteOptions(
        outputDir: output,
        session: fakeHFSession(),
        rangeChunkBytes: 4096)
    ).run()

    #expect(result.plan.arch.family == .maple)
    #expect(result.excludedMultimodalTensorCount == 4)
    #expect(result.outputBytes > result.remoteBytesToDownload)
    #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)

    let residentPath = (output as NSString).appendingPathComponent("model_weights.bin")
    let resident = try Data(contentsOf: URL(fileURLWithPath: residentPath))
    let entries = try mapleResidentEntries(in: resident)
    let centroids = try #require(entries["lm_head_flash.centroids.weight"])
    let tokenMap = try #require(entries["lm_head_flash.token_map"])
    #expect(centroids.size == UInt64(4_748 * 2_048 / 2))
    #expect(centroids.scaleSize == UInt64(4_748 * (2_048 / 64) * 2))
    #expect(centroids.biasSize == centroids.scaleSize)
    #expect(tokenMap.size == UInt64(4_748 * 32 * 4))
    #expect(entries["lm_head_flash.cluster_scale"] == nil)
    #expect(entries["lm_head_flash.head.weight"] == nil)
    for (name, offset, size) in [
      ("lm_head_flash.centroids.weight", centroids.offset, centroids.size),
      ("lm_head_flash.centroids.scales", centroids.scaleOffset, centroids.scaleSize),
      ("lm_head_flash.centroids.biases", centroids.biasOffset, centroids.biasSize),
      ("lm_head_flash.token_map", tokenMap.offset, tokenMap.size),
    ] {
      let sourceTensor = try #require(source[name])
      #expect(try mapleBytes(resident, offset: offset, count: size)
        == mapleBytes(at: snapshot.shardPath, tensor: sourceTensor))
    }
    let q = try #require(entries["model.layers.0.self_attn.q_proj.weight"])
    let sourceQ = try #require(source["model.layers.0.self_attn.q_proj.weight"])
    let sourceAlpha = try #require(source["model.layers.0.self_attn.q_proj.row_alpha"])
    #expect(try mapleBytes(resident, offset: q.offset, count: q.size)
      == mapleExpandedInt2(try mapleBytes(at: snapshot.shardPath, tensor: sourceQ)))
    #expect(try mapleBytes(resident, offset: q.scaleOffset, count: q.scaleSize)
      == mapleRepeatedBF16(try mapleBytes(at: snapshot.shardPath, tensor: sourceAlpha),
                           count: 2, negated: false))
    #expect(try mapleBytes(resident, offset: q.biasOffset, count: q.biasSize)
      == mapleRepeatedBF16(try mapleBytes(at: snapshot.shardPath, tensor: sourceAlpha),
                           count: 2, negated: true))

    let layout = try mapleJSON(
      (output as NSString).appendingPathComponent("packed_experts/layout.json"))
    let layer = try #require((layout["layers"] as? [[String: Any]])?.first)
    let experts = try #require(layer["experts"] as? [[String: Any]])
    for expert in [0, 255] {
      let layoutExpert = try #require(experts.first { $0["expert"] as? Int == expert })
      let tensors = try #require(layoutExpert["tensors"] as? [String: [String: Any]])
      let expertOffset = try #require(layoutExpert["offset"] as? Int)
      let expertPath = ((output as NSString).appendingPathComponent("packed_experts") as NSString)
        .appendingPathComponent(try #require(layer["file"] as? String))
      let expertBytes = try Data(contentsOf: URL(fileURLWithPath: expertPath))
      for role in ["gate", "up", "down"] {
        let tensor = try #require(tensors[role])
        let tensorOffset = try #require(tensor["offset"] as? Int)
        let tensorSize = try #require(tensor["size"] as? Int)
        let sourceWeight = try #require(source[
          "model.layers.0.mlp.switch_mlp.\(role)_proj.weight"])
        let sourceOffset = sourceWeight.offset + UInt64(expert * Int(sourceWeight.size) / 256)
        let expected = try mapleBytes(
          at: snapshot.shardPath, offset: sourceOffset, count: UInt64(tensorSize))
        #expect(try mapleBytes(expertBytes,
                               offset: UInt64(expertOffset + tensorOffset),
                               count: UInt64(tensorSize)) == expected)
      }
    }

    let manifest = try mapleJSON((output as NSString).appendingPathComponent("manifest.json"))
    try mapleExpectManifest(manifest, expertStride: try #require(manifest["expertStride"] as? Int))
    let flashHead = try #require(manifest["flashHead"] as? [String: Any])
    #expect(flashHead["nClusters"] as? Int == 4_748)
    #expect(flashHead["clusterSize"] as? Int == 32)
    #expect(flashHead["nProbes"] as? Int == 512)

    let verify = try VerifiedInstallTool.run(options: VerifyInstallOptions(inputGTurbo: output))
    #expect(verify.unexpectedEntries.isEmpty)
    #expect(FileManager.default.fileExists(atPath: verify.receiptPath))
  }

  @Test func mapleResumeSkipsCommittedTransformedSourceRange() async throws {
    let snapshotDir = tmpDirForRemote("maple-snap-resume")
    let output = tmpPathForRemote("maple-remote-resume")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snapshot = try SyntheticSnapshot.buildMaple(
      at: snapshotDir,
      seed: 0x4D41_504C_4500_0002)
    let sourceStart = try mapleSourceTensors(in: snapshot.shardPath).values.map(\.offset).min()!

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snapshot,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    let task = Task {
      try await RemoteStreamingRepacker(
        options: remoteOptions(outputDir: output, session: fakeHFSession())
      ).run { progress in
        guard case .copyingPayload(_, let downloaded, _) = progress,
              downloaded > 0,
              let checkpoint = try? RemoteInstallCheckpoint.load(from: output + ".resume.json"),
              checkpoint.completedRanges.contains(where: { $0.destinationBytes > $0.sourceBytes }) else {
          return
        }
        withUnsafeCurrentTask { $0?.cancel() }
      }
    }
    await #expect(throws: CancellationError.self) { _ = try await task.value }

    let checkpoint = try RemoteInstallCheckpoint.load(from: output + ".resume.json")
    #expect(checkpoint.completedRanges.count == 1)
    #expect(checkpoint.completedRanges[0].destinationBytes > checkpoint.completedRanges[0].sourceBytes)
    let before = maplePayloadRanges(startingAt: sourceStart)

    _ = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: output, session: fakeHFSession(), resume: true)
    ).run()

    let after = maplePayloadRanges(startingAt: sourceStart)
    #expect(after.dropFirst(before.count).allSatisfy { !before.contains($0) })
  }
}

private struct MapleRemoteTensor {
  let offset: UInt64
  let size: UInt64
}

private struct MapleResidentEntry {
  let offset: UInt64
  let size: UInt64
  let scaleOffset: UInt64
  let scaleSize: UInt64
  let biasOffset: UInt64
  let biasSize: UInt64
}

private func mapleSourceTensors(in path: String) throws -> [String: MapleRemoteTensor] {
  let data = try Data(contentsOf: URL(fileURLWithPath: path))
  let headerSize = try mapleUInt64(data, at: 0)
  let header = try JSONSerialization.jsonObject(with: mapleBytes(
    data, offset: 8, count: headerSize)) as! [String: [String: Any]]
  return try header.reduce(into: [:]) { result, item in
    if item.key == "__metadata__" { return }
    guard let offsets = item.value["data_offsets"] as? [Any], offsets.count == 2,
          let start = offsets[0] as? NSNumber,
          let end = offsets[1] as? NSNumber else {
      throw NSError(domain: "RemoteMapleInstallTests", code: 1)
    }
    result[item.key] = MapleRemoteTensor(
      offset: 8 + headerSize + start.uint64Value,
      size: end.uint64Value - start.uint64Value)
  }
}

private func mapleResidentEntries(in data: Data) throws -> [String: MapleResidentEntry] {
  let count = try mapleUInt64(data, at: 16)
  return try (0..<Int(count)).reduce(into: [:]) { result, index in
    let base = 24 + index * 72
    let nameOffset = try mapleUInt32(data, at: base)
    let nameLength = try mapleUInt16(data, at: base + 4)
    let name = String(decoding: try mapleBytes(data, offset: UInt64(nameOffset),
                                                count: UInt64(nameLength)), as: UTF8.self)
    result[name] = MapleResidentEntry(
      offset: try mapleUInt64(data, at: base + 8),
      size: try mapleUInt64(data, at: base + 16),
      scaleOffset: try mapleUInt64(data, at: base + 40),
      scaleSize: try mapleUInt64(data, at: base + 48),
      biasOffset: try mapleUInt64(data, at: base + 56),
      biasSize: try mapleUInt64(data, at: base + 64))
  }
}

private func mapleJSON(_ path: String) throws -> [String: Any] {
  try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as! [String: Any]
}

private func mapleBytes(at path: String, tensor: MapleRemoteTensor) throws -> Data {
  try mapleBytes(at: path, offset: tensor.offset, count: tensor.size)
}

private func mapleBytes(at path: String, offset: UInt64, count: UInt64) throws -> Data {
  try mapleBytes(Data(contentsOf: URL(fileURLWithPath: path)), offset: offset, count: count)
}

private func mapleBytes(_ data: Data, offset: UInt64, count: UInt64) throws -> Data {
  guard offset <= UInt64(data.count), count <= UInt64(data.count) - offset else {
    throw NSError(domain: "RemoteMapleInstallTests", code: 2)
  }
  return Data(data[Int(offset)..<Int(offset + count)])
}

private func mapleUInt16(_ data: Data, at offset: Int) throws -> UInt16 {
  let bytes = try mapleBytes(data, offset: UInt64(offset), count: 2)
  return UInt16(bytes[bytes.startIndex]) | UInt16(bytes[bytes.startIndex + 1]) << 8
}

private func mapleUInt32(_ data: Data, at offset: Int) throws -> UInt32 {
  let bytes = try mapleBytes(data, offset: UInt64(offset), count: 4)
  return (0..<4).reduce(0) { $0 | UInt32(bytes[bytes.startIndex + $1]) << UInt32($1 * 8) }
}

private func mapleUInt64(_ data: Data, at offset: Int) throws -> UInt64 {
  let bytes = try mapleBytes(data, offset: UInt64(offset), count: 8)
  return (0..<8).reduce(0) { $0 | UInt64(bytes[bytes.startIndex + $1]) << UInt64($1 * 8) }
}

private func mapleExpandedInt2(_ source: Data) -> Data {
  precondition(source.count.isMultiple(of: 4))
  var output = Data(capacity: source.count * 2)
  for offset in stride(from: 0, to: source.count, by: 4) {
    let word = UInt32(source[source.startIndex + offset])
      | UInt32(source[source.startIndex + offset + 1]) << 8
      | UInt32(source[source.startIndex + offset + 2]) << 16
      | UInt32(source[source.startIndex + offset + 3]) << 24
    for half in 0..<2 {
      // Explicit loop, not a closed-form reduce: Swift 6.1 (the macOS 15 CI
      // floor) cannot type-check the reduce form in reasonable time.
      var expanded: UInt32 = 0
      for nibble in 0..<8 {
        let pair = (word >> UInt32((half * 8 + nibble) * 2)) & 3
        expanded |= pair << UInt32(nibble * 4)
      }
      output.append(UInt8(truncatingIfNeeded: expanded))
      output.append(UInt8(truncatingIfNeeded: expanded >> 8))
      output.append(UInt8(truncatingIfNeeded: expanded >> 16))
      output.append(UInt8(truncatingIfNeeded: expanded >> 24))
    }
  }
  return output
}

private func mapleRepeatedBF16(_ source: Data, count: Int, negated: Bool) -> Data {
  precondition(source.count.isMultiple(of: 2))
  var output = Data(capacity: source.count * count)
  for offset in stride(from: 0, to: source.count, by: 2) {
    for _ in 0..<count {
      output.append(source[source.startIndex + offset])
      output.append(source[source.startIndex + offset + 1] ^ (negated ? 0x80 : 0))
    }
  }
  return output
}

private func maplePayloadRanges(startingAt sourceStart: UInt64) -> [String] {
  (FakeHFURLProtocol.requestedRanges["model-00001-of-00001.safetensors"] ?? []).filter {
    guard let range = $0.split(separator: "=").last?.split(separator: "-").first,
          let offset = UInt64(range) else { return false }
    return offset >= sourceStart
  }
}

private func mapleExpectManifest(_ manifest: [String: Any], expertStride: Int) throws {
  let expectedArch: [String: Any] = [
    "hiddenSize": 2_048, "ffnIntermediate": 512, "moeIntermediateSize": 512,
    "numHeads": 16, "numKVHeads": 4, "numFullKVHeads": 4, "headDim": 128,
    "fullHeadDim": 128, "vocabSize": 151_936, "slidingWindow": 512,
    "finalLogitSoftcap": 0.0, "ropeTheta": 10_000.0, "fullRopeTheta": 0.0,
    "partialRotaryFactor": 0.5, "numLayers": 24, "numExperts": 256,
    "topKExperts": 8, "tieWordEmbeddings": false, "attentionKEqV": false,
    "hiddenActivation": "silu",
    "fullAttentionLayerMask": (0..<24).map { $0 % 4 == 3 ? 1 : 0 },
    "family": "maple", "attnOutputGate": false,
    "attentionScale": 1.0 / Double(128).squareRoot(),
    "embeddingScaledBySqrtHidden": false, "routerScaled": false,
    "ffnSandwichNorms": false, "sharedExpertGated": false, "ropeNeoxSubdim": true,
    "linearNumKHeads": 0, "linearNumVHeads": 0, "linearKeyHeadDim": 0,
    "linearValueHeadDim": 0, "linearConvKernelSize": 0,
    "routerScoringFunc": "softmax", "routedScalingFactor": 1.0,
    "swigluLimit": 7.0, "numSharedExperts": 0, "numDenseLayers": 0,
    "routerNormAfterTopK": true,
  ]
  let affine: [String: Any] = [
    "weightBits": 4, "scheme": "affine", "scaleType": "BF16", "biasType": "BF16", "groupSize": 64,
  ]
  let expectedQuant: [String: Any] = [
    "embedding": affine, "attention": affine,
    "router": ["weightBits": 16, "scheme": "unquantized", "scaleType": "none", "biasType": "none", "groupSize": 0],
    "sharedExpert": ["weightBits": 0, "scheme": "none", "scaleType": "none", "biasType": "none", "groupSize": 0],
    "routedExpert": ["weightBits": 2, "scheme": "affine", "scaleType": "BF16", "biasType": "BF16", "groupSize": 64],
  ]
  #expect(try JSONSerialization.data(
    withJSONObject: #require(manifest["arch"] as? [String: Any]), options: [.sortedKeys])
    == JSONSerialization.data(withJSONObject: expectedArch, options: [.sortedKeys]))
  #expect(try JSONSerialization.data(
    withJSONObject: #require(manifest["quant"] as? [String: Any]), options: [.sortedKeys])
    == JSONSerialization.data(withJSONObject: expectedQuant, options: [.sortedKeys]))
  #expect(manifest["magic"] as? String == "GTURBO")
  #expect(manifest["versionMajor"] as? Int == 1)
  #expect(manifest["versionMinor"] as? Int == 0)
  #expect(manifest["modelID"] as? String == "unknown/snapshot")
  #expect(manifest["expertsPerLayer"] as? Int == 256)
  #expect(manifest["numLayers"] as? Int == 24)
  #expect(manifest["expertStride"] as? Int == expertStride)
  #expect(manifest["bitWidthOverridesHonored"] as? Int == 2)
}
