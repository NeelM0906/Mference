import Foundation
import Testing

@testable import MferenceRepackCore

extension RemotePayloadCopyTests {
  @Test func qwen38RemoteInstallCompletesWithDenseManifest() async throws {
    let snapshotDir = tmpDirForRemote("qwen38-snap")
    let remoteOutput = tmpPathForRemote("qwen38-remote")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snapshot = try SyntheticSnapshot.buildQwen38(
      at: snapshotDir,
      seed: 0x3800_0102_0304_0506)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snapshot,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: true)
    let recorder = InstallProgressRecorder()

    let result = try await RemoteStreamingRepacker(
      options: remoteOptions(
        outputDir: remoteOutput,
        session: fakeHFSession())
    ).run { recorder.append($0) }

    #expect(result.reusedBytes == 0)
    #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)
    #expect(result.expertLayerCount == 4)
    #expect(result.excludedMultimodalTensorCount == 2)
    for relativePath in [
      "model_weights.bin",
      "packed_experts/layout.json",
      "manifest.json",
    ] {
      let remote = (remoteOutput as NSString).appendingPathComponent(relativePath)
      #expect(FileManager.default.fileExists(atPath: remote))
    }
    // Dense: no packed-expert blob is created for any layer.
    for layer in 0..<4 {
      let blob = (remoteOutput as NSString).appendingPathComponent(
        String(format: "packed_experts/layer_%02d.bin", layer))
      #expect(!FileManager.default.fileExists(atPath: blob))
    }
    #expect(recorder.values.contains(.finalizing))
    try assertRemoteTokenizerFilesRecorded(
      outputDir: remoteOutput,
      expectsOptionalSpecialTokens: true)

    // The manifest carries the qwen38 family extension fields and the
    // zeroed MoE geometry.
    let manifestData = try Data(contentsOf: URL(fileURLWithPath:
      (remoteOutput as NSString).appendingPathComponent("manifest.json")))
    let manifest = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
    #expect(manifest["expertsPerLayer"] as? Int == 0)
    #expect((manifest["expertStride"] as? NSNumber)?.uint64Value == 0)
    let arch = manifest["arch"] as! [String: Any]
    #expect(arch["family"] as? String == "qwen38")
    #expect(arch["attnOutputGate"] as? Bool == true)
    #expect(arch["attentionScale"] as? Double == 0.125)
    #expect(arch["sharedExpertGated"] as? Bool == false)
    #expect(arch["ropeNeoxSubdim"] as? Bool == true)
    #expect(arch["numExperts"] as? Int == 0)
    #expect(arch["topKExperts"] as? Int == 0)
    #expect(arch["numSharedExperts"] as? Int == 0)
    #expect(arch["numDenseLayers"] as? Int == 4)
    #expect(arch["denseIntermediateSize"] as? Int == 64)
    #expect(arch["linearNumKHeads"] as? Int == 2)
    #expect(arch["linearNumVHeads"] as? Int == 4)
    #expect(arch["linearKeyHeadDim"] as? Int == 32)
    #expect(arch["linearValueHeadDim"] as? Int == 32)
    #expect(arch["linearConvKernelSize"] as? Int == 4)
    #expect(arch["fullAttentionLayerMask"] as? [Int] == [2, 2, 2, 1])
    #expect(arch["tieWordEmbeddings"] as? Bool == false)
    #expect(arch["hiddenActivation"] as? String == "silu")

    let quant = manifest["quant"] as! [String: [String: Any]]
    #expect(quant["embedding"]?["weightBits"] as? Int == 4)
    #expect(quant["attention"]?["weightBits"] as? Int == 4)
    for slot in ["router", "sharedExpert", "routedExpert"] {
      #expect(quant[slot]?["weightBits"] as? Int == 0)
      #expect(quant[slot]?["scheme"] as? String == "none")
      #expect(quant[slot]?["groupSize"] as? Int == 0)
    }

    // The finished install passes post-hoc verification.
    let verify = try VerifiedInstallTool.run(
      options: VerifyInstallOptions(inputGTurbo: remoteOutput))
    #expect(verify.unexpectedEntries.isEmpty)
    #expect(verify.fileCount > 0)
  }
}
