import Foundation
import Synchronization
import Testing

@testable import MferenceRepackCore

extension RemotePayloadCopyTests {
  /// End-to-end synthetic install for the DeepSeek-V4-Flash family: exercises
  /// the real-checkpoint quirks the planner must survive — I64 `tid2eid`,
  /// an unquantized BF16 router gate, BF16 `ape` position biases, and mixed
  /// 32/64 gate_proj quant groups — through plan, byte copy, manifest
  /// encode, and post-hoc verification.
  @Test func deepseekRemoteInstallCompletesWithFamilyManifest() async throws {
    let snapshotDir = tmpDirForRemote("dsv4-snap")
    let remoteOutput = tmpPathForRemote("dsv4-remote")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snapshot = try SyntheticSnapshot.buildDeepseekV4(at: snapshotDir)

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
    for relativePath in [
      "model_weights.bin",
      "packed_experts/layout.json",
      "packed_experts/layer_00.bin",
      "packed_experts/layer_01.bin",
      "packed_experts/layer_02.bin",
      "packed_experts/layer_03.bin",
      "manifest.json",
    ] {
      let remote = (remoteOutput as NSString).appendingPathComponent(relativePath)
      #expect(FileManager.default.fileExists(atPath: remote))
    }
    #expect(recorder.values.contains(.finalizing))

    let manifestData = try Data(contentsOf: URL(fileURLWithPath:
      (remoteOutput as NSString).appendingPathComponent("manifest.json")))
    let manifest = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
    let arch = manifest["arch"] as! [String: Any]
    #expect(arch["family"] as? String == "deepseekV4Flash")
    #expect(arch["numHashRoutedLayers"] as? Int == 1)
    #expect(arch["routerScoringFunc"] as? String == "sqrtsoftplus")

    let quant = manifest["quant"] as! [String: [String: Any]]
    #expect(quant["embedding"]?["weightBits"] as? Int == 4)
    #expect(quant["attention"]?["weightBits"] as? Int == 4)
    #expect(quant["routedExpert"]?["weightBits"] as? Int == 2)

    // The layout carries mixed per-layer gate scale sizes (group 32 on
    // layers 0-2, group 64 on layer 3) — the runtime derives the gate
    // group size from these byte sizes.
    let layoutData = try Data(contentsOf: URL(fileURLWithPath:
      (remoteOutput as NSString).appendingPathComponent("packed_experts/layout.json")))
    let layout = try JSONSerialization.jsonObject(with: layoutData) as! [String: Any]
    let layers = layout["layers"] as! [[String: Any]]
    #expect(layers.count == 4)
    func gateScalesSize(_ layer: [String: Any]) -> Int? {
      let experts = layer["experts"] as! [[String: Any]]
      let tensors = experts[0]["tensors"] as! [String: [String: Any]]
      return tensors["gate_scales"]?["size"] as? Int
    }
    #expect(gateScalesSize(layers[0]) == 64 * 4 * 2)
    #expect(gateScalesSize(layers[3]) == 64 * 2 * 2)

    // The finished install passes post-hoc verification.
    let verify = try VerifiedInstallTool.run(
      options: VerifyInstallOptions(inputGTurbo: remoteOutput))
    #expect(verify.unexpectedEntries.isEmpty)
    #expect(verify.fileCount > 0)
  }
}
