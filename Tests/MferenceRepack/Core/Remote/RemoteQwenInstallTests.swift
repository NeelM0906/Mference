import Foundation
import Synchronization
import Testing

@testable import MferenceRepackCore

extension RemotePayloadCopyTests {
  @Test func qwenRemoteInstallCompletesWithFamilyManifest() async throws {
    let snapshotDir = tmpDirForRemote("qwen-snap")
    let remoteOutput = tmpPathForRemote("qwen-remote")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snapshot = try SyntheticSnapshot.buildQwen(
      at: snapshotDir,
      seed: 0x0102_0304_0506_0708)

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
    try assertRemoteTokenizerFilesRecorded(
      outputDir: remoteOutput,
      expectsOptionalSpecialTokens: true)

    // The manifest carries the qwen36 family extension fields.
    let manifestData = try Data(contentsOf: URL(fileURLWithPath:
      (remoteOutput as NSString).appendingPathComponent("manifest.json")))
    let manifest = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
    let arch = manifest["arch"] as! [String: Any]
    #expect(arch["family"] as? String == "qwen36")
    #expect(arch["attnOutputGate"] as? Bool == true)
    #expect(arch["attentionScale"] as? Double == 0.125)
    #expect(arch["embeddingScaledBySqrtHidden"] as? Bool == false)
    #expect(arch["routerScaled"] as? Bool == false)
    #expect(arch["ffnSandwichNorms"] as? Bool == false)
    #expect(arch["sharedExpertGated"] as? Bool == true)
    #expect(arch["ropeNeoxSubdim"] as? Bool == true)
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
    #expect(quant["router"]?["weightBits"] as? Int == 8)
    #expect(quant["sharedExpert"]?["weightBits"] as? Int == 4)
    #expect(quant["routedExpert"]?["weightBits"] as? Int == 4)

    // The finished install passes post-hoc verification.
    let verify = try VerifiedInstallTool.run(
      options: VerifyInstallOptions(inputGTurbo: remoteOutput))
    #expect(verify.unexpectedEntries.isEmpty)
    #expect(verify.fileCount > 0)
  }

  @Test func qwenCancellationPreservesCommittedRangesForResume() async throws {
    let snapshotDir = tmpDirForRemote("qwen-snap-resume")
    let output = tmpPathForRemote("qwen-remote-resume")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snapshot = try SyntheticSnapshot.buildQwen(
      at: snapshotDir,
      seed: 0x99_1122_3344)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snapshot,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    let seen = Mutex<[UInt64: Int]>([:])
    let task = Task {
      try await RemoteStreamingRepacker(
        options: remoteOptions(outputDir: output, session: fakeHFSession())
      ).run { progress in
        guard case .copyingPayload(_, let downloaded, _) = progress,
              downloaded > 0 else { return }
        let count = seen.withLock {
          $0[downloaded, default: 0] += 1
          return $0[downloaded] ?? 0
        }
        if count == 3 {
          withUnsafeCurrentTask { $0?.cancel() }
        }
      }
    }
    await #expect(throws: CancellationError.self) {
      _ = try await task.value
    }

    let checkpoint = try RemoteInstallCheckpoint.load(from: output + ".resume.json")
    #expect(!checkpoint.completedRanges.isEmpty)

    let result = try await RemoteStreamingRepacker(
      options: remoteOptions(
        outputDir: output,
        session: fakeHFSession(),
        resume: true)
    ).run()
    #expect(result.reusedBytes > 0)
    #expect(result.downloadedThisRunBytes < result.remoteBytesToDownload)
    let manifestPath = (output as NSString).appendingPathComponent("manifest.json")
    #expect(FileManager.default.fileExists(atPath: manifestPath))
    let manifest = try JSONSerialization.jsonObject(
      with: Data(contentsOf: URL(fileURLWithPath: manifestPath))) as! [String: Any]
    let arch = manifest["arch"] as! [String: Any]
    #expect(arch["family"] as? String == "qwen36")
  }

  /// The free-space reserve refuses the install before anything is published:
  /// no output directory and no resume checkpoint. Moved from the removed
  /// Gemma 4 QAT install suite onto the Qwen synthetic snapshot.
  @Test func qwenSpaceFailureDoesNotPublishAnInstall() async throws {
    let snapshotDir = tmpDirForRemote("qwen-snap-space")
    let output = tmpPathForRemote("qwen-remote-space")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snapshot = try SyntheticSnapshot.buildQwen(
      at: snapshotDir,
      seed: 0x5ACE_0001)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snapshot,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: true)
    await #expect(throws: (any Error).self) {
      _ = try await RemoteStreamingRepacker(
        options: .init(
          repoID: "owner/model",
          revision: "main",
          outputDir: output,
          minFreeReserveBytes: 1 << 60,
          downloadSession: fakeHFSession(),
          baseURL: URL(string: "https://hf.test")!)
      ).run()
    }
    #expect(!FileManager.default.fileExists(atPath: output))
    #expect(!FileManager.default.fileExists(atPath: output + ".resume.json"))
  }

  /// End-to-end resume after cancellation: a resume whose plan changed (a
  /// different range chunk size) is refused without publishing, and a resume
  /// over saved destinations whose bytes were damaged re-downloads them rather
  /// than trusting the checkpoint's digests. Moved from the removed Gemma 4 QAT
  /// install suite onto the Qwen synthetic snapshot.
  @Test func qwenResumeRejectsAChangedPlanAndRedownloadsDamagedRanges() async throws {
    let snapshotDir = tmpDirForRemote("qwen-snap-damaged")
    let output = tmpPathForRemote("qwen-remote-damaged")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snapshot = try SyntheticSnapshot.buildQwen(
      at: snapshotDir,
      seed: 0x5ACE_0002)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snapshot,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: true)
    let seen = Mutex<[UInt64: Int]>([:])
    let task = Task {
      try await RemoteStreamingRepacker(
        options: remoteOptions(outputDir: output, session: fakeHFSession())
      ).run { progress in
        guard case .copyingPayload(_, let downloaded, _) = progress,
              downloaded > 0 else { return }
        let count = seen.withLock {
          $0[downloaded, default: 0] += 1
          return $0[downloaded] ?? 0
        }
        if count == 3 {
          withUnsafeCurrentTask { $0?.cancel() }
        }
      }
    }
    await #expect(throws: CancellationError.self) {
      _ = try await task.value
    }
    let checkpoint = try RemoteInstallCheckpoint.load(from: output + ".resume.json")
    #expect(!checkpoint.completedRanges.isEmpty)
    #expect(!FileManager.default.fileExists(atPath: output))

    // A changed plan must not resume onto the saved ranges.
    await #expect(throws: (any Error).self) {
      _ = try await RemoteStreamingRepacker(
        options: remoteOptions(
          outputDir: output,
          session: fakeHFSession(),
          resume: true,
          rangeChunkBytes: 8192)
      ).run()
    }
    #expect(!FileManager.default.fileExists(atPath: output))

    // Corrupt every saved destination while retaining file lengths. Resume
    // must re-download the damaged ranges instead of trusting the saved digest.
    let partial = output + ".partial"
    for name in try FileManager.default.subpathsOfDirectory(atPath: partial)
      where name.hasSuffix(".bin") {
      let url = URL(fileURLWithPath: partial + "/" + name)
      let data = try Data(contentsOf: url)
      try Data(repeating: 0xee, count: data.count).write(to: url)
    }
    let result = try await RemoteStreamingRepacker(
      options: remoteOptions(
        outputDir: output,
        session: fakeHFSession(),
        resume: true)
    ).run()
    #expect(result.reusedBytes == 0)
    let verify = try VerifiedInstallTool.run(
      options: VerifyInstallOptions(inputGTurbo: output))
    #expect(verify.unexpectedEntries.isEmpty)

    // Byte for byte, the resumed payload equals an uninterrupted install of
    // the same snapshot: no damaged byte survived the resume. (The original
    // QAT suite compared against the source tensors with a QAT-layout parser;
    // the uninterrupted install is the Qwen equivalent.)
    let reference = tmpPathForRemote("qwen-remote-damaged-reference")
    defer { cleanUpRemote([reference]) }
    _ = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: reference, session: fakeHFSession())
    ).run()
    func payloadFiles(_ root: String) throws -> [String] {
      try FileManager.default.subpathsOfDirectory(atPath: root)
        .filter { $0.hasSuffix(".bin") || $0 == "packed_experts/layout.json" }
        .sorted()
    }
    let files = try payloadFiles(output)
    #expect(files.contains("model_weights.bin"))
    #expect(files == (try payloadFiles(reference)))
    for name in files {
      let resumed = try Data(contentsOf: URL(fileURLWithPath: output + "/" + name))
      let uninterrupted = try Data(contentsOf: URL(fileURLWithPath: reference + "/" + name))
      #expect(resumed == uninterrupted, "\(name)")
    }
  }
}
