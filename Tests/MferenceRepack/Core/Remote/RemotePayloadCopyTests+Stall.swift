import Foundation
import Testing

@testable import MferenceRepackCore

/// A range request that is accepted and never answered (no response, no bytes,
/// no completion) is what a URLSession task waiting for connectivity looks like
/// from the installer: the hang seen on two real installs. The no-progress
/// watchdog must turn it into a retryable `remoteRangeStalled`, and the
/// bounded retries must end in that clear error instead of a silent wait.
extension RemotePayloadCopyTests {
  private static let shardKey = "GET:model-00001-of-00001.safetensors"

  private func stallingSession(seconds: TimeInterval = 0.3) -> RemoteDownloadSession {
    RemoteDownloadSession(
      policy: RemoteDownloadSessionPolicy(stallTimeoutSeconds: seconds),
      protocolClasses: [FakeHFURLProtocol.self])
  }

  @Test(.timeLimit(.minutes(1))) func remoteStalledRangeIsCancelledAndRetried() async throws {
    let snapshotDir = tmpDirForRemote("snap-stall-retry")
    let output = tmpPathForRemote("remote-stall-retry")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0x5741_4C4C_0001)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    // The first two shard GETs are the snapshot loader's header reads (the
    // 8-byte prefix, then the header); the third is the first payload range.
    FakeHFURLProtocol.failures[Self.shardKey] = [.pass, .pass, .stall]
    let audit = RepackAudit()

    let started = Date()
    let result = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: output, session: stallingSession()),
      audit: audit
    ).run()

    #expect(Date().timeIntervalSince(started) < 30)
    #expect(audit.remoteRangeRetries == 1)
    #expect(audit.remoteRetries.first.map { $0.detail.contains("made no progress") } == true)
    #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)
    #expect(FileManager.default.fileExists(
      atPath: (output as NSString).appendingPathComponent("manifest.json")))
    try assertNoInternalRemoteDirs(outputDir: output)
  }

  @Test(.timeLimit(.minutes(1))) func remoteRangeThatNeverProgressesFailsClearlyAndResumes() async throws {
    let snapshotDir = tmpDirForRemote("snap-stall-fail")
    let output = tmpPathForRemote("remote-stall-fail")
    defer { cleanUpRemote([snapshotDir, output, output + ".resume.json"]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0x5741_4C4C_0002)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    // The two header reads and two payload ranges land, then one payload
    // range never progresses on any attempt.
    FakeHFURLProtocol.failures[Self.shardKey] = [.pass, .pass, .pass, .pass, .stall, .stall, .stall]
    let audit = RepackAudit()

    let started = Date()
    do {
      _ = try await RemoteStreamingRepacker(
        options: remoteOptions(
          outputDir: output,
          session: stallingSession(),
          rangeRetryAttempts: 3,
          rangeChunkBytes: 1024),
        audit: audit
      ).run()
      Issue.record("a range that never progresses completed the install")
    } catch RepackError.remoteRangeStalled(let path, let seconds) {
      #expect(path == "model-00001-of-00001.safetensors")
      #expect(seconds == 0.3)
    }
    #expect(Date().timeIntervalSince(started) < 30)
    #expect(audit.remoteRangeRetries == 2)
    #expect(FakeHFURLProtocol.requestCounts[Self.shardKey] == 7)

    // The checkpoint keeps the two verified ranges, the stalled range is not
    // recorded, and its temp file is gone.
    let checkpoint = try RemoteInstallCheckpoint.load(from: output + ".resume.json")
    #expect(!checkpoint.completedRanges.isEmpty)
    #expect(!FileManager.default.fileExists(
      atPath: ((output + ".partial") as NSString).appendingPathComponent(".range.tmp")))

    // A resume with a healthy source reuses those ranges and finishes.
    let resumed = try await RemoteStreamingRepacker(
      options: remoteOptions(
        outputDir: output,
        session: stallingSession(),
        resume: true,
        rangeChunkBytes: 1024)
    ).run()
    #expect(resumed.reusedBytes > 0)
    #expect(resumed.downloadedThisRunBytes < resumed.remoteBytesToDownload)
    #expect(FileManager.default.fileExists(
      atPath: (output as NSString).appendingPathComponent("manifest.json")))
  }
}
