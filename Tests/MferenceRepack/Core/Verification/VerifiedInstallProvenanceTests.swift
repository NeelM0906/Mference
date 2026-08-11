import Foundation
import Testing

@testable import MferenceRepackCore

// Joins the serialized remote suite: every test that mutates the shared
// `FakeHFURLProtocol` statics must run in `RemotePayloadCopyTests`, whose
// `.serialized` trait is the only thing keeping those statics race-free.
// A separate `@Suite(.serialized)` type still runs in parallel WITH that
// suite and produced flaky Content-Range/416 failures under full-suite
// parallelism.
extension RemotePayloadCopyTests {
    @Test func verificationRetainsOnlyBoundInstallerProvenance() async throws {
        let snapshotDir = tmpDirForRemote("verified-provenance-snapshot")
        let output = tmpPathForRemote("verified-provenance-output")
        let movedOutput = tmpPathForRemote("verified-provenance-moved")
        defer { cleanUpRemote([snapshotDir, output, movedOutput]) }

        let snapshot = try SyntheticSnapshot.buildQwen(at: snapshotDir)
        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDir,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: false)
        _ = try await RemoteStreamingRepacker(
            options: remoteOptions(outputDir: output, session: fakeHFSession())
        ).run()

        let receiptPath = receiptPath(for: output)
        let originalReceipt = try Data(contentsOf: URL(fileURLWithPath: receiptPath))
        let manifestPath = (output as NSString).appendingPathComponent("manifest.json")
        let originalManifest = try Data(contentsOf: URL(fileURLWithPath: manifestPath))

        try FileManager.default.removeItem(atPath: receiptPath)
        var manifest = try jsonObject(at: manifestPath)
        manifest["sourceSnapshotHash"] = "manifest-is-not-provenance"
        try writeJSONObject(manifest, to: manifestPath)
        try assertStrippedProvenance(at: output)

        try originalManifest.write(to: URL(fileURLWithPath: manifestPath), options: .atomic)
        try originalReceipt.write(to: URL(fileURLWithPath: receiptPath), options: .atomic)
        try assertProvenance(at: output, repoID: "owner/model", revision: FakeHFURLProtocol.commit)
        try assertProvenance(at: output, repoID: "owner/model", revision: FakeHFURLProtocol.commit)

        var receipt = try jsonObject(at: receiptPath)
        receipt["manifestSha256"] = String(repeating: "0", count: 64)
        try writeJSONObject(receipt, to: receiptPath)
        try assertStrippedProvenance(at: output)

        receipt = try jsonObject(from: originalReceipt)
        receipt.removeValue(forKey: "sourceRevision")
        try writeJSONObject(receipt, to: receiptPath)
        try assertStrippedProvenance(at: output)

        receipt = try jsonObject(from: originalReceipt)
        var incompleteFiles = receipt["files"] as! [String: Any]
        incompleteFiles.removeValue(forKey: incompleteFiles.keys.sorted().first!)
        receipt["files"] = incompleteFiles
        try writeJSONObject(receipt, to: receiptPath)
        try assertStrippedProvenance(at: output)

        receipt = try jsonObject(from: originalReceipt)
        var extraFiles = receipt["files"] as! [String: Any]
        extraFiles["not-in-manifest.bin"] = ["size": 0, "sha256": "00"]
        receipt["files"] = extraFiles
        try writeJSONObject(receipt, to: receiptPath)
        try assertStrippedProvenance(at: output)

        receipt = try jsonObject(from: originalReceipt)
        var mismatchedFiles = receipt["files"] as! [String: Any]
        let payloadPath = mismatchedFiles.keys.sorted().first(where: { $0 != "manifest.json" })!
        var payload = mismatchedFiles[payloadPath] as! [String: Any]
        payload["size"] = 0
        mismatchedFiles[payloadPath] = payload
        receipt["files"] = mismatchedFiles
        try writeJSONObject(receipt, to: receiptPath)
        try assertStrippedProvenance(at: output)

        receipt = try jsonObject(from: originalReceipt)
        receipt["toolVersion"] = "other-tool"
        try writeJSONObject(receipt, to: receiptPath)
        try assertStrippedProvenance(at: output)

        try Data("not json".utf8).write(to: URL(fileURLWithPath: receiptPath), options: .atomic)
        try assertStrippedProvenance(at: output)

        FileManager.default.createFile(atPath: receiptPath, contents: Data())
        let oversizedReceipt = try FileHandle(forWritingTo: URL(fileURLWithPath: receiptPath))
        try oversizedReceipt.seek(toOffset: VerifiedInstallTool.metadataMaxBytes)
        try oversizedReceipt.write(contentsOf: Data([0]))
        try oversizedReceipt.close()
        try assertStrippedProvenance(at: output)

        try originalReceipt.write(to: URL(fileURLWithPath: receiptPath), options: .atomic)
        try FileManager.default.moveItem(atPath: output, toPath: movedOutput)
        try assertStrippedProvenance(at: movedOutput)
    }
}

private func assertProvenance(at output: String, repoID: String, revision: String) throws {
    _ = try VerifiedInstallTool.run(options: VerifyInstallOptions(inputGTurbo: output))
    let receipt = try jsonObject(at: receiptPath(for: output))
    #expect(receipt["sourceRepoID"] as? String == repoID)
    #expect(receipt["sourceRevision"] as? String == revision)
}

private func assertStrippedProvenance(at output: String) throws {
    _ = try VerifiedInstallTool.run(options: VerifyInstallOptions(inputGTurbo: output))
    let receipt = try jsonObject(at: receiptPath(for: output))
    #expect(receipt["sourceRepoID"] == nil)
    #expect(receipt["sourceRevision"] == nil)
}

private func receiptPath(for output: String) -> String {
    (output as NSString).appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
}

private func jsonObject(at path: String) throws -> [String: Any] {
    try jsonObject(from: Data(contentsOf: URL(fileURLWithPath: path)))
}

private func jsonObject(from data: Data) throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: data) as! [String: Any]
}

private func writeJSONObject(_ object: [String: Any], to path: String) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: path), options: .atomic)
}
