import Foundation
import Testing
@testable import Mference

@Suite struct GemmaQATManifestTests {
    @Test func nativeQATManifestValidatesItsSeparateIdentity() throws {
        let (directory, arch) = try Self.makeInstall()
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = try ManifestReader.load(directoryURL: directory, expecting: arch)
        #expect(manifest.quant?.embedding.groupSize == 32)
        #expect(manifest.quant?.router.weightBits == 16)
        #expect(manifest.modelID == CheckpointIdentity.gemma4QAT)
    }

    @Test func missingOrAlteredQATAssetIsInvalidBeforeCapabilityRefusal() throws {
        let (directory, arch) = try Self.makeInstall()
        defer { try? FileManager.default.removeItem(at: directory) }
        let asset = directory.appendingPathComponent("tokenizer/generation_config.json")
        try Data("[]".utf8).write(to: asset)
        #expect(throws: ModelError.checksumMismatch(file: "tokenizer/generation_config.json")) {
            _ = try ManifestReader.load(directoryURL: directory, expecting: arch)
        }
        try FileManager.default.removeItem(at: asset)
        #expect(throws: (any Error).self) {
            _ = try ManifestReader.load(directoryURL: directory, expecting: arch)
        }
    }

    @Test func incompleteQATReceiptFailsBeforeCapabilityRefusal() throws {
        let (directory, arch) = try Self.makeInstall()
        defer { try? FileManager.default.removeItem(at: directory) }
        _ = try ManifestReader.load(directoryURL: directory, expecting: arch)
        let receipt = directory.appendingPathComponent("verified-install.json")
        let data = try Data(contentsOf: receipt)
        try FileManager.default.removeItem(at: receipt)
        #expect(throws: ModelError.trustedReceiptInvalid(detail: "verified-install.json is missing")) {
            _ = try ManifestReader.load(directoryURL: directory, expecting: arch)
        }
        var object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["manifestSha256"] = String(repeating: "0", count: 64)
        try JSONSerialization.data(withJSONObject: object).write(to: receipt)
        #expect(throws: ModelError.trustedReceiptInvalid(detail: "manifest SHA mismatch")) {
            _ = try ManifestReader.load(directoryURL: directory, expecting: arch)
        }
    }

    static func makeInstall() throws -> (URL, ArchConfig) {
        let affine: [String: Any] = ["weightBits": 4, "scheme": "affine",
            "scaleType": "BF16", "biasType": "BF16", "groupSize": 32]
        let router: [String: Any] = ["weightBits": 16, "scheme": "unquantized",
            "scaleType": "none", "biasType": "none", "groupSize": 0]
        let (directory, arch) = try ManifestReaderTests.writeToyManifest([
            "modelID": "gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned",
            "sourceSnapshotHash": "sha256:7dbbeef0345505798abcf0ac54434116a48c2f1e7aad828071c17a7a871adfe7",
            "quant": ["embedding": affine, "attention": affine, "router": router,
                      "sharedExpert": affine, "routedExpert": affine],
        ])
        let manifestURL = directory.appendingPathComponent("manifest.json")
        var json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
        var files = try #require(json["files"] as? [String: [String: Any]])
        for (name, entry) in files {
            let size = try #require(entry["size"] as? Int)
            let data = Data(repeating: 0, count: size)
            try data.write(to: directory.appendingPathComponent(name))
            if name == "packed_experts/layout.json" {
                files[name]?["sha256"] = Sha256Verifier.hashData(data)
            }
        }
        let tokenizer = directory.appendingPathComponent("tokenizer")
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"] {
            let data = Data((name == "chat_template.jinja" ? "{{ messages }}" : "{}").utf8)
            try data.write(to: tokenizer.appendingPathComponent(name))
            files["tokenizer/" + name] = ["size": data.count, "sha256": Sha256Verifier.hashData(data)]
        }
        json["files"] = files
        let manifestData = try JSONSerialization.data(withJSONObject: json)
        try manifestData.write(to: manifestURL)
        let manifestHash = Sha256Verifier.hashData(manifestData)
        files["manifest.json"] = ["size": manifestData.count, "sha256": manifestHash]
        let receipt: [String: Any] = ["schemaVersion": 1,
            "manifestSha256": manifestHash, "modelDirectoryPath": directory.standardizedFileURL.path,
            "verificationTimestamp": "2026-09-20T00:00:00Z", "toolVersion": "GemmaQATManifestTests",
            "files": files]
        try JSONSerialization.data(withJSONObject: receipt)
            .write(to: directory.appendingPathComponent("verified-install.json"))
        return (directory, arch)
    }
}
