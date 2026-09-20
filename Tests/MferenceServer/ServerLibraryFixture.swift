import Darwin
import Foundation
import Mference

/// Builds installs on disk that `ServerLibraryProbe` accepts, so the probe runs
/// against real manifests and receipts. Nothing here is loadable — the weight
/// files are absent and no test ever asks a runtime to open one.
enum ServerLibraryFixture {
    /// A format-valid QAT installation fixture with real, tiny files and a
    /// complete receipt. It is never passed to a model runner.
    static func makeQATInstall(in root: URL, named name: String) throws -> URL {
        let directory = try makeCompleteInstall(in: root, named: name)
        let manifestURL = directory.appendingPathComponent("manifest.json")
        var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
        manifest["modelID"] = CheckpointIdentity.gemma4QAT
        manifest["sourceSnapshotHash"] = "sha256:7dbbeef0345505798abcf0ac54434116a48c2f1e7aad828071c17a7a871adfe7"
        var quant: [String: Any] = [:]
        for slot in ["embedding", "attention", "sharedExpert", "routedExpert"] {
            var value = quantSlot(4)
            value["groupSize"] = 32
            quant[slot] = value
        }
        quant["router"] = ["weightBits": 16, "scheme": "unquantized",
            "scaleType": "none", "biasType": "none", "groupSize": 0]
        manifest["quant"] = quant
        var files = manifest["files"] as! [String: Any]
        for name in Array(files.keys) {
            let data = name.hasSuffix("layout.json") ? Data("{}".utf8) : Data()
            try data.write(to: directory.appendingPathComponent(name))
            files[name] = ["size": data.count, "sha256": Sha256Verifier.hashData(data)]
        }
        let tokenizer = directory.appendingPathComponent("tokenizer")
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        for name in GemmaQATCheckpoint.requiredAssets {
            let data = Data("{}".utf8)
            try data.write(to: tokenizer.appendingPathComponent(name))
            files["tokenizer/" + name] = ["size": data.count, "sha256": Sha256Verifier.hashData(data)]
        }
        manifest["files"] = files
        try writeManifestAndReceipt(manifest, in: directory)
        let manifestData = try Data(contentsOf: manifestURL)
        files["manifest.json"] = ["size": manifestData.count, "sha256": Sha256Verifier.hashData(manifestData)]
        let receiptURL = directory.appendingPathComponent("verified-install.json")
        var receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as! [String: Any]
        receipt["files"] = files
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]).write(to: receiptURL)
        return directory
    }

    static func makeRoot(_ tag: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mference-library-\(tag)-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.standardizedFileURL
    }

    /// A complete Gemma 4 install named `<name>.gturbo` under `root`.
    @discardableResult
    static func makeCompleteInstall(in root: URL, named name: String) throws -> URL {
        let directory = root.appendingPathComponent("\(name).gturbo", isDirectory: true)
            .standardizedFileURL
        let experts = directory.appendingPathComponent("packed_experts", isDirectory: true)
        try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: experts.appendingPathComponent("layout.json"))

        let arch = ArchConfig.gemma4_26B_A4B
        var files: [String: Any] = [
            "model_weights.bin": ["size": 0, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 2,
                                           "sha256": String(repeating: "0", count: 64)],
        ]
        for layer in 0..<arch.numLayers {
            files[String(format: "packed_experts/layer_%02d.bin", layer)] = [
                "size": 0,
                "sha256": String(repeating: "0", count: 64),
            ]
        }
        let manifest: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true],
            "modelID": "test/\(name)",
            "sourceSnapshotHash": "sha256:" + String(repeating: "a", count: 64),
            "quant": [
                "embedding": quantSlot(4),
                "attention": quantSlot(4),
                "router": quantSlot(8),
                "sharedExpert": quantSlot(4),
                "routedExpert": quantSlot(4),
            ],
            "arch": [
                "family": ModelFamily.gemma4.rawValue,
                "hiddenSize": arch.hiddenSize,
                "ffnIntermediate": arch.intermediateSize,
                "moeIntermediateSize": arch.moeIntermediateSize,
                "numHeads": arch.numHeads,
                "numKVHeads": arch.numKVHeads,
                "numFullKVHeads": arch.numFullKVHeads,
                "headDim": arch.headDim,
                "fullHeadDim": arch.fullHeadDim,
                "vocabSize": arch.vocabSize,
                "slidingWindow": arch.slidingWindow,
                "finalLogitSoftcap": arch.finalLogitSoftcap,
                "ropeTheta": arch.ropeTheta,
                "fullRopeTheta": arch.fullRopeTheta,
                "partialRotaryFactor": arch.partialRotaryFactor,
                "numLayers": arch.numLayers,
                "numExperts": arch.numExperts,
                "topKExperts": arch.topKExperts,
                "tieWordEmbeddings": arch.tieWordEmbeddings,
                "attentionKEqV": arch.attentionKEqV,
                "hiddenActivation": arch.hiddenActivation,
                "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
            ],
            "files": files,
            "expertsPerLayer": arch.numExperts,
            "numLayers": arch.numLayers,
            "expertStride": UInt64(getpagesize()),
        ]
        try writeManifestAndReceipt(manifest, in: directory)
        return directory
    }

    /// A directory whose manifest names a family the runtime installs but has
    /// A stand-in capability gate table for tests: the runtime's own table is
    /// empty since the Flash-Next lift, so the not-runnable branch needs an
    /// injected entry to be exercised.
    static let gate: [String: [String]] = ["not-yet-family": ["someAxis"]]

    /// no runner for. Deliberately minimal: the probe classifies it from
    /// `arch.family` alone, before any strict decode.
    @discardableResult
    static func makeGatedInstall(in root: URL,
                                 named name: String,
                                 family: String = ModelFamily.qwen38flashnext.rawValue)
        throws -> URL {
        let directory = root.appendingPathComponent("\(name).gturbo", isDirectory: true)
            .standardizedFileURL
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        let manifest: [String: Any] = ["arch": ["family": family]]
        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try data.write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    /// A directory with a manifest that decodes to nothing usable.
    @discardableResult
    static func makeCorruptInstall(in root: URL, named name: String) throws -> URL {
        let directory = root.appendingPathComponent("\(name).gturbo", isDirectory: true)
            .standardizedFileURL
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        try Data("{\"arch\":{}}".utf8)
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    /// Marks `<name>.gturbo` as an install in progress, the way `InstallLock`
    /// does: an exclusive `flock` on a sibling lock file next to the directory
    /// being written. The returned descriptor holds the lock; close it to
    /// release.
    static func holdInstallLock(in root: URL, named name: String) throws -> Int32 {
        let lock = root.appendingPathComponent("\(name).gturbo.install.lock")
        try Data().write(to: lock)
        let descriptor = open(lock.path, O_RDWR)
        guard descriptor >= 0, flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            throw CocoaError(.fileWriteUnknown)
        }
        return descriptor
    }

    /// The file `InstallLock` leaves behind after a completed install: present,
    /// empty, and held by nobody.
    static func leaveStaleInstallLock(in root: URL, named name: String) throws {
        try Data().write(to: root.appendingPathComponent("\(name).gturbo.install.lock"))
    }

    private static func writeManifestAndReceipt(_ manifest: [String: Any],
                                                in directory: URL) throws {
        let manifestData = try JSONSerialization.data(withJSONObject: manifest,
                                                      options: [.sortedKeys])
        let manifestURL = directory.appendingPathComponent("manifest.json")
        try manifestData.write(to: manifestURL)
        let receipt: [String: Any] = [
            "schemaVersion": 1,
            "manifestSha256": Sha256Verifier.hashData(manifestData),
            "modelDirectoryPath": directory.standardizedFileURL.path,
            "verificationTimestamp": "2026-09-10T00:00:00Z",
            "toolVersion": "MferenceServerTests",
            "files": [:],
        ]
        let receiptData = try JSONSerialization.data(withJSONObject: receipt,
                                                     options: [.sortedKeys])
        try receiptData.write(to: directory.appendingPathComponent("verified-install.json"))
    }

    private static func quantSlot(_ weightBits: Int) -> [String: Any] {
        [
            "weightBits": weightBits,
            "scheme": "affine",
            "scaleType": "bf16",
            "biasType": "bf16",
            "groupSize": Quantization.groupSize,
        ]
    }
}
