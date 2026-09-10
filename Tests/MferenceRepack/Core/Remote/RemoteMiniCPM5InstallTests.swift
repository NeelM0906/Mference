import Foundation
import Testing
@testable import MferenceRepackCore

/// End-to-end MiniCPM5 installs through the fake remote, both paths: the
/// vendor's BF16 upload quantized in flight (the shipped `minicpm5` entry) and
/// the vendor's MLX INT4 conversion through the pre-quantized path (the W2.1b
/// control, `minicpm5mlx`). Joins the serialized remote suite because it
/// mutates the shared `FakeHFURLProtocol` statics.
extension RemotePayloadCopyTests {

    @Test func miniCPM5InstallQuantizesInFlightFromTheVendorRepo() async throws {
        let snapshotDir = tmpDirForRemote("minicpm5-snap")
        let output = tmpPathForRemote("minicpm5-remote")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.buildMiniCPM5(at: snapshotDir)
        let source = try miniCPM5SourceTensors(in: snapshot.shardPath)

        resetFakeHF()
        FakeHFURLProtocol.files = try miniCPM5RemoteFiles(snapshotDir: snapshotDir,
                                                          snapshot: snapshot)
        let recorder = InstallProgressRecorder()
        // The pinned repo id is what vouches for the missing `quantization`
        // block; the fingerprint check is off because the snapshot is synthetic.
        let result = try await RemoteStreamingRepacker(
            options: remoteOptions(outputDir: output,
                                   session: fakeHFSession(),
                                   repoID: "openbmb/MiniCPM5-2B")
        ).run { recorder.append($0) }

        #expect(result.plan.arch.family == .minicpm5)
        #expect(result.plan.quantizedAtInstall)
        #expect(result.outputBytes < result.remoteBytesToDownload)
        #expect(result.expertLayerCount == 4)
        #expect(result.excludedMultimodalTensorCount == 0)
        #expect(recorder.values.contains(.finalizing))
        for layer in 0..<4 {
            let blob = (output as NSString).appendingPathComponent(
                String(format: "packed_experts/layer_%02d.bin", layer))
            #expect(!FileManager.default.fileExists(atPath: blob))
        }
        try assertRemoteTokenizerFilesRecorded(outputDir: output,
                                               expectsOptionalSpecialTokens: true)

        // --- Resident bytes: projections INT4 g64 byte-identical to the
        // reference encoder; norms verbatim (no fold), including the head.
        let resident = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("model_weights.bin")))
        let entries = try miniCPM5ResidentEntries(in: resident)
        #expect(entries.count == 4 * 9 + 3)
        for (name, rows, columns) in [
            ("model.embed_tokens.weight", 256, 128),
            ("lm_head.weight", 256, 128),
            ("model.layers.2.self_attn.q_proj.weight", 128, 128),
            ("model.layers.2.self_attn.k_proj.weight", 128, 128),
            ("model.layers.2.mlp.gate_proj.weight", 64, 128),
            ("model.layers.2.mlp.down_proj.weight", 128, 64),
        ] {
            let entry = try #require(entries[name], Comment(rawValue: name))
            let tensor = try #require(source[name], Comment(rawValue: name))
            #expect(entry.dtype == 0, Comment(rawValue: name))
            #expect(entry.shape == [UInt32(rows), UInt32(columns), 0, 0], Comment(rawValue: name))
            #expect(entry.size == UInt64(rows * columns / 2), Comment(rawValue: name))
            #expect(entry.scaleSize == UInt64(rows * (columns / 64) * 2), Comment(rawValue: name))
            #expect(entry.biasSize == entry.scaleSize, Comment(rawValue: name))
            let expected = miniCPM5Reference(
                bytes: try miniCPM5Slice(source.data, tensor.offset, tensor.size),
                rowLength: columns)
            #expect(try miniCPM5Slice(resident, entry.offset, entry.size) == expected.packed, Comment(rawValue: name))
            #expect(try miniCPM5Slice(resident, entry.scaleOffset, entry.scaleSize)
                == expected.scales, Comment(rawValue: name))
            #expect(try miniCPM5Slice(resident, entry.biasOffset, entry.biasSize)
                == expected.biases, Comment(rawValue: name))
        }
        for name in ["model.norm.weight",
                     "model.layers.0.input_layernorm.weight",
                     "model.layers.3.post_attention_layernorm.weight"] {
            let entry = try #require(entries[name], Comment(rawValue: name))
            let tensor = try #require(source[name], Comment(rawValue: name))
            #expect(entry.dtype == 1, Comment(rawValue: name))    // BF16
            #expect(entry.scaleSize == 0 && entry.biasSize == 0, Comment(rawValue: name))
            #expect(entry.size == tensor.size, Comment(rawValue: name))
            // Byte-equal to the source: the (1 + w) fold must NOT have run.
            #expect(try miniCPM5Slice(resident, entry.offset, entry.size)
                == miniCPM5Slice(source.data, tensor.offset, tensor.size), Comment(rawValue: name))
        }

        // --- Manifest: the family's axes, the dense quant contract, and the
        // quantize-in-flight provenance block.
        let manifest = try miniCPM5JSON(
            (output as NSString).appendingPathComponent("manifest.json"))
        #expect(manifest["expertsPerLayer"] as? Int == 0)
        #expect((manifest["expertStride"] as? NSNumber)?.uint64Value == 0)
        let arch = try #require(manifest["arch"] as? [String: Any])
        #expect(arch["family"] as? String == "minicpm5")
        #expect(arch["qkNorm"] as? Bool == false)
        #expect(arch["attnOutputGate"] as? Bool == false)
        #expect(arch["attentionScale"] as? Double == 0.125)
        #expect(arch["ropeNeoxSubdim"] as? Bool == true)
        #expect(arch["partialRotaryFactor"] as? Double == 1.0)
        #expect(arch["ropeTheta"] as? Double == 5_000_000.0)
        #expect(arch["numExperts"] as? Int == 0)
        #expect(arch["numSharedExperts"] as? Int == 0)
        #expect(arch["numDenseLayers"] as? Int == 4)
        #expect(arch["denseIntermediateSize"] as? Int == 64)
        #expect(arch["fullAttentionLayerMask"] as? [Int] == [1, 1, 1, 1])
        #expect(arch["tieWordEmbeddings"] as? Bool == false)
        #expect(arch["linearNumKHeads"] as? Int == 0)
        let quant = try #require(manifest["quant"] as? [String: [String: Any]])
        #expect(quant["embedding"]?["weightBits"] as? Int == 4)
        #expect(quant["attention"]?["weightBits"] as? Int == 4)
        for slot in ["router", "sharedExpert", "routedExpert"] {
            #expect(quant[slot]?["weightBits"] as? Int == 0, Comment(rawValue: slot))
            #expect(quant[slot]?["scheme"] as? String == "none", Comment(rawValue: slot))
        }
        let quantized = try #require(manifest["quantizedAtInstall"] as? [String: Any])
        #expect(quantized["weightBits"] as? Int == 4)
        #expect(quantized["groupSize"] as? Int == 64)
        #expect(quantized["sourceDtype"] as? String == "BF16")
        #expect(quantized["overriddenTensorCount"] == nil)
        #expect(manifest["bitWidthOverridesHonored"] as? Int == 0)

        let verify = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: output))
        #expect(verify.unexpectedEntries.isEmpty)
        #expect(verify.fileCount > 0)
    }

    /// The same BF16 files from a repo the installer does not know are refused
    /// for the missing `quantization` block: `model_type "llama"` alone never
    /// opens the quantize-in-flight path.
    @Test func miniCPM5BF16SnapshotFromAnUnknownRepoIsRefused() async throws {
        let snapshotDir = tmpDirForRemote("minicpm5-unknown-snap")
        let output = tmpPathForRemote("minicpm5-unknown-remote")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.buildMiniCPM5(at: snapshotDir)

        resetFakeHF()
        FakeHFURLProtocol.files = try miniCPM5RemoteFiles(snapshotDir: snapshotDir,
                                                          snapshot: snapshot)
        var thrown: Error?
        do {
            _ = try await RemoteStreamingRepacker(
                options: remoteOptions(outputDir: output,
                                       session: fakeHFSession(),
                                       repoID: "someone/llama-lookalike")
            ).run()
        } catch {
            thrown = error
        }
        guard case let RepackError.configJsonInvalid(_, detail)? = thrown else {
            Issue.record("expected configJsonInvalid, got \(String(describing: thrown))")
            return
        }
        #expect(detail.contains("no quantization slot"))
    }

    @Test func miniCPM5MLXControlInstallsThroughThePreQuantizedPath() async throws {
        let snapshotDir = tmpDirForRemote("minicpm5mlx-snap")
        let output = tmpPathForRemote("minicpm5mlx-remote")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.buildMiniCPM5MLX(at: snapshotDir)

        resetFakeHF()
        var files = try miniCPM5RemoteFiles(snapshotDir: snapshotDir, snapshot: snapshot)
        files["model.safetensors"] = files.removeValue(forKey: "model-00000-of-00001.safetensors")
        FakeHFURLProtocol.files = files
        let result = try await RemoteStreamingRepacker(
            options: remoteOptions(outputDir: output,
                                   session: fakeHFSession(),
                                   repoID: "openbmb/MiniCPM5-2B-MLX")
        ).run()

        #expect(result.plan.arch.family == .minicpm5)
        #expect(!result.plan.quantizedAtInstall)
        #expect(result.expertLayerCount == 4)

        let resident = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("model_weights.bin")))
        let entries = try miniCPM5ResidentEntries(in: resident)
        #expect(entries.count == 4 * 9 + 3)
        // Same names as the in-flight install, so the runner needs one spelling.
        let q = try #require(entries["model.layers.1.self_attn.q_proj.weight"])
        #expect(q.dtype == 0)
        #expect(q.shape == [128, 128, 0, 0])
        #expect(q.scaleSize == UInt64(128 * 2 * 2))
        #expect(entries["lm_head.weight"]?.dtype == 0)
        #expect(entries["model.norm.weight"]?.dtype == 1)

        let manifest = try miniCPM5JSON(
            (output as NSString).appendingPathComponent("manifest.json"))
        let arch = try #require(manifest["arch"] as? [String: Any])
        #expect(arch["family"] as? String == "minicpm5")
        #expect(arch["qkNorm"] as? Bool == false)
        #expect(manifest["quantizedAtInstall"] == nil)
        let quant = try #require(manifest["quant"] as? [String: [String: Any]])
        #expect(quant["embedding"]?["weightBits"] as? Int == 4)
        #expect(quant["attention"]?["weightBits"] as? Int == 4)
        #expect(quant["router"]?["weightBits"] as? Int == 0)

        let verify = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: output))
        #expect(verify.unexpectedEntries.isEmpty)
    }
}

// MARK: - Fixtures

private func miniCPM5RemoteFiles(snapshotDir: String,
                                 snapshot: SyntheticSnapshot.Snapshot) throws
    -> [String: Data] {
    var files: [String: Data] = [
        "config.json": try Data(contentsOf: URL(fileURLWithPath:
            (snapshotDir as NSString).appendingPathComponent("config.json"))),
        "model.safetensors.index.json": try Data(contentsOf: URL(fileURLWithPath:
            (snapshotDir as NSString).appendingPathComponent("model.safetensors.index.json"))),
        "tokenizer.json": remoteTokenizerJSON,
        "tokenizer_config.json": remoteTokenizerConfigJSON,
        "special_tokens_map.json": remoteSpecialTokensMapJSON,
        "chat_template.jinja": remoteChatTemplateJinja,
    ]
    files["model-00000-of-00001.safetensors"] =
        try Data(contentsOf: URL(fileURLWithPath: snapshot.shardPath))
    return files
}

private struct MiniCPM5SourceTensor {
    let offset: UInt64
    let size: UInt64
}

private struct MiniCPM5Source {
    let data: Data
    let tensors: [String: MiniCPM5SourceTensor]
    subscript(name: String) -> MiniCPM5SourceTensor? { tensors[name] }
}

private func miniCPM5SourceTensors(in shardPath: String) throws -> MiniCPM5Source {
    let data = try Data(contentsOf: URL(fileURLWithPath: shardPath))
    let headerLength = try miniCPM5UInt64(data, at: 0)
    let header = try JSONSerialization.jsonObject(
        with: try miniCPM5Slice(data, 8, headerLength)) as! [String: Any]
    var tensors: [String: MiniCPM5SourceTensor] = [:]
    for (name, value) in header where name != "__metadata__" {
        guard let entry = value as? [String: Any],
              let offsets = entry["data_offsets"] as? [Int], offsets.count == 2 else { continue }
        tensors[name] = MiniCPM5SourceTensor(
            offset: 8 + headerLength + UInt64(offsets[0]),
            size: UInt64(offsets[1] - offsets[0]))
    }
    return MiniCPM5Source(data: data, tensors: tensors)
}

private struct MiniCPM5ResidentEntry {
    let dtype: UInt8
    let shape: [UInt32]
    let offset: UInt64
    let size: UInt64
    let scaleOffset: UInt64
    let scaleSize: UInt64
    let biasOffset: UInt64
    let biasSize: UInt64
}

private func miniCPM5ResidentEntries(in data: Data) throws
    -> [String: MiniCPM5ResidentEntry] {
    let count = try miniCPM5UInt64(data, at: 16)
    var result: [String: MiniCPM5ResidentEntry] = [:]
    for index in 0..<Int(count) {
        let base = 24 + index * 72
        let nameOffset = try miniCPM5UInt32(data, at: base)
        let nameLength = try miniCPM5UInt16(data, at: base + 4)
        let name = String(decoding: try miniCPM5Slice(data, UInt64(nameOffset),
                                                      UInt64(nameLength)), as: UTF8.self)
        let shape = try (0..<4).map { try miniCPM5UInt32(data, at: base + 24 + $0 * 4) }
        result[name] = MiniCPM5ResidentEntry(
            dtype: try miniCPM5Slice(data, UInt64(base + 6), 1).first!,
            shape: shape,
            offset: try miniCPM5UInt64(data, at: base + 8),
            size: try miniCPM5UInt64(data, at: base + 16),
            scaleOffset: try miniCPM5UInt64(data, at: base + 40),
            scaleSize: try miniCPM5UInt64(data, at: base + 48),
            biasOffset: try miniCPM5UInt64(data, at: base + 56),
            biasSize: try miniCPM5UInt64(data, at: base + 64))
    }
    return result
}

/// The reference result for a `[rows, rowLength]` BF16 tensor, produced by
/// the whole-tensor encoder the streaming path is locked to (W2.1a).
private func miniCPM5Reference(bytes: Data, rowLength: Int)
    -> (packed: Data, scales: Data, biases: Data) {
    var values = [Float]()
    values.reserveCapacity(bytes.count / 2)
    for index in stride(from: 0, to: bytes.count, by: 2) {
        let lo = UInt32(bytes[bytes.startIndex + index])
        let hi = UInt32(bytes[bytes.startIndex + index + 1])
        values.append(Float(bitPattern: (lo | hi << 8) << 16))
    }
    let encoded = values.withUnsafeBufferPointer {
        Int4AffineEncoder.encodeTensor($0, rowLength: rowLength)
    }
    func widen(_ companions: [UInt16]) -> Data {
        var out = Data(capacity: companions.count * 2)
        for value in companions {
            out.append(UInt8(truncatingIfNeeded: value))
            out.append(UInt8(truncatingIfNeeded: value >> 8))
        }
        return out
    }
    return (Data(encoded.packed), widen(encoded.scales), widen(encoded.biases))
}

private func miniCPM5JSON(_ path: String) throws -> [String: Any] {
    try JSONSerialization.jsonObject(
        with: Data(contentsOf: URL(fileURLWithPath: path))) as! [String: Any]
}

private func miniCPM5Slice(_ data: Data, _ offset: UInt64, _ count: UInt64) throws -> Data {
    guard offset <= UInt64(data.count), count <= UInt64(data.count) - offset else {
        throw NSError(domain: "RemoteMiniCPM5InstallTests", code: 2)
    }
    return Data(data[(data.startIndex + Int(offset))..<(data.startIndex + Int(offset + count))])
}

private func miniCPM5UInt16(_ data: Data, at offset: Int) throws -> UInt16 {
    let bytes = try miniCPM5Slice(data, UInt64(offset), 2)
    return UInt16(bytes[bytes.startIndex]) | UInt16(bytes[bytes.startIndex + 1]) << 8
}

private func miniCPM5UInt32(_ data: Data, at offset: Int) throws -> UInt32 {
    let bytes = try miniCPM5Slice(data, UInt64(offset), 4)
    return (0..<4).reduce(UInt32(0)) {
        $0 | UInt32(bytes[bytes.startIndex + $1]) << UInt32($1 * 8)
    }
}

private func miniCPM5UInt64(_ data: Data, at offset: Int) throws -> UInt64 {
    let bytes = try miniCPM5Slice(data, UInt64(offset), 8)
    return (0..<8).reduce(UInt64(0)) {
        $0 | UInt64(bytes[bytes.startIndex + $1]) << UInt64($1 * 8)
    }
}
