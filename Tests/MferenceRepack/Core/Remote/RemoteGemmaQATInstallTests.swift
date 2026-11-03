import Foundation
import Synchronization
import Testing
@testable import MferenceRepackCore

extension RemotePayloadCopyTests {
    @Test func gemmaQATInstallPreservesNativeFormatAndRequiredAssets() async throws {
        let source = tmpDirForRemote("qat-source")
        let output = tmpPathForRemote("qat-output")
        defer { cleanUpRemote([source, output]) }
        let snapshot = try SyntheticSnapshot.build(at: source, gemmaQAT: true)
        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(snapshotDir: source, snap: snapshot,
            includeRequiredTokenizer: true, includeOptionalTokenizer: true)
        let generation = Data(#"{"do_sample":true,"temperature":1.0,"top_k":64,"top_p":0.95,"bos_token_id":2,"eos_token_id":[1,106,50],"pad_token_id":0}"#.utf8)
        FakeHFURLProtocol.files["generation_config.json"] = generation
        let result = try await RemoteStreamingRepacker(options: remoteOptions(
            outputDir: output, session: fakeHFSession())).run()

        let root = URL(fileURLWithPath: output)
        let manifest = try qatJSON(root.appendingPathComponent("manifest.json"))
        let quant = try #require(manifest["quant"] as? [String: [String: Any]])
        for slot in ["embedding", "attention", "sharedExpert", "routedExpert"] {
            #expect(quant[slot]?["weightBits"] as? Int == 4)
            #expect(quant[slot]?["groupSize"] as? Int == 32)
            #expect(quant[slot]?["scaleType"] as? String == "BF16")
            // Routed experts are stored without their -8 * scale biases.
            #expect(quant[slot]?["biasType"] as? String
                    == (slot == "routedExpert" ? QATImpliedBiasConverter.impliedBiasType : "BF16"))
        }
        #expect(quant["router"]?["weightBits"] as? Int == 16)
        #expect(quant["router"]?["scheme"] as? String == "unquantized")
        #expect(quant["router"]?["groupSize"] as? Int == 0)
        #expect(quant["router"]?["scaleType"] as? String == "none")
        #expect(quant["router"]?["biasType"] as? String == "none")
        let files = try #require(manifest["files"] as? [String: Any])
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja", "generation_config.json"] {
            #expect(files["tokenizer/" + name] != nil)
            let copied = try Data(contentsOf: root.appendingPathComponent("tokenizer/" + name))
            #expect(copied == FakeHFURLProtocol.files[name])
        }
        try qatExpectEveryCopiedTensor(source: URL(fileURLWithPath: snapshot.shardPath), output: root)
        let installedBytes = try FileManager.default.subpathsOfDirectory(atPath: output)
            .reduce(UInt64(0)) { sum, path in
                let attrs = try FileManager.default.attributesOfItem(atPath: output + "/" + path)
                return sum + (attrs[.type] as? FileAttributeType == .typeRegular
                    ? (attrs[.size] as? NSNumber)?.uint64Value ?? 0 : 0)
            }
        #expect(result.outputBytes == installedBytes)
        #expect(result.sourceMetadataBytes > UInt64(generation.count))
        let verified = try VerifiedInstallTool.run(options: .init(inputGTurbo: output))
        #expect(verified.unexpectedEntries.isEmpty)
        for relative in ["model_weights.bin", "tokenizer/generation_config.json"] {
            let url = root.appendingPathComponent(relative)
            let original = try Data(contentsOf: url)
            var damaged = original
            damaged[damaged.startIndex] ^= 1
            try damaged.write(to: url)
            #expect(throws: (any Error).self) {
                _ = try VerifiedInstallTool.run(options: .init(inputGTurbo: output))
            }
            try original.write(to: url)
        }
    }

    @Test func gemmaQATRejectsMissingOrMalformedRequiredAssetsBeforePayload() async throws {
        let source = tmpDirForRemote("qat-assets")
        let output = tmpPathForRemote("qat-assets-output")
        defer { cleanUpRemote([source, output]) }
        let snapshot = try SyntheticSnapshot.build(at: source, gemmaQAT: true)
        for name in GemmaQATSource.requiredAssets {
            for missing in [true, false] {
                try qatResetRemote(source: source, snapshot: snapshot)
                FakeHFURLProtocol.files[name] = missing ? nil : Data([0xff])
                let progress = InstallProgressRecorder()
                await #expect(throws: (any Error).self) {
                    _ = try await RemoteStreamingRepacker(options: remoteOptions(
                        outputDir: output, session: fakeHFSession(), rangeRetryAttempts: 1))
                        .run { progress.append($0) }
                }
                #expect(!FileManager.default.fileExists(atPath: output))
                #expect(!FileManager.default.fileExists(atPath: output + ".resume.json"))
                #expect(!progress.values.contains { if case .copyingPayload = $0 { true } else { false } })
            }
        }
    }

    @Test func gemmaQATRejectsWrongSourcePinAndTensorGeometry() async throws {
        let source = tmpDirForRemote("qat-invalid-source")
        let output = tmpPathForRemote("qat-invalid-output")
        defer { cleanUpRemote([source, output]) }
        let snapshot = try SyntheticSnapshot.build(at: source, gemmaQAT: true)
        let revision = try #require(SupportedModelSource.gemma4QAT.revision)
        for badCommit in [true, false] {
            try qatResetRemote(source: source, snapshot: snapshot)
            if !badCommit { FakeHFURLProtocol.commit = revision }
            await #expect(throws: (any Error).self) {
                _ = try await RemoteStreamingRepacker(options: remoteOptions(
                    outputDir: output, session: fakeHFSession(),
                    repoID: SupportedModelSource.gemma4QAT.repoID,
                    revision: revision)).run()
            }
            #expect(!FileManager.default.fileExists(atPath: output))
        }
        for mutation in ["router-dtype", "companion-shape", "missing-tensor"] {
            try qatResetRemote(source: source, snapshot: snapshot)
            let filename = URL(fileURLWithPath: snapshot.shardPath).lastPathComponent
            let data = try #require(FakeHFURLProtocol.files[filename])
            let size = (0..<8).reduce(0) { $0 | Int(data[$1]) << ($1 * 8) }
            var header = try #require(JSONSerialization.jsonObject(with: data.subdata(in: 8..<(8 + size))) as? [String: Any])
            let name = "language_model.model.layers.0." + (mutation == "router-dtype"
                ? "router.proj.weight" : "self_attn.q_proj.scales")
            var tensor = try #require(header[name] as? [String: Any])
            if mutation == "router-dtype" { tensor["dtype"] = "F16" }
            if mutation == "companion-shape" {
                tensor["shape"] = Array((try #require(tensor["shape"] as? [Int])).reversed())
            }
            header[name] = mutation == "missing-tensor" ? nil : tensor
            let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            var length = UInt64(json.count).littleEndian
            var changed = withUnsafeBytes(of: &length) { Data($0) }
            changed.append(json)
            changed.append(data.dropFirst(8 + size))
            FakeHFURLProtocol.files[filename] = changed
            await #expect(throws: (any Error).self) {
                _ = try await RemoteStreamingRepacker(options: remoteOptions(
                    outputDir: output, session: fakeHFSession())).run()
            }
            #expect(!FileManager.default.fileExists(atPath: output))
        }
    }

    @Test func gemmaQATSpaceFailureDoesNotPublishAnInstall() async throws {
        let source = tmpDirForRemote("qat-space")
        let output = tmpPathForRemote("qat-space-output")
        defer { cleanUpRemote([source, output]) }
        let snapshot = try SyntheticSnapshot.build(at: source, gemmaQAT: true)
        try qatResetRemote(source: source, snapshot: snapshot)
        await #expect(throws: (any Error).self) {
            _ = try await RemoteStreamingRepacker(options: .init(repoID: "owner/model",
                revision: "main", outputDir: output, minFreeReserveBytes: 1 << 60,
                downloadSession: fakeHFSession(), baseURL: URL(string: "https://hf.test")!)).run()
        }
        #expect(!FileManager.default.fileExists(atPath: output))
        #expect(!FileManager.default.fileExists(atPath: output + ".resume.json"))
    }

    @Test func gemmaQATCancellationResumeRevalidatesDamagedRangesAndRejectsChangedPlan() async throws {
        let source = tmpDirForRemote("qat-resume")
        let output = tmpPathForRemote("qat-resume-output")
        defer { cleanUpRemote([source, output]) }
        let snapshot = try SyntheticSnapshot.build(at: source, gemmaQAT: true)
        try qatResetRemote(source: source, snapshot: snapshot)
        let seen = Mutex<[UInt64: Int]>([:])
        let task = Task {
            try await RemoteStreamingRepacker(options: remoteOptions(
                outputDir: output, session: fakeHFSession())).run { progress in
                guard case .copyingPayload(_, let downloaded, _) = progress, downloaded > 0 else { return }
                let count = seen.withLock { $0[downloaded, default: 0] += 1; return $0[downloaded]! }
                if count == 3 { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        let checkpoint = try RemoteInstallCheckpoint.load(from: output + ".resume.json")
        #expect(!checkpoint.completedRanges.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: output))
        await #expect(throws: (any Error).self) {
            _ = try await RemoteStreamingRepacker(options: remoteOptions(outputDir: output,
                session: fakeHFSession(), resume: true, rangeChunkBytes: 8192)).run()
        }
        #expect(!FileManager.default.fileExists(atPath: output))
        // Corrupt every saved destination while retaining file lengths. Resume must
        // re-download damaged ranges instead of trusting the saved digest.
        let partial = output + ".partial"
        for name in try FileManager.default.subpathsOfDirectory(atPath: partial) where name.hasSuffix(".bin") {
            let url = URL(fileURLWithPath: partial + "/" + name)
            let data = try Data(contentsOf: url)
            try Data(repeating: 0xee, count: data.count).write(to: url)
        }
        let result = try await RemoteStreamingRepacker(options: remoteOptions(
            outputDir: output, session: fakeHFSession(), resume: true)).run()
        #expect(result.reusedBytes == 0)
        try qatExpectEveryCopiedTensor(source: URL(fileURLWithPath: snapshot.shardPath),
            output: URL(fileURLWithPath: output))
        _ = try VerifiedInstallTool.run(options: .init(inputGTurbo: output))
    }
}

private func qatResetRemote(source: String, snapshot: SyntheticSnapshot.Snapshot) throws {
    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(snapshotDir: source, snap: snapshot,
        includeRequiredTokenizer: true, includeOptionalTokenizer: true)
    FakeHFURLProtocol.files["generation_config.json"] = Data(#"{"do_sample":true,"temperature":1.0,"top_k":64,"top_p":0.95,"bos_token_id":2,"eos_token_id":[1,106,50],"pad_token_id":0}"#.utf8)
}

private func qatJSON(_ path: URL) throws -> [String: Any] {
    try #require(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
}

/// Parse the source header and destination index independently of RepackPlan.
private func qatExpectEveryCopiedTensor(source: URL, output: URL) throws {
    let input = try Data(contentsOf: source)
    func uint(_ data: Data, _ offset: Int, _ size: Int = 8) -> Int {
        (0..<size).reduce(0) { $0 | Int(data[offset + $1]) << (8 * $1) }
    }
    let headerSize = uint(input, 0)
    var tensors = try #require(JSONSerialization.jsonObject(
        with: input.subdata(in: 8..<(8 + headerSize))) as? [String: [String: Any]])
    // Safetensors reserves this entry for file metadata; it is not a tensor.
    tensors.removeValue(forKey: "__metadata__")
    func sourceBytes(_ name: String) throws -> Data {
        let bounds = try #require(tensors[name]?["data_offsets"] as? [Int])
        return input.subdata(in: (8 + headerSize + bounds[0])..<(8 + headerSize + bounds[1]))
    }
    var checked = Set<String>()
    let resident = try Data(contentsOf: output.appendingPathComponent("model_weights.bin"))
    for row in 0..<uint(resident, 16) {
        let base = 24 + row * 72
        let start = uint(resident, base, 4)
        let length = uint(resident, base + 4, 2)
        let name = try #require(String(data: resident.subdata(in: start..<(start + length)), encoding: .utf8))
        for (offsetField, sizeField, sourceName) in [
            (8, 16, name),
            (40, 48, String(name.dropLast(".weight".count)) + ".scales"),
            (56, 64, String(name.dropLast(".weight".count)) + ".biases"),
        ] {
            let size = uint(resident, base + sizeField)
            if size == 0 { continue }
            let offset = uint(resident, base + offsetField)
            #expect(resident.subdata(in: offset..<(offset + size)) == (try sourceBytes(sourceName)))
            checked.insert(sourceName)
        }
        if name.hasSuffix(".router.proj.weight") {
            #expect(resident[base + 6] == 1)
            #expect(uint(resident, base + 48) == 0)
            #expect(uint(resident, base + 64) == 0)
        }
    }
    let layout = try qatJSON(output.appendingPathComponent("packed_experts/layout.json"))
    // Experts are stored as segments of the expanded expert; biases are implied.
    let storage = try #require(layout["expertStorage"] as? [String: Any])
    #expect(storage["format"] as? String == QATImpliedBiasConverter.storageFormat)
    let segments = try #require(storage["segments"] as? [[String: Int]])
    let implied = Set(try #require(storage["impliedBiases"] as? [[String: String]]).compactMap { $0["biases"] })
    #expect(implied == ["gate_biases", "up_biases", "down_biases"])
    func storedOffset(_ memory: Int, _ size: Int) throws -> Int {
        let segment = try #require(segments.first {
            memory >= $0["memoryOffset"]! && memory + size <= $0["memoryOffset"]! + $0["size"]!
        })
        return segment["storedOffset"]! + memory - segment["memoryOffset"]!
    }
    for layer in try #require(layout["layers"] as? [[String: Any]]) {
        let index = try #require(layer["layer"] as? Int)
        let file = try #require(layer["file"] as? String)
        let data = try Data(contentsOf: output.appendingPathComponent("packed_experts/" + file))
        let experts = try #require(layer["experts"] as? [[String: Any]])
        for expert in experts {
            let id = try #require(expert["expert"] as? Int)
            let base = try #require(expert["offset"] as? Int)
            let regions = try #require(expert["tensors"] as? [String: [String: Any]])
            for role in ["gate", "up", "down"] {
                for (suffix, sourceSuffix) in [("", "weight"), ("_scales", "scales"), ("_biases", "biases")] {
                    let region = try #require(regions[role + suffix])
                    let offset = try #require(region["offset"] as? Int)
                    let size = try #require(region["size"] as? Int)
                    let name = "language_model.model.layers.\(index).experts.switch_glu.\(role)_proj.\(sourceSuffix)"
                    let original = try sourceBytes(name)
                    #expect(size * experts.count == original.count)
                    let expected = original.subdata(in: (id * size)..<((id + 1) * size))
                    if implied.contains(role + suffix) {
                        // Not stored: the runtime rebuilds -8 * scale.
                        let scales = try sourceBytes(String(name.dropLast("biases".count)) + "scales")
                            .subdata(in: (id * size)..<((id + 1) * size))
                        let rebuilt = Data(stride(from: 0, to: size, by: 2).flatMap { index -> [UInt8] in
                            let scale = UInt16(scales[scales.startIndex + index])
                                | UInt16(scales[scales.startIndex + index + 1]) << 8
                            let bias = QATImpliedBiasConverter.neg8ScaleBits(scale)
                            return [UInt8(bias & 0xFF), UInt8(bias >> 8)]
                        })
                        #expect(rebuilt == expected)
                    } else {
                        let at = base + (try storedOffset(offset, size))
                        #expect(data.subdata(in: at..<(at + size)) == expected)
                    }
                    checked.insert(name)
                }
            }
        }
    }
    #expect(checked == Set(tensors.keys))
}
