import Foundation
import Testing

@testable import MferenceRepackCore

/// Workstream 2 end-to-end: plan and run a streamed install of a synthetic
/// Qwen3.8-Flash-Next checkpoint — original-repo BF16, three shards, no
/// `config.json -> quantization` — through the quantize-in-flight path into a
/// real temporary `.gturbo`, then check every claim the layout makes.
///
/// Everything is byte-checked against `Int4AffineEncoder` applied to the
/// synthetic BF16 source, which `Int4AffineEncoderParityTests` and
/// `Int4AffineStreamingParityTests` lock to the runtime's reference quantizer.
extension RemotePayloadCopyTests {

    @Test func flashNextInstallQuantizesInFlightAndBuildsRowPool() async throws {
        let snapshotDir = tmpDirForRemote("flashnext-snap")
        let output = tmpPathForRemote("flashnext-remote")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.buildQwen38FlashNext(at: snapshotDir)
        let source = try flashNextSourceTensors(in: snapshot.shardPaths)

        resetFakeHF()
        FakeHFURLProtocol.files = try flashNextRemoteFiles(snapshotDir: snapshotDir,
                                                           snapshot: snapshot)
        let result = try await RemoteStreamingRepacker(
            options: flashNextOptions(outputDir: output)
        ).run()

        #expect(result.plan.arch.family == .qwen38flashnext)
        #expect(result.plan.quantizedAtInstall)
        // BF16 in, INT4 + BF16 companions out: the install must be far smaller
        // than the bytes it read.
        #expect(result.outputBytes < result.remoteBytesToDownload)

        // --- The vision tower is gone; the MTP group came along.
        #expect(result.plan.excludedMultimodalTensorNames
            == ["model.visual.blocks.0.norm1.weight",
                "model.visual.patch_embed.proj.weight"])

        let residentPath = (output as NSString).appendingPathComponent("model_weights.bin")
        let resident = try Data(contentsOf: URL(fileURLWithPath: residentPath))
        let entries = try flashNextResidentEntries(in: resident)
        #expect(entries.keys.contains { $0.hasPrefix("model.visual.") } == false)
        #expect(entries["mtp.fc_embedding.weight"] != nil)
        #expect(entries["mtp.layers.0.self_attn.q_proj.weight"] != nil)

        // --- A quantized resident projection: INT4 payload plus BF16
        // companions, byte-identical to the reference quantizer.
        let qName = "model.language_model.layers.1.self_attn.q_proj.weight"
        let q = try #require(entries[qName])
        let qSource = try #require(source[qName])
        #expect(q.dtype == 0)
        #expect(q.shape == [256, 128, 0, 0])
        #expect(q.size == UInt64(256 * 128 / 2))
        #expect(q.scaleSize == UInt64(256 * (128 / 64) * 2))
        #expect(q.biasSize == q.scaleSize)
        let qExpected = flashNextReference(bytes: try flashNextBytes(at: qSource.path,
                                                                     offset: qSource.offset,
                                                                     count: qSource.size),
                                           rowLength: 128)
        #expect(try flashNextSlice(resident, q.offset, q.size) == qExpected.packed)
        #expect(try flashNextSlice(resident, q.scaleOffset, q.scaleSize) == qExpected.scales)
        #expect(try flashNextSlice(resident, q.biasOffset, q.biasSize) == qExpected.biases)

        // --- Norms, 1-D vectors, conv kernels and the I64 n-gram head tables
        // ride through untouched at their source dtype. dtype codes: 1 = BF16,
        // 4 = I64. Getting I64 wrong would silently corrupt the tables the PLE
        // hash lookup indexes with, so each is byte-compared to its source.
        for (name, dtype) in [
            ("model.language_model.layers.1.input_layernorm.weight", UInt8(1)),
            ("model.language_model.layers.0.linear_attn.conv1d.weight", UInt8(1)),
            ("model.language_model.layers.0.linear_attn.A_log", UInt8(1)),
            ("model.language_model.layers.1.ple.ple_embedding.layer_multipliers", UInt8(4)),
            ("model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets", UInt8(4)),
            ("model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
             UInt8(4)),
        ] {
            let entry = try #require(entries[name], "missing resident entry \(name)")
            let tensor = try #require(source[name])
            #expect(entry.dtype == dtype, "\(name) dtype")
            #expect(entry.scaleSize == 0 && entry.biasSize == 0, "\(name) companions")
            #expect(entry.size == tensor.size, "\(name) size")
            #expect(try flashNextSlice(resident, entry.offset, entry.size)
                == flashNextBytes(at: tensor.path, offset: tensor.offset, count: tensor.size),
                "\(name) bytes")
        }
        // I64 is 8 bytes per element: 3 layer multipliers, 2 heads.
        #expect(try #require(entries[
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"]).size == 24)
        #expect(try #require(entries[
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"]).size == 16)

        // --- Fused expert tensors split into per-expert page-aligned blobs.
        let layout = try flashNextJSON(
            (output as NSString).appendingPathComponent("packed_experts/layout.json"))
        #expect(layout["numLayers"] as? Int == 2)
        #expect(layout["expertsPerLayer"] as? Int == 4)
        let expertStride = try #require((layout["expertStride"] as? NSNumber)?.uint64Value)
        #expect(expertStride % 16_384 == 0)

        let layer1 = try #require((layout["layers"] as? [[String: Any]])?
            .first { $0["layer"] as? Int == 1 })
        let layerFile = ((output as NSString).appendingPathComponent("packed_experts")
            as NSString).appendingPathComponent(try #require(layer1["file"] as? String))
        let layerBytes = try Data(contentsOf: URL(fileURLWithPath: layerFile))
        let gateUp = try #require(source["model.language_model.layers.1.mlp.experts.gate_up_proj"])
        let down = try #require(source["model.language_model.layers.1.mlp.experts.down_proj"])
        let gateUpSlab = UInt64(2 * 64 * 128 * 2)
        let downSlab = UInt64(128 * 64 * 2)

        for expert in [0, 3] {
            let entry = try #require((layer1["experts"] as? [[String: Any]])?
                .first { $0["expert"] as? Int == expert })
            let base = try #require((entry["offset"] as? NSNumber)?.uint64Value)
            #expect(base == UInt64(expert) * expertStride)
            #expect(base % 16_384 == 0, "expert \(expert) blob is not page-aligned")
            let tensors = try #require(entry["tensors"] as? [String: [String: Any]])

            // gate = the first half of the fused slab, up = the second half.
            for (role, sliceOffset) in [("gate", UInt64(0)), ("up", gateUpSlab / 2)] {
                let expected = flashNextReference(
                    bytes: try flashNextBytes(
                        at: gateUp.path,
                        offset: gateUp.offset + UInt64(expert) * gateUpSlab + sliceOffset,
                        count: gateUpSlab / 2),
                    rowLength: 128)
                try flashNextExpectSlice(layerBytes, base: base, tensors: tensors,
                                         role: role, expected: expected)
            }
            let downExpected = flashNextReference(
                bytes: try flashNextBytes(at: down.path,
                                          offset: down.offset + UInt64(expert) * downSlab,
                                          count: downSlab),
                rowLength: 64)
            try flashNextExpectSlice(layerBytes, base: base, tensors: tensors,
                                     role: "down", expected: downExpected)
        }

        // --- The PLE n-gram table became a page-aligned row-lookup pool.
        let manifest = try flashNextJSON(
            (output as NSString).appendingPathComponent("manifest.json"))
        let pool = try #require(manifest["plePool"] as? [String: Any])
        #expect(pool["kind"] as? String == "rowLookupPoolV1")
        let poolLayer = try #require((pool["layers"] as? [[String: Any]])?.first)
        #expect(poolLayer["layer"] as? Int == 1)
        #expect(poolLayer["rows"] as? Int == 260)
        // 160 is not a multiple of 64, so group-64 cannot quantize these rows
        // and the pool must fall back to BF16 — the production case.
        #expect(poolLayer["rowDim"] as? Int == 160)
        #expect(poolLayer["storage"] as? String == "bf16")
        #expect(poolLayer["weightBits"] as? Int == 16)
        #expect(poolLayer["groupSize"] as? Int == 0)
        #expect(poolLayer["scheme"] as? String == "none")
        #expect(poolLayer["rowScaleBytes"] as? Int == 0)
        #expect(poolLayer["rowBiasBytes"] as? Int == 0)
        // 160 BF16 values = 320 bytes, so a 16 KB page holds 51 rows with
        // 64 bytes of slack and no row straddling it.
        let rowStride = try #require((poolLayer["rowStride"] as? NSNumber)?.uint64Value)
        let rowsPerBlock = try #require(poolLayer["rowsPerBlock"] as? Int)
        let blockStride = try #require((poolLayer["blockStride"] as? NSNumber)?.uint64Value)
        #expect(rowStride == 320)
        #expect(rowsPerBlock == 51)
        #expect(blockStride == 16_384)
        #expect(UInt64(rowsPerBlock) * rowStride <= blockStride)

        let poolPath = (output as NSString)
            .appendingPathComponent(try #require(poolLayer["file"] as? String))
        let poolBytes = try Data(contentsOf: URL(fileURLWithPath: poolPath))
        #expect(UInt64(poolBytes.count)
            == (try #require((poolLayer["fileSize"] as? NSNumber)?.uint64Value)))

        let poolShards = try #require(poolLayer["shards"] as? [[String: Any]])
        #expect(poolShards.count == 2)
        for shard in poolShards {
            let index = try #require(shard["shard"] as? Int)
            let regionOffset = try #require((shard["offset"] as? NSNumber)?.uint64Value)
            #expect(shard["rows"] as? Int == 130)
            #expect(regionOffset % 16_384 == 0, "shard \(index) region is not page-aligned")
            let tensor = try #require(source[
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
                    + ".shard_\(index).weight"])
            // Rows in the first block, at the start of the second block, at the
            // start of the trailing partial block, and the very last row.
            for row in [0, 50, rowsPerBlock, 2 * rowsPerBlock, 129] {
                let offset = regionOffset
                    + UInt64(row / rowsPerBlock) * blockStride
                    + UInt64(row % rowsPerBlock) * rowStride
                // A BF16 pool row is its source row, byte for byte.
                #expect(try flashNextSlice(poolBytes, offset, rowStride)
                    == flashNextBytes(at: tensor.path,
                                      offset: tensor.offset + UInt64(row) * rowStride,
                                      count: rowStride),
                    "shard \(index) row \(row)")
                // No row may cross a page boundary.
                #expect(offset / 16_384 == (offset + rowStride - 1) / 16_384,
                        "shard \(index) row \(row) straddles a page")
            }
            // The block slack is zero-filled, not left as stale bytes.
            let slackOffset = regionOffset + UInt64(rowsPerBlock) * rowStride
            #expect(try flashNextSlice(poolBytes, slackOffset,
                                       blockStride - UInt64(rowsPerBlock) * rowStride)
                .allSatisfy { $0 == 0 }, "shard \(index) block slack is not zeroed")
        }

        // --- Sidecar decisions are recorded, not implied.
        let sidecars = try #require(manifest["sidecars"] as? [String: Any])
        let mtp = try #require(sidecars["mtp"] as? [String: Any])
        #expect(mtp["carried"] as? Bool == true)
        #expect((mtp["tensorCount"] as? Int ?? 0) > 0)
        let vision = try #require(sidecars["vision"] as? [String: Any])
        #expect(vision["carried"] as? Bool == false)
        #expect(vision["tensorCount"] as? Int == 2)
        let quantized = try #require(manifest["quantizedAtInstall"] as? [String: Any])
        #expect(quantized["weightBits"] as? Int == 4)
        #expect(quantized["groupSize"] as? Int == 64)
        #expect(quantized["sourceDtype"] as? String == "BF16")
        #expect(quantized["qualityGate"] as? String == "W2.1b-kld-open")

        // The MTP draft layer's own routed experts land in their own additive
        // pool, outside packed_experts/ so the shipped layout is untouched.
        let auxiliary = try #require(manifest["auxiliaryExpertPools"] as? [[String: Any]])
        #expect(auxiliary.count == 1)
        #expect(auxiliary[0]["name"] as? String == "mtp")
        #expect(auxiliary[0]["directory"] as? String == "packed_experts_mtp")
        let files = try #require(manifest["files"] as? [String: Any])
        #expect(files["packed_experts_mtp/layer_00.bin"] != nil)
        #expect(files["ple/layer_01_ngram_rows.bin"] != nil)

        // --- The new axes are published by name for the runtime's gate.
        let arch = try #require(manifest["arch"] as? [String: Any])
        #expect(arch["family"] as? String == "qwen38flashnext")
        #expect(arch["hcCount"] as? Int == 4)
        #expect(arch["hcLowRank"] as? Int == 64)
        #expect(arch["indexerBudget"] as? Int == 128)
        #expect(arch["indexerCompressRatio"] as? Int == 4)
        #expect(arch["pleLayerIDs"] as? [Int] == [1])
        #expect(arch["pleNgramShardCount"] as? Int == 2)
        // Verbatim `ngram_vocab_size_base`: the per-head base vocab, NOT the
        // row count. True row counts come from the shard headers and live in
        // plePool (260 here), so the two must be allowed to differ.
        #expect(arch["pleNgramVocabSizeBase"] as? Int == 4_096)
        #expect(poolLayer["rows"] as? Int == 260)
        #expect(arch["requiredAxes"] as? [String]
            == ["hyperConnectionsLowRank", "attentionIndexer", "pleNgramEmbedding"])

        // --- The install verifies, and nothing unexpected is on disk.
        let verify = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: output))
        #expect(verify.unexpectedEntries.isEmpty)
        #expect(FileManager.default.fileExists(atPath: verify.receiptPath))
        try assertNoInternalRemoteDirs(outputDir: output)
    }

    @Test func flashNextSkipMTPDropsTheDraftGroupAndSaysSo() async throws {
        let snapshotDir = tmpDirForRemote("flashnext-snap-nomtp")
        let output = tmpPathForRemote("flashnext-remote-nomtp")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.buildQwen38FlashNext(at: snapshotDir)

        resetFakeHF()
        FakeHFURLProtocol.files = try flashNextRemoteFiles(snapshotDir: snapshotDir,
                                                           snapshot: snapshot)
        let result = try await RemoteStreamingRepacker(
            options: flashNextOptions(outputDir: output,
                                      sidecarPolicy: SidecarPolicy(carryMTP: false))
        ).run()

        let resident = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("model_weights.bin")))
        let entries = try flashNextResidentEntries(in: resident)
        #expect(entries.keys.contains { $0.hasPrefix("mtp.") } == false)
        #expect(!FileManager.default.fileExists(
            atPath: (output as NSString).appendingPathComponent("packed_experts_mtp")))

        let manifest = try flashNextJSON(
            (output as NSString).appendingPathComponent("manifest.json"))
        let sidecars = try #require(manifest["sidecars"] as? [String: Any])
        let mtp = try #require(sidecars["mtp"] as? [String: Any])
        #expect(mtp["carried"] as? Bool == false)
        #expect((mtp["tensorCount"] as? Int ?? 0) > 0)
        #expect(manifest["auxiliaryExpertPools"] == nil)
        // Skipped MTP tensors join the excluded list alongside the vision tower.
        #expect(result.plan.excludedMultimodalTensorNames
            .contains("mtp.fc_embedding.weight"))

        // The PLE pool and the main expert pool are unaffected by the flag.
        #expect(FileManager.default.fileExists(atPath: (output as NSString)
            .appendingPathComponent("ple/layer_01_ngram_rows.bin")))
        let verify = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: output))
        #expect(verify.unexpectedEntries.isEmpty)
    }

    /// The fused-expert axis order is an assumption (see `FlashNextPlanner`);
    /// a checkpoint that violates it must fail loudly, not install silently.
    @Test func flashNextRejectsAnUnexpectedFusedExpertShape() async throws {
        let snapshotDir = tmpDirForRemote("flashnext-snap-badshape")
        let output = tmpPathForRemote("flashnext-remote-badshape")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.buildQwen38FlashNext(
            at: snapshotDir, mutation: .brokenFusedExpertShape)

        resetFakeHF()
        FakeHFURLProtocol.files = try flashNextRemoteFiles(snapshotDir: snapshotDir,
                                                           snapshot: snapshot)
        await #expect(throws: RepackError.self) {
            _ = try await RemoteStreamingRepacker(
                options: flashNextOptions(outputDir: output)).run()
        }
        #expect(!FileManager.default.fileExists(
            atPath: (output as NSString).appendingPathComponent("manifest.json")))
    }

    @Test func flashNextTensorClassifiersSeparateFusedExpertsFromTheSharedFFN() {
        #expect(FlashNextPlanner.fusedExpertRole(
            in: "model.language_model.layers.0.mlp.experts.gate_up_proj") == "gate_up")
        #expect(FlashNextPlanner.fusedExpertRole(
            in: "model.language_model.layers.0.mlp.experts.down_proj.weight") == "down")
        // The shared FFN shares the `.mlp.` segment and must stay resident.
        #expect(FlashNextPlanner.fusedExpertRole(
            in: "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight") == nil)
        #expect(FlashNextPlanner.fusedExpertRole(
            in: "model.language_model.layers.0.mlp.shared_expert.down_proj.weight") == nil)
        #expect(FlashNextPlanner.ngramShardIndex(
            in: "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
                + ".shard_17.weight") == 17)
        #expect(FlashNextPlanner.ngramShardIndex(
            in: "model.language_model.layers.1.ple.key_proj.weight") == nil)
    }

    /// The pool's storage is chosen from the row width, not hard-coded. The
    /// real table is 160 wide and must land on BF16; a width the group size
    /// divides must still quantize. Only the BF16 branch runs end to end, so
    /// this is what keeps the other one honest.
    @Test func plePoolStorageFollowsTheRowWidth() throws {
        func pool(rowDim: Int, rowsPerShard: Int) throws -> PleRowPoolPlan {
            let shards = (0..<2).map { index in
                (index: index,
                 tensor: SourceTensor(name: "ngram.shard_\(index).weight",
                                      shardPath: "shard",
                                      dtype: .bf16,
                                      shape: [UInt64(rowsPerShard), UInt64(rowDim)],
                                      absoluteOffset: 0,
                                      sizeBytes: UInt64(rowsPerShard * rowDim * 2)),
                 rows: rowsPerShard)
            }
            return try PleRowPoolPlan.make(layerIndex: 1,
                                           outputDir: "/tmp/does-not-exist",
                                           sourceTensorPrefix: "ngram",
                                           rowDim: rowDim,
                                           shards: shards)
        }

        // Production width: 16 n-gram heads x 160 = 2560, and 160 % 64 != 0.
        let wide = try pool(rowDim: 160, rowsPerShard: 130)
        #expect(wide.storage == .bf16)
        #expect(wide.bits == 16)
        #expect(wide.groupSize == 0)
        #expect(wide.rowCompanionBytes == 0)
        #expect(wide.rowStride == 320)
        #expect(wide.rowsPerBlock == 51)
        #expect(wide.blockStride == 16_384)
        #expect(wide.tailTransform == .identity)
        #expect(wide.blockTransform == .bf16RowBlocks(rowSourceBytes: 320,
                                                      rowsPerBlock: 51,
                                                      blockStride: 16_384))
        // 130 rows = 2 whole blocks + a 28-row tail, 3 blocks per shard region.
        #expect(wide.shards[0].fullBlocks == 2)
        #expect(wide.shards[0].tailRows == 28)
        #expect(wide.shards[1].regionOffset == 3 * 16_384)
        #expect(wide.fileSize == 6 * 16_384)

        // A width the group size divides still quantizes.
        let narrow = try pool(rowDim: 128, rowsPerShard: 130)
        #expect(narrow.storage == .int4AffineG64)
        #expect(narrow.bits == 4)
        #expect(narrow.groupSize == 64)
        #expect(narrow.rowStride == 72)          // 64 packed + 4 scale + 4 bias
        #expect(narrow.rowsPerBlock == 227)
        #expect(narrow.tailTransform == .quantizeInt4G64Rows(rowSourceBytes: 256))

        // Whatever the storage, a row never straddles a page.
        for plan in [wide, narrow] {
            #expect(UInt64(plan.rowsPerBlock) * plan.rowStride <= plan.blockStride)
            #expect(plan.blockStride % 16_384 == 0)
        }
    }

    /// Every resident tensor is either a genuine 2-D projection (quantized) or
    /// rides through at its source dtype. Getting this backwards would quantize
    /// a norm or leave a projection at BF16, and neither shows up as a crash.
    @Test func flashNextResidentQuantizationRuleIsExplicit() {
        func tensor(_ name: String, _ dtype: SourceTensor.Dtype,
                    _ shape: [UInt64]) -> SourceTensor {
            SourceTensor(name: name, shardPath: "shard", dtype: dtype, shape: shape,
                         absoluteOffset: 0,
                         sizeBytes: shape.reduce(1, *) * UInt64(dtype.elementBytes))
        }
        #expect(FlashNextPlanner.quantizesResident(
            tensor("model.language_model.layers.0.self_attn.q_proj.weight", .bf16, [256, 128])))
        #expect(FlashNextPlanner.quantizesResident(
            tensor("model.language_model.layers.0.attn_hyper_connection"
                + ".input_mix_weight_up.weight", .bf16, [128, 64])))
        // Norms by name, conv kernels by rank, 1-D vectors by rank, integer
        // tables by dtype, and any width the group size does not divide.
        #expect(!FlashNextPlanner.quantizesResident(
            tensor("model.language_model.layers.0.input_layernorm.weight", .bf16, [128])))
        #expect(!FlashNextPlanner.quantizesResident(
            tensor("model.language_model.layers.0.self_attn.indexer.q_layernorm.weight",
                   .bf16, [64, 128])))
        #expect(!FlashNextPlanner.quantizesResident(
            tensor("model.language_model.layers.0.linear_attn.conv1d.weight",
                   .bf16, [256, 4, 1])))
        #expect(!FlashNextPlanner.quantizesResident(
            tensor("model.language_model.layers.0.linear_attn.A_log", .bf16, [4])))
        #expect(!FlashNextPlanner.quantizesResident(
            tensor("model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
                   .i32, [2])))
        #expect(!FlashNextPlanner.quantizesResident(
            tensor("model.language_model.layers.0.mlp.gate.weight", .bf16, [4, 100])))
    }
}

// MARK: - Fixtures

private func flashNextOptions(outputDir: String,
                              sidecarPolicy: SidecarPolicy = .default)
    -> RemoteStreamingRepackOptions {
    // One PLE row block is 227 rows x 256 BF16 bytes = 58,112 bytes, so the
    // range chunk has to be at least that; production's default is 64 MB.
    RemoteStreamingRepackOptions(
        repoID: "owner/model",
        revision: "main",
        outputDir: outputDir,
        requireKnownSource: false,
        rangeChunkBytes: 1 << 20,
        minFreeReserveBytes: 0,
        overwrite: true,
        downloadSession: fakeHFSession(),
        baseURL: URL(string: "https://hf.test")!,
        retryBaseDelayNs: 0,
        sidecarPolicy: sidecarPolicy)
}

private func flashNextRemoteFiles(snapshotDir: String,
                                  snapshot: SyntheticSnapshot.Snapshot) throws
    -> [String: Data] {
    var files: [String: Data] = [
        "config.json": try Data(contentsOf: URL(fileURLWithPath:
            (snapshotDir as NSString).appendingPathComponent("config.json"))),
        "model.safetensors.index.json": try Data(contentsOf: URL(fileURLWithPath:
            (snapshotDir as NSString).appendingPathComponent("model.safetensors.index.json"))),
        "tokenizer.json": remoteTokenizerJSON,
        "tokenizer_config.json": remoteTokenizerConfigJSON,
    ]
    for path in snapshot.shardPaths {
        files[(path as NSString).lastPathComponent] =
            try Data(contentsOf: URL(fileURLWithPath: path))
    }
    return files
}

private struct FlashNextSourceTensor {
    let path: String
    let offset: UInt64
    let size: UInt64
}

private func flashNextSourceTensors(in shardPaths: [String]) throws
    -> [String: FlashNextSourceTensor] {
    var result: [String: FlashNextSourceTensor] = [:]
    for path in shardPaths {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let headerSize = try flashNextUInt64(data, at: 0)
        let header = try JSONSerialization.jsonObject(
            with: flashNextSlice(data, 8, headerSize)) as! [String: Any]
        for (name, value) in header {
            if name == "__metadata__" { continue }
            guard let entry = value as? [String: Any],
                  let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
                  let start = (offsets[0] as? NSNumber)?.uint64Value,
                  let end = (offsets[1] as? NSNumber)?.uint64Value else {
                throw NSError(domain: "RemoteFlashNextInstallTests", code: 1)
            }
            result[name] = FlashNextSourceTensor(path: path,
                                                 offset: 8 + headerSize + start,
                                                 size: end - start)
        }
    }
    return result
}

private struct FlashNextResidentEntry {
    let dtype: UInt8
    let shape: [UInt32]
    let offset: UInt64
    let size: UInt64
    let scaleOffset: UInt64
    let scaleSize: UInt64
    let biasOffset: UInt64
    let biasSize: UInt64
}

private func flashNextResidentEntries(in data: Data) throws
    -> [String: FlashNextResidentEntry] {
    let count = try flashNextUInt64(data, at: 16)
    var result: [String: FlashNextResidentEntry] = [:]
    for index in 0..<Int(count) {
        let base = 24 + index * 72
        let nameOffset = try flashNextUInt32(data, at: base)
        let nameLength = try flashNextUInt16(data, at: base + 4)
        let name = String(decoding: try flashNextSlice(data, UInt64(nameOffset),
                                                       UInt64(nameLength)), as: UTF8.self)
        let shape = try (0..<4).map { try flashNextUInt32(data, at: base + 24 + $0 * 4) }
        result[name] = FlashNextResidentEntry(
            dtype: try flashNextSlice(data, UInt64(base + 6), 1).first!,
            shape: shape,
            offset: try flashNextUInt64(data, at: base + 8),
            size: try flashNextUInt64(data, at: base + 16),
            scaleOffset: try flashNextUInt64(data, at: base + 40),
            scaleSize: try flashNextUInt64(data, at: base + 48),
            biasOffset: try flashNextUInt64(data, at: base + 56),
            biasSize: try flashNextUInt64(data, at: base + 64))
    }
    return result
}

/// The reference result for a `[rows, rowLength]` BF16 tensor, produced by the
/// whole-tensor encoder the streaming path is locked to.
private func flashNextReference(bytes: Data, rowLength: Int)
    -> (packed: Data, scales: Data, biases: Data) {
    var values = [Float]()
    values.reserveCapacity(bytes.count / 2)
    for index in stride(from: 0, to: bytes.count, by: 2) {
        let bits = UInt32(bytes[bytes.startIndex + index])
            | UInt32(bytes[bytes.startIndex + index + 1]) << 8
        values.append(Float(bitPattern: bits << 16))
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

private func flashNextExpectSlice(
    _ layerBytes: Data,
    base: UInt64,
    tensors: [String: [String: Any]],
    role: String,
    expected: (packed: Data, scales: Data, biases: Data),
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    for (key, want) in [(role, expected.packed),
                        (role + "_scales", expected.scales),
                        (role + "_biases", expected.biases)] {
        let tensor = try #require(tensors[key], "missing \(key)",
                                  sourceLocation: sourceLocation)
        let offset = UInt64(try #require(tensor["offset"] as? Int))
        let size = UInt64(try #require(tensor["size"] as? Int))
        #expect(size == UInt64(want.count), "\(key) size", sourceLocation: sourceLocation)
        #expect(try flashNextSlice(layerBytes, base + offset, size) == want,
                "\(key) bytes", sourceLocation: sourceLocation)
    }
}

private func flashNextJSON(_ path: String) throws -> [String: Any] {
    try JSONSerialization.jsonObject(
        with: Data(contentsOf: URL(fileURLWithPath: path))) as! [String: Any]
}

private func flashNextBytes(at path: String, offset: UInt64, count: UInt64) throws -> Data {
    try flashNextSlice(Data(contentsOf: URL(fileURLWithPath: path)), offset, count)
}

private func flashNextSlice(_ data: Data, _ offset: UInt64, _ count: UInt64) throws -> Data {
    guard offset <= UInt64(data.count), count <= UInt64(data.count) - offset else {
        throw NSError(domain: "RemoteFlashNextInstallTests", code: 2)
    }
    return Data(data[(data.startIndex + Int(offset))..<(data.startIndex + Int(offset + count))])
}

private func flashNextUInt16(_ data: Data, at offset: Int) throws -> UInt16 {
    let bytes = try flashNextSlice(data, UInt64(offset), 2)
    return UInt16(bytes[bytes.startIndex]) | UInt16(bytes[bytes.startIndex + 1]) << 8
}

private func flashNextUInt32(_ data: Data, at offset: Int) throws -> UInt32 {
    let bytes = try flashNextSlice(data, UInt64(offset), 4)
    return (0..<4).reduce(UInt32(0)) {
        $0 | UInt32(bytes[bytes.startIndex + $1]) << UInt32($1 * 8)
    }
}

private func flashNextUInt64(_ data: Data, at offset: Int) throws -> UInt64 {
    let bytes = try flashNextSlice(data, UInt64(offset), 8)
    return (0..<8).reduce(UInt64(0)) {
        $0 | UInt64(bytes[bytes.startIndex + $1]) << UInt64($1 * 8)
    }
}
