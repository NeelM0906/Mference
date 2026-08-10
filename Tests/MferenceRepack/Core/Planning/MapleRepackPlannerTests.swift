import Foundation
import Testing
@testable import MferenceRepackCore

@Suite
struct MapleRepackPlannerTests {

    @Test func pinnedMapleConfigParsesExactly() throws {
        let root = try makeSnapshot("arch")
        defer { try? FileManager.default.removeItem(atPath: root) }

        let arch = try ArchInfo.load(configPath: configPath(in: root))

        #expect(arch.family == .maple)
        #expect(arch.hiddenSize == 2_048)
        #expect(arch.intermediateSize == 512)
        #expect(arch.moeIntermediateSize == 512)
        #expect(arch.numHeads == 16)
        #expect(arch.numKVHeads == 4)
        #expect(arch.headDim == 128)
        #expect(arch.vocabSize == 151_936)
        #expect(arch.slidingWindow == 512)
        #expect(arch.numLayers == 24)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 8)
        #expect(arch.ropeTheta == 10_000.0)
        #expect(arch.fullRopeTheta == 0.0)
        #expect(arch.partialRotaryFactor == 0.5)
        #expect(arch.fullAttentionLayerMask ==
            (0..<24).map { $0 % 4 == 3 ? 1 : 0 })
        #expect(arch.attentionScale == 1.0 / Double(128).squareRoot())
        #expect(arch.routerScoringFunc == "softmax")
        #expect(arch.routedScalingFactor == 1.0)
        #expect(arch.swigluLimit == 7.0)
        #expect(arch.numSharedExperts == 0)
        #expect(arch.numDenseLayers == 0)
        #expect(arch.routerNormAfterTopK)
    }

    @Test func mapleConfigRejectsOneFieldMutations() throws {
        for (key, value) in [
            ("hidden_size", NSNumber(value: 2_048.5) as Any),
            ("moe_shared_expert_intermediate_size", 511 as Any),
            ("moe_intermediate_size", 511 as Any),
            ("num_attention_heads", 15 as Any),
            ("num_key_value_heads", 3 as Any),
            ("head_dim", 127 as Any),
            ("vocab_size", 151_935 as Any),
            ("max_position_embeddings", 127_999 as Any),
            ("sliding_window", 511 as Any),
            ("rope_theta", 10_001.0 as Any),
            ("partial_rotary_factor", 0.25 as Any),
            ("num_hidden_layers", 23 as Any),
            ("num_experts", true as Any),
            ("num_experts_per_tok", 7 as Any),
            ("num_shared_experts", 1 as Any),
            ("first_k_dense_replace", 1 as Any),
            ("rms_norm_eps", 0.000_01 as Any),
            ("use_qk_norm", false as Any),
            ("norm_topk_prob", false as Any),
            ("tie_word_embeddings", true as Any),
            ("use_rmsnorm", false as Any),
            ("use_bias", true as Any),
            ("router_dtype", "bf16" as Any),
            ("hidden_act", "gelu" as Any),
            ("layer_types", (0..<24).map {
                $0 % 4 == 0 ? "full_attention" : "sliding_attention"
            } as Any),
        ] {
            let root = try makeSnapshot("arch-mutation") { config in
                config[key] = value
            }
            defer { try? FileManager.default.removeItem(atPath: root) }

            #expect(throws: RepackError.self) {
                _ = try ArchInfo.load(configPath: configPath(in: root))
            }
        }
    }

    @Test func mapleQuantizationIsExact() throws {
        let root = try makeSnapshot("quant")
        defer { try? FileManager.default.removeItem(atPath: root) }

        let meta = try IndexLoader.load(snapshotDir: root)
        #expect(meta.baseBits == 2)
        #expect(meta.baseGroupSize == 128)
        #expect(meta.baseMode == "affine")
        #expect(meta.bitsOverrides == [
            "model.word_embeddings": QuantSpec(bits: 4, groupSize: 64),
            "lm_head": QuantSpec(bits: 4, groupSize: 64),
        ])
    }

    @Test func mapleQuantizationRejectsSchemaMutations() throws {
        for kind in ["missing", "extra", "fractionalBase", "fractionalOverride"] {
            let root = try makeSnapshot("quant-\(kind)") { config in
                var quant = config["quantization"] as! [String: Any]
                switch kind {
                case "missing": quant.removeValue(forKey: "lm_head")
                case "extra": quant["model.layers.0.self_attn.q_proj"] = ["bits": 4, "group_size": 64]
                case "fractionalBase": quant["group_size"] = NSNumber(value: 128.5)
                default:
                    quant["lm_head"] = ["bits": NSNumber(value: 4.5), "group_size": 64]
                }
                config["quantization"] = quant
            }
            defer { try? FileManager.default.removeItem(atPath: root) }

            #expect(throws: RepackError.self) {
                _ = try IndexLoader.load(snapshotDir: root)
            }
        }
    }

    @Test func mapleSourceIsPinnedAndRegistered() throws {
        let source = try #require(SupportedModelSource.named("maple"))
        #expect(source.repoID == "deepgrove/maple-preview-2bit-mlx")
        #expect(source.revision == "361db5da5e74ff6fcdd852d478e1f266ce11013a")
        #expect(source.sourceIndexSHA256 ==
            "56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95")
        #expect(source.modelID == "maple-preview-2bit-mlx")
        #expect(source.isPinned)
        #expect(SourceFingerprint.modelID(forIndexSha256: source.sourceIndexSHA256!) == source.modelID)
    }

    @Test func mapleFlashHeadI32TensorIsParsedAndConditionallyRetained() throws {
        let headerData = try JSONSerialization.data(withJSONObject: [
            "lm_head_flash.token_map": [
                "dtype": "I32", "shape": [2], "data_offsets": [0, 8],
            ],
        ])
        let header = try Safetensors.parseHeaderBytes(
            path: "maple.safetensors", fileSize: UInt64(8 + headerData.count + 8),
            headerBytes: headerData)

        #expect(header.tensors == [SourceTensor(
            name: "lm_head_flash.token_map", shardPath: "maple.safetensors",
            dtype: .i32, shape: [2], absoluteOffset: UInt64(8 + headerData.count),
            sizeBytes: 8)])
        #expect(RepackPlanner.classify(
            "lm_head_flash.token_map", numLayers: 24, family: .maple) == .excludedMultimodal)
        #expect(RepackPlanner.classify(
            "lm_head_flash.token_map", numLayers: 24, family: .maple,
            includeMapleFlashHead: true) == .lmResident)
        #expect(RepackPlanner.classify(
            "lm_head_flash.head.weight", numLayers: 24, family: .maple,
            includeMapleFlashHead: true) == .excludedMultimodal)
        #expect(RepackPlanner.classify(
            "lm_head_flash.token_map", numLayers: 24, family: .deepseekV4Flash) == .unknown)
    }

    @Test func mapleTernaryFixturePlansResidentWideningAndRoutedSlices() throws {
        let root = temporaryRoot("plan")
        defer { try? FileManager.default.removeItem(atPath: root) }

        let snapshot = try SyntheticSnapshot.buildMaple(at: root)
        let plan = try maplePlan(snapshot: snapshot, root: root)
        let resident = plan.resident
        let q = try #require(resident.entries.first {
            $0.name == "model.layers.0.self_attn.q_proj.weight"
        })

        #expect(resident.entries.map(\.name) == [
            "model.word_embeddings.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
        ])
        #expect(q.logicalShape4 == [2, 128, 0, 0])
        #expect(q.sourceWeight.sizeBytes == 64)
        #expect(q.sizeBytes == 128)
        #expect(q.scaleSize == 8)
        #expect(q.biasSize == 8)
        #expect(q.quantSpec == QuantSpec(bits: 4, groupSize: 64))
        #expect(q.weightTransform == .unpackInt2ToInt4)
        #expect(q.scaleTransform == .repeatBF16(count: 2, negated: false))
        #expect(q.biasTransform == .repeatBF16(count: 2, negated: true))
        #expect(q.scaleOffset == q.fileOffset + q.sizeBytes)
        #expect(q.biasOffset == q.scaleOffset + q.scaleSize)

        #expect(plan.layers.count == 24)
        #expect(plan.layers.allSatisfy {
            $0.expertsPerLayer == 256 && $0.expertStride == 16_384 && $0.subTensors.count == 9
        })
        let layer = plan.layers[0]
        #expect(layer.expertsPerLayer == 256)
        #expect(layer.expertStride == 16_384)
        #expect(layer.fileSize == 4_194_304)
        #expect([0, 1, 255].map(layer.physicalRank(for:)) == [0, 1, 255])
        #expect(layer.subTensors.map { "\($0.role):\($0.component)" } == [
            "gate:weights", "gate:scales", "gate:biases",
            "up:weights", "up:scales", "up:biases",
            "down:weights", "down:scales", "down:biases",
        ])
        #expect(layer.subTensors.map(\.offsetInExpertBlob) == [
            0, 64, 72, 80, 144, 152, 160, 224, 232,
        ])
        #expect(layer.subTensors.map(\.sizeInExpertBlob) == [
            64, 8, 8, 64, 8, 8, 64, 8, 8,
        ])
        #expect(layer.subTensors.map(\.sourceOffsetPerExpert) == [
            64, 4, 4, 64, 4, 4, 64, 4, 4,
        ])
        #expect(layer.subTensors.map(\.logicalShape) == [
            [2, 128], [2, 2], [2, 2],
            [2, 128], [2, 2], [2, 2],
            [2, 128], [2, 2], [2, 2],
        ])
        #expect(layer.subTensors.map(\.bitsForWeights) == [
            2, nil, nil, 2, nil, nil, 2, nil, nil,
        ])
        #expect(layer.subTensors.map(\.transform) == [
            .identity, .repeatBF16(count: 2, negated: false), .repeatBF16(count: 2, negated: true),
            .identity, .repeatBF16(count: 2, negated: false), .repeatBF16(count: 2, negated: true),
            .identity, .repeatBF16(count: 2, negated: false), .repeatBF16(count: 2, negated: true),
        ])
        #expect(plan.excludedMultimodalTensorNames.isEmpty)
    }

    @Test func mapleFlashHeadPlanRetainsOnlyItsRequiredSourceTensors() throws {
        let root = temporaryRoot("flash-head-plan")
        defer { try? FileManager.default.removeItem(atPath: root) }

        let snapshot = try SyntheticSnapshot.buildMaple(at: root, includeFlashHead: true)
        let plan = try maplePlan(snapshot: snapshot, root: root)
        let flashHead = try #require(plan.flashHead)
        let centroid = try #require(plan.resident.entries.first {
            $0.name == "lm_head_flash.centroids.weight"
        })
        let tokenMap = try #require(plan.resident.entries.first {
            $0.name == "lm_head_flash.token_map"
        })

        #expect(flashHead.nClusters == 4_748)
        #expect(flashHead.clusterSize == 32)
        #expect(flashHead.nProbes == 512)
        #expect(flashHead.forceTokens == [151_645, 151_668, 151_643])
        #expect(centroid.dtype == 0)
        #expect(centroid.logicalShape4 == [4_748, 2_048, 0, 0])
        #expect(centroid.quantSpec == QuantSpec(bits: 4, groupSize: 64))
        #expect(centroid.sourceWeight.name == "lm_head_flash.centroids.weight")
        #expect(centroid.sourceScales?.name == "lm_head_flash.centroids.scales")
        #expect(centroid.sourceBiases?.name == "lm_head_flash.centroids.biases")
        #expect(tokenMap.dtype == 5)
        #expect(tokenMap.logicalShape4 == [4_748, 32, 0, 0])
        #expect(tokenMap.sourceWeight.name == "lm_head_flash.token_map")
        #expect(tokenMap.sourceScales == nil)
        #expect(tokenMap.sourceBiases == nil)
        #expect(plan.resident.entries.map(\.name).contains {
            $0.hasPrefix("lm_head_flash.head.") || $0 == "lm_head_flash.cluster_scale"
        } == false)
        #expect(plan.excludedMultimodalTensorNames == [
            "lm_head_flash.cluster_scale",
            "lm_head_flash.head.biases",
            "lm_head_flash.head.scales",
            "lm_head_flash.head.weight",
        ])
    }

    @Test func mapleFlashHeadRejectsMalformedSourceTensors() throws {
        let cases: [(String, [SyntheticSnapshot.MapleTensorMutation])] = [
            ("missing token map", [.remove("lm_head_flash.token_map")]),
            ("bad centroid dtype", [.replaceDtype("lm_head_flash.centroids.weight", "BF16")]),
            ("bad centroid shape", [.replaceShape("lm_head_flash.centroids.weight", [4_748, 255])]),
            ("bad token-map dtype", [.replaceDtype("lm_head_flash.token_map", "BF16")]),
            ("bad token-map shape", [.replaceShape("lm_head_flash.token_map", [4_748, 31])]),
        ]
        for (name, mutations) in cases {
            let root = temporaryRoot("flash-head-malformed-\(name)")
            defer { try? FileManager.default.removeItem(atPath: root) }
            let snapshot = try SyntheticSnapshot.buildMaple(
                at: root, includeFlashHead: true, mutations: mutations)

            #expect(throws: RepackError.self) {
                _ = try maplePlan(snapshot: snapshot, root: root)
            }
        }
    }

    @Test func mapleFlashHeadRejectsMalformedSourceMetadata() throws {
        let mutations: [(String, (inout [String: Any]) -> Void)] = [
            ("missing field", { $0.removeValue(forKey: "n_probes") }),
            ("bad dimensions", { $0["n_clusters"] = 4_747 }),
            ("too many probes", { $0["n_probes"] = 4_749 }),
            ("unscaled centroids", { $0["scaled_centroids"] = false }),
            ("duplicate forced token", { $0["force_tokens"] = [151_645, 151_645] }),
        ]
        for (name, mutate) in mutations {
            let root = temporaryRoot("flash-head-metadata-\(name)")
            defer { try? FileManager.default.removeItem(atPath: root) }
            _ = try SyntheticSnapshot.buildMaple(at: root, includeFlashHead: true)
            let url = URL(fileURLWithPath: configPath(in: root))
            var config = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
            var flashHead = try #require(config["flash_head"] as? [String: Any])
            mutate(&flashHead)
            config["flash_head"] = flashHead
            try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]).write(to: url)

            #expect(throws: RepackError.self) {
                _ = try IndexLoader.load(snapshotDir: root)
            }
        }
    }

    @Test func mapleTernaryRangePlanMapsExpandedDestinationsWithinBounds() throws {
        let root = temporaryRoot("ranges")
        defer { try? FileManager.default.removeItem(atPath: root) }

        let snapshot = try SyntheticSnapshot.buildMaple(at: root)
        let plan = try maplePlan(snapshot: snapshot, root: root)
        let ranges = try RangeCopyPlanner.plan(repackPlan: plan, rangeChunkBytes: 4_096)
        let q = try #require(plan.resident.entries.first {
            $0.name == "model.layers.0.self_attn.q_proj.weight"
        })
        let layer = plan.layers[0]
        let transformed = ranges.scalarCopies.filter { $0.transform != .identity }
        let qCopies = transformed.filter { $0.destinationPath == plan.resident.path }

        #expect(qCopies.map(\.destinationOffset) == [q.fileOffset, q.scaleOffset, q.biasOffset])
        #expect(qCopies.map(\.transform) == [
            .unpackInt2ToInt4,
            .repeatBF16(count: 2, negated: false),
            .repeatBF16(count: 2, negated: true),
        ])
        #expect(transformed.count == 36_867)
        #expect(Set(transformed.filter { $0.destinationPath.contains("/packed_experts/") }
            .map(\.destinationPath)).count == 24)
        #expect(transformed.allSatisfy { copy in
            let limit = copy.destinationPath == plan.resident.path
                ? plan.resident.totalSize : layer.fileSize
            return (try? copy.destinationByteCount()).map {
                copy.destinationOffset + $0 <= limit
            } ?? false
        })

        let firstExpert = transformed.filter {
            $0.destinationPath == layer.path && $0.destinationOffset < layer.expertStride
        }
        let lastBase = 255 * layer.expertStride
        let lastExpert = transformed.filter {
            $0.destinationPath == layer.path && $0.destinationOffset >= lastBase
        }
        #expect(firstExpert.map(\.destinationOffset) == [64, 72, 144, 152, 224, 232])
        #expect(lastExpert.map { $0.destinationOffset - lastBase } == [64, 72, 144, 152, 224, 232])
    }

    @Test func mapleTernaryFixtureRejectsMalformedCompanions() throws {
        let q = "model.layers.0.self_attn.q_proj"
        let routed = "model.layers.0.mlp.switch_mlp.gate_proj"
        let cases: [(String, [SyntheticSnapshot.MapleTensorMutation])] = [
            ("wrong routed weight dtype", [.replaceDtype(routed + ".weight", "BF16")]),
            ("wrong alpha dtype", [.replaceDtype(routed + ".row_alpha", "F32")]),
            ("wrong alpha shape", [.replaceShape(routed + ".row_alpha", [256, 1])]),
            ("wrong resident alpha dtype", [.replaceDtype(q + ".row_alpha", "F32")]),
            ("wrong resident alpha shape", [.replaceShape(q + ".row_alpha", [1])]),
            ("wrong resident BF16 weight", [.replaceDtype(q + ".weight", "BF16")]),
            ("wrong resident F32 weight", [.replaceDtype(q + ".weight", "F32")]),
            ("packed input alignment", [.replaceShape(q + ".weight", [2, 7])]),
            ("missing alpha", [.remove(routed + ".row_alpha")]),
            ("missing resident alpha", [.remove(q + ".row_alpha")]),
            ("conflicting affine companions", [.addAffineCompanions(q)]),
        ]

        for (name, mutations) in cases {
            let root = temporaryRoot("malformed-\(name)")
            defer { try? FileManager.default.removeItem(atPath: root) }
            let snapshot = try SyntheticSnapshot.buildMaple(at: root, mutations: mutations)

            #expect(throws: RepackError.self) {
                _ = try maplePlan(snapshot: snapshot, root: root)
            }
        }
    }

    @Test func localMapleSnapshotPlannerAuditWhenProvided() throws {
        guard let root = ProcessInfo.processInfo.environment["MFERENCE_MAPLE_SOURCE_SNAPSHOT"],
              !root.isEmpty else { return }

        let metadata = try IndexLoader.load(snapshotDir: root)
        let arch = try ArchInfo.load(configPath: configPath(in: root))
        let headers = try metadata.shardFilenames.map {
            try mapleHeader(path: (root as NSString).appendingPathComponent($0))
        }
        let plan = try RepackPlanner.plan(
            meta: metadata, arch: arch, shardHeaders: headers,
            outputDir: (root as NSString).appendingPathComponent("maple-planner-audit"))

        #expect(metadata.indexSha256Hex ==
            "56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95")
        #expect(arch.family == .maple)
        #expect(plan.matchedModelID == "maple-preview-2bit-mlx")
        #expect(plan.layers.count == 24)
        #expect(plan.layers.allSatisfy { $0.expertsPerLayer == 256 && $0.subTensors.count == 9 })
        #expect(plan.layers.allSatisfy { layer in
            layer.subTensors.filter { $0.component == "weights" }.allSatisfy {
                $0.bitsForWeights == 2 && $0.transform == .identity
            }
        })
        let ternaryResident = plan.resident.entries.filter {
            $0.weightTransform == .unpackInt2ToInt4
        }
        #expect(!ternaryResident.isEmpty)
        #expect(ternaryResident.allSatisfy {
            $0.quantSpec == QuantSpec(bits: 4, groupSize: 64)
                && $0.scaleTransform == .repeatBF16(count: 32, negated: false)
                && $0.biasTransform == .repeatBF16(count: 32, negated: true)
        })
        #expect(plan.flashHead?.nClusters == 4_748)
        #expect(!plan.excludedMultimodalTensorNames.contains("lm_head_flash.token_map"))
    }

    @Test func mapleManifestUsesFixedQuantSlotsAndBehavior() throws {
        let root = try makeSnapshot("manifest")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let arch = try ArchInfo.load(configPath: configPath(in: root))
        let plan = RepackPlan(
            arch: arch, baseMode: "bogus", baseGroupSize: 999, bitsOverrideCount: 17,
            resident: ResidentFilePlan(
                path: "weights.bin", entries: [], stringTable: [], stringTableOffsets: [],
                indexSize: 0, residentSize: 0),
            layers: [], matchedModelID: nil, excludedMultimodalTensorNames: [], flashHead: nil)

        let data = try GTurboJSON.encodeManifest(
            plan: plan, modelID: "maple-preview-2bit-mlx", sourceSnapshotHash: "sha256:0",
            files: [], expertsPerLayer: 256, numLayers: 24, expertStride: 16_384,
            bitWidths: .init(embedding: 7, attention: 8, router: 9, sharedExpert: 10, routedExpert: 11))
        let manifest = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let archDict = manifest["arch"] as! [String: Any]
        let quant = manifest["quant"] as! [String: [String: Any]]

        #expect(archDict["family"] as? String == "maple")
        #expect(archDict["routerScoringFunc"] as? String == "softmax")
        #expect(archDict["routedScalingFactor"] as? Double == 1.0)
        #expect(archDict["swigluLimit"] as? Double == 7.0)
        #expect(archDict["numSharedExperts"] as? Int == 0)
        #expect(archDict["numDenseLayers"] as? Int == 0)
        #expect(archDict["routerNormAfterTopK"] as? Bool == true)
        #expect(quant.count == 5)
        expectQuant(quant["embedding"], bits: 4, scheme: "affine", scale: "BF16", bias: "BF16", group: 64)
        expectQuant(quant["attention"], bits: 4, scheme: "affine", scale: "BF16", bias: "BF16", group: 64)
        expectQuant(quant["router"], bits: 16, scheme: "unquantized", scale: "none", bias: "none", group: 0)
        expectQuant(quant["sharedExpert"], bits: 0, scheme: "none", scale: "none", bias: "none", group: 0)
        expectQuant(quant["routedExpert"], bits: 2, scheme: "affine", scale: "BF16", bias: "BF16", group: 64)
    }

    private func makeSnapshot(
        _ label: String,
        mutate: (inout [String: Any]) -> Void = { _ in }) throws -> String {
        let root = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-maple-\(label)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        var config: [String: Any] = [
            "model_type": "maple",
            "hidden_size": 2_048,
            "moe_shared_expert_intermediate_size": 512,
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 151_936,
            "max_position_embeddings": 128_000,
            "sliding_window": 512,
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
            "num_hidden_layers": 24,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_shared_experts": 0,
            "first_k_dense_replace": 0,
            "rms_norm_eps": 0.000_001,
            "layer_types": (0..<24).map {
                $0 % 4 == 3 ? "full_attention" : "sliding_attention"
            },
            "use_qk_norm": true,
            "norm_topk_prob": true,
            "tie_word_embeddings": false,
            "use_rmsnorm": true,
            "use_bias": false,
            "router_dtype": "fp32",
            "hidden_act": "silu",
            "quantization": [
                "bits": 2,
                "group_size": 128,
                "mode": "affine",
                "model.word_embeddings": ["bits": 4, "group_size": 64],
                "lm_head": ["bits": 4, "group_size": 64],
            ],
        ]
        mutate(&config)
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: configPath(in: root)))
        try JSONSerialization.data(withJSONObject: ["weight_map": [String: String]()])
            .write(to: URL(fileURLWithPath: (root as NSString)
                .appendingPathComponent("model.safetensors.index.json")))
        return root
    }

    private func maplePlan(snapshot: SyntheticSnapshot.Snapshot,
                           root: String) throws -> RepackPlan {
        let metadata = try IndexLoader.load(snapshotDir: root)
        let arch = try ArchInfo.load(configPath: configPath(in: root))
        return try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [try mapleHeader(path: snapshot.shardPath)],
            outputDir: (root as NSString).appendingPathComponent("repacked"))
    }

    private func mapleHeader(path: String) throws -> Safetensors.Header {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw RepackError.safetensorsHeaderInvalid(path: path, detail: "missing header length")
        }
        let headerSize = [UInt8](lengthData).enumerated().reduce(UInt64(0)) {
            $0 | (UInt64($1.element) << UInt64(8 * $1.offset))
        }
        guard headerSize <= Safetensors.maxHeaderBytes,
              let headerData = try handle.read(upToCount: Int(headerSize)),
              headerData.count == Int(headerSize),
              let fileSize = (try FileManager.default.attributesOfItem(atPath: path)[.size]
                as? NSNumber)?.uint64Value else {
            throw RepackError.safetensorsHeaderInvalid(path: path, detail: "truncated header")
        }
        return try Safetensors.parseHeaderBytes(path: path, fileSize: fileSize,
                                                headerBytes: headerData)
    }

    private func temporaryRoot(_ label: String) -> String {
        let root = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-maple-\(label)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        return root
    }

    private func configPath(in root: String) -> String {
        (root as NSString).appendingPathComponent("config.json")
    }

    private func expectQuant(_ quant: [String: Any]?, bits: Int, scheme: String,
                             scale: String, bias: String, group: Int) {
        #expect(quant?["weightBits"] as? Int == bits)
        #expect(quant?["scheme"] as? String == scheme)
        #expect(quant?["scaleType"] as? String == scale)
        #expect(quant?["biasType"] as? String == bias)
        #expect(quant?["groupSize"] as? Int == group)
    }
}
