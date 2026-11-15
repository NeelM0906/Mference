import Foundation

/// `layout.json` and manifest additions for an install planned from GGUF bytes
/// (WS4). The schema is the existing one plus fields an old reader rejects or
/// ignores; see `scratch/program/design/ws4-u2-installer.md` §2.
extension GTurboJSON {
    /// ggml type name for a planned dtype code (`FlashNextGGUFPlanner.dtypeCode`).
    static func ggufFormatName(dtypeCode: UInt8) -> String {
        switch dtypeCode {
        case 1: return "BF16"
        case 2: return "F16"
        case 3: return "F32"
        default:
            let id = UInt32(dtypeCode & ~FlashNextGGUFPlanner.quantizedDtypeBase)
            return GGMLType(rawValue: id)?.name ?? "unknown"
        }
    }

    /// Per-layer expert stride and per-component ggml format. The top-level
    /// `expertStride` is the widest layer's, for slot sizing.
    static func encodeGGUFLayout(plan: RepackPlan) throws -> Data {
        var layersArr: [[String: Any]] = []
        for lp in plan.layers {
            var experts: [[String: Any]] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                var tensors: [String: Any] = [:]
                for slice in lp.subTensors {
                    let format = ggufFormatName(dtypeCode: slice.dtype)
                    tensors[slice.role] = [
                        "offset": slice.offsetInExpertBlob,
                        "size": slice.sizeInExpertBlob,
                        "dtype": "GGML",
                        "format": format,
                        "blockElements": GGMLType.allCases.first { $0.name == format }?.blockElements ?? 0,
                        "blockBytes": GGMLType.allCases.first { $0.name == format }?.blockBytes ?? 0,
                        "shape": slice.logicalShape.map { Int($0) },
                    ] as [String: Any]
                }
                experts.append(["expert": e, "offset": UInt64(e) * lp.expertStride,
                                "size": lp.expertStride, "tensors": tensors])
            }
            layersArr.append(["layer": lp.layerIndex, "file": (lp.path as NSString).lastPathComponent,
                              "expertStride": lp.expertStride, "experts": experts])
        }
        let obj: [String: Any] = [
            "expertStride": plan.layers.map(\.expertStride).max() ?? 0,
            "numLayers": plan.arch.numLayers,
            "expertsPerLayer": plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            "layers": layersArr,
        ]
        return try JSONSerialization.data(withJSONObject: obj,
                                          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    /// The PLE n-gram hash constants a GGUF keeps as metadata arrays; the
    /// vendor install keeps them as resident I64 tensors instead.
    struct GGUFPleHash: Equatable {
        let layerMultipliers: [Int64]
        let headOffsets: [Int64]
        let headVocabSizes: [Int64]
        let ngramSize: Int
        let headsPerNgram: Int
        let eosTokenID: Int

        static func from(metadata md: [String: GGUF.Value], path: String) throws -> GGUFPleHash {
            func ints(_ key: String) throws -> [Int64] {
                guard case .array(let a)? = md["qwen4exp." + key] else {
                    throw RepackError.ggufInvalid(path: path, detail: "missing qwen4exp.\(key)")
                }
                return try a.map { v in
                    switch v {
                    case .i64(let x): return x
                    case .u64(let x) where x <= UInt64(Int64.max): return Int64(x)
                    case .i32(let x): return Int64(x)
                    case .u32(let x): return Int64(x)
                    default: throw RepackError.ggufInvalid(path: path, detail: "qwen4exp.\(key) entry")
                    }
                }
            }
            func int(_ key: String) throws -> Int {
                guard let v = md["qwen4exp." + key]?.uint64, let n = Int(exactly: v) else {
                    throw RepackError.ggufInvalid(path: path, detail: "missing qwen4exp.\(key)")
                }
                return n
            }
            let h = GGUFPleHash(layerMultipliers: try ints("ple.layer_multipliers"),
                                headOffsets: try ints("ple.head_offsets"),
                                headVocabSizes: try ints("ple.head_vocab_sizes"),
                                ngramSize: try int("ple.ngram_size"),
                                headsPerNgram: try int("ple.heads_per_ngram"),
                                eosTokenID: try int("ple.eos_token_id"))
            // Heads tile the table: offsets are the running sum of vocab sizes.
            var running: Int64 = 0
            for (off, size) in zip(h.headOffsets, h.headVocabSizes) {
                guard off == running, size > 0 else {
                    throw RepackError.ggufInvalid(path: path, detail: "PLE head offsets do not tile the table")
                }
                running += size
            }
            guard h.headOffsets.count == h.headVocabSizes.count,
                  h.layerMultipliers.count == h.ngramSize,
                  h.headOffsets.count == (h.ngramSize - 1) * h.headsPerNgram else {
                throw RepackError.ggufInvalid(path: path, detail: "PLE hash constant counts are inconsistent")
            }
            return h
        }

        var totalRows: Int64 { headVocabSizes.reduce(0, +) }
    }

    /// Adds the GGUF-only fields to an encoded manifest:
    /// - `source{kind:"gguf", repo, commit, pinFingerprint, shards}`;
    /// - `expertStrides` (per layer);
    /// - `quant` slots with `scheme:"gguf"` and the ggml format of the slot's
    ///   tensor (an old reader's `validateQuant` rejects these, which is the
    ///   intended fail-closed behaviour);
    /// - `plePool.layers[0].hash` with the n-gram hash constants.
    static func addGGUFManifestFields(_ data: Data, plan: RepackPlan, repoID: String, pin: GGUFSourcePin,
                                      pleHash: GGUFPleHash?) throws -> Data {
        guard var m = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configurationInvalid(detail: "manifest is not a JSON object")
        }
        m["source"] = [
            "kind": "gguf", "repo": repoID, "commit": pin.commit, "pinFingerprint": pin.fingerprint,
            "shards": pin.shards.map { ["path": $0.path, "sha256": $0.sha256, "size": $0.size] as [String: Any] },
        ] as [String: Any]
        m["expertStrides"] = plan.layers.map(\.expertStride)
        func format(_ name: String) -> String {
            plan.resident.entries.first { $0.name == name }.map { ggufFormatName(dtypeCode: $0.dtype) } ?? "absent"
        }
        func routedFormats() -> [String] {
            Array(Set(plan.layers.flatMap { $0.subTensors.map { ggufFormatName(dtypeCode: $0.dtype) } })).sorted()
        }
        let slot: (String) -> [String: Any] = { f in
            ["scheme": "gguf", "format": f, "weightBits": 0, "scaleType": "none", "biasType": "none", "groupSize": 0]
        }
        var routed = slot(routedFormats().joined(separator: "+"))
        routed["formats"] = routedFormats()
        m["quant"] = [
            "embedding": slot(format("token_embd.weight")),
            "attention": slot(format("blk.3.attn_q.weight")),
            "router": slot(format("blk.0.ffn_gate_inp.weight")),
            "sharedExpert": slot(format("blk.0.ffn_gate_shexp.weight")),
            "routedExpert": routed,
        ] as [String: Any]
        if let pleHash, var ple = m["plePool"] as? [String: Any], var layers = ple["layers"] as? [[String: Any]],
           !layers.isEmpty {
            // The heads address the first sum(vocab sizes) rows; the table
            // keeps the vendor's padding to 128 equal shards (90 rows in the
            // UD-IQ3_XXS file), so the sum must fit, not match.
            guard pleHash.totalRows <= Int64(plan.plePools.first?.totalRows ?? -1) else {
                throw RepackError.ggufInvalid(path: pin.shards.first?.path ?? "",
                                              detail: "PLE heads address more rows than the table holds")
            }
            layers[0]["hash"] = [
                "layerMultipliers": pleHash.layerMultipliers, "headOffsets": pleHash.headOffsets,
                "headVocabSizes": pleHash.headVocabSizes, "ngramSize": pleHash.ngramSize,
                "headsPerNgram": pleHash.headsPerNgram, "eosTokenID": pleHash.eosTokenID,
            ] as [String: Any]
            ple["layers"] = layers
            m["plePool"] = ple
        }
        return try JSONSerialization.data(withJSONObject: m,
                                          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
