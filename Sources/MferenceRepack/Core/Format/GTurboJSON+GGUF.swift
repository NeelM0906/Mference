import Foundation

/// `layout.json` and manifest additions for an install planned from GGUF bytes
/// (WS4). The schema is the existing one plus fields an old reader rejects or
/// ignores; see docs/families/QWEN38_FLASH_NEXT_UNSLOTH.md.
extension GTurboJSON {
    /// ggml type name for a planned dtype code (`FlashNextGGUFPlanner.dtypeCode`).
    static func ggufFormatName(dtypeCode: UInt8) -> String {
        switch dtypeCode {
        case 1: return "BF16"
        case 2: return "F16"
        case 3: return "F32"
        default:
            if dtypeCode & FlashNextGGUFPlanner.int8TwoLevelDtypeBase != 0 {
                let id = UInt32(dtypeCode & ~FlashNextGGUFPlanner.int8TwoLevelDtypeBase)
                return "INT8-2L(" + (GGMLType(rawValue: id)?.name ?? "unknown") + ")"
            }
            let id = UInt32(dtypeCode & ~FlashNextGGUFPlanner.quantizedDtypeBase)
            return GGMLType(rawValue: id)?.name ?? "unknown"
        }
    }

    /// Nominal bits per weight of a planned dtype code, for the advisory quant
    /// slots (block scales excluded).
    static func ggufNominalBits(dtypeCode: UInt8) -> Int {
        let name = ggufFormatName(dtypeCode: dtypeCode)
        if name.hasPrefix("INT8-2L") { return 8 }
        switch name {
        case "F32": return 32
        case "F16", "BF16": return 16
        case "Q8_0": return 8
        case "Q6_K": return 6
        case "IQ4_XS", "IQ4_NL": return 4
        case "IQ3_XXS": return 3
        case "IQ2_S", "Q2_0": return 2
        default: return 0
        }
    }

    /// The `expertStorage.format` of a GGUF install (ruling R2). An old reader's
    /// `parseStorage` refuses an unknown format, so this install cannot be
    /// mistaken for INT4 or BF16 experts.
    static let ggufExpertStorageFormat = "ggufNativeBlocksV1"

    /// Per-layer expert stride and per-component ggml format. The top-level
    /// `expertStride` is the max *stored* stride; it never sizes a runtime
    /// slot, which holds the execution format (a LUT4 slot is larger than
    /// any GGUF disk stride).
    static func encodeGGUFLayout(plan: RepackPlan) throws -> Data {
        let slotStrides = try plan.layers.map(LUT4SlotGeometry.slotStride)
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
            "expertStorage": [
                "format": ggufExpertStorageFormat,
                "storedExpertStride": plan.layers.map(\.expertStride).max() ?? 0,
                "storedExpertStrides": plan.layers.map(\.expertStride),
                // Runtime slots hold LUT4 (U1 §3a); size them from these.
                "slotFormat": "lut4",
                "slotStrides": slotStrides,
            ] as [String: Any],
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
        // R3: every norm in the GGUF is already in full form. The converter
        // baked +1 into the zero-centred gammas (qwen4exp.py:165 and the
        // shared Qwen norm rule), and the pinned graph multiplies by the
        // stored weight directly (qwen4exp.cpp:362, ggml_mul(rms_norm(x), w)).
        // A runtime must never add 1 again.
        m["zeroCenteredNormsBakedAtInstall"] = true
        // Disk and slot strides per layer (U1 §3a): the top-level
        // `expertStride` is the max disk stride and never sizes a slot.
        m["slotStrides"] = try plan.layers.map(LUT4SlotGeometry.slotStride)
        // R7: advisory slots. Formats vary by layer in a UD quant, so each slot
        // lists every format its tensors use; the runtime dispatches per tensor
        // on the index dtype, never on these slots.
        func slot(_ codes: [UInt8]) -> [String: Any] {
            let formats = Array(Set(codes.map { ggufFormatName(dtypeCode: $0) })).sorted()
            return ["scheme": "gguf", "advisory": true, "formats": formats,
                    "format": formats.joined(separator: "+"),
                    "weightBits": codes.map { ggufNominalBits(dtypeCode: $0) }.max() ?? 0,
                    "scaleType": codes.contains { ggufNominalBits(dtypeCode: $0) < 16 } ? "FP16" : "none",
                    "biasType": "none", "groupSize": 0]
        }
        func resident(_ match: (String) -> Bool) -> [UInt8] {
            plan.resident.entries.filter { match($0.name) }.map(\.dtype)
        }
        let attention = ["attn_q.", "attn_k.", "attn_v.", "attn_output.", "attn_qkv.", "attn_gate.", "ssm_out."]
        m["quant"] = [
            "embedding": slot(resident { $0 == "token_embd.weight" }),
            "attention": slot(resident { n in attention.contains { n.contains("." + $0) } }),
            "router": slot(resident { $0.hasSuffix(".ffn_gate_inp.weight") }),
            "sharedExpert": slot(resident { $0.contains("_shexp.weight") && !$0.contains("gate_inp") }),
            "routedExpert": slot(plan.layers.flatMap { $0.subTensors.map(\.dtype) }),
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
            // Exact Int64 JSON integers: Foundation round-trips all 64 bits
            // (tested), so no decimal-string encoding is needed.
            layers[0]["hash"] = [
                "layerMultipliers": pleHash.layerMultipliers, "headOffsets": pleHash.headOffsets,
                "headVocabSizes": pleHash.headVocabSizes, "ngramSize": pleHash.ngramSize,
                "headsPerNgram": pleHash.headsPerNgram,
            ] as [String: Any]
            ple["layers"] = layers
            m["plePool"] = ple
            // R6: the n-gram EOS reset token goes in the existing, validated
            // `arch.pleEosTokenID` axis rather than a second copy in `hash`.
            if var arch = m["arch"] as? [String: Any] {
                arch["pleEosTokenID"] = pleHash.eosTokenID
                m["arch"] = arch
            }
        }
        return try JSONSerialization.data(withJSONObject: m,
                                          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
