import Foundation

/// Plans a Qwen3.8-Flash-Next install from a pinned GGUF upload (WS4, the
/// Unsloth UD-IQ3_XXS quant). Every byte is an identity range copy of the
/// GGUF's own bytes (U1 §3: expansion to the LUT4 / INT8-2L execution formats
/// happens in the runtime), so the installed tensors are bit-identical to the
/// source by construction.
///
/// - Routed experts `blk.L.ffn_{gate,up,down}_exps.weight` (GGML `ne` =
///   `[cols, rows, experts]`) go to `packed_experts/layer_LL.bin` as one blob
///   per expert: gate, up, down slices back to back, each 256-byte aligned,
///   padded to 16 KiB. The ggml types vary by layer, so each layer keeps its
///   own stride (no padding to the widest).
/// - `per_layer_token_embd.weight` (IQ4_NL `[160, rows]`) goes to the PLE row
///   pool with `iq4nl` storage.
/// - Every other tensor is resident under its GGUF name, 256-byte aligned,
///   with dtype code `0x40 | ggml_type` for quantized types.
enum FlashNextGGUFPlanner {
    static let residentAlignment: UInt64 = 256
    static let expertComponentAlignment: UInt64 = 256
    static let quantizedDtypeBase: UInt8 = 0x40
    static let pleTensorName = "per_layer_token_embd.weight"

    static func dtypeCode(_ t: GGMLType) -> UInt8 {
        switch t {
        case .f32: 3
        case .f16: 2
        case .bf16: 1
        default: quantizedDtypeBase | UInt8(t.rawValue)
        }
    }

    static func plan(headers: [GGUF.Header],
                     shardPaths: [String],
                     arch: ArchInfo,
                     outputDir: String,
                     matchedModelID: String?) throws -> RepackPlan {
        guard headers.count == shardPaths.count else {
            throw RepackError.configurationInvalid(detail: "GGUF headers and shard paths differ in count")
        }
        var all: [(GGUF.TensorInfo, String)] = []
        for (h, path) in zip(headers, shardPaths) { all += h.tensors.map { ($0, path) } }
        func source(_ t: GGUF.TensorInfo, _ shard: String) -> SourceTensor {
            SourceTensor(name: t.name, shardPath: shard, dtype: .ggmlBlocks, shape: [t.sizeBytes],
                         absoluteOffset: t.absoluteOffset, sizeBytes: t.sizeBytes)
        }

        var expertTensors: [Int: [String: (GGUF.TensorInfo, String)]] = [:]
        var ple: (GGUF.TensorInfo, String)?
        var resident: [(GGUF.TensorInfo, String)] = []
        for (t, shard) in all {
            if let (layer, role) = expertRole(t.name) {
                expertTensors[layer, default: [:]][role] = (t, shard)
            } else if t.name == pleTensorName {
                ple = (t, shard)
            } else {
                resident.append((t, shard))
            }
        }
        // Deterministic order: source order (shard, then offset).
        resident.sort { ($0.1, $0.0.absoluteOffset) < ($1.1, $1.0.absoluteOffset) }

        let residentPlan = try planResident(resident.map { ($0.0, source($0.0, $0.1)) },
                                            path: (outputDir as NSString).appendingPathComponent("model_weights.bin"))

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layers: [LayerFilePlan] = []
        for layer in 0..<arch.numLayers {
            let path = (layersDir as NSString).appendingPathComponent(String(format: "layer_%02d.bin", layer))
            guard let roles = expertTensors.removeValue(forKey: layer) else {
                throw RepackError.missingTensor(name: "blk.\(layer).ffn_gate_exps.weight")
            }
            layers.append(try planLayer(layer: layer, path: path, roles: roles, arch: arch, source: source))
        }
        if let stray = expertTensors.keys.min() {
            throw RepackError.unknownTensorPrefix(name: "blk.\(stray).ffn_*_exps (layer outside block_count)")
        }

        var pools: [PleRowPoolPlan] = []
        if let pleLayers = arch.flashNext?.pleLayerIDs, !pleLayers.isEmpty {
            guard pleLayers.count == 1, let (t, shard) = ple else {
                throw RepackError.missingTensor(name: pleTensorName)
            }
            guard t.type == .iq4_nl, t.ne.count == 2, let rows = Int(exactly: t.ne[1]) else {
                throw RepackError.dtypeMismatch(name: t.name, detail: "expected IQ4_NL [rowDim, rows], got \(t.type.name) \(t.ne)")
            }
            pools.append(try PleRowPoolPlan.make(layerIndex: pleLayers[0] - 1, outputDir: outputDir,
                                                 sourceTensorPrefix: t.name, rowDim: Int(t.ne[0]),
                                                 shards: [(index: 0, tensor: source(t, shard), rows: rows)],
                                                 sourceStorage: .iq4nl))
        } else if ple != nil {
            throw RepackError.configurationInvalid(detail: "\(pleTensorName) present but arch has no PLE layer")
        }

        return RepackPlan(arch: arch, baseMode: "gguf", baseGroupSize: 0, bitsOverrideCount: 0,
                          resident: residentPlan, layers: layers, matchedModelID: matchedModelID,
                          excludedMultimodalTensorNames: [], flashHead: nil, plePools: pools,
                          sidecarOutcomes: [], quantizedAtInstall: false, auxiliaryExpertPools: [])
    }

    /// `(layer, "gate" | "up" | "down")` for a routed-expert tensor name.
    static func expertRole(_ name: String) -> (Int, String)? {
        let parts = name.split(separator: ".")
        guard parts.count == 4, parts[0] == "blk", let layer = Int(parts[1]), parts[3] == "weight" else { return nil }
        switch parts[2] {
        case "ffn_gate_exps": return (layer, "gate")
        case "ffn_up_exps": return (layer, "up")
        case "ffn_down_exps": return (layer, "down")
        default: return nil
        }
    }

    private static func alignUp(_ v: UInt64, _ a: UInt64) -> UInt64 { (v + a - 1) / a * a }

    private static func planResident(_ tensors: [(GGUF.TensorInfo, SourceTensor)], path: String) throws
        -> ResidentFilePlan {
        guard Set(tensors.map { $0.0.name }).count == tensors.count else {
            throw RepackError.configurationInvalid(detail: "duplicate resident tensor name")
        }
        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        for (t, _) in tensors {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: t.name.utf8)
        }
        let rawIndex = UInt64(GTurboBinary.indexHeaderBytes + tensors.count * GTurboBinary.indexEntryBytes
                              + stringTable.count)
        let indexSize = RepackPlanner.roundUpToPage(rawIndex)
        var cursor = indexSize
        var entries: [ResidentEntry] = []
        for (t, src) in tensors {
            guard t.ne.count <= 4 else { throw RepackError.shapeMismatch(name: t.name, detail: "rank > 4") }
            cursor = alignUp(cursor, residentAlignment)
            // Row-major logical shape: GGML ne reversed.
            entries.append(ResidentEntry(name: t.name, dtype: dtypeCode(t.type),
                                         logicalShape4: RepackPlanner.padTo4(t.ne.reversed()),
                                         fileOffset: cursor, sizeBytes: t.sizeBytes,
                                         scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
                                         quantSpec: nil, sourceWeight: src, sourceScales: nil, sourceBiases: nil))
            cursor += t.sizeBytes
        }
        return ResidentFilePlan(path: path, entries: entries, stringTable: stringTable,
                                stringTableOffsets: offsets, indexSize: indexSize,
                                residentSize: cursor - indexSize)
    }

    private static func planLayer(layer: Int, path: String, roles: [String: (GGUF.TensorInfo, String)],
                                  arch: ArchInfo,
                                  source: (GGUF.TensorInfo, String) -> SourceTensor) throws -> LayerFilePlan {
        let hidden = UInt64(arch.hiddenSize), inter = UInt64(arch.moeIntermediateSize)
        let experts = UInt64(arch.numExperts)
        // GGML ne: gate/up [hidden, inter, E] (rows = inter); down [inter, hidden, E] (rows = hidden).
        let expected: [(String, [UInt64])] = [("gate", [hidden, inter, experts]),
                                              ("up", [hidden, inter, experts]),
                                              ("down", [inter, hidden, experts])]
        guard Set(roles.keys) == ["gate", "up", "down"] else {
            throw RepackError.missingTensor(name: "blk.\(layer).ffn_{gate,up,down}_exps.weight")
        }
        guard roles["gate"]!.0.type == roles["up"]!.0.type else {
            throw RepackError.dtypeMismatch(name: "blk.\(layer).ffn_up_exps.weight", detail: "gate and up types differ")
        }
        var cursor: UInt64 = 0
        var slices: [PerExpertTensorSlice] = []
        for (role, ne) in expected {
            let (t, shard) = roles[role]!
            guard t.ne == ne else {
                throw RepackError.shapeMismatch(name: t.name, detail: "expected ne \(ne), got \(t.ne)")
            }
            guard let rowBytes = t.type.rowBytes(elements: ne[0]) else {
                throw RepackError.shapeMismatch(name: t.name, detail: "row is not whole \(t.type.name) blocks")
            }
            let perExpert = rowBytes * ne[1]
            guard perExpert * experts == t.sizeBytes else {
                throw RepackError.shapeMismatch(name: t.name, detail: "per-expert bytes do not tile the tensor")
            }
            cursor = alignUp(cursor, expertComponentAlignment)
            slices.append(PerExpertTensorSlice(role: role, component: "weights", dtype: dtypeCode(t.type),
                                               logicalShape: [ne[1], ne[0]], offsetInExpertBlob: cursor,
                                               sizeInExpertBlob: perExpert, sourceOffsetPerExpert: perExpert,
                                               sourceTensor: source(t, shard), bitsForWeights: nil))
            cursor += perExpert
        }
        return LayerFilePlan(layerIndex: layer, path: path, expertsPerLayer: arch.numExperts,
                             expertStride: RepackPlanner.roundUpToPage(cursor), subTensors: slices)
    }
}
