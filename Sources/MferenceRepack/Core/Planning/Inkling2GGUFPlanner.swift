import Foundation

/// Plans an Inkling 2 install from Unsloth's pinned UD-Q4_K_XL GGUF. Every
/// byte is an identity range copy of the GGUF's own bytes: nothing is
/// re-quantized, converted or renamed, so the installed tensors equal the
/// source by construction (and `--verify-install` plus the per-tensor check
/// prove it after the fact).
///
/// - Routed experts `blk.L.ffn_gate_up_exps.weight` (GGML `ne` =
///   `[hidden, 2 * width, experts]`: rows 0 ..< width are the gate, the rest
///   the up projection) and `blk.L.ffn_down_exps.weight` (`[width, hidden,
///   experts]`) go to `packed_experts/layer_LL.bin`, one blob per expert:
///   `gate_up` then `down`, each starting 256-byte aligned, the blob padded to
///   16 KiB. A layer's block format (NVFP4 or Q8_0) sets its own stride, so
///   the 4-bit layers are not padded to the 8-bit stride.
/// - Every other tensor is resident under its GGUF name, 256-byte aligned,
///   with dtype `GGMLType.residentDtype` (F32 3, BF16 1, F16 2, blocks
///   `0x40 | type`), in source order. That includes the NVFP4 per-expert
///   factors `blk.L.ffn_{gate_up,down}_exps.scale` (`[experts]` F32).
enum Inkling2GGUFPlanner {
    static let residentAlignment: UInt64 = 256
    static let expertComponentAlignment: UInt64 = 256

    /// `(layer, "gate_up" | "down")` for a routed-expert weight tensor name.
    static func expertRole(_ name: String) -> (Int, String)? {
        let parts = name.split(separator: ".")
        guard parts.count == 4, parts[0] == "blk", let layer = Int(parts[1]), parts[3] == "weight" else { return nil }
        switch parts[2] {
        case "ffn_gate_up_exps": return (layer, "gate_up")
        case "ffn_down_exps": return (layer, "down")
        default: return nil
        }
    }

    /// The source tensor a planned range reads: opaque GGML bytes addressed by byte.
    static func source(_ t: GGUF.TensorInfo, shard: String) -> SourceTensor {
        SourceTensor(name: t.name, shardPath: shard, dtype: .ggmlBlocks, shape: [t.sizeBytes],
                     absoluteOffset: t.absoluteOffset, sizeBytes: t.sizeBytes)
    }

    /// Plans the install. `shardPath` is the file's path in the pinned repo;
    /// it names the source in every range copy whether the bytes then come
    /// from the Hub or from a local copy of the same file.
    static func plan(header: GGUF.Header,
                     shardPath: String,
                     arch: ArchInfo,
                     outputDir: String,
                     matchedModelID: String?) throws -> RepackPlan {
        guard arch.family == .inkling2Loop, let ink = arch.inkling2 else {
            throw RepackError.configurationInvalid(detail: "Inkling2GGUFPlanner needs an inkling2Loop arch")
        }
        var experts: [Int: [String: GGUF.TensorInfo]] = [:]
        var resident: [GGUF.TensorInfo] = []
        for t in header.tensors {
            if let (layer, role) = expertRole(t.name) {
                guard experts[layer, default: [:]][role] == nil else {
                    throw RepackError.configurationInvalid(detail: "duplicate expert tensor \(t.name)")
                }
                experts[layer, default: [:]][role] = t
            } else {
                resident.append(t)
            }
        }
        resident.sort { $0.absoluteOffset < $1.absoluteOffset }
        let residentPlan = try planResident(resident, shard: shardPath,
                                            path: (outputDir as NSString).appendingPathComponent("model_weights.bin"))

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layers: [LayerFilePlan] = []
        for layer in 0..<arch.numLayers {
            let path = (layersDir as NSString).appendingPathComponent(String(format: "layer_%02d.bin", layer))
            guard let roles = experts.removeValue(forKey: layer) else {
                throw RepackError.missingTensor(name: "blk.\(layer).ffn_gate_up_exps.weight")
            }
            layers.append(try planLayer(layer: layer, path: path, roles: roles, arch: arch,
                                        format: ink.routedExpertFormats[layer], shard: shardPath))
        }
        if let stray = experts.keys.min() {
            throw RepackError.unknownTensorPrefix(name: "blk.\(stray).ffn_*_exps (layer outside block_count)")
        }
        return RepackPlan(arch: arch, baseMode: "gguf", baseGroupSize: 0, bitsOverrideCount: 0,
                          resident: residentPlan, layers: layers, matchedModelID: matchedModelID,
                          excludedMultimodalTensorNames: [], flashHead: nil)
    }

    private static func alignUp(_ v: UInt64, _ a: UInt64) -> UInt64 { (v + a - 1) / a * a }

    private static func planResident(_ tensors: [GGUF.TensorInfo], shard: String, path: String) throws
        -> ResidentFilePlan {
        guard Set(tensors.map(\.name)).count == tensors.count else {
            throw RepackError.configurationInvalid(detail: "duplicate resident tensor name")
        }
        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        for t in tensors {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: t.name.utf8)
        }
        let rawIndex = UInt64(GTurboBinary.indexHeaderBytes + tensors.count * GTurboBinary.indexEntryBytes
                              + stringTable.count)
        let indexSize = RepackPlanner.roundUpToPage(rawIndex)
        var cursor = indexSize
        var entries: [ResidentEntry] = []
        for t in tensors {
            guard t.ne.count <= 4 else { throw RepackError.shapeMismatch(name: t.name, detail: "rank > 4") }
            cursor = alignUp(cursor, residentAlignment)
            // Row-major logical shape: GGML ne reversed.
            let shape = RepackPlanner.padTo4(t.ne.reversed())
            entries.append(ResidentEntry(name: t.name, dtype: t.type.residentDtype,
                                         logicalShape4: shape,
                                         fileOffset: cursor, sizeBytes: t.sizeBytes,
                                         scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
                                         quantSpec: nil, sourceWeight: source(t, shard: shard),
                                         sourceScales: nil, sourceBiases: nil))
            cursor += t.sizeBytes
        }
        return ResidentFilePlan(path: path, entries: entries, stringTable: stringTable,
                                stringTableOffsets: offsets, indexSize: indexSize,
                                residentSize: cursor - indexSize)
    }

    private static func planLayer(layer: Int, path: String, roles: [String: GGUF.TensorInfo],
                                  arch: ArchInfo, format: String, shard: String) throws -> LayerFilePlan {
        let hidden = UInt64(arch.hiddenSize), width = UInt64(arch.moeIntermediateSize)
        let experts = UInt64(arch.numExperts)
        guard Set(roles.keys) == ["gate_up", "down"] else {
            throw RepackError.missingTensor(name: "blk.\(layer).ffn_{gate_up,down}_exps.weight")
        }
        // GGML ne: gate_up [hidden, 2 * width, E] (rows = 2 * width);
        // down [width, hidden, E] (rows = hidden).
        let expected: [(String, [UInt64])] = [("gate_up", [hidden, 2 * width, experts]),
                                              ("down", [width, hidden, experts])]
        var cursor: UInt64 = 0
        var slices: [PerExpertTensorSlice] = []
        for (role, ne) in expected {
            let t = roles[role]!
            guard t.type.name == format else {
                throw RepackError.dtypeMismatch(name: t.name, detail: "expected \(format), got \(t.type.name)")
            }
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
            slices.append(PerExpertTensorSlice(role: role, component: "weights", dtype: t.type.residentDtype,
                                               logicalShape: [ne[1], ne[0]], offsetInExpertBlob: cursor,
                                               sizeInExpertBlob: perExpert, sourceOffsetPerExpert: perExpert,
                                               sourceTensor: source(t, shard: shard), bitsForWeights: nil))
            cursor += perExpert
        }
        return LayerFilePlan(layerIndex: layer, path: path, expertsPerLayer: arch.numExperts,
                             expertStride: RepackPlanner.roundUpToPage(cursor), subTensors: slices)
    }
}
