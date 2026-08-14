import Foundation

/// On-disk page alignment unit for `.gturbo` files. Fixed at 16 KB regardless
/// of host page size — the format is the contract, not the kernel.
enum Layout {
    static let pageBytes: UInt64 = 16_384
}

// MARK: - Plan data types

struct ResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    /// File offset where the (packed) weight bytes start.
    let fileOffset: UInt64
    /// Size in bytes of the weight bytes.
    let sizeBytes: UInt64
    /// Offset where BF16 scales start (0 if none).
    let scaleOffset: UInt64
    let scaleSize: UInt64
    /// Offset where BF16 biases start (0 if none).
    let biasOffset: UInt64
    let biasSize: UInt64
    /// Quantization spec (nil for unquantized scalars/norms).
    let quantSpec: QuantSpec?

    /// Source tensors that supply this entry's bytes.
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor?
    let sourceBiases: SourceTensor?
    let weightTransform: RangeCopyTransform
    let scaleTransform: RangeCopyTransform
    let biasTransform: RangeCopyTransform

    init(name: String, dtype: UInt8, logicalShape4: [UInt32],
         fileOffset: UInt64, sizeBytes: UInt64,
         scaleOffset: UInt64, scaleSize: UInt64,
         biasOffset: UInt64, biasSize: UInt64,
         quantSpec: QuantSpec?,
         sourceWeight: SourceTensor,
         sourceScales: SourceTensor?, sourceBiases: SourceTensor?,
         weightTransform: RangeCopyTransform = .identity,
         scaleTransform: RangeCopyTransform = .identity,
         biasTransform: RangeCopyTransform = .identity) {
        self.name = name
        self.dtype = dtype
        self.logicalShape4 = logicalShape4
        self.fileOffset = fileOffset
        self.sizeBytes = sizeBytes
        self.scaleOffset = scaleOffset
        self.scaleSize = scaleSize
        self.biasOffset = biasOffset
        self.biasSize = biasSize
        self.quantSpec = quantSpec
        self.sourceWeight = sourceWeight
        self.sourceScales = sourceScales
        self.sourceBiases = sourceBiases
        self.weightTransform = weightTransform
        self.scaleTransform = scaleTransform
        self.biasTransform = biasTransform
    }
}

struct ResidentFilePlan: Sendable {
    let path: String
    let entries: [ResidentEntry]
    let stringTable: [UInt8]
    let stringTableOffsets: [UInt32]   // per-entry offsets into the table
    let indexSize: UInt64              // header + entries + table + padding
    let residentSize: UInt64           // tensor payload region
    var totalSize: UInt64 { indexSize + residentSize }
}

struct PerExpertTensorSlice: Sendable {
    let role: String                   // "gate" | "up" | "down"
    let component: String              // "weights" | "scales" | "biases"
    let dtype: UInt8                   // 0=U32, 1=BF16
    let logicalShape: [UInt64]         // per-expert logical shape
    let offsetInExpertBlob: UInt64     // within each expert blob
    let sizeInExpertBlob: UInt64
    /// For each expert e (0..<expertsPerLayer): source byte offset & size.
    let sourceOffsetPerExpert: UInt64  // stride per expert in source
    let sourceTensor: SourceTensor
    let bitsForWeights: Int?           // 4 for routed expert weight; nil for scales/biases
    let transform: RangeCopyTransform

    init(role: String, component: String, dtype: UInt8,
         logicalShape: [UInt64], offsetInExpertBlob: UInt64,
         sizeInExpertBlob: UInt64, sourceOffsetPerExpert: UInt64,
         sourceTensor: SourceTensor, bitsForWeights: Int?,
         transform: RangeCopyTransform = .identity) {
        self.role = role
        self.component = component
        self.dtype = dtype
        self.logicalShape = logicalShape
        self.offsetInExpertBlob = offsetInExpertBlob
        self.sizeInExpertBlob = sizeInExpertBlob
        self.sourceOffsetPerExpert = sourceOffsetPerExpert
        self.sourceTensor = sourceTensor
        self.bitsForWeights = bitsForWeights
        self.transform = transform
    }
}

struct LayerFilePlan: Sendable {
    let layerIndex: Int
    let path: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    let subTensors: [PerExpertTensorSlice]  // 9 entries: gate/up/down × {weights, scales, biases}
    var fileSize: UInt64 { UInt64(expertsPerLayer) * expertStride }

    func physicalRank(for logicalExpert: Int) -> Int {
        logicalExpert
    }

    init(layerIndex: Int,
                path: String,
                expertsPerLayer: Int,
                expertStride: UInt64,
                subTensors: [PerExpertTensorSlice]) {
        self.layerIndex = layerIndex
        self.path = path
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.subTensors = subTensors
    }
}

struct RepackPlan: Sendable {
    let arch: ArchInfo
    let baseMode: String                  // "affine"
    let baseGroupSize: Int                // 64
    let bitsOverrideCount: Int
    let resident: ResidentFilePlan
    let layers: [LayerFilePlan]
    let matchedModelID: String?
    let excludedMultimodalTensorNames: [String]
    let flashHead: IndexLoader.MapleFlashHeadMetadata?
}

// MARK: - Planner

enum RepackPlanner {

    /// Classify a tensor name. Routed-expert tensors split off the LM bucket.
    enum Bucket: Equatable {
        case lmResident
        case routedExpert(role: String, layer: Int)   // role = "gate"|"up"|"down"
        case excludedMultimodal
        case unknown
    }

    static func classify(_ name: String, numLayers: Int,
                         family: RepackModelFamily,
                         includeMapleFlashHead: Bool = false) -> Bucket {
        if family == .maple, name.hasPrefix("lm_head_flash.") {
            if isExcludedMapleFlashTensor(name) || !includeMapleFlashHead {
                return .excludedMultimodal
            }
            return isMapleFlashResidentTensor(name) ? .lmResident : .unknown
        }
        if isExcludedTensorName(name, family: family) {
            return .excludedMultimodal
        }
        if hasResidentPrefix(name, family: family) {
            // Routed expert?
            if let role = routedExpertRole(in: name, family: family),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(role: role, layer: layer)
            }
            return .lmResident
        }
        return .unknown
    }

    /// The LM prefix is a family contract: Gemma and Qwen ship multimodal
    /// checkpoints whose text tower lives under `language_model.`; DeepSeek
    /// V4 is text-only with a plain `model.` prefix and a top-level
    /// `lm_head.` (mlx-lm conversion naming).
    private static func hasResidentPrefix(_ name: String,
                                          family: RepackModelFamily) -> Bool {
        switch family {
        case .gemma4, .qwen36, .qwen38:
            return name.hasPrefix("language_model.")
        case .deepseekV4Flash, .maple:
            return name.hasPrefix("model.") || name.hasPrefix("lm_head.")
        case .inklingSmall:
            // Inkling is multimodal but names its towers as siblings, so the
            // text tower is exactly `model.llm.`; `model.visual.` and
            // `model.audio.` fall through to the multimodal exclusion.
            return name.hasPrefix("model.llm.")
        }
    }

    private static func routedExpertRole(in name: String,
                                         family: RepackModelFamily) -> String? {
        let routedContainer: String
        switch family {
        case .gemma4:          routedContainer = ".experts.switch_glu."
        case .qwen36:          routedContainer = ".mlp.switch_mlp."
        // Dense family: every `.mlp.*_proj` is the layer's own FFN, resident.
        case .qwen38:          return nil
        case .deepseekV4Flash: routedContainer = ".ffn.switch_mlp."
        case .maple:           routedContainer = ".mlp.switch_mlp."
        // `.mlp.experts.` does not match the shared experts, which sit under
        // `.mlp.shared_experts.`.
        case .inklingSmall:    routedContainer = ".mlp.experts."
        }
        guard name.contains(routedContainer) else { return nil }
        if name.contains(".gate_proj.") { return "gate" }
        if name.contains(".up_proj.")   { return "up" }
        if name.contains(".down_proj.") { return "down" }
        return nil
    }

    private static func layerIndex(in name: String) -> Int? {
        // matches "...layers.<N>...."
        guard let r = name.range(of: ".layers.") else { return nil }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// Build the plan from parsed shard headers + source metadata.
    /// - throws: classification + companion + override count failures.
    static func plan(meta: IndexLoader.SourceMetadata,
                            arch: ArchInfo,
                            shardHeaders: [Safetensors.Header],
                            outputDir: String) throws -> RepackPlan {

        // Companion tensors may live in different shards, so resolve them
        // through one global registry.
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        // Source allowlisting owns exact fingerprint validation. Preserve the
        // declared override count for the output manifest audit.
        let bitsOverrideCount = meta.bitsOverrides.count

        var lmResidentBases: [String] = []
        var excludedMultimodalNames: [String] = []
        var routedByLayerAndRole: [Int: [String: String]] = [:]
        if let flashHead = meta.flashHead {
            try validateMapleFlashHead(flashHead, registry: registry, arch: arch)
        }
        for (name, _) in registry {
            if isExcludedTensorName(name, family: arch.family)
                || (arch.family == .maple && meta.flashHead == nil
                    && name.hasPrefix("lm_head_flash.")) {
                excludedMultimodalNames.append(name)
            }
            if name.hasSuffix(".scales") || name.hasSuffix(".biases")
                || (arch.family == .maple && name.hasSuffix(".row_alpha")) { continue }
            let b = classify(name, numLayers: arch.numLayers, family: arch.family,
                             includeMapleFlashHead: meta.flashHead != nil)
            switch b {
            case .lmResident:                   lmResidentBases.append(name)
            case .routedExpert(let role, let layer):
                var byRole = routedByLayerAndRole[layer] ?? [:]
                if byRole[role] != nil {
                    throw RepackError.configurationInvalid(detail:
                        "two routed-expert tensors for layer \(layer) role \(role)")
                }
                byRole[role] = name
                routedByLayerAndRole[layer] = byRole
            case .excludedMultimodal:           continue
            case .unknown:                      throw RepackError.unknownTensorPrefix(name: name)
            }
        }

        // Sort deterministically. The LM order follows a fixed template.
        lmResidentBases.sort(by: lmResidentOrdering(family: arch.family))
        excludedMultimodalNames.sort()

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath,
                                            baseNames: lmResidentBases,
                                            registry: registry, meta: meta,
                                            family: arch.family)

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layerPlans: [LayerFilePlan] = []
        layerPlans.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let bundle = routedByLayerAndRole[layer] ?? [:]
            // Synthetic snapshots may legitimately have no routed experts.
            guard let gName = bundle["gate"], let uName = bundle["up"], let dName = bundle["down"] else {
                if bundle.isEmpty {
                    layerPlans.append(LayerFilePlan(layerIndex: layer,
                                                    path: (layersDir as NSString).appendingPathComponent("layer_\(String(format: "%02d", layer)).bin"),
                                                    expertsPerLayer: 0,
                                                    expertStride: 0,
                                                    subTensors: []))
                    continue
                }
                throw RepackError.configurationInvalid(detail:
                    "layer \(layer) routed-expert bundle incomplete: \(bundle)")
            }
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            let lp = try planLayerFile(path: path, layer: layer,
                                       gateName: gName, upName: uName, downName: dName,
                                       registry: registry, meta: meta, arch: arch)
            layerPlans.append(lp)
        }

        // The .gturbo format keeps one expertStride across every layer
        // (manifest root + runtime streamers assume it). Mixed per-tensor
        // quant group sizes make raw blob sizes differ per layer — DeepSeek
        // V4's group-64 gate_proj on the last layer packs 0.5 MiB tighter
        // than the group-32 layers — so pad every layer to the widest
        // stride. Sub-tensor offsets are blob-relative and unaffected.
        let maxStride = layerPlans.map(\.expertStride).max() ?? 0
        layerPlans = layerPlans.map { lp in
            guard lp.expertsPerLayer > 0, lp.expertStride != maxStride else { return lp }
            return LayerFilePlan(layerIndex: lp.layerIndex,
                                 path: lp.path,
                                 expertsPerLayer: lp.expertsPerLayer,
                                 expertStride: maxStride,
                                 subTensors: lp.subTensors)
        }

        let matched = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex)

        return RepackPlan(arch: arch,
                          baseMode: meta.baseMode,
                          baseGroupSize: arch.family == .maple ? 64 : meta.baseGroupSize,
                          bitsOverrideCount: bitsOverrideCount,
                          resident: resident,
                          layers: layerPlans,
                          matchedModelID: matched,
                          excludedMultimodalTensorNames: excludedMultimodalNames,
                          flashHead: meta.flashHead)
    }

    private static func isExcludedTensorName(_ name: String,
                                             family: RepackModelFamily) -> Bool {
        name.hasPrefix("vision_tower.") ||
            name.hasPrefix("embed_vision.") ||
            name.hasPrefix("audio_tower.") ||
            name.hasPrefix("model.visual.") ||
            name.hasPrefix("model.audio.") ||
            (family == .maple && isExcludedMapleFlashTensor(name))
    }

    private static func isMapleFlashResidentTensor(_ name: String) -> Bool {
        switch name {
        case "lm_head_flash.centroids.weight", "lm_head_flash.token_map":
            return true
        default:
            return false
        }
    }

    private static func isExcludedMapleFlashTensor(_ name: String) -> Bool {
        name == "lm_head_flash.cluster_scale" || name.hasPrefix("lm_head_flash.head.")
    }

    // MARK: - Resident planning

    private static func planResidentFile(path: String,
                                         baseNames: [String],
                                         registry: [String: SourceTensor],
                                         meta: IndexLoader.SourceMetadata,
                                         family: RepackModelFamily) throws
                                        -> ResidentFilePlan {
        let entryCount = baseNames.count

        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(entryCount)
        for n in baseNames {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: n.utf8)
        }

        // Index size includes the fixed header, fixed-width entries, and the
        // string table, padded to a 16 KB page boundary.
        let rawIdx = UInt64(GTurboBinary.indexHeaderBytes
            + entryCount * GTurboBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        var fileCursor = indexSize
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(entryCount)

        for name in baseNames {
            guard let weight = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            if family == .maple,
               isMaplePackedResidentWeight(name),
               weight.dtype != .u32 {
                throw RepackError.dtypeMismatch(
                    name: name,
                    detail: "expected U32 packed Maple weight, got \(weight.dtype)")
            }
            let dtype = ietnyDtype(weight.dtype)
            let isQuantizedPacked = (weight.dtype == .u32) && name.hasSuffix(".weight")

            if isQuantizedPacked {
                let base = String(name.dropLast(".weight".count))
                if family == .maple {
                    let sourceSpec = IndexLoader.quantSpec(forTensor: name, meta: meta)
                    if sourceSpec.bits == 2 {
                        guard let alpha = registry[base + ".row_alpha"] else {
                            throw RepackError.configurationInvalid(
                                detail: "Maple INT2 tensor \(name) is missing row_alpha")
                        }
                        guard registry[base + ".scales"] == nil,
                              registry[base + ".biases"] == nil else {
                            throw RepackError.configurationInvalid(
                                detail: "Maple INT2 tensor \(name) has affine companions")
                        }
                        guard alpha.dtype == .bf16 else {
                            throw RepackError.dtypeMismatch(
                                name: alpha.name,
                                detail: "expected BF16 row_alpha, got \(alpha.dtype)")
                        }
                        try validateMapleStorage(weight, elementBytes: 4)
                        try validateMapleStorage(alpha, elementBytes: 2)
                        let logical = try mapleLogicalShape(forPackedSource: weight.shape,
                                                            name: name)
                        guard let input = logical.last, input.isMultiple(of: 64), input > 0,
                              alpha.shape == Array(logical.dropLast()) else {
                            throw RepackError.shapeMismatch(
                                name: name,
                                detail: "row_alpha shape \(alpha.shape) does not match ternary output rows \(Array(logical.dropLast()))")
                        }
                        let repetitionCount = input / 64
                        let repetitions = try exactInt(repetitionCount,
                                                       name: name,
                                                       detail: "Maple row_alpha repeat count is not representable as Int")
                        let wOff = fileCursor
                        let wSize = try checkedProduct(weight.sizeBytes, 2,
                                                       detail: "Maple widened weight size overflows UInt64")
                        let sOff = try checkedAdd(wOff, wSize,
                                                   detail: "Maple resident scale offset overflows UInt64")
                        let sSize = try checkedProduct(alpha.sizeBytes, repetitionCount,
                                                       detail: "Maple resident companion size overflows UInt64")
                        let bOff = try checkedAdd(sOff, sSize,
                                                   detail: "Maple resident bias offset overflows UInt64")
                        let bSize = sSize
                        fileCursor = try checkedAdd(bOff, bSize,
                                                    detail: "Maple resident file size overflows UInt64")
                        entries.append(ResidentEntry(
                            name: name, dtype: 0,
                            logicalShape4: try mapleShape4(logical, name: name),
                            fileOffset: wOff, sizeBytes: wSize,
                            scaleOffset: sOff, scaleSize: sSize,
                            biasOffset: bOff, biasSize: bSize,
                            quantSpec: QuantSpec(bits: 4, groupSize: 64),
                            sourceWeight: weight,
                            sourceScales: alpha,
                            sourceBiases: alpha,
                            weightTransform: .unpackInt2ToInt4,
                            scaleTransform: .repeatBF16(count: repetitions, negated: false),
                            biasTransform: .repeatBF16(count: repetitions, negated: true)))
                        continue
                    }
                    guard sourceSpec.bits == 4, sourceSpec.groupSize == 64 else {
                        throw RepackError.configJsonInvalid(
                            path: meta.configPath,
                            detail: "Maple packed tensor \(name) is neither INT2 ternary nor INT4/group-64 affine")
                    }
                    guard registry[base + ".row_alpha"] == nil else {
                        throw RepackError.configurationInvalid(
                            detail: "Maple INT4 tensor \(name) has a row_alpha companion")
                    }
                }
                guard let scales = registry[base + ".scales"] else {
                    throw RepackError.missingScalesCompanion(name: name)
                }
                guard let biases = registry[base + ".biases"] else {
                    throw RepackError.missingBiasesCompanion(name: name)
                }
                if scales.dtype != .bf16 || biases.dtype != .bf16 {
                    throw RepackError.dtypeMismatch(name: name,
                        detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
                }
                let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
                let logical = logicalShape(forPackedSource: weight.shape, bits: spec.bits)

                let wOff = fileCursor
                let wSize = weight.sizeBytes
                let sOff = wOff + wSize
                let sSize = scales.sizeBytes
                let bOff = sOff + sSize
                let bSize = biases.sizeBytes
                fileCursor = bOff + bSize

                entries.append(ResidentEntry(
                    name: name, dtype: 0,
                    logicalShape4: padTo4(logical),
                    fileOffset: wOff, sizeBytes: wSize,
                    scaleOffset: sOff, scaleSize: sSize,
                    biasOffset: bOff, biasSize: bSize,
                    quantSpec: spec,
                    sourceWeight: weight, sourceScales: scales, sourceBiases: biases))
            } else {
                // Unquantized (BF16 norm / scalar) — no companions.
                let off = fileCursor
                let size = weight.sizeBytes
                fileCursor = off + size

                entries.append(ResidentEntry(
                    name: name, dtype: dtype,
                    logicalShape4: padTo4(weight.shape),
                    fileOffset: off, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }

        let residentSize = fileCursor - indexSize

        return ResidentFilePlan(path: path,
                                entries: entries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: residentSize)
    }

    // MARK: - Layer planning

    private static func planLayerFile(path: String, layer: Int,
                                      gateName: String, upName: String, downName: String,
                                      registry: [String: SourceTensor],
                                      meta: IndexLoader.SourceMetadata,
                                      arch: ArchInfo) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        let roles: [(role: String, name: String)] = [
            ("gate", gateName), ("up", upName), ("down", downName)
        ]
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(9)
        var blobCursor: UInt64 = 0

        for (role, name) in roles {
            guard let w = registry[name] else { throw RepackError.missingTensor(name: name) }
            guard let sourceExpertCount = Int(exactly: w.shape.first ?? 0),
                  w.dtype == .u32, w.shape.count == 3,
                  sourceExpertCount == expertCount else {
                throw RepackError.shapeMismatch(name: name,
                    detail: "expected U32 rank-3 with leading \(expertCount), got \(w.dtype) \(w.shape)")
            }
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            if arch.family == .maple {
                let sourceSpec = IndexLoader.quantSpec(forTensor: name, meta: meta)
                guard sourceSpec.bits == 2 else {
                    throw RepackError.configJsonInvalid(
                        path: meta.configPath,
                        detail: "Maple routed tensor \(name) is not INT2")
                }
                guard let alpha = registry[base + ".row_alpha"] else {
                    throw RepackError.configurationInvalid(
                        detail: "Maple INT2 routed tensor \(name) is missing row_alpha")
                }
                guard registry[base + ".scales"] == nil,
                      registry[base + ".biases"] == nil else {
                    throw RepackError.configurationInvalid(
                        detail: "Maple INT2 routed tensor \(name) has affine companions")
                }
                guard alpha.dtype == .bf16 else {
                    throw RepackError.dtypeMismatch(
                        name: alpha.name,
                        detail: "expected BF16 row_alpha, got \(alpha.dtype)")
                }
                try validateMapleStorage(w, elementBytes: 4)
                try validateMapleStorage(alpha, elementBytes: 2)
                let expertCount64 = try exactUInt64(expertCount,
                                                    name: name,
                                                    detail: "Maple expert count is not representable as UInt64")
                guard w.sizeBytes.isMultiple(of: expertCount64),
                      alpha.sizeBytes.isMultiple(of: expertCount64) else {
                    throw RepackError.shapeMismatch(
                        name: name,
                        detail: "source bytes not evenly divisible by \(expertCount) experts")
                }
                let perExpertWeightSize = w.sizeBytes / expertCount64
                let perExpertAlphaSize = alpha.sizeBytes / expertCount64
                let logicalPerExpert = try mapleLogicalShape(
                    forPackedSource: Array(w.shape.dropFirst()), name: name)
                guard let input = logicalPerExpert.last, input.isMultiple(of: 64), input > 0,
                      alpha.shape == [expertCount64] + Array(logicalPerExpert.dropLast()) else {
                    throw RepackError.shapeMismatch(
                        name: name,
                        detail: "row_alpha shape \(alpha.shape) does not match routed output rows")
                }
                let repetitionCount = input / 64
                let repetitions = try exactInt(repetitionCount,
                                               name: name,
                                               detail: "Maple row_alpha repeat count is not representable as Int")
                let perExpertCompanionSize = try checkedProduct(
                    perExpertAlphaSize, repetitionCount,
                    detail: "Maple routed companion size overflows UInt64")
                let companionLogical = Array(logicalPerExpert.dropLast()) + [repetitionCount]

                let wSlice = PerExpertTensorSlice(
                    role: role, component: "weights", dtype: 0,
                    logicalShape: logicalPerExpert,
                    offsetInExpertBlob: blobCursor,
                    sizeInExpertBlob: perExpertWeightSize,
                    sourceOffsetPerExpert: perExpertWeightSize,
                    sourceTensor: w,
                    bitsForWeights: 2)
                blobCursor = try checkedAdd(blobCursor, perExpertWeightSize,
                                            detail: "Maple routed weight cursor overflows UInt64")
                let sSlice = PerExpertTensorSlice(
                    role: role, component: "scales", dtype: 1,
                    logicalShape: companionLogical,
                    offsetInExpertBlob: blobCursor,
                    sizeInExpertBlob: perExpertCompanionSize,
                    sourceOffsetPerExpert: perExpertAlphaSize,
                    sourceTensor: alpha,
                    bitsForWeights: nil,
                    transform: .repeatBF16(count: repetitions, negated: false))
                blobCursor = try checkedAdd(blobCursor, perExpertCompanionSize,
                                            detail: "Maple routed scale cursor overflows UInt64")
                let bSlice = PerExpertTensorSlice(
                    role: role, component: "biases", dtype: 1,
                    logicalShape: companionLogical,
                    offsetInExpertBlob: blobCursor,
                    sizeInExpertBlob: perExpertCompanionSize,
                    sourceOffsetPerExpert: perExpertAlphaSize,
                    sourceTensor: alpha,
                    bitsForWeights: nil,
                    transform: .repeatBF16(count: repetitions, negated: true))
                blobCursor = try checkedAdd(blobCursor, perExpertCompanionSize,
                                            detail: "Maple routed bias cursor overflows UInt64")
                subs.append(wSlice); subs.append(sSlice); subs.append(bSlice)
                continue
            }
            guard let s = registry[base + ".scales"] else { throw RepackError.missingScalesCompanion(name: name) }
            guard let b = registry[base + ".biases"] else { throw RepackError.missingBiasesCompanion(name: name) }
            if s.dtype != .bf16 || b.dtype != .bf16 {
                throw RepackError.dtypeMismatch(name: name,
                    detail: "expected BF16 scales/biases, got \(s.dtype)/\(b.dtype)")
            }

            let perExpertWeightSize = w.sizeBytes / UInt64(expertCount)
            let perExpertScaleSize  = s.sizeBytes / UInt64(expertCount)
            let perExpertBiasSize   = b.sizeBytes / UInt64(expertCount)
            if perExpertWeightSize * UInt64(expertCount) != w.sizeBytes ||
               perExpertScaleSize  * UInt64(expertCount) != s.sizeBytes ||
               perExpertBiasSize   * UInt64(expertCount) != b.sizeBytes {
                throw RepackError.shapeMismatch(name: name,
                    detail: "source bytes not evenly divisible by \(expertCount) experts")
            }

            let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
            let perExpertSourceShape = Array(w.shape.dropFirst())
            let logicalPerExpert = logicalShape(forPackedSource: perExpertSourceShape, bits: spec.bits)
            let scalesLogical = Array(s.shape.dropFirst())
            let biasesLogical = Array(b.shape.dropFirst())

            let wSlice = PerExpertTensorSlice(
                role: role, component: "weights", dtype: 0,
                logicalShape: logicalPerExpert,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertWeightSize,
                sourceOffsetPerExpert: perExpertWeightSize, sourceTensor: w,
                bitsForWeights: spec.bits)
            blobCursor += perExpertWeightSize
            let sSlice = PerExpertTensorSlice(
                role: role, component: "scales", dtype: 1,
                logicalShape: scalesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertScaleSize,
                sourceOffsetPerExpert: perExpertScaleSize, sourceTensor: s,
                bitsForWeights: nil)
            blobCursor += perExpertScaleSize
            let bSlice = PerExpertTensorSlice(
                role: role, component: "biases", dtype: 1,
                logicalShape: biasesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertBiasSize,
                sourceOffsetPerExpert: perExpertBiasSize, sourceTensor: b,
                bitsForWeights: nil)
            blobCursor += perExpertBiasSize

            subs.append(wSlice); subs.append(sSlice); subs.append(bSlice)
        }

        let expertStride: UInt64
        if arch.family == .maple {
            expertStride = try roundUpToPageChecked(
                blobCursor,
                detail: "Maple expert stride alignment overflows UInt64")
            let expertCount64 = try exactUInt64(
                expertCount,
                name: "layer_\(layer)",
                detail: "Maple expert count is not representable as UInt64")
            _ = try checkedProduct(
                expertCount64, expertStride,
                detail: "Maple expert file size overflows UInt64")
        } else {
            expertStride = roundUpToPage(blobCursor)
        }
        return LayerFilePlan(layerIndex: layer, path: path,
                             expertsPerLayer: expertCount,
                             expertStride: expertStride,
                             subTensors: subs)
    }

    // MARK: - Helpers

    private static func ietnyDtype(_ d: SourceTensor.Dtype) -> UInt8 {
        switch d {
        case .u32: 0
        case .bf16: 1
        case .fp16: 2
        case .fp32: 3
        case .i64: 4
        case .i32: 5
        }
    }

    private static func isMaplePackedResidentWeight(_ name: String) -> Bool {
        if name == "model.word_embeddings.weight" || name == "lm_head.weight"
            || name == "lm_head_flash.centroids.weight" {
            return true
        }
        let components = name.split(separator: ".")
        guard components.count == 6,
              components[0] == "model",
              components[1] == "layers",
              Int(components[2]) != nil,
              components[3] == "self_attn",
              components[5] == "weight" else {
            return false
        }
        return components[4] == "q_proj" || components[4] == "k_proj"
            || components[4] == "v_proj" || components[4] == "o_proj"
    }

    private static func validateMapleFlashHead(_ flashHead: IndexLoader.MapleFlashHeadMetadata,
                                               registry: [String: SourceTensor],
                                               arch: ArchInfo) throws {
        guard arch.family == .maple,
              flashHead.nClusters > 0, flashHead.clusterSize > 0,
              flashHead.nClusters <= Int.max / flashHead.clusterSize,
              flashHead.nClusters * flashHead.clusterSize == arch.vocabSize,
              flashHead.nProbes > 0, flashHead.nProbes <= flashHead.nClusters,
              flashHead.groupSize == 64, flashHead.bits == 4,
              flashHead.headGroupSize == 64, flashHead.headBits == 4,
              flashHead.scaledCentroids,
              flashHead.forceTokens.allSatisfy({ $0 >= 0 && $0 < arch.vocabSize }) else {
            throw RepackError.configurationInvalid(detail: "Maple FlashHead metadata is invalid")
        }
        let packedColumns = arch.hiddenSize * flashHead.bits / 32
        let groups = arch.hiddenSize / flashHead.groupSize
        let expectedCentroids = [UInt64(flashHead.nClusters), UInt64(packedColumns)]
        let expectedParameters = [UInt64(flashHead.nClusters), UInt64(groups)]
        let expectedMap = [UInt64(flashHead.nClusters), UInt64(flashHead.clusterSize)]

        func require(_ name: String, dtype: SourceTensor.Dtype,
                     shape: [UInt64], elementBytes: UInt64) throws {
            guard let tensor = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            guard tensor.dtype == dtype, tensor.shape == shape else {
                throw RepackError.shapeMismatch(
                    name: name,
                    detail: "expected \(dtype) shape \(shape), got \(tensor.dtype) \(tensor.shape)")
            }
            let elements = try shape.reduce(UInt64(1)) { partial, dimension in
                try checkedProduct(partial, dimension,
                                   detail: "Maple FlashHead shape product overflows UInt64")
            }
            let bytes = try checkedProduct(elements, elementBytes,
                                           detail: "Maple FlashHead byte count overflows UInt64")
            guard tensor.sizeBytes == bytes else {
                throw RepackError.shapeMismatch(
                    name: name,
                    detail: "shape requires \(bytes) bytes, got \(tensor.sizeBytes)")
            }
        }

        try require("lm_head_flash.centroids.weight", dtype: .u32,
                    shape: expectedCentroids, elementBytes: 4)
        try require("lm_head_flash.centroids.scales", dtype: .bf16,
                    shape: expectedParameters, elementBytes: 2)
        try require("lm_head_flash.centroids.biases", dtype: .bf16,
                    shape: expectedParameters, elementBytes: 2)
        try require("lm_head_flash.token_map", dtype: .i32,
                    shape: expectedMap, elementBytes: 4)
    }

    private static func roundUpToPage(_ v: UInt64) -> UInt64 {
        let p = Layout.pageBytes
        return ((v + p - 1) / p) * p
    }

    private static func roundUpToPageChecked(_ value: UInt64,
                                             detail: String) throws -> UInt64 {
        let adjusted = try checkedAdd(value, Layout.pageBytes - 1, detail: detail)
        return try checkedProduct(adjusted / Layout.pageBytes, Layout.pageBytes,
                                  detail: detail)
    }

    private static func padTo4(_ s: [UInt64]) -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(4)
        for v in s.prefix(4) { out.append(UInt32(v)) }
        while out.count < 4 { out.append(0) }
        return out
    }

    /// Logical shape of a packed quantized tensor whose source is `[D0,..,Dn-1, Dn/factor]`.
    private static func logicalShape(forPackedSource source: [UInt64], bits: Int) -> [UInt64] {
        let factor = UInt64(32 / bits)
        guard !source.isEmpty else { return source }
        var out = source
        out[out.count - 1] = source[source.count - 1] * factor
        return out
    }

    private static func mapleLogicalShape(forPackedSource source: [UInt64],
                                          name: String) throws -> [UInt64] {
        guard let packedInput = source.last, packedInput.isMultiple(of: 4) else {
            throw RepackError.shapeMismatch(
                name: name,
                detail: "INT2 packed input dimension must be a nonempty multiple of four U32 values")
        }
        var logical = source
        logical[logical.count - 1] = try checkedProduct(
            packedInput, 16,
            detail: "Maple logical input dimension overflows UInt64")
        return logical
    }

    private static func mapleShape4(_ shape: [UInt64], name: String) throws -> [UInt32] {
        guard shape.count <= 4 else {
            throw RepackError.shapeMismatch(name: name,
                                            detail: "Maple logical rank exceeds four")
        }
        var result = try shape.map {
            guard let value = UInt32(exactly: $0) else {
                throw RepackError.shapeMismatch(
                    name: name,
                    detail: "Maple logical dimension is not representable as UInt32")
            }
            return value
        }
        while result.count < 4 { result.append(0) }
        return result
    }

    private static func validateMapleStorage(_ tensor: SourceTensor,
                                             elementBytes: UInt64) throws {
        let elements = try tensor.shape.reduce(UInt64(1)) { partial, dimension in
            try checkedProduct(partial, dimension,
                               detail: "Maple tensor shape product overflows UInt64")
        }
        let expectedSize = try checkedProduct(elements, elementBytes,
                                              detail: "Maple tensor byte size overflows UInt64")
        guard tensor.sizeBytes == expectedSize else {
            throw RepackError.shapeMismatch(
                name: tensor.name,
                detail: "shape requires \(expectedSize) bytes, got \(tensor.sizeBytes)")
        }
    }

    private static func exactInt(_ value: UInt64, name: String, detail: String) throws -> Int {
        guard let exact = Int(exactly: value) else {
            throw RepackError.shapeMismatch(name: name, detail: detail)
        }
        return exact
    }

    private static func exactUInt64(_ value: Int, name: String, detail: String) throws -> UInt64 {
        guard let exact = UInt64(exactly: value) else {
            throw RepackError.shapeMismatch(name: name, detail: detail)
        }
        return exact
    }

    private static func checkedAdd(_ lhs: UInt64, _ rhs: UInt64,
                                   detail: String) throws -> UInt64 {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else { throw RepackError.configurationInvalid(detail: detail) }
        return sum
    }

    private static func checkedProduct(_ lhs: UInt64, _ rhs: UInt64,
                                       detail: String) throws -> UInt64 {
        let (product, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw RepackError.configurationInvalid(detail: detail) }
        return product
    }

    /// Stable order for the resident LM tensor list. Embedding first, then
    /// per-layer groups in layer index order, then the final norm (and, for
    /// families with an untied head, `lm_head` last).
    private static func lmResidentOrdering(family: RepackModelFamily)
        -> (String, String) -> Bool {
        // Compute a sort key per name; we order by (group rank, layer, slot rank, name).
        // Group 2 holds top-level tensors between the layers and the final
        // norm (DeepSeek V4's `model.hc_head.*` stream collapse lands there).
        func key(_ n: String) -> (Int, Int, Int, String) {
            switch family {
            case .gemma4, .qwen36, .qwen38:
                if n == "language_model.model.embed_tokens.weight" { return (0, 0, 0, n) }
                if n == "language_model.model.norm.weight"          { return (3, 0, 0, n) }
                if n == "language_model.lm_head.weight"             { return (4, 0, 0, n) }
            case .deepseekV4Flash:
                if n == "model.embed_tokens.weight" { return (0, 0, 0, n) }
                if n == "model.norm.weight"          { return (3, 0, 0, n) }
                if n == "lm_head.weight"             { return (4, 0, 0, n) }
            case .maple:
                if n == "model.word_embeddings.weight" { return (0, 0, 0, n) }
                if n == "model.norm.weight"             { return (3, 0, 0, n) }
                if n == "lm_head.weight"                { return (4, 0, 0, n) }
            case .inklingSmall:
                if n == "model.llm.embed.weight"      { return (0, 0, 0, n) }
                if n == "model.llm.embed_norm.weight" { return (0, 0, 1, n) }
                if n == "model.llm.norm.weight"       { return (3, 0, 0, n) }
                if n == "model.llm.unembed.weight"    { return (4, 0, 0, n) }
            }
            if let li = layerIndex(in: n) {
                let slot: Int
                switch family {
                case .gemma4:          slot = slotRank(in: n)
                case .qwen36:          slot = qwenSlotRank(in: n)
                case .qwen38:          slot = qwen38SlotRank(in: n)
                case .deepseekV4Flash: slot = deepseekV4SlotRank(in: n)
                case .inklingSmall:    slot = inklingSlotRank(in: n)
                case .maple:           slot = mapleSlotRank(in: n)
                }
                return (1, li, slot, n)
            }
            return (2, 0, 0, n)
        }
        return { a, b in
            let ka = key(a), kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    private static func mapleSlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight") { return 0 }
        if n.contains(".self_attn.k_proj.weight") { return 1 }
        if n.contains(".self_attn.v_proj.weight") { return 2 }
        if n.contains(".self_attn.o_proj.weight") { return 3 }
        if n.contains(".self_attn.q_norm.weight") { return 4 }
        if n.contains(".self_attn.k_norm.weight") { return 5 }
        if n.contains(".mlp.gate.weight") { return 6 }
        if n.hasSuffix(".input_layernorm.weight") { return 7 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 8 }
        return 99
    }

    /// Within-layer slot order for Inkling: the attention bundle (QKV/O, the
    /// relative-position projection and its profile bank, the per-head norms,
    /// and the K/V short convs), then the router, then the shared experts,
    /// then the dense FFN carried by layers 0–1, then the block short convs
    /// and the two layer norms.
    ///
    /// Ordered before `.mlp.gate_proj` so the dense-FFN checks cannot swallow
    /// the shared-expert tensors, which share the `.mlp.` segment.
    private static func inklingSlotRank(in n: String) -> Int {
        if n.contains(".attn.wq_du.weight")                 { return 0 }
        if n.contains(".attn.wk_dv.weight")                 { return 1 }
        if n.contains(".attn.wv_dv.weight")                 { return 2 }
        if n.contains(".attn.wo_ud.weight")                 { return 3 }
        if n.contains(".attn.wr_du.weight")                 { return 4 }
        if n.hasSuffix(".attn.rel_logits_proj.proj")        { return 5 }
        if n.contains(".attn.q_norm.weight")                { return 6 }
        if n.contains(".attn.k_norm.weight")                { return 7 }
        if n.contains(".attn.k_sconv.weight")               { return 8 }
        if n.contains(".attn.v_sconv.weight")               { return 9 }
        if n.contains(".mlp.gate.weight")                   { return 10 }
        if n.hasSuffix(".mlp.gate.bias")                    { return 11 }
        if n.hasSuffix(".mlp.gate.global_scale")            { return 12 }
        if n.contains(".mlp.shared_experts.gate_proj.weight") { return 13 }
        if n.contains(".mlp.shared_experts.up_proj.weight")   { return 14 }
        if n.contains(".mlp.shared_experts.down_proj.weight") { return 15 }
        if n.contains(".mlp.gate_proj.weight")              { return 16 }
        if n.contains(".mlp.up_proj.weight")                { return 17 }
        if n.contains(".mlp.down_proj.weight")              { return 18 }
        if n.hasSuffix(".mlp.global_scale")                 { return 19 }
        if n.contains(".attn_sconv.weight")                 { return 20 }
        if n.contains(".mlp_sconv.weight")                  { return 21 }
        if n.hasSuffix(".attn_norm.weight")                 { return 22 }
        if n.hasSuffix(".mlp_norm.weight")                  { return 23 }
        return 99
    }

    /// Within-layer slot order for the Qwen 3.6 family: full-attention
    /// projections/norms, then the gated-DeltaNet linear-attention bundle,
    /// then router, shared-expert gate and MLP, then the two layer norms.
    private static func qwenSlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight")   { return 0 }
        if n.contains(".self_attn.k_proj.weight")   { return 1 }
        if n.contains(".self_attn.v_proj.weight")   { return 2 }
        if n.contains(".self_attn.o_proj.weight")   { return 3 }
        if n.contains(".self_attn.q_norm.weight")   { return 4 }
        if n.contains(".self_attn.k_norm.weight")   { return 5 }
        if n.contains(".linear_attn.in_proj_qkv.weight") { return 6 }
        if n.contains(".linear_attn.in_proj_z.weight")   { return 7 }
        if n.contains(".linear_attn.in_proj_a.weight")   { return 8 }
        if n.contains(".linear_attn.in_proj_b.weight")   { return 9 }
        if n.contains(".linear_attn.conv1d.weight")      { return 10 }
        if n.hasSuffix(".linear_attn.A_log")             { return 11 }
        if n.hasSuffix(".linear_attn.dt_bias")           { return 12 }
        if n.contains(".linear_attn.norm.weight")        { return 13 }
        if n.contains(".linear_attn.out_proj.weight")    { return 14 }
        if n.contains(".mlp.gate.weight")                { return 15 }
        if n.contains(".mlp.shared_expert_gate.weight")  { return 16 }
        if n.contains(".mlp.shared_expert.gate_proj.weight") { return 17 }
        if n.contains(".mlp.shared_expert.up_proj.weight")   { return 18 }
        if n.contains(".mlp.shared_expert.down_proj.weight") { return 19 }
        if n.hasSuffix(".input_layernorm.weight")        { return 20 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 21 }
        return 100
    }

    /// Within-layer slot order for the dense Qwen 3.8 family: the Qwen 3.6
    /// attention and gated-DeltaNet ranks unchanged, then the layer's own
    /// SwiGLU MLP where the router/shared-expert bundle would sit, then the
    /// two layer norms.
    private static func qwen38SlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight")   { return 0 }
        if n.contains(".self_attn.k_proj.weight")   { return 1 }
        if n.contains(".self_attn.v_proj.weight")   { return 2 }
        if n.contains(".self_attn.o_proj.weight")   { return 3 }
        if n.contains(".self_attn.q_norm.weight")   { return 4 }
        if n.contains(".self_attn.k_norm.weight")   { return 5 }
        if n.contains(".linear_attn.in_proj_qkv.weight") { return 6 }
        if n.contains(".linear_attn.in_proj_z.weight")   { return 7 }
        if n.contains(".linear_attn.in_proj_a.weight")   { return 8 }
        if n.contains(".linear_attn.in_proj_b.weight")   { return 9 }
        if n.contains(".linear_attn.conv1d.weight")      { return 10 }
        if n.hasSuffix(".linear_attn.A_log")             { return 11 }
        if n.hasSuffix(".linear_attn.dt_bias")           { return 12 }
        if n.contains(".linear_attn.norm.weight")        { return 13 }
        if n.contains(".linear_attn.out_proj.weight")    { return 14 }
        if n.contains(".mlp.gate_proj.weight")           { return 15 }
        if n.contains(".mlp.up_proj.weight")             { return 16 }
        if n.contains(".mlp.down_proj.weight")           { return 17 }
        if n.hasSuffix(".input_layernorm.weight")        { return 18 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 19 }
        return 100
    }

    /// Within-layer slot order for the DeepSeek-V4-Flash family: the low-rank
    /// attention path in pipeline order, then the CSA/HCA compressor and its
    /// indexer, then router (with hash table / correction bias), shared
    /// expert, layer norms, and the two hyper-connection mix sites. Names
    /// follow the mlx-community conversion (`attn.wq_a`, `attn.indexer.*`,
    /// `ffn.gate`, `ffn.shared_experts`). Indexer patterns are matched
    /// before the compressor's so the shared component names (`wkv`,
    /// `wgate`, `norm`, `ape`) cannot cross-match. Unquantized tensors
    /// (norms, sinks, position biases, hc mixes, e_score_correction_bias,
    /// tid2eid) carry no `.scales`/`.biases` companions and flow through
    /// the planner's non-U32/non-`.weight` branch.
    private static func deepseekV4SlotRank(in n: String) -> Int {
        if n.contains(".attn.wq_a.weight")                      { return 0 }
        if n.contains(".attn.q_norm.weight")                    { return 1 }
        if n.contains(".attn.wq_b.weight")                      { return 2 }
        if n.contains(".attn.wkv.weight")                       { return 3 }
        if n.contains(".attn.kv_norm.weight")                   { return 4 }
        if n.hasSuffix(".attn.attn_sink")                       { return 5 }
        if n.contains(".attn.wo_a.weight")                      { return 6 }
        if n.contains(".attn.wo_b.weight")                      { return 7 }
        if n.contains(".attn.indexer.compressor.wkv.weight")    { return 12 }
        if n.contains(".attn.indexer.compressor.wgate.weight")  { return 13 }
        if n.contains(".attn.indexer.compressor.norm.weight")   { return 14 }
        if n.hasSuffix(".attn.indexer.compressor.ape")          { return 15 }
        if n.contains(".attn.indexer.wq_b.weight")              { return 16 }
        if n.contains(".attn.indexer.weights_proj.weight")      { return 17 }
        if n.contains(".attn.compressor.wkv.weight")            { return 8 }
        if n.contains(".attn.compressor.wgate.weight")          { return 9 }
        if n.contains(".attn.compressor.norm.weight")           { return 10 }
        if n.hasSuffix(".attn.compressor.ape")                  { return 11 }
        if n.contains(".ffn.gate.weight")                       { return 18 }
        if n.hasSuffix(".ffn.gate.e_score_correction_bias")     { return 19 }
        // tid2eid is an I64 lookup table; it rides the resident file as raw
        // I64 bytes (no `.weight` suffix, hence no quant companions) and the
        // runtime's CPU-side hash lookup reads it dtype-aware.
        if n.hasSuffix(".ffn.gate.tid2eid")                     { return 20 }
        if n.contains(".ffn.shared_experts.gate_proj.weight")   { return 21 }
        if n.contains(".ffn.shared_experts.up_proj.weight")     { return 22 }
        if n.contains(".ffn.shared_experts.down_proj.weight")   { return 23 }
        if n.hasSuffix(".attn_norm.weight")                     { return 24 }
        if n.hasSuffix(".ffn_norm.weight")                      { return 25 }
        if n.hasSuffix(".attn_hc.fn")                           { return 26 }
        if n.hasSuffix(".attn_hc.base")                         { return 27 }
        if n.hasSuffix(".attn_hc.scale")                        { return 28 }
        if n.hasSuffix(".ffn_hc.fn")                            { return 29 }
        if n.hasSuffix(".ffn_hc.base")                          { return 30 }
        if n.hasSuffix(".ffn_hc.scale")                         { return 31 }
        return 100
    }

    /// Within-layer slot order. Mirrors the per-layer description in the
    /// architecture doc.
    private static func slotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight") { return 0 }
        if n.contains(".self_attn.k_proj.weight") { return 1 }
        if n.contains(".self_attn.v_proj.weight") { return 2 }
        if n.contains(".self_attn.o_proj.weight") { return 3 }
        if n.contains(".self_attn.q_norm.weight") { return 4 }
        if n.contains(".self_attn.k_norm.weight") { return 5 }
        if n.contains(".router.proj.weight")      { return 6 }
        if n.contains(".router.scale")            { return 7 }
        if n.contains(".router.per_expert_scale") { return 8 }
        if n.contains(".mlp.gate_proj.weight")    { return 9 }
        if n.contains(".mlp.up_proj.weight")      { return 10 }
        if n.contains(".mlp.down_proj.weight")    { return 11 }
        if n.hasSuffix(".input_layernorm.weight") { return 12 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 13 }
        if n.hasSuffix(".pre_feedforward_layernorm.weight") { return 14 }
        if n.hasSuffix(".pre_feedforward_layernorm_2.weight") { return 15 }
        if n.hasSuffix(".post_feedforward_layernorm.weight") { return 16 }
        if n.hasSuffix(".post_feedforward_layernorm_1.weight") { return 17 }
        if n.hasSuffix(".post_feedforward_layernorm_2.weight") { return 18 }
        if n.hasSuffix(".layer_scalar")           { return 19 }
        return 100
    }
}
