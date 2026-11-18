import Foundation

/// An additive expert pool that is not the main routed pool: same per-expert
/// blob layout, its own directory, its own manifest block. It is deliberately
/// absent from `packed_experts/layout.json`, so the shipped layout validator
/// and the runtime's routed-expert reader see exactly what they saw before.
struct AuxiliaryExpertPoolPlan: Sendable {
    /// Manifest key and directory suffix, e.g. `mtp`.
    let name: String
    let directoryName: String
    let layers: [LayerFilePlan]
}

/// Planner for checkpoints read from the model vendor's **original BF16 repo**
/// rather than from a pre-quantized MLX conversion. `qwen36original` — the
/// same Qwen 3.6 checkpoint the `qwen36` entry installs from mlx-community —
/// is what the W2.1b quantizer-quality gate measures
/// (docs/QUANTIZER_QUALITY.md); `minicpm5` is the plain-llama second entry.
///
/// `RepackPlanner.plan` routes here on `meta.sourceIsUnquantized`, not on the
/// family: what this planner handles is a property of the *source*, and Qwen
/// 3.6 proves the point by arriving on both paths.
///
/// Two things make an original-repo source structurally different from a
/// conversion, and they are why it gets its own planner instead of more
/// branches in `RepackPlanner`:
///
/// 1. **Quantize-in-flight.** Nothing in the source is quantized. Two-dimensional
///    projection weights become INT4 affine group-64 through
///    `StreamingInt4Quantizer`; norms, 1-D vectors, conv kernels and integer
///    lookup tables stay BF16 (or their source dtype), matching what every
///    existing family's conversion already does.
/// 2. **Fused expert tensors.** `mlp.experts.gate_up_proj` holds all experts
///    with the gate and up halves concatenated, and `mlp.experts.down_proj`
///    holds all experts. Both split into the per-expert page-aligned blobs the
///    runtime's routed-expert reader already consumes, in the same
///    `gate | up | down` x `weights | scales | biases` order Qwen 3.6 emits.
///
/// # Fused-expert axis order — checked against a real vendor checkpoint
///
/// The fused-expert axis order is taken to be `[experts, 2 * moeIntermediate,
/// hidden]` for `gate_up_proj` (gate rows first, then up rows) and
/// `[experts, hidden, moeIntermediate]` for `down_proj`. That is the only
/// layout in which a pure byte-range split yields the per-expert `gate`, `up`
/// and `down` matrices the existing kernels expect.
///
/// `Qwen/Qwen3.6-35B-A3B` rev `995ad96e` declares `gate_up_proj` as
/// `[256, 1024, 2048]` and `down_proj` as `[256, 2048, 512]` with
/// `moe_intermediate_size` 512 and `hidden_size` 2048, which is exactly
/// `[E, 2 * I, H]` and `[E, H, I]`. Splitting `gate_up_proj` at the halfway row
/// also reproduces mlx-community's independently converted
/// `switch_mlp.gate_proj` / `switch_mlp.up_proj` tensors, so the gate-first
/// ordering is confirmed too, not just the shape. `planLayerFile` still fails
/// loudly on any other shape rather than producing a plausible-looking wrong
/// install.
enum OriginalRepoPlanner {

    static let textPrefix = "model.language_model."
    static let lmHeadName = "lm_head.weight"

    /// The trunk prefix the vendor's original repo uses. Qwen ships inside a
    /// multimodal container (`model.language_model.`); a plain
    /// `LlamaForCausalLM` upload uses the bare `model.`.
    static func trunkPrefix(for family: RepackModelFamily) -> String {
        family == .minicpm5 ? "model." : textPrefix
    }
    static let visionPrefix = "model.visual."
    static let mtpPrefix = "mtp."
    static let expertsSegment = ".mlp.experts."

    // MARK: - Entry point

    static func plan(meta: IndexLoader.SourceMetadata,
                     arch: ArchInfo,
                     registry: [String: SourceTensor],
                     outputDir: String,
                     sidecarPolicy: SidecarPolicy) throws -> RepackPlan {
        guard meta.sourceIsUnquantized else {
            throw RepackError.configJsonInvalid(
                path: meta.configPath,
                detail: "\(arch.family.rawValue) reached the original-repo BF16 planner, "
                    + "but this config declares a quantization block; the "
                    + "quantize-in-flight path would double-quantize already-packed "
                    + "weights")
        }
        guard !sidecarPolicy.carryVision else {
            throw RepackError.configurationInvalid(
                detail: "carrying the vision tower is not supported: the runtime is "
                    + "text-only and no .gturbo region holds a tower")
        }
        var residentNames: [String] = []
        var mtpResidentNames: [String] = []
        var visionNames: [String] = []
        var skippedMTPNames: [String] = []
        var fusedByLayer: [Int: [String: SourceTensor]] = [:]   // role -> tensor
        var fusedMTPByLayer: [Int: [String: SourceTensor]] = [:]

        for name in registry.keys.sorted() {
            guard let tensor = registry[name] else { continue }
            if name.hasPrefix(visionPrefix) {
                visionNames.append(name)
                continue
            }
            if name.hasPrefix(mtpPrefix) {
                guard sidecarPolicy.carryMTP else {
                    skippedMTPNames.append(name)
                    continue
                }
                if let role = fusedExpertRole(in: name) {
                    guard let layer = RepackPlanner.layerIndex(in: name) else {
                        throw RepackError.unknownTensorPrefix(name: name)
                    }
                    try record(role: role, tensor: tensor, layer: layer,
                               into: &fusedMTPByLayer)
                    continue
                }
                mtpResidentNames.append(name)
                continue
            }
            guard name.hasPrefix(trunkPrefix(for: arch.family)) || name == lmHeadName else {
                throw RepackError.unknownTensorPrefix(name: name)
            }
            if let role = fusedExpertRole(in: name) {
                guard let layer = RepackPlanner.layerIndex(in: name),
                      layer >= 0, layer < arch.numLayers else {
                    throw RepackError.unknownTensorPrefix(name: name)
                }
                try record(role: role, tensor: tensor, layer: layer, into: &fusedByLayer)
                continue
            }
            residentNames.append(name)
        }

        let family = arch.family
        residentNames.sort { residentOrdering($0, $1, family: family) }
        mtpResidentNames.sort()
        visionNames.sort()
        skippedMTPNames.sort()

        // Per-tensor width. The policy is consulted for resident tensors only:
        // the fused expert pools are quantized at the base width, and a rule
        // that reached into them would half-apply silently, so any rule that
        // matches an expert tensor is a configuration error rather than a
        // partially-honoured policy.
        let policy = try QuantBitPolicy.originalRepo(family: arch.family)
            .validated(for: arch.family)
        for bundle in [fusedByLayer, fusedMTPByLayer] {
            for tensors in bundle.values {
                for tensor in tensors.values where policy.overrides(tensor.name) {
                    throw RepackError.configurationInvalid(
                        detail: "quantization bit policy overrides the fused expert "
                            + "tensor \(tensor.name) to \(policy.bits(forTensorNamed: tensor.name))"
                            + " bits, but the expert pools are planned at the base "
                            + "width \(policy.defaultBits)")
                }
            }
        }

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath,
                                            names: residentNames + mtpResidentNames,
                                            registry: registry,
                                            policy: policy,
                                            family: arch.family)

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layers: [LayerFilePlan] = []
        layers.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            guard let bundle = fusedByLayer[layer] else {
                layers.append(LayerFilePlan(layerIndex: layer, path: path,
                                            expertsPerLayer: 0, expertStride: 0,
                                            subTensors: []))
                continue
            }
            layers.append(try planLayerFile(path: path, layer: layer,
                                            bundle: bundle,
                                            expertCount: arch.numExperts))
        }
        layers = paddedToWidestStride(layers)

        var auxiliaryPools: [AuxiliaryExpertPoolPlan] = []
        if !fusedMTPByLayer.isEmpty {
            let mtpDir = (outputDir as NSString).appendingPathComponent("packed_experts_mtp")
            var mtpLayers: [LayerFilePlan] = []
            for layer in fusedMTPByLayer.keys.sorted() {
                let path = (mtpDir as NSString)
                    .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
                mtpLayers.append(try planLayerFile(path: path, layer: layer,
                                                   bundle: fusedMTPByLayer[layer]!,
                                                   expertCount: arch.numExperts))
            }
            auxiliaryPools.append(AuxiliaryExpertPoolPlan(
                name: "mtp",
                directoryName: "packed_experts_mtp",
                layers: paddedToWidestStride(mtpLayers)))
        }

        let carriedMTPCount = mtpResidentNames.count
            + fusedMTPByLayer.values.reduce(0) { $0 + $1.count }
        var outcomes: [SidecarGroupOutcome] = []
        if carriedMTPCount > 0 || !skippedMTPNames.isEmpty {
            outcomes.append(SidecarGroupOutcome(
                group: "mtp",
                carried: sidecarPolicy.carryMTP,
                tensorCount: sidecarPolicy.carryMTP ? carriedMTPCount : skippedMTPNames.count))
        }
        if !visionNames.isEmpty {
            outcomes.append(SidecarGroupOutcome(group: "vision",
                                                carried: false,
                                                tensorCount: visionNames.count))
        }

        // An original-repo source declares no quantization block, so
        // `meta.bitsOverrides` is always empty here and the honest audit value
        // is what the *policy* actually produced. Counting the planned entries
        // rather than the policy's rule count makes the number directly
        // comparable to a community conversion's own `bitWidthOverridesHonored`
        // — mlx-community's Qwen 3.6 records 80, one `mlp.gate` and one
        // `mlp.shared_expert_gate` per layer.
        let bitsOverrideCount = resident.entries.filter {
            guard let spec = $0.quantSpec else { return false }
            return spec.bits != policy.defaultBits
        }.count

        return RepackPlan(
            arch: arch,
            baseMode: "affine",
            baseGroupSize: StreamingInt4Quantizer.groupSize,
            bitsOverrideCount: bitsOverrideCount,
            resident: resident,
            layers: layers,
            matchedModelID: SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex),
            excludedMultimodalTensorNames: (visionNames + skippedMTPNames).sorted(),
            flashHead: nil,
            sidecarOutcomes: outcomes,
            quantizedAtInstall: meta.sourceIsUnquantized,
            auxiliaryExpertPools: auxiliaryPools)
    }

    // MARK: - Classification

    /// `gate_up` or `down` for the two fused routed-expert tensors, else nil.
    /// The `.mlp.experts.` segment cannot match `.mlp.shared_expert.`, so the
    /// shared FFN stays resident.
    static func fusedExpertRole(in name: String) -> String? {
        guard name.contains(expertsSegment) else { return nil }
        let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        if base.hasSuffix(".gate_up_proj") { return "gate_up" }
        if base.hasSuffix(".down_proj") { return "down" }
        return nil
    }

    private static func record(role: String,
                               tensor: SourceTensor,
                               layer: Int,
                               into table: inout [Int: [String: SourceTensor]]) throws {
        var bundle = table[layer] ?? [:]
        guard bundle[role] == nil else {
            throw RepackError.configurationInvalid(
                detail: "two fused \(role) expert tensors for layer \(layer)")
        }
        bundle[role] = tensor
        table[layer] = bundle
    }

    // MARK: - Resident naming

    /// The name a resident tensor is *written under*, which is not always the
    /// name the source shipped it under.
    ///
    /// A `.gturbo` resident index is looked up by exact name
    /// (`Model.resident(name:)` is a dictionary hit or `tensorNotFound`), and
    /// each family's runner asks for one fixed spelling — for Qwen 3.6,
    /// `Model.trunkPrefix` is `language_model.model.` and the head is
    /// `language_model.lm_head.weight`. The pre-quantized path inherits those
    /// names for free because mlx-lm's conversion already uses them. A vendor's
    /// original repo does not: `Qwen/Qwen3.6-35B-A3B` ships the trunk as
    /// `model.language_model.` and the head as a top-level `lm_head.weight`.
    ///
    /// So the two orderings have to be reconciled somewhere, and the install is
    /// the right place: it happens once, it is what makes an original-repo
    /// install a drop-in replacement for a conversion of the same checkpoint,
    /// and the alternative — teaching every runtime accessor a second spelling —
    /// would spread the source's naming accident across the runtime forever.
    ///
    /// This is a *renaming*, not a remapping: `quantizer-mixture-compare.py`
    /// checks that the two installs' resident sets are identical under
    /// normalization, so every name has exactly one counterpart.
    static func residentName(for sourceName: String,
                             family: RepackModelFamily) -> String {
        switch family {
        case .minicpm5:
            // The vendor already spells the trunk `model.` and the head
            // `lm_head.weight`, which is what the runner looks up.
            return sourceName
        case .qwen36, .qwen38:
            if sourceName == lmHeadName {
                return "language_model.lm_head.weight"
            }
            if sourceName.hasPrefix(textPrefix) {
                return "language_model.model."
                    + sourceName.dropFirst(textPrefix.count)
            }
            return sourceName
        case .maple:
            // No original-repo entry exists for Maple, so no runner contract
            // has been checked. Passing the source name through unchanged
            // fails loudly at load (`tensorNotFound`) rather than quietly
            // installing something a runner cannot address.
            return sourceName
        }
    }

    // MARK: - RMSNorm weight convention

    /// `true` when this tensor is an RMSNorm gain vector.
    static func isNormWeight(_ name: String) -> Bool {
        name.contains("norm") && name.hasSuffix(".weight")
    }

    /// `true` when the stored weight must be `1 + w` rather than the vendor's
    /// bare `w`.
    ///
    /// Qwen 3.6 applies most of its RMSNorms as `(1 + w) * x̂`. mlx-lm's
    /// conversion folds that `+1` into the stored weight so the kernel can just
    /// multiply, and the Mference runtime — built against that conversion —
    /// multiplies by the stored weight directly. An install read from the
    /// vendor's original repo therefore has to fold it too. It is not a
    /// cosmetic difference: without it every norm in the model is off by one,
    /// the install still verifies and still *loads*, and the model emits
    /// confident nonsense.
    ///
    /// This is exactly the class of defect the model-level half of W2.1b exists
    /// to catch. The weight-level half cannot see it — norms are not quantized,
    /// so they are not in its sample — and neither `--verify-install` nor
    /// `ManifestReader` has any opinion about the numeric convention of a
    /// passthrough tensor.
    ///
    /// **The exception is the gated-DeltaNet block's own norm.**
    /// `linear_attn.norm` is a different module with a different convention and
    /// is stored bare on both sides. Measured against mlx-community's
    /// conversion of the same checkpoint: folding `bf16(w + 1)` reproduces the
    /// control's stored bytes **bit-exactly on all 101** folded tensors (40
    /// `input_layernorm`, 40 `post_attention_layernorm`, 10 `self_attn.q_norm`,
    /// 10 `self_attn.k_norm`, and the final `norm`), while all 30
    /// `linear_attn.norm` tensors are already byte-identical unfolded.
    static func foldsNormBias(_ sourceName: String,
                              family: RepackModelFamily) -> Bool {
        switch family {
        case .qwen36:
            return isNormWeight(sourceName)
                && !sourceName.hasSuffix(".linear_attn.norm.weight")
        case .minicpm5:
            // `LlamaRMSNorm` (transformers v5.6.2) is plain `w * x_hat`, and
            // the vendor's own MLX conversion stores the bare weights. No
            // fold. See docs/families/MINICPM5.md.
            return false
        case .qwen38, .maple:
            // No original-repo entry exists for these, so no conversion has
            // been compared and no fold can be justified. A family arriving
            // here must check its own norm convention first.
            return false
        }
    }

    // MARK: - Resident planning

    /// A source tensor becomes INT4 affine group-64 only when it is a genuine
    /// two-dimensional BF16 projection whose input dimension the group size
    /// divides. Everything else — norms, 1-D vectors (`A_log`, `dt_bias`,
    /// `layer_multipliers`), conv kernels, and the integer n-gram head tables —
    /// rides through at its source dtype, which is what every shipped family's
    /// conversion also does.
    static func quantizesResident(_ tensor: SourceTensor) -> Bool {
        guard tensor.dtype == .bf16,
              tensor.name.hasSuffix(".weight"),
              tensor.shape.count == 2,
              !isForcedBF16(tensor.name),
              let columns = tensor.shape.last,
              let rows = tensor.shape.first,
              rows > 0,
              let width = Int(exactly: columns),
              StreamingInt4Quantizer.isQuantizableRowDim(width) else {
            return false
        }
        return true
    }

    private static func isForcedBF16(_ name: String) -> Bool {
        name.contains("norm") || name.hasSuffix(".conv1d.weight")
    }

    private static func planResidentFile(path: String,
                                         names: [String],
                                         registry: [String: SourceTensor],
                                         policy: QuantBitPolicy,
                                         family: RepackModelFamily) throws
        -> ResidentFilePlan {
        // `names` are the source's own; the index is written under the names
        // the family's runner looks up. See `residentName(for:family:)`.
        let emitted = names.map { residentName(for: $0, family: family) }
        guard Set(emitted).count == emitted.count else {
            throw RepackError.configurationInvalid(
                detail: "resident renaming collided: two source tensors map to "
                    + "the same emitted name")
        }
        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(names.count)
        for name in emitted {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: name.utf8)
        }
        let rawIndex = UInt64(GTurboBinary.indexHeaderBytes
            + names.count * GTurboBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = RepackPlanner.roundUpToPage(rawIndex)

        var cursor = indexSize
        var entries: [ResidentEntry] = []
        var foldedNormCount = 0
        entries.reserveCapacity(names.count)
        for (index, name) in names.enumerated() {
            guard let tensor = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            let entryName = emitted[index]
            let shape = tensor.shape
            if quantizesResident(tensor) {
                let bits = policy.bits(forTensorNamed: entryName)
                let rows = tensor.shape[0]
                let columns = tensor.shape[1]
                let groups = columns / UInt64(StreamingInt4Quantizer.groupSize)
                // The width changes the weight-side byte count and nothing
                // else: the companion arrays are one BF16 scale and one BF16
                // bias per group at either width.
                let weightBytes = rows * columns * UInt64(bits) / 8
                let companionBytes = rows * groups
                    * UInt64(StreamingInt4Quantizer.companionBytesPerGroup)
                let transform: (RangeCopyTransform, RangeCopyTransform, RangeCopyTransform)
                switch bits {
                case 4:
                    transform = (.quantizeInt4G64(component: .weights),
                                 .quantizeInt4G64(component: .scales),
                                 .quantizeInt4G64(component: .biases))
                case 8:
                    transform = (.quantizeInt8G64(component: .weights),
                                 .quantizeInt8G64(component: .scales),
                                 .quantizeInt8G64(component: .biases))
                default:
                    // `QuantBitPolicy.validated(for:)` runs before planning, so
                    // this is unreachable; it is here so a future width cannot
                    // be added to the policy without adding its transform.
                    throw RepackError.configurationInvalid(
                        detail: "no streaming quantizer for \(bits)-bit \(name)")
                }
                let weightOffset = cursor
                let scaleOffset = weightOffset + weightBytes
                let biasOffset = scaleOffset + companionBytes
                cursor = biasOffset + companionBytes
                entries.append(ResidentEntry(
                    name: entryName,
                    dtype: 0,
                    logicalShape4: RepackPlanner.padTo4(shape),
                    fileOffset: weightOffset,
                    sizeBytes: weightBytes,
                    scaleOffset: scaleOffset,
                    scaleSize: companionBytes,
                    biasOffset: biasOffset,
                    biasSize: companionBytes,
                    quantSpec: QuantSpec(bits: bits,
                                         groupSize: StreamingInt4Quantizer.groupSize),
                    sourceWeight: tensor,
                    sourceScales: tensor,
                    sourceBiases: tensor,
                    weightTransform: transform.0,
                    scaleTransform: transform.1,
                    biasTransform: transform.2))
            } else {
                // Passthrough — but not always verbatim. An RMSNorm gain that
                // the runtime expects as `1 + w` is folded in flight; see
                // `foldsNormBias`. Everything else rides through untouched.
                let fold = foldsNormBias(name, family: family)
                if fold {
                    guard tensor.dtype == .bf16 else {
                        throw RepackError.dtypeMismatch(
                            name: name,
                            detail: "the RMSNorm +1 fold is defined for BF16 "
                                + "weights, got \(tensor.dtype)")
                    }
                    foldedNormCount += 1
                }
                let offset = cursor
                cursor += tensor.sizeBytes
                entries.append(ResidentEntry(
                    name: entryName,
                    dtype: try dtypeCode(tensor.dtype),
                    logicalShape4: RepackPlanner.padTo4(shape),
                    fileOffset: offset,
                    sizeBytes: tensor.sizeBytes,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: tensor,
                    sourceScales: nil, sourceBiases: nil,
                    weightTransform: fold ? .addOneBF16 : .identity))
            }
        }
        if foldedNormCount > 0 {
            FileHandle.standardError.write(Data(
                "[repack] folded +1 into \(foldedNormCount) RMSNorm weights\n".utf8))
        }
        return ResidentFilePlan(path: path,
                                entries: entries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: cursor - indexSize)
    }

    // MARK: - Fused-expert planning

    private static func planLayerFile(path: String,
                                      layer: Int,
                                      bundle: [String: SourceTensor],
                                      expertCount: Int) throws -> LayerFilePlan {
        guard let gateUp = bundle["gate_up"], let down = bundle["down"] else {
            throw RepackError.configurationInvalid(
                detail: "layer \(layer) fused-expert bundle incomplete: "
                    + "\(bundle.keys.sorted())")
        }
        let experts = UInt64(expertCount)
        guard gateUp.dtype == .bf16, down.dtype == .bf16 else {
            throw RepackError.dtypeMismatch(
                name: gateUp.name,
                detail: "expected BF16 fused expert tensors, got "
                    + "\(gateUp.dtype)/\(down.dtype)")
        }
        guard gateUp.shape.count == 3, down.shape.count == 3,
              gateUp.shape[0] == experts, down.shape[0] == experts else {
            throw RepackError.shapeMismatch(
                name: gateUp.name,
                detail: "expected rank-3 [\(expertCount), ...] fused expert tensors, "
                    + "got \(gateUp.shape) and \(down.shape)")
        }
        let hidden = down.shape[1]
        let intermediate = down.shape[2]
        guard gateUp.shape[1] == 2 * intermediate, gateUp.shape[2] == hidden else {
            throw RepackError.shapeMismatch(
                name: gateUp.name,
                detail: "gate_up_proj \(gateUp.shape) is not "
                    + "[\(expertCount), 2 * \(intermediate), \(hidden)]; the fused axis "
                    + "order assumed by the original-repo planner does not hold for this "
                    + "checkpoint")
        }
        let group = UInt64(StreamingInt4Quantizer.groupSize)
        guard hidden % group == 0, intermediate % group == 0 else {
            throw RepackError.shapeMismatch(
                name: gateUp.name,
                detail: "expert dims \(hidden)x\(intermediate) are not multiples of "
                    + "\(group); they cannot be group-64 quantized")
        }
        let halfBytes = intermediate * hidden * 2
        let gateUpStride = 2 * halfBytes
        let downBytes = hidden * intermediate * 2
        guard gateUp.sizeBytes == experts * gateUpStride,
              down.sizeBytes == experts * downBytes else {
            throw RepackError.shapeMismatch(
                name: gateUp.name,
                detail: "fused expert byte counts do not match their shapes")
        }

        var subTensors: [PerExpertTensorSlice] = []
        subTensors.reserveCapacity(9)
        var cursor: UInt64 = 0
        let roles: [(role: String,
                     tensor: SourceTensor,
                     rows: UInt64,
                     columns: UInt64,
                     stride: UInt64,
                     sliceOffset: UInt64,
                     sourceBytes: UInt64)] = [
            ("gate", gateUp, intermediate, hidden, gateUpStride, 0, halfBytes),
            ("up", gateUp, intermediate, hidden, gateUpStride, halfBytes, halfBytes),
            ("down", down, hidden, intermediate, downBytes, 0, downBytes),
        ]
        for role in roles {
            let groups = role.columns / group
            let weightBytes = role.rows * role.columns / 2
            let companionBytes = role.rows * groups
                * UInt64(StreamingInt4Quantizer.companionBytesPerGroup)
            let components: [(component: StreamingInt4Quantizer.Component,
                              label: String,
                              dtype: UInt8,
                              shape: [UInt64],
                              bytes: UInt64,
                              bits: Int?)] = [
                (.weights, "weights", 0, [role.rows, role.columns], weightBytes, 4),
                (.scales, "scales", 1, [role.rows, groups], companionBytes, nil),
                (.biases, "biases", 1, [role.rows, groups], companionBytes, nil),
            ]
            for component in components {
                subTensors.append(PerExpertTensorSlice(
                    role: role.role,
                    component: component.label,
                    dtype: component.dtype,
                    logicalShape: component.shape,
                    offsetInExpertBlob: cursor,
                    sizeInExpertBlob: component.bytes,
                    sourceOffsetPerExpert: role.sourceBytes,
                    sourceTensor: role.tensor,
                    bitsForWeights: component.bits,
                    transform: .quantizeInt4G64(component: component.component),
                    sourceStridePerExpert: role.stride,
                    sourceSliceOffset: role.sliceOffset))
                cursor += component.bytes
            }
        }
        return LayerFilePlan(layerIndex: layer,
                             path: path,
                             expertsPerLayer: expertCount,
                             expertStride: RepackPlanner.roundUpToPage(cursor),
                             subTensors: subTensors)
    }

    /// The `.gturbo` format keeps one `expertStride` across a pool, so pad
    /// every populated layer to the widest one.
    private static func paddedToWidestStride(_ layers: [LayerFilePlan]) -> [LayerFilePlan] {
        let widest = layers.map(\.expertStride).max() ?? 0
        return layers.map { layer in
            guard layer.expertsPerLayer > 0, layer.expertStride != widest else { return layer }
            return LayerFilePlan(layerIndex: layer.layerIndex,
                                 path: layer.path,
                                 expertsPerLayer: layer.expertsPerLayer,
                                 expertStride: widest,
                                 subTensors: layer.subTensors)
        }
    }

    // MARK: - Ordering

    private static func dtypeCode(_ dtype: SourceTensor.Dtype) throws -> UInt8 {
        switch dtype {
        case .u32: 0
        case .bf16: 1
        case .fp16: 2
        case .fp32: 3
        case .i64: 4
        case .i32: 5
        }
    }

    /// Deterministic resident order: embedding, then per-layer groups in
    /// pipeline order, then the final norm and the untied head.
    static func residentOrdering(_ lhs: String, _ rhs: String,
                                 family: RepackModelFamily) -> Bool {
        let a = orderKey(lhs, family: family), b = orderKey(rhs, family: family)
        if a.0 != b.0 { return a.0 < b.0 }
        if a.1 != b.1 { return a.1 < b.1 }
        if a.2 != b.2 { return a.2 < b.2 }
        return a.3 < b.3
    }

    private static func orderKey(_ name: String,
                                 family: RepackModelFamily) -> (Int, Int, Int, String) {
        let prefix = trunkPrefix(for: family)
        if name == prefix + "embed_tokens.weight" { return (0, 0, 0, name) }
        if name == prefix + "norm.weight" { return (3, 0, 0, name) }
        if name == lmHeadName { return (4, 0, 0, name) }
        if let layer = RepackPlanner.layerIndex(in: name) {
            let slot = family == .minicpm5
                ? RepackPlanner.miniCPM5SlotRank(in: name)
                : slotRank(in: name)
            return (1, layer, slot, name)
        }
        return (2, 0, 0, name)
    }

    /// Within-layer slot order, following the block's own pipeline: the
    /// attention bundle, the gated-DeltaNet bundle, the router and shared
    /// expert, then the two layer norms.
    private static func slotRank(in name: String) -> Int {
        let ranks: [(String, Int)] = [
            (".self_attn.q_proj.weight", 4),
            (".self_attn.k_proj.weight", 5),
            (".self_attn.v_proj.weight", 6),
            (".self_attn.o_proj.weight", 7),
            (".self_attn.q_norm.weight", 8),
            (".self_attn.k_norm.weight", 9),
            (".linear_attn.in_proj_qkv.weight", 13),
            (".linear_attn.in_proj_z.weight", 14),
            (".linear_attn.in_proj_a.weight", 15),
            (".linear_attn.in_proj_b.weight", 16),
            (".linear_attn.conv1d.weight", 17),
            (".linear_attn.A_log", 18),
            (".linear_attn.dt_bias", 19),
            (".linear_attn.norm.weight", 20),
            (".linear_attn.out_proj.weight", 21),
            (".mlp.gate.weight", 26),
            (".mlp.shared_expert_gate.weight", 27),
            (".mlp.shared_expert.gate_proj.weight", 28),
            (".mlp.shared_expert.up_proj.weight", 29),
            (".mlp.shared_expert.down_proj.weight", 30),
            (".input_layernorm.weight", 40),
            (".post_attention_layernorm.weight", 41),
        ]
        for (suffix, rank) in ranks where name.hasSuffix(suffix) { return rank }
        return 100
    }
}
