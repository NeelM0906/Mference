import Foundation

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum GTurboJSON {

    static let magic = "GTURBO"
    static let versionMajor = 1
    static let versionMinor = 0

    struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    struct QuantBitWidths {
        var embedding: Int
        var attention: Int
        var router: Int
        var sharedExpert: Int
        var routedExpert: Int
    }

    /// The W2.1b provenance an original-repo install carries. The quantizer
    /// nucleus is shared, so the qwen36 measurement covers every family that
    /// has no control of its own; a family measured against its own vendor
    /// control records that measurement instead (docs/QUANTIZER_QUALITY.md).
    static func qualityGateStamp(for family: RepackModelFamily) -> String {
        switch family {
        case .minicpm5:
            return "W2.1b-weight+kld-2026-09-10-vs-openbmb-MiniCPM5-2B-MLX"
        case .qwen36, .qwen38, .maple:
            return "W2.1b-weight+kld-2026-09-02-vs-mlx-community-qwen36"
        case .inkling2Loop:
            // Never quantized at install: its tensors are the GGUF's bytes.
            return "none-gguf-bytes-copied"
        case .minicpm5_1b:
            // No original-repo source: installed only from the vendor's MLX
            // conversion, whose quantized bytes are copied unchanged.
            return "none-mlx-bytes-copied"
        }
    }

    static func encodeManifest(plan: RepackPlan,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        let arch = plan.arch
        var archDict: [String: Any] = [
            "hiddenSize": arch.hiddenSize,
            "ffnIntermediate": arch.intermediateSize,
            "moeIntermediateSize": arch.moeIntermediateSize,
            "numHeads": arch.numHeads,
            "numKVHeads": arch.numKVHeads,
            "numFullKVHeads": arch.numFullKVHeads,
            "headDim": arch.headDim,
            "fullHeadDim": arch.fullHeadDim,
            "vocabSize": arch.vocabSize,
            "slidingWindow": arch.slidingWindow,
            "finalLogitSoftcap": arch.finalLogitSoftcap,
            "ropeTheta": arch.ropeTheta,
            "fullRopeTheta": arch.fullRopeTheta,
            "partialRotaryFactor": arch.partialRotaryFactor,
            "numLayers": arch.numLayers,
            "numExperts": arch.numExperts,
            "topKExperts": arch.topKExperts,
            "tieWordEmbeddings": arch.tieWordEmbeddings,
            "attentionKEqV": arch.attentionKEqV,
            "hiddenActivation": arch.hiddenActivation,
            "fullAttentionLayerMask": arch.fullAttentionLayerMask.map { Int($0) }
        ]
        // Family extension fields, written for every family.
        archDict["family"] = arch.family.rawValue
        archDict["attnOutputGate"] = arch.attnOutputGate
        archDict["attentionScale"] = arch.attentionScale
        archDict["embeddingScaledBySqrtHidden"] = arch.embeddingScaledBySqrtHidden
        archDict["routerScaled"] = arch.routerScaled
        archDict["ffnSandwichNorms"] = arch.ffnSandwichNorms
        archDict["sharedExpertGated"] = arch.sharedExpertGated
        archDict["ropeNeoxSubdim"] = arch.ropeNeoxSubdim
        archDict["linearNumKHeads"] = arch.linearNumKHeads
        archDict["linearNumVHeads"] = arch.linearNumVHeads
        archDict["linearKeyHeadDim"] = arch.linearKeyHeadDim
        archDict["linearValueHeadDim"] = arch.linearValueHeadDim
        archDict["linearConvKernelSize"] = arch.linearConvKernelSize
        // Qwen 3.8 is dense: the reader validates numSharedExperts against 0
        // (absence would default to 1) and numDenseLayers/denseIntermediateSize
        // against the full-depth dense FFN, so all three must be written.
        if arch.family == .qwen38 {
            archDict["numSharedExperts"] = arch.numSharedExperts
            archDict["numDenseLayers"] = arch.numDenseLayers
            archDict["denseIntermediateSize"] = arch.denseIntermediateSize
        }
        // MiniCPM5: the same dense trio, plus the one axis it is the first
        // family to set away from the default. Emitted for this family only,
        // so every other manifest stays byte-identical.
        if arch.family.isMiniCPM5 {
            archDict["numSharedExperts"] = arch.numSharedExperts
            archDict["numDenseLayers"] = arch.numDenseLayers
            archDict["denseIntermediateSize"] = arch.denseIntermediateSize
            archDict["qkNorm"] = arch.qkNorm
        }
        if arch.family == .inkling2Loop {
            try addInkling2ArchFields(&archDict, arch: arch)
        }
        if arch.family == .maple {
            archDict["routerScoringFunc"] = arch.routerScoringFunc
            archDict["routedScalingFactor"] = arch.routedScalingFactor
            archDict["swigluLimit"] = arch.swigluLimit
            archDict["numSharedExperts"] = arch.numSharedExperts
            archDict["numDenseLayers"] = arch.numDenseLayers
            archDict["routerNormAfterTopK"] = arch.routerNormAfterTopK
        }
        let quantBits = [
            "embedding": bitWidths.embedding,
            "attention": bitWidths.attention,
            "router": bitWidths.router,
            "sharedExpert": bitWidths.sharedExpert,
            "routedExpert": bitWidths.routedExpert,
        ]
        var quantDict: [String: Any] = [:]
        for (slot, bits) in quantBits {
            quantDict[slot] = [
                "weightBits": bits,
                "scheme": plan.baseMode,
                "scaleType": "BF16",
                "biasType": "BF16",
                "groupSize": plan.baseGroupSize
            ]
        }
        // Dense family: there is no router, shared expert or routed expert to
        // quantize. Mark the slots absent (Maple's sharedExpert convention);
        // embedding/attention keep the affine INT4 entries from the loop.
        if arch.family == .qwen38 || arch.family.isMiniCPM5 {
            let absent: [String: Any] = [
                "weightBits": 0,
                "scheme": "none",
                "scaleType": "none",
                "biasType": "none",
                "groupSize": 0,
            ]
            quantDict["router"] = absent
            quantDict["sharedExpert"] = absent
            quantDict["routedExpert"] = absent
        }
        if arch.family == .maple {
            let affineInt4: [String: Any] = [
                "weightBits": 4,
                "scheme": "affine",
                "scaleType": "BF16",
                "biasType": "BF16",
                "groupSize": 64,
            ]
            quantDict["embedding"] = affineInt4
            quantDict["attention"] = affineInt4
            quantDict["router"] = [
                "weightBits": 16,
                "scheme": "unquantized",
                "scaleType": "none",
                "biasType": "none",
                "groupSize": 0,
            ]
            quantDict["sharedExpert"] = [
                "weightBits": 0,
                "scheme": "none",
                "scaleType": "none",
                "biasType": "none",
                "groupSize": 0,
            ]
            quantDict["routedExpert"] = [
                "weightBits": 2,
                "scheme": "affine",
                "scaleType": "BF16",
                "biasType": "BF16",
                "groupSize": 64,
            ]
        }

        if arch.family == .inkling2Loop {
            quantDict = inkling2QuantSlots
        }

        var filesDict: [String: Any] = [:]
        for (path, info) in files {
            filesDict[path] = ["size": info.size, "sha256": info.sha256]
        }

        var manifest: [String: Any] = [
            "magic": GTurboJSON.magic,
            "versionMajor": GTurboJSON.versionMajor,
            "versionMinor": GTurboJSON.versionMinor,
            "flags": [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false
            ],
            "modelID": modelID,
            "sourceSnapshotHash": sourceSnapshotHash,
            "arch": archDict,
            "quant": quantDict,
            "files": filesDict,
            "expertsPerLayer": expertsPerLayer,
            "numLayers": numLayers,
            "expertStride": expertStride,
            "bitWidthOverridesHonored": plan.bitsOverrideCount
        ]
        // --- Additive blocks. Every family that predates them emits none of
        // these keys, so existing manifests stay byte-identical.
        if plan.quantizedAtInstall {
            var quantized: [String: Any] = [
                "scheme": "affine",
                // The base width. A mixed-width install records its overrides
                // below rather than changing this, so the field keeps meaning
                // the same thing it always did.
                "weightBits": 4,
                "groupSize": StreamingInt4Quantizer.groupSize,
                "sourceDtype": "BF16",
                // Gate W2.1a is enforced in CI. W2.1b now records BOTH halves;
                // the string names the control, the method and the date so an
                // install carries its own provenance rather than a bare "pass".
                // Both halves are measured on the shared
                // `Int4AffineEncoder.encodeGroup` nucleus every original-repo
                // family funnels through, which is why the stamp is not
                // per-family. Method and numbers: docs/QUANTIZER_QUALITY.md.
                "parityGate": "W2.1a-bit-parity",
                "qualityGate": qualityGateStamp(for: arch.family),
            ]
            if plan.bitsOverrideCount > 0 {
                quantized["overrideWeightBits"] = 8
                quantized["overriddenTensorCount"] = plan.bitsOverrideCount
            }
            manifest["quantizedAtInstall"] = quantized
        }
        if !plan.sidecarOutcomes.isEmpty {
            var sidecars: [String: Any] = [:]
            for outcome in plan.sidecarOutcomes {
                sidecars[outcome.group] = [
                    "carried": outcome.carried,
                    "tensorCount": outcome.tensorCount,
                ]
            }
            manifest["sidecars"] = sidecars
        }
        if !plan.auxiliaryExpertPools.isEmpty {
            manifest["auxiliaryExpertPools"] = plan.auxiliaryExpertPools.map { pool in
                [
                    "name": pool.name,
                    "directory": pool.directoryName,
                    "expertsPerLayer": pool.layers.first(where: { $0.expertsPerLayer > 0 })?
                        .expertsPerLayer ?? 0,
                    "expertStride": pool.layers.first(where: { $0.expertsPerLayer > 0 })?
                        .expertStride ?? 0,
                    "layers": pool.layers.filter { $0.expertsPerLayer > 0 }.map {
                        [
                            "layer": $0.layerIndex,
                            "file": ($0.path as NSString).lastPathComponent,
                        ] as [String: Any]
                    },
                ] as [String: Any]
            }
        }
        if let flashHead = plan.flashHead {
            manifest["flashHead"] = [
                "nClusters": flashHead.nClusters,
                "clusterSize": flashHead.clusterSize,
                "nProbes": flashHead.nProbes,
                "groupSize": flashHead.groupSize,
                "bits": flashHead.bits,
                "headGroupSize": flashHead.headGroupSize,
                "headBits": flashHead.headBits,
                "scaledCentroids": flashHead.scaledCentroids,
                "forceTokens": flashHead.forceTokens,
            ]
        }
        return try JSONSerialization.data(withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        let arch = plan.arch
        var layersArr: [[String: Any]] = []
        layersArr.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [[String: Any]] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let base = UInt64(e) * lp.expertStride
                var tensors: [String: Any] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    var t: [String: Any] = [
                        "offset": slice.offsetInExpertBlob,
                        "size":   slice.sizeInExpertBlob,
                        "dtype":  slice.dtype == 0 ? "U32" : "BF16",
                        "shape":  slice.logicalShape.map { Int($0) }
                    ]
                    if let bits = slice.bitsForWeights { t["bits"] = bits }
                    tensors[key] = t
                }
                let expertEntry: [String: Any] = [
                    "expert": e,
                    "offset": base,
                    "size":   lp.expertStride,
                    "tensors": tensors
                ]
                experts.append(expertEntry)
            }
            layersArr.append([
                "layer": lp.layerIndex,
                "file":  layerFile,
                "experts": experts
            ])
        }
        let obj: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": arch.numLayers,
            // Skip layers that carry no routed experts (every layer of a dense
            // family). Taking `layers.first` would publish 0 here and
            // contradict the manifest, which selects the same way.
            "expertsPerLayer": plan.layers.first(where: { $0.expertsPerLayer > 0 })?
                .expertsPerLayer ?? 0,
            "layers": layersArr
        ]
        return try JSONSerialization.data(withJSONObject: obj,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
