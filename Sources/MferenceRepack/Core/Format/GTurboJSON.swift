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
        case .gemma4, .qwen36, .qwen38, .deepseekV4Flash, .inklingSmall, .maple,
             .qwen38flashnext, .glm53Flash:
            return "W2.1b-weight+kld-2026-09-02-vs-mlx-community-qwen36"
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
        // Family extension fields. Gemma manifests omit them (byte-identical
        // to the pre-family format); the reader treats absence as the Gemma
        // defaults.
        if arch.family != .gemma4 {
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
        }
        // DeepSeek-V4 extension fields. Gated on the family so existing Qwen
        // manifests stay byte-identical; the reader treats absence as the
        // zeroed/"softmax"/1.0/0.0 defaults these fields hold for Gemma and
        // Qwen anyway.
        if arch.family == .deepseekV4Flash {
            archDict["caQLoraRank"] = arch.caQLoraRank
            archDict["caOLoraRank"] = arch.caOLoraRank
            archDict["caOGroups"] = arch.caOGroups
            archDict["caRopeHeadDim"] = arch.caRopeHeadDim
            archDict["caIndexNHeads"] = arch.caIndexNHeads
            archDict["caIndexHeadDim"] = arch.caIndexHeadDim
            archDict["caIndexTopK"] = arch.caIndexTopK
            archDict["caCSACompressRate"] = arch.caCSACompressRate
            archDict["caHCACompressRate"] = arch.caHCACompressRate
            archDict["caCompressRopeTheta"] = arch.caCompressRopeTheta
            archDict["caRopeScalingFactor"] = arch.caRopeScalingFactor
            archDict["caRopeScalingOriginalMax"] = arch.caRopeScalingOriginalMax
            archDict["caRopeScalingBetaFast"] = arch.caRopeScalingBetaFast
            archDict["caRopeScalingBetaSlow"] = arch.caRopeScalingBetaSlow
            archDict["hcMult"] = arch.hcMult
            archDict["hcSinkhornIters"] = arch.hcSinkhornIters
            archDict["hcEps"] = arch.hcEps
            archDict["numHashRoutedLayers"] = arch.numHashRoutedLayers
            archDict["routerScoringFunc"] = arch.routerScoringFunc
            archDict["routedScalingFactor"] = arch.routedScalingFactor
            archDict["swigluLimit"] = arch.swigluLimit
        }
        // Inkling extension fields, gated on the family for the same reason:
        // the other three manifests stay byte-identical, and the reader treats
        // absence as the defaults those families already hold.
        if arch.family == .inklingSmall {
            archDict["routerScoringFunc"] = arch.routerScoringFunc
            archDict["routedScalingFactor"] = arch.routedScalingFactor
            archDict["relDRel"] = arch.relDRel
            archDict["relExtent"] = arch.relExtent
            archDict["relProjDim"] = arch.relProjDim
            archDict["relLogScalingFloor"] = arch.relLogScalingFloor
            archDict["relLogScalingAlpha"] = arch.relLogScalingAlpha
            archDict["sconvKernelSize"] = arch.sconvKernelSize
            archDict["numSharedExperts"] = arch.numSharedExperts
            archDict["numDenseLayers"] = arch.numDenseLayers
            archDict["denseIntermediateSize"] = arch.denseIntermediateSize
            archDict["sharedExpertSink"] = arch.sharedExpertSink
            archDict["embedNormEnabled"] = arch.embedNormEnabled
            archDict["logitsWidthMultiplier"] = arch.logitsWidthMultiplier
            archDict["routerGateBias"] = arch.routerGateBias
            archDict["routerNormAfterTopK"] = arch.routerNormAfterTopK
            archDict["routerGlobalScale"] = arch.routerGlobalScale
            archDict["unpaddedVocabSize"] = arch.unpaddedVocabSize
        }
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
        if arch.family == .minicpm5 {
            archDict["numSharedExperts"] = arch.numSharedExperts
            archDict["numDenseLayers"] = arch.numDenseLayers
            archDict["denseIntermediateSize"] = arch.denseIntermediateSize
            archDict["qkNorm"] = arch.qkNorm
        }
        // Qwen3.8-Flash-Next: the covered axes are value changes the non-Gemma
        // block above already publishes. These are the three NEW axes, written
        // under their own names plus a `requiredAxes` list, so a runtime that
        // cannot execute them can say which ones it is missing instead of
        // reporting an unrecognised family.
        // GLM-5.3-Flash: the low-rank query rank and indexer head shape ride
        // the V4 `ca*` keys, the KDA geometry the `linear*` keys (already
        // emitted above), mHC and router fields their DSV4 keys, the dense
        // leading layers the Inkling keys, then its own axes and the
        // `requiredAxes` list the capability gate refuses it by.
        if arch.family == .glm53Flash, let axes = arch.glm53 {
            archDict["caQLoraRank"] = arch.caQLoraRank
            archDict["caIndexNHeads"] = arch.caIndexNHeads
            archDict["caIndexHeadDim"] = arch.caIndexHeadDim
            archDict["caIndexTopK"] = arch.caIndexTopK
            archDict["hcMult"] = arch.hcMult
            archDict["hcSinkhornIters"] = arch.hcSinkhornIters
            archDict["hcEps"] = arch.hcEps
            archDict["numHashRoutedLayers"] = arch.numHashRoutedLayers
            archDict["routerScoringFunc"] = arch.routerScoringFunc
            archDict["routedScalingFactor"] = arch.routedScalingFactor
            archDict["swigluLimit"] = arch.swigluLimit
            archDict["numSharedExperts"] = arch.numSharedExperts
            archDict["numDenseLayers"] = arch.numDenseLayers
            archDict["denseIntermediateSize"] = arch.denseIntermediateSize
            archDict["routerGateBias"] = arch.routerGateBias
            archDict["routerNormAfterTopK"] = arch.routerNormAfterTopK
            archDict["qkNorm"] = arch.qkNorm
            archDict["kvLoraRank"] = axes.kvLoraRank
            archDict["qkNopeHeadDim"] = axes.qkNopeHeadDim
            archDict["vHeadDim"] = axes.vHeadDim
            archDict["indexKPool"] = axes.indexKPool
            archDict["indexKPoolAlwaysSelectTail"] = axes.indexKPoolAlwaysSelectTail
            archDict["indexerKNormEps"] = axes.indexerKNormEps
            archDict["kdaGateLowerBound"] = axes.kdaGateLowerBound
            archDict["rmsNormEps"] = axes.rmsNormEps
            archDict["unpaddedVocabSize"] = arch.unpaddedVocabSize
            archDict["requiredAxes"] = Glm53Axes.requiredAxisNames
        }
        if arch.family == .qwen38flashnext, let axes = arch.flashNext {
            archDict["numSharedExperts"] = arch.numSharedExperts
            archDict["numDenseLayers"] = arch.numDenseLayers
            archDict["unpaddedVocabSize"] = arch.unpaddedVocabSize
            archDict["hcCount"] = axes.hcCount
            archDict["hcLowRank"] = axes.hcLowRank
            archDict["indexerNumHeads"] = axes.indexerNumHeads
            archDict["indexerHeadDim"] = axes.indexerHeadDim
            archDict["indexerNumKVHeads"] = axes.indexerNumKVHeads
            archDict["indexerBudget"] = axes.indexerBudget
            archDict["indexerCompressRatio"] = axes.indexerCompressRatio
            archDict["pleLayerIDs"] = axes.pleLayerIDs
            archDict["pleNgramShardCount"] = axes.pleNgramShardCount
            archDict["pleNgramVocabSizeBase"] = axes.pleNgramVocabSizeBase
            archDict["pleConvKernelSize"] = axes.pleConvKernelSize
            archDict["requiredAxes"] = FlashNextAxes.requiredAxisNames
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
        // GLM-5.3-Flash's router gate is unquantized BF16; the manifest
        // records the slot as such rather than the 8-bit default a BF16
        // tensor would otherwise leave in place.
        if arch.family == .glm53Flash
            || (arch.family == .gemma4 && bitWidths.router == 16) {
            quantDict["router"] = [
                "weightBits": 16,
                "scheme": "unquantized",
                "scaleType": "none",
                "biasType": "none",
                "groupSize": 0,
            ]
        }
        if arch.family == .qwen38 || arch.family == .minicpm5 {
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
        if !plan.plePools.isEmpty {
            manifest["plePool"] = encodePlePools(plan.plePools)
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

    /// The additive `plePool` manifest block. `kind` names the pool format so a
    /// future revision can be told apart; readers must refuse an unknown kind.
    ///
    /// Row `i` of shard `s` lives at
    /// `shards[s].offset + (i / rowsPerBlock) * blockStride + (i % rowsPerBlock) * rowStride`.
    /// A row's record is `[rowWeightBytes | rowScaleBytes | rowBiasBytes]` — for
    /// a BF16 pool the companion sizes are 0 and the record is just the raw row.
    /// Blocks are page-aligned and rows never straddle a page, so one row costs
    /// one page fault and a cached page serves `rowsPerBlock` neighbours.
    static func encodePlePools(_ pools: [PleRowPoolPlan]) -> [String: Any] {
        [
            "kind": "rowLookupPoolV1",
            "layers": pools.map { pool in
                let quantized = pool.storage == .int4AffineG64
                return [
                    "layer": pool.layerIndex,
                    "file": pool.relativePath,
                    "sourceTensor": pool.sourceTensorPrefix,
                    "rows": pool.totalRows,
                    "rowDim": pool.rowDim,
                    "storage": pool.storage.rawValue,
                    "weightBits": pool.bits,
                    "scheme": quantized ? "affine" : "none",
                    "scaleType": quantized ? "BF16" : "none",
                    "biasType": quantized ? "BF16" : "none",
                    "groupSize": pool.groupSize,
                    "rowWeightBytes": pool.rowWeightBytes,
                    "rowScaleBytes": pool.rowCompanionBytes,
                    "rowBiasBytes": pool.rowCompanionBytes,
                    "rowStride": pool.rowStride,
                    "rowsPerBlock": pool.rowsPerBlock,
                    "blockStride": pool.blockStride,
                    "fileSize": pool.fileSize,
                    "shards": pool.shards.map {
                        [
                            "shard": $0.shardIndex,
                            "rows": $0.rows,
                            "offset": $0.regionOffset,
                            "size": $0.regionBytes,
                        ] as [String: Any]
                    },
                ] as [String: Any]
            },
        ]
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
            // Skip leading dense-FFN layers, which carry no routed experts
            // (Inkling's layers 0-1). Taking `layers.first` would publish 0
            // here and contradict the manifest, which selects the same way.
            "expertsPerLayer": plan.layers.first(where: { $0.expertsPerLayer > 0 })?
                .expertsPerLayer ?? 0,
            "layers": layersArr
        ]
        return try JSONSerialization.data(withJSONObject: obj,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
