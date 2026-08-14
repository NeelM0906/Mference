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
        if arch.family == .qwen38 {
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
