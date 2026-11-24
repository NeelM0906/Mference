import Foundation

/// The manifest fields an `inkling2Loop` install adds. Written only for that
/// family, so every other family's manifest stays byte-identical.
extension GTurboJSON {
    /// The axes a runner must implement, published as `arch.requiredAxes`.
    /// Advisory: the runtime's `ManifestReader.familiesWithoutRunner` is the
    /// authority and holds the same list.
    static let inkling2RequiredAxes: [String] = [
        "loopedCore", "blockAttentionResiduals", "relativePositionBias",
        "projectedAttentionSink", "shortConvolution", "situExpertActivation",
        "nvfp4RoutedExperts", "q8_0Weights",
    ]

    /// `manifest.quant` of an install copied from the GGUF: each slot names
    /// the GGUF's own format (twin of the runtime's
    /// `ManifestReader.ggufQuantSlots`). Embedding, head and attention are
    /// Q8_0, the router F32, no shared expert, and the routed experts' slot
    /// names the dominant NVFP4; `arch.routedExpertFormats` and `layout.json`
    /// give each layer's format.
    static var inkling2QuantSlots: [String: Any] { [
        "embedding": ["weightBits": 8, "scheme": "gguf", "scaleType": "F16",
                      "biasType": "none", "groupSize": 32] as [String: Any],
        "attention": ["weightBits": 8, "scheme": "gguf", "scaleType": "F16",
                      "biasType": "none", "groupSize": 32] as [String: Any],
        "router": ["weightBits": 32, "scheme": "gguf", "scaleType": "none",
                   "biasType": "none", "groupSize": 0] as [String: Any],
        "sharedExpert": ["weightBits": 0, "scheme": "none", "scaleType": "none",
                         "biasType": "none", "groupSize": 0] as [String: Any],
        "routedExpert": ["weightBits": 4, "scheme": "gguf", "scaleType": "E4M3",
                         "biasType": "none", "groupSize": 16] as [String: Any],
    ] }

    /// Writes every arch key the family sets away from its absent default.
    /// The runtime requires each of them to be present for this family.
    static func addInkling2ArchFields(_ archDict: inout [String: Any], arch: ArchInfo) throws {
        guard let ink = arch.inkling2 else {
            throw RepackError.configurationInvalid(detail: "inkling2Loop arch without its Inkling 2 axes")
        }
        archDict["routerScoringFunc"] = arch.routerScoringFunc
        archDict["routedScalingFactor"] = arch.routedScalingFactor
        archDict["numSharedExperts"] = arch.numSharedExperts
        archDict["numDenseLayers"] = arch.numDenseLayers
        archDict["routerNormAfterTopK"] = arch.routerNormAfterTopK
        archDict["qkNorm"] = arch.qkNorm
        archDict["relDRel"] = ink.relDRel
        archDict["relExtent"] = ink.relExtent
        archDict["relProjDim"] = ink.relProjDim
        archDict["relLogScalingFloor"] = ink.relLogScalingFloor
        archDict["relLogScalingAlpha"] = ink.relLogScalingAlpha
        archDict["sconvKernelSize"] = ink.sconvKernelSize
        archDict["embedNormEnabled"] = ink.embedNormEnabled
        archDict["routerGateBias"] = ink.routerGateBias
        archDict["unpaddedVocabSize"] = ink.unpaddedVocabSize
        archDict["attentionSinks"] = ink.attentionSinks
        archDict["loopCoreStart"] = ink.loopCoreStart
        archDict["loopCoreEnd"] = ink.loopCoreEnd
        archDict["loopNumPasses"] = ink.loopNumPasses
        archDict["loopGateRank"] = ink.loopGateRank
        archDict["attnResBlockSize"] = ink.attnResBlockSize
        archDict["qkClipThreshold"] = ink.qkClipThreshold
        archDict["relClipThreshold"] = ink.relClipThreshold
        archDict["attentionSinkProjected"] = ink.attentionSinkProjected
        archDict["expertGateSoftcap"] = ink.expertGateSoftcap
        archDict["expertUpSoftcap"] = ink.expertUpSoftcap
        archDict["routedExpertFormats"] = ink.routedExpertFormats
        archDict["requiredAxes"] = inkling2RequiredAxes
    }
}
