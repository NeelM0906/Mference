import Foundation

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct ManifestArch: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let ffnIntermediate: Int
    public let moeIntermediateSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let hiddenActivation: String
    public let fullAttentionLayerMask: [Int]

    // Family extensions. Optional so legacy Gemma manifests decode unchanged;
    // absent values validate against the Gemma defaults in `ArchConfig`.
    public let family: String?
    public let attnOutputGate: Bool?
    public let attentionScale: Double?
    public let embeddingScaledBySqrtHidden: Bool?
    public let routerScaled: Bool?
    public let ffnSandwichNorms: Bool?
    public let sharedExpertGated: Bool?
    public let ropeNeoxSubdim: Bool?
    public let linearNumKHeads: Int?
    public let linearNumVHeads: Int?
    public let linearKeyHeadDim: Int?
    public let linearValueHeadDim: Int?
    public let linearConvKernelSize: Int?

    // DeepSeek-V4 compressed-attention / mHC / router extensions. Optional
    // for the same reason: absent values validate against zeroed defaults.
    public let caQLoraRank: Int?
    public let caOLoraRank: Int?
    public let caOGroups: Int?
    public let caRopeHeadDim: Int?
    public let caIndexNHeads: Int?
    public let caIndexHeadDim: Int?
    public let caIndexTopK: Int?
    public let caCSACompressRate: Int?
    public let caHCACompressRate: Int?
    public let caCompressRopeTheta: Double?
    public let caRopeScalingFactor: Double?
    public let caRopeScalingOriginalMax: Int?
    public let caRopeScalingBetaFast: Double?
    public let caRopeScalingBetaSlow: Double?
    public let hcMult: Int?
    public let hcSinkhornIters: Int?
    public let hcEps: Double?
    public let numHashRoutedLayers: Int?
    public let routerScoringFunc: String?
    public let routedScalingFactor: Double?
    public let swigluLimit: Double?

    // Inkling relative-position / short-conv / router extensions. Optional for
    // the same reason: absent values validate against the defaults the other
    // three families hold (one shared expert, unit logit scale, rest zeroed).
    public let relDRel: Int?
    public let relExtent: Int?
    public let relProjDim: Int?
    public let relLogScalingFloor: Int?
    public let relLogScalingAlpha: Double?
    public let sconvKernelSize: Int?
    public let numSharedExperts: Int?
    public let numDenseLayers: Int?
    public let denseIntermediateSize: Int?
    public let sharedExpertSink: Bool?
    public let embedNormEnabled: Bool?
    public let logitsWidthMultiplier: Double?
    public let routerGateBias: Bool?
    public let routerNormAfterTopK: Bool?
    public let routerGlobalScale: Bool?
    public let unpaddedVocabSize: Int?
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot
}

/// Optional sparse singleton-decode head retained from a Maple source
/// checkpoint. Prefill always uses the exact full vocabulary head.
public struct ManifestMapleFlashHead: Decodable, Equatable, Sendable {
    public let nClusters: Int
    public let clusterSize: Int
    public let nProbes: Int
    public let groupSize: Int
    public let bits: Int
    public let headGroupSize: Int
    public let headBits: Int
    public let scaledCentroids: Bool
    public let forceTokens: [Int]
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let flashHead: ManifestMapleFlashHead?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}

public enum ManifestReader {
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = [
        "streamingPresent", "turboQuantKV", "aneSharedExpert"
    ]

    /// Required file entries (relative to `model.gturbo/`). Layer files
    /// `packed_experts/layer_<L>.bin` for L in 0..<numLayers are checked
    /// after decode against `numLayers` (with the zero-padded "layer_%02d"
    /// naming the writer produces; falling back to plain "layer_<L>" when
    /// only the unpadded form is present, for toy synthetics).
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(directoryURL: URL,
                            expecting: ArchConfig,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let size = try metadataFileSize(manifestURL, fileName: "manifest.json")
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "manifest.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        try validate(manifest, against: expecting,
                     directoryURL: directoryURL)
        return manifest
    }

    private static func metadataFileSize(_ url: URL,
                                         fileName: String) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attrs[.size] as? NSNumber else {
            throw ModelError.indexCorrupt(detail: "\(fileName): file size unavailable")
        }
        return number.uint64Value
    }

    static func validate(_ m: Manifest,
                         against expected: ArchConfig,
                         directoryURL: URL) throws {
        guard m.magic == "GTURBO" else { throw ModelError.notAGTurboDirectory }
        guard m.versionMajor == 1 else {
            throw ModelError.unsupportedVersion(major: m.versionMajor, minor: m.versionMinor)
        }
        for key in m.flags.keys {
            if !knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
        }
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArch(m.arch, expected: expected)
        if let quant = m.quant {
            try validateQuant(quant, expected: expected)
        } else if isProductionArch(expected) {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the production architecture")
        }
        if let flashHead = m.flashHead {
            try validateMapleFlashHead(flashHead, expected: expected)
        }
        let pageSize = UInt64(getpagesize())
        guard m.expertStride % pageSize == 0 else {
            throw ModelError.expertStrideNotPageAligned(stride: m.expertStride,
                                                        pageSize: Int(pageSize))
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
        // Leading dense-FFN layers carry no routed experts, so the writer
        // emits no blob for them (Inkling's layers 0-1). `validateArch` has
        // already confirmed the manifest agrees with the baseline on the
        // count, so it is safe to skip exactly that many.
        for L in expected.numDenseLayers..<m.numLayers {
            let padded = String(format: "packed_experts/layer_%02d.bin", L)
            let plain  = "packed_experts/layer_\(L).bin"
            if m.files[padded] == nil && m.files[plain] == nil {
                throw ModelError.missingFile(name: padded)
            }
        }
    }

    /// A manifest matching one of the shipped production baselines must carry
    /// quantization metadata; toy/synthetic manifests may omit it.
    private static func isProductionArch(_ expected: ArchConfig) -> Bool {
        for baseline in ArchConfig.knownArchitectures.values {
            if expected.numLayers == baseline.numLayers,
               expected.hiddenSize == baseline.hiddenSize {
                return true
            }
        }
        return false
    }

    private static func validateQuant(_ quant: ManifestQuant,
                                      expected: ArchConfig) throws {
        if expected.family == .qwen38 {
            let affine: [(String, ManifestQuantSlot)] = [
                ("embedding", quant.embedding),
                ("attention", quant.attention),
            ]
            for (name, slot) in affine {
                guard slot.weightBits == 4,
                      slot.scheme.lowercased() == "affine",
                      slot.scaleType.lowercased() == "bf16",
                      slot.biasType.lowercased() == "bf16",
                      slot.groupSize == Quantization.groupSize else {
                    throw ModelError.indexCorrupt(
                        detail: "unsupported Qwen3.8 quantization for \(name)")
                }
            }
            // Dense: the manifest must mark the router, shared-expert and
            // routed-expert slots absent.
            let absent: [(String, ManifestQuantSlot)] = [
                ("router", quant.router),
                ("sharedExpert", quant.sharedExpert),
                ("routedExpert", quant.routedExpert),
            ]
            for (name, slot) in absent {
                guard slot.weightBits == 0,
                      slot.scheme.lowercased() == "none",
                      slot.scaleType.lowercased() == "none",
                      slot.biasType.lowercased() == "none",
                      slot.groupSize == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "Qwen3.8 manifest must mark \(name) absent")
                }
            }
            return
        }
        if expected.family == .maple {
            let affine: [(String, ManifestQuantSlot, Int)] = [
                ("embedding", quant.embedding, 4),
                ("attention", quant.attention, 4),
                ("routedExpert", quant.routedExpert, 2),
            ]
            for (name, slot, bits) in affine {
                guard slot.weightBits == bits,
                      slot.scheme.lowercased() == "affine",
                      slot.scaleType.lowercased() == "bf16",
                      slot.biasType.lowercased() == "bf16",
                      slot.groupSize == Quantization.groupSize else {
                    throw ModelError.indexCorrupt(
                        detail: "unsupported Maple quantization for \(name)")
                }
            }
            guard quant.router.weightBits == 16,
                  quant.router.scheme.lowercased() == "unquantized",
                  quant.router.scaleType.lowercased() == "none",
                  quant.router.biasType.lowercased() == "none",
                  quant.router.groupSize == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "unsupported Maple quantization for router")
            }
            guard quant.sharedExpert.weightBits == 0,
                  quant.sharedExpert.scheme.lowercased() == "none",
                  quant.sharedExpert.scaleType.lowercased() == "none",
                  quant.sharedExpert.biasType.lowercased() == "none",
                  quant.sharedExpert.groupSize == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "Maple manifest must mark sharedExpert absent")
            }
            return
        }
        // Routed experts additionally allow 2-bit: the DeepSeek-V4-Flash
        // dynamic-quant checkpoint ships Q2 experts under a Q4 core, and the
        // MoE runtime dispatches on `quant.routedExpert.weightBits`.
        let slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("embedding", quant.embedding, [4]),
            ("attention", quant.attention, [4]),
            ("router", quant.router, [8]),
            ("sharedExpert", quant.sharedExpert, [4, 8]),
            ("routedExpert", quant.routedExpert, [2, 4]),
        ]
        for (name, slot, allowedBits) in slots {
            guard allowedBits.contains(slot.weightBits),
                  slot.scheme.lowercased() == "affine",
                  slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16",
                  slot.groupSize == Quantization.groupSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
    }

    private static func validateMapleFlashHead(_ flashHead: ManifestMapleFlashHead,
                                               expected: ArchConfig) throws {
        guard expected.family == .maple,
              flashHead.nClusters > 0, flashHead.clusterSize > 0,
              flashHead.nClusters <= Int.max / flashHead.clusterSize,
              flashHead.nClusters * flashHead.clusterSize == expected.vocabSize,
              flashHead.nProbes > 0, flashHead.nProbes <= flashHead.nClusters,
              flashHead.groupSize == Quantization.groupSize, flashHead.bits == 4,
              flashHead.headGroupSize == Quantization.groupSize, flashHead.headBits == 4,
              flashHead.scaledCentroids,
              Set(flashHead.forceTokens).count == flashHead.forceTokens.count,
              flashHead.forceTokens.allSatisfy({ $0 >= 0 && $0 < expected.vocabSize }) else {
            throw ModelError.indexCorrupt(detail: "invalid Maple FlashHead metadata")
        }
    }

    private static func validateArch(_ a: ManifestArch,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        guard a.fullAttentionLayerMask.allSatisfy({ UInt8(exactly: $0) != nil }) else {
            throw ModelError.archMismatch(
                field: "fullAttentionLayerMask",
                expected: e.fullAttentionLayerMask.description,
                actual: a.fullAttentionLayerMask.description)
        }
        let actualMask = a.fullAttentionLayerMask.compactMap { UInt8(exactly: $0) }
        try check("fullAttentionLayerMask",
                  actualMask.description,
                  e.fullAttentionLayerMask.description)

        if e.family == .maple {
            let requiredExtensions: [(String, Bool)] = [
                ("family", a.family != nil),
                ("attnOutputGate", a.attnOutputGate != nil),
                ("attentionScale", a.attentionScale != nil),
                ("embeddingScaledBySqrtHidden", a.embeddingScaledBySqrtHidden != nil),
                ("routerScaled", a.routerScaled != nil),
                ("ffnSandwichNorms", a.ffnSandwichNorms != nil),
                ("ropeNeoxSubdim", a.ropeNeoxSubdim != nil),
                ("routerScoringFunc", a.routerScoringFunc != nil),
                ("routedScalingFactor", a.routedScalingFactor != nil),
                ("swigluLimit", a.swigluLimit != nil),
                ("numSharedExperts", a.numSharedExperts != nil),
                ("routerNormAfterTopK", a.routerNormAfterTopK != nil),
            ]
            if let missing = requiredExtensions.first(where: { !$0.1 }) {
                throw ModelError.archMismatch(field: missing.0,
                                              expected: "present",
                                              actual: "missing")
            }
        }

        // Family extensions: absent fields mean the Gemma defaults.
        let gemmaDefaults = ArchConfig.gemma4_26B_A4B
        try check("family",
                  a.family ?? ModelFamily.gemma4.rawValue,
                  e.family.rawValue)
        try check("attnOutputGate",
                  a.attnOutputGate ?? gemmaDefaults.attnOutputGate,
                  e.attnOutputGate)
        try check("attentionScale",
                  a.attentionScale ?? gemmaDefaults.attentionScale,
                  e.attentionScale)
        try check("embeddingScaledBySqrtHidden",
                  a.embeddingScaledBySqrtHidden ?? gemmaDefaults.embeddingScaledBySqrtHidden,
                  e.embeddingScaledBySqrtHidden)
        try check("routerScaled",
                  a.routerScaled ?? gemmaDefaults.routerScaled,
                  e.routerScaled)
        try check("ffnSandwichNorms",
                  a.ffnSandwichNorms ?? gemmaDefaults.ffnSandwichNorms,
                  e.ffnSandwichNorms)
        try check("sharedExpertGated",
                  a.sharedExpertGated ?? gemmaDefaults.sharedExpertGated,
                  e.sharedExpertGated)
        try check("ropeNeoxSubdim",
                  a.ropeNeoxSubdim ?? gemmaDefaults.ropeNeoxSubdim,
                  e.ropeNeoxSubdim)
        try check("linearNumKHeads",
                  a.linearNumKHeads ?? 0, e.linearAttention.numKHeads)
        try check("linearNumVHeads",
                  a.linearNumVHeads ?? 0, e.linearAttention.numVHeads)
        try check("linearKeyHeadDim",
                  a.linearKeyHeadDim ?? 0, e.linearAttention.keyHeadDim)
        try check("linearValueHeadDim",
                  a.linearValueHeadDim ?? 0, e.linearAttention.valueHeadDim)
        try check("linearConvKernelSize",
                  a.linearConvKernelSize ?? 0, e.linearAttention.convKernelSize)
        try check("caQLoraRank",
                  a.caQLoraRank ?? 0, e.compressedAttention.qLoraRank)
        try check("caOLoraRank",
                  a.caOLoraRank ?? 0, e.compressedAttention.oLoraRank)
        try check("caOGroups",
                  a.caOGroups ?? 0, e.compressedAttention.oGroups)
        try check("caRopeHeadDim",
                  a.caRopeHeadDim ?? 0, e.compressedAttention.ropeHeadDim)
        try check("caIndexNHeads",
                  a.caIndexNHeads ?? 0, e.compressedAttention.indexNHeads)
        try check("caIndexHeadDim",
                  a.caIndexHeadDim ?? 0, e.compressedAttention.indexHeadDim)
        try check("caIndexTopK",
                  a.caIndexTopK ?? 0, e.compressedAttention.indexTopK)
        try check("caCSACompressRate",
                  a.caCSACompressRate ?? 0, e.compressedAttention.csaCompressRate)
        try check("caHCACompressRate",
                  a.caHCACompressRate ?? 0, e.compressedAttention.hcaCompressRate)
        try check("caCompressRopeTheta",
                  a.caCompressRopeTheta ?? 0, e.compressedAttention.compressRopeTheta)
        try check("caRopeScalingFactor",
                  a.caRopeScalingFactor ?? 0, e.compressedAttention.ropeScalingFactor)
        try check("caRopeScalingOriginalMax",
                  a.caRopeScalingOriginalMax ?? 0,
                  e.compressedAttention.ropeScalingOriginalMax)
        try check("caRopeScalingBetaFast",
                  a.caRopeScalingBetaFast ?? 0, e.compressedAttention.ropeScalingBetaFast)
        try check("caRopeScalingBetaSlow",
                  a.caRopeScalingBetaSlow ?? 0, e.compressedAttention.ropeScalingBetaSlow)
        try check("hcMult",
                  a.hcMult ?? 0, e.hyperConnections.mult)
        try check("hcSinkhornIters",
                  a.hcSinkhornIters ?? 0, e.hyperConnections.sinkhornIters)
        try check("hcEps",
                  a.hcEps ?? 0, e.hyperConnections.eps)
        try check("numHashRoutedLayers",
                  a.numHashRoutedLayers ?? 0, e.numHashRoutedLayers)
        try check("routerScoringFunc",
                  a.routerScoringFunc ?? "softmax", e.routerScoringFunc)
        try check("routedScalingFactor",
                  a.routedScalingFactor ?? 1.0, e.routedScalingFactor)
        try check("swigluLimit",
                  a.swigluLimit ?? 0.0, e.swigluLimit)
        try check("relDRel",
                  a.relDRel ?? 0, e.relativePosition.dRel)
        try check("relExtent",
                  a.relExtent ?? 0, e.relativePosition.extent)
        try check("relProjDim",
                  a.relProjDim ?? 0, e.relativePosition.projDim)
        try check("relLogScalingFloor",
                  a.relLogScalingFloor ?? 0, e.relativePosition.logScalingFloor)
        try check("relLogScalingAlpha",
                  a.relLogScalingAlpha ?? 0.0, e.relativePosition.logScalingAlpha)
        try check("sconvKernelSize",
                  a.sconvKernelSize ?? 0, e.sconvKernelSize)
        try check("numSharedExperts",
                  a.numSharedExperts ?? 1, e.numSharedExperts)
        try check("numDenseLayers",
                  a.numDenseLayers ?? 0, e.numDenseLayers)
        try check("denseIntermediateSize",
                  a.denseIntermediateSize ?? 0, e.denseIntermediateSize)
        try check("sharedExpertSink",
                  a.sharedExpertSink ?? false, e.sharedExpertSink)
        try check("embedNormEnabled",
                  a.embedNormEnabled ?? false, e.embedNormEnabled)
        try check("logitsWidthMultiplier",
                  a.logitsWidthMultiplier ?? 1.0, e.logitsWidthMultiplier)
        try check("routerGateBias",
                  a.routerGateBias ?? false, e.routerGateBias)
        try check("routerNormAfterTopK",
                  a.routerNormAfterTopK ?? false, e.routerNormAfterTopK)
        try check("routerGlobalScale",
                  a.routerGlobalScale ?? false, e.routerGlobalScale)
        try check("unpaddedVocabSize",
                  a.unpaddedVocabSize ?? 0, e.unpaddedVocabSize)
    }

    /// Decode just enough of `manifest.json` to identify the model family,
    /// without arch validation. Used by `Model.load` auto-detection.
    public static func peekFamily(directoryURL: URL,
                                  maxBytes: UInt64 = defaultMaxBytes) throws -> ModelFamily {
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let size = try metadataFileSize(manifestURL, fileName: "manifest.json")
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "manifest.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        guard let raw = manifest.arch.family else { return .gemma4 }
        guard let family = ModelFamily(rawValue: raw) else {
            throw ModelError.indexCorrupt(detail: "unknown arch.family \"\(raw)\"")
        }
        return family
    }
}
