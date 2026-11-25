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

    // Family extensions. Optional so manifests that omit a key decode
    // unchanged; absent values validate against `absentFieldDefaults`.
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
    public let noRopeLayerMask: [UInt8]?
    public let attentionSinks: Bool?

    // Qwen3.8-Flash-Next extensions. Optional for the same reason: absent
    // values validate against the zeroed `FlashNextConfig.none`. Field names
    // match what `MferenceRepack`'s `FlashNextAxes` publishes verbatim.
    public let hcCount: Int?
    public let hcLowRank: Int?
    public let indexerNumHeads: Int?
    public let indexerHeadDim: Int?
    public let indexerNumKVHeads: Int?
    public let indexerBudget: Int?
    public let indexerCompressRatio: Int?
    public let pleLayerIDs: [Int]?
    public let pleNgramShardCount: Int?
    public let pleNgramVocabSizeBase: Int?
    public let pleConvKernelSize: Int?
    /// Not emitted by installs predating this axis, so `validateArch` checks
    /// it only when present. See `FlashNextConfig.pleEosTokenID`.
    public let pleEosTokenID: Int?
    /// Axis names the install claims a runner must implement. Advisory only —
    /// `ManifestReader.familiesWithoutRunner` is the authority, so an install
    /// publishing this field does not by itself make the family unloadable.
    public let requiredAxes: [String]?
    /// Per-head q/k RMSNorm gains on full-attention layers. Optional so every
    /// manifest predating the axis decodes unchanged; absent validates against
    /// the Gemma default (`true`).
    public let qkNorm: Bool?

    // GLM-5.3-Flash extensions. Optional for the same reason: absent values
    // validate against the zeroed `Glm53Config.none`. Field names match what
    // `MferenceRepack`'s `Glm53Axes` publishes.
    public let kvLoraRank: Int?
    public let qkNopeHeadDim: Int?
    public let vHeadDim: Int?
    public let indexKPool: Int?
    public let indexKPoolAlwaysSelectTail: Bool?
    public let indexerKNormEps: Double?
    public let kdaGateLowerBound: Double?
    public let rmsNormEps: Double?

    // Inkling 2 extensions. Optional for the same reason: absent values
    // validate against the zeroed `Inkling2Config.none`. Field names match
    // what `MferenceRepack`'s `Inkling2Axes` publishes.
    public let loopCoreStart: Int?
    public let loopCoreEnd: Int?
    public let loopNumPasses: Int?
    public let loopGateRank: Int?
    public let attnResBlockSize: Int?
    public let qkClipThreshold: Double?
    public let relClipThreshold: Double?
    public let attentionSinkProjected: Bool?
    public let expertGateSoftcap: Double?
    public let expertUpSoftcap: Double?
    public let routedExpertFormats: [String]?
}

/// Whether an optional tensor group from the source checkpoint was carried
/// into the install (`mtp`) or skipped (`vision`).
public struct ManifestSidecar: Decodable, Equatable, Sendable {
    public let carried: Bool
    public let tensorCount: Int
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

    // Additive install blocks. Every family that predates them emits none of
    // these keys, so existing manifests decode byte-identically.

    /// Optional source tensor groups and whether the install carried them.
    public let sidecars: [String: ManifestSidecar]?
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
        // emits no blob for them. `validateArch` has
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

    /// The quant contract of an install copied from a GGUF (`inkling2Loop`):
    /// nothing is re-quantized, so each slot names the GGUF's own format.
    /// Embedding, output head and attention are Q8_0 (8-bit codes with one F16
    /// scale per 32 values); the router is F32; there is no shared expert; the
    /// routed experts are mixed per layer, so their slot names the dominant
    /// NVFP4 (4-bit E2M1 codes, one E4M3 scale per 16 values) and
    /// `arch.routedExpertFormats` with `layout.json` carry each layer's format.
    /// The runtime dispatches on each tensor's own index dtype, never on these.
    static let ggufQuantSlots: [String: ManifestQuantSlot] = [
        "embedding": ManifestQuantSlot(weightBits: 8, scheme: "gguf", scaleType: "F16",
                                       biasType: "none", groupSize: 32),
        "attention": ManifestQuantSlot(weightBits: 8, scheme: "gguf", scaleType: "F16",
                                       biasType: "none", groupSize: 32),
        "router": ManifestQuantSlot(weightBits: 32, scheme: "gguf", scaleType: "none",
                                    biasType: "none", groupSize: 0),
        "sharedExpert": ManifestQuantSlot(weightBits: 0, scheme: "none", scaleType: "none",
                                          biasType: "none", groupSize: 0),
        "routedExpert": ManifestQuantSlot(weightBits: 4, scheme: "gguf", scaleType: "E4M3",
                                          biasType: "none", groupSize: 16),
    ]

    private static func validateQuant(_ quant: ManifestQuant,
                                      expected: ArchConfig) throws {
        if expected.family == .inkling2Loop {
            let slots: [(String, ManifestQuantSlot)] = [
                ("embedding", quant.embedding), ("attention", quant.attention),
                ("router", quant.router), ("sharedExpert", quant.sharedExpert),
                ("routedExpert", quant.routedExpert),
            ]
            for (name, slot) in slots {
                guard let want = ggufQuantSlots[name],
                      slot.weightBits == want.weightBits,
                      slot.scheme.lowercased() == want.scheme.lowercased(),
                      slot.scaleType.lowercased() == want.scaleType.lowercased(),
                      slot.biasType.lowercased() == want.biasType.lowercased(),
                      slot.groupSize == want.groupSize else {
                    throw ModelError.indexCorrupt(
                        detail: "unsupported Inkling 2 quantization for \(name)")
                }
            }
            return
        }
        if expected.family.isMiniCPM5 {
            // Dense, like Qwen 3.8: INT4 affine group-64 embedding and
            // attention slots (the MLP shares the attention slot), and the
            // router / shared-expert / routed-expert slots marked absent.
            try validateDenseInt4Quant(quant, familyName: "MiniCPM5")
            return
        }
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
        // Routed experts additionally allow 2-bit: the MoE runtime dispatches
        // on `quant.routedExpert.weightBits`.
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

    /// The dense-family quant contract: INT4 affine group-64 `embedding` and
    /// `attention` slots, and every MoE slot marked absent.
    private static func validateDenseInt4Quant(_ quant: ManifestQuant,
                                               familyName: String) throws {
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
                    detail: "unsupported \(familyName) quantization for \(name)")
            }
        }
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
                    detail: "\(familyName) manifest must mark \(name) absent")
            }
        }
    }

    /// What an absent family-extension key in `manifest.arch` means: the
    /// original install format's values (Gemma 4's, a family retired on
    /// 2026-11-18; AGENTS.md lists the retired families). They equal the
    /// `ArchConfig.init` defaults and must not change, or kept installs that
    /// omit a key would validate against different values.
    private struct AbsentFieldDefaults {
        let attnOutputGate = false
        let attentionScale = 1.0
        let embeddingScaledBySqrtHidden = true
        let routerScaled = true
        let ffnSandwichNorms = true
        let sharedExpertGated = false
        let ropeNeoxSubdim = false
        let qkNorm = true
    }
    private static let absentFieldDefaults = AbsentFieldDefaults()

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
        // Every axis Inkling 2 sets away from its absent default must be
        // written, so a manifest that drops one cannot validate by default.
        if e.family == .inkling2Loop {
            let requiredExtensions: [(String, Bool)] = [
                ("attentionScale", a.attentionScale != nil),
                ("embeddingScaledBySqrtHidden", a.embeddingScaledBySqrtHidden != nil),
                ("routerScaled", a.routerScaled != nil),
                ("ffnSandwichNorms", a.ffnSandwichNorms != nil),
                ("qkNorm", a.qkNorm != nil),
                ("routerScoringFunc", a.routerScoringFunc != nil),
                ("routedScalingFactor", a.routedScalingFactor != nil),
                ("relDRel", a.relDRel != nil),
                ("relExtent", a.relExtent != nil),
                ("relProjDim", a.relProjDim != nil),
                ("relLogScalingFloor", a.relLogScalingFloor != nil),
                ("relLogScalingAlpha", a.relLogScalingAlpha != nil),
                ("sconvKernelSize", a.sconvKernelSize != nil),
                ("numSharedExperts", a.numSharedExperts != nil),
                ("embedNormEnabled", a.embedNormEnabled != nil),
                ("routerGateBias", a.routerGateBias != nil),
                ("routerNormAfterTopK", a.routerNormAfterTopK != nil),
                ("unpaddedVocabSize", a.unpaddedVocabSize != nil),
                ("attentionSinks", a.attentionSinks != nil),
                ("loopCoreStart", a.loopCoreStart != nil),
                ("loopCoreEnd", a.loopCoreEnd != nil),
                ("loopNumPasses", a.loopNumPasses != nil),
                ("loopGateRank", a.loopGateRank != nil),
                ("attnResBlockSize", a.attnResBlockSize != nil),
                ("qkClipThreshold", a.qkClipThreshold != nil),
                ("relClipThreshold", a.relClipThreshold != nil),
                ("attentionSinkProjected", a.attentionSinkProjected != nil),
                ("expertGateSoftcap", a.expertGateSoftcap != nil),
                ("expertUpSoftcap", a.expertUpSoftcap != nil),
                ("routedExpertFormats", a.routedExpertFormats != nil),
            ]
            if let missing = requiredExtensions.first(where: { !$0.1 }) {
                throw ModelError.archMismatch(field: missing.0,
                                              expected: "present",
                                              actual: "missing")
            }
        }

        // Family extensions: absent fields mean the original install format's
        // values (`absentFieldDefaults`).
        let absentDefaults = Self.absentFieldDefaults
        try check("family",
                  a.family ?? "",
                  e.family.rawValue)
        try check("attnOutputGate",
                  a.attnOutputGate ?? absentDefaults.attnOutputGate,
                  e.attnOutputGate)
        try check("attentionScale",
                  a.attentionScale ?? absentDefaults.attentionScale,
                  e.attentionScale)
        try check("embeddingScaledBySqrtHidden",
                  a.embeddingScaledBySqrtHidden ?? absentDefaults.embeddingScaledBySqrtHidden,
                  e.embeddingScaledBySqrtHidden)
        try check("routerScaled",
                  a.routerScaled ?? absentDefaults.routerScaled,
                  e.routerScaled)
        try check("ffnSandwichNorms",
                  a.ffnSandwichNorms ?? absentDefaults.ffnSandwichNorms,
                  e.ffnSandwichNorms)
        try check("sharedExpertGated",
                  a.sharedExpertGated ?? absentDefaults.sharedExpertGated,
                  e.sharedExpertGated)
        try check("ropeNeoxSubdim",
                  a.ropeNeoxSubdim ?? absentDefaults.ropeNeoxSubdim,
                  e.ropeNeoxSubdim)
        try check("qkNorm",
                  a.qkNorm ?? absentDefaults.qkNorm,
                  e.qkNorm)
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
        // The unpadded vocabulary is a head-masking fact, not a layout one:
        // a manifest written before a family recorded it (GLM-5.3-Flash
        // installs from before 2026-09-11) leaves the baseline's value in
        // force; one that records it must agree.
        if let recorded = a.unpaddedVocabSize {
            try check("unpaddedVocabSize", recorded, e.unpaddedVocabSize)
        }

        let fn = e.flashNext
        try check("hcCount",              a.hcCount ?? 0,              fn.hcCount)
        try check("hcLowRank",            a.hcLowRank ?? 0,            fn.hcLowRank)
        try check("indexerNumHeads",      a.indexerNumHeads ?? 0,      fn.indexerNumHeads)
        try check("indexerHeadDim",       a.indexerHeadDim ?? 0,       fn.indexerHeadDim)
        try check("indexerNumKVHeads",    a.indexerNumKVHeads ?? 0,    fn.indexerNumKVHeads)
        try check("indexerBudget",        a.indexerBudget ?? 0,        fn.indexerBudget)
        try check("indexerCompressRatio", a.indexerCompressRatio ?? 0, fn.indexerCompressRatio)
        try check("pleLayerIDs",
                  (a.pleLayerIDs ?? []).description, fn.pleLayerIDs.description)
        try check("pleNgramShardCount",   a.pleNgramShardCount ?? 0,   fn.pleNgramShardCount)
        try check("pleNgramVocabSizeBase",
                  a.pleNgramVocabSizeBase ?? 0, fn.pleNgramVocabSizeBase)
        try check("pleConvKernelSize",    a.pleConvKernelSize ?? 0,    fn.pleConvKernelSize)
        // Checked only when the install publishes it: the shipped repack path
        // predates this axis, and defaulting to the baseline would turn a real
        // mismatch into silent agreement. See `FlashNextConfig.pleEosTokenID`.
        if let published = a.pleEosTokenID {
            try check("pleEosTokenID", published, fn.pleEosTokenID)
        }
        try check("noRopeLayerMask",
                  (a.noRopeLayerMask ?? []).description,
                  e.noRopeLayerMask.description)
        try check("attentionSinks",
                  a.attentionSinks ?? false, e.attentionSinks)

        let g = e.glm53
        try check("kvLoraRank",        a.kvLoraRank ?? 0,        g.kvLoraRank)
        try check("qkNopeHeadDim",     a.qkNopeHeadDim ?? 0,     g.qkNopeHeadDim)
        try check("vHeadDim",          a.vHeadDim ?? 0,          g.vHeadDim)
        try check("indexKPool",        a.indexKPool ?? 0,        g.indexKPool)
        try check("indexKPoolAlwaysSelectTail",
                  a.indexKPoolAlwaysSelectTail ?? false, g.indexKPoolAlwaysSelectTail)
        try check("indexerKNormEps",   a.indexerKNormEps ?? 0,   g.indexerKNormEps)
        try check("kdaGateLowerBound", a.kdaGateLowerBound ?? 0, g.kdaGateLowerBound)
        try check("rmsNormEps",        a.rmsNormEps ?? 0,        g.rmsNormEps)

        let ink = e.inkling2
        try check("loopCoreStart",    a.loopCoreStart ?? 0,    ink.loopCoreStart)
        try check("loopCoreEnd",      a.loopCoreEnd ?? 0,      ink.loopCoreEnd)
        try check("loopNumPasses",    a.loopNumPasses ?? 0,    ink.loopNumPasses)
        try check("loopGateRank",     a.loopGateRank ?? 0,     ink.loopGateRank)
        try check("attnResBlockSize", a.attnResBlockSize ?? 0, ink.attnResBlockSize)
        try check("qkClipThreshold",  a.qkClipThreshold ?? 0,  ink.qkClipThreshold)
        try check("relClipThreshold", a.relClipThreshold ?? 0, ink.relClipThreshold)
        try check("attentionSinkProjected",
                  a.attentionSinkProjected ?? false, ink.attentionSinkProjected)
        try check("expertGateSoftcap", a.expertGateSoftcap ?? 0, ink.expertGateSoftcap)
        try check("expertUpSoftcap",   a.expertUpSoftcap ?? 0,   ink.expertUpSoftcap)
        try check("routedExpertFormats",
                  (a.routedExpertFormats ?? []).description,
                  ink.routedExpertFormats.description)
    }

    /// Metadata-only checkpoint identity, bounded like the family probe.
    public static func peekModelID(directoryURL: URL,
                                   maxBytes: UInt64 = defaultMaxBytes) throws -> String {
        let url = directoryURL.appendingPathComponent("manifest.json")
        let size = try metadataFileSize(url, fileName: "manifest.json")
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(detail: "manifest.json exceeds metadata cap")
        }
        struct Identity: Decodable { let modelID: String }
        return try JSONDecoder().decode(Identity.self, from: Data(contentsOf: url)).modelID
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
        // The capability gate runs off a minimal decode, before the full
        // `Manifest` shape is required. A family the runner cannot execute must
        // say so by name even if its manifest carries fields (or omits ones)
        // the strict decoder does not expect — otherwise the first thing the
        // user sees is a decode error that reads like a corrupt install.
        if let declared = try? JSONDecoder().decode(FamilyPeek.self, from: data),
           let raw = declared.arch.family,
           let refusal = capabilityRefusal(family: raw) {
            throw refusal
        }
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        // A retired checkpoint on a kept architecture (Swift-Qwen3.8 is a
        // `qwen38` install) would otherwise load as that kept family.
        if retiredCheckpoints[manifest.modelID] != nil {
            throw ModelError.retiredCheckpoint(modelID: manifest.modelID)
        }
        // A manifest without `arch.family` was a Gemma 4 install, a retired
        // family, as is every name in `retiredFamilies`.
        guard let raw = manifest.arch.family else { throw ModelError.retiredFamily(family: "gemma4") }
        guard let family = ModelFamily(rawValue: raw) else {
            if retiredFamilies.contains(raw) { throw ModelError.retiredFamily(family: raw) }
            throw ModelError.indexCorrupt(detail: "unknown arch.family \"\(raw)\"")
        }
        return family
    }

    /// Just enough of `manifest.json` to read `arch.family`.
    private struct FamilyPeek: Decodable {
        struct Arch: Decodable { let family: String? }
        let arch: Arch
    }

    /// The axes a raw `arch.family` string is missing a runner for, or nil when
    /// the runtime can execute it.
    ///
    /// `peekFamily` refuses a gated family by throwing, which is right for a
    /// load but not for a caller enumerating a library: it has to tell "this
    /// install has no runner yet" from "this install is broken" without
    /// matching on error text. `familiesWithoutRunner` stays the authority;
    /// this only reads it.
    public static func missingRunnerAxes(forRawFamily raw: String) -> [String]? {
        familiesWithoutRunner[raw]
    }

    /// Families `MferenceRepack` can install but no forward runner can execute,
    /// mapped to the axes whose kernels are missing.
    ///
    /// Holds `inkling2Loop` while `Inkling2ForwardRunner` is built. Remove an
    /// entry only when `ForwardRunnerFactory` actually dispatches to the
    /// family's runner.
    ///
    /// Add an entry the moment a new family gains an `ArchInfo` install path,
    /// keyed by its `manifest.arch.family` string and valued with the axis
    /// names whose kernels are still missing. Remove it only when that family's
    /// runner actually lands and `ForwardRunnerFactory` dispatches to it.
    ///
    /// The gate lives here because `peekFamily` is the single funnel every
    /// entry point uses — `Model.load`, the CLI, the loopback server, the Mac
    /// app's installation probe and the tokenizer loader all call it — so one
    /// check makes the failure named and actionable everywhere instead of
    /// surfacing as "unknown arch.family" or, worse, matching some other
    /// family's runner.
    ///
    /// The axis names are the runtime's own; the install also publishes them as
    /// `manifest.arch.requiredAxes`, but this table is deliberately the
    /// authority — a manifest does not get to tell the runtime what it can run.
    ///
    /// A gated family may still carry an `ArchConfig` baseline, a `ModelFamily`
    /// case, tensor accessors and manifest validation — that is what the
    /// runner is built *against*. Presence in this table is the single fact
    /// that decides whether it can be loaded, and it is checked in
    /// `peekFamily` before any of that machinery is reached.
    static let familiesWithoutRunner: [String: [String]] = [:]

    /// The mechanisms the Inkling 2 runner implements, by axis name. The
    /// installer publishes the same list as `manifest.arch.requiredAxes`; it
    /// held `inkling2Loop` in `familiesWithoutRunner` until
    /// `Inkling2ForwardRunner` landed.
    public static let inkling2RequiredAxes: [String] = [
        "loopedCore", "blockAttentionResiduals", "relativePositionBias",
        "projectedAttentionSink", "shortConvolution", "situExpertActivation",
        "nvfp4RoutedExperts", "q8_0Weights",
    ]

    /// `arch.family` values of the families retired on 2026-11-18 (see
    /// AGENTS.md). Their installs on disk are left alone;
    /// loading one fails with `ModelError.retiredFamily` instead of "unknown
    /// arch.family", so the library and the CLI can say what happened.
    public static let retiredFamilies: Set<String> = [
        "gemma4", "deepseekV4Flash", "inklingSmall", "flint",
        "qwen38flashnext", "qwen38flashnextudiq3xxs", "glm53Flash",
    ]

    /// Retired checkpoints whose installs carry a kept `arch.family`, keyed by
    /// `manifest.modelID`, valued with the retired family's name. Refused by
    /// ID (`ModelError.retiredCheckpoint`) because their family would load.
    public static let retiredCheckpoints: [String: String] = [
        "swift-qwen3.8-27b-int4g64": "Swift-Qwen3.8",
    ]

    /// An install on disk that belongs to a retired family: the identifier a
    /// library lists it under and the reason it is not served.
    public struct RetiredInstall: Equatable, Sendable {
        public let identifier: String
        public let detail: String
    }

    /// Classifies a `manifest.json` as a retired install, or nil when it is
    /// not one (or cannot be read: the caller's own checks report that).
    /// Library probes use it ahead of `peekFamily` so a retired install is
    /// listed as not runnable rather than as a broken one:
    /// - a retired checkpoint ID (`retiredCheckpoints`), under that ID;
    /// - a manifest without `arch.family` (the original Gemma 4 format) that
    ///   names a `modelID`, under that ID;
    /// - a retired `arch.family` (`retiredFamilies`), under that family name.
    public static func retiredInstall(manifestURL: URL,
                                      maxBytes: UInt64 = defaultMaxBytes) -> RetiredInstall? {
        struct Peek: Decodable {
            struct Arch: Decodable { let family: String? }
            let arch: Arch?
            let modelID: String?
        }
        guard let size = try? metadataFileSize(manifestURL, fileName: "manifest.json"),
              size <= maxBytes,
              let data = try? Data(contentsOf: manifestURL),
              let peek = try? JSONDecoder().decode(Peek.self, from: data) else { return nil }
        let notServed = "(see AGENTS.md); the install is left on disk but not served"
        if let id = peek.modelID, let name = retiredCheckpoints[id] {
            return RetiredInstall(identifier: id,
                                  detail: "\(id) is a \(name) checkpoint, a retired family \(notServed)")
        }
        guard let arch = peek.arch else { return nil }
        guard let raw = arch.family else {
            // Every Gemma 4 install names its checkpoint; a manifest without
            // one is not an install of any family, so it stays incomplete.
            guard let id = peek.modelID else { return nil }
            return RetiredInstall(identifier: id,
                                  detail: "\(id) has no arch.family: a Gemma 4 install, a retired family \(notServed)")
        }
        if retiredFamilies.contains(raw) {
            return RetiredInstall(identifier: raw, detail: "\(raw) is a retired family \(notServed)")
        }
        return nil
    }

    /// The gate lookup, factored out of `peekFamily` so it can be exercised
    /// against an injected table, so the mechanism stays testable whatever the
    /// shipped table holds.
    ///
    /// Returns the named refusal for a gated family, or nil for one the runtime
    /// can execute.
    static func capabilityRefusal(family raw: String,
                                  in table: [String: [String]] = familiesWithoutRunner)
        -> ModelError? {
        guard let missingAxes = table[raw] else { return nil }
        return .familyRunnerNotImplemented(family: raw, missingAxes: missingAxes)
    }
}
