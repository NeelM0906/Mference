import Testing
import Foundation
@testable import Mference

@Suite struct ModelTypesTests {

    /// The `ArchConfig.init` defaults are what an absent manifest key means
    /// (`ManifestReader.AbsentFieldDefaults` mirrors them). They are the
    /// original install format's values, Gemma 4's, kept after that family's
    /// retirement so every kept install that omits a key decodes unchanged.
    @Test func initDefaultsAreTheOriginalFormatValues() {
        let a = ArchConfig(
            hiddenSize: 64, intermediateSize: 64, moeIntermediateSize: 64,
            numHeads: 2, numKVHeads: 1, numFullKVHeads: 1, headDim: 32,
            fullHeadDim: 32, vocabSize: 128, slidingWindow: 0,
            finalLogitSoftcap: 0, ropeTheta: 10_000, fullRopeTheta: 10_000,
            partialRotaryFactor: 1, numLayers: 1, numExperts: 0, topKExperts: 0,
            tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: [1], hiddenActivation: "silu")
        #expect(a.family == .qwen36)
        #expect(a.attnOutputGate == false)
        #expect(a.attentionScale == 1.0)
        #expect(a.embeddingScaledBySqrtHidden == true)
        #expect(a.routerScaled == true)
        #expect(a.ffnSandwichNorms == true)
        #expect(a.sharedExpertGated == false)
        #expect(a.ropeNeoxSubdim == false)
        #expect(a.qkNorm == true)
        #expect(a.inkling2 == .none)
    }

    /// Only the kept families have baselines.
    @Test func knownArchitecturesAreTheKeptFamilies() {
        #expect(Set(ArchConfig.knownArchitectures.keys)
                == [.qwen36, .qwen38, .maple, .minicpm5, .inkling2Loop])
        #expect(Set(ModelFamily.allCases) == Set(ArchConfig.knownArchitectures.keys))
        for (family, config) in ArchConfig.knownArchitectures {
            #expect(config.family == family)
        }
    }

    /// Pins the baseline against `openbmb/MiniCPM5-2B` revision `cd199ce3`
    /// `config.json` (a flat `LlamaForCausalLM` config) and the attention
    /// conventions read from `transformers` v5.6.2 `modeling_llama.py`; see
    /// `docs/families/MINICPM5.md`.
    @Test func archConfigMiniCPM5BaselineMatchesCheckpoint() {
        let a = ArchConfig.miniCPM5_2B
        #expect(a.family == .minicpm5)
        #expect(a.hiddenSize == 2048)
        #expect(a.intermediateSize == 6144)
        #expect(a.denseIntermediateSize == 6144)
        #expect(a.moeIntermediateSize == 0)
        #expect(a.numLayers == 42)
        #expect(a.numDenseLayers == 42)
        #expect(a.numHeads == 16)
        #expect(a.numKVHeads == 2)
        #expect(a.numFullKVHeads == 2)
        #expect(a.headDim == 128)
        #expect(a.fullHeadDim == 128)
        #expect(a.vocabSize == 130_560)
        #expect(a.unpaddedVocabSize == 0)
        #expect(a.tieWordEmbeddings == false)
        #expect(a.attentionKEqV == false)
        #expect(a.hiddenActivation == "silu")
        #expect(a.finalLogitSoftcap == 0.0)
        #expect(a.slidingWindow == 0)
        // Plain-llama attention: no experts, no router, no shared expert, no
        // output gate, no q/k norm, 1/sqrt(head_dim) softmax scale.
        #expect(a.numExperts == 0)
        #expect(a.topKExperts == 0)
        #expect(a.numSharedExperts == 0)
        #expect(a.attnOutputGate == false)
        #expect(a.qkNorm == false)
        // `LlamaAttention.scaling = head_dim ** -0.5`; `pow` reproduces that
        // value bit-exactly, `1 / sqrt` lands one ulp away, and the manifest
        // check is exact.
        #expect(a.attentionScale == pow(128.0, -0.5))
        #expect(a.embeddingScaledBySqrtHidden == false)
        #expect(a.routerScaled == false)
        #expect(a.ffnSandwichNorms == false)
        #expect(a.sharedExpertGated == false)
        // `rope_theta` 5e6, `rope_scaling` null, rotate_half over the whole
        // 128-wide head: NeoX sub-dim convention with rotaryDim == headDim.
        #expect(a.ropeTheta == 5_000_000.0)
        #expect(a.fullRopeTheta == 5_000_000.0)
        #expect(a.partialRotaryFactor == 1.0)
        #expect(a.ropeNeoxSubdim == true)
        #expect(a.linearAttention == .none)
        #expect(a.compressedAttention == .none)
        #expect(a.hyperConnections == .none)
        #expect(a.relativePosition == .none)
        #expect(a.flashNext == .none)
        // Every layer is full attention.
        #expect(a.fullAttentionLayerMask.count == 42)
        #expect(a.fullAttentionLayerMask.allSatisfy { $0 == 1 })
        #expect(!a.hasLinearAttentionLayers)
        #expect(!a.hasCompressedAttentionLayers)
        #expect(!a.hasLowRankHyperConnections)
    }

    /// `qkNorm` defaults to the Gemma behavior so every existing baseline —
    /// and every manifest that omits the key — keeps its meaning.
    @Test func qkNormDefaultsToTrueForEveryShippedFamily() {
        // MiniCPM5 is plain-llama attention; Inkling 2 clips q and k
        // (`inkling2.qkClipThreshold`) instead of norming them.
        let plainQK: Set<ModelFamily> = [.minicpm5, .inkling2Loop]
        for (family, config) in ArchConfig.knownArchitectures where !plainQK.contains(family) {
            #expect(config.qkNorm, "\(family.rawValue) should keep q/k norms")
        }
        #expect(ArchConfig.knownArchitectures[.minicpm5]?.qkNorm == false)
    }

    @Test func miniCPM5IsRegisteredForAutoDetection() {
        #expect(ArchConfig.knownArchitectures[.minicpm5]?.family == .minicpm5)
        #expect(ModelFamily(rawValue: "minicpm5") == .minicpm5)
    }

    @Test func modelErrorDescriptionsContainKeyFacts() {
        let e1 = ModelError.archMismatch(field: "hiddenSize", expected: "2816", actual: "4096")
        #expect(e1.description.contains("2816") && e1.description.contains("4096"))
        let e2 = ModelError.unsupportedVersion(major: 2, minor: 0)
        #expect(e2.description.contains("2"))
        let e3 = ModelError.checksumMismatch(file: "model_weights.bin")
        #expect(e3.description.contains("model_weights.bin"))
    }
}
