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
                == [.qwen36, .qwen38, .maple, .minicpm5, .minicpm5_1b, .inkling2Loop])
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

    /// Pins the baseline against `thinkingmachines/Inkling-2-Mini` @ `b85c3cf`
    /// `config.json` and the metadata of Unsloth's UD-Q4_K_XL GGUF of it.
    @Test func archConfigInkling2MiniBaselineMatchesCheckpoint() {
        let a = ArchConfig.inkling2Mini
        #expect(a.family == .inkling2Loop)
        #expect(a.hiddenSize == 2304)
        #expect(a.numLayers == 24)
        #expect(a.numHeads == 24 && a.numKVHeads == 8 && a.numFullKVHeads == 8)
        #expect(a.headDim == 128 && a.fullHeadDim == 128)
        #expect(a.vocabSize == 201_024 && a.unpaddedVocabSize == 200_058)
        #expect(a.tieWordEmbeddings == false)
        #expect(a.slidingWindow == 1024)
        #expect(a.fullAttentionLayerMask.count == 24)
        #expect(a.fullAttentionLayerMask.enumerated().filter { $0.element == 1 }.map(\.offset)
                == [5, 11, 17, 23])
        #expect(a.fullAttentionLayerMask.allSatisfy { $0 == 0 || $0 == 1 })
        #expect(a.finalLogitSoftcap == 30.0)
        #expect(a.attentionScale == pow(128.0, -0.5))
        // No RoPE: the relative bias carries positions.
        #expect(a.ropeTheta == 0 && a.fullRopeTheta == 0 && a.partialRotaryFactor == 0)
        #expect(a.relativePosition == RelativePositionConfig(
            dRel: 16, extent: 1024, projDim: 24 * 16,
            logScalingFloor: 8192, logScalingAlpha: Double(Float(0.1))))
        #expect(a.sconvKernelSize == 4)
        #expect(a.attentionSinks && a.inkling2.attentionSinkProjected)
        #expect(a.qkNorm == false)
        #expect(a.embedNormEnabled)
        // MoE: 128 experts, top-8, width 1152, no shared expert.
        #expect(a.numExperts == 128 && a.topKExperts == 8)
        #expect(a.moeIntermediateSize == 1152)
        #expect(a.numSharedExperts == 0 && a.numDenseLayers == 0)
        #expect(a.routerScoringFunc == "sigmoid")
        #expect(a.routerGateBias && a.routerNormAfterTopK)
        #expect(a.routedScalingFactor == 8.0)
        #expect(a.hiddenActivation == "situ")
        #expect(a.inkling2.expertGateSoftcap == 4.0 && a.inkling2.expertUpSoftcap == 25.0)
        // The loop.
        #expect(a.inkling2.loopCoreStart == 6 && a.inkling2.loopCoreEnd == 18)
        #expect(a.inkling2.loopNumPasses == 3 && a.inkling2.loopGateRank == 64)
        #expect(a.inkling2.attnResBlockSize == 6)
        #expect(a.inkling2.loopCoreStart % a.inkling2.attnResBlockSize == 0)
        #expect(a.inkling2.loopCoreEnd % a.inkling2.attnResBlockSize == 0)
        #expect(a.inkling2.qkClipThreshold == Double(Float(512.0.squareRoot())))
        #expect(a.inkling2.relClipThreshold == 8.0)
        // Formats: NVFP4 experts in layers 0 to 19, Q8_0 in 20 to 23.
        #expect(a.inkling2.routedExpertFormats.count == 24)
        #expect(a.inkling2.routedExpertFormats.prefix(20).allSatisfy { $0 == "NVFP4" })
        #expect(a.inkling2.routedExpertFormats.suffix(4).allSatisfy { $0 == "Q8_0" })
    }

    /// `qkNorm` defaults to the Gemma behavior so every existing baseline —
    /// and every manifest that omits the key — keeps its meaning.
    @Test func qkNormDefaultsToTrueForEveryShippedFamily() {
        // MiniCPM5 is plain-llama attention; Inkling 2 clips q and k
        // (`inkling2.qkClipThreshold`) instead of norming them.
        let plainQK: Set<ModelFamily> = [.minicpm5, .minicpm5_1b, .inkling2Loop]
        for (family, config) in ArchConfig.knownArchitectures where !plainQK.contains(family) {
            #expect(config.qkNorm, "\(family.rawValue) should keep q/k norms")
        }
        #expect(ArchConfig.knownArchitectures[.minicpm5]?.qkNorm == false)
        #expect(ArchConfig.knownArchitectures[.minicpm5_1b]?.qkNorm == false)
    }

    /// Pins the 1B baseline against `openbmb/MiniCPM5-1B-MLX` revision
    /// `c973e24a` `config.json`: hidden 1536, intermediate 4608, 24 layers,
    /// and every other field the 2B's.
    @Test func archConfigMiniCPM5_1BBaselineMatchesCheckpoint() {
        let a = ArchConfig.miniCPM5_1B
        let b = ArchConfig.miniCPM5_2B
        #expect(a.family == .minicpm5_1b)
        #expect(a.family.isMiniCPM5 && b.family.isMiniCPM5)
        #expect(a.hiddenSize == 1536)
        #expect(a.intermediateSize == 4608)
        #expect(a.denseIntermediateSize == 4608)
        #expect(a.numLayers == 24)
        #expect(a.numDenseLayers == 24)
        #expect(a.fullAttentionLayerMask == [UInt8](repeating: 1, count: 24))
        #expect(a.moeIntermediateSize == b.moeIntermediateSize)
        #expect(a.numHeads == b.numHeads)
        #expect(a.numKVHeads == b.numKVHeads)
        #expect(a.numFullKVHeads == b.numFullKVHeads)
        #expect(a.headDim == b.headDim)
        #expect(a.fullHeadDim == b.fullHeadDim)
        #expect(a.vocabSize == b.vocabSize)
        #expect(a.unpaddedVocabSize == b.unpaddedVocabSize)
        #expect(a.tieWordEmbeddings == b.tieWordEmbeddings)
        #expect(a.attentionKEqV == b.attentionKEqV)
        #expect(a.hiddenActivation == b.hiddenActivation)
        #expect(a.finalLogitSoftcap == b.finalLogitSoftcap)
        #expect(a.slidingWindow == b.slidingWindow)
        #expect(a.numExperts == b.numExperts)
        #expect(a.topKExperts == b.topKExperts)
        #expect(a.numSharedExperts == b.numSharedExperts)
        #expect(a.attnOutputGate == b.attnOutputGate)
        #expect(a.qkNorm == b.qkNorm)
        #expect(a.attentionScale == b.attentionScale)
        #expect(a.embeddingScaledBySqrtHidden == b.embeddingScaledBySqrtHidden)
        #expect(a.routerScaled == b.routerScaled)
        #expect(a.ffnSandwichNorms == b.ffnSandwichNorms)
        #expect(a.sharedExpertGated == b.sharedExpertGated)
        #expect(a.ropeTheta == b.ropeTheta)
        #expect(a.fullRopeTheta == b.fullRopeTheta)
        #expect(a.partialRotaryFactor == b.partialRotaryFactor)
        #expect(a.ropeNeoxSubdim == b.ropeNeoxSubdim)
        #expect(a.linearAttention == .none)
        #expect(a.compressedAttention == .none)
        #expect(a.hyperConnections == .none)
        #expect(a.relativePosition == .none)
        #expect(a.flashNext == .none)
        #expect(a.inkling2 == .none)
    }

    @Test func miniCPM5IsRegisteredForAutoDetection() {
        #expect(ArchConfig.knownArchitectures[.minicpm5]?.family == .minicpm5)
        #expect(ModelFamily(rawValue: "minicpm5") == .minicpm5)
        #expect(ArchConfig.knownArchitectures[.minicpm5_1b]?.family == .minicpm5_1b)
        #expect(ModelFamily(rawValue: "minicpm5_1b") == .minicpm5_1b)
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
