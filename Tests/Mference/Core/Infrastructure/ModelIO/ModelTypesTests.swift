import Testing
import Foundation
@testable import Mference

@Suite struct ModelTypesTests {

    @Test func archConfigGemma4BaselineMatchesDocs() {
        let a = ArchConfig.gemma4_26B_A4B
        #expect(a.hiddenSize == 2816)
        #expect(a.intermediateSize == 2112)
        #expect(a.moeIntermediateSize == 704)
        #expect(a.numLayers == 30)
        #expect(a.numExperts == 128)
        #expect(a.topKExperts == 8)
        #expect(a.vocabSize == 262144)
        #expect(a.tieWordEmbeddings == true)
        #expect(a.finalLogitSoftcap == 30.0)
        #expect(a.fullAttentionLayerMask.count == 30)
        let fullCount = a.fullAttentionLayerMask.reduce(0) { $0 + Int($1) }
        #expect(fullCount == 5, "Gemma 4 has 5 full-attention layers, got \(fullCount)")
        // Mask flags layers 5, 11, 17, 23, 29.
        for L in [5, 11, 17, 23, 29] {
            #expect(a.fullAttentionLayerMask[L] == 1, "layer \(L) should be full-attention")
        }
    }

    /// Pins the baseline against `pipenetwork/Inkling-Small-MLX-4bit`
    /// revision `9d6e4720` `config.json -> text_config`.
    @Test func archConfigInklingSmallBaselineMatchesCheckpoint() {
        let a = ArchConfig.inklingSmall_276B_A12B
        #expect(a.hiddenSize == 4096)
        #expect(a.numLayers == 42)
        #expect(a.vocabSize == 201024)
        #expect(a.numHeads == 32)
        #expect(a.numKVHeads == 8)
        #expect(a.headDim == 128)
        #expect(a.slidingWindow == 512)
        #expect(a.numExperts == 256)
        #expect(a.topKExperts == 6)
        #expect(a.moeIntermediateSize == 2048)
        #expect(a.tieWordEmbeddings == false)

        // Position is carried entirely by the learned relative bias, so the
        // RoPE knobs must stay zeroed or the attention path would apply both.
        #expect(a.ropeTheta == 0.0)
        #expect(a.partialRotaryFactor == 0.0)
        #expect(a.relativePosition.dRel == 16)
        #expect(a.relativePosition.extent == 1024)
        #expect(a.relativePosition.projDim == 512)
        #expect(a.relativePosition.logScalingFloor == 128_000)

        #expect(a.sconvKernelSize == 4)
        #expect(a.numSharedExperts == 2)
        #expect(a.sharedExpertSink == true)
        #expect(a.numDenseLayers == 2)
        #expect(a.denseIntermediateSize == 16_384)
        #expect(a.embedNormEnabled == true)
        #expect(a.logitsWidthMultiplier == 16.0)
        // Per-head RMS-normalized q/k: scale is 1/d, not 1/sqrt(d).
        #expect(a.attentionScale == 1.0 / 128.0)
        #expect(a.unpaddedVocabSize == 200_058)
        #expect(a.routerScoringFunc == "sigmoid")
        #expect(a.routedScalingFactor == 8.0)
        #expect(a.routerGateBias == true)
        #expect(a.routerNormAfterTopK == true)
        #expect(a.routerGlobalScale == true)

        // `local_layer_ids` lists every layer except 5, 11, 17, 23, 29, 35, 41.
        #expect(a.fullAttentionLayerMask.count == 42)
        let localIDs: Set<Int> = [
            0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21,
            22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
        for L in 0..<42 {
            let expected: UInt8 = localIDs.contains(L) ? 0 : 1
            #expect(a.fullAttentionLayerMask[L] == expected,
                    "layer \(L) attention kind")
        }
        #expect(a.fullAttentionLayerMask.reduce(0) { $0 + Int($1) } == 7)
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
        // MiniCPM5 is plain-llama attention; GLM-5.3-Flash's sparse layers
        // norm the low-rank query latent (`q_a_layernorm`) rather than the
        // per-head projections, and its KDA layers have no q/k norm at all.
        let plainQK: Set<ModelFamily> = [.minicpm5, .glm53Flash]
        for (family, config) in ArchConfig.knownArchitectures where !plainQK.contains(family) {
            #expect(config.qkNorm, "\(family.rawValue) should keep q/k norms")
        }
        #expect(ArchConfig.knownArchitectures[.minicpm5]?.qkNorm == false)
        #expect(ArchConfig.knownArchitectures[.glm53Flash]?.qkNorm == false)
    }

    @Test func miniCPM5IsRegisteredForAutoDetection() {
        #expect(ArchConfig.knownArchitectures[.minicpm5]?.family == .minicpm5)
        #expect(ModelFamily(rawValue: "minicpm5") == .minicpm5)
    }

    @Test func inklingSmallIsRegisteredForAutoDetection() {
        #expect(ArchConfig.knownArchitectures[.inklingSmall]?.family
                == .inklingSmall)
        #expect(ModelFamily(rawValue: "inklingSmall") == .inklingSmall)
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
