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
