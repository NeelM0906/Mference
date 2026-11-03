import Foundation
import Testing
@testable import Mference

@Suite
struct FlintModelTypesTests {
    @Test func baselineIsRegistered() throws {
        let config = try #require(ArchConfig.knownArchitectures[.flint])
        #expect(config == .flint40B_A2_5B_JANG_4M)
        #expect(config.numLayers == 49)
        #expect(config.numDenseLayers == 2)
        #expect(config.numExperts == 256)
        #expect(config.topKExperts == 6)
        #expect(config.slidingWindow == 513)
        #expect(config.attentionSinks)
        #expect(config.noRopeLayerMask == config.fullAttentionLayerMask)
        #expect(config.fullAttentionLayerMask.reduce(0) { $0 + Int($1) } == 14)
        #expect(config.fullAttentionLayerMask[48] == 1)
    }

    @Test func manifestValidationAcceptsTheFlintContract() throws {
        let config = ArchConfig.flint40B_A2_5B_JANG_4M
        let affine8: [String: Any] = [
            "weightBits": 8, "scheme": "affine",
            "scaleType": "BF16", "biasType": "BF16", "groupSize": 64,
        ]
        let affine4 = affine8.merging(["weightBits": 4]) { _, new in new }
        var files: [String: Any] = [
            "model_weights.bin": ["size": 1, "sha256": "00"],
            "packed_experts/layout.json": ["size": 1, "sha256": "00"],
        ]
        for layer in config.numDenseLayers..<config.numLayers {
            files[String(format: "packed_experts/layer_%02d.bin", layer)] =
                ["size": 1, "sha256": "00"]
        }
        let root: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true],
            "modelID": "flint-40b-a2.5b-jang-4m",
            "arch": [
                "hiddenSize": config.hiddenSize,
                "ffnIntermediate": config.intermediateSize,
                "moeIntermediateSize": config.moeIntermediateSize,
                "numHeads": config.numHeads,
                "numKVHeads": config.numKVHeads,
                "numFullKVHeads": config.numFullKVHeads,
                "headDim": config.headDim,
                "fullHeadDim": config.fullHeadDim,
                "vocabSize": config.vocabSize,
                "slidingWindow": config.slidingWindow,
                "finalLogitSoftcap": config.finalLogitSoftcap,
                "ropeTheta": config.ropeTheta,
                "fullRopeTheta": config.fullRopeTheta,
                "partialRotaryFactor": config.partialRotaryFactor,
                "numLayers": config.numLayers,
                "numExperts": config.numExperts,
                "topKExperts": config.topKExperts,
                "tieWordEmbeddings": config.tieWordEmbeddings,
                "attentionKEqV": config.attentionKEqV,
                "hiddenActivation": config.hiddenActivation,
                "fullAttentionLayerMask": config.fullAttentionLayerMask.map(Int.init),
                "family": "flint",
                "attnOutputGate": false,
                "attentionScale": config.attentionScale,
                "embeddingScaledBySqrtHidden": false,
                "routerScaled": false,
                "ffnSandwichNorms": false,
                "sharedExpertGated": false,
                "ropeNeoxSubdim": true,
                "routerScoringFunc": "sigmoid",
                "routedScalingFactor": 1.0,
                "numSharedExperts": 1,
                "numDenseLayers": 2,
                "denseIntermediateSize": 6_144,
                "routerNormAfterTopK": true,
                "noRopeLayerMask": config.noRopeLayerMask.map(Int.init),
                "attentionSinks": true,
            ],
            "quant": [
                "embedding": affine8,
                "attention": affine8,
                "router": [
                    "weightBits": 16, "scheme": "unquantized",
                    "scaleType": "none", "biasType": "none", "groupSize": 0,
                ],
                "sharedExpert": affine8,
                "routedExpert": affine4,
            ],
            "files": files,
            "expertsPerLayer": 256,
            "numLayers": 49,
            "expertStride": 1_769_472,
        ]
        let data = try JSONSerialization.data(withJSONObject: root)
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        try ManifestReader.validate(
            manifest,
            against: config,
            directoryURL: URL(fileURLWithPath: "/tmp/flint.gturbo"))
    }
}
