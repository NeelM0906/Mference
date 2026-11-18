import Foundation
@testable import Mference

extension ArchConfig {
    /// Toy GLM-5.3-Flash baseline, the shape PipeNetwork's tiny parity config
    /// takes once every quantized inner dimension is widened to the group
    /// size (INT8 / INT4 group-64 needs multiples of 64): hidden 128, four
    /// layers — three Kimi Delta Attention (mask 7) and one NoPE latent
    /// sparse attention (mask 8) — one leading dense layer of width 128, then
    /// 16 routed experts of width 64 (top-8, the production width the INT4
    /// expert reduce implements) plus one shared expert, 2
    /// attention heads of 64 over a 64-wide latent, a 2-head 64-wide indexer
    /// with `index_topk` 4 pooled in pairs, 2 KDA heads of 64 with a 4-tap
    /// conv, a 4-stream mHC, and a deliberately low swiglu clamp (0.5) so
    /// the clamp is load-bearing. Vocabulary 256. `SyntheticSnapshot.buildGlm53`
    /// writes the same geometry in PipeNetwork's layout.
    static func glm53Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 128,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHeads: 2,
            numKVHeads: 1,
            numFullKVHeads: 1,
            headDim: 64,
            fullHeadDim: 64,
            vocabSize: 256,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 0.0,
            fullRopeTheta: 0.0,
            partialRotaryFactor: 0.0,
            numLayers: 4,
            numExperts: 16,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: true,
            fullAttentionLayerMask: [7, 7, 7, 8],
            hiddenActivation: "silu",
            family: .glm53Flash,
            attnOutputGate: false,
            attentionScale: 0.125,                  // 64^-0.5
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearAttention: LinearAttentionConfig(
                numKHeads: 2, numVHeads: 2, keyHeadDim: 64, valueHeadDim: 64,
                convKernelSize: 4),
            compressedAttention: CompressedAttentionConfig(
                qLoraRank: 64, oLoraRank: 0, oGroups: 0,
                ropeHeadDim: 0,
                indexNHeads: 2, indexHeadDim: 64, indexTopK: 4,
                csaCompressRate: 0, hcaCompressRate: 0,
                compressRopeTheta: 0.0),
            hyperConnections: HyperConnectionConfig(mult: 4, sinkhornIters: 20, eps: 1e-6),
            numHashRoutedLayers: 0,
            routerScoringFunc: "sigmoid",
            routedScalingFactor: 2.5,
            swigluLimit: 0.5,
            numSharedExperts: 1,
            numDenseLayers: 1,
            denseIntermediateSize: 128,
            routerGateBias: true,
            routerNormAfterTopK: true,
            qkNorm: false,
            glm53: Glm53Config(
                kvLoraRank: 64,
                qkNopeHeadDim: 64,
                vHeadDim: 64,
                indexKPool: 2,
                indexKPoolAlwaysSelectTail: true,
                indexerKNormEps: 1e-6,
                kdaGateLowerBound: -5.0,
                rmsNormEps: 1e-5))
    }
}
