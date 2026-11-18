@testable import Mference

extension ArchConfig {
    /// The retired Gemma 4 26B-A4B baseline's geometry (see
    /// `docs/RETIRED_FAMILIES.md`), kept as a synthetic shape for tests of
    /// shared kernels and KV storage: 30 layers of 1024-token sliding-window
    /// attention with a full-attention layer every sixth, K = V on full
    /// layers, and every family-extension axis at its `ArchConfig.init`
    /// default. No install of this shape loads any more; nothing here is a
    /// supported family.
    static let retiredGemma4Shape = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: (0..<30).map { $0 % 6 == 5 ? 1 : 0 },
        hiddenActivation: "gelu_pytorch_tanh"
    )
}
