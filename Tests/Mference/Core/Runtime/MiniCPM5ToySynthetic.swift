import Foundation
@testable import Mference

extension ArchConfig {
    /// Tiny MiniCPM5 baseline: the plain-llama layer graph at toy width. Four
    /// full-attention layers, 4 query heads over 2 KV heads of 16 (GQA 2:1),
    /// full-head NeoX RoPE (`partialRotaryFactor` 1.0, rotaryDim == headDim),
    /// no q/k norm, no output gate, one SwiGLU MLP per layer, untied
    /// `lm_head`, no experts. Numbers respect every kernel constraint the
    /// production shape does (`hiddenSize % 64`, even rotaryDim, `numHeads %
    /// numKVHeads == 0`) and match the toy `LlamaConfig` the reference-parity
    /// goldens are captured from (`Scripts/parity/minicpm5_make_goldens.py`).
    static func miniCPM5Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 128,
            moeIntermediateSize: 0,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 16,
            fullHeadDim: 16,
            vocabSize: 128,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 5_000_000.0,
            fullRopeTheta: 5_000_000.0,
            partialRotaryFactor: 1.0,
            numLayers: 4,
            numExperts: 0,
            topKExperts: 0,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [1, 1, 1, 1],
            hiddenActivation: "silu",
            family: .minicpm5,
            attnOutputGate: false,
            attentionScale: 0.25,   // 16^-0.5
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true,
            numSharedExperts: 0,
            numDenseLayers: 4,
            denseIntermediateSize: 128,
            qkNorm: false
        )
    }
}
