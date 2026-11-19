extension Native {
    /// Hyperparameters of a llama.cpp `qwen35` model, decoded from GGUF
    /// metadata (`qwen35.*` keys). Quick (Qwen3.5-4B UD-Q4_K_XL) values are
    /// given in brackets.
    public struct Qwen35Config: Codable, Equatable, Sendable {
        /// Main blocks: GGUF `block_count` minus `nextn_predict_layers`
        /// (llama.cpp's `n_layer`) [32; Deep 64]. The `nextn` blocks follow at
        /// indices `blockCount ..< blockCount + nextnPredictLayers`.
        public var blockCount: Int
        /// `embedding_length` [2560].
        public var hiddenSize: Int
        /// `feed_forward_length` [9216].
        public var ffnSize: Int
        /// `attention.head_count` [16].
        public var headCount: Int
        /// `attention.head_count_kv` [4].
        public var kvHeadCount: Int
        /// `attention.key_length` [256].
        public var keyLength: Int
        /// `attention.value_length` [256].
        public var valueLength: Int
        /// `attention.layer_norm_rms_epsilon` [1e-6].
        public var rmsNormEpsilon: Float
        /// `rope.freq_base` [1e7].
        public var ropeFreqBase: Float
        /// `rope.dimension_count`: rotated dims per head [64].
        public var ropeDimensionCount: Int
        /// `rope.dimension_sections` (IMROPE; llama.cpp gives a text token at
        /// position p the section positions [p, p, p, 0]) [11, 11, 10, 0].
        public var ropeSections: [Int]
        /// `full_attention_interval`: layer `i` is full attention iff
        /// `(i + 1) % interval == 0`, otherwise Gated DeltaNet [4].
        public var fullAttentionInterval: Int
        /// `ssm.conv_kernel` [4].
        public var ssmConvKernel: Int
        /// `ssm.state_size`: key/value head dim of the delta rule [128].
        public var ssmStateSize: Int
        /// `ssm.group_count`: q/k heads of the delta rule [16].
        public var ssmGroupCount: Int
        /// `ssm.time_step_rank`: value heads of the delta rule [32].
        public var ssmTimeStepRank: Int
        /// `ssm.inner_size`: value width, `timeStepRank * stateSize` [4096].
        public var ssmInnerSize: Int
        /// `context_length` [262144].
        public var contextLength: Int
        /// `nextn_predict_layers` (MTP blocks after `blockCount`) [0].
        public var nextnPredictLayers: Int
        /// Vocabulary size (rows of `token_embd.weight`) [248320].
        public var vocabSize: Int
        /// True when there is no `output.weight` and the LM head reuses
        /// `token_embd.weight` [true].
        public var tiedEmbeddings: Bool

        public init(blockCount: Int, hiddenSize: Int, ffnSize: Int, headCount: Int,
                    kvHeadCount: Int, keyLength: Int, valueLength: Int, rmsNormEpsilon: Float,
                    ropeFreqBase: Float, ropeDimensionCount: Int, ropeSections: [Int],
                    fullAttentionInterval: Int, ssmConvKernel: Int, ssmStateSize: Int,
                    ssmGroupCount: Int, ssmTimeStepRank: Int, ssmInnerSize: Int,
                    contextLength: Int, nextnPredictLayers: Int, vocabSize: Int,
                    tiedEmbeddings: Bool) {
            self.blockCount = blockCount
            self.hiddenSize = hiddenSize
            self.ffnSize = ffnSize
            self.headCount = headCount
            self.kvHeadCount = kvHeadCount
            self.keyLength = keyLength
            self.valueLength = valueLength
            self.rmsNormEpsilon = rmsNormEpsilon
            self.ropeFreqBase = ropeFreqBase
            self.ropeDimensionCount = ropeDimensionCount
            self.ropeSections = ropeSections
            self.fullAttentionInterval = fullAttentionInterval
            self.ssmConvKernel = ssmConvKernel
            self.ssmStateSize = ssmStateSize
            self.ssmGroupCount = ssmGroupCount
            self.ssmTimeStepRank = ssmTimeStepRank
            self.ssmInnerSize = ssmInnerSize
            self.contextLength = contextLength
            self.nextnPredictLayers = nextnPredictLayers
            self.vocabSize = vocabSize
            self.tiedEmbeddings = tiedEmbeddings
        }

        public enum LayerKind: String, Codable, Sendable {
            case fullAttention
            case gatedDeltaNet
        }

        /// Kind of main block `index` (0-based, `0..<blockCount`).
        public func layerKind(_ index: Int) -> LayerKind {
            (index + 1) % fullAttentionInterval == 0 ? .fullAttention : .gatedDeltaNet
        }

        /// Width of the delta-rule conv input: q, k (group heads) and v.
        public var ssmConvChannels: Int { 2 * ssmGroupCount * ssmStateSize + ssmInnerSize }

        /// Indices of full-attention blocks, ascending.
        public var fullAttentionLayers: [Int] { (0..<blockCount).filter { layerKind($0) == .fullAttention } }

        /// Indices of Gated DeltaNet blocks, ascending.
        public var gatedDeltaNetLayers: [Int] { (0..<blockCount).filter { layerKind($0) == .gatedDeltaNet } }
    }
}
