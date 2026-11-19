import Metal

extension Native {
    /// Weights a full-attention block's `AttentionOp` reads (Quick shapes in
    /// brackets). The q/k/v/output projections are `QuantGEMM` work outside
    /// the op.
    public struct AttentionLayerWeights {
        /// `blk.N.attn_q_norm.weight`, F32 [keyLength = 256].
        public var qNorm: QuantTensor
        /// `blk.N.attn_k_norm.weight`, F32 [keyLength = 256].
        public var kNorm: QuantTensor
        public init(qNorm: QuantTensor, kNorm: QuantTensor) {
            self.qNorm = qNorm
            self.kNorm = kNorm
        }
    }

    /// Full attention of one `qwen35` full-attention block, from projections to
    /// the gated output, over the paged KV (`KVPageGeometry`).
    ///
    /// Inputs, all F32 with `rows` rows at consecutive positions
    /// `startPosition ..< startPosition + rows` (the KV store has already
    /// reserved them, so `kv.tokenCount == startPosition + rows`):
    /// - `qGate`: output of `attn_q` [rows x headCount*keyLength*2]; per head
    ///   `h`, `q` is elements `h*2*keyLength ..< h*2*keyLength + keyLength` and
    ///   the output gate is the next `keyLength` elements
    /// - `k`: output of `attn_k` [rows x kvHeadCount*keyLength]
    /// - `v`: output of `attn_v` [rows x kvHeadCount*valueLength]
    ///
    /// Computation, matching llama.cpp `qwen35` at the pin:
    /// 1. per-head RMSNorm of q and k with `qNorm` / `kNorm` (eps from config)
    /// 2. RoPE on the first `ropeDimensionCount` dims of q and k at each row's
    ///    position (multi-section rope with identical text positions)
    /// 3. append k (after 1-2) and v to the KV pages at the rows' positions
    ///    (F16, or INT8 + scale when the geometry says so)
    /// 4. causal GQA attention of each row over positions `0...its own`,
    ///    scale `1/sqrt(keyLength)`, reading K/V from the pages
    /// 5. multiply by `sigmoid(gate)`
    ///
    /// Output: F32 [rows x headCount*valueLength], the input of `attn_output`.
    /// Decode is `rows == 1`, verify `2...16`, prefill any count.
    public protocol AttentionOp: AnyObject {
        func prepare(config: Qwen35Config, geometry: KVPageGeometry) throws
        func encodeDecode(_ encoder: MTLComputeCommandEncoder, attnLayer: Int,
                          weights: AttentionLayerWeights, qGate: TensorRef, k: TensorRef,
                          v: TensorRef, startPosition: Int, kv: KVPageTable,
                          output: TensorRef) throws
        func encodeVerify(_ encoder: MTLComputeCommandEncoder, attnLayer: Int,
                          weights: AttentionLayerWeights, qGate: TensorRef, k: TensorRef,
                          v: TensorRef, startPosition: Int, kv: KVPageTable,
                          output: TensorRef) throws
        func encodePrefill(_ encoder: MTLComputeCommandEncoder, attnLayer: Int,
                           weights: AttentionLayerWeights, qGate: TensorRef, k: TensorRef,
                           v: TensorRef, startPosition: Int, kv: KVPageTable,
                           output: TensorRef) throws
    }
}
