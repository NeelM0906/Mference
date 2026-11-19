import Metal

extension Native {
    /// Weights a Gated DeltaNet block's `GDNOp` reads (Quick shapes in
    /// brackets). The `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta` and
    /// `ssm_out` projections are `QuantGEMM` work outside the op.
    public struct GDNLayerWeights {
        /// `blk.N.ssm_conv1d.weight`, F32 ne [convKernel = 4, convChannels = 8192].
        public var conv1d: QuantTensor
        /// `blk.N.ssm_dt.bias`, F32 [timeStepRank = 32].
        public var dtBias: QuantTensor
        /// `blk.N.ssm_a`, F32 [timeStepRank = 32] (stored as `-exp(A_log)`).
        public var a: QuantTensor
        /// `blk.N.ssm_norm.weight`, F32 [stateSize = 128].
        public var norm: QuantTensor
        public init(conv1d: QuantTensor, dtBias: QuantTensor, a: QuantTensor, norm: QuantTensor) {
            self.conv1d = conv1d
            self.dtBias = dtBias
            self.a = a
            self.norm = norm
        }
    }

    /// The Gated DeltaNet (linear attention) core of one `qwen35` block, from
    /// projections to the gated-norm output, over the layer's recurrent state
    /// (`RecurrentLayout`).
    ///
    /// Inputs, all F32 with `rows` rows at consecutive positions:
    /// - `qkv`: output of `attn_qkv` [rows x convChannels] (q, k, v as llama.cpp orders them)
    /// - `z`: output of `attn_gate` [rows x ssmInnerSize]
    /// - `alpha`: output of `ssm_alpha` [rows x timeStepRank]
    /// - `beta`: output of `ssm_beta` [rows x timeStepRank]
    ///
    /// Computation, matching llama.cpp `qwen35` at the pin: causal conv1d over
    /// `qkv` with the conv state, SiLU; split into q, k, v; L2-normalize q and
    /// k per head; gate `g = softplus(alpha + dtBias) * a`, `beta = sigmoid(beta)`;
    /// delta-rule recurrence per value head (q/k head mapping as llama.cpp);
    /// per-head RMSNorm with `norm` times `silu(z)`.
    ///
    /// Output: F32 [rows x ssmInnerSize], the input of `ssm_out`. The live
    /// state is advanced past all rows. `encodeVerify` additionally writes the
    /// state after each row `r` into `arena` slot `r`.
    public protocol GDNOp: AnyObject {
        func prepare(config: Qwen35Config, layout: RecurrentLayout) throws
        func encodeDecode(_ encoder: MTLComputeCommandEncoder, weights: GDNLayerWeights,
                          qkv: TensorRef, z: TensorRef, alpha: TensorRef, beta: TensorRef,
                          state: RecurrentStateRef, output: TensorRef) throws
        func encodeVerify(_ encoder: MTLComputeCommandEncoder, weights: GDNLayerWeights,
                          qkv: TensorRef, z: TensorRef, alpha: TensorRef, beta: TensorRef,
                          state: RecurrentStateRef, arena: RecurrentVerifyArena,
                          output: TensorRef) throws
        func encodePrefill(_ encoder: MTLComputeCommandEncoder, weights: GDNLayerWeights,
                           qkv: TensorRef, z: TensorRef, alpha: TensorRef, beta: TensorRef,
                           state: RecurrentStateRef, output: TensorRef) throws
    }
}
