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
    /// Computation, matching llama.cpp `qwen35` at the pin (H = value heads
    /// = `timeStepRank`, G = q/k heads = `groupCount`, d = `stateSize`):
    /// 1. causal depthwise conv1d (kernel `convKernel`) over `qkv` with the
    ///    conv state, then SiLU; the conv state becomes the last
    ///    `convKernel - 1` input rows
    /// 2. split the conv output into q [G x d], k [G x d], v [H x d]
    /// 3. L2-normalize q and k per head: `x / sqrt(sum(x^2) + eps)` with the
    ///    model's `rmsNormEpsilon`
    /// 4. `beta = sigmoid(beta)`, `g = softplus(alpha + dtBias) * a`, decay `exp(g)`
    /// 5. value head `h` uses q/k head `h % G` (tiled, as ggml's
    ///    `gated_delta_net`, not Hugging Face's `repeat_interleave`)
    /// 6. per row and head: `S *= exp(g)`; `delta = (v - S k) * beta`;
    ///    `S += delta k^T`; `o = S q / sqrt(d)` (S per `RecurrentLayout`)
    /// 7. per-head RMSNorm of `o` with `norm`, times `silu(z)`
    ///
    /// Output: F32 [rows x ssmInnerSize], the input of `ssm_out`. The op reads
    /// state bank `stateIn` and writes the state after all rows to `stateOut`
    /// (the two may alias; the `KVStore` passes distinct banks so the step
    /// can be undone without a copy). `encodeVerify` also writes the state
    /// after each row `r` into `arena` slot `r`.
    public protocol GDNOp: AnyObject {
        func prepare(config: Qwen35Config, layout: RecurrentLayout) throws
        func encodeDecode(_ encoder: MTLComputeCommandEncoder, weights: GDNLayerWeights,
                          qkv: TensorRef, z: TensorRef, alpha: TensorRef, beta: TensorRef,
                          stateIn: RecurrentStateRef, stateOut: RecurrentStateRef,
                          output: TensorRef) throws
        func encodeVerify(_ encoder: MTLComputeCommandEncoder, weights: GDNLayerWeights,
                          qkv: TensorRef, z: TensorRef, alpha: TensorRef, beta: TensorRef,
                          stateIn: RecurrentStateRef, stateOut: RecurrentStateRef,
                          arena: RecurrentVerifyArena, output: TensorRef) throws
        func encodePrefill(_ encoder: MTLComputeCommandEncoder, weights: GDNLayerWeights,
                           qkv: TensorRef, z: TensorRef, alpha: TensorRef, beta: TensorRef,
                           stateIn: RecurrentStateRef, stateOut: RecurrentStateRef,
                           output: TensorRef) throws
    }
}
