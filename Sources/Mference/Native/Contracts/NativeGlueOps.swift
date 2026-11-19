import Metal

extension Native {
    /// Element-wise and normalization work between the contracted ops: the
    /// block pre-norms, residual adds and the SwiGLU of the FFN. All F32.
    ///
    /// The reference implements it in Phase 0; the fast implementation is a
    /// Phase 2 integration item (`docs/NATIVE_ENGINE.md`, "Glue").
    public protocol GlueOps: AnyObject {
        func prepare(config: Qwen35Config) throws
        /// `output[r] = input[r] / sqrt(mean(input[r]^2) + eps) * weight` (ggml `rms_norm` then `mul`).
        func encodeRMSNorm(_ encoder: MTLComputeCommandEncoder, input: TensorRef,
                           weight: QuantTensor, output: TensorRef) throws
        /// `output = a + b`.
        func encodeAdd(_ encoder: MTLComputeCommandEncoder, a: TensorRef, b: TensorRef,
                       output: TensorRef) throws
        /// `output = silu(gate) * up`.
        func encodeSwiGLU(_ encoder: MTLComputeCommandEncoder, gate: TensorRef, up: TensorRef,
                          output: TensorRef) throws
        /// Copy `source` into `destination` (same shape and dtype).
        func encodeCopy(_ encoder: MTLComputeCommandEncoder, source: TensorRef,
                        destination: TensorRef) throws
    }
}
