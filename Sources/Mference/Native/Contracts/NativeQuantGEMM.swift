import Metal

extension Native {
    /// How many activation rows one GEMM call carries.
    public enum GEMMRowClass: Hashable, Sendable {
        /// 1 row: decode.
        case gemv
        /// 2...16 rows: speculative verify (flat cost per row is the goal).
        case verify(rows: Int)
        /// Any row count: prefill tiles.
        case prefill(rows: Int)
    }

    /// Whether a GEMM writes or adds into its output.
    public enum GEMMOutputMode: Sendable {
        /// `y = W x`.
        case overwrite
        /// `y += W x` (residual fusion).
        case accumulate
    }

    /// Quantized matrix products over packed weights.
    ///
    /// Semantics for every entry point: `y[r][o] = sum_i W[o][i] * x[r][i]`,
    /// where `W` is the tensor dequantized exactly as ggml's
    /// `dequantize_row_<type>` at the pin, `x` is `input` (F32,
    /// `rows x cols`) and `y` is `output` (F32, `rows x weight.rows`).
    /// Accuracy is judged against the reference and the golden per-op tensors
    /// within the tolerances in `docs/NATIVE_PARITY.md`.
    ///
    /// Layouts: the fast implementation supports, for each type it handles,
    /// the `PackLayoutTable` layout; the reference supports every layout.
    ///
    /// Encoding rules (all ops): encode compute work only, into the given
    /// encoder; never end it, commit, wait, or allocate per call. Pipelines are
    /// built in `prepare` (or the initializer), not while encoding.
    public protocol QuantGEMM: AnyObject {
        /// Types and layouts this implementation handles on `family`.
        func supports(type: GGMLType, layout: TileLayout, family: GPUFamilyClass) -> Bool
        /// Build every pipeline needed for `types` up front.
        func prepare(types: Set<GGMLType>) throws
        /// One row (`input.rows == 1`).
        func encodeGEMV(_ encoder: MTLComputeCommandEncoder, weight: QuantTensor,
                        input: TensorRef, output: TensorRef, mode: GEMMOutputMode) throws
        /// 2...16 rows.
        func encodeVerify(_ encoder: MTLComputeCommandEncoder, weight: QuantTensor,
                          input: TensorRef, output: TensorRef, mode: GEMMOutputMode) throws
        /// Any number of rows (prefill chunk).
        func encodePrefill(_ encoder: MTLComputeCommandEncoder, weight: QuantTensor,
                           input: TensorRef, output: TensorRef, mode: GEMMOutputMode) throws
        /// Embedding lookup: `output[r] = dequant(weight row tokens[r])` (F32,
        /// `tokens.count x weight.cols`), matching ggml `get_rows`.
        func encodeGetRows(_ encoder: MTLComputeCommandEncoder, weight: QuantTensor,
                           tokens: TokenSlot, output: TensorRef) throws
    }

    /// Per-GPU-family kernel choices (Splash-style calibration table). The
    /// pack layout is not a calibration choice: it is `PackLayoutTable`, the
    /// same on every family, so a pack is portable across Macs.
    public protocol KernelCalibration: AnyObject {
        var family: GPUFamilyClass { get }
        /// Opaque name of the kernel variant used for `type` at `rows`
        /// (recorded in benchmark tables).
        func variant(for type: GGMLType, rows: GEMMRowClass) -> String
    }
}
