import Foundation

/// Arithmetic and batching choices for Gemma-family prefill.
///
/// The QAT checkpoint keeps MLX's FP16 reduction order (`sourceFP16`) for
/// decode, routing, normalization and sliding-window attention. Its three
/// expensive prefill matmul families (projections, shared expert, routed
/// experts) use the normal kernels: they change only floating-point summation
/// order, which measured as no perplexity change, and the source-order kernels
/// cost 3-5x at long prompts. Its full-attention prefill layers use the
/// tensor-ops kernel where it builds, which accumulates in FP32 where the
/// source rounds scores and probabilities to FP16.
struct GemmaPrefillPolicy: Equatable, Sendable {
    /// MLX FP16 reduction order outside the prefill matmuls (QAT only).
    let sourceFP16: Bool
    /// The shipped QAT profile for prefill matmuls; `MFERENCE_QAT_EXACT_PREFILL=1`.
    let prefillMatmulSourceFP16: Bool
    /// The shipped QAT profile for full-attention prefill; same switch.
    let prefillAttentionSourceFP16: Bool
    /// Batched shared expert and grouped-GEMM routed experts. Source arithmetic
    /// has no grouped form; `MFERENCE_GEMMA_PREFILL_LEGACY=1` also disables it.
    let batchedExperts: Bool

    init(modelID: String,
         environment: [String: String] = ProcessInfo.processInfo.environment) {
        sourceFP16 = modelID == CheckpointIdentity.gemma4QAT
        prefillMatmulSourceFP16 = sourceFP16 && environment["MFERENCE_QAT_EXACT_PREFILL"] == "1"
        prefillAttentionSourceFP16 = prefillMatmulSourceFP16
        batchedExperts = !prefillMatmulSourceFP16 && environment["MFERENCE_GEMMA_PREFILL_LEGACY"] != "1"
    }
}

/// Grouped GEMM computes whole 64-row matrix tiles per expert, so it pays only
/// while padding stays small. Measured on M2: ~17.5 us per padded row against
/// ~29.6 us per real row for the row kernel, a break-even near 1.69x padding.
enum PrefillGroupedExpertGate {
    static let tileRows = 64

    static func usesGroupedGEMM(pairCounts: [Int]) -> Bool {
        let real = pairCounts.reduce(0, +)
        guard real > 0 else { return false }
        let padded = pairCounts.reduce(0) { $0 + ($1 + tileRows - 1) / tileRows * tileRows }
        return padded * 2 <= real * 3
    }
}
