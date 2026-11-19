import Foundation

/// The numbers of the D1 bar and the per-op tolerances. Their justification
/// (llama.cpp's own noise floor at the pin) is in `docs/NATIVE_PARITY.md`,
/// "ε and tolerances". D1 itself is the maintainer's: 99% greedy agreement,
/// per-token KL ≤ ε, agentic parity. Only ε, the mean-KL check and the per-op
/// tolerances are set here.
struct ParityTolerances: Equatable {
    /// D1: greedy agreement over every generated position of the prompt set.
    var minimumAgreement: Double
    /// ε: the largest per-token KL(golden || candidate), in nats, at any
    /// position.
    var maxTokenKLD: Double
    /// Mean per-token KL over the set: a tighter check on the bulk.
    var meanKLD: Double

    /// The D1 bar for a native backend against the llama.cpp Metal golden.
    ///
    /// llama.cpp's own spread, teacher-forced on quick-v1:
    /// - Metal configuration changes (rerun, ubatch 128, flash attention off):
    ///   max per-token KL ≤ 5.8e-6, agreement ≥ 99.96%.
    /// - llama.cpp's CPU backend (Q8_K-quantized activations): mean 2.4e-3,
    ///   p99 1.7e-2, max 0.29, agreement 98.17%. It fails D1.
    ///
    /// ε sits more than 1,000 times above the Metal noise, leaving room for
    /// another accumulation order and staging precision, and below the CPU
    /// backend's p99, so a path as far from ggml-metal as the CPU backend
    /// fails.
    static let d1 = ParityTolerances(minimumAgreement: 0.99, maxTokenKLD: 0.01, meanKLD: 0.001)
}

/// Per-op tolerance classes. The **local** bound applies to a component fed
/// the golden's own inputs at an op boundary (`GoldenTapMap.Boundary`) and
/// compared with the golden output: that is the per-op gate. The
/// **accumulated** bound is a diagnostic for tap points of an end-to-end run,
/// where every block's input already carries the earlier blocks' differences;
/// the D1 logits bar is the gate there.
enum OpToleranceClass: String, CaseIterable {
    /// Embedding lookup (dequantize a row).
    case exact
    /// RMSNorm outputs.
    case norm
    /// Quantized GEMM outputs: projections, FFN, LM head.
    case gemm
    /// Attention / GDN core outputs (F16 KV or recurrent state involved).
    case mixer
    /// Residual adds.
    case residual
    /// GDN recurrent state after a step.
    case state

    /// Largest allowed `||c - g|| / ||g||` for a component fed golden inputs.
    ///
    /// Measured locally at block 0 of the CPU-backend run, whose input (the
    /// embedding) is bit-exact against the golden:
    /// - RMSNorm: 6.7e-8.
    /// - GEMMs: 3.6e-3 to 5.4e-3 (`attn_qkv`, `attn_gate`).
    /// - GDN core: 4.9e-3 to 8.2e-3.
    /// The bounds are about two times those, and float rounding for the
    /// lossless ops.
    var maxLocalRelativeL2: Double {
        switch self {
        case .exact: return 1e-6
        case .norm: return 1e-5
        case .gemm: return 1e-2
        case .mixer: return 2e-2
        case .residual: return 1e-5
        case .state: return 2e-2
        }
    }

    /// Diagnostic bound for an end-to-end run's tap points: llama.cpp's CPU
    /// backend reaches 4.7e-2 to 5.0e-2 at the last block, final norm and
    /// logits.
    static let maxAccumulatedRelativeL2 = 0.06

    /// The class of a llama.cpp graph tensor base name (`docs/NATIVE_PARITY.md`).
    static func of(base: String) -> OpToleranceClass {
        switch base {
        case "model.input_embed": return .exact
        case "attn_norm", "attn_post_norm", "result_norm": return .norm
        case "Qcur_full", "Kcur", "Vcur", "linear_attn_qkv_mixed", "z", "alpha", "beta",
             "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_out", "attn_output", "linear_attn_out", "result_output":
            return .gemm
        case "attn_gated", "final_output": return .mixer
        case "l_out": return .residual
        case "conv_states", "state_predelta": return .state
        default: return .gemm
        }
    }
}
