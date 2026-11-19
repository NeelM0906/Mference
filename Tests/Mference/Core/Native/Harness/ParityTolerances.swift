import Foundation
@testable import Mference

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
    /// Mean per-token KL over the set: a tighter check on the bulk. An extra
    /// gate beyond D1's "per-token KL ≤ ε": PENDING MAINTAINER SIGN-OFF at the
    /// Phase 0 exit.
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

/// Per-op tolerance classes. The bound applies to a component fed the
/// golden's own inputs at an op boundary (`GoldenTapMap.Boundary`, or a tap
/// point whose input is golden) and compared with the golden output: that is
/// the per-op gate. Tap points of an end-to-end run accumulate every earlier
/// block's difference and have no per-op bound; the D1 logits bar is their
/// gate (`docs/NATIVE_PARITY.md`, "Per-op bounds").
enum OpToleranceClass: String, CaseIterable {
    /// Embedding lookup (dequantize a row).
    case exact
    /// RMSNorm outputs.
    case norm
    /// Quantized GEMM outputs: projections, FFN, LM head.
    case gemm
    /// Attention / GDN core outputs (F16 KV or recurrent state involved).
    case mixer
    /// Element-wise glue: residual adds, SwiGLU, copies.
    case glue
    /// GDN recurrent state after a step.
    case state

    /// Largest allowed `||c - g|| / ||g||` for a component fed golden inputs.
    /// PROVISIONAL until the perturbation runs confirm them
    /// (`mfe-golden --perturb`); the derivation is in `docs/NATIVE_PARITY.md`.
    ///
    /// The bounds are chosen so that meeting them makes D1 likely, not from a
    /// backend that fails it. llama.cpp's CPU backend has a local GEMM error
    /// of about 4.5e-3 (block 0, input bit-exact) and misses D1 (98.17%
    /// agreement, max per-token KL 0.29, mean 2.4e-3). With KL scaling as the
    /// square of the per-op error and flips linearly, every one of its
    /// differences attributed to the GEMMs (conservative), a GEMM error of
    /// 1e-3 predicts:
    /// - flips 0.41% (agreement 99.6%)
    /// - mean KL 1.2e-4
    /// - max KL 0.014, borderline, which the perturbation runs must settle
    /// The golden's own half-precision staging in ggml-metal's prefill GEMM
    /// is about 3e-4, so a bound much below 1e-3 would fail correct F32
    /// kernels. The mixer and state bounds are 2e-3 (fewer, larger ops); the
    /// lossless ops keep float rounding.
    var maxLocalRelativeL2: Double {
        switch self {
        case .exact: return 1e-6
        case .norm: return 1e-5
        case .gemm: return 1e-3
        case .mixer: return 2e-3
        case .glue: return 1e-5
        case .state: return 2e-3
        }
    }

    /// The class of a llama.cpp graph tensor, from its base name and, where
    /// the name is ambiguous, its ggml op (as `golden.json` records it).
    /// `attn_output` is the `attn_output` GEMM in a full-attention block
    /// (`MUL_MAT`) but the raw delta-rule output in a GDN block (`VIEW`).
    /// Nil for a name the harness does not classify: never guessed.
    static func of(base: String, op: String? = nil) -> OpToleranceClass? {
        switch base {
        case "model.input_embed": return .exact
        case "attn_norm", "attn_post_norm", "result_norm": return .norm
        case "Qcur_full", "Kcur", "Vcur", "linear_attn_qkv_mixed", "z", "alpha", "beta",
             "ffn_gate", "ffn_up", "ffn_out", "linear_attn_out", "result_output":
            return .gemm
        case "attn_output":
            switch op {
            case "MUL_MAT": return .gemm
            case "VIEW": return .mixer
            default: return nil
            }
        case "attn_gated", "final_output": return .mixer
        case "ffn_swiglu", "l_out": return .glue
        case "conv_states", "state_predelta": return .state
        default: return nil
        }
    }

    /// The class of a captured golden tensor.
    static func of(_ tensor: GoldenManifest.Tensor) -> OpToleranceClass? {
        of(base: tensor.base, op: tensor.op)
    }

    /// The class a component's output at a `Native.TapPoint` is held to. A
    /// fused kernel is held to the class of its last contracted op: a GEMM
    /// fused with the residual add is `gemm`; a fused RMSNorm+GEMM is `gemm`.
    static func of(_ point: Native.TapPoint) -> OpToleranceClass {
        switch point {
        case .embedding: return .exact
        case .mixerInput, .finalNorm: return .norm
        case .mixerOutput, .ffnOutput, .logits: return .gemm
        case .layerOutput: return .glue
        }
    }

    /// The class of an op boundary (`GoldenTapMap.Boundary`).
    static func of(_ boundary: GoldenTapMap.Boundary) -> OpToleranceClass {
        switch boundary {
        case .attentionQGate, .attentionK, .attentionV, .gdnQKV, .gdnZ, .gdnAlpha, .gdnBeta,
             .ffnGate, .ffnUp, .ffnDown:
            return .gemm
        case .attentionOutput, .gdnOutput: return .mixer
        case .gdnConvState, .gdnState: return .state
        case .ffnInput: return .norm
        case .ffnSwiGLU: return .glue
        }
    }
}
