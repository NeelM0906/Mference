import Foundation
@testable import Mference

/// The Flash-Next (`Qwen3NextSparseMoeBlock`) **expert compute** in float32,
/// extracted so the Metal routed-expert kernels can be gated against the same
/// arithmetic the CPU reference forward runs.
///
/// This is a transcription of `FlashNextReferenceRunner.sparseMoE`'s expert
/// block — not a re-derivation, and not a copy of `MoeRef`, which activates with
/// `gelu_pytorch_tanh` and knows nothing about the sigmoid-gated shared expert.
/// The runner is oracle code and is not modified to share this; instead
/// `FlashNextExpertReferenceTieBackTests` proves the two agree by replaying the
/// runner's own captured `mlp_hc_mixed` / `router_*` through this type and
/// checking the result against its captured `moe_out`.
///
/// Semantics (design doc, "MoE"):
///
/// * `gate, up = chunk2(W_gate_up[e] . x)` with **gate first** — the split is
///   rows `[0, I)` and `[I, 2I)` of the fused tensor, which the installer has
///   already separated into `gate` and `up` sub-tensors.
/// * `expert_out = W_down[e] . (silu(gate) * up)`.
/// * The routed outputs are accumulated in **router rank order**, each scaled by
///   its renormalized weight; the shared expert is added last.
/// * Shared expert: a plain SwiGLU MLP at `intermediateSize`, scaled by
///   `sigmoid(shared_expert_gate . x)` — a single row, so a scalar per token.
///
/// Accumulation order is load-bearing: the runner adds `scale * y[d]` into `out`
/// one expert at a time, then the shared term. The Metal reduce seeds with the
/// residual and sums rank 0..k in the same order, so the two agree to float
/// rounding rather than by luck.
enum FlashNextExpertReference {

    /// One routed expert's three projections, row-major float32.
    struct Expert {
        let gate: [Float]   // [moeIntermediate, hidden]
        let up: [Float]     // [moeIntermediate, hidden]
        let down: [Float]   // [hidden, moeIntermediate]
    }

    /// The shared expert's weights, including its `[1, hidden]` gate row.
    struct Shared {
        let gateRow: [Float]    // [1, hidden]
        let gateProj: [Float]   // [intermediate, hidden]
        let upProj: [Float]     // [intermediate, hidden]
        let downProj: [Float]   // [hidden, intermediate]
    }

    static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
    static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }

    /// `down . (silu(gate . x) * (up . x))` for one expert.
    static func expertFFN(_ e: Expert, x: [Float],
                          hidden: Int, intermediate: Int) -> [Float] {
        precondition(x.count == hidden)
        var g = matVec(e.gate, rows: intermediate, cols: hidden, x: x)
        let u = matVec(e.up, rows: intermediate, cols: hidden, x: x)
        for i in 0..<intermediate { g[i] = silu(g[i]) * u[i] }
        return matVec(e.down, rows: hidden, cols: intermediate, x: g)
    }

    /// `sum over ranks of weight[rank] * expertFFN(experts[rank], x)`, seeded
    /// with `residual` and accumulated rank by rank — the order both the runner
    /// and `flashnext_moe_phase2_down_reduce_bf16` / `moe_phase2_down_reduce_k10`
    /// use.
    static func routedSum(experts: [Expert], weights: [Float],
                          x: [Float], residual: [Float],
                          hidden: Int, intermediate: Int) -> [Float] {
        precondition(experts.count == weights.count)
        precondition(residual.count == hidden)
        var out = residual
        for rank in 0..<experts.count {
            let y = expertFFN(experts[rank], x: x,
                              hidden: hidden, intermediate: intermediate)
            let scale = weights[rank]
            for d in 0..<hidden { out[d] += scale * y[d] }
        }
        return out
    }

    /// `sigmoid(gateRow . x) * down(silu(gate . x) * (up . x))`.
    static func sharedExpert(_ s: Shared, x: [Float],
                             hidden: Int, intermediate: Int) -> [Float] {
        var g = matVec(s.gateProj, rows: intermediate, cols: hidden, x: x)
        let u = matVec(s.upProj, rows: intermediate, cols: hidden, x: x)
        for i in 0..<intermediate { g[i] = silu(g[i]) * u[i] }
        let y = matVec(s.downProj, rows: hidden, cols: intermediate, x: g)
        let gate = sigmoid(matVec(s.gateRow, rows: 1, cols: hidden, x: x)[0])
        return y.map { gate * $0 }
    }

    /// The whole `Qwen3NextSparseMoeBlock` output for one token, in the runner's
    /// order: routed experts rank by rank into a zero accumulator, then the
    /// gated shared expert.
    static func block(experts: [Expert], weights: [Float], shared: Shared,
                      x: [Float], hidden: Int,
                      moeIntermediate: Int, sharedIntermediate: Int) -> [Float] {
        var out = routedSum(experts: experts, weights: weights, x: x,
                            residual: [Float](repeating: 0, count: hidden),
                            hidden: hidden, intermediate: moeIntermediate)
        let s = sharedExpert(shared, x: x,
                             hidden: hidden, intermediate: sharedIntermediate)
        for d in 0..<hidden { out[d] += s[d] }
        return out
    }

    /// `out[r] = W[r, :] . x`, the runner's `matVec` accumulation order.
    static func matVec(_ w: [Float], rows: Int, cols: Int, x: [Float]) -> [Float] {
        precondition(w.count == rows * cols && x.count == cols)
        var out = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            var acc: Float = 0
            let base = r * cols
            for c in 0..<cols { acc += w[base + c] * x[c] }
            out[r] = acc
        }
        return out
    }
}
