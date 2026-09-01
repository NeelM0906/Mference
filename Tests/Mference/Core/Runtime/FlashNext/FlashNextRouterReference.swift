import Foundation
@testable import Mference

/// The Flash-Next (`qwen4_exp` / `Qwen3NextSparseMoeBlock`) router, in float32,
/// extracted so the Metal router kernels can be gated against the same
/// arithmetic the CPU reference forward runs.
///
/// This is a transcription of `FlashNextReferenceRunner.sparseMoE`'s routing
/// block and of its `descendingTopK`, not a re-derivation. The runner is oracle
/// code and is not modified to share this; instead
/// `FlashNextRouterReferenceTieBackTests` proves the two agree by replaying the
/// runner's own captured `router_indices` / `router_weights` through this type.
///
/// Semantics (design doc, "MoE"): float32 softmax over **all** experts, top-k of
/// the **probs**, then renormalize the k to sum to 1 (`norm_topk_prob = true`).
enum FlashNextRouterReference {

    struct Selection: Equatable {
        let indices: [Int]
        /// Renormalized, summing to 1 over the k.
        let weights: [Float]
    }

    /// `probs = softmax_fp32(logits)`, exactly as the runner computes it: a
    /// max-subtraction, `expf`, one accumulation pass, then a division pass.
    static func probabilities(_ logits: [Float]) -> [Float] {
        precondition(!logits.isEmpty)
        var maximum = logits[0]
        for i in 1..<logits.count where logits[i] > maximum { maximum = logits[i] }
        var probs = [Float](repeating: 0, count: logits.count)
        var sum: Float = 0
        for i in 0..<logits.count {
            probs[i] = expf(logits[i] - maximum)
            sum += probs[i]
        }
        for i in 0..<logits.count { probs[i] /= sum }
        return probs
    }

    static func select(logits: [Float], k: Int) -> Selection {
        let probs = probabilities(logits)
        let chosen = descendingTopK(probs, k: k)
        var total: Float = 0
        for j in 0..<k { total += probs[chosen[j]] }
        return Selection(indices: chosen,
                         weights: (0..<k).map { probs[chosen[$0]] / total })
    }

    /// The smallest gap between the k-th and (k+1)-th largest probability. Zero
    /// means the selection boundary is a tie and the *ordering policy*, not the
    /// model, decides the outcome — a parity test that wants exact index
    /// agreement across two different top-k implementations must assert this is
    /// non-zero rather than hope.
    static func boundaryGap(logits: [Float], k: Int) -> Float {
        guard logits.count > k else { return .infinity }
        let ranked = probabilities(logits).sorted(by: >)
        return ranked[k - 1] - ranked[k]
    }

    /// Verbatim from `FlashNextReferenceRunner.descendingTopK` — including the
    /// libc++ `nth_element` short-circuit branches PyTorch's CPU `topk` takes at
    /// small `n`. At `n > 7` (every case the Metal kernels are gated at) the
    /// default branch is a stable descending sort with ties resolved
    /// lowest-index-first, which is exactly the Metal selection's tie rule.
    static func descendingTopK(_ values: [Float], k: Int) -> [Int] {
        var a = values.enumerated().map { (index: $0.offset, value: $0.element) }
        let n = a.count
        if k >= n { return a.map(\.index) }
        func greater(_ x: Int, _ y: Int) -> Bool { a[x].value > a[y].value }
        switch n {
        case 0, 1:
            break
        case 2:
            if greater(1, 0) { a.swapAt(0, 1) }
        case 3:
            if !greater(1, 0) {
                if greater(2, 1) {
                    a.swapAt(1, 2)
                    if greater(1, 0) { a.swapAt(0, 1) }
                }
            } else if greater(2, 1) {
                a.swapAt(0, 2)
            } else {
                a.swapAt(0, 1)
                if greater(2, 1) { a.swapAt(1, 2) }
            }
        case 4...7:
            for i in 0..<(n - 1) {
                var best = i
                for m in (i + 1)..<n where greater(m, best) { best = m }
                if best != i { a.swapAt(i, best) }
            }
        default:
            a.sort { $0.value != $1.value ? $0.value > $1.value : $0.index < $1.index }
        }
        return (0..<k).map { a[$0].index }
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
