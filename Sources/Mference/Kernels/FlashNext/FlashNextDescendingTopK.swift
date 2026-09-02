import Foundation

/// `torch.topk(v, k, largest: true)`'s CPU result order, including its tie
/// behaviour — the selection policy the Flash-Next QSA indexer inherits.
///
/// # Why this is on the CPU at all
///
/// The indexer's block scores decide which KV a layer may read, and a flipped
/// selection is not a rounding error: it changes the attention support set, so
/// the port's gate is *exact* selection against the reference, not a tolerance.
/// The score computation is GPU work (a few thousand blocks per query), but the
/// ranking is a few thousand FP32 values read back — cheap next to the attention
/// it gates, and the only way to reproduce libc++'s `nth_element` short-circuits
/// bit for bit. Relu-zero ties at small block counts are real and common, so
/// "close enough" ordering is not available.
///
/// # The algorithm
///
/// PyTorch's CPU topk takes the `std::nth_element` branch whenever `k * 64 > n`
/// — always, at these widths — calling `nth_element(begin, begin + k - 1, end,
/// greater)` and reading the first `k` entries. libc++'s `__nth_element`
/// short-circuits by length: `2` swaps if out of order, `3` runs `__sort3`,
/// `<= 7` runs `__selection_sort` (first maximum, swapped into place — the swap
/// displaces whatever was there, which is why ties do NOT resolve to
/// lowest-index-first), and larger ranges run a median-of-3 quickselect.
///
/// Only the first three branches are reproduced; above 7 candidates the
/// documented lowest-index-first fallback applies, and it is unobservable unless
/// a tie occurs there. This is a transcription of
/// `FlashNextReferenceRunner.descendingTopK`, which is oracle code and is not
/// modified to share it; `FlashNextIndexerReferenceTieBackTests` discharges the
/// transcription against the runner's own captured selections.
enum FlashNextDescendingTopK {

    /// The indices `torch.topk` puts in the first `k` slots, in its order.
    ///
    /// Note the `k >= n` short-circuit returns **identity order**, not a
    /// descending sort — a real divergence from `torch.topk` that is
    /// unobservable here because the indexer sorts the chosen positions
    /// afterwards. `RouterWideTopK10Tests` pins the same property.
    static func indices(_ values: [Float], k: Int) -> [Int] {
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

    /// Whether the k-th and (k+1)-th largest scores are bit-equal, i.e. the
    /// selection boundary is decided by the ordering policy rather than by the
    /// model. Callers that need to attribute a parity failure record this.
    static func boundaryIsTied(_ values: [Float], k: Int) -> Bool {
        guard values.count > k, k >= 1 else { return false }
        let ranked = values.sorted(by: >)
        return ranked[k - 1] == ranked[k]
    }
}
