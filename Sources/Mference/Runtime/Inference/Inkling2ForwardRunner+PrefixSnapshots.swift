import Foundation

/// Inkling 2's prefix snapshots for the multi-prefix cache (`PrefixStateCache`).
///
/// - KV rows: the 8 global visits' linear rows `[0, L)` (32 KiB a token),
///   append-only like Qwen 3.6's full-attention rows.
/// - Recurrent state: a light snapshot of `Inkling2AttentionState` at the
///   cursor: the convolution tails (3.8 MB) and the last `window - 1` rows of
///   the 40 window rings, about 86 MB at 500 tokens and 171 MB from 1,023 on.
///   The rings cannot be in a row image: a ring keeps only its last
///   `window + maxChunkTokens` positions.
///
/// Nothing else the runner keeps carries from one token to the next: the
/// loop and depth-mix arena, the decode partials and the other scratch are
/// written before they are read within a call, and the expert source holds
/// weights (the same bits from any slot), so a restore leaves the decode slot
/// caches as they are and only abandons visits or a prefetched read left
/// open. Every call needs a drained, clean runner (no chunk left dirty by a
/// failure: `reset()` first) and refuses once the expert source failed.
extension Inkling2ForwardRunner: PrefixStateSnapshotting {
    var supportsPrefixStateSnapshots: Bool { true }

    func prefixKVRowBytes(tokens: Int) -> Int { Inkling2AttentionState.globalRowBytes(tokens: tokens) }

    /// At the cursor, where `PrefixStateCache` reads it to capture.
    var prefixRecurrentStateBytes: Int { state.snapshotBytes(includeGlobalRows: false) }

    private func requireSnapshotReady(_ operation: String) throws {
        try Task.checkCancellation()
        try chunkState.requireClean(operation: operation)
        if let failure = expertsFailed {
            throw Inkling2ForwardRunnerError.commandFailed("Inkling 2 \(failure); reload the model")
        }
    }

    func capturePrefixKVRows(count: Int, into region: PrefixSnapshotRegion) throws {
        try requireSnapshotReady("prefix KV capture")
        try state.copyGlobalRows(count: count, into: region)
    }

    func capturePrefixRecurrentState(into region: PrefixSnapshotRegion) throws {
        try requireSnapshotReady("prefix recurrent capture")
        _ = try state.takeSnapshot(into: region, includeGlobalRows: false)
    }

    func restorePrefixState(rows: (region: PrefixSnapshotRegion, count: Int)?, position: Int,
                            recurrentState: PrefixSnapshotRegion) throws {
        try requireSnapshotReady("prefix restore")
        try state.checkPrefixRestore(globalRows: rows, position: position, light: recurrentState)
        // A staging read left armed must not outlive the prefix it was for: a
        // prefill whose last chunk threw before it began leaves the chunk
        // clean and the early read made for it still armed. (A prefill
        // cancelled between spans has given its read back already.)
        do {
            try w.experts.abandonOpenVisits()
        } catch {
            expertsFailed = "abandoning open expert visits failed: \(error)"
            throw Inkling2ForwardRunnerError.commandFailed("Inkling 2 \(expertsFailed!); reload the model")
        }
        try state.restorePrefix(globalRows: rows, position: position, light: recurrentState)
    }
}
