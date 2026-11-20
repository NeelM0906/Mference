import Darwin
import Foundation

// MARK: Memory admission seam
//
// The multi-prefix cache takes no live-memory reading of its own. Every
// retained region is admitted through a gate. The server passes
// `PrefixCacheMemoryGovernor`, which refuses growth under memory pressure or
// swap growth; `UnlimitedPrefixCacheMemoryGate` leaves only the cache's own
// byte ceiling in force.

/// Admits one retained region of `bytes`, or refuses it.
public protocol PrefixCacheMemoryGate: AnyObject {
    /// False while the gate refuses any growth whatever its size: the cache
    /// then keeps nothing new and evicts nothing to make room.
    var admitsGrowth: Bool { get }
    func reserve(bytes: Int) -> (any PrefixCacheMemoryGrant)?
}

extension PrefixCacheMemoryGate {
    public var admitsGrowth: Bool { true }
}

/// One admitted region. `commit()` once its pages are resident (after the
/// copy into it); `release()` exactly once, when it is unmapped.
public protocol PrefixCacheMemoryGrant: AnyObject {
    func commit()
    func release()
}

public final class UnlimitedPrefixCacheMemoryGate: PrefixCacheMemoryGate {
    public init() {}
    private final class Grant: PrefixCacheMemoryGrant {
        func commit() {}
        func release() {}
    }
    public func reserve(bytes: Int) -> (any PrefixCacheMemoryGrant)? { Grant() }
}

// MARK: The multi-prefix cache

/// Retained turn-boundary prefixes for one session and one snapshotting
/// runner (the server's `--prompt-cache-mode multi-prefix`).
///
/// Each finished turn that `PromptPrefixCache.makeEntry` accepts becomes a
/// checkpoint: its entry, plus a copy of the runner's recurrent state at the
/// turn's end. Its KV rows are either still in the runner (the live lineage:
/// full-attention KV is append-only, so rows below a turn's end stay valid
/// while the live conversation descends from it) or in an image, an
/// off-runner copy of rows [0, L) shared by checkpoints of one lineage.
///
/// A request continues from the checkpoint with the longest hit under the
/// unchanged per-entry rules. Restoring it puts the runner back bit for bit
/// in the state it held when that turn finished, so the request then runs
/// exactly the single-prefix server's `.resume` from that state. Before a
/// request writes rows a retained checkpoint still needs, those rows are
/// copied out to an image, or the checkpoints are dropped when the budget
/// refuses the copy.
///
/// The checkpoint the runner holds right now needs no copy to be used: with a
/// zero budget this is exactly the single-prefix cache.
///
/// Not thread-safe: the owning session actor serializes every call, and calls
/// it only while no generation is running.
public final class PrefixStateCache {
    final class Retained {
        let region: PrefixSnapshotRegion
        private let grant: any PrefixCacheMemoryGrant

        init(region: PrefixSnapshotRegion, grant: any PrefixCacheMemoryGrant) {
            self.region = region
            self.grant = grant
        }

        func commit() { grant.commit() }
        deinit { grant.release() }
    }

    final class Checkpoint {
        let id: Int
        let entry: PromptPrefixCacheEntry
        /// nil: the KV rows are in the runner's live buffers.
        var image: Int?
        /// nil only for the live checkpoint when no copy was admitted.
        var recurrent: Retained?
        var lastUsed: UInt64

        var position: Int { entry.kvPosition }

        init(id: Int, entry: PromptPrefixCacheEntry, lastUsed: UInt64) {
            self.id = id
            self.entry = entry
            self.lastUsed = lastUsed
        }
    }

    struct Image {
        let rows: Int
        let storage: Retained
    }

    /// What `begin` decided, for diagnostics and tests.
    public enum Outcome: String, Equatable, Sendable {
        case miss
        /// The runner already held the checkpoint's state.
        case live
        /// The live KV was rewound and the recurrent state copied back.
        case rewound
        /// An image's rows and the recurrent state were copied in.
        case restored
    }

    /// What the last request's start and finish cost (diagnostics).
    public struct Report: Sendable, Equatable {
        public var outcome: Outcome = .miss
        /// Seconds `begin` spent copying state out of and into the runner.
        public var beginSeconds: Double = 0
        /// KV row bytes copied out of the runner into a new image.
        public var displacedBytes = 0
        /// KV row and recurrent bytes copied into the runner.
        public var restoredBytes = 0
        /// Recurrent bytes copied out at the turn's end, and the seconds it took.
        public var capturedBytes = 0
        public var captureSeconds: Double = 0
    }

    public private(set) var lastReport = Report()

    public let ceilingBytes: Int
    private let gate: any PrefixCacheMemoryGate
    private(set) var checkpoints: [Checkpoint] = []
    private(set) var images: [Int: Image] = [:]
    /// The checkpoint whose exact state the runner holds now.
    private(set) var liveCheckpoint: Checkpoint?
    private var nextID = 0
    private var clock: UInt64 = 0

    public init(ceilingBytes: Int, gate: any PrefixCacheMemoryGate = UnlimitedPrefixCacheMemoryGate()) {
        self.ceilingBytes = max(0, ceilingBytes)
        self.gate = gate
    }

    public var checkpointCount: Int { checkpoints.count }
    public var imageCount: Int { images.count }

    /// Bytes mapped for images and recurrent copies.
    public var retainedBytes: Int {
        images.values.reduce(0) { $0 + $1.storage.region.mappedLength }
            + checkpoints.reduce(0) { $0 + ($1.recurrent?.region.mappedLength ?? 0) }
    }

    // MARK: Request start

    /// Chooses the retained prefix for a request and puts the runner in its
    /// state. Returns the effective prompt and how `runRawCompletion` starts.
    /// A restore that fails for any reason but cancellation falls back to a
    /// miss (the runner is reset, as the miss path would reset it). A throw
    /// leaves the runner needing `reset()`; call `didFail()`.
    public func begin(domain: PromptPrefixCacheDomain,
               turn: PromptPrefixCacheTurn,
               renderedPromptIDs: [Int32],
               tokenizer: MFTokenizer,
               maxContext: Int,
               runner: any PrefixStateSnapshotting) throws
        -> (effectivePromptIDs: [Int32], start: RawCompletionStart, outcome: Outcome) {
        let started = DispatchTime.now().uptimeNanoseconds
        lastReport = Report()
        defer {
            lastReport.beginSeconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        }
        // A request cancelled while queued must not cost the live
        // conversation its prefix: nothing below runs for it.
        try Task.checkCancellation()
        var best: (checkpoint: Checkpoint, effective: [Int32], cached: Int)?
        for checkpoint in checkpoints where checkpoint.recurrent != nil || checkpoint === liveCheckpoint {
            guard case .hit(let effective, let cached) = PromptPrefixCache.match(
                entry: checkpoint.entry, domain: domain, turn: turn,
                renderedPromptIDs: renderedPromptIDs, tokenizer: tokenizer),
                  // A hit that no longer fits is a miss, as on the server's
                  // single-prefix path (`fitting(maxContext:)`).
                  effective.count < maxContext,
                  // Retained state exists only at the turn's end.
                  cached == checkpoint.position else { continue }
            if let current = best, !Self.prefers(checkpoint, cached, over: current.checkpoint,
                                                current.cached, live: liveCheckpoint) {
                continue
            }
            best = (checkpoint, effective, cached)
        }

        guard let best else {
            displace(checkpoints.filter { $0.image == nil }, runner: runner, protecting: [])
            clearLive()
            lastReport.outcome = .miss
            return (renderedPromptIDs, .reset, .miss)
        }
        let chosen = best.checkpoint
        clock += 1
        chosen.lastUsed = clock
        let outcome: Outcome
        do {
            outcome = try resume(from: chosen, runner: runner)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // The live rows, if any were retained, were moved out before the
            // restore began; only the live lineage is lost, as on a failure.
            runner.reset()
            didFail()
            lastReport.outcome = .miss
            return (renderedPromptIDs, .reset, .miss)
        }
        lastReport.outcome = outcome
        return (best.effective, .resume(cachedPromptTokens: best.cached), outcome)
    }

    /// Puts the runner in `chosen`'s state, moving out first whatever live
    /// rows a retained checkpoint still needs.
    private func resume(from chosen: Checkpoint, runner: any PrefixStateSnapshotting) throws -> Outcome {
        let outcome: Outcome
        if let imageID = chosen.image {
            // Another conversation is live: its rows leave the runner first.
            displace(checkpoints.filter { $0.image == nil }, runner: runner, protecting: [chosen])
            clearLive()
            guard let image = images[imageID], let recurrent = chosen.recurrent else {
                throw PrefixSnapshotError.outOfRange("retained prefix lost its state")
            }
            // As on a miss: release the previous conversation's grown KV
            // before the image's rows grow it again to what they need.
            runner.reset()
            try runner.restorePrefixState(rows: (image.storage.region, image.rows),
                                          position: chosen.position,
                                          recurrentState: recurrent.region)
            lastReport.restoredBytes = runner.prefixKVRowBytes(tokens: image.rows)
                + runner.prefixRecurrentStateBytes
            // The image's checkpoints up to this one are live now; later ones
            // keep the image, since this request overwrites their rows.
            for checkpoint in checkpoints where checkpoint.image == imageID
                && checkpoint.position <= chosen.position {
                checkpoint.image = nil
            }
            dropUnusedImages()
            outcome = .restored
        } else {
            // A branch: the request overwrites rows of later live turns.
            displace(checkpoints.filter { $0.image == nil && $0.position > chosen.position },
                     runner: runner, protecting: [chosen])
            if chosen === liveCheckpoint {
                outcome = .live
            } else {
                guard let recurrent = chosen.recurrent else {
                    throw PrefixSnapshotError.outOfRange("retained prefix lost its state")
                }
                try runner.restorePrefixState(rows: nil, position: chosen.position,
                                              recurrentState: recurrent.region)
                lastReport.restoredBytes = runner.prefixRecurrentStateBytes
                outcome = .rewound
            }
            clearLive()
        }
        return outcome
    }

    /// Longest hit first; then the checkpoint the runner holds, then the
    /// checkpoint whose rows are live, then the most recently used.
    private static func prefers(_ candidate: Checkpoint, _ cached: Int,
                                over current: Checkpoint, _ currentCached: Int,
                                live: Checkpoint?) -> Bool {
        if cached != currentCached { return cached > currentCached }
        if (candidate === live) != (current === live) { return candidate === live }
        if (candidate.image == nil) != (current.image == nil) { return candidate.image == nil }
        return candidate.lastUsed > current.lastUsed
    }

    // MARK: Request end

    /// A finished turn: retains `entry` (nil when the turn cannot be
    /// continued) as the runner's live checkpoint, with a recurrent copy when
    /// the budget admits one.
    public func didFinish(entry: PromptPrefixCacheEntry?, runner: any PrefixStateSnapshotting) {
        clearLive()
        guard let entry, entry.kvPosition == runner.continuationPosition else { return }
        // A regenerated turn that ended identically replaces the older copy.
        checkpoints.removeAll { $0.entry == entry }
        dropUnusedImages()
        clock += 1
        let checkpoint = Checkpoint(id: nextID, entry: entry, lastUsed: clock)
        nextID += 1
        let started = DispatchTime.now().uptimeNanoseconds
        if let storage = allocate(bytes: runner.prefixRecurrentStateBytes, protecting: []) {
            do {
                try runner.capturePrefixRecurrentState(into: storage.region)
                storage.commit()
                checkpoint.recurrent = storage
                lastReport.capturedBytes = runner.prefixRecurrentStateBytes
                lastReport.captureSeconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
            } catch {
                // Kept without a copy: usable only while it stays live.
            }
        }
        checkpoints.append(checkpoint)
        liveCheckpoint = checkpoint
    }

    /// A request failed or was cancelled and the runner was reset: the live
    /// rows are gone. Images and their checkpoints survive.
    public func didFail() {
        liveCheckpoint = nil
        checkpoints.removeAll { $0.image == nil }
        dropUnusedImages()
    }

    /// Releases everything retained except the live checkpoint's entry, which
    /// stays usable without a copy. `PrefixCacheMemoryGovernor` calls it
    /// between requests after a memory-pressure event.
    public func shed() {
        let live = liveCheckpoint
        live?.recurrent = nil
        checkpoints.removeAll { $0 !== live }
        images.removeAll()
    }

    /// Evicts least recently used retained state until at most `target`
    /// bytes remain. The live checkpoint's entry stays usable without a copy.
    public func shrink(toRetainedBytes target: Int) {
        while retainedBytes > max(0, target) {
            guard evictLeastRecentlyUsed(protecting: []) else { return }
        }
    }

    // MARK: Internals

    /// The live checkpoint stops being the runner's state. One without a
    /// copy cannot be restored, so it goes.
    private func clearLive() {
        if let live = liveCheckpoint, live.recurrent == nil {
            checkpoints.removeAll { $0 === live }
        }
        liveCheckpoint = nil
    }

    /// Moves live checkpoints out of the runner before their rows are
    /// overwritten: into one new image when the budget admits it, dropped
    /// otherwise. A checkpoint without a recurrent copy cannot be kept.
    private func displace(_ moving: [Checkpoint], runner: any PrefixStateSnapshotting,
                          protecting: [Checkpoint]) {
        guard !moving.isEmpty else { return }
        let keep = moving.filter { $0.recurrent != nil }
        func drop(_ list: [Checkpoint]) {
            checkpoints.removeAll { candidate in
                list.contains { $0 === candidate } && candidate !== liveCheckpoint
            }
            for checkpoint in list where checkpoint === liveCheckpoint {
                checkpoint.recurrent = nil
            }
        }
        guard let rows = keep.map(\.position).max(), rows <= runner.continuationPosition,
              let storage = allocate(bytes: runner.prefixKVRowBytes(tokens: rows),
                                     protecting: keep + protecting) else {
            drop(moving)
            return
        }
        do {
            try runner.capturePrefixKVRows(count: rows, into: storage.region)
        } catch {
            drop(moving)
            return
        }
        storage.commit()
        lastReport.displacedBytes += runner.prefixKVRowBytes(tokens: rows)
        let imageID = nextID
        nextID += 1
        images[imageID] = Image(rows: rows, storage: storage)
        for checkpoint in keep { checkpoint.image = imageID }
        drop(moving.filter { $0.recurrent == nil })
    }

    private func dropUnusedImages() {
        let used = Set(checkpoints.compactMap(\.image))
        images = images.filter { used.contains($0.key) }
    }

    /// Maps a region of `bytes` within the ceiling and the gate, evicting
    /// least recently used retained state (never `protecting`) to make room.
    private func allocate(bytes: Int, protecting: [Checkpoint]) -> Retained? {
        guard bytes > 0, gate.admitsGrowth else { return nil }
        let page = Int(getpagesize())
        let mapped = (bytes + page - 1) / page * page
        // Evict nothing when even evicting everything unprotected would not
        // make room.
        let protectedImages = Set(protecting.compactMap(\.image))
        let protectedBytes = protectedImages.reduce(0) {
            $0 + (images[$1]?.storage.region.mappedLength ?? 0)
        } + protecting.reduce(0) { $0 + ($1.recurrent?.region.mappedLength ?? 0) }
        guard protectedBytes + mapped <= ceilingBytes else { return nil }
        while retainedBytes + mapped > ceilingBytes {
            guard evictLeastRecentlyUsed(protecting: protecting) else { return nil }
        }
        var grant = gate.reserve(bytes: mapped)
        while grant == nil {
            guard evictLeastRecentlyUsed(protecting: protecting) else { return nil }
            grant = gate.reserve(bytes: mapped)
        }
        guard let grant else { return nil }
        guard let region = try? PrefixSnapshotRegion(byteCount: bytes) else {
            grant.release()
            return nil
        }
        return Retained(region: region, grant: grant)
    }

    /// Frees the least recently used unit: a whole image with its
    /// checkpoints, or one live checkpoint's recurrent copy.
    private func evictLeastRecentlyUsed(protecting: [Checkpoint]) -> Bool {
        let protectedImages = Set(protecting.compactMap(\.image))
        var victim: (lastUsed: UInt64, image: Int?, checkpoint: Checkpoint?)?
        for (imageID, _) in images where !protectedImages.contains(imageID) {
            let members = checkpoints.filter { $0.image == imageID }
            guard !members.contains(where: { member in protecting.contains { $0 === member } }) else {
                continue
            }
            let lastUsed = members.map(\.lastUsed).max() ?? 0
            if victim == nil || lastUsed < victim!.lastUsed { victim = (lastUsed, imageID, nil) }
        }
        for checkpoint in checkpoints where checkpoint.image == nil && checkpoint.recurrent != nil
            && !protecting.contains(where: { $0 === checkpoint }) {
            if victim == nil || checkpoint.lastUsed < victim!.lastUsed {
                victim = (checkpoint.lastUsed, nil, checkpoint)
            }
        }
        guard let victim else { return false }
        if let imageID = victim.image {
            checkpoints.removeAll { $0.image == imageID }
            images[imageID] = nil
        } else if let checkpoint = victim.checkpoint {
            if checkpoint === liveCheckpoint {
                checkpoint.recurrent = nil
            } else {
                checkpoints.removeAll { $0 === checkpoint }
            }
        }
        return true
    }
}
