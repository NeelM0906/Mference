import Metal

/// Decode attribution for `MFERENCE_PHASES=1` (the CLI's phases report): what
/// the Inkling 2 runner's decode steps (one-token chunks) cost, summed.
/// The runner keeps it only while that variable is set (`tracesPhases`, read
/// once at load), so a run without it records nothing and times nothing new.
public struct Inkling2DecodePhaseStats: Sendable, Equatable {
    /// One-token chunks, as the decode timeline counts them: a prefill whose
    /// last chunk is one token adds one too (rare; balanced chunks differ by
    /// at most one token).
    public var decodeSteps = 0
    /// The runner's time per step, from before the first buffer is encoded
    /// to after every buffer completed and the step's deferred visits were
    /// collected.
    public var wallSeconds: Double = 0
    /// Of the wall time, the CPU in the runner's waits for its committed
    /// buffers to complete (a split staged visit's route readback is one).
    /// Waits inside the store are in `slotLoadSeconds`, not here.
    public var blockedSeconds: Double = 0
    /// The GPU's own execution time, `gpuStartTime` to `gpuEndTime` per
    /// command buffer, read after the buffers completed. A deferred visit's
    /// buffer waits on the GPU for its slot loads, and that wait can fall
    /// inside the buffer's span, so with deferred visits this includes it.
    public var gpuBusySeconds: Double = 0
    /// The GPU's idle time between a step's consecutive buffers.
    public var gpuGapSeconds: Double = 0
    /// Staged layers' decode visits, and of them those served deferred.
    public var stagedVisits = 0
    public var deferredVisits = 0
    public var hits = 0
    public var misses = 0
    /// The store's slot preparation: slot assignment, its waits for the
    /// slots' previous reads, and the miss reads. On the runner thread,
    /// inside the wall time, for split readbacks; on the decode worker,
    /// overlapped with the GPU, for deferred ones.
    public var slotLoadSeconds: Double = 0
    /// The split staged visits' route readbacks: committing the buffer and
    /// waiting for it, so mostly inside `blockedSeconds`, not added to it
    /// (0 for deferred visits).
    public var readbackSeconds: Double = 0

    public init() {}
}

/// A runner that keeps `Inkling2DecodePhaseStats` (the Inkling 2 runner,
/// whose class is internal).
public protocol Inkling2DecodePhaseReporting: AnyObject {
    var decodePhaseStats: Inkling2DecodePhaseStats { get }
}

extension Inkling2ForwardRunner: Inkling2DecodePhaseReporting {
    /// Adds one decode step, called with `tracesPhases` set, after the step's
    /// buffers completed and its deferred visits were collected. It reads the
    /// completed buffers' GPU times and the decode-cache deltas only: no
    /// wait, no commit, no GPU work.
    func recordDecodePhases(_ buffers: [MTLCommandBuffer], started: Double, ended: Double, blocked: Double,
                            before: DecodeCacheStats) {
        let spans = buffers.map { ($0.gpuStartTime, $0.gpuEndTime) }
        let after = decodeCacheStats
        decodePhaseStats.decodeSteps += 1
        decodePhaseStats.wallSeconds += max(0, ended - started)
        decodePhaseStats.blockedSeconds += blocked
        decodePhaseStats.gpuBusySeconds += spans.reduce(0.0) { $0 + max(0, $1.1 - $1.0) }
        decodePhaseStats.gpuGapSeconds += spans.indices.dropFirst()
            .reduce(0.0) { $0 + max(0, spans[$1].0 - spans[$1 - 1].1) }
        decodePhaseStats.stagedVisits += after.visits - before.visits
        decodePhaseStats.deferredVisits += after.deferred - before.deferred
        decodePhaseStats.hits += after.hits - before.hits
        decodePhaseStats.misses += after.misses - before.misses
        decodePhaseStats.slotLoadSeconds += after.seconds - before.seconds
        decodePhaseStats.readbackSeconds += after.readbackSeconds - before.readbackSeconds
    }
}
