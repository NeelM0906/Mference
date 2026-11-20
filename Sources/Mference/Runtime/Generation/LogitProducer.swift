import Metal

/// Produces next-token logits for the `Generator`. The production
/// implementation is `RealForwardRunner`; tests use scripted logits so decode
/// behavior stays independent of the kernel stack.
public protocol LogitProducer: AnyObject, Sendable {
    /// Clear any per-generation state, such as KV cache.
    func reset()
    /// Run one token at `position`, leaving FP16 logits in `logits`.
    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws
}

public protocol ContinuableLogitProducer: LogitProducer {
    var continuationPosition: Int { get }
    func prepareForContinuation(expectedPosition: Int) throws
}

protocol FusedHeadLogitProducer: LogitProducer {
    var usesFusedGreedyHead: Bool { get }
    var lastGreedyToken: UInt32 { get }
}

protocol ContextWindowReporting: Sendable {
    var maxContext: Int { get }
}

public enum PrefillOutputMode: Sendable, Equatable {
    case logits
    case greedyIfAvailable
}

public enum PrefillSeed: Sendable, Equatable {
    case logitsWritten
    case greedyToken(UInt32)
}

public struct PrefillResult: Sendable, Equatable {
    public let newPosition: Int
    public let seed: PrefillSeed
    /// Nil means the producer does not report execution, never an inferred batch.
    public let execution: PrefillExecutionReport?

    public init(newPosition: Int, seed: PrefillSeed,
                execution: PrefillExecutionReport? = nil) {
        self.newPosition = newPosition
        self.seed = seed
        self.execution = execution
    }
}

protocol ChunkedPrefillRunner: LogitProducer {
    /// Prefill a prompt slice using the chunked production runtime.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult
}

protocol HeadlessSequentialPrefillRunner: LogitProducer {
    /// Advance one prompt token without producing vocabulary logits.
    func produceWithoutLogits(token: Int32, position: Int) async throws
}

/// Produces the final prompt-token logits without enabling a decode-only
/// approximate head.
protocol ExactPrefillLogitProducer: LogitProducer {
    func produceExactPrefill(token: Int32, position: Int, into logits: MTLBuffer) async throws
}

/// Optional capability behind the server's multi-prefix cache: copy the
/// complete numerical state at the current position out of the runner, and
/// put such a copy back. A restore must leave the runner bit-for-bit in the
/// state it held when the copy was taken, so continuing from it is the same
/// computation as continuing the live runner. Every call requires a drained,
/// clean runner (no prefill chunk in flight). Runners that cannot guarantee
/// that report `supportsPrefixStateSnapshots == false`.
public protocol PrefixStateSnapshotting: ContinuableLogitProducer {
    var supportsPrefixStateSnapshots: Bool { get }
    /// Bytes of a KV row image covering positions [0, tokens).
    func prefixKVRowBytes(tokens: Int) -> Int
    /// Bytes of one recurrent-state checkpoint.
    var prefixRecurrentStateBytes: Int { get }
    /// Copies KV rows [0, count) (`count <= continuationPosition`).
    func capturePrefixKVRows(count: Int, into region: PrefixSnapshotRegion) throws
    /// Copies the recurrent state at `continuationPosition`.
    func capturePrefixRecurrentState(into region: PrefixSnapshotRegion) throws
    /// Restores a prefix ending at `position`. With `rows`, KV rows
    /// [0, rows.count) are copied in first (`position <= rows.count`); without,
    /// the live rows below `position` must still be the prefix's rows and only
    /// the cursor moves back (`position <= continuationPosition`). Then the
    /// recurrent state is replaced. A throw after copying began requires
    /// `reset()`.
    func restorePrefixState(rows: (region: PrefixSnapshotRegion, count: Int)?,
                            position: Int,
                            recurrentState: PrefixSnapshotRegion) throws
}
