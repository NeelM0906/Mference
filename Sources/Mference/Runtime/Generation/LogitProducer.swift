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

/// Optional, server-only recovery. Implementations must reject other model
/// families and unfinished GPU state; ordinary continuation stays exact.
public enum GemmaPrefixRecoverySource: String, Sendable { case current, snapshot }

public protocol GemmaPrefixRecovering: ContinuableLogitProducer {
    var supportsGemmaPrefixRecovery: Bool { get }
    var gemmaRecoveryBytes: UInt64 { get }
    func gemmaRecoverablePrefix(upTo limit: Int) -> Int
    func captureGemmaPrefix() throws -> Bool
    func recoverGemmaPrefix(to position: Int) throws -> GemmaPrefixRecoverySource
    func discardGemmaPrefix()
}
