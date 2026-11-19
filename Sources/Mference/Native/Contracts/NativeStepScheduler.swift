import Metal

extension Native {
    /// One generation request as the scheduler sees it (already tokenized).
    public struct GenerationJob: Sendable {
        public var promptTokens: [Int32]
        public var params: SamplingParams
        public var maxNewTokens: Int
        /// Tokens that end generation (EOS / end-of-generation ids).
        public var stopTokens: Set<Int32>
        /// Use and update the prefix cache.
        public var reusePrefix: Bool
        /// Prefill chunk rows (<= the model's `maxPrefillChunk`).
        public var prefillChunk: Int

        public init(promptTokens: [Int32], params: SamplingParams, maxNewTokens: Int,
                    stopTokens: Set<Int32>, reusePrefix: Bool, prefillChunk: Int) {
            self.promptTokens = promptTokens
            self.params = params
            self.maxNewTokens = maxNewTokens
            self.stopTokens = stopTokens
            self.reusePrefix = reusePrefix
            self.prefillChunk = prefillChunk
        }
    }

    public enum StopReason: String, Sendable, Equatable {
        case stopToken
        case maxTokens
        case cancelled
        case contextFull
    }

    /// Wall-clock and overhead figures for one generation.
    public struct StepTimings: Sendable, Equatable {
        public var prefillSeconds: Double
        public var timeToFirstTokenSeconds: Double
        public var decodeSeconds: Double
        public var decodeSteps: Int
        /// Median GPU idle time between consecutive decode command buffers.
        public var medianGPUGapMicroseconds: Double
        /// Median CPU time per decode step on the generation thread, excluding
        /// the model's own `encodeStep`.
        public var medianSchedulerCPUMicroseconds: Double
        public init(prefillSeconds: Double = 0, timeToFirstTokenSeconds: Double = 0,
                    decodeSeconds: Double = 0, decodeSteps: Int = 0,
                    medianGPUGapMicroseconds: Double = 0, medianSchedulerCPUMicroseconds: Double = 0) {
            self.prefillSeconds = prefillSeconds
            self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
            self.decodeSeconds = decodeSeconds
            self.decodeSteps = decodeSteps
            self.medianGPUGapMicroseconds = medianGPUGapMicroseconds
            self.medianSchedulerCPUMicroseconds = medianSchedulerCPUMicroseconds
        }
    }

    public struct GenerationOutcome: Sendable, Equatable {
        public var tokens: [Int32]
        public var stopReason: StopReason
        public var promptTokenCount: Int
        /// Prompt tokens restored from the prefix cache instead of prefilled.
        public var reusedPrefixTokens: Int
        public var timings: StepTimings
        public init(tokens: [Int32], stopReason: StopReason, promptTokenCount: Int,
                    reusedPrefixTokens: Int, timings: StepTimings) {
            self.tokens = tokens
            self.stopReason = stopReason
            self.promptTokenCount = promptTokenCount
            self.reusedPrefixTokens = reusedPrefixTokens
            self.timings = timings
        }
    }

    public enum StepEvent: Sendable {
        case prefillProgress(done: Int, total: Int)
        case token(Int32)
    }

    /// Runs generations over one `ForwardModel`, `Sampler` and
    /// `DecodingStrategy` on a dedicated high-QoS generation thread.
    ///
    /// It owns command-buffer creation and pipelining (step t+1 encoded and
    /// committed while step t runs, with GPU-side token choice when the
    /// sampler allows), chunked prefill, verify-row batching, prefix-cache use
    /// through the model's `KVStore`, and cancellation (checked between steps;
    /// a cancelled generation leaves the store consistent at a step boundary).
    /// Stop strings are the caller's (it detokenizes and calls `cancel`).
    ///
    /// Exactness: output is byte-identical to a sequential, unpipelined loop
    /// over the same components.
    ///
    /// Concurrency: the model, sampler, strategy and store are not `Sendable`.
    /// A scheduler takes them as `sending` arguments (`StepSchedulerFactory`),
    /// confines them to its generation thread, and is itself a
    /// `final class ...: @unchecked Sendable` whose only cross-thread state is
    /// its job hand-off and cancellation flag (lock- or atomic-protected).
    public protocol StepScheduler: AnyObject, Sendable {
        /// Run `job` to completion; `onEvent` is called on the generation
        /// thread in order. Throws `NativeError.cancelled` only if cancelled
        /// before the first step; later cancellation returns `.cancelled`.
        func run(_ job: GenerationJob, onEvent: @escaping @Sendable (StepEvent) -> Void) async throws -> GenerationOutcome
        /// Request cancellation of the running job (thread-safe, idempotent).
        func cancel()
    }

    /// Builds a scheduler over the given components (reference or fast).
    public protocol StepSchedulerFactory: AnyObject {
        func make(model: sending any ForwardModel, sampler: sending any Sampler,
                  strategy: sending any DecodingStrategy,
                  queue: MTLCommandQueue) throws -> any StepScheduler
    }
}
