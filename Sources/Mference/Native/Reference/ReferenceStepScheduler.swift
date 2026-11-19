import Foundation
import Metal

extension Native {
    /// Builds the reference scheduler over any model, sampler and strategy.
    public final class ReferenceStepSchedulerFactory: StepSchedulerFactory {
        public init() {}

        public func make(model: sending any ForwardModel, sampler: sending any Sampler,
                         strategy: sending any DecodingStrategy, queue: MTLCommandQueue) throws -> any StepScheduler {
            ReferenceStepScheduler(model: model, sampler: sampler, strategy: strategy, queue: queue)
        }
    }

    /// A dedicated high-QoS thread that runs submitted work in order.
    final class ReferenceGenerationThread: @unchecked Sendable {
        private final class State: @unchecked Sendable {
            let condition = NSCondition()
            var work: [@Sendable () -> Void] = []
            var stopped = false
        }

        private let state = State()

        init(name: String) {
            let state = self.state
            let thread = Thread {
                while true {
                    state.condition.lock()
                    while state.work.isEmpty && !state.stopped { state.condition.wait() }
                    if state.work.isEmpty && state.stopped {
                        state.condition.unlock()
                        return
                    }
                    let next = state.work.removeFirst()
                    state.condition.unlock()
                    next()
                }
            }
            thread.name = name
            thread.qualityOfService = .userInteractive
            thread.start()
        }

        func submit(_ work: @escaping @Sendable () -> Void) {
            state.condition.lock()
            state.work.append(work)
            state.condition.signal()
            state.condition.unlock()
        }

        deinit {
            state.condition.lock()
            state.stopped = true
            state.condition.signal()
            state.condition.unlock()
        }
    }

    /// The reference `StepScheduler`: sequential and unpipelined. Each step is
    /// one command buffer, committed and waited on before the next; the token
    /// is always chosen on the CPU with `Sampler.sample`. Prefill runs in
    /// `job.prefillChunk` chunks, the prompt is accepted into the sampler's
    /// penalty history (as the llama.cpp backend does), and with
    /// `reusePrefix` the longest cached prefix of `prompt.dropLast()` is
    /// attached and the prompt state is cached after prefill.
    ///
    /// A stop token ends generation and is neither emitted nor included in
    /// `GenerationOutcome.tokens` (matching the engine's llama.cpp backend).
    public final class ReferenceStepScheduler: StepScheduler, @unchecked Sendable {
        private let model: any ForwardModel
        private let sampler: any Sampler
        private let strategy: any DecodingStrategy
        private let queue: MTLCommandQueue
        let thread = ReferenceGenerationThread(name: "mference.native.reference.generation")
        private let cancelLock = NSLock()

        /// One submitted generation's cancellation flag (guarded by `cancelLock`).
        final class Generation: @unchecked Sendable {
            var cancelled = false
        }
        /// The generation executing on the thread, if any.
        private var active: Generation?
        /// Submitted generations not yet started, in submission order.
        private var pending: [Generation] = []

        init(model: any ForwardModel, sampler: any Sampler, strategy: any DecodingStrategy, queue: MTLCommandQueue) {
            self.model = model
            self.sampler = sampler
            self.strategy = strategy
            self.queue = queue
        }

        /// Cancels the running generation; when none has started yet, the
        /// next one to start (which then throws `cancelled` before its first
        /// step). Cancellation belongs to one generation: it never carries
        /// over to a later one.
        public func cancel() {
            cancelLock.withLock { (active ?? pending.first)?.cancelled = true }
        }

        var pendingCount: Int { cancelLock.withLock { pending.count } }

        private func isCancelled(_ g: Generation) -> Bool {
            cancelLock.withLock { g.cancelled }
        }

        public func run(_ job: GenerationJob,
                        onEvent: @escaping @Sendable (StepEvent) -> Void) async throws -> GenerationOutcome {
            let generation = Generation()
            cancelLock.withLock { pending.append(generation) }
            return try await withCheckedThrowingContinuation { continuation in
                thread.submit {
                    self.cancelLock.withLock {
                        self.pending.removeAll { $0 === generation }
                        self.active = generation
                    }
                    defer { self.cancelLock.withLock { self.active = nil } }
                    do {
                        continuation.resume(returning: try autoreleasepool {
                            try self.execute(job, generation, onEvent: onEvent)
                        })
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        }

        private func commitAndWait(_ cb: MTLCommandBuffer) throws {
            cb.commit()
            cb.waitUntilCompleted()
            if cb.status != .completed {
                let message = cb.error.map { "\($0)" } ?? "status \(cb.status.rawValue)"
                // State can no longer be trusted: clear the sequence.
                if let reset = queue.makeCommandBuffer() {
                    try? model.kv.reset(commandBuffer: reset)
                    reset.commit()
                    reset.waitUntilCompleted()
                }
                throw NativeError.commandBufferFailed(message)
            }
        }

        private func makeCommandBuffer() throws -> MTLCommandBuffer {
            guard let cb = queue.makeCommandBuffer() else { throw NativeError.resource("command buffer") }
            return cb
        }

        struct StepResult {
            var logits: TensorRef?
            var gpuStart: CFTimeInterval
            var gpuEnd: CFTimeInterval
            var encodeNanos: Double
        }

        /// Encode and run one step. Inside its own autorelease pool: the
        /// generation thread is long-lived, and an autoreleased command buffer
        /// keeps every buffer it used alive until its pool drains.
        private func step(_ plan: StepPlan) throws -> StepResult {
            try autoreleasepool {
                let cb = try makeCommandBuffer()
                try model.kv.beginStep(rows: plan.rows, kind: plan.kind)
                guard let encoder = cb.makeComputeCommandEncoder() else {
                    try undo(plan)
                    throw NativeError.resource("compute encoder")
                }
                let encodeStart = DispatchTime.now().uptimeNanoseconds
                let logits: TensorRef?
                do {
                    logits = try model.encodeStep(plan, encoder: encoder)
                } catch {
                    // beginStep already advanced the store: end the encoder
                    // without committing, then undo the step.
                    encoder.endEncoding()
                    try undo(plan)
                    throw error
                }
                let encodeNanos = Double(DispatchTime.now().uptimeNanoseconds - encodeStart)
                encoder.endEncoding()
                try commitAndWait(cb)
                return StepResult(logits: logits, gpuStart: cb.gpuStartTime, gpuEnd: cb.gpuEndTime,
                                  encodeNanos: encodeNanos)
            }
        }

        /// Roll the store back to before `plan`'s step (it was never committed).
        private func undo(_ plan: StepPlan) throws {
            let cb = try makeCommandBuffer()
            try model.kv.rollback(to: plan.startPosition, commandBuffer: cb)
            cb.commit()
            cb.waitUntilCompleted()
        }

        private func execute(_ job: GenerationJob, _ generation: Generation,
                             onEvent: @Sendable (StepEvent) -> Void) throws -> GenerationOutcome {
            guard strategy.maxDraftRows == 0 else {
                throw NativeError.unsupported("the reference scheduler runs plain decoding only (\(strategy.name))")
            }
            let prompt = job.promptTokens
            guard !prompt.isEmpty else { throw NativeError.invalidShape("empty prompt") }
            guard (1...model.maxPrefillChunk).contains(job.prefillChunk) else {
                throw NativeError.invalidShape("prefill chunk \(job.prefillChunk) (max \(model.maxPrefillChunk))")
            }
            let kv = model.kv
            guard prompt.count < kv.capacity else {
                throw NativeError.stateOutOfRange("prompt of \(prompt.count) tokens, capacity \(kv.capacity)")
            }
            if isCancelled(generation) { throw NativeError.cancelled }
            let start = DispatchTime.now().uptimeNanoseconds
            func seconds(since t: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t) / 1e9 }

            sampler.reset(params: job.params)
            strategy.reset(prompt: prompt)
            var reused = 0
            let setup = try makeCommandBuffer()
            try kv.reset(commandBuffer: setup)
            if job.reusePrefix, prompt.count > 1, let match = kv.lookupPrefix(Array(prompt.dropLast())) {
                try kv.attachPrefix(match, commandBuffer: setup)
                reused = match.length
            }
            try commitAndWait(setup)
            for t in prompt { sampler.accept(t) }

            var timings = StepTimings()
            var position = reused
            var logits: TensorRef?
            while position < prompt.count {
                let n = min(job.prefillChunk, prompt.count - position)
                let last = position + n == prompt.count
                let plan = StepPlan(kind: .prefill, startPosition: position, rows: n,
                                    input: .host(Array(prompt[position ..< position + n])),
                                    logits: last ? .last : .none)
                logits = try step(plan).logits
                position += n
                onEvent(.prefillProgress(done: position, total: prompt.count))
                if !last && isCancelled(generation) {
                    timings.prefillSeconds = seconds(since: start)
                    return GenerationOutcome(tokens: [], stopReason: .cancelled, promptTokenCount: prompt.count,
                                             reusedPrefixTokens: reused, timings: timings)
                }
            }
            timings.prefillSeconds = seconds(since: start)
            if job.reusePrefix {
                try autoreleasepool {
                    let cb = try makeCommandBuffer()
                    do {
                        _ = try kv.insertPrefix(prompt, commandBuffer: cb)
                        try commitAndWait(cb)
                    } catch NativeError.memoryBudgetExceeded {
                        // The prompt does not fit the prefix budget: generate without caching it.
                    }
                }
            }

            var generated: [Int32] = []
            var reason: StopReason = .maxTokens
            var previousEnd: Double?
            var gaps: [Double] = []
            var cpu: [Double] = []
            let decodeStart = DispatchTime.now().uptimeNanoseconds
            while true {
                if job.maxNewTokens <= generated.count { reason = .maxTokens; break }
                if isCancelled(generation) { reason = .cancelled; break }
                guard let current = logits else { throw NativeError.invalidShape("model returned no logits") }
                let stepStart = DispatchTime.now().uptimeNanoseconds
                let row = UnsafeBufferPointer(
                    start: current.buffer.contents().advanced(by: current.offset).assumingMemoryBound(to: Float.self),
                    count: current.cols)
                let token = sampler.sample(logits: row)
                if generated.isEmpty { timings.timeToFirstTokenSeconds = seconds(since: start) }
                if job.stopTokens.contains(token) { reason = .stopToken; break }
                sampler.accept(token)
                generated.append(token)
                onEvent(.token(token))
                if job.maxNewTokens <= generated.count { reason = .maxTokens; break }
                if kv.length >= kv.capacity { reason = .contextFull; break }
                if isCancelled(generation) { reason = .cancelled; break }
                let r = try step(StepPlan(kind: .decode, startPosition: kv.length, rows: 1,
                                          input: .host([token]), logits: .last))
                logits = r.logits
                timings.decodeSteps += 1
                if let previousEnd { gaps.append((r.gpuStart - previousEnd) * 1e6) }
                previousEnd = r.gpuEnd
                cpu.append((Double(DispatchTime.now().uptimeNanoseconds - stepStart) - r.encodeNanos) / 1e3
                           - (r.gpuEnd - r.gpuStart) * 1e6)
            }
            timings.decodeSeconds = seconds(since: decodeStart)
            func median(_ v: [Double]) -> Double {
                guard !v.isEmpty else { return 0 }
                let s = v.sorted()
                return s[s.count / 2]
            }
            timings.medianGPUGapMicroseconds = median(gaps)
            timings.medianSchedulerCPUMicroseconds = median(cpu)
            return GenerationOutcome(tokens: generated, stopReason: reason, promptTokenCount: prompt.count,
                                     reusedPrefixTokens: reused, timings: timings)
        }
    }
}
