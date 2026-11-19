import Foundation
import Metal

extension Native {
    /// Builds the pipelined scheduler over any model, sampler and strategy.
    public final class PipelinedStepSchedulerFactory: StepSchedulerFactory {
        public init() {}

        public func make(model: sending any ForwardModel, sampler: sending any Sampler,
                         strategy: sending any DecodingStrategy, queue: MTLCommandQueue) throws -> any StepScheduler {
            try PipelinedStepScheduler(model: model, sampler: sampler, strategy: strategy, queue: queue)
        }
    }

    /// The fast `StepScheduler`. Its output is byte-identical to the
    /// sequential loop of `ReferenceStepScheduler` over the same components;
    /// what differs is how the GPU is kept busy:
    ///
    /// - **Pipelined decode with GPU-side choice.** When the sampler can choose
    ///   on the GPU (`canChooseOnGPU`, plain decoding), every step also encodes
    ///   `Sampler.encodeChoice` into a scheduler-owned `TokenSlot` ring, and
    ///   the step after it, which reads that slot as its input, is committed
    ///   before the CPU waits for the first. The CPU reads each token one step
    ///   behind the GPU, for streaming and the stop checks. A step is committed
    ///   ahead only when the sequential loop would run it unless the token it
    ///   consumes ends the generation (a stop token, or a cancellation seen
    ///   after it); in that case it is rolled back (`rollback(to:)` its start),
    ///   so the store ends where the sequential loop leaves it.
    /// - **Pipelined prefill.** Prompt chunks are committed two deep; each
    ///   chunk's progress event follows its completion, and a cancellation
    ///   between chunks rolls back the one committed ahead.
    /// - **Verify-row batching.** When the strategy drafts tokens (and the
    ///   model has verify rows), one `.verify` step checks the last token and
    ///   the drafts; rows are sampled in order on the CPU exactly as plain
    ///   decoding would sample them, and the store is rolled back to the last
    ///   kept row (`rollback(to: start + r + 1)`), so the output equals plain
    ///   decoding for every sampler setting.
    /// - Otherwise (CPU sampling) each step is committed and waited on, as in
    ///   the reference: the next input depends on the CPU's choice.
    ///
    /// Prefix reuse, stop tokens (neither emitted nor returned), prompt
    /// acceptance into the penalty history and timings follow the reference.
    /// Every step runs inside its own autorelease pool and every job inside
    /// another. A command buffer that completes with an error ends the
    /// generation with `commandBufferFailed` after the sequence is reset.
    ///
    /// Timings: `medianGPUGapMicroseconds` is the median GPU idle time between
    /// consecutive kept decode command buffers; `medianSchedulerCPUMicroseconds`
    /// is the median, per generated token, of the generation thread's busy
    /// time (wall time minus time blocked waiting on the GPU and minus the
    /// model's own `encodeStep`).
    public final class PipelinedStepScheduler: StepScheduler, @unchecked Sendable {
        private let model: any ForwardModel
        private let sampler: any Sampler
        private let strategy: any DecodingStrategy
        private let queue: MTLCommandQueue
        let thread = SchedulerGenerationThread(name: "mference.native.generation")

        /// Shared-storage `u32` ring for GPU-chosen tokens.
        private let slots: MTLBuffer
        static let slotCount = 4
        private var nextSlot = 0

        private let cancelLock = NSLock()
        /// One submitted generation's cancellation flag (guarded by `cancelLock`).
        final class Generation: @unchecked Sendable {
            var cancelled = false
        }
        /// The generation executing on the thread, if any.
        private var active: Generation?
        /// Submitted generations not yet started, in submission order.
        private var pending: [Generation] = []

        /// A committed step.
        struct InFlight {
            var cb: MTLCommandBuffer
            var plan: StepPlan
            var logits: TensorRef?
            /// Token slot its GPU choice writes, if any.
            var slot: Int?
        }

        // Generation-thread state.
        /// Committed steps not yet waited on, oldest first (at most two).
        private var inflight: [InFlight] = []
        /// Nanoseconds blocked in `waitUntilCompleted` / spent in the model's
        /// `encodeStep` since the last counter reset.
        private var waitNanos: UInt64 = 0
        private var encodeNanos: UInt64 = 0
        /// Set when an undo failed and the sequence had to be reset.
        private var undoReset = false

        init(model: any ForwardModel, sampler: any Sampler, strategy: any DecodingStrategy,
             queue: MTLCommandQueue) throws {
            self.model = model
            self.sampler = sampler
            self.strategy = strategy
            self.queue = queue
            guard let slots = queue.device.makeBuffer(length: Self.slotCount * 4, options: .storageModeShared) else {
                throw NativeError.resource("token slots")
            }
            slots.label = "native.scheduler.tokenSlots"
            self.slots = slots
        }

        /// Cancels the running generation; when none has started yet, the
        /// next one to start (which then throws `cancelled` before its first
        /// step). Cancellation belongs to one generation.
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
                            defer { self.drain() }
                            return try self.execute(job, generation, onEvent: onEvent)
                        })
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        }

        // MARK: Command buffers

        @inline(__always) private static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

        private func makeCommandBuffer() throws -> MTLCommandBuffer {
            guard let cb = queue.makeCommandBuffer() else { throw NativeError.resource("command buffer") }
            return cb
        }

        private static func failure(_ cb: MTLCommandBuffer) -> String {
            cb.error.map { "\($0)" } ?? "status \(cb.status.rawValue)"
        }

        /// Wait for every committed step, ignoring status (error paths only).
        private func drain() {
            for s in inflight { s.cb.waitUntilCompleted() }
            inflight.removeAll()
        }

        /// The state can no longer be trusted: wait for everything, clear the
        /// sequence.
        private func resetSequence() {
            drain()
            autoreleasepool {
                if let cb = queue.makeCommandBuffer() {
                    try? model.kv.reset(commandBuffer: cb)
                    cb.commit()
                    cb.waitUntilCompleted()
                }
            }
        }

        /// Commit `cb` (store work outside any step) and wait; a failure resets
        /// the sequence and throws.
        private func commitAndWait(_ cb: MTLCommandBuffer) throws {
            cb.commit()
            cb.waitUntilCompleted()
            if cb.status != .completed {
                let message = Self.failure(cb)
                resetSequence()
                throw NativeError.commandBufferFailed(message)
            }
        }

        /// Wait for a committed step and check it.
        private func wait(_ step: InFlight) throws {
            let t0 = Self.now()
            step.cb.waitUntilCompleted()
            waitNanos &+= Self.now() &- t0
            inflight.removeAll { $0.cb === step.cb }
            if step.cb.status != .completed {
                let message = Self.failure(step.cb)
                resetSequence()
                throw NativeError.commandBufferFailed(message)
            }
        }

        private func takeSlot() -> Int {
            let s = nextSlot
            nextSlot = (nextSlot + 1) % Self.slotCount
            return s
        }

        private func tokenSlot(_ i: Int) -> TokenSlot { TokenSlot(buffer: slots, offset: i * 4, count: 1) }

        private func readSlot(_ i: Int) -> Int32 {
            Int32(bitPattern: slots.contents().load(fromByteOffset: i * 4, as: UInt32.self))
        }

        /// Begin, encode and commit one step (with the GPU choice into `slot`
        /// when given). On an encode failure the step is undone and the error
        /// rethrown.
        @discardableResult
        private func submit(_ plan: StepPlan, choice slot: Int?) throws -> InFlight {
            try autoreleasepool {
                let cb = try makeCommandBuffer()
                try model.kv.beginStep(rows: plan.rows, kind: plan.kind)
                guard let encoder = cb.makeComputeCommandEncoder() else {
                    undoOrReset(plan)
                    throw NativeError.resource("compute encoder")
                }
                let logits: TensorRef?
                do {
                    let t0 = Self.now()
                    logits = try model.encodeStep(plan, encoder: encoder)
                    encodeNanos &+= Self.now() &- t0
                    if let slot {
                        guard let logits else { throw NativeError.invalidShape("model returned no logits") }
                        try sampler.encodeChoice(encoder, logits: logits, output: tokenSlot(slot))
                    }
                } catch {
                    // beginStep already advanced the store: end the encoder
                    // without committing, then undo the step.
                    encoder.endEncoding()
                    undoOrReset(plan)
                    throw error
                }
                encoder.endEncoding()
                cb.commit()
                let step = InFlight(cb: cb, plan: plan, logits: logits, slot: slot)
                inflight.append(step)
                return step
            }
        }

        /// Undo `plan`'s uncommitted step; if that fails, clear the sequence.
        private func undoOrReset(_ plan: StepPlan) {
            do {
                try autoreleasepool {
                    let cb = try makeCommandBuffer()
                    try model.kv.rollback(to: plan.startPosition, commandBuffer: cb)
                    cb.commit()
                    cb.waitUntilCompleted()
                }
            } catch {
                undoReset = true
                resetSequence()
            }
        }

        /// Roll the store back to `position` (after the most recent step,
        /// which may still be running: the rollback is ordered after it).
        private func rollback(to position: Int) throws {
            try autoreleasepool {
                let cb = try makeCommandBuffer()
                do {
                    try model.kv.rollback(to: position, commandBuffer: cb)
                } catch {
                    resetSequence()
                    throw error
                }
                try commitAndWait(cb)
            }
            // The rollback ran after every committed step.
            drain()
        }

        // MARK: Generation

        private func execute(_ job: GenerationJob, _ generation: Generation,
                             onEvent: @Sendable (StepEvent) -> Void) throws -> GenerationOutcome {
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
            let start = Self.now()
            func seconds(since t: UInt64) -> Double { Double(Self.now() - t) / 1e9 }
            undoReset = false
            inflight.removeAll()

            sampler.reset(params: job.params)
            strategy.reset(prompt: prompt)
            let draftLimit = max(0, min(strategy.maxDraftRows, model.maxVerifyRows - 1, 15))
            let gpuChoice = draftLimit == 0 && sampler.canChooseOnGPU
            let reused: Int = try autoreleasepool {
                var reused = 0
                let setup = try makeCommandBuffer()
                try kv.reset(commandBuffer: setup)
                if job.reusePrefix, prompt.count > 1, let match = kv.lookupPrefix(Array(prompt.dropLast())) {
                    try kv.attachPrefix(match, commandBuffer: setup)
                    reused = match.length
                }
                try commitAndWait(setup)
                return reused
            }
            for t in prompt { sampler.accept(t) }

            var timings = StepTimings()
            func cancelledDuringPrefill() -> GenerationOutcome {
                timings.prefillSeconds = seconds(since: start)
                return GenerationOutcome(tokens: [], stopReason: .cancelled, promptTokenCount: prompt.count,
                                         reusedPrefixTokens: reused, timings: timings)
            }
            func prefillCompleted(_ step: InFlight) {
                let done = step.plan.startPosition + step.plan.rows
                onEvent(.prefillProgress(done: done, total: prompt.count))
                if done == prompt.count { timings.prefillSeconds = seconds(since: start) }
            }

            // Prefill: chunks committed two deep.
            var position = reused
            var lastPrefill: InFlight?
            while position < prompt.count {
                let n = min(job.prefillChunk, prompt.count - position)
                let last = position + n == prompt.count
                let plan = StepPlan(kind: .prefill, startPosition: position, rows: n,
                                    input: .host(Array(prompt[position ..< position + n])),
                                    logits: last ? .last : .none)
                let step = try submit(plan, choice: last && gpuChoice ? takeSlot() : nil)
                position += n
                if inflight.count == 2 {
                    let older = inflight[0]
                    try wait(older)
                    prefillCompleted(older)
                    if isCancelled(generation) {
                        try rollback(to: step.plan.startPosition)
                        return cancelledDuringPrefill()
                    }
                }
                if last { lastPrefill = step }
            }
            guard let lastPrefill else { throw NativeError.invalidShape("nothing to prefill") }
            // The last chunk stays in flight into a pipelined decode, unless
            // its state is cached first.
            if !(gpuChoice && !job.reusePrefix) {
                try wait(lastPrefill)
                prefillCompleted(lastPrefill)
            }
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

            var d = Decode(timings: timings)
            let decodeStart = Self.now()
            if gpuChoice {
                try decodeOnGPU(job, generation, prompt: prompt, producer: lastPrefill, start: start,
                                state: &d, prefillCompleted: prefillCompleted, onEvent: onEvent)
            } else {
                try decodeOnCPU(job, generation, prompt: prompt, last: lastPrefill, start: start,
                                draftLimit: draftLimit, state: &d, onEvent: onEvent)
            }
            // `prefillCompleted` may have run inside the GPU loop.
            d.timings.prefillSeconds = max(d.timings.prefillSeconds, timings.prefillSeconds)
            d.timings.decodeSeconds = seconds(since: decodeStart)
            d.timings.medianGPUGapMicroseconds = Self.median(d.gaps)
            d.timings.medianSchedulerCPUMicroseconds = Self.median(d.cpu)
            return GenerationOutcome(tokens: d.generated, stopReason: d.reason, promptTokenCount: prompt.count,
                                     reusedPrefixTokens: reused, timings: d.timings)
        }

        struct Decode {
            var timings: StepTimings
            var generated: [Int32] = []
            var reason: StopReason = .maxTokens
            var gaps: [Double] = []
            var cpu: [Double] = []
            var previousEnd: CFTimeInterval?

            /// Record a completed, kept decode or verify step.
            mutating func kept(_ step: InFlight) {
                guard step.plan.kind != .prefill else { return }
                timings.decodeSteps += 1
                let (s, e) = (step.cb.gpuStartTime, step.cb.gpuEndTime)
                if let previousEnd { gaps.append(max(0, s - previousEnd) * 1e6) }
                previousEnd = e
            }
        }

        static func median(_ v: [Double]) -> Double {
            guard !v.isEmpty else { return 0 }
            let s = v.sorted()
            return s[s.count / 2]
        }

        private func startCounters() -> UInt64 {
            waitNanos = 0
            encodeNanos = 0
            return Self.now()
        }

        private func busyMicroseconds(since t0: UInt64) -> Double {
            let wall = Self.now() &- t0
            return Double(wall) / 1e3 - Double(waitNanos) / 1e3 - Double(encodeNanos) / 1e3
        }

        /// Plain decoding with the choice on the GPU: the step consuming each
        /// token is committed before the CPU reads that token.
        private func decodeOnGPU(_ job: GenerationJob, _ generation: Generation, prompt: [Int32],
                                 producer first: InFlight, start: UInt64, state d: inout Decode,
                                 prefillCompleted: (InFlight) -> Void,
                                 onEvent: @Sendable (StepEvent) -> Void) throws {
            let kv = model.kv
            // The step whose choice slot holds the next token.
            var producer = first
            // Position at which that token would be consumed.
            var position = prompt.count
            func plan(at position: Int, reading slot: Int) -> StepPlan {
                StepPlan(kind: .decode, startPosition: position, rows: 1, input: .device(tokenSlot(slot)),
                         logits: .last)
            }
            func complete(_ step: InFlight) throws {
                guard inflight.contains(where: { $0.cb === step.cb }) else { return }
                try wait(step)
                if step.plan.kind == .prefill { prefillCompleted(step) } else { d.kept(step) }
            }
            while true {
                if job.maxNewTokens <= d.generated.count { d.reason = .maxTokens; break }
                if isCancelled(generation) { d.reason = .cancelled; break }
                let t0 = startCounters()
                guard let slot = producer.slot else { throw NativeError.invalidShape("model returned no logits") }
                // Commit the step that consumes the unread token, if the
                // sequential loop would run it unless that token ends the
                // generation.
                var ahead: InFlight?
                if d.generated.count + 1 < job.maxNewTokens && position < kv.capacity {
                    do {
                        ahead = try submit(plan(at: position, reading: slot), choice: takeSlot())
                    } catch {
                        // Retried below in sequence, where the loop would run it.
                        if undoReset { throw error }
                    }
                }
                try complete(producer)
                let token = readSlot(slot)
                if d.generated.isEmpty { d.timings.timeToFirstTokenSeconds = Double(Self.now() - start) / 1e9 }
                if job.stopTokens.contains(token) {
                    d.reason = .stopToken
                    if let ahead { try rollback(to: ahead.plan.startPosition) }
                    break
                }
                sampler.accept(token)
                d.generated.append(token)
                onEvent(.token(token))
                if job.maxNewTokens <= d.generated.count { d.reason = .maxTokens; break }
                if position >= kv.capacity { d.reason = .contextFull; break }
                if isCancelled(generation) {
                    d.reason = .cancelled
                    if let ahead { try rollback(to: ahead.plan.startPosition) }
                    break
                }
                let next = try ahead ?? submit(plan(at: position, reading: slot), choice: takeSlot())
                d.cpu.append(busyMicroseconds(since: t0))
                producer = next
                position += 1
            }
            // A step the sequential loop also runs may still be in flight
            // (stopped at the top of the loop): finish it.
            try complete(producer)
        }

        /// Sampling on the CPU: each step is committed and waited on; with
        /// drafts, one verify step checks them and the store keeps the
        /// accepted rows.
        private func decodeOnCPU(_ job: GenerationJob, _ generation: Generation, prompt: [Int32],
                                 last first: InFlight, start: UInt64, draftLimit: Int, state d: inout Decode,
                                 onEvent: @Sendable (StepEvent) -> Void) throws {
            let kv = model.kv
            var step = first
            var t0 = startCounters()
            outer: while true {
                guard let logits = step.logits else { throw NativeError.invalidShape("model returned no logits") }
                let rows = logits.rows
                // Input tokens of the step's logit rows (row r consumed
                // inputs[r]); drafts are inputs[1...].
                let inputs: [Int32]
                if case .host(let ids) = step.plan.input { inputs = Array(ids.suffix(rows)) } else { inputs = [] }
                let firstRowEnd = step.plan.startPosition + step.plan.rows - rows
                var next: Int32?
                var accepted = 0
                func keep(_ r: Int) throws {
                    let end = firstRowEnd + r + 1
                    if end < kv.length { try rollback(to: end) }
                }
                for r in 0..<rows {
                    if job.maxNewTokens <= d.generated.count { d.reason = .maxTokens; try keep(r); break outer }
                    if isCancelled(generation) { d.reason = .cancelled; try keep(r); break outer }
                    let row = UnsafeBufferPointer(
                        start: logits.buffer.contents().advanced(by: logits.offset + r * logits.rowStride)
                            .assumingMemoryBound(to: Float.self),
                        count: logits.cols)
                    let token = sampler.sample(logits: row)
                    if d.generated.isEmpty { d.timings.timeToFirstTokenSeconds = Double(Self.now() - start) / 1e9 }
                    if job.stopTokens.contains(token) { d.reason = .stopToken; try keep(r); break outer }
                    sampler.accept(token)
                    d.generated.append(token)
                    onEvent(.token(token))
                    if job.maxNewTokens <= d.generated.count { d.reason = .maxTokens; try keep(r); break outer }
                    if firstRowEnd + r + 1 >= kv.capacity { d.reason = .contextFull; try keep(r); break outer }
                    if isCancelled(generation) { d.reason = .cancelled; try keep(r); break outer }
                    if r + 1 < rows && r + 1 < inputs.count && inputs[r + 1] == token {
                        accepted += 1
                        continue
                    }
                    try keep(r)
                    next = token
                    break
                }
                if step.plan.kind == .verify { strategy.observe(proposed: rows - 1, accepted: accepted) }
                guard let token = next else { throw NativeError.invalidShape("verify step produced no token") }
                let position = kv.length
                var drafts: [Int32] = []
                if draftLimit > 0 {
                    let room = min(draftLimit, kv.capacity - position - 1, job.maxNewTokens - d.generated.count - 1)
                    if room > 0 {
                        drafts = Array(strategy.draft(history: prompt + d.generated).prefix(room))
                    }
                }
                let plan = drafts.isEmpty
                    ? StepPlan(kind: .decode, startPosition: position, rows: 1, input: .host([token]), logits: .last)
                    : StepPlan(kind: .verify, startPosition: position, rows: 1 + drafts.count,
                               input: .host([token] + drafts), logits: .all)
                let s = try submit(plan, choice: nil)
                try wait(s)
                d.kept(s)
                d.cpu.append(busyMicroseconds(since: t0))
                t0 = startCounters()
                step = s
            }
        }
    }
}
