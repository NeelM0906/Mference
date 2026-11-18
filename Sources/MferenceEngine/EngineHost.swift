import Foundation
import Synchronization

// MARK: - Status

/// What the host is doing with its one model slot.
public enum EngineHostPhase: String, Sendable, Hashable {
    /// No model resident.
    case empty
    /// A model is being loaded, or swapped in: the old one is released first.
    case loading
    case ready
    /// The resident model is being released (idle, requested, or shutdown).
    case unloading
}

/// Why a resident model was released.
public enum EngineUnloadReason: Sendable, Hashable {
    /// Another model was requested; `to` is the one that loads next.
    case swap(to: String)
    /// Nothing used it for `idleUnloadAfter`.
    case idle
    /// `EngineHost.unload()`.
    case requested
    case shutdown
}

public struct EngineLoadFailure: Sendable, Hashable {
    public let modelID: String
    public let message: String

    public init(modelID: String, message: String) {
        self.modelID = modelID
        self.message = message
    }
}

/// `GET /health` in the server's spelling: `("loading", target)` while a model
/// is being loaded or swapped in, otherwise `("ok", resident model or nil)`.
public struct EngineHealth: Sendable, Hashable {
    public let status: String
    public let modelID: String?

    public init(status: String, modelID: String?) {
        self.status = status
        self.modelID = modelID
    }
}

/// A consistent picture of the host, readable synchronously from any thread
/// (a NIO channel callback, a SwiftUI view) while a load runs on the actor.
public struct EngineHostStatus: Sendable, Hashable {
    public var phase: EngineHostPhase = .empty
    /// The model that is loaded and usable; nil while loading or unloading.
    public var residentModelID: String?
    /// The model being loaded (`loading`) or released (`unloading`).
    public var targetModelID: String?
    /// 1 while a request holds the model, else 0.
    public var activeRequests = 0
    /// Requests waiting their turn.
    public var queuedRequests = 0
    /// The most recent load that failed, cleared by the next successful load.
    public var lastLoadFailure: EngineLoadFailure?
    /// `shutdown()` has begun; no new work is accepted.
    public var isShuttingDown = false

    public init() {}

    public var health: EngineHealth {
        if phase == .loading { return EngineHealth(status: "loading", modelID: targetModelID) }
        return EngineHealth(status: "ok", modelID: residentModelID)
    }
}

/// Lifecycle notices, in the order they happen. A swap is
/// `unloadStarted(A, .swap(to: B))`, `unloadFinished(A, ...)`,
/// `loadStarted(B, replacing: A)`, then `loadFinished(B)` or `loadFailed(B)`:
/// the release of the old model always completes before the new load starts.
public enum EngineHostEvent: Sendable, Hashable {
    /// `replacing` is the model a swap released first, nil for a cold load.
    case loadStarted(modelID: String, replacing: String?)
    case loadFinished(modelID: String, seconds: Double)
    case loadFailed(modelID: String, message: String)
    case unloadStarted(modelID: String, reason: EngineUnloadReason)
    case unloadFinished(modelID: String, reason: EngineUnloadReason)
}

/// One element of an `EngineGenerationStream`.
public enum EngineStreamElement: Sendable {
    /// The request was rendered and accepted; generation starts next.
    case prepared(promptTokens: Int)
    case event(EngineEvent)
    /// The last element of a successful stream.
    case completed(EngineCompletion)
}

private final class EngineHostStatusBoard: Sendable {
    private let state = Mutex(EngineHostStatus())

    var current: EngineHostStatus { state.withLock { $0 } }

    func update(_ body: (inout EngineHostStatus) -> Void) {
        state.withLock { body(&$0) }
    }
}

// MARK: - Host

/// Owns the one resident model and gives it to one request at a time.
///
/// - **One model.** At most one model is loaded across every backend (spec
///   D4). A request for another model releases the resident one (its
///   `unload()` returns and the host drops its reference) before the next
///   load starts, so peak memory is one model, not two.
/// - **One request at a time, first come first served.** Requests, explicit
///   loads and unloads, and idle unloads all pass one FIFO gate. A swap never
///   happens under a running request.
/// - **Cancellation.** Cancelling the calling task, or `cancelActiveRequest()`,
///   stops a running request at the next token and throws
///   `CancellationError`; a cancelled waiter leaves the line without running.
///   A load in progress is not interrupted: the model stays resident.
/// - **Load failure** leaves the host empty, with the failure in
///   `status.lastLoadFailure`, and rethrows the backend's error.
/// - **Idle unload.** With `idleUnloadAfter` set, the resident model is
///   released once nothing has used it for that long.
public actor EngineHost {
    public typealias Sleeper = @Sendable (Duration) async throws -> Void

    private struct Waiter {
        let id: UInt64
        let isRequest: Bool
        /// Resumes the waiter: with nil when it is handed the gate, with the
        /// error when it leaves the line without it.
        let resume: @Sendable ((any Error)?) -> Void
    }

    public nonisolated let catalog: EngineCatalog
    public nonisolated let loadOptions: EngineLoadOptions
    public nonisolated let idleUnloadAfter: Duration?

    private nonisolated let board = EngineHostStatusBoard()
    private let backends: [EngineKind: any EngineBackend]
    private let maxQueuedRequests: Int?
    private let sleep: Sleeper
    private let observer: @Sendable (EngineHostEvent) -> Void

    private var resident: (id: String, model: any EngineModel)?
    private var gateBusy = false
    private var waiters: [Waiter] = []
    private var nextWaiterID: UInt64 = 0
    private var cancelActive: (@Sendable () -> Void)?
    /// True while a request holds the gate, from `acquire` to the end of its
    /// body, including the load it may trigger.
    private var requestHoldsGate = false
    /// Set by `cancelActiveRequest()` or `shutdown()` while a request holds
    /// the gate but its body has not started: it is checked after the load.
    private var holderCancelRequested = false
    private var shutdownTask: Task<Void, Never>?
    private var idleTask: Task<Void, Never>?
    private var idleEpoch: UInt64 = 0
    private var shuttingDown = false

    /// - Parameters:
    ///   - backends: one per engine kind; a later duplicate kind is ignored.
    ///   - loadOptions: applied to every load.
    ///   - idleUnloadAfter: release the resident model after this long unused;
    ///     nil keeps it until a swap, `unload()`, or `shutdown()`.
    ///   - maxQueuedRequests: requests allowed to wait behind the running one;
    ///     one more is refused with `EngineError.queueFull`. Nil is unbounded.
    ///   - sleep: the idle timer's clock; tests inject a controllable one.
    ///   - observer: lifecycle notices, called on the host's executor; keep it
    ///     quick (log, or hand off).
    public init(catalog: EngineCatalog,
                backends: [any EngineBackend],
                loadOptions: EngineLoadOptions = EngineLoadOptions(),
                idleUnloadAfter: Duration? = nil,
                maxQueuedRequests: Int? = nil,
                sleep: @escaping Sleeper = { try await Task.sleep(for: $0) },
                observer: @escaping @Sendable (EngineHostEvent) -> Void = { _ in }) {
        var byKind: [EngineKind: any EngineBackend] = [:]
        for backend in backends where byKind[backend.engine] == nil {
            byKind[backend.engine] = backend
        }
        self.catalog = catalog
        self.backends = byKind
        self.loadOptions = loadOptions
        self.idleUnloadAfter = idleUnloadAfter
        self.maxQueuedRequests = maxQueuedRequests
        self.sleep = sleep
        self.observer = observer
    }

    /// Current status, readable without awaiting the actor.
    public nonisolated var status: EngineHostStatus { board.current }

    // MARK: Requests

    /// Runs `body` with `modelID` resident and exclusively held, loading or
    /// swapping it in first if needed.
    ///
    /// This is the primitive every request goes through: validate against
    /// `model.traits`, `prepare`, commit a response, `generate`, all inside
    /// one `body`, so no swap can come between them. `onQueued` is called if
    /// the request has to wait behind another. Do not keep the model after
    /// `body` returns: the host may release it at any later point.
    public func withModel<T: Sendable>(
        _ modelID: String,
        onQueued: @escaping @Sendable () -> Void = {},
        _ body: @escaping @Sendable (any EngineModel) async throws -> T
    ) async throws -> T {
        let descriptor = try catalog.resolve(modelID)
        guard backends[descriptor.engine] != nil else {
            throw EngineError.backendUnavailable(descriptor.engine)
        }
        try await acquire(isRequest: true, onQueued: onQueued)
        defer { release() }
        requestHoldsGate = true
        holderCancelRequested = false
        board.update { $0.activeRequests = 1 }
        defer {
            requestHoldsGate = false
            board.update { $0.activeRequests = 0 }
        }
        try checkHolderMayProceed()
        let model = try await makeResident(descriptor)
        try checkHolderMayProceed()

        // The body runs in its own task so `cancelActiveRequest()` can reach
        // it; cancelling the caller is forwarded to it too.
        let task = Task<T, Error> { try await body(model) }
        cancelActive = { task.cancel() }
        defer { cancelActive = nil }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    /// Prepares and generates one request on `modelID`. `onPrepared` runs
    /// after the prompt is accepted and before the first event: the point at
    /// which a server commits its response status.
    public func generate(
        modelID: String,
        request: EngineRequest,
        onQueued: @escaping @Sendable () -> Void = {},
        onPrepared: @escaping @Sendable (EnginePreparedPrompt) async throws -> Void = { _ in },
        onEvent: @escaping @Sendable (EngineEvent) -> Void
    ) async throws -> EngineCompletion {
        try await withModel(modelID, onQueued: onQueued) { model in
            let prepared = try await model.prepare(request)
            try await onPrepared(prepared)
            return try await model.generate(prepared, onEvent: onEvent)
        }
    }

    /// `generate` as an async sequence. Each iteration is one request,
    /// started when iteration starts. It is cancelled when the iterating
    /// task is cancelled, or when iteration ends early and the iterator is
    /// released, even if the sequence itself is kept.
    public nonisolated func stream(modelID: String, request: EngineRequest) -> EngineGenerationStream {
        EngineGenerationStream(host: self, modelID: modelID, request: request)
    }

    /// Cancels the request that holds the model, if any, also while it is
    /// still loading that model (the load itself completes). Waiting requests
    /// are not affected.
    public func cancelActiveRequest() {
        if requestHoldsGate { holderCancelRequested = true }
        cancelActive?()
    }

    /// After the gate is handed over and again after the load: the caller's
    /// task, `cancelActiveRequest()`, and `shutdown()` may all have spoken.
    private func checkHolderMayProceed() throws {
        try Task.checkCancellation()
        if holderCancelRequested { throw CancellationError() }
        if shuttingDown { throw EngineError.shuttingDown }
    }

    // MARK: Lifecycle

    /// Makes `modelID` resident without running a request: the server's
    /// startup preload, or the app's "warm up" when a model is picked.
    public func load(_ modelID: String) async throws {
        let descriptor = try catalog.resolve(modelID)
        guard backends[descriptor.engine] != nil else {
            throw EngineError.backendUnavailable(descriptor.engine)
        }
        try await acquire(isRequest: false, onQueued: {})
        defer { release() }
        try Task.checkCancellation()
        if shuttingDown { throw EngineError.shuttingDown }
        _ = try await makeResident(descriptor)
    }

    /// Releases the resident model once the running request, if any, and the
    /// requests already waiting have finished.
    public func unload() async throws {
        try await acquire(isRequest: false, onQueued: {})
        defer { release() }
        if shuttingDown { throw EngineError.shuttingDown }
        await unloadResident(reason: .requested)
    }

    /// Refuses new work, fails waiting requests with `EngineError.shuttingDown`,
    /// cancels the running one, and releases the resident model. Returns once
    /// the model is released. Cancelling the caller does not stop it, and a
    /// second call waits for the first.
    public func shutdown() async {
        if shutdownTask == nil {
            shuttingDown = true
            board.update { $0.isShuttingDown = true }
            cancelIdleTimer()
            let queued = waiters
            waiters.removeAll()
            publishQueue()
            for waiter in queued { waiter.resume(EngineError.shuttingDown) }
            if requestHoldsGate { holderCancelRequested = true }
            cancelActive?()
            shutdownTask = Task { await self.finishShutdown() }
        }
        await shutdownTask?.value
    }

    /// Waits for the gate without regard to cancellation (nothing else can
    /// join the line now, so this is next), then releases the model.
    private func finishShutdown() async {
        if gateBusy {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                waiters.append(Waiter(id: nextWaiterID, isRequest: false, resume: { _ in continuation.resume() }))
                nextWaiterID &+= 1
            }
        } else {
            gateBusy = true
        }
        await unloadResident(reason: .shutdown)
        release()
    }

    // MARK: Gate

    private func acquire(isRequest: Bool, onQueued: @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw EngineError.shuttingDown }
        cancelIdleTimer()
        if !gateBusy {
            gateBusy = true
            return
        }
        if isRequest, let maxQueuedRequests,
           waiters.lazy.filter(\.isRequest).count >= maxQueuedRequests {
            throw EngineError.queueFull
        }
        let id = nextWaiterID
        nextWaiterID &+= 1
        if isRequest { onQueued() }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                // Runs on the actor, before `cancelWaiter` can: a task already
                // cancelled leaves at once instead of waiting for nobody.
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                waiters.append(Waiter(id: id, isRequest: isRequest, resume: { error in
                    if let error { continuation.resume(throwing: error) } else { continuation.resume() }
                }))
                publishQueue()
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        // Resumed by `release`, which handed the gate over without clearing it.
    }

    private func cancelWaiter(_ id: UInt64) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        publishQueue()
        waiter.resume(CancellationError())
    }

    private func release() {
        if !waiters.isEmpty {
            let next = waiters.removeFirst()
            publishQueue()
            next.resume(nil)
        } else {
            gateBusy = false
            scheduleIdleTimer()
        }
    }

    private func publishQueue() {
        let count = waiters.lazy.filter(\.isRequest).count
        board.update { $0.queuedRequests = count }
    }

    // MARK: Residency (called with the gate held)

    private func makeResident(_ descriptor: EngineModelDescriptor) async throws -> any EngineModel {
        if let resident, resident.id == descriptor.id { return resident.model }
        let replacing = resident?.id
        board.update {
            $0.phase = .loading
            $0.targetModelID = descriptor.id
            $0.residentModelID = nil
        }
        await unloadResident(reason: .swap(to: descriptor.id))
        observer(.loadStarted(modelID: descriptor.id, replacing: replacing))
        let started = ContinuousClock.now
        do {
            guard let backend = backends[descriptor.engine] else {
                throw EngineError.backendUnavailable(descriptor.engine)
            }
            // The load runs in its own task so that cancelling the request
            // that triggered it cannot abandon it halfway: the old model is
            // already gone, and a half-finished load would leave the host
            // empty for nothing. The request checks cancellation afterwards.
            let options = loadOptions
            let model = try await Task { try await backend.load(descriptor, options: options) }.value
            resident = (id: descriptor.id, model: model)
            board.update {
                $0.phase = .ready
                $0.residentModelID = descriptor.id
                $0.targetModelID = nil
                $0.lastLoadFailure = nil
            }
            let elapsed = started.duration(to: .now)
            observer(.loadFinished(modelID: descriptor.id,
                                   seconds: Double(elapsed.components.seconds)
                                       + Double(elapsed.components.attoseconds) / 1e18))
            return model
        } catch {
            let message = String(describing: error)
            board.update {
                $0.phase = .empty
                $0.residentModelID = nil
                $0.targetModelID = nil
                $0.lastLoadFailure = EngineLoadFailure(modelID: descriptor.id, message: message)
            }
            observer(.loadFailed(modelID: descriptor.id, message: message))
            throw error
        }
    }

    /// Releases the resident model, if any. The host's reference is gone when
    /// this returns: the only other one was this function's local.
    private func unloadResident(reason: EngineUnloadReason) async {
        guard let outgoing = resident else { return }
        resident = nil
        let isSwap: Bool
        if case .swap = reason { isSwap = true } else { isSwap = false }
        if !isSwap {
            board.update {
                $0.phase = .unloading
                $0.residentModelID = nil
                $0.targetModelID = outgoing.id
            }
        }
        observer(.unloadStarted(modelID: outgoing.id, reason: reason))
        await outgoing.model.unload()
        observer(.unloadFinished(modelID: outgoing.id, reason: reason))
        if !isSwap {
            board.update {
                $0.phase = .empty
                $0.targetModelID = nil
            }
        }
    }

    // MARK: Idle unload

    private func scheduleIdleTimer() {
        guard let idleUnloadAfter, resident != nil, !gateBusy, !shuttingDown else { return }
        idleEpoch &+= 1
        let epoch = idleEpoch
        let sleep = self.sleep
        idleTask = Task { [weak self] in
            do { try await sleep(idleUnloadAfter) } catch { return }
            await self?.idleTimerFired(epoch)
        }
    }

    private func cancelIdleTimer() {
        idleEpoch &+= 1
        idleTask?.cancel()
        idleTask = nil
    }

    private func idleTimerFired(_ epoch: UInt64) async {
        guard epoch == idleEpoch, !gateBusy, resident != nil, !shuttingDown else { return }
        idleTask = nil
        gateBusy = true
        await unloadResident(reason: .idle)
        release()
    }
}

/// The sequence `EngineHost.stream(modelID:request:)` returns.
///
/// The request belongs to the iterator, not to the sequence: making an
/// iterator starts it, and releasing the last copy of that iterator cancels
/// it. Breaking out of a `for try await` loop therefore stops generation and
/// frees the host's gate, whether or not the caller keeps the sequence.
public struct EngineGenerationStream: AsyncSequence, Sendable {
    public typealias Element = EngineStreamElement

    let host: EngineHost
    let modelID: String
    let request: EngineRequest

    public func makeAsyncIterator() -> Iterator {
        let (elements, continuation) = AsyncThrowingStream<Element, Error>.makeStream()
        let host = self.host
        let modelID = self.modelID
        let request = self.request
        let task = Task {
            do {
                let completion = try await host.generate(
                    modelID: modelID,
                    request: request,
                    onPrepared: { continuation.yield(.prepared(promptTokens: $0.promptTokens.count)) },
                    onEvent: { continuation.yield(.event($0)) })
                continuation.yield(.completed(completion))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return Iterator(base: elements.makeAsyncIterator(), request: RequestHandle(task: task))
    }

    public struct Iterator: AsyncIteratorProtocol {
        var base: AsyncThrowingStream<Element, Error>.Iterator
        let request: RequestHandle
        var completed = false

        /// Ends with `.completed`, or throws: a request that stops because
        /// the iterating task was cancelled throws `CancellationError` rather
        /// than ending quietly without a completion.
        public mutating func next() async throws -> Element? {
            let element = try await base.next()
            if case .completed? = element { completed = true }
            if element == nil, !completed {
                try Task.checkCancellation()
            }
            return element
        }
    }

    /// Cancels the request when the last iterator copy goes away. Cancelling
    /// a request that already finished does nothing.
    final class RequestHandle: Sendable {
        let task: Task<Void, Never>

        init(task: Task<Void, Never>) { self.task = task }

        deinit { task.cancel() }
    }
}
