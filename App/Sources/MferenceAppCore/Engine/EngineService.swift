import Foundation
import Mference
import MferenceEngine
import Synchronization

/// The live `EngineServicing` over one shared `EngineHost` (spec 4.3.1).
///
/// - **Discovery is lazy and off the caller's thread.** Building the service
///   does no file-system work; the first call that needs the catalog runs
///   discovery on a detached task (it walks install directories for their
///   sizes, which is too slow for the main thread).
/// - **One host, one resident model, one generation at a time.** The host
///   queues requests first come first served and releases the old model
///   before a swap loads the next.
/// - **Refresh after a download.** `refreshCatalog()` rediscovers. The host's
///   catalog is fixed, so a changed catalog gets a new host: at once when
///   nothing is resident or running, otherwise when a request needs a model
///   the old host cannot serve, after the old host has finished its running
///   and queued work and released its model. A model the old host already
///   serves as discovered now keeps being served by it without a reload.
/// - **Idle unload** is the host's own timer, with an `IdleUnloadControl` as
///   its clock so Settings can change the delay while a model is resident.
public actor EngineService: EngineServicing {
    /// Ten minutes after the last token (decision L13).
    public static let defaultIdleUnload: Duration = .seconds(600)

    public typealias HostFactory = @Sendable (EngineCatalog, @escaping @Sendable (EngineHostEvent) -> Void)
        -> EngineHost

    private let discover: @Sendable () -> EngineCatalog
    private let makeHost: HostFactory
    private let broadcaster: StatusBroadcaster
    /// Deep is offered, and may be loaded, only on Macs with more than 16 GB
    /// (L4, spec 4.3).
    private let physicalMemory: UInt64

    /// The newest discovery result; picker rows come from it.
    private var catalog: EngineCatalog?
    private var discovery: Task<EngineCatalog, Never>?
    /// Each refresh takes the next number; only the newest one stores its
    /// catalog, so overlapping refreshes never install an older result.
    private var refreshGeneration: UInt64 = 0
    private var host: EngineHost?
    /// Requests and loads handed a host and not yet finished, per host.
    private var inFlight: [ObjectIdentifier: Int] = [:]
    private var drainWaiters: [ObjectIdentifier: [CheckedContinuation<Void, Never>]] = [:]
    /// A host replacement in progress; new work waits for it.
    private var replacement: Task<Void, Never>?
    private var isShutDown = false

    /// The production service: default discovery roots, native and llama.cpp
    /// backends, idle unload driven by `idleUnload`.
    public static func live(idleUnload: IdleUnloadControl = IdleUnloadControl()) -> EngineService {
        EngineService(
            discover: { EngineCatalogDiscovery.discoverDefault(scope: .product) },
            makeHost: { catalog, observer in
                EngineHost(catalog: catalog,
                           backends: [NativeBackend(), LlamaBackend()],
                           idleUnloadAfter: defaultIdleUnload,
                           maxQueuedRequests: 8,
                           sleep: idleUnload.sleeper,
                           observer: observer)
            })
    }

    /// Tests and previews: inject discovery and the host (with
    /// `FakeEngineBackend`). Neither is called until the service needs them.
    public init(discover: @escaping @Sendable () -> EngineCatalog, makeHost: @escaping HostFactory,
                physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) {
        self.broadcaster = StatusBroadcaster()
        self.discover = discover
        self.makeHost = makeHost
        self.physicalMemory = physicalMemory
    }

    /// Deep needs more than 16 GB of memory (L4), and a model with an engine
    /// memory floor needs that much (`EngineProductSlot.minimumPhysicalMemory`:
    /// Inkling 2 Mini, 32 GB); the reason when `model` may not run on a Mac
    /// with `physicalMemory`, else nil.
    public static func memoryGateReason(_ model: AppModel, physicalMemory: UInt64) -> String? {
        if model == .deep && physicalMemory <= 16 << 30 { return "Needs a Mac with more than 16 GB of memory." }
        return EngineProductSlot(modelID: model.modelID)?.memoryFloorReason(physicalMemory: physicalMemory)
    }

    /// The runtime's own estimate of what `model` keeps resident with
    /// `sizeBytes` on disk on a Mac with `physicalMemory` (32K context), for
    /// the models it can size before loading (Inkling 2 Mini); nil otherwise.
    public static func runtimeResidentEstimate(_ model: AppModel, sizeBytes: UInt64,
                                               physicalMemory: UInt64) -> UInt64? {
        EngineProductSlot(modelID: model.modelID)?
            .estimatedResidentBytes(installBytes: sizeBytes, physicalMemory: physicalMemory)
    }

    /// What a picker row says when `model` is not installed and the catalog
    /// gives no reason: native installs come from `MferenceRepack`, so the
    /// row says how to make one where the app cannot download it.
    static func notInstalledReason(_ model: AppModel) -> String {
        switch model {
        case .inkling2Mini: "Not installed. Install it with MferenceRepack --model inkling2mini."
        default: "Not installed on this Mac."
        }
    }

    // MARK: Catalog

    public func models() async -> [ModelOption] {
        let catalog = await currentCatalog()
        return Self.options(catalog: catalog, status: host?.status ?? EngineHostStatus(),
                            physicalMemory: physicalMemory)
    }

    public func refreshCatalog() async {
        guard !isShutDown else { return }
        refreshGeneration &+= 1
        let generation = refreshGeneration
        let discover = self.discover
        let fresh = await Task.detached { discover() }.value
        // A newer refresh started meanwhile: its catalog wins.
        guard !isShutDown, generation == refreshGeneration else { return }
        discovery = nil
        setCatalog(fresh)
        // Replace an idle, empty host now so its rows and the next load agree;
        // a busy or loaded one is replaced when a request needs it.
        if let host, host.catalog != fresh, replacement == nil {
            let status = host.status
            if status.phase == .empty, status.activeRequests == 0, status.queuedRequests == 0,
               inFlight[ObjectIdentifier(host), default: 0] == 0 {
                await replaceHost(with: fresh)
            }
        }
        broadcaster.publish()
    }

    static func options(catalog: EngineCatalog, status: EngineHostStatus,
                        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> [ModelOption] {
        AppModel.engineModels.map { model in
            guard let descriptor = catalog.descriptor(for: model.modelID) else {
                let reason = catalog.unavailableReason(for: model.modelID) ?? notInstalledReason(model)
                return ModelOption(model: model, availability: .notInstalled(reason: reason))
            }
            if let reason = memoryGateReason(model, physicalMemory: physicalMemory) {
                return ModelOption(model: model, availability: .notInstalled(reason: reason),
                                   sizeBytes: descriptor.sizeBytes, location: descriptor.location)
            }
            let availability: ModelAvailability
            if status.residentModelID == model.modelID, status.phase == .ready {
                availability = .loaded
            } else if status.targetModelID == model.modelID, status.phase == .loading {
                availability = .loading
            } else if status.targetModelID == model.modelID, status.phase == .unloading {
                availability = .unloading
            } else {
                availability = .ready
            }
            return ModelOption(model: model, availability: availability,
                               sizeBytes: descriptor.sizeBytes,
                               contextLength: descriptor.contextLength,
                               location: descriptor.location,
                               seesImages: model.supportsImageUnderstanding && descriptor.visionProjector != nil,
                               imageUnderstandingBytes: descriptor.visionProjector?.sizeBytes ?? 0,
                               imageUnderstandingLocation: model.supportsImageUnderstanding
                                   ? descriptor.visionProjector?.location : nil)
        }
    }

    private func currentCatalog() async -> EngineCatalog {
        if let catalog { return catalog }
        let task: Task<EngineCatalog, Never>
        if let discovery {
            task = discovery
        } else {
            let discover = self.discover
            task = Task.detached { discover() }
            discovery = task
        }
        let result = await task.value
        // A refresh may have stored a newer catalog while this one ran.
        if let catalog { return catalog }
        if discovery == task { discovery = nil }
        setCatalog(result)
        return result
    }

    /// Stores `fresh` and tells subscribers when it differs.
    private func setCatalog(_ fresh: EngineCatalog) {
        let changed = catalog != fresh
        catalog = fresh
        if changed { broadcaster.bumpCatalogRevision() }
    }

    // MARK: Host

    /// The host that should serve `model`, created or replaced as needed,
    /// reserved for the caller until `release(_:)`: a replacement waits for
    /// every reservation on the old host before shutting it down.
    private func reserveHost(for model: AppModel) async throws -> EngineHost {
        while true {
            if isShutDown { throw EngineServiceError.shuttingDown }
            if let replacement {
                await replacement.value
                continue
            }
            let catalog = await currentCatalog()
            if isShutDown { throw EngineServiceError.shuttingDown }
            if replacement != nil { continue }
            guard let host else {
                let fresh = makeHost(catalog, observer)
                host = fresh
                broadcaster.setHost(fresh)
                return reserve(fresh)
            }
            if host.catalog == catalog { return reserve(host) }
            // The old host still serves this model exactly as discovered now.
            if host.catalog.descriptor(for: model.modelID) == catalog.descriptor(for: model.modelID) {
                return reserve(host)
            }
            await replaceHost(with: catalog)
        }
    }

    private func reserve(_ host: EngineHost) -> EngineHost {
        inFlight[ObjectIdentifier(host), default: 0] += 1
        return host
    }

    private func release(_ host: EngineHost) {
        let id = ObjectIdentifier(host)
        let count = inFlight[id, default: 1] - 1
        guard count == 0 else {
            inFlight[id] = count
            return
        }
        inFlight[id] = nil
        for waiter in drainWaiters.removeValue(forKey: id) ?? [] { waiter.resume() }
    }

    /// Returns once nothing holds a reservation on `host`.
    private func drain(_ host: EngineHost) async {
        let id = ObjectIdentifier(host)
        guard inFlight[id, default: 0] > 0 else { return }
        await withCheckedContinuation { drainWaiters[id, default: []].append($0) }
    }

    /// Waits until every request and load already handed the old host has
    /// finished, shuts it down (releasing its model), then installs a host
    /// over `catalog`. Concurrent callers wait; no request is cancelled.
    private func replaceHost(with catalog: EngineCatalog) async {
        if let replacement { return await replacement.value }
        let old = host
        let task = Task {
            if let old {
                await self.drain(old)
                await old.shutdown()
            }
            self.installHost(catalog)
        }
        replacement = task
        await task.value
    }

    private func installHost(_ catalog: EngineCatalog) {
        replacement = nil
        guard !isShutDown else { return }
        let fresh = makeHost(catalog, observer)
        host = fresh
        broadcaster.setHost(fresh)
        broadcaster.publish()
    }

    private nonisolated var observer: @Sendable (EngineHostEvent) -> Void {
        { [broadcaster] event in broadcaster.publish(after: event) }
    }

    // MARK: Status

    public nonisolated var status: EngineServiceStatus { broadcaster.current }

    public nonisolated func statusUpdates() -> AsyncStream<EngineServiceStatus> {
        broadcaster.subscribe()
    }

    // MARK: Lifecycle

    public func load(_ model: AppModel) async throws {
        try await withReservedHost(model) { host in
            do { try await host.load(model.modelID) } catch { throw Self.map(error, stage: .load) }
        }
        broadcaster.publish()
    }

    public func unload() async throws {
        if isShutDown { throw EngineServiceError.shuttingDown }
        await replacement?.value
        // Nothing can be resident without a host.
        guard let host else { return }
        _ = reserve(host)
        defer { release(host) }
        do { try await host.unload() } catch { throw Self.map(error, stage: .load) }
        broadcaster.publish()
    }

    public func cancelActiveGeneration() async {
        await host?.cancelActiveRequest()
    }

    /// Stops the current host first (cancelling its request and releasing its
    /// model), so quitting never waits behind a replacement still draining.
    public func shutdown() async {
        guard !isShutDown else { return }
        isShutDown = true
        await host?.shutdown()
        await replacement?.value
        await host?.shutdown()
        broadcaster.publish()
        broadcaster.finish()
    }

    /// Runs `body` on the host for `model`, reserved for the duration.
    private func withReservedHost<T: Sendable>(_ model: AppModel,
                                               _ body: (EngineHost) async throws -> T) async throws -> T {
        if let reason = Self.memoryGateReason(model, physicalMemory: physicalMemory) {
            throw EngineServiceError.notInstalled(model: model.name, reason: reason)
        }
        let host = try await reserveHost(for: model)
        defer { release(host) }
        return try await body(host)
    }

    // MARK: Generation

    public nonisolated func generate(model: AppModel, turns: [ChatTurn],
                                     options: GenerationOptions) -> GenerationStream {
        GenerationStream { continuation in
            Task {
                do {
                    let summary = try await self.run(model: model, turns: turns, options: options) { update in
                        continuation.yield(update)
                    }
                    continuation.yield(.completed(summary))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func run(model: AppModel, turns: [ChatTurn], options: GenerationOptions,
                     emit: @escaping @Sendable (GenerationUpdate) -> Void) async throws -> GenerationSummary {
        let broadcaster = self.broadcaster
        let messages = turns.map(Self.message)
        let tools = options.tools.map {
            MFTokenizer.FunctionDefinition(name: $0.name, description: $0.description,
                                           parameters: Self.engineJSON($0.parameters))
        }
        broadcaster.publish()
        defer { broadcaster.publish() }
        let completion = try await withReservedHost(model) { host in
            let started = StartFlag()
            do {
                return try await host.withModel(model.modelID,
                                                onQueued: { broadcaster.publishQueued() }) { engineModel in
                    started.set()
                    broadcaster.publish()
                    var config = engineModel.traits.generationDefaults
                    if let temperature = options.temperature { config.temperature = temperature }
                    if let seed = options.seed { config.seed = seed }
                    let effort: QwenReasoningEffort? = switch options.thinking {
                    case .some(true) where engineModel.traits.acceptsReasoningEffort: .xhigh
                    case .some(false) where engineModel.traits.acceptsReasoningEffort: .off
                    default: nil
                    }
                    let request = EngineRequest(messages: messages, tools: tools, generationConfig: config,
                                                maximumCompletionTokens: options.maximumCompletionTokens,
                                                reasoningEffort: effort,
                                                // Replay earlier turns' reasoning too (Qwen templates
                                                // drop it before the last user turn otherwise).
                                                preserveThinking: engineModel.traits.acceptsReasoningEffort,
                                                media: engineModel.traits.supportsVision
                                                    ? Self.media(turns) : [:])
                    let prepared = try await engineModel.prepare(request)
                    emit(.prepared(promptTokens: prepared.promptTokens.count))
                    return try await engineModel.generate(prepared) { event in
                        switch event {
                        case .content(let text): emit(.content(text))
                        case .reasoning(let text): emit(.reasoning(text))
                        case .toolCall(let call): emit(.toolCall(Self.toolCall(call)))
                        }
                    }
                }
            } catch {
                throw Self.map(error, stage: started.isSet ? .generation : .load)
            }
        }
        let finish: GenerationFinish = switch completion.finishReason {
        case .stop: .stop
        case .length: .length
        case .toolCalls: .toolCalls
        }
        return GenerationSummary(
            content: completion.content,
            reasoning: completion.reasoningContent,
            toolCalls: completion.toolCalls.map(Self.toolCall),
            finish: finish,
            promptTokens: completion.usage.promptTokens,
            completionTokens: completion.usage.completionTokens,
            prefillSeconds: completion.timings.prefillSeconds,
            decodeSeconds: completion.timings.decodeSeconds)
    }

    // MARK: Mapping

    /// The pictures of the user turns, decoded, by turn index. A picture
    /// ImageIO cannot read is left out (the turn's text still names it).
    static func media(_ turns: [ChatTurn]) -> [Int: [EngineMedia]] {
        var media: [Int: [EngineMedia]] = [:]
        for (index, turn) in turns.enumerated() where turn.role == .user && !turn.images.isEmpty {
            let decoded = turn.images.compactMap { EngineMedia.image(encoded: $0.data) }
            if !decoded.isEmpty { media[index] = decoded }
        }
        return media
    }

    static func message(_ turn: ChatTurn) -> MFTokenizer.Message {
        let role: MFTokenizer.Role = switch turn.role {
        case .system: .system
        case .user: .user
        case .assistant: .assistant
        case .tool: .tool
        }
        return MFTokenizer.Message(
            role: role, content: turn.content,
            toolCalls: turn.toolCalls.map {
                MFTokenizer.HistoricalToolCall(id: $0.id, name: $0.name, arguments: engineJSON($0.arguments))
            },
            toolCallID: turn.toolCallID, name: turn.toolName, reasoningContent: turn.reasoning)
    }

    static func toolCall(_ call: ParsedToolCall) -> ToolCall {
        ToolCall(id: call.id, name: call.name, arguments: appJSON(call.arguments), argumentsJSON: call.argumentsJSON)
    }

    static func engineJSON(_ value: ToolJSON) -> JSONValue {
        switch value {
        case .null: .null
        case .bool(let b): .bool(b)
        case .integer(let i): .integer(i)
        case .number(let d): .number(d)
        case .string(let s): .string(s)
        case .array(let a): .array(a.map(engineJSON))
        case .object(let o): .object(o.mapValues(engineJSON))
        }
    }

    static func appJSON(_ value: JSONValue) -> ToolJSON {
        switch value {
        case .null: .null
        case .bool(let b): .bool(b)
        case .integer(let i): .integer(i)
        case .unsignedInteger(let u): u <= UInt64(Int64.max) ? .integer(Int64(u)) : .number(Double(u))
        case .decimal(let d): .number(NSDecimalNumber(decimal: d).doubleValue)
        case .number(let d): .number(d)
        case .string(let s): .string(s)
        case .array(let a): .array(a.map(appJSON))
        case .object(let o): .object(o.mapValues(appJSON))
        }
    }

    /// Where an error came from: getting the model resident, or running the
    /// request on it.
    enum Stage { case load, generation }

    /// Maps every error to an `EngineServiceError`, except cancellation,
    /// which stays `CancellationError` so callers can tell a stop apart.
    static func map(_ error: any Error, stage: Stage = .generation) -> any Error {
        if error is CancellationError || error is EngineServiceError { return error }
        guard let engineError = error as? EngineError else {
            let message = String(describing: error)
            return switch stage {
            case .load: EngineServiceError.loadFailed(message)
            case .generation: EngineServiceError.generationFailed(message)
            }
        }
        return switch engineError {
        case .unknownModel(let id): EngineServiceError.unknownModel(id)
        case .modelNotRunnable(let id, let reason):
            EngineServiceError.notInstalled(model: AppModel(modelID: id)?.name ?? id, reason: reason)
        case .backendUnavailable: EngineServiceError.engineUnavailable
        case .invalidRequest(let message, _, let code):
            code == "context_length_exceeded"
                ? EngineServiceError.requestTooLarge(message)
                : EngineServiceError.invalidRequest(message)
        case .queueFull: EngineServiceError.busy
        case .shuttingDown: EngineServiceError.shuttingDown
        case .modelUnloaded: EngineServiceError.modelUnloaded
        }
    }

    static func status(_ status: EngineHostStatus) -> EngineServiceStatus {
        let phase: EngineServiceStatus.Phase = switch status.phase {
        case .empty: .empty
        case .loading: .loading
        case .ready: .ready
        case .unloading: .unloading
        }
        return EngineServiceStatus(
            phase: phase,
            residentModel: status.residentModelID.flatMap(AppModel.init(modelID:)),
            targetModel: status.targetModelID.flatMap(AppModel.init(modelID:)),
            isGenerating: status.activeRequests > 0,
            queuedRequests: status.queuedRequests,
            lastLoadFailure: status.lastLoadFailure?.message,
            lastLoadFailureModel: status.lastLoadFailure.flatMap { AppModel(modelID: $0.modelID) })
    }
}

/// Set once the request body runs, so an error before it counts as a load
/// failure and one after it as a generation failure.
private final class StartFlag: Sendable {
    private let value = Mutex(false)
    var isSet: Bool { value.withLock { $0 } }
    func set() { value.withLock { $0 = true } }
}

/// Fans host status out to every subscriber. Publishing re-reads the host's
/// synchronous snapshot, so subscribers always see a consistent status.
final class StatusBroadcaster: Sendable {
    private struct State {
        var host: EngineHost?
        var last = EngineServiceStatus.empty
        var subscribers: [UUID: AsyncStream<EngineServiceStatus>.Continuation] = [:]
        var isFinished = false
        var catalogRevision = 0

        /// The host's status with the service's own fields.
        var snapshot: EngineServiceStatus {
            var status = host.map { EngineService.status($0.status) } ?? last
            status.catalogRevision = catalogRevision
            return status
        }
    }

    private let state = Mutex(State())

    var current: EngineServiceStatus {
        state.withLock { $0.snapshot }
    }

    /// The catalog changed: subscribers re-read `models()`.
    func bumpCatalogRevision() {
        state.withLock { $0.catalogRevision += 1 }
        publish()
    }

    func setHost(_ host: EngineHost) {
        state.withLock { $0.host = host }
    }

    /// Publishes the status a host event leads to. The host reports
    /// `unloadFinished` (idle, requested, shutdown) just before it moves its
    /// status from unloading to empty, so that one is published as empty.
    func publish(after event: EngineHostEvent) {
        guard case .unloadFinished(let modelID, let reason) = event else { return publish() }
        // A swap keeps the status at loading; the next load publishes it.
        if case .swap = reason { return publish() }
        publish { status in
            guard status.phase == .unloading, status.targetModel?.modelID == modelID else { return }
            status.phase = .empty
            status.targetModel = nil
        }
    }

    /// The host calls `onQueued` just before it counts the new waiter and
    /// publishes nothing after, so this publishes the count it is about to
    /// have.
    func publishQueued() {
        publish { $0.queuedRequests += 1 }
    }

    func publish(adjust: (inout EngineServiceStatus) -> Void = { _ in }) {
        state.withLock { s in
            guard !s.isFinished else { return }
            var next = s.snapshot
            adjust(&next)
            guard next != s.last else { return }
            s.last = next
            for continuation in s.subscribers.values { continuation.yield(next) }
        }
    }

    /// Yields the current status first. After `finish()` the stream yields
    /// that one status and ends.
    func subscribe() -> AsyncStream<EngineServiceStatus> {
        let (stream, continuation) = AsyncStream<EngineServiceStatus>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let id = UUID()
        let finished = state.withLock { s -> Bool in
            continuation.yield(s.snapshot)
            guard !s.isFinished else { return true }
            s.subscribers[id] = continuation
            return false
        }
        if finished {
            continuation.finish()
            return stream
        }
        continuation.onTermination = { [weak self] _ in
            self?.state.withLock { _ = $0.subscribers.removeValue(forKey: id) }
        }
        return stream
    }

    func finish() {
        let subscribers = state.withLock { s in
            s.isFinished = true
            defer { s.subscribers.removeAll() }
            return Array(s.subscribers.values)
        }
        for continuation in subscribers { continuation.finish() }
    }
}
