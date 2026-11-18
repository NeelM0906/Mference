import Foundation
import Mference
import MferenceEngine
import Synchronization

/// The live `EngineServicing` over one shared `EngineHost` (spec 4.3.1).
///
/// Owned by U2. The A0 version is complete enough for U3 to build chat
/// against: catalog to picker rows, status stream, load/unload, streamed
/// generation with the model's own sampling defaults, cancellation.
public actor EngineService: EngineServicing {
    /// Ten minutes after the last token (decision L13).
    public static let defaultIdleUnload: Duration = .seconds(600)

    private var host: EngineHost
    private let makeHost: @Sendable (EngineCatalog, @escaping @Sendable (EngineHostEvent) -> Void) -> EngineHost
    private let discover: @Sendable () -> EngineCatalog
    private let broadcaster: StatusBroadcaster

    /// The production service: default discovery roots, native and llama.cpp
    /// backends, idle unload after `idleUnloadAfter`.
    public static func live(idleUnloadAfter: Duration? = defaultIdleUnload) -> EngineService {
        EngineService(
            discover: { EngineCatalogDiscovery.discoverDefault(scope: .product) },
            makeHost: { catalog, observer in
                EngineHost(catalog: catalog,
                           backends: [NativeBackend(), LlamaBackend()],
                           idleUnloadAfter: idleUnloadAfter,
                           maxQueuedRequests: 8,
                           observer: observer)
            })
    }

    /// Tests and previews: inject discovery and the host (with
    /// `FakeEngineBackend`).
    public init(discover: @escaping @Sendable () -> EngineCatalog,
                makeHost: @escaping @Sendable (EngineCatalog, @escaping @Sendable (EngineHostEvent) -> Void) -> EngineHost) {
        let broadcaster = StatusBroadcaster()
        self.broadcaster = broadcaster
        self.discover = discover
        self.makeHost = makeHost
        let host = makeHost(discover()) { broadcaster.publish(after: $0) }
        self.host = host
        broadcaster.setHost(host)
    }

    // MARK: Catalog

    public func models() async -> [ModelOption] {
        Self.options(catalog: host.catalog, status: host.status)
    }

    public func refreshCatalog() async {
        // Release the resident model first (after running and queued
        // requests), so the old and new host never hold a model at once; then
        // swap synchronously, so no request reaches the old host afterwards.
        let old = host
        try? await old.unload()
        guard host === old else { return }  // another refresh won the race
        let broadcaster = self.broadcaster
        let fresh = makeHost(discover()) { broadcaster.publish(after: $0) }
        host = fresh
        broadcaster.setHost(fresh)
        broadcaster.publish()
        await old.shutdown()
    }

    static func options(catalog: EngineCatalog, status: EngineHostStatus) -> [ModelOption] {
        AppModel.allCases.map { model in
            guard let descriptor = catalog.descriptor(for: model.modelID) else {
                let reason = catalog.unavailableReason(for: model.modelID) ?? "Not installed on this Mac."
                return ModelOption(model: model, availability: .notInstalled(reason: reason))
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
                               location: descriptor.location)
        }
    }

    // MARK: Status

    public nonisolated var status: EngineServiceStatus { broadcaster.current }

    public nonisolated func statusUpdates() -> AsyncStream<EngineServiceStatus> {
        broadcaster.subscribe()
    }

    // MARK: Lifecycle

    public func load(_ model: AppModel) async throws {
        do { try await host.load(model.modelID) } catch { throw Self.map(error) }
        broadcaster.publish()
    }

    public func unload() async throws {
        do { try await host.unload() } catch { throw Self.map(error) }
        broadcaster.publish()
    }

    public func cancelActiveGeneration() async {
        await host.cancelActiveRequest()
    }

    public func shutdown() async {
        await host.shutdown()
        broadcaster.publish()
        broadcaster.finish()
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
                    continuation.finish(throwing: Self.map(error))
                }
            }
        }
    }

    private func run(model: AppModel, turns: [ChatTurn], options: GenerationOptions,
                     emit: @escaping @Sendable (GenerationUpdate) -> Void) async throws -> GenerationSummary {
        let host = self.host
        let broadcaster = self.broadcaster
        let messages = turns.map(Self.message)
        let tools = options.tools.map {
            MFTokenizer.FunctionDefinition(name: $0.name, description: $0.description,
                                           parameters: Self.engineJSON($0.parameters))
        }
        broadcaster.publish()
        defer { broadcaster.publish() }
        let completion = try await host.withModel(model.modelID, onQueued: {
            // Fires before the host counts the waiter; publish after it has.
            Task { await Task.yield(); broadcaster.publish() }
        }) { engineModel in
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
                                        preserveThinking: engineModel.traits.acceptsReasoningEffort)
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

    static func map(_ error: any Error) -> any Error {
        guard let engineError = error as? EngineError else { return error }
        return switch engineError {
        case .unknownModel(let id): EngineServiceError.unknownModel(id)
        case .modelNotRunnable(let id, let reason):
            EngineServiceError.notInstalled(model: AppModel(modelID: id)?.name ?? id, reason: reason)
        case .backendUnavailable: EngineServiceError.engineUnavailable
        case .invalidRequest(let message, _, let code):
            code == "context_length_exceeded" ? EngineServiceError.requestTooLarge(message)
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
            lastLoadFailure: status.lastLoadFailure?.message)
    }
}

/// Fans host status out to every subscriber. Publishing re-reads the host's
/// synchronous snapshot, so subscribers always see a consistent status.
final class StatusBroadcaster: Sendable {
    private struct State {
        var host: EngineHost?
        var last = EngineServiceStatus.empty
        var subscribers: [UUID: AsyncStream<EngineServiceStatus>.Continuation] = [:]
    }

    private let state = Mutex(State())

    var current: EngineServiceStatus {
        state.withLock { s in s.host.map { EngineService.status($0.status) } ?? s.last }
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

    func publish(adjust: (inout EngineServiceStatus) -> Void = { _ in }) {
        state.withLock { s in
            var next = s.host.map { EngineService.status($0.status) } ?? s.last
            adjust(&next)
            guard next != s.last else { return }
            s.last = next
            for continuation in s.subscribers.values { continuation.yield(next) }
        }
    }

    func subscribe() -> AsyncStream<EngineServiceStatus> {
        let (stream, continuation) = AsyncStream<EngineServiceStatus>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let id = UUID()
        state.withLock { s in
            s.subscribers[id] = continuation
            continuation.yield(s.host.map { EngineService.status($0.status) } ?? s.last)
        }
        continuation.onTermination = { [weak self] _ in
            self?.state.withLock { _ = $0.subscribers.removeValue(forKey: id) }
        }
        return stream
    }

    func finish() {
        let subscribers = state.withLock { s in
            defer { s.subscribers.removeAll() }
            return Array(s.subscribers.values)
        }
        for continuation in subscribers { continuation.finish() }
    }
}
