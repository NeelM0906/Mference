import Foundation
import Synchronization
import Mference
import MferenceEngine

// Library mode runs on `EngineHost`. The HTTP layer speaks the server's own
// request types (`ValidatedChatRequest`, `PreparedGeneration`,
// `ServerCompletion`); this file maps them onto the engine's in both
// directions:
//
// - `ServerSessionEngineModel` / `ServerLoaderBackend` put a model the server
//   loaded itself (`ServerModelSession`, or a test fake) behind the engine
//   protocols, so the host owns its lifetime. The HTTP path unwraps it again
//   and calls it directly, so a native model behind this bridge runs exactly
//   the code it ran before the engine existed.
// - `EngineModelAdapter` puts any other `EngineModel` (the engine's own native
//   and llama.cpp backends) behind `ServerLoadedModel` for the HTTP path.

// MARK: - Request and result mapping

extension EngineRequest {
    init(_ request: ValidatedChatRequest) {
        self.init(messages: request.messages,
                  tools: request.tools,
                  generationConfig: request.generationConfig,
                  maximumCompletionTokens: request.maximumCompletionTokens,
                  reasoningEffort: request.reasoningEffort,
                  preserveThinking: request.preserveThinking,
                  inkling2Effort: request.inkling2Effort)
    }
}

extension ValidatedChatRequest {
    /// A request built from an engine request, for the bridged model. The
    /// streaming flags only shape the HTTP response, which the engine does
    /// not write, so they carry their non-streaming values.
    init(_ request: EngineRequest) {
        self.init(reasoningEffort: request.reasoningEffort,
                  preserveThinking: request.preserveThinking,
                  inkling2Effort: request.inkling2Effort,
                  messages: request.messages,
                  tools: request.tools,
                  stream: false,
                  includeUsage: false,
                  generationConfig: request.generationConfig,
                  maximumCompletionTokens: request.maximumCompletionTokens)
    }
}

extension ServerInferenceEvent {
    init(_ event: EngineEvent) {
        switch event {
        case .content(let text): self = .content(text)
        case .reasoning(let text): self = .reasoning(text)
        case .toolCall(let call): self = .toolCall(call)
        }
    }
}

extension EngineEvent {
    init(_ event: ServerInferenceEvent) {
        switch event {
        case .content(let text): self = .content(text)
        case .reasoning(let text): self = .reasoning(text)
        case .toolCall(let call): self = .toolCall(call)
        }
    }
}

extension ServerCompletion {
    init(_ completion: EngineCompletion) {
        let usage = completion.usage
        // The details object exists only where the backend counts reasoning
        // tokens: inventing a zero would claim a count nobody made.
        let details = usage.reasoningTokens.map {
            OpenAIUsage.CompletionTokensDetails(reasoningTokens: $0,
                                                visibleTokens: usage.visibleTokens)
        }
        self.init(content: completion.content,
                  toolCalls: completion.toolCalls,
                  finishReason: completion.finishReason.rawValue,
                  usage: OpenAIUsage(promptTokens: usage.promptTokens,
                                     completionTokens: usage.completionTokens,
                                     totalTokens: usage.totalTokens,
                                     cachedTokens: usage.cachedPromptTokens ?? 0,
                                     completionTokensDetails: details),
                  diagnostics: completion.diagnostics)
        reasoningContent = completion.reasoningContent
        prefillSeconds = completion.timings.prefillSeconds
        decodeSeconds = completion.timings.decodeSeconds
    }
}

extension EngineCompletion {
    init(_ completion: ServerCompletion) {
        let usage = completion.usage
        self.init(content: completion.content,
                  reasoningContent: completion.reasoningContent,
                  toolCalls: completion.toolCalls,
                  finishReason: EngineFinishReason(rawValue: completion.finishReason) ?? .stop,
                  usage: EngineUsage(promptTokens: usage.promptTokens,
                                     completionTokens: usage.completionTokens,
                                     cachedPromptTokens: usage.promptTokensDetails.cachedTokens,
                                     reasoningTokens: usage.completionTokensDetails?.reasoningTokens,
                                     visibleTokens: usage.completionTokensDetails?.visibleTokens),
                  timings: EngineTimings(prefillSeconds: completion.prefillSeconds,
                                         decodeSeconds: completion.decodeSeconds),
                  diagnostics: completion.diagnostics)
    }
}

extension ServerRequestError {
    /// The server's error for an engine error that has one; nil leaves the
    /// error as it is (a backend failure is a 500, as before).
    init?(_ error: EngineError) {
        switch error {
        case .unknownModel:
            self = .unknownModel
        case .modelNotRunnable(_, let reason):
            self = .invalid(message: reason, param: "model", code: "model_not_runnable")
        case .backendUnavailable(let engine):
            self = .invalid(message: "no \(engine.rawValue) backend in this build",
                            param: "model", code: "model_not_runnable")
        case .invalidRequest(let message, let param, let code):
            self = .invalid(message: message, param: param, code: code)
        case .queueFull:
            self = .queueFull
        case .shuttingDown, .modelUnloaded:
            return nil
        }
    }
}

/// Engine errors become the server's own request errors, and a host that is
/// shutting down becomes cancellation, which is how the server has always
/// ended work at shutdown.
func serverError(_ error: any Error) -> any Error {
    guard let engineError = error as? EngineError else { return error }
    if engineError == .shuttingDown { return CancellationError() }
    return ServerRequestError(engineError).map { $0 as any Error } ?? error
}

// MARK: - A server-loaded model behind the engine

/// A model the server loaded itself, owned by the engine host.
///
/// `unload()` drops the only strong reference the host chain holds, so a
/// `ServerModelSession` (Metal buffers, expert cache, KV) is released at that
/// point, before the host starts the next load.
public final class ServerSessionEngineModel: EngineModel {
    public let descriptor: EngineModelDescriptor
    public let traits: EngineModelTraits
    private let model: Mutex<(any ServerLoadedModel)?>

    public init(descriptor: EngineModelDescriptor,
                model: any ServerLoadedModel,
                maxContext: Int) {
        self.descriptor = descriptor
        self.traits = EngineModelTraits(
            chatDialect: model.chatDialect,
            generationDefaults: model.generationDefaults,
            maxContext: maxContext,
            acceptsReasoningEffort: model.acceptsReasoningEffort,
            supportsQwenReasoningEffort: model.supportsQwenReasoningEffort)
        self.model = Mutex(model)
    }

    /// The bridged model, for the HTTP path; nil once unloaded.
    public var serverModel: (any ServerLoadedModel)? {
        model.withLock { $0 }
    }

    private func loaded() throws -> any ServerLoadedModel {
        guard let model = serverModel else { throw EngineError.modelUnloaded(descriptor.id) }
        return model
    }

    public func prepare(_ request: EngineRequest) async throws -> EnginePreparedPrompt {
        try request.validateMedia(supportsVision: false)
        let prepared: PreparedGeneration
        do {
            prepared = try await loaded().prepare(ValidatedChatRequest(request))
        } catch let error as ServerRequestError {
            if case .invalid(let message, let param, let code) = error {
                throw EngineError.invalidRequest(message: message, param: param, code: code)
            }
            throw error
        }
        return EnginePreparedPrompt(request: request,
                                    promptTokens: prepared.promptIDs,
                                    needsToolTemplate: prepared.needsToolTemplate)
    }

    public func generate(_ prepared: EnginePreparedPrompt,
                         onEvent: @escaping @Sendable (EngineEvent) -> Void) async throws -> EngineCompletion {
        let generation = PreparedGeneration(request: ValidatedChatRequest(prepared.request),
                                            promptIDs: prepared.promptTokens,
                                            needsToolTemplate: prepared.needsToolTemplate)
        let completion = try await loaded().generate(generation) { onEvent(EngineEvent($0)) }
        return EngineCompletion(completion)
    }

    public func unload() async {
        model.withLock { $0 = nil }
    }
}

/// Loads `.gturbo` installs with a server loader: in production
/// `ServerModelSession.load`, in tests an injected fake.
public struct ServerLoaderBackend: EngineBackend {
    public typealias Loader = @Sendable (URL) async throws -> any ServerLoadedModel

    public let engine = EngineKind.mference
    private let loader: Loader

    public init(loader: @escaping Loader) {
        self.loader = loader
    }

    public func load(_ descriptor: EngineModelDescriptor,
                     options: EngineLoadOptions) async throws -> any EngineModel {
        let model = try await loader(descriptor.location)
        let maxContext = (model as? ServerModelSession)?.maxContext
            ?? descriptor.resolvedContext(options: options)
        return ServerSessionEngineModel(descriptor: descriptor, model: model, maxContext: maxContext)
    }
}

// MARK: - An engine model behind the server

/// Any `EngineModel` presented to the HTTP path as a `ServerLoadedModel`.
struct EngineModelAdapter: ServerLoadedModel {
    let model: any EngineModel

    var chatDialect: ChatDialect { model.traits.chatDialect }
    var generationDefaults: GenerationConfig { model.traits.generationDefaults }
    var acceptsReasoningEffort: Bool { model.traits.acceptsReasoningEffort }
    var supportsQwenReasoningEffort: Bool { model.traits.supportsQwenReasoningEffort }

    func prepare(_ request: ValidatedChatRequest) async throws -> PreparedGeneration {
        let prepared: EnginePreparedPrompt
        do {
            prepared = try await model.prepare(EngineRequest(request))
        } catch {
            throw serverError(error)
        }
        return PreparedGeneration(request: request,
                                  promptIDs: prepared.promptTokens,
                                  needsToolTemplate: prepared.needsToolTemplate)
    }

    func generate(_ prepared: PreparedGeneration,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        let engineRequest = EngineRequest(prepared.request)
        let prompt = EnginePreparedPrompt(request: engineRequest,
                                          promptTokens: prepared.promptIDs,
                                          needsToolTemplate: prepared.needsToolTemplate)
        do {
            let completion = try await model.generate(prompt) { onEvent(ServerInferenceEvent($0)) }
            return ServerCompletion(completion)
        } catch {
            throw serverError(error)
        }
    }
}

/// The server-side view of a resident engine model.
func serverLoadedModel(for model: any EngineModel) throws -> any ServerLoadedModel {
    if let bridged = model as? ServerSessionEngineModel {
        guard let inner = bridged.serverModel else {
            throw EngineError.modelUnloaded(model.descriptor.id)
        }
        return inner
    }
    return EngineModelAdapter(model: model)
}
