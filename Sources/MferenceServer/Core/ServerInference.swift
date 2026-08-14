import CryptoKit
import Foundation
import Mference

public enum ServerInferenceEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

public struct ServerCompletion: Equatable, Sendable {
    public let content: String
    public let toolCalls: [ParsedToolCall]
    public let finishReason: String
    public let usage: OpenAIUsage

    public init(content: String,
                toolCalls: [ParsedToolCall],
                finishReason: String,
                usage: OpenAIUsage) {
        self.content = content
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
    }
}

/// A request that has been rendered and measured against the context window.
public struct PreparedGeneration: Sendable {
    public let request: ValidatedChatRequest
    public let promptIDs: [Int32]
    public let needsToolTemplate: Bool

    public init(request: ValidatedChatRequest,
                promptIDs: [Int32] = [],
                needsToolTemplate: Bool = false) {
        self.request = request
        self.promptIDs = promptIDs
        self.needsToolTemplate = needsToolTemplate
    }
}

public protocol ServerInferenceBackend: Sendable {
    /// Everything that can reject a request must happen here, because the
    /// caller commits the response status once `generate` starts: a streaming
    /// request has `200` and the SSE head on the wire by then, and no status
    /// left to send.
    func prepare(_ request: ValidatedChatRequest) async throws -> PreparedGeneration

    func generate(_ prepared: PreparedGeneration,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
}

extension ServerInferenceBackend {
    /// Backends that do not tokenize inherit a pass-through. A backend that
    /// renders a prompt must override this, or `generate` receives no tokens.
    public func prepare(_ request: ValidatedChatRequest) async throws -> PreparedGeneration {
        PreparedGeneration(request: request)
    }
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    private var active = false
    private var waiters: [Waiter] = []
    private var claims = 0
    private var shuttingDown = false

    public init(queueLimit: Int) {
        self.queueLimit = queueLimit
    }

    /// Claims a place, renders, then waits its turn. Rendering stays outside
    /// the gate so it never stalls the request that is generating, but the
    /// place is claimed before it starts: a request the queue has no room for
    /// is turned away without having paid for tokenization first.
    public func run<Rendered: Sendable, T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        render: @escaping @Sendable () async throws -> Rendered,
        _ operation: @escaping @Sendable (Rendered) async throws -> T
    ) async throws -> T {
        try claim()
        let rendered: Rendered
        do {
            rendered = try await render()
        } catch {
            claims -= 1
            throw error
        }
        try await acquire(onQueued: onQueued)
        defer { release() }
        return try await operation(rendered)
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await run(onQueued: onQueued, render: {}) { _ in try await operation() }
    }

    /// Occupancy is the one generating request plus the queue behind it.
    /// Outstanding claims count towards it from the moment they are taken, so
    /// two requests rendering at once cannot both take the same free place.
    private func claim() throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        let occupancy = (active ? 1 : 0) + waiters.count + claims
        guard occupancy < queueLimit + 1 else { throw ServerRequestError.queueFull }
        claims += 1
    }

    private func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        // The claim becomes the active slot or a place in the queue.
        claims -= 1
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if !active {
            active = true
            return
        }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release() {
        if waiters.isEmpty {
            active = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { active }
}

public actor ServerModelSession: ServerInferenceBackend {
    /// Chat dialect of the loaded tokenizer; drives request-validation rules.
    public nonisolated let chatDialect: ChatDialect
    /// Family-derived API model identifier used when --model-id is absent.
    public nonisolated var defaultModelID: String {
        switch modelFamily {
        case .gemma4: return "gemma-4-26b-a4b-it"
        case .qwen36: return "qwen3.6-35b-a3b"
        case .qwen38: return "qwen3.8-27b-4bit"
        case .deepseekV4Flash: return "deepseek-v4-flash-2bit-dq"
        case .inklingSmall: return "inkling-small-4bit"
        case .maple: return "maple-preview-2bit-mlx"
        }
    }
    private nonisolated let modelFamily: ModelFamily

    private let context: MetalContext
    private let model: Model
    private let tokenizer: MFTokenizer
    private let runner: any ContinuableLogitProducer
    private let scratch: RawCompletionScratch
    private let prefillConfig: PrefillRuntimeConfig
    private let maxContext: Int
    private let promptCacheMode: ServerPromptCacheMode
    private let promptCacheDomain: ServerPromptCacheDomain
    private var promptCache = ServerPromptCache()

    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            promptCacheMode: ServerPromptCacheMode = .singlePrefix) async throws -> ServerModelSession {
        let family = try ManifestReader.peekFamily(directoryURL: modelDirectory)
        let tokenizerFolder = MFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw MFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        let tokenizer = try await MFTokenizer.load(from: tokenizerFolder, family: family)
        // DeepSeek ships no chat_template.jinja — its chat framing is native
        // Swift — so the prompt-cache identity hashes a pinned constant that
        // changes only when that native render does. Every other dialect
        // still requires the bundled template.
        let templateData: Data
        if FileManager.default.fileExists(atPath: templateURL.path) {
            templateData = try Data(contentsOf: templateURL)
        } else if tokenizer.dialect == .deepseek {
            templateData = Data("native:deepseek:v1".utf8)
        } else {
            throw MFTokenizerError.missingToolTemplate
        }
        let context = try MetalContext()
        let streamingMode = RuntimeConfiguration.defaultExpertStreamingMode(
            for: family,
            expertPoolBytes: try ExpertPoolInspector.poolByteSize(
                directoryURL: modelDirectory),
            coreWeightsBytes: try ExpertPoolInspector.coreWeightsByteSize(
                directoryURL: modelDirectory))
        let configSlots: Int
        switch streamingMode {
        case .pread(let slots): configSlots = slots
        case .resident:
            configSlots = RuntimeConfiguration.allowedExpertCacheSlots.max()!
        }
        let runtime = RuntimeConfiguration(
            expertCacheSlots: configSlots,
            forceLogitsHead: true)
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            streamingMode: streamingMode,
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let forwardRuntime = try ForwardRunnerFactory.make(model: model,
                                                            context: context,
                                                            maxContext: maxContext,
                                                            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let templateDigest = SHA256.hash(data: templateData)
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            forwardRuntime.prefillConfig.mode.rawValue,
            String(forwardRuntime.prefillConfig.chunkTokens),
            runtime.headPath.rawValue,
        ].joined(separator: ":")
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: forwardRuntime.kvStorageMode.rawValue,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        return ServerModelSession(context: context,
                                  model: model,
                                  tokenizer: tokenizer,
                                  runner: forwardRuntime.producer,
                                  scratch: scratch,
                                  prefillConfig: forwardRuntime.prefillConfig,
                                  maxContext: maxContext,
                                  promptCacheMode: promptCacheMode,
                                  promptCacheDomain: promptCacheDomain)
    }

    private init(context: MetalContext,
                 model: Model,
                 tokenizer: MFTokenizer,
                 runner: any ContinuableLogitProducer,
                 scratch: RawCompletionScratch,
                 prefillConfig: PrefillRuntimeConfig,
                 maxContext: Int,
                 promptCacheMode: ServerPromptCacheMode,
                 promptCacheDomain: ServerPromptCacheDomain) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.chatDialect = tokenizer.dialect
        self.modelFamily = model.config.family
        self.runner = runner
        self.scratch = scratch
        self.prefillConfig = prefillConfig
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheDomain = promptCacheDomain
    }

    /// Renders the prompt and checks it against the context window. Runs
    /// before the response status is committed, and touches no generation
    /// state, so it is safe to run while another request is generating.
    public func prepare(_ request: ValidatedChatRequest) throws -> PreparedGeneration {
        let needsToolTemplate = !request.tools.isEmpty
            || request.messages.contains {
                $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
            }
        let promptIDs: [Int32]
        if needsToolTemplate {
            promptIDs = try tokenizer.encodeToolChat(messages: request.messages, tools: request.tools)
        } else {
            let rendered = try tokenizer.applyChatTemplate(request.messages)
            promptIDs = tokenizer.encode(rendered, addBOS: false)
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return PreparedGeneration(request: request,
                                  promptIDs: promptIDs,
                                  needsToolTemplate: needsToolTemplate)
    }

    public func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let request = prepared.request
        let needsToolTemplate = prepared.needsToolTemplate
        let promptIDs = prepared.promptIDs
        var completed = false
        defer {
            if !completed {
                promptCache.invalidate()
                runner.reset()
            }
        }

        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        if promptCacheMode == .singlePrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: request,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer) {
            case .miss:
                promptCache.invalidate()
                effectivePromptIDs = promptIDs
                completionStart = .reset
            case .hit(let effective, let cached):
                effectivePromptIDs = effective
                completionStart = .resume(cachedPromptTokens: cached)
            }
        } else {
            promptCache.invalidate()
            effectivePromptIDs = promptIDs
            completionStart = .reset
        }
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        config.stopStrings = []

        let decoder = needsToolTemplate || tokenizer.generationPromptStartsInThinking
            ? StructuredAssistantDecoder(
                tokenizer: tokenizer,
                allowedTools: Set(request.tools.map(\.name)),
                startsInThought: tokenizer.generationPromptStartsInThinking)
            : nil
        var stopMatcher = StreamingStopMatcher(stops: request.generationConfig.stopStrings)
        var content = ""
        var calls: [ParsedToolCall] = []
        var decodingError: Error?
        var shouldStop = false

        let result = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: effectivePromptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: completionStart,
            shouldStop: { shouldStop }) { progress in
                guard decodingError == nil else { return }
                do {
                    switch progress {
                    case .prefill:
                        break
                    case .token(_, let tokenID, let delta):
                        let events = if let decoder {
                            try decoder.consume(tokenID: tokenID, delta: delta)
                        } else {
                            delta.isEmpty ? [] : [StructuredAssistantEvent.content(delta)]
                        }
                        for event in events {
                            switch event {
                            case .content(let text):
                                let visible = stopMatcher.push(text)
                                if !visible.isEmpty {
                                    content += visible
                                    onEvent(.content(visible))
                                }
                                if stopMatcher.isStopped { shouldStop = true }
                            case .toolCall(let call):
                                calls.append(call)
                                onEvent(.toolCall(call))
                            }
                        }
                    case .tail(let text):
                        // Flush text must pass through the decoder like any
                        // delta: committing it directly would reorder it
                        // ahead of a withheld DSML-prefix tail and skip
                        // marker scanning.
                        let events = if let decoder {
                            try decoder.consumeFlushedText(text)
                        } else {
                            text.isEmpty ? [] : [StructuredAssistantEvent.content(text)]
                        }
                        for event in events {
                            switch event {
                            case .content(let flushed):
                                let visible = stopMatcher.push(flushed)
                                if !visible.isEmpty {
                                    content += visible
                                    onEvent(.content(visible))
                                }
                                if stopMatcher.isStopped { shouldStop = true }
                            case .toolCall(let call):
                                calls.append(call)
                                onEvent(.toolCall(call))
                            }
                        }
                    }
                } catch {
                    decodingError = error
                    shouldStop = true
                }
        }
        if let decodingError { throw decodingError }
        if let decoder {
            for event in try decoder.finish() {
                if case .content(let text) = event {
                    let visible = stopMatcher.push(text)
                    if !visible.isEmpty {
                        content += visible
                        onEvent(.content(visible))
                    }
                }
            }
        }
        if needsToolTemplate, result.reason == .toolCalls, calls.isEmpty {
            throw GemmaToolCallParserError.malformed
        }
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
        let reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        if promptCacheMode == .singlePrefix {
            promptCache.publish(
                domain: promptCacheDomain,
                request: request,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopMatcher.isStopped)
        }
        completed = true
        return ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: result.prefillTokens,
                               completionTokens: result.newTokens,
                               totalTokens: result.prefillTokens + result.newTokens,
                               cachedTokens: result.cachedPromptTokens))
    }
}
