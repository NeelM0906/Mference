import CryptoKit
import Foundation
import Mference

/// Something the native backend reports for an operator log.
public enum NativeBackendNotice: Sendable, Hashable {
    /// `--verify` fell back from the receipt to hashing; the runtime's text.
    case verifyAutoFallback(String)
    /// The multi-prefix cache released retained state.
    case promptCacheShed(reason: String, releasedBytes: Int)
}

/// Runs `.gturbo` installs on Mference's own runtime.
///
/// An adapter, not a new implementation: `NativeEngineModel` is
/// `MferenceServer`'s `ServerModelSession` lifted into the engine, with the
/// same load sequence (`Model.load`, `ForwardRunnerFactory.make`), the same
/// rendering, single- and multi-prefix prompt reuse, structured decoding, stop
/// handling, and `runRawCompletion` loop, so its output is byte-identical to
/// the server's.
public struct NativeBackend: EngineBackend {
    public let engine = EngineKind.mference
    private let onNotice: @Sendable (NativeBackendNotice) -> Void

    public init(onNotice: @escaping @Sendable (NativeBackendNotice) -> Void = { _ in }) {
        self.onNotice = onNotice
    }

    public func load(_ descriptor: EngineModelDescriptor,
                     options: EngineLoadOptions) async throws -> any EngineModel {
        guard case .gturbo = descriptor.format else {
            throw EngineError.backendUnavailable(descriptor.engine)
        }
        return try await NativeEngineModel.load(descriptor: descriptor, options: options, onNotice: onNotice)
    }

    /// The runtime configuration a native model is served with.
    public static func runtimeConfiguration(
        family: ModelFamily,
        expertCacheSlots: Int,
        shadowPrefetchBudget: Int? = nil,
        prefillChunkTokens: Int? = nil,
        reserveFullKV: Bool = false,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> RuntimeConfiguration {
        RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            prefillChunkTokens: prefillChunkTokens ?? RuntimeConfiguration.defaultServerPrefillChunkTokens(
                for: family, physicalMemoryBytes: physicalMemoryBytes, environment: environment),
            forceLogitsHead: true,
            shadowPrefetchBudget: shadowPrefetchBudget,
            kvGrowthTokens: reserveFullKV ? nil : RuntimeConfiguration.defaultKVGrowthTokens)
    }

    /// `multi-prefix` needs a runner that snapshots its prefix state; any
    /// other runner is served exactly as `single-prefix`.
    public static func effectivePromptCacheMode(_ requested: EnginePromptCacheMode,
                                                runnerSnapshotsPrefixes: Bool) -> EnginePromptCacheMode {
        requested == .multiPrefix && !runnerSnapshotsPrefixes ? .singlePrefix : requested
    }
}

extension EnginePromptCacheBytes {
    static let gibibyte = 1 << 30

    /// `auto` is min(4 GiB, physical / 8) on hosts with at least 24 GiB,
    /// 1 GiB on 16 to under 24 GiB, and 0 (single-prefix behaviour) below.
    public func resolved(family: ModelFamily? = nil,
                         physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Int {
        switch self {
        case .bytes(let value):
            return max(0, value)
        case .auto:
            let physical = Int(clamping: physicalMemoryBytes)
            let general: Int
            if physical >= 24 * Self.gibibyte {
                general = min(4 * Self.gibibyte, physical / 8)
            } else if physical >= 16 * Self.gibibyte {
                general = Self.gibibyte
            } else {
                general = 0
            }
            return general
        }
    }
}

extension EngineRequest {
    /// The prompt-prefix cache's identity of this turn.
    var promptCacheTurn: PromptPrefixCacheTurn {
        PromptPrefixCacheTurn(messages: messages,
                              tools: tools,
                              reasoningEffort: reasoningEffort,
                              preserveThinking: preserveThinking)
    }
}

extension PromptPrefixCacheMatch {
    /// `prepare` admitted the rendered prompt, but a hit can be longer: the KV
    /// cache keeps generated tokens, such as a thought, that the re-render
    /// drops. A hit that no longer fits is treated as a miss.
    fileprivate func fitting(maxContext: Int) -> PromptPrefixCacheMatch {
        if case .hit(let effective, _) = self, effective.count >= maxContext {
            return .miss
        }
        return self
    }
}

/// One loaded `.gturbo` model. See `NativeBackend`.
public actor NativeEngineModel: EngineModel {
    public nonisolated let descriptor: EngineModelDescriptor
    public nonisolated let traits: EngineModelTraits

    /// Everything the loaded model holds. Dropped by `unload()`, which frees
    /// the weights' Metal buffers, the expert cache, and the KV with it.
    private struct Loaded {
        let context: MetalContext
        let model: Model
        let tokenizer: MFTokenizer
        let runner: any ContinuableLogitProducer
        let scratch: RawCompletionScratch
        let prefillConfig: PrefillRuntimeConfig
        /// The context this model runs with.
        let maxContext: Int
        /// `multi-prefix` on a runner without prefix snapshots is stored as
        /// `singlePrefix`, so that runner is served exactly as single-prefix.
        let promptCacheMode: EnginePromptCacheMode
        let promptCacheDomain: PromptPrefixCacheDomain
        /// Non-nil exactly when `promptCacheMode == .multiPrefix`.
        let prefixStates: PrefixStateCache?
        let snapshotRunner: (any PrefixStateSnapshotting)?
        /// The multi-prefix cache's live-memory policy and pressure
        /// registration; nil without the cache.
        let memoryGovernor: PrefixCacheMemoryGovernor?
    }

    private var loaded: Loaded?
    private var promptCache = PromptPrefixCache()
    private let onNotice: @Sendable (NativeBackendNotice) -> Void

    /// The context a load runs with: the descriptor's resolution (the
    /// request, or the native context for `max`, lowered to its host cap),
    /// lowered again by the family's cap for this Mac's memory, so a
    /// descriptor not built by discovery is capped too. For every family
    /// without a host cap this is `options.nativeMaxContext ??
    /// family.maximumContext`, as before.
    static func loadContext(descriptor: EngineModelDescriptor, options: EngineLoadOptions, family: ModelFamily,
                            physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Int {
        let resolved = descriptor.resolvedContext(options: options)
        return family.memoryContextCap(physicalMemory: physicalMemory).map { min(resolved, $0) } ?? resolved
    }

    static func load(descriptor: EngineModelDescriptor,
                     options: EngineLoadOptions,
                     onNotice: @escaping @Sendable (NativeBackendNotice) -> Void) async throws -> NativeEngineModel {
        let modelDirectory = descriptor.location
        let family = try ManifestReader.peekFamily(directoryURL: modelDirectory)
        let maxContext = loadContext(descriptor: descriptor, options: options, family: family)
        let tokenizerFolder = MFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw MFTokenizerError.missingTokenizerSidecar(
                modelDirectory.appendingPathComponent("tokenizer", isDirectory: true).path)
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        let tokenizer = try await MFTokenizer.load(forModelDirectory: modelDirectory)
        // The prompt-cache identity hashes the installed chat template.
        let templateData: Data
        if FileManager.default.fileExists(atPath: templateURL.path) {
            templateData = try Data(contentsOf: templateURL)
        } else {
            throw MFTokenizerError.missingToolTemplate
        }
        let context = try MetalContext()
        let streamingMode = RuntimeConfiguration.defaultExpertStreamingMode(
            for: family,
            expertPoolBytes: try ExpertPoolInspector.poolByteSize(directoryURL: modelDirectory),
            coreWeightsBytes: try ExpertPoolInspector.coreWeightsByteSize(directoryURL: modelDirectory))
        let configSlots: Int
        switch streamingMode {
        case .pread(let slots): configSlots = slots
        case .resident:
            configSlots = RuntimeConfiguration.allowedExpertCacheSlots.max()!
        }
        let runtime = NativeBackend.runtimeConfiguration(family: family, expertCacheSlots: configSlots,
                                                         shadowPrefetchBudget: options.shadowPrefetchBudget,
                                                         prefillChunkTokens: options.prefillChunkTokens,
                                                         reserveFullKV: options.reserveFullKV)
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            streamingMode: streamingMode,
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: options.integrityPolicy)
        if let notice = model.verifyAutoFallbackNotice {
            onNotice(.verifyAutoFallback(notice))
        }
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
        let promptCacheDomain = PromptPrefixCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: forwardRuntime.kvStorageMode.rawValue,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        let session = NativeEngineModel(descriptor: descriptor,
                                        context: context,
                                        model: model,
                                        tokenizer: tokenizer,
                                        runner: forwardRuntime.producer,
                                        scratch: scratch,
                                        prefillConfig: forwardRuntime.prefillConfig,
                                        maxContext: maxContext,
                                        promptCacheMode: options.promptCacheMode,
                                        promptCacheBytes: options.promptCacheBytes.resolved(family: family),
                                        promptCacheDomain: promptCacheDomain,
                                        onNotice: onNotice)
        await session.startMemoryPressureRelief()
        return session
    }

    private init(descriptor: EngineModelDescriptor,
                 context: MetalContext,
                 model: Model,
                 tokenizer: MFTokenizer,
                 runner: any ContinuableLogitProducer,
                 scratch: RawCompletionScratch,
                 prefillConfig: PrefillRuntimeConfig,
                 maxContext: Int,
                 promptCacheMode requestedPromptCacheMode: EnginePromptCacheMode,
                 promptCacheBytes: Int,
                 promptCacheDomain: PromptPrefixCacheDomain,
                 onNotice: @escaping @Sendable (NativeBackendNotice) -> Void) {
        self.descriptor = descriptor
        self.traits = EngineModelTraits(
            chatDialect: tokenizer.dialect,
            generationDefaults: tokenizer.generationDefaults,
            maxContext: maxContext,
            acceptsReasoningEffort: tokenizer.acceptsReasoningEffort,
            supportsQwenReasoningEffort: tokenizer.supportsQwenReasoningEffort,
            reusesPromptPrefixes: requestedPromptCacheMode != .off,
            usesDraftDecoding: false)
        self.onNotice = onNotice
        let snapshots = (runner as? any PrefixStateSnapshotting).flatMap {
            $0.supportsPrefixStateSnapshots ? $0 : nil
        }
        let promptCacheMode = NativeBackend.effectivePromptCacheMode(
            requestedPromptCacheMode, runnerSnapshotsPrefixes: snapshots != nil)
        let snapshotRunner: (any PrefixStateSnapshotting)?
        let memoryGovernor: PrefixCacheMemoryGovernor?
        let prefixStates: PrefixStateCache?
        if promptCacheMode == .multiPrefix, let snapshots {
            snapshotRunner = snapshots
            let governor = PrefixCacheMemoryGovernor()
            memoryGovernor = governor
            prefixStates = PrefixStateCache(ceilingBytes: promptCacheBytes, gate: governor)
        } else {
            snapshotRunner = nil
            memoryGovernor = nil
            prefixStates = nil
        }
        self.loaded = Loaded(context: context,
                             model: model,
                             tokenizer: tokenizer,
                             runner: runner,
                             scratch: scratch,
                             prefillConfig: prefillConfig,
                             maxContext: maxContext,
                             promptCacheMode: promptCacheMode,
                             promptCacheDomain: promptCacheDomain,
                             prefixStates: prefixStates,
                             snapshotRunner: snapshotRunner,
                             memoryGovernor: memoryGovernor)
    }

    /// Points the governor's pressure events at an idle safe point on this
    /// actor. The handler only schedules the hop; the hop defers to the end of
    /// a running request.
    func startMemoryPressureRelief() {
        loaded?.memoryGovernor?.setRaiseAction { [weak self] in
            Task { await self?.idleMemorySafePoint() }
        }
    }

    private func idleMemorySafePoint() {
        guard let prefixStates = loaded?.prefixStates, let memoryGovernor = loaded?.memoryGovernor else { return }
        let before = prefixStates.retainedBytes
        logShed(memoryGovernor.idleSafePoint(prefixStates), from: before)
    }

    private func logShed(_ reason: PrefixCacheShedReason?, from before: Int) {
        guard let reason, let prefixStates = loaded?.prefixStates else { return }
        onNotice(.promptCacheShed(reason: reason.rawValue, releasedBytes: before - prefixStates.retainedBytes))
    }

    public func unload() async {
        loaded?.memoryGovernor?.setRaiseAction {}
        loaded = nil
        promptCache.invalidate()
    }

    /// Renders the prompt and checks it against the context window. Touches
    /// no generation state.
    public func prepare(_ request: EngineRequest) throws -> EnginePreparedPrompt {
        guard let loaded else { throw EngineError.modelUnloaded(descriptor.id) }
        let tokenizer = loaded.tokenizer
        let maxContext = loaded.maxContext
        let needsToolTemplate = request.needsToolTemplate
        let promptIDs: [Int32]
        do {
            promptIDs = try tokenizer.encodeChat(messages: request.messages, tools: request.tools,
                                                reasoningEffort: request.reasoningEffort,
                                                preserveThinking: request.preserveThinking)
        } catch {
            throw EngineError.invalidRequest(message: String(describing: error),
                                             param: "messages", code: "invalid_chat_template")
        }
        guard promptIDs.count < maxContext else {
            throw EngineError.invalidRequest(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return EnginePreparedPrompt(request: request,
                                    promptTokens: promptIDs,
                                    needsToolTemplate: needsToolTemplate)
    }

    public func generate(
        _ prepared: EnginePreparedPrompt,
        onEvent: @escaping @Sendable (EngineEvent) -> Void
    ) async throws -> EngineCompletion {
        guard let loaded else { throw EngineError.modelUnloaded(descriptor.id) }
        let context = loaded.context
        let model = loaded.model
        let tokenizer = loaded.tokenizer
        let runner = loaded.runner
        let scratch = loaded.scratch
        let prefillConfig = loaded.prefillConfig
        let maxContext = loaded.maxContext
        let promptCacheMode = loaded.promptCacheMode
        let promptCacheDomain = loaded.promptCacheDomain
        let prefixStates = loaded.prefixStates
        let snapshotRunner = loaded.snapshotRunner
        let memoryGovernor = loaded.memoryGovernor

        let request = prepared.request
        let needsToolTemplate = prepared.needsToolTemplate
        let promptIDs = prepared.promptTokens
        var completed = false
        var ended = false
        // The end-of-request safe point. A finished request runs it before
        // its diagnostics are captured, so they show any shed; a failed one
        // runs it from the defer, after `didFail`.
        func endRequest() {
            guard !ended else { return }
            ended = true
            if let prefixStates, let memoryGovernor {
                let before = prefixStates.retainedBytes
                logShed(memoryGovernor.requestEnded(prefixStates), from: before)
            }
        }
        if let prefixStates, let memoryGovernor {
            let before = prefixStates.retainedBytes
            logShed(memoryGovernor.requestStarted(prefixStates), from: before)
        }
        defer {
            if !completed {
                promptCache.invalidate()
                runner.reset()
                prefixStates?.didFail()
            }
            endRequest()
        }

        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        if let prefixStates, let snapshotRunner {
            let begun = try prefixStates.begin(
                domain: promptCacheDomain,
                turn: request.promptCacheTurn,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer,
                maxContext: maxContext,
                runner: snapshotRunner)
            effectivePromptIDs = begun.effectivePromptIDs
            completionStart = begun.start
        } else if promptCacheMode == .singlePrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                turn: request.promptCacheTurn,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer).fitting(maxContext: maxContext) {
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
            throw EngineError.invalidRequest(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        config.stopStrings = []

        let startsInThinking = tokenizer.startsInThinking(
            reasoningEffort: request.reasoningEffort, promptIDs: effectivePromptIDs)
        let countsPayloadTokens = tokenizer.dialect == .chatml
            || tokenizer.dialect == .minicpm
            || tokenizer.dialect == .inkling2
        let decoder = countsPayloadTokens || needsToolTemplate || startsInThinking
            ? StructuredAssistantDecoder(
                tokenizer: tokenizer,
                allowedTools: Set(request.tools.map(\.name)),
                startsInThought: startsInThinking,
                toolDefinitions: request.tools)
            : nil
        var stopMatcher = StreamingStopMatcher(stops: request.generationConfig.stopStrings)
        var content = ""
        var reasoning = ""
        // Inkling 2 always frames its thinking in blocks the template renders
        // back, so its reasoning is reported like the source templates'.
        if tokenizer.usesSourceTemplate(reasoningEffort: request.reasoningEffort)
            || tokenizer.dialect == .inkling2 {
            decoder?.onReasoning = { text in
                reasoning += text
                onEvent(.reasoning(text))
            }
        }
        var calls: [ParsedToolCall] = []
        var decodingError: Error?
        var shouldStop = false

        let checkpoint: (position: Int, capture: () throws -> Void)? = nil

        let result = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: effectivePromptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: completionStart,
            prefillCheckpoint: checkpoint,
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
            throw ToolCallParserError.malformed
        }
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
        let reason: EngineFinishReason
        if !calls.isEmpty {
            reason = .toolCalls
        } else if result.reason == .maxTokens {
            reason = .length
        } else {
            reason = .stop
        }
        try Task.checkCancellation()
        if let prefixStates, let snapshotRunner {
            prefixStates.didFinish(
                entry: PromptPrefixCache.makeEntry(
                    domain: promptCacheDomain,
                    turn: request.promptCacheTurn,
                    content: content,
                    calls: calls,
                    result: result,
                    reasoningContent: reasoning.isEmpty ? nil : reasoning,
                    stopStringFiltered: stopMatcher.isStopped),
                runner: snapshotRunner)
        } else if promptCacheMode == .singlePrefix {
            promptCache.publish(
                domain: promptCacheDomain,
                turn: request.promptCacheTurn,
                content: content,
                calls: calls,
                result: result,
                reasoningContent: reasoning.isEmpty ? nil : reasoning,
                stopStringFiltered: stopMatcher.isStopped)
        }
        completed = true
        endRequest()
        let payload = decoder?.payloadTokenCounts
        return EngineCompletion(
            content: content,
            reasoningContent: reasoning.isEmpty ? nil : reasoning,
            toolCalls: calls,
            finishReason: reason,
            usage: EngineUsage(promptTokens: result.prefillTokens,
                               completionTokens: result.newTokens,
                               cachedPromptTokens: result.cachedPromptTokens,
                               reasoningTokens: payload?.reasoning,
                               visibleTokens: payload.flatMap { stopMatcher.isStopped ? nil : $0.visible }),
            timings: EngineTimings(prefillSeconds: result.prefillSeconds,
                                   decodeSeconds: result.decodeSeconds),
            diagnostics: RuntimeDiagnostics.enabled ? RuntimeDiagnostics(
                result: result,
                memory: .capture(model: model, producer: runner, scratch: scratch),
                prefixCache: prefixStates.map {
                    PrefixCacheDiagnostics(cache: $0, memory: memoryGovernor?.report)
                }) : nil)
    }
}
