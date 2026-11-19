import Foundation
import LlamaCpp
import Mference
import Synchronization

/// A GGUF model loaded into llama.cpp: one model, one context, one sequence.
///
/// `prepare` renders the GGUF's own chat template and tokenizes with the
/// model's vocabulary; `generate` prefills (reusing whatever prefix of the
/// prompt the context already holds), samples one token at a time, and turns
/// the tokens into the same events, finish reasons and usage as the native
/// path (`StructuredAssistantDecoder`, `StreamingStopMatcher`).
actor LlamaModel: EngineModel {
    nonisolated let descriptor: EngineModelDescriptor
    nonisolated let traits: EngineModelTraits

    private let handle: LlamaModelHandle
    private let contextOwner: LlamaContextHandle
    /// MTP speculative decoding (Deep), when the GGUF has an MTP head and
    /// draft decoding is on; nil runs plain decoding.
    private let drafter: LlamaDrafter?
    private var context: OpaquePointer?
    private let vocabulary: LlamaVocabulary
    private let tokenizer: MFTokenizer
    private let template: LlamaChatTemplate
    private let batchSize: Int
    private let abortFlag = LlamaAbortFlag()

    /// What the context's memory holds, for prompt-prefix reuse.
    private var promptCache = LlamaPromptCache()
    /// Every token the last generation sampled, the end-of-generation token
    /// included. Kept for the token-identity checks against upstream tools.
    private(set) var lastSampledTokens: [Int32] = []

    /// The prompt text `prepare` tokenizes: the GGUF's template rendered for
    /// `request`.
    func renderedPrompt(_ request: EngineRequest) throws -> String {
        let rendered = try template.render(
            messages: request.messages,
            tools: request.tools,
            keywordArguments: LlamaChatTemplate.keywordArguments(
                reasoningEffort: request.reasoningEffort,
                preserveThinking: request.preserveThinking,
                readsEnableThinking: template.readsEnableThinking,
                readsReasoningEffort: template.readsReasoningEffort))
        return template.strippingImplicitSpecialTokens(rendered, addsBOS: vocabulary.addsBOS,
                                                       addsEOS: vocabulary.addsEOS)
    }

    init(descriptor: EngineModelDescriptor,
         handle: LlamaModelHandle,
         context: OpaquePointer,
         vocabulary: LlamaVocabulary,
         tokenizer: MFTokenizer,
         template: LlamaChatTemplate,
         batchSize: Int,
         mtp: OpaquePointer? = nil,
         draftLimit: Int = LlamaBackend.draftLimit) {
        self.descriptor = descriptor
        self.handle = handle
        let contextOwner = LlamaContextHandle(context, model: handle)
        self.contextOwner = contextOwner
        let drafter = mtp.map { LlamaDrafter($0, draftLimit: draftLimit, context: contextOwner) }
        self.drafter = drafter
        self.context = context
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.template = template
        self.batchSize = batchSize
        traits = EngineModelTraits(
            chatDialect: tokenizer.dialect,
            generationDefaults: .defaults,
            maxContext: Int(llama_n_ctx(context)),
            acceptsReasoningEffort: template.readsEnableThinking || template.readsReasoningEffort,
            // Qwen 3.8's source template, which reads the effort level and
            // rejects developer messages, as the server's Qwen 3.8 rules do.
            supportsQwenReasoningEffort: template.readsReasoningEffort,
            reusesPromptPrefixes: drafter == nil,
            usesDraftDecoding: drafter != nil)
        llama_set_abort_callback(context, { data in
            guard let data else { return false }
            return Unmanaged<LlamaAbortFlag>.fromOpaque(data).takeUnretainedValue().isSet
        }, Unmanaged.passUnretained(abortFlag).toOpaque())
        if let draftContext = drafter?.pointer.flatMap(mference_mtp_draft_context) {
            llama_set_abort_callback(draftContext, { data in
                guard let data else { return false }
                return Unmanaged<LlamaAbortFlag>.fromOpaque(data).takeUnretainedValue().isSet
            }, Unmanaged.passUnretained(abortFlag).toOpaque())
        }
    }

    // MARK: EngineModel

    func prepare(_ request: EngineRequest) async throws -> EnginePreparedPrompt {
        guard context != nil else { throw EngineError.modelUnloaded(descriptor.id) }
        if request.reasoningEffort != nil, !traits.acceptsReasoningEffort {
            throw EngineError.invalidRequest(message: "reasoning_effort is not supported by this model",
                                             param: "reasoning_effort", code: "unsupported_value")
        }
        let tokens = vocabulary.tokenIDs(try renderedPrompt(request), addSpecial: true)
        // (`renderedPrompt` already dropped any BOS/EOS tokenization adds.)
        guard !tokens.isEmpty else {
            throw EngineError.invalidRequest(message: "the rendered prompt is empty",
                                             param: "messages", code: "invalid_chat_template")
        }
        guard tokens.count < traits.maxContext else {
            throw EngineError.invalidRequest(message: "effective prompt exceeds the configured context",
                                             param: "messages", code: "context_length_exceeded")
        }
        return EnginePreparedPrompt(request: request, promptTokens: tokens,
                                    needsToolTemplate: request.needsToolTemplate)
    }

    func generate(_ prepared: EnginePreparedPrompt,
                  onEvent: @escaping @Sendable (EngineEvent) -> Void) async throws -> EngineCompletion {
        guard let context else { throw EngineError.modelUnloaded(descriptor.id) }
        abortFlag.clear()
        return try await withTaskCancellationHandler {
            try run(prepared, context: context, onEvent: onEvent)
        } onCancel: {
            abortFlag.set()
        }
    }

    /// The text of `tokens` with special tokens spelled out, as llama.cpp's
    /// tools print a generation, without a final end-of-generation token.
    func printedText(_ tokens: [Int32]) -> String {
        var tokens = tokens
        if let last = tokens.last, vocabulary.isEndOfGeneration(last) { tokens.removeLast() }
        return String(decoding: tokens.flatMap { vocabulary.pieceBytes($0, special: true) }, as: UTF8.self)
    }

    /// Forgets the prompt prefix the memory holds, so the next request
    /// prefills from scratch (used by the token-identity checks, which compare
    /// cold runs with the upstream tools).
    func discardCachedPrompt() {
        promptCache.invalidate()
    }

    func unload() async {
        abortFlag.set()
        context = nil
        drafter?.free()
        contextOwner.free()
        promptCache.invalidate()
        handle.free()
    }

    // MARK: Generation

    private func run(_ prepared: EnginePreparedPrompt,
                     context: OpaquePointer,
                     onEvent: @escaping @Sendable (EngineEvent) -> Void) throws -> EngineCompletion {
        let request = prepared.request
        let prompt = prepared.promptTokens
        let maxContext = traits.maxContext
        guard !prompt.isEmpty, prompt.count < maxContext else {
            throw EngineError.invalidRequest(message: "effective prompt exceeds the configured context",
                                             param: "messages", code: "context_length_exceeded")
        }
        try request.generationConfig.validate()
        let config = request.generationConfig
        let maxNewTokens = min(request.maximumCompletionTokens, maxContext - prompt.count)

        // Tokens become events the same way on both paths.
        let decoder = StructuredAssistantDecoder(
            tokenizer: tokenizer,
            allowedTools: Set(request.tools.map(\.name)),
            startsInThought: tokenizer.startsInThinking(reasoningEffort: request.reasoningEffort,
                                                        promptIDs: prompt),
            toolDefinitions: request.tools)
        var assembler = LlamaEventAssembler(stops: config.stopStrings, onEvent: onEvent)
        // Reasoning bypasses the stop matcher, as on the native path.
        var reasoning = ""
        decoder.onReasoning = { [onEvent] text in
            reasoning += text
            onEvent(.reasoning(text))
        }
        var utf8 = LlamaUTF8Stream()
        lastSampledTokens = []
        var generated = 0
        var hitLimit = false

        /// Records one sampled token; false once generation must end (end of
        /// generation, a stop string, or the token budget).
        func take(_ token: Int32) throws -> Bool {
            lastSampledTokens.append(token)
            generated += 1
            if vocabulary.isEndOfGeneration(token)
                || tokenizer.stopTokenIDs.contains(token)
                || config.extraStopTokens.contains(token) {
                return false
            }
            let delta = utf8.push(vocabulary.pieceBytes(token, special: false))
            try assembler.take(decoder.consume(tokenID: token, delta: delta))
            if assembler.isStopped { return false }
            guard generated < maxNewTokens else {
                hitLimit = true
                return false
            }
            return true
        }

        let reused: Int
        let clock = ContinuousClock()
        let prefillSeconds: Double
        let decodeStart: ContinuousClock.Instant
        var draftProposed: Int?
        var draftAccepted: Int?

        if let drafter, let mtp = drafter.pointer {
            // MTP: llama.cpp's speculative loop owns both memories and starts
            // from empty ones (no prefix reuse).
            reused = 0
            // From here until a clean return, the memory may hold anything.
            promptCache.invalidate()
            let prefillStart = clock.now
            let status = prompt.withUnsafeBufferPointer { buffer in
                mference_mtp_begin(mtp, buffer.baseAddress, Int32(buffer.count),
                                   LlamaSampler.mtpSampling(config, contextSize: Int32(maxContext)))
            }
            if status == 2 || abortFlag.isSet { throw CancellationError() }
            guard status == 0 else { throw LlamaBackendError.decodeFailed(status) }
            llama_synchronize(context)
            prefillSeconds = (clock.now - prefillStart).seconds
            decodeStart = clock.now
            var proposed = 0
            var accepted = 0
            var buffer = [Int32](repeating: 0, count: drafter.stepCapacity)
            stepping: while true {
                try Task.checkCancellation()
                guard generated < maxNewTokens else {
                    hitLimit = true
                    break
                }
                var drafted: Int32 = 0
                let count = mference_mtp_step(mtp, Int32(maxNewTokens - generated),
                                              &buffer, Int32(buffer.count), &drafted)
                if count == -2 || abortFlag.isSet { throw CancellationError() }
                guard count > 0 else { throw LlamaBackendError.decodeFailed(count) }
                proposed += Int(drafted)
                // Of the step's tokens, all but the target's last one are
                // accepted drafts; count only those that reached the output.
                var taken = 0
                defer { accepted += min(taken, Int(count) - 1) }
                for token in buffer.prefix(Int(count)) {
                    taken += 1
                    guard try take(token) else { break stepping }
                }
            }
            draftProposed = proposed
            draftAccepted = accepted
        } else {
            // Decides what to reuse, trims or clears the memory to match, and
            // leaves the cache untrusted until a clean return.
            let memory = llama_get_memory(context)
            reused = promptCache.begin(
                prompt: prompt,
                trim: { llama_memory_seq_rm(memory, 0, llama_pos($0), -1) },
                clear: { llama_memory_clear(memory, true) })
            let prefillStart = clock.now
            var position = reused
            while position < prompt.count {
                try Task.checkCancellation()
                let end = min(position + batchSize, prompt.count)
                try decode(Array(prompt[position..<end]), context: context)
                position = end
            }
            // llama_decode only queues GPU work; wait for it so the prefill time
            // is the prefill's, not the first sampled token's.
            llama_synchronize(context)
            prefillSeconds = (clock.now - prefillStart).seconds

            let sampler = LlamaSampler(config: config, vocabularySize: Int32(vocabulary.tokenCount),
                                       contextSize: Int32(maxContext),
                                       suppressedTokens: vocabulary.suppressedTokens)
            for token in prompt { sampler.accept(token) }
            decodeStart = clock.now
            while true {
                try Task.checkCancellation()
                guard generated < maxNewTokens else {
                    hitLimit = true
                    break
                }
                guard try take(sampler.sample(context: context)) else { break }
                try decode([lastSampledTokens[lastSampledTokens.count - 1]], context: context)
            }
        }
        try assembler.take(decoder.consumeFlushedText(utf8.finish()))
        try assembler.take(decoder.finish())
        assembler.finish()
        let decodeSeconds = (clock.now - decodeStart).seconds
        try Task.checkCancellation()

        let finishReason: EngineFinishReason =
            !assembler.toolCalls.isEmpty ? .toolCalls : (hitLimit && !assembler.isStopped ? .length : .stop)
        let counts = decoder.payloadTokenCounts
        // The plain path's memory now holds exactly what the cache recorded;
        // the MTP loop's is not tracked, so the next request starts from empty.
        if drafter == nil { promptCache.commit() }
        return EngineCompletion(
            content: assembler.content,
            reasoningContent: reasoning.isEmpty ? nil : reasoning,
            toolCalls: assembler.toolCalls,
            finishReason: finishReason,
            usage: EngineUsage(promptTokens: prompt.count,
                               completionTokens: generated,
                               cachedPromptTokens: reused,
                               reasoningTokens: counts?.reasoning,
                               visibleTokens: assembler.isStopped ? nil : counts?.visible),
            timings: EngineTimings(prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds,
                                   draftTokensProposed: draftProposed,
                                   draftTokensAccepted: draftAccepted))
    }

    private func decode(_ tokens: [Int32], context: OpaquePointer) throws {
        var tokens = tokens
        let status = tokens.withUnsafeMutableBufferPointer { buffer in
            llama_decode(context, llama_batch_get_one(buffer.baseAddress, Int32(buffer.count)))
        }
        if status == 2 || abortFlag.isSet { throw CancellationError() }
        guard status == 0 else { throw LlamaBackendError.decodeFailed(status) }
        promptCache.append(tokens)
    }
}

/// The tokens a llama.cpp context's memory holds, for prompt-prefix reuse.
///
/// A request calls `begin` once: it decides how much of the new prompt the
/// memory already holds (while the record is still trusted), trims or clears
/// the memory to match, and only then marks the record untrusted. Decoded
/// tokens are `append`ed, and `commit` trusts the record again after a clean
/// finish; a request that fails or is cancelled never commits, so the next one
/// starts from an empty memory. At least one prompt token is always left to
/// decode, for its logits.
struct LlamaPromptCache {
    private(set) var tokens: [Int32] = []
    private(set) var isTrusted = true

    /// Returns how many leading prompt tokens are reused. `trim(n)` drops
    /// everything from position `n` on and reports whether the memory could
    /// (a recurrent state cannot roll back); `clear()` empties it.
    mutating func begin(prompt: [Int32], trim: (Int) -> Bool, clear: () -> Void) -> Int {
        defer { isTrusted = false }
        var common = 0
        if isTrusted {
            let limit = min(tokens.count, prompt.count - 1)
            while common < limit, tokens[common] == prompt[common] { common += 1 }
            if common == tokens.count {
                return common
            }
            if common > 0, trim(common) {
                tokens.removeLast(tokens.count - common)
                return common
            }
        }
        clear()
        tokens = []
        return 0
    }

    mutating func append(_ decoded: [Int32]) {
        tokens += decoded
    }

    mutating func commit() {
        isTrusted = true
    }

    /// Forgets the record; the next `begin` clears the memory.
    mutating func invalidate() {
        isTrusted = false
        tokens = []
    }
}

/// Collects decoder events into the completion and forwards them, applying
/// stop strings to visible text the same way the server's native path does.
struct LlamaEventAssembler {
    private var stopMatcher: StreamingStopMatcher
    private let onEvent: @Sendable (EngineEvent) -> Void
    private(set) var content = ""
    private(set) var toolCalls: [ParsedToolCall] = []

    init(stops: [String], onEvent: @escaping @Sendable (EngineEvent) -> Void) {
        stopMatcher = StreamingStopMatcher(stops: stops)
        self.onEvent = onEvent
    }

    var isStopped: Bool { stopMatcher.isStopped }

    mutating func take(_ events: [StructuredAssistantEvent]) {
        for event in events {
            switch event {
            case .content(let text):
                let visible = stopMatcher.push(text)
                if !visible.isEmpty {
                    content += visible
                    onEvent(.content(visible))
                }
            case .toolCall(let call):
                toolCalls.append(call)
                onEvent(.toolCall(call))
            }
        }
    }

    /// Releases text the stop matcher was holding back as a possible prefix.
    mutating func finish() {
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
    }
}

/// Set from a cancellation handler on another thread; read by llama.cpp's
/// abort callback between graph nodes.
final class LlamaAbortFlag: Sendable {
    private let flag = Atomic<Bool>(false)
    var isSet: Bool { flag.load(ordering: .acquiring) }
    func set() { flag.store(true, ordering: .releasing) }
    func clear() { flag.store(false, ordering: .releasing) }
}

fileprivate extension Duration {
    var seconds: Double {
        let (seconds, attoseconds) = components
        return Double(seconds) + Double(attoseconds) / 1e18
    }
}
