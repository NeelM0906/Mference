import Foundation

/// What must be identical for a cached prefix to be reusable at all.
public struct PromptPrefixCacheDomain: Sendable, Equatable {
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let runtimeProfileHash: String
    public let maximumContext: Int
    public let kvStorage: String
    public let fp16RingEnabled: Bool
    public let templateSHA256: String

    public init(modelID: String,
                sourceSnapshotHash: String?,
                runtimeProfileHash: String,
                maximumContext: Int,
                kvStorage: String,
                fp16RingEnabled: Bool,
                templateSHA256: String) {
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.runtimeProfileHash = runtimeProfileHash
        self.maximumContext = maximumContext
        self.kvStorage = kvStorage
        self.fp16RingEnabled = fp16RingEnabled
        self.templateSHA256 = templateSHA256
    }
}

/// The part of a chat turn the cache compares: the conversation and the
/// template controls that change how it renders.
public struct PromptPrefixCacheTurn: Sendable, Equatable {
    public let messages: [MFTokenizer.Message]
    public let tools: [MFTokenizer.FunctionDefinition]
    public let reasoningEffort: QwenReasoningEffort?
    public let preserveThinking: Bool

    public init(messages: [MFTokenizer.Message],
                tools: [MFTokenizer.FunctionDefinition] = [],
                reasoningEffort: QwenReasoningEffort? = nil,
                preserveThinking: Bool = false) {
        self.messages = messages
        self.tools = tools
        self.reasoningEffort = reasoningEffort
        self.preserveThinking = preserveThinking
    }
}

public struct CachedAssistantTurn: Sendable, Equatable {
    public let message: MFTokenizer.Message
    public let rawStopReason: StopReason
}

public struct PromptPrefixCacheEntry: Sendable, Equatable {
    public let domain: PromptPrefixCacheDomain
    public let inputMessages: [MFTokenizer.Message]
    public let tools: [MFTokenizer.FunctionDefinition]
    public let assistantTurn: CachedAssistantTurn
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
    public let kvPosition: Int
    public var reasoningEffort: QwenReasoningEffort? = nil
    public var preserveThinking: Bool = false
}

public enum PromptPrefixCacheMatch: Sendable, Equatable {
    case miss
    case hit(effectivePromptIDs: [Int32], cachedPromptTokens: Int)
}

/// One reusable KV prefix: the last finished turn. `match` decides whether the
/// next turn may continue from it and with which effective prompt; the rules
/// are per chat dialect because templates re-render earlier turns differently
/// from how they were generated.
public struct PromptPrefixCache: Sendable {
    public private(set) var entry: PromptPrefixCacheEntry?

    public init() {}

    public mutating func invalidate() {
        entry = nil
    }

    public mutating func publish(
        domain: PromptPrefixCacheDomain,
        turn: PromptPrefixCacheTurn,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        reasoningContent: String? = nil,
        stopStringFiltered: Bool = false
    ) {
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              !stopStringFiltered,
              result.reason == .endOfTurn
                || result.reason == .toolCalls
                || result.reason == .maxTokens else {
            entry = nil
            return
        }
        let historicalCalls = calls.map {
            MFTokenizer.HistoricalToolCall(
                id: $0.id,
                name: $0.name,
                arguments: $0.arguments)
        }
        let assistant = MFTokenizer.Message(
            role: .assistant,
            content: calls.isEmpty ? content : nil,
            toolCalls: historicalCalls,
            reasoningContent: reasoningContent)
        entry = PromptPrefixCacheEntry(
            domain: domain,
            inputMessages: turn.messages,
            tools: turn.tools,
            assistantTurn: CachedAssistantTurn(
                message: assistant,
                rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition,
            reasoningEffort: turn.reasoningEffort,
            preserveThinking: turn.preserveThinking)
    }

    public func match(
        domain: PromptPrefixCacheDomain,
        turn: PromptPrefixCacheTurn,
        renderedPromptIDs: [Int32],
        tokenizer: MFTokenizer,
        gemmaRecoverablePrefix: ((Int) -> Int)? = nil
    ) -> PromptPrefixCacheMatch {
        guard let entry,
              entry.domain == domain,
              entry.tools == turn.tools,
              entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1 else {
            return .miss
        }

        if tokenizer.dialect == .gemma {
            let wasThinking = (entry.reasoningEffort ?? .off) != .off
            let isThinking = (turn.reasoningEffort ?? .off) != .off
            guard wasThinking == isThinking,
                  entry.preserveThinking == turn.preserveThinking else { return .miss }
        }

        if renderedPromptIDs.count > entry.kvPosition,
           renderedPromptIDs.prefix(entry.kvPosition)
            .elementsEqual(entry.kvBackedTokenIDs) {
            return .hit(
                effectivePromptIDs: renderedPromptIDs,
                cachedPromptTokens: entry.kvPosition)
        }

        if tokenizer.dialect == .gemma {
            let inputCount = entry.inputMessages.count
            guard turn.messages.count > inputCount + 1,
                  turn.messages.prefix(inputCount).elementsEqual(entry.inputMessages),
                  assistantMatches(turn.messages[inputCount], entry.assistantTurn.message) else {
                return .miss
            }
            // Only tool results from this same user turn may append to the
            // generated prefix. A new user must use the canonical re-render,
            // which can remove earlier thoughts.
            let continuation = Array(turn.messages.dropFirst(inputCount + 1))
            if !tokenizer.isGemmaQAT {
                let toolMatch = matchToolContinuation(entry: entry, turn: turn,
                    continuation: continuation, tokenizer: tokenizer)
                if case .hit = toolMatch { return toolMatch }
            }
            // Results may arrive together with a new user message. Such a
            // request starts a new turn too, and must use canonical recovery
            // rather than the append-only tool bridge.
            // QAT's source can normalize even the current tool-call turn
            // (thought closure whitespace and the non-thinking channel).
            // Its resumed input must equal a fresh source render; appending a
            // bridge to raw generated tokens cannot establish that equality.
            guard tokenizer.isGemmaQAT || continuation.contains(where: { $0.role == .user }),
                  let gemmaRecoverablePrefix else { return .miss }
            let common = zip(entry.kvBackedTokenIDs, renderedPromptIDs).prefix { $0 == $1 }.count
            // Leave at least one token for the next-token logits. Only the
            // backend can prove which ring rows or snapshot remain available.
            let limit = min(common, renderedPromptIDs.count - 1)
            guard limit > 0 else { return .miss }
            let cached = gemmaRecoverablePrefix(limit)
            guard cached > 0, cached <= limit else { return .miss }
            return .hit(effectivePromptIDs: renderedPromptIDs, cachedPromptTokens: cached)
        }

        // Qwen 3.8 source-template requests require an exact rendered prefix;
        // the legacy bridge cannot prove equivalence for those checkpoints.
        // Qwen 3.6 has its own verified source-template continuation below.
        guard !tokenizer.isSwiftQwen else { return .miss }
        if tokenizer.supportsQwenReasoningEffort,
           turn.reasoningEffort != nil || entry.reasoningEffort != nil {
            return .miss
        }
        let inputCount = entry.inputMessages.count
        guard turn.messages.count > inputCount + 1,
              turn.messages.prefix(inputCount)
                .elementsEqual(entry.inputMessages),
              assistantMatches(
                turn.messages[inputCount],
                entry.assistantTurn.message) else {
            return .miss
        }
        if tokenizer.usesSourceTemplate(reasoningEffort: turn.reasoningEffort) {
            return matchSourceTemplateContinuation(
                entry: entry,
                turn: turn,
                cachedTurnIndex: inputCount,
                tokenizer: tokenizer)
        }
        let continuation = Array(turn.messages.dropFirst(inputCount + 1))

        if entry.assistantTurn.message.toolCalls.isEmpty {
            return matchTextContinuation(
                entry: entry,
                continuation: continuation,
                tokenizer: tokenizer)
        }
        return matchToolContinuation(
            entry: entry,
            turn: turn,
            continuation: continuation,
            tokenizer: tokenizer)
    }

    private func assistantMatches(
        _ incoming: MFTokenizer.Message,
        _ cached: MFTokenizer.Message
    ) -> Bool {
        guard incoming.role == .assistant,
              cached.role == .assistant,
              incoming.toolCalls == cached.toolCalls,
              incoming.reasoningContent == cached.reasoningContent,
              incoming.toolCallID == cached.toolCallID,
              incoming.name == cached.name else {
            return false
        }
        if !cached.toolCalls.isEmpty {
            return (incoming.content ?? "").isEmpty
                && (cached.content ?? "").isEmpty
        }
        return incoming.content == cached.content
    }

    /// Qwen 3.6 rendered through its source template. The hand-written bridges
    /// below end in the non-thinking generation prompt, and the recurrent
    /// state cannot be rewound to re-render the turn, so the cached turn stays
    /// as generated and only what follows it is prefilled. ChatML ends a
    /// tool-call turn with `<|im_end|>` too, so both shapes stop `.endOfTurn`.
    private func matchSourceTemplateContinuation(
        entry: PromptPrefixCacheEntry,
        turn: PromptPrefixCacheTurn,
        cachedTurnIndex: Int,
        tokenizer: MFTokenizer
    ) -> PromptPrefixCacheMatch {
        guard entry.assistantTurn.rawStopReason == .endOfTurn,
              let bridge = try? tokenizer.encodeSourceTemplateContinuation(
                messages: turn.messages,
                cachedTurnIndex: cachedTurnIndex,
                tools: turn.tools,
                reasoningEffort: turn.reasoningEffort),
              bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }

    private func matchTextContinuation(
        entry: PromptPrefixCacheEntry,
        continuation: [MFTokenizer.Message],
        tokenizer: MFTokenizer
    ) -> PromptPrefixCacheMatch {
        guard continuation.count == 1,
              continuation[0].role == .user,
              let content = continuation[0].content,
              continuation[0].toolCalls.isEmpty,
              continuation[0].toolCallID == nil,
              entry.assistantTurn.rawStopReason == .endOfTurn
                || entry.assistantTurn.rawStopReason == .maxTokens else {
            return .miss
        }
        var bridge = tokenizer.encodeTextContinuation(userContent: content)
        if entry.assistantTurn.rawStopReason == .maxTokens {
            bridge = entry.uncommittedBoundaryTokenIDs + bridge
        } else if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }

    private func matchToolContinuation(
        entry: PromptPrefixCacheEntry,
        turn: PromptPrefixCacheTurn,
        continuation: [MFTokenizer.Message],
        tokenizer: MFTokenizer
    ) -> PromptPrefixCacheMatch {
        let calls = entry.assistantTurn.message.toolCalls
        guard entry.assistantTurn.rawStopReason == .toolCalls,
              continuation.count == calls.count,
              zip(continuation, calls).allSatisfy({ message, call in
                  message.role == .tool
                    && message.toolCallID == call.id
                    && (message.name == nil || message.name == call.name)
                    && message.content != nil
                    && message.toolCalls.isEmpty
              }) else {
            return .miss
        }
        guard let bridge = try? tokenizer.encodeToolResultContinuation(
            cachedMessages: entry.inputMessages,
            assistant: entry.assistantTurn.message,
            incomingMessages: turn.messages,
            tools: turn.tools,
            reasoningEffort: turn.reasoningEffort,
            preserveThinking: turn.preserveThinking),
              bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }
}
