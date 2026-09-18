import Foundation
import Mference

public enum ServerPromptCacheMode: String, Sendable, Equatable {
    case off
    case singlePrefix = "single-prefix"
}

struct ServerPromptCacheDomain: Sendable, Equatable {
    let modelID: String
    let sourceSnapshotHash: String?
    let runtimeProfileHash: String
    let maximumContext: Int
    let kvStorage: String
    let fp16RingEnabled: Bool
    let templateSHA256: String
}

struct CachedAssistantTurn: Sendable, Equatable {
    let message: MFTokenizer.Message
    let rawStopReason: StopReason
}

struct ServerPromptCacheEntry: Sendable, Equatable {
    let domain: ServerPromptCacheDomain
    let inputMessages: [MFTokenizer.Message]
    let tools: [MFTokenizer.FunctionDefinition]
    let assistantTurn: CachedAssistantTurn
    let kvBackedTokenIDs: [Int32]
    let uncommittedBoundaryTokenIDs: [Int32]
    let kvPosition: Int
    var reasoningEffort: QwenReasoningEffort? = nil
}

enum ServerPromptCacheMatch: Sendable, Equatable {
    case miss
    case hit(effectivePromptIDs: [Int32], cachedPromptTokens: Int)
}

struct ServerPromptCache: Sendable {
    private(set) var entry: ServerPromptCacheEntry?

    mutating func invalidate() {
        entry = nil
    }

    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
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
        entry = ServerPromptCacheEntry(
            domain: domain,
            inputMessages: request.messages,
            tools: request.tools,
            assistantTurn: CachedAssistantTurn(
                message: assistant,
                rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition,
            reasoningEffort: request.reasoningEffort)
    }

    func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: MFTokenizer
    ) -> ServerPromptCacheMatch {
        guard let entry,
              entry.domain == domain,
              entry.tools == request.tools,
              entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1 else {
            return .miss
        }

        if renderedPromptIDs.count > entry.kvPosition,
           renderedPromptIDs.prefix(entry.kvPosition)
            .elementsEqual(entry.kvBackedTokenIDs) {
            return .hit(
                effectivePromptIDs: renderedPromptIDs,
                cachedPromptTokens: entry.kvPosition)
        }

        // Qwen 3.8 source-template requests require an exact rendered prefix;
        // the legacy bridge cannot prove equivalence for those checkpoints.
        // Qwen 3.6 has its own verified source-template continuation below.
        guard !tokenizer.isSwiftQwen else { return .miss }
        if tokenizer.supportsQwenReasoningEffort,
           request.reasoningEffort != nil || entry.reasoningEffort != nil {
            return .miss
        }
        let inputCount = entry.inputMessages.count
        guard request.messages.count > inputCount + 1,
              request.messages.prefix(inputCount)
                .elementsEqual(entry.inputMessages),
              assistantMatches(
                request.messages[inputCount],
                entry.assistantTurn.message) else {
            return .miss
        }
        if tokenizer.usesSourceTemplate(reasoningEffort: request.reasoningEffort) {
            return matchSourceTemplateContinuation(
                entry: entry,
                request: request,
                cachedTurnIndex: inputCount,
                tokenizer: tokenizer)
        }
        let continuation = Array(request.messages.dropFirst(inputCount + 1))

        if entry.assistantTurn.message.toolCalls.isEmpty {
            return matchTextContinuation(
                entry: entry,
                continuation: continuation,
                tokenizer: tokenizer)
        }
        return matchToolContinuation(
            entry: entry,
            request: request,
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
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        cachedTurnIndex: Int,
        tokenizer: MFTokenizer
    ) -> ServerPromptCacheMatch {
        guard entry.assistantTurn.rawStopReason == .endOfTurn,
              let bridge = try? tokenizer.encodeSourceTemplateContinuation(
                messages: request.messages,
                cachedTurnIndex: cachedTurnIndex,
                tools: request.tools,
                reasoningEffort: request.reasoningEffort),
              bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }

    private func matchTextContinuation(
        entry: ServerPromptCacheEntry,
        continuation: [MFTokenizer.Message],
        tokenizer: MFTokenizer
    ) -> ServerPromptCacheMatch {
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
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        continuation: [MFTokenizer.Message],
        tokenizer: MFTokenizer
    ) -> ServerPromptCacheMatch {
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
            incomingMessages: request.messages,
            tools: request.tools),
              bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }
}
