import Foundation

/// The pinned Qwen 3.8 template's request-level choices. "none" closes the
/// thinking block; medium deliberately adds no effort instruction.
public enum QwenReasoningEffort: String, Codable, Sendable, CaseIterable {
    case xhigh, medium, low
    case off = "none"
}

extension MFTokenizer {
    /// Copy-on-value profile selection; does not mutate the cached tokenizer.
    public func forCheckpoint(_ modelID: String) throws -> Self {
        var value = self
        value.isSwiftQwen = modelID == CheckpointIdentity.swiftQwen38
        value.isBaseQwen38 = modelID == CheckpointIdentity.baseQwen38
        if value.supportsQwenReasoningEffort, value.dialect != .chatml {
            throw MFTokenizerError.unsupportedForDialect("Qwen 3.8 requires ChatML")
        }
        return value
    }

    public var supportsQwenReasoningEffort: Bool { isSwiftQwen || isBaseQwen38 }

    /// Whether a request may carry `reasoning_effort` at all.
    public var acceptsReasoningEffort: Bool {
        supportsQwenReasoningEffort || supportsOptInThinking
    }

    /// Swift always uses its source template. Base Qwen 3.8 opts into that
    /// same contract with an explicit effort; omitted effort keeps legacy behavior.
    public func usesSourceQwenTemplate(reasoningEffort: QwenReasoningEffort?) -> Bool {
        isSwiftQwen || (isBaseQwen38 && reasoningEffort != nil)
    }

    /// Swift-Qwen always renders through its source template. Qwen 3.6 does so
    /// only when the request opts in; an omitted effort keeps its native
    /// non-thinking render. Its template has no effort levels: `none` closes
    /// the thinking block and every other value opens it.
    public func usesSourceTemplate(reasoningEffort: QwenReasoningEffort?) -> Bool {
        usesSourceQwenTemplate(reasoningEffort: reasoningEffort)
            || (supportsOptInThinking && reasoningEffort != nil)
    }

    /// Tokens that follow the assistant turn at `cachedTurnIndex` in the full
    /// render: its closing `<|im_end|>`, the new tool results or user turn, and
    /// the generation prompt. The KV cache already holds that turn as the
    /// model generated it, which a re-render of the parsed turn need not match
    /// token for token, so the cache is extended rather than compared. This is
    /// only sound because `preserve_thinking` makes the render of a turn
    /// independent of what follows it.
    public func encodeSourceTemplateContinuation(messages: [Message],
                                                 cachedTurnIndex: Int,
                                                 tools: [FunctionDefinition],
                                                 reasoningEffort: QwenReasoningEffort?) throws -> [Int32] {
        guard usesSourceTemplate(reasoningEffort: reasoningEffort) else {
            throw MFTokenizerError.unsupportedForDialect("source-template KV continuation")
        }
        guard messages.indices.contains(cachedTurnIndex),
              messages[cachedTurnIndex].role == .assistant,
              cachedTurnIndex < messages.count - 1 else {
            throw MFTokenizerError.invalidChatTemplate(
                "KV continuation needs a cached assistant turn followed by new messages")
        }
        let full = try encodeToolChat(messages: messages, tools: tools,
                                      reasoningEffort: reasoningEffort)
        let head = try encodeToolChat(messages: Array(messages[...cachedTurnIndex]), tools: tools,
                                      reasoningEffort: reasoningEffort,
                                      addGenerationPrompt: false)
        guard let end = head.lastIndex(of: endOfTurnID), end < full.count,
              full[...end].elementsEqual(head[...end]) else {
            throw MFTokenizerError.invalidChatTemplate(
                "cached assistant turn is not a prefix of the full render")
        }
        return Array(full[end...])
    }

    public func startsInThinking(reasoningEffort: QwenReasoningEffort?,
                                 promptIDs: [Int32]? = nil) -> Bool {
        // The actual generation suffix wins over a family's ordinary-chat
        // default. In particular base Qwen opens thinking for ordinary chat,
        // but its tool/history template explicitly closes it. Initializing the
        // decoder as thinking in that case silently discards the visible answer.
        if let promptIDs {
            for id in promptIDs.reversed() {
                if id == thinkStartID { return true }
                if id == thinkEndID { return false }
                // Inspect only trailing whitespace, never markers in history
                // or user text before the assistant-generation boundary.
                if !decode([id], skipSpecialTokens: false)
                    .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { break }
            }
        }
        return usesSourceTemplate(reasoningEffort: reasoningEffort)
            ? reasoningEffort != .off : generationPromptStartsInThinking
    }

    /// Uses the installed source template for both ordinary and tool chat.
    /// No hand-written framing or forced thinking-off path for Swift-Qwen.
    public func encodeChat(messages: [Message], tools: [FunctionDefinition] = [],
                           reasoningEffort: QwenReasoningEffort? = nil) throws -> [Int32] {
        if usesSourceQwenTemplate(reasoningEffort: reasoningEffort) {
            // The source rejects developer turns. Do not silently rewrite their
            // role or discard their instructions.
            guard !messages.contains(where: { $0.role == .developer }) else {
                throw MFTokenizerError.invalidChatTemplate(
                    "Qwen 3.8's source template does not accept developer messages; use a leading system message")
            }
            return try encodeToolChat(messages: messages, tools: tools,
                                      reasoningEffort: reasoningEffort)
        }
        if usesSourceTemplate(reasoningEffort: reasoningEffort) {
            return try encodeToolChat(messages: messages, tools: tools,
                                      reasoningEffort: reasoningEffort)
        }
        guard reasoningEffort == nil else {
            throw MFTokenizerError.unsupportedForDialect(
                "reasoning_effort is not supported by this model")
        }
        if !tools.isEmpty || messages.contains(where: {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }) {
            return try encodeToolChat(messages: messages, tools: tools)
        }
        return encode(try applyChatTemplate(messages), addBOS: false)
    }
}
