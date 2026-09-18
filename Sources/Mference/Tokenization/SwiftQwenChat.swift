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

    /// Swift always uses its source template. Base Qwen opts into that same
    /// contract only with an explicit effort; omitted effort keeps legacy behavior.
    public func usesSourceQwenTemplate(reasoningEffort: QwenReasoningEffort?) -> Bool {
        isSwiftQwen || (isBaseQwen38 && reasoningEffort != nil)
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
        return usesSourceQwenTemplate(reasoningEffort: reasoningEffort)
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
        guard reasoningEffort == nil else {
            throw MFTokenizerError.unsupportedForDialect(
                "reasoning_effort requires base or Swift Qwen 3.8")
        }
        if !tools.isEmpty || messages.contains(where: {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }) {
            return try encodeToolChat(messages: messages, tools: tools)
        }
        return encode(try applyChatTemplate(messages), addBOS: false)
    }
}
