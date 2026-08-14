import Foundation

public enum StructuredAssistantEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

public final class StructuredAssistantDecoder: @unchecked Sendable {
    private enum Channel {
        case thought
        case visible
        case label
    }

    private let tokenizer: MFTokenizer
    private let allowedTools: Set<String>
    private let idGenerator: @Sendable () -> String
    private var channel: Channel = .visible
    private var label = ""
    private var toolTokens: [Int32]?
    /// DeepSeek text-stream state. DSML markers are plain text, not special
    /// tokens, so the fork scans deltas: `heldText` is a tail withheld while
    /// it could still open a marker, `dsmlText` buffers an open block.
    private var heldText = ""
    private var dsmlText: String?
    private var emittedCalls = 0
    private var failed = false

    public init(tokenizer: MFTokenizer,
                allowedTools: Set<String>,
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.tokenizer = tokenizer
        self.allowedTools = allowedTools
        self.idGenerator = idGenerator
    }

    public convenience init(tokenizer: MFTokenizer,
                            allowedTools: Set<String>,
                            startsInThought: Bool,
                            idGenerator: @escaping @Sendable () -> String = {
                                "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                            }) {
        self.init(tokenizer: tokenizer, allowedTools: allowedTools, idGenerator: idGenerator)
        channel = startsInThought ? .thought : .visible
    }

    /// Route detokenizer flush text (no backing token, e.g. the end-of-stream
    /// `.tail`) through the same dialect scanning as token deltas, so a flush
    /// cannot bypass DSML marker withholding or land out of order relative
    /// to a held tail. `-1` never matches a special-token ID.
    public func consumeFlushedText(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !text.isEmpty else { return [] }
        return try consume(tokenID: -1, delta: text)
    }

    public func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        if tokenizer.dialect == .chatml {
            return try consumeChatML(tokenID: tokenID, delta: delta)
        }
        if tokenizer.dialect == .deepseek {
            return try consumeDeepseek(tokenID: tokenID, delta: delta)
        }
        if tokenID == tokenizer.channelStartID {
            label = ""
            channel = .label
            return []
        }
        if tokenID == tokenizer.channelEndID {
            channel = .visible
            return []
        }
        if tokenID == tokenizer.toolCallStartID {
            guard toolTokens == nil else {
                failed = true
                throw ToolCallParserError.malformed
            }
            toolTokens = []
            return []
        }
        if tokenID == tokenizer.toolCallEndID {
            guard let tokens = toolTokens else {
                failed = true
                throw ToolCallParserError.malformed
            }
            toolTokens = nil
            let text = tokenizer.decode(tokens, skipSpecialTokens: false)
            do {
                let call = try GemmaToolCallParser().parse(
                    text, allowedTools: allowedTools, id: idGenerator())
                emittedCalls += 1
                return [.toolCall(call)]
            } catch {
                failed = true
                throw error
            }
        }
        if tokenID == tokenizer.toolResponseID || tokenID == tokenizer.toolResponseEndID {
            guard emittedCalls > 0, toolTokens == nil else {
                failed = true
                throw ToolCallParserError.malformed
            }
            return []
        }
        if var tokens = toolTokens {
            tokens.append(tokenID)
            guard tokens.count * MemoryLayout<Int32>.size <= GemmaToolCallParser.maximumBytes else {
                failed = true
                throw ToolCallParserError.oversized
            }
            toolTokens = tokens
            return []
        }
        switch channel {
        case .thought:
            return []
        case .visible:
            return delta.isEmpty ? [] : [.content(delta)]
        case .label:
            label += delta
            guard let newline = label.firstIndex(of: "\n") else { return [] }
            let name = label[..<newline].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let contentStart = label.index(after: newline)
            let content = String(label[contentStart...])
            channel = name == "final" || name == "answer" ? .visible : .thought
            label = ""
            if channel == .visible, !content.isEmpty {
                return [.content(content)]
            }
            return []
        }
    }

    /// ChatML transitions: `<think>`…`</think>` suppress thought text, and
    /// `<tool_call>`…`</tool_call>` buffer the dialect's tool payload.
    private func consumeChatML(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        if tokenID == tokenizer.toolCallStartID {
            guard toolTokens == nil else {
                failed = true
                throw ToolCallParserError.malformed
            }
            toolTokens = []
            return []
        }
        if tokenID == tokenizer.toolCallEndID {
            guard let tokens = toolTokens else {
                failed = true
                throw ToolCallParserError.malformed
            }
            toolTokens = nil
            let text = tokenizer.decode(tokens, skipSpecialTokens: false)
            do {
                let call = try parseChatMLToolCall(text)
                emittedCalls += 1
                return [.toolCall(call)]
            } catch {
                failed = true
                throw error
            }
        }
        if var tokens = toolTokens {
            tokens.append(tokenID)
            guard tokens.count * MemoryLayout<Int32>.size <= chatMLToolCallMaximumBytes else {
                failed = true
                throw ToolCallParserError.oversized
            }
            toolTokens = tokens
            return []
        }
        if tokenID == tokenizer.thinkStartID {
            channel = .thought
            return []
        }
        if tokenID == tokenizer.thinkEndID {
            channel = .visible
            return []
        }
        guard channel != .thought else { return [] }
        return delta.isEmpty ? [] : [.content(delta)]
    }

    private var chatMLToolCallMaximumBytes: Int {
        tokenizer.generationPromptStartsInThinking
            ? MapleToolCallParser.maximumBytes
            : QwenToolCallParser.maximumBytes
    }

    private func parseChatMLToolCall(_ text: String) throws -> ParsedToolCall {
        if tokenizer.generationPromptStartsInThinking {
            return try MapleToolCallParser().parse(
                text, allowedTools: allowedTools, id: idGenerator())
        }
        return try QwenToolCallParser().parse(
            text, allowedTools: allowedTools, id: idGenerator())
    }

    /// DeepSeek transitions: `<think>`…`</think>` suppress thought text like
    /// ChatML, but tool calls arrive as a plain-text
    /// `<｜DSML｜tool_calls>`…`</｜DSML｜tool_calls>` block with no special
    /// tokens, so this fork scans the delta stream instead of matching IDs.
    /// A tail that could still begin the open marker is withheld — a chunk
    /// ending in a bare `<` must not stream ahead of the marker it may start —
    /// and a completed block goes to the DSML parser, which can yield several
    /// calls at once.
    private func consumeDeepseek(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        if tokenID == tokenizer.thinkStartID || tokenID == tokenizer.thinkEndID {
            channel = tokenID == tokenizer.thinkStartID ? .thought : .visible
            // A special token interrupts the text run, so a withheld tail can
            // no longer complete the open marker; release it.
            guard dsmlText == nil, !heldText.isEmpty else { return [] }
            let visible = heldText
            heldText = ""
            return [.content(visible)]
        }
        guard channel != .thought, !delta.isEmpty else { return [] }
        heldText += delta
        var events: [StructuredAssistantEvent] = []
        scanning: while !heldText.isEmpty {
            if dsmlText != nil {
                dsmlText! += heldText
                heldText = ""
                guard let close = dsmlText!.range(of: Self.dsmlCloseMark) else {
                    guard dsmlText!.utf8.count <= DeepseekToolCallParser.maximumBytes else {
                        failed = true
                        throw ToolCallParserError.oversized
                    }
                    break scanning
                }
                let body = String(dsmlText![..<close.lowerBound])
                heldText = String(dsmlText![close.upperBound...])
                dsmlText = nil
                do {
                    let calls = try DeepseekToolCallParser().parse(
                        body, allowedTools: allowedTools, idGenerator: idGenerator)
                    emittedCalls += calls.count
                    events += calls.map { .toolCall($0) }
                } catch {
                    failed = true
                    throw error
                }
                continue scanning
            }
            if let open = heldText.range(of: Self.dsmlOpenMark) {
                let visible = String(heldText[..<open.lowerBound])
                if !visible.isEmpty { events.append(.content(visible)) }
                heldText = String(heldText[open.upperBound...])
                dsmlText = ""
                continue scanning
            }
            // Stream everything except a tail that is still a prefix of the
            // open marker; it completes or releases on a later chunk.
            let held = Self.openMarkerPrefixLength(of: heldText)
            if held < heldText.count {
                events.append(.content(String(heldText.dropLast(held))))
                heldText = String(heldText.suffix(held))
            }
            break scanning
        }
        return events
    }

    private static let dsmlOpenMark = DeepseekToolCallParser.toolCallsOpenMark
    private static let dsmlCloseMark = DeepseekToolCallParser.toolCallsCloseMark

    /// Length of the longest suffix of `text` that is a proper prefix of the
    /// DSML open marker. The marker is short, so the scan is a few characters.
    private static func openMarkerPrefixLength(of text: String) -> Int {
        let longest = min(text.count, dsmlOpenMark.count - 1)
        guard longest > 0 else { return 0 }
        for length in stride(from: longest, through: 1, by: -1)
        where dsmlOpenMark.hasPrefix(text.suffix(length)) {
            return length
        }
        return 0
    }

    /// Release any tail withheld as a potential DSML-open prefix: a reply
    /// that legitimately ends in `<`, `</`, `<｜`, … would otherwise
    /// silently lose those characters. `finish()` calls this itself, so
    /// callers only need `drain()` directly when they want the released
    /// text before deciding whether to finish.
    public func drain() -> [StructuredAssistantEvent] {
        guard !failed, dsmlText == nil, !heldText.isEmpty else { return [] }
        let visible = heldText
        heldText = ""
        return [.content(visible)]
    }

    /// End of stream: releases any withheld plain-text tail and validates
    /// that no tool-call block was left unclosed. Emitting the returned
    /// events is required for byte-faithful output — dropping them
    /// truncates a reply that ends in a DSML-open prefix.
    public func finish() throws -> [StructuredAssistantEvent] {
        let released = drain()
        guard !failed, toolTokens == nil, dsmlText == nil else {
            throw ToolCallParserError.malformed
        }
        return released
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }
}
