import Foundation

public enum StructuredAssistantEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

public final class StructuredAssistantDecoder: @unchecked Sendable {
    /// Optional separate reasoning channel; never mixed into visible content.
    /// Source-template Qwen callers use this to round-trip the template's history.
    public var onReasoning: ((String) -> Void)?
    /// Generated payload tokens, not a re-tokenization of rendered text. Markers,
    /// tool syntax and EOS are excluded; buffered byte tokens still count.
    public struct PayloadTokenCounts: Equatable, Sendable {
        public fileprivate(set) var reasoning = 0
        public fileprivate(set) var visible = 0
    }
    private var payloadCounts = PayloadTokenCounts()
    public var payloadTokenCounts: PayloadTokenCounts? { payloadCounts }
    private enum Channel {
        case thought
        case visible
    }

    private let tokenizer: MFTokenizer
    private let allowedTools: Set<String>
    private let toolSchemas: [String: JSONValue]
    private let idGenerator: @Sendable () -> String
    private var channel: Channel = .visible
    private var toolTokens: [Int32]?
    /// The source template frames a turn as `</think>\n\n{content}`. When
    /// reasoning is reported separately those newlines are framing, not
    /// content: echoed back they would double the separator on re-render.
    private var afterThoughtFraming = false
    private var emittedCalls = 0
    private var failed = false

    /// Inkling 2: where the generation is. Between blocks, in a
    /// `<|message_model|>` header (a tool name may follow), or inside a
    /// thinking, visible or tool-call block; every block ends at
    /// `<|end_message|>`.
    private enum Inkling2Block { case between, header, thinking, visible, toolCall }
    private var inkling2Block: Inkling2Block = .between
    private var inkling2Header = ""
    private var inkling2CallHeader = ""

    public convenience init(tokenizer: MFTokenizer,
                            allowedTools: Set<String>,
                            idGenerator: @escaping @Sendable () -> String = {
                                "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                            }) {
        self.init(tokenizer: tokenizer, allowedTools: allowedTools,
                  toolDefinitions: [], idGenerator: idGenerator)
    }

    public init(tokenizer: MFTokenizer,
                allowedTools: Set<String>,
                toolDefinitions: [MFTokenizer.FunctionDefinition],
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.tokenizer = tokenizer
        self.allowedTools = allowedTools
        self.toolSchemas = toolDefinitions.reduce(into: [:]) { $0[$1.name] = $1.parameters }
        self.idGenerator = idGenerator
    }

    public convenience init(tokenizer: MFTokenizer,
                            allowedTools: Set<String>,
                            startsInThought: Bool,
                            idGenerator: @escaping @Sendable () -> String = {
                                "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                            }) {
        self.init(tokenizer: tokenizer, allowedTools: allowedTools,
                  startsInThought: startsInThought, toolDefinitions: [], idGenerator: idGenerator)
    }

    public convenience init(tokenizer: MFTokenizer,
                            allowedTools: Set<String>,
                            startsInThought: Bool,
                            toolDefinitions: [MFTokenizer.FunctionDefinition],
                            idGenerator: @escaping @Sendable () -> String = {
                                "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                            }) {
        self.init(tokenizer: tokenizer, allowedTools: allowedTools,
                  toolDefinitions: toolDefinitions, idGenerator: idGenerator)
        channel = startsInThought ? .thought : .visible
    }

    /// Route detokenizer flush text (no backing token, e.g. the end-of-stream
    /// `.tail`) through the same dialect handling as token deltas. `-1` never
    /// matches a special-token ID.
    public func consumeFlushedText(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !text.isEmpty else { return [] }
        return try consume(tokenID: -1, delta: text)
    }

    public func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        if tokenizer.dialect == .inkling2 {
            return try consumeInkling2(tokenID: tokenID, delta: delta)
        }
        if payloadTokenCounts != nil, tokenID >= 0,
           !tokenizer.stopTokenIDs.contains(tokenID),
           tokenID != tokenizer.thinkStartID, tokenID != tokenizer.thinkEndID,
           tokenID != tokenizer.toolCallStartID, tokenID != tokenizer.toolCallEndID,
           tokenID != tokenizer.toolResponseID, tokenID != tokenizer.toolResponseEndID,
           toolTokens == nil {
            if channel == .thought { payloadCounts.reasoning += 1 }
            else if channel == .visible { payloadCounts.visible += 1 }
        }
        switch tokenizer.dialect {
        case .chatml: return try consumeChatML(tokenID: tokenID, delta: delta)
        case .minicpm: return try consumeMiniCPM(tokenID: tokenID, delta: delta)
        case .inkling2: return try consumeInkling2(tokenID: tokenID, delta: delta)
        }
    }

    /// ChatML transitions: `<think>`…`</think>` suppress thought text, and
    /// `<tool_call>`…`</tool_call>` buffer the dialect's tool payload.
    private func consumeChatML(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        // Reasoning can discuss tool syntax; only the visible channel may
        // issue a call. Do not parse or reject examples inside <think>.
        // Once a real call is open, its payload owns all tokens (including
        // literal thinking markers in string arguments) until </tool_call>.
        if toolTokens == nil {
            if tokenID == tokenizer.thinkStartID {
                channel = .thought
                return []
            }
            if tokenID == tokenizer.thinkEndID {
                channel = .visible
                afterThoughtFraming = onReasoning != nil
                return []
            }
            if channel == .thought {
                if !delta.isEmpty { onReasoning?(delta) }
                return []
            }
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
        guard afterThoughtFraming else {
            return delta.isEmpty ? [] : [.content(delta)]
        }
        let visible = String(delta.drop(while: { $0 == "\n" }))
        guard !visible.isEmpty else { return [] }
        afterThoughtFraming = false
        return [.content(visible)]
    }

    private var chatMLToolCallMaximumBytes: Int {
        usesJSONToolCalls
            ? MapleToolCallParser.maximumBytes
            : QwenToolCallParser.maximumBytes
    }

    // Thinking mode does not determine tool syntax: base Qwen 3.8 opens
    // thought blocks but its pinned template uses XML payloads.
    private var usesJSONToolCalls: Bool {
        tokenizer.usesJSONChatMLToolCalls
    }

    private func parseChatMLToolCall(_ text: String) throws -> ParsedToolCall {
        if usesJSONToolCalls {
            return try MapleToolCallParser().parse(
                text, allowedTools: allowedTools, id: idGenerator())
        }
        return try QwenToolCallParser().parse(
            text, allowedTools: allowedTools, id: idGenerator(), toolSchemas: toolSchemas)
    }

    /// MiniCPM transitions: `<think>`…`</think>` (ids 8 / 9, added tokens
    /// that are not flagged special but still arrive as single ids) suppress
    /// thought text, and `<function`…`</function>` (special tokens) buffer
    /// the XML tool-call body for `MiniCPMToolCallParser`. The `<param`
    /// special token inside the body decodes to its text and is parsed as
    /// part of the XML.
    private func consumeMiniCPM(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
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
                let call = try MiniCPMToolCallParser().parse(
                    text, allowedTools: allowedTools, id: idGenerator())
                emittedCalls += 1
                return [.toolCall(call)]
            } catch {
                failed = true
                throw error
            }
        }
        if var tokens = toolTokens {
            tokens.append(tokenID)
            guard tokens.count * MemoryLayout<Int32>.size <= MiniCPMToolCallParser.maximumBytes else {
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

    /// Inkling 2 transitions. A turn is a list of blocks,
    ///   `<|message_model|><|content_thinking|>…<|end_message|>`
    ///   `<|message_model|><|content_text|>…<|end_message|>`
    ///   `<|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":…}<|end_message|>`
    /// then `<|content_model_end_sampling|>`. Thinking goes to `onReasoning`;
    /// text, tool-error and text-invoke blocks are visible (the vendor's
    /// response template reads all three as content); a JSON block is a tool
    /// call for `Inkling2ToolCallParser`. Header text is a tool name, never
    /// content. Text outside any block is shown rather than dropped.
    private func consumeInkling2(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard let t = tokenizer.inkling2Tokens else {
            failed = true
            throw ToolCallParserError.malformed
        }
        if inkling2Block == .toolCall {
            if tokenID == t.endMessage {
                let tokens = toolTokens ?? []
                toolTokens = nil
                inkling2Block = .between
                inkling2Header = ""
                let text = tokenizer.decode(tokens, skipSpecialTokens: false)
                do {
                    let call = try Inkling2ToolCallParser().parse(
                        text, headerName: inkling2CallHeader,
                        allowedTools: allowedTools, id: idGenerator())
                    emittedCalls += 1
                    return [.toolCall(call)]
                } catch {
                    failed = true
                    throw error
                }
            }
            var tokens = toolTokens ?? []
            if tokenID >= 0 {
                tokens.append(tokenID)
            } else if !delta.isEmpty {
                tokens += tokenizer.encode(delta, addBOS: false)
            }
            guard tokens.count * MemoryLayout<Int32>.size <= Inkling2ToolCallParser.maximumBytes else {
                failed = true
                throw ToolCallParserError.oversized
            }
            toolTokens = tokens
            return []
        }
        switch tokenID {
        case t.messageModel:
            inkling2Block = .header
            inkling2Header = ""
            return []
        case t.contentThinking:
            inkling2Block = .thinking
            return []
        case t.contentText, t.contentToolError, t.contentInvokeToolText:
            inkling2Block = .visible
            return []
        case t.contentInvokeToolJSON:
            inkling2Block = .toolCall
            inkling2CallHeader = inkling2Header
            toolTokens = []
            return []
        case t.endMessage:
            // A header names only the block it opens.
            inkling2Block = .between
            inkling2Header = ""
            return []
        case _ where tokenizer.stopTokenIDs.contains(tokenID):
            return []
        default:
            break
        }
        switch inkling2Block {
        case .header:
            inkling2Header += delta
            return []
        case .thinking:
            if tokenID >= 0 { payloadCounts.reasoning += 1 }
            if !delta.isEmpty { onReasoning?(delta) }
            return []
        case .visible, .between:
            if tokenID >= 0 { payloadCounts.visible += 1 }
            return delta.isEmpty ? [] : [.content(delta)]
        case .toolCall:
            return []
        }
    }

    /// Release any withheld text. Every supported dialect frames tool calls
    /// with special tokens, so nothing is ever withheld; kept so callers can
    /// drain before deciding whether to finish.
    public func drain() -> [StructuredAssistantEvent] { [] }

    /// End of stream: validates that no tool-call block was left unclosed.
    public func finish() throws -> [StructuredAssistantEvent] {
        guard !failed, toolTokens == nil else {
            throw ToolCallParserError.malformed
        }
        return []
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }
}
