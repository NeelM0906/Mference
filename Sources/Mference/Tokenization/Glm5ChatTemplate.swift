import Foundation

/// The `glm5` dialect: a hand-port of GLM-5.3-Flash's `chat_template.jinja`
/// (`pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit` at `d43ea8b4`, the file
/// zai-org ships), byte-matched to `transformers` 5.17 `apply_chat_template`
/// on the committed fixture set
/// (`Tests/Mference/Core/Tokenization/Fixtures/Glm5Tokenizer/renders.json`).
///
/// Shape of a render (`trim_blocks` / `lstrip_blocks` are on in HF's Jinja,
/// so no block tag leaves a newline behind):
///
///     [gMASK]<sop><|system|>Reasoning Effort: Max
///     [<|system|>\n# Tools ... </tool_call>]            when tools are given
///     <|system|>SYSTEM  <|user|>USER                     content untrimmed
///     <|assistant|><think>REASONING</think>CONTENT<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>
///     <|observation|><tool_response>RESULT</tool_response>...
///     <|assistant|><think>                               generation prompt
///
/// Facts pinned by the fixtures: the effort line always opens the prompt
/// (`max` when the caller says nothing); an assistant turn without a
/// `</think>` in its content re-renders with an empty `<think></think>`, and
/// with `clear_thinking` every assistant turn at or before the last user turn
/// does; assistant content is `strip()`ped, user and system content is not;
/// no separator is written between content, tool calls or turns; a block of
/// consecutive `tool` messages renders under one `<|observation|>`, ordered
/// by the preceding assistant's call order when every result carries a
/// unique matching `tool_call_id`, in message order otherwise; the assistant
/// turn has no closing token — the model ends it by emitting `<|user|>`,
/// `<|observation|>` or `<|endoftext|>`, all three of which stop generation.
///
/// Two deviations by construction, recorded on docs/families/GLM53_FLASH.md:
/// `JSONValue` objects are unordered, so tool-schema keys and call arguments
/// render in **sorted key order** (Jinja keeps insertion order); and
/// `developer` guidance renders as a `<|system|>` turn (the server already
/// folds it there for every non-Gemma dialect).
extension MFTokenizer {

    /// The template's `reasoning_effort` kwarg; anything but `low` / `high`
    /// renders as `Max`, the template's own fallback.
    public enum Glm5ReasoningEffort: String, Sendable {
        case low = "Low", high = "High", max = "Max"
    }

    /// The effort the chat and tool encoders pin. `Max` is what the template
    /// renders when a caller passes nothing, so it is the vendor's default.
    public static let glm5DefaultReasoningEffort: Glm5ReasoningEffort = .max

    static let glm5Prefix = "[gMASK]<sop>"
    static let glm5SystemMark = "<|system|>"
    static let glm5UserMark = "<|user|>"
    static let glm5AssistantMark = "<|assistant|>"
    static let glm5ObservationMark = "<|observation|>"
    static let glm5ThinkOpen = "<think>"
    static let glm5ThinkClose = "</think>"
    /// The generation prompt: the assistant turn opens inside its think block.
    static let glm5GenerationSuffix = glm5AssistantMark + glm5ThinkOpen

    private static let glm5ToolsHead = "\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        + "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
    private static let glm5ToolsTail = "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n"
        + "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
        + "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>"

    /// Full render, text and tools, with or without the generation prompt.
    func glm5Render(messages: [Message],
                    tools: [FunctionDefinition],
                    reasoningEffort: Glm5ReasoningEffort = MFTokenizer.glm5DefaultReasoningEffort,
                    clearThinking: Bool = false,
                    addGenerationPrompt: Bool = true) throws -> String {
        var s = Self.glm5Prefix + Self.glm5SystemMark + "Reasoning Effort: " + reasoningEffort.rawValue
        if !tools.isEmpty {
            s += Self.glm5SystemMark + Self.glm5ToolsHead
            for tool in tools {
                let object = JSONValue.object([
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "parameters": tool.parameters,
                ])
                s += Self.pythonJSONDumps(object) + "\n"
            }
            s += Self.glm5ToolsTail
        }
        let lastUserIndex = messages.lastIndex { $0.role == .user } ?? -1
        var index = 0
        while index < messages.count {
            let message = messages[index]
            switch message.role {
            case .user:
                s += Self.glm5UserMark + (message.content ?? "")
            case .system, .developer:
                s += Self.glm5SystemMark + (message.content ?? "")
            case .assistant:
                s += try Self.glm5AssistantTurn(
                    message,
                    keepReasoning: !clearThinking || index > lastUserIndex)
            case .tool:
                // One <|observation|> for the whole run of tool messages.
                var end = index
                while end + 1 < messages.count, messages[end + 1].role == .tool { end += 1 }
                let block = Array(messages[index...end])
                let calls = index > 0 && messages[index - 1].role == .assistant
                    ? messages[index - 1].toolCalls : []
                s += Self.glm5ObservationMark
                for result in Self.glm5OrderedResults(block, calls: calls) {
                    s += "<tool_response>" + (result.content ?? "") + "</tool_response>"
                }
                index = end
            }
            index += 1
        }
        if addGenerationPrompt { s += Self.glm5GenerationSuffix }
        return s
    }

    private static func glm5AssistantTurn(_ message: Message, keepReasoning: Bool) throws -> String {
        var content = message.content ?? ""
        var reasoning: String?
        if let close = content.range(of: glm5ThinkClose) {
            // reasoning = content.split('</think>')[0].split('<think>')[-1]
            var head = String(content[..<close.lowerBound])
            if let open = head.range(of: glm5ThinkOpen, options: .backwards) {
                head = String(head[open.upperBound...])
            }
            reasoning = head
            // content = content.split('</think>')[-1]
            if let last = content.range(of: glm5ThinkClose, options: .backwards) {
                content = String(content[last.upperBound...])
            }
        }
        var s = glm5AssistantMark
        if keepReasoning, let reasoning {
            s += glm5ThinkOpen + reasoning + glm5ThinkClose
        } else {
            s += glm5ThinkOpen + glm5ThinkClose
        }
        let stripped = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if !stripped.isEmpty { s += stripped }
        for call in message.toolCalls {
            s += try glm5ToolCall(call)
        }
        return s
    }

    /// `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`;
    /// a string value passes through raw, anything else as `tojson`.
    static func glm5ToolCall(_ call: HistoricalToolCall) throws -> String {
        guard case .object(let arguments) = call.arguments else {
            throw MFTokenizerError.invalidChatTemplate(
                "historical tool arguments must be a JSON object")
        }
        var s = "<tool_call>" + call.name
        for key in arguments.keys.sorted() {
            s += "<arg_key>" + key + "</arg_key><arg_value>"
            if case .string(let raw) = arguments[key]! {
                s += raw
            } else {
                s += pythonJSONDumps(arguments[key]!)
            }
            s += "</arg_value>"
        }
        return s + "</tool_call>"
    }

    /// The template's `can_sort` rule: results follow the preceding
    /// assistant's call order when every result and every call carries a
    /// unique id and each result's id names one of the calls; otherwise
    /// they stay in message order.
    private static func glm5OrderedResults(_ block: [Message],
                                           calls: [HistoricalToolCall]) -> [Message] {
        guard !calls.isEmpty else { return block }
        let callIDs = calls.map(\.id)
        guard Set(callIDs).count == callIDs.count, !callIDs.contains("") else { return block }
        let resultIDs = block.map { $0.toolCallID ?? "" }
        guard !resultIDs.contains(""), Set(resultIDs).count == resultIDs.count,
              resultIDs.allSatisfy({ callIDs.contains($0) }) else { return block }
        return callIDs.flatMap { id in block.filter { $0.toolCallID == id } }
    }
}

/// Parses the GLM-5 tool-call body — the text BETWEEN the `<tool_call>` and
/// `</tool_call>` tokens:
///
///     NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>…
///
/// A VALUE that parses as a JSON number / `true` / `false` / `null` /
/// object / array becomes typed; anything else stays a string, mirroring the
/// template's asymmetric serialization (strings raw, everything else via
/// `tojson`). The `<arg_key>` … `</arg_value>` markers are added tokens that
/// decode back to their text, so the body is parsed as text.
public struct Glm5ToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    private static let keyOpen = "<arg_key>"
    private static let keyClose = "</arg_key>"
    private static let valueOpen = "<arg_value>"
    private static let valueClose = "</arg_value>"

    public init() {}

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
        var body = Substring(text)
        trimOuterWhitespace(&body)

        let nameEnd = body.range(of: Self.keyOpen)?.lowerBound ?? body.endIndex
        let name = body[..<nameEnd].trimmingCharacters(in: .whitespacesAndNewlines)
        body = body[nameEnd...]
        guard name.range(of: #"^[A-Za-z0-9_.\-]{1,64}$"#, options: .regularExpression) != nil else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else { throw ToolCallParserError.unknownTool(name) }

        var arguments: [String: JSONValue] = [:]
        while true {
            trimOuterWhitespace(&body)
            if body.isEmpty { break }
            guard body.hasPrefix(Self.keyOpen) else { throw ToolCallParserError.malformed }
            body.removeFirst(Self.keyOpen.count)
            guard let keyEnd = body.range(of: Self.keyClose) else { throw ToolCallParserError.malformed }
            let key = String(body[..<keyEnd.lowerBound])
            body = body[keyEnd.upperBound...]
            guard !key.isEmpty, arguments[key] == nil else { throw ToolCallParserError.malformed }
            guard body.hasPrefix(Self.valueOpen) else { throw ToolCallParserError.malformed }
            body.removeFirst(Self.valueOpen.count)
            guard let valueEnd = body.range(of: Self.valueClose) else { throw ToolCallParserError.malformed }
            arguments[key] = Self.typedValue(String(body[..<valueEnd.lowerBound]))
            body = body[valueEnd.upperBound...]
        }
        let argumentsValue = JSONValue.object(arguments)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: argumentsValue,
                              argumentsJSON: try argumentsValue.encoded())
    }

    static func typedValue(_ raw: String) -> JSONValue {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let first = trimmed.first,
           first == "{" || first == "[" || first == "-" || first.isNumber
            || trimmed == "true" || trimmed == "false" || trimmed == "null",
           let decoded = try? JSONDecoder().decode(JSONValue.self, from: Data(trimmed.utf8)) {
            return decoded
        }
        return .string(raw)
    }

    private func trimOuterWhitespace(_ body: inout Substring) {
        while let first = body.first, first.isWhitespace { body.removeFirst() }
        while let last = body.last, last.isWhitespace { body.removeLast() }
    }
}
