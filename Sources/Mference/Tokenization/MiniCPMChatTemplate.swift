import Foundation

/// The `minicpm` dialect: a hand-port of `openbmb/MiniCPM5-2B`'s
/// `chat_template.jinja` (revision `cd199ce3`, SHA-256 `cc945752…`), byte-matched
/// to `transformers` 5.6.2 `apply_chat_template` on the committed fixture set
/// (`Tests/Mference/Core/Tokenization/Fixtures/MiniCPM5Tokenizer/renders.json`).
///
/// ChatML framing with `<s>` first. Three facts of the *rendered* template that
/// are easy to get wrong from reading it, and are pinned by the fixtures:
///
/// 1. **Historical tool calls render once, after the text before the first
///    `<tool_sep>`.** The template's `processed_content` loop assigns inside a
///    `{% for %}`, and Jinja2 `set` does not leak out of a loop, so the
///    interleaved form never survives; only `content.split('<tool_sep>')[0]`
///    does, followed by every call in order. `has_tool_sep` is never defined,
///    so `not has_tool_sep` is always true.
/// 2. **An assistant turn without a think block is re-rendered with an empty
///    one** (`<think>\n\n</think>\n\n`); an inline `<think>…</think>` becomes
///    `<think>\n{reasoning}\n</think>\n\n{content}` with `\n`-stripping on both
///    parts; a `<think>` with no closing tag is rendered verbatim.
/// 3. **Consecutive tool results share one `user` turn**, each wrapped in
///    `\n<tool_response>\n…\n</tool_response>`.
///
/// Two deviations by construction, both recorded on docs/families/MINICPM5.md:
/// `JSONValue` objects are unordered, so tool-schema keys and call parameters
/// render in **sorted key order** (Jinja keeps insertion order); and a
/// non-string, non-scalar parameter value renders as JSON where Jinja would
/// print Python's `repr`.
extension MFTokenizer {

    /// The template's `enable_thinking` kwarg, three-valued exactly as Jinja
    /// sees it. The family default is `.enabled` (the vendor's canonical usage).
    enum MiniCPMThinking {
        case enabled, disabled, unspecified
    }

    static let miniCPMDefaultThinking: MiniCPMThinking = .enabled

    private static let miniCPMBOS = "<s>"
    private static let miniCPMToolDefinitionsHead =
        "# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    private static let miniCPMToolDefinitionsTail = """
        \n</tools>\n\nTool usage guidelines:
        - You may call zero or more functions. If no function calls are needed, just answer normally and do not include any <function ... </function>.
        - When calling a function, return an XML object within <function ... </function> using:
        <function name="function-name"><param name="param-name">param-value</param></function>
        - param-value may be multi-line. If it contains <, & or newline characters, wrap it in a CDATA block: <param name="param-name"><![CDATA[...multi-line value...]]></param>
        """
    private static let miniCPMToolDefSep = "<tool_def_sep>"
    private static let miniCPMToolSep = "<tool_sep>"

    /// Full render, text and tools, with or without the generation prompt.
    func miniCPMRender(messages: [Message],
                       tools: [FunctionDefinition],
                       thinking: MiniCPMThinking,
                       addGenerationPrompt: Bool = true) throws -> String {
        var s = Self.miniCPMBOS
        if !tools.isEmpty {
            let definitions = try Self.miniCPMToolDefinitions(tools)
            s += Self.imStartMark + "system\n"
            if let first = messages.first, first.role == .system {
                let content = first.content ?? ""
                if content.contains(Self.miniCPMToolDefSep) {
                    s += content.replacingOccurrences(of: Self.miniCPMToolDefSep, with: definitions)
                } else {
                    s += content + "\n\n" + definitions
                }
            } else {
                s += Self.pythonLstrip(definitions)
            }
            s += Self.imEndMark + "\n"
        } else if let first = messages.first, first.role == .system {
            s += Self.imStartMark + "system\n" + (first.content ?? "") + Self.imEndMark + "\n"
        }

        for (index, message) in messages.enumerated() {
            let content = message.content ?? ""
            switch message.role {
            case .user, .developer:
                s += Self.imStartMark + "user\n" + content + Self.imEndMark + "\n"
            case .system:
                if index != 0 {
                    s += Self.imStartMark + "system\n" + content + Self.imEndMark + "\n"
                }
            case .assistant:
                s += try Self.miniCPMAssistantTurn(content: content, calls: message.toolCalls)
            case .tool:
                let previousIsTool = index > 0 && messages[index - 1].role == .tool
                let nextIsTool = index + 1 < messages.count && messages[index + 1].role == .tool
                if !previousIsTool { s += Self.imStartMark + "user" }
                s += "\n<tool_response>\n" + content + "\n</tool_response>"
                if !nextIsTool { s += Self.imEndMark + "\n" }
            }
        }

        if addGenerationPrompt {
            s += Self.imStartMark + "assistant\n"
            switch thinking {
            case .enabled: s += "<think>\n"
            case .disabled: s += "<think>\n\n</think>\n\n"
            case .unspecified: break
            }
        }
        return s
    }

    private static func miniCPMAssistantTurn(content raw: String,
                                             calls: [HistoricalToolCall]) throws -> String {
        var content = raw
        var reasoning = ""
        if let close = content.range(of: "</think>") {
            // reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
            var head = String(content[..<close.lowerBound])
            head = pythonRstripNewlines(head)
            if let open = head.range(of: "<think>", options: .backwards) {
                head = String(head[open.upperBound...])
            }
            reasoning = pythonLstripNewlines(head)
            // content = content.split('</think>')[-1].lstrip('\n')
            if let last = content.range(of: "</think>", options: .backwards) {
                content = pythonLstripNewlines(String(content[last.upperBound...]))
            }
        }
        if !calls.isEmpty {
            // Jinja scoping: only the text before the first <tool_sep> survives.
            if let sep = content.range(of: miniCPMToolSep) {
                content = String(content[..<sep.lowerBound])
            }
        }
        var s: String
        if !reasoning.isEmpty {
            s = imStartMark + "assistant\n<think>\n" + pythonStripNewlines(reasoning)
                + "\n</think>\n\n" + pythonLstripNewlines(content)
        } else if !content.contains("<think>") && !content.contains("</think>") {
            s = imStartMark + "assistant\n<think>\n\n</think>\n\n" + pythonLstripNewlines(content)
        } else {
            s = imStartMark + "assistant\n" + content
        }
        for (index, call) in calls.enumerated() {
            if (index == 0 && !content.isEmpty) || index > 0 { s += "\n" }
            s += try miniCPMFunctionXML(call)
        }
        s += imEndMark + "\n"
        return s
    }

    /// `<function name="…"><param name="…">value</param>…</function>`, CDATA
    /// around a string value that contains `<`, `&` or a newline.
    static func miniCPMFunctionXML(_ call: HistoricalToolCall) throws -> String {
        guard case .object(let arguments) = call.arguments else {
            throw MFTokenizerError.invalidChatTemplate(
                "historical tool arguments must be a JSON object")
        }
        var s = "<function name=\"\(call.name)\">"
        for key in arguments.keys.sorted() {
            s += "<param name=\"\(key)\">"
            s += try miniCPMParamValue(arguments[key]!)
            s += "</param>"
        }
        s += "</function>"
        return s
    }

    private static func miniCPMParamValue(_ value: JSONValue) throws -> String {
        switch value {
        case .string(let text):
            if text.contains("<") || text.contains("&") || text.contains("\n") {
                return "<![CDATA[" + text + "]]>"
            }
            return text
        case .integer(let n): return String(n)
        case .unsignedInteger(let n): return String(n)
        case .bool(let b): return b ? "True" : "False"
        case .null: return "None"
        case .number, .decimal, .array, .object:
            // Jinja prints Python's repr here; JSON is the closest portable
            // spelling for the composite cases (recorded deviation).
            return try value.encoded()
        }
    }

    /// `tojson(ensure_ascii=False)` per tool, joined by newlines inside the
    /// `<tools>` block. HF's chat-template `tojson` is plain `json.dumps` with
    /// `", "` / `": "` separators and no HTML escaping.
    private static func miniCPMToolDefinitions(_ tools: [FunctionDefinition]) throws -> String {
        var s = miniCPMToolDefinitionsHead
        for tool in tools {
            let object = JSONValue.object([
                "type": .string("function"),
                "function": .object([
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "parameters": tool.parameters,
                ]),
            ])
            s += "\n" + pythonJSONDumps(object)
        }
        s += miniCPMToolDefinitionsTail
        return s
    }

    /// `json.dumps(value, ensure_ascii=False)`: sorted keys (the only order the
    /// unordered `JSONValue.object` can offer), `", "` and `": "` separators,
    /// Python's escaping of control characters and `"` / `\`.
    static func pythonJSONDumps(_ value: JSONValue) -> String {
        switch value {
        case .object(let object):
            let parts = object.keys.sorted().map { key in
                pythonJSONString(key) + ": " + pythonJSONDumps(object[key]!)
            }
            return "{" + parts.joined(separator: ", ") + "}"
        case .array(let values):
            return "[" + values.map(pythonJSONDumps).joined(separator: ", ") + "]"
        case .string(let text):
            return pythonJSONString(text)
        case .integer(let n): return String(n)
        case .unsignedInteger(let n): return String(n)
        case .bool(let b): return b ? "true" : "false"
        case .null: return "null"
        case .number(let d):
            if d == d.rounded(), abs(d) < 1e16 { return String(format: "%.1f", d) }
            return "\(d)"
        case .decimal(let d): return "\(d)"
        }
    }

    private static func pythonJSONString(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }

    // Python string helpers, restricted to what the template uses.
    private static func pythonLstrip(_ s: String) -> String {
        String(s.drop(while: { $0.isWhitespace }))
    }
    private static func pythonLstripNewlines(_ s: String) -> String {
        String(s.drop(while: { $0 == "\n" }))
    }
    private static func pythonRstripNewlines(_ s: String) -> String {
        var out = Substring(s)
        while out.last == "\n" { out.removeLast() }
        return String(out)
    }
    private static func pythonStripNewlines(_ s: String) -> String {
        pythonRstripNewlines(pythonLstripNewlines(s))
    }
}

/// Parses the MiniCPM XML tool-call body — the text BETWEEN the `<function`
/// and `</function>` special tokens:
///
///     ` name="NAME"><param name="KEY">VALUE</param>…`
///
/// A VALUE is either a CDATA block (`<![CDATA[…]]>`, always a string) or
/// bare text. Bare text that parses as a JSON number / `true` / `false` /
/// `null` / object / array — or Python's `True` / `False` / `None`, which is
/// what the template prints for those types — becomes typed; anything else
/// stays a string.
public struct MiniCPMToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
        var body = Substring(text)
        trimOuterWhitespace(&body)

        guard body.hasPrefix("name=\"") else { throw ToolCallParserError.malformed }
        body.removeFirst("name=\"".count)
        guard let nameEnd = body.firstIndex(of: "\"") else { throw ToolCallParserError.malformed }
        let name = String(body[..<nameEnd])
        body = body[body.index(after: nameEnd)...]
        guard body.first == ">" else { throw ToolCallParserError.malformed }
        body.removeFirst()
        guard name.range(of: #"^[A-Za-z0-9_.\-]{1,64}$"#, options: .regularExpression) != nil else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else { throw ToolCallParserError.unknownTool(name) }

        var arguments: [String: JSONValue] = [:]
        while true {
            trimOuterWhitespace(&body)
            if body.isEmpty { break }
            guard body.hasPrefix("<param name=\"") else { throw ToolCallParserError.malformed }
            body.removeFirst("<param name=\"".count)
            guard let keyEnd = body.firstIndex(of: "\"") else { throw ToolCallParserError.malformed }
            let key = String(body[..<keyEnd])
            body = body[body.index(after: keyEnd)...]
            guard body.first == ">" else { throw ToolCallParserError.malformed }
            body.removeFirst()
            guard !key.isEmpty, arguments[key] == nil else { throw ToolCallParserError.malformed }
            let value: JSONValue
            if body.hasPrefix("<![CDATA[") {
                body.removeFirst("<![CDATA[".count)
                guard let end = body.range(of: "]]>") else { throw ToolCallParserError.malformed }
                value = .string(String(body[..<end.lowerBound]))
                body = body[end.upperBound...]
                guard body.hasPrefix("</param>") else { throw ToolCallParserError.malformed }
            } else {
                guard let end = body.range(of: "</param>") else { throw ToolCallParserError.malformed }
                value = Self.typedValue(String(body[..<end.lowerBound]))
                body = body[end.lowerBound...]
            }
            body.removeFirst("</param>".count)
            arguments[key] = value
        }
        let argumentsValue = JSONValue.object(arguments)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: argumentsValue,
                              argumentsJSON: try argumentsValue.encoded())
    }

    static func typedValue(_ raw: String) -> JSONValue {
        switch raw {
        case "True": return .bool(true)
        case "False": return .bool(false)
        case "None": return .null
        default: break
        }
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
