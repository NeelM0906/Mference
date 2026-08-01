import Foundation

/// Parses the DeepSeek-V4 DSML tool-call body — the text BETWEEN the
/// `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>` markers:
///
///     \n<｜DSML｜invoke name="NAME">\n
///     <｜DSML｜parameter name="KEY" string="true|false">VALUE</｜DSML｜parameter>\n
///     ...
///     </｜DSML｜invoke>\n
///
/// One block may hold several invokes; each becomes its own `ParsedToolCall`.
/// `string="true"` keeps VALUE as the raw string — literal quotes and JSON
/// look-alikes included — while `string="false"` parses VALUE as JSON,
/// mirroring the reference encoder's asymmetric serialization. Whitespace
/// between structural elements is tolerated; anything else is malformed.
public struct DeepseekToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    /// DSML framing text, shared with the tokenizer's native tool-chat render
    /// and the streaming decoder's delta scan. The `｜` bars are fullwidth
    /// (U+FF5C), matching the model's training data.
    public static let dsmlMark = "｜DSML｜"
    public static let toolCallsOpenMark = "<｜DSML｜tool_calls>"
    public static let toolCallsCloseMark = "</｜DSML｜tool_calls>"

    private static let invokeOpenPrefix = "<｜DSML｜invoke name=\""
    private static let invokeCloseMark = "</｜DSML｜invoke>"
    private static let parameterOpenPrefix = "<｜DSML｜parameter name=\""
    private static let parameterCloseMark = "</｜DSML｜parameter>"

    public init() {}

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      idGenerator: () -> String) throws -> [ParsedToolCall] {
        guard text.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
        var body = Substring(text)
        trimOuterWhitespace(&body)
        guard !body.isEmpty else { throw ToolCallParserError.malformed }

        var calls: [ParsedToolCall] = []
        while !body.isEmpty {
            calls.append(try invoke(&body,
                                    allowedTools: allowedTools,
                                    id: idGenerator()))
            trimLeadingWhitespace(&body)
        }
        return calls
    }

    private func trimOuterWhitespace(_ body: inout Substring) {
        while let first = body.first, first.isWhitespace { body.removeFirst() }
        while let last = body.last, last.isWhitespace { body.removeLast() }
    }

    private func trimLeadingWhitespace(_ body: inout Substring) {
        while let first = body.first, first.isWhitespace { body.removeFirst() }
    }

    private func invoke(_ body: inout Substring,
                        allowedTools: Set<String>,
                        id: String) throws -> ParsedToolCall {
        guard body.hasPrefix(Self.invokeOpenPrefix) else {
            throw ToolCallParserError.malformed
        }
        body.removeFirst(Self.invokeOpenPrefix.count)
        guard let quote = body.firstIndex(of: "\"") else {
            throw ToolCallParserError.malformed
        }
        let name = String(body[..<quote])
        body = body[body.index(after: quote)...]
        guard body.first == ">" else { throw ToolCallParserError.malformed }
        body.removeFirst()
        guard isValidFunctionName(name) else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else {
            throw ToolCallParserError.unknownTool(name)
        }

        var arguments: [String: JSONValue] = [:]
        trimLeadingWhitespace(&body)
        while !body.hasPrefix(Self.invokeCloseMark) {
            let (key, value) = try parameter(&body)
            arguments[key] = value
            trimLeadingWhitespace(&body)
        }
        body.removeFirst(Self.invokeCloseMark.count)

        let argumentsValue = JSONValue.object(arguments)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: argumentsValue,
                              argumentsJSON: try argumentsValue.encoded())
    }

    private func parameter(_ body: inout Substring) throws -> (String, JSONValue) {
        guard body.hasPrefix(Self.parameterOpenPrefix) else {
            throw ToolCallParserError.malformed
        }
        body.removeFirst(Self.parameterOpenPrefix.count)
        guard let quote = body.firstIndex(of: "\"") else {
            throw ToolCallParserError.malformed
        }
        let key = String(body[..<quote])
        guard !key.isEmpty, !key.contains("\n"), !key.contains("<") else {
            throw ToolCallParserError.malformed
        }
        body = body[body.index(after: quote)...]
        guard body.hasPrefix(" string=\"") else {
            throw ToolCallParserError.malformed
        }
        body.removeFirst(" string=\"".count)
        let isString: Bool
        if body.hasPrefix("true\">") {
            isString = true
            body.removeFirst("true\">".count)
        } else if body.hasPrefix("false\">") {
            isString = false
            body.removeFirst("false\">".count)
        } else {
            throw ToolCallParserError.malformed
        }
        // VALUE runs to the closing tag; raw strings may span lines.
        guard let closeRange = body.range(of: Self.parameterCloseMark) else {
            throw ToolCallParserError.malformed
        }
        let raw = String(body[..<closeRange.lowerBound])
        body = body[closeRange.upperBound...]
        if isString { return (key, .string(raw)) }
        guard let value = try? JSONDecoder().decode(JSONValue.self,
                                                    from: Data(raw.utf8)) else {
            throw ToolCallParserError.malformed
        }
        return (key, value)
    }

    private func isValidFunctionName(_ name: String) -> Bool {
        name.range(of: "^[A-Za-z0-9_-]{1,64}$", options: .regularExpression) != nil
    }
}
