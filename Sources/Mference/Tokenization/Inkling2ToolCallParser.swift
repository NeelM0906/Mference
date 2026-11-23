import Foundation

/// Parses an Inkling 2 tool call: the JSON body between
/// `<|content_invoke_tool_json|>` and `<|end_message|>`, written by the model
/// as `{"name":"NAME","args":{…}}`, after a `<|message_model|>NAME` header.
public struct Inkling2ToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    /// `headerName` is the text between `<|message_model|>` and the content
    /// token; when present it must name the same tool as the body.
    public func parse(_ text: String,
                      headerName: String,
                      allowedTools: Set<String>,
                      id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
        var scanner = JSONDuplicateKeyScanner(text)
        guard let envelope = try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)),
              case .object(let fields) = envelope,
              Set(fields.keys) == ["name", "args"],
              case .string(let name)? = fields["name"],
              case .object(let arguments)? = fields["args"],
              scanner.hasNoDuplicateObjectKeys() else {
            throw ToolCallParserError.malformed
        }
        let header = headerName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard header.isEmpty || header == name else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else {
            throw ToolCallParserError.unknownTool(name)
        }
        let value = JSONValue.object(arguments)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: value,
                              argumentsJSON: try value.encoded())
    }
}
