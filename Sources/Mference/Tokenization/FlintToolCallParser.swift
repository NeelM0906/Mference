import Foundation

/// Parses Flint's GLM XML body between `<tool_call>` and `</tool_call>`:
///
///     function-name
///     <arg_key>name</arg_key><arg_value>value</arg_value>...
///
/// The template writes string values raw and non-string values as JSON.
public struct FlintToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      toolSchemas: [String: JSONValue] = [:],
                      id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
        var body = Substring(text)
        trim(&body)
        let firstArgument = body.range(of: "<arg_key>")?.lowerBound
        let rawName = firstArgument.map { String(body[..<$0]) } ?? String(body)
        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty,
              name.range(of: #"^[A-Za-z0-9_.:-]{1,128}$"#,
                         options: .regularExpression) != nil else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else {
            throw ToolCallParserError.unknownTool(name)
        }
        if let firstArgument { body = body[firstArgument...] } else { body = body[body.endIndex...] }

        var arguments: [String: JSONValue] = [:]
        while !body.isEmpty {
            trim(&body)
            if body.isEmpty { break }
            try consume("<arg_key>", from: &body)
            guard let keyEnd = body.range(of: "</arg_key>") else {
                throw ToolCallParserError.malformed
            }
            let key = String(body[..<keyEnd.lowerBound])
            guard !key.isEmpty, !key.contains("<"), arguments[key] == nil else {
                throw ToolCallParserError.malformed
            }
            body = body[keyEnd.upperBound...]
            try consume("<arg_value>", from: &body)
            guard let valueEnd = body.range(of: "</arg_value>") else {
                throw ToolCallParserError.malformed
            }
            let rawValue = String(body[..<valueEnd.lowerBound])
            guard !rawValue.contains("</arg_key>"),
                  !rawValue.contains("</arg_value>"),
                  !rawValue.contains("</tool_call>"),
                  !rawValue.contains("<tool_response>"),
                  !rawValue.contains("</tool_response>") else {
                throw ToolCallParserError.malformed
            }
            arguments[key] = try parsedValue(
                rawValue,
                key: key,
                schema: toolSchemas[name])
            body = body[valueEnd.upperBound...]
        }

        let value = JSONValue.object(arguments)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: value,
                              argumentsJSON: try value.encoded())
    }

    private func trim(_ body: inout Substring) {
        while let first = body.first, first.isWhitespace { body.removeFirst() }
        while let last = body.last, last.isWhitespace { body.removeLast() }
    }

    private func consume(_ literal: String, from body: inout Substring) throws {
        guard body.hasPrefix(literal) else { throw ToolCallParserError.malformed }
        body.removeFirst(literal.count)
    }

    private func parsedValue(_ raw: String,
                             key: String,
                             schema: JSONValue?) throws -> JSONValue {
        if declaredType(in: schema, key: key) == "string" {
            return .string(raw)
        }
        guard let first = raw.first,
              "{[-0123456789tfn".contains(first) else {
            return .string(raw)
        }
        guard let value = try? JSONDecoder().decode(
            JSONValue.self, from: Data(raw.utf8)) else {
            return .string(raw)
        }
        // Unquoted strings are the template's normal string representation.
        if case .string = value { return .string(raw) }
        return value
    }

    private func declaredType(in schema: JSONValue?,
                              key: String) -> String? {
        guard case .object(let root)? = schema,
              case .object(let properties)? = root["properties"],
              let property = properties[key] else {
            return nil
        }
        let object = property.objectValue ?? [:]
        if let type = preferredType(object["type"]) {
            return type
        }

        for keyword in ["oneOf", "anyOf"] {
            if case .array(let branches)? = object[keyword] {
                let types = branches.compactMap {
                    declaredType(
                        in: .object(["properties": .object([key: $0])]),
                        key: key)
                }
                if let type = preferredType(types) { return type }
            }
        }

        // allOf is an intersection: only return a type every branch agrees on.
        if case .array(let branches)? = object["allOf"] {
            let types = branches.compactMap {
                declaredType(
                    in: .object(["properties": .object([key: $0])]),
                    key: key)
            }
            let concrete = types.filter { $0 != "null" }
            if let first = concrete.first,
               concrete.allSatisfy({ $0 == first }) {
                return first
            }
        }
        return nil
    }

    private func preferredType(_ value: JSONValue?) -> String? {
        switch value {
        case .string(let type)?:
            return type
        case .array(let members)?:
            let types = members.compactMap { member -> String? in
                guard case .string(let type) = member, type != "null" else {
                    return nil
                }
                return type
            }
            return preferredType(types)
        default:
            return nil
        }
    }

    private func preferredType(_ types: [String]) -> String? {
        types.contains("string") ? "string" : types.first
    }
}
