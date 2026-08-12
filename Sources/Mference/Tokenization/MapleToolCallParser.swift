import Foundation

/// Parses Maple's JSON body between `<tool_call>` markers.
public struct MapleToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
        let data = Data(text.utf8)
        guard let envelope = try? JSONDecoder().decode(JSONValue.self, from: data),
              case .object(let fields) = envelope,
              Set(fields.keys) == ["name", "arguments"],
              case .string(let name)? = fields["name"],
              case .object(let arguments)? = fields["arguments"],
              hasNoDuplicateObjectKeys(in: text) else {
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

    /// `JSONDecoder` accepts duplicate object keys and keeps one value. Scan
    /// the already validated, depth-bounded JSON and reject that ambiguity at
    /// every object level, including escaped spellings of the same key.
    private func hasNoDuplicateObjectKeys(in text: String) -> Bool {
        var scanner = JSONDuplicateKeyScanner(text)
        return scanner.hasNoDuplicateObjectKeys()
    }
}

private struct JSONDuplicateKeyScanner {
    private let characters: [Character]
    private var index = 0

    init(_ text: String) { characters = Array(text) }

    mutating func hasNoDuplicateObjectKeys() -> Bool {
        skipWhitespace()
        guard let duplicate = value() else { return false }
        skipWhitespace()
        return index == characters.count && !duplicate
    }

    /// Returns whether this value contains a duplicate object key, or nil if
    /// the scanner disagrees with the JSONDecoder validation above.
    private mutating func value() -> Bool? {
        skipWhitespace()
        guard index < characters.count else { return nil }
        switch characters[index] {
        case "{": return object()
        case "[": return array()
        case "\"": return string() == nil ? nil : false
        default:
            let start = index
            while index < characters.count,
                  !characters[index].isWhitespace,
                  !",]}".contains(characters[index]) {
                index += 1
            }
            return index == start ? nil : false
        }
    }

    private mutating func object() -> Bool? {
        guard take("{") else { return nil }
        var seen = Set<String>()
        var duplicate = false
        if take("}") { return duplicate }
        while true {
            guard let key = string(), take(":"), let nestedDuplicate = value() else {
                return nil
            }
            if !seen.insert(key).inserted { duplicate = true }
            if nestedDuplicate { duplicate = true }
            if take("}") { return duplicate }
            guard take(",") else { return nil }
        }
    }

    private mutating func array() -> Bool? {
        guard take("[") else { return nil }
        var duplicate = false
        if take("]") { return duplicate }
        while true {
            guard let nestedDuplicate = value() else { return nil }
            if nestedDuplicate { duplicate = true }
            if take("]") { return duplicate }
            guard take(",") else { return nil }
        }
    }

    private mutating func skipWhitespace() {
        while index < characters.count, characters[index].isWhitespace { index += 1 }
    }

    private mutating func take(_ character: Character) -> Bool {
        skipWhitespace()
        guard index < characters.count, characters[index] == character else { return false }
        index += 1
        return true
    }

    private mutating func string() -> String? {
        skipWhitespace()
        guard index < characters.count, characters[index] == "\"" else { return nil }
        let start = index
        index += 1
        var escaped = false
        while index < characters.count {
            let character = characters[index]
            index += 1
            if escaped {
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == "\"" {
                return try? JSONDecoder().decode(
                    String.self,
                    from: Data(String(characters[start..<index]).utf8))
            }
        }
        return nil
    }

}
