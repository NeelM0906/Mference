import Foundation

public struct ParsedToolCall: Equatable, Sendable {
    public let id: String
    public let name: String
    public let arguments: JSONValue
    public let argumentsJSON: String
    /// Inkling 2 only: the arguments' text as `argumentsJSON` gives it, kept
    /// on the history entry built from this call (`PromptPrefixCache.makeEntry`)
    /// so the next turn renders it as the model wrote it. Nil from every
    /// other parser.
    public let argumentsText: String?

    public init(id: String,
                name: String,
                arguments: JSONValue,
                argumentsJSON: String,
                argumentsText: String? = nil) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.argumentsJSON = argumentsJSON
        self.argumentsText = argumentsText
    }
}

public enum ToolCallParserError: Error, Equatable {
    case malformed
    case unknownTool(String)
    case oversized
}
