import Foundation

public struct ChatMessage: Identifiable, Equatable, Sendable {
    public enum Role: String, Sendable {
        case user
        case assistant
    }

    public let id: UUID
    public let role: Role
    public var content: String

    public init(id: UUID = UUID(), role: Role, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }
}

public struct Chat: Identifiable, Equatable, Sendable {
    public let id: UUID
    public var title: String
    public var messages: [ChatMessage]
    public var lastActivity: Date
    public var starred: Bool

    public init(
        id: UUID = UUID(),
        title: String,
        messages: [ChatMessage] = [],
        lastActivity: Date = Date(),
        starred: Bool = false
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.lastActivity = lastActivity
        self.starred = starred
    }
}

public struct ChatModelInfo: Identifiable, Equatable, Sendable {
    public let id: String
    public let label: String
    public let subtitle: String?

    public init(id: String, label: String, subtitle: String? = nil) {
        self.id = id
        self.label = label
        self.subtitle = subtitle
    }
}

public enum ModelCatalog {
    /// Shown until `/v1/models` responds (or when it is unreachable).
    public static let fallback: [ChatModelInfo] = [
        ChatModelInfo(
            id: "gemma-4-26b-a4b-it",
            label: "Gemma 4",
            subtitle: "26B-A4B instruct"),
    ]

    public static let mock = ChatModelInfo(
        id: "mock-model",
        label: "Mock",
        subtitle: "Canned streaming responses")
}
