import Foundation

// Rich per-message values owned by Chat/Web (U3), stored as JSON `Data` in
// `StoredMessage.stepsJSON` and `StoredMessage.citationsJSON` (CONTRACTS.md:
// features evolve these without a schema migration). Every field added later
// must be optional so older rows still decode.

/// One numbered source an answer cites ("[1]" in the text, a chip below it).
public struct ChatCitation: Sendable, Hashable, Codable, Identifiable {
    /// 1-based number the answer uses.
    public var number: Int
    public var url: URL
    public var title: String
    /// A short excerpt for the hover card.
    public var snippet: String?

    public var id: Int { number }

    public init(number: Int, url: URL, title: String, snippet: String? = nil) {
        self.number = number
        self.url = url
        self.title = title
        self.snippet = snippet
    }

    /// The chip label: host plus a short path ("oakline.example/sale").
    public var label: String {
        var host = url.host() ?? url.absoluteString
        if host.hasPrefix("www.") { host.removeFirst(4) }
        let path = url.path().trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !path.isEmpty else { return host }
        let first = path.split(separator: "/").first.map(String.init) ?? path
        return first.count <= 18 ? "\(host)/\(first)" : host
    }
}

/// One step the assistant took before answering (the steps disclosure).
public struct ChatStep: Sendable, Hashable, Codable {
    public enum Kind: String, Sendable, Hashable, Codable {
        /// A `web_search` call: `query`, `resultCount`.
        case search
        /// An `open_page` call: `url`, `title`.
        case readPage
        /// A tool that failed: `detail` says why.
        case failed
    }

    public var kind: Kind
    public var query: String?
    public var resultCount: Int?
    public var url: URL?
    public var title: String?
    public var detail: String?

    public init(kind: Kind, query: String? = nil, resultCount: Int? = nil, url: URL? = nil,
                title: String? = nil, detail: String? = nil) {
        self.kind = kind
        self.query = query
        self.resultCount = resultCount
        self.url = url
        self.title = title
        self.detail = detail
    }

    public static func search(_ query: String, results: Int) -> ChatStep {
        ChatStep(kind: .search, query: query, resultCount: results)
    }

    public static func readPage(_ url: URL, title: String?) -> ChatStep {
        ChatStep(kind: .readPage, url: url, title: title)
    }
}

/// The one-line header of the steps disclosure.
public enum ChatStepsSummary {
    /// "Searched the web · 3 searches · 7 pages read"; "Reasoning" when the
    /// model only thought; nil when there is nothing to disclose.
    public static func title(steps: [ChatStep], hasReasoning: Bool) -> String? {
        let searches = steps.filter { $0.kind == .search }.count
        let pages = steps.filter { $0.kind == .readPage }.count
        if searches > 0 || pages > 0 {
            var parts = ["Searched the web"]
            if searches > 0 { parts.append(searches == 1 ? "1 search" : "\(searches) searches") }
            if pages > 0 { parts.append(pages == 1 ? "1 page read" : "\(pages) pages read") }
            return parts.joined(separator: " · ")
        }
        if !steps.isEmpty { return "Tried the web" }
        return hasReasoning ? "Reasoning" : nil
    }

    /// Shown under the queries whenever the web was used.
    public static let privacyNote = "These searches went to search engines through SearXNG on this Mac, and the pages were fetched from this Mac (web search is opt-in). The model and what it read stay here."
}

/// What a finished or stopped assistant message records in `finishRawValue`.
public enum ChatFinish: String, Sendable, Hashable, Codable {
    case stop, length, toolCalls, cancelled, failed

    public init(_ finish: GenerationFinish) {
        switch finish {
        case .stop: self = .stop
        case .length: self = .length
        case .toolCalls: self = .toolCalls
        }
    }
}

enum ChatJSON {
    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    static func encode<T: Encodable>(_ value: T) -> Data? {
        try? encoder.encode(value)
    }

    static func decode<T: Decodable>(_ type: T.Type, from data: Data?) -> T? {
        guard let data else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }
}

extension StoredMessage {
    public var role: ChatTurn.Role { ChatTurn.Role(rawValue: roleRawValue) ?? .user }

    public var toolCalls: [ToolCall] {
        get { ChatJSON.decode([ToolCall].self, from: toolCallsJSON) ?? [] }
        set { toolCallsJSON = newValue.isEmpty ? nil : ChatJSON.encode(newValue) }
    }

    public var citations: [ChatCitation] {
        get { ChatJSON.decode([ChatCitation].self, from: citationsJSON) ?? [] }
        set { citationsJSON = newValue.isEmpty ? nil : ChatJSON.encode(newValue) }
    }

    public var steps: [ChatStep] {
        get { ChatJSON.decode([ChatStep].self, from: stepsJSON) ?? [] }
        set { stepsJSON = newValue.isEmpty ? nil : ChatJSON.encode(newValue) }
    }

    public var finish: ChatFinish? { finishRawValue.flatMap(ChatFinish.init(rawValue:)) }

    public var model: AppModel? { modelRawValue.flatMap(AppModel.init(rawValue:)) }

    /// The turn this message contributes to the model's history.
    public var turn: ChatTurn {
        ChatTurn(role: role, content: content, reasoning: reasoning, toolCalls: toolCalls,
                 toolCallID: toolCallID, toolName: toolName)
    }
}

extension Conversation {
    public var model: AppModel {
        get { AppModel(rawValue: modelRawValue) ?? .quick }
        set { modelRawValue = newValue.rawValue }
    }

    /// Messages in order.
    public var orderedMessages: [StoredMessage] {
        messages.sorted { $0.index < $1.index }
    }
}
