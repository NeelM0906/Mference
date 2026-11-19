import Foundation

/// Where a model runs, as the phone sees it.
public enum ModelPlace: Sendable, Hashable {
    case thisIPhone
    case pairedMac
}

/// A model the phone can pick for a chat.
public struct PhoneModel: Identifiable, Hashable, Sendable {
    public let id: String
    /// "Quick", "Deep", "Apple's on-device model".
    public let name: String
    /// "Qwen3.5 4B" (empty for Apple's model).
    public let detail: String
    /// What it is good for, one line.
    public let summary: String
    /// Short label for the composer chip and engine label ("Quick 4B").
    public let shortName: String
    public let place: ModelPlace
    /// Shown as a Preview model.
    public let isPreview: Bool

    public init(id: String, name: String, detail: String, summary: String, shortName: String,
                place: ModelPlace, isPreview: Bool = false) {
        self.id = id
        self.name = name
        self.detail = detail
        self.summary = summary
        self.shortName = shortName
        self.place = place
        self.isPreview = isPreview
    }

    public static let quick = PhoneModel(
        id: "quick", name: "Quick", detail: "Qwen3.5 4B · default",
        summary: "Everyday chat, web answers and scanning.", shortName: "Quick 4B", place: .thisIPhone)
    public static let apple = PhoneModel(
        id: "apple", name: "Apple's on-device model", detail: "",
        summary: "Short replies, titles and summaries. Needs Apple Intelligence.", shortName: "Apple model",
        place: .thisIPhone)
    public static let deep = PhoneModel(
        id: "deep", name: "Deep", detail: "Qwen3.8 27B",
        summary: "Harder reasoning and long documents. Your Mac answers; the reply comes back here.",
        shortName: "Deep 27B", place: .pairedMac)
    public static let qwen36 = PhoneModel(
        id: "qwen36", name: "Qwen 3.6", detail: "35B-A3B", summary: "Large mixture-of-experts.",
        shortName: "Qwen 3.6", place: .pairedMac)
    public static let miniCPM = PhoneModel(
        id: "minicpm5", name: "MiniCPM5", detail: "2B", summary: "Very small and quick.",
        shortName: "MiniCPM5", place: .pairedMac)
    public static let maple = PhoneModel(
        id: "maple", name: "Maple", detail: "", summary: "Experimental model.", shortName: "Maple",
        place: .pairedMac, isPreview: true)

    /// The engine label under an answer: "Quick 4B · on this iPhone".
    public var engineLabel: String {
        switch place {
        case .thisIPhone: "\(shortName) · on this iPhone"
        case .pairedMac: "\(shortName) · on your Mac"
        }
    }
}

/// A table inside an answer.
public struct AnswerTable: Hashable, Sendable {
    public var headers: [String]
    public var rows: [[String]]
}

/// The "Better on your Mac" offer and what the person did with it.
public struct HandoffOffer: Hashable, Sendable {
    public enum State: Hashable, Sendable { case offered, sentToMac, returnedFromMac, keptOnIPhone }

    public var reason: String
    public var state: State = .offered
}

/// One message in a conversation.
public struct ChatMessage: Identifiable, Hashable, Sendable {
    public enum Role: Hashable, Sendable { case user, assistant }

    public let id: UUID
    public var role: Role
    /// Body text. `[n]` marks a citation to source n.
    public var text: String
    /// "Searched via paired Mac · 3 searches", when the answer used the web.
    public var webSteps: String?
    public var table: AnswerTable?
    /// A second paragraph after the table.
    public var closing: String?
    public var sources: [String]
    public var actions: [String]
    /// "Quick 4B · on this iPhone".
    public var engineLabel: String?
    public var handoff: HandoffOffer?
    /// True while the reply is still arriving.
    public var isStreaming: Bool

    public init(id: UUID = UUID(), role: Role, text: String, webSteps: String? = nil,
                table: AnswerTable? = nil, closing: String? = nil, sources: [String] = [],
                actions: [String] = [], engineLabel: String? = nil, handoff: HandoffOffer? = nil,
                isStreaming: Bool = false) {
        self.id = id
        self.role = role
        self.text = text
        self.webSteps = webSteps
        self.table = table
        self.closing = closing
        self.sources = sources
        self.actions = actions
        self.engineLabel = engineLabel
        self.handoff = handoff
        self.isStreaming = isStreaming
    }
}

/// A conversation in the Chats list.
public struct Conversation: Identifiable, Hashable, Sendable {
    public enum ListGroup: String, CaseIterable, Sendable {
        case today = "Today", skills = "Skills", privateChats = "Private"
    }

    public let id: UUID
    public var title: String
    public var group: ListGroup
    public var messages: [ChatMessage]
    /// Shown as "On your Mac" while the Mac works on it.
    public var isRunningOnMac: Bool
    public var updatedLabel: String

    public init(id: UUID = UUID(), title: String, group: ListGroup, messages: [ChatMessage] = [],
                isRunningOnMac: Bool = false, updatedLabel: String = "") {
        self.id = id
        self.title = title
        self.group = group
        self.messages = messages
        self.isRunningOnMac = isRunningOnMac
        self.updatedLabel = updatedLabel
    }

    /// The last answer's first line, for the list.
    public var preview: String {
        if isRunningOnMac { return "Reading 40 PDFs with Deep · 12 of 40" }
        let last = messages.last(where: { $0.role == .assistant }) ?? messages.last
        return last.map { CitationText.plain($0.closing ?? $0.text) } ?? ""
    }
}

/// Decides when a request is better done on the Mac. A placeholder rule for
/// the sample build: many or long documents go to the Mac. The real rule
/// comes with the on-device engine (context size, attachments, model fit).
public enum HandoffAdvisor {
    public static func reason(for request: String) -> String? {
        let lowered = request.lowercased()
        let mentionsDocuments = ["pdf", "documents", "files", "spreadsheets"].contains { lowered.contains($0) }
        let mentionsMany = lowered.range(of: #"\b([2-9]\d|\d{3,})\b"#, options: .regularExpression) != nil
        guard mentionsDocuments && mentionsMany else { return nil }
        return "That many long documents is more than this iPhone can read well. Your Mac can do it with Deep and send the result back here."
    }
}

/// Splits answer text into plain runs and citation numbers (`[2]`).
public enum CitationText {
    public enum Run: Hashable, Sendable {
        case text(String)
        case citation(Int)
    }

    public static func runs(_ text: String) -> [Run] {
        var runs: [Run] = []
        var rest = Substring(text)
        while let range = rest.range(of: #"\[(\d+)\]"#, options: .regularExpression) {
            let before = rest[rest.startIndex..<range.lowerBound]
            if !before.isEmpty { runs.append(.text(String(before))) }
            let digits = rest[range].dropFirst().dropLast()
            if let number = Int(digits) { runs.append(.citation(number)) }
            rest = rest[range.upperBound...]
        }
        if !rest.isEmpty { runs.append(.text(String(rest))) }
        return runs
    }

    /// The text without citation marks.
    public static func plain(_ text: String) -> String {
        runs(text).compactMap { run in
            if case .text(let string) = run { return string }
            return nil
        }
        .joined()
        .replacingOccurrences(of: " .", with: ".")
        .trimmingCharacters(in: .whitespaces)
    }
}
