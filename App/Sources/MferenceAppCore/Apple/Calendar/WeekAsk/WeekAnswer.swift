import Foundation

/// Which reply shape the model chose (spec "Model reply"), or prose when the
/// reply was not JSON.
public enum WeekReplyRoute: String, Sendable, Hashable, Codable {
    case freeQuery = "free_query"
    case answer
    case prose
}

/// What an answer stores for a follow-up: the JSON the model (or a chip)
/// produced, replayed as the assistant turn. Never the shown sentence.
public enum WeekAnswerRecord: Sendable, Hashable {
    /// A free-time search Swift answered.
    case freeQuery(FreeQuery)
    /// A lookup: the model's sentence (or a chip's) and the facts it cited.
    case lookup(answer: String, refs: [WeekFact])
    /// A reply that was not JSON: shown, but never replayed.
    case prose
}

/// Where an answer came from.
public enum WeekAnswerSource: Sendable, Hashable {
    /// The model wrote the sentence (a lookup, or prose).
    case model(label: String)
    /// Mference worked it out from the calendar: a chip, or a free-time
    /// question the model passed on (`askedBy`).
    case calendar(askedBy: String?)

    /// The small line under the answer.
    public var caption: String {
        switch self {
        case .model(let label): "\(label) · from your calendar"
        case .calendar(nil): "Mference · worked out from your calendar"
        case .calendar(let label?): "\(label) read the question · Mference worked out the times"
        }
    }
}

/// How a model answered, for the log and the checkpoint.
public struct WeekAnswerTrace: Sendable, Hashable {
    public var model: String
    /// The reply, exactly as it came back.
    public var reply: String
    public var route: WeekReplyRoute
    /// The reply parsed as one of the two JSON shapes.
    public var jsonParsed: Bool
    public var wasCut: Bool
    public var latencyMilliseconds: Int
    public var promptTokens: Int?
    public var completionTokens: Int?

    public init(model: String, reply: String, route: WeekReplyRoute, jsonParsed: Bool, wasCut: Bool,
                latencyMilliseconds: Int, promptTokens: Int? = nil, completionTokens: Int? = nil) {
        self.model = model
        self.reply = reply
        self.route = route
        self.jsonParsed = jsonParsed
        self.wasCut = wasCut
        self.latencyMilliseconds = latencyMilliseconds
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
}

/// One answered question.
public struct WeekAnswer: Sendable, Hashable, Identifiable {
    public var id: UUID
    public var question: String
    public var sentence: String
    /// What the answer points at: open blocks Swift found, or the events
    /// and to-dos it cited.
    public var rows: [WeekFact]
    /// Notes under the answer (mode B's times not in the calendar, a cut-off
    /// answer).
    public var warnings: [String]
    public var source: WeekAnswerSource
    public var record: WeekAnswerRecord
    /// The model call behind it; nil for a chip.
    public var trace: WeekAnswerTrace?

    public init(id: UUID = UUID(), question: String, sentence: String, rows: [WeekFact], warnings: [String] = [],
                source: WeekAnswerSource, record: WeekAnswerRecord, trace: WeekAnswerTrace? = nil) {
        self.id = id
        self.question = question
        self.sentence = sentence
        self.rows = rows
        self.warnings = warnings
        self.source = source
        self.record = record
        self.trace = trace
    }

    /// The free-time search behind the answer, if Swift ran one.
    public var freeQuery: FreeQuery? {
        if case .freeQuery(let query) = record { query } else { nil }
    }

    /// The open blocks among the rows (drawn on the grid, with Block).
    public var openTimes: [DateInterval] {
        rows.compactMap { if case .openTime(let interval) = $0.kind { interval } else { nil } }
    }

    /// The events among the rows (ringed on the grid).
    public var eventIDs: Set<String> {
        Set(rows.compactMap { if case .event(let event) = $0.kind { event.id } else { nil } })
    }

    /// The assistant turn a follow-up replays: the stored JSON, its refs in
    /// `context`'s IDs. Nil for prose, which is never replayed.
    public func replay(in context: WeekContext) -> String? {
        switch record {
        case .freeQuery(let query): return query.json
        case .lookup(let answer, let refs):
            let ids = refs.compactMap { context.fact(matching: $0)?.id }
            let object: [String: Any] = ["answer": answer, "refs": ids]
            guard let data = try? JSONSerialization.data(withJSONObject: object,
                                                         options: [.sortedKeys, .withoutEscapingSlashes]) else {
                return nil
            }
            return String(data: data, encoding: .utf8)
        case .prose: return nil
        }
    }
}

/// What the answerer is doing, for the line under the question.
public enum WeekAskStep: Sendable, Hashable {
    case reading
    /// The engine is answering Chat first.
    case waiting
    case answering

    public var label: String {
        switch self {
        case .reading: "Reading your week"
        case .waiting: "Waiting for Chat"
        case .answering: "Writing the answer"
        }
    }
}

/// A question that got no answer. None of these uses up a question: the
/// question goes back in the box.
public enum WeekAskError: Error, Sendable, Hashable {
    /// No model can run on this Mac now.
    case noModel
    /// The model answered nothing.
    case emptyReply

    public var userMessage: String {
        switch self {
        case .noModel: "No on-device model can answer right now."
        case .emptyReply: "The model couldn't answer. Try again."
        }
    }
}
