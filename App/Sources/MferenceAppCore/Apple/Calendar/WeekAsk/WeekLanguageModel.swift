import Foundation

/// What a model said back, and whether it ran out of room.
public struct WeekModelReply: Sendable, Hashable {
    public var content: String
    /// Stopped at the token limit: the text may be cut off.
    public var wasCut: Bool
    /// For the log, when the model reports them.
    public var promptTokens: Int?
    public var completionTokens: Int?

    public init(content: String, wasCut: Bool = false, promptTokens: Int? = nil, completionTokens: Int? = nil) {
        self.content = content
        self.wasCut = wasCut
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
}

/// A model Ask about your week can use: it reads the schedule and the
/// thread and answers. Thinking is always off and temperature 0, so the same
/// week and question give the same answer.
public protocol WeekLanguageModel: Sendable {
    /// Shown under the answer ("Quick 4B", "Deep 27B", "Apple on-device").
    var label: String { get }
    /// Another answer (Chat's) runs first: the step reads "Waiting for
    /// Chat" until `started`.
    var isBusyElsewhere: Bool { get async }
    /// `turns` is the system turn, then the thread (user, assistant, ...,
    /// user). `started` is called once the model begins on this request.
    func respond(turns: [ChatTurn], maximumTokens: Int,
                 started: @escaping @Sendable () async -> Void) async throws -> WeekModelReply
}

extension WeekLanguageModel {
    public var isBusyElsewhere: Bool { get async { false } }

    public func respond(turns: [ChatTurn], maximumTokens: Int) async throws -> WeekModelReply {
        try await respond(turns: turns, maximumTokens: maximumTokens, started: {})
    }
}

/// An engine model, used only once it is resident (or about to be), so Ask
/// never makes the engine swap the person's chat model out.
public struct EngineWeekLanguageModel: WeekLanguageModel {
    public let engine: any EngineServicing
    public let model: AppModel

    public init(engine: any EngineServicing, model: AppModel) {
        self.engine = engine
        self.model = model
    }

    public var label: String { model.shortName }

    /// The engine runs one answer at a time: one running or queued now goes
    /// first.
    public var isBusyElsewhere: Bool {
        get async { engine.status.isGenerating || engine.status.queuedRequests > 0 }
    }

    public func respond(turns: [ChatTurn], maximumTokens: Int,
                        started: @escaping @Sendable () async -> Void) async throws -> WeekModelReply {
        let options = GenerationOptions(thinking: false, maximumCompletionTokens: maximumTokens, temperature: 0)
        var content = ""
        var began = false
        for try await update in engine.generate(model: model, turns: turns, options: options) {
            if !began, update.beginsThisRequest {
                began = true
                await started()
            }
            switch update {
            case .content(let text): content += text
            case .completed(let summary):
                try Task.checkCancellation()
                return WeekModelReply(content: summary.content.isEmpty ? content : summary.content,
                                      wasCut: summary.finish == .length, promptTokens: summary.promptTokens,
                                      completionTokens: summary.completionTokens)
            case .prepared, .reasoning, .toolCall: break
            }
        }
        try Task.checkCancellation()
        throw AssistantModelError.unreadableAnswer
    }
}

extension GenerationUpdate {
    /// The engine has taken this request up (its prompt is read, or text
    /// is coming).
    var beginsThisRequest: Bool {
        switch self {
        case .prepared, .content, .reasoning, .completed: true
        case .toolCall: false
        }
    }
}

/// Apple's on-device model, which takes instructions and one prompt: the
/// thread is written out in order, each question labeled "Earlier
/// question" or "Current question" just before its `<USER QUERY>` block,
/// and each earlier reply as the JSON it was.
public struct AssistantWeekLanguageModel: WeekLanguageModel {
    public let model: any AssistantTextModel

    public init(model: any AssistantTextModel) {
        self.model = model
    }

    public var label: String { model.kind.label }

    /// The one prompt Apple's model reads for `turns`.
    static func prompt(_ turns: [ChatTurn]) -> (instructions: String, prompt: String) {
        let instructions = turns.filter { $0.role == .system }.compactMap(\.content).joined(separator: "\n\n")
        let thread = turns.filter { $0.role != .system }
        let lastUser = thread.lastIndex { $0.role == .user }
        var parts: [String] = []
        for (index, turn) in thread.enumerated() {
            let content = turn.content ?? ""
            switch turn.role {
            case .assistant: parts.append("Your earlier reply: " + content)
            case .user:
                let label = index == lastUser ? "Current question:" : "Earlier question:"
                // Titles and questions cannot hold the tag (unfenced), so the
                // first one is the question's own block.
                if let range = content.range(of: "<USER QUERY>") {
                    parts.append(content.replacingCharacters(in: range, with: label + "\n<USER QUERY>"))
                } else {
                    parts.append(label + "\n" + content)
                }
            default: parts.append(content)
            }
        }
        return (instructions, parts.joined(separator: "\n\n"))
    }

    public func respond(turns: [ChatTurn], maximumTokens: Int,
                        started: @escaping @Sendable () async -> Void) async throws -> WeekModelReply {
        let (instructions, prompt) = Self.prompt(turns)
        await started()
        let text = try await model.complete(AssistantRequest(instructions: instructions, prompt: prompt,
                                                             maximumTokens: maximumTokens, temperature: 0))
        return WeekModelReply(content: text)
    }
}

/// Which model answers (canvas board 4): the one already in memory,
/// whatever it is; with none, Apple's on-device model when it is switched
/// on and ready, then Quick (L12). Never a swap. Nil when no model can run:
/// the chips still answer, in Swift.
public struct WeekModelPicker: Sendable {
    private let engine: (any EngineServicing)?
    private let appleModel: (any AssistantTextModel)?
    private let useAppleModel: @Sendable () async -> Bool
    private var fixed: (any WeekLanguageModel)?

    public init(engine: (any EngineServicing)?, appleModel: (any AssistantTextModel)?,
                useAppleModel: @escaping @Sendable () async -> Bool) {
        self.engine = engine
        self.appleModel = appleModel
        self.useAppleModel = useAppleModel
    }

    /// Always `model` (the checkpoint, tests).
    public static func fixed(_ model: any WeekLanguageModel) -> WeekModelPicker {
        var picker = WeekModelPicker(engine: nil, appleModel: nil, useAppleModel: { false })
        picker.fixed = model
        return picker
    }

    /// No model at all (previews, and Macs with nothing installed).
    public static let none = WeekModelPicker(engine: nil, appleModel: nil, useAppleModel: { false })

    public func pick() async -> (any WeekLanguageModel)? {
        if let fixed { return fixed }
        if let engine {
            let status = engine.status
            // Resident, or being loaded already: using it costs no swap.
            if status.phase == .ready, let resident = status.residentModel, resident.isEngineModel {
                return EngineWeekLanguageModel(engine: engine, model: resident)
            }
            if status.phase == .loading, let target = status.targetModel, target.isEngineModel {
                return EngineWeekLanguageModel(engine: engine, model: target)
            }
        }
        if let appleModel, await useAppleModel(), await appleModel.availability() == .available {
            return AssistantWeekLanguageModel(model: appleModel)
        }
        guard let engine else { return nil }
        // Nothing resident (or it is being released): loading Quick swaps
        // nothing the person is using.
        let quick = await engine.models().first { $0.model == .quick }
        switch quick?.availability {
        case .loaded, .ready, .loading, .unloading: return EngineWeekLanguageModel(engine: engine, model: .quick)
        case .notInstalled, nil: return nil
        }
    }
}
