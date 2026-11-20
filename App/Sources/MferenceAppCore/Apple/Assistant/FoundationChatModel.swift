import Foundation
import FoundationModels

/// Chat with Apple's on-device model through FoundationModels. Each answer
/// is a fresh session over the chat's history, trimmed from the oldest end to
/// the model's context (`SystemLanguageModel.contextSize`).
public struct FoundationChatModel: AppleChatModeling {
    public init() {}

    public func info() async -> AppleModelInfo {
        let model = SystemLanguageModel.default
        let state: AppleModelInfo.State = switch model.availability {
        case .available: .ready
        case .unavailable(.deviceNotEligible): .notEligible
        case .unavailable(.appleIntelligenceNotEnabled): .turnedOff
        case .unavailable(.modelNotReady): .gettingReady
        case .unavailable: .gettingReady
        }
        guard state != .notEligible else { return AppleModelInfo(state: state) }
        let labels = Self.labels(of: model)
        AppleModelLabels.update(labels)
        return AppleModelInfo(state: state, labels: labels)
    }

    static func labels(of model: SystemLanguageModel) -> AppleModelLabels {
        #if compiler(>=6.4)
        if #available(macOS 27, *) {
            let variant = model.variant
            let short = switch variant {
            case .core3: "Apple 3B"
            case .coreAdvanced3: "Apple 20B"
            default: "Apple"
            }
            return AppleModelLabels(variantName: variant.displayName, shortName: short,
                                    contextSize: model.contextSize)
        }
        #endif
        return AppleModelLabels(variantName: AppleModelLabels.fallback.variantName,
                                shortName: AppleModelLabels.fallback.shortName, contextSize: model.contextSize)
    }

    public func generate(turns: [ChatTurn], options: GenerationOptions) -> GenerationStream {
        GenerationStream { continuation in
            Task {
                do {
                    try await Self.answer(turns: turns, options: options, into: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: Self.map(error))
                }
            }
        }
    }

    private static func answer(turns: [ChatTurn], options: GenerationOptions,
                               into continuation: GenerationStream.Continuation) async throws {
        let model = SystemLanguageModel.default
        guard model.isAvailable else {
            let info = await FoundationChatModel().info()
            throw AssistantModelError.noModelAvailable(reason: info.state == .turnedOff
                ? AppleChatRouting.turnedOffReason : AppleChatRouting.gettingReadyReason)
        }
        guard let split = AppleChatContext.split(turns) else { throw ChatError.endedEarly }
        let reserve = AppleChatContext.answerReserve(contextSize: model.contextSize,
                                                     requested: options.maximumCompletionTokens)
        let budget = model.contextSize - reserve
        guard let plan = try await AppleChatContext.fit(split, budget: budget, cost: { plan in
            try await tokens(in: entries(for: plan, includingPrompt: true), model: model)
        }) else {
            throw EngineServiceError.requestTooLarge(
                "Apple’s model reads about \(AppleModelLabels.current.contextLabel) tokens at once. Try Quick or Deep.")
        }
        let promptTokens = try await tokens(in: entries(for: plan, includingPrompt: true), model: model)
        continuation.yield(.prepared(promptTokens: promptTokens))

        let session = LanguageModelSession(model: model, tools: [],
                                           transcript: Transcript(entries: entries(for: plan, includingPrompt: false)))
        let generation = FoundationModels.GenerationOptions(temperature: options.temperature.map(Double.init),
                                                            maximumResponseTokens: reserve)
        var sent = ""
        for try await snapshot in session.streamResponse(to: plan.prompt, options: generation) {
            try Task.checkCancellation()
            if let delta = AppleChatContext.delta(sent: sent, snapshot: snapshot.content) {
                continuation.yield(.content(delta))
            }
            sent = snapshot.content
        }
        try Task.checkCancellation()
        let completionTokens = (try? await tokens(in: [response(sent)], model: model)) ?? 0
        continuation.yield(.completed(GenerationSummary(
            content: sent, reasoning: nil, toolCalls: [], finish: .stop,
            promptTokens: promptTokens, completionTokens: completionTokens, historyTrimmed: plan.trimmed)))
    }

    // MARK: Transcript

    private static func entries(for plan: AppleChatContext.Plan, includingPrompt: Bool) -> [Transcript.Entry] {
        var entries: [Transcript.Entry] = []
        if let instructions = plan.instructions {
            entries.append(.instructions(Transcript.Instructions(segments: [text(instructions)], toolDefinitions: [])))
        }
        for turn in plan.history {
            let content = turn.content ?? ""
            entries.append(turn.role == .user ? .prompt(Transcript.Prompt(segments: [text(content)]))
                                              : response(content))
        }
        if includingPrompt { entries.append(.prompt(Transcript.Prompt(segments: [text(plan.prompt)]))) }
        return entries
    }

    private static func text(_ content: String) -> Transcript.Segment {
        .text(Transcript.TextSegment(content: content))
    }

    private static func response(_ content: String) -> Transcript.Entry {
        .response(Transcript.Response(assetIDs: [], segments: [text(content)]))
    }

    /// Exact counts where macOS can give them (26.4 and later), else a
    /// cautious estimate of three characters per token.
    private static func tokens(in entries: [Transcript.Entry], model: SystemLanguageModel) async throws -> Int {
        if #available(macOS 26.4, *) {
            return try await model.tokenCount(for: entries)
        }
        let characters = entries.reduce(0) { total, entry in
            total + String(describing: entry).count
        }
        return characters / 3 + 1
    }

    // MARK: Errors

    private static func map(_ error: any Error) -> any Error {
        if error is CancellationError || error is EngineServiceError || error is AssistantModelError
            || error is ChatError {
            return error
        }
        if let error = error as? LanguageModelSession.GenerationError {
            if case .exceededContextWindowSize = error {
                return EngineServiceError.requestTooLarge("Apple’s model reads about \(AppleModelLabels.current.contextLabel) tokens at once. Try Quick or Deep.")
            }
            return FoundationTextModel.map(error)
        }
        return AssistantModelError.noModelAvailable(reason: "Apple’s on-device model could not answer.")
    }
}
