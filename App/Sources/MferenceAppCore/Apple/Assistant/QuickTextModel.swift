import Foundation

/// Quick 4B through the app's one engine door (`EngineServicing`), the
/// fallback of decision L12. Thinking is off: these are short jobs.
public struct QuickTextModel: AssistantTextModel {
    private let engine: any EngineServicing

    public init(engine: any EngineServicing) {
        self.engine = engine
    }

    public var kind: AssistantModelKind { .quick }

    public func availability() async -> AssistantAvailability {
        let option = await engine.models().first { $0.model == .quick }
        switch option?.availability {
        case .loaded, .ready, .loading, .unloading:
            return .available
        case .notInstalled(let reason):
            return .unavailable(reason: "Apple Intelligence is off and Quick is not installed. \(reason)")
        case nil:
            return .unavailable(reason: "Apple Intelligence is off and Quick is not installed.")
        }
    }

    public func complete(_ request: AssistantRequest) async throws -> String {
        let turns = [ChatTurn(role: .system, content: request.instructions),
                     ChatTurn(role: .user, content: request.prompt)]
        let options = MferenceAppCore.GenerationOptions(thinking: false,
                                                        maximumCompletionTokens: request.maximumTokens,
                                                        temperature: Float(request.temperature))
        var streamed = ""
        do {
            for try await update in engine.generate(model: .quick, turns: turns, options: options) {
                switch update {
                case .content(let text): streamed += text
                case .completed(let summary): return summary.content.isEmpty ? streamed : summary.content
                case .prepared, .reasoning, .toolCall: break
                }
            }
        } catch let error as EngineServiceError {
            switch error {
            case .requestTooLarge: throw AssistantModelError.inputTooLong
            default: throw AssistantModelError.noModelAvailable(reason: error.userMessage)
            }
        }
        return streamed
    }
}
