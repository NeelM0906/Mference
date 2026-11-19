import Foundation

/// What one web question through the chat produced (checkpoint C-A2).
public struct WebCheckpointOutcome: Sendable, Hashable, Codable {
    public struct Source: Sendable, Hashable, Codable {
        public var number: Int
        public var url: String
        public var title: String
    }

    public var question: String
    public var model: String
    public var answer: String = ""
    public var reasoningCharacters = 0
    public var finish: String?
    public var sources: [Source] = []
    public var searches: [String] = []
    public var pagesRead = 0
    public var failedSteps: [String] = []
    public var promptTokens: Int?
    public var completionTokens: Int?
    public var error: String?
    /// Finished normally, answered, searched at least once, and cited at
    /// least one source that it read or found.
    public var passed = false
}

/// Asks one question through the real chat path (`ChatViewModel` with web on,
/// the web tools, the given engine) and reports what came back. The
/// checkpoint harness (`MferenceSmoke --web`) runs it on a real model under
/// the machine lock; tests run it on `ScriptedEngineService`.
@MainActor
public enum WebCheckpoint {
    /// The C-A2 question: needs current facts, so a model that answers well
    /// has to search and cite.
    public static let defaultQuestion =
        "What is the latest stable version of the Swift programming language, and when was it released? Cite your sources."

    public static func run(question: String = defaultQuestion, model: AppModel, service: any EngineServicing,
                           tools: any ChatToolboxProvider, timeout: Duration = .seconds(900)) async -> WebCheckpointOutcome {
        var outcome = WebCheckpointOutcome(question: question, model: model.name)
        do {
            let container = try AppModelContainer.inMemory()
            let store = ConversationStore(container: container)
            let conversation = try store.createConversation(model: model, webEnabled: true)
            let chat = ChatViewModel(service: service, store: store, tools: tools)
            chat.select(conversation.id)
            tools.webUsageChanged(enabled: true)
            chat.draft = question
            chat.send()

            // Stops the answer if it runs past the timeout.
            let watchdog = Task { @MainActor in
                try? await Task.sleep(for: timeout)
                guard !Task.isCancelled else { return false }
                chat.stop()
                return true
            }
            await chat.waitUntilIdle()
            watchdog.cancel()
            if await watchdog.value { outcome.error = "no answer within \(timeout)" }

            let stored = conversation.orderedMessages
            if let answer = stored.last(where: { $0.role == .assistant && $0.toolCalls.isEmpty }) {
                outcome.answer = answer.content
                outcome.reasoningCharacters = answer.reasoning?.count ?? 0
                outcome.finish = answer.finishRawValue
                outcome.promptTokens = answer.promptTokens
                outcome.completionTokens = answer.completionTokens
                outcome.sources = answer.citations.map {
                    WebCheckpointOutcome.Source(number: $0.number, url: $0.url.absoluteString, title: $0.title)
                }
                outcome.searches = answer.steps.filter { $0.kind == .search }.compactMap(\.query)
                outcome.pagesRead = answer.steps.filter { $0.kind == .readPage }.count
                outcome.failedSteps = answer.steps.filter { $0.kind == .failed }.compactMap(\.detail)
            }
            if outcome.error == nil, let message = chat.errorMessage { outcome.error = message }
        } catch {
            outcome.error = "could not set up the chat: \(error.localizedDescription)"
        }

        let answered = !outcome.answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        outcome.passed = outcome.error == nil && outcome.finish == ChatFinish.stop.rawValue && answered
            && !outcome.searches.isEmpty && !outcome.sources.isEmpty
        if outcome.error == nil, !outcome.passed {
            outcome.error = !answered ? "empty answer"
                : outcome.searches.isEmpty ? "the model did not search"
                : outcome.sources.isEmpty ? "the answer cites no sources"
                : "answer did not finish normally (\(outcome.finish ?? "none"))"
        }
        return outcome
    }
}
