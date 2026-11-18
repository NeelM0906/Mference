#if DEBUG
import Foundation
import Synchronization

/// An `EngineServicing` that plays back scripted answers, for previews,
/// snapshots and view-model tests that need several different rounds (a tool
/// call, then an answer). It loads no model and touches no network. Tests of
/// the real service path use `EngineService` over `FakeEngineBackend`.
/// Debug builds only: it never ships in the release app.
public final class ScriptedEngineService: EngineServicing {
    /// One scripted generation.
    public struct Reply: Sendable {
        public var updates: [GenerationUpdate]
        /// Thrown after `updates` instead of completing.
        public var error: EngineServiceError?
        /// Pause before each update (simulated decode speed).
        public var delay: Duration

        public init(content: String, reasoning: String? = nil, toolCalls: [ToolCall] = [],
                    chunk: Int = 6, delay: Duration = .zero, error: EngineServiceError? = nil) {
            var updates: [GenerationUpdate] = [.prepared(promptTokens: 1)]
            if let reasoning { updates += Self.chunks(reasoning, size: chunk).map(GenerationUpdate.reasoning) }
            updates += Self.chunks(content, size: chunk).map(GenerationUpdate.content)
            updates += toolCalls.map(GenerationUpdate.toolCall)
            if error == nil {
                updates.append(.completed(GenerationSummary(
                    content: content, reasoning: reasoning, toolCalls: toolCalls,
                    finish: toolCalls.isEmpty ? .stop : .toolCalls,
                    promptTokens: 1, completionTokens: updates.count)))
            }
            self.updates = updates
            self.error = error
            self.delay = delay
        }

        static func chunks(_ text: String, size: Int) -> [String] {
            guard size > 0, !text.isEmpty else { return text.isEmpty ? [] : [text] }
            var result: [String] = []
            var index = text.startIndex
            while index < text.endIndex {
                let end = text.index(index, offsetBy: size, limitedBy: text.endIndex) ?? text.endIndex
                result.append(String(text[index..<end]))
                index = end
            }
            return result
        }
    }

    private struct State {
        var options: [ModelOption]
        var status: EngineServiceStatus
        var replies: [Reply]
        var fallback: Reply
        var requests: [(model: AppModel, turns: [ChatTurn], options: GenerationOptions)] = []
        var subscribers: [UUID: AsyncStream<EngineServiceStatus>.Continuation] = [:]
    }

    private let state: Mutex<State>

    /// `options` defaults to every model ready; `replies` play in order, then
    /// `fallback` repeats.
    public init(options: [ModelOption]? = nil, status: EngineServiceStatus = .empty,
                replies: [Reply] = [], fallback: Reply = Reply(content: "Hello from a scripted model.")) {
        let rows = options ?? AppModel.allCases.map { ModelOption(model: $0, availability: .ready) }
        state = Mutex(State(options: rows, status: status, replies: replies, fallback: fallback))
    }

    /// Requests seen so far (tests).
    public var requests: [(model: AppModel, turns: [ChatTurn], options: GenerationOptions)] {
        state.withLock { $0.requests }
    }

    public func enqueue(_ reply: Reply) {
        state.withLock { $0.replies.append(reply) }
    }

    public func setStatus(_ status: EngineServiceStatus) {
        state.withLock { s in
            s.status = status
            for continuation in s.subscribers.values { continuation.yield(status) }
        }
    }

    public func models() async -> [ModelOption] {
        state.withLock { s in
            s.options.map { option in
                var option = option
                guard case .notInstalled = option.availability else {
                    if s.status.residentModel == option.model, s.status.phase == .ready { option.availability = .loaded }
                    else if s.status.targetModel == option.model, s.status.phase == .loading { option.availability = .loading }
                    else { option.availability = .ready }
                    return option
                }
                return option
            }
        }
    }

    public func refreshCatalog() async {}

    public var status: EngineServiceStatus { state.withLock { $0.status } }

    public func statusUpdates() -> AsyncStream<EngineServiceStatus> {
        let (stream, continuation) = AsyncStream<EngineServiceStatus>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let id = UUID()
        state.withLock { s in
            s.subscribers[id] = continuation
            continuation.yield(s.status)
        }
        continuation.onTermination = { [weak self] _ in
            self?.state.withLock { _ = $0.subscribers.removeValue(forKey: id) }
        }
        return stream
    }

    public func load(_ model: AppModel) async throws {
        let installed = state.withLock { s in
            s.options.first { $0.model == model }.map { option -> Bool in
                if case .notInstalled = option.availability { return false }
                return true
            } ?? false
        }
        guard installed else { throw EngineServiceError.notInstalled(model: model.name, reason: "Not installed.") }
        setStatus(EngineServiceStatus(phase: .ready, residentModel: model))
    }

    public func unload() async throws {
        setStatus(.empty)
    }

    public func generate(model: AppModel, turns: [ChatTurn],
                         options: GenerationOptions) -> GenerationStream {
        GenerationStream { [self] continuation in
            let reply = self.state.withLock { s -> Reply in
                s.requests.append((model, turns, options))
                return s.replies.isEmpty ? s.fallback : s.replies.removeFirst()
            }
            return Task {
                do {
                    for update in reply.updates {
                        if reply.delay > .zero { try await Task.sleep(for: reply.delay) }
                        try Task.checkCancellation()
                        continuation.yield(update)
                        await Task.yield()
                    }
                    if let error = reply.error { throw error }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func cancelActiveGeneration() async {}

    public func shutdown() async {
        let subscribers = state.withLock { s in
            defer { s.subscribers.removeAll() }
            return Array(s.subscribers.values)
        }
        for continuation in subscribers { continuation.finish() }
    }
}
#endif
